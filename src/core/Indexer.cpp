#include "Indexer.h"

#include "core/db/Layout2D.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <queue>
#include <unordered_map>

namespace svdb
{

Indexer::Indexer (db::SampleDatabase& database, db::VectorIndex& index, ml::ClapEncoder& clap)
    : db (database), vectorIndex (index), clapEncoder (clap)
{
}

Indexer::~Indexer()
{
    cancel();
}

void Indexer::cancel()
{
    cancelFlag.store (true);
    if (worker.joinable())
        worker.join();
    running.store (false);
}

IndexProgress Indexer::getProgress() const
{
    std::lock_guard<std::mutex> lock (progMtx);
    return progress;
}

void Indexer::setProgress (IndexPhase phase, int done, int total, const std::string& msg,
                           const std::function<void (const IndexProgress&)>& cb)
{
    IndexProgress p { phase, done, total, msg };
    {
        std::lock_guard<std::mutex> lock (progMtx);
        progress = p;
    }
    if (cb)
        cb (p);
}

void Indexer::startAsync (const std::string& folderPath,
                          const Options& opts,
                          std::function<void()> onIndexUpdated)
{
    cancel();
    cancelFlag.store (false);
    running.store (true);

    worker = std::thread ([this, folderPath, opts, onIndexUpdated = std::move (onIndexUpdated)] {
        runSync (folderPath, opts, [&] (const IndexProgress& p) {
            if (onIndexUpdated && (p.phase == IndexPhase::Done
                                   || (p.phase == IndexPhase::DspExtracting && p.done > 0 && p.done % 512 == 0)
                                   || (p.phase == IndexPhase::ClapEmbedding && p.done > 0 && p.done % 256 == 0)))
            {
                onIndexUpdated();
            }
        });
        running.store (false);
        if (onIndexUpdated)
            onIndexUpdated();
    });
}

void Indexer::runSync (const std::string& folderPath,
                       const Options& opts,
                       std::function<void (const IndexProgress&)> onProgress)
{
    cancelFlag.store (false);
    const auto rootDir = juce::File (juce::String::fromUTF8 (folderPath.c_str()));
    if (! rootDir.isDirectory())
    {
        setProgress (IndexPhase::Error, 0, 0, "Folder not found: " + folderPath, onProgress);
        return;
    }

    db.addFolder (folderPath);
    setProgress (IndexPhase::Scanning, 0, 0, "Scanning folder...", onProgress);

    // 1. Recursively discover supported audio files.
    struct DiscoveredFile
    {
        juce::File file;
        std::string path;
        std::string filename;
        std::string folder;
        int64_t size = 0;
        int64_t mtime = 0;
    };

    std::vector<DiscoveredFile> files;
    std::unordered_map<std::string, bool> presentPaths;

    for (const auto& entry : juce::RangedDirectoryIterator (rootDir, true, "*", juce::File::findFiles))
    {
        if (cancelFlag.load()) return;
        const auto f = entry.getFile();
        const auto ext = f.getFileExtension().toLowerCase();
        if (! isSupportedAudioExtension (ext))
            continue;

        DiscoveredFile df;
        df.file     = f;
        df.path     = f.getFullPathName().toStdString();
        df.filename = f.getFileName().toStdString();
        df.folder   = f.getParentDirectory().getFileName().toStdString();
        df.size     = f.getSize();
        df.mtime    = f.getLastModificationTime().toMilliseconds();
        presentPaths.emplace (df.path, true);
        files.push_back (std::move (df));
    }

    const auto existing = db.getExistingStamps();
    const int pruned = db.deletePathsNotIn (rootDir.getFullPathName().toStdString(), presentPaths);

    std::vector<DiscoveredFile> toDsp;
    toDsp.reserve (files.size());
    for (auto& f : files)
    {
        auto it = existing.find (f.path);
        if (it == existing.end() || it->second.fileSize != f.size || it->second.mtime != f.mtime)
            toDsp.push_back (std::move (f));
    }

    // 2. Phase 1: Parallel DSP extraction.
    bool dbMutated = (pruned > 0);
    if (! toDsp.empty())
    {
        dbMutated = true;
        const int total = static_cast<int> (toDsp.size());
        setProgress (IndexPhase::DspExtracting, 0, total, "Extracting DSP features...", onProgress);

        std::atomic<int> nextIdx { 0 };
        std::atomic<int> completed { 0 };
        std::mutex batchMtx;
        std::vector<SampleRecord> pendingBatch;
        pendingBatch.reserve (256);

        const int numWorkers = std::max (1, std::min (12, static_cast<int> (std::thread::hardware_concurrency()) - 1));
        std::vector<std::thread> pool;
        pool.reserve (static_cast<size_t> (numWorkers));

        for (int w = 0; w < numWorkers; ++w)
        {
            pool.emplace_back ([&] {
                while (! cancelFlag.load())
                {
                    const int i = nextIdx.fetch_add (1);
                    if (i >= total) break;

                    const auto& df = toDsp[static_cast<size_t> (i)];
                    const auto decoded = decoder.decode (df.file, 12.0);
                    if (decoded.ok)
                    {
                        const auto feat = dspExtractor.extract (decoded, df.filename, df.folder, df.path);
                        SampleRecord rec;
                        rec.meta.path        = df.path;
                        rec.meta.filename    = df.filename;
                        rec.meta.folder      = df.folder;
                        rec.meta.fileSize    = df.size;
                        rec.meta.mtime       = df.mtime;
                        rec.meta.durationMs  = decoded.fullDurationMs;
                        rec.meta.sampleRate  = static_cast<int> (decoded.sampleRate);
                        rec.meta.channels    = decoded.channels;
                        rec.meta.isLoop      = feat.isLoop;
                        rec.meta.category    = feat.heuristicCategory;
                        rec.meta.categorySrc = 0;
                        rec.meta.midiNote    = feat.midiNote;
                        rec.meta.brightness  = feat.brightness;
                        rec.meta.punch       = feat.punch;
                        rec.meta.peaks       = feat.peaks;
                        rec.dspVec           = feat.vec;

                        std::vector<SampleRecord> toCommit;
                        {
                            std::lock_guard<std::mutex> lock (batchMtx);
                            pendingBatch.push_back (std::move (rec));
                            if (pendingBatch.size() >= 256)
                                toCommit.swap (pendingBatch);
                        }
                        if (! toCommit.empty())
                            db.upsertDspBatch (toCommit);
                    }

                    const int doneNow = completed.fetch_add (1) + 1;
                    if (doneNow % 64 == 0 || doneNow == total)
                        setProgress (IndexPhase::DspExtracting, doneNow, total,
                                     "DSP " + std::to_string (doneNow) + "/" + std::to_string (total), onProgress);
                }
            });
        }
        for (auto& t : pool) t.join();
        if (! pendingBatch.empty())
            db.upsertDspBatch (pendingBatch);

        vectorIndex.reloadFromDatabase (db);
    }

    if (cancelFlag.load())
        return;

    // 3. Phase 2: Batched CLAP neural embeddings (if enabled & models available).
    if (opts.runClap)
    {
        const std::string mDir = opts.modelDir.empty() ? defaultModelDir().getFullPathName().toStdString() : opts.modelDir;
        if (ml::ClapEncoder::modelsExistIn (mDir))
        {
            std::string err;
            if (! clapEncoder.isAudioReady() || ! clapEncoder.isTextReady())
                clapEncoder.loadModels (mDir, opts.preferDirectML, &err);

            if (clapEncoder.isAudioReady())
            {
                const auto pending = db.getPendingClap();
                const int total = static_cast<int> (pending.size());
                if (total > 0)
                {
                    dbMutated = true;
                    const int batchSize = clapEncoder.isUsingDirectML() ? 16 : 4;
                    setProgress (IndexPhase::ClapEmbedding, 0, total,
                                 clapEncoder.providerLabel() + " 0/" + std::to_string (total), onProgress);

                    std::vector<db::ClapUpdate> commitBuf;
                    commitBuf.reserve (64);

                    for (int offset = 0; offset < total && ! cancelFlag.load(); offset += batchSize)
                    {
                        const int count = std::min (batchSize, total - offset);
                        std::vector<std::vector<float>> clips48k;
                        std::vector<int> validOffsets;
                        clips48k.reserve (static_cast<size_t> (count));
                        validOffsets.reserve (static_cast<size_t> (count));

                        for (int j = 0; j < count; ++j)
                        {
                            const auto& item = pending[static_cast<size_t> (offset + j)];
                            const auto dec = decoder.decode (juce::File (juce::String::fromUTF8 (item.path.c_str())), 10.0);
                            if (dec.ok && ! dec.mono.empty())
                            {
                                clips48k.push_back (dsp::AudioDecoder::resampleLinear (dec.mono, dec.sampleRate, 48000.0));
                                validOffsets.push_back (offset + j);
                            }
                        }

                        if (! clips48k.empty())
                        {
                            std::vector<std::array<float, kClapDim>> embeds;
                            if (clapEncoder.encodeAudioBatch (clips48k, embeds, &err))
                            {
                                for (size_t j = 0; j < embeds.size(); ++j)
                                {
                                    const auto& item = pending[static_cast<size_t> (validOffsets[j])];
                                    db::ClapUpdate u;
                                    u.id = item.id;
                                    u.clapVec = embeds[j];
                                    u.category = clapEncoder.classifyAudioEmbedding (
                                        embeds[j], item.filename, item.folder, item.durationMs, item.isLoop);
                                    commitBuf.push_back (std::move (u));
                                }
                            }
                        }

                        if (commitBuf.size() >= 64)
                        {
                            db.updateClapBatch (commitBuf);
                            commitBuf.clear();
                        }

                        const int doneNow = std::min (total, offset + count);
                        setProgress (IndexPhase::ClapEmbedding, doneNow, total,
                                     clapEncoder.providerLabel() + " " + std::to_string (doneNow) + "/" + std::to_string (total),
                                     onProgress);
                    }

                    if (! commitBuf.empty())
                        db.updateClapBatch (commitBuf);

                    vectorIndex.reloadFromDatabase (db);
                }
            }
        }
    }

    if (cancelFlag.load())
        return;

    // 4. Phase 3: Compute 2D UMAP layout if DB changed or coordinates are uninitialized.
    if (opts.computeLayout)
    {
        const auto allRecs = db.loadAll();
        bool needsLayout = dbMutated;
        if (! needsLayout && allRecs.size() > 1)
        {
            bool allZero = true;
            for (const auto& r : allRecs)
            {
                if (std::abs (r.meta.mapX) > 1e-4f || std::abs (r.meta.mapY) > 1e-4f)
                {
                    allZero = false;
                    break;
                }
            }
            needsLayout = allZero;
        }

        if (needsLayout && ! allRecs.empty())
        {
            setProgress (IndexPhase::ComputingLayout, 0, 390, "Computing 2D maps...", onProgress);
            const auto coords = db::Layout2D::computeMultiBlend (allRecs, 15, 130, [&] (int d, int t) {
                setProgress (IndexPhase::ComputingLayout, d, t,
                             "2D map " + std::to_string (d) + "/" + std::to_string (t), onProgress);
            });
            db.updateLayoutBatch (coords);
            vectorIndex.reloadFromDatabase (db);
        }
    }

    const auto stats = db.getStats();
    setProgress (IndexPhase::Done, stats.total, stats.total,
                 std::to_string (stats.total) + " samples (" + std::to_string (stats.withClap) + " CLAP)",
                 onProgress);
}

} // namespace svdb
