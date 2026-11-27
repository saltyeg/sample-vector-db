// svdb-index: CLI indexer, model downloader, semantic query & similarity test tool.
//
// Usage:
//   svdb-index download-models [--models <dir>]
//   svdb-index scan <folder> [--no-clap] [--cpu] [--db <path>] [--models <dir>]
//   svdb-index query "<text>" [--top N] [--all|--oneshots|--loops] [--db <path>]
//   svdb-index similar <file.wav> [--blend 0.55] [--top N] [--db <path>]
//   svdb-index layout [--db <path>]
//   svdb-index stats [--db <path>]

#include "core/Indexer.h"
#include "core/Types.h"
#include "core/db/Layout2D.h"
#include "core/db/SampleDatabase.h"
#include "core/db/VectorIndex.h"
#include "core/dsp/AudioDecoder.h"
#include "core/dsp/DspExtractor.h"
#include "core/ml/ClapEncoder.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#endif

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{

#ifdef _WIN32
bool downloadHttpsFile (const std::wstring& host,
                        const std::wstring& path,
                        const std::filesystem::path& destFile,
                        const std::string& label)
{
    std::error_code ec;
    if (destFile.has_parent_path())
        std::filesystem::create_directories (destFile.parent_path(), ec);

    const auto tmpFile = destFile.wstring() + L".part";

    HINTERNET hSession = WinHttpOpen (L"sample-vector-db/0.1",
                                      WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (! hSession) { std::cerr << "WinHttpOpen failed\n"; return false; }

    DWORD redirectPolicy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption (hSession, WINHTTP_OPTION_REDIRECT_POLICY, &redirectPolicy, sizeof (redirectPolicy));
    WinHttpSetTimeouts (hSession, 15000, 15000, 30000, 120000);

    HINTERNET hConnect = WinHttpConnect (hSession, host.c_str(), INTERNET_DEFAULT_HTTPS_PORT, 0);
    if (! hConnect) { WinHttpCloseHandle (hSession); std::cerr << "WinHttpConnect failed\n"; return false; }

    HINTERNET hRequest = WinHttpOpenRequest (hConnect, L"GET", path.c_str(), nullptr,
                                             WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                             WINHTTP_FLAG_SECURE);
    if (! hRequest)
    {
        WinHttpCloseHandle (hConnect);
        WinHttpCloseHandle (hSession);
        std::cerr << "WinHttpOpenRequest failed\n";
        return false;
    }

    WinHttpSetOption (hRequest, WINHTTP_OPTION_REDIRECT_POLICY, &redirectPolicy, sizeof (redirectPolicy));

    bool ok = false;
    if (WinHttpSendRequest (hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
        && WinHttpReceiveResponse (hRequest, nullptr))
    {
        DWORD statusCode = 0;
        DWORD statusSize = sizeof (statusCode);
        WinHttpQueryHeaders (hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &statusSize, WINHTTP_NO_HEADER_INDEX);

        uint64_t totalLen = 0;
        wchar_t clBuf[64] {};
        DWORD clSize = sizeof (clBuf);
        if (WinHttpQueryHeaders (hRequest, WINHTTP_QUERY_CONTENT_LENGTH,
                                 WINHTTP_HEADER_NAME_BY_INDEX, clBuf, &clSize, WINHTTP_NO_HEADER_INDEX))
        {
            totalLen = std::wcstoull (clBuf, nullptr, 10);
        }

        if (statusCode == 200)
        {
            std::ofstream out (tmpFile, std::ios::binary);
            std::vector<char> buf (65536);
            uint64_t downloaded = 0;
            uint64_t lastPrintMb = 0;

            DWORD avail = 0;
            ok = true;
            while (WinHttpQueryDataAvailable (hRequest, &avail) && avail > 0)
            {
                if (avail > buf.size()) buf.resize (avail);
                DWORD readBytes = 0;
                if (! WinHttpReadData (hRequest, buf.data(), avail, &readBytes) || readBytes == 0)
                {
                    ok = false;
                    break;
                }
                out.write (buf.data(), static_cast<std::streamsize> (readBytes));
                downloaded += readBytes;

                const uint64_t mb = downloaded / (1024 * 1024);
                if (mb >= lastPrintMb + 10)
                {
                    lastPrintMb = mb;
                    if (totalLen > 0)
                        std::cout << "\r  " << label << ": " << mb << " / " << (totalLen / (1024 * 1024)) << " MB   " << std::flush;
                    else
                        std::cout << "\r  " << label << ": " << mb << " MB   " << std::flush;
                }
            }
            out.close();

            if (ok && downloaded > 0)
            {
                std::filesystem::rename (tmpFile, destFile, ec);
                if (ec)
                {
                    std::filesystem::remove (destFile, ec);
                    std::filesystem::rename (tmpFile, destFile, ec);
                }
                std::cout << "\r  " << label << ": " << std::fixed << std::setprecision (1)
                          << (static_cast<double> (downloaded) / (1024.0 * 1024.0)) << " MB [OK]         \n";
            }
        }
        else
        {
            std::cerr << "HTTP status " << statusCode << " for " << label << "\n";
        }
    }
    else
    {
        std::cerr << "HTTP request failed for " << label << " (err " << GetLastError() << ")\n";
    }

    WinHttpCloseHandle (hRequest);
    WinHttpCloseHandle (hConnect);
    WinHttpCloseHandle (hSession);
    return ok;
}
#endif

void printUsage()
{
    std::cout << "Usage:\n"
              << "  svdb-index download-models [--models <dir>]\n"
              << "  svdb-index scan <folder> [--no-clap] [--cpu] [--db <path>] [--models <dir>]\n"
              << "  svdb-index query \"<text>\" [--top N] [--all|--oneshots|--loops] [--db <path>]\n"
              << "  svdb-index similar <file.wav> [--blend 0.55] [--top N] [--db <path>]\n"
              << "  svdb-index autoname <folder> [--preview N] [--apply] [--out <dir>] [--db <path>]\n"
              << "  svdb-index layout [--db <path>]\n"
              << "  svdb-index stats [--db <path>]\n";
}

} // namespace

int main (int argc, char* argv[])
{
    if (argc < 2)
    {
        printUsage();
        return 1;
    }

    const std::string cmd = argv[1];
    std::string dbPath    = svdb::defaultDbFile().getFullPathName().toStdString();
    std::string modelDir  = svdb::defaultModelDir().getFullPathName().toStdString();
    bool noClap = false;
    bool forceCpu = false;
    bool applyRename = false;
    int topK = 15;
    int previewCount = 20;
    float blend = 0.55f;
    std::string outDir;
    svdb::LengthFilter lenFilter = svdb::LengthFilter::All;

    std::vector<std::string> positional;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--no-clap")          noClap = true;
        else if (a == "--cpu")         forceCpu = true;
        else if (a == "--apply")       applyRename = true;
        else if (a == "--all")         lenFilter = svdb::LengthFilter::All;
        else if (a == "--oneshots")    lenFilter = svdb::LengthFilter::Oneshots;
        else if (a == "--loops")       lenFilter = svdb::LengthFilter::Loops;
        else if (a == "--db" && i + 1 < argc)      dbPath = argv[++i];
        else if (a == "--models" && i + 1 < argc)  modelDir = argv[++i];
        else if (a == "--out" && i + 1 < argc)     outDir = argv[++i];
        else if (a == "--top" && i + 1 < argc)     topK = std::stoi (argv[++i]);
        else if (a == "--preview" && i + 1 < argc) previewCount = std::stoi (argv[++i]);
        else if (a == "--blend" && i + 1 < argc)   blend = std::stof (argv[++i]);
        else positional.push_back (a);
    }

    if (cmd == "download-models")
    {
#ifdef _WIN32
        const auto base = std::filesystem::u8path (modelDir);
        std::cout << "Downloading Xenova/clap-htsat-unfused ONNX models to: " << modelDir << "\n";
        struct DlItem { std::wstring remote; std::string local; };
        const DlItem items[] = {
            { L"/Xenova/clap-htsat-unfused/resolve/main/tokenizer.json",         "tokenizer.json" },
            { L"/Xenova/clap-htsat-unfused/resolve/main/onnx/audio_model.onnx",  "audio_model.onnx" },
            { L"/Xenova/clap-htsat-unfused/resolve/main/onnx/text_model.onnx",   "text_model.onnx" },
        };
        for (const auto& it : items)
        {
            const auto dst = base / it.local;
            if (std::filesystem::exists (dst) && std::filesystem::file_size (dst) > 10000)
            {
                std::cout << "  " << it.local << ": already exists ("
                          << (std::filesystem::file_size (dst) / (1024 * 1024)) << " MB), skipping.\n";
                continue;
            }
            if (! downloadHttpsFile (L"huggingface.co", it.remote, dst, it.local))
                return 1;
        }
        std::cout << "All CLAP models ready.\n";
        return 0;
#else
        std::cerr << "download-models is implemented for Windows WinHTTP.\n";
        return 1;
#endif
    }

    svdb::db::SampleDatabase db;
    std::string err;
    if (! db.open (dbPath, &err))
    {
        std::cerr << "Failed to open DB " << dbPath << ": " << err << "\n";
        return 1;
    }

    if (cmd == "stats")
    {
        const auto s = db.getStats();
        std::cout << "Database: " << dbPath << "\n"
                  << "  Total:    " << s.total << "\n"
                  << "  Oneshots: " << s.oneshots << "\n"
                  << "  Loops:    " << s.loops << "\n"
                  << "  CLAP:     " << s.withClap << "\n"
                  << "  Categories:\n";
        for (int c = 0; c < svdb::kNumCategories; ++c)
            std::cout << "    " << std::setw (8) << std::left << svdb::categoryName (static_cast<svdb::Category> (c))
                      << s.byCategory[static_cast<size_t> (c)] << "\n";
        return 0;
    }

    if (cmd == "scan")
    {
        if (positional.empty())
        {
            std::cerr << "Missing <folder> argument.\n";
            return 1;
        }
        svdb::db::VectorIndex index;
        svdb::ml::ClapEncoder clap;
        svdb::Indexer indexer (db, index, clap);

        svdb::Indexer::Options opts;
        opts.runClap = ! noClap;
        opts.computeLayout = true;
        opts.preferDirectML = ! forceCpu;
        opts.modelDir = modelDir;

        const auto t0 = std::chrono::steady_clock::now();
        svdb::IndexPhase lastPhase = svdb::IndexPhase::Idle;
        indexer.runSync (positional[0], opts, [&] (const svdb::IndexProgress& p) {
            if (p.phase != lastPhase)
            {
                if (lastPhase != svdb::IndexPhase::Idle) std::cout << "\n";
                lastPhase = p.phase;
            }
            std::cout << "\r" << p.message << "          " << std::flush;
        });
        const double secs = std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count();
        std::cout << "\nFinished in " << std::fixed << std::setprecision (2) << secs << " s.\n";
        return 0;
    }

    if (cmd == "retag")
    {
        svdb::ml::ClapEncoder clap;
        const bool clapOk = clap.loadTextModelOnly (modelDir, &err);
        const auto all = db.loadAll();

        std::vector<svdb::db::LoopFlagUpdate> loopUpdates;
        loopUpdates.reserve (all.size());
        std::vector<svdb::db::ClapUpdate> clapUpdates;
        clapUpdates.reserve (all.size());

        for (const auto& r : all)
        {
            const bool isLoop = svdb::dsp::DspExtractor::classifyIsLoop (
                r.meta.path, r.meta.filename, r.meta.durationMs, r.meta.peaks);
            loopUpdates.push_back ({ r.meta.id, isLoop });

            if (clapOk && r.meta.hasClap)
            {
                svdb::db::ClapUpdate u;
                u.id = r.meta.id;
                u.clapVec = r.clapVec;
                u.category = clap.classifyAudioEmbedding (
                    r.clapVec, r.meta.filename, r.meta.folder, r.meta.durationMs, isLoop);
                clapUpdates.push_back (u);
            }
        }
        db.updateLoopFlagBatch (loopUpdates);
        if (! clapUpdates.empty())
            db.updateClapBatch (clapUpdates);
        std::cout << "Re-tagged " << loopUpdates.size() << " samples (loop flags"
                  << (clapUpdates.empty() ? "" : " + CLAP categories") << ").\n";
        return 0;
    }

    if (cmd == "autoname")
    {
        if (positional.empty())
        {
            std::cerr << "Missing <folder> argument.\n";
            return 1;
        }
        svdb::ml::ClapEncoder clap;
        if (! clap.loadTextModelOnly (modelDir, &err))
        {
            std::cerr << "Failed to load CLAP text model: " << err << "\n";
            return 1;
        }

        // Embed concise contrasting producer vibe descriptors via CLAP.
        struct VibePrompt { const char* tag; const char* prompt; };
        static constexpr VibePrompt kVibes[] = {
            { "Raw",    "raw gritty vintage lo-fi analog drum break texture" },
            { "Punchy", "punchy hard hitting tight transient impact sound" },
            { "Crisp",  "crisp bright clean sharp high frequency shimmer" },
            { "Dark",   "dark moody deep murky low-pass filtered tone" },
            { "Warm",   "warm lush smooth mellow analog pad chord" },
            { "Sub",    "deep heavy sub bass rumble low end boom" },
            { "Dirty",  "distorted crunchy overdriven aggressive saturated sound" },
            { "Metal",  "metallic clanging resonant industrial bell cymbal" },
            { "Airy",   "airy atmospheric reverb ambient spacious tail" },
            { "Tight",  "short dry tight staccato percussive snap" },
            { "Heavy",  "heavy thick slamming weighty dense low midrange texture" },
            { "Acid",   "squelchy resonant synth acid electronic zap" },
        };
        std::vector<std::pair<std::string, std::array<float, svdb::kClapDim>>> vibeProtos;
        for (const auto& vp : kVibes)
        {
            std::array<float, svdb::kClapDim> emb {};
            if (clap.encodeText (vp.prompt, emb))
                vibeProtos.emplace_back (vp.tag, emb);
        }

        // 5 ms dual-band transient onset-flux autocorrelation comb (1/2, 1, 2, 4, 8 beat).
        // Immune to extra hits or reverb tails at the end of sample-CD breaks.
        svdb::dsp::AudioDecoder bpmDecoder;
        auto detectBpm = [&bpmDecoder] (const std::string& filePath, float durationMs) -> int {
            if (durationMs < 1500.0f) return 0;
            const auto dec = bpmDecoder.decode (juce::File (juce::String::fromUTF8 (filePath.c_str())), 12.0);
            if (! dec.ok || dec.mono.empty() || dec.sampleRate <= 0.0) return 0;

            const int hop = std::max (1, static_cast<int> (std::lround (dec.sampleRate * 0.005))); // 5 ms -> 200 Hz
            const int nwin = static_cast<int> (dec.mono.size()) / hop;
            if (nwin < 260) return 0;

            std::vector<float> eLow (static_cast<size_t> (nwin), 0.0f);
            std::vector<float> eHi  (static_cast<size_t> (nwin), 0.0f);
            for (int w = 0; w < nwin; ++w)
            {
                const int base = w * hop;
                float el = 0.0f, eh = 0.0f, prev = 0.0f;
                for (int i = 0; i < hop; ++i)
                {
                    const float v = dec.mono[static_cast<size_t> (base + i)];
                    el += v * v;
                    const float d = v - prev;
                    eh += d * d;
                    prev = v;
                }
                eLow[static_cast<size_t> (w)] = std::log1p (el * 20.0f);
                eHi[static_cast<size_t> (w)]  = std::log1p (eh * 20.0f);
            }

            std::vector<float> onset (static_cast<size_t> (nwin), 0.0f);
            double sumO = 0.0;
            for (int w = 1; w < nwin; ++w)
            {
                const float dl = std::max (0.0f, eLow[static_cast<size_t> (w)] - eLow[static_cast<size_t> (w - 1)]);
                const float dh = std::max (0.0f, eHi[static_cast<size_t> (w)]  - eHi[static_cast<size_t> (w - 1)]);
                onset[static_cast<size_t> (w)] = dl + 0.8f * dh;
                sumO += onset[static_cast<size_t> (w)];
            }
            const float meanO = static_cast<float> (sumO / nwin);
            double var = 1e-9;
            for (float& x : onset) { x -= meanO; var += static_cast<double> (x) * x; }

            const int maxLag = std::min (nwin - 30, 820);
            std::vector<float> acTbl (static_cast<size_t> (maxLag + 1), 0.0f);
            for (int lag = 25; lag <= maxLag; ++lag)
            {
                const int cnt = nwin - lag;
                double acc = 0.0;
                for (int i = 0; i < cnt; ++i)
                    acc += static_cast<double> (onset[static_cast<size_t> (i)]) * onset[static_cast<size_t> (i + lag)];
                acTbl[static_cast<size_t> (lag)] = static_cast<float> (acc / (var * (static_cast<double> (cnt) / nwin)));
            }

            auto ac = [&] (double lagF) -> float {
                const int k = static_cast<int> (std::lround (lagF));
                if (k < 25 || k > maxLag) return 0.0f;
                float best = acTbl[static_cast<size_t> (k)];
                if (k > 25)     best = std::max (best, acTbl[static_cast<size_t> (k - 1)]);
                if (k < maxLag) best = std::max (best, acTbl[static_cast<size_t> (k + 1)]);
                return best;
            };

            double bestBpm = 0.0;
            float bestSc = -1e9f;
            const double hz = dec.sampleRate / static_cast<double> (hop);
            for (int bpmX2 = 260; bpmX2 <= 380; ++bpmX2) // 130.0 .. 190.0 step 0.5
            {
                const double bpm = bpmX2 * 0.5;
                const double lag = (60.0 * hz) / bpm;
                const float sc = 0.35f * ac (lag * 0.5)
                               + 0.70f * ac (lag)
                               + 0.85f * ac (lag * 2.0)
                               + 1.25f * ac (lag * 4.0)
                               + 1.15f * ac (lag * 8.0);
                if (sc > bestSc)
                {
                    bestSc = sc;
                    bestBpm = bpm;
                }
            }
            return (bestSc >= 0.95f) ? static_cast<int> (std::lround (bestBpm)) : 0;
        };

        auto isDrumCat = [] (svdb::Category c) {
            return c == svdb::Category::Kick || c == svdb::Category::Snare || c == svdb::Category::Rim
                || c == svdb::Category::Clap || c == svdb::Category::ClosedHat || c == svdb::Category::OpenHat
                || c == svdb::Category::Cymbal || c == svdb::Category::Tom || c == svdb::Category::Perc;
        };

        // Filter DB records under <folder> (or all if folder matches prefix)
        const std::string rootPrefix = juce::File (juce::String::fromUTF8 (positional[0].c_str())).getFullPathName().toStdString();
        const auto all = db.loadAll();
        std::vector<const svdb::SampleRecord*> matching;
        for (const auto& r : all)
        {
            if (r.meta.path.rfind (rootPrefix, 0) == 0)
                matching.push_back (&r);
        }
        if (matching.empty())
        {
            std::cerr << "No indexed samples under: " << rootPrefix << " (run 'scan' first)\n";
            return 1;
        }

        std::unordered_map<std::string, int> seqByPrefix;
        std::vector<std::pair<const svdb::SampleRecord*, std::string>> plan;
        plan.reserve (matching.size());

        for (const auto* r : matching)
        {
            const auto& m = r->meta;
            // 1. Short 3-letter code
            std::string code = (m.isLoop && isDrumCat (m.category)) ? "BRK" : svdb::categoryShort (m.category);

            // 2. Best CLAP vibe descriptor
            std::string vibe = "Raw";
            if (m.hasClap && ! vibeProtos.empty())
            {
                float bestSc = -1e9f;
                for (const auto& vp : vibeProtos)
                {
                    const float sc = svdb::db::VectorIndex::dotProduct (r->clapVec.data(), vp.second.data(), svdb::kClapDim);
                    if (sc > bestSc)
                    {
                        bestSc = sc;
                        vibe = vp.first;
                    }
                }
            }

            // 3. Real transient-comb BPM (only for rhythmic drum breaks) or Musical Note (for tonal oneshots)
            std::string extra;
            if (code == "BRK")
            {
                const int bpm = detectBpm (m.path, m.durationMs);
                if (bpm > 0) extra = "_" + std::to_string (bpm);
            }
            else if (! m.isLoop && m.midiNote >= 0)
            {
                extra = "_" + svdb::midiNoteToName (m.midiNote);
            }

            const std::string key = code + "_" + vibe + extra;
            const int seq = ++seqByPrefix[key];
            char seqBuf[16] {};
            std::snprintf (seqBuf, sizeof (seqBuf), "%03d", seq);
            const std::string newName = key + "_" + seqBuf + ".wav";
            plan.emplace_back (r, newName);
        }

        if (! applyRename)
        {
            const size_t total = plan.size();
            const size_t showN = std::min<size_t> (total, static_cast<size_t> (std::max (1, previewCount)));
            std::cout << "Preview (" << showN << " of " << total << " samples, style: <CAT>_<Vibe>[_<BPM|Note>]_<###>.wav):\n";
            for (size_t k = 0; k < showN; ++k)
            {
                const size_t idx = (showN >= total) ? k : (k * total) / showN;
                const auto* r = plan[idx].first;
                std::cout << "  " << std::setw (20) << std::left << r->meta.filename
                          << " (" << std::setw (6) << std::right << std::fixed << std::setprecision (2)
                          << (r->meta.durationMs / 1000.0f) << "s " << (r->meta.isLoop ? "Loop" : "One ") << ")  ->  "
                          << plan[idx].second << "\n";
            }
            return 0;
        }

        const auto dstRoot = outDir.empty() ? std::filesystem::u8path (rootPrefix) : std::filesystem::u8path (outDir);
        std::error_code ec;
        std::filesystem::create_directories (dstRoot, ec);
        int moved = 0;
        for (const auto& [r, newName] : plan)
        {
            const auto src = std::filesystem::u8path (r->meta.path);
            const auto dst = dstRoot / newName;
            if (outDir.empty())
                std::filesystem::rename (src, dst, ec);
            else
                std::filesystem::copy_file (src, dst, std::filesystem::copy_options::overwrite_existing, ec);
            if (! ec) ++moved;
        }
        std::cout << (outDir.empty() ? "Renamed " : "Copied ") << moved << " / " << plan.size()
                  << " files into " << dstRoot.string() << "\n";
        return 0;
    }

    if (cmd == "layout")
    {
        const auto t0 = std::chrono::steady_clock::now();
        const auto all = db.loadAll();
        std::cout << "Computing 3 Procrustes-aligned 2D UMAP maps (DSP / Hybrid / CLAP) for " << all.size() << " samples...\n";
        const auto coords = svdb::db::Layout2D::computeMultiBlend (all, 15, 130, [] (int d, int t) {
            std::cout << "\r  step " << d << "/" << t << "   " << std::flush;
        });
        db.updateLayoutBatch (coords);
        const double secs = std::chrono::duration<double> (std::chrono::steady_clock::now() - t0).count();
        std::cout << "\nMulti-blend layouts saved in " << std::fixed << std::setprecision (2) << secs << " s.\n";
        return 0;
    }

    if (cmd == "query")
    {
        if (positional.empty())
        {
            std::cerr << "Missing \"<text>\" argument.\n";
            return 1;
        }
        svdb::db::VectorIndex index;
        index.reloadFromDatabase (db);
        const auto snap = index.getSnapshot();

        svdb::SearchQuery q;
        q.text = positional[0];
        q.lengthFilter = lenFilter;
        q.maxResults = topK;

        svdb::ml::ClapEncoder clap;
        if (svdb::ml::ClapEncoder::modelsExistIn (modelDir))
        {
            if (clap.loadTextModelOnly (modelDir, &err))
            {
                std::array<float, svdb::kClapDim> emb {};
                if (clap.encodeText (q.text, emb, &err))
                    q.textClapVec.assign (emb.begin(), emb.end());
                else
                    std::cerr << "[CLAP text encode error] " << err << "\n";
            }
            else
            {
                std::cerr << "[CLAP text load error] " << err << "\n";
            }
        }

        const auto t0 = std::chrono::steady_clock::now();
        const auto results = index.search (q);
        const double ms = std::chrono::duration<double, std::milli> (std::chrono::steady_clock::now() - t0).count();

        std::cout << "Query: \"" << q.text << "\" (" << (q.textClapVec.empty() ? "lexical" : "CLAP")
                  << ", " << std::fixed << std::setprecision (2) << ms << " ms)\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& m = snap->meta[static_cast<size_t> (results[i].index)];
            std::cout << "  " << std::setw (2) << (i + 1) << ". [" << std::setw (3) << svdb::categoryShort (m.category) << "] "
                      << std::setw (4) << svdb::midiNoteToName (m.midiNote) << " "
                      << std::setw (6) << std::fixed << std::setprecision (0) << m.durationMs << "ms  "
                      << std::setprecision (3) << results[i].score << "  " << m.filename
                      << " (" << m.folder << ")\n";
        }
        return 0;
    }

    if (cmd == "similar")
    {
        if (positional.empty())
        {
            std::cerr << "Missing <file.wav> argument.\n";
            return 1;
        }
        svdb::db::VectorIndex index;
        index.reloadFromDatabase (db);
        const auto snap = index.getSnapshot();

        svdb::dsp::AudioDecoder dec;
        svdb::dsp::DspExtractor ext;
        const juce::File f (juce::String::fromUTF8 (positional[0].c_str()));
        const auto audio = dec.decode (f, 12.0);
        if (! audio.ok)
        {
            std::cerr << "Cannot decode " << positional[0] << "\n";
            return 1;
        }
        const auto feat = ext.extract (audio, f.getFileName().toStdString(), f.getParentDirectory().getFileName().toStdString());

        svdb::SearchQuery q;
        q.refDspVec.assign (feat.vec.begin(), feat.vec.end());
        q.clapBlend = blend;
        q.lengthFilter = lenFilter;
        q.maxResults = topK;

        svdb::ml::ClapEncoder clap;
        if (! noClap && svdb::ml::ClapEncoder::modelsExistIn (modelDir)
            && clap.loadAudioModelOnly (modelDir, ! forceCpu, &err))
        {
            std::vector<std::vector<float>> batch { svdb::dsp::AudioDecoder::resampleLinear (audio.mono, audio.sampleRate, 48000.0) };
            std::vector<std::array<float, svdb::kClapDim>> embeds;
            if (clap.encodeAudioBatch (batch, embeds) && ! embeds.empty())
                q.refClapVec.assign (embeds[0].begin(), embeds[0].end());
        }

        const auto results = index.search (q);
        std::cout << "Similar to: " << f.getFileName().toStdString()
                  << " (blend=" << blend << ", clap=" << (! q.refClapVec.empty() ? "on" : "off") << ")\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& m = snap->meta[static_cast<size_t> (results[i].index)];
            std::cout << "  " << std::setw (2) << (i + 1) << ". [" << std::setw (3) << svdb::categoryShort (m.category) << "] "
                      << std::setw (4) << svdb::midiNoteToName (m.midiNote) << " "
                      << std::setw (6) << std::fixed << std::setprecision (0) << m.durationMs << "ms  "
                      << std::setprecision (3) << results[i].score << "  " << m.filename
                      << " (" << m.folder << ")\n";
        }
        return 0;
    }

    printUsage();
    return 1;
}
