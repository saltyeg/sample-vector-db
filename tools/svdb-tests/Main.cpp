// svdb-tests: unit tests for DSP extractor, CLAP mel frontend, RoBERTa BPE tokenizer,
// SQLite WAL database, AVX2 VectorIndex, and 2D UMAP Layout.

#include "core/Types.h"
#include "core/db/Layout2D.h"
#include "core/db/SampleDatabase.h"
#include "core/db/VectorIndex.h"
#include "core/dsp/DspExtractor.h"
#include "core/ml/ClapMelFrontend.h"
#include "core/ml/RobertaTokenizer.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace
{
int gFailures = 0;

void expect (bool cond, const char* msg)
{
    if (! cond)
    {
        std::cerr << "[FAIL] " << msg << "\n";
        ++gFailures;
    }
    else
    {
        std::cout << "[PASS] " << msg << "\n";
    }
}

void testDspExtractor()
{
    svdb::dsp::DspExtractor ext;
    constexpr double sr = 44100.0;
    constexpr double kPi = 3.14159265358979323846;

    // 1. 440 Hz sine (A4 = MIDI 69)
    svdb::dsp::DecodedAudio sine;
    sine.ok = true;
    sine.sampleRate = sr;
    sine.channels = 1;
    sine.fullDurationMs = 500.0f;
    sine.mono.resize (static_cast<size_t> (sr * 0.5));
    for (size_t i = 0; i < sine.mono.size(); ++i)
        sine.mono[i] = static_cast<float> (0.8 * std::sin (2.0 * kPi * 440.0 * static_cast<double> (i) / sr));

    const auto fSine = ext.extract (sine, "synth_A4.wav", "tonal");
    expect (fSine.midiNote == 69, "YIN detects 440 Hz sine as MIDI 69 (A4)");
    expect (fSine.pitchConfidence > 0.8f, "Pure sine has high pitch confidence");
    expect (! fSine.isLoop, "500ms sine classified as oneshot");

    // 2. White noise burst (hi-hat-like)
    svdb::dsp::DecodedAudio noise;
    noise.ok = true;
    noise.sampleRate = sr;
    noise.channels = 1;
    noise.fullDurationMs = 150.0f;
    noise.mono.resize (static_cast<size_t> (sr * 0.15));
    std::mt19937 rng (123);
    std::uniform_real_distribution<float> dist (-1.0f, 1.0f);
    for (size_t i = 0; i < noise.mono.size(); ++i)
        noise.mono[i] = dist (rng) * std::exp (-static_cast<float> (i) / 1800.0f);

    const auto fNoise = ext.extract (noise, "hat_01.wav", "hats");
    expect (fNoise.midiNote == -1, "White noise burst is unpitched");
    expect (fNoise.brightness > fSine.brightness, "White noise is brighter than 440 Hz sine");

    // 3. Cosine similarity: sine closer to slightly detuned sine than to noise.
    svdb::dsp::DecodedAudio sine2 = sine;
    for (size_t i = 0; i < sine2.mono.size(); ++i)
        sine2.mono[i] = static_cast<float> (0.8 * std::sin (2.0 * kPi * 466.16 * static_cast<double> (i) / sr));
    const auto fSine2 = ext.extract (sine2, "synth_A#4.wav", "tonal");

    const float simSS = svdb::db::VectorIndex::dotProduct (fSine.vec.data(), fSine2.vec.data(), svdb::kDspDim);
    const float simSN = svdb::db::VectorIndex::dotProduct (fSine.vec.data(), fNoise.vec.data(), svdb::kDspDim);
    expect (simSS > simSN + 0.2f, "DSP embedding separates tonal sines from noisy hats");

    // 4. Loop vs Oneshot classification on long FX/impacts vs rhythmic breaks
    std::array<int8_t, svdb::kWaveformPeaks * 2> decayPeaks {};
    for (int i = 0; i < svdb::kWaveformPeaks; ++i)
    {
        const int8_t v = static_cast<int8_t> (std::max (1, static_cast<int> (120.0f * std::exp (-0.06f * i))));
        decayPeaks[static_cast<size_t> (i * 2)] = -v;
        decayPeaks[static_cast<size_t> (i * 2 + 1)] = v;
    }
    expect (! svdb::dsp::DspExtractor::classifyIsLoop ("C:\\Samples\\Impacts\\Cymatics - Impact 29.wav",
                                                       "Cymatics - Impact 29.wav", 6500.0f, decayPeaks),
            "Long impact FX (6.5s) is classified as oneshot, not loop");
    expect (! svdb::dsp::DspExtractor::classifyIsLoop ("C:\\Samples\\Risers\\Cymatics - Riser 2 - 150 BPM.wav",
                                                       "Cymatics - Riser 2 - 150 BPM.wav", 12000.0f, decayPeaks),
            "Tempo-tagged riser FX is classified as oneshot, not loop");
    expect (svdb::dsp::DspExtractor::classifyIsLoop ("C:\\Samples\\dnb-breaks\\DNB_BREAK_101.wav",
                                                     "DNB_BREAK_101.wav", 2759.0f, decayPeaks),
            "Drum break is classified as loop");
}

void testClapMelFrontend()
{
    svdb::ml::ClapMelFrontend fe;
    std::vector<float> clip (4800, 0.0f); // 100ms 1kHz sine
    for (size_t i = 0; i < clip.size(); ++i)
        clip[i] = static_cast<float> (0.5 * std::sin (2.0 * 3.14159265358979323846 * 1000.0 * static_cast<double> (i) / 48000.0));

    std::vector<float> mel (svdb::ml::ClapMelFrontend::kFeatureCount, 0.0f);
    fe.compute (clip, mel.data());

    bool allFinite = true;
    float minVal = mel[0], maxVal = mel[0];
    for (float v : mel)
    {
        if (! std::isfinite (v)) allFinite = false;
        minVal = std::min (minVal, v);
        maxVal = std::max (maxVal, v);
    }
    expect (allFinite, "ClapMelFrontend produces all-finite 1001x64 log-mel features");
    expect (maxVal > minVal + 20.0f, "ClapMelFrontend exhibits strong dynamic range on 1kHz tone");
}

void testRobertaTokenizer()
{
    const auto tmp = std::filesystem::temp_directory_path() / "svdb_test_tok.json";
    {
        std::ofstream out (tmp);
        out << R"JSON({
            "model": {
                "vocab": {
                    "<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
                    "k": 10, "i": 11, "c": 12, "ki": 13, "ck": 14, "kick": 15,
                    "\u0120": 20, "d": 21, "r": 22, "u": 23, "m": 24,
                    "\u0120d": 25, "\u0120dr": 26, "\u0120dru": 27, "\u0120drum": 28
                },
                "merges": [
                    "k i", "c k", "ki ck",
                    "\u0120 d", "\u0120d r", "\u0120dr u", "\u0120dru m"
                ]
            }
        })JSON";
    }

    svdb::ml::RobertaTokenizer tok;
    std::string err;
    expect (tok.loadFromFile (tmp.string(), &err), "RobertaTokenizer loads tokenizer.json");
    const auto enc = tok.encode ("kick drum");
    expect (enc.inputIds.size() == 4 && enc.inputIds[0] == 0 && enc.inputIds[1] == 15
            && enc.inputIds[2] == 28 && enc.inputIds[3] == 2,
            "RobertaTokenizer merges 'kick drum' -> [0, 15, 28, 2]");
    std::filesystem::remove (tmp);
}

void testDbSearchAndLayout()
{
    const auto tmpDb = std::filesystem::temp_directory_path() / "svdb_unit_test.db";
    std::filesystem::remove (tmpDb);

    svdb::db::SampleDatabase db;
    std::string err;
    expect (db.open (tmpDb.string(), &err), "SQLite WAL database opens cleanly");

    std::vector<svdb::SampleRecord> recs;
    std::mt19937 rng (7);
    std::normal_distribution<float> nd (0.0f, 0.1f);

    // Build 2 clear clusters (kicks vs hats) of 16 samples each.
    for (int i = 0; i < 32; ++i)
    {
        svdb::SampleRecord r;
        const bool isKick = i < 16;
        r.meta.path = isKick ? ("C:\\samples\\kick_" + std::to_string (i) + ".wav")
                             : ("C:\\samples\\hat_" + std::to_string (i) + ".wav");
        r.meta.filename = isKick ? ("kick_" + std::to_string (i) + ".wav") : ("hat_" + std::to_string (i) + ".wav");
        r.meta.folder = isKick ? "kicks" : "hats";
        r.meta.fileSize = 10000 + i;
        r.meta.mtime = 1000;
        r.meta.durationMs = isKick ? 220.0f : 80.0f;
        r.meta.sampleRate = 44100;
        r.meta.channels = 1;
        r.meta.category = isKick ? svdb::Category::Kick : svdb::Category::ClosedHat;

        float normSq = 0.0f;
        for (int d = 0; d < svdb::kDspDim; ++d)
        {
            float base = (isKick ? (d < 16 ? 1.0f : 0.0f) : (d >= 32 && d < 48 ? 1.0f : 0.0f));
            r.dspVec[static_cast<size_t> (d)] = base + nd (rng);
            normSq += r.dspVec[static_cast<size_t> (d)] * r.dspVec[static_cast<size_t> (d)];
        }
        const float inv = 1.0f / std::sqrt (normSq);
        for (float& v : r.dspVec) v *= inv;
        recs.push_back (r);
    }

    expect (db.upsertDspBatch (recs), "SQLite upserts batch of 32 records");
    auto loaded = db.loadAll();
    expect (loaded.size() == 32, "SQLite reloads all 32 records");

    // AVX2 vs scalar dot product check.
    float scalar = 0.0f;
    for (int d = 0; d < svdb::kDspDim; ++d)
        scalar += loaded[0].dspVec[static_cast<size_t> (d)] * loaded[1].dspVec[static_cast<size_t> (d)];
    const float simd = svdb::db::VectorIndex::dotProduct (loaded[0].dspVec.data(), loaded[1].dspVec.data(), svdb::kDspDim);
    expect (std::abs (scalar - simd) < 1e-5f, "AVX2 dotProduct matches scalar reference within 1e-5");

    // Search similarity check: kick_0's top 10 neighbors should all be kicks.
    svdb::db::VectorIndex idx;
    idx.rebuildFromRecords (loaded);
    svdb::SearchQuery q;
    q.referenceId = loaded[0].meta.id;
    q.maxResults = 10;
    const auto res = idx.search (q);
    const auto snap = idx.getSnapshot();
    bool allKicks = (res.size() == 10);
    for (const auto& sr : res)
        if (snap->meta[static_cast<size_t> (sr.index)].category != svdb::Category::Kick)
            allKicks = false;
    expect (allKicks, "VectorIndex similarity query retrieves same-cluster neighbors");

    // 2D Multi-blend Layout check (DSP + Hybrid + CLAP aligned keyframes).
    for (size_t i = 0; i < loaded.size(); ++i)
    {
        loaded[i].meta.hasClap = true;
        const bool isKick = loaded[i].meta.category == svdb::Category::Kick;
        float ns = 0.0f;
        for (int d = 0; d < svdb::kClapDim; ++d)
        {
            float v = (isKick ? (d < 64 ? 1.0f : 0.0f) : (d >= 256 && d < 320 ? 1.0f : 0.0f)) + nd (rng);
            loaded[i].clapVec[static_cast<size_t> (d)] = v;
            ns += v * v;
        }
        const float inv = 1.0f / std::sqrt (ns);
        for (float& v : loaded[i].clapVec) v *= inv;
    }
    const auto coords = svdb::db::Layout2D::computeMultiBlend (loaded, 8, 60);
    bool validCoords = (coords.size() == 32);
    for (const auto& c : coords)
    {
        if (! std::isfinite (c.x) || ! std::isfinite (c.y)
            || ! std::isfinite (c.dspX) || ! std::isfinite (c.dspY)
            || ! std::isfinite (c.clapX) || ! std::isfinite (c.clapY)
            || std::abs (c.x) > 1.01f || std::abs (c.dspX) > 1.01f || std::abs (c.clapX) > 1.01f)
            validCoords = false;
    }
    expect (validCoords, "Layout2D::computeMultiBlend produces finite normalized DSP/Hybrid/CLAP coordinates");
    expect (db.updateLayoutBatch (coords), "SQLite persists multi-blend 2D keyframes");

    db.close();
    std::filesystem::remove (tmpDb);
    std::filesystem::remove (tmpDb.string() + "-wal");
    std::filesystem::remove (tmpDb.string() + "-shm");
}
} // namespace

int main()
{
    testDspExtractor();
    testClapMelFrontend();
    testRobertaTokenizer();
    testDbSearchAndLayout();

    if (gFailures == 0)
    {
        std::cout << "\nALL TESTS PASSED.\n";
        return 0;
    }
    std::cerr << "\n" << gFailures << " TEST(S) FAILED.\n";
    return 1;
}
