#pragma once

#include "core/Types.h"
#include "core/db/SampleDatabase.h"
#include "core/db/VectorIndex.h"
#include "core/dsp/AudioDecoder.h"
#include "core/dsp/DspExtractor.h"
#include "core/ml/ClapEncoder.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace svdb
{

class Indexer
{
public:
    Indexer (db::SampleDatabase& database, db::VectorIndex& index, ml::ClapEncoder& clap);
    ~Indexer();

    struct Options
    {
        bool runClap = true;
        bool computeLayout = true;
        bool preferDirectML = true;
        std::string modelDir;
    };

    // Synchronous run (used by CLI and background worker thread).
    void runSync (const std::string& folderPath,
                  const Options& opts,
                  std::function<void (const IndexProgress&)> onProgress = nullptr);

    // Asynchronous run (used by VST3/Standalone UI).
    void startAsync (const std::string& folderPath,
                     const Options& opts,
                     std::function<void()> onIndexUpdated = nullptr);
    void cancel();
    bool isRunning() const { return running.load(); }
    IndexProgress getProgress() const;

private:
    db::SampleDatabase& db;
    db::VectorIndex& vectorIndex;
    ml::ClapEncoder& clapEncoder;
    dsp::AudioDecoder decoder;
    dsp::DspExtractor dspExtractor;

    std::atomic<bool> running { false };
    std::atomic<bool> cancelFlag { false };
    std::thread worker;
    mutable std::mutex progMtx;
    IndexProgress progress;

    void setProgress (IndexPhase phase, int done, int total, const std::string& msg,
                      const std::function<void (const IndexProgress&)>& cb);
};

} // namespace svdb
