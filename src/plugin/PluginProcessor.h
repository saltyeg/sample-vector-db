#pragma once

#include "AuditionEngine.h"
#include "core/Indexer.h"
#include "core/Types.h"
#include "core/db/SampleDatabase.h"
#include "core/db/VectorIndex.h"
#include "core/dsp/AudioDecoder.h"
#include "core/dsp/DspExtractor.h"
#include "core/ml/ClapEncoder.h"

#include <juce_audio_processors/juce_audio_processors.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace svdb
{

class SampleVectorDBProcessor final : public juce::AudioProcessor
{
public:
    SampleVectorDBProcessor();
    ~SampleVectorDBProcessor() override;

    // AudioProcessor interface
    const juce::String getName() const override { return "sample-vector-db"; }
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // Library & search helpers for UI / tools
    bool openDatabase (const std::string& dbPath);
    db::VectorIndex& getVectorIndex() { return vectorIndex; }
    db::SampleDatabase& getDatabase() { return database; }
    AuditionEngine& getAuditionEngine() { return audition; }
    Indexer& getIndexer() { return indexer; }
    ml::ClapEncoder& getClapEncoder() { return clapEncoder; }

    // Encodes text with CLAP (if loaded) and executes search.
    std::vector<SearchResult> runSearch (SearchQuery query);

    // Extracts DSP (+ CLAP if ready) vector from an arbitrary dropped audio file.
    bool encodeExternalAudio (const juce::File& file,
                              std::vector<float>& outDsp,
                              std::vector<float>& outClap);

    void startIndexingFolder (const std::string& folderPath);
    std::string getDefaultSampleFolder() const;
    std::string getStatusSummary() const;

    // Monotonic counter bumped whenever the in-memory VectorIndex reloads.
    uint64_t getIndexRevision() const { return indexRevision.load(); }

    // Persisted UI settings
    float uiClapBlend = 0.55f;
    LengthFilter uiLengthFilter = LengthFilter::Oneshots;
    int uiCategoryFilter = -1;
    std::string lastSampleFolder;

private:
    db::SampleDatabase database;
    db::VectorIndex vectorIndex;
    ml::ClapEncoder clapEncoder;
    Indexer indexer;
    AuditionEngine audition;
    dsp::AudioDecoder externalDecoder;
    dsp::DspExtractor externalDsp;

    std::atomic<uint64_t> indexRevision { 0 };
    std::thread modelWarmupThread;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SampleVectorDBProcessor)
};

} // namespace svdb
