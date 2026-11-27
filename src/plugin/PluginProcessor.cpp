#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <nlohmann/json.hpp>

namespace svdb
{

SampleVectorDBProcessor::SampleVectorDBProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      indexer (database, vectorIndex, clapEncoder)
{
    openDatabase (defaultDbFile().getFullPathName().toStdString());

    // Lazy-load CLAP text encoder in background so opening the plugin in FL Studio is instant.
    const std::string modelDir = defaultModelDir().getFullPathName().toStdString();
    if (ml::ClapEncoder::modelsExistIn (modelDir))
    {
        modelWarmupThread = std::thread ([this, modelDir] {
            std::string err;
            clapEncoder.loadTextModelOnly (modelDir, &err);
        });
    }
}

SampleVectorDBProcessor::~SampleVectorDBProcessor()
{
    indexer.cancel();
    if (modelWarmupThread.joinable())
        modelWarmupThread.join();
}

bool SampleVectorDBProcessor::openDatabase (const std::string& dbPath)
{
    std::string err;
    if (! database.open (dbPath, &err))
        return false;
    vectorIndex.reloadFromDatabase (database);
    const auto folders = database.getFolders();
    if (! folders.empty() && lastSampleFolder.empty())
        lastSampleFolder = folders.front();
    indexRevision.fetch_add (1);
    return true;
}

void SampleVectorDBProcessor::prepareToPlay (double sampleRate, int /*samplesPerBlock*/)
{
    audition.prepare (sampleRate);
}

bool SampleVectorDBProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
}

void SampleVectorDBProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    audition.processBlock (buffer, midiMessages);
}

juce::AudioProcessorEditor* SampleVectorDBProcessor::createEditor()
{
    return new SampleVectorDBEditor (*this);
}

void SampleVectorDBProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    nlohmann::json j;
    j["clapBlend"] = uiClapBlend;
    j["lengthFilter"] = static_cast<int> (uiLengthFilter);
    j["categoryFilter"] = uiCategoryFilter;
    j["gain"] = audition.getGain();
    j["chromaticPitch"] = audition.isChromaticPitch();
    j["lastFolder"] = lastSampleFolder;
    const std::string s = j.dump();
    destData.replaceAll (s.data(), s.size());
}

void SampleVectorDBProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (! data || sizeInBytes <= 0) return;
    try
    {
        const std::string s (static_cast<const char*> (data), static_cast<size_t> (sizeInBytes));
        const auto j = nlohmann::json::parse (s);
        if (j.contains ("clapBlend"))      uiClapBlend = j["clapBlend"].get<float>();
        if (j.contains ("lengthFilter"))   uiLengthFilter = static_cast<LengthFilter> (j["lengthFilter"].get<int>());
        if (j.contains ("categoryFilter")) uiCategoryFilter = j["categoryFilter"].get<int>();
        if (j.contains ("gain"))           audition.setGain (j["gain"].get<float>());
        if (j.contains ("chromaticPitch")) audition.setChromaticPitch (j["chromaticPitch"].get<bool>());
        if (j.contains ("lastFolder"))     lastSampleFolder = j["lastFolder"].get<std::string>();
    }
    catch (...) {}
}

std::vector<SearchResult> SampleVectorDBProcessor::runSearch (SearchQuery query)
{
    if (! query.text.empty() && query.textClapVec.empty() && clapEncoder.isTextReady())
    {
        std::array<float, kClapDim> emb {};
        if (clapEncoder.encodeText (query.text, emb))
            query.textClapVec.assign (emb.begin(), emb.end());
    }
    return vectorIndex.search (query);
}

bool SampleVectorDBProcessor::encodeExternalAudio (const juce::File& file,
                                                   std::vector<float>& outDsp,
                                                   std::vector<float>& outClap)
{
    outDsp.clear();
    outClap.clear();

    const auto dec = externalDecoder.decode (file, 12.0);
    if (! dec.ok)
        return false;

    const auto feat = externalDsp.extract (dec, file.getFileName().toStdString(), file.getParentDirectory().getFileName().toStdString());
    outDsp.assign (feat.vec.begin(), feat.vec.end());

    const std::string mDir = defaultModelDir().getFullPathName().toStdString();
    if (! clapEncoder.isAudioReady() && ml::ClapEncoder::modelsExistIn (mDir))
        clapEncoder.loadAudioModelOnly (mDir, true);

    if (clapEncoder.isAudioReady())
    {
        std::vector<std::vector<float>> batch;
        batch.push_back (dsp::AudioDecoder::resampleLinear (dec.mono, dec.sampleRate, 48000.0));
        std::vector<std::array<float, kClapDim>> embeds;
        if (clapEncoder.encodeAudioBatch (batch, embeds) && ! embeds.empty())
            outClap.assign (embeds[0].begin(), embeds[0].end());
    }
    return true;
}

void SampleVectorDBProcessor::startIndexingFolder (const std::string& folderPath)
{
    lastSampleFolder = folderPath;
    Indexer::Options opts;
    opts.runClap = true;
    opts.computeLayout = true;
    opts.preferDirectML = true;
    opts.modelDir = defaultModelDir().getFullPathName().toStdString();

    indexer.startAsync (folderPath, opts, [this] {
        vectorIndex.reloadFromDatabase (database);
        indexRevision.fetch_add (1);
    });
}

std::string SampleVectorDBProcessor::getDefaultSampleFolder() const
{
    if (! lastSampleFolder.empty())
        return lastSampleFolder;
    const auto docs = juce::File::getSpecialLocation (juce::File::userDocumentsDirectory).getChildFile ("Samples");
    if (docs.isDirectory())
        return docs.getFullPathName().toStdString();
    return juce::File::getSpecialLocation (juce::File::userDocumentsDirectory).getFullPathName().toStdString();
}

std::string SampleVectorDBProcessor::getStatusSummary() const
{
    if (indexer.isRunning())
        return indexer.getProgress().message;
    const auto snap = vectorIndex.getSnapshot();
    std::string s = std::to_string (snap->count()) + " samples";
    if (snap->clapCount > 0)
        s += " | CLAP";
    else
        s += " | DSP";
    return s;
}

} // namespace svdb

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new svdb::SampleVectorDBProcessor();
}
