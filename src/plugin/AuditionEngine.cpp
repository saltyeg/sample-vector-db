#include "AuditionEngine.h"

#include <algorithm>
#include <cmath>

namespace svdb
{

AuditionEngine::AuditionEngine()
{
    formats.registerBasicFormats();
    loaderThread = std::thread ([this] { loaderLoop(); });
}

AuditionEngine::~AuditionEngine()
{
    {
        std::lock_guard<std::mutex> lock (reqMtx);
        stopping = true;
    }
    reqCv.notify_all();
    if (loaderThread.joinable())
        loaderThread.join();
}

void AuditionEngine::prepare (double hostSampleRate)
{
    hostSr = (hostSampleRate > 0.0) ? hostSampleRate : 44100.0;
}

void AuditionEngine::loadSample (const std::string& filePath, bool autoPlay)
{
    // If already loaded, just trigger immediately with zero disk I/O.
    {
        const juce::SpinLock::ScopedLockType sl (sampleLock);
        if (currentSample && currentSample->path == filePath)
        {
            if (autoPlay)
                pendingPreviewCount.fetch_add (1);
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lock (reqMtx);
        requestedPath = filePath;
        requestedAutoPlay = autoPlay;
        hasRequest = true;
    }
    reqCv.notify_one();
}

void AuditionEngine::triggerPreview()
{
    triggerPreviewAt (0.0f);
}

void AuditionEngine::triggerPreviewAt (float normStartPos)
{
    pendingPreviewStartNorm.store (std::clamp (normStartPos, 0.0f, 0.99f));
    pendingPreviewCount.fetch_add (1);
}

void AuditionEngine::stopAll()
{
    pendingStop.store (true);
}

void AuditionEngine::loaderLoop()
{
    while (true)
    {
        std::string pathToLoad;
        bool autoPlay = false;

        {
            std::unique_lock<std::mutex> lock (reqMtx);
            reqCv.wait (lock, [this] { return stopping || hasRequest; });
            if (stopping)
                return;
            pathToLoad = std::move (requestedPath);
            autoPlay = requestedAutoPlay;
            hasRequest = false;
        }

        // Release any previously retired sample safely on this non-audio thread.
        std::shared_ptr<const LoadedSample> toFree;
        {
            const juce::SpinLock::ScopedLockType sl (sampleLock);
            toFree = std::move (retiredSample);
        }
        toFree.reset();

        const juce::File f (juce::String::fromUTF8 (pathToLoad.c_str()));
        if (! f.existsAsFile())
            continue;

        std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (f));
        if (! reader || reader->sampleRate <= 0.0 || reader->lengthInSamples <= 0)
            continue;

        // Cap audition buffer to 30 seconds to keep RAM bounded even on huge stems.
        const int64_t maxLen = static_cast<int64_t> (reader->sampleRate * 30.0);
        const int numSamples = static_cast<int> (std::min<int64_t> (reader->lengthInSamples, maxLen));
        const int ch = std::clamp (static_cast<int> (reader->numChannels), 1, 2);

        auto loaded = std::make_shared<LoadedSample>();
        loaded->path = pathToLoad;
        loaded->sampleRate = reader->sampleRate;
        loaded->buffer.setSize (2, numSamples);
        loaded->buffer.clear();

        reader->read (&loaded->buffer, 0, numSamples, 0, true, ch > 1);
        if (ch == 1)
            loaded->buffer.copyFrom (1, 0, loaded->buffer, 0, 0, numSamples);

        {
            const juce::SpinLock::ScopedLockType sl (sampleLock);
            retiredSample = std::move (currentSample);
            currentSample = std::move (loaded);
        }

        if (autoPlay)
        {
            pendingPreviewStartNorm.store (0.0f);
            pendingPreviewCount.fetch_add (1);
        }
    }
}

void AuditionEngine::startVoice (const std::shared_ptr<const LoadedSample>& s,
                                 int midiNote,
                                 float vel,
                                 bool isPreview,
                                 float startNorm)
{
    if (! s || s->buffer.getNumSamples() <= 0)
        return;

    // Preview clicks/scrubs strictly allow only one sample playing at a time.
    int slot = -1;
    if (isPreview)
    {
        for (auto& voice : voices)
            voice.active = false;
        slot = 0;
    }
    if (slot < 0)
    {
        for (int i = 0; i < kMaxVoices; ++i)
        {
            if (! voices[static_cast<size_t> (i)].active)
            {
                slot = i;
                break;
            }
        }
    }
    if (slot < 0)
    {
        slot = nextVoice;
        nextVoice = (nextVoice + 1) % kMaxVoices;
    }

    const double pitchRatio = chromaticPitch.load()
        ? std::pow (2.0, static_cast<double> (midiNote - 60) / 12.0)
        : 1.0;

    const int maxStart = std::max (0, s->buffer.getNumSamples() - 2);
    auto& v = voices[static_cast<size_t> (slot)];
    v.sample = s;
    v.pos = std::clamp (static_cast<double> (startNorm), 0.0, 0.99) * static_cast<double> (maxStart);
    v.step = (s->sampleRate / hostSr) * pitchRatio;
    v.velocity = std::clamp (vel, 0.0f, 1.0f);
    v.active = true;
    v.isPreview = isPreview;
}

void AuditionEngine::processBlock (juce::AudioBuffer<float>& out, const juce::MidiBuffer& midi)
{
    out.clear();
    const int numSamples = out.getNumSamples();
    const int numCh = out.getNumChannels();
    if (numSamples <= 0 || numCh <= 0)
        return;

    if (pendingStop.exchange (false))
    {
        for (auto& v : voices)
            v.active = false;
    }

    std::shared_ptr<const LoadedSample> snap;
    if (sampleLock.tryEnter())
    {
        snap = currentSample;
        sampleLock.exit();
    }

    if (pendingPreviewCount.exchange (0) > 0 && snap)
        startVoice (snap, 60, 1.0f, true, pendingPreviewStartNorm.load());

    for (const auto metadata : midi)
    {
        const auto msg = metadata.getMessage();
        if (msg.isNoteOn() && snap)
            startVoice (snap, msg.getNoteNumber(), msg.getFloatVelocity(), false);
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
        {
            for (auto& v : voices)
                v.active = false;
        }
    }

    const float g = gain.load();
    float latestNorm = -1.0f;

    float* outL = out.getWritePointer (0);
    float* outR = (numCh > 1) ? out.getWritePointer (1) : nullptr;

    for (auto& v : voices)
    {
        if (! v.active || ! v.sample)
            continue;

        const auto& buf = v.sample->buffer;
        const int len = buf.getNumSamples();
        const float* srcL = buf.getReadPointer (0);
        const float* srcR = buf.getReadPointer (1);
        const float amp = g * v.velocity;

        for (int i = 0; i < numSamples; ++i)
        {
            const int idx = static_cast<int> (v.pos);
            if (idx >= len - 1)
            {
                v.active = false;
                break;
            }
            const float frac = static_cast<float> (v.pos - static_cast<double> (idx));
            const float sL = (srcL[idx] + frac * (srcL[idx + 1] - srcL[idx])) * amp;
            const float sR = (srcR[idx] + frac * (srcR[idx + 1] - srcR[idx])) * amp;

            outL[i] += sL;
            if (outR) outR[i] += sR;

            v.pos += v.step;
        }

        if (v.active && len > 0)
            latestNorm = std::clamp (static_cast<float> (v.pos / static_cast<double> (len)), 0.0f, 1.0f);
    }

    playheadNorm.store (latestNorm);
}

} // namespace svdb
