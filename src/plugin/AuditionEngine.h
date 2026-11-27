#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace svdb
{

struct LoadedSample
{
    std::string path;
    juce::AudioBuffer<float> buffer;
    double sampleRate = 44100.0;
};

// Lock-free 4-voice oneshot audition engine playable via UI click or FL Studio MIDI.
// MIDI note 60 (FL Studio "C5") plays at original pitch (1.0x).
class AuditionEngine
{
public:
    AuditionEngine();
    ~AuditionEngine();

    void prepare (double hostSampleRate);
    void processBlock (juce::AudioBuffer<float>& out, const juce::MidiBuffer& midi);

    // Loads file asynchronously on a background thread and optionally triggers preview on ready.
    void loadSample (const std::string& filePath, bool autoPlay);
    void triggerPreview();
    void triggerPreviewAt (float normStartPos);
    void stopAll();

    void setGain (float g) { gain.store (g); }
    float getGain() const { return gain.load(); }

    void setChromaticPitch (bool enabled) { chromaticPitch.store (enabled); }
    bool isChromaticPitch() const { return chromaticPitch.load(); }

    // Normalized [0, 1] playhead of the most recent active voice, or -1 if silent.
    float getPlayheadNormalized() const { return playheadNorm.load(); }
    bool isPlaying() const { return playheadNorm.load() >= 0.0f; }

private:
    struct Voice
    {
        std::shared_ptr<const LoadedSample> sample;
        double pos = 0.0;
        double step = 1.0;
        float velocity = 1.0f;
        bool active = false;
        bool isPreview = false;
    };

    static constexpr int kMaxVoices = 4;

    juce::AudioFormatManager formats;
    double hostSr = 44100.0;

    std::atomic<float> gain { 0.85f };
    std::atomic<bool> chromaticPitch { false };
    std::atomic<float> playheadNorm { -1.0f };
    std::atomic<uint32_t> pendingPreviewCount { 0 };
    std::atomic<float> pendingPreviewStartNorm { 0.0f };
    std::atomic<bool> pendingStop { false };

    juce::SpinLock sampleLock;
    std::shared_ptr<const LoadedSample> currentSample;
    std::shared_ptr<const LoadedSample> retiredSample; // released on loader thread, never audio thread

    std::array<Voice, kMaxVoices> voices {};
    int nextVoice = 0;

    // Background loader thread.
    std::mutex reqMtx;
    std::condition_variable reqCv;
    std::string requestedPath;
    bool requestedAutoPlay = false;
    bool hasRequest = false;
    bool stopping = false;
    std::thread loaderThread;

    void loaderLoop();
    void startVoice (const std::shared_ptr<const LoadedSample>& s, int midiNote, float vel, bool isPreview, float startNorm = 0.0f);
};

} // namespace svdb
