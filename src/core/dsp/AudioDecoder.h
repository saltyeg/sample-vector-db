#pragma once

#include <juce_audio_formats/juce_audio_formats.h>

#include <vector>

namespace svdb::dsp
{

struct DecodedAudio
{
    std::vector<float> mono;     // Mono mixdown (truncated to maxSeconds if long)
    float stereoWidth = 0.0f;    // Side / (Mid + Side) energy ratio in [0, 1]
    double sampleRate = 44100.0;
    int channels = 1;
    float fullDurationMs = 0.0f; // True duration of the file in ms
    bool ok = false;
};

class AudioDecoder
{
public:
    AudioDecoder();

    // Thread-safe: creates a local reader from registered formats.
    DecodedAudio decode (const juce::File& file, double maxSeconds = 12.0) const;

    // Linear resampling (fast, deterministic, adequate for feature extraction).
    static std::vector<float> resampleLinear (const std::vector<float>& input,
                                              double srcRate,
                                              double dstRate);

private:
    juce::AudioFormatManager formats;
};

} // namespace svdb::dsp
