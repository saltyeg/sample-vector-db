#pragma once

#include "AudioDecoder.h"
#include "core/Types.h"

#include <array>
#include <string>

namespace svdb::dsp
{

struct DspFeatures
{
    std::array<float, kDspDim> vec {};
    std::array<int8_t, kWaveformPeaks * 2> peaks {};
    int midiNote = -1;
    float pitchHz = 0.0f;
    float pitchConfidence = 0.0f;
    float brightness = 0.5f;
    float punch = 0.5f;
    bool isLoop = false;
    Category heuristicCategory = Category::Other;
};

class DspExtractor
{
public:
    DspExtractor();

    DspFeatures extract (const DecodedAudio& audio,
                         const std::string& filename = {},
                         const std::string& folder = {},
                         const std::string& fullPath = {}) const;

    // Classifies loop vs oneshot using word-bounded path tokens + 96-bin envelope shape.
    static bool classifyIsLoop (const std::string& fullPath,
                                const std::string& filename,
                                float durationMs,
                                const std::array<int8_t, kWaveformPeaks * 2>& peaks);

private:
    static constexpr int kWorkSr = 44100;
    static constexpr int kFftOrder = 11;          // 2048-point FFT
    static constexpr int kFftSize = 1 << kFftOrder;
    static constexpr int kNumBins = kFftSize / 2 + 1;
    static constexpr int kMelBands = 16;

    std::array<float, kFftSize> hannWindow {};
    // Dense mel weights: kMelBands * kNumBins
    std::vector<float> melFilterbank;
    std::array<float, kNumBins> binFreqs {};

    void computeSpectrum (const float* samples, int count, float* magOut) const;
    void estimatePitchYin (const float* samples, int count, float& outHz, float& outConf) const;
    static Category classifyHeuristic (float durationMs,
                                       bool isLoop,
                                       float subRatio,
                                       float lowRatio,
                                       float midRatio,
                                       float highMidRatio,
                                       float airRatio,
                                       float flatness,
                                       float crest,
                                       float centroidHz,
                                       float pitchConf,
                                       const std::string& filename,
                                       const std::string& folder);
};

} // namespace svdb::dsp
