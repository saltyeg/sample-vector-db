#pragma once

#include <array>
#include <vector>

namespace svdb::ml
{

// Exact C++ port of Hugging Face ClapFeatureExtractor for Xenova/clap-htsat-unfused:
//   sampling_rate = 48000, n_fft = 1024, hop_length = 480,
//   nb_max_samples = 480000 (10s), frames = 1001, mel_bins = 64,
//   frequency_min = 50, frequency_max = 14000, HTK mel scale, power = 2.0,
//   10 * log10(max(x, 1e-10)), repeatpad padding.
class ClapMelFrontend
{
public:
    static constexpr int kSampleRate = 48000;
    static constexpr int kMaxSamples = 480000;
    static constexpr int kFftOrder = 10;       // 1024
    static constexpr int kFftSize = 1024;
    static constexpr int kHopLength = 480;
    static constexpr int kNumBins = kFftSize / 2 + 1; // 513
    static constexpr int kNumFrames = 1001;
    static constexpr int kNumMels = 64;
    static constexpr size_t kFeatureCount = static_cast<size_t> (kNumFrames * kNumMels);

    ClapMelFrontend();

    // Computes [1, 1001, 64] flattened row-major (frame-major) float32 features.
    // Input must already be mono at 48 kHz.
    void compute (const std::vector<float>& audio48k, float* out1001x64) const;

private:
    std::array<float, kFftSize> hannWindow {};
    // Dense [kNumMels * kNumBins] HTK mel filterbank
    std::vector<float> melWeights;
};

} // namespace svdb::ml
