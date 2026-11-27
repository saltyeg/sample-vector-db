#include "ClapMelFrontend.h"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <cmath>

namespace svdb::ml
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

double hzToHtkMel (double hz)
{
    return 2595.0 * std::log10 (1.0 + hz / 700.0);
}

double htkMelToHz (double mel)
{
    return 700.0 * (std::pow (10.0, mel / 2595.0) - 1.0);
}
} // namespace

ClapMelFrontend::ClapMelFrontend()
{
    // Periodic Hann window matching torch.hann_window(1024) / librosa hann(fftbins=True).
    for (int i = 0; i < kFftSize; ++i)
        hannWindow[static_cast<size_t> (i)] = static_cast<float> (0.5 * (1.0 - std::cos (2.0 * kPi * static_cast<double> (i) / static_cast<double> (kFftSize))));

    // HTK triangular mel filterbank [50 Hz .. 14000 Hz], matching Hugging Face
    // transformers.audio_utils.mel_filter_bank(..., mel_scale="htk", norm=None).
    melWeights.assign (static_cast<size_t> (kNumMels * kNumBins), 0.0f);

    const double melMin = hzToHtkMel (50.0);
    const double melMax = hzToHtkMel (14000.0);
    std::array<double, kNumMels + 2> filterFreqs {};
    for (int i = 0; i < kNumMels + 2; ++i)
    {
        const double m = melMin + (melMax - melMin) * static_cast<double> (i) / static_cast<double> (kNumMels + 1);
        filterFreqs[static_cast<size_t> (i)] = htkMelToHz (m);
    }

    std::array<double, kNumBins> fftFreqs {};
    for (int b = 0; b < kNumBins; ++b)
        fftFreqs[static_cast<size_t> (b)] = static_cast<double> (b) * static_cast<double> (kSampleRate) / static_cast<double> (kFftSize);

    for (int m = 0; m < kNumMels; ++m)
    {
        const double fLeft   = filterFreqs[static_cast<size_t> (m)];
        const double fCenter = filterFreqs[static_cast<size_t> (m + 1)];
        const double fRight  = filterFreqs[static_cast<size_t> (m + 2)];
        const double dLeft   = fCenter - fLeft;
        const double dRight  = fRight - fCenter;

        for (int b = 0; b < kNumBins; ++b)
        {
            const double f = fftFreqs[static_cast<size_t> (b)];
            const double downSlope = (dLeft > 0.0) ? ((f - fLeft) / dLeft) : 0.0;
            const double upSlope   = (dRight > 0.0) ? ((fRight - f) / dRight) : 0.0;
            const double w = std::max (0.0, std::min (downSlope, upSlope));
            melWeights[static_cast<size_t> (m * kNumBins + b)] = static_cast<float> (w);
        }
    }
}

void ClapMelFrontend::compute (const std::vector<float>& audio48k, float* out1001x64) const
{
    // 1. Repeatpad / truncate to 480,000 samples (matching ClapFeatureExtractor padding="repeatpad").
    //    For audio shorter than 480,000: tile whole copies as many times as fit, then zero-pad remainder.
    //    (Matches HF: n_repeat = nb_max_samples // len; tile(n_repeat) then pad right with zeros.)
    constexpr int kPadHalf = kFftSize / 2; // 512 center-padding
    std::vector<float> padded (static_cast<size_t> (kMaxSamples + 2 * kPadHalf), 0.0f);

    if (! audio48k.empty())
    {
        float* core = padded.data() + kPadHalf;
        const size_t srcLen = audio48k.size();
        if (srcLen >= static_cast<size_t> (kMaxSamples))
        {
            std::copy_n (audio48k.data(), kMaxSamples, core);
        }
        else
        {
            const size_t nRepeat = static_cast<size_t> (kMaxSamples) / srcLen;
            size_t dst = 0;
            for (size_t r = 0; r < nRepeat; ++r)
            {
                std::copy_n (audio48k.data(), srcLen, core + dst);
                dst += srcLen;
            }
            // Remaining tail stays 0.0f (exact HF repeatpad behavior).
        }

        // Reflect-pad both ends by kPadHalf (matching librosa/HF center=True, pad_mode="reflect").
        for (int i = 0; i < kPadHalf; ++i)
        {
            padded[static_cast<size_t> (kPadHalf - 1 - i)] = core[static_cast<size_t> (i + 1)];
            padded[static_cast<size_t> (kPadHalf + kMaxSamples + i)] = core[static_cast<size_t> (kMaxSamples - 2 - i)];
        }
    }

    // 2. STFT + Mel projection + 10*log10(max(x, 1e-10)).
    juce::dsp::FFT fft (kFftOrder);
    std::array<float, kFftSize * 2> fftWork {};
    std::array<float, kNumBins> power {};

    for (int frame = 0; frame < kNumFrames; ++frame)
    {
        const int start = frame * kHopLength;
        fftWork.fill (0.0f);
        for (int i = 0; i < kFftSize; ++i)
            fftWork[static_cast<size_t> (i)] = padded[static_cast<size_t> (start + i)] * hannWindow[static_cast<size_t> (i)];

        fft.performFrequencyOnlyForwardTransform (fftWork.data());
        for (int b = 0; b < kNumBins; ++b)
            power[static_cast<size_t> (b)] = fftWork[static_cast<size_t> (b)] * fftWork[static_cast<size_t> (b)];

        float* outRow = out1001x64 + static_cast<size_t> (frame * kNumMels);
        for (int m = 0; m < kNumMels; ++m)
        {
            const float* wRow = melWeights.data() + static_cast<size_t> (m * kNumBins);
            float acc = 0.0f;
            for (int b = 0; b < kNumBins; ++b)
                acc += wRow[b] * power[static_cast<size_t> (b)];
            outRow[m] = 10.0f * std::log10 (std::max (acc, 1e-10f));
        }
    }
}

} // namespace svdb::ml
