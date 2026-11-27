#include "AudioDecoder.h"

#include <algorithm>
#include <cmath>

namespace svdb::dsp
{

AudioDecoder::AudioDecoder()
{
    formats.registerBasicFormats();
}

DecodedAudio AudioDecoder::decode (const juce::File& file, double maxSeconds) const
{
    DecodedAudio out;
    if (! file.existsAsFile())
        return out;

    // Create a local AudioFormatManager-compatible reader via createReaderFor.
    // AudioFormatManager::createReaderFor is safe to call concurrently once formats are registered.
    std::unique_ptr<juce::AudioFormatReader> reader (const_cast<juce::AudioFormatManager&> (formats).createReaderFor (file));
    if (! reader || reader->sampleRate <= 0.0 || reader->lengthInSamples <= 0)
        return out;

    out.sampleRate = reader->sampleRate;
    out.channels = static_cast<int> (reader->numChannels);
    out.fullDurationMs = static_cast<float> ((static_cast<double> (reader->lengthInSamples) / reader->sampleRate) * 1000.0);

    const int64_t maxSamples = static_cast<int64_t> (std::ceil (maxSeconds * reader->sampleRate));
    const int numSamples = static_cast<int> (std::min<int64_t> (reader->lengthInSamples, maxSamples));
    if (numSamples <= 0)
        return out;

    const int readCh = std::min (out.channels, 2);
    juce::AudioBuffer<float> buf (readCh, numSamples);
    buf.clear();
    if (! reader->read (&buf, 0, numSamples, 0, true, readCh > 1))
        return out;

    out.mono.resize (static_cast<size_t> (numSamples));
    if (readCh == 1)
    {
        const float* ch0 = buf.getReadPointer (0);
        std::copy (ch0, ch0 + numSamples, out.mono.begin());
        out.stereoWidth = 0.0f;
    }
    else
    {
        const float* L = buf.getReadPointer (0);
        const float* R = buf.getReadPointer (1);
        double midEnergy = 0.0;
        double sideEnergy = 0.0;
        for (int i = 0; i < numSamples; ++i)
        {
            const float m = 0.5f * (L[i] + R[i]);
            const float s = 0.5f * (L[i] - R[i]);
            out.mono[static_cast<size_t> (i)] = m;
            midEnergy += static_cast<double> (m) * m;
            sideEnergy += static_cast<double> (s) * s;
        }
        const double denom = midEnergy + sideEnergy + 1e-12;
        out.stereoWidth = static_cast<float> (std::sqrt (sideEnergy / denom));
    }

    out.ok = true;
    return out;
}

std::vector<float> AudioDecoder::resampleLinear (const std::vector<float>& input,
                                                 double srcRate,
                                                 double dstRate)
{
    if (input.empty() || srcRate <= 0.0 || dstRate <= 0.0)
        return {};
    if (std::abs (srcRate - dstRate) < 1.0)
        return input;

    const double ratio = srcRate / dstRate;
    const size_t outLen = std::max<size_t> (1, static_cast<size_t> (std::llround (static_cast<double> (input.size()) / ratio)));
    std::vector<float> out (outLen);
    const size_t last = input.size() - 1;

    for (size_t i = 0; i < outLen; ++i)
    {
        const double pos = static_cast<double> (i) * ratio;
        const size_t idx = static_cast<size_t> (pos);
        if (idx >= last)
        {
            out[i] = input[last];
        }
        else
        {
            const float frac = static_cast<float> (pos - static_cast<double> (idx));
            out[i] = input[idx] + frac * (input[idx + 1] - input[idx]);
        }
    }
    return out;
}

} // namespace svdb::dsp
