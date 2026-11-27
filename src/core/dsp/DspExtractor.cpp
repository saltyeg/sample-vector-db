#include "DspExtractor.h"

#include <juce_dsp/juce_dsp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <numeric>

namespace svdb::dsp
{
namespace
{
constexpr float kPi = 3.14159265358979323846f;

float hzToMel (float hz)
{
    return 2595.0f * std::log10 (1.0f + hz / 700.0f);
}

float melToHz (float mel)
{
    return 700.0f * (std::pow (10.0f, mel / 2595.0f) - 1.0f);
}

std::string toLowerAscii (const std::string& s)
{
    std::string out = s;
    for (char& c : out)
        c = static_cast<char> (std::tolower (static_cast<unsigned char> (c)));
    return out;
}

bool containsWordOrSub (const std::string& hay, const char* needle)
{
    return hay.find (needle) != std::string::npos;
}

bool hasWord (const std::string& hay, const char* word)
{
    const size_t wlen = std::strlen (word);
    if (wlen == 0) return false;
    auto isAlnum = [] (char c) { return std::isalnum (static_cast<unsigned char> (c)) != 0; };
    size_t pos = hay.find (word, 0);
    while (pos != std::string::npos)
    {
        const bool leftOk  = (pos == 0) || ! isAlnum (hay[pos - 1]);
        const bool rightOk = (pos + wlen >= hay.size()) || ! isAlnum (hay[pos + wlen]);
        if (leftOk && rightOk)
            return true;
        pos = hay.find (word, pos + 1);
    }
    return false;
}

bool hasAnyWord (const std::string& hay, std::initializer_list<const char*> words)
{
    for (const char* w : words)
        if (hasWord (hay, w))
            return true;
    return false;
}

float clamp01 (float x)
{
    return std::clamp (x, 0.0f, 1.0f);
}

} // namespace

DspExtractor::DspExtractor()
{
    for (int i = 0; i < kFftSize; ++i)
        hannWindow[static_cast<size_t> (i)] = 0.5f * (1.0f - std::cos (2.0f * kPi * static_cast<float> (i) / static_cast<float> (kFftSize)));

    for (int b = 0; b < kNumBins; ++b)
        binFreqs[static_cast<size_t> (b)] = static_cast<float> (b) * static_cast<float> (kWorkSr) / static_cast<float> (kFftSize);

    // Build 16-band triangular Mel filterbank from 30 Hz to 18000 Hz.
    melFilterbank.assign (static_cast<size_t> (kMelBands * kNumBins), 0.0f);
    const float melMin = hzToMel (30.0f);
    const float melMax = hzToMel (18000.0f);
    std::array<float, kMelBands + 2> edges {};
    for (int m = 0; m < kMelBands + 2; ++m)
        edges[static_cast<size_t> (m)] = melToHz (melMin + (melMax - melMin) * static_cast<float> (m) / static_cast<float> (kMelBands + 1));

    for (int m = 0; m < kMelBands; ++m)
    {
        const float fL = edges[static_cast<size_t> (m)];
        const float fC = edges[static_cast<size_t> (m + 1)];
        const float fR = edges[static_cast<size_t> (m + 2)];
        float sum = 0.0f;
        for (int b = 0; b < kNumBins; ++b)
        {
            const float f = binFreqs[static_cast<size_t> (b)];
            float w = 0.0f;
            if (f >= fL && f <= fC && fC > fL)
                w = (f - fL) / (fC - fL);
            else if (f > fC && f <= fR && fR > fC)
                w = (fR - f) / (fR - fC);
            melFilterbank[static_cast<size_t> (m * kNumBins + b)] = w;
            sum += w;
        }
        if (sum > 1e-8f)
        {
            for (int b = 0; b < kNumBins; ++b)
                melFilterbank[static_cast<size_t> (m * kNumBins + b)] /= sum;
        }
    }
}

void DspExtractor::computeSpectrum (const float* samples, int count, float* magOut) const
{
    std::fill (magOut, magOut + kNumBins, 0.0f);
    if (count <= 0)
        return;

    juce::dsp::FFT fft (kFftOrder);
    std::array<float, kFftSize * 2> work {};

    const int hop = kFftSize / 2;
    int frames = 0;
    for (int start = 0; start < count; start += hop)
    {
        work.fill (0.0f);
        const int n = std::min (kFftSize, count - start);
        for (int i = 0; i < n; ++i)
            work[static_cast<size_t> (i)] = samples[start + i] * hannWindow[static_cast<size_t> (i)];

        fft.performFrequencyOnlyForwardTransform (work.data());
        for (int b = 0; b < kNumBins; ++b)
            magOut[b] += work[static_cast<size_t> (b)] * work[static_cast<size_t> (b)];

        ++frames;
        if (start + kFftSize >= count)
            break;
    }

    if (frames > 0)
    {
        const float inv = 1.0f / static_cast<float> (frames);
        for (int b = 0; b < kNumBins; ++b)
            magOut[b] *= inv;
    }
}

void DspExtractor::estimatePitchYin (const float* samples, int count, float& outHz, float& outConf) const
{
    outHz = 0.0f;
    outConf = 0.0f;

    // YIN over up to 4096 samples (~93 ms) targeting 32 Hz .. 2000 Hz.
    const int minLag = kWorkSr / 2000; // ~22
    const int maxLag = kWorkSr / 32;   // ~1378
    const int win = std::min (count, 4096);
    if (win < maxLag + 64)
        return;

    const int half = std::min (win / 2, maxLag);
    if (half <= minLag + 4)
        return;

    std::vector<float> diff (static_cast<size_t> (half), 0.0f);
    for (int tau = 1; tau < half; ++tau)
    {
        double sum = 0.0;
        const int limit = win - half;
        for (int i = 0; i < limit; ++i)
        {
            const float d = samples[i] - samples[i + tau];
            sum += static_cast<double> (d) * d;
        }
        diff[static_cast<size_t> (tau)] = static_cast<float> (sum);
    }

    // Cumulative mean normalized difference function (CMNDF).
    std::vector<float> cmnd (static_cast<size_t> (half), 1.0f);
    double runningSum = 0.0;
    for (int tau = 1; tau < half; ++tau)
    {
        runningSum += diff[static_cast<size_t> (tau)];
        cmnd[static_cast<size_t> (tau)] = (runningSum > 1e-12)
            ? static_cast<float> (diff[static_cast<size_t> (tau)] * static_cast<double> (tau) / runningSum)
            : 1.0f;
    }

    // Find first local minimum below threshold 0.25, else global min.
    int bestTau = -1;
    float bestVal = 1.0f;
    for (int tau = minLag; tau < half - 1; ++tau)
    {
        const float v = cmnd[static_cast<size_t> (tau)];
        if (v < 0.25f && v <= cmnd[static_cast<size_t> (tau - 1)] && v <= cmnd[static_cast<size_t> (tau + 1)])
        {
            bestTau = tau;
            bestVal = v;
            break;
        }
        if (v < bestVal)
        {
            bestVal = v;
            bestTau = tau;
        }
    }

    if (bestTau <= minLag || bestTau >= half - 1)
        return;

    // Parabolic interpolation around bestTau.
    const float s0 = cmnd[static_cast<size_t> (bestTau - 1)];
    const float s1 = cmnd[static_cast<size_t> (bestTau)];
    const float s2 = cmnd[static_cast<size_t> (bestTau + 1)];
    const float denom = 2.0f * (s0 - 2.0f * s1 + s2);
    float betterTau = static_cast<float> (bestTau);
    if (std::abs (denom) > 1e-8f)
        betterTau += std::clamp ((s0 - s2) / denom, -0.5f, 0.5f);

    outHz = static_cast<float> (kWorkSr) / std::max (1.0f, betterTau);
    outConf = clamp01 (1.0f - bestVal);
}

Category DspExtractor::classifyHeuristic (float durationMs,
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
                                          const std::string& folder)
{
    // Word-bounded filename hints first (overridden by CLAP in Phase 2).
    const std::string name = toLowerAscii (filename);
    const std::string dir  = toLowerAscii (folder);
    const std::string both = name + " " + dir;

    if (hasAnyWord (both, { "808", "808s" }))                                                return Category::Bass808;
    if (hasAnyWord (both, { "kick", "kicks", "kck", "bd", "bassdrum" }))                     return Category::Kick;
    if (hasAnyWord (both, { "rim", "rims", "rimshot", "rimshots", "sidestick" }))            return Category::Rim;
    if (hasAnyWord (both, { "snare", "snares", "snr", "sd" }))                               return Category::Snare;
    if (hasAnyWord (both, { "clap", "claps", "clp", "snap", "snaps" }))                      return Category::Clap;
    if (hasAnyWord (both, { "open", "oh", "ohh" }) && (containsWordOrSub (both, "hat") || hasAnyWord (both, { "oh", "ohh" })))
        return Category::OpenHat;
    if (hasAnyWord (both, { "closed", "ch", "chh", "hh", "hihat", "hihats", "hat", "hats" })) return Category::ClosedHat;
    if (hasAnyWord (both, { "crash", "crashes", "ride", "rides", "cymbal", "cymbals", "cym" })) return Category::Cymbal;
    if (hasAnyWord (both, { "tom", "toms" }))                                                return Category::Tom;
    if (hasAnyWord (both, { "vocal", "vocals", "vox", "chant", "chants", "acapella" }))      return Category::Vocal;
    if (hasAnyWord (both, { "foley", "toy", "toys", "texture", "textures" }))                return Category::Foley;
    if (hasAnyWord (both, { "riser", "risers", "uplifter", "sweep", "sweeps", "transition", "transitions" })
        || containsWordOrSub (both, "white noise"))                                          return Category::Riser;
    if (hasAnyWord (both, { "impact", "impacts", "downlifter", "fx", "sfx", "boom" }))       return Category::Impact;
    if (hasAnyWord (both, { "stab", "stabs", "stb", "orch", "orchestra" }))                  return Category::Stab;
    if (hasAnyWord (both, { "chord", "chords", "pad", "pads", "guitar", "piano" }))          return Category::Chord;
    if (hasAnyWord (both, { "bass", "basses", "reese", "sub" }))                            return Category::Bass;
    if (hasAnyWord (both, { "synth", "pluck", "plucks", "lead", "arp", "bell", "keys" }))    return Category::Synth;

    if (isLoop || durationMs > 4000.0f)
        return pitchConf > 0.55f ? Category::Chord : Category::Impact;

    const float bassTotal = subRatio + lowRatio;
    const float highTotal = highMidRatio + airRatio;

    if (bassTotal > 0.55f)
    {
        if (durationMs > 380.0f && pitchConf > 0.55f)
            return Category::Bass808;
        return Category::Kick;
    }
    if (highTotal > 0.50f && centroidHz > 3500.0f)
    {
        if (durationMs > 750.0f)  return Category::Cymbal;
        if (durationMs > 280.0f)  return Category::OpenHat;
        return Category::ClosedHat;
    }
    if (pitchConf > 0.65f && flatness < 0.15f)
        return (subRatio > 0.25f) ? Category::Bass : Category::Synth;
    if (flatness > 0.18f && midRatio + highMidRatio > 0.45f)
        return (crest > 5.5f && durationMs < 450.0f) ? Category::Snare : Category::Clap;
    if (durationMs < 500.0f && crest > 4.0f)
        return Category::Perc;

    return durationMs > 1800.0f ? Category::Impact : Category::Foley;
}

bool DspExtractor::classifyIsLoop (const std::string& fullPath,
                                   const std::string& filename,
                                   float durationMs,
                                   const std::array<int8_t, kWaveformPeaks * 2>& peaks)
{
    if (durationMs < 1200.0f)
        return false;

    const std::string pLow = toLowerAscii (fullPath.empty() ? filename : fullPath);
    const std::string fLow = toLowerAscii (filename);

    const bool hasExplicitLoopWord = hasAnyWord (pLow, {
        "loop", "loops", "drumloop", "drumloops", "toploop", "toploops",
        "break", "breaks", "breakbeat", "breakbeats"
    });
    const bool hasFxOneshotWord = hasAnyWord (pLow, {
        "impact", "impacts", "riser", "risers", "uplifter", "uplifters",
        "downlifter", "downlifters", "sweep", "sweeps", "whoosh",
        "transition", "transitions", "subdrop", "crash", "crashes"
    }) || containsWordOrSub (pLow, "white noise") || containsWordOrSub (pLow, "sub drop");

    // Explicit "loop"/"break" wins unless it's clearly a tempo-synced riser/impact without "loop".
    if (hasExplicitLoopWord)
        return true;

    // Tempo-synced FX (e.g. "Cymatics - Dubstep Riser 2 - 150 BPM.wav") are oneshots, not loops.
    if (hasFxOneshotWord)
        return false;

    // BPM tag (when not an impact/riser) strongly indicates a loop.
    if (hasWord (pLow, "bpm"))
        return true;

    const bool hasSoftOneshotWord = hasAnyWord (pLow, {
        "oneshot", "oneshots", "shot", "shots", "hit", "hits", "stab", "stabs",
        "foley", "fx", "sfx", "808", "808s", "kick", "kicks", "snare", "snares",
        "clap", "claps", "hat", "hats", "hihat", "hihats", "cymbal", "cymbals"
    }) || containsWordOrSub (pLow, "one shot") || containsWordOrSub (pLow, "one-shot")
       || containsWordOrSub (pLow, "one_shot");

    const bool hasSoftLoopWord = hasAnyWord (pLow, {
        "melody", "melodies", "groove", "grooves", "riff", "riffs",
        "progression", "progressions", "buildup", "buildups", "stem", "stems", "arp", "arps"
    });

    if (hasSoftOneshotWord && ! hasSoftLoopWord)
        return false;

    // Acoustic envelope test using the 96-bin peak contour:
    // True loops have sustained energy at both start & end plus multiple rhythmic peaks,
    // unlike impacts/long oneshots (monotonic decay) or risers (monotonic ramp).
    float env[kWaveformPeaks] {};
    float mx = 1e-6f;
    for (int i = 0; i < kWaveformPeaks; ++i)
    {
        const float a = std::abs (static_cast<float> (peaks[static_cast<size_t> (i * 2)])) / 127.0f;
        const float b = std::abs (static_cast<float> (peaks[static_cast<size_t> (i * 2 + 1)])) / 127.0f;
        env[i] = std::max (a, b);
        mx = std::max (mx, env[i]);
    }

    float first = 0.0f, mid = 0.0f, last = 0.0f;
    for (int i = 0; i < 32; ++i)  first += env[i];
    for (int i = 32; i < 64; ++i) mid   += env[i];
    for (int i = 64; i < 96; ++i) last  += env[i];
    first /= 32.0f;
    mid   /= 32.0f;
    last  /= 32.0f;

    int onsetPeaks = 0;
    float valley = env[0];
    bool rising = false;
    for (int i = 1; i < kWaveformPeaks; ++i)
    {
        const float v = env[i];
        if (v < valley)
        {
            valley = v;
            rising = false;
        }
        else if (v - valley > 0.15f * mx && ! rising)
        {
            ++onsetPeaks;
            rising = true;
            valley = v;
        }
        else if (rising && v > valley)
        {
            valley = v;
        }
    }

    const bool sustained = (first >= 0.18f * mx) && (mid >= 0.20f * mx) && (last >= 0.22f * mx);
    if (durationMs >= 1800.0f && sustained && onsetPeaks >= 4)
        return true;
    if (durationMs >= 7000.0f && sustained)
        return true;

    (void) fLow;
    return false;
}

DspFeatures DspExtractor::extract (const DecodedAudio& audio,
                                   const std::string& filename,
                                   const std::string& folder,
                                   const std::string& fullPath) const
{
    DspFeatures out;
    if (! audio.ok || audio.mono.empty())
        return out;

    // 1. Waveform peaks from original decoded mono (96 min/max pairs).
    {
        const size_t n = audio.mono.size();
        float absMax = 1e-6f;
        for (float s : audio.mono)
            absMax = std::max (absMax, std::abs (s));
        const float inv = 127.0f / absMax;

        for (int bin = 0; bin < kWaveformPeaks; ++bin)
        {
            const size_t i0 = (static_cast<size_t> (bin) * n) / kWaveformPeaks;
            const size_t i1 = std::max (i0 + 1, (static_cast<size_t> (bin + 1) * n) / kWaveformPeaks);
            float mn = 1.0f, mx = -1.0f;
            for (size_t i = i0; i < std::min (i1, n); ++i)
            {
                mn = std::min (mn, audio.mono[i]);
                mx = std::max (mx, audio.mono[i]);
            }
            out.peaks[static_cast<size_t> (bin * 2)]     = static_cast<int8_t> (std::clamp (std::lround (mn * inv), -127L, 127L));
            out.peaks[static_cast<size_t> (bin * 2 + 1)] = static_cast<int8_t> (std::clamp (std::lround (mx * inv), -127L, 127L));
        }
    }

    // 2. Resample to 44.1 kHz & peak-normalize for scale-invariant DSP features.
    std::vector<float> sig = AudioDecoder::resampleLinear (audio.mono, audio.sampleRate, kWorkSr);
    if (sig.empty())
        return out;

    float peak = 0.0f;
    size_t peakIdx = 0;
    for (size_t i = 0; i < sig.size(); ++i)
    {
        const float a = std::abs (sig[i]);
        if (a > peak)
        {
            peak = a;
            peakIdx = i;
        }
    }
    if (peak < 1e-6f)
        return out;

    const float invPeak = 1.0f / peak;
    for (float& s : sig)
        s *= invPeak;

    // 3. Onset trimming: first sample above -48 dB (0.004) relative to peak.
    size_t onset = 0;
    for (size_t i = 0; i < sig.size(); ++i)
    {
        if (std::abs (sig[i]) >= 0.004f)
        {
            onset = i;
            break;
        }
    }
    const float* trimmed = sig.data() + onset;
    const int trimmedLen = static_cast<int> (sig.size() - onset);
    if (trimmedLen < 32)
        return out;

    // Effective active duration (last sample above -48 dB).
    int activeEnd = trimmedLen;
    for (int i = trimmedLen - 1; i >= 0; --i)
    {
        if (std::abs (trimmed[i]) >= 0.004f)
        {
            activeEnd = i + 1;
            break;
        }
    }
    const float activeDurMs = static_cast<float> (activeEnd) * 1000.0f / static_cast<float> (kWorkSr);

    // 4. Temporal envelope descriptors: attack time, decay -12dB time, crest factor, ZCR.
    const int peakRel = static_cast<int> (peakIdx >= onset ? (peakIdx - onset) : 0);
    int idx10 = 0, idx90 = peakRel;
    for (int i = 0; i <= peakRel; ++i)
    {
        if (std::abs (trimmed[i]) >= 0.10f) { idx10 = i; break; }
    }
    for (int i = idx10; i <= peakRel; ++i)
    {
        if (std::abs (trimmed[i]) >= 0.90f) { idx90 = i; break; }
    }
    const float attackMs = static_cast<float> (std::max (0, idx90 - idx10)) * 1000.0f / static_cast<float> (kWorkSr);

    // RMS in 10ms hops to find -12dB decay after peak.
    const int envHop = kWorkSr / 100; // 441 samples (10 ms)
    float peakRms = 1e-6f;
    int peakFrame = 0;
    const int numEnvFrames = std::max (1, activeEnd / envHop);
    std::vector<float> envRms (static_cast<size_t> (numEnvFrames), 0.0f);
    for (int f = 0; f < numEnvFrames; ++f)
    {
        const int i0 = f * envHop;
        const int i1 = std::min (activeEnd, i0 + envHop);
        double e = 0.0;
        for (int i = i0; i < i1; ++i)
            e += static_cast<double> (trimmed[i]) * trimmed[i];
        const float r = std::sqrt (static_cast<float> (e / std::max (1, i1 - i0)));
        envRms[static_cast<size_t> (f)] = r;
        if (r > peakRms)
        {
            peakRms = r;
            peakFrame = f;
        }
    }
    int decayFrame = numEnvFrames - 1;
    const float thresh12 = peakRms * 0.2512f; // -12 dB
    for (int f = peakFrame; f < numEnvFrames; ++f)
    {
        if (envRms[static_cast<size_t> (f)] < thresh12)
        {
            decayFrame = f;
            break;
        }
    }
    const float decayMs = static_cast<float> (std::max (1, decayFrame - peakFrame)) * 10.0f;

    double totalEnergy = 0.0;
    int zeroCrossings = 0;
    for (int i = 0; i < activeEnd; ++i)
    {
        totalEnergy += static_cast<double> (trimmed[i]) * trimmed[i];
        if (i > 0 && ((trimmed[i] >= 0.0f) != (trimmed[i - 1] >= 0.0f)))
            ++zeroCrossings;
    }
    const float rms = std::sqrt (static_cast<float> (totalEnergy / std::max (1, activeEnd)));
    const float crestFactor = 1.0f / std::max (rms, 1e-4f);
    const float zcr = static_cast<float> (zeroCrossings) / static_cast<float> (std::max (1, activeEnd));

    // 5. 3-Slice Temporal Log-Mel Energies (48 dims = 3 x 16).
    const int attackSamples = static_cast<int> (0.035 * kWorkSr); // 0 - 35 ms
    const int bodyEnd       = static_cast<int> (0.200 * kWorkSr); // 35 - 200 ms

    std::array<std::array<float, kNumBins>, 3> slicePower {};
    computeSpectrum (trimmed, std::min (trimmedLen, attackSamples), slicePower[0].data());
    if (trimmedLen > attackSamples)
        computeSpectrum (trimmed + attackSamples, std::min (trimmedLen - attackSamples, bodyEnd - attackSamples), slicePower[1].data());
    if (trimmedLen > bodyEnd)
        computeSpectrum (trimmed + bodyEnd, std::min (trimmedLen - bodyEnd, kWorkSr * 2), slicePower[2].data());

    for (int s = 0; s < 3; ++s)
    {
        std::array<float, kMelBands> mel {};
        float meanLog = 0.0f;
        for (int m = 0; m < kMelBands; ++m)
        {
            double acc = 0.0;
            const float* row = melFilterbank.data() + static_cast<size_t> (m * kNumBins);
            for (int b = 0; b < kNumBins; ++b)
                acc += static_cast<double> (row[b]) * slicePower[static_cast<size_t> (s)][static_cast<size_t> (b)];
            mel[static_cast<size_t> (m)] = std::log10 (static_cast<float> (acc) + 1e-8f);
            meanLog += mel[static_cast<size_t> (m)];
        }
        meanLog /= static_cast<float> (kMelBands);
        // Zero-mean per slice so spectral shape dominates over loudness.
        float sliceScale = 0.0f;
        for (int m = 0; m < kMelBands; ++m)
        {
            mel[static_cast<size_t> (m)] -= meanLog;
            sliceScale += mel[static_cast<size_t> (m)] * mel[static_cast<size_t> (m)];
        }
        sliceScale = std::sqrt (sliceScale) + 1e-6f;
        const float weight = (s == 0) ? 1.15f : (s == 1 ? 1.0f : 0.70f);
        for (int m = 0; m < kMelBands; ++m)
            out.vec[static_cast<size_t> (s * kMelBands + m)] = weight * (mel[static_cast<size_t> (m)] / sliceScale);
    }

    // 6. Full-sample power spectrum for spectral descriptors & band ratios.
    std::array<float, kNumBins> fullPower {};
    computeSpectrum (trimmed, std::min (activeEnd, kWorkSr * 4), fullPower.data());

    double sumP = 0.0, sumMag = 0.0, sumFreqMag = 0.0, logGeoSum = 0.0;
    double subE = 0.0, lowE = 0.0, midE = 0.0, hiMidE = 0.0, airE = 0.0;
    for (int b = 1; b < kNumBins; ++b)
    {
        const double p = static_cast<double> (fullPower[static_cast<size_t> (b)]) + 1e-12;
        const double mag = std::sqrt (p);
        const double f = static_cast<double> (binFreqs[static_cast<size_t> (b)]);
        sumP += p;
        sumMag += mag;
        sumFreqMag += f * mag;
        logGeoSum += std::log (p);

        if (f < 80.0)        subE += p;
        else if (f < 250.0)  lowE += p;
        else if (f < 2000.0) midE += p;
        else if (f < 6000.0) hiMidE += p;
        else                 airE += p;
    }

    const float centroidHz = static_cast<float> (sumFreqMag / std::max (sumMag, 1e-12));
    double spreadAcc = 0.0;
    for (int b = 1; b < kNumBins; ++b)
    {
        const double mag = std::sqrt (static_cast<double> (fullPower[static_cast<size_t> (b)]) + 1e-12);
        const double df = static_cast<double> (binFreqs[static_cast<size_t> (b)]) - centroidHz;
        spreadAcc += df * df * mag;
    }
    const float spreadHz = static_cast<float> (std::sqrt (spreadAcc / std::max (sumMag, 1e-12)));

    double cumP = 0.0;
    float rolloffHz = 18000.0f;
    for (int b = 1; b < kNumBins; ++b)
    {
        cumP += static_cast<double> (fullPower[static_cast<size_t> (b)]);
        if (cumP >= 0.85 * sumP)
        {
            rolloffHz = binFreqs[static_cast<size_t> (b)];
            break;
        }
    }

    const double arithMean = sumP / static_cast<double> (kNumBins - 1);
    const double geoMean = std::exp (logGeoSum / static_cast<double> (kNumBins - 1));
    const float flatness = clamp01 (static_cast<float> (geoMean / std::max (arithMean, 1e-12)));

    const float subRatio   = static_cast<float> (subE / std::max (sumP, 1e-12));
    const float lowRatio   = static_cast<float> (lowE / std::max (sumP, 1e-12));
    const float midRatio   = static_cast<float> (midE / std::max (sumP, 1e-12));
    const float hiMidRatio = static_cast<float> (hiMidE / std::max (sumP, 1e-12));
    const float airRatio   = static_cast<float> (airE / std::max (sumP, 1e-12));

    // 7. YIN pitch detection (start slightly past initial click if sample is long enough).
    const int pitchStart = (activeEnd > 2400) ? std::min (activeEnd - 2048, static_cast<int> (0.015 * kWorkSr)) : 0;
    estimatePitchYin (trimmed + pitchStart, activeEnd - pitchStart, out.pitchHz, out.pitchConfidence);
    if (out.pitchConfidence >= 0.55f && out.pitchHz >= 30.0f && out.pitchHz <= 2200.0f)
    {
        const int note = static_cast<int> (std::lround (69.0 + 12.0 * std::log2 (out.pitchHz / 440.0)));
        if (note >= 12 && note <= 108)
            out.midiNote = note;
    }

    // 8. Pack 16 global descriptors into dims [48..63] with calibrated scaling.
    const float logDur     = std::clamp (std::log10 (std::max (activeDurMs, 10.0f) / 1000.0f), -2.0f, 1.2f) / 1.5f;
    const float logAttack  = std::clamp (std::log10 (attackMs + 1.0f), 0.0f, 3.0f) / 3.0f;
    const float logDecay   = std::clamp (std::log10 (decayMs + 1.0f), 0.5f, 3.5f) / 3.5f;
    const float normCrest  = std::clamp ((crestFactor - 1.5f) / 10.0f, 0.0f, 1.0f);
    const float normZcr    = std::clamp (zcr * 4.0f, 0.0f, 1.0f);
    const float normCent   = std::clamp ((std::log2 (std::max (centroidHz, 40.0f)) - 5.3f) / 8.5f, 0.0f, 1.0f);
    const float normSpread = std::clamp (spreadHz / 6000.0f, 0.0f, 1.0f);
    const float normRoll   = std::clamp ((std::log2 (std::max (rolloffHz, 80.0f)) - 6.3f) / 8.0f, 0.0f, 1.0f);

    constexpr float kDescWeight = 0.65f;
    out.vec[48] = kDescWeight * logDur;
    out.vec[49] = kDescWeight * logAttack;
    out.vec[50] = kDescWeight * logDecay;
    out.vec[51] = kDescWeight * normCrest;
    out.vec[52] = kDescWeight * normZcr;
    out.vec[53] = kDescWeight * normCent;
    out.vec[54] = kDescWeight * normSpread;
    out.vec[55] = kDescWeight * normRoll;
    out.vec[56] = kDescWeight * flatness;
    out.vec[57] = kDescWeight * subRatio;
    out.vec[58] = kDescWeight * lowRatio;
    out.vec[59] = kDescWeight * midRatio;
    out.vec[60] = kDescWeight * hiMidRatio;
    out.vec[61] = kDescWeight * airRatio;
    out.vec[62] = kDescWeight * out.pitchConfidence;
    out.vec[63] = kDescWeight * audio.stereoWidth;

    // L2-normalize final 64D DSP vector.
    float normSq = 0.0f;
    for (float v : out.vec)
        normSq += v * v;
    const float invNorm = 1.0f / (std::sqrt (normSq) + 1e-8f);
    for (float& v : out.vec)
        v *= invNorm;

    out.brightness = normCent;
    out.punch = clamp01 (0.6f * normCrest + 0.4f * (1.0f - logAttack));

    // 9. Oneshot vs Loop detection (word-bounded path cues + acoustic envelope).
    const std::string pathContext = fullPath.empty() ? (folder + "/" + filename) : fullPath;
    out.isLoop = classifyIsLoop (pathContext, filename, audio.fullDurationMs, out.peaks);

    out.heuristicCategory = classifyHeuristic (audio.fullDurationMs, out.isLoop,
                                               subRatio, lowRatio, midRatio, hiMidRatio, airRatio,
                                               flatness, crestFactor, centroidHz, out.pitchConfidence,
                                               filename, folder);
    return out;
}

} // namespace svdb::dsp
