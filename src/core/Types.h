#pragma once

#include <juce_core/juce_core.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace svdb
{

inline constexpr int kDspDim = 64;
inline constexpr int kClapDim = 512;
inline constexpr int kWaveformPeaks = 96; // 96 min/max int8 pairs = 192 bytes

enum class Category : int
{
    Other = 0,
    Kick,
    Snare,
    Rim,
    Clap,
    ClosedHat,
    OpenHat,
    Cymbal,
    Tom,
    Perc,
    Bass808,
    Bass,
    Synth,
    Chord,
    Stab,
    Vocal,
    Foley,
    Riser,
    Impact,
    Count
};

inline constexpr int kNumCategories = static_cast<int> (Category::Count);

inline const char* categoryName (Category c)
{
    switch (c)
    {
        case Category::Kick:      return "Kick";
        case Category::Snare:     return "Snare";
        case Category::Rim:       return "Rim";
        case Category::Clap:      return "Clap";
        case Category::ClosedHat: return "Closed Hat";
        case Category::OpenHat:   return "Open Hat";
        case Category::Cymbal:    return "Cymbal";
        case Category::Tom:       return "Tom";
        case Category::Perc:      return "Perc";
        case Category::Bass808:   return "808";
        case Category::Bass:      return "Bass";
        case Category::Synth:     return "Synth";
        case Category::Chord:     return "Chord";
        case Category::Stab:      return "Stab";
        case Category::Vocal:     return "Vocal";
        case Category::Foley:     return "Foley";
        case Category::Riser:     return "Riser";
        case Category::Impact:    return "Impact";
        case Category::Other:
        default:                  return "Other";
    }
}

inline const char* categoryShort (Category c)
{
    switch (c)
    {
        case Category::Kick:      return "KCK";
        case Category::Snare:     return "SNR";
        case Category::Rim:       return "RIM";
        case Category::Clap:      return "CLP";
        case Category::ClosedHat: return "CHH";
        case Category::OpenHat:   return "OHH";
        case Category::Cymbal:    return "CYM";
        case Category::Tom:       return "TOM";
        case Category::Perc:      return "PRC";
        case Category::Bass808:   return "808";
        case Category::Bass:      return "BAS";
        case Category::Synth:     return "SYN";
        case Category::Chord:     return "CHD";
        case Category::Stab:      return "STB";
        case Category::Vocal:     return "VOX";
        case Category::Foley:     return "FOL";
        case Category::Riser:     return "RSR";
        case Category::Impact:    return "IMP";
        case Category::Other:
        default:                  return "---";
    }
}

inline std::string midiNoteToName (int midiNote)
{
    if (midiNote < 0 || midiNote > 127)
        return "-";
    static constexpr const char* kNames[12] = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };
    const int octave = (midiNote / 12) - 1;
    return std::string (kNames[midiNote % 12]) + std::to_string (octave);
}

enum class LengthFilter : int
{
    All = 0,
    Oneshots = 1,
    Loops = 2
};

struct SampleMeta
{
    int64_t id = 0;
    std::string path;
    std::string filename;
    std::string folder;
    int64_t fileSize = 0;
    int64_t mtime = 0;
    float durationMs = 0.0f;
    int sampleRate = 0;
    int channels = 0;
    bool isLoop = false;
    Category category = Category::Other;
    int categorySrc = 0; // 0 = DSP heuristic, 1 = CLAP zero-shot
    int midiNote = -1;
    float brightness = 0.5f;
    float punch = 0.5f;
    float mapX = 0.0f;       // Hybrid (0.5) UMAP X
    float mapY = 0.0f;       // Hybrid (0.5) UMAP Y
    float mapDspX = 0.0f;    // Pure DSP (0.0) UMAP X (Procrustes-aligned to Hybrid)
    float mapDspY = 0.0f;    // Pure DSP (0.0) UMAP Y
    float mapClapX = 0.0f;   // Pure CLAP (1.0) UMAP X (Procrustes-aligned to Hybrid)
    float mapClapY = 0.0f;   // Pure CLAP (1.0) UMAP Y
    bool hasClap = false;
    std::array<int8_t, kWaveformPeaks * 2> peaks {}; // [min0, max0, min1, max1, ...]

    void mapPosForBlend (float blend, float& outX, float& outY) const
    {
        const bool hasKeyframes = (mapDspX != 0.0f || mapDspY != 0.0f || mapClapX != 0.0f || mapClapY != 0.0f);
        if (! hasKeyframes)
        {
            outX = mapX;
            outY = mapY;
            return;
        }
        const float b = (blend < 0.0f) ? 0.0f : (blend > 1.0f ? 1.0f : blend);
        if (b <= 0.5f)
        {
            const float t = b * 2.0f;
            outX = mapDspX + t * (mapX - mapDspX);
            outY = mapDspY + t * (mapY - mapDspY);
        }
        else
        {
            const float t = (b - 0.5f) * 2.0f;
            outX = mapX + t * (mapClapX - mapX);
            outY = mapY + t * (mapClapY - mapY);
        }
    }
};

struct SampleRecord
{
    SampleMeta meta;
    std::array<float, kDspDim> dspVec {};
    std::array<float, kClapDim> clapVec {};
};

struct SearchQuery
{
    std::string text;
    std::vector<float> textClapVec;      // Optional 512D CLAP text embedding
    int64_t referenceId = -1;            // Optional in-DB reference sample
    std::vector<float> refDspVec;        // Optional ad-hoc external WAV DSP vector
    std::vector<float> refClapVec;       // Optional ad-hoc external WAV CLAP vector
    float clapBlend = 0.55f;             // 0 = pure DSP, 1 = pure CLAP
    LengthFilter lengthFilter = LengthFilter::All;
    int categoryFilter = -1;             // -1 = all, else Category int
    int maxResults = 500;
};

struct SearchResult
{
    int index = -1;     // Index into VectorIndex::Snapshot
    int64_t id = 0;
    float score = 0.0f; // Similarity or match score
};

enum class IndexPhase : int
{
    Idle = 0,
    Scanning,
    DspExtracting,
    ClapEmbedding,
    ComputingLayout,
    Done,
    Error
};

struct IndexProgress
{
    IndexPhase phase = IndexPhase::Idle;
    int done = 0;
    int total = 0;
    std::string message;
};

inline juce::File defaultDataDir()
{
    auto dir = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                   .getParentDirectory()
                   .getChildFile ("Local")
                   .getChildFile ("sample-vector-db");
    if (! dir.isDirectory())
        dir = juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
                  .getChildFile ("sample-vector-db");
    dir.createDirectory();
    return dir;
}

inline juce::File defaultDbFile()
{
    return defaultDataDir().getChildFile ("library.db");
}

inline juce::File defaultModelDir()
{
    auto dir = defaultDataDir().getChildFile ("models");
    dir.createDirectory();
    return dir;
}

inline bool isSupportedAudioExtension (const juce::String& extLower)
{
    return extLower == ".wav" || extLower == ".mp3" || extLower == ".flac"
        || extLower == ".aif" || extLower == ".aiff" || extLower == ".ogg";
}

// Extracts the top-level sample pack folder (first segment after "\Samples\" or "/Samples/"),
// falling back to immediate parent folder name.
inline std::string samplePackName (const std::string& fullPath, const std::string& fallbackFolder)
{
    std::string lower = fullPath;
    for (char& c : lower)
    {
        if (c == '/') c = '\\';
        else if (c >= 'A' && c <= 'Z') c = static_cast<char> (c - 'A' + 'a');
    }
    const std::string marker = "\\samples\\";
    const size_t pos = lower.find (marker);
    if (pos != std::string::npos)
    {
        const size_t start = pos + marker.size();
        const size_t sep = lower.find ('\\', start);
        if (sep != std::string::npos && sep > start)
            return fullPath.substr (start, sep - start);
    }
    return fallbackFolder;
}

inline std::string sampleFolderLabel (const std::string& fullPath, const std::string& folder)
{
    const std::string pack = samplePackName (fullPath, folder);
    if (pack.empty()) return folder;
    if (folder.empty() || pack == folder) return pack;
    return pack + " / " + folder;
}

} // namespace svdb
