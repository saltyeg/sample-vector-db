#include "ClapEncoder.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace svdb::ml
{
namespace
{
std::wstring utf8ToWide (const std::string& s)
{
    return std::filesystem::u8path (s).wstring();
}

std::string toLower (const std::string& s)
{
    std::string r = s;
    for (char& c : r)
        c = static_cast<char> (std::tolower (static_cast<unsigned char> (c)));
    return r;
}

float dot512 (const float* a, const float* b)
{
    float s = 0.0f;
    for (int i = 0; i < kClapDim; ++i)
        s += a[i] * b[i];
    return s;
}
} // namespace

ClapEncoder::ClapEncoder()
{
    auto& ort = OrtLoader::instance();
    if (ort.isAvailable())
    {
        OrtStatus* st = ort.getApi()->CreateCpuMemoryInfo (OrtArenaAllocator, OrtMemTypeDefault, &cpuMemInfo);
        OrtLoader::checkStatus (ort.getApi(), st);
    }
}

ClapEncoder::~ClapEncoder()
{
    auto& ort = OrtLoader::instance();
    if (const OrtApi* api = ort.getApi())
    {
        if (audioSession) api->ReleaseSession (audioSession);
        if (textSession)  api->ReleaseSession (textSession);
        if (cpuMemInfo)   api->ReleaseMemoryInfo (cpuMemInfo);
    }
}

bool ClapEncoder::modelsExistIn (const std::string& modelDir)
{
    const auto base = std::filesystem::u8path (modelDir);
    return std::filesystem::exists (base / "audio_model.onnx")
        && std::filesystem::exists (base / "text_model.onnx")
        && std::filesystem::exists (base / "tokenizer.json");
}

std::string ClapEncoder::providerLabel() const
{
    if (! OrtLoader::instance().isAvailable())
        return "DSP only";
    if (! isAudioReady() && ! isTextReady())
        return "No CLAP models";
    return audioUsingDml ? "CLAP DirectML" : "CLAP CPU";
}

void ClapEncoder::l2Normalize (float* v, int dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim; ++i)
        sum += static_cast<double> (v[i]) * v[i];
    const float inv = 1.0f / (static_cast<float> (std::sqrt (sum)) + 1e-8f);
    for (int i = 0; i < dim; ++i)
        v[i] *= inv;
}

void ClapEncoder::inspectSessionIO (const OrtApi* api,
                                    OrtSession* sess,
                                    std::vector<std::string>& inputs,
                                    std::vector<std::string>& outputs)
{
    OrtAllocator* alloc = nullptr;
    OrtStatus* st = api->GetAllocatorWithDefaultOptions (&alloc);
    if (st || ! alloc)
    {
        if (st) api->ReleaseStatus (st);
        return;
    }

    size_t nIn = 0, nOut = 0;
    (void) api->SessionGetInputCount (sess, &nIn);
    (void) api->SessionGetOutputCount (sess, &nOut);

    for (size_t i = 0; i < nIn; ++i)
    {
        char* name = nullptr;
        if (auto* s = api->SessionGetInputName (sess, i, alloc, &name); s == nullptr && name)
        {
            inputs.emplace_back (name);
            (void) api->AllocatorFree (alloc, name);
        }
        else if (s) api->ReleaseStatus (s);
    }
    for (size_t i = 0; i < nOut; ++i)
    {
        char* name = nullptr;
        if (auto* s = api->SessionGetOutputName (sess, i, alloc, &name); s == nullptr && name)
        {
            outputs.emplace_back (name);
            (void) api->AllocatorFree (alloc, name);
        }
        else if (s) api->ReleaseStatus (s);
    }
}

bool ClapEncoder::loadTextModelOnly (const std::string& modelDir, std::string* errOut)
{
    std::lock_guard<std::mutex> lock (textMutex);
    auto& ort = OrtLoader::instance();
    if (! ort.isAvailable() || ! cpuMemInfo)
    {
        if (errOut) *errOut = ort.statusText();
        return false;
    }

    const auto base = std::filesystem::u8path (modelDir);
    const auto tokPath = (base / "tokenizer.json").u8string();
    const auto textPath = base / "text_model.onnx";

    if (! std::filesystem::exists (textPath))
    {
        if (errOut) *errOut = "Missing text_model.onnx in " + modelDir;
        return false;
    }

    if (! tokenizer.isLoaded())
    {
        std::string tokStr (reinterpret_cast<const char*> (tokPath.data()), tokPath.size());
        if (! tokenizer.loadFromFile (tokStr, errOut))
            return false;
    }

    if (! textSession)
    {
        // Text encoder is tiny & fast (~15ms on CPU); CPU avoids DirectML dynamic shape overhead on variable text lengths.
        bool dummyDml = false;
        OrtSessionOptions* opts = ort.createSessionOptions (false, 4, dummyDml);
        const OrtApi* api = ort.getApi();
        OrtStatus* st = api->CreateSession (ort.getEnv(), textPath.c_str(), opts, &textSession);
        api->ReleaseSessionOptions (opts);
        if (st != nullptr)
        {
            OrtLoader::checkStatus (api, st, errOut);
            textSession = nullptr;
            return false;
        }

        std::vector<std::string> ins, outs;
        inspectSessionIO (api, textSession, ins, outs);
        textInputIdsName = ! ins.empty() ? ins[0] : "input_ids";
        textMaskName.clear();
        textHasMask      = false;
        textOutputName   = ! outs.empty() ? outs[0] : "text_embeds";
        for (const auto& n : ins)
        {
            if (n.find ("input_ids") != std::string::npos) textInputIdsName = n;
            if (n.find ("attention_mask") != std::string::npos)
            {
                textMaskName = n;
                textHasMask = true;
            }
        }
        for (const auto& n : outs)
        {
            if (n.find ("text_embeds") != std::string::npos || n.find ("embed") != std::string::npos)
                textOutputName = n;
        }
    }

    return true;
}

bool ClapEncoder::loadAudioModelOnly (const std::string& modelDir, bool preferDirectML, std::string* errOut)
{
    std::lock_guard<std::mutex> lock (audioMutex);
    auto& ort = OrtLoader::instance();
    if (! ort.isAvailable() || ! cpuMemInfo)
    {
        if (errOut) *errOut = ort.statusText();
        return false;
    }

    const auto audioPath = std::filesystem::u8path (modelDir) / "audio_model.onnx";
    if (! std::filesystem::exists (audioPath))
    {
        if (errOut) *errOut = "Missing audio_model.onnx in " + modelDir;
        return false;
    }

    if (audioSession)
        return true;

    const OrtApi* api = ort.getApi();
    auto tryCreate = [&] (bool useDml) -> bool {
        bool usingDml = false;
        OrtSessionOptions* opts = ort.createSessionOptions (useDml, 8, usingDml);
        OrtStatus* st = api->CreateSession (ort.getEnv(), audioPath.c_str(), opts, &audioSession);
        api->ReleaseSessionOptions (opts);
        if (st != nullptr)
        {
            OrtLoader::checkStatus (api, st, errOut);
            audioSession = nullptr;
            return false;
        }
        audioUsingDml = usingDml;
        return true;
    };

    if (! tryCreate (preferDirectML))
    {
        if (! preferDirectML || ! tryCreate (false))
            return false;
    }

    std::vector<std::string> ins, outs;
    inspectSessionIO (api, audioSession, ins, outs);
    audioInputName  = ! ins.empty() ? ins[0] : "input_features";
    audioOutputName = ! outs.empty() ? outs[0] : "audio_embeds";
    for (const auto& n : ins)
        if (n.find ("input_features") != std::string::npos) audioInputName = n;
    for (const auto& n : outs)
        if (n.find ("audio_embeds") != std::string::npos || n.find ("embed") != std::string::npos)
            audioOutputName = n;

    return true;
}

bool ClapEncoder::loadModels (const std::string& modelDir, bool preferDirectML, std::string* errOut)
{
    const bool okText  = loadTextModelOnly (modelDir, errOut);
    const bool okAudio = loadAudioModelOnly (modelDir, preferDirectML, errOut);
    if (okText && categoryProtos.empty())
        initCategoryPrototypes();
    return okText && okAudio;
}

bool ClapEncoder::encodeText (const std::string& query,
                              std::array<float, kClapDim>& outEmbed,
                              std::string* errOut)
{
    std::lock_guard<std::mutex> lock (textMutex);
    auto& ort = OrtLoader::instance();
    if (! textSession || ! tokenizer.isLoaded() || ! ort.isAvailable())
    {
        if (errOut) *errOut = "CLAP text model not loaded";
        return false;
    }

    const auto enc = tokenizer.encode (query, 77);
    const int64_t seqLen = static_cast<int64_t> (enc.inputIds.size());
    const int64_t shape[2] = { 1, seqLen };

    const OrtApi* api = ort.getApi();
    OrtValue* idsTensor = nullptr;
    OrtValue* maskTensor = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue (
        cpuMemInfo, const_cast<int64_t*> (enc.inputIds.data()),
        enc.inputIds.size() * sizeof (int64_t), shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &idsTensor);
    if (st) { OrtLoader::checkStatus (api, st, errOut); return false; }

    if (textHasMask)
    {
        st = api->CreateTensorWithDataAsOrtValue (
            cpuMemInfo, const_cast<int64_t*> (enc.attentionMask.data()),
            enc.attentionMask.size() * sizeof (int64_t), shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &maskTensor);
        if (st) { api->ReleaseValue (idsTensor); OrtLoader::checkStatus (api, st, errOut); return false; }
    }

    const char* inNames[2] = { textInputIdsName.c_str(), textMaskName.c_str() };
    const OrtValue* inVals[2] = { idsTensor, maskTensor };
    const size_t numInputs = textHasMask ? 2 : 1;
    const char* outNames[1] = { textOutputName.c_str() };
    OrtValue* outVals[1] = { nullptr };

    st = api->Run (textSession, nullptr, inNames, inVals, numInputs, outNames, 1, outVals);
    api->ReleaseValue (idsTensor);
    if (maskTensor) api->ReleaseValue (maskTensor);

    if (st || ! outVals[0])
    {
        OrtLoader::checkStatus (api, st, errOut);
        if (outVals[0]) api->ReleaseValue (outVals[0]);
        return false;
    }

    float* data = nullptr;
    st = api->GetTensorMutableData (outVals[0], reinterpret_cast<void**> (&data));
    if (st || ! data)
    {
        OrtLoader::checkStatus (api, st, errOut);
        api->ReleaseValue (outVals[0]);
        return false;
    }

    std::copy_n (data, kClapDim, outEmbed.data());
    l2Normalize (outEmbed.data(), kClapDim);
    api->ReleaseValue (outVals[0]);
    return true;
}

bool ClapEncoder::encodeAudioBatch (const std::vector<std::vector<float>>& clips48k,
                                    std::vector<std::array<float, kClapDim>>& outEmbeds,
                                    std::string* errOut)
{
    std::lock_guard<std::mutex> lock (audioMutex);
    outEmbeds.clear();
    if (clips48k.empty())
        return true;

    auto& ort = OrtLoader::instance();
    if (! audioSession || ! ort.isAvailable())
    {
        if (errOut) *errOut = "CLAP audio model not loaded";
        return false;
    }

    const int64_t B = static_cast<int64_t> (clips48k.size());
    std::vector<float> batchBuf (static_cast<size_t> (B) * ClapMelFrontend::kFeatureCount);
    for (size_t i = 0; i < clips48k.size(); ++i)
        frontend.compute (clips48k[i], batchBuf.data() + i * ClapMelFrontend::kFeatureCount);

    const int64_t shape[4] = { B, 1, ClapMelFrontend::kNumFrames, ClapMelFrontend::kNumMels };
    const OrtApi* api = ort.getApi();
    OrtValue* inTensor = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue (
        cpuMemInfo, batchBuf.data(), batchBuf.size() * sizeof (float),
        shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inTensor);
    if (st) { OrtLoader::checkStatus (api, st, errOut); return false; }

    const char* inNames[1] = { audioInputName.c_str() };
    const OrtValue* inVals[1] = { inTensor };
    const char* outNames[1] = { audioOutputName.c_str() };
    OrtValue* outVals[1] = { nullptr };

    st = api->Run (audioSession, nullptr, inNames, inVals, 1, outNames, 1, outVals);
    api->ReleaseValue (inTensor);

    if (st || ! outVals[0])
    {
        OrtLoader::checkStatus (api, st, errOut);
        if (outVals[0]) api->ReleaseValue (outVals[0]);
        return false;
    }

    float* data = nullptr;
    st = api->GetTensorMutableData (outVals[0], reinterpret_cast<void**> (&data));
    if (st || ! data)
    {
        OrtLoader::checkStatus (api, st, errOut);
        api->ReleaseValue (outVals[0]);
        return false;
    }

    outEmbeds.resize (clips48k.size());
    for (size_t i = 0; i < clips48k.size(); ++i)
    {
        std::copy_n (data + i * kClapDim, kClapDim, outEmbeds[i].data());
        l2Normalize (outEmbeds[i].data(), kClapDim);
    }

    api->ReleaseValue (outVals[0]);
    return true;
}

void ClapEncoder::initCategoryPrototypes()
{
    struct PromptDef { Category cat; const char* text; };
    static constexpr PromptDef kPrompts[] = {
        { Category::Kick,      "a punchy kick drum one-shot hit" },
        { Category::Kick,      "electronic bass drum thump" },
        { Category::Kick,      "acoustic kick drum beat" },

        { Category::Snare,     "a crisp snare drum hit" },
        { Category::Snare,     "acoustic snare drum crack" },
        { Category::Snare,     "electronic trap snare drum" },

        { Category::Rim,       "snare rimshot sidestick wood click" },
        { Category::Rim,       "sharp dry drum rim hit" },

        { Category::Clap,      "hand clap drum sample" },
        { Category::Clap,      "electronic layered clap snap sound" },

        { Category::ClosedHat, "short tight closed hi-hat cymbal tick" },
        { Category::ClosedHat, "crisp trap closed hihat hit" },

        { Category::OpenHat,   "open hi-hat cymbal sizzle decay" },
        { Category::OpenHat,   "sustained open hihat drum sample" },

        { Category::Cymbal,    "crash cymbal splash sound" },
        { Category::Cymbal,    "ride cymbal metallic shimmer ring" },

        { Category::Tom,       "acoustic tom drum hit fill" },
        { Category::Tom,       "low floor tom percussion drum" },

        { Category::Perc,      "percussion bongo conga shaker tambourine hit" },
        { Category::Perc,      "woodblock cowbell triangle percussion" },

        { Category::Bass808,   "deep 808 sub bass boom note" },
        { Category::Bass808,   "distorted trap 808 sub bass glide" },

        { Category::Bass,      "electronic synth bass reese wobble growl" },
        { Category::Bass,      "electric bass guitar low note" },

        { Category::Synth,     "synthesizer lead pluck arpeggio bell note" },
        { Category::Synth,     "electronic synth melody analog tone" },

        { Category::Chord,     "piano guitar synth chord progression" },
        { Category::Chord,     "warm atmospheric synth pad sustained chord" },

        { Category::Stab,      "short punchy rave synth chord stab hit" },
        { Category::Stab,      "brass horn orchestra hit jungle stab sample" },
        { Category::Stab,      "electronic dub chord stab one-shot" },

        { Category::Vocal,     "human vocal chop singing voice sample" },
        { Category::Vocal,     "vocal shout chant adlib voice" },

        { Category::Foley,     "organic foley found sound knock tap rustle" },
        { Category::Foley,     "household object toy mechanical click texture" },

        { Category::Riser,     "electronic riser uplifter sweep buildup white noise" },
        { Category::Riser,     "rising pitch transition whoosh swell" },

        { Category::Impact,    "cinematic impact boom explosion hit" },
        { Category::Impact,    "downlifter sub drop sound effect crash hit" },
    };

    std::array<std::array<float, kClapDim>, kNumCategories> sums {};
    std::array<int, kNumCategories> counts {};

    for (const auto& p : kPrompts)
    {
        std::array<float, kClapDim> e {};
        if (encodeText (p.text, e))
        {
            const size_t idx = static_cast<size_t> (p.cat);
            for (int d = 0; d < kClapDim; ++d)
                sums[idx][static_cast<size_t> (d)] += e[static_cast<size_t> (d)];
            counts[idx]++;
        }
    }

    categoryProtos.clear();
    for (int c = 1; c < kNumCategories; ++c)
    {
        if (counts[static_cast<size_t> (c)] > 0)
        {
            CategoryProto proto;
            proto.category = static_cast<Category> (c);
            proto.embed = sums[static_cast<size_t> (c)];
            l2Normalize (proto.embed.data(), kClapDim);
            categoryProtos.push_back (proto);
        }
    }
}

namespace
{
bool hasWordCI (const std::string& hayLower, const char* word)
{
    const size_t wlen = std::strlen (word);
    if (wlen == 0) return false;
    auto isAlnum = [] (char c) { return std::isalnum (static_cast<unsigned char> (c)) != 0; };
    size_t pos = hayLower.find (word, 0);
    while (pos != std::string::npos)
    {
        const bool leftOk  = (pos == 0) || ! isAlnum (hayLower[pos - 1]);
        const bool rightOk = (pos + wlen >= hayLower.size()) || ! isAlnum (hayLower[pos + wlen]);
        if (leftOk && rightOk)
            return true;
        pos = hayLower.find (word, pos + 1);
    }
    return false;
}

bool hasAnyWordCI (const std::string& hayLower, std::initializer_list<const char*> words)
{
    for (const char* w : words)
        if (hasWordCI (hayLower, w))
            return true;
    return false;
}
} // namespace

Category ClapEncoder::classifyAudioEmbedding (const std::array<float, kClapDim>& audioEmbed,
                                              const std::string& filename,
                                              const std::string& folder,
                                              float durationMs,
                                              bool /*isLoop*/,
                                              float* outConfidence)
{
    if (categoryProtos.empty() && isTextReady())
        initCategoryPrototypes();

    if (categoryProtos.empty())
        return Category::Other;

    const std::string fLow = toLower (filename);
    const std::string dLow = toLower (folder);

    auto cueScore = [&] (const std::string& text, Category c) -> bool {
        const bool hasHatToken = hasAnyWordCI (text, { "hat", "hats", "hihat", "hihats", "hh", "ch", "chh", "oh", "ohh" })
                              || text.find ("hi-hat") != std::string::npos;
        switch (c)
        {
            case Category::Kick:
                return hasAnyWordCI (text, { "kick", "kicks", "kck", "bd", "bassdrum" });
            case Category::Snare:
                return hasAnyWordCI (text, { "snare", "snares", "snr", "sd" });
            case Category::Rim:
                return hasAnyWordCI (text, { "rim", "rims", "rimshot", "rimshots", "sidestick", "stick" });
            case Category::Clap:
                return hasAnyWordCI (text, { "clap", "claps", "clp", "snap", "snaps" });
            case Category::ClosedHat:
                return (hasHatToken && (hasAnyWordCI (text, { "closed", "ch", "chh" })
                                        || (! hasAnyWordCI (text, { "open", "oh", "ohh" }) && (durationMs <= 0.0f || durationMs < 260.0f))));
            case Category::OpenHat:
                return (hasAnyWordCI (text, { "oh", "ohh" })
                        || (hasHatToken && (hasAnyWordCI (text, { "open" }) || durationMs >= 260.0f)));
            case Category::Cymbal:
                return hasAnyWordCI (text, { "crash", "crashes", "ride", "rides", "cymbal", "cymbals", "cym", "splash" });
            case Category::Tom:
                return hasAnyWordCI (text, { "tom", "toms" });
            case Category::Perc:
                return hasAnyWordCI (text, { "perc", "percs", "percussion", "shaker", "shakers", "conga", "bongo", "cowbell", "tambourine", "woodblock", "clave", "bell" });
            case Category::Bass808:
                return hasAnyWordCI (text, { "808", "808s" });
            case Category::Bass:
                return hasAnyWordCI (text, { "bass", "basses", "reese", "growl", "wobble", "sub" })
                    && ! hasAnyWordCI (text, { "808", "808s", "bassdrum" });
            case Category::Synth:
                return hasAnyWordCI (text, { "synth", "synths", "pluck", "plucks", "lead", "leads", "arp", "arps", "keys", "key", "melody", "brass" });
            case Category::Chord:
                return hasAnyWordCI (text, { "chord", "chords", "pad", "pads", "piano", "guitar", "strings", "rhodes", "organ" });
            case Category::Stab:
                return hasAnyWordCI (text, { "stab", "stabs", "stb", "orch", "orchestra", "horn", "horns", "brass", "rave" })
                    || text.find ("stab") != std::string::npos;
            case Category::Vocal:
                return hasAnyWordCI (text, { "vocal", "vocals", "vox", "voice", "chant", "chants", "shout", "adlib", "acapella", "choir" });
            case Category::Foley:
                return hasAnyWordCI (text, { "foley", "toy", "toys", "texture", "textures", "ambient", "ambience", "click", "clicks" })
                    || text.find ("foley") != std::string::npos;
            case Category::Riser:
                return hasAnyWordCI (text, { "riser", "risers", "uplifter", "uplifters", "sweep", "sweeps", "transition", "transitions", "whoosh", "buildup", "swell" })
                    || text.find ("white noise") != std::string::npos;
            case Category::Impact:
                return hasAnyWordCI (text, { "impact", "impacts", "downlifter", "downlifters", "boom", "fx", "sfx", "exhaust", "subdrop" })
                    || text.find ("sub drop") != std::string::npos;
            default:
                return false;
        }
    };

    Category bestCat = Category::Other;
    float bestScore = -1e9f;
    for (const auto& p : categoryProtos)
    {
        float sc = dot512 (audioEmbed.data(), p.embed.data());
        if (cueScore (fLow, p.category)) sc += 0.13f;
        if (cueScore (dLow, p.category)) sc += 0.09f;

        // Subtle duration nudge for Closed vs Open Hat, Stab vs Chord, and FX tails.
        if (durationMs > 0.0f)
        {
            if (p.category == Category::ClosedHat && durationMs < 240.0f) sc += 0.02f;
            if (p.category == Category::OpenHat   && durationMs > 300.0f && durationMs < 1800.0f) sc += 0.02f;
            if (p.category == Category::Stab      && durationMs < 1800.0f) sc += 0.015f;
            if ((p.category == Category::Riser || p.category == Category::Impact) && durationMs > 2500.0f) sc += 0.02f;
        }

        if (sc > bestScore)
        {
            bestScore = sc;
            bestCat = p.category;
        }
    }
    if (outConfidence)
        *outConfidence = bestScore;
    return bestCat;
}

} // namespace svdb::ml
