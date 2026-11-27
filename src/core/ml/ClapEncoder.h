#pragma once

#include "ClapMelFrontend.h"
#include "OrtLoader.h"
#include "RobertaTokenizer.h"
#include "core/Types.h"

#include <array>
#include <mutex>
#include <string>
#include <vector>

namespace svdb::ml
{

class ClapEncoder
{
public:
    ClapEncoder();
    ~ClapEncoder();

    bool loadModels (const std::string& modelDir, bool preferDirectML = true, std::string* errOut = nullptr);
    bool loadTextModelOnly (const std::string& modelDir, std::string* errOut = nullptr);
    bool loadAudioModelOnly (const std::string& modelDir, bool preferDirectML = true, std::string* errOut = nullptr);

    bool isAudioReady() const { return audioSession != nullptr; }
    bool isTextReady() const { return textSession != nullptr && tokenizer.isLoaded(); }
    bool isUsingDirectML() const { return audioUsingDml; }
    std::string providerLabel() const;

    // Encodes a batch of 48kHz mono audio clips -> N x 512 L2-normalized embeddings.
    bool encodeAudioBatch (const std::vector<std::vector<float>>& clips48k,
                           std::vector<std::array<float, kClapDim>>& outEmbeds,
                           std::string* errOut = nullptr);

    // Encodes a natural-language query -> 512 L2-normalized embedding.
    bool encodeText (const std::string& query,
                     std::array<float, kClapDim>& outEmbed,
                     std::string* errOut = nullptr);

    // Zero-shot category classification using pre-embedded CLAP text prototypes
    // combined with word-bounded filename/folder and duration tie-breakers.
    Category classifyAudioEmbedding (const std::array<float, kClapDim>& audioEmbed,
                                     const std::string& filename,
                                     const std::string& folder = {},
                                     float durationMs = 0.0f,
                                     bool isLoop = false,
                                     float* outConfidence = nullptr);

    static bool modelsExistIn (const std::string& modelDir);

private:
    struct CategoryProto
    {
        Category category = Category::Other;
        std::array<float, kClapDim> embed {};
    };

    ClapMelFrontend frontend;
    RobertaTokenizer tokenizer;

    OrtSession* audioSession = nullptr;
    OrtSession* textSession = nullptr;
    OrtMemoryInfo* cpuMemInfo = nullptr;

    std::string audioInputName;
    std::string audioOutputName;
    std::string textInputIdsName;
    std::string textMaskName;
    std::string textOutputName;
    bool textHasMask = false;
    bool audioUsingDml = false;

    std::vector<CategoryProto> categoryProtos;
    std::mutex audioMutex;
    std::mutex textMutex;

    void initCategoryPrototypes();
    static void l2Normalize (float* v, int dim);
    static void inspectSessionIO (const OrtApi* api,
                                  OrtSession* sess,
                                  std::vector<std::string>& inputs,
                                  std::vector<std::string>& outputs);
};

} // namespace svdb::ml
