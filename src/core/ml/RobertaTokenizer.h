#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace svdb::ml
{

// Pure C++ byte-level RoBERTa BPE tokenizer loading Hugging Face tokenizer.json.
class RobertaTokenizer
{
public:
    bool loadFromFile (const std::string& tokenizerJsonPath, std::string* errOut = nullptr);
    bool isLoaded() const { return loaded; }

    struct Encoded
    {
        std::vector<int64_t> inputIds;
        std::vector<int64_t> attentionMask;
    };

    Encoded encode (const std::string& text, int maxLength = 77) const;

private:
    bool loaded = false;
    int64_t bosId = 0; // <s>
    int64_t padId = 1; // <pad>
    int64_t eosId = 2; // </s>
    int64_t unkId = 3; // <unk>

    std::string byteToUnicode[256];
    std::unordered_map<std::string, int64_t> vocab;
    std::unordered_map<std::string, int> mergeRanks; // "a\tb" -> rank

    void initByteEncoder();
    std::vector<std::string> preTokenize (const std::string& text) const;
    std::vector<std::string> bpe (const std::string& token) const;
};

} // namespace svdb::ml
