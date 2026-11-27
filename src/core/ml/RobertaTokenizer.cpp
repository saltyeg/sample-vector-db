#include "RobertaTokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>

namespace svdb::ml
{
namespace
{
std::string codepointToUtf8 (int cp)
{
    std::string out;
    if (cp < 0x80)
    {
        out.push_back (static_cast<char> (cp));
    }
    else if (cp < 0x800)
    {
        out.push_back (static_cast<char> (0xC0 | (cp >> 6)));
        out.push_back (static_cast<char> (0x80 | (cp & 0x3F)));
    }
    else
    {
        out.push_back (static_cast<char> (0xE0 | (cp >> 12)));
        out.push_back (static_cast<char> (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back (static_cast<char> (0x80 | (cp & 0x3F)));
    }
    return out;
}

bool isAsciiLetter (unsigned char c) { return std::isalpha (c) != 0; }
bool isAsciiDigit  (unsigned char c) { return std::isdigit (c) != 0; }
bool isAsciiSpace  (unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
} // namespace

void RobertaTokenizer::initByteEncoder()
{
    // Exact GPT-2 / RoBERTa bytes_to_unicode() table.
    bool direct[256] {};
    for (int b = 33; b <= 126; ++b) direct[b] = true;
    for (int b = 161; b <= 172; ++b) direct[b] = true;
    for (int b = 174; b <= 255; ++b) direct[b] = true;

    int extra = 0;
    for (int b = 0; b < 256; ++b)
    {
        const int cp = direct[b] ? b : (256 + extra++);
        byteToUnicode[b] = codepointToUtf8 (cp);
    }
}

bool RobertaTokenizer::loadFromFile (const std::string& tokenizerJsonPath, std::string* errOut)
{
    loaded = false;
    vocab.clear();
    mergeRanks.clear();
    initByteEncoder();

    std::ifstream in (tokenizerJsonPath, std::ios::binary);
    if (! in)
    {
        if (errOut) *errOut = "Cannot open tokenizer.json: " + tokenizerJsonPath;
        return false;
    }

    try
    {
        nlohmann::json j;
        in >> j;

        const auto& model = j.at ("model");
        const auto& vObj = model.at ("vocab");
        vocab.reserve (vObj.size());
        for (auto it = vObj.begin(); it != vObj.end(); ++it)
            vocab.emplace (it.key(), it.value().get<int64_t>());

        const auto& merges = model.at ("merges");
        mergeRanks.reserve (merges.size());
        int rank = 0;
        for (const auto& m : merges)
        {
            if (m.is_string())
            {
                const std::string s = m.get<std::string>();
                const auto sp = s.find (' ');
                if (sp != std::string::npos)
                    mergeRanks.emplace (s.substr (0, sp) + '\t' + s.substr (sp + 1), rank);
            }
            else if (m.is_array() && m.size() == 2)
            {
                mergeRanks.emplace (m[0].get<std::string>() + '\t' + m[1].get<std::string>(), rank);
            }
            ++rank;
        }

        auto findSpecial = [&] (const char* tok, int64_t fallback) -> int64_t {
            auto it = vocab.find (tok);
            return it != vocab.end() ? it->second : fallback;
        };
        bosId = findSpecial ("<s>", 0);
        padId = findSpecial ("<pad>", 1);
        eosId = findSpecial ("</s>", 2);
        unkId = findSpecial ("<unk>", 3);

        loaded = ! vocab.empty();
        return loaded;
    }
    catch (const std::exception& e)
    {
        if (errOut) *errOut = std::string ("Failed to parse tokenizer.json: ") + e.what();
        return false;
    }
}

std::vector<std::string> RobertaTokenizer::preTokenize (const std::string& text) const
{
    // GPT-2 / RoBERTa pre-tokenization: contractions, optional leading space + letters,
    // optional leading space + digits, optional leading space + punctuation, trailing spaces.
    std::vector<std::string> pieces;
    const size_t n = text.size();
    size_t i = 0;

    auto matchContraction = [&] (size_t pos) -> size_t {
        if (pos >= n || text[pos] != '\'')
            return 0;
        const size_t rem = n - pos;
        auto eq2 = [&] (const char* s) {
            return rem >= 2 && std::tolower (static_cast<unsigned char> (text[pos + 1])) == s[1];
        };
        auto eq3 = [&] (const char* s) {
            return rem >= 3
                && std::tolower (static_cast<unsigned char> (text[pos + 1])) == s[1]
                && std::tolower (static_cast<unsigned char> (text[pos + 2])) == s[2];
        };
        if (eq3 ("'re") || eq3 ("'ve") || eq3 ("'ll")) return 3;
        if (eq2 ("'s") || eq2 ("'t") || eq2 ("'m") || eq2 ("'d")) return 2;
        return 0;
    };

    while (i < n)
    {
        if (const size_t cLen = matchContraction (i); cLen > 0)
        {
            pieces.push_back (text.substr (i, cLen));
            i += cLen;
            continue;
        }

        const size_t start = i;
        if (text[i] == ' ' && i + 1 < n && ! isAsciiSpace (static_cast<unsigned char> (text[i + 1])))
            ++i;

        const unsigned char c = static_cast<unsigned char> (text[i]);
        if (isAsciiLetter (c) || c >= 0x80)
        {
            while (i < n)
            {
                const unsigned char ch = static_cast<unsigned char> (text[i]);
                if (isAsciiLetter (ch) || ch >= 0x80) ++i;
                else break;
            }
            pieces.push_back (text.substr (start, i - start));
        }
        else if (isAsciiDigit (c))
        {
            while (i < n && isAsciiDigit (static_cast<unsigned char> (text[i])))
                ++i;
            pieces.push_back (text.substr (start, i - start));
        }
        else if (! isAsciiSpace (c))
        {
            while (i < n)
            {
                const unsigned char ch = static_cast<unsigned char> (text[i]);
                if (! isAsciiSpace (ch) && ! isAsciiLetter (ch) && ! isAsciiDigit (ch) && ch < 0x80)
                    ++i;
                else
                    break;
            }
            pieces.push_back (text.substr (start, i - start));
        }
        else
        {
            while (i < n && isAsciiSpace (static_cast<unsigned char> (text[i])))
                ++i;
            pieces.push_back (text.substr (start, i - start));
        }
    }
    return pieces;
}

std::vector<std::string> RobertaTokenizer::bpe (const std::string& token) const
{
    std::vector<std::string> word;
    word.reserve (token.size());
    for (unsigned char b : token)
        word.push_back (byteToUnicode[b]);

    if (word.size() <= 1)
        return word;

    while (word.size() > 1)
    {
        int bestRank = std::numeric_limits<int>::max();
        size_t bestIdx = std::string::npos;

        for (size_t i = 0; i + 1 < word.size(); ++i)
        {
            const std::string key = word[i] + '\t' + word[i + 1];
            auto it = mergeRanks.find (key);
            if (it != mergeRanks.end() && it->second < bestRank)
            {
                bestRank = it->second;
                bestIdx = i;
            }
        }

        if (bestIdx == std::string::npos)
            break;

        const std::string first = word[bestIdx];
        const std::string second = word[bestIdx + 1];
        std::vector<std::string> nextWord;
        nextWord.reserve (word.size());

        size_t i = 0;
        while (i < word.size())
        {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second)
            {
                nextWord.push_back (first + second);
                i += 2;
            }
            else
            {
                nextWord.push_back (word[i]);
                ++i;
            }
        }
        word = std::move (nextWord);
    }

    return word;
}

RobertaTokenizer::Encoded RobertaTokenizer::encode (const std::string& text, int maxLength) const
{
    Encoded out;
    out.inputIds.push_back (bosId);

    const auto pieces = preTokenize (text);
    for (const auto& piece : pieces)
    {
        const auto toks = bpe (piece);
        for (const auto& tok : toks)
        {
            if (static_cast<int> (out.inputIds.size()) >= maxLength - 1)
                break;
            auto it = vocab.find (tok);
            out.inputIds.push_back (it != vocab.end() ? it->second : unkId);
        }
        if (static_cast<int> (out.inputIds.size()) >= maxLength - 1)
            break;
    }

    out.inputIds.push_back (eosId);
    out.attentionMask.assign (out.inputIds.size(), 1);
    return out;
}

} // namespace svdb::ml
