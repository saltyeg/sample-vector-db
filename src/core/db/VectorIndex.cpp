#include "VectorIndex.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace svdb::db
{
namespace
{
std::string toLower (const std::string& s)
{
    std::string out = s;
    for (char& c : out)
        c = static_cast<char> (std::tolower (static_cast<unsigned char> (c)));
    return out;
}

std::vector<std::string> splitTokens (const std::string& s)
{
    std::vector<std::string> toks;
    std::string cur;
    for (char c : s)
    {
        if (std::isspace (static_cast<unsigned char> (c)) || c == '_' || c == '-' || c == '/')
        {
            if (! cur.empty())
            {
                toks.push_back (cur);
                cur.clear();
            }
        }
        else
        {
            cur.push_back (static_cast<char> (std::tolower (static_cast<unsigned char> (c))));
        }
    }
    if (! cur.empty())
        toks.push_back (cur);
    return toks;
}
} // namespace

VectorIndex::VectorIndex()
    : snapshot (std::make_shared<IndexSnapshot>())
{
}

float VectorIndex::dotProduct (const float* a, const float* b, int dim)
{
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8)
    {
        const __m256 va = _mm256_loadu_ps (a + i);
        const __m256 vb = _mm256_loadu_ps (b + i);
        acc = _mm256_fmadd_ps (va, vb, acc);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps (lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3]
              + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dim; ++i)
        sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i)
        sum += a[i] * b[i];
    return sum;
#endif
}

void VectorIndex::rebuildFromRecords (const std::vector<SampleRecord>& records)
{
    auto next = std::make_shared<IndexSnapshot>();
    const size_t n = records.size();
    next->meta.reserve (n);
    next->lowerSearchText.reserve (n);
    next->dspVecs.resize (n * kDspDim, 0.0f);
    next->clapVecs.resize (n * kClapDim, 0.0f);
    next->idToIndex.reserve (n);

    for (size_t i = 0; i < n; ++i)
    {
        const auto& r = records[i];
        next->meta.push_back (r.meta);
        next->idToIndex[r.meta.id] = static_cast<int> (i);

        std::string searchStr = r.meta.filename + " " + r.meta.folder + " " + categoryName (r.meta.category);
        if (r.meta.midiNote >= 0)
            searchStr += " " + midiNoteToName (r.meta.midiNote);
        next->lowerSearchText.push_back (toLower (searchStr));

        std::copy_n (r.dspVec.data(), kDspDim, next->dspVecs.data() + i * kDspDim);
        if (r.meta.hasClap)
        {
            std::copy_n (r.clapVec.data(), kClapDim, next->clapVecs.data() + i * kClapDim);
            next->clapCount++;
        }
    }

    std::lock_guard<std::mutex> lock (mtx);
    snapshot = std::move (next);
}

void VectorIndex::reloadFromDatabase (SampleDatabase& db)
{
    rebuildFromRecords (db.loadAll());
}

std::shared_ptr<const IndexSnapshot> VectorIndex::getSnapshot() const
{
    std::lock_guard<std::mutex> lock (mtx);
    return snapshot;
}

std::vector<SearchResult> VectorIndex::search (const SearchQuery& q) const
{
    auto snap = getSnapshot();
    const int n = snap->count();
    if (n == 0)
        return {};

    // Determine query mode.
    const float* refDsp = nullptr;
    const float* refClap = nullptr;
    bool hasRef = false;
    bool hasRefClap = false;

    if (q.referenceId >= 0)
    {
        auto it = snap->idToIndex.find (q.referenceId);
        if (it != snap->idToIndex.end())
        {
            const int idx = it->second;
            refDsp = snap->dspAt (idx);
            hasRef = true;
            if (snap->meta[static_cast<size_t> (idx)].hasClap)
            {
                refClap = snap->clapAt (idx);
                hasRefClap = true;
            }
        }
    }
    else if (q.refDspVec.size() == kDspDim)
    {
        refDsp = q.refDspVec.data();
        hasRef = true;
        if (q.refClapVec.size() == kClapDim)
        {
            refClap = q.refClapVec.data();
            hasRefClap = true;
        }
    }

    const bool hasTextQuery = ! q.text.empty();
    const bool hasTextClap  = (q.textClapVec.size() == kClapDim) && (snap->clapCount > 0);
    const auto queryTokens  = hasTextQuery ? splitTokens (q.text) : std::vector<std::string> {};
    const std::string lowerQuery = hasTextQuery ? toLower (q.text) : std::string {};
    const float blend = std::clamp (q.clapBlend, 0.0f, 1.0f);

    std::vector<SearchResult> results;
    results.reserve (static_cast<size_t> (n));

    for (int i = 0; i < n; ++i)
    {
        const auto& m = snap->meta[static_cast<size_t> (i)];

        // Length filter.
        if (q.lengthFilter == LengthFilter::Oneshots && m.isLoop) continue;
        if (q.lengthFilter == LengthFilter::Loops && ! m.isLoop)  continue;

        // Category filter.
        if (q.categoryFilter >= 0 && static_cast<int> (m.category) != q.categoryFilter)
            continue;

        float score = 0.0f;

        if (hasRef)
        {
            const float dspSim = dotProduct (refDsp, snap->dspAt (i), kDspDim);
            if (hasRefClap && m.hasClap)
            {
                const float clapSim = dotProduct (refClap, snap->clapAt (i), kClapDim);
                score = (1.0f - blend) * dspSim + blend * clapSim;
            }
            else
            {
                score = dspSim;
            }
        }
        else if (hasTextQuery)
        {
            const auto& hay = snap->lowerSearchText[static_cast<size_t> (i)];
            int matchedTokens = 0;
            for (const auto& tok : queryTokens)
                if (hay.find (tok) != std::string::npos)
                    ++matchedTokens;
            const float lexRatio = queryTokens.empty()
                ? 0.0f
                : static_cast<float> (matchedTokens) / static_cast<float> (queryTokens.size());

            if (hasTextClap && m.hasClap)
            {
                // CLAP text-audio cosine similarity typically falls in [-0.1, 0.70].
                // Keep headroom before 1.0 so top matches preserve clear ranking nuance.
                const float rawClap = dotProduct (q.textClapVec.data(), snap->clapAt (i), kClapDim);
                const float normClap = std::clamp ((rawClap + 0.05f) / 0.75f, 0.0f, 0.90f);
                score = normClap + 0.08f * lexRatio;
                if (hay.find (lowerQuery) != std::string::npos)
                    score += 0.04f;
                score = std::clamp (score, 0.0f, 0.99f);
            }
            else
            {
                // Pure lexical fallback when CLAP isn't loaded/indexed yet.
                if (matchedTokens == 0)
                    continue;
                score = lexRatio;
                if (toLower (m.filename).find (lowerQuery) != std::string::npos)
                    score += 0.25f;
            }
        }
        else
        {
            // No active query: stable default ordering (oneshots first, then filename).
            score = 1.0f;
        }

        results.push_back ({ i, m.id, score });
    }

    if (hasRef || hasTextQuery)
    {
        const size_t limit = (q.maxResults > 0) ? std::min<size_t> (results.size(), static_cast<size_t> (q.maxResults)) : results.size();
        std::partial_sort (results.begin(), results.begin() + static_cast<std::ptrdiff_t> (limit), results.end(),
                           [] (const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
        results.resize (limit);
    }
    else
    {
        std::sort (results.begin(), results.end(), [&] (const SearchResult& a, const SearchResult& b) {
            const auto& ma = snap->meta[static_cast<size_t> (a.index)];
            const auto& mb = snap->meta[static_cast<size_t> (b.index)];
            if (ma.category != mb.category)
                return static_cast<int> (ma.category) < static_cast<int> (mb.category);
            return ma.filename < mb.filename;
        });
        if (q.maxResults > 0 && static_cast<int> (results.size()) > q.maxResults)
            results.resize (static_cast<size_t> (q.maxResults));
    }

    return results;
}

} // namespace svdb::db
