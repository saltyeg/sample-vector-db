#pragma once

#include "SampleDatabase.h"
#include "core/Types.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace svdb::db
{

struct IndexSnapshot
{
    std::vector<SampleMeta> meta;
    std::vector<std::string> lowerSearchText; // pre-lowercased "filename folder category"
    std::vector<float> dspVecs;               // size = N * kDspDim
    std::vector<float> clapVecs;              // size = N * kClapDim
    std::unordered_map<int64_t, int> idToIndex;
    int clapCount = 0;

    int count() const { return static_cast<int> (meta.size()); }
    const float* dspAt (int i) const { return dspVecs.data() + static_cast<size_t> (i) * kDspDim; }
    const float* clapAt (int i) const { return clapVecs.data() + static_cast<size_t> (i) * kClapDim; }
};

class VectorIndex
{
public:
    VectorIndex();

    void rebuildFromRecords (const std::vector<SampleRecord>& records);
    void reloadFromDatabase (SampleDatabase& db);

    std::shared_ptr<const IndexSnapshot> getSnapshot() const;

    std::vector<SearchResult> search (const SearchQuery& q) const;

    static float dotProduct (const float* a, const float* b, int dim);

private:
    mutable std::mutex mtx;
    std::shared_ptr<const IndexSnapshot> snapshot;
};

} // namespace svdb::db
