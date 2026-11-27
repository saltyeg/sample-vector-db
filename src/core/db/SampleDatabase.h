#pragma once

#include "core/Types.h"

#include <sqlite3.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svdb::db
{

struct FileStamp
{
    int64_t id = 0;
    int64_t fileSize = 0;
    int64_t mtime = 0;
    bool hasClap = false;
};

struct ClapUpdate
{
    int64_t id = 0;
    std::array<float, kClapDim> clapVec {};
    Category category = Category::Other;
};

struct LayoutUpdate
{
    int64_t id = 0;
    float x = 0.0f;
    float y = 0.0f;
    float dspX = 0.0f;
    float dspY = 0.0f;
    float clapX = 0.0f;
    float clapY = 0.0f;
};

struct LoopFlagUpdate
{
    int64_t id = 0;
    bool isLoop = false;
};

struct DbStats
{
    int total = 0;
    int oneshots = 0;
    int loops = 0;
    int withClap = 0;
    std::array<int, kNumCategories> byCategory {};
};

class SampleDatabase
{
public:
    SampleDatabase() = default;
    ~SampleDatabase();

    SampleDatabase (const SampleDatabase&) = delete;
    SampleDatabase& operator= (const SampleDatabase&) = delete;

    bool open (const std::string& dbPath, std::string* errOut = nullptr);
    void close();
    bool isOpen() const { return db != nullptr; }
    std::string path() const { return currentPath; }

    bool addFolder (const std::string& folderPath);
    std::vector<std::string> getFolders();

    std::unordered_map<std::string, FileStamp> getExistingStamps();
    bool upsertDspBatch (const std::vector<SampleRecord>& records);
    bool updateClapBatch (const std::vector<ClapUpdate>& updates);
    bool updateLayoutBatch (const std::vector<LayoutUpdate>& updates);
    bool updateLoopFlagBatch (const std::vector<LoopFlagUpdate>& updates);
    int deletePathsNotIn (const std::string& folderPrefix,
                          const std::unordered_map<std::string, bool>& presentPaths);

    // Samples missing CLAP embeddings (id + path + filename + isLoop).
    struct PendingClapItem
    {
        int64_t id = 0;
        std::string path;
        std::string filename;
        std::string folder;
        float durationMs = 0.0f;
        bool isLoop = false;
    };
    std::vector<PendingClapItem> getPendingClap (int limit = 100000);

    std::vector<SampleRecord> loadAll();
    DbStats getStats();

private:
    sqlite3* db = nullptr;
    std::string currentPath;
    std::mutex mtx;

    bool execSql (const char* sql, std::string* errOut = nullptr);
};

} // namespace svdb::db
