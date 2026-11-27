#include "SampleDatabase.h"

#include <cstring>
#include <filesystem>

namespace svdb::db
{

SampleDatabase::~SampleDatabase()
{
    close();
}

bool SampleDatabase::execSql (const char* sql, std::string* errOut)
{
    char* errMsg = nullptr;
    const int rc = sqlite3_exec (db, sql, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK)
    {
        if (errOut && errMsg) *errOut = errMsg;
        if (errMsg) sqlite3_free (errMsg);
        return false;
    }
    return true;
}

bool SampleDatabase::open (const std::string& dbPath, std::string* errOut)
{
    std::lock_guard<std::mutex> lock (mtx);
    if (db)
    {
        sqlite3_close (db);
        db = nullptr;
    }

    const auto p = std::filesystem::u8path (dbPath);
    if (p.has_parent_path())
    {
        std::error_code ec;
        std::filesystem::create_directories (p.parent_path(), ec);
    }

    const int rc = sqlite3_open_v2 (dbPath.c_str(), &db,
                                    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                                    nullptr);
    if (rc != SQLITE_OK)
    {
        if (errOut) *errOut = db ? sqlite3_errmsg (db) : "sqlite3_open_v2 failed";
        if (db) { sqlite3_close (db); db = nullptr; }
        return false;
    }

    currentPath = dbPath;
    sqlite3_busy_timeout (db, 5000);

    execSql ("PRAGMA journal_mode=WAL;");
    execSql ("PRAGMA synchronous=NORMAL;");
    execSql ("PRAGMA temp_store=MEMORY;");

    static constexpr const char* kSchema = R"SQL(
        CREATE TABLE IF NOT EXISTS samples (
            id              INTEGER PRIMARY KEY,
            path            TEXT UNIQUE NOT NULL,
            filename        TEXT NOT NULL,
            folder          TEXT NOT NULL,
            file_size       INTEGER NOT NULL,
            mtime           INTEGER NOT NULL,
            duration_ms     REAL NOT NULL,
            sample_rate     INTEGER NOT NULL,
            channels        INTEGER NOT NULL,
            is_loop         INTEGER NOT NULL DEFAULT 0,
            category        INTEGER NOT NULL DEFAULT 0,
            category_src    INTEGER NOT NULL DEFAULT 0,
            midi_note       INTEGER NOT NULL DEFAULT -1,
            brightness      REAL NOT NULL DEFAULT 0.5,
            punch           REAL NOT NULL DEFAULT 0.5,
            dsp_vec         BLOB,
            clap_vec        BLOB,
            peaks           BLOB,
            map_x           REAL NOT NULL DEFAULT 0.0,
            map_y           REAL NOT NULL DEFAULT 0.0,
            map_dsp_x       REAL NOT NULL DEFAULT 0.0,
            map_dsp_y       REAL NOT NULL DEFAULT 0.0,
            map_clap_x      REAL NOT NULL DEFAULT 0.0,
            map_clap_y      REAL NOT NULL DEFAULT 0.0,
            has_clap        INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_samples_has_clap ON samples(has_clap);
        CREATE TABLE IF NOT EXISTS folders (
            path            TEXT PRIMARY KEY,
            last_scan       INTEGER NOT NULL DEFAULT 0
        );
    )SQL";

    if (! execSql (kSchema, errOut))
        return false;

    // Non-destructive migration for existing DBs created before multi-blend keyframes.
    execSql ("ALTER TABLE samples ADD COLUMN map_dsp_x  REAL NOT NULL DEFAULT 0.0;");
    execSql ("ALTER TABLE samples ADD COLUMN map_dsp_y  REAL NOT NULL DEFAULT 0.0;");
    execSql ("ALTER TABLE samples ADD COLUMN map_clap_x REAL NOT NULL DEFAULT 0.0;");
    execSql ("ALTER TABLE samples ADD COLUMN map_clap_y REAL NOT NULL DEFAULT 0.0;");
    return true;
}

void SampleDatabase::close()
{
    std::lock_guard<std::mutex> lock (mtx);
    if (db)
    {
        sqlite3_close (db);
        db = nullptr;
    }
}

bool SampleDatabase::addFolder (const std::string& folderPath)
{
    std::lock_guard<std::mutex> lock (mtx);
    if (! db) return false;
    sqlite3_stmt* st = nullptr;
    const char* sql = "INSERT OR REPLACE INTO folders(path, last_scan) VALUES (?, strftime('%s','now'));";
    if (sqlite3_prepare_v2 (db, sql, -1, &st, nullptr) != SQLITE_OK)
        return false;
    sqlite3_bind_text (st, 1, folderPath.c_str(), -1, SQLITE_TRANSIENT);
    const bool ok = sqlite3_step (st) == SQLITE_DONE;
    sqlite3_finalize (st);
    return ok;
}

std::vector<std::string> SampleDatabase::getFolders()
{
    std::lock_guard<std::mutex> lock (mtx);
    std::vector<std::string> out;
    if (! db) return out;
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, "SELECT path FROM folders ORDER BY path;", -1, &st, nullptr) != SQLITE_OK)
        return out;
    while (sqlite3_step (st) == SQLITE_ROW)
    {
        const auto* txt = reinterpret_cast<const char*> (sqlite3_column_text (st, 0));
        if (txt) out.emplace_back (txt);
    }
    sqlite3_finalize (st);
    return out;
}

std::unordered_map<std::string, FileStamp> SampleDatabase::getExistingStamps()
{
    std::lock_guard<std::mutex> lock (mtx);
    std::unordered_map<std::string, FileStamp> map;
    if (! db) return map;
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, "SELECT id, path, file_size, mtime, has_clap FROM samples;", -1, &st, nullptr) != SQLITE_OK)
        return map;

    while (sqlite3_step (st) == SQLITE_ROW)
    {
        FileStamp fs;
        fs.id       = sqlite3_column_int64 (st, 0);
        const auto* p = reinterpret_cast<const char*> (sqlite3_column_text (st, 1));
        fs.fileSize = sqlite3_column_int64 (st, 2);
        fs.mtime    = sqlite3_column_int64 (st, 3);
        fs.hasClap  = sqlite3_column_int (st, 4) != 0;
        if (p) map.emplace (p, fs);
    }
    sqlite3_finalize (st);
    return map;
}

bool SampleDatabase::upsertDspBatch (const std::vector<SampleRecord>& records)
{
    if (records.empty()) return true;
    std::lock_guard<std::mutex> lock (mtx);
    if (! db) return false;

    if (! execSql ("BEGIN IMMEDIATE;"))
        return false;

    static constexpr const char* kSql = R"SQL(
        INSERT INTO samples (
            path, filename, folder, file_size, mtime, duration_ms, sample_rate, channels,
            is_loop, category, category_src, midi_note, brightness, punch, dsp_vec, peaks, has_clap
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(path) DO UPDATE SET
            filename=excluded.filename,
            folder=excluded.folder,
            file_size=excluded.file_size,
            mtime=excluded.mtime,
            duration_ms=excluded.duration_ms,
            sample_rate=excluded.sample_rate,
            channels=excluded.channels,
            is_loop=excluded.is_loop,
            category=excluded.category,
            category_src=0,
            midi_note=excluded.midi_note,
            brightness=excluded.brightness,
            punch=excluded.punch,
            dsp_vec=excluded.dsp_vec,
            peaks=excluded.peaks,
            has_clap=0;
    )SQL";

    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, kSql, -1, &st, nullptr) != SQLITE_OK)
    {
        execSql ("ROLLBACK;");
        return false;
    }

    for (const auto& rec : records)
    {
        const auto& m = rec.meta;
        sqlite3_reset (st);
        sqlite3_clear_bindings (st);
        sqlite3_bind_text   (st, 1,  m.path.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text   (st, 2,  m.filename.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text   (st, 3,  m.folder.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64  (st, 4,  m.fileSize);
        sqlite3_bind_int64  (st, 5,  m.mtime);
        sqlite3_bind_double (st, 6,  m.durationMs);
        sqlite3_bind_int    (st, 7,  m.sampleRate);
        sqlite3_bind_int    (st, 8,  m.channels);
        sqlite3_bind_int    (st, 9,  m.isLoop ? 1 : 0);
        sqlite3_bind_int    (st, 10, static_cast<int> (m.category));
        sqlite3_bind_int    (st, 11, m.categorySrc);
        sqlite3_bind_int    (st, 12, m.midiNote);
        sqlite3_bind_double (st, 13, m.brightness);
        sqlite3_bind_double (st, 14, m.punch);
        sqlite3_bind_blob   (st, 15, rec.dspVec.data(), static_cast<int> (rec.dspVec.size() * sizeof (float)), SQLITE_TRANSIENT);
        sqlite3_bind_blob   (st, 16, m.peaks.data(), static_cast<int> (m.peaks.size()), SQLITE_TRANSIENT);
        sqlite3_step (st);
    }

    sqlite3_finalize (st);
    return execSql ("COMMIT;");
}

bool SampleDatabase::updateClapBatch (const std::vector<ClapUpdate>& updates)
{
    if (updates.empty()) return true;
    std::lock_guard<std::mutex> lock (mtx);
    if (! db) return false;

    if (! execSql ("BEGIN IMMEDIATE;"))
        return false;

    const char* sql = "UPDATE samples SET clap_vec=?, category=?, category_src=1, has_clap=1 WHERE id=?;";
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, sql, -1, &st, nullptr) != SQLITE_OK)
    {
        execSql ("ROLLBACK;");
        return false;
    }

    for (const auto& u : updates)
    {
        sqlite3_reset (st);
        sqlite3_clear_bindings (st);
        sqlite3_bind_blob  (st, 1, u.clapVec.data(), static_cast<int> (u.clapVec.size() * sizeof (float)), SQLITE_TRANSIENT);
        sqlite3_bind_int   (st, 2, static_cast<int> (u.category));
        sqlite3_bind_int64 (st, 3, u.id);
        sqlite3_step (st);
    }

    sqlite3_finalize (st);
    return execSql ("COMMIT;");
}

bool SampleDatabase::updateLayoutBatch (const std::vector<LayoutUpdate>& updates)
{
    if (updates.empty()) return true;
    std::lock_guard<std::mutex> lock (mtx);
    if (! db) return false;

    if (! execSql ("BEGIN IMMEDIATE;"))
        return false;

    const char* sql = "UPDATE samples SET map_x=?, map_y=?, map_dsp_x=?, map_dsp_y=?, map_clap_x=?, map_clap_y=? WHERE id=?;";
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, sql, -1, &st, nullptr) != SQLITE_OK)
    {
        execSql ("ROLLBACK;");
        return false;
    }

    for (const auto& u : updates)
    {
        sqlite3_reset (st);
        sqlite3_clear_bindings (st);
        sqlite3_bind_double (st, 1, u.x);
        sqlite3_bind_double (st, 2, u.y);
        sqlite3_bind_double (st, 3, u.dspX);
        sqlite3_bind_double (st, 4, u.dspY);
        sqlite3_bind_double (st, 5, u.clapX);
        sqlite3_bind_double (st, 6, u.clapY);
        sqlite3_bind_int64  (st, 7, u.id);
        sqlite3_step (st);
    }

    sqlite3_finalize (st);
    return execSql ("COMMIT;");
}

bool SampleDatabase::updateLoopFlagBatch (const std::vector<LoopFlagUpdate>& updates)
{
    if (updates.empty()) return true;
    std::lock_guard<std::mutex> lock (mtx);
    if (! db) return false;

    if (! execSql ("BEGIN IMMEDIATE;"))
        return false;

    const char* sql = "UPDATE samples SET is_loop=? WHERE id=?;";
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, sql, -1, &st, nullptr) != SQLITE_OK)
    {
        execSql ("ROLLBACK;");
        return false;
    }

    for (const auto& u : updates)
    {
        sqlite3_reset (st);
        sqlite3_clear_bindings (st);
        sqlite3_bind_int   (st, 1, u.isLoop ? 1 : 0);
        sqlite3_bind_int64 (st, 2, u.id);
        sqlite3_step (st);
    }

    sqlite3_finalize (st);
    return execSql ("COMMIT;");
}

int SampleDatabase::deletePathsNotIn (const std::string& folderPrefix,
                                      const std::unordered_map<std::string, bool>& presentPaths)
{
    std::lock_guard<std::mutex> lock (mtx);
    if (! db) return 0;

    std::vector<int64_t> toDelete;
    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, "SELECT id, path FROM samples;", -1, &st, nullptr) == SQLITE_OK)
    {
        while (sqlite3_step (st) == SQLITE_ROW)
        {
            const int64_t id = sqlite3_column_int64 (st, 0);
            const auto* p = reinterpret_cast<const char*> (sqlite3_column_text (st, 1));
            if (! p) continue;
            const std::string path (p);
            if (path.rfind (folderPrefix, 0) == 0 && presentPaths.find (path) == presentPaths.end())
                toDelete.push_back (id);
        }
        sqlite3_finalize (st);
    }

    if (toDelete.empty())
        return 0;

    execSql ("BEGIN IMMEDIATE;");
    if (sqlite3_prepare_v2 (db, "DELETE FROM samples WHERE id=?;", -1, &st, nullptr) == SQLITE_OK)
    {
        for (int64_t id : toDelete)
        {
            sqlite3_reset (st);
            sqlite3_bind_int64 (st, 1, id);
            sqlite3_step (st);
        }
        sqlite3_finalize (st);
    }
    execSql ("COMMIT;");
    return static_cast<int> (toDelete.size());
}

std::vector<SampleDatabase::PendingClapItem> SampleDatabase::getPendingClap (int limit)
{
    std::lock_guard<std::mutex> lock (mtx);
    std::vector<PendingClapItem> out;
    if (! db) return out;

    sqlite3_stmt* st = nullptr;
    const char* sql = "SELECT id, path, filename, folder, duration_ms, is_loop FROM samples WHERE has_clap=0 ORDER BY id LIMIT ?;";
    if (sqlite3_prepare_v2 (db, sql, -1, &st, nullptr) != SQLITE_OK)
        return out;
    sqlite3_bind_int (st, 1, limit);

    while (sqlite3_step (st) == SQLITE_ROW)
    {
        PendingClapItem item;
        item.id = sqlite3_column_int64 (st, 0);
        if (auto* p = reinterpret_cast<const char*> (sqlite3_column_text (st, 1))) item.path = p;
        if (auto* f = reinterpret_cast<const char*> (sqlite3_column_text (st, 2))) item.filename = f;
        if (auto* d = reinterpret_cast<const char*> (sqlite3_column_text (st, 3))) item.folder = d;
        item.durationMs = static_cast<float> (sqlite3_column_double (st, 4));
        item.isLoop = sqlite3_column_int (st, 5) != 0;
        out.push_back (std::move (item));
    }
    sqlite3_finalize (st);
    return out;
}

std::vector<SampleRecord> SampleDatabase::loadAll()
{
    std::lock_guard<std::mutex> lock (mtx);
    std::vector<SampleRecord> out;
    if (! db) return out;

    static constexpr const char* kSql = R"SQL(
        SELECT id, path, filename, folder, file_size, mtime, duration_ms, sample_rate, channels,
               is_loop, category, category_src, midi_note, brightness, punch,
               dsp_vec, clap_vec, peaks, map_x, map_y, map_dsp_x, map_dsp_y, map_clap_x, map_clap_y, has_clap
        FROM samples ORDER BY id;
    )SQL";

    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, kSql, -1, &st, nullptr) != SQLITE_OK)
        return out;

    while (sqlite3_step (st) == SQLITE_ROW)
    {
        SampleRecord rec;
        auto& m = rec.meta;
        m.id = sqlite3_column_int64 (st, 0);
        if (auto* t = reinterpret_cast<const char*> (sqlite3_column_text (st, 1))) m.path = t;
        if (auto* t = reinterpret_cast<const char*> (sqlite3_column_text (st, 2))) m.filename = t;
        if (auto* t = reinterpret_cast<const char*> (sqlite3_column_text (st, 3))) m.folder = t;
        m.fileSize    = sqlite3_column_int64 (st, 4);
        m.mtime       = sqlite3_column_int64 (st, 5);
        m.durationMs  = static_cast<float> (sqlite3_column_double (st, 6));
        m.sampleRate  = sqlite3_column_int (st, 7);
        m.channels    = sqlite3_column_int (st, 8);
        m.isLoop      = sqlite3_column_int (st, 9) != 0;
        const int cat = sqlite3_column_int (st, 10);
        m.category    = (cat >= 0 && cat < kNumCategories) ? static_cast<Category> (cat) : Category::Other;
        m.categorySrc = sqlite3_column_int (st, 11);
        m.midiNote    = sqlite3_column_int (st, 12);
        m.brightness  = static_cast<float> (sqlite3_column_double (st, 13));
        m.punch       = static_cast<float> (sqlite3_column_double (st, 14));

        const void* dspBlob = sqlite3_column_blob (st, 15);
        const int dspBytes  = sqlite3_column_bytes (st, 15);
        if (dspBlob && dspBytes == static_cast<int> (kDspDim * sizeof (float)))
            std::memcpy (rec.dspVec.data(), dspBlob, static_cast<size_t> (dspBytes));

        const void* clapBlob = sqlite3_column_blob (st, 16);
        const int clapBytes  = sqlite3_column_bytes (st, 16);
        if (clapBlob && clapBytes == static_cast<int> (kClapDim * sizeof (float)))
            std::memcpy (rec.clapVec.data(), clapBlob, static_cast<size_t> (clapBytes));

        const void* peakBlob = sqlite3_column_blob (st, 17);
        const int peakBytes  = sqlite3_column_bytes (st, 17);
        if (peakBlob && peakBytes == static_cast<int> (m.peaks.size()))
            std::memcpy (m.peaks.data(), peakBlob, static_cast<size_t> (peakBytes));

        m.mapX     = static_cast<float> (sqlite3_column_double (st, 18));
        m.mapY     = static_cast<float> (sqlite3_column_double (st, 19));
        m.mapDspX  = static_cast<float> (sqlite3_column_double (st, 20));
        m.mapDspY  = static_cast<float> (sqlite3_column_double (st, 21));
        m.mapClapX = static_cast<float> (sqlite3_column_double (st, 22));
        m.mapClapY = static_cast<float> (sqlite3_column_double (st, 23));
        m.hasClap  = sqlite3_column_int (st, 24) != 0;

        out.push_back (std::move (rec));
    }
    sqlite3_finalize (st);
    return out;
}

DbStats SampleDatabase::getStats()
{
    std::lock_guard<std::mutex> lock (mtx);
    DbStats s;
    if (! db) return s;

    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2 (db, "SELECT is_loop, category, has_clap FROM samples;", -1, &st, nullptr) != SQLITE_OK)
        return s;

    while (sqlite3_step (st) == SQLITE_ROW)
    {
        ++s.total;
        if (sqlite3_column_int (st, 0) != 0) ++s.loops;
        else                                 ++s.oneshots;
        const int c = sqlite3_column_int (st, 1);
        if (c >= 0 && c < kNumCategories) s.byCategory[static_cast<size_t> (c)]++;
        if (sqlite3_column_int (st, 2) != 0) ++s.withClap;
    }
    sqlite3_finalize (st);
    return s;
}

} // namespace svdb::db
