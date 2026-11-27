#include "OrtLoader.h"
#include "core/Types.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <filesystem>
#include <vector>

namespace svdb::ml
{

namespace
{
void anchorSymbol() {}

#ifdef _WIN32
std::filesystem::path getThisModuleDirectory()
{
    HMODULE hm = nullptr;
    if (GetModuleHandleExW (GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR> (&anchorSymbol), &hm))
    {
        wchar_t buf[MAX_PATH * 2] {};
        const DWORD len = GetModuleFileNameW (hm, buf, static_cast<DWORD> (std::size (buf)));
        if (len > 0)
            return std::filesystem::path (buf).parent_path();
    }
    return {};
}
#endif
} // namespace

OrtLoader& OrtLoader::instance()
{
    static OrtLoader inst;
    return inst;
}

OrtLoader::OrtLoader()
{
#ifdef _WIN32
    std::vector<std::filesystem::path> searchDirs;
    if (const auto dir = getThisModuleDirectory(); ! dir.empty())
        searchDirs.push_back (dir);
    searchDirs.push_back (std::filesystem::u8path (defaultDataDir().getChildFile ("runtime").getFullPathName().toStdString()));

    for (const auto& dir : searchDirs)
    {
        if (! dmlModule)
        {
            const auto dmlPath = dir / L"DirectML.dll";
            if (std::filesystem::exists (dmlPath))
                dmlModule = LoadLibraryExW (dmlPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
        if (! ortModule)
        {
            const auto ortPath = dir / L"onnxruntime.dll";
            if (std::filesystem::exists (ortPath))
                ortModule = LoadLibraryExW (ortPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
        if (ortModule)
            break;
    }

    if (! ortModule)
        ortModule = LoadLibraryW (L"onnxruntime.dll");

    if (! ortModule)
    {
        status = "onnxruntime.dll not found (DSP-only mode)";
        return;
    }

    using GetApiBaseFn = const OrtApiBase* (ORT_API_CALL*) ();
    auto getBase = reinterpret_cast<GetApiBaseFn> (GetProcAddress (static_cast<HMODULE> (ortModule), "OrtGetApiBase"));
    if (! getBase)
    {
        status = "OrtGetApiBase missing";
        return;
    }

    const OrtApiBase* base = getBase();
    if (! base)
    {
        status = "OrtApiBase null";
        return;
    }

    api = base->GetApi (ORT_API_VERSION);
    if (! api)
    {
        // Try older API version fallback if DLL version differs slightly.
        api = base->GetApi (17);
    }
    if (! api)
    {
        status = "Unsupported ORT API version";
        return;
    }

    appendDmlFn = reinterpret_cast<AppendDmlFn> (
        GetProcAddress (static_cast<HMODULE> (ortModule), "OrtSessionOptionsAppendExecutionProvider_DML"));

    OrtStatus* st = api->CreateEnv (ORT_LOGGING_LEVEL_ERROR, "svdb", &env);
    if (st != nullptr)
    {
        status = api->GetErrorMessage (st);
        api->ReleaseStatus (st);
        api = nullptr;
        env = nullptr;
        return;
    }

    status = appendDmlFn ? "ORT Ready (DirectML + CPU)" : "ORT Ready (CPU)";
#else
    status = "ORT dynamic loader only configured for Windows";
#endif
}

OrtLoader::~OrtLoader()
{
    if (api && env)
        api->ReleaseEnv (env);
    // Intentionally keep DLLs loaded until process exit to avoid teardown races in VST hosts.
}

OrtSessionOptions* OrtLoader::createSessionOptions (bool useDirectML, int cpuThreads, bool& outUsingDml) const
{
    outUsingDml = false;
    if (! api)
        return nullptr;

    OrtSessionOptions* opts = nullptr;
    OrtStatus* st = api->CreateSessionOptions (&opts);
    if (st != nullptr || ! opts)
    {
        if (st) api->ReleaseStatus (st);
        return nullptr;
    }

    (void) api->SetSessionGraphOptimizationLevel (opts, ORT_ENABLE_ALL);

    if (useDirectML && appendDmlFn != nullptr)
    {
        // DirectML requires sequential execution and disabled memory pattern.
        (void) api->SetSessionExecutionMode (opts, ORT_SEQUENTIAL);
        (void) api->DisableMemPattern (opts);
        OrtStatus* dmlSt = appendDmlFn (opts, 0);
        if (dmlSt == nullptr)
        {
            outUsingDml = true;
            return opts;
        }
        api->ReleaseStatus (dmlSt);
    }

    (void) api->SetIntraOpNumThreads (opts, cpuThreads > 0 ? cpuThreads : 4);
    (void) api->SetInterOpNumThreads (opts, 1);
    return opts;
}

void OrtLoader::checkStatus (const OrtApi* a, OrtStatus* st, std::string* errOut)
{
    if (! st)
        return;
    if (errOut && a)
        *errOut = a->GetErrorMessage (st);
    if (a)
        a->ReleaseStatus (st);
}

} // namespace svdb::ml
