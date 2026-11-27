#pragma once

#include <onnxruntime_c_api.h>

#include <string>

namespace svdb::ml
{

// Dynamically loads onnxruntime.dll (and DirectML.dll if present) from the
// plugin/executable's own directory so VST3 hosts never fail to load the plugin
// if ORT DLLs are missing.
class OrtLoader
{
public:
    static OrtLoader& instance();

    bool isAvailable() const { return api != nullptr && env != nullptr; }
    bool hasDirectML() const { return appendDmlFn != nullptr; }
    const OrtApi* getApi() const { return api; }
    OrtEnv* getEnv() const { return env; }
    std::string statusText() const { return status; }

    // Creates session options configured for DirectML (if available and requested)
    // or CPU with the specified thread count. Caller owns returned pointer.
    OrtSessionOptions* createSessionOptions (bool useDirectML, int cpuThreads, bool& outUsingDml) const;

    static void checkStatus (const OrtApi* api, OrtStatus* st, std::string* errOut = nullptr);

private:
    OrtLoader();
    ~OrtLoader();

    OrtLoader (const OrtLoader&) = delete;
    OrtLoader& operator= (const OrtLoader&) = delete;

    using AppendDmlFn = OrtStatus* (ORT_API_CALL*) (OrtSessionOptions*, int);

    void* ortModule = nullptr;
    void* dmlModule = nullptr;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    AppendDmlFn appendDmlFn = nullptr;
    std::string status;
};

} // namespace svdb::ml
