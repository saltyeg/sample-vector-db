cmake_minimum_required(VERSION 3.25)

project(SampleVectorDB VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

include(FetchContent)

# ---------------------------------------------------------------- JUCE
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.16-security
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(JUCE)

# ---------------------------------------------------------------- nlohmann/json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------- SQLite3 amalgamation
set(SQLITE_DIR "${CMAKE_BINARY_DIR}/_deps/sqlite3")
if(NOT EXISTS "${SQLITE_DIR}/sqlite3.c")
    file(MAKE_DIRECTORY "${SQLITE_DIR}")
    set(SQLITE_ZIP "${CMAKE_BINARY_DIR}/_deps/sqlite-amalgamation-3460100.zip")
    if(NOT EXISTS "${SQLITE_ZIP}")
        message(STATUS "Downloading SQLite3 amalgamation...")
        file(DOWNLOAD
            "https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip"
            "${SQLITE_ZIP}"
            STATUS SQLITE_DL_STATUS)
        list(GET SQLITE_DL_STATUS 0 SQLITE_DL_CODE)
        if(NOT SQLITE_DL_CODE EQUAL 0)
            message(FATAL_ERROR "Failed to download SQLite3: ${SQLITE_DL_STATUS}")
        endif()
    endif()
    file(ARCHIVE_EXTRACT INPUT "${SQLITE_ZIP}" DESTINATION "${CMAKE_BINARY_DIR}/_deps/sqlite_extract")
    file(COPY "${CMAKE_BINARY_DIR}/_deps/sqlite_extract/sqlite-amalgamation-3460100/sqlite3.c"
              "${CMAKE_BINARY_DIR}/_deps/sqlite_extract/sqlite-amalgamation-3460100/sqlite3.h"
         DESTINATION "${SQLITE_DIR}")
endif()

add_library(sqlite3 STATIC "${SQLITE_DIR}/sqlite3.c")
target_include_directories(sqlite3 PUBLIC "${SQLITE_DIR}")
target_compile_definitions(sqlite3 PRIVATE
    SQLITE_THREADSAFE=1
    SQLITE_ENABLE_FTS5=1
    SQLITE_OMIT_LOAD_EXTENSION=1)
if(MSVC)
    target_compile_options(sqlite3 PRIVATE /W0)
endif()

# ---------------------------------------------------------------- ONNX Runtime DirectML + DirectML (NuGet)
set(ORT_VERSION "1.24.4")
set(DML_VERSION "1.15.4")
set(ORT_DIR "${CMAKE_BINARY_DIR}/_deps/ort_dml_${ORT_VERSION}")
set(DML_DIR "${CMAKE_BINARY_DIR}/_deps/directml_${DML_VERSION}")

if(WIN32)
    if(NOT EXISTS "${ORT_DIR}/build/native/include/onnxruntime_c_api.h")
        set(ORT_ZIP "${CMAKE_BINARY_DIR}/_deps/Microsoft.ML.OnnxRuntime.DirectML.${ORT_VERSION}.zip")
        if(NOT EXISTS "${ORT_ZIP}")
            message(STATUS "Downloading Microsoft.ML.OnnxRuntime.DirectML ${ORT_VERSION}...")
            file(DOWNLOAD
                "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ORT_VERSION}"
                "${ORT_ZIP}"
                STATUS ORT_DL_STATUS)
            list(GET ORT_DL_STATUS 0 ORT_DL_CODE)
            if(NOT ORT_DL_CODE EQUAL 0)
                message(FATAL_ERROR "Failed to download ONNX Runtime DirectML: ${ORT_DL_STATUS}")
            endif()
        endif()
        file(ARCHIVE_EXTRACT INPUT "${ORT_ZIP}" DESTINATION "${ORT_DIR}")
    endif()

    if(NOT EXISTS "${DML_DIR}/bin/x64-win/DirectML.dll")
        set(DML_ZIP "${CMAKE_BINARY_DIR}/_deps/Microsoft.AI.DirectML.${DML_VERSION}.zip")
        if(NOT EXISTS "${DML_ZIP}")
            message(STATUS "Downloading Microsoft.AI.DirectML ${DML_VERSION}...")
            file(DOWNLOAD
                "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/${DML_VERSION}"
                "${DML_ZIP}"
                STATUS DML_DL_STATUS)
            list(GET DML_DL_STATUS 0 DML_DL_CODE)
            if(NOT DML_DL_CODE EQUAL 0)
                message(FATAL_ERROR "Failed to download DirectML: ${DML_DL_STATUS}")
            endif()
        endif()
        file(ARCHIVE_EXTRACT INPUT "${DML_ZIP}" DESTINATION "${DML_DIR}")
    endif()

    set(ORT_INCLUDE_DIR "${ORT_DIR}/build/native/include")
    set(ORT_DLL "${ORT_DIR}/runtimes/win-x64/native/onnxruntime.dll")
    set(DML_DLL "${DML_DIR}/bin/x64-win/DirectML.dll")
endif()

# Helper function to copy runtime DLLs next to an executable or VST3 binary
function(svdb_copy_runtime_dlls TARGET_NAME DEST_DIR)
    if(WIN32 AND EXISTS "${ORT_DLL}" AND EXISTS "${DML_DLL}")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DLL}" "${DEST_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DML_DLL}" "${DEST_DIR}"
            COMMENT "Copying ONNX Runtime + DirectML DLLs for ${TARGET_NAME}")
    endif()
endfunction()

# Dev VST3 install folder (linked into Common Files\VST3 for FL Studio).
set(SVDB_VST3_DIR "$ENV{USERPROFILE}/VST3-dev" CACHE PATH "Where built VST3 bundles are copied")

set(SVDB_COMMON_DEFS
    _SILENCE_CXX20_U8PATH_DEPRECATION_WARNING
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_USE_MP3AUDIOFORMAT=1
    JUCE_USE_FLAC=1
    JUCE_USE_OGGVORBIS=1
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_VST3_CAN_REPLACE_VST2=0)

# ---------------------------------------------------------------- Core library
add_library(svdb_core STATIC
    src/core/dsp/AudioDecoder.cpp
    src/core/dsp/DspExtractor.cpp
    src/core/ml/OrtLoader.cpp
    src/core/ml/ClapMelFrontend.cpp
    src/core/ml/RobertaTokenizer.cpp
    src/core/ml/ClapEncoder.cpp
    src/core/db/SampleDatabase.cpp
    src/core/db/VectorIndex.cpp
    src/core/db/Layout2D.cpp
    src/core/Indexer.cpp)

target_include_directories(svdb_core PUBLIC src "${ORT_INCLUDE_DIR}")
target_compile_definitions(svdb_core PUBLIC ${SVDB_COMMON_DEFS})
if(MSVC)
    target_compile_options(svdb_core PRIVATE /arch:AVX2 /W3)
endif()

target_link_libraries(svdb_core
    PUBLIC
        sqlite3
        nlohmann_json::nlohmann_json
        juce::juce_audio_formats
        juce::juce_dsp
        juce::juce_core)

# ---------------------------------------------------------------- Plugin (VST3 + Standalone)
juce_add_plugin(SampleVectorDB
    COMPANY_NAME "mikey"
    PLUGIN_MANUFACTURER_CODE Mkey
    PLUGIN_CODE SvDb
    PRODUCT_NAME "sample-vector-db"
    ICON_BIG "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.png"
    ICON_SMALL "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.png"
    FORMATS VST3 Standalone
    VST3_CATEGORIES Instrument Sampler
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    VST3_COPY_DIR "${SVDB_VST3_DIR}")

target_sources(SampleVectorDB PRIVATE
    src/plugin/AuditionEngine.cpp
    src/plugin/PluginProcessor.cpp
    src/plugin/PluginEditor.cpp
    src/plugin/ui/SampleMap.cpp
    src/plugin/ui/SampleList.cpp
    src/plugin/ui/AuditionBar.cpp)

target_include_directories(SampleVectorDB PRIVATE src src/plugin)
target_compile_definitions(SampleVectorDB PUBLIC
    ${SVDB_COMMON_DEFS}
    JUCE_MODAL_LOOPS_PERMITTED=1)

target_link_libraries(SampleVectorDB
    PRIVATE
        svdb_core
        juce::juce_audio_utils
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

if(WIN32)
    svdb_copy_runtime_dlls(SampleVectorDB_Standalone "$<TARGET_FILE_DIR:SampleVectorDB_Standalone>")
    # Keep ONNX Runtime + DirectML DLLs outside the VST3 bundle (in %APPDATA%/sample-vector-db/runtime)
    # so DAW VST2 scanners never mistake them for plugins.
    set(SVDB_RUNTIME_DIR "$ENV{APPDATA}/sample-vector-db/runtime")
    if(EXISTS "${ORT_DLL}" AND EXISTS "${DML_DLL}")
        add_custom_command(TARGET SampleVectorDB_VST3 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SVDB_RUNTIME_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DLL}" "${SVDB_RUNTIME_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DML_DLL}" "${SVDB_RUNTIME_DIR}"
            COMMENT "Copying ONNX Runtime + DirectML DLLs to ${SVDB_RUNTIME_DIR}")
    endif()
endif()

# ---------------------------------------------------------------- CLI indexer & query tool
juce_add_console_app(svdb-index PRODUCT_NAME "svdb-index")
target_sources(svdb-index PRIVATE tools/svdb-index/Main.cpp)
target_include_directories(svdb-index PRIVATE src)
target_link_libraries(svdb-index
    PRIVATE
        svdb_core
        winhttp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
svdb_copy_runtime_dlls(svdb-index "$<TARGET_FILE_DIR:svdb-index>")

# ---------------------------------------------------------------- Offscreen UI snapshot tool
add_executable(svdb-ui tools/svdb-ui/Main.cpp)
target_include_directories(svdb-ui PRIVATE $<TARGET_PROPERTY:SampleVectorDB,INCLUDE_DIRECTORIES>)
target_compile_definitions(svdb-ui PRIVATE $<TARGET_PROPERTY:SampleVectorDB,COMPILE_DEFINITIONS>)
target_link_libraries(svdb-ui PRIVATE SampleVectorDB)
svdb_copy_runtime_dlls(svdb-ui "$<TARGET_FILE_DIR:svdb-ui>")

# ---------------------------------------------------------------- Unit tests
juce_add_console_app(svdb-tests PRODUCT_NAME "svdb-tests")
target_sources(svdb-tests PRIVATE tools/svdb-tests/Main.cpp)
target_include_directories(svdb-tests PRIVATE src)
target_link_libraries(svdb-tests
    PRIVATE
        svdb_core
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
svdb_copy_runtime_dlls(svdb-tests "$<TARGET_FILE_DIR:svdb-tests>")
