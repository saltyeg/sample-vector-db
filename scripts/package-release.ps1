# Packages a self-contained Windows x64 release zip from build/Release.
# Usage: .\scripts\package-release.ps1 [-Version "0.1.0"] [-OutDir "build\release-pkg"]
param(
    [string]$Version = "0.1.0",
    [string]$OutDir = ""
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $repoRoot "build\Release"
if (-not $OutDir) { $OutDir = Join-Path $repoRoot "build\release-pkg" }

$vst3Src = Join-Path $buildDir "SampleVectorDB_artefacts\Release\VST3\sample-vector-db.vst3"
$standaloneDir = Join-Path $buildDir "SampleVectorDB_artefacts\Release\Standalone"
$cliDir = Join-Path $buildDir "svdb-index_artefacts\Release"

foreach ($p in @($vst3Src, (Join-Path $standaloneDir "sample-vector-db.exe"), (Join-Path $cliDir "svdb-index.exe"))) {
    if (-not (Test-Path $p)) { throw "Missing build artifact: $p (run .\build.ps1 first)" }
}

$pkgName = "sample-vector-db-v$Version-win64"
$stage = Join-Path $OutDir $pkgName
$zipPath = Join-Path $OutDir "$pkgName.zip"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

New-Item -ItemType Directory -Path $stage, "$stage\bin" -Force | Out-Null

# 1. VST3 bundle (clean, no stray DLLs inside)
Copy-Item $vst3Src -Destination "$stage\sample-vector-db.vst3" -Recurse -Force
Get-ChildItem "$stage\sample-vector-db.vst3" -Filter "*.dll" -Recurse | Remove-Item -Force -ErrorAction SilentlyContinue

# 2. Standalone app + CLI indexer sharing one copy of ONNX Runtime + DirectML DLLs
Copy-Item (Join-Path $standaloneDir "sample-vector-db.exe") "$stage\bin\" -Force
Copy-Item (Join-Path $cliDir "svdb-index.exe")              "$stage\bin\" -Force
Copy-Item (Join-Path $standaloneDir "onnxruntime.dll")      "$stage\bin\" -Force
Copy-Item (Join-Path $standaloneDir "DirectML.dll")         "$stage\bin\" -Force

# 3. License
Copy-Item (Join-Path $repoRoot "LICENSE") "$stage\LICENSE" -Force

# 4. 1-click Windows installer script
@'
# Installs sample-vector-db VST3, ONNX Runtime/DirectML DLLs, and optionally downloads CLAP models.
# Run in PowerShell (run as Administrator if installing VST3 to C:\Program Files\Common Files\VST3).
param([switch]$SkipModels)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot

# 1. Install ONNX Runtime + DirectML DLLs into %APPDATA%\sample-vector-db\runtime
$runtimeDst = Join-Path $env:APPDATA "sample-vector-db\runtime"
New-Item -ItemType Directory -Path $runtimeDst -Force | Out-Null
Copy-Item (Join-Path $here "bin\onnxruntime.dll") $runtimeDst -Force
Copy-Item (Join-Path $here "bin\DirectML.dll")    $runtimeDst -Force
Write-Host "[OK] Installed ONNX Runtime + DirectML DLLs to $runtimeDst"

# 2. Install VST3 bundle to C:\Program Files\Common Files\VST3
$vst3Dst = "C:\Program Files\Common Files\VST3\sample-vector-db.vst3"
try {
    if (Test-Path $vst3Dst) { Remove-Item $vst3Dst -Recurse -Force }
    Copy-Item (Join-Path $here "sample-vector-db.vst3") $vst3Dst -Recurse -Force
    Write-Host "[OK] Installed VST3 plugin to $vst3Dst"
} catch {
    Write-Warning "Could not write to $vst3Dst (re-run PowerShell as Administrator, or copy sample-vector-db.vst3 there manually)."
}

# 3. Download Xenova/clap-htsat-unfused ONNX models (~590 MB)
if (-not $SkipModels) {
    Write-Host "Downloading CLAP ONNX models (~590 MB)..."
    & (Join-Path $here "bin\svdb-index.exe") download-models
}

Write-Host ""
Write-Host "Done! Open bin\sample-vector-db.exe or scan VST3 plugins in FL Studio."
'@ | Set-Content -Path "$stage\install.ps1" -Encoding UTF8

# 5. Quick-start readme
@"
sample-vector-db v$Version (Windows x64)
========================================

Quick Start:
  1. Run in an Administrator PowerShell:
       powershell -ExecutionPolicy Bypass -File .\install.ps1
     This copies sample-vector-db.vst3 into C:\Program Files\Common Files\VST3,
     installs ONNX Runtime + DirectML DLLs into %APPDATA%\sample-vector-db\runtime,
     and downloads the CLAP ONNX models (~590 MB) for semantic search & auto-tagging.

  2. Index your sample library (either click [Index] inside the app, or via CLI):
       .\bin\svdb-index.exe scan "C:\path\to\your\Samples"

  3. Use in FL Studio:
       - Options -> Manage Plugins -> Find installed plugins
       - Load "sample-vector-db" under Generators -> VST3
       - Or run bin\sample-vector-db.exe and toggle [Pin] to float over FL Studio.

Controls:
  Left-click       Select + audition sample
  Right-drag       Scrub-audition across the 2D point cloud (single-voice)
  Middle-drag      Pan map (wheel zooms, R resets view)
  Left-drag        Drag sample directly into FL Studio Channel Rack / Playlist
  Double-click     Find Similar samples (Spacebar clears)
  Click waveform   Start audition playback at clicked position
  Click header     Sort browser by Name / Pack / Cat / Note / Len / Sim
"@ | Set-Content -Path "$stage\QUICKSTART.txt" -Encoding UTF8

Compress-Archive -Path $stage -DestinationPath $zipPath -CompressionLevel Optimal
$mb = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "Created release archive: $zipPath ($mb MB)"
