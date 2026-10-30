# SR-HPM installer for Windows: llama.cpp at the pinned commit + the SR-HPM patch, built with CUDA when the CUDA
# toolkit is installed, the CPU backend rebuilt with clang-cl (-march=native) when LLVM is installed, the dGPU
# co-verifier DLL, and the Python launcher's dependencies.
#
# Requirements: Git, CMake >= 3.24, Visual Studio 2022 (or Build Tools) with "Desktop development with C++",
# Python >= 3.10. Optional: NVIDIA CUDA Toolkit 12.x (GPU offload), LLVM/clang-cl (~10-20 % faster CPU kernels).
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1 [-Dest <dir>] [-NoCuda] [-Jobs 16]
#       [-Upstream <git url or local clone>] [-NoPip]
param(
    [string]$Dest = "",
    [switch]$NoCuda,
    [int]$Jobs = [Environment]::ProcessorCount,
    [string]$Upstream = "https://github.com/ggml-org/llama.cpp.git",
    [switch]$NoPip
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $Dest) { $Dest = Join-Path $Repo "llama.cpp-srhpm" }
$Commit = (Get-Content (Join-Path $Repo "patch\UPSTREAM_COMMIT") -First 1).Trim()
$Patch = Join-Path $Repo "patch\srhpm.patch"

function Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Need($exe, $hint) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { throw "$exe not found: $hint" }
}
Need git "install Git for Windows"
Need cmake "install CMake (cmake.org) and add it to PATH"

# Visual Studio environment (cl, link, ninja) through vswhere
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Visual Studio 2022 / Build Tools not found" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio C++ tools not found (workload 'Desktop development with C++')" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
Step "Visual Studio: $vs"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) }
}

Step "llama.cpp $Commit -> $Dest"
if (-not (Test-Path (Join-Path $Dest ".git"))) {
    git clone --no-checkout $Upstream $Dest
}
git -C $Dest fetch --quiet origin $Commit
git -C $Dest checkout --quiet --force $Commit
git -C $Dest clean -fdq -e build -e build-clang
Step "applying the SR-HPM patch"
git -C $Dest apply --whitespace=nowarn $Patch

$cuda = (-not $NoCuda) -and (Get-Command nvcc -ErrorAction SilentlyContinue)
$cmakeArgs = @("-S", $Dest, "-B", "$Dest\build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF",
          "-DLLAMA_BUILD_TESTS=OFF", "-DGGML_NATIVE=ON")
if ($cuda) { $cmakeArgs += @("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=native") }
Step ("configure (" + ($(if ($cuda) { "CUDA" } else { "CPU only" })) + ")")
cmake @cmakeArgs
Step "build"
cmake --build "$Dest\build" --config Release -j $Jobs --target llama-server llama-cli llama-bench llama-quantize llama-perplexity
$bin = "$Dest\build\bin"

# The SR-HPM CPU kernels are 10-20 % faster with clang-cl than with MSVC: rebuild only ggml-cpu with clang-cl
$clang = Get-Command clang-cl -ErrorAction SilentlyContinue
if (-not $clang -and (Test-Path "$env:ProgramFiles\LLVM\bin\clang-cl.exe")) { $env:PATH = "$env:ProgramFiles\LLVM\bin;$env:PATH"; $clang = $true }
if ($clang) {
    Step "ggml-cpu with clang-cl (-march=native)"
    cmake -S $Dest -B "$Dest\build-clang" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl `
        -DCMAKE_CXX_COMPILER=clang-cl -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF `
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    cmake --build "$Dest\build-clang" --config Release -j $Jobs --target ggml-cpu
    Copy-Item "$Dest\build-clang\bin\ggml-cpu.dll" "$bin\ggml-cpu.dll" -Force
} else {
    Write-Host "    LLVM not found: keeping the MSVC ggml-cpu (install LLVM for faster CPU kernels)" -ForegroundColor Yellow
}

if ($cuda) {
    Step "dGPU co-verifier (srhpm_dgpu.dll)"
    Push-Location (Join-Path $Repo "dgpu")
    nvcc -O3 -arch=native --shared -cudart static -o "$bin\srhpm_dgpu.dll" dgpu.cu
    Pop-Location
}

if (-not $NoPip) {
    Step "Python dependencies"
    python -m pip install --quiet -r (Join-Path $Repo "requirements.txt")
}

Write-Host ""
Write-Host "SR-HPM is ready: $bin\llama-server.exe" -ForegroundColor Green
Write-Host "  python -m srhpm.launch <model.gguf> [--draft <drafter.gguf>] --run"
Write-Host "  (set SRHPM_LLAMA_DIR=$Dest if you installed outside the repository)"
Write-Host "  Optional: scripts\grant_large_pages.ps1 (as administrator) for 2 MB pages"
