@echo off
rem Builds srhpm_dgpu.dll (nvcc) and the test / bandwidth tools (clang++) in this directory.
rem Needs: CUDA toolkit, LLVM, and LLAMA set to the SR-HPM llama.cpp checkout (for dgpu_bw.cpp's kernel headers).
rem Run from a "x64 Native Tools Command Prompt for VS 2022".
cd /d %~dp0
if "%LLAMA%"=="" set LLAMA=%~dp0..\llama.cpp-srhpm
nvcc -O3 -arch=native --shared -o srhpm_dgpu.dll dgpu.cu -cudart static || exit /b 1
clang++ -O3 -march=native -std=c++17 -Wno-deprecated-declarations -I%LLAMA%\ggml\src -I%LLAMA%\ggml\include -I%LLAMA%\ggml\src\ggml-cpu dgpu_bw.cpp -o dgpu_bw.exe || exit /b 1
clang++ -O2 -std=c++17 dgpu_test.cpp -o dgpu_test.exe || exit /b 1
echo BUILD_OK
