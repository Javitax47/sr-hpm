# dGPU co-verifier

While the CPU computes the layers that live in RAM, the discrete GPU is idle. `srhpm_dgpu.dll` lets it compute the
first rows of each IQ4_XS / Q3_K matrix product of those layers, reading a pinned copy of the rows straight from RAM
over PCIe (zero copy, no VRAM used for weights), while the CPU threads compute the remaining rows.

- On the reference laptop (PCIe 3.0 x8) the GPU adds ~6–7 GB/s of weight reads without slowing the CPU.
- With 4 tokens per pass the CPU already saturates the RAM (35 of ~38 GB/s), so PCIe only competes with it; from 6
  tokens on the CPU is compute-bound and the GPU's rows are a net gain (−15 % per product at 8 tokens). The launcher
  sets `GGML_SRV_DGPU=0,0,0,0,0,0.12,0.16,0.2` (share of rows for 1..8 tokens).
- The GPU rows use exact float activations (the CPU path uses 8-bit activations), so they are slightly *more*
  precise; the greedy output may differ by rounding.

`ggml-cpu` loads the DLL at run time from its own directory (`ggml/src/ggml-cpu/srv-dgpu.inc` in the patch). The
Windows installer builds it when the CUDA toolkit is present:

```powershell
nvcc -O3 -arch=native --shared -cudart static -o <llama.cpp>\build\bin\srhpm_dgpu.dll dgpu.cu
```

Tests (correctness against ggml's dequantization, and bandwidth): `dgpu_test.cpp`, `dgpu_bw.cpp`
(`build.bat` builds them with clang++ next to the DLL).
