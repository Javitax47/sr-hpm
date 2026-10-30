// Can the discrete GPU add weight-read throughput over PCIe while the CPU kernels are compute-bound (8 tokens)?
// Measures: dGPU streaming read of pinned host memory alone, CPU SR-HPM kernel at T=8 alone, and both at once
// on separate 1 GB buffers. build: f3/dgpu/build.bat
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <windows.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
static inline float fp16_to_f32(ggml_half h) { return _cvtsh_ss(h); }
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_f32(x)
#include "srv-kernels.inc"
using namespace ggml::cpu::repack;

int main(int argc, char ** argv) {
    HMODULE h = LoadLibraryA("srhpm_dgpu.dll");
    if (!h) { printf("no srhpm_dgpu.dll\n"); return 1; }
    auto init  = (int (*)()) GetProcAddress(h, "dgpu_init");
    auto halloc = (void * (*)(size_t)) GetProcAddress(h, "dgpu_host_alloc");
    auto rd    = (int (*)(void *, size_t, int, int)) GetProcAddress(h, "dgpu_read");
    if (init()) { printf("init failed\n"); return 1; }
    const size_t GB = (size_t) 1 << 30;
    uint8_t * hp = (uint8_t *) halloc(GB);
    if (!hp) { printf("host alloc failed\n"); return 1; }
    for (size_t i = 0; i < GB; i += 64) hp[i] = (uint8_t) i;

    const int blocks = argc > 1 ? atoi(argv[1]) : 120, threads = 256;
    rd(hp, GB, blocks, threads);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; r++) rd(hp, GB, blocks, threads);
    const double g_alone = 5.0 * GB / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9;
    printf("dGPU (PCIe, zero copy) alone: %.1f GB/s\n", g_alone);
    // small reads: latency of one ~10 MB slice (the share of one matrix)
    const size_t sl = 10u << 20;
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 100; r++) rd(hp + (r % 50) * sl, sl, blocks, threads);
    const double us = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 100 * 1e6;
    printf("dGPU 10 MB slice: %.0f us (%.1f GB/s)\n", us, sl / us / 1e3);

    using B = block_iq4_xs;
    const int64_t K = 5120, nsb = K / QK_K, N = (int64_t) (GB / (nsb * sizeof(B)));
    std::vector<B> W(N * nsb);
    std::mt19937 rng(1);
    for (auto & b : W) { uint8_t * p = (uint8_t *) &b; for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng(); }
    std::vector<int8_t> xq(8 * K, 3);
    std::vector<float> xs(8 * nsb, 1.f), out_c(N * 8);
    const int8_t * xp[8]; const float * sp[8];
    for (int t = 0; t < 8; t++) { xp[t] = xq.data() + t * K; sp[t] = xs.data() + t * nsb; }
    auto cpu_pass = [&] {
        std::atomic<int64_t> next{0};
        std::vector<std::thread> th;
        for (int i = 0; i < 8; i++)
            th.emplace_back([&] {
                for (int64_t s; (s = next.fetch_add(16)) < N;)
                    for (int64_t r = s; r < std::min<int64_t>(N, s + 16); r++) srv_row<srv_iq4_xs_p, 8>(W.data() + r * nsb, nsb, xp, sp, out_c.data() + r * 8);
            });
        for (auto & t : th) t.join();
    };
    cpu_pass();
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 3; r++) cpu_pass();
    const double c_alone = 3.0 * GB / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9;
    printf("CPU T=8 alone: %.1f GB/s\n", c_alone);

    std::atomic<bool> stop{false};
    std::atomic<long long> gbytes{0};
    std::thread gt([&] { while (!stop) { rd(hp, GB, blocks, threads); gbytes += GB; } });
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 3; r++) cpu_pass();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop = true;
    gt.join();
    printf("together: CPU %.1f GB/s + dGPU %.1f GB/s = %.1f GB/s\n", 3.0 * GB / sec / 1e9, gbytes / sec / 1e9,
           (3.0 * GB + gbytes) / sec / 1e9);
}
