// Test of the llama.cpp integration code for the integrated GPU (ggml-cpu/srv-igpu.inc): launch + collect on random
// IQ4_XS and C4 (IQ3_S, IQ3_XXS) weights against a scalar f32 reference, then the time of one call.
// build: clang++ -O2 -std=c++17 -I<llama.cpp>/ggml/src -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src/ggml-cpu igpu_test.cpp
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
static inline float fp16_to_f32(ggml_half h) { return _cvtsh_ss(h); }
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_f32(x)
#include "srv-kernels.inc"
#include "srv-igpu.inc"
using namespace ggml::cpu::repack;

// f32 value of weight k of a row, for the two block layouts
static float w_iq4(const block_iq4_xs * row, int64_t k) {
    const block_iq4_xs & x = row[k / 256];
    const int b = (int) (k % 256) / 32, j = (int) (k % 32);
    const int ls = ((x.scales_l[b / 2] >> 4 * (b % 2)) & 0xf) | (((x.scales_h >> 2 * b) & 3) << 4);
    const uint8_t c = x.qs[16 * b + (j % 16)];
    return fp16_to_f32(x.d) * (ls - 32) * kvalues_iq4nl[j < 16 ? (c & 15) : (c >> 4)];
}
static float w_c4(const block_srv_c4 * row, int64_t k, int kind) {
    const block_srv_c4 & x = row[k / 256];
    const int b = (int) (k % 256) / 32, j = (int) (k % 32);
    const uint8_t c = x.qs[16 * b + (j % 16)];
    return x.d * x.sc[b] * srv_c4_tab[kind][j < 16 ? (c & 15) : (c >> 4)];
}

template <class B, class F>
static void check(const char * name, int kind, std::vector<B> & W, int64_t N, int64_t K, int M, F wval) {
    srv_igpu_t & g = srv_igpu();
    std::vector<float> X(M * K);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : X) v = nd(rng);
    const int64_t nr = (int64_t) (0.5 * N) / 64 * 64;
    std::vector<float> dst((size_t) M * N, -999.f);
    auto t0 = std::chrono::steady_clock::now();
    const bool ok = g.launch(kind, W.data(), sizeof(B), N, K, M, nr, X.data(), K * sizeof(float));
    g.collect(M, nr, (char *) dst.data(), N * sizeof(float));
    const double first = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 10; r++) {
        g.launch(kind, W.data(), sizeof(B), N, K, M, nr, X.data(), K * sizeof(float));
        g.collect(M, nr, (char *) dst.data(), N * sizeof(float));
    }
    const double each = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 10;
    double md = 0, mv = 0;
    for (int64_t r = 0; r < nr; r += std::max<int64_t>(1, nr / 97)) {
        for (int m = 0; m < M; m++) {
            double ref = 0;
            for (int64_t k = 0; k < K; k++) ref += (double) wval(&W[r * (K / 256)], k) * X[m * K + k];
            md = std::max(md, std::fabs(ref - dst[(size_t) m * N + r]));
            mv = std::max(mv, std::fabs(ref));
        }
    }
    printf("%-8s M=%d launch %d  first call %.0f ms (builds the copy)  then %.2f ms/call = %.1f GB/s of its rows   max rel err %.2g\n",
           name, M, (int) ok, first * 1e3, each * 1e3, (double) nr * (K / 256) * sizeof(B) / each / 1e9, md / mv);
}

// a big IQ4_XS matrix (like ffn_up, 17408 x 5120) with 8 tokens, 35 % of the rows on the iGPU, called in three
// regimes: back to back; with 3 ms idle between calls; with the 8 CPU cores busy (AVX2) while the iGPU runs
static void regimes() {
    srv_igpu_t & g = srv_igpu();
    const int64_t K = 5120, N = 17408, nsb = K / 256, M = 8;
    std::vector<block_iq4_xs> W(N * nsb);
    std::mt19937 rng(3);
    for (auto & b : W) { uint8_t * p = (uint8_t *) &b; for (size_t i = 0; i < sizeof(b); i++) p[i] = (uint8_t) rng(); b.d = _cvtss_sh(0.01f, 0); }
    std::vector<float> X(M * K, 0.5f), dst((size_t) M * N);
    const int64_t nr = (int64_t) (0.35 * N) / 64 * 64;
    g.launch(0, W.data(), sizeof(block_iq4_xs), N, K, M, nr, X.data(), K * 4);
    g.collect(M, nr, (char *) dst.data(), N * 4);
    auto spin_cpu = [](double ms) {
        std::atomic<bool> stop{false};
        std::vector<std::thread> th;
        for (int i = 0; i < 8; i++) th.emplace_back([&] { __m256i a = _mm256_set1_epi32(i); while (!stop) for (int j = 0; j < 1000; j++) a = _mm256_madd_epi16(a, a); volatile int s = _mm256_extract_epi32(a, 0); (void) s; });
        std::this_thread::sleep_for(std::chrono::microseconds((int) (ms * 1000)));
        stop = true;
        for (auto & t : th) t.join();
    };
    for (int mode = 0; mode < 3; mode++) {
        const auto t0 = std::chrono::steady_clock::now();
        double busy = 0;
        for (int r = 0; r < 50; r++) {
            const auto a = std::chrono::steady_clock::now();
            g.launch(0, W.data(), sizeof(block_iq4_xs), N, K, M, nr, X.data(), K * 4);
            if (mode == 2) spin_cpu(1.0);
            g.collect(M, nr, (char *) dst.data(), N * 4);
            busy += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            if (mode == 1) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        (void) t0;
        const double per = busy / 50;
        printf("%-28s %.2f ms per call = %.1f GB/s on its %lld rows\n", mode == 0 ? "back to back" : mode == 1 ? "3 ms idle between calls" : "CPU cores busy meanwhile",
               per * 1e3, (double) nr * nsb * sizeof(block_iq4_xs) / per / 1e9, (long long) nr);
    }
}

int main(int argc, char ** argv) {
    if (!srv_igpu().ok) { printf("iGPU not available (set GGML_SRV_IGPU=1)\n"); return 1; }
    if (argc > 1 && std::string(argv[1]) == "regimes") { regimes(); return 0; }
    const int64_t K = 5120, N = 4096, nsb = K / 256;
    std::mt19937 rng(1);
    std::vector<block_iq4_xs> W4(N * nsb);
    for (auto & b : W4) { uint8_t * p = (uint8_t *) &b; for (size_t i = 0; i < sizeof(b); i++) p[i] = (uint8_t) rng(); b.d = _cvtss_sh(0.01f, 0); }
    std::vector<block_iq3_s> W3(N * nsb);
    for (auto & b : W3) { uint8_t * p = (uint8_t *) &b; for (size_t i = 0; i < sizeof(b); i++) p[i] = (uint8_t) rng(); b.d = _cvtss_sh(0.01f, 0); }
    std::vector<block_srv_c4> C3(N * nsb);
    for (int64_t i = 0; i < N * nsb; i++) srv_to_c4<srv_iq3_s, 0>(&W3[i], &C3[i]);
    for (int M : { 5, 8 }) {
        check("IQ4_XS", 0, W4, N, K, M, [](const block_iq4_xs * r, int64_t k) { return w_iq4(r, k); });
        check("C4/IQ3S", 1, C3, N, K, M, [](const block_srv_c4 * r, int64_t k) { return w_c4(r, k, 0); });
    }
}
