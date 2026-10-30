// DRAM read-bandwidth ceiling on this machine: N threads sweep a 4 GiB buffer with AVX2 loads, handing out
// chunks dynamically (like the SR-HPM kernels do with rows). Variants: chunk size, software prefetch distance.
// build: clang-cl /O2 /arch:AVX2 /EHsc bw_bench.cpp
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double run(const uint8_t * buf, size_t n, int nth, size_t chunk, int pf) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    std::atomic<long long> sink{0};
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < nth; t++) {
        th.emplace_back([&] {
            __m256i acc = _mm256_setzero_si256();
            for (;;) {
                size_t c = next.fetch_add(chunk);
                if (c >= n) break;
                size_t e = c + chunk < n ? c + chunk : n;
                for (size_t i = c; i < e; i += 128) {
                    if (pf) _mm_prefetch((const char *) buf + i + pf, _MM_HINT_T0);
                    __m256i a = _mm256_load_si256((const __m256i *) (buf + i));
                    __m256i b = _mm256_load_si256((const __m256i *) (buf + i + 32));
                    __m256i c2 = _mm256_load_si256((const __m256i *) (buf + i + 64));
                    __m256i d = _mm256_load_si256((const __m256i *) (buf + i + 96));
                    acc = _mm256_add_epi8(acc, _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c2, d)));
                }
            }
            sink += _mm256_extract_epi64(acc, 0);
        });
    }
    for (auto & x : th) x.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return n / s / 1e9;
}

int main() {
    const size_t n = (size_t) 4 << 30;
    uint8_t * buf = (uint8_t *) _aligned_malloc(n, 4096);
    for (size_t i = 0; i < n; i += 64) buf[i] = (uint8_t) i;
    run(buf, n, 16, 1 << 20, 0);  // warm-up
    int threads[] = {4, 8, 12, 16};
    size_t chunks[] = {16 << 10, 64 << 10, 256 << 10, 4 << 20};
    int pfs[] = {0, 512, 2048};
    for (int pf : pfs)
        for (size_t ch : chunks)
            for (int t : threads) {
                double best = 0;
                for (int r = 0; r < 3; r++) { double g = run(buf, n, t, ch, pf); if (g > best) best = g; }
                printf("pf %5d chunk %6zu KiB threads %2d : %.1f GB/s\n", pf, ch >> 10, t, best);
                fflush(stdout);
            }
}
