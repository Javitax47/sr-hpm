// srhpm_dgpu.dll kernels: correctness against a CPU dequantization (ggml's algorithms) and speed over PCIe.
// usage: dgpu_test.exe [K] [rows]   build: f3/dgpu/build.bat
#include <windows.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m | 1024), (int) e - 25);
    return s ? -v : v;
}
static const int8_t kv[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

// ggml dequantize_row_iq4_xs
static void deq_iq4_xs(const uint8_t * b, float * y) {
    uint16_t d; memcpy(&d, b, 2);
    uint16_t sh; memcpy(&sh, b + 2, 2);
    const uint8_t * sl = b + 4, * qs = b + 8;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
        const float dl = h2f(d) * (ls - 32);
        for (int j = 0; j < 16; j++) {
            y[32 * ib + j]      = dl * kv[qs[16 * ib + j] & 0xf];
            y[32 * ib + j + 16] = dl * kv[qs[16 * ib + j] >> 4];
        }
    }
}

// ggml dequantize_row_q3_K (aux / kmask version)
static void deq_q3_k(const uint8_t * b, float * y) {
    const uint8_t * hm = b, * q = b + 32;
    uint32_t aux[4];
    memcpy(aux, b + 96, 12);
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t * sc = (const int8_t *) aux;
    uint16_t dh; memcpy(&dh, b + 108, 2);
    const float d = h2f(dh);
    int is = 0; uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d * (sc[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
            dl = d * (sc[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2; m <<= 1;
        }
        q += 32;
    }
}

int main(int argc, char ** argv) {
    HMODULE h = LoadLibraryA("srhpm_dgpu.dll");
    if (!h) { printf("no srhpm_dgpu.dll\n"); return 1; }
    auto init    = (int (*)()) GetProcAddress(h, "dgpu_init");
    auto launch  = (int (*)(int, const void *, const void *, int64_t, int64_t, int, const float *, size_t, size_t)) GetProcAddress(h, "dgpu_launch");
    auto collect = (int (*)(int, int64_t, char *, size_t)) GetProcAddress(h, "dgpu_collect");
    if (init()) { printf("init failed\n"); return 1; }
    const int64_t K = argc > 1 ? atoll(argv[1]) : 5120, R = argc > 2 ? atoll(argv[2]) : 4352, nsb = K / 256;
    std::mt19937 rng(7);
    for (int kind = 0; kind < 2; kind++) {
        const size_t bsz = kind == 0 ? 136 : 110;
        std::vector<uint8_t> w(R * nsb * bsz);
        for (auto & c : w) c = (uint8_t) rng();
        for (int64_t i = 0; i < R * nsb; i++) {  // sane fp16 scales (~1e-3)
            const uint16_t d = (uint16_t) (0x1400 + (rng() & 0x3ff));
            memcpy(&w[i * bsz + (kind == 0 ? 0 : 108)], &d, 2);
        }
        std::vector<float> x(8 * K), y(8 * R), ref(8 * R), deq(K);
        std::normal_distribution<float> nd;
        for (auto & v : x) v = nd(rng);
        for (int64_t r = 0; r < R; r++) {
            for (int64_t s = 0; s < nsb; s++) (kind == 0 ? deq_iq4_xs : deq_q3_k)(&w[(r * nsb + s) * bsz], &deq[s * 256]);
            for (int t = 0; t < 8; t++) { double a = 0; for (int64_t k = 0; k < K; k++) a += (double) deq[k] * x[t * K + k]; ref[t * R + r] = (float) a; }
        }
        for (int M : { 1, 4, 8 }) {
            if (launch(kind, w.data(), w.data(), R, K, M, x.data(), K * 4, (size_t) 4 << 30)) { printf("launch failed\n"); return 1; }
            collect(M, R, (char *) y.data(), R * 4);
            double err = 0, nrm = 0;
            for (int t = 0; t < M; t++) for (int64_t r = 0; r < R; r++) { err += std::pow(y[t * R + r] - ref[t * R + r], 2); nrm += std::pow(ref[t * R + r], 2); }
            // speed (weights already pinned)
            const int reps = 20;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; i++) { launch(kind, w.data(), w.data(), R, K, M, x.data(), K * 4, (size_t) 4 << 30); collect(M, R, (char *) y.data(), R * 4); }
            const double us = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps * 1e6;
            printf("%s K=%lld rows=%lld M=%d: rel err %.2e, %.0f us, %.1f GB/s\n", kind == 0 ? "IQ4_XS" : "Q3_K", (long long) K, (long long) R, M,
                   std::sqrt(err / nrm), us, w.size() / us / 1e3);
        }
    }
}
