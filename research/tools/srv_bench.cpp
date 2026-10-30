// Standalone check + throughput benchmark of the SR-HPM small-batch kernels (llama.cpp srv-kernels.inc).
// For each type: random weights, scalar reference (dequantize as in ggml-quants.c, float dot), then
// GB/s of weights for T = 1, 2, 4, 8 tokens with all threads.
// build: clang++ -O3 -march=native -std=c++17 -I<llama.cpp>/ggml/src -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src/ggml-cpu f3/srv_bench.cpp -o srv_bench.exe
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <string>
#include <type_traits>
#include <vector>

static inline float fp16_to_f32(ggml_half h) { return _cvtsh_ss(h); }
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_f32(x)
#include "srv-kernels.inc"

using namespace ggml::cpu::repack;

// ---- scalar references (same formulas as ggml-quants.c dequantize_row_*) ----
static void ref_iq4_xs(const block_iq4_xs * x, float * y) {
    const float d = fp16_to_f32(x->d);
    for (int ib = 0; ib < 8; ib++) {
        const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
        for (int j = 0; j < 16; j++) {
            y[32 * ib + j]      = d * (ls - 32) * kvalues_iq4nl[x->qs[16 * ib + j] & 0xf];
            y[32 * ib + j + 16] = d * (ls - 32) * kvalues_iq4nl[x->qs[16 * ib + j] >> 4];
        }
    }
}
static void ref_iq3_s(const block_iq3_s * x, float * y) {
    const float d = fp16_to_f32(x->d);
    const uint8_t *qs = x->qs, *qh = x->qh, *signs = x->signs;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const float db[2] = { d * (1 + 2 * (x->scales[ib32 / 2] & 0xf)), d * (1 + 2 * (x->scales[ib32 / 2] >> 4)) };
        for (int h = 0; h < 2; h++) {
            for (int l = 0; l < 4; l++) {
                const uint8_t * g1 = (const uint8_t *) (iq3s_grid + (qs[2 * l + 0] | ((qh[h] << (8 - 2 * l)) & 256)));
                const uint8_t * g2 = (const uint8_t *) (iq3s_grid + (qs[2 * l + 1] | ((qh[h] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    y[j + 0] = db[h] * g1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db[h] * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}
static void ref_iq3_xxs(const block_iq3_xxs * x, float * y) {
    const float     d  = fp16_to_f32(x->d);
    const uint8_t * qs = x->qs;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux;
        memcpy(&aux, x->qs + 64 + 4 * ib32, 4);
        const float db = d * (0.5f + (aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            const uint8_t   s  = ksigns_iq2xs[(aux >> 7 * l) & 127];
            const uint8_t * g1 = (const uint8_t *) (iq3xxs_grid + qs[2 * l + 0]);
            const uint8_t * g2 = (const uint8_t *) (iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; j++) {
                y[j + 0] = db * g1[j] * (s & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                y[j + 4] = db * g2[j] * (s & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
    }
}
static void ref_q3_k(const block_q3_K * x, float * y) {
    srv_q3_k::sb_t s;
    const float    d = srv_q3_k::prep(x, s);  // scale unpacking is copied verbatim from dequantize_row_q3_K
    const uint8_t *q = x->qs, *hm = x->hmask;
    uint8_t        m  = 1;
    int            is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            float dl = d * s.sc[is++];
            for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t) ((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
            dl = d * s.sc[is++];
            for (int l = 0; l < 16; l++) *y++ = dl * ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

static void ref_iq2_xxs(const block_iq2_xxs * x, float * y) {
    const float d = fp16_to_f32(x->d);
    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux32[2];
        memcpy(aux32, x->qs + 4 * ib32, 8);
        const uint8_t * aux8 = (const uint8_t *) aux32;
        const float     db   = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const uint8_t * grid  = (const uint8_t *) (iq2xxs_grid + aux8[l]);
            const uint8_t   signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
            for (int j = 0; j < 8; j++) y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}
static void ref_iq2_xs(const block_iq2_xs * x, float * y) {
    const float d = fp16_to_f32(x->d);
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const float db[2] = { d * (0.5f + (x->scales[ib32] & 0xf)) * 0.25f, d * (0.5f + (x->scales[ib32] >> 4)) * 0.25f };
        for (int l = 0; l < 4; l++) {
            const uint8_t * grid  = (const uint8_t *) (iq2xs_grid + (x->qs[4 * ib32 + l] & 511));
            const uint8_t   signs = ksigns_iq2xs[x->qs[4 * ib32 + l] >> 9];
            for (int j = 0; j < 8; j++) y[j] = db[l / 2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}
static void ref_iq2_s(const block_iq2_s * x, float * y) {
    const float     d  = fp16_to_f32(x->d);
    const uint8_t * qs = x->qs;
    const uint8_t * qh = x->qh;
    const uint8_t * signs = qs + 32;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const float db[2] = { d * (0.5f + (x->scales[ib32] & 0xf)) * 0.25f, d * (0.5f + (x->scales[ib32] >> 4)) * 0.25f };
        for (int l = 0; l < 4; l++) {
            const uint8_t * grid = (const uint8_t *) (iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
            for (int j = 0; j < 8; j++) y[j] = db[l / 2] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

template <class D> struct ref;
template <> struct ref<srv_iq4_xs>  { static void f(const block_iq4_xs * x, float * y) { ref_iq4_xs(x, y); } static constexpr int doff = 0; };
template <> struct ref<srv_iq3_s>   { static void f(const block_iq3_s * x, float * y) { ref_iq3_s(x, y); } static constexpr int doff = 0; };
template <> struct ref<srv_iq3_xxs> { static void f(const block_iq3_xxs * x, float * y) { ref_iq3_xxs(x, y); } static constexpr int doff = 0; };
template <> struct ref<srv_iq2_xxs> { static void f(const block_iq2_xxs * x, float * y) { ref_iq2_xxs(x, y); } static constexpr int doff = 0; };
template <> struct ref<srv_iq2_xs>  { static void f(const block_iq2_xs * x, float * y) { ref_iq2_xs(x, y); } static constexpr int doff = 0; };
template <> struct ref<srv_iq2_s>   { static void f(const block_iq2_s * x, float * y) { ref_iq2_s(x, y); } static constexpr int doff = 0; };
template <> struct ref<srv_q3_k>    { static void f(const block_q3_K * x, float * y) { ref_q3_k(x, y); } static constexpr int doff = offsetof(block_q3_K, d); };

template <class D, int T>
static void run_rows(const typename D::block * W, int64_t r0, int64_t r1, int64_t nsb, const int8_t * xq, const float * xs,
                     float * out) {
    const int8_t * xp[T];
    const float *  sp[T];
    for (int t = 0; t < T; t++) { xp[t] = xq + t * nsb * QK_K; sp[t] = xs + t * nsb; }
    for (int64_t r = r0; r < r1; r++) srv_row<D, T>(W + r * nsb, nsb, xp, sp, out + r * T);
}

template <class D>
static void bench(const char * name, int nth) {
    using B          = typename D::block;
    const int64_t K = 5120, nsb = K / QK_K;
    const int64_t N = (int64_t) (512e6 / (nsb * sizeof(B)));  // ~512 MB of weights, far beyond the caches
    std::vector<B> W(N * nsb);
    std::mt19937   rng(1);
    for (auto & b : W) {
        uint8_t * p = (uint8_t *) &b;
        for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng();
        const ggml_half d = _cvtss_sh(0.01f + 0.01f * (rng() % 100) / 100.f, 0);
        memcpy(p + ref<D>::doff, &d, 2);
    }
    // activations: 8 tokens
    std::vector<float> x(8 * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    std::vector<int8_t> xq(8 * K);
    std::vector<float>  xs(8 * nsb);
    for (int t = 0; t < 8; t++) srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K);

    // correctness on the first 64 rows, T = 8
    std::vector<float> out(N * 8), y(K);
    run_rows<D, 8>(W.data(), 0, 64, nsb, xq.data(), xs.data(), out.data());
    double maxrel = 0;
    for (int r = 0; r < 64; r++) {
        for (int64_t ib = 0; ib < nsb; ib++) ref<D>::f(&W[r * nsb + ib], y.data() + ib * QK_K);
        for (int t = 0; t < 8; t++) {
            double ref_dot = 0, norm = 0;
            for (int64_t k = 0; k < K; k++) { ref_dot += (double) y[k] * x[t * K + k]; norm += std::fabs((double) y[k] * x[t * K + k]); }
            maxrel = std::max(maxrel, std::fabs(out[r * 8 + t] - ref_dot) / (norm / std::sqrt((double) K) + 1e-9));
        }
    }
    printf("%-8s N=%lld  max err (in units of a typical |term| x sqrt K) = %.4f  %s\n", name, (long long) N, maxrel,
           maxrel < 0.05 ? "OK" : "MISMATCH");

    auto timed = [&](auto fn) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (int i = 0; i < nth; i++) {
                const int64_t per = (N + nth - 1) / nth, a = i * per, b = std::min(N, a + per);
                th.emplace_back([&, a, b] { fn(a, b); });
            }
            for (auto & t : th) t.join();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    const double bytes = (double) N * nsb * sizeof(B);
    double t1 = timed([&](int64_t a, int64_t b) { run_rows<D, 1>(W.data(), a, b, nsb, xq.data(), xs.data(), out.data()); });
    double t2 = timed([&](int64_t a, int64_t b) { run_rows<D, 2>(W.data(), a, b, nsb, xq.data(), xs.data(), out.data()); });
    double t4 = timed([&](int64_t a, int64_t b) { run_rows<D, 4>(W.data(), a, b, nsb, xq.data(), xs.data(), out.data()); });
    double t8 = timed([&](int64_t a, int64_t b) { run_rows<D, 8>(W.data(), a, b, nsb, xq.data(), xs.data(), out.data()); });
    printf("%-8s threads=%d  GB/s  T=1 %.1f  T=2 %.1f  T=4 %.1f  T=8 %.1f   cost vs T=1: %.2f %.2f %.2f\n", name, nth,
           bytes / t1 / 1e9, bytes / t2 / 1e9, bytes / t4 / 1e9, bytes / t8 / 1e9, t2 / t1, t4 / t1, t8 / t1);
}


// C4 re-encoding of IQ3 types: same results as the native kernel, then throughput (GB/s of native bytes)
template <class D, int KIND>
static void bench_c4(const char * name, int nth) {
    using B          = typename D::block;
    const int64_t K = 5120, nsb = K / QK_K;
    const int64_t N = (int64_t) (512e6 / (nsb * sizeof(B)));
    std::vector<B> W(N * nsb);
    std::mt19937   rng(1);
    for (auto & b : W) {
        uint8_t * p = (uint8_t *) &b;
        for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng();
        const ggml_half d = _cvtss_sh(0.01f + 0.01f * (rng() % 100) / 100.f, 0);
        memcpy(p + ref<D>::doff, &d, 2);
    }
    std::vector<block_srv_c4> C(N * nsb);
    for (int64_t i = 0; i < N * nsb; i++) srv_to_c4<D, KIND>(&W[i], &C[i]);
    std::vector<float> x(8 * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    std::vector<int8_t> xq(8 * K);
    std::vector<float>  xs(8 * nsb);
    for (int t = 0; t < 8; t++) srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K);
    const int8_t * xp[8]; const float * sp[8];
    for (int t = 0; t < 8; t++) { xp[t] = xq.data() + t * K; sp[t] = xs.data() + t * nsb; }
    double maxd = 0, maxv = 0;
    for (int64_t r = 0; r < 256; r++) {
        float a[8], b[8];
        srv_row<D, 8>(W.data() + r * nsb, nsb, xp, sp, a);
        srv_row<srv_c4<KIND>, 8>(C.data() + r * nsb, nsb, xp, sp, b);
        for (int t = 0; t < 8; t++) { maxd = std::max(maxd, (double) std::fabs(a[t] - b[t])); maxv = std::max(maxv, (double) std::fabs(a[t])); }
    }
    printf("%-8s C4 vs native: max |diff| %.3g (max |out| %.3g)  %s\n", name, maxd, maxv, maxd <= 1e-5 * maxv ? "EXACT" : "MISMATCH");
    auto timed = [&](auto fn) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (int i = 0; i < nth; i++) {
                const int64_t per = (N + nth - 1) / nth, a = i * per, b = std::min(N, a + per);
                th.emplace_back([&, a, b] { fn(a, b); });
            }
            for (auto & t : th) t.join();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    std::vector<float> out(N * 8);
    const double native = (double) N * nsb * sizeof(B);
    auto run = [&](auto tag, int64_t a, int64_t b) {
        using DD = decltype(tag);
        const typename DD::block * base = (const typename DD::block *) (std::is_same<DD, D>::value ? (const void *) W.data() : (const void *) C.data());
        for (int64_t r = a; r < b; r++) srv_row<DD, 4>(base + r * nsb, nsb, xp, sp, out.data() + r * 8);
    };
    double tn = timed([&](int64_t a, int64_t b) { run(D{}, a, b); });
    double tc = timed([&](int64_t a, int64_t b) { run(srv_c4<KIND>{}, a, b); });
    printf("%-8s T=4 native %.1f GB/s   C4 %.1f GB/s-native-equivalent (%.2fx faster, %.2fx bytes)\n", name, native / tn / 1e9,
           native / tc / 1e9, tn / tc, (double) sizeof(block_srv_c4) / sizeof(B));
}

// cost curve T = 1..8 (time relative to T = 1, GB/s), rows handed out in chunks of 16 as in srv-verify.inc;
// random blocks (only throughput matters here; float fields set to small sane values)
template <class D, int T, bool OFFP = false>
static void rows_t(const typename D::block * W, int64_t r0, int64_t r1, int64_t nsb, const int8_t * xq, const float * xs,
                   const int16_t * xb, float * out, const int16_t * xb16 = nullptr) {
    const int8_t * xp[T]; const float * sp[T]; const int16_t * bp[T]; const int16_t * b16[T];
    for (int t = 0; t < T; t++) {
        xp[t] = xq + t * nsb * QK_K; sp[t] = xs + t * nsb; bp[t] = xb + t * nsb * 8; b16[t] = xb16 ? xb16 + t * nsb * 16 : nullptr;
    }
    for (int64_t r = r0; r < r1; r++) srv_row<D, T, OFFP>(W + r * nsb, nsb, xp, sp, out + r * 8, bp, b16);
}

template <class D, bool OFFP = false>
static void curve(const char * name, int nth, int fpos) {
    using B         = typename D::block;
    const int64_t K = 5120, nsb = K / QK_K;
    const int64_t N = (int64_t) (1024e6 / (nsb * sizeof(B)));
    std::vector<B> W(N * nsb);
    std::mt19937   rng(1);
    for (auto & b : W) {
        uint8_t * p = (uint8_t *) &b;
        for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng();
        if (fpos >= 0) { const ggml_half d = _cvtss_sh(0.01f, 0); memcpy(p + fpos, &d, 2); if (fpos + 4 <= (int) sizeof(B)) memcpy(p + fpos + 2, &d, 2); }
        else { const float d = 0.01f; memcpy(p + sizeof(B) - 4, &d, 4); }
    }
    std::vector<float> x(8 * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    std::vector<int8_t> xq(8 * K);
    std::vector<float>  xs(8 * nsb);
    std::vector<int16_t> xb(8 * K / 32), xb16(8 * K / 16);
    for (int t = 0; t < 8; t++)
        srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K, xb.data() + t * K / 32, xb16.data() + t * K / 16);
    if constexpr (OFFP) {  // exactness of the offset path against the sign path, all T
        std::vector<float> a(256 * 8), b(256 * 8);
        double maxd = 0;
        auto cmp = [&](auto tt) {
            constexpr int TT = decltype(tt)::value;
            rows_t<D, TT, false>(W.data(), 0, 256, nsb, xq.data(), xs.data(), xb.data(), a.data());
            rows_t<D, TT, true>(W.data(), 0, 256, nsb, xq.data(), xs.data(), xb.data(), b.data(), xb16.data());
            double md = 0, mv = 0;
            for (int r = 0; r < 256; r++) for (int t = 0; t < TT; t++) {
                md = std::max(md, (double) std::fabs(a[r * 8 + t] - b[r * 8 + t])); mv = std::max(mv, (double) std::fabs(a[r * 8 + t]));
            }
            printf("  T=%d max|diff| %.3g max|out| %.3g\n", TT, md, mv);
            maxd = std::max(maxd, md);
        };
        cmp(std::integral_constant<int, 1>{}); cmp(std::integral_constant<int, 3>{}); cmp(std::integral_constant<int, 4>{});
        cmp(std::integral_constant<int, 5>{}); cmp(std::integral_constant<int, 8>{});
        printf("%-9s offset path vs sign path: max |diff| %.3g  %s\n", name, maxd, maxd == 0 ? "EXACT" : "MISMATCH");
    }
    std::vector<float> out(N * 8);
    auto timed = [&](auto fn) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            std::atomic<int64_t> next{0};
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (int i = 0; i < nth; i++) {
                th.emplace_back([&] {
                    for (int64_t a; (a = next.fetch_add(16)) < N;) fn(a, std::min<int64_t>(N, a + 16));
                });
            }
            for (auto & t : th) t.join();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    const double bytes = (double) N * nsb * sizeof(B);
    double t[9];
    t[1] = timed([&](int64_t a, int64_t b) { rows_t<D, 1, OFFP>(W.data(), a, b, nsb, xq.data(), xs.data(), xb.data(), out.data(), xb16.data()); });
    t[2] = timed([&](int64_t a, int64_t b) { rows_t<D, 2, OFFP>(W.data(), a, b, nsb, xq.data(), xs.data(), xb.data(), out.data(), xb16.data()); });
    t[3] = timed([&](int64_t a, int64_t b) { rows_t<D, 3, OFFP>(W.data(), a, b, nsb, xq.data(), xs.data(), xb.data(), out.data(), xb16.data()); });
    t[4] = timed([&](int64_t a, int64_t b) { rows_t<D, 4, OFFP>(W.data(), a, b, nsb, xq.data(), xs.data(), xb.data(), out.data(), xb16.data()); });
    t[5] = timed([&](int64_t a, int64_t b) { rows_t<D, 5, OFFP>(W.data(), a, b, nsb, xq.data(), xs.data(), xb.data(), out.data(), xb16.data()); });
    t[6] = timed([&](int64_t a, int64_t b) { rows_t<D, 6, OFFP>(W.data(), a, b, nsb, xq.data(), xs.data(), xb.data(), out.data(), xb16.data()); });
    t[7] = timed([&](int64_t a, int64_t b) { rows_t<D, 7, OFFP>(W.data(), a, b, nsb, xq.data(), xs.data(), xb.data(), out.data(), xb16.data()); });
    t[8] = timed([&](int64_t a, int64_t b) { rows_t<D, 8, OFFP>(W.data(), a, b, nsb, xq.data(), xs.data(), xb.data(), out.data(), xb16.data()); });
    printf("%-9s thr %2d  T=1 %.1f GB/s  cost:", name, nth, bytes / t[1] / 1e9);
    for (int i = 2; i <= 8; i++) printf(" %.2f", t[i] / t[1]);
    printf("   GB/s T=4 %.1f T=8 %.1f\n", bytes / t[4] / 1e9, bytes / t[8] / 1e9);
    fflush(stdout);
}

// paired IQ4_XS (two blocks per load, permuted activations) against the plain kernel: results and T = 1..8 cost
static void bench_pair(int nth) {
    using B         = block_iq4_xs;
    const int64_t K = 5120, nsb = K / QK_K;
    const int64_t N = (int64_t) (1024e6 / (nsb * sizeof(B)));
    std::vector<B> W(N * nsb);
    std::mt19937   rng(1);
    for (auto & b : W) {
        uint8_t * p = (uint8_t *) &b;
        for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng();
        const ggml_half d = _cvtss_sh(0.01f, 0);
        memcpy(p, &d, 2);
    }
    std::vector<float> x(8 * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    std::vector<int8_t> xq(8 * K), xp(8 * K);
    std::vector<float>  xs(8 * nsb);
    std::vector<int16_t> xb(8 * K / 32);
    for (int t = 0; t < 8; t++) srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K);
    xp = xq;
    for (int t = 0; t < 8; t++) srv_permute_pairs(xp.data() + t * K, K);
    std::vector<float> a(256 * 8), b(256 * 8);
    rows_t<srv_iq4_xs, 8>(W.data(), 0, 256, nsb, xq.data(), xs.data(), xb.data(), a.data());
    rows_t<srv_iq4_xs_p, 8>(W.data(), 0, 256, nsb, xp.data(), xs.data(), xb.data(), b.data());
    double md = 0, mv = 0;
    for (int i = 0; i < 256 * 8; i++) { md = std::max(md, (double) std::fabs(a[i] - b[i])); mv = std::max(mv, (double) std::fabs(a[i])); }
    printf("IQ4_XS paired vs plain: max |diff| %.3g (max |out| %.3g)\n", md, mv);
    std::vector<float> out(N * 8);
    auto timed = [&](auto fn) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            std::atomic<int64_t> next{0};
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (int i = 0; i < nth; i++)
                th.emplace_back([&] { for (int64_t s; (s = next.fetch_add(16)) < N;) fn(s, std::min<int64_t>(N, s + 16)); });
            for (auto & t : th) t.join();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    const double bytes = (double) N * nsb * sizeof(B);
    auto line = [&](const char * name, auto tag, const int8_t * q) {
        using D = decltype(tag);
        double t[9];
        t[1] = timed([&](int64_t s, int64_t e) { rows_t<D, 1>(W.data(), s, e, nsb, q, xs.data(), xb.data(), out.data()); });
        t[2] = timed([&](int64_t s, int64_t e) { rows_t<D, 2>(W.data(), s, e, nsb, q, xs.data(), xb.data(), out.data()); });
        t[3] = timed([&](int64_t s, int64_t e) { rows_t<D, 3>(W.data(), s, e, nsb, q, xs.data(), xb.data(), out.data()); });
        t[4] = timed([&](int64_t s, int64_t e) { rows_t<D, 4>(W.data(), s, e, nsb, q, xs.data(), xb.data(), out.data()); });
        t[5] = timed([&](int64_t s, int64_t e) { rows_t<D, 5>(W.data(), s, e, nsb, q, xs.data(), xb.data(), out.data()); });
        t[6] = timed([&](int64_t s, int64_t e) { rows_t<D, 6>(W.data(), s, e, nsb, q, xs.data(), xb.data(), out.data()); });
        t[8] = timed([&](int64_t s, int64_t e) { rows_t<D, 8>(W.data(), s, e, nsb, q, xs.data(), xb.data(), out.data()); });
        printf("%-14s thr %2d  GB/s T=1 %.1f T=2 %.1f T=3 %.1f T=4 %.1f T=5 %.1f T=6 %.1f T=8 %.1f\n", name, nth, bytes / t[1] / 1e9,
               bytes / t[2] / 1e9, bytes / t[3] / 1e9, bytes / t[4] / 1e9, bytes / t[5] / 1e9, bytes / t[6] / 1e9, bytes / t[8] / 1e9);
        fflush(stdout);
    };
    for (int r = 0; r < 2; r++) {
        line("IQ4_XS plain", srv_iq4_xs{}, xq.data());
        line("IQ4_XS paired", srv_iq4_xs_p{}, xp.data());
    }
}

// IQ3_S three ways on the same weights: native (scalar grid loads), native with a gather, C4 re-encoding.
// Reported as time per pass relative to native T=1, so the byte difference of C4 is included.
static void bench_iq3(int nth) {
    using B         = block_iq3_s;
    const int64_t K = 5120, nsb = K / QK_K;
    const int64_t N = (int64_t) (1024e6 / (nsb * sizeof(B)));
    std::vector<B> W(N * nsb);
    std::mt19937   rng(1);
    for (auto & b : W) {
        uint8_t * p = (uint8_t *) &b;
        for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng();
        const ggml_half d = _cvtss_sh(0.01f, 0);
        memcpy(p, &d, 2);
    }
    std::vector<block_srv_c4> C(N * nsb);
    for (int64_t i = 0; i < N * nsb; i++) srv_to_c4<srv_iq3_s, 0>(&W[i], &C[i]);
    std::vector<float> x(8 * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    std::vector<int8_t> xq(8 * K);
    std::vector<float>  xs(8 * nsb);
    std::vector<int16_t> xb(8 * K / 32);
    for (int t = 0; t < 8; t++) srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K);
    {
        std::vector<float> a(256 * 8), b(256 * 8);
        rows_t<srv_iq3_s, 8>(W.data(), 0, 256, nsb, xq.data(), xs.data(), xb.data(), a.data());
        rows_t<srv_iq3_s_g, 8>(W.data(), 0, 256, nsb, xq.data(), xs.data(), xb.data(), b.data());
        double md = 0;
        for (int i = 0; i < 256 * 8; i++) md = std::max(md, (double) std::fabs(a[i] - b[i]));
        printf("IQ3_S gather vs native: max |diff| %.3g  %s\n", md, md == 0 ? "EXACT" : "MISMATCH");
    }
    std::vector<float> out(N * 8);
    auto timed = [&](auto fn) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            std::atomic<int64_t> next{0};
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (int i = 0; i < nth; i++)
                th.emplace_back([&] { for (int64_t s; (s = next.fetch_add(16)) < N;) fn(s, std::min<int64_t>(N, s + 16)); });
            for (auto & t : th) t.join();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    double ref = 0;
    auto line = [&](const char * name, auto tag, const void * w) {
        using D = decltype(tag);
        const auto * Wd = (const typename D::block *) w;
        double t[9];
        t[1] = timed([&](int64_t s, int64_t e) { rows_t<D, 1>(Wd, s, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
        t[2] = timed([&](int64_t s, int64_t e) { rows_t<D, 2>(Wd, s, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
        t[3] = timed([&](int64_t s, int64_t e) { rows_t<D, 3>(Wd, s, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
        t[4] = timed([&](int64_t s, int64_t e) { rows_t<D, 4>(Wd, s, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
        t[5] = timed([&](int64_t s, int64_t e) { rows_t<D, 5>(Wd, s, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
        t[6] = timed([&](int64_t s, int64_t e) { rows_t<D, 6>(Wd, s, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
        t[8] = timed([&](int64_t s, int64_t e) { rows_t<D, 8>(Wd, s, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
        if (ref == 0) ref = t[1];
        printf("%-14s thr %2d  time vs native T=1:  T=1 %.2f T=2 %.2f T=3 %.2f T=4 %.2f T=5 %.2f T=6 %.2f T=8 %.2f\n", name, nth,
               t[1] / ref, t[2] / ref, t[3] / ref, t[4] / ref, t[5] / ref, t[6] / ref, t[8] / ref);
        fflush(stdout);
    };
    for (int r = 0; r < 2; r++) {
        line("IQ3_S native", srv_iq3_s{}, W.data());
        line("IQ3_S gather", srv_iq3_s_g{}, W.data());
        line("C4", srv_c4<0>{}, C.data());
    }
}

// SMT helper threads: nth compute threads on logical CPUs 0,2,4,..., each with a helper on its SMT sibling
// (2i+1) that streams the compute thread's weight rows into the shared L2 a fixed distance ahead. Compute and
// memory traffic then overlap instead of adding up. Static row ranges; progress is shared per pair.
#define NOMINMAX
#include <windows.h>
struct alignas(64) helper_pair {
    std::atomic<int64_t> done{0};     // bytes the compute thread has finished
    std::atomic<bool>    stop{false};
};

template <class D, int T>
static double run_helper(const typename D::block * W, int64_t N, int64_t nsb, const int8_t * xq, const float * xs,
                         const int16_t * xb, float * out, int nth, int64_t ahead, bool helpers) {
    std::vector<helper_pair> pr(nth);
    std::vector<std::thread> th;
    const size_t row_bytes = nsb * sizeof(typename D::block);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < nth; i++) {
        const int64_t per = (N + nth - 1) / nth, a = i * per, b = std::min(N, a + per);
        th.emplace_back([&, i, a, b] {
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (2 * i));
            const int8_t * xp[T]; const float * sp[T]; const int16_t * bp[T];
            for (int t = 0; t < T; t++) { xp[t] = xq + t * nsb * QK_K; sp[t] = xs + t * nsb; bp[t] = xb + t * nsb * 8; }
            for (int64_t r = a; r < b; r++) {
                srv_row<D, T>(W + r * nsb, nsb, xp, sp, out + r * 8, bp);
                pr[i].done.store((r + 1 - a) * row_bytes, std::memory_order_relaxed);
            }
            pr[i].stop.store(true);
        });
        if (helpers) {
            th.emplace_back([&, i, a, b] {
                SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (2 * i + 1));
                const char * base = (const char *) (W + a * nsb);
                const int64_t len = (b - a) * row_bytes;
                volatile char sink = 0;
                for (int64_t p = 0; p < len; p += 64) {
                    while (p > pr[i].done.load(std::memory_order_relaxed) + ahead) {
                        if (pr[i].stop.load(std::memory_order_relaxed)) return;
                        _mm_pause();
                    }
                    sink += base[p];  // a demand load: the line lands in the L1/L2 the sibling shares
                }
            });
        }
    }
    for (auto & t : th) t.join();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <class D>
static void bench_helper(const char * name, int nth, int fpos) {
    using B         = typename D::block;
    const int64_t K = 5120, nsb = K / QK_K;
    const int64_t N = (int64_t) (1024e6 / (nsb * sizeof(B)));
    std::vector<B> W(N * nsb);
    std::mt19937   rng(1);
    for (auto & b : W) {
        uint8_t * p = (uint8_t *) &b;
        for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng();
        if (fpos >= 0) { const ggml_half d = _cvtss_sh(0.01f, 0); memcpy(p + fpos, &d, 2); }
        else { const float d = 0.01f; memcpy(p + sizeof(B) - 4, &d, 4); }
    }
    std::vector<float> x(8 * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    std::vector<int8_t> xq(8 * K);
    std::vector<float>  xs(8 * nsb);
    std::vector<int16_t> xb(8 * K / 32);
    for (int t = 0; t < 8; t++) srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K, xb.data() + t * K / 32);
    std::vector<float> out(N * 8);
    const double bytes = (double) N * nsb * sizeof(B);
    auto best = [&](auto fn) { double b = 1e9; for (int r = 0; r < 4; r++) b = std::min(b, fn()); return b; };
    for (int64_t ahead : {0LL, 32LL << 10, 96LL << 10, 192LL << 10}) {
        const bool h = ahead > 0;
        double t[9];
        t[1] = best([&] { return run_helper<D, 1>(W.data(), N, nsb, xq.data(), xs.data(), xb.data(), out.data(), nth, ahead, h); });
        t[4] = best([&] { return run_helper<D, 4>(W.data(), N, nsb, xq.data(), xs.data(), xb.data(), out.data(), nth, ahead, h); });
        t[6] = best([&] { return run_helper<D, 6>(W.data(), N, nsb, xq.data(), xs.data(), xb.data(), out.data(), nth, ahead, h); });
        t[8] = best([&] { return run_helper<D, 8>(W.data(), N, nsb, xq.data(), xs.data(), xb.data(), out.data(), nth, ahead, h); });
        printf("%-9s %d compute%s ahead %3lld KB: GB/s T=1 %.1f T=4 %.1f T=6 %.1f T=8 %.1f   cost T=4 %.2f T=6 %.2f T=8 %.2f\n", name,
               nth, h ? " + helpers" : "          ", (long long) (ahead >> 10), bytes / t[1] / 1e9, bytes / t[4] / 1e9,
               bytes / t[6] / 1e9, bytes / t[8] / 1e9, t[4] / t[1], t[6] / t[1], t[8] / t[1]);
        fflush(stdout);
    }
}

// pure compute cost of the real kernels: 48 rows (~130 KB, L2-resident) swept many times on one thread,
// in cycles per (32 weights x token) at the measured clock
template <class D, bool OFFP = false>
static void bench_l2(const char * name, int fpos, double ghz, int64_t K = 5120) {
    using B         = typename D::block;
    const int64_t nsb = K / QK_K, N = 48 * 5120 / K;
    std::vector<B> W(N * nsb);
    std::mt19937   rng(1);
    for (auto & b : W) {
        uint8_t * p = (uint8_t *) &b;
        for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng();
        if (fpos >= 0) { const ggml_half d = _cvtss_sh(0.01f, 0); memcpy(p + fpos, &d, 2); }
        else { const float d = 0.01f; memcpy(p + sizeof(B) - 4, &d, 4); }
    }
    std::vector<float> x(8 * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    std::vector<int8_t> xq(8 * K);
    std::vector<float>  xs(8 * nsb);
    std::vector<int16_t> xb(8 * K / 32), x16(8 * K / 16);
    for (int t = 0; t < 8; t++)
        srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K, xb.data() + t * K / 32, x16.data() + t * K / 16);
    std::vector<float> out(N * 8);
    const int reps = 400;
    auto cyc = [&](auto tt) {
        constexpr int T = decltype(tt)::value;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++) rows_t<D, T, OFFP>(W.data(), 0, N, nsb, xq.data(), xs.data(), xb.data(), out.data(), x16.data());
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return s * ghz * 1e9 / ((double) reps * N * K / 32 * T);
    };
    printf("%-9s K=%5lld L2-resident, cycles per 32w x token: T=1 %.2f T=2 %.2f T=4 %.2f T=5 %.2f T=6 %.2f T=8 %.2f\n", name, (long long) K,
           cyc(std::integral_constant<int, 1>{}), cyc(std::integral_constant<int, 2>{}), cyc(std::integral_constant<int, 4>{}),
           cyc(std::integral_constant<int, 5>{}), cyc(std::integral_constant<int, 6>{}), cyc(std::integral_constant<int, 8>{}));
}

// IQ4_XS with unsigned weights (no sign step) against the sign kernel, both on 7-bit activations: exactness and
// L2-resident cycles per (32 weights x token), then streaming throughput from RAM with all threads
static void bench_iq4u(double ghz, int nth) {
    using B         = block_iq4_xs;
    const int64_t K = 5120, nsb = K / QK_K;
    for (int big = 0; big < 2; big++) {
        const int64_t N = big ? (int64_t) (1024e6 / (nsb * sizeof(B))) : 48;
        std::vector<B> W(N * nsb);
        std::mt19937   rng(1);
        for (auto & b : W) {
            uint8_t * p = (uint8_t *) &b;
            for (size_t i = 0; i < sizeof(B); i++) p[i] = (uint8_t) rng();
            const ggml_half d = _cvtss_sh(0.01f, 0);
            memcpy(p, &d, 2);
        }
        std::vector<float> x(8 * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : x) v = nd(rng);
        std::vector<int8_t> xq(8 * K);
        std::vector<float>  xs(8 * nsb);
        std::vector<int16_t> xb(8 * K / 32), x16(8 * K / 16);
        for (int t = 0; t < 8; t++)
            srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K, xb.data() + t * K / 32, x16.data() + t * K / 16, 63);
        std::vector<float> out(N * 8);
        if (!big) {
            std::vector<float> a(48 * 8), b(48 * 8);
            rows_t<srv_iq4_xs, 8>(W.data(), 0, 48, nsb, xq.data(), xs.data(), xb.data(), a.data());
            rows_t<srv_iq4_xs_u, 8, true>(W.data(), 0, 48, nsb, xq.data(), xs.data(), xb.data(), b.data(), x16.data());
            double md = 0, mv = 0;
            for (int i = 0; i < 48 * 8; i++) { md = std::max(md, (double) std::fabs(a[i] - b[i])); mv = std::max(mv, (double) std::fabs(a[i])); }
            printf("IQ4_XS unsigned vs sign (7-bit x): max |diff| %.3g of max |out| %.3g\n", md, mv);
            auto cyc = [&](auto tag, auto tt) {
                using D = decltype(tag);
                constexpr int T = decltype(tt)::value;
                constexpr bool U = std::is_same<D, srv_iq4_xs_u>::value;
                const int reps = 400;
                auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; r++) rows_t<D, T, U>(W.data(), 0, N, nsb, xq.data(), xs.data(), xb.data(), out.data(), x16.data());
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                return sec * ghz * 1e9 / ((double) reps * N * K / 32 * T);
            };
            for (int r = 0; r < 2; r++) {
                printf("L2 cycles/32w x tok  sign:     T=1 %.2f T=4 %.2f T=6 %.2f T=8 %.2f\n", cyc(srv_iq4_xs{}, std::integral_constant<int, 1>{}),
                       cyc(srv_iq4_xs{}, std::integral_constant<int, 4>{}), cyc(srv_iq4_xs{}, std::integral_constant<int, 6>{}),
                       cyc(srv_iq4_xs{}, std::integral_constant<int, 8>{}));
                printf("L2 cycles/32w x tok  unsigned: T=1 %.2f T=4 %.2f T=6 %.2f T=8 %.2f\n", cyc(srv_iq4_xs_u{}, std::integral_constant<int, 1>{}),
                       cyc(srv_iq4_xs_u{}, std::integral_constant<int, 4>{}), cyc(srv_iq4_xs_u{}, std::integral_constant<int, 6>{}),
                       cyc(srv_iq4_xs_u{}, std::integral_constant<int, 8>{}));
            }
        } else {
            auto timed = [&](auto fn) {
                double best = 1e9;
                for (int rep = 0; rep < 4; rep++) {
                    std::atomic<int64_t> next{0};
                    auto t0 = std::chrono::steady_clock::now();
                    std::vector<std::thread> th;
                    for (int i = 0; i < nth; i++)
                        th.emplace_back([&] { for (int64_t s; (s = next.fetch_add(16)) < N;) fn(s, std::min<int64_t>(N, s + 16)); });
                    for (auto & t : th) t.join();
                    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                return best;
            };
            const double bytes = (double) N * nsb * sizeof(B);
            for (int r = 0; r < 2; r++) {
                double s1 = timed([&](int64_t a, int64_t e) { rows_t<srv_iq4_xs, 1>(W.data(), a, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
                double s4 = timed([&](int64_t a, int64_t e) { rows_t<srv_iq4_xs, 4>(W.data(), a, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
                double s6 = timed([&](int64_t a, int64_t e) { rows_t<srv_iq4_xs, 6>(W.data(), a, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
                double s8 = timed([&](int64_t a, int64_t e) { rows_t<srv_iq4_xs, 8>(W.data(), a, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
                double u4 = timed([&](int64_t a, int64_t e) { rows_t<srv_iq4_xs_u, 4, true>(W.data(), a, e, nsb, xq.data(), xs.data(), xb.data(), out.data(), x16.data()); });
                double u6 = timed([&](int64_t a, int64_t e) { rows_t<srv_iq4_xs_u, 6, true>(W.data(), a, e, nsb, xq.data(), xs.data(), xb.data(), out.data(), x16.data()); });
                double u8 = timed([&](int64_t a, int64_t e) { rows_t<srv_iq4_xs_u, 8, true>(W.data(), a, e, nsb, xq.data(), xs.data(), xb.data(), out.data(), x16.data()); });
                printf("RAM %d thr, cost vs sign T=1:  sign T=4 %.2f T=6 %.2f T=8 %.2f   unsigned T=4 %.2f T=6 %.2f T=8 %.2f  (T=1 %.1f GB/s)\n", nth,
                       s4 / s1, s6 / s1, s8 / s1, u4 / s1, u6 / s1, u8 / s1, bytes / s1 / 1e9);
            }
        }
    }
}

// IQ4_XS re-encoded into the C4 layout (int8 scales unpacked once): exactness, L2 cycles, RAM curve
static void bench_iq4c4(double ghz, int nth) {
    const int64_t K = 5120, nsb = K / QK_K;
    for (int big = 0; big < 2; big++) {
        const int64_t N = big ? (int64_t) (1024e6 / (nsb * sizeof(block_iq4_xs))) : 48;
        std::vector<block_iq4_xs> W(N * nsb);
        std::mt19937 rng(1);
        for (auto & b : W) {
            uint8_t * p = (uint8_t *) &b;
            for (size_t i = 0; i < sizeof(b); i++) p[i] = (uint8_t) rng();
            const ggml_half d = _cvtss_sh(0.01f, 0);
            memcpy(p, &d, 2);
        }
        std::vector<block_srv_c4> C(N * nsb);
        for (int64_t i = 0; i < N * nsb; i++) srv_iq4_to_c4(&W[i], &C[i]);
        std::vector<float> x(8 * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : x) v = nd(rng);
        std::vector<int8_t> xq(8 * K);
        std::vector<float>  xs(8 * nsb);
        std::vector<int16_t> xb(8 * K / 32);
        for (int t = 0; t < 8; t++) srv_quantize_row(x.data() + t * K, xq.data() + t * K, xs.data() + t * nsb, K);
        std::vector<float> out(N * 8);
        if (!big) {
            std::vector<float> a(48 * 8), b(48 * 8);
            rows_t<srv_iq4_xs, 8>(W.data(), 0, 48, nsb, xq.data(), xs.data(), xb.data(), a.data());
            rows_t<srv_c4<2>, 8>(C.data(), 0, 48, nsb, xq.data(), xs.data(), xb.data(), b.data());
            double md = 0;
            for (int i = 0; i < 48 * 8; i++) md = std::max(md, (double) std::fabs(a[i] - b[i]));
            printf("IQ4_XS in C4 layout vs IQ4_XS: max |diff| %.3g  %s\n", md, md == 0 ? "EXACT" : "MISMATCH");
            auto cyc = [&](auto tag, const void * w, auto tt) {
                using D = decltype(tag);
                constexpr int T = decltype(tt)::value;
                const int reps = 400;
                auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < reps; r++) rows_t<D, T>((const typename D::block *) w, 0, N, nsb, xq.data(), xs.data(), xb.data(), out.data());
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                return sec * ghz * 1e9 / ((double) reps * N * K / 32 * T);
            };
            auto line = [&](const char * nm, auto tag, const void * w) {
                printf("L2 %-12s T=1 %.2f T=2 %.2f T=4 %.2f T=6 %.2f T=8 %.2f\n", nm, cyc(tag, w, std::integral_constant<int, 1>{}),
                       cyc(tag, w, std::integral_constant<int, 2>{}), cyc(tag, w, std::integral_constant<int, 4>{}),
                       cyc(tag, w, std::integral_constant<int, 6>{}), cyc(tag, w, std::integral_constant<int, 8>{}));
            };
            for (int r = 0; r < 2; r++) {
                line("iq4_xs", srv_iq4_xs{}, W.data());
                line("iq4_xs-p", srv_iq4_xs_p{}, W.data());
                line("c4(iq4)", srv_c4<2>{}, C.data());
                line("c4(iq4)-p", srv_c4_p<2>{}, C.data());
            }
        } else {
            auto timed = [&](auto fn) {
                double best = 1e9;
                for (int rep = 0; rep < 4; rep++) {
                    std::atomic<int64_t> next{0};
                    auto t0 = std::chrono::steady_clock::now();
                    std::vector<std::thread> th;
                    for (int i = 0; i < nth; i++)
                        th.emplace_back([&] { for (int64_t s; (s = next.fetch_add(16)) < N;) fn(s, std::min<int64_t>(N, s + 16)); });
                    for (auto & t : th) t.join();
                    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                return best;
            };
            auto curve = [&](const char * nm, auto tag, const void * w, double ref) {
                using D = decltype(tag);
                const auto * Wd = (const typename D::block *) w;
                double t1 = timed([&](int64_t a, int64_t e) { rows_t<D, 1>(Wd, a, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
                double t4 = timed([&](int64_t a, int64_t e) { rows_t<D, 4>(Wd, a, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
                double t6 = timed([&](int64_t a, int64_t e) { rows_t<D, 6>(Wd, a, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
                double t8 = timed([&](int64_t a, int64_t e) { rows_t<D, 8>(Wd, a, e, nsb, xq.data(), xs.data(), xb.data(), out.data()); });
                if (ref == 0) ref = t1;
                printf("RAM %-12s %d thr, time vs iq4_xs T=1:  T=1 %.2f T=4 %.2f T=6 %.2f T=8 %.2f\n", nm, nth, t1 / ref, t4 / ref, t6 / ref, t8 / ref);
                return ref;
            };
            for (int r = 0; r < 2; r++) {
                double ref = curve("iq4_xs", srv_iq4_xs{}, W.data(), 0);
                curve("iq4_xs-p", srv_iq4_xs_p{}, W.data(), ref);
                curve("c4(iq4)", srv_c4<2>{}, C.data(), ref);
                curve("c4(iq4)-p", srv_c4_p<2>{}, C.data(), ref);
            }
        }
    }
}

int main(int argc, char ** argv) {
    const int nth = argc > 1 ? atoi(argv[1]) : 16;
    if (argc > 2 && std::string(argv[2]) == "iq4c4") {
        bench_iq4c4(argc > 3 ? atof(argv[3]) : 4.2, nth);
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "iq4u") {
        bench_iq4u(argc > 3 ? atof(argv[3]) : 4.2, nth);
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "l2q3") {
        const double ghz = argc > 3 ? atof(argv[3]) : 3.0;
        for (int r = 0; r < 2; r++) {
            bench_l2<srv_iq4_xs_p>("IQ4_XS-p", 0, ghz);
            bench_l2<srv_q3_k>("Q3_K", offsetof(block_q3_K, d), ghz);
            bench_l2<srv_q3_k, true>("Q3_K+off", offsetof(block_q3_K, d), ghz);
        }
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "l2k") {
        const double ghz = argc > 3 ? atof(argv[3]) : 3.0;
        for (int64_t K : {1024LL, 2048LL, 5120LL, 17408LL}) {
            bench_l2<srv_iq4_xs_p>("IQ4_XS-p", 0, ghz, K);
            bench_l2<srv_c4<0>>("C4", -1, ghz, K);
        }
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "l2") {
        const double ghz = argc > 3 ? atof(argv[3]) : 3.0;
        for (int r = 0; r < 2; r++) {
            bench_l2<srv_iq4_xs>("IQ4_XS", 0, ghz);
            bench_l2<srv_iq4_xs_p>("IQ4_XS-p", 0, ghz);
            bench_l2<srv_c4<0>>("C4", -1, ghz);
            bench_l2<srv_c4<0>, true>("C4+off", -1, ghz);
        }
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "helper") {
        bench_helper<srv_iq4_xs_p>("IQ4_XS-p", nth, 0);
        bench_helper<srv_c4<0>>("C4(IQ3S)", nth, -1);
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "iq3") {
        bench_iq3(nth);
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "pair") {
        bench_pair(nth);
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "off") {
        curve<srv_c4<0>, true>("C4S+off", nth, -1);
        curve<srv_q6_k, true>("Q6_K+off", nth, offsetof(block_q6_K, d));
        curve<srv_q45_k<false>>("Q4_K", nth, 0);
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "curve") {
        curve<srv_iq4_xs>("IQ4_XS", nth, 0);
        curve<srv_c4<0>>("C4(IQ3S)", nth, -1);
        curve<srv_c4<1>>("C4(IQ3XX)", nth, -1);
        curve<srv_iq3_xxs>("IQ3_XXS", nth, 0);
        curve<srv_q6_k>("Q6_K", nth, offsetof(block_q6_K, d));
        curve<srv_q45_k<false>>("Q4_K", nth, 0);
        curve<srv_q45_k<true>>("Q5_K", nth, 0);
        curve<srv_c4<0>, true>("C4S+off", nth, -1);
        curve<srv_c4<1>, true>("C4XX+off", nth, -1);
        curve<srv_iq3_xxs, true>("IQ3XX+off", nth, 0);
        curve<srv_q6_k, true>("Q6_K+off", nth, offsetof(block_q6_K, d));
        return 0;
    }
    if (argc > 2 && std::string(argv[2]) == "c4") {
        bench_c4<srv_iq3_s, 0>("IQ3_S", nth);
        bench_c4<srv_iq3_xxs, 1>("IQ3_XXS", nth);
        return 0;
    }
    bench<srv_iq4_xs>("IQ4_XS", nth);
    bench<srv_iq3_s>("IQ3_S", nth);
    bench<srv_iq3_xxs>("IQ3_XXS", nth);
    bench<srv_q3_k>("Q3_K", nth);
    bench<srv_iq2_xxs>("IQ2_XXS", nth);
    bench<srv_iq2_xs>("IQ2_XS", nth);
    bench<srv_iq2_s>("IQ2_S", nth);
    return 0;
}
