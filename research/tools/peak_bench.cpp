// Compute ceiling of the small-batch int8 kernels on this CPU, built up step by step towards the real C4 kernel, to
// find which part costs the cycles: cycles per (32 weights x 1 token), data in L1/L2, single thread at a given clock.
//   core    : sign + maddubs + madd + add per token, weights from a register-ready buffer
//   +dec    : the C4 block decode (load 16 B, nibble split, pshufb, abs, scale broadcast) per block
//   +sb     : per super-block (8 blocks) float scaling: cvt + fmadd per token
//   +ptr    : activations through a pointer array per token (as srv_row does)
// build: clang++ -O3 -march=native peak_bench.cpp -o peak_bench.exe
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

struct c4blk { uint8_t qs[128]; int8_t sc[8]; float d; };

alignas(64) static int8_t  W[64][32];
alignas(64) static int8_t  X[8][8192];
alignas(64) static c4blk   C[1024];
alignas(64) static float   XS[8][64];
alignas(64) static __m256  FM[8];

static const int8_t t0[16] = { 1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15 };

template <int T, int MODE>  // MODE: 0 core, 1 +dec, 2 +dec+sb, 3 +dec+sb+ptr, 4 = 3 with the float sums in memory
static double run(long nsb_total, double ghz) {
    const __m256i tab = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *) t0));
    __m256 f[T];
    __m256i acc[T];
    for (int t = 0; t < T; t++) { f[t] = _mm256_setzero_ps(); acc[t] = _mm256_setzero_si256(); }
    const int8_t * xp[T];
    for (int t = 0; t < T; t++) xp[t] = X[t];
    auto t_0 = std::chrono::steady_clock::now();
    for (long i = 0; i < nsb_total; i++) {
        const int sb = (int) (i & 1023);
        const c4blk * x = &C[sb];
        const int xo = (int) ((i & 31) * 256);
        if (MODE >= 2) for (int t = 0; t < T; t++) acc[t] = _mm256_setzero_si256();
        for (int b = 0; b < 8; b++) {
            __m256i aw, sg, sc;
            if (MODE == 0) {
                aw = _mm256_load_si256((const __m256i *) W[(sb + b) & 63]);
                sg = _mm256_load_si256((const __m256i *) W[(sb + b + 7) & 63]);
                sc = _mm256_set1_epi16(7);
            } else {
                const __m128i v   = _mm_loadu_si128((const __m128i *) (x->qs + 16 * b));
                const __m128i m4  = _mm_set1_epi8(0x0f);
                const __m256i idx = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m4), _mm_and_si128(v, m4));
                sg = _mm256_shuffle_epi8(tab, idx);
                aw = _mm256_abs_epi8(sg);
                sc = _mm256_set1_epi16(x->sc[b]);
            }
            for (int t = 0; t < T; t++) {
                const int8_t * src = MODE >= 3 ? xp[t] + xo + 32 * b : &X[t][xo + 32 * b];
                const __m256i xv = _mm256_loadu_si256((const __m256i *) src);
                acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, sg)), sc));
            }
        }
        if (MODE == 4) {
            __m256 * volatile fm = FM;  // escapes: the float sums live in memory, not in registers
            for (int t = 0; t < T; t++) fm[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[t]), _mm256_set1_ps(x->d * XS[t][i & 63]), fm[t]);
        } else if (MODE >= 2) {
            for (int t = 0; t < T; t++) f[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[t]), _mm256_set1_ps(x->d * XS[t][i & 63]), f[t]);
        }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_0).count();
    float sink = 0;
    for (int t = 0; t < T; t++) sink += _mm256_cvtss_f32(f[t]) + (float) _mm256_extract_epi32(acc[t], 1);
    if (sink == 1.2345f) printf(" ");
    return s * ghz * 1e9 / ((double) nsb_total * 8 * T);
}

int main(int argc, char ** argv) {
    const double ghz = argc > 1 ? atof(argv[1]) : 4.2;
    for (int i = 0; i < 64; i++) for (int j = 0; j < 32; j++) W[i][j] = (int8_t) (rand() % 15 + 1);
    for (int t = 0; t < 8; t++) { for (int j = 0; j < 8192; j++) X[t][j] = (int8_t) (rand() % 255 - 127); for (int j = 0; j < 64; j++) XS[t][j] = 0.01f; }
    for (int i = 0; i < 1024; i++) { for (int j = 0; j < 128; j++) C[i].qs[j] = (uint8_t) rand(); for (int b = 0; b < 8; b++) C[i].sc[b] = (int8_t) (rand() % 31 + 1); C[i].d = 0.01f; }
    const long n = 4000000;
    printf("cycles per 32w x token @ %.1f GHz (T = 4 / 8)\n", ghz);
    printf("core              %.2f / %.2f\n", run<4, 0>(n, ghz), run<8, 0>(n, ghz));
    printf("+ C4 decode       %.2f / %.2f\n", run<4, 1>(n, ghz), run<8, 1>(n, ghz));
    printf("+ sb float scale  %.2f / %.2f\n", run<4, 2>(n, ghz), run<8, 2>(n, ghz));
    printf("+ pointer array   %.2f / %.2f\n", run<4, 3>(n, ghz), run<8, 3>(n, ghz));
    printf("  float sums in memory  %.2f / %.2f\n", run<4, 4>(n, ghz), run<8, 4>(n, ghz));
}
