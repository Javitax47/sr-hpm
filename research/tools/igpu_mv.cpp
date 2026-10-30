// IQ4_XS weights x T tokens on the integrated GPU (OpenCL, zero-copy host memory): correctness against a scalar
// reference and throughput in GB/s of weights for T = 1, 4, 8, 16. Activations are f32 (no activation quantization).
// build: clang++ -O2 -std=c++17 -I<cuda>/include -I<llama.cpp>/ggml/src -I<llama.cpp>/ggml/include igpu_mv.cpp -L<cuda>/lib/x64 -lOpenCL
#define CL_TARGET_OPENCL_VERSION 120
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <CL/cl.h>
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <malloc.h>
#include <atomic>
#include <thread>

#define CK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { printf("OpenCL error %d at line %d: %s\n", e_, __LINE__, #x); exit(1); } } while (0)

// one work-group of 64 = 4 rows x 16 lanes; a lane takes every 16th 32-weight block of its row and keeps T sums;
// the 16 lanes of a row are reduced in local memory
// v2 (default): one row per work-item, 64 rows per group; per super-block the T x 256 activations are staged in local
// memory and read by all lanes at the same address (broadcast); the value table also lives in local memory
static const char * KSRC2 = R"CLC(
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void mv(__global const uchar * W, __global const float * X, __global float * Y, const int nsb, const int K, const int N) {
    __local float xs[T * 256];
    __local float kv[16];
    const int lid = get_local_id(0);
    const int row = get_group_id(0) * 64 + lid;
    if (lid < 16) {
        const float KV[16] = { -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f };
        kv[lid] = KV[lid];
    }
    float acc[T];
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    __global const uchar * rp = W + (size_t) min(row, N - 1) * nsb * 136;
    for (int ib = 0; ib < nsb; ib++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < T * 256; i += 64) xs[i] = X[(i >> 8) * K + ib * 256 + (i & 255)];
        barrier(CLK_LOCAL_MEM_FENCE);
        __global const uchar * sb = rp + ib * 136;
        const float d  = vload_half(0, (__global const half *) sb);
        const int   sh = sb[2] | (sb[3] << 8);
        const uint  sl = sb[4] | (sb[5] << 8) | (sb[6] << 16) | ((uint) sb[7] << 24);
        for (int b = 0; b < 8; b++) {
            const int   ls = (((sl >> (4 * b)) & 15) | (((sh >> (2 * b)) & 3) << 4)) - 32;
            const float dl = d * (float) ls;
            const uint4 q4 = vload4(0, (__global const uint *) (sb + 8 + 16 * b));
            const uint  qq[4] = { q4.x, q4.y, q4.z, q4.w };
            float s[T];
            for (int t = 0; t < T; t++) s[t] = 0.f;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    const uint  c  = (qq[i] >> (8 * j)) & 255;
                    const float w0 = kv[c & 15], w1 = kv[c >> 4];
                    const int   k  = b * 32 + 4 * i + j;
                    for (int t = 0; t < T; t++) s[t] = fma(w0, xs[t * 256 + k], fma(w1, xs[t * 256 + k + 16], s[t]));
                }
            }
            for (int t = 0; t < T; t++) acc[t] = fma(dl, s[t], acc[t]);
        }
    }
    if (row < N) for (int t = 0; t < T; t++) Y[(size_t) t * N + row] = acc[t];
}
)CLC";

// v3: the block scale folded into the 32 decoded weights (one multiply per weight, shared by all tokens), activations
// read from local memory as float4 (one LDS read per 4 FMAs), no per-block partial sums (fewer registers)
static const char * KSRC3 = R"CLC(
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void mv(__global const uchar * W, __global const float * X, __global float * Y, const int nsb, const int K, const int N) {
    __local float xs[T * 256];
    __local float kv[16];
    const int lid = get_local_id(0);
    const int row = get_group_id(0) * 64 + lid;
    if (lid < 16) {
        const float KV[16] = { -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f };
        kv[lid] = KV[lid];
    }
    float acc[T];
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    __global const uchar * rp = W + (size_t) min(row, N - 1) * nsb * 136;
    for (int ib = 0; ib < nsb; ib++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < T * 256; i += 64) xs[i] = X[(i >> 8) * K + ib * 256 + (i & 255)];
        barrier(CLK_LOCAL_MEM_FENCE);
        __global const uchar * sb = rp + ib * 136;
        const float d  = vload_half(0, (__global const half *) sb);
        const int   sh = sb[2] | (sb[3] << 8);
        const uint  sl = sb[4] | (sb[5] << 8) | (sb[6] << 16) | ((uint) sb[7] << 24);
        for (int b = 0; b < 8; b++) {
            const float dl = d * (float) ((int) (((sl >> (4 * b)) & 15) | (((sh >> (2 * b)) & 3) << 4)) - 32);
            const uint4 q4 = vload4(0, (__global const uint *) (sb + 8 + 16 * b));
            const uint  qq[4] = { q4.x, q4.y, q4.z, q4.w };
            for (int i = 0; i < 4; i++) {
                const uint   c  = qq[i];  // weights b*32 + 4i .. +3 (low nibbles) and +16 (high nibbles)
                float4 wl, wh;
                wl.s0 = kv[c & 15] * dl;         wl.s1 = kv[(c >> 8) & 15] * dl;
                wl.s2 = kv[(c >> 16) & 15] * dl; wl.s3 = kv[(c >> 24) & 15] * dl;
                wh.s0 = kv[(c >> 4) & 15] * dl;  wh.s1 = kv[(c >> 12) & 15] * dl;
                wh.s2 = kv[(c >> 20) & 15] * dl; wh.s3 = kv[(c >> 28) & 15] * dl;
                for (int t = 0; t < T; t++) {
                    const float4 xl = vload4(0, xs + t * 256 + b * 32 + 4 * i);
                    const float4 xh = vload4(0, xs + t * 256 + b * 32 + 16 + 4 * i);
                    acc[t] = fma(wl.s0, xl.s0, fma(wl.s1, xl.s1, fma(wl.s2, xl.s2, fma(wl.s3, xl.s3, acc[t]))));
                    acc[t] = fma(wh.s0, xh.s0, fma(wh.s1, xh.s1, fma(wh.s2, xh.s2, fma(wh.s3, xh.s3, acc[t]))));
                }
            }
        }
    }
    if (row < N) for (int t = 0; t < T; t++) Y[(size_t) t * N + row] = acc[t];
}
)CLC";

// v4: v3 on weights interleaved by groups of 64 rows (super-block ib of row 64 g + l at ((g nsb + ib) 64 + l) 136):
// the 64 lanes of a group read 64 consecutive blocks instead of 64 rows 2.7 KB apart
static const char * KSRC4 = R"CLC(
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void mv(__global const uchar * W, __global const float * X, __global float * Y, const int nsb, const int K, const int N) {
    __local float xs[T * 256];
    __local float kv[16];
    const int lid = get_local_id(0);
    const int row = get_group_id(0) * 64 + lid;
    if (lid < 16) {
        const float KV[16] = { -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f };
        kv[lid] = KV[lid];
    }
    float acc[T];
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    __global const uchar * gp = W + (size_t) get_group_id(0) * nsb * 64 * 136 + lid * 136;
    for (int ib = 0; ib < nsb; ib++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < T * 256; i += 64) xs[i] = X[(i >> 8) * K + ib * 256 + (i & 255)];
        barrier(CLK_LOCAL_MEM_FENCE);
        __global const uchar * sb = gp + (size_t) ib * 64 * 136;
        const uint2 hd = vload2(0, (__global const uint *) sb);
        const float d  = vload_half(0, (__global const half *) sb);
        const uint  sh = hd.x >> 16;
        const uint  sl = hd.y;
        for (int b = 0; b < 8; b++) {
            const float dl = d * (float) ((int) (((sl >> (4 * b)) & 15) | (((sh >> (2 * b)) & 3) << 4)) - 32);
            const uint4 q4 = vload4(0, (__global const uint *) (sb + 8 + 16 * b));
            const uint  qq[4] = { q4.x, q4.y, q4.z, q4.w };
            for (int i = 0; i < 4; i++) {
                const uint c = qq[i];
                float4 wl, wh;
                wl.s0 = kv[c & 15] * dl;         wl.s1 = kv[(c >> 8) & 15] * dl;
                wl.s2 = kv[(c >> 16) & 15] * dl; wl.s3 = kv[(c >> 24) & 15] * dl;
                wh.s0 = kv[(c >> 4) & 15] * dl;  wh.s1 = kv[(c >> 12) & 15] * dl;
                wh.s2 = kv[(c >> 20) & 15] * dl; wh.s3 = kv[(c >> 28) & 15] * dl;
                for (int t = 0; t < T; t++) {
                    const float4 xl = vload4(0, xs + t * 256 + b * 32 + 4 * i);
                    const float4 xh = vload4(0, xs + t * 256 + b * 32 + 16 + 4 * i);
                    acc[t] = fma(wl.s0, xl.s0, fma(wl.s1, xl.s1, fma(wl.s2, xl.s2, fma(wl.s3, xl.s3, acc[t]))));
                    acc[t] = fma(wh.s0, xh.s0, fma(wh.s1, xh.s1, fma(wh.s2, xh.s2, fma(wh.s3, xh.s3, acc[t]))));
                }
            }
        }
    }
    if (row < N) for (int t = 0; t < T; t++) Y[(size_t) t * N + row] = acc[t];
}
)CLC";

// v5: v4 with R row groups per work-item (rows 64 (R g' + j) + l for j < R): every activation read from local memory
// feeds R rows. Interleaved layout as v4.
static const char * KSRC5 = R"CLC(
#ifndef R
#define R 2
#endif
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void mv(__global const uchar * W, __global const float * X, __global float * Y, const int nsb, const int K, const int N) {
    __local float xs[T * 256];
    __local float kv[16];
    const int lid = get_local_id(0);
    if (lid < 16) {
        const float KV[16] = { -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f };
        kv[lid] = KV[lid];
    }
    const int ng = (N + 63) / 64;
    float acc[R][T];
    for (int j = 0; j < R; j++) for (int t = 0; t < T; t++) acc[j][t] = 0.f;
    __global const uchar * gp[R];
    for (int j = 0; j < R; j++) gp[j] = W + (size_t) min((int) get_group_id(0) * R + j, ng - 1) * nsb * 64 * 136 + lid * 136;
    for (int ib = 0; ib < nsb; ib++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < T * 256; i += 64) xs[i] = X[(i >> 8) * K + ib * 256 + (i & 255)];
        barrier(CLK_LOCAL_MEM_FENCE);
        float d[R];
        uint  sh[R], sl[R];
        for (int j = 0; j < R; j++) {
            __global const uchar * sb = gp[j] + (size_t) ib * 64 * 136;
            const uint2 hd = vload2(0, (__global const uint *) sb);
            d[j]  = vload_half(0, (__global const half *) sb);
            sh[j] = hd.x >> 16;
            sl[j] = hd.y;
        }
        for (int b = 0; b < 8; b++) {
            uint4 q4[R];
            float dl[R];
            for (int j = 0; j < R; j++) {
                dl[j] = d[j] * (float) ((int) (((sl[j] >> (4 * b)) & 15) | (((sh[j] >> (2 * b)) & 3) << 4)) - 32);
                q4[j] = vload4(0, (__global const uint *) (gp[j] + (size_t) ib * 64 * 136 + 8 + 16 * b));
            }
            for (int i = 0; i < 4; i++) {
                float4 wl[R], wh[R];
                for (int j = 0; j < R; j++) {
                    const uint c = i == 0 ? q4[j].x : i == 1 ? q4[j].y : i == 2 ? q4[j].z : q4[j].w;
                    wl[j].s0 = kv[c & 15] * dl[j];         wl[j].s1 = kv[(c >> 8) & 15] * dl[j];
                    wl[j].s2 = kv[(c >> 16) & 15] * dl[j]; wl[j].s3 = kv[(c >> 24) & 15] * dl[j];
                    wh[j].s0 = kv[(c >> 4) & 15] * dl[j];  wh[j].s1 = kv[(c >> 12) & 15] * dl[j];
                    wh[j].s2 = kv[(c >> 20) & 15] * dl[j]; wh[j].s3 = kv[(c >> 28) & 15] * dl[j];
                }
                for (int t = 0; t < T; t++) {
                    const float4 xl = vload4(0, xs + t * 256 + b * 32 + 4 * i);
                    const float4 xh = vload4(0, xs + t * 256 + b * 32 + 16 + 4 * i);
                    for (int j = 0; j < R; j++) {
                        acc[j][t] = fma(wl[j].s0, xl.s0, fma(wl[j].s1, xl.s1, fma(wl[j].s2, xl.s2, fma(wl[j].s3, xl.s3, acc[j][t]))));
                        acc[j][t] = fma(wh[j].s0, xh.s0, fma(wh[j].s1, xh.s1, fma(wh[j].s2, xh.s2, fma(wh[j].s3, xh.s3, acc[j][t]))));
                    }
                }
            }
        }
    }
    for (int j = 0; j < R; j++) {
        const int row = (get_group_id(0) * R + j) * 64 + lid;
        if (row < N) for (int t = 0; t < T; t++) Y[(size_t) t * N + row] = acc[j][t];
    }
}
)CLC";

static const char * KSRC6 = R"CLC(
#ifndef R
#define R 2
#endif
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void mv(__global const uchar * W, __global const float * X, __global float * Y, const int nsb, const int K, const int N) {
    __local float xs[T * 256];
    __local float kv[16];
    const int lid = get_local_id(0);
    if (lid < 16) {
        const float KV[16] = { -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f };
        kv[lid] = KV[lid];
    }
    const int ng = (N + 63) / 64;
    float acc[R][T];
    for (int j = 0; j < R; j++) for (int t = 0; t < T; t++) acc[j][t] = 0.f;
    __global const uchar * gp[R];
    for (int j = 0; j < R; j++) gp[j] = W + (size_t) min(((int) get_group_id(0) * R + j) * 64 + lid, N - 1) * nsb * 136;
    for (int ib = 0; ib < nsb; ib++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = lid; i < T * 256; i += 64) xs[i] = X[(i >> 8) * K + ib * 256 + (i & 255)];
        barrier(CLK_LOCAL_MEM_FENCE);
        float d[R];
        uint  sh[R], sl[R];
        for (int j = 0; j < R; j++) {
            __global const uchar * sb = gp[j] + (size_t) ib * 136;
            const uint2 hd = vload2(0, (__global const uint *) sb);
            d[j]  = vload_half(0, (__global const half *) sb);
            sh[j] = hd.x >> 16;
            sl[j] = hd.y;
        }
        for (int b = 0; b < 8; b++) {
            uint4 q4[R];
            float dl[R];
            for (int j = 0; j < R; j++) {
                dl[j] = d[j] * (float) ((int) (((sl[j] >> (4 * b)) & 15) | (((sh[j] >> (2 * b)) & 3) << 4)) - 32);
                q4[j] = vload4(0, (__global const uint *) (gp[j] + (size_t) ib * 136 + 8 + 16 * b));
            }
            for (int i = 0; i < 4; i++) {
                float4 wl[R], wh[R];
                for (int j = 0; j < R; j++) {
                    const uint c = i == 0 ? q4[j].x : i == 1 ? q4[j].y : i == 2 ? q4[j].z : q4[j].w;
                    wl[j].s0 = kv[c & 15] * dl[j];         wl[j].s1 = kv[(c >> 8) & 15] * dl[j];
                    wl[j].s2 = kv[(c >> 16) & 15] * dl[j]; wl[j].s3 = kv[(c >> 24) & 15] * dl[j];
                    wh[j].s0 = kv[(c >> 4) & 15] * dl[j];  wh[j].s1 = kv[(c >> 12) & 15] * dl[j];
                    wh[j].s2 = kv[(c >> 20) & 15] * dl[j]; wh[j].s3 = kv[(c >> 28) & 15] * dl[j];
                }
                for (int t = 0; t < T; t++) {
                    const float4 xl = vload4(0, xs + t * 256 + b * 32 + 4 * i);
                    const float4 xh = vload4(0, xs + t * 256 + b * 32 + 16 + 4 * i);
                    for (int j = 0; j < R; j++) {
                        acc[j][t] = fma(wl[j].s0, xl.s0, fma(wl[j].s1, xl.s1, fma(wl[j].s2, xl.s2, fma(wl[j].s3, xl.s3, acc[j][t]))));
                        acc[j][t] = fma(wh[j].s0, xh.s0, fma(wh[j].s1, xh.s1, fma(wh[j].s2, xh.s2, fma(wh[j].s3, xh.s3, acc[j][t]))));
                    }
                }
            }
        }
    }
    for (int j = 0; j < R; j++) {
        const int row = (get_group_id(0) * R + j) * 64 + lid;
        if (row < N) for (int t = 0; t < T; t++) Y[(size_t) t * N + row] = acc[j][t];
    }
}
)CLC";

static const char * KSRC = R"CLC(
__constant float KV[16] = { -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f };
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void mv(__global const uchar * W, __global const float * X, __global float * Y, const int nsb, const int K, const int N) {
    const int lid = get_local_id(0), rin = lid >> 4, sub = lid & 15;
    const int row = get_group_id(0) * 4 + rin;
    float acc[T];
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    if (row < N) {
        __global const uchar * rp = W + (size_t) row * nsb * 136;
        for (int blk = sub; blk < nsb * 8; blk += 16) {
            const int ib = blk >> 3, b = blk & 7;
            __global const uchar * sb = rp + ib * 136;
            const float d  = vload_half(0, (__global const half *) sb);
            const int   sh = sb[2] | (sb[3] << 8);
            const int   ls = (((sb[4 + (b >> 1)] >> (4 * (b & 1))) & 15) | (((sh >> (2 * b)) & 3) << 4)) - 32;
            const float dl = d * (float) ls;
            const uint4 q4 = vload4(0, (__global const uint *) (sb + 8 + 16 * b));
            const uint  qq[4] = { q4.x, q4.y, q4.z, q4.w };
            const int   k0 = ib * 256 + b * 32;
            float s[T];
            for (int t = 0; t < T; t++) s[t] = 0.f;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    const uint  c  = (qq[i] >> (8 * j)) & 255;
                    const float w0 = KV[c & 15], w1 = KV[c >> 4];
                    const int   k  = k0 + 4 * i + j;
                    for (int t = 0; t < T; t++) s[t] = fma(w0, X[t * K + k], fma(w1, X[t * K + k + 16], s[t]));
                }
            }
            for (int t = 0; t < T; t++) acc[t] = fma(dl, s[t], acc[t]);
        }
    }
    __local float red[64 * T];
    for (int t = 0; t < T; t++) red[lid * T + t] = acc[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sub == 0 && row < N) {
        for (int t = 0; t < T; t++) {
            float v = 0.f;
            for (int l = 0; l < 16; l++) v += red[(lid + l) * T + t];
            Y[(size_t) t * N + row] = v;
        }
    }
}
)CLC";

static float h2f(ggml_half h) { return _cvtsh_ss(h); }
static inline float fp16_to_f32(ggml_half h) { return _cvtsh_ss(h); }
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_f32(x)
#include "srv-kernels.inc"
using namespace ggml::cpu::repack;

static void ref_row(const block_iq4_xs * x, int64_t nsb, const float * X, int64_t K, int T, float * y) {
    for (int t = 0; t < T; t++) y[t] = 0;
    for (int64_t ib = 0; ib < nsb; ib++) {
        const float d = h2f(x[ib].d);
        for (int b = 0; b < 8; b++) {
            const int ls = ((x[ib].scales_l[b / 2] >> 4 * (b % 2)) & 0xf) | (((x[ib].scales_h >> 2 * b) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; j++) {
                const uint8_t c = x[ib].qs[16 * b + j];
                const int64_t k = ib * 256 + b * 32 + j;
                for (int t = 0; t < T; t++) y[t] += dl * (kvalues_iq4nl[c & 15] * X[t * K + k] + kvalues_iq4nl[c >> 4] * X[t * K + k + 16]);
            }
        }
    }
}

int main(int argc, char ** argv) {
    cl_uint np = 0;
    CK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> ps(np);
    CK(clGetPlatformIDs(np, ps.data(), nullptr));
    cl_device_id dev = nullptr;
    for (auto p : ps) {
        cl_device_id d; cl_uint nd = 0; cl_bool um = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, &nd) == CL_SUCCESS && nd) {
            clGetDeviceInfo(d, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(um), &um, nullptr);
            if (um) dev = d;
        }
    }
    if (!dev) { printf("no iGPU\n"); return 1; }
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err); CK(err);

    const int64_t K = 5120, nsb = K / QK_K;
    const int64_t N = (int64_t) (1024e6 / (nsb * sizeof(block_iq4_xs)));
    const size_t wbytes = (size_t) N * nsb * sizeof(block_iq4_xs);
    // ordinary cacheable host memory shared by CPU and iGPU (CL_MEM_USE_HOST_PTR), like the model's weights
    auto * W = (block_iq4_xs *) _aligned_malloc(wbytes, 4096);
    std::mt19937 rng(1);
    for (int64_t i = 0; i < N * nsb; i++) {
        uint8_t * p = (uint8_t *) &W[i];
        for (size_t j = 0; j < sizeof(block_iq4_xs); j++) p[j] = (uint8_t) rng();
        W[i].d = _cvtss_sh(0.01f, 0);
    }
    std::vector<block_iq4_xs> Wc(W, W + 64 * nsb);  // copy of the first rows for the reference
    const bool interleave = argc > 1 && (std::string(argv[1]) == "v4" || std::string(argv[1]).rfind("v5", 0) == 0);
    const int64_t NG = (N + 63) / 64;
    block_iq4_xs * Wi = W;
    if (interleave) {
        Wi = (block_iq4_xs *) _aligned_malloc((size_t) NG * 64 * nsb * sizeof(block_iq4_xs), 4096);
        for (int64_t g = 0; g < NG; g++)
            for (int64_t ib = 0; ib < nsb; ib++)
                for (int l = 0; l < 64; l++) Wi[(g * nsb + ib) * 64 + l] = W[std::min<int64_t>(N - 1, g * 64 + l) * nsb + ib];
    }
    cl_mem wb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, interleave ? (size_t) NG * 64 * nsb * sizeof(block_iq4_xs) : wbytes, Wi, &err); CK(err);

    const int TMAX = 16;
    std::vector<float> X(TMAX * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : X) v = nd(rng);
    cl_mem xb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, X.size() * 4, X.data(), &err); CK(err);
    cl_mem yb = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) TMAX * N * 4, nullptr, &err); CK(err);

    for (int T : { 1, 4, 8, 16 }) {
        const std::string ver = argc > 1 ? argv[1] : "v3";
        const char * src = ver == "v1" ? KSRC : ver == "v2" ? KSRC2 : ver == "v4" ? KSRC4 : ver.rfind("v5", 0) == 0 ? KSRC5 : ver.rfind("v6", 0) == 0 ? KSRC6 : KSRC3;
        cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err); CK(err);
        const std::string vv = argc > 1 ? argv[1] : "";
        const int RR = (vv.rfind("v5r", 0) == 0 || vv.rfind("v6r", 0) == 0) ? atoi(vv.c_str() + 3) : 2;
        const std::string opt = "-cl-fast-relaxed-math -D T=" + std::to_string(T) + " -D R=" + std::to_string(RR);
        if (clBuildProgram(prog, 1, &dev, opt.c_str(), nullptr, nullptr) != CL_SUCCESS) {
            char log[8192]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr); printf("%s\n", log); return 1;
        }
        cl_kernel k = clCreateKernel(prog, "mv", &err); CK(err);
        const int insb = (int) nsb, iK = (int) K, iN = (int) N;
        CK(clSetKernelArg(k, 0, sizeof(wb), &wb)); CK(clSetKernelArg(k, 1, sizeof(xb), &xb)); CK(clSetKernelArg(k, 2, sizeof(yb), &yb));
        CK(clSetKernelArg(k, 3, sizeof(int), &insb)); CK(clSetKernelArg(k, 4, sizeof(int), &iK)); CK(clSetKernelArg(k, 5, sizeof(int), &iN));
        const bool v1 = argc > 1 && std::string(argv[1]) == "v1";
        const bool v5 = vv.rfind("v5", 0) == 0 || vv.rfind("v6", 0) == 0;
        const size_t gws = v1 ? (size_t) ((N + 3) / 4) * 64 : v5 ? (size_t) (((N + 63) / 64 + RR - 1) / RR) * 64 : (size_t) ((N + 63) / 64) * 64, lws = 64;
        CK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr));
        CK(clFinish(q));
        // correctness on the first 64 rows
        std::vector<float> Y((size_t) T * N);
        CK(clEnqueueReadBuffer(q, yb, CL_TRUE, 0, Y.size() * 4, Y.data(), 0, nullptr, nullptr));
        double md = 0, mv = 0;
        for (int r = 0; r < 64; r++) {
            float y[TMAX];
            ref_row(&Wc[r * nsb], nsb, X.data(), K, T, y);
            for (int t = 0; t < T; t++) { md = std::max(md, (double) std::fabs(y[t] - Y[(size_t) t * N + r])); mv = std::max(mv, (double) std::fabs(y[t])); }
            if (getenv("IGPU_DEBUG") && r < 3) printf("  row %d: ref %g got %g\n", r, y[0], Y[r]);
        }
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            auto t0 = std::chrono::steady_clock::now();
            CK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr));
            CK(clFinish(q));
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("T=%2d  %.1f GB/s  (%.1f ms per GB)   max rel err %.2g\n", T, wbytes / best / 1e9, best * 1e3 * 1e9 / wbytes, md / mv);
        // dispatch latency: a one-group launch, enqueue + finish
        {
            const size_t g1 = 64;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 200; r++) {
                CK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g1, &lws, 0, nullptr, nullptr));
                CK(clFinish(q));
            }
            printf("      dispatch + finish of one group: %.0f us\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 200 * 1e6);
        }
        if (T == 8 || T == 4) {
            // split: the iGPU takes rows [0, r), 8 CPU threads rows [r, N) of the same zero-copy weights
            auto * Wm = W;
            std::vector<int8_t> xq(T * K);
            std::vector<float> xsc(T * nsb), outc((size_t) N * 8);
            for (int t = 0; t < T; t++) srv_quantize_row(X.data() + t * K, xq.data() + t * K, xsc.data() + t * nsb, K);
            const int8_t * xp[16]; const float * sp[16];
            for (int t = 0; t < T; t++) { xp[t] = xq.data() + t * K; sp[t] = xsc.data() + t * nsb; }
            auto cpu_rows = [&](int64_t r0) {
                std::atomic<int64_t> next{r0};
                std::vector<std::thread> th;
                for (int i = 0; i < 8; i++)
                    th.emplace_back([&] {
                        for (int64_t s0; (s0 = next.fetch_add(16)) < N;)
                            for (int64_t rr = s0; rr < std::min<int64_t>(N, s0 + 16); rr++) {
                                if (T == 8) srv_row<srv_iq4_xs, 8>(Wm + rr * nsb, nsb, xp, sp, outc.data() + rr * 8);
                                else        srv_row<srv_iq4_xs, 4>(Wm + rr * nsb, nsb, xp, sp, outc.data() + rr * 8);
                            }
                    });
                for (auto & t : th) t.join();
            };
            for (double frac : { 0.0, 0.3, 0.45, 0.6, 1.0 }) {
                const int64_t r = ((int64_t) (frac * N) / 64) * 64;
                double best2 = 1e9;
                for (int rep = 0; rep < 3; rep++) {
                    auto t0 = std::chrono::steady_clock::now();
                    if (r > 0) {
                        const size_t gr = (size_t) r;
                        CK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gr, &lws, 0, nullptr, nullptr));
                        CK(clFlush(q));
                    }
                    if (r < N) cpu_rows(r);
                    CK(clFinish(q));
                    best2 = std::min(best2, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                printf("      split iGPU %3.0f %% / CPU: %.1f GB/s\n", 100.0 * r / N, wbytes / best2 / 1e9);
            }

        }
        fflush(stdout);
        clReleaseKernel(k);
        clReleaseProgram(prog);
    }
}
