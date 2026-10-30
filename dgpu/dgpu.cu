// SR-HPM: the discrete GPU as a second verifier for the layers whose weights stay in RAM.
// The rows it takes are copied once into pinned, mapped host memory; the kernels read them over PCIe (zero copy),
// so no VRAM is used for weights. While the GPU computes its rows the CPU threads compute the others.
// Loaded at run time (LoadLibrary) by ggml-cpu (srv-dgpu.inc) and by the tools. build: f3/dgpu/build.bat
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

#define DGPU_API extern "C" __declspec(dllexport)

static cudaStream_t g_stream = nullptr;

DGPU_API int dgpu_init() {
    if (g_stream) {
        return 0;
    }
    if (cudaSetDeviceFlags(cudaDeviceMapHost) != cudaSuccess) {
        cudaGetLastError();  // context already created by another user of the device in this process: mapping still works (UVA)
    }
    return cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking) == cudaSuccess ? 0 : -1;
}

// pinned + mapped host allocation; returns the host pointer
DGPU_API void * dgpu_host_alloc(size_t n) {
    void * p = nullptr;
    if (cudaHostAlloc(&p, n, cudaHostAllocMapped | cudaHostAllocWriteCombined) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return p;
}

DGPU_API void * dgpu_dev_ptr(void * host) {
    void * d = nullptr;
    return cudaHostGetDevicePointer(&d, host, 0) == cudaSuccess ? d : nullptr;
}

__global__ void k_read(const uint4 * __restrict__ p, size_t n16, unsigned * out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) {
        out[0] = acc;
    }
}

// streaming read of n bytes at host pointer p (mapped); blocking
DGPU_API int dgpu_read(void * host, size_t n, int blocks, int threads) {
    static unsigned * out = nullptr;
    if (!out) {
        cudaMalloc(&out, 64);
    }
    const uint4 * d = (const uint4 *) dgpu_dev_ptr(host);
    if (!d) {
        return -1;
    }
    k_read<<<blocks, threads, 0, g_stream>>>(d, n / 16, out);
    return cudaStreamSynchronize(g_stream) == cudaSuccess ? 0 : -2;
}

// ---------------------------------------------------------------------------------------------------------------
// matrix x few vectors. One warp per weight row; x (M rows of K floats) in VRAM; y[t * ldy + row]

#define QK 256
#define MMAX 8

__constant__ int8_t c_kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

struct blk_iq4_xs { __half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };       // 136 bytes
struct blk_q3_k   { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; __half d; };           // 110 bytes
static_assert(sizeof(blk_iq4_xs) == 136, "iq4_xs");
static_assert(sizeof(blk_q3_k) == 110, "q3_k");

template <int M>
__device__ __forceinline__ void warp_store(float (&acc)[MMAX], float * y, int64_t ldy, int64_t row) {
    #pragma unroll
    for (int t = 0; t < M; t++) {
        float v = acc[t];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, o);
        }
        if ((threadIdx.x & 31) == 0) {
            y[t * ldy + row] = v;
        }
    }
}

// each warp streams its row through shared memory in chunks of CH blocks with consecutive 4-byte loads: over PCIe,
// scattered small loads of headers and scales would waste most of every transaction
#define CH 8
#define WARPS 4

__device__ __forceinline__ void warp_fill(uint32_t * sm, const uint32_t * __restrict__ g, int nwords, int lane) {
    for (int w = lane; w < nwords; w += 32) {
        sm[w] = g[w];
    }
    __syncwarp();
}

template <int M>
__global__ void k_iq4_xs(const blk_iq4_xs * __restrict__ w, int64_t nrows, int64_t nsb, const float * __restrict__ x, int64_t K,
                         float * __restrict__ y, int64_t ldy) {
    __shared__ uint32_t sm_all[WARPS][CH * sizeof(blk_iq4_xs) / 4];
    const int64_t row  = blockIdx.x * (int64_t) WARPS + threadIdx.x / 32;
    const int     lane = threadIdx.x & 31;
    if (row >= nrows) {
        return;
    }
    uint32_t * sm = sm_all[threadIdx.x / 32];
    float acc[MMAX] = {};
    const blk_iq4_xs * rb = w + row * nsb;
    const int ib = lane / 4;              // 32-weight sub-block
    const int j0 = 4 * (lane % 4);        // first of the 4 low-nibble weights of this lane
    for (int64_t c = 0; c < nsb; c += CH) {
        const int nb = (int) min((int64_t) CH, nsb - c);
        __syncwarp();
        warp_fill(sm, (const uint32_t *) (rb + c), nb * (int) sizeof(blk_iq4_xs) / 4, lane);
        for (int bi = 0; bi < nb; bi++) {
            const blk_iq4_xs * b = (const blk_iq4_xs *) (sm + bi * sizeof(blk_iq4_xs) / 4);
            const uint32_t q  = *(const uint32_t *) (b->qs + 4 * lane);
            const int      ls = ((b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((b->scales_h >> (2 * ib)) & 3) << 4);
            const float    dl = __half2float(b->d) * (float) (ls - 32);
            float wl[4], wh[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int byte = (q >> (8 * i)) & 0xff;
                wl[i] = dl * c_kvalues_iq4nl[byte & 0xf];
                wh[i] = dl * c_kvalues_iq4nl[byte >> 4];
            }
            const int64_t k0 = (c + bi) * QK + 32 * ib + j0;
            #pragma unroll
            for (int t = 0; t < M; t++) {
                const float4 xl = *(const float4 *) (x + t * K + k0);
                const float4 xh = *(const float4 *) (x + t * K + k0 + 16);
                acc[t] += wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w
                        + wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
            }
        }
    }
    warp_store<M>(acc, y, ldy, row);
}

// Q3_K rows must start on a 4-byte boundary (nsb even), checked by the caller
template <int M>
__global__ void k_q3_k(const blk_q3_k * __restrict__ w, int64_t nrows, int64_t nsb, const float * __restrict__ x, int64_t K,
                       float * __restrict__ y, int64_t ldy) {
    __shared__ uint32_t sm_all[WARPS][CH * sizeof(blk_q3_k) / 4 + 1];
    const int64_t row  = blockIdx.x * (int64_t) WARPS + threadIdx.x / 32;
    const int     l    = threadIdx.x & 31;
    if (row >= nrows) {
        return;
    }
    uint32_t * sm = sm_all[threadIdx.x / 32];
    float acc[MMAX] = {};
    const blk_q3_k * rb = w + row * nsb;
    for (int64_t c = 0; c < nsb; c += CH) {
        const int nb = (int) min((int64_t) CH, nsb - c);
        __syncwarp();
        warp_fill(sm, (const uint32_t *) (rb + c), (nb * (int) sizeof(blk_q3_k) + 3) / 4, l);
        for (int bi = 0; bi < nb; bi++) {
            const blk_q3_k * b = (const blk_q3_k *) ((const uint8_t *) sm + bi * sizeof(blk_q3_k));
            const float d  = __half2float(b->d);
            const int   hm = b->hmask[l];
            #pragma unroll
            for (int nn = 0; nn < 2; nn++) {
                const int qb = b->qs[32 * nn + l];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int is = 8 * nn + 2 * j + (l >> 4);
                    const int lo = is < 8 ? (b->scales[is] & 0xf) : (b->scales[is - 8] >> 4);
                    const int hi = (b->scales[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
                    const int q  = ((qb >> (2 * j)) & 3) - ((hm >> (4 * nn + j)) & 1 ? 0 : 4);
                    const float wv = d * (float) ((lo | (hi << 4)) - 32) * (float) q;
                    const int64_t k = (c + bi) * QK + 128 * nn + 32 * j + l;
                    #pragma unroll
                    for (int t = 0; t < M; t++) {
                        acc[t] += wv * x[t * K + k];
                    }
                }
            }
        }
    }
    warp_store<M>(acc, y, ldy, row);
}

// ---------------------------------------------------------------------------------------------------------------
// weights: one pinned copy of the GPU's rows per tensor, keyed by the tensor's data pointer

struct dgpu_w { void * host; const void * dev; size_t bytes; };
static std::mutex                              g_mtx;
static std::unordered_map<const void *, dgpu_w> g_w;
static size_t                                  g_pinned = 0;

// staging: x (pinned) -> x (device); y written by the kernel straight into pinned mapped memory
static float * g_xh = nullptr;
static float * g_xd = nullptr;
static float * g_yh = nullptr;
static float * g_yd = nullptr;
static size_t  g_xcap = 0, g_ycap = 0;
static cudaEvent_t g_done = nullptr;

static bool ensure_stage(size_t xn, size_t yn) {
    if (xn > g_xcap) {
        if (g_xh) { cudaFreeHost(g_xh); cudaFree(g_xd); }
        if (cudaHostAlloc((void **) &g_xh, xn * 4, cudaHostAllocDefault) != cudaSuccess || cudaMalloc((void **) &g_xd, xn * 4) != cudaSuccess) {
            g_xcap = 0;
            return false;
        }
        g_xcap = xn;
    }
    if (yn > g_ycap) {
        if (g_yh) { cudaFreeHost(g_yh); }
        if (cudaHostAlloc((void **) &g_yh, yn * 4, cudaHostAllocMapped) != cudaSuccess) {
            g_ycap = 0;
            return false;
        }
        cudaHostGetDevicePointer((void **) &g_yd, g_yh, 0);
        g_ycap = yn;
    }
    if (!g_done) {
        cudaEventCreateWithFlags(&g_done, cudaEventDisableTiming);
    }
    return true;
}

DGPU_API size_t dgpu_pinned_bytes() { return g_pinned; }

DGPU_API void dgpu_mem_info(size_t * free_b, size_t * total_b) {
    if (cudaMemGetInfo(free_b, total_b) != cudaSuccess) {
        *free_b = *total_b = 0;
    }
}

// kind 0 = IQ4_XS, 1 = Q3_K. rows: the first nrows rows of the tensor (native layout, nsb blocks of 256 per row).
// x: M rows of K floats, row stride x_nb bytes. Asynchronous: dgpu_collect() waits and copies y out.
// Returns 0 on success; on failure nothing was launched.
DGPU_API int dgpu_launch(int kind, const void * key, const void * rows, int64_t nrows, int64_t K, int M,
                         const float * x, size_t x_nb, size_t max_pinned) {
    if (M < 1 || M > MMAX || K % QK) {
        return -1;
    }
    const int64_t nsb = K / QK;
    const size_t  bsz = kind == 0 ? sizeof(blk_iq4_xs) : sizeof(blk_q3_k);
    const size_t  bytes = (size_t) nrows * nsb * bsz;
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_w.find(key);
    if (it == g_w.end() || it->second.bytes < bytes) {
        if (g_pinned + bytes > max_pinned) {
            return -2;
        }
        void * h = dgpu_host_alloc(bytes);
        if (!h) {
            return -3;
        }
        memcpy(h, rows, bytes);
        if (it != g_w.end()) {
            g_pinned -= it->second.bytes;
            cudaFreeHost(it->second.host);
        }
        g_w[key] = { h, dgpu_dev_ptr(h), bytes };
        g_pinned += bytes;
        it = g_w.find(key);
    }
    if (!ensure_stage((size_t) M * K, (size_t) M * nrows)) {
        return -4;
    }
    for (int t = 0; t < M; t++) {
        memcpy(g_xh + (size_t) t * K, (const char *) x + t * x_nb, K * sizeof(float));
    }
    cudaMemcpyAsync(g_xd, g_xh, (size_t) M * K * 4, cudaMemcpyHostToDevice, g_stream);
    if (kind == 1 && (nsb & 1)) {
        return -6;  // Q3_K rows would not be 4-byte aligned
    }
    const dim3 grid((unsigned) ((nrows + WARPS - 1) / WARPS)), block(32 * WARPS);
#define LAUNCH(KER, BT) \
    switch (M) { \
        case 1: KER<1><<<grid, block, 0, g_stream>>>((const BT *) it->second.dev, nrows, nsb, g_xd, K, g_yd, nrows); break; \
        case 2: KER<2><<<grid, block, 0, g_stream>>>((const BT *) it->second.dev, nrows, nsb, g_xd, K, g_yd, nrows); break; \
        case 3: KER<3><<<grid, block, 0, g_stream>>>((const BT *) it->second.dev, nrows, nsb, g_xd, K, g_yd, nrows); break; \
        case 4: KER<4><<<grid, block, 0, g_stream>>>((const BT *) it->second.dev, nrows, nsb, g_xd, K, g_yd, nrows); break; \
        case 5: KER<5><<<grid, block, 0, g_stream>>>((const BT *) it->second.dev, nrows, nsb, g_xd, K, g_yd, nrows); break; \
        case 6: KER<6><<<grid, block, 0, g_stream>>>((const BT *) it->second.dev, nrows, nsb, g_xd, K, g_yd, nrows); break; \
        case 7: KER<7><<<grid, block, 0, g_stream>>>((const BT *) it->second.dev, nrows, nsb, g_xd, K, g_yd, nrows); break; \
        default: KER<8><<<grid, block, 0, g_stream>>>((const BT *) it->second.dev, nrows, nsb, g_xd, K, g_yd, nrows); break; \
    }
    if (kind == 0) {
        LAUNCH(k_iq4_xs, blk_iq4_xs)
    } else {
        LAUNCH(k_q3_k, blk_q3_k)
    }
#undef LAUNCH
    cudaEventRecord(g_done, g_stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -5;
}

// waits for the last launch and writes y[t][0..nrows) to dst + t * dst_nb (bytes)
DGPU_API int dgpu_collect(int M, int64_t nrows, char * dst, size_t dst_nb) {
    // spin rather than block: the wait is short and the caller's other threads are spinning too
    while (cudaEventQuery(g_done) == cudaErrorNotReady) {
    }
    for (int t = 0; t < M; t++) {
        memcpy(dst + t * dst_nb, g_yh + (size_t) t * nrows, nrows * sizeof(float));
    }
    return 0;
}
