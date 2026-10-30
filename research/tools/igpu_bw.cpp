// Can the integrated GPU add DRAM throughput while the CPU kernels are compute-bound (8 tokens)?
// Measures: iGPU streaming read bandwidth alone (OpenCL, zero-copy host buffer), CPU SR-HPM kernel at T=8 alone, and
// both at the same time on separate 1 GB buffers.
// build: clang++ -O3 -march=native -std=c++17 with the CUDA toolkit OpenCL headers, linking OpenCL.lib
#define CL_TARGET_OPENCL_VERSION 120
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <CL/cl.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
static inline float fp16_to_f32(ggml_half h) { return _cvtsh_ss(h); }
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_f32(x)
#include "srv-kernels.inc"
using namespace ggml::cpu::repack;

static const char * KSRC = R"(
__kernel void rd(__global const uint4 * p, __global uint * out, const uint n16) {
    uint acc = 0;
    for (uint i = get_global_id(0); i < n16; i += get_global_size(0)) {
        uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    out[get_global_id(0)] = acc;
}
)";

#define CK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { printf("OpenCL error %d at %s\n", e_, #x); exit(1); } } while (0)

int main() {
    cl_uint np = 0;
    CK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> ps(np);
    CK(clGetPlatformIDs(np, ps.data(), nullptr));
    cl_device_id dev = nullptr;
    for (auto p : ps) {
        char name[256];
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        cl_device_id d;
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, &nd) == CL_SUCCESS && nd) {
            char dn[256];
            cl_bool um = 0;
            clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(um), &um, nullptr);
            printf("platform %s: device %s (unified memory %d)\n", name, dn, (int) um);
            if (um) dev = d;
        }
    }
    if (!dev) { printf("no integrated GPU with unified memory\n"); return 1; }
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err); CK(err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KSRC, nullptr, &err); CK(err);
    CK(clBuildProgram(prog, 1, &dev, "", nullptr, nullptr));
    cl_kernel k = clCreateKernel(prog, "rd", &err); CK(err);

    const size_t GB = (size_t) 1 << 30;
    // zero-copy: the driver allocates host memory the iGPU reads directly
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, GB, nullptr, &err); CK(err);
    void * hp = clEnqueueMapBuffer(q, buf, CL_TRUE, CL_MAP_WRITE, 0, GB, 0, nullptr, nullptr, &err); CK(err);
    for (size_t i = 0; i < GB; i += 64) ((uint8_t *) hp)[i] = (uint8_t) i;
    CK(clEnqueueUnmapMemObject(q, buf, hp, 0, nullptr, nullptr));
    const size_t gws = 8 * 8 * 64 * 32;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, gws * 4, nullptr, &err); CK(err);
    const cl_uint n16 = (cl_uint) (GB / 16);
    CK(clSetKernelArg(k, 0, sizeof(buf), &buf));
    CK(clSetKernelArg(k, 1, sizeof(out), &out));
    CK(clSetKernelArg(k, 2, sizeof(n16), &n16));
    auto gpu_pass = [&] {
        CK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr));
        CK(clFinish(q));
    };
    gpu_pass();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; r++) gpu_pass();
    const double g_alone = 5.0 * GB / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9;
    printf("iGPU alone: %.1f GB/s\n", g_alone);

    // CPU: IQ4_XS paired kernel at T=8 over its own 1 GB, 8 threads
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

    // both at once: the iGPU loops over its buffer while the CPU does 3 passes over its own
    std::atomic<bool> stop{false};
    std::atomic<long long> gbytes{0};
    std::thread gt([&] { while (!stop) { gpu_pass(); gbytes += GB; } });
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 3; r++) cpu_pass();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop = true;
    gt.join();
    printf("together: CPU %.1f GB/s + iGPU %.1f GB/s = %.1f GB/s\n", 3.0 * GB / sec / 1e9, gbytes / sec / 1e9,
           (3.0 * GB + gbytes) / sec / 1e9);
}
