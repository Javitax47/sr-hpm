// Two feasibility checks for using the integrated GPU on the model's own weights:
//  1. iGPU read bandwidth over ordinary cacheable host memory (_aligned_malloc + CL_MEM_USE_HOST_PTR), and the CPU's
//     read bandwidth over the same memory (the model's weights live in such memory)
//  2. launch overhead: enqueue + finish of an empty kernel
// build: clang++ -O2 -std=c++17 -I<cuda>/include igpu_hostptr.cpp -L<cuda>/lib/x64 -lOpenCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <immintrin.h>
#include <malloc.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

#define CK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { printf("OpenCL error %d at line %d\n", e_, __LINE__); exit(1); } } while (0)

static const char * KSRC = R"CLC(
__kernel void rd(__global const uint4 * p, __global uint * out, const uint n16) {
    uint acc = 0;
    for (uint i = get_global_id(0); i < n16; i += get_global_size(0)) { uint4 v = p[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    out[get_global_id(0)] = acc;
}
__kernel void nop(__global uint * out) { if (get_global_id(0) == 123456789) out[0] = 1; }
)CLC";

int main() {
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
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err); CK(err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KSRC, nullptr, &err); CK(err);
    CK(clBuildProgram(prog, 1, &dev, "", nullptr, nullptr));
    cl_kernel krd = clCreateKernel(prog, "rd", &err); CK(err);
    cl_kernel knop = clCreateKernel(prog, "nop", &err); CK(err);

    const size_t GB = (size_t) 1 << 30;
    auto * host = (uint8_t *) _aligned_malloc(GB, 4096);
    for (size_t i = 0; i < GB; i += 64) host[i] = (uint8_t) i;
    auto t0 = std::chrono::steady_clock::now();
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, GB, host, &err); CK(err);
    printf("clCreateBuffer(USE_HOST_PTR, 1 GB): %.1f ms\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3);
    const size_t gws = 8 * 8 * 64 * 32;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, gws * 4, nullptr, &err); CK(err);
    const cl_uint n16 = (cl_uint) (GB / 16);
    CK(clSetKernelArg(krd, 0, sizeof(buf), &buf)); CK(clSetKernelArg(krd, 1, sizeof(out), &out)); CK(clSetKernelArg(krd, 2, sizeof(n16), &n16));
    CK(clSetKernelArg(knop, 0, sizeof(out), &out));
    auto gpu = [&] { CK(clEnqueueNDRangeKernel(q, krd, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr)); CK(clFinish(q)); };
    gpu();
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; r++) gpu();
    printf("iGPU over malloc'd memory: %.1f GB/s\n", 5.0 * GB / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9);

    // CPU over the same memory
    auto cpu = [&] {
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        std::atomic<uint64_t> sink{0};
        for (int i = 0; i < 8; i++)
            th.emplace_back([&] {
                __m256i a = _mm256_setzero_si256();
                for (size_t c; (c = next.fetch_add(1 << 20)) < GB;)
                    for (size_t j = c; j < c + (1 << 20); j += 32) a = _mm256_xor_si256(a, _mm256_load_si256((const __m256i *) (host + j)));
                sink += _mm256_extract_epi64(a, 0);
            });
        for (auto & t : th) t.join();
    };
    cpu();
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 5; r++) cpu();
    printf("CPU over the same memory: %.1f GB/s\n", 5.0 * GB / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e9);

    // launch overhead
    const size_t g1 = 64;
    for (int r = 0; r < 20; r++) { CK(clEnqueueNDRangeKernel(q, knop, 1, nullptr, &g1, &g1, 0, nullptr, nullptr)); CK(clFinish(q)); }
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 500; r++) { CK(clEnqueueNDRangeKernel(q, knop, 1, nullptr, &g1, &g1, 0, nullptr, nullptr)); CK(clFinish(q)); }
    printf("empty kernel enqueue + finish: %.0f us\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 500 * 1e6);
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < 500; r++) CK(clEnqueueNDRangeKernel(q, knop, 1, nullptr, &g1, &g1, 0, nullptr, nullptr));
    CK(clFinish(q));
    printf("empty kernel, 500 queued then one finish: %.0f us each\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 500 * 1e6);
}
