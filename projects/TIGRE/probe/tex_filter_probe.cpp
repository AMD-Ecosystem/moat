// tex_filter_probe.cpp -- per-arch probe for hardware linear filtering of an
// fp32, element-read 3D texture (the TIGRE / popsift texture class).
//
// CUDA supports cudaFilterModeLinear + cudaReadModeElementType over a float
// array; on gfx90a (CDNA2, ROCm 7.2.1) hipCreateTextureObject REJECTS it at
// creation, which is why TIGRE does the trilinear lerp in software. This probe
// establishes the per-arch truth on the HARDWARE path (no software fallback):
// does creation succeed, and if so does the sampler actually interpolate or
// silently point-sample (the dangerous wrong-but-no-error case)?
//
// Build (HIP): hipcc -o tex_filter_probe tex_filter_probe.cpp
// Build (CUDA reference): nvcc -x cu -o tex_filter_probe tex_filter_probe.cpp
// Run: ./tex_filter_probe   (prints one of REJECTED / LINEAR / POINT)
//
// The array is filled so texel[x][y][z] = x. With CUDA's unnormalized -0.5
// texel-center convention, sampling at x=2.0 yields 1.5 under linear filtering
// and 2.0 under point sampling -- an unambiguous discriminator.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily

#include <cstdio>
#include <cmath>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define API "HIP"
#else
#include <cuda_runtime.h>
// Map the HIP names this file uses onto the CUDA runtime for the reference build.
#define hipArray_t                         cudaArray_t
#define hipExtent                          cudaExtent
#define make_hipExtent                     make_cudaExtent
#define hipChannelFormatDesc               cudaChannelFormatDesc
#define hipCreateChannelDesc               cudaCreateChannelDesc
#define hipMalloc3DArray                   cudaMalloc3DArray
#define hipMemcpy3DParms                   cudaMemcpy3DParms
#define make_hipPitchedPtr                 make_cudaPitchedPtr
#define hipMemcpy3D                        cudaMemcpy3D
#define hipMemcpyHostToDevice              cudaMemcpyHostToDevice
#define hipPos                             cudaPos
#define make_hipPos                        make_cudaPos
#define hipResourceDesc                    cudaResourceDesc
#define hipResourceTypeArray              cudaResourceTypeArray
#define hipTextureDesc                     cudaTextureDesc
#define hipAddressModeBorder               cudaAddressModeBorder
#define hipFilterModeLinear                cudaFilterModeLinear
#define hipReadModeElementType             cudaReadModeElementType
#define hipTextureObject_t                 cudaTextureObject_t
#define hipCreateTextureObject            cudaCreateTextureObject
#define hipDestroyTextureObject           cudaDestroyTextureObject
#define hipMalloc                          cudaMalloc
#define hipMemcpy                          cudaMemcpy
#define hipMemcpyDeviceToHost              cudaMemcpyDeviceToHost
#define hipDeviceSynchronize              cudaDeviceSynchronize
#define hipGetErrorString                 cudaGetErrorString
#define hipError_t                         cudaError_t
#define hipSuccess                         cudaSuccess
#define hipGetLastError                    cudaGetLastError
#define API "CUDA"
#endif

__global__ void sample_kernel(hipTextureObject_t tex, float* out) {
    // texel[x]=x; at x=2.0 linear->1.5, point->2.0 (y,z held at texel centers).
    *out = tex3D<float>(tex, 2.0f, 0.5f, 0.5f);
}

int main() {
    const int N = 4;
    float host[N * N * N];
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x)
                host[(z * N + y) * N + x] = (float)x;

    hipChannelFormatDesc ch = hipCreateChannelDesc<float>();
    hipArray_t arr;
    hipExtent ext = make_hipExtent(N, N, N);
    if (hipMalloc3DArray(&arr, &ch, ext, 0) != hipSuccess) {
        printf("%s: ERROR array alloc\n", API); return 2;
    }
    hipMemcpy3DParms cp = {};
    cp.srcPtr = make_hipPitchedPtr(host, N * sizeof(float), N, N);
    cp.dstArray = arr;
    cp.extent = ext;
    cp.kind = hipMemcpyHostToDevice;
    if (hipMemcpy3D(&cp) != hipSuccess) { printf("%s: ERROR memcpy\n", API); return 2; }

    hipResourceDesc rd = {};
    rd.resType = hipResourceTypeArray;
    rd.res.array.array = arr;
    hipTextureDesc td = {};
    td.addressMode[0] = hipAddressModeBorder;
    td.addressMode[1] = hipAddressModeBorder;
    td.addressMode[2] = hipAddressModeBorder;
    td.filterMode = hipFilterModeLinear;       // the capability under test
    td.readMode = hipReadModeElementType;      // raw fp32, not normalized
    td.normalizedCoords = 0;

    hipTextureObject_t tex = 0;
    hipError_t cerr = hipCreateTextureObject(&tex, &rd, &td, nullptr);
    if (cerr != hipSuccess) {
        printf("%s: REJECTED  (hipCreateTextureObject -> %s)\n", API, hipGetErrorString(cerr));
        return 0;  // the gfx90a behavior
    }

    float* dout; hipMalloc(&dout, sizeof(float));
    sample_kernel<<<1, 1>>>(tex, dout);
    hipDeviceSynchronize();
    float v = -1.0f;
    hipMemcpy(&v, dout, sizeof(float), hipMemcpyDeviceToHost);
    hipDestroyTextureObject(tex);

    const char* verdict =
        (fabsf(v - 1.5f) < 0.05f) ? "LINEAR (hardware trilinear works)" :
        (fabsf(v - 2.0f) < 0.05f) ? "POINT  (creation OK but SILENTLY point-samples -- wrong, no error)" :
                                     "OTHER";
    printf("%s: ACCEPTED, sample@2.0 = %.4f -> %s\n", API, v, verdict);
    return 0;
}
