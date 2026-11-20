/* Diagnostic only, never part of the port: global-memory emulation of the HIP 3D texture API, so TIGRE's
 * projectors build and run on gfx94x (MI300), whose HIP has no image/texture API (__HIP_NO_IMAGE_SUPPORT).
 * Paste this block into Common/CUDA/cuda_to_hip.h just before the "Runtime verdict for fp32" comment (inside the
 * USE_HIP branch, after the cuda* -> hip* defines), and make tigre_hw_linear_supported() return false in
 * gpuUtils.cu. Every tex3D is then a point fetch with border-zero addressing from a linear buffer, so
 * tex3D_TIGRE takes its software 8-tap trilinear path, numerically the same as MI250X's point-texture fallback.
 * Used 2026-11-20 (gfx942) to reproduce the gfx90a full-turn test failure without a texture unit. */
#define TIGRE_TEXEMU 1
#if defined(TIGRE_TEXEMU)
struct TigreEmuArr { float* d; size_t w, h, z; };
struct TigreEmuTex { const float* d; int w, h, z; };
static inline hipError_t tigre_emu_Malloc3DArray(hipArray_t* arr, const hipChannelFormatDesc*, hipExtent e) {
    TigreEmuArr* a = new TigreEmuArr; a->w = e.width; a->h = e.height ? e.height : 1; a->z = e.depth ? e.depth : 1;
    hipError_t r = hipMalloc((void**)&a->d, a->w * a->h * a->z * sizeof(float));
    if (r != hipSuccess) { delete a; *arr = nullptr; return r; }
    *arr = (hipArray_t)a; return hipSuccess;
}
static inline hipError_t tigre_emu_FreeArray(hipArray_t arr) {
    TigreEmuArr* a = (TigreEmuArr*)arr; if (!a) return hipSuccess; hipFree(a->d); delete a; return hipSuccess;
}
static inline hipError_t tigre_emu_ArrayGetInfo(hipChannelFormatDesc* d, hipExtent* e, unsigned int* f, hipArray_t arr) {
    TigreEmuArr* a = (TigreEmuArr*)arr; *e = make_hipExtent(a->w, a->h, a->z); if (f) *f = 0;
    if (d) *d = hipCreateChannelDesc<float>(); return hipSuccess;
}
static inline hipMemcpy3DParms tigre_emu_fix(const hipMemcpy3DParms* p) {
    hipMemcpy3DParms q = *p;
    if (p->dstArray) { TigreEmuArr* a = (TigreEmuArr*)p->dstArray; q.dstArray = nullptr;
        q.dstPtr = make_hipPitchedPtr(a->d, a->w * sizeof(float), a->w, a->h); }
    if (p->srcArray) { TigreEmuArr* a = (TigreEmuArr*)p->srcArray; q.srcArray = nullptr;
        q.srcPtr = make_hipPitchedPtr(a->d, a->w * sizeof(float), a->w, a->h); }
    q.extent.width *= sizeof(float); q.srcPos.x *= sizeof(float); q.dstPos.x *= sizeof(float); /* linear copies count bytes */
    if (q.kind == hipMemcpyHostToDevice || q.kind == hipMemcpyDeviceToDevice || q.kind == hipMemcpyDefault) q.kind = hipMemcpyDefault;
    return q;
}
static inline hipError_t tigre_emu_Memcpy3DAsync(const hipMemcpy3DParms* p, hipStream_t s = 0) {
    hipMemcpy3DParms q = tigre_emu_fix(p); return hipMemcpy3DAsync(&q, s);
}
static inline hipError_t tigre_emu_Memcpy3D(const hipMemcpy3DParms* p) {
    hipMemcpy3DParms q = tigre_emu_fix(p); return hipMemcpy3D(&q);
}
static inline hipError_t tigre_emu_CreateTextureObject(hipTextureObject_t* t, const hipResourceDesc* r, const hipTextureDesc*, const void*) {
    TigreEmuArr* a = (TigreEmuArr*)r->res.array.array;
    TigreEmuTex h{a->d, (int)a->w, (int)a->h, (int)a->z};
    void* dh = nullptr; hipError_t e = hipMalloc(&dh, sizeof h); if (e) return e;
    e = hipMemcpy(dh, &h, sizeof h, hipMemcpyHostToDevice); *t = (hipTextureObject_t)dh; return e;
}
static inline hipError_t tigre_emu_DestroyTextureObject(hipTextureObject_t t) { return hipFree((void*)t); }
#undef cudaMalloc3DArray
#undef cudaFreeArray
#undef cudaMemcpy3D
#undef cudaMemcpy3DAsync
#undef cudaCreateTextureObject
#undef cudaDestroyTextureObject
#undef cudaArrayGetInfo
#define cudaMalloc3DArray(arr, desc, ext) tigre_emu_Malloc3DArray((arr), (desc), (ext))
#define cudaFreeArray                   tigre_emu_FreeArray
#define cudaMemcpy3D                    tigre_emu_Memcpy3D
#define cudaMemcpy3DAsync               tigre_emu_Memcpy3DAsync
#define cudaCreateTextureObject         tigre_emu_CreateTextureObject
#define cudaDestroyTextureObject        tigre_emu_DestroyTextureObject
#define cudaArrayGetInfo                tigre_emu_ArrayGetInfo
#if defined(__HIPCC__)
template <typename T>
__device__ __forceinline__ T tigre_emu_tex3D(hipTextureObject_t t, float x, float y, float z) {
    const TigreEmuTex* h = (const TigreEmuTex*)t;
    const int i = (int)floorf(x), j = (int)floorf(y), k = (int)floorf(z);
    if (i < 0 || j < 0 || k < 0 || i >= h->w || j >= h->h || k >= h->z) return T(0);
    return h->d[((size_t)k * h->h + j) * h->w + i];
}
#define tex3D tigre_emu_tex3D
#endif
#endif
