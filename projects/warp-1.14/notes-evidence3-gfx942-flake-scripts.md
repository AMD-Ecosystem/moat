# warp-1.14 evidence session 3 (2026-11-20, linux-gfx942): versions, experiments, report scripts

Under the job worktree's `agent_space/e3/`. `lost_writes.hip` (the final reproducer) is inline in the draft in `notes.md`. Builds: `<rocm root>/bin/hipcc --offload-arch=gfx942 -O2`, one binary set per ROCm version (`r101/`, `r102/`, `nightly/`), run with `LD_LIBRARY_PATH=<rocm root>/lib`.

## `exp.hip`

```
// exp.hip: copy_flake2's sequence with two knobs, run under external load (8 copy_flake load processes).
// Usage: exp <iterations> <n floats> <reader>   reader = full (A read by every block, as in copy_flake2),
//        one (A read by one 256-thread workgroup only), part (only the first 1/8 of A read, by every block of that part),
//        none (A written by the H2D copy and never read by a kernel; Y filled from host instead)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)
__global__ void copy_strided(float* dst, const float* src, size_t ds, size_t ss, size_t n)
{ size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) dst[i * ds] = src[i * ss]; }
__global__ void copy_one_wg(float* dst, const float* src, size_t ds, size_t ss, size_t n)
{ for (size_t i = threadIdx.x; i < n; i += blockDim.x) dst[i * ds] = src[i * ss]; }
__global__ void zero_strided(float* dst, size_t s, size_t n)
{ size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) dst[i * s] = 0.0f; }
int main(int argc, char** argv)
{
    int iters = atoi(argv[1]); size_t n = strtoull(argv[2], 0, 0); const char* rd = argv[3];
    int blocks = (int)((n + 255) / 256);
    hipStream_t S; CK(hipStreamCreateWithPriority(&S, 0, 0));
    float* src = (float*)malloc(n * 4); float* zeros = (float*)calloc(n, 4); float* out = (float*)malloc(n * 4);
    float* src2 = (float*)malloc(2 * n * 4);
    int bad = 0, reused = 0, aligned = 0;
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < n; ++i) src[i] = float(it % 1000 + i);
        float *A, *Y, *D;
        CK(hipMalloc(&A, n * 4));
        CK(hipMemcpyAsync(A, src, n * 4, hipMemcpyHostToDevice, S));
        CK(hipMalloc(&Y, 2 * n * 4));
        if (!strcmp(rd, "none")) {
            for (size_t i = 0; i < n; ++i) { src2[2 * i] = src[i]; src2[2 * i + 1] = 0; }
            CK(hipMemcpyAsync(Y, src2, 2 * n * 4, hipMemcpyHostToDevice, S));
        } else {
            zero_strided<<<blocks, 256, 0, S>>>(Y, 2, n);
            if (!strcmp(rd, "one")) copy_one_wg<<<1, 256, 0, S>>>(Y, A, 2, 1, n);
            else if (!strcmp(rd, "part")) {
                copy_strided<<<(int)((n / 8 + 255) / 256), 256, 0, S>>>(Y, A, 2, 1, n / 8);
                for (size_t i = n / 8; i < n; ++i) { src2[2 * i] = src[i]; src2[2 * i + 1] = 0; }
                CK(hipMemcpyAsync(Y + 2 * (n / 8), src2 + 2 * (n / 8), (2 * n - 2 * (n / 8)) * 4, hipMemcpyHostToDevice, S));
            } else copy_strided<<<blocks, 256, 0, S>>>(Y, A, 2, 1, n);
        }
        CK(hipFree(A)); CK(hipMalloc(&D, n * 4));
        if (D == A) ++reused;
        if (((uintptr_t)D & ((2u << 20) - 1)) == 0) ++aligned;
        CK(hipMemcpyAsync(D, zeros, n * 4, hipMemcpyHostToDevice, S));
        CK(hipDeviceSynchronize());
        copy_strided<<<blocks, 256, 0, S>>>(D, Y, 1, 2, n);
        CK(hipStreamSynchronize(S));
        CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost));
        int cls[8] = {0}; size_t w = 0;
        for (size_t i = 0; i < n; ++i) if (out[i] != src[i]) { ++w; ++cls[(i / 256) % 8]; }
        if (w) { ++bad; printf("iter %d D=%p wrong %zu per class %d %d %d %d %d %d %d %d\n", it, (void*)D, w, cls[0], cls[1], cls[2], cls[3], cls[4], cls[5], cls[6], cls[7]); fflush(stdout); }
        CK(hipFree(D)); CK(hipFree(Y));
    }
    printf("%d of %d iterations wrong (n=%zu floats = %zu bytes, reader %s; D reused A's address %d times, 2 MiB-aligned %d times)\n", bad, iters, n, n * 4, rd, reused, aligned);
    return bad ? 1 : 0;
}
```

## `vrun.sh`

```
#!/bin/bash
# vrun.sh <label> <bindir> <rocm root>: copy_flake2 twice under 8 copy_flake load processes, then lost_writes once,
# all built with and run against the ROCm at <rocm root> (LD_LIBRARY_PATH=<root>/lib). Load PIDs kept, killed by PID.
ulimit -c 0; D=$(dirname $(readlink -f $0)); L=$1; B=$2; R=$3; O=$D/logs/$L; mkdir -p $O
export LD_LIBRARY_PATH=$R/lib
{ echo "hipconfig --version: $($R/bin/hipconfig --version)"; echo "kernel: $(uname -r)"; echo "amdgpu: $(cat /sys/module/amdgpu/version)"
  echo "libamdhip64: $(ldd $B/copy_flake2 | grep amdhip)"; } > $O/versions.txt
for r in 1 2; do
  pids=(); for k in $(seq 8); do $B/copy_flake load 7200 4 > /dev/null 2>&1 & pids+=($!); done; sleep 2
  timeout 1500 stdbuf -oL $B/copy_flake2 40000 > $O/copy_flake2_$r.log 2>&1
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done; wait "${pids[@]}" 2>/dev/null
  echo "$L copy_flake2 run $r: $(tail -1 $O/copy_flake2_$r.log)"
done
timeout 1500 stdbuf -oL $B/lost_writes 40000 8 > $O/lost_writes.log 2>&1; echo "$L lost_writes: rc=$? $(tail -1 $O/lost_writes.log)"
```

## `batch3.sh`

```
#!/bin/bash
# batch3.sh: ROCm 10.1 (image) runs, then size / last-reader experiments and environment variables on the nightly
ulimit -c 0; D=$(dirname $(readlink -f $0))
R1=$(/opt/python-3.11-venv/bin/python -m rocm_sdk path --root); RN=$($FLEET_VENV/bin/python -m rocm_sdk path --root)
$D/vrun.sh rocm10.1.0-image $D/r101 $R1
export LD_LIBRARY_PATH=$RN/lib; B=$D/nightly; O=$D/logs/exp; mkdir -p $O
ex() { L=$1; shift
  pids=(); for k in $(seq 8); do $B/copy_flake load 7200 4 > /dev/null 2>&1 & pids+=($!); done; sleep 2
  timeout 1500 stdbuf -oL $B/exp "$@" > $O/$L.log 2>&1
  for p in "${pids[@]}"; do kill $p 2>/dev/null; done; wait "${pids[@]}" 2>/dev/null
  echo "exp $L: $(tail -1 $O/$L.log)"
}
ex size_1MiB 40000 262144 full
ex size_2MiB 40000 524288 full
ex size_4MB 40000 1000000 full
ex size_4MiB 40000 1048576 full
ex size_16MiB 10000 4194304 full
ex reader_one 40000 1000000 one
ex reader_part 40000 1000000 part
ex reader_none 40000 1000000 none
for e in "HSA_XNACK=1" "HSA_ENABLE_SDMA=0" "GPU_MAX_HW_QUEUES=1"; do
  env $e timeout 1500 stdbuf -oL $B/lost_writes 40000 8 > $O/env_${e%%=*}.log 2>&1
  echo "env $e lost_writes: rc=$? $(tail -1 $O/env_${e%%=*}.log)"
done
```

## `batch4.sh`

```
#!/bin/bash
# batch4.sh: lost_writes on ROCm 10.2.0: default, noreuse control, no-load control
ulimit -c 0; D=$(dirname $(readlink -f $0)); R=$($FLEET_VENV/bin/python -m rocm_sdk path --root); export LD_LIBRARY_PATH=$R/lib
O=$D/logs/rocm10.2.0; mkdir -p $O
{ echo "hipconfig --version: $($R/bin/hipconfig --version)"; echo "kernel: $(uname -r)"; echo "amdgpu: $(cat /sys/module/amdgpu/version)"; } > $O/versions.txt
for spec in "default 40000 8" "noreuse 40000 8 noreuse" "noload 40000 0"; do
  set -- $spec; L=$1; shift
  timeout 1500 stdbuf -oL $D/r102/lost_writes "$@" > $O/lost_writes_$L.log 2>&1
  echo "rocm10.2.0 lost_writes $L: rc=$? $(tail -1 $O/lost_writes_$L.log)"
done
```

## `batch5.sh`

```
#!/bin/bash
# batch5.sh: final lost_writes (no kernel touches the freed buffer): 10.2.0 default x2, noreuse, no load; 10.1.0 default x2
ulimit -c 0; D=$(dirname $(readlink -f $0))
R=$($FLEET_VENV/bin/python -m rocm_sdk path --root); R1=$(/opt/python-3.11-venv/bin/python -m rocm_sdk path --root)
run() { L=$1; B=$2; LIB=$3; shift 3
  LD_LIBRARY_PATH=$LIB timeout 1500 stdbuf -oL $B "$@" > $D/logs/final_$L.log 2>&1
  echo "$L: rc=$? $(tail -1 $D/logs/final_$L.log)"
}
run r102_default_1 $D/r102/lost_writes_final $R/lib 40000
run r102_default_2 $D/r102/lost_writes_final $R/lib 40000
run r102_noreuse $D/r102/lost_writes_final $R/lib 40000 8 noreuse
run r102_noload $D/r102/lost_writes_final $R/lib 40000 0
run r101_default_1 $D/r101/lost_writes_final $R1/lib 40000
run r101_default_2 $D/r101/lost_writes_final $R1/lib 40000
```

## `lw_dbg.hip`

```
// lost_writes.hip plus, on the first failure, a dump of the read-back and the expected values to lw_dump.bin, then stop
104a105,106
>             { FILE* f = fopen("lw_dump.bin", "wb"); fwrite(out, 4, n, f); fwrite(src, 4, n, f); fclose(f); }
>             iters = it + 1;
```

