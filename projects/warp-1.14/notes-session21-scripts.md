# warp-1.14 porting session 21: scripts (linux-gfx950)

Scripts referenced by "Porting session 21" in notes.md (agent_space/ is wiped). Session 20's runner scripts are in notes-session20-scripts.md.

## `relbuild.sh`

```
#!/bin/bash
# Release-configuration HIP build of tree $1 (default target list from the ROCm SDK), -j $2; log to $1/build_rel_j$2.log
T=$1; J=$2
P=$FLEET_VENV/bin/python
R=$($P -m rocm_sdk path --root)
cd "$T" && $P build_lib.py --rocm-path "$R" -j $J > build_rel_j$J.log 2>&1
rc=$?; tail -1 build_rel_j$J.log; exit $rc
```

## `suite_wheel.sh`

```
#!/bin/bash
# Full default suite from the installed wheel ($FLEET_VENV site-packages), run from a directory outside the source tree.
J=/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt
ulimit -c 0
mkdir -p $J/wheelrun && cd $J/wheelrun
HIP_VISIBLE_DEVICES=1 WARP_CACHE_PATH=$J/wheelrun/cache $FLEET_VENV/bin/python -c "import warp, os; print(warp.__file__); assert 'site-packages' in warp.__file__"
HIP_VISIBLE_DEVICES=1 WARP_CACHE_PATH=$J/wheelrun/cache $FLEET_VENV/bin/python -m warp.tests -s autodetect
```

## `suite_ff.sh`

```
#!/bin/bash
# Full default suite from the fp-contract scratch tree agent_space/wff, pinned GPU, private kernel cache.
A=/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/moat/agent_space
ulimit -c 0
cd $A/wff
HIP_VISIBLE_DEVICES=1 WARP_CACHE_PATH=$A/cache_ff $FLEET_VENV/bin/python -m warp.tests -s autodetect
```

## `flake_a.sh`

```
#!/bin/bash
# Flake reproduction (a): the full default suite runs on the pinned GPU in the background (its parallel workers),
# while the test_async s2f_d2d / s2c_d2d subset runs in a loop beside it. Tree $1, log dir $2.
T=$1; L=$2; mkdir -p $L
ulimit -c 0
export HIP_VISIBLE_DEVICES=1
cd $T
WARP_CACHE_PATH=$L/cache_suite $FLEET_VENV/bin/python -m warp.tests -s autodetect > $L/suite.log 2>&1 &
S=$!
i=0
while kill -0 $S 2>/dev/null; do
  i=$((i+1))
  WARP_CACHE_PATH=$L/cache_subset $FLEET_VENV/bin/python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d > $L/subset_$i.log 2>&1
  echo "iter $i $(date +%T) $(grep -E '^(OK|FAILED)' $L/subset_$i.log)"
done
echo "suite: $(grep -E '^(Ran|OK|FAILED)' $L/suite.log | tr '\n' ' ')"
```

## `flake_b.sh`

```
#!/bin/bash
# Flake reproduction: the test_async s2f_d2d / s2c_d2d subset (192 tests) run N times in tree $1, either with
# $3 background copy_flake load processes (4 streams each) on the same GPU or with none. Log dir $2.
T=$1; L=$2; NL=$3; N=$4; mkdir -p $L
ulimit -c 0
export HIP_VISIBLE_DEVICES=1
A=/var/lib/jenkins/fleet/jobs/mi355-20261119-154230-warp114-perf-porter-mi355-prompt/moat/agent_space
pids=()
for k in $(seq $NL); do $A/flake/copy_flake load 3600 4 > /dev/null 2>&1 & pids+=($!); done
f=0
cd $T
for i in $(seq $N); do
  WARP_CACHE_PATH=$L/cache $FLEET_VENV/bin/python -m warp.tests -s autodetect -p test_async.py -k s2f_d2d -k s2c_d2d > $L/run_$i.log 2>&1
  if ! grep -qE "^OK" $L/run_$i.log; then f=$((f+1)); echo "run $i: $(grep -E '^(OK|FAILED)' $L/run_$i.log) $(grep -E '\.\.\. (FAIL|ERROR)' $L/run_$i.log | sed -E 's/ \(warp.*//' | tr '\n' ' ') $(grep -oE 'Mismatched elements: [0-9]+' $L/run_$i.log | tr '\n' ' ')"; fi
done
for p in "${pids[@]}"; do kill $p 2>/dev/null; done
echo "$f of $N runs failed with $NL load processes"
```

## `flake/copy_flake.hip`

```
// Mirrors Warp's test_async s2c_d2d copy with memory pools off:
//   src: 2-strided float array (hipMalloc), dst: contiguous float array (hipMalloc) zero-filled from pageable memory,
//   device sync, strided->contiguous copy kernel on a blocking stream, hipStreamSynchronize(stream),
//   hipMemcpyAsync device-to-host on the null stream into a fresh malloc'd buffer, then the buffer is checked
//   immediately (Warp relies on CUDA's rule that a D2H async copy to pageable memory returns once it completes).
// Usage: copy_flake check <iterations> [sync|pool]   ("sync" adds hipStreamSynchronize(0) before the check;
//        "pool" allocates the source arrays with hipMallocAsync on the stream, freed with hipFreeAsync)
//        copy_flake load <seconds> [streams]   (background load: kernels and copies on several streams of the same device)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(2); } } while (0)

__global__ void strided_copy(float* dst, const float* src, int n, int src_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[(size_t)i * src_stride];
}

__global__ void scatter2(float* s, const float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        s[2 * (size_t)i] = c[i];
}

__global__ void burn(float* x, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        for (int k = 0; k < iters; ++k)
            v = v * 0.999f + 0.5f;
        x[i] = v;
    }
}

int main(int argc, char** argv)
{
    const int n = 1000000;
    if (argc >= 3 && !strcmp(argv[1], "load")) {
        // load: <streams> streams, each with its own buffer, kernels and copies queued round robin
        int secs = atoi(argv[2]);
        int ns = argc > 3 ? atoi(argv[3]) : 1;
        size_t m = 16 << 20;
        float* h = (float*)malloc(m * 4); memset(h, 0, m * 4);
        hipStream_t s[16]; float* a[16]; float* b[16];
        for (int k = 0; k < ns; ++k) {
            CK(hipStreamCreate(&s[k])); CK(hipMalloc(&a[k], m * 4)); CK(hipMalloc(&b[k], m * 4));
        }
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(secs)) {
            for (int k = 0; k < ns; ++k) {
                burn<<<(m + 255) / 256, 256, 0, s[k]>>>(a[k], (int)m, 100);
                CK(hipMemcpyAsync(b[k], a[k], m * 4, hipMemcpyDeviceToDevice, s[k]));
                CK(hipMemcpyAsync(h, b[k], (m / 16) * 4, hipMemcpyDeviceToHost, s[k]));
            }
            for (int k = 0; k < ns; ++k)
                CK(hipStreamSynchronize(s[k]));
        }
        return 0;
    }
    if (argc < 3) { printf("usage: copy_flake check <iterations> [sync] | load <seconds>\n"); return 1; }
    int iters = atoi(argv[2]);
    bool sync = argc > 3 && !strcmp(argv[3], "sync");
    bool pool = argc > 3 && !strcmp(argv[3], "pool");  // source and its staging array from the stream-ordered pool
    hipStream_t stream; CK(hipStreamCreateWithPriority(&stream, hipStreamDefault, 0));
    int bad = 0;
    for (int it = 0; it < iters; ++it) {
        float* host_src = (float*)malloc(n * 4);
        for (int i = 0; i < n; ++i) host_src[i] = (float)(it + i);
        float* zeros = (float*)calloc(n, 4);
        float *a, *src, *dst;
        if (pool) {
            CK(hipMallocAsync((void**)&a, n * 4, stream));
            CK(hipMallocAsync((void**)&src, 2 * (size_t)n * 4, stream));
        } else {
            CK(hipMalloc(&a, n * 4));
            CK(hipMalloc(&src, 2 * (size_t)n * 4));
        }
        CK(hipMalloc(&dst, n * 4));
        CK(hipMemcpyAsync(a, host_src, n * 4, hipMemcpyHostToDevice, stream));
        CK(hipMemsetAsync(src, 0, 2 * (size_t)n * 4, stream));
        // as_strided_array: copy the contiguous source into every second element
        scatter2<<<(n + 255) / 256, 256, 0, stream>>>(src, a, n);
        CK(hipMemcpyAsync(dst, zeros, n * 4, hipMemcpyHostToDevice, stream));
        CK(hipDeviceSynchronize());
        if (pool) CK(hipFreeAsync(a, stream)); else CK(hipFree(a));
        strided_copy<<<(n + 255) / 256, 256, 0, stream>>>(dst, src, n, 2);
        CK(hipStreamSynchronize(stream));
        float* out = (float*)malloc(n * 4);  // fresh pages: zero until written
        CK(hipMemcpyAsync(out, dst, n * 4, hipMemcpyDeviceToHost, 0));
        if (sync)
            CK(hipStreamSynchronize(0));
        int wrong = 0, first = -1, last = -1;
        for (int i = 0; i < n; ++i)
            if (out[i] != host_src[i]) { if (first < 0) first = i; last = i; ++wrong; }
        if (wrong) {
            ++bad;
            int mod8[8] = {0};
            for (int i = 0; i < n; ++i)
                if (out[i] != host_src[i]) mod8[(i / 256) % 8]++;
            printf("iter %d: %d wrong, first %d last %d, out[first]=%g, wrong per (block %% 8): %d %d %d %d %d %d %d %d\n", it, wrong,
                   first, last, out[first], mod8[0], mod8[1], mod8[2], mod8[3], mod8[4], mod8[5], mod8[6], mod8[7]);
        }
        free(out); free(host_src); free(zeros);
        if (pool) CK(hipFreeAsync(src, stream)); else CK(hipFree(src));
        CK(hipFree(dst));
    }
    printf("%d of %d iterations wrong%s\n", bad, iters, sync ? " (with null-stream sync)" : pool ? " (pooled source)" : "");
    return bad ? 1 : 0;
}
```

## `bench/banner.py`

```
# Print the wp.init() banner and the runtime version attributes.
import warp as wp
wp.config.log_level = wp.LOG_INFO
wp.init()
r = wp._src.context.runtime
print("rocm_version", r.rocm_version, "hip_version", r.hip_version, "toolkit_version", r.toolkit_version, "driver_version", r.driver_version)
```

## `bench/s2c_loop.py`

```
# usage: s2c_loop.py <iterations> [pool_src 0|1]; the test_async s2c_d2d copy (pool off for the destination, default
# stream, no graph) in a loop. On a mismatch: the wrong index range, then a second read-back of the same array, a
# read-back after a device synchronize, and a device-side comparison (kernel) to locate where the stale data is.
import sys
import numpy as np
import warp as wp
wp.config.quiet = True
wp.init()
dev = wp.get_device("cuda:0")
N = 1000000
iters = int(sys.argv[1]); pool_src = len(sys.argv) > 2 and sys.argv[2] == "1"

@wp.kernel
def count_wrong(dst: wp.array(dtype=float), off: float, wrong: wp.array(dtype=int)):
    i = wp.tid()
    if dst[i] != off + float(i):
        wp.atomic_add(wrong, 0, 1)

bad = 0
for it in range(iters):
    off = float(it * 7 % 1000)
    src_data = np.arange(off, off + N, dtype=np.float32)
    with wp.ScopedMempool(dev, pool_src):
        a = wp.array(data=src_data, device=dev)
        strided = wp.zeros(shape=a.shape, strides=(2 * a.strides[0],), dtype=a.dtype, device=dev)
        wp.copy(strided, a)
        del a
    with wp.ScopedMempool(dev, False):
        dst = wp.array(data=np.zeros(N, dtype=np.float32), device=dev, copy=True)
        wp.synchronize()
        wp.copy(dst, strided)
        wp.synchronize_stream(dev.stream)
        x = dst.numpy()
        w = np.nonzero(x != src_data)[0]
        if len(w):
            bad += 1
            x2 = dst.numpy()
            w2 = np.nonzero(x2 != src_data)[0]
            wp.synchronize_device(dev)
            x3 = dst.numpy()
            w3 = np.nonzero(x3 != src_data)[0]
            wrong = wp.zeros(1, dtype=int, device=dev)
            wp.launch(count_wrong, dim=N, inputs=[dst, off, wrong])
            print(f"iter {it}: {len(w)} wrong [{w[0]}, {w[-1]}] blocks [{w[0] // 256}, {w[-1] // 256}] values {x[w[0]]} | "
                  f"second read {len(w2)} wrong | after device sync {len(w3)} wrong | on device {int(wrong.numpy()[0])} wrong", flush=True)
        del dst, strided
print(f"{bad} of {iters} iterations wrong (source pool {'on' if pool_src else 'off'})")
```

## Scratch diagnostic in agent_space/w43 (never committed): `git -C agent_space/w43 diff`

```diff
diff --git a/warp/tests/cuda/test_async.py b/warp/tests/cuda/test_async.py
index 196424ae..5d4fffe0 100644
--- a/warp/tests/cuda/test_async.py
+++ b/warp/tests/cuda/test_async.py
@@ -431,7 +431,19 @@ def copy_template(test, src_ctor, dst_ctor, src_device, dst_device, n, params: C
             if stream is not None:
                 wp.synchronize_stream(stream)
 
-            assert_np_equal(dst.numpy(), src.numpy())
+            x1 = dst.numpy()
+            ref = src.numpy()
+            w1 = np.nonzero(x1 != ref)[0]
+            if len(w1):
+                # SCRATCH DIAGNOSTIC (not for commit): where is the stale data?
+                x2 = dst.numpy()
+                wp.synchronize()
+                x3 = dst.numpy()
+                w2 = np.nonzero(x2 != ref)[0]
+                w3 = np.nonzero(x3 != ref)[0]
+                print(f"FLAKEDIAG {test.id()}: first read {len(w1)} wrong [{w1[0]}, {w1[-1]}] blocks [{w1[0] // 256}, {w1[-1] // 256}] "
+                      f"values {x1[w1[0]]}..{x1[w1[-1]]}; second read {len(w2)} wrong; after wp.synchronize {len(w3)} wrong", flush=True)
+            assert_np_equal(x1, ref)
 
             if params.with_grad:
                 assert_np_equal(dst.grad.numpy(), src.grad.numpy())
```
