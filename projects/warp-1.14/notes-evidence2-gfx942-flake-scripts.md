# warp-1.14 evidence session 2 (2026-11-20, linux-gfx942): copy flake narrowing scripts

Under the job worktree's `agent_space/e2/`; `w` is the fork clone at `7059a340` (gfx942 build, with the first session's uncommitted test diagnostic, which `loop.py` does not use). HIP programs built with `$(python -m rocm_sdk path --root)/bin/hipcc --offload-arch=gfx942 -O2` from the job venv's ROCm 10.2 SDK. `copy_flake2.hip` itself is inline in `notes.md`.

## `loop.py`

```
# loop.py <iterations> [options]: the test_async s2c_d2d copy in a loop (session 21's s2c_loop.py, extended).
# options: pool_src  pool_dst  sync_before_copy  kernel_zero (dst zero-filled by wp.zeros instead of H2D copy)
#          keep (no free/realloc of dst and strided between iterations)  rtc (copy with a hipRTC Warp kernel instead of wp.copy)
# On a mismatch: wrong count per (block % 8) of the host read, re-read, read after device sync, then device-side checks:
# a kernel reading each element from the block with the same index (shift 0) and from a block shifted by 1..7.
import sys, time
import numpy as np
import warp as wp
wp.config.log_level = wp.LOG_WARNING
wp.init()
dev = wp.get_device("cuda:0")
N = 1000000
iters = int(sys.argv[1]); opt = set(sys.argv[2:])

@wp.kernel
def count_wrong(dst: wp.array(dtype=float), off: float, shift: int, n: int, wrong: wp.array(dtype=int)):
    # thread i (block i // 256) reads element (i + shift * 256) % n
    i = wp.tid()
    j = (i + shift * 256) % n
    if dst[j] != off + float(j):
        wp.atomic_add(wrong, (j // 256) % 8, 1)

@wp.kernel
def gather(out: wp.array(dtype=float), dst: wp.array(dtype=float), shift: int, n: int):
    # thread i (block i // 256, so the XCD of block i) reads element (i + shift * 256) % n into out at that element
    i = wp.tid()
    j = (i + shift * 256) % n
    out[j] = dst[j]

@wp.kernel
def rtc_copy(dst: wp.array(dtype=float), src: wp.array(dtype=float)):
    i = wp.tid()
    dst[i] = src[i]

def classes(x, ref):
    w = np.nonzero(x != ref)[0]
    return len(w), np.bincount((w // 256) % 8, minlength=8).tolist()

bad = 0
dst = strided = None
t0 = time.time()
for it in range(iters):
    off = float(it * 7 % 1000)
    src_data = np.arange(off, off + N, dtype=np.float32)
    with wp.ScopedMempool(dev, "pool_src" in opt):
        a = wp.array(data=src_data, device=dev)
        if strided is None or "keep" not in opt:
            strided = wp.zeros(shape=a.shape, strides=(2 * a.strides[0],), dtype=a.dtype, device=dev)
        wp.copy(strided, a)
        del a
    with wp.ScopedMempool(dev, "pool_dst" in opt):
        if "kernel_zero" in opt:
            if dst is None or "keep" not in opt:
                dst = wp.zeros(N, dtype=float, device=dev)
            else:
                dst.zero_()
        else:
            if dst is None or "keep" not in opt:
                dst = wp.array(data=np.zeros(N, dtype=np.float32), device=dev, copy=True)
            else:
                dst.assign(np.zeros(N, dtype=np.float32))
        wp.synchronize()
        if "sync_before_copy" in opt:
            wp.synchronize_device(dev)
        if "rtc" in opt:
            wp.launch(rtc_copy, dim=N, inputs=[dst, strided])
        else:
            wp.copy(dst, strided)
        wp.synchronize_stream(dev.stream)
        x = dst.numpy()
        if not np.array_equal(x, src_data):
            bad += 1
            n1, c1 = classes(x, src_data)
            n2, c2 = classes(dst.numpy(), src_data)
            wp.synchronize_device(dev)
            n3, c3 = classes(dst.numpy(), src_data)
            # what the XCDs see, by value: for shift s the reader of class c is the XCD that runs class (c - s) mod 8
            prev = float((it - 1) * 7 % 1000)
            ref = np.arange(N, dtype=np.float32)
            for s in (0, 1, 0, 4):
                out = wp.full(N, -1.0, dtype=float, device=dev)
                wp.launch(gather, dim=N, inputs=[out, dst, s, N])
                y = out.numpy()
                desc = []
                for c in range(8):
                    m = ((np.arange(N) // 256) % 8) == c
                    yc = y[m]; rc = ref[m]
                    desc.append(f"c{c}:ok{int((yc == rc + off).sum())}/z{int((yc == 0).sum())}/prev{int((yc == rc + prev).sum())}/other{int(m.sum() - (yc == rc + off).sum() - (yc == 0).sum() - (yc == rc + prev).sum())}")
                print(f"    values seen with shift {s}: " + " ".join(desc), flush=True)
            dv = []
            for s in range(8):
                wrong = wp.zeros(8, dtype=int, device=dev)
                wp.launch(count_wrong, dim=N, inputs=[dst, off, s, N, wrong])
                dv.append(wrong.numpy().tolist())
            time.sleep(1.0)
            n4, c4 = classes(dst.numpy(), src_data)
            print(f"iter {it} t={time.time()-t0:.1f}s ptr=0x{dst.ptr:x} strided=0x{strided.ptr:x}: read1 {n1} {c1} | read2 {n2} {c2} | after sync {n3} {c3} | after 1 s {n4} {c4}", flush=True)
            for s in range(8):
                print(f"    device check, read shift {s} blocks: wrong per class {dv[s]}", flush=True)
    if "keep" not in opt:
        dst = strided = None
print(f"{bad} of {iters} iterations wrong ({' '.join(sorted(opt)) or 'baseline'}) in {time.time()-t0:.0f} s")
```

## `ablate.sh`

```
#!/bin/bash
# ablate.sh <iterations> <label> [loop.py options] : loop.py under 8 copy_flake load processes on the pin
ulimit -c 0; D=$(dirname $(readlink -f $0)); export PYTHONPATH=$D/../w WARP_CACHE_PATH=$D/cache
N=$1; L=$2; shift 2
$D/load.sh start 8 copy
timeout 1800 $FLEET_VENV/bin/python $D/loop.py $N "$@" > $D/logs/abl_$L.log 2>&1
echo "rc=$? $(tail -1 $D/logs/abl_$L.log)"
$D/load.sh stop
```

## `load.sh`

```
#!/bin/bash
# load.sh start <n> <kind> [gpu] | stop : background load processes, PIDs kept in e2/load.pids (never pattern-killed)
D=$(dirname $(readlink -f $0)); P=$D/load.pids
case $1 in
start) G=${4:-$HIP_VISIBLE_DEVICES}; : > $P
  for k in $(seq $2); do
    case $3 in
      copy) HIP_VISIBLE_DEVICES=$G setsid $D/../flake/copy_flake load 7200 4 > /dev/null 2>&1 & ;;
      copy_only) HIP_VISIBLE_DEVICES=$G setsid $D/loadkind copy 7200 > /dev/null 2>&1 & ;;
      *) HIP_VISIBLE_DEVICES=$G setsid $D/loadkind $3 7200 > /dev/null 2>&1 & ;;
    esac
    echo $! >> $P
  done; sleep 2 ;;
stop) for p in $(cat $P 2>/dev/null); do kill $p 2>/dev/null; done; sleep 1; for p in $(cat $P 2>/dev/null); do kill -9 $p 2>/dev/null; done; : > $P ;;
esac
```

## `loadkind.hip`

```
// loadkind <kind> <seconds>: background load of one kind on the current device.
//   alloc  : hipMalloc / hipMemset / hipFree of 16-64 MiB buffers in a loop (no kernels of its own)
//   kernel : a compute kernel over a resident 64 MiB buffer on 4 streams, no allocation, no copies
//   copy   : D2D and D2H hipMemcpyAsync on 4 streams between resident buffers, no kernels, no allocation
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s -> %s\n", #x, hipGetErrorString(e)); exit(2); } } while (0)
__global__ void burn(float* x, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; for (int k = 0; k < iters; ++k) v = v * 0.999f + 0.5f; x[i] = v; }
}
int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: loadkind alloc|kernel|copy <seconds>\n"); return 1; }
    const char* kind = argv[1]; int secs = atoi(argv[2]);
    auto t0 = std::chrono::steady_clock::now();
    auto running = [&] { return std::chrono::steady_clock::now() - t0 < std::chrono::seconds(secs); };
    size_t m = 16 << 20;
    hipStream_t s[4]; for (int k = 0; k < 4; ++k) CK(hipStreamCreate(&s[k]));
    if (!strcmp(kind, "alloc")) {
        unsigned r = 1;
        while (running()) {
            r = r * 1103515245u + 12345u; size_t bytes = (size_t)(16 + (r >> 16) % 49) << 20;
            void* p; CK(hipMalloc(&p, bytes)); CK(hipMemset(p, 0, bytes)); CK(hipFree(p));
        }
    } else if (!strcmp(kind, "kernel")) {
        float* a[4]; for (int k = 0; k < 4; ++k) { CK(hipMalloc(&a[k], m * 4)); CK(hipMemset(a[k], 0, m * 4)); }
        while (running()) {
            for (int k = 0; k < 4; ++k) burn<<<(m + 255) / 256, 256, 0, s[k]>>>(a[k], (int)m, 100);
            for (int k = 0; k < 4; ++k) CK(hipStreamSynchronize(s[k]));
        }
    } else if (!strcmp(kind, "copy")) {
        float *a[4], *b[4]; float* h = (float*)malloc(m * 4);
        for (int k = 0; k < 4; ++k) { CK(hipMalloc(&a[k], m * 4)); CK(hipMalloc(&b[k], m * 4)); CK(hipMemset(a[k], 0, m * 4)); }
        while (running()) {
            for (int k = 0; k < 4; ++k) {
                CK(hipMemcpyAsync(b[k], a[k], m * 4, hipMemcpyDeviceToDevice, s[k]));
                CK(hipMemcpyAsync(h, b[k], (m / 16) * 4, hipMemcpyDeviceToHost, s[k]));
            }
            for (int k = 0; k < 4; ++k) CK(hipStreamSynchronize(s[k]));
        }
    } else { printf("unknown kind\n"); return 1; }
    return 0;
}
```

## `copy_flake2e.hip`

```
// copy_flake2: Warp's per-iteration HIP call sequence for test_async's s2c_d2d copy (from an AMD_LOG_LEVEL=3 trace of
// warp 1.14 at 7059a340), with Warp's array_copy_1d_kernel body, 3907 x 256 launches and its stream setup:
//   A = hipMalloc(n*4); hipMemcpyAsync(A <- pageable src, H2D, S)
//   Y = hipMalloc(2n*4); fill kernel (zero every 2nd float of Y) on S; copy kernel A -> Y (stride 8 B) on S
//   hipFree(A)                                   <- no explicit sync: relies on hipFree's implicit synchronization
//   D = hipMalloc(n*4)  (normally returns A's address); hipMemcpyAsync(D <- pageable zeros, H2D, S)
//   hipDeviceSynchronize(); copy kernel Y -> D on S; hipStreamSynchronize(S)
//   hipEventRecord(e, S); hipStreamWaitEvent(null, e); hipMemcpyAsync(pageable out <- D, D2H, null); check out
//   hipFree(D); hipFree(Y)
// S = hipStreamCreateWithPriority(0, 0), e = hipEventCreateWithFlags(hipEventDisableTiming).
// Usage: copy_flake2 <iterations> [noreuse] [syncfree] [memsetfill]
//   noreuse: allocate D before freeing A, so D never gets A's address; syncfree: hipDeviceSynchronize() before hipFree(A);
//   memsetfill: zero D with hipMemsetAsync on S instead of the pageable H2D copy
// On a mismatch: wrong count per (block % 8) for the first read, a second read, a read after hipDeviceSynchronize,
// and a device-side count by a kernel whose block b reads block b + s (s = 0..7).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <dirent.h>
#include <set>
#include <string>
// eviction sampling: this process's KFD directory is the one that appears across the first HIP call
static std::set<std::string> kfd_procs() {
    std::set<std::string> r; DIR* d = opendir("/sys/class/kfd/kfd/proc");
    if (d) { while (dirent* e = readdir(d)) if (e->d_name[0] != '.') r.insert(e->d_name); closedir(d); }
    return r;
}
static std::string g_kfd_dir;
static long evicted_ms() {  // sum over this process's per-GPU stats_*/evicted_ms
    long sum = 0; DIR* d = opendir(g_kfd_dir.c_str()); if (!d) return -1;
    while (dirent* e = readdir(d)) if (!strncmp(e->d_name, "stats_", 6)) {
        FILE* f = fopen((g_kfd_dir + "/" + e->d_name + "/evicted_ms").c_str(), "r");
        if (f) { long v = 0; if (fscanf(f, "%ld", &v) == 1) sum += v; fclose(f); }
    }
    closedir(d); return sum;
}
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); exit(2); } } while (0)

static __global__ void array_copy_1d_kernel(void* dst, const void* src, size_t dst_stride, size_t src_stride,
                                            const int* dst_indices, const int* src_indices, size_t n, size_t elem_size)
{
    size_t i = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (i < n) {
        size_t src_idx = src_indices ? src_indices[i] : i;
        size_t dst_idx = dst_indices ? dst_indices[i] : i;
        const char* p = (const char*)src + src_idx * src_stride;
        char* q = (char*)dst + dst_idx * dst_stride;
        memcpy(q, p, elem_size);
    }
}

static __global__ void fill_strided(float* dst, size_t stride_elems, size_t n)
{
    size_t i = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
    if (i < n)
        dst[i * stride_elems] = 0.0f;
}

static __global__ void count_wrong(const float* d, float off, int shift, int n, int* wrong)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (i + shift * 256) % n;
        if (d[j] != off + float(j))
            atomicAdd(&wrong[(j / 256) % 8], 1);
    }
}

static void classes(const float* x, const float* ref, int n, int* cnt, int* tot)
{
    memset(cnt, 0, 8 * sizeof(int)); *tot = 0;
    for (int i = 0; i < n; ++i) if (x[i] != ref[i]) { cnt[(i / 256) % 8]++; (*tot)++; }
}

int main(int argc, char** argv)
{
    const int n = 1000000; const int blocks = (n + 255) / 256;
    int iters = argc > 1 ? atoi(argv[1]) : 1000;
    bool noreuse = false, syncfree = false, memsetfill = false;
    for (int a = 2; a < argc; ++a) {
        if (!strcmp(argv[a], "noreuse")) noreuse = true;
        else if (!strcmp(argv[a], "syncfree")) syncfree = true;      // hipDeviceSynchronize() before hipFree(A)
        else if (!strcmp(argv[a], "memsetfill")) memsetfill = true;  // zero D with hipMemsetAsync instead of a pageable H2D copy
    }
    std::set<std::string> before = kfd_procs();
    CK(hipFree(nullptr));
    for (auto& p : kfd_procs()) if (!before.count(p)) g_kfd_dir = "/sys/class/kfd/kfd/proc/" + p;
    printf("kfd dir %s, evicted_ms %ld\n", g_kfd_dir.c_str(), evicted_ms());
    long ev_pass_iters = 0, ev_fail_iters = 0, ev_total = 0;
    hipStream_t S; CK(hipStreamCreateWithPriority(&S, 0, 0));
    hipEvent_t ev; CK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    int* wrong; CK(hipMalloc(&wrong, 8 * sizeof(int)));
    int bad = 0, reused = 0;
    for (int it = 0; it < iters; ++it) {
        float off = float(it * 7 % 1000);
        float* src = (float*)malloc(n * 4); for (int i = 0; i < n; ++i) src[i] = off + float(i);
        float* zeros = (float*)calloc(n, 4);
        float *A, *Y, *D;
        long ev0 = evicted_ms();
        CK(hipMalloc(&A, n * 4));
        CK(hipMemcpyAsync(A, src, n * 4, hipMemcpyHostToDevice, S));
        CK(hipMalloc(&Y, 2 * (size_t)n * 4));
        fill_strided<<<blocks, 256, 0, S>>>(Y, 2, n);
        array_copy_1d_kernel<<<blocks, 256, 0, S>>>(Y, A, 8, 4, nullptr, nullptr, n, 4);
        if (syncfree) CK(hipDeviceSynchronize());
        if (noreuse) { CK(hipMalloc(&D, n * 4)); CK(hipFree(A)); }
        else { CK(hipFree(A)); CK(hipMalloc(&D, n * 4)); }
        if (D == A) ++reused;
        if (memsetfill) CK(hipMemsetAsync(D, 0, n * 4, S));
        else CK(hipMemcpyAsync(D, zeros, n * 4, hipMemcpyHostToDevice, S));
        CK(hipDeviceSynchronize());
        array_copy_1d_kernel<<<blocks, 256, 0, S>>>(D, Y, 4, 8, nullptr, nullptr, n, 4);
        CK(hipStreamSynchronize(S));
        float* out = (float*)malloc(n * 4);
        CK(hipEventRecord(ev, S)); CK(hipStreamWaitEvent(nullptr, ev, 0));
        CK(hipMemcpyAsync(out, D, n * 4, hipMemcpyDeviceToHost, nullptr));
        long ev1 = evicted_ms(); long dev = ev1 - ev0; ev_total += dev;
        int c1[8], t1; classes(out, src, n, c1, &t1);
        if (dev > 0) { if (t1) ++ev_fail_iters; else ++ev_pass_iters; }
        if (t1) {
            ++bad;
            int c2[8], t2, c3[8], t3, c4[8], t4;
            CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost)); classes(out, src, n, c2, &t2);
            CK(hipDeviceSynchronize());
            CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost)); classes(out, src, n, c3, &t3);
            int dv[8][8];
            for (int s = 0; s < 8; ++s) {
                CK(hipMemset(wrong, 0, 8 * sizeof(int)));
                count_wrong<<<blocks, 256>>>(D, off, s, n, wrong);
                CK(hipMemcpy(dv[s], wrong, 8 * sizeof(int), hipMemcpyDeviceToHost));
            }
            usleep(1000000);
            CK(hipMemcpy(out, D, n * 4, hipMemcpyDeviceToHost)); classes(out, src, n, c4, &t4);
            printf("    eviction during this iteration: %ld ms (evicted_ms %ld -> %ld)\n", dev, ev0, ev1);
            printf("iter %d D=%p A-reuse=%d: read1 %d [%d %d %d %d %d %d %d %d] | read2 %d [%d %d %d %d %d %d %d %d] | after sync %d [%d %d %d %d %d %d %d %d] | after 1 s %d [%d %d %d %d %d %d %d %d]\n",
                   it, (void*)D, D == A, t1, c1[0], c1[1], c1[2], c1[3], c1[4], c1[5], c1[6], c1[7], t2, c2[0], c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7],
                   t3, c3[0], c3[1], c3[2], c3[3], c3[4], c3[5], c3[6], c3[7], t4, c4[0], c4[1], c4[2], c4[3], c4[4], c4[5], c4[6], c4[7]);
            for (int s = 0; s < 8; ++s)
                printf("    device check, block b reads block b+%d: wrong per class %d %d %d %d %d %d %d %d\n", s, dv[s][0], dv[s][1], dv[s][2], dv[s][3], dv[s][4], dv[s][5], dv[s][6], dv[s][7]);
            fflush(stdout);
        }
        free(out); free(src); free(zeros);
        CK(hipFree(D)); CK(hipFree(Y));
        if ((it + 1) % 5000 == 0) { printf("progress %d iterations, %d wrong\n", it + 1, bad); fflush(stdout); }
    }
    printf("evictions: %ld ms in total; iterations with an eviction: %ld passing, %ld failing\n", ev_total, ev_pass_iters, ev_fail_iters);
    printf("%d of %d iterations wrong%s%s%s (D got A's address in %d)\n", bad, iters, noreuse ? " (noreuse)" : "", syncfree ? " (syncfree)" : "", memsetfill ? " (memsetfill)" : "", reused);
    return bad ? 1 : 0;
}
```

## `cf2_batch.sh`

```
#!/bin/bash
# cf2_batch.sh: copy_flake2 under different loads on the pin; 40000 iterations each
ulimit -c 0; D=$(dirname $(readlink -f $0))
run() { L=$1; NL=$2; K=$3; shift 3
  [ $NL -gt 0 ] && $D/load.sh start $NL $K
  timeout 1500 $D/copy_flake2 40000 "$@" > $D/logs/cf2_$L.log 2>&1
  echo "$L ($NL x $K): $(tail -1 $D/logs/cf2_$L.log)"
  [ $NL -gt 0 ] && $D/load.sh stop
}
run noreuse_load8 8 copy noreuse
run alone 0 none
run load1 1 copy
run alloc8 8 alloc
run kernel8 8 kernel
run copyonly8 8 copy_only
```

## `cf2_batch3.sh`

```
#!/bin/bash
# cf2_batch3.sh: eviction sampling and the syncfree / memsetfill variants under 8 copy_flake load processes on the pin
ulimit -c 0; D=$(dirname $(readlink -f $0))
run() { L=$1; B=$2; N=$3; shift 3
  $D/load.sh start 8 copy
  timeout 1500 stdbuf -oL $D/$B $N "$@" > $D/logs/cf2c_$L.log 2>&1
  echo "$L: $(grep -E '^(evictions|[0-9]+ of)' $D/logs/cf2c_$L.log | tr '\n' ' ') failures logged: $(grep -c '^iter' $D/logs/cf2c_$L.log)"
  $D/load.sh stop
}
run evict_load8 copy_flake2e 40000
run syncfree_load8 copy_flake2v 40000 syncfree
run memsetfill_load8 copy_flake2v 40000 memsetfill
```

