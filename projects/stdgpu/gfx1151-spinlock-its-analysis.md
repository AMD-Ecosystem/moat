# gfx1151 concurrent-container spinlock: ITS reconvergence analysis (Class A)

Self-contained writeup of the verified gfx1151 stdgpu failure class A: a lane-contended
atomic spinlock that relies on independent-thread-scheduling (ITS) forward progress.
Reproduced standalone (raw HIP, no stdgpu) and confirmed in the generated ISA. The stdgpu
port already mitigates this in source via `wave_lock_serialize`; this document records the
evidence.

Host: XSJJDAILYL02, AMD Radeon 8060S (gfx1151, RDNA3.5, wave32), Windows 11, ROCm 7.x
(TheRock pip SDK). Control architecture for ISA comparison: gfx1100 (RDNA3, wave32).
Reproducer artifacts: `agent_space/gfx1151-repro/` (t3.cpp, harness.h, t3.dev.gfx1151.s,
t3.dev.gfx1100.s) -- gitignored scratch.

## Summary

stdgpu's concurrent containers use a per-slot spinlock (`mutex_array` over an atomic
bitset): `try_lock` = `atomicOr(block, bit)` (acquired iff the bit was previously clear),
`unlock` = `atomicAnd(block, ~bit)`. When multiple lanes of one wavefront contend for the
same lock, the natural retry loop only terminates if a lane that wins the lock can make
forward progress to its `unlock` while the losing lanes keep retrying -- i.e. it depends on
CUDA Volta+ independent thread scheduling. On the AMD SIMT execution model the loop's
critical section and unlock are placed in a post-loop block that the compiler reaches only
after every lane has left the loop, and a lane leaves only by acquiring; the holder thus
cannot reach its unlock until the still-spinning losers acquire, which they cannot while it
holds the lock. The portable fix is to serialize lane attempts so only one lane is in the
loop at a time (ballot/`__ffsll` election).

## Reproducer (raw HIP, no stdgpu)

`<<<1,32>>>` -- 32 lanes of one wavefront contend for a single lock. Progress is observed
via host-pinned coherent memory (`hipHostMallocCoherent`) polled live by a host watchdog,
so a non-terminating kernel still yields a per-lane progress fingerprint (no device printf).

```cpp
// try_lock = atomicOr(block, bit); acquired iff (prev & bit)==0.  unlock = atomicAnd(block, ~bit)
__global__ void k_contend(volatile int* prog, unsigned int* block, int* counter) {
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    prog[g] = 1; __threadfence_system();              // phase 1: pre-acquire
    bool done = false;
    while (!done) {
        unsigned int prev = atomicOr(block, 1u);
        if ((prev & 1u) == 0u) {                      // got the lock
            prog[g] = 2; __threadfence_system();      // phase 2: acquired
            atomicAdd(counter, 1);
            prog[g] = 3; __threadfence_system();      // phase 3: critical section
            atomicAnd(block, ~1u);                    // unlock
            prog[g] = 4; __threadfence_system();      // phase 4: released
            done = true;
        }
    }
}
```

`wave_lock_serialize` (the stdgpu mitigation) elects one active lane at a time:

```cpp
template <typename F> __device__ void wave_lock_serialize(F&& body) {
    int lane = (int)__lane_id();
    unsigned long long active = __ballot(1);
    while (active) {
        int elected = __ffsll((long long)active) - 1;
        if (lane == elected) body();
        active &= ~(1ull << elected);
    }
}
```

## Observed behavior (gfx1151, 15s watchdog, deterministic)

| Variant | Shape | Result |
| --- | --- | --- |
| i (`k_uncontended`) | 1 lane, uncontended | COMPLETED (phase 4) |
| ii (`k_contend`) | 32 lanes contend, no serialization | DOES NOT TERMINATE; all 32 lanes pinned at phase 1 (3/3 runs) |
| iii (`k_contend_serial`) | 32 lanes wrapped in `wave_lock_serialize` | COMPLETED (phase 4) |

Single-lane and serialized variants terminate; the unserialized 32-lane contention does
not. gfx90a and gfx1100 complete all stdgpu concurrent-container tests of this family.

## ISA (gfx1151 vs gfx1100): the spinlock loop region is identical

`clang++ -x hip --cuda-device-only -O2 -S` for both targets. The `k_contend` loop region
(`.LBB1_*`) is instruction-for-instruction identical between gfx1151 and gfx1100
(differences elsewhere in the kernel are register allocation and dual-issue scheduling only):

```
.LBB1_1:
        s_or_b32 exec_lo, exec_lo, s3          ; restore EXEC after the elected-lane atomic
        ... compute "did this lane acquire?" predicate into vcc_lo ...
        s_or_b32 s2, vcc_lo, s2                ; accumulate the "done" (acquired) mask
        s_and_not1_b32 exec_lo, exec_lo, s2    ; a lane LEAVES the loop only once it has acquired
        s_cbranch_execz .LBB1_4                ; exit the loop only when EXEC == 0 (all acquired)
.LBB1_2:
        v_mbcnt_lo_u32_b32 v4, exec_lo, 0      ; elect lowest active lane
        v_cmp_eq_u32_e32 vcc_lo, 0, v4
        s_and_saveexec_b32 s3, vcc_lo          ; narrow EXEC to the elected lane
        s_cbranch_execz .LBB1_1
; %bb.3:
        global_atomic_or_b32 v4, v3, v2, s[6:7] glc   ; try_lock under narrowed EXEC
        s_branch .LBB1_1
.LBB1_4:
        ... phase-2/3/4: critical section + atomicAnd unlock ...   ; reached only after EXEC==0
```

The critical section and `unlock` (`.LBB1_4`) are reached only after `s_cbranch_execz`
sees `exec_lo == 0`, i.e. after every lane has cleared its EXEC bit via
`s_and_not1_b32 exec_lo, exec_lo, s2`. A lane clears its bit only by acquiring. The first
lane to acquire holds the lock and is removed from EXEC, but its `unlock` lives in
`.LBB1_4`, which does not run until the remaining lanes also acquire -- which they cannot
while the holder keeps the bit set. The loop therefore does not terminate without per-thread
forward progress (ITS).

## Mechanism

- A lane-contended `atomicOr`/`atomicAnd` spinlock where the unlock is in code reached only
  after all lanes leave the retry loop.
- Termination requires that a lane which has acquired can advance to its unlock while peers
  continue retrying -- independent thread scheduling semantics.
- Because the device ISA of the loop is identical on the two architectures examined, the
  observed difference is not in the emitted instruction stream.

## Resolution (already in the stdgpu port)

Serialize the lock attempt so only one lane executes the retry loop at a time
(`wave_lock_serialize`, ballot/`__ffsll` election). Variant iii confirms this terminates;
the shipping stdgpu HIP backend wraps the contended container operations this way, and the
real `insert_single` kernel's disassembly shows the same election + save-atomic-restore
structure around its bitset `try_lock`.

## Scope

This (Class A) accounts for the contended concurrent-container tests
(`insert_parallel_while_one_free`, the `emplace_*` variants, deque same-end push+pop). The
single-thread `insert_while_full` family (Class B) is a separate, unrelated phenomenon
localized to `valid()`'s `values_reachable` check; see notes.md (its mechanism is not
confidently isolated and it has no trustworthy standalone reproducer on this host). The
stdgpu port is correct and validated on gfx90a and gfx1100; windows-gfx1151 is blocked.

## Public gist

An observational, neutral-tone version of this Class A analysis (CUDA->HIP portability note;
reproducer + ISA + portable fix; no defect framing) is published at:
https://gist.github.com/jeffdaily/f2763b46b4a1c45234901ef9b459f3d4
