# Quest notes

planner: perf-critical kernels -- assess a mechanical HIP port vs an AMD-native (rocWMMA/CK/MFMA) rewrite of the hot kernels; a correctness-first port is a valid first step.

## Porting attempt 2026-06-04

### Scope
Stage 1 only: `rms_norm.cu` and `topk.cu`. The flashinfer-dependent attention kernels (Stage 2) use NVIDIA-specific PTX intrinsics (mma.sync, ldmatrix, cp.async) that require an AMD-native rewrite.

### Build environment
- ROCm 7.2.1, PyTorch 2.13.0a0+gitb5e90ff (development build)
- a ported raft (no longer tracked here) built and installed at `_deps/raft/install`)

### Issues encountered

1. **PyTorch TensorImpl.h C++20 concepts incompatibility**: The PyTorch headers use `requires` clauses that fail to parse with the HIP compiler unless `-std=gnu++20` is used. Worked around by setting `CMAKE_HIP_STANDARD 20`.

2. **Multi-arch flags from Torch CMake**: `find_package(Torch)` adds `--offload-arch=gfx90a;gfx942;gfx950;gfx1100` from the PyTorch build. Worked around by setting `PYTORCH_ROCM_ARCH=gfx90a` environment variable.

3. **__HIP_NO_HALF_CONVERSIONS__ breaks static_cast<float>(__half)**: PyTorch ROCm sets this flag to disable implicit half conversions. The `rms_norm.cu` kernel uses `static_cast<float>(h->x)` which fails. Required explicit `__half2float()` calls.

4. **Name collision with PyTorch symbols**: A `half_to_float` macro clashed with `at::attr::half_to_float` in PyTorch's interned_strings.h. Renamed to `quest_h2f`.

5. **raft radix_topk_one_block_kernel overload resolution**: The raft kernel function pointer capture fails with "incompatible initializer of type '<overloaded function type>'". This is a blocking issue in how Quest's `decode_select_k.cuh` captures the raft kernel.

### Blocking issue
The raft select_k integration needs investigation. The Quest code captures a function pointer to raft's templated kernel but the HIP compiler cannot resolve the overload. This may require changes to how Quest calls raft's select_k API.

### Files modified (uncommitted, on moat-port branch)
- `quest/ops/CMakeLists.txt`: USE_HIP option, HIP language enable, raft via find_package
- `quest/ops/cmake/hip_build.cmake`: HIP-specific build config (unused, early return approach)
- `quest/ops/csrc/rms_norm.cu`: wave64 warp-size abstraction, half<->float conversion traits
- `quest/ops/csrc/bsk_ops.h`: USE_HIP guards for flashinfer-dependent declarations
- `quest/ops/csrc/bsk_ops.cu`: USE_HIP guards for pybind module
- `quest/ops/csrc/pytorch_extension_utils.h`: USE_HIP guard for DISPATCH macro
- `kernels/include/topk/decode_select_k.cuh`: HIP runtime aliases

## 2026-08-06 dependency

`depends_on: [raft]` cleared. raft is no longer a MOAT project (the ROCm-DS team owns
the RAPIDS domain), so it is an ordinary external build dependency here rather than
something this pipeline builds first. If the build needs it, install it from the
environment like any other third-party library.

## Resuming (2026-08-07)

The port continues: this was judged a port worth finishing rather than an unportable
codebase, so linux-gfx90a is no longer marked blocked. The blocker as last recorded, which
is where to pick it up:

raft select_k API incompatibility: radix_topk_one_block_kernel overload resolution fails in Quest's decode_select_k wrapper. PyTorch ROCm 2.13.0a0 development build has __HIP_NO_HALF_CONVERSIONS__ requiring explicit conversion calls. Need to investigate raft select_k integration pattern.

## Port on linux-gfx1100 (2026-08-08)

Fork sha `ff80217` on `moat-port`, replacing the earlier WIP commit (nothing had validated
it, so it was collapsed rather than built on).

Environment: Radeon Pro W7800 (gfx1100, wave32), ROCm 7.2.3, PyTorch 2.14.0a0 ROCm build,
python 3.12. The test suite needs `transformers==4.37.2` with `tokenizers==0.15.2`.

Build (from `quest/ops`, and note that a stale build directory must be wiped after editing
anything under `kernels/include/hip_compat`, since two of those headers are force-included
and carry no dependency edge):

```
mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja
ln -sf $PWD/_kernels*.so ../../     # setup.sh puts it in quest/, NOT quest/ops/
```

Do NOT pass `-DCMAKE_HIP_COMPILER=...`: cmake then reports "variables have changed, cache
must be deleted", re-runs configure without the command-line `-DUSE_HIP=ON`, and falls into
the CUDA branch looking for nvcc. Leave the compiler to `enable_language(HIP)`.

GPU results (`HIP_VISIBLE_DEVICES=1`, from the repo root with it on PYTHONPATH):

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| test_topk.py | 49 failed (topk_filtering not built) |
| test_prefill_attention.py | 29 failed (prefill_with_paged_kv_cache not built) |

What is out of scope and why, so the next session does not re-derive it:

- `topk.cu` -> RAFT `radix_topk_one_block_kernel`. The previous session's blocker (an
  overload-resolution failure capturing the kernel function pointer) was never resolved and
  is now moot in a different way: no RAFT is installed on this host and RAFT is no longer a
  MOAT project. Replacing it with a self-contained per-block top-k over the page scores
  (batch 32, len 1024-8192, k 64-256) is the obvious next deliverable and needs no external
  library. `test_topk.py` is its gate.
- `prefill.cuh` -> `mma_sync_m16n16k16_*` and `ldmatrix_m8n8x4`. Genuine AMD-native rewrite
  (WMMA on gfx11, MFMA on CDNA). `test_prefill_attention.py` is its gate.

Things that turned out NOT to be problems, contrary to the plan:

- `cp_async` needed nothing. flashinfer's `cp_async.cuh` guards its PTX on
  `__CUDACC_VER_MAJOR__ >= 11` and already carries a plain vectorized-load fallback, which
  is what a HIP compile selects. That alone is what put decode attention in reach.
- The `rms_norm.cu` reduction is wave-portable as written, because the shuffles state an
  explicit width of 32. Only the mask literal had to change. The `>>5` / `&0x1f` / `[33]`
  shapes the plan flagged are correct on wave64 and were left alone.

## Review 2026-08-08

Reviewed fork sha `ff80217` (moat-port vs `01c1623`) on linux-gfx1100. Verdict:
changes-requested. Review PR (never merged, feedback lives as line comments):
https://github.com/AMD-Ecosystem/Quest/pull/1

The shim-header approach is sound and the analysis behind it checks out; the defects are
all in the build file and the README.

1. `quest/ops/CMakeLists.txt:72` -- `-DCMAKE_HIP_ARCHITECTURES` is INERT. `find_package(Torch)`
   pulls `Caffe2/public/LoadHIP.cmake:107`, `set(CMAKE_HIP_ARCHITECTURES ${PYTORCH_ROCM_ARCH})`,
   a normal variable that shadows the cache entry, so `set_target_properties(... HIP_ARCHITECTURES
   "${CMAKE_HIP_ARCHITECTURES}")` re-applies torch's list. Measured: configuring with
   `-DCMAKE_HIP_ARCHITECTURES=gfx1201` produces `--offload-arch=gfx90a,gfx942,gfx950,gfx1100`
   and no gfx1201; `PYTORCH_ROCM_ARCH=gfx908` with no `-D` produces gfx908 alone. So the env
   var is the only working knob, the reverse of what the README says, and any user on an arch
   outside torch's default list gets hipErrorNoBinaryForGpu. The comment "Override Torch's
   multi-arch flags" says the opposite of what the line does.
2. `CMakeLists.txt:17` -- the `PYTORCH_ROCM_ARCH` / FATAL_ERROR block is dead. `enable_language(HIP)`
   on line 14 already defines `CMAKE_HIP_ARCHITECTURES` (CMakeDetermineHIPCompiler.cmake:296-334,
   via rocm_agent_enumerator, else CMake's own fatal error). Cache after configure reads
   `gfx1100;gfx1100;gfx1100;gfx1100` even with PYTORCH_ROCM_ARCH=gfx908 set. Hoist above
   `enable_language(HIP)`.
3. `CMakeLists.txt:29` -- same ordering bug on the CUDA leg, and it changes the NVIDIA build:
   `enable_language(CUDA)` runs first and CMakeDetermineCUDACompiler.cmake:261-267 caches nvcc's
   default arch, so `set(CMAKE_CUDA_ARCHITECTURES native)` never applies. Upstream had it before
   `project()` where it did. Likely a hard failure, not a slow path: flashinfer emits cp.async /
   mma.sync PTX that ptxas rejects below sm_80. Read off the CMake module; no nvcc on this host.
4. `CMakeLists.txt:4` -- the upstream `gcc-11`/`g++-11` compiler pins were deleted rather than
   guarded, contradicting the commit message's "nothing the NVIDIA build compiles or includes
   changes". `option(USE_HIP)` precedes `project()`, so `if(NOT USE_HIP)` restores them.
5. `README.md:69` -- the PYTORCH_ROCM_ARCH sentence is backwards (follows from 1).
6. `README.md:60` -- does not say the end-to-end path is unavailable. QuestAttention.py:117,127,151
   reach `_kernels.prefill_with_paged_kv_cache` and `_kernels.topk_filtering`, so scripts/passkey.sh
   and everything under evaluation/ -- the section immediately below this text -- cannot run.
7. `quest/ops/csrc/bsk_ops.{cu,h}` -- gratuitous reordering (apply_rope_in_place moved in both
   files, the guarded m.defs relocated to the bottom) turns a four-line additive diff into a
   whole-file reshuffle. Guard in place.

Checked and correct, so nobody re-derives it:

- Force-include mechanism: both shadow headers reuse the original guard macros
  (`FLASHINFER_MATH_CUH_`, `VEC_DTYPES_CUH_`), so the sibling-relative include of the original
  collapses. vec_dtypes.cuh differs from the vendored file ONLY in the vec_cast dispatch.
- math.cuh replacement is symbol-complete and numerically appropriate: ex2.approx -> exp2f
  (`v_exp_f32`), rcp.approx -> `__frcp_rn` (correctly rounded); both at least as accurate as the
  approximate instructions, against a 5e-3 reference. shfl.sync.bfly with clamp operand 0x1f is
  a 32-lane segment, which `__shfl_xor(x, mask, 32)` reproduces on both wavefront widths.
- rms_norm.cu mask change is required: amd_warp_sync_functions.h:307 static_asserts
  `sizeof(MaskT) == 8`, and torch's Caffe2Targets.cmake puts `-DHIP_ENABLE_WARP_SYNC_BUILTINS`
  on every extension compile.
- **Item 3 adjudicated: the porter is right, the plan was wrong.** Every shuffle in
  `blockReduceSum` states width 32, so `lane`/`wid` keep their meaning on wave64 and
  `shared[NUM][33]` still bounds `blockDim.x/32 <= 32` entries. Rewriting to `warpSize` would
  have been a regression for no gain. Same for the in-scope decode path: decode_attn.cuh:102,158
  reduce over offsets < bdx with `static_assert(bdx <= 32)` at 824/923/1118.
- Scoped-out suites are honest: 49 + 29 failures re-run on gfx1100, all 78 `AttributeError:
  module 'quest._kernels' has no attribute ...`. Exported list is exactly the six ops the
  README names.
- Empty `cuda/pipeline` shim is correct: included by permuted_smem.cuh / decode.cuh /
  decode_attn.cuh, but no `cuda::pipeline` object is declared in the compiled sources.
- Commit hygiene clean: `[ROCm]` title 50 chars, Claude named, no noreply trailer, jargon clean,
  no non-ASCII and no em-dash in added lines, fork tree clean.
- Lesson promotion (6d42e1d) is correctly filed and item 3's outcome is reflected accurately:
  fault-classes.md states that an explicit width is already wave-agnostic and that rewriting it
  to `warpSize` is a regression risk, which is the correction the plan needed.

## Porter pass 2 (2026-08-08, linux-gfx1100) -- review defects 1-7

Fork sha `3465b34` on `moat-port`, amended over `ff80217` (nothing had validated it: both
`validated_sha` were null at the time). All seven review defects addressed; the porting
approach itself was not revisited.

1. **`-DCMAKE_HIP_ARCHITECTURES` was inert.** Root cause exactly as the reviewer diagnosed:
   `Caffe2/public/LoadHIP.cmake:107` `set(CMAKE_HIP_ARCHITECTURES ${PYTORCH_ROCM_ARCH})` is a
   NORMAL variable that shadows the cache entry for the rest of the directory scope, so
   `set_target_properties(... HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")` after
   `find_package(Torch)` re-applied torch's list. Fix: resolve the target before
   `enable_language(HIP)` (env `PYTORCH_ROCM_ARCH` consulted only when the cache is empty, so
   an explicit `-D` wins), snapshot it into `QUEST_HIP_ARCHITECTURES` before
   `find_package(Torch)`, and set the target property from the snapshot.
   `list(REMOVE_DUPLICATES)` because `rocm_agent_enumerator` repeats an arch per device.

   Measured off `compile_commands.json` (`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`), with the
   host env carrying `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;gfx1100` throughout:

   | case | before (`ff80217`) | after (`3465b34`) |
   |---|---|---|
   | `-DCMAKE_HIP_ARCHITECTURES=gfx1201` | `gfx90a gfx942 gfx950 gfx1100` | `gfx1201` |
   | no `-D`, `PYTORCH_ROCM_ARCH=gfx908` | `gfx908` | `gfx908` |
   | no `-D`, env var unset | (n/a, FATAL_ERROR path dead) | `gfx1100` (deduped from 4) |

   `llvm-objdump --offloading` on the built module shows `hipv4-amdgcn-amd-amdhsa--gfx1100`
   only. Note this bug HIDES on any host whose GPU is in the torch wheel's list, which is why
   reading the compile line is the gate rather than a successful local run.

2. **Dead `PYTORCH_ROCM_ARCH` / `FATAL_ERROR` block.** Made reachable rather than removed:
   the env-var branch now sits ABOVE `enable_language(HIP)`, which is the only place it can
   do anything (`CMakeDetermineHIPCompiler.cmake:296` is `elseif(NOT DEFINED
   CMAKE_HIP_ARCHITECTURES)`). The `FATAL_ERROR` is gone: with the ordering fixed, falling
   through to CMake's own `rocm_agent_enumerator` detection is the better default, and CMake
   raises its own fatal error when that finds nothing.

3. **CUDA `native` lost.** `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES) set(... native)` moved
   back above `project()`, where upstream had it, under `if(NOT USE_HIP)`.
   `CMakeDetermineCUDACompiler.cmake:261` is `if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")`,
   so a value set before the language is enabled survives and one set after is ignored.
   **UNVERIFIED and stated as such in the commit message: there is no nvcc on this host, so
   the CUDA configure cannot be run at all** (`/usr/bin/gcc-11` is also absent). What WAS
   verified: (a) the guard logic itself, with a standalone `project(probe LANGUAGES NONE)`
   probe -- `USE_HIP=OFF` yields the gcc-11 pins and `native`, an explicit
   `-DCMAKE_CUDA_ARCHITECTURES=80` still wins, `USE_HIP=ON` leaves all three untouched; and
   (b) the same normal-variable-before-`enable_language` mechanism on the HIP side, which is
   the identical code path in the sibling CMake module and is proven by the table above.

4. **gcc-11 / g++-11 pins.** Restored above `project()` under the same `if(NOT USE_HIP)`
   guard. `option(USE_HIP ...)` already precedes `project()`, so the guard is available.

5. **README arch sentence.** Rewritten to the real precedence: `CMAKE_HIP_ARCHITECTURES` when
   passed, else `PYTORCH_ROCM_ARCH`, else the GPUs in the build machine.

6. **README end-to-end gap.** The paragraph now says that both missing operators are reached
   from `quest/models/QuestAttention.py`, so the model wrappers, the accuracy evaluation
   described immediately below, the end-to-end efficiency scripts and the examples do not run
   on ROCm yet, and points at `quest/tests` for what does.

7. **bsk_ops reordering.** Upstream declaration and `m.def` order restored; the guards are now
   wrapped in place. The `.cu` diff is 4 added `#if`/`#endif` lines and the `.h` diff is the
   three guards plus the explanatory comment. No moved lines in either file.

GPU re-run after the fix (`HIP_VISIBLE_DEVICES=1`, gfx1100, ROCm 7.2.3, torch 2.14.0a0):
`test_rope.py` + `test_estimate.py` + `test_decode_attention.py` + `test_approx_attention.py`
= **121 passed in 4.72s**. Unchanged from the pre-fix run, which is the point: the build
changes did not disturb them.

Lesson promoted to the `cuda-to-rocm` skill, `references/strategy-b-torch.md`, as
"CMake-driven torch extension: `find_package(Torch)` takes the GPU architecture away from
you" -- it covers both the LoadHIP shadowing and the enable_language ordering, and it will
bite any torch extension built through CMake rather than `CUDAExtension`.

## Review 2026-08-08 (pass 2)

Re-reviewed fork sha `3465b34` (moat-port vs `01c1623`) on linux-gfx1100, scoped to the seven
defects from the first round. The delta `ff80217..3465b34` touches only README.md,
quest/ops/CMakeLists.txt and csrc/bsk_ops.{cu,h}, so the shim headers, the force-include
mechanism and the rms_norm mask are unchanged and were not re-litigated. Verdict:
changes-requested, for one defect introduced by the fix to item 1.

Reproduction of the item 1 measurement (independent, off `compile_commands.json`, configure
only, host env carrying `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;gfx1100`):

| case | `--offload-arch` |
|---|---|
| `-DCMAKE_HIP_ARCHITECTURES=gfx1201` | `gfx1201` |
| no `-D`, `PYTORCH_ROCM_ARCH=gfx908` | `gfx908` |
| no `-D`, env unset | `gfx1100` (cache holds `gfx1100;gfx1100;gfx1100;gfx1100`, so REMOVE_DUPLICATES is load bearing) |

All three match the porter's table. The precedence the README states is the precedence the
code produces.

### Defect: the PYTORCH_ROCM_ARCH path is not sticky, and a later reconfigure hard-fails

`quest/ops/CMakeLists.txt:28-30` sets `CMAKE_HIP_ARCHITECTURES` from the environment as a
NORMAL variable. Nothing ever writes it to the cache on that path:
`CMakeDetermineHIPCompiler.cmake:296` is `elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES)`, so the
normal variable suppresses the enumerator branch that would have cached it, and
`CMakeCache.txt` ends up with no `CMAKE_HIP_ARCHITECTURES` entry at all. The value therefore
exists only for the configure that saw the environment variable.

Any later cmake run in that build directory without the variable in the environment -- which
ninja does by itself whenever a listfile changes -- resolves `CMAKE_HIP_ARCHITECTURES` to
empty, so `QUEST_HIP_ARCHITECTURES` at :35 is empty and :83 sets an empty target property.
Reproduced on this branch:

```
PYTORCH_ROCM_ARCH=gfx908 cmake -S quest/ops -B b -GNinja -DUSE_HIP=ON ...   # ok, gfx908
env -u PYTORCH_ROCM_ARCH cmake -S quest/ops -B b                            # exit 1
  HIP_ARCHITECTURES is empty for target "_kernels".
  CMake Generate step failed.  Build files cannot be regenerated correctly.
```

This is a new failure mode: at `ff80217` the environment variable reached the compile line
through torch's own `LoadHIP.cmake`, which re-reads it every configure and falls back to
`rocm_agent_enumerator`, so the build directory never went empty. It is also the middle rung of
the precedence the README now documents at README.md:69.

Fix, at :28-30 -- make the resolved value a cache entry so it survives the configure that set
it. The surrounding `if(NOT CMAKE_HIP_ARCHITECTURES ...)` means `FORCE` can never clobber a
user's `-D`:

```
if(NOT CMAKE_HIP_ARCHITECTURES AND DEFINED ENV{PYTORCH_ROCM_ARCH})
  string(REPLACE " " ";" _quest_hip_archs "$ENV{PYTORCH_ROCM_ARCH}")
  set(CMAKE_HIP_ARCHITECTURES "${_quest_hip_archs}" CACHE STRING "HIP architectures" FORCE)
endif()
```

Verified against all three precedence cases plus a reconfigure with the variable removed:
`-D` still wins, the environment variable is still consulted only when nothing else set it,
and the second configure now reports `gfx908` instead of empty.

The same fix is needed in the code block the lesson promotes to
`.claude/skills/cuda-to-rocm/references/strategy-b-torch.md`, which currently reproduces the
non-cached form verbatim.

### Items 2-7

Verified fixed and correct.

- Item 2: `CMakeDetermineHIPCompiler.cmake:296-334` confirms the reachability claim and confirms
  CMake raises `Failed to find a default HIP architecture.` itself, so dropping the project's own
  FATAL_ERROR loses no diagnostic.
- Item 3: reasoning holds. `CMakeDetermineCUDACompiler.cmake:261` is
  `if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")`, and `project()` does not clear normal variables,
  so `native` set at CMakeLists.txt:11-13 survives to `enable_language(CUDA)` at :41 and to target
  creation at :79. The mechanism is the same one measured on the HIP side above. The commit
  message's Test Plan discloses that the CUDA build was not exercised and why; that is adequate.
  Residual untested surface, not a defect and nothing to change: the rapids-cmake `include()`
  calls now sit after `project()`/`enable_language(CUDA)` (CMakeLists.txt:43-51) rather than before
  `project()`; they only define functions and `rapids_cuda_init_architectures` is not called, and
  get_raft.cmake's relative order to CUDA enablement is unchanged.
- Item 4: pins restored at CMakeLists.txt:7-8 under the same guard.
- Item 5: README.md:69 matches the measured precedence exactly.
- Item 6: README.md:71 is accurate. `quest/utils/__init__.py:157,229` reference the two missing
  ops inside function bodies, so `import quest.utils` still succeeds and only the call paths fail;
  `quest/models/QuestAttention.py:117` and the decode_topk path reach both through those wrappers.
- Item 7: `git diff 01c1623...HEAD` on both files is additions only -- 4 lines in bsk_ops.cu, 10 in
  bsk_ops.h -- with no moved upstream line and no trailing-newline change.

Hygiene re-checked at this sha: `[ROCm]` title 50 chars, Claude named, no noreply trailer, no
added copyright or author lines (the two `Copyright (c) 2023 by FlashInfer team.` lines are the
preserved upstream headers on the copied shims), jargon clean over `01c1623..HEAD`, added lines
ASCII, no arch-conditional code anywhere under `kernels/include/hip_compat`, fork tree clean.

### Needs a PR against moat main, not this branch

`.claude/skills/cuda-to-rocm/**` is global and is currently only on `port/Quest` (commits
`6d42e1d`, `d7a6a61`). The `strategy-b-torch.md` section "CMake-driven torch extension:
`find_package(Torch)` takes the GPU architecture away from you" is correct and well stated -- its
claims about `Caffe2/public/LoadHIP.cmake:107` (`set(CMAKE_HIP_ARCHITECTURES ${PYTORCH_ROCM_ARCH})`
as a normal variable) and `Caffe2/public/utils.cmake:280` (`torch_hip_get_arch_list`, environment
first then `rocm_agent_enumerator`) both check out against the installed torch, as does the
"read the compile line, not the CMake" gate. Lift it to main once its code block carries the
`CACHE ... FORCE` correction above.

## Port round 3 (2026-08-08, linux-gfx1100) -- sticky architecture cache entry

Fixes the one defect from review pass 2. Fork head `3465b34` -> `c1d7fff` (amended; both
platforms' `validated_sha` were null, so no validated commit was orphaned).

`quest/ops/CMakeLists.txt:28-33` now writes the resolved architecture to the cache:

```cmake
if(NOT CMAKE_HIP_ARCHITECTURES AND DEFINED ENV{PYTORCH_ROCM_ARCH})
  string(REPLACE " " ";" _quest_hip_archs "$ENV{PYTORCH_ROCM_ARCH}")
  set(CMAKE_HIP_ARCHITECTURES "${_quest_hip_archs}" CACHE STRING "HIP architectures" FORCE)
endif()
```

Defect reproduced on the pre-fix tree first, so the regression test is known to bite: with
the old normal-variable form the first configure succeeded with no `CMAKE_HIP_ARCHITECTURES`
line in `CMakeCache.txt` at all, and `env -u PYTORCH_ROCM_ARCH cmake -S quest/ops -B <same>`
exited 1 with `HIP_ARCHITECTURES is empty for target "_kernels"`.

Four cases re-measured off `compile_commands.json` and `CMakeCache.txt`, configure only,
host env carrying `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;gfx1100`:

| case | `--offload-arch` | `CMakeCache.txt` | exit |
|---|---|---|---|
| `-DCMAKE_HIP_ARCHITECTURES=gfx1201` | `gfx1201` | `UNINITIALIZED=gfx1201` | 0 |
| no `-D`, `PYTORCH_ROCM_ARCH=gfx908` | `gfx908` | `STRING=gfx908` | 0 |
| no `-D`, env unset | `gfx1100` | `STRING=gfx1100;gfx1100;gfx1100;gfx1100` | 0 |
| **reconfigure of case 2 with env unset** | **`gfx908`** | `STRING=gfx908` | **0** |

The last row is the regression test for this defect. Case 1 keeps `UNINITIALIZED` because a
`-D` on the command line never reaches the `set(... FORCE)` -- the guard excludes it -- which
is the same evidence that `FORCE` cannot clobber a user's choice. Case 3 shows the enumerator
still writes the per-device duplicate list, so `REMOVE_DUPLICATES` at :39 stays load bearing.

Clean rebuild for gfx1100 and the four suites re-run on GPU 1: **121 passed in 4.69s**,
unchanged. Jargon clean over `01c1623..moat-port` for both commit messages and diff content.

The skill entries from this port (`SKILL.md`, `fault-classes.md`, `strategy-a-cmake.md`,
`strategy-b-torch.md`) stay on `port/Quest` rather than being lifted to main: they reach main
when this branch's own MOAT PR merges, which is after a reviewer has vetted them. This round
is the argument for that policy -- the `strategy-b-torch.md` code block reproduced the
non-sticky `set()` form verbatim, and promoting it early would have made this defect the
documented recipe for the next CMake-driven torch extension port. That block now carries the
`CACHE ... FORCE` form and a paragraph on why the cache entry matters, plus the
configure-then-reconfigure test as the way to check it.

## Review 2026-08-08 (pass 3)

Re-reviewed fork sha `c1d7fff` (moat-port vs `01c1623`) on linux-gfx1100, scoped to the one
defect pass 2 raised and to the skill entries this branch publishes. The delta
`3465b34..c1d7fff` is six lines of `quest/ops/CMakeLists.txt` plus commit-message text, so the
shim headers, the force-include mechanism, the rms_norm mask and the scoped-out suites were not
re-litigated. Verdict: changes-requested, for one defect in a global skill reference. The fork
code itself is correct at this sha and needs no change.

### The architecture fix: independently reproduced, both directions

Ran the regression test myself rather than reading the porter's table. Fresh build directories,
configure only, host env carrying `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;gfx1100` except where
noted, `--offload-arch` read off `compile_commands.json`.

Pre-fix tree (`3465b34`, checked out into a scratch worktree with `kernels/3rdparty` linked in
so the pybind submodule resolves) -- the defect is real and bites exactly as described:

| step | result |
|---|---|
| `PYTORCH_ROCM_ARCH=gfx908 cmake -S quest/ops -B b_pre ...` | exit 0, `--offload-arch=gfx908`, **no `CMAKE_HIP_ARCHITECTURES` line in `CMakeCache.txt`** |
| `env -u PYTORCH_ROCM_ARCH cmake -S quest/ops -B b_pre` | **exit 1**, `HIP_ARCHITECTURES is empty for target "_kernels"` |

Fixed tree (`c1d7fff`), four cases plus one the porter did not run:

| case | `--offload-arch` | `CMakeCache.txt` | exit |
|---|---|---|---|
| `-DCMAKE_HIP_ARCHITECTURES=gfx1201` | `gfx1201` | `UNINITIALIZED=gfx1201` | 0 |
| env `PYTORCH_ROCM_ARCH=gfx908`, no `-D` | `gfx908` | `STRING=gfx908` | 0 |
| neither set | `gfx1100` | `STRING=gfx1100;gfx1100;gfx1100;gfx1100` | 0 |
| **reconfigure of case 2 with `env -u PYTORCH_ROCM_ARCH`** | **`gfx908`** | `STRING=gfx908` | **0** |
| reconfigure of case 1 with the wheel list still in the env | `gfx1201` | `UNINITIALIZED=gfx1201` | 0 |

All match the porter's table. The last row is mine: it is the direct test that `FORCE` cannot
clobber a user's `-D`, and it holds because `if(NOT CMAKE_HIP_ARCHITECTURES ...)` at
`quest/ops/CMakeLists.txt:32` is already false on the second configure, so the `set(... FORCE)`
at :34 is never reached. `build.ninja` in that directory carries a `RERUN_CMAKE` edge on
`quest/ops/CMakeLists.txt`, so the reconfigure the defect breaks is one ninja performs by itself.

Behaviour worth knowing but not a defect: with the cache entry in place, changing
`PYTORCH_ROCM_ARCH` in an EXISTING build directory no longer changes the architecture
(reconfiguring the gfx908 directory with `PYTORCH_ROCM_ARCH=gfx942` still yields gfx908). That is
ordinary CMake cache stickiness, it is the same thing `-D` does, and it is the price of the fix.

Independent re-run of the four in-scope suites on GPU 1 (gfx1100, ROCm 7.2.3, torch 2.14.0a0):
`test_rope.py` + `test_estimate.py` + `test_decode_attention.py` + `test_approx_attention.py` =
**121 passed in 4.67s**, unchanged. `llvm-objdump --offloading` on the built module shows
`hipv4-amdgcn-amd-amdhsa--gfx1100` only. Jargon clean over `01c1623..moat-port` for both commit
messages and diff content; `HEAD == origin/moat-port == status.json.head_sha == c1d7fff`; fork
tree clean; title `[ROCm] Build the end-to-end operators for AMD GPUs` is 50 chars, Claude named,
no noreply trailer, commit message ASCII with no em-dash.

### Defect: the skill entry states, wrongly, that a set after enable_language is dead code

`.claude/skills/cuda-to-rocm/references/strategy-b-torch.md:56-57` -- "Anything that sets it
AFTER that line is dead code." That is false, and it is contradicted by point 2 of the same
section eleven lines further down, which is entirely about a `set()` after
`enable_language(HIP)` taking effect and stealing the architecture.

`enable_language(HIP)` writes CMAKE_HIP_ARCHITECTURES to the CACHE. A NORMAL variable set after
it shadows that cache entry and still initialises the target property at target creation.
Measured with a standalone probe on this host:

```
project(probe LANGUAGES CXX)
enable_language(HIP)                  # cache becomes gfx1100;gfx1100;gfx1100;gfx1100
set(CMAKE_HIP_ARCHITECTURES gfx908)   # normal variable, AFTER enable_language
add_library(probe_lib STATIC k.cpp)   # TARGET HIP_ARCHITECTURES=gfx908, --offload-arch=gfx908
```

That is precisely what `Caffe2/public/LoadHIP.cmake:107` does, and why this port has to snapshot
into QUEST_HIP_ARCHITECTURES at all. What IS dead after `enable_language()` is a set GUARDED on
the variable being unset, and a `set(... CACHE ...)` without FORCE. Quest's own CUDA leg proves
the distinction: at `ff80217` the line was `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` at :29
wrapping `set(... native)` at :30, immediately after `enable_language(CUDA)` at :27 -- dead
because of its own guard, not because a later set cannot win. Review pass 1's item 3 reached the
right conclusion by a mechanism the skill has now generalised incorrectly.

An agent reading the sentence as written concludes that torch's post-`enable_language` assignment
is harmless and skips the snapshot, which reproduces the exact bug the section exists to prevent.
Fix: replace the sentence with the guarded/cache distinction, e.g. "after that line the variable
is DEFINED, so any set guarded on it being unset is dead and a `set(... CACHE ...)` without FORCE
is a no-op; an unguarded normal set still wins at target creation, which is what point 2 below is
about."

### The rest of the skill delta is accurate

Checked against the sources it names, not against the porter's summary.

- `CMakeDetermineHIPCompiler.cmake:296` is `elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES)` in both
  cmake 3.28 and 3.31, and :328/:330 are the only writes of that cache entry, so the
  "suppresses the branch that would have persisted it" argument at strategy-b-torch.md:92-107 is
  exactly right, and the code block at :75-84 is the fixed `CACHE STRING ... FORCE` form.
- `CMakeDetermineCUDACompiler.cmake:261` is `if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")`.
- `Caffe2/public/LoadHIP.cmake:107` and `Caffe2/public/utils.cmake:280`
  (`torch_hip_get_arch_list`, environment first then `rocm_agent_enumerator`) are as quoted.
- `amd_warp_sync_functions.h:177` static_asserts `sizeof(MaskT) == 8` with "The mask must be a
  64-bit integer", and `-DHIP_ENABLE_WARP_SYNC_BUILTINS` is on this extension's compile line via
  `Caffe2Targets.cmake:109`, so the fault-classes.md entry and its "explicit width is already
  wave-agnostic" correction both hold. `rms_norm.cu` widens the literal on the AMD side only and
  leaves the CUDA value numerically identical.
- SKILL.md:102-105 index lines match the reference entries they point at.

### README.md on this branch: restore main's copy, do not leave it to the sync

Not a port defect; recorded because it was asked. `README.md` is generated output covering EVERY
project, and a port branch owns only its own `projects/<name>/`, so this branch's copy is
authoritative for nothing. It has diverged in BOTH directions against main -- this branch is
ahead for alien and catboost, main is ahead for faster-gaussian-splatting, GooFit, GPU_IPC and
HEonGPU -- so a textual merge at sync time can silently regress main's table for the second
group. Restore it now (`git checkout origin/main -- README.md` on `port/Quest`) and let main
regenerate; that turns a conflict with a wrong-answer resolution into a no-op. Regenerating the
table on a port branch is the thing to stop doing, not just this instance of it.

Observation, not a defect and no change requested: the commit title's "the end-to-end operators"
echoes upstream README.md:51 ("4. Build end-to-end operators with PyBind"), which is the build
step this change touches, but the same README uses "end-to-end" for the full model pipeline in
five other places, and that pipeline is exactly what this port cannot run. The body says so
plainly in its fifth paragraph, so a maintainer reading both is not misled. Weigh a retitle
before the PR opens rather than amending for it now.

## Port round 4 (2026-08-08, linux-gfx1100) -- control-plane only, fork untouched

Review pass 3 found the fork code correct at `c1d7fff`, so nothing in `projects/Quest/src`
changed and `head_sha` is unchanged. No rebuild and no re-run: the reviewer independently
re-ran the four in-scope suites at this sha (121 passed on GPU 1). Fork tree clean,
`HEAD == origin/moat-port == status.json.head_sha == c1d7fff`.

Corrected the false claim in `.claude/skills/cuda-to-rocm/references/strategy-b-torch.md`,
point 1 of the CMake-driven-torch-extension entry. It said "Anything that sets it AFTER that
line is dead code", which point 2 eleven lines below already contradicted. What
`enable_language(HIP)` does is make the variable DEFINED, and that kills exactly two
spellings -- a `set()` guarded on it being unset, and a `set(... CACHE ...)` without `FORCE`.
An unguarded normal `set()` afterwards still wins: it shadows the cache entry and initialises
`HIP_ARCHITECTURES` on every target created after it, which is exactly the
`Caffe2/public/LoadHIP.cmake:107` assignment point 2 is about. Point 1 now says that, names
point 2 as the case where the late set does win, and point 2 opens by naming itself as that
unguarded set, so the two read as one account. The Quest CUDA-leg example now states the
actual mechanism: at `ff80217` the `set(CMAKE_CUDA_ARCHITECTURES native)` carried its own
`if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard, so moving `enable_language(CUDA)` above it
made the guard false -- dead because of the guard, not because a later set cannot win. The
rest of that entry was verified accurate by the reviewer against the CMake and torch sources
and is unchanged.

Why this mattered enough to bounce the port: an agent who believes a late set is inert
concludes torch's post-`enable_language` assignment is harmless, skips the snapshot into
`QUEST_HIP_ARCHITECTURES`, and reproduces the exact bug the section exists to prevent.

`README.md` restore was a no-op. `git checkout origin/main -- README.md` changed nothing:
this branch's copy and `origin/main`'s are already the same blob (`562ad7a`), because main's
own regeneration (`048c6ba`) reads project records across the port branches and so picked up
the alien and catboost rows this branch's copy had. The standing rule holds anyway -- a port
branch owns only its own `projects/<name>/` and must not regenerate the board -- and the
three "Regenerate the README table" commits on this branch (`a837a4e`, `6f04063`, `0b0954a`)
are the instances to stop repeating, not to revert now that the content agrees.

### Open item for whoever opens the upstream PR

The commit title `[ROCm] Build the end-to-end operators for AMD GPUs` keeps its wording for
now, by the reviewer's judgement, and is the one thing to weigh before the PR goes out.
"End-to-end operators" is upstream's own name for the build step this change touches
(`README.md:51`, "4. Build end-to-end operators with PyBind"), but the same README uses
"end-to-end" for the full model pipeline in five other places, and the full pipeline is
precisely what this port does not claim to run. The body says so plainly in its fifth
paragraph. Decide it at PR time; if you retitle, it is a new commit on top, never an amend,
because by then an arch may have validated `c1d7fff`.

## Review 2026-08-08 (pass 4, linux-gfx1100) -- narrow confirmation, PASS

No problems found. Round 4 is control-plane only: `head_sha` unchanged at `c1d7fff`, fork
tree clean, `HEAD == origin/moat-port == status.json.head_sha`, `porting` null, and
`jargon.py` clean over both `origin/main..HEAD` here and `01c1623..c1d7fff` on the fork.
The port code, the build fix and the test results were not re-reviewed; pass 3 settled them.

The corrected sentence in `strategy-b-torch.md:56-59` is now true, and the NEGATIVE half --
that the two named spellings are the dead ones -- was checked here rather than taken on
trust. A standalone probe (cmake 3.31.6, `cmake_minimum_required(VERSION 3.21)`, real
`enable_language(HIP)` on this host) ran all three spellings against one cache entry left by
`enable_language(HIP)` (`gfx1100;gfx1100;gfx1100;gfx1100`):

- `if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)` evaluated FALSE, so a guarded `set()` never runs.
  DEAD.
- `set(CMAKE_HIP_ARCHITECTURES gfx801 CACHE STRING "nf")` left both the cache entry and the
  read value unchanged. DEAD. It also does not clobber an existing normal-variable shadow
  (CMP0126 is NEW at this minimum), so it is inert in both directions.
- An unguarded `set(CMAKE_HIP_ARCHITECTURES gfx1100)` won: the read value became `gfx1100`
  while `get_property(... CACHE ...)` still showed the four-entry cache, i.e. a shadow, and
  it propagated into a subdirectory added after it.

The "every target created after it" qualifier is exact, not loose. A second probe created one
HIP executable before the unguarded `set()` and one after: the earlier target got the cache
value (`gfx1100;gfx1100;gfx1100;gfx1100`) and kept it on re-read after the `set()`, while the
later target got `gfx942`. So the shadow initialises `HIP_ARCHITECTURES` at target-creation
time for the rest of the directory scope and cannot reach back.

The upstream citations behind the entry hold: `CMakeDetermineHIPCompiler.cmake:296`
(`elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES)`), `CMakeDetermineCUDACompiler.cmake:261`
(`if("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")` -> cache set at :264, and it FATAL_ERRORs
rather than leaving the variable undefined, so a later `if(NOT DEFINED ...)` guard is always
false on a configure that succeeds), and `LoadHIP.cmake:107`
(`set(CMAKE_HIP_ARCHITECTURES ${PYTORCH_ROCM_ARCH})`, unguarded and normal, with that file's
own `enable_language(HIP)` at :139 below it -- so point 2's parenthetical is correctly
ordered).

The rewritten Quest CUDA-leg example is accurate against the fork: at `ff80217`,
`quest/ops/CMakeLists.txt` has `enable_language(CUDA)` immediately above its own
`if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES) set(CMAKE_CUDA_ARCHITECTURES native)`, so the guard
is what killed it. The pre-port file at `01c1623` had that same guarded set ABOVE
`project(_kernels LANGUAGES CUDA CXX)`, where it worked -- which is exactly why the old wording
read as if bare ordering were the mechanism.

Points 1 and 2 read as one account with no residual contradiction: point 1 forward-references
point 2 as the case where a late set does win, and point 2 opens by naming itself as that
unguarded normal `set()`. The rest of the file is untouched -- the round-4 delta is two hunks,
23 insertions / 13 deletions, confined to those two paragraphs, and a word-diff shows the
surviving CUDA and "belong above `project()`" sentences are reflow only. Longest added line is
94 chars, matching the file's wrap.

`README.md` needed no restore: blob `562ad7a9b88b66b2faf6e84f1912c9df46e6ae61` on all three of
`origin/main:README.md`, `HEAD:README.md`, and the worktree. The porter's MECHANISM is right --
`gen_readme.py:33-34` builds every row from `moatlib.all_projects()` + `project_record()`, both of
which resolve across the port branches, so a regeneration run on main already carries in-flight
branches, which is why main's copy converged on this branch's content by itself (main's README
carries the Quest row at line 166 today).

The CONCLUSION the porter drew from it is wrong, and this review nearly repeated it. "A port
branch must not regenerate the board, and the three `Regenerate the README table` commits
(`a837a4e`, `6f04063`, `0b0954a`) are the instances to stop repeating" does not survive contact
with `utils/check.py:109-115`: the readme gate runs on EVERY pre-push from a full clone,
regardless of branch, and `gen_readme.py --check` sees this branch's own state.json. Setting a
stage on a port branch therefore stales the branch's table and the next push is refused -- which
is exactly what happened to this review's push, and is what those three commits were doing. They
are the gate being satisfied, not a habit to break.

Both statements are true at once and they are about different things. The board CONVERGES without
a port branch's help, so a branch never has to regenerate it to make main correct; the gate still
REQUIRES the branch's own copy to be fresh at push time, so in practice every state transition on
a port branch carries a regeneration with it. Do not treat a `Regenerate the README table` commit
on a port branch as noise, and do not reach for `git push --no-verify` when the readme gate fires
-- run `python3 utils/gen_readme.py` and include the result in the same commit.

Verdict: review-passed. The one thing still open for whoever opens the upstream PR is the
commit-title wording recorded above; it is a judgement call at PR time, not a defect.

## Validation 2026-08-08 (linux-gfx1100)

Real-GPU validation of fork sha `c1d7fff` on Radeon Pro W7800 (gfx1100, wave32), ROCm 7.2.3,
PyTorch 2.14.0a0+gitb81488e, python 3.12, `HIP_VISIBLE_DEVICES=1`. Clean rebuild (wiped
`quest/ops/build` first, per the note above about force-included headers under
`kernels/include/hip_compat`):

```
mkdir -p quest/ops/build && cd quest/ops/build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

Wrapped `utils/timeit.sh Quest compile -- ninja -C quest/ops/build`. Configure and build both
clean (6/6 objects, two pre-existing nodiscard-`hipError_t` warnings, no errors).
`llvm-objdump --offloading` on the built `.so` shows `hipv4-amdgcn-amd-amdhsa--gfx1100` in
every offload bundle and no other arch -- confirms the architecture-selection fix independently,
by inspecting the linked module rather than re-deriving the CMake mechanism (already measured
five ways by the porter and two reviewers; not repeated here).

GPU tests, wrapped `utils/timeit.sh Quest test -- python3 -m pytest ...`:

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| **in-scope total** | **121 passed in 4.67s** |
| test_topk.py | 49 failed, 15 skipped |
| test_prefill_attention.py | 29 failed, 13 skipped |
| **scoped-out total** | **78 failed** (matches prior sessions exactly) |

Confirmed the scoped-out failure mode is still `AttributeError`, not a wrong answer, with `-x`
on both files: `AttributeError: module 'quest._kernels' has no attribute 'topk_filtering'` and
`AttributeError: module 'quest._kernels' has no attribute 'prefill_with_paged_kv_cache'`. No
non-GPU regression to check against an upstream baseline -- this project has no CPU-only test
path; all six suites require the compiled extension and a GPU.

### CUDA no-regression gate: partial, walled by an external dependency, recorded honestly

No CUDA gate recorded at `c1d7fff` before this run, so it was owed once for this head_sha.
This host has no NVIDIA GPU. `/opt/conda/envs/cuda-12.8` already existed with nvcc 12.8.93.
`gcc-11`/`g++-11` (the project's pinned CUDA-leg host compiler, `quest/ops/CMakeLists.txt:7-8`)
were absent and installed via `apt-get install -y gcc-11 g++-11` (standard Ubuntu noble
universe packages, within the standing "install missing build deps via apt" allowance).

Configure attempt, throwaway build dir, arch pinned per the standing instruction rather than
left to `native` (no NVIDIA GPU on this host to autodetect against):

```
cd quest/ops && mkdir build-cuda && cd build-cuda
cmake -GNinja -DUSE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
```

Result: `enable_language(CUDA)` succeeded with the conda nvcc
(`CMAKE_CUDA_COMPILER:FILEPATH=/opt/conda/envs/cuda-12.8/bin/nvcc` in `CMakeCache.txt`), and the
pinned arch survived to the cache untouched (`CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=80`) --
new evidence beyond the porter's/reviewer's standalone `project(probe LANGUAGES NONE)` probes,
because this is the real project's own `CMakeLists.txt` guard (`if(NOT DEFINED
CMAKE_CUDA_ARCHITECTURES) set(... native)` at line 11) exercised for real: with an explicit `-D`
already defining the variable, the guard is false, `native` is correctly skipped, and the `-D`
value reaches `enable_language(CUDA)` unclobbered. This confirms the *mechanism* item 3 relies on
(a value set before `enable_language(CUDA)` survives; `project()` does not clear it) end to end
for the first time, though it still does not exercise the actual `native` default path -- that
remains correctly untested, deliberately, per the standing instruction to always pin.

Configure then failed downstream, before reaching any of Quest's own `.cu` files or the
`pybind11_add_module(_kernels ...)` target:

```
CMake Error at build-cuda/_deps/raft-src/cpp/CMakeLists.txt:262 (target_link_libraries):
  The link interface of target "raft" contains:
    CUDA::nvToolsExt
  but the target was not found.
```

Root cause: RAFT is pinned to `branch-24.02` (`quest/ops/cmake/get_raft.cmake`, ~early 2024) and
its CMakeLists still links the legacy `CUDA::nvToolsExt` imported target. Modern CMake's
`FindCUDAToolkit` (3.31, shipped with this environment) no longer creates that target against a
CUDA 12.8 toolkit layout -- NVTX is header-only (`nvtx3/nvToolsExt.h`, present here under
`nsight-compute-2025.1.1/.../nvtx/include/`, not at the legacy top-level include path
`FindCUDAToolkit` probes) and the shared-library target it backed was retired. This is a version
mismatch between an old pinned RAPIDS dependency and a modern CUDA toolkit, entirely external to
Quest's and this port's own code -- it would equally block an upstream CUDA build attempted with
this same CUDA 12.8 + CMake 3.31 combination, independent of anything USE_HIP touches. Not
attempted as a local workaround (e.g. stubbing the imported target) because that patches RAFT's
build system, not Quest's, and the fix belongs with RAPIDS/raft's own CMake modernization, not
with this port.

**`cuda-not-validated: RAFT (pinned branch-24.02) requires the legacy CUDA::nvToolsExt CMake
target, which FindCUDAToolkit no longer creates against a CUDA 12.8 toolkit (NVTX is
header-only in 12.x); configure fails inside RAFT's own CMakeLists before Quest's own .cu
sources or the pybind11 target are ever reached.`** The CUDA path is therefore unexercised
beyond `enable_language(CUDA)` and the architecture-cache mechanism above: no nvcc invocation
for any of Quest's own kernels was observed, and item 3's fix (native-arch preservation via
ordering) remains reasoned-and-partially-probed rather than fully run, exactly as recorded by
the porter and reviewers. Budget: approx 15 minutes on this leg, consistent with the standing
"do not grind" instruction; stopped after establishing the wall rather than patching RAFT.

Throwaway `quest/ops/build-cuda` directory removed before completion; `gcc-11`/`g++-11` left
installed at the host level (apt package, not part of the fork tree).

### Jargon and documentation

`python3 utils/jargon.py --commits 01c1623..HEAD -C projects/Quest/src` and
`python3 utils/jargon.py --diff 01c1623..HEAD -C projects/Quest/src` (this checkout predates
`--port`): both `jargon: clean`.

README.md's "Building on AMD GPUs (ROCm)" section matches the commands run above verbatim,
states the real precedence (`CMAKE_HIP_ARCHITECTURES` when passed, else `PYTORCH_ROCM_ARCH`,
else the build machine's GPUs), and names both scoped-out ops and everything downstream of them
(model wrappers, accuracy evaluation, end-to-end scripts, examples) as not running on ROCm yet.
No inaccuracy found; nothing sent back.

`git -C projects/Quest/src status --porcelain`: clean. `HEAD` at `c1d7fff`, matching
`status.json.head_sha`.

State: `review-passed` -> `completed` on linux-gfx1100, `validated_sha = c1d7fff`. wave32 gate
satisfied by this arch. wave64 and windows remain open (`pr-ready` still reports
linux-gfx90a=None and windows=no viable arch yet); this arch does not carry either attribute
and gates neither. CUDA gate: not satisfied (environmental wall, recorded above), carries no
penalty per the gate's own rules -- it is secondary and compile-only.

## Validation 2026-08-09 (linux-gfx90a)

First real-GPU validation of fork sha `c1d7fff` on this arch (`port-ready`, no prior
`validated_sha`). MI250X (gfx90a, wave64), `HIP_VISIBLE_DEVICES=1` (index 1 confirmed gfx90a
via `rocm-smi --showhw`), ROCm 7.2.1, PyTorch 2.14.0a0+gitb6b444c (hip 7.2.53211), python
3.12.13. Cloned the fork fresh into `projects/Quest/src` and checked out `moat-port`
(`HEAD == c1d7fff == status.json.head_sha`). Submodules `kernels/3rdparty/{flashinfer,pybind}`
initialized (needed for the shim force-include mechanism and the vendored pybind11 CMake
target respectively; `raft`/`googletest`/`nvbench` left uninitialized, matching the scoped-out
CUDA-only/RAFT-only surface). Installed `transformers==4.37.2` `tokenizers==0.15.2` per this
file's earlier recipe.

Build, wrapped `utils/timeit.sh Quest compile -- ninja -C quest/ops/build`:

```
mkdir -p quest/ops/build && cd quest/ops/build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

Configure and build both clean (6/6 objects, same two pre-existing nodiscard-`hipError_t`
warnings on `decode_handler.cuh:165` and `approx_attn.cu:65` as every prior session, no new
warnings, no errors). `llvm-objdump --offloading _kernels*.so` shows
`hipv4-amdgcn-amd-amdhsa--gfx90a` in every offload bundle and no other arch -- independent
reconfirmation of the architecture-selection fix (item 1) on a second, wave64 arch.
`python3 -c "import quest._kernels as k; print(dir(k))"` lists exactly the six ops the README
names (`BatchDecodeWithPagedKVCachePyTorchWrapper`, `append_kv_cache_decode`,
`append_kv_cache_prefill`, `apply_rope_in_place`, `estimate_attn_score`, `rms_norm_forward`).

GPU tests, wrapped `utils/timeit.sh Quest test -- python3 -m pytest ...`:

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| **in-scope total** | **121 passed in 8.56s** |
| test_topk.py | 49 failed, 15 skipped |
| test_prefill_attention.py | 29 failed, 13 skipped |
| **scoped-out total** | **78 failed, 28 skipped** (matches gfx1100 exactly) |

Confirmed real GPU dispatch, not a CPU/skip pass: `AMD_LOG_LEVEL=3` on `test_decode_attention.py`
and on `test_rope.py`+`test_estimate.py`+`test_approx_attention.py` shows `hipLaunchKernel`
calls returning `hipSuccess` with `ShaderName` entries for Quest's own compiled kernels --
`flashinfer::AppendPagedKVCacheDecodeKernel`, `AppendPagedKVCachePrefillKernel`,
`BatchDecodeWithPagedKVCacheKernel`, `VariableLengthMergeStatesKernel`,
`MaxPossibleSampleWithPagedKVCacheKernel`, `QKApplyRotaryInPlaceKernel` -- interleaved with
PyTorch's own RNG/copy/fill kernels, i.e. actual device execution, not a mock. No dedicated
`rms_norm` test exists in this suite (pre-existing gap, not introduced by the port); the op
still built and links (symbol present in `dir(quest._kernels)`).

Confirmed the scoped-out failure mode is still `AttributeError`, not a wrong answer, with `-x`
on both files: `AttributeError: module 'quest._kernels' has no attribute 'topk_filtering'`
(`test_topk.py:53`) and `AttributeError: module 'quest._kernels' has no attribute
'prefill_with_paged_kv_cache'` (`quest/utils/__init__.py:157`). No non-GPU regression to check
against an upstream baseline -- this project has no CPU-only test path; all six suites require
the compiled extension and a GPU.

### CUDA no-regression gate

Not repeated: already recorded at this head_sha (`c1d7fff`) by the linux-gfx1100 validation
above as `cuda-not-validated` (environmental wall -- RAFT branch-24.02 pinning
`CUDA::nvToolsExt`, which `FindCUDAToolkit` no longer creates against a CUDA 12.8 toolkit; the
configure fails inside RAFT's own CMakeLists before Quest's own `.cu` sources are reached).
Runs once per head_sha per the standing rule; not re-run here.

### Jargon and documentation

`python3 utils/jargon.py --port Quest`: `jargon: clean`.

README.md's "Building on AMD GPUs (ROCm)" section matches the commands run above verbatim
(only the `--offload-arch` differs, gfx90a vs gfx1100, which is exactly the
`CMAKE_HIP_ARCHITECTURES` argument the doc says to pass) and correctly states that
`topk_filtering` and `prefill_with_paged_kv_cache` -- and everything downstream of them -- do
not run on ROCm yet. No inaccuracy found; nothing sent back.

`git -C projects/Quest/src status --porcelain`: clean. `HEAD` at `c1d7fff`, matching
`status.json.head_sha`. No source or build edit was needed on this arch (the CMake
architecture-cache fix from the gfx1100 rounds already covers gfx90a; verified by the
`--offload-arch=gfx90a`-only objdump result above).

State: `port-ready` -> `completed` on linux-gfx90a, `validated_sha = c1d7fff`. wave64 gate now
satisfied (this arch carries wave64). windows remains open. CUDA gate unaffected (recorded once
already at this head_sha, per the standing rule); no penalty. Wall clock: build ~50s, in-scope
tests ~9s, scoped-out re-runs and AMD_LOG_LEVEL evidence runs a few minutes more; well under the
~60 minute budget.

## Validation attempt 2026-08-13 (windows-gfx1151) -- builds and loads, tests blocked on the harness

Platform: AMD Radeon 8060S (gfx1151, RDNA3.5, wave32), Windows 11. torch
2.12.0+rocm7.14.0a20260519 with the `[device-gfx1151]` extra, ROCm 7.14 devel,
Python 3.13.14.

Outcome: the extension now BUILDS and LOADS on Windows -- which it did not
before this round -- but the test suite could not be run, for a reason that has
nothing to do with the GPU. Recorded `validation-failed` because no real-GPU
pass was obtained; the blocker and the remaining work are both concrete.

### Windows is not the CMake-clang-cl case; this project needs the GNU driver

Configuring with `clang-cl` for C/CXX/HIP succeeds under CMake 4.3, and then
`csrc/page.cu` fails deep inside the ORIGINAL flashinfer headers
(`invalid output constraint '=f' in asm`, `function template partial
specialization is not allowed`) -- as if the port's compat shims were not there.
They were not. `quest/ops/CMakeLists.txt:111-113` force-includes them with
GNU-style `-include<path>`, which `clang-cl` does not honour (its spelling is
`/FI`), so the shims silently do nothing and the upstream CUDA headers compile
raw. Nothing in the error names the cause.

Configure with `amdclang++` for both CXX and HIP instead and the shims apply.
The general rule: a project that passes GNU-style flags from its own CMakeLists
forces the GNU driver regardless of what the CMake version would otherwise
allow. `-fuse-ld=lld-link` still has to be stripped from `build.ninja` (1
occurrence) as in the other Windows CMake ports.

### Real source defect found and fixed: LLP64 vs LP64 in the decode kernels

With the shims working, the build failed in the project's OWN headers:

```
kernels/include/decode/decode_attn.cuh:920:32: error: no matching function for call to 'max'
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeIn), HEAD_DIM / 32UL);
```

`unsigned long` is 64-bit on Linux (LP64) and 32-bit on Windows (LLP64), while
`sizeof` is always `size_t`. On Linux the two arguments happen to share a type
and deduction succeeds; on Windows they do not and every instantiation fails.
Eight sites across `decode_attn.cuh` and `decode_page.cuh`.

This is upstream code, untouched by the port -- `git log` on the file shows only
"[Major] Initial commit", and the port's diff against its base does not touch
these lines -- so a CUDA build on Windows fails identically today. Fixed by
naming the template argument, `std::max<size_t>(...)`, which leaves every
already-compiling instantiation unchanged. Committed as `ac7382c`; head_sha
c1d7fff -> ac7382c. Expect linux-gfx90a and linux-gfx1100 to carry forward:
the change is a no-op on LP64, so `codeobj_diff` should report identical.

After it, all 6 objects compile and link, and `import quest._kernels` exposes
the expected symbols (`BatchDecodeWithPagedKVCachePyTorchWrapper`,
`append_kv_cache_decode/prefill`, `apply_rope_in_place`, `estimate_attn_score`,
`rms_norm_forward`).

### Test harness: resolved, needed Python 3.12

All four in-scope suites import `transformers`, and their reference paths use the
pre-4.38 rotary API (`LlamaRotaryEmbedding(head_dim)`,
`rotary_emb(q, seq_len=...)`). The working version is `transformers==4.37.2` as
the earlier notes record -- `pyproject.toml`'s `transformers==4.45.2` pin is
stale and fails with `AttributeError: 'int' object has no attribute
'max_position_embeddings'` then `TypeError: forward() got an unexpected keyword
argument 'seq_len'`.

4.37.2 needs `tokenizers>=0.14,<0.19`, which has no cp313 wheel and is enforced
at import (so `--no-deps` does not dodge it). Resolved by installing Python 3.12
and building a second environment on it; TheRock publishes cp312 wheels of the
same torch build. Env script: `agent_space/win_torch312_env.sh`. The kernels were
rebuilt against that interpreter (`_kernels.cp312-win_amd64.pyd`).

### Test results -- 73 passed, 48 failed. The decode attention path is WRONG on this arch

```
python -m pytest quest/tests/test_rope.py quest/tests/test_estimate.py                  quest/tests/test_decode_attention.py                  quest/tests/test_approx_attention.py -q
# 48 failed, 73 passed in 5.52s
```

| suite | linux-gfx1100 / gfx90a | windows-gfx1151 |
|---|---|---|
| test_rope.py | 64 passed | **64 passed** |
| test_estimate.py | 9 passed | **9 passed** |
| test_decode_attention.py | 6 passed | **6 FAILED** |
| test_approx_attention.py | 42 passed | **42 FAILED** |

The failures are not tolerance drift. They are structural:

```
AssertionError: Tensor-likes are not close!
Mismatched elements: 4036 / 4096 (98.5%)
Greatest absolute difference: 1.36328125 (up to 0.005 allowed)
Greatest relative difference: 6472.0
```

98.5-98.6% of elements wrong, absolute differences of 1.4-2.5 against a
reference whose values are order 0.1-0.5. The output is unrelated to the answer,
not a slightly-off version of it -- so this is NOT the RDNA3.5 floating-point
accumulation class recorded elsewhere in the skill, which produces small
divergences.

The split is informative: `apply_rope_in_place`, `estimate_attn_score` and the
KV-cache append path all produce correct results, and the two failing suites are
exactly the ones that run `BatchDecodeWithPagedKVCache` -- the flashinfer-derived
kernel, and the only one reached through the `kernels/include/hip_compat/flashinfer`
shims (`math.cuh`, `vec_dtypes.cuh`).

### Root cause: the single-block decode path is broken (FIXED, 9be60fc)

**An earlier section of this file blamed no-op `cp_async` primitives and a
multi-wavefront race. That was wrong.** It is true that hipcc does not define
`__CUDACC_VER_MAJOR__`, so `commit_group`/`wait_group` compile to nothing -- but
that is not what breaks this kernel, and every "fix" derived from it failed:
adding a real `__syncthreads()` to `wait_group` changed nothing (48 failed,
byte-identical). The bdz and num_stages_smem experiments that pointed there were
reading a shadow: both knobs change `smem_size`, `smem_size` changes occupancy,
and occupancy is what selects the launch shape.

The sparse decode has two launch shapes. The **split** shape gives each block a
slice of the pages and merges partial states afterwards. The **single-block**
shape hands one block the whole range, and is chosen when

```
batch_size * num_kv_heads >= max_grid_size     // = blocks_per_cu * cu_count
```

The single-block shape is broken. It launches `dim3 nblks(batch_size,
num_kv_heads)` so `gridDim.x == 1`, and the kernel derives its slice length from

```
(batch_idx == gridDim.x - 1) ? paged_kv.last_page_len : paged_kv.page_size
```

which is written for the split shape, where only the final slice ends on a
partial page. With one block that test is always true, so the slice length
collapses to the trailing page length of the whole cache. The caller passes
`kv_indices_without_last` -- the trailing partial page already removed -- so the
value is unrelated to what the block should read.

Measured directly with in-kernel `printf`, `kv_len=17`:

| config | gridDim.x | `batch_idx==gridDim.x-1` | `cur_last_page_len` | `kv_chunk_len` |
|---|---|---|---|---|
| split (ns=1) | 2 | false | page_size 16 | **16** correct |
| single (ns=2) | 1 | **true** | last_page_len 1 | **1** wrong |

A 577-token sequence attends over 1 position instead of 576.

### Why only this GPU

`max_grid_size = blocks_per_cu * cu_count`, and these shapes need
`1 * 32 = 32` blocks:

| GPU | CUs | blocks/CU @36864 B smem | max_grid | path taken |
|---|---|---|---|---|
| gfx1151 (APU) | 20 | 1 | 20 | **single-block -- broken** |
| gfx1100 (W7800) | 70 | 1 | 70 | split -- fine |
| gfx942 (MI300X) | 304 | 1 | 304 | split -- fine |

So this is a **CU-count-dependent latent defect**, not an RDNA3.5 fault, not a
compiler bug, and not a race. The larger parts never execute the broken path,
which is why they validated clean. Their results are sound -- the earlier claim
in this file that they "pass by scheduling luck" was part of the wrong diagnosis
and is withdrawn.

### Fix (9be60fc)

Take the split path unconditionally. Correcting the `cur_last_page_len`
expression alone was tried and is **not** sufficient -- with it applied,
`kv_chunk_len` reads the correct 576 and the tests still fail, so the path has at
least one further defect. Disabling it is the honest change; repairing and
re-enabling it is registered as `quest-decode-single-block-path-broken`.

On a GPU where the old condition was false this changes nothing at all, so
gfx1100 and gfx942 behaviour is untouched.

### Result: 121/121, all in-scope suites

```
python -m pytest quest/tests/test_rope.py quest/tests/test_estimate.py   quest/tests/test_decode_attention.py quest/tests/test_approx_attention.py -q
# 121 passed
```

| suite | expected | windows-gfx1151 |
|---|---|---|
| test_rope.py | 64 | **64 passed** |
| test_estimate.py | 9 | **9 passed** |
| test_decode_attention.py | 6 | **6 passed** |
| test_approx_attention.py | 42 | **42 passed** |

Matches the linux-gfx1100 / linux-gfx90a reference exactly. Swept `kv_len` over
17..1541: peak absolute error **1.2e-4** (fp16 rounding) and output/reference
standard-deviation ratio **1.000** at every length, against 1.0-4.0 absolute
error and ratios 0.78-6.57 before the fix.

`test_topk.py` and `test_prefill_attention.py` remain scoped out and fail with
`AttributeError` on absent symbols -- the recorded expected shape, unchanged.

CUDA no-regression gate: not run (Windows host, no CUDA toolkit). Owed at this
head_sha by whichever Linux arch revalidates first; the change is a host-side
launch decision, compiled on both paths.
Not applied here: it changes kernel configuration for every platform, so it
wants a porter's commit and review, and it will require revalidating the Linux
arches -- the right outcome, since their current pass rests on the race.
Integrity: `git status --porcelain` in src clean at `ac7382c`; the only fork
change this round is that commit.

## Validation 2026-08-13 (linux-gfx90a, revalidate after the LLP64 fix)

Re-validation of fork sha `ac7382c` on this arch after `head_sha` advanced from
`c1d7fff` (this platform's prior `validated_sha`) via the windows-gfx1151
session's `std::max<size_t>` fix. MI250X (gfx90a, wave64), 4 GPUs visible and
idle, `HIP_VISIBLE_DEVICES=1`. ROCm 7.14 (`hip 7.14.60850`), PyTorch
2.14.0a0+git7d05abc, python 3.12.13. Cloned the fork fresh into
`projects/Quest/src`, checked out `moat-port` (`HEAD == ac7382c ==
status.json.head_sha`), submodules `kernels/3rdparty/{flashinfer,pybind}`
initialized (same subset as prior sessions).

### Carry-forward shortcut attempted, then abandoned for a full run

`python3 utils/moatlib.py classify Quest c1d7fff ac7382c` returned
`class=mixed arch_independent=False` (`decode_attn.cuh`/`decode_page.cuh`
token counts differ), which is the case the carry-forward binary-equivalence
check exists for. Built `ac7382c` cleanly into `build-head` first. Building
`c1d7fff` into a second tree for `codeobj_diff` required a second worktree and
a second submodule checkout of `flashinfer`/`pybind`; the network-bound
submodule clone stalled/partially failed twice in this environment (empty
working tree after an interrupted fetch, then a wrong-commit checkout on
retry). Rather than keep fighting the network for a shortcut whose only
benefit is skipping a GPU run that costs under 10 seconds once the extension
is built, abandoned the codeobj_diff path and ran the normal full real-GPU
revalidation instead, per the validator's own fallback ("if you cannot build
both shas, do the normal full real-GPU revalidation"). Worktree and its build
directory removed.

### Build

Wrapped `utils/timeit.sh Quest compile -- ninja -C quest/ops/build-head`:

```
mkdir -p quest/ops/build-head && cd quest/ops/build-head
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

Clean (6/6 objects, no errors). `grep -o -- '--offload-arch=[a-z0-9]*'
build.ninja` shows `gfx90a` only, confirming the architecture-selection fix
still holds on this delta. `import quest._kernels` exposes the same six ops
as every prior session.

### GPU tests

Wrapped `utils/timeit.sh Quest test -- python3 -m pytest ...` (absolute test
paths, since `timeit.sh` cds to the MOAT repo root before running the wrapped
command):

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| **in-scope total** | **121 passed in 9.43s** |
| test_topk.py | 49 failed, 15 skipped |
| test_prefill_attention.py | 29 failed, 13 skipped |
| **scoped-out total** | **78 failed, 28 skipped** (matches every prior session exactly) |

Scoped-out failures confirmed still `AttributeError` (`topk_filtering` /
`prefill_with_paged_kv_cache` not built), not a wrong answer, with `-x` on
both files. Real device dispatch confirmed with `AMD_LOG_LEVEL=3` on
`test_decode_attention.py`: 216 `hipLaunchKernel ... Returned hipSuccess`
entries, including Quest's own compiled kernels by name
(`flashinfer::AppendPagedKVCacheDecodeKernel`,
`flashinfer::AppendPagedKVCachePrefillKernel`,
`flashinfer::BatchDecodeWithPagedKVCacheKernel`,
`flashinfer::VariableLengthMergeStatesKernel`) interleaved with PyTorch's own
kernels -- real execution, not a mock. No non-GPU regression to check: this
project has no CPU-only test path.

### CUDA no-regression gate

The delta `c1d7fff..ac7382c` touches only `kernels/include/decode/{decode_attn,decode_page}.cuh`
(the `std::max<size_t>` fix). The CUDA gate recorded at `c1d7fff` was
`cuda-not-validated`: RAFT (pinned branch-24.02) requires the legacy
`CUDA::nvToolsExt` CMake target, which `FindCUDAToolkit` no longer creates
against a CUDA 12.8 toolkit, so `cmake` fails inside RAFT's own CMakeLists at
configure time, before Quest's own `.cu`/`.cuh` sources -- including the two
files this commit touches -- are ever processed by the compiler.

Attempted to reproduce the wall fresh at this head_sha for the record: no
`cuda-12.8` conda env existed on this host, so created one per the standing
instruction (`conda create -y -n cuda-12.8 -c nvidia cuda-toolkit=12.8`,
~470MB+ of packages, several minutes). `gcc-11`/`g++-11` (the project's
pinned CUDA-leg host compiler) were not available from this host's apt
sources (`apt-get update` itself did not complete within the tool timeout),
which is a separate environmental gap from the RAFT wall and not worth
grinding on for a compile-only, secondary, once-per-head_sha gate whose
outcome this specific delta cannot change: the RAFT CMakeLists failure is
strictly upstream, in dependency-resolution order, of the two files this
commit edits, so nothing in `ac7382c` can alter whether or where that
configure-time failure occurs. Stopped pursuing a second full reproduction
after the toolchain-setup steps alone exceeded the gate's ~15 minute budget.

**`cuda-not-validated: unchanged from c1d7fff -- RAFT (pinned branch-24.02)
requires the legacy CUDA::nvToolsExt CMake target, which FindCUDAToolkit no
longer creates against a CUDA 12.8 toolkit; the configure fails inside RAFT's
own CMakeLists before any Quest source, including the two files
ac7382c touches, is ever reached. This delta cannot change that outcome.`**
Not a gate; carries no penalty per the gate's own rules.

### Jargon and documentation

`python3 utils/jargon.py --port Quest`: `jargon: clean`.

README.md's "Building on AMD GPUs (ROCm)" section is unchanged from the prior
validation and still accurately describes the build, the architecture
precedence, and the scoped-out ops. No inaccuracy found; nothing sent back.

`git -C projects/Quest/src status --porcelain`: clean except the untracked
`quest/ops/build-head/` build directory. `HEAD` at `ac7382c`, matching
`status.json.head_sha`.

State: `revalidate` -> `completed` on linux-gfx90a, `validated_sha = ac7382c`.
wave64 gate remains satisfied by this arch at the new head. windows remains
open (still `validation-failed` at `ac7382c`, the decode-attention
wrong-answer fault on gfx1151 recorded above -- an unruled deferral, not
something this arch's evidence resolves). CUDA gate: `cuda-not-validated`,
unchanged reasoning, no penalty. Wall clock: build ~1 min, in-scope tests
~10s, scoped-out and log-evidence re-runs a few minutes, CUDA toolchain setup
the majority of the session; overall within the ~60 minute budget.

## Validation 2026-08-13 (linux-gfx90a, revalidate after the split-path fix, 9be60fc)

Re-validation of fork sha `9be60fc` on this arch after `head_sha` advanced
from `ac7382c` (this platform's prior `validated_sha`, validated a few hours
earlier the same day) via the porter's commit `9be60fc`, which makes the
sparse decode kernel always take the split launch shape (the fix for the
single-block-path wrong-answer fault the windows-gfx1151 session found and
recorded above). MI250X (gfx90a, wave64), 4 GPUs visible and idle,
`HIP_VISIBLE_DEVICES=1`. ROCm 7.14 (`hip 7.14.60850`, TheRock-style pip
install under `/opt/conda/envs/py_3.12`, not `/opt/rocm`), PyTorch
2.14.0a0+git7d05abc, python 3.12.13, `transformers==4.37.2`
`tokenizers==0.15.2` already present in the environment from the prior
session. `git fetch origin` on the existing `projects/Quest/src` checkout,
fast-forwarded `moat-port` from `ac7382c` to `9be60fc` (`git diff` is exactly
the one file the commit message states: `kernels/include/decode/decode_attn.cuh`,
10 insertions/1 deletion -- a `constexpr bool kSingleBlockPathIsCorrect =
false;` guard on the existing condition). Submodules
`kernels/3rdparty/{flashinfer,pybind}` already initialized from the prior
session; no re-init needed.

### Classify: mixed, not arch-independent -- full GPU run, no carry-forward attempted

`python3 utils/moatlib.py classify Quest ac7382c 9be60fc` -> `class=mixed
arch_independent=False`, correctly: the change alters which launch shape a
kernel takes based on `batch_size * num_kv_heads` vs `blocks_per_cu *
cu_count`, a runtime/hardware-dependent decision, not a textual no-op. Went
straight to a full real-GPU run rather than attempting `codeobj_diff`
carry-forward -- MI250X has far more CUs than the 32-block threshold these
test shapes need (see the CU table in the windows-gfx1151 section above:
gfx1100 at 70 CUs and gfx942 at 304 CUs both already always took the split
path), so the OLD condition was already false on this arch for the tested
shapes and the compiled difference is a dead `constexpr false &&` term on
the taken path -- but that is a *behavioral* argument, not something
`codeobj_diff`'s device-code-object/symbol comparison would certify, and the
prior session's own submodule-clone flakiness in the second worktree made
the shortcut not worth chasing again for a run this cheap.

### Build

Wrapped `utils/timeit.sh Quest compile -- ninja -C quest/ops/build` (cleared
the leftover `build-head/` directory from the prior session's worktree
experiment first; built into a plain `build/` this time):

```
cd quest/ops && mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

Clean (6/6 objects, the same two pre-existing nodiscard-`hipError_t`
warnings on `decode_handler.cuh:165` and `approx_attn.cu:65` as every prior
session, no new warnings, no errors). `grep -o -- '--offload-arch=[a-z0-9]*'
build.ninja` -> `gfx90a` only. `import quest._kernels` exposes the same six
ops as every prior session.

### GPU tests

Wrapped `utils/timeit.sh Quest test -- python3 -m pytest ...`:

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| **in-scope total** | **121 passed in 8.44s** |
| test_topk.py | 49 failed, 15 skipped |
| test_prefill_attention.py | 29 failed, 13 skipped |
| **scoped-out total** | **78 failed, 28 skipped** (matches every prior session exactly) |

No behavior change from the pre-fix `ac7382c` run on this arch, as
predicted: this GPU never took the single-block path in these test shapes,
so the fix is a no-op here and the 121/9.43s-ish result is unchanged within
run-to-run noise. Confirmed the scoped-out failures are still `AttributeError`
(`topk_filtering` / `prefill_with_paged_kv_cache` not built), not a wrong
answer, with `-x` on both files. Confirmed real device dispatch with
`AMD_LOG_LEVEL=3 ... -s` (plain `-q` swallows the trace; `-s` is required to
see it) on `test_decode_attention.py`: 432 `hipLaunchKernel` lines, including
Quest's own compiled kernels by name (`flashinfer::AppendPagedKVCacheDecodeKernel`,
`flashinfer::AppendPagedKVCachePrefillKernel`,
`flashinfer::BatchDecodeWithPagedKVCacheKernel`,
`flashinfer::VariableLengthMergeStatesKernel`) interleaved with PyTorch's own
kernels. No non-GPU regression to check: this project has no CPU-only test
path.

### CUDA no-regression gate: re-attempted fresh at this head_sha, same environmental wall

Not owed automatically -- the CUDA gate was last recorded at `ac7382c`
(reasoned, not freshly reproduced there either, per that session's note),
and this is a new head_sha, so it was attempted fresh here rather than
carried forward on reasoning alone. `/opt/conda/envs/cuda-12.8` already
existed (nvcc 12.8.93) from the prior session. `gcc-11`/`g++-11` (the
project's pinned CUDA-leg host compiler) were missing, and unlike the prior
session's apt failure, `sudo apt-get update` and `sudo apt-get install -y
gcc-11 g++-11` both worked cleanly this time (~1 min).

Configure, throwaway build dir, arch pinned per the standing instruction
(`-DCMAKE_CUDA_ARCHITECTURES=80`, no NVIDIA GPU on this host to autodetect
against), wrapped `utils/timeit.sh Quest cuda-compile -- cmake ...`:

```
cmake -GNinja -DUSE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc \
  -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') \
  -S quest/ops -B quest/ops/build-cuda
```

`enable_language(CUDA)` succeeded with the conda nvcc and the pinned arch
survived to the cache untouched, confirming the ordering fix (item 3, from
the review rounds) still holds at this head. Configure then failed in the
exact same place as every prior session, before any of Quest's own `.cu`
files or the `pybind11_add_module(_kernels ...)` target were reached:

```
CMake Error at build-cuda/_deps/raft-src/cpp/CMakeLists.txt:262 (target_link_libraries):
  The link interface of target "raft" contains:
    CUDA::nvToolsExt
  but the target was not found.
```

Same root cause as recorded at `c1d7fff` and `ac7382c`: RAFT pinned to
`branch-24.02` links the legacy `CUDA::nvToolsExt` imported target, which
modern `FindCUDAToolkit` (3.31) no longer creates against a CUDA 12.8
toolkit layout. This confirms directly, rather than by dependency-order
reasoning alone, that the `9be60fc` delta (confined to
`kernels/include/decode/decode_attn.cuh`, never reached before this
configure-time failure) does not change the CUDA gate's outcome.

**`cuda-not-validated: unchanged from ac7382c/c1d7fff -- RAFT (pinned
branch-24.02) requires the legacy CUDA::nvToolsExt CMake target, which
FindCUDAToolkit no longer creates against a CUDA 12.8 toolkit; the configure
fails inside RAFT's own CMakeLists before any Quest source, including the
file 9be60fc touches, is ever reached. Reproduced fresh at this head_sha,
not just reasoned.`** Not a gate; carries no penalty per the gate's own
rules. Throwaway `quest/ops/build-cuda` directory removed before completion;
`gcc-11`/`g++-11` left installed at the host level (apt package, not part of
the fork tree).

### Jargon and documentation

`python3 utils/jargon.py --port Quest`: `jargon: clean`.

README.md's "Building on AMD GPUs (ROCm)" section is unchanged from the
prior validation (the delta touched only a `.cuh` header, no build or doc
files) and still accurately describes the build, the architecture
precedence, and the scoped-out ops. No inaccuracy found; nothing sent back.

`git -C projects/Quest/src status --porcelain`: clean. `HEAD` at `9be60fc`,
matching `status.json.head_sha`.

State: `revalidate` -> `completed` on linux-gfx90a, `validated_sha = 9be60fc`.
wave64 gate remains satisfied by this arch at the new head. windows is now
`completed` at this same head_sha (the porter's fix this arch just
revalidated is the same commit windows validated its own fix against).
CUDA gate: `cuda-not-validated`, reproduced fresh this round (not just
carried by reasoning), no penalty. Wall clock: build ~1 min, in-scope tests
~10s, scoped-out/log-evidence re-runs a few minutes, CUDA toolchain
(apt install + configure) the majority of the session; overall within the
~60 minute budget.

## Validation 2026-08-19 (linux-gfx1100, revalidate after the split-path fix, 9be60fc)

Re-validation of fork sha `9be60fc` on this arch after `head_sha` advanced from
`c1d7fff` (this platform's prior `validated_sha`, from the original 2026-08-08
session). 4x Radeon Pro W7800 (gfx1100, wave32) visible and idle, no other KFD
processes running (`rocm-smi --showpids`), `HIP_VISIBLE_DEVICES=1`. ROCm
`hip 7.2.53211`, PyTorch 2.14.0a0+gitb81488e, python 3.12.13,
`transformers==4.37.2` `tokenizers==0.15.2` already present from a prior
session on this host. `git fetch origin` + `git merge --ff-only origin/moat-port`
on the existing `projects/Quest/src` checkout, `c1d7fff` -> `9be60fc`
(`git log --oneline c1d7fff..9be60fc` shows exactly the two commits recorded
above, `ac7382c` then `9be60fc`; `git diff --stat` touches only
`kernels/include/decode/{decode_attn,decode_page}.cuh`). Submodules
`kernels/3rdparty/{flashinfer,pybind}` already initialized from the prior
session.

Went straight to a full real-GPU run rather than `classify`+`codeobj_diff`:
this platform's `validated_sha` predates BOTH commits in the delta (unlike
the gfx90a sessions, which revalidated one commit at a time), and the
gfx90a session already established that a full run at this delta is cheap
(build ~1 min, tests ~10s) and that the split-path fix is behaviorally a
no-op on high-CU parts (gfx1100 at 70 CUs already always took the split
path pre-fix, per the CU table in the windows-gfx1151 section above) --
predicting the same outcome here, worth confirming on GPU rather than
asserting.

### Host environment hazard, unrelated to the port: broken `find_package(Torch)` cmake export on this shared host

`$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')` (the
README's own recipe, used cleanly by every prior session on this host)
resolved to `/var/lib/jenkins/pytorch/torch/share/cmake`, which does not
exist: the installed `torch` on this host is an editable
(scikit-build-core) checkout at `/var/lib/jenkins/pytorch`, and `torch.__file__`
now resolves into that source tree rather than a proper site-packages
install (`_editable_skbc_torch.pth`/`.py` reroute `import torch` there). The
source tree's own `build/` directory is a raw ninja build output, not a
`cmake --install` tree: its `TorchConfig.cmake`/`Caffe2Targets.cmake`
compute `_IMPORT_PREFIX`/`TORCH_INSTALL_PREFIX` by walking a fixed number of
parent directories up from the `.cmake` file's own location, which is only
correct for the specific nested path CMake originally exported it to
(`build/CMakeFiles/Export/<hash>/Caffe2Targets.cmake`, `_IMPORT_PREFIX`
walks 4 levels to reach `build/`); pointing `CMAKE_PREFIX_PATH` straight at
`build/` breaks that assumption and manifests as `TorchConfig.cmake` handing
`find_package(Torch)` a nonexistent `INTERFACE_INCLUDE_DIRECTORIES` path
(observed: `/var/lib/include`, then `/var/lib/jenkins/pytorch/build/include/torch/csrc/api/include`,
neither of which exists -- `build/include` on this host holds only `sleef.h`,
nothing was ever installed there). This is unrelated to Quest, to `USE_HIP`,
and to anything this port's CMakeLists touches -- it is purely this host's
PyTorch dev tree missing its install step, so `torch.utils.cmake_prefix_path`
lies about where a real Torch cmake package lives.

Diagnostic method that generalizes: when `find_package(Torch)` fails or
resolves to nonexistent include/lib paths, check whether `torch.__file__`
is inside an editable/source checkout (`_editable_*` marker files in
`site-packages`) rather than a normal wheel install, and whether that same
`site-packages` directory ALSO contains a complete separate `torch/` payload
(headers, libs, `share/cmake`) alongside the editable shim -- a from-wheel
install and an editable dev pointer can coexist in one environment, and the
complete one is usually the one to build against for a downstream C++
extension. Here it did:
`/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake`
carries a real `TorchConfig.cmake` with real `include/`, `lib/*.so` (15
libraries) beside it, untouched by the editable-import shadowing (which
only affects the *Python* `import torch` path, not the directory's
contents). Used
`-DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake`
instead of the README's literal recipe; configure then succeeded cleanly
(`-- Found Torch: .../site-packages/torch/lib/libtorch.so`, only the
pre-existing unrelated `libhipcxx not found` optional-package warning and a
`kineto not found` warning, both present in every prior session's log too).
Tried two dead ends first that are worth recording so a future session does
not repeat them: (1) copying `Caffe2Targets.cmake` up one directory level
into `build/` does not work -- the `_IMPORT_PREFIX` walk-count is baked in
per export location, so it silently computes a *different* wrong prefix
(`/var/lib` instead of `/var/lib/jenkins/pytorch/build`) rather than the
right one; (2) setting `TORCH_INSTALL_PREFIX` in the environment (which
`TorchConfig.cmake` does honor, `if(DEFINED ENV{TORCH_INSTALL_PREFIX})`)
fixes `TorchConfig.cmake`'s own prefix walk but not `Caffe2Targets.cmake`'s
independent one, so `Found Torch` succeeds while the include-path error
persists one step later at generate time -- both cmake files need the same
prefix but derive it independently, so patch/point at the right layout
directly rather than chasing each symptom. No fork file and no MOAT-tracked
file needed to change; the one Quest source edit made along the way
(temporarily repointing `Caffe2Config.cmake`'s include of
`Caffe2Targets.cmake` at the nested export path, to test hypothesis (1))
was reverted before building Quest, and is not part of the fork tree in any
case (`/var/lib/jenkins/pytorch/build/Caffe2Config.cmake` is host-local
build output, not git-tracked by this port). `git -C
/var/lib/jenkins/pytorch status --porcelain` shows only pre-existing
modifications unrelated to this session (RNN.cpp, submodule pointers,
`.ci/pytorch/build.sh`), confirming nothing here touched pytorch's tracked
source.

### Build

Wrapped `utils/timeit.sh Quest compile -- ninja -C
/var/lib/jenkins/moat-worktrees/Quest/projects/Quest/src/quest/ops/build`
(absolute path -- `timeit.sh` cds to the MOAT repo root first):

```
cd quest/ops && rm -rf build && mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

Clean (6/6 objects, the same two pre-existing nodiscard-`hipError_t`
warnings on `decode_handler.cuh:165` and `approx_attn.cu:65` as every prior
session, no new warnings, no errors). `grep -o -- '--offload-arch=[a-z0-9]*'
build/build.ninja` -> `gfx1100` only, reconfirming the architecture-selection
fix still holds. `import quest._kernels` exposes the same six ops as every
prior session.

### GPU tests

Wrapped `utils/timeit.sh Quest test -- python3 -m pytest ...`:

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| **in-scope total** | **121 passed in 4.54s** |
| test_topk.py | `AttributeError: module 'quest._kernels' has no attribute 'topk_filtering'` (`-x`) |
| test_prefill_attention.py | `AttributeError: module 'quest._kernels' has no attribute 'prefill_with_paged_kv_cache'` (`-x`) |

Matches every prior gfx1100/gfx90a session exactly; the split-path fix is
behaviorally a no-op here as predicted (this GPU's 70 CUs already always
took the split path pre-fix). Confirmed real device dispatch with
`AMD_LOG_LEVEL=3 ... -s` on `test_decode_attention.py`: 432
`hipLaunchKernel` lines, 3921 `hipSuccess` occurrences, and Quest's own
compiled kernels present by name (`flashinfer::AppendPagedKVCacheDecodeKernel`,
`flashinfer::AppendPagedKVCachePrefillKernel`,
`flashinfer::BatchDecodeWithPagedKVCacheKernel`,
`flashinfer::VariableLengthMergeStatesKernel`) -- real execution, not a
mock. No non-GPU regression to check: this project has no CPU-only test
path.

### CUDA no-regression gate

Not repeated: already recorded at this exact head_sha (`9be60fc`) by the
linux-gfx90a validation above (`cuda-not-validated`, reproduced fresh there
this same head_sha -- RAFT branch-24.02's `CUDA::nvToolsExt` wall). Runs
once per head_sha per the standing rule.

### Jargon and documentation

`python3 utils/jargon.py --port Quest`: needed a local `main` branch in the
fork checkout first (`git branch main origin/main`; the checkout only had
`origin/main`, and the tool resolves `main..moat-port` locally) -- then
`jargon: clean`.

README.md's "Building on AMD GPUs (ROCm)" section is unchanged from the
prior validation (the delta touches only two `.cuh` files, no build or doc
files) and still accurately describes the build, the architecture
precedence (`CMAKE_HIP_ARCHITECTURES` / `PYTORCH_ROCM_ARCH` / build-machine
default), and the scoped-out ops. No inaccuracy found; nothing sent back.

`git -C projects/Quest/src status --porcelain`: clean. `HEAD` at `9be60fc`,
matching `status.json.head_sha`.

State: `revalidate` -> `completed` on linux-gfx1100, `validated_sha = 9be60fc`.
wave32 gate remains satisfied by this arch at the new head. CUDA gate:
`cuda-not-validated`, already recorded at this head_sha by linux-gfx90a, no
penalty, not re-run. Wall clock: build ~1 min, in-scope tests ~5s,
scoped-out/log-evidence re-runs a few minutes, the host cmake-environment
diagnosis the majority of the session; overall within the ~60 minute
budget.

## The single-block decode path: the second defect, root-caused 2026-08-19

`quest-decode-single-block-path-broken` recorded that correcting `cur_last_page_len`
was not sufficient and that "the path has at least one further defect", without
naming it. It is named now, and it is structural rather than a second typo.

### The design the kernel actually encodes

Quest's `paged_kv_t` carries `last_page_len` as a SCALAR (`decode_page.cuh:91`), not
the per-request array upstream FlashInfer uses
(`3rdparty/flashinfer/include/flashinfer/decode.cuh:515`). The caller strips the
trailing partial page -- `kv_indices_without_last = kv_cache.indicies[:-1]`
(`quest/utils/controller.py:106`) -- and that page is re-attached as an EXTRA CHUNK by
the partition step:

```c++
// Manually append information of last page
chunk_indptr_h.back() += 1;
new_page_indptr_h.push_back( new_page_indptr_h.back() + 1 );
batch_idx_map_h.push_back( 0 );
```

So in the split shape the final block IS the appended trailing page, and every
`blockIdx.x == gridDim.x - 1` test in the kernel means "this block is that appended
page". Both such tests are correct there.

### Why the single-block shape cannot work as written

`tmp == nullptr` launches with the CALLER's `paged_kv`, whose indptr is
`kv_indptr_for_approx_decode = [0, inference_page_budget - 1]`. The partition step
never runs, so the appended page is simply not in the block's range, and:

1. `decode_attn.cuh:469` -- `cur_last_page_len` picks `last_page_len`, which describes
   a page that is not in the list. This is the defect already recorded.
2. `decode_page.cuh:331` -- `protective_get_k_ptr_heads` redirects the read to
   `last_page_idx` whenever `blockIdx.x == gridDim.x - 1`. With `gridDim.x == 1` that
   is block 0, so EVERY page access in the whole range resolves to the trailing page
   and `indices` is never consulted.

(2) is the missing piece. It is why fixing (1) alone changed nothing: the chunk length
became correct while every data pointer still came from one wrong page.

### What a repair costs

Both tests can be made shape-independent, and both reduce to current behaviour in the
split shape:

- `cur_last_page_len`: test `cur_page_indptr_end == last_indptr` rather than the block
  index -- true only for the chunk that ends at the global end.
- the accessor: test `page_iter == last_indptr - 1` rather than the block index -- in
  the split shape only the appended chunk contains that page, so it is equivalent.

But that is not enough on its own. The single block must also COVER the appended page,
which means `BeginForward` has to allocate an index buffer and publish an extended
indptr even when `tmp_size == 0`, and the dispatch has to install it when
`tmp == nullptr`. That is four coordinated host and device changes.

### Scope

Nothing here is HIP-specific. `paged_kv_t`, the scalar `last_page_len`, the append and
both `gridDim.x - 1` tests are upstream Quest code, identical on the CUDA path, and the
selecting condition `batch_size * num_kv_heads >= blocks_per_cu * cu_count` is reachable
on a low-SM NVIDIA card too. A repair is an upstream fix, not a ROCm one.

## Repair 2026-08-19 (windows-gfx1151): single-block decode path fixed and re-enabled

Ruled `now` by Jeff Daily, and ruled to land on this branch rather than as a separate
upstream contribution, with the scope caveat below recorded rather than acted on.
`cbc3890`, on top of `9be60fc`, which it supersedes: the path is repaired instead of
disabled, so `kSingleBlockPathIsCorrect` is gone.

### The fix

Three changes, following the root-cause section above:

1. `decode_attn.cuh` -- the chunk length asks `cur_page_indptr_end == last_indptr`
   instead of `batch_idx == gridDim.x - 1`.
2. `decode_page.cuh` -- `protective_get_k_ptr_heads` asks `page_iter == last_indptr - 1`
   instead of the block index. This was the unnamed second defect.
3. `decode_handler.cuh` -- `BeginForward` publishes an indptr one page longer when it
   does not partition, and the dispatch installs it, so the single block actually spans
   the appended trailing page. Without this the first two changes are inert, because the
   page is simply not in the block's range.

(1) and (2) pick out the same block and page as before whenever the range is
partitioned, so the split shape is untouched by construction.

### Evidence

The launch-shape decision was instrumented to prove the repaired path is the one
running here rather than inferring it from a green suite:

```
QUESTPATH bs=1 kvh=32 max_grid=20 -> SINGLE
```

32 >= 20, so gfx1151's 20 CUs select the single-block shape, as the CU table above
predicts. The instrumentation was removed before committing.

| run | result |
|---|---|
| four in-scope suites, single-block shape (what this host takes) | **121 passed** |
| same suites with the split shape forced, to check (1) and (2) are no-ops there | **121 passed** |
| `kv_len` sweep 17..1541 | peak abs diff **2.4e-4**, std ratio **1.000** at every length, tolerance 5e-3 |

The forced-split run matters because gfx1100 and gfx90a take that shape: it is direct
evidence here that the two retargeted tests do not disturb them, rather than a claim
for them to discover.

### Scope caveat, recorded and not acted on

Nothing in this change is HIP-specific. `paged_kv_t`, the scalar `last_page_len`, the
manual append and both `gridDim.x - 1` tests are upstream code that the CUDA path
compiles identically, and the selecting condition is reachable on a low-SM NVIDIA card.
Landing it here means the branch carries an upstream fix that changes NVIDIA behaviour.
That was put to Jeff Daily with the alternative of a separate branch cut from upstream
master, and landing it here was the decision. It follows that
`quest-single-block-disable-changes-cuda` is answered by repair rather than by a HIP
guard: there is no disable left to guard, and the path is now correct on both backends.

The CUDA no-regression gate is still owed at this head and is not available on this
host (no CUDA toolkit); it is recorded `cuda-not-validated` at the previous head for a
RAFT branch-24.02 `CUDA::nvToolsExt` wall, which is a host limitation rather than a
property of this change.

### Integrity

`git -C projects/Quest/src status --porcelain` clean at `cbc3890`; the only fork change
this round is that commit.

## Review 2026-08-19 (windows-gfx1151, `moat-port` cbc3890 vs upstream main) -- CHANGES REQUESTED

Problems only. Reviewed the whole branch, not the newest commit. Conflict of interest
declared: this session also wrote `cbc3890`, so findings 1 and 5 are self-reported and a
second reviewer on another host is worth having before publication.

1. **`decode_handler.cuh` -- `int_buffer_ != nullptr` no longer means "partitioned".**
   `GetChunkIndPtr` and `GetBatchIdxMap` use exactly that as their nullptr sentinel and
   index past the end of the `batch_size + 1` buffer `cbc3890` allocates on the
   non-partition path. Not a live bug: every caller is already inside a `tmp`/`tmp_size`
   guard, so neither runs when there is no partition. But the sentinel is now untrue, and
   the next caller to trust it gets a valid-looking pointer into a two-element
   allocation. Fix: allocate the chunk/map regions too, or gate those accessors on
   something that still means partitioned.

2. **`hip_compat/cuda_fp16.h` -- `#undef __HIP_NO_HALF_CONVERSIONS__` is an ODR hazard.**
   In this SDK the macro guards MEMBER functions of `__half` (`amd_hip_fp16.h:87,101,137`
   -- converting constructors and `operator=`), so a translation unit reaching this shim
   sees a different `__half` from one that only pulls torch's headers. It is also
   include-order dependent: a TU that reached `hip/hip_fp16.h` through torch first would
   make the `#undef` a no-op behind the include guard. Verified load-bearing -- removing
   both `#undef`s gives 36 errors at `decode_page.cuh:436,527` ("invalid argument type
   '__half' to unary expression") and `decode_attn.cuh:164` ("no matching conversion for
   static_cast from 'float' to '__half'"). The alternative is explicit
   `__half2float`/`__float2half` at those sites, as `rms_norm.cu` already does, trading
   footprint in vendored files for the hazard. At minimum the shim must state that it has
   to precede any torch header.

3. **No test can select the single-block launch shape.** It is chosen by
   `batch_size * num_kv_heads >= blocks_per_cu * cu_count`: 32 >= 20 on gfx1151, but
   32 >= 70 (gfx1100) and 32 >= 304 (gfx942) are false. So the path repaired in `cbc3890`
   is exercised on exactly one platform in the fleet and a regression is invisible to the
   other two and to any NVIDIA CI. The split shape was checked here only by editing the
   condition and rebuilding, which no test run reproduces. Wanted: a test-visible override
   of the launch shape.

4. **The CUDA path's CMake was restructured and never compiled.** `project()` moved from
   `LANGUAGES CUDA CXX` to `LANGUAGES CXX` plus a conditional `enable_language(CUDA)`, and
   `fetch_rapids.cmake`, the `rapids-*` includes, `rapids_cpm_init()` and `get_raft.cmake`
   moved from before `project()` to after it -- all on the NVIDIA build. The usual
   breakage vector is absent (`rapids_cuda_init_architectures()` is never called, and it is
   the one entry point that must precede `project()`), so this is expected to be fine, but
   the CUDA gate is `cuda-not-validated` against a pre-existing RAFT branch-24.02
   `CUDA::nvToolsExt` wall, so nobody has configured it since the rewrite.

5. **`cbc3890` changes NVIDIA behaviour**, per the scope caveat recorded above. Raised here
   because an upstream reviewer will ask, and because it turns finding 4 from "owed" into
   load-bearing.

6. **`c1d7fff` carries no AI-assistance disclosure.** The other three commits do, and it is
   the commit the branch is built on.

Checked and clear, recorded so a later round does not redo them: the width-32 shuffle
grouping in `rms_norm.cu` against `tid>>5`/`tid&0x1f`/`shared[NUM][33]`; the decode
reductions (`offset = bdx/2 .. 1` with `static_assert(bdx <= 32)` and `tx` fastest-varying,
so the butterfly stays in an aligned sub-group on wave64); the `hip_compat/flashinfer/math.cuh`
symbol set against upstream (identical); `cp_async`'s non-PTX fallback being a synchronous
`uint4` store with `block.sync()` present in the main loop, which is why the no-op
`commit_group`/`wait_group` and the empty `cuda/pipeline` shim are safe; `protective_get_k_ptr_heads`
having no user outside the sparse decode kernel, so `prefill.cuh` and the CUDA prefill path
are untouched by the accessor change; and V pointers being derived as
`k_ptrs[j] + kv_offset_delta()`, so the K fix covers V.

## Porter round 2026-08-19 (windows-gfx1151): review findings addressed -- `f50e33a`

Taken here rather than handed to another host; the conflict of interest declared in the
review above still stands and is not resolved by this round.

**Finding 1 (int_buffer_ sentinel) -- fixed.** `GetChunkIndPtr` and `GetBatchIdxMap` now
require `float_buffer_` as well. That is the temporary the partitioned kernel merges
through and only the partitioning branch allocates it, so it still means what
`int_buffer_` used to mean before the un-partitioned branch started allocating one too.

**Finding 2 (`__HIP_NO_HALF_CONVERSIONS__`) -- the ODR claim was WRONG; corrected and
documented instead.** The review asserted that translation units disagree on `__half`.
They do not. The macros are on the command line for all five sources
(`grep __HIP_NO_HALF_CONVERSIONS__ quest/ops/build/build.ninja` -> 5), and
`torch/extension.h` is reached first in every one of them via `bsk_ops.h`. If torch pulled
`hip/hip_fp16.h` at that point, the shim's undefs would be inert behind the include guard
and the build would fail -- and when the undefs were removed as an experiment the 36
errors landed in `decode_page.cuh` and `decode_attn.cuh`, headers that ALL five sources
include, which shows the shim is what first pulls `hip_fp16.h` in every one of them. So
the units agree and there is no ODR violation; `__half` also does not cross the libtorch
boundary, whose public surface is `at::Half`. What is real is the include-order
dependency, now written into the shim along with the loud failure mode and the remedy.

**Finding 3 (neither launch shape testable everywhere) -- fixed.**
`QUEST_DECODE_LAUNCH_SHAPE=single|split` overrides the occupancy decision; unset, nothing
changes. This is what closes the coverage gap for gfx1100 and gfx942, which cannot reach
the single-block shape by occupancy (32 >= 70 and 32 >= 304 are false) and therefore never
exercised the `cbc3890` repair.

**Finding 4 (CUDA CMake restructured, never compiled) -- not addressable here.** No CUDA
toolchain on this host, and the gate is blocked by a pre-existing RAFT branch-24.02
`CUDA::nvToolsExt` incompatibility rather than by anything the port did. Still owed.

**Finding 5 (`cbc3890` changes NVIDIA behaviour) -- no action, recorded decision.**

**Finding 6 (`c1d7fff` has no AI-assistance disclosure) -- WITHDRAWN. The commit does
disclose; the finding was a false positive, and so was the deferral raised from it.**

`c1d7fff` line 50 reads "This work was authored with the assistance of Claude, an AI
assistant." The review checked with `grep -ci "AI coding agent"` -- one phrasing out of at
least ten in use across the fleet -- and read its absence as a missing disclosure.
Everything built on that is void with it: the claim that the PR-prep squash resolved the
finding, and the correction that nothing enforces the squash. That second observation is
true on its own and is recorded in `moat-no-gate-on-commit-message-rules`; it simply never
applied to this commit.

Found by building the gate that would have caught a real instance:
`python3 utils/moatlib.py audit-commits` matches the naming rather than one sentence shape
and reports this branch clean.

### Evidence

```
for shape in "" single split; do
  QUEST_DECODE_LAUNCH_SHAPE=$shape python -m pytest quest/tests/test_rope.py \
    quest/tests/test_estimate.py quest/tests/test_decode_attention.py \
    quest/tests/test_approx_attention.py -q
done
```

| configuration | result |
|---|---|
| default (occupancy picks single-block here) | **121 passed** |
| `QUEST_DECODE_LAUNCH_SHAPE=single` | **121 passed** |
| `QUEST_DECODE_LAUNCH_SHAPE=split` | **121 passed** |
| `kv_len` sweep 17..1541, split forced | peak abs diff **2.4e-4**, std ratio **1.000** |

Integrity: `git -C projects/Quest/src status --porcelain` clean at `f50e33a`.

## Review 2026-08-20 (linux-gfx1100, delta `cbc3890..f50e33a`) -- CHANGES REQUESTED

Problems only. Scope: the round's delta plus its interaction with the whole branch.
Reviewed on a different host from the one that wrote `cbc3890`/`f50e33a`, so the conflict
of interest declared in the 2026-08-19 review does not apply here. `pr-state Quest` is
`none` and `status.json` has no `fix` block: this is the pre-PR branch, not a fix round.

Re-verified and holding, so a later round does not redo them: `float_buffer_` really is a
sound "partitioned" sentinel (`decode_handler.cuh:110` is the only assignment, ctor
`:192` and `EndForward:161` null it), so finding 1 of the previous round is closed;
`protective_get_v_ptr` has no caller (V comes from `k_ptrs[j] + kv_offset_delta()` at
`decode_attn.cuh:545,622`), so the accessor retarget covers V; `protective_get_k_ptr_heads`
still bounds the `indices` read at `page_iter < last_indptr - 1` (`decode_page.cuh:344`)
against a `[num_kv_heads, page_budget]` array, so the extended indptr adds no out-of-bounds
neighbour read; the estimate kernel uses the untouched `protective_get_k_ptr`
(`decode_attn.cuh:301,344`) so `MaxPossibleSample` is unaffected; only the appended chunk
can satisfy `cur_page_indptr_end == last_indptr` with a non-zero length, so the split shape
is unchanged by construction; the `cp_async` fallback really is a synchronous `uint4` store
(`kernels/3rdparty/flashinfer/include/flashinfer/cp_async.cuh:80,128`) with a `block.sync()`
between every smem read and the write that reuses the stage, so the no-op
`commit_group`/`wait_group` remain safe; `jargon.py --port Quest` is clean; all five titles
are `[ROCm]`-prefixed and under 72 characters with no agent `Co-Authored-By` trailer.

### 1. BLOCKING -- the launch-shape override has three silent no-op modes, so its own evidence cannot distinguish "both shapes ran" from "the override was ignored"

`kernels/include/decode/decode_attn.cuh:871-880`. This is the whole point of the round, and
nothing in the built artifact confirms it took effect.

- `:876` -- an unrecognised value returns `0` and falls through to the occupancy decision.
  `QUEST_DECODE_LAUNCH_SHAPE=Single`, `=SPLIT` or a typo is silently no override, and the
  suite is green either way.
- `:871` -- `static const int forced_shape` latches the first `getenv` per template
  instantiation. Setting the variable in-process (the natural `monkeypatch.setenv` fixture
  that "test-visible override" invites) reads the latched value and silently exercises the
  wrong shape. This is a host setup path called once per `begin_forward`, so a plain
  `std::getenv` per call costs nothing.
- `:885-911` -- with `split` forced, the existing `if(new_batch_size == batch_size)
  { tmp_size = 0; }` fallback at `:905` reverts to the single-block shape. It cannot fire
  at the current `batch_size == 1` (`approx_attn.cu:33,78`, so `new_batch_size >= 2`), but
  it is an unchecked path out of the requested shape.

Nothing observable reports which shape ran. The pybind surface is `begin_forward`,
`end_forward`, `forward` only (`bsk_ops.cu:18-23`, `bsk_ops.h:98-125`) -- no accessor for
`tmp_size`, `float_buffer_` or the resolved shape -- and the `QUESTPATH` instrumentation
that proved it for `cbc3890` was removed before committing. So the "121 passed in each of
the three configurations" table (notes above, and f50e33a's Test Plan) is consistent with
the override having done nothing in two of the three runs, and the same will be true of
the gfx1100 and gfx90a runs that are the reason the override exists.

Minimal fix, no new public API: in `BatchDecodeWithPagedKVCacheWorkEstimation`, hard-error
on an unrecognised `QUEST_DECODE_LAUNCH_SHAPE` value, and hard-error when `split` was
requested but `tmp_size` ends at `0`. Then a green three-configuration run is itself the
proof. Drop the `static` while there. If you would rather assert positively from Python, a
`partitioned()` getter on `BatchDecodeWithPagedKVCachePyTorchWrapper` returning
`handler_.GetTempFloatBuffer<void>() != nullptr` works too, but it grows the upstream
surface.

### 2. The override is exercised by nothing in the tree and documented nowhere

`grep -rn QUEST_DECODE_LAUNCH_SHAPE` over the fork returns exactly two lines, both inside
`decode_attn.cuh` (`:868` comment, `:872` the read). No test uses it, `README.md`'s new
ROCm section (`README.md:57-71`) does not mention it, and `plan.md`'s test plan
(`plan.md:96-102`) does not either. So the coverage it buys depends on a validator reading
`f50e33a`'s commit body, and an upstream maintainer sees an undocumented `getenv` added to
a vendored FlashInfer header by a `[ROCm]` commit, with no consumer. Either drive it from
the test suite (a pytest that re-execs the decode/approx tests once per shape as
subprocesses, which also sidesteps the latching in finding 1) or document it in the README
ROCm section as a test/debug knob. Note it is not HIP-guarded, so it is a new runtime knob
on the NVIDIA build as well; that is inert when unset, but it belongs in the same
"changes NVIDIA behaviour" conversation as the recorded scope caveat.

### 3. Recorded, not this round's to fix

- ~~`c1d7fff` carries no AI-assistance disclosure.~~ RETRACTED, same day, before anyone
  acted on it. Raised here off `grep -ci "AI coding agent"` -- the same too-narrow probe
  that produced the original false positive one round earlier. `c1d7fff` does disclose, in
  different words: "This work was authored with the assistance of Claude, an AI assistant."
  The deferral `quest-c1d7fff-missing-ai-disclosure` was already withdrawn in `41d1deb`,
  which landed from another host while this review was in progress. Two reviewers have now
  made the identical mistake with the identical command: check disclosure with
  `moatlib.py audit-commits`, never a grep for one phrasing.
- What survives that line of thought, independent of disclosure: `9be60fc "[ROCm] Always
  split the decode KV range across blocks"` is contradicted by `cbc3890`, which re-enables
  the single-block path. Publishing unsquashed puts a self-reverting series in front of the
  maintainer, so the PR-prep squash is worth doing on its own merits.
- The CUDA no-regression gate is still owed at this head. Independently checked here and
  the previous round's risk assessment stands: `rapids_cuda_init_architectures` appears
  nowhere under `quest/ops/cmake` and `fetch_rapids.cmake` only downloads `RAPIDS.cmake`,
  so moving the `rapids-*` includes after `project()` has no known breakage vector. No
  `nvcc` on this host either, so it stays owed rather than answered.
- All three platforms carry `validated_sha` `9be60fc`, and `cbc3890` changed unguarded
  shared decode code (`decode_attn.cuh:476`, `decode_page.cuh:337`,
  `decode_handler.cuh:123-150`). The gfx1100 and gfx90a evidence does not carry to
  `f50e33a`; both need a fresh run, and per finding 1 that run should cover both shapes.

## Porter round 2026-08-20 (linux-gfx1100): review findings 1-2 addressed -- `c9828e2`

Written on a different host from the one that wrote `cbc3890`/`f50e33a`, so this round
carries no conflict of interest with the code it is fixing. `pr-state Quest` is `none`, so
`moat-port` itself moved; no staging branch.

**Finding 1 (three silent no-op modes in the launch-shape override) -- fixed**, exactly the
minimal remedy the review specified, in `BatchDecodeWithPagedKVCacheWorkEstimation`
(`kernels/include/decode/decode_attn.cuh:864-931`):

- an unrecognized value now throws `std::invalid_argument` instead of returning 0;
- the `static const int forced_shape` lambda is gone -- `std::getenv` runs on every call,
  which costs nothing at a host-side setup site called once per `begin_forward`, and makes
  an in-process `setenv` take effect;
- `forced_shape < 0 && tmp_size == 0` after the partition search throws, closing the
  `if(new_batch_size == batch_size) { tmp_size = 0; }` exit from a requested `split`.

Empty is treated as unset alongside absent, and that is documented in both the header
comment and the README. This is deliberate: `env VAR= cmd` and CI systems that materialize
an unset variable as `""` should mean "no override", not a hard error, and the failure mode
the finding is about is a typo, not an empty string. The previous round's own evidence loop
(`for shape in "" single split`) depends on it too.

No new public API: no `partitioned()` getter, no pybind surface change. The throws reach
Python through pybind11 as `ValueError` at `quest/utils/decode_wrapper.py:43`.

**Finding 2 (override exercised by nothing, documented nowhere, not HIP-guarded) -- fixed,
both remedies rather than one.** The review offered "either drive it from the test suite or
document it in the README"; both were cheap and they answer different halves of the finding
(no in-tree consumer; an upstream maintainer meeting an undocumented `getenv`).

- NEW `quest/tests/test_decode_launch_shape.py` (44 lines): re-runs `test_decode_attention.py`
  and `test_approx_attention.py` once per shape as subprocesses with `QUEST_DECODE_LAUNCH_SHAPE`
  set, and asserts that a bad value fails with the new message. Subprocesses, not
  `monkeypatch.setenv`: it also keeps working if the caching in finding 1 ever comes back.
  It runs the two named files by path, so collecting the whole directory cannot recurse.
  `cwd=REPO_ROOT` so `import quest` resolves the same way the README's `python -m pytest`
  recipe does.
- README ROCm section (`README.md:73`) documents the knob in the section's own prose style.

**Not HIP-guarded, deliberately, now stated where the decision is visible** (README, commit
body, here) rather than left implicit. The gap the knob closes is a property of the device's
CU count, not of the platform -- an NVIDIA part with a high SM count never reaches the
single-block shape either -- so guarding it would remove the coverage from exactly the leg
that has no other way to get it. It is inert unless set. It does add a runtime knob to the
NVIDIA build, so it belongs in the same scope caveat as `cbc3890`, and the CUDA
no-regression gate is still owed at this head (finding 3, unchanged: no `nvcc` on this
host, and the RAFT branch-24.02 `CUDA::nvToolsExt` wall is still the blocker).

### Evidence (Radeon Pro W7800, gfx1100, wave32, `HIP_VISIBLE_DEVICES=1`)

Build: `utils/timeit.sh Quest compile -- ninja -C .../quest/ops/build` into the existing
configure from the 2026-08-19 session (`-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
-DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake`
-- the host's editable-torch `cmake_prefix_path` hazard recorded in that session still
applies). 6/6 objects, no new warnings: the same two pre-existing nodiscard-`hipError_t`
warnings at `decode_handler.cuh:197` and `approx_attn.cu:65`.

| configuration | in-scope suites + new test |
|---|---|
| default (occupancy picks split here, 70 CUs) | **124 passed** in 16.7s |
| `QUEST_DECODE_LAUNCH_SHAPE=single` | **124 passed** in 16.9s |
| `QUEST_DECODE_LAUNCH_SHAPE=split` | **124 passed** in 16.7s |
| `QUEST_DECODE_LAUNCH_SHAPE=` (empty) | 6 passed (decode only), override inert as documented |
| `test_decode_launch_shape.py` alone | 3 passed in 12.5s |

121 of those 124 are the unchanged baseline (`test_rope.py` 64, `test_estimate.py` 9,
`test_decode_attention.py` 6, `test_approx_attention.py` 42), matching every prior
gfx1100/gfx90a session; the 3 new ones are the launch-shape file. Out-of-scope suites
unchanged: `test_topk.py` and `test_prefill_attention.py` still stop at the missing
`topk_filtering` / `prefill_with_paged_kv_cache` attributes.

The override provably takes effect, which is what the finding actually asked for and does
not depend on the new hard-errors:

```
QUEST_DECODE_LAUNCH_SHAPE=$shape AMD_LOG_LEVEL=3 python3 -m pytest \
  quest/tests/test_decode_attention.py -q -s 2>&1 | grep -o "MergeStatesKernel"
```

| forced shape | `BatchDecodeWithPagedKVCacheKernel` | `MergeStatesKernel` |
|---|---|---|
| single | 6 | **0** |
| split | 6 | **6** |

Both hard-errors were observed, not just written:

- `QUEST_DECODE_LAUNCH_SHAPE=Single` -> `ValueError: QUEST_DECODE_LAUNCH_SHAPE must be
  "single" or "split", got "Single"`, 1 failed. The review's own example of a silent typo.
- the split guard cannot be reached at `batch_size == 1`, so it was checked by temporarily
  inverting its condition to `tmp_size != 0`, rebuilding, and running forced split ->
  `ValueError: QUEST_DECODE_LAUNCH_SHAPE=split cannot be honored: partitioning a batch of 1
  left the batch size unchanged, so this call runs the single-block shape`. Condition
  restored, rebuilt, and the whole table above re-run after the revert.

`python3 utils/jargon.py --port Quest` -> clean. `moatlib.py audit-commits Quest` -> OK
(all six titles `[ROCm]`, <= 72 chars, disclosure present, no agent `Co-Authored-By`).
`git -C projects/Quest/src status --porcelain` clean at `c9828e2`.

Lesson promoted to the `cuda-to-rocm` skill (`references/validation.md`, "A path only some
GPUs reach: force it, with an override that cannot be ignored"): the whole shape of this --
a device-property branch that leaves one path untested fleet-wide, the three ways a forcing
knob silently no-ops, and confirming the forced paths differ with `AMD_LOG_LEVEL=3` before
trusting the knob -- applies to any port with an occupancy- or wavefront-dependent branch.

State: `changes-requested` -> `porting` -> `ported`, `head_sha` `f50e33a` -> `c9828e2`. All
three platforms carried `validated_sha` `9be60fc` and now read `revalidate`; they needed a
fresh run at `f50e33a` already (per the review's last bullet) and this head adds shared
unguarded changes to `decode_attn.cuh`, so nothing was lost. gfx1100 evidence above is a
porter's build+test record, not a validation record.

## Review 2026-08-20 (linux-gfx1100, delta `f50e33a..c9828e2`, whole branch `origin/main...c9828e2`) -- PASS

Follow-up on the `changes-requested` round. Scope: the delta that answers the previous
review's findings, plus a re-read of the whole branch for anything the churn disturbed.
`pr-state Quest` is `none` and `status.json` carries no `fix` block, so this is still the
pre-PR branch and `moat-port` itself is the working branch. Six commits, `origin/moat-port`
== `c9828e2`, working tree clean.

### Previous round's findings

**Finding 1 (three silent no-op modes) -- closed, all three verified in the code, not from
the porter's summary.**

- Unrecognized value: `decode_attn.cuh:878-888` throws `std::invalid_argument` before the
  occupancy decision at `:890`. Same idiom and same exception type the file's own
  `SWITCH_*` macros already use (`decode_attn.cuh:1089,1145,1151`; `<sstream>`/`<stdexcept>`
  arrive through `flashinfer/utils.cuh:21-22`), so nothing new is required on the include
  path and the throw crosses only ordinary lambdas -- `DISPATCH_PYTORCH_DTYPE_TO_CTYPE` and
  `SWITCH_LAYOUT` in `approx_attn.cu:41-56`, none of them `noexcept` -- to reach pybind11 as
  a `ValueError`.
- Static latch: gone. `decode_attn.cuh:876` is a plain per-call `std::getenv`, and
  `:890` is the only consumer of the resulting `forced_shape`. This host function is
  reached once per `begin_forward` (`decode_handler.cuh:91-107` is its only caller
  fork-wide), so the per-call read costs nothing.
- Requested split reverting: `decode_attn.cuh:924-931` throws when `forced_shape < 0` and
  `tmp_size` ended at `0`, immediately after the `if(new_batch_size == batch_size)` exit at
  `:917`. The guard cannot fire from Python -- `approx_attn.cu:33` fixes
  `batch_size = 1`, and `PartitionPagedKVCacheBinarySearchMinNumPagePerBatch`
  (`decode_attn.cuh:758-783`) plus the `new_batch_size += 1` append cannot return `1` -- so
  the porter's inverted-condition check is the right way to have exercised it.

With those in place a green forced run is real evidence: forced `single` sets
`tmp_size = 0` unconditionally (`:893-896`), the dispatch reads exactly that through
`GetTempFloatBuffer` (`decode_handler.cuh:248-260`), and there is no second occupancy
decision anywhere on the branch (`max_grid_size` is consumed only at `:892` and `:914`).

**Finding 2 (no in-tree consumer, undocumented, not HIP-guarded) -- closed.**
`quest/tests/test_decode_launch_shape.py` drives both shapes as subprocesses, which also
survives the caching in finding 1 ever returning; it names the two suites by path
(`:32`), so collecting the directory cannot recurse. Its third case
(`test_unrecognized_launch_shape_is_rejected`, `:42-45`) is worth more than it looks: it
fails unless the loaded module really reads the variable, so it is a canary against a
stale `_kernels*.so` under the other two cases. `README.md:73` documents the knob. The
not-HIP-guarded decision is stated in the README, the commit body and the notes; it is the
right call for the reason given (the gap is a CU-count property, not a platform one) and
it is inert unless set.

**Finding 3 items -- carried, unchanged.** See "Standing" below.

### Minor -- not worth a porter round, fold into the PR-prep pass

1. `quest/tests/test_decode_launch_shape.py:27-28` -- the `shape is None` branch of
   `_run_suites` is unreachable. The only two callers (`:38`, `:43`) pass a string. Drop
   the branch and the `else`.
2. `README.md:73` -- the knob is documented inside `### Building on AMD GPUs (ROCm)`
   (`README.md:58`) but is live on the CUDA build too, as its own third sentence says. A
   reader of the CUDA build instructions never reaches it. One line under the CUDA
   instructions, or a move to a neutral spot, when the branch is squashed for the PR.

### Standing, carried from the previous round and still owed

- The CUDA no-regression gate at this head. The branch now changes NVIDIA behaviour in
  three places -- the `cbc3890` single-block repair, the `getenv` on every
  `begin_forward`, and the two new throws -- and the CMake restructure
  (`quest/ops/CMakeLists.txt:1-60`) has never been configured with a CUDA toolchain. No
  `nvcc` on this host either, and the blocker remains the pre-existing RAFT branch-24.02
  `CUDA::nvToolsExt` wall rather than anything the port did. Re-checked here:
  `rapids_cuda_init_architectures` appears nowhere under `quest/ops/cmake`, which is the
  one entry point that must precede `project()`, so there is still no known breakage
  vector -- but it is unverified, not verified.
- All three platforms read `revalidate` at `c9828e2`. The new test file means a plain
  suite run now covers both launch shapes on every platform, so no validator needs to
  remember the knob.
- The series is self-reverting for a reader (`9be60fc "Always split the decode KV range
  across blocks"` against `cbc3890 "Fix the single-block decode path and re-enable it"`).
  The PR-prep squash stays worth doing on its own merits.

### Re-verified on the moved tip, so a later round does not redo them

The delta touches only the host work-estimation function, `README.md` and the new test, so
the device-side items were re-read rather than assumed: `protective_get_k_ptr_heads` still
selects the trailing page by position and still bounds the `indices` read at
`page_iter < last_indptr - 1` (`decode_page.cuh:337-350`); the extended indptr is installed
on the un-partitioned path only (`decode_handler.cuh:258-260`) and `float_buffer_` remains
the sound "partitioned" sentinel for `GetChunkIndPtr`/`GetBatchIdxMap`
(`decode_handler.cuh:55-70`, assigned only at `:110`); `BeginForward` allocates nothing
before the throw can happen (`:90-108`), so a rejected value leaks no buffer;
`jargon.py --port Quest` clean; `moatlib.py audit-commits Quest` OK (six `[ROCm]` titles,
all under 72 characters, disclosure present, no agent `Co-Authored-By`); no non-ASCII in
the delta. The lesson promoted to `cuda-to-rocm/references/validation.md` ("A path only
some GPUs reach") was checked against the code it describes rather than the summary: the
three no-op modes it lists are the ones this branch actually shipped and fixed, and its
code fragments are the labelled anti-patterns plus the remedy, not a recipe.

## Validation 2026-08-20 (linux-gfx1100, revalidate at `c9828e2`)

Re-validation triggered by `head_sha` advancing from `9be60fc` (this platform's prior
`validated_sha`) to `c9828e2`. `python3 utils/moatlib.py classify Quest 9be60fc c9828e2` ->
`class=mixed arch_independent=False inert=False` (the three `decode/*.cuh` files are
token-level mixed, `hip_compat/cuda_fp16.h` is comment-only, the new test file is added) --
not doc/comment/rename-only, so no carry-forward shortcut; went straight to a full real-GPU
run as instructed.

4x Radeon Pro W7800 (gfx1100, wave32) visible via `rocm-smi --showhw`, none showing KFD
activity (`rocm-smi --showpids` empty before starting), `HIP_VISIBLE_DEVICES=1`. ROCm
`hip 7.2.53211`, PyTorch 2.14.0a0+gitb81488e, python 3.12.13 (`/opt/conda/envs/py_3.12`),
`transformers==4.37.2` `tokenizers==0.15.2` already present. Fork checkout already at
`c9828e2` (`HEAD == origin/moat-port == status.json.head_sha`), clean, submodules
(`flashinfer`, `pybind`) already initialized from a prior session.

### Host hazard, same as recorded 2026-08-19: broken `find_package(Torch)` export on this
shared host

`torch.utils.cmake_prefix_path` resolves into `/var/lib/jenkins/pytorch`'s editable
scikit-build-core dev checkout, which has no proper `cmake --install` tree. Same host as
the earlier gfx1100 sessions; used the same known-good workaround,
`-DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake`
(a complete from-wheel torch install coexisting in the same environment). No fork or
MOAT-tracked file touched.

### Build

Clean rebuild (wiped `quest/ops/build` first, per the standing note about force-included
headers under `kernels/include/hip_compat`), wrapped `utils/timeit.sh Quest compile --
ninja -C .../quest/ops/build`:

```
cd quest/ops && rm -rf build && mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

Configure and build both clean (6/6 objects, same two pre-existing nodiscard-`hipError_t`
warnings at `decode_handler.cuh:197` and `approx_attn.cu:65` as every prior session, no new
warnings, no errors). `grep -o -- '--offload-arch=[a-z0-9]*' build/build.ninja` -> `gfx1100`
only. `import quest._kernels` exposes the same six ops as every prior session.

### GPU tests

Wrapped `utils/timeit.sh Quest test -- python3 -m pytest ...` (`PYTHONPATH` set to the fork
checkout root, matching the README's `PYTHONPATH` recipe):

```
HIP_VISIBLE_DEVICES=1 PYTHONPATH=<fork-root> python3 -m pytest \
  quest/tests/test_rope.py quest/tests/test_estimate.py quest/tests/test_decode_attention.py \
  quest/tests/test_approx_attention.py quest/tests/test_decode_launch_shape.py -q
```

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| test_decode_launch_shape.py | 3 passed |
| **in-scope total** | **124 passed in 16.75-16.92s** |
| test_topk.py | `-x`: `AttributeError: module 'quest._kernels' has no attribute 'topk_filtering'` |
| test_prefill_attention.py | `-x`: `AttributeError: module 'quest._kernels' has no attribute 'prefill_with_paged_kv_cache'` |

Scoped-out failure mode confirmed unchanged (`AttributeError`, not a wrong answer),
matching every prior session. No non-GPU regression to check against an upstream baseline
-- this project has no CPU-only test path; all suites require the compiled extension and a
GPU.

The new `test_decode_launch_shape.py` means the default run above already covers both
launch shapes, per the review's note that no validator needs to remember the knob. Verified
the override provably takes effect anyway, independently of the porter's evidence, with
`AMD_LOG_LEVEL=3` on `test_decode_attention.py`:

| forced shape | `BatchDecodeWithPagedKVCacheKernel` | `MergeStatesKernel` |
|---|---|---|
| single | 6 | **0** |
| split | 6 | **6** |

Matches the porter's table exactly (default here is `split`, gfx1100 at 70 CUs). Confirmed
real device dispatch (not a mock) on `test_decode_attention.py` with `AMD_LOG_LEVEL=3`: 432
`hipLaunchKernel` lines, 3921 `hipSuccess` occurrences, Quest's own compiled kernels present
by name (`flashinfer::AppendPagedKVCacheDecodeKernel`,
`flashinfer::AppendPagedKVCachePrefillKernel`, `flashinfer::BatchDecodeWithPagedKVCacheKernel`,
`flashinfer::VariableLengthMergeStatesKernel`) -- identical counts to the 2026-08-19 session
at the prior head, as expected since the delta does not touch the decode kernel bodies
themselves.

### CUDA no-regression gate

Not skipped: no record of this gate at `c9828e2` yet (the last recording was at `9be60fc`),
this platform's toolchain still has the CUDA 12.8 conda env and `gcc-11`/`g++-11`, so it was
owed and available. Reproduced the standing wall fresh at this head, wrapped
`utils/timeit.sh Quest cuda-compile -- cmake ...`:

```
cd quest/ops && mkdir build-cuda && cd build-cuda
cmake -GNinja -DUSE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake \
  -DCMAKE_CUDA_COMPILER=/opt/conda/envs/cuda-12.8/bin/nvcc ..
```

`enable_language(CUDA)` succeeded, the pinned `-DCMAKE_CUDA_ARCHITECTURES=80` survived to
the cache, configure reached "Configuring done" -- then failed in the Generate step inside
RAFT's own CMakeLists, identical to every prior recording:

```
CMake Error at build-cuda/_deps/raft-src/cpp/CMakeLists.txt:262 (target_link_libraries):
  The link interface of target "raft" contains:
    CUDA::nvToolsExt
  but the target was not found.
```

**`cuda-not-validated: unchanged from 9be60fc/c1d7fff -- RAFT (pinned branch-24.02) requires
the legacy CUDA::nvToolsExt CMake target, which FindCUDAToolkit no longer creates against a
CUDA 12.8 toolkit; the configure fails inside RAFT's own CMakeLists before Quest's own .cu
sources or the pybind11 target are ever reached.`** Same wall, same file, same line as every
prior recording at every prior head_sha -- confirms the previous rounds' risk assessment
that moving the `rapids-*` includes after `project()` in this branch's CMake restructure
introduced no new breakage vector, since the failure is unrelated to that restructure and
occurs at the identical RAFT-internal line. Throwaway `quest/ops/build-cuda` removed before
completion.

### Jargon and documentation

`python3 utils/jargon.py --port Quest` -> `jargon: clean`. `python3 utils/moatlib.py
audit-commits Quest` -> OK (fork commit messages conform). README.md's "Building on AMD
GPUs (ROCm)" section matches the commands run above verbatim, documents the
`QUEST_DECODE_LAUNCH_SHAPE` knob, and names both scoped-out ops and everything downstream of
them as not running on ROCm yet. No inaccuracy found; nothing sent back.

`git -C projects/Quest/src status --porcelain`: clean. `HEAD` at `c9828e2`, matching
`status.json.head_sha`.

State: `revalidate` -> `completed` on linux-gfx1100, `validated_sha = c9828e2`. wave32 gate
remains satisfied by this arch at the new head. CUDA gate: `cuda-not-validated`, recorded
fresh at this head_sha (environmental wall, pre-existing, external to the port); no penalty,
secondary and compile-only per the gate's own rules. Wall clock: build ~1 min, in-scope
tests ~17s, CUDA-gate reproduction and log-evidence re-runs a few minutes; well within the
~60 minute budget.

## Validation 2026-08-20 (windows-gfx1151, revalidate at `c9828e2`)

Re-validation triggered by `head_sha` advancing from `9be60fc` (this platform's prior
`validated_sha`, from the 2026-08-13 session and its own follow-on porter/repair round at
`cbc3890`) to `c9828e2` via the two linux-gfx1100 porter/review rounds. This platform's own
`cbc3890` repair (single-block decode path fix) is folded into `c9828e2`; the delta since
`9be60fc` is `mixed`/not arch-independent (per the linux-gfx1100 classify call recorded
above), so no carry-forward shortcut -- full real-GPU run.

AMD Radeon 8060S (gfx1151, RDNA3.5 APU, wave32, 20 CUs), Windows 11. Reused the existing
`torch312_venv` (torch 2.12.0+rocm7.14.0a20260519, ROCm 7.14 devel, python 3.12.13,
`transformers==4.37.2` `tokenizers==0.15.2`), all already set up per
`agent_space/win_torch312_env.sh` from the 2026-08-13 session; no re-provisioning needed.
`moat-port` checkout fast-forwarded from `f50e33a` (stale local tip) to `origin/moat-port`
`c9828e2` (`git fetch` + `git merge --ff-only`); submodules already initialized, matched
`git submodule status`.

### Build: same amdclang++/GNU-driver recipe as 2026-08-13, plus two fixes this round found

Configure, wrapped `utils/timeit.sh Quest compile -- cmake ...`:

```
cmake -GNinja -S quest/ops -B quest/ops/build \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/amdclang++.exe \
  -DPython_EXECUTABLE=<venv>/Scripts/python.exe \
  -DCMAKE_PREFIX_PATH=<venv>/Lib/site-packages/torch/share/cmake
ninja -C quest/ops/build   # after stripping ' -fuse-ld=lld-link' from build.ninja (1 occurrence)
```

Two environment issues surfaced and were fixed, neither a port defect -- both are `find_package(Python
REQUIRED COMPONENTS Interpreter Development)` (`quest/ops/CMakeLists.txt:61`) with no venv
hint, on a host with more than one Python on PATH/registry:

1. Without `-DPython_EXECUTABLE`, CMake picked up the system `Python313` (registry/PATH,
   not the `torch312_venv` interpreter) and linked against its `.lib`, producing a
   `_kernels.cp313-win_amd64.pyd` that would not match the `transformers==4.37.2`-compatible
   cp312 environment. Fixed by pinning `-DPython_EXECUTABLE` to the venv interpreter.
2. With Python fixed, CMake's default `CMAKE_BUILD_TYPE` (unset -> `Debug` here) drove the
   pybind Python::Python target to look for `python312_d.lib`, which does not exist for a
   normal (non-debug) CPython install -- `lld-link: error: could not open 'python312_d.lib'`.
   Fixed by pinning `-DCMAKE_BUILD_TYPE=Release`. Neither knob touches the fork; both are
   validator-side CMake invocation flags, recorded here so a later Windows session does not
   rediscover them.

6/6 objects, link succeeded. Same two pre-existing nodiscard-`hipError_t` warnings as every
prior session (`decode_handler.cuh:197`, `approx_attn.cu:65`), no new warnings, no errors.
`import quest._kernels` (after staging `amdhip64_7.dll`/`amd_comgr*.dll`/`hiprtc*.dll`/
`rocm_kpack.dll` next to it, per the standing System32-DLL-shadowing workaround) exposes the
same six ops as every prior session: `BatchDecodeWithPagedKVCachePyTorchWrapper`,
`append_kv_cache_decode/prefill`, `apply_rope_in_place`, `estimate_attn_score`,
`rms_norm_forward`.

### GPU tests: 124/124, matches the linux-gfx1100 reference exactly

Wrapped `utils/timeit.sh Quest test -- python -m pytest ...`:

```
python -m pytest quest/tests/test_rope.py quest/tests/test_estimate.py \
  quest/tests/test_decode_attention.py quest/tests/test_approx_attention.py \
  quest/tests/test_decode_launch_shape.py -q
# 124 passed in 14.97s
```

| suite | linux-gfx1100 @ `c9828e2` | windows-gfx1151 @ `c9828e2` |
|---|---|---|
| test_rope.py | 64 passed | **64 passed** |
| test_estimate.py | 9 passed | **9 passed** |
| test_decode_attention.py | 6 passed | **6 passed** |
| test_approx_attention.py | 42 passed | **42 passed** |
| test_decode_launch_shape.py | 3 passed | **3 passed** |
| **total** | **124 passed** | **124 passed** |

Scoped-out suites unchanged: `test_topk.py` / `test_prefill_attention.py` still stop at
`AttributeError` on the RAFT/prefill symbols that are not part of the ROCm build (recorded
expected shape, no wrong-answer failure).

### Real device dispatch, and this GPU's own launch-shape default confirmed live

`batch_size * num_kv_heads = 32 >= blocks_per_cu * cu_count = 1 * 20 = 20` on this 20-CU
APU, so the CU table above predicts the **single-block** shape by default -- the exact path
this platform's own `cbc3890` repaired. Confirmed rather than assumed, with
`AMD_LOG_LEVEL=3` on `test_decode_attention.py`, counting `MergeStatesKernel` occurrences
(present only in the split shape's merge step):

| configuration | `MergeStatesKernel` count |
|---|---|
| default (no override) | **0** -- single-block, as the CU table predicts |
| `QUEST_DECODE_LAUNCH_SHAPE=single` | **0** |
| `QUEST_DECODE_LAUNCH_SHAPE=split` | **6** |

420 `hipLaunchKernel` lines in the default run (comparable order of magnitude to gfx1100's
432 at the same head), including Quest's own compiled kernels by name -- real device
dispatch, not a mock. The bad-value hard error (`QUEST_DECODE_LAUNCH_SHAPE=Single` ->
`ValueError`) and the "split requested but unusable" guard are both exercised by
`test_decode_launch_shape.py`, which passed as part of the 124 above; not re-derived by hand
this round since the new test file is exactly built to cover them and already ran green
here.

No non-GPU regression to check: this project has no CPU-only test path, unchanged from
every prior session.

### CUDA no-regression gate: not re-run, already recorded fresh at this head_sha

`cuda-not-validated` was reproduced fresh at `c9828e2` by the linux-gfx1100 validator earlier
today (RAFT branch-24.02 `CUDA::nvToolsExt` wall against a CUDA 12.8 toolkit, failing inside
RAFT's own CMakeLists before any Quest source is reached). This host has no CUDA toolkit
regardless. Per the gate's own rule (once per head_sha, already satisfied), not repeated
here.
## Validation 2026-08-20 (linux-gfx90a, revalidate at `c9828e2`)

Re-validation triggered by `head_sha` advancing from `9be60fc` (this platform's prior
`validated_sha`) to `c9828e2`. `python3 utils/moatlib.py classify Quest 9be60fc c9828e2` ->
`class=mixed arch_independent=False inert=False` (`decode_attn.cuh`, `decode_handler.cuh`,
`decode_page.cuh` token-level mixed; `hip_compat/cuda_fp16.h` comment-only;
`test_decode_launch_shape.py` added) -- matches the linux-gfx1100 classification of the same
delta exactly. Not doc/comment/rename-only, so no carry-forward shortcut considered; this is
the real revalidation the dispatch called for, given the delta's device-visible changes
(repaired single-block decode launch shape, the `__HIP_NO_HALF_CONVERSIONS__` include-order
fix, the `int_buffer_`/`float_buffer_` sentinel fix, and the launch-shape override plus its
hard-error paths).

4x MI250X (gfx90a, wave64) visible via `rocm-smi --showhw`, all idle (`rocm-smi --showpids`
empty before starting), `HIP_VISIBLE_DEVICES=1` (104 compute units per `rocminfo`). ROCm
7.14 (TheRock-style pip install under `/opt/conda/envs/py_3.12`, `hip 7.14.60850`, no
`/opt/rocm` on this host), PyTorch 2.14.0a0+git7d05abc, python 3.12.13, `transformers==4.37.2`
`tokenizers==0.15.2` already present in the environment. `projects/Quest/src` already existed;
`git fetch origin` then `git checkout -B moat-port origin/moat-port` (another host had
advanced/rewritten the branch earlier the same day) landed cleanly at `c9828e2`. Submodules
`kernels/3rdparty/{flashinfer,pybind}` already initialized from a prior session.

### Build

Clean rebuild (wiped `quest/ops/build` first, per the standing note about force-included
headers under `kernels/include/hip_compat`), wrapped `utils/timeit.sh Quest compile --
ninja -C .../quest/ops/build`:

```
cd quest/ops && rm -rf build && mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_PREFIX_PATH=/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/share/cmake ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

No editable-torch `cmake_prefix_path` hazard on this host (unlike the shared gfx1100 host):
`torch.utils.cmake_prefix_path` resolves straight into the same site-packages tree used here.
Configure and build both clean (6/6 objects, the same two pre-existing nodiscard-`hipError_t`
warnings at `decode_handler.cuh:197` and `approx_attn.cu:65` as every prior session, no new
warnings, no errors). `grep -o -- '--offload-arch=[a-z0-9]*' build.ninja` -> `gfx90a` only.
`import quest._kernels` exposes the same six ops as every prior session.

### GPU tests

Wrapped `utils/timeit.sh Quest test -- python3 -m pytest ...`:

```
HIP_VISIBLE_DEVICES=1 PYTHONPATH=<fork-root> python3 -m pytest \
  quest/tests/test_rope.py quest/tests/test_estimate.py quest/tests/test_decode_attention.py \
  quest/tests/test_approx_attention.py quest/tests/test_decode_launch_shape.py -q
```

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| test_decode_launch_shape.py | 3 passed |
| **in-scope total** | **124 passed in 27.17s** |
| test_topk.py | `-x`: `AttributeError: module 'quest._kernels' has no attribute 'topk_filtering'` |
| test_prefill_attention.py | `-x`: `AttributeError: module 'quest._kernels' has no attribute 'prefill_with_paged_kv_cache'` |

Scoped-out failure mode confirmed unchanged (`AttributeError`, not a wrong answer), matching
every prior session. No non-GPU regression to check against an upstream baseline -- this
project has no CPU-only test path; all suites require the compiled extension and a GPU.

Per the dispatch, MI250X is high-CU (104 CUs/die), so this is exactly the arch the review's
finding 3 identified as unable to reach the single-block shape by occupancy alone (32 >=
104*blocks_per_cu is false for any realistic `blocks_per_cu`) -- confirmed directly with
`AMD_LOG_LEVEL=3` on the unforced run before touching the override:

| run | `BatchDecodeWithPagedKVCacheKernel` | `MergeStatesKernel` |
|---|---|---|
| default (no override) | 6 | 6 |

`MergeStatesKernel` present means the default occupancy decision picks the split shape here,
same as gfx1100 and as every prior gfx90a session at `9be60fc`. So the repaired single-block
path (the CHANGES-behavior-on-low-CU-GPUs finding from the dispatch) is invisible to a plain
suite run on this GPU by device property, exactly as documented -- which is why the override
and `test_decode_launch_shape.py` exist. Forced both ways to prove the override actually
flips the kernel selection on this arch too, independently of the porter's and gfx1100
validator's evidence:

| forced shape | `BatchDecodeWithPagedKVCacheKernel` | `MergeStatesKernel` |
|---|---|---|
| single | 6 | **0** |
| split | 6 | **6** |

Matches the documented mechanism exactly: forcing `single` suppresses `MergeStatesKernel`
(no partition, no merge step) and forcing `split` restores it. `QUEST_DECODE_LAUNCH_SHAPE=Single`
(bad value) run standalone through `test_decode_launch_shape.py`'s own third case: 3 passed,
confirming the hard-error path is exercised here too, not just on gfx1100. Confirmed real
device dispatch (not a mock): the `AMD_LOG_LEVEL=3` traces above show live kernel launches by
name on the real device, not a skip/mock path.

### CUDA no-regression gate

Not repeated: already recorded fresh at this exact head_sha (`c9828e2`) by the linux-gfx1100
validation above (`cuda-not-validated` -- RAFT branch-24.02's `CUDA::nvToolsExt` wall,
pre-existing and external to the port). Runs once per head_sha per the standing rule; this
host also lacks a CUDA toolkit clone under `/opt/conda`, so it would have been an
environmental skip here regardless.

### Jargon and documentation

`python3 utils/jargon.py --port Quest` -> `jargon: clean`. `python3 utils/moatlib.py
audit-commits Quest` -> OK. README.md's "Building on AMD GPUs (ROCm)" section (already
reviewed for accuracy against this exact head by the 2026-08-20 linux-gfx1100 review) was
not re-derived; the commands run above match it, modulo the venv-Python-executable and
build-type pins that are validator-environment specifics rather than anything the recipe
gets wrong. No inaccuracy found; nothing sent back.

### Integrity

`git -C projects/Quest/src status --porcelain`: clean at completion. `HEAD` at `c9828e2`,
matching `status.json.head_sha`, fast-forwarded from the stale local `f50e33a` this session
found on arrival. Removed the throwaway `-fuse-ld=lld-link` edit's build directory and the
staged runtime DLLs/`.pyd` symlink before finishing (all untracked, gitignored except the
five staged DLLs, which were deleted directly) -- no tracked file touched.

State: `revalidate` -> `completed` on windows-gfx1151, `validated_sha = c9828e2`. windows
gate remains satisfied by this arch at the new head. CUDA gate: `cuda-not-validated`,
carried from the linux-gfx1100 recording at this same head_sha, no penalty. Wall clock:
env/fetch ~2 min, two failed configure/link cycles diagnosing the Python-executable and
build-type issues ~2 min each, final clean build ~1 min, tests ~15s, evidence re-runs a few
minutes; well within the ~60 minute budget.
audit-commits Quest` -> OK (fork commit messages conform). README.md's "Building on AMD GPUs
(ROCm)" section (`README.md:58-73`) matches the commands run above verbatim and documents the
`QUEST_DECODE_LAUNCH_SHAPE` knob accurately, including that it raises rather than silently
no-oping and that `test_decode_launch_shape.py` covers both shapes. No inaccuracy found;
nothing sent back.

`git -C projects/Quest/src status --porcelain`: clean. `HEAD` at `c9828e2`, matching
`status.json.head_sha`.

State: `revalidate` -> `completed` on linux-gfx90a, `validated_sha = c9828e2`. wave64 gate
remains satisfied by this arch at the new head. CUDA gate: `cuda-not-validated`, already
recorded fresh at this head_sha by linux-gfx1100, no penalty, not re-run here. Wall clock:
build ~1 min, in-scope tests ~30s (includes the subprocess-based launch-shape suite), forced
single/split log-evidence re-runs a few minutes; well within the ~60 minute budget.

## Ruling 2026-11-18 (Jeff Daily): finish the port, do not ship it partial

Jeff Daily ruled on 2026-11-18 that the Quest port is to be finished rather than sent upstream or kept partial. At `c9828e2` the HIP build compiles out `topk_filtering` and `prefill_with_paged_kv_cache` (`quest/ops/csrc/bsk_ops.{cu,h}`); both are reached from `quest/models/QuestAttention.py`, so no Quest model, evaluation or example runs on an AMD GPU, and `test_topk.py` and `test_prefill_attention.py` fail with `AttributeError`. This round (linux-gfx942, MI300) implements both operators for HIP, removes the compile-outs, folds in the two minor review leftovers (unreachable branch in `test_decode_launch_shape.py`, placement of the `QUEST_DECODE_LAUNCH_SHAPE` documentation), and runs all seven suites plus an end-to-end model run. No upstream PR is open, so `moat-port` is private and pushed directly.

## Porter round 2026-11-19 (linux-gfx942): topk_filtering and prefill for HIP

Progress is recorded here as it happens.

### topk_filtering: done, fork `016d24c`

Environment for this round: MI300X (gfx942, wave64; this device is a compute partition reporting 38 CUs -- "304 CUs" was first written here in error), `HIP_VISIBLE_DEVICES=0`; PyTorch 2.16.0a0+git6c3af02 with the pip ROCm SDK 10.1.0 (`torch.version.hip` 7.16.26385, no `/opt/rocm`), Python 3.11.17 from the image venv exposed to the job venv through a `.pth` file, `transformers==4.37.2` and `tokenizers==0.15.2` installed in the job venv. Baseline at `c9828e2` on this GPU: the five in-scope suites 124 passed.

Under `USE_HIP`, `kernels/include/topk/decode_select_k.cuh` carries a one-block radix select of its own (RAFT stays the CUDA path, behind `#else`). It is RAFT kRadix8bits' algorithm: cub key twiddle inverted for "greater", 8 bits per pass, values beating the k-th go to `out[0 : k-needed]`, ties fill `out[k-needed : k]` from the back. RAFT orders both groups by `atomicAdd`, so its tie choice is unspecified; the HIP kernel takes both groups in input order (scan over contiguous per-thread slices), so the lowest-positioned ties win, which is one of the orders RAFT can produce. Block barriers and shared atomics only, so wave-size independent. The `buf` argument is unused on HIP.

Verification: `test_topk.py` 49 passed, 15 skipped (the test skips k > kv_len). `agent_space/quest/topk_semantics.py` (not committed) checked 8704 rows against a reference of RAFT's rule: fp16 random, heavily tied (9 distinct values), all-equal, and 5% NaN / 5% +inf / 5% -inf / 10% +0 / 10% -0 inputs; lengths 1, 2, 13, 255-257, 1000, 1023, 4096, 8191, 8192; k in {1, 2, 7, 63, 64, 255, 256, len}; random non-contiguous `in_idx`. 0 failures: front group equals the set of values beating the k-th, back group is `needed` distinct ties, and `out` is bitwise equal to the input at each returned index.

### prefill_with_paged_kv_cache: builds and passes its suite (uncommitted at time of writing)

Approach: keep `kernels/include/prefill/prefill.cuh` (vendored FlashInfer FA2) as it is and give HIP a replacement for `flashinfer/mma.cuh` in `kernels/include/hip_compat/flashinfer/mma.cuh`, same include guard, force-included like `math.cuh` and `vec_dtypes.cuh` (`permuted_smem.cuh` reaches it by a sibling-relative include, and the original's `#error` branches fire at preprocessing under HIP). The shim reproduces the PTX fragment layouts exactly: `ldmatrix.m8n8.x4` (plain and `.trans`) and `stmatrix` emulated by shuffling the row address from the lane that supplied it and reading/writing LDS directly (bit-identical data movement); `mma.sync.m16n8k16` (f32 and f16 accumulate) by gathering the lane's A rows and B columns with width-32 shuffles and fp32 FMA; `rowsum` by a 4-lane butterfly. All shuffles state width 32, so the two 32-lane warps a wave64 holds stay independent: one code path for wave32 and wave64. This is the TornadoVM recipe in `cuda-to-rocm/references/no-hip-equivalent.md` without the matrix-instruction step (see "Why no MFMA/WMMA" below).

Two more HIP-only fixes the kernel needed:
- `__hmax2(half2, half2)` does not exist in HIP's fp16 header (only the bf16 overload). Added to the math shim inside `namespace flashinfer`, with CUDA NaN semantics via `__hmax`. (A future HIP global overload would make the call ambiguous through argument-dependent lookup -- a loud compile error, fixed by deleting this one; the first version of this note claimed the namespace prevented a collision, which the 2026-11-19 review disproved.)
- Tile sizing: the dispatcher assumes two blocks per SM and halves `cudaDevAttrMaxSharedMemoryPerMultiprocessor`. MI300X reports 64 KB per CU, so half leaves `max_num_frags_z_smem = 0` at `num_frags_x = 2`, and `SWITCH_NUM_FRAGS_Z` then only prints `Unsupported max_frags_z` to stderr and returns success with `o` uninitialized (a silent upstream failure mode on any GPU with that little shared memory). Under `USE_HIP` the full per-CU amount is used (one block per CU); frags_z then caps at 4 from the register limit and the largest tile is exactly 64 KB at head_dim 128. Alias `cudaDevAttrMaxSharedMemoryPerMultiprocessor` added to the runtime shim.

`test_prefill_attention.py`: 29 passed, 13 skipped (qo_len > kv_len, by the test's design), MI300X, first build.

### Performance investigation: the ROCm build never optimized device code

The first wider prefill sweep showed correct results but ~1 s per call at qo=1, kv=1024 (rocprofv3 kernel trace: `BatchPrefillWithPagedKVCacheKernel` 957 ms; `AppendPagedKVCachePrefillKernel`, an operator validated since August, 11 ms for 1024 tokens). Counters (`SQ_INSTS_VALU` 35M, `SQ_INSTS_SALU` 51M, `SQ_INSTS_FLAT` 13M per wave over 16 KV iterations) were ~100x what the optimized ISA contains. Cause: `quest/ops/CMakeLists.txt` sets no build type, the compile lines carry no `-O` flag, and clang compiles HIP device code at `-O0` then (single-instantiation probe without `-O3`: 77 VGPR + 256 AGPR, 7384 B/lane scratch, matching the module). nvcc optimizes device code whatever the build type, so the CUDA build never had this problem. The Windows validator hit the same thing as a Debug default and pinned `-DCMAKE_BUILD_TYPE=Release` on its command line (validation 2026-08-20) without the recipe changing. Fix: under `USE_HIP`, an unset build type defaults to `Release` in the CMake file (single-config generators only). Every earlier ROCm validation ran unoptimized kernels; correctness evidence stands, any timing taken from them does not.

Two kernel-level changes found on the way, both measured with a one-instantiation probe (`agent_space/quest/probe.sh`, `-Rpass-analysis=kernel-resource-usage`, gfx942, `-O3`):
- `__launch_bounds__(num_warps * warp_size)` on the prefill kernel under `USE_HIP`. Without it the compiler budgets for 1024-thread blocks (64 VGPR + 64 AGPR at this tile shape) though the kernel launches 128 threads.
- `#pragma unroll 1` on the k loop inside the shim's `m16n8k16`: full unrolling there let the scheduler keep every shuffle result of the caller's unrolled fragment loops live, 4456 B/lane scratch at nfx=2; with it, 0 B (256 VGPR + 212 AGPR, occupancy 1).

### Round result: fork `661449c` (four new commits on `c9828e2`, nothing rewritten)

| commit | what |
|---|---|
| `016d24c` | `[ROCm] Build top-k page filtering with a self-contained radix select` |
| `c79b3cd` | `[ROCm] Optimize device code when no build type is given` (CMake: unset build type -> Release under `USE_HIP`) |
| `70492bc` | `[ROCm] Build prefill attention for AMD GPUs` (mma shim, `__hmax2`, tile sizing, launch bounds, all `USE_HIP` compile-outs and the explicit HIP source list removed) |
| `661449c` | `Document the decode launch-shape knob beside the CUDA build` (review leftovers 1 and 2 of the 2026-08-20 review) |

The final shim keeps `m16n8k16` `__noinline__` and its k loop `#pragma unroll 1`. Build (clean, `ninja`, gfx942, Release by the new default) 18m50s wall: `batch_prefill.cu` is one translation unit with 768 prefill instantiations compiled serially; fully inlining the emulation pushed it past 30 minutes (stopped), and the optimized kernel body dominates even with the helper out of line (probe: ~0.2 s per instantiation net of ~1.5 s header parse with the helper out of line, ~2 s inlined, ~0.4 s with a trivial body). The CUDA build compiles the same 768 instantiations.

GPU tests at `661449c`, MI300X, `HIP_VISIBLE_DEVICES=0`, from the fork root:

```
PYTHONPATH=$PWD python -m pytest quest/tests -q -p no:cacheprovider
```

| suite | result |
|---|---|
| test_rope.py | 64 passed |
| test_estimate.py | 9 passed |
| test_decode_attention.py | 6 passed |
| test_approx_attention.py | 42 passed |
| test_decode_launch_shape.py | 3 passed |
| test_topk.py | 49 passed, 15 skipped (k > kv_len, the test's own skip) |
| test_prefill_attention.py | 29 passed, 13 skipped (qo_len > kv_len, the test's own skip) |
| **all seven** | **202 passed, 28 skipped, 0 failed** (23 s) |

Correction (2026-11-19, after the review): the full-suite and per-suite runs above were made with `transformers==4.37.2`, BEFORE the venv was switched to the project's own pins (`transformers==4.45.2`, `accelerate==1.0.1`, `sentencepiece==0.2.0`, from `pyproject.toml`) for the end-to-end run; the sentence first written here said the reverse. On 4.45.2 all 64 `test_rope.py` cases fail at `quest/tests/test_rope.py:24` (`LlamaRotaryEmbedding.forward() got an unexpected keyword argument 'seq_len'`) in the pure-torch reference: an upstream test written for 4.37, not a port defect. Run the suites with 4.37.2 and the end-to-end script with the pyproject pins (two venvs).

Prefill accuracy sweep (`agent_space/quest/prefill_sweep.py`, not committed): qo_len x kv_len in {1x17, 1x4096, 16x16, 63x700, 64x64, 65x1000, 128x129, 511x2048, 1024x4096, 2048x2048, 3000x8000}, causal and not, fp32 and fp16 QK accumulation (the fp16 form is compiled but `quest.utils.prefill_forward` hard-wires `False`; the sweep calls `_kernels` directly), against fp64: 44/44 within `5e-3 + 5e-3*max|ref|`, worst 4.37e-3 (2048x2048 causal fp16-accumulate). With fp32 accumulation the max error was <= torch's own fp16 attention error on the same inputs in every case.

### End-to-end: `scripts/example_textgen.py`, Llama-2-7b-chat

Model: `NousResearch/Llama-2-7b-chat-hf` at revision `351844e75ed0bcbbe3f10671b3c808d2b83894ee` (an ungated copy of `meta-llama/Llama-2-7b-chat-hf`, the model the script names), downloaded into the job directory (`models/Llama-2-7b-chat-hf`, safetensors). It is the smallest model the Quest path accepts: `topk_filtering` is instantiated for exactly 32 heads (`decode_select_k<..., 32>` and `CHECK_EQ(32, num_heads)`) and prefill supports head_dim 64 and 128 only, so a Llama-2-7B shape is the floor. Two deviations from the stock script, both outside the fork: a copy with only `MODEL_PATH` changed (`agent_space/quest/e2e/example_textgen.py`), and `do_sample` set to `false` in the downloaded `generation_config.json` (the original is kept as `generation_config.json.orig`) so the two methods are compared under greedy decoding instead of sampling.

```
PYTHONPATH=<fork> python example_textgen.py --method hg
PYTHONPATH=<fork> python example_textgen.py --method quest --token_budget 1024
PYTHONPATH=<fork> python example_textgen.py --method quest --token_budget 128
```

All three print the identical continuation (103-token prompt; "Reason: The rabbit is the only one who is not working hard. ... can also mean \"lazy.\""). Wall 13.9 s / 11.9 s / 11.8 s including model load. `rocprofv3 --kernel-trace --stats` on the budget-128 run shows the Quest path really ran: `BatchPrefillWithPagedKVCacheKernel<1,16,true,kNone,2,8,4,4,...>` 32 calls (one per layer, 942 us each), `quest_radix_topk_one_block_kernel<__half,int,256>` 2130 calls (19.9 us), `MaxPossibleSampleWithPagedKVCacheKernel` (page estimate) 2130 calls = 71 sparse decode steps x 30 layers (Quest skips layers 0-1), `BatchDecodeWithPagedKVCacheKernel` 3072, `VariableLengthMergeStatesKernel` 3072, `AppendPagedKVCachePrefillKernel` 32 at 26 us (11 ms at `-O0`). At budget 1024 the context (~200 tokens) never exceeds the budget, so that run exercises prefill and dense decode only.

### Prefill performance: correct, but on the vector units -- a decision for Jeff

On MI300X, causal, 32 heads x 128 (`agent_space/quest/prefill_time.py`, mean of 5 after warmup):

| qo x kv | HIP prefill | torch SDPA |
|---|---|---|
| 1 x 4096 | 13.8 ms | 0.26 ms |
| 128 x 4096 | 29.1 ms | 0.55 ms |
| 1024 x 1024 | 34.5 ms | 1.03 ms |
| 4096 x 4096 | 431 ms | 14.1 ms |
| 8192 x 8192 | 1664 ms | 38.3 ms |

So 30-50x slower than a native AMD attention kernel. Decode (Quest's contribution) is unaffected; a long-context prefill (the evaluation's 32k prompts) will take tens of seconds per prompt. Why no MFMA/WMMA in this round: the kernel's softmax, masking and output code index the accumulator in the PTX m16n8 per-lane layout, so a matrix instruction would need the same operand and result shuffles around it (the skill's TornadoVM recipe: ~64 shuffles per 16x16x16 versus the emulation's 48 shuffles + 128 FMAs), and on CDNA the 16x16 MFMA is wavefront-wide while the kernel's warp is 32 lanes, which forces the 4x4x4 block form; the measured headroom is the shuffle-bound part, not the FMAs. A real speedup means reworking `prefill.cuh` itself around MFMA (wave64) and WMMA (wave32, gfx11 and gfx12 layouts differ) fragments, which is the planner's "Stage 2 AMD-native rewrite", not a small diff. A cheaper alternative under `USE_HIP`: gather the request's pages and call `at::scaled_dot_product_attention` (CK/aotriton flash attention on ROCm) inside `prefill_with_paged_kv_cache`. Neither was done; recorded as deferral `quest-prefill-matrix-cores` for Jeff to rule on.

### CUDA path and nvcc

No `nvcc` on this host (`which nvcc` empty, no CUDA toolkit), so the changed CUDA-visible files were not compiled with nvcc here. What the CUDA build sees from this round: `kernels/include/topk/decode_select_k.cuh` (RAFT path moved under `#else`, otherwise byte-identical plus a trailing newline), `kernels/include/prefill/prefill.cuh` (two `#if defined(USE_HIP)` blocks only), `quest/ops/CMakeLists.txt` (the source glob now precedes the `if(USE_HIP)` and the HIP-only build-type default), `quest/ops/csrc/bsk_ops.{cu,h}` (now byte-identical to upstream). Jeff will have a direct nvcc compile of the changed `.cu`/`.cuh` done on the host that has nvcc. The standing CUDA no-regression gap (RAFT branch-24.02 `CUDA::nvToolsExt` wall) is unchanged.

### For pull-request preparation

The commit series contains a self-reverting pair: `9be60fc` ("Always split the decode KV range across blocks") is undone by `cbc3890` ("Fix the single-block decode path and re-enable it"). The pull-request preparation must squash them. The four commits of this round add no further pair.

## Lesson for cuda-to-rocm: an unset CMake build type runs every HIP kernel unoptimized

Promoted to cuda-to-rocm/references/build-cmake-link.md in bb61ae96.

For `references/build-cmake-link.md` (cross-reference from `faults-memory-streams.md`, which already notes device `-O0` as a fault trigger). Symptom: a HIP port passes its tests but its kernels are tens to hundreds of times slower than expected; `-Rpass-analysis=kernel-resource-usage` shows kilobytes of scratch per lane. Cause: nvcc optimizes device code whatever `CMAKE_BUILD_TYPE` says, but with no build type CMake passes no `-O` flag and clang compiles HIP device code at `-O0`; a README configure command without `-DCMAKE_BUILD_TYPE` is the usual trigger, and a Windows generator defaulting to Debug is the same thing. Fix: in the project's CMake, when the HIP backend is selected and `NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES`, set the cache entry to `Release`; check `ninja -t commands` for `-O3`. Seen: Quest (MI300X, ROCm SDK 10.1): paged KV append 11 ms -> 26 us, FA2 prefill at qo=1 went from 957 ms for kv=1024 (unoptimized) to 13.8 ms for kv=4096 once combined with the next lesson.

## Lesson for cuda-to-rocm: a kernel launched with fewer than 1024 threads needs __launch_bounds__ on HIP to get its registers

Promoted to cuda-to-rocm/references/platform-limits.md in bb61ae96.

For `references/platform-limits.md`, next to the existing `__launch_bounds__(1024)` entry (that one says 1024 changes nothing on HIP; the converse matters). HIP-clang assumes a 1024-thread block when a kernel has no launch bounds and caps registers to fit it (64 VGPR + 64 AGPR on gfx942), where nvcc gives a kernel up to 255 registers regardless. A register-heavy kernel (FlashAttention-2 tiles, 128 fp32 accumulators per lane) launched with 128 threads then spills to scratch. Fix: `__launch_bounds__(<actual block size>)` under the HIP guard; verify with `-Rpass-analysis=kernel-resource-usage`. A second spill source showed up in the same kernel: emulating a matrix instruction with shuffles inside fully unrolled fragment loops let the scheduler keep every shuffle result live (4.4 KB/lane scratch at 256 VGPR + 256 AGPR); `#pragma unroll 1` on the emulation's inner loop took it to 0 in a one-instantiation probe with the helper inlined; the shipped module keeps the helper out of line (compile time) and its prefill kernels report 240-848 B/lane of private segment (848 B for the instantiation the end-to-end run uses; reviewer, `llvm-readelf --notes` on the gfx942 code object). (Quest)

## Lesson for cuda-to-rocm: AMD's 64 KB LDS breaks "half the per-SM shared memory" tile heuristics

Promoted to cuda-to-rocm/references/platform-limits.md in bb61ae96.

For `references/platform-limits.md`. FlashInfer-derived dispatchers size tiles from `cudaDevAttrMaxSharedMemoryPerMultiprocessor / 2` (two blocks per SM). MI300X reports 64 KB per CU (verified; RDNA values not yet recorded, check `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor` on gfx1100/gfx1151), so half leaves room for no tile at head_dim 128, and FlashInfer's `SWITCH_NUM_FRAGS_Z` then prints `Unsupported max_frags_z` to stderr and returns success without launching: the output tensor is whatever `torch::empty` held. Size for one block per CU under the HIP guard (the largest tile is exactly 64 KB). Check every such `SWITCH_*` default branch in a port: an unlaunched kernel looks like a numerics bug. (Quest)

## Lesson for cuda-to-rocm: HIP has no half2 overload of __hmax2

Promoted to cuda-to-rocm/references/build-headers-macros.md in bb61ae96.

For `references/build-headers-macros.md` (or the half-precision operator traps). `amd_hip_fp16.h` provides `__hmax`/`__hmin` for `__half` and `__hmax2` only for `__hip_bfloat162`, so `__hmax2(half2, half2)` fails with "no matching function" and a conversion error from `__hip_bfloat162`. Define it from `__hmax` on `__low2half`/`__high2half`, which keeps CUDA's NaN rule (a NaN yields the other input). Putting it in the project's namespace does not protect against a future HIP overload: `__half2` is a global type, so argument-dependent lookup adds a global `__hmax2(__half2, __half2)` and the call becomes ambiguous (reviewer verified with a simulated overload and `-fsyntax-only`). The failure is a loud compile error at the call; the fix then is to delete the local definition. (Quest, ROCm SDK 10.1)

## Review 2026-11-19 (linux-gfx942, whole branch main...661449c)

Verdict: **CHANGES REQUESTED.** Scope: `git diff main...moat-port` (ten commits, 661449c), with this round's four commits read line by line. The prefill shim, the HIP top-k, the tile sizing and the CMake build-type default are correct; the findings are one narrow wave64 code defect, two documentation inaccuracies in the fork, and records/lessons that would carry wrong facts into the skill.

### Problems

1. **`quest/ops/csrc/rms_norm.cu:50,61` -- full 64-bit `__shfl_xor_sync` mask traps in a HIP Debug build on wave64 when the last wavefront is partial.** `rms_norm_forward` launches `min(1024, (columns/8 + 31)/32*32)` threads (`rms_norm.cu:179`). When that is an odd multiple of 32 (hidden size not a multiple of 512, e.g. 2304, 2816, 3328), the block's last wave64 has only 32 lanes, and `__hip_check_mask` (`amd_warp_sync_functions.h:311`, ROCm SDK 10.1) asserts `mask == __ballot(true)`, which `0xffffffffffffffff` fails; `__hip_assert` is `__builtin_trap()` unless `NDEBUG`. Release (now the default) is unaffected, and hidden 4096 (512 threads) is too. Reproduced with a standalone kernel doing the same five `__shfl_xor_sync(0xffff...ffffULL, v, m, 32)` steps on gfx942: 96 threads without `-DNDEBUG` -> `HSA_STATUS_ERROR_EXCEPTION` / `unspecified launch failure`; 128 threads, or 96 with `-DNDEBUG`, clean (`agent_space/quest-review/mask.hip`). Fix: under `USE_HIP` pass the caller's 32-lane group mask, `0xffffffffull << (__lane_id() & ~31u)` (cuda-to-rocm `faults-wavefront.md`, "A full 64-bit `*_sync` mask in 32-lane group code traps in a HIP debug build on wave64"), or use mask-free `__shfl_xor(v, mask, 32)`.

2. **`kernels/include/hip_compat/README.md:3-7` is inaccurate (this round edited it).** It says the flashinfer headers are picked up because the directory is "placed ahead of the CUDA include path", but `quest/ops/CMakeLists.txt:102-109` says the opposite: `math.cuh`, `mma.cuh` and `vec_dtypes.cuh` are force-included because the sibling-relative include cannot be shadowed. It says "without any edit to the kernel sources themselves", but the branch edits `prefill.cuh`, `decode_attn.cuh`, `decode_page.cuh`, `decode_handler.cuh`, `decode_select_k.cuh` and `rms_norm.cu`. It omits `vec_dtypes.cuh`, `cuda_bf16.h` and `cuda/pipeline`. Rewrite to list every shim and say which are force-included.

3. **`README.md:60-71` (ROCm section) is silent on the accuracy-evaluation dependency.** This round deleted the sentence saying the evaluation does not run on ROCm, so the README now implies it does. But `evaluation/llama.py:18` and `evaluation/mistral.py:17` import `flash_attn` at module scope (`evaluation/passkey/passkey.py:13`, `evaluation/LongBench/pred.py:17` import them), and step 2 installs it with `pip install flash-attn==2.6.3 --no-build-isolation`, a source build that targets CUDA. The job venv has no `flash_attn` (`ModuleNotFoundError`), so no passkey/LongBench/pg19 script was run on ROCm this round; only `scripts/example_textgen.py` was. Either name the ROCm flash-attention install in the ROCm section and run one evaluation (passkey is the shortest), or state that the evaluation scripts need a ROCm build of flash-attention.

4. **Notes (`## Porter round 2026-11-19`, test table): the `test_rope.py 64 passed` result does not hold under the transformers pin the notes say was used.** With the job venv on `transformers==4.45.2` (the `pyproject.toml` pin), `PYTHONPATH=$PWD python -m pytest quest/tests -q -p no:cacheprovider` at 661449c gives **64 failed, 138 passed, 28 skipped**. All 64 are `test_rope.py`: `TypeError: LlamaRotaryEmbedding.forward() got an unexpected keyword argument 'seq_len'` at `quest/tests/test_rope.py:24`, in the pure-torch reference. That is an upstream test written for transformers 4.37 and is not a port defect, but the table must say test_rope passes only on 4.37.x, so a validator on the pinned 4.45.2 does not read 64 failures as a regression. The other six suites reproduce exactly (138 passed, 28 skipped).

5. **Lesson "HIP has no half2 overload of __hmax2": the no-collision claim is wrong.** "Define it in the project's own namespace (inner-scope lookup wins, so a future HIP definition cannot collide)" ignores argument-dependent lookup: `__half2` is a global-namespace type, so ADL adds a global `__hmax2(__half2, __half2)` to the candidate set and the call is ambiguous. Verified with hipcc 10.1 `-fsyntax-only` (`agent_space/quest-review/adl.hip`): with a simulated global overload added -> `error: call to '__hmax2' is ambiguous`. The failure would be a loud compile error, not a silent one, so the code is acceptable as it is. Reword the lesson (and the same sentence in the porter-round notes): a future HIP overload breaks the build at the call, and the fix then is to delete the local one.

6. **Lesson "a kernel launched with fewer than 1024 threads needs __launch_bounds__": the scratch-elimination claim does not describe what shipped.** "`#pragma unroll 1` on the emulation's inner loop took it to 0" is a probe result. The shipped module keeps `m16n8k16` out of line (one local function `flashinfer::mma::detail::m16n8k16<__half>` in the code object) and every prefill kernel reports a nonzero `.private_segment_fixed_size`: 240 to 848 bytes per lane across all instantiations, 848 B for the end-to-end instantiation `<1,16,true,kNone,2,8,4,4,...,__half,float,__half>` (llvm-readelf `--notes` on the gfx942 bundle of `quest/ops/build/_kernels*.so`). State the shipped figure, or limit the claim to the inlined probe. The `__launch_bounds__` half of the lesson stands.

7. **Lesson "AMD's 64 KB LDS breaks ... tile heuristics": "MI300X and RDNA report 64 KB per CU" is checked only for MI300X.** No RDNA value of `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor` is recorded anywhere in these notes. Verify it on gfx1100/gfx1151 when the validators run, or drop "and RDNA".

### Standing (not new, not counted above)

- No nvcc compile of the changed CUDA-visible files (`decode_select_k.cuh`, `prefill.cuh`, `CMakeLists.txt`) at this head; it is owed before the port goes to jeff.
- Prefill is 30-50x slower than torch SDPA. Re-measured with `agent_space/quest/prefill_time.py` on MI300X: 13.8 / 29.1 / 34.6 / 430 / 1665 ms against SDPA 0.27 / 0.55 / 1.03 / 14.1 / 38.5 ms. The notes' table and the 70492bc commit message (431 ms vs 14 ms) are accurate. Deferral `quest-prefill-matrix-cores` is for jeff to rule on.
- PR prep still has to squash the self-reverting pair 9be60fc/cbc3890.

### Checked and correct (do not redo)

- `hip_compat/flashinfer/mma.cuh` checked against the PTX ISA fragment tables. A regs {g,2t..}, {g+8,2t..}, {g,2t+8..}, {g+8,2t+8..}; B {2t..,g}, {2t+8..,g}; f32 D {g,2t},{g,2t+1},{g+8,2t},{g+8,2t+1}; f16 C packing into `acc[2i..2i+1]` matches the m16n8k16 `acc[n]`/`acc[2+n]` indexing. Column `2t+n` of B is read from lanes `4(2t+n)+q`, and lo/hi element order is right. ldmatrix x4 (lane l <- row l/4, cols 2(l%4)..+1, address from lane 8i+l/4) and `.trans` (rows 2(l%4), 2(l%4)+1, col l/4) match, and stmatrix is the inverse. rowsum equals the ones-B mma (d0 += row g, d1 += row g+8, bf16 ones 0x3F80). bf16 lo/hi bit tricks are right. Wave-size safety: HIP `__shfl(v, s, 32)` computes `(s & 31) + (self & ~31)` (`amd_warp_functions.h:151-153`), `__lane_id() % 32 == threadIdx.x` for blockDim (32, 4), branches around the shuffles are warp-uniform, and `__launch_bounds__(num_warps*warp_size)` = 128 matches `nthrs(32, num_warps)`.
- `decode_select_k.cuh` HIP top-k checked against RAFT kRadix8bits. The cub twiddle (inverted for greater) matches, as do front/back group placement (`k-1-tie_pos`) and `needed >= 1` after every pass. The histogram and Hillis-Steele scan have no races (read / sync / write / sync; `s_prefix` rewritten only two barriers after it is read). k == len works, and `CHECK_GE(num_pages, page_budget)` excludes k > len. Same null stream as the RAFT path. Independent check (`agent_space/quest-review/topk_check.py`): 232 cases (n 1..20000, k 1..n, random / ties / all-negative / 10% NaN) against a torch sort, 0 failures (values bitwise equal at the returned indices, distinct indices, value multiset equal to the top k).
- Tile sizing: with LDS >= 64 KB, `min(max_num_frags_z_smem, reg)` is always 2 or 4 at head_dim 64/128 for nfx 1 and 2, so `SWITCH_NUM_FRAGS_Z` cannot fall through on any current AMD GPU. The z = 4 cap keeps the dynamic LDS at 64 KB or less, so 160 KB parts never exceed the per-workgroup limit.
- CMake: the build-type default is HIP-only and single-config, and an explicit value wins. The glob move does not change the CUDA source set. The CUDA path of `prefill.cuh`/`decode_select_k.cuh` preprocesses to the original (plus a trailing newline), and `bsk_ops.{cu,h}` are byte-identical to upstream.
- Commits: `jargon.py --port Quest` clean; `audit-commits` OK; `[ROCm]` on the three ROCm commits and not on 661449c; disclosure present; no agent Co-Authored-By; no non-ASCII. HIP `__hmax` is a `_Float16` compare with CUDA's NaN rule (`amd_hip_fp16.h:758-765`), so the shim's semantics claim holds.
- Full suite at 661449c, MI300X, transformers 4.45.2: 138 passed / 28 skipped outside test_rope (see finding 4).

## Porter round 2026-11-19 (linux-gfx942), answering the 2026-11-19 review -- fork `6e7d8db`

Two new commits on `661449c`, nothing rewritten:

- `2ab01dd` `[ROCm] Name only the calling warp in the RMSNorm shuffle mask` (finding 1). `QUEST_FULL_WARP_MASK` under `USE_HIP` is now `0xffffffffULL << (__lane_id() & ~31u)`. Verified through the real module with `agent_space/quest/rmsnorm_check.py` (`rms_norm_forward` on [1, 7, hidden] fp16 vs a float reference, hidden 256/2304/2816/3328/4096/5120/8192, i.e. 32/288/352/416/512/640/1024 threads): Debug build (`-DCMAKE_BUILD_TYPE=Debug`, tree in `agent_space/quest/build_dbg`, 20m14s) passes every size, max abs error 3.84e-3; negative control, the same Debug tree with only the old full mask restored in `rms_norm.cu`: the first call (32 threads) fails with `unspecified launch failure`; Release build passes every size with the same errors. The reviewer's standalone reproducer (`agent_space/quest-review/mask.hip`, compiled without `-DNDEBUG`) traps at 96 and 160 threads with the old mask and is clean at 96/128/160/1024 with the group mask (`agent_space/quest/mask_fix.hip`).
- `6e7d8db` `[ROCm] Correct the HIP shim and evaluation notes in the documentation` (findings 2 and 3): `kernels/include/hip_compat/README.md` lists all eight shims and says which are force-included; the README's ROCm section says the model wrappers and `scripts/example_textgen.py` run on ROCm and that the evaluation scripts import `flash_attn`, so they need a ROCm build of flash-attention. No evaluation script was run (no `flash_attn` in this environment).

Findings 4-7 (records and lessons) are corrected in place above: the transformers-version sentence under "Round result", the `__hmax2` note and lesson (argument-dependent lookup), the `__launch_bounds__` lesson (probe figure versus the shipped 240-848 B/lane), and the LDS lesson (MI300X only).

**Commit-audit gap, left for PR preparation:** `moatlib.py audit-commits Quest` reports `Quest 6e7d8db: Test Plan has no fenced command block` (the commit is documentation only and its Test Plan is prose). I pushed before reading the full audit output (a `| tail -1` hid the exit status), and the round's instruction is never to rewrite existing commits, so it stands; the PR-preparation squash (already owed for `9be60fc`/`cbc3890`) must give the squashed message a fenced Test Plan.

Tests at `6e7d8db`, MI300X, Release build (incremental from the reviewed tree plus `rms_norm.cu`), job venv on `transformers==4.37.2`/`tokenizers==0.15.2`: rope 64 passed, estimate 9, decode_attention 6, approx_attention 42, decode_launch_shape 3, topk 49 passed + 15 skipped, prefill_attention 29 passed + 13 skipped: **202 passed, 28 skipped, 0 failed**. End-to-end, venv switched to `transformers==4.45.2` (`tokenizers` 0.20.3), `accelerate==1.0.1`: `example_textgen.py --method hg`, `--method quest --token_budget 1024`, `--method quest --token_budget 128` all exit 0 with byte-identical text from "Input Sequence Length: 103" on (md5 prefix `b900af27eeb4`), the same continuation as at `661449c`.

## Review 2026-11-19 pass 2 (linux-gfx942, delta 661449c..6e7d8db, whole branch main...6e7d8db)

Verdict: **PASS.** Two new commits on 661449c (`2ab01dd`, `6e7d8db`), nothing rewritten; `moat-port` == `origin/moat-port` == `6e7d8db`, working tree clean. I checked every previous finding against the code, the build output and the records, not against the porter's summary.

### Previous findings

1. **rms_norm mask: closed.** `rms_norm.cu:51` is now `(0xffffffffULL << (__lane_id() & ~31u))` under `USE_HIP`. That names one half of a wave64 and the whole wave on wave32; the CUDA branch is unchanged. Both `warpReduceSum` calls in `blockReduceSum` are reached by every thread, so each 32-lane half calls with exactly its own mask. I verified it through the real module with my own script (`agent_space/quest-review/rms_dbg.py`): `rms_norm_forward` on [1, 7, h] fp16 against a float reference, h = 256/2304/2816/3328/4096/5120/8192 (32 to 1024 threads, including the odd multiples of 32). On the Debug tree `agent_space/quest/build_dbg` (`CMAKE_BUILD_TYPE=Debug`, its `rms_norm.cu.o` newer than the source) every size passes, max abs error <= 3.0e-3. On the Release module every size passes, max abs error <= 3.3e-3. The negative control (old mask traps on a partial wave) was shown last round with `mask.hip`.
2. **hip_compat README: closed.** `kernels/include/hip_compat/README.md` lists all eight shims, matching `find kernels/include/hip_compat -type f`: cuda_runtime.h, cuda_fp16.h, cuda_bf16.h, cooperative_groups.h, cuda/pipeline, flashinfer/{math,mma,vec_dtypes}.cuh. It separates the include-path shims from the three force-included ones, as `CMakeLists.txt:97-109` does, and no longer claims the kernel sources are unedited.
3. **README evaluation note: closed.** `README.md:73` says the model wrappers and `example_textgen.py` run on ROCm and that the evaluation scripts need a ROCm flash-attention. That matches the evidence: example_textgen was run, the evaluation scripts were not, and they import flash_attn.
4. **test_rope / transformers records: closed.** The correction under "Round result" says the suites need 4.37.2. I reproduced it at 6e7d8db with the job venv switched to transformers 4.37.2 / tokenizers 0.15.2: `PYTHONPATH=$PWD python -m pytest quest/tests -q -p no:cacheprovider` -> **202 passed, 28 skipped**. I restored the venv to 4.45.2 / 0.20.3 afterwards.
5. **`__hmax2` lesson and note: closed.** Both now state the argument-dependent-lookup ambiguity and the loud compile failure. No "cannot collide" claim remains outside the previous review's text.
6. **`__launch_bounds__` lesson: closed.** It limits "took it to 0" to the inlined probe and gives the shipped 240-848 B/lane figure.
7. **LDS lesson: closed.** It says MI300X only and asks for the RDNA value to be checked on gfx1100/gfx1151.

### Commit-audit gap (6e7d8db Test Plan has no fenced block): does not block

`moatlib.py audit-commits Quest` fails on `6e7d8db: Test Plan has no fenced command block`; the commit is documentation only and its Test Plan is prose. Fixing it means rewriting a pushed commit, which jeff's standing rule (no amend, no force-push; new commits go on top) forbids. The branch already owes a PR-preparation squash for 9be60fc/cbc3890, and that squash writes new messages. So this is carried as a PR-preparation requirement, not a porter round: the squashed message must carry a fenced Test Plan, and `audit-commits` must pass before the review PR is opened. `jargon.py --port Quest` is clean.

### Re-read of the whole branch

The delta touches only the `USE_HIP` mask macro and two Markdown files, so the device code and CMake checked last round are unaffected. Last round's "checked and correct" list stands for 6e7d8db (mma shim layouts and wave-size safety, HIP top-k semantics and race freedom, tile sizing, the CMake build-type default, the CUDA path preprocessing to the original).

### Still owed (not review findings)

- nvcc compile of the changed CUDA-visible files before the port goes to jeff.
- PR-preparation squash, including the 9be60fc/cbc3890 pair and the fenced Test Plan above.
- jeff's ruling on deferral `quest-prefill-matrix-cores` (prefill 30-50x slower than SDPA; numbers re-measured and accurate last round).
- Wave32 (gfx1100) and Windows (gfx1151) validation of the new prefill and top-k paths, which no RDNA GPU has run yet. That run should also record the RDNA LDS attribute the lesson asks for.

## Validation 2026-11-19 (linux-gfx942, at `6e7d8db`)

AMD Instinct MI300X, `gcnArchName` `gfx942:sramecc+:xnack-`, wave64, a compute partition reporting 38 CUs and 64 KB shared memory per CU; `HIP_VISIBLE_DEVICES=0` (pinned by the job). PyTorch 2.16.0a0+git6c3af02 (`torch.version.hip` 7.16.26385) on the pip ROCm SDK `rocm`/`rocm-sdk-core` 10.1.0 (no `/opt/rocm`), Python 3.11.17. Fork at `6e7d8db` == `origin/moat-port` == `head_sha`, tree clean. Review passed at this sha (pass 2).

### Build

Clean (build directory wiped), from `quest/ops`:

```
mkdir build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DPython_EXECUTABLE=$FLEET_VENV/bin/python \
  -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja      # wrapped: utils/timeit.sh Quest compile
ln -sf $PWD/_kernels*.so ../../
```

`CMAKE_BUILD_TYPE:STRING=Release` from the new default (none passed); all 8 compile lines carry `-O3` and `--offload-arch=gfx942` only. 18m51s wall, 0 errors, the two pre-existing nodiscard-`hipError_t` warnings.

### GPU tests (transformers 4.37.2, tokenizers 0.15.2)

```
PYTHONPATH=<fork> python -m pytest quest/tests -p no:cacheprovider -q     # wrapped: timeit.sh Quest test
```

**202 passed, 28 skipped, 0 failed** (23.7 s). Per suite: test_rope 64 passed; test_estimate 9; test_decode_attention 6; test_approx_attention 42; test_decode_launch_shape 3; test_topk 49 passed, 15 skipped; test_prefill_attention 29 passed, 13 skipped. The 28 skips are the tests' own `k > kv_len` / `qo_len > kv_len` skips. Compared with the `c9828e2` baseline on this GPU (five suites, 124 passed, the other two `AttributeError`), the two previously missing suites now pass and nothing regressed. The project has no CPU-only test path.

### End-to-end (transformers 4.45.2, tokenizers 0.20.3, accelerate 1.0.1 -- the pyproject pins)

`scripts/example_textgen.py` (copy with only `MODEL_PATH` changed) on `NousResearch/Llama-2-7b-chat-hf@351844e7` in the job directory, greedy (`do_sample: false` in the downloaded `generation_config.json`), each wrapped in `timeit.sh Quest test`:

```
python example_textgen.py --method hg
python example_textgen.py --method quest --token_budget 1024
python example_textgen.py --method quest --token_budget 128
```

All exit 0, prompt 103 tokens, byte-identical text from "Input Sequence Length" on (md5 prefix `b900af27eeb4`, same as at `661449c`). The kernel trace recorded at `661449c` (prefill 32 calls, HIP top-k and page estimate 2130 calls each at budget 128) is not repeated: the delta since (`2ab01dd`, `6e7d8db`) changes only the `rms_norm` mask expression and Markdown.

### CUDA no-regression gate

`cuda-not-validated: no CUDA toolkit on this host` (no nvcc, no `/opt/conda`, and creating one would write outside the job). Owed at this head: Jeff will have the changed CUDA-visible files compiled with nvcc on the host that has it (see "CUDA path and nvcc" above). The pre-existing RAFT branch-24.02 `CUDA::nvToolsExt` wall is unchanged.

### Content checks and integrity

`jargon.py --port Quest` clean; the ROCm build is documented in `README.md` beside the CUDA build. `audit-commits` reports only the known `6e7d8db` fenced-Test-Plan gap recorded for PR preparation. `git -C projects/Quest/src status --porcelain` empty at completion.

State: linux-gfx942 `completed` at `6e7d8db` (wave64 gate). linux-gfx90a, linux-gfx1100 and windows-gfx1151 read `revalidate` at this head; the wave32 and Windows runs are the first on the new prefill and top-k paths, and should record `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor` for the LDS lesson.

## CUDA no-regression compile 2026-11-19 (linux-gfx950 host, real nvcc, at `6e7d8db`) -- PASS, no port regression

Asked for by Jeff Daily before the port is offered upstream: show that the CUDA path still compiles with real nvcc at the current head. Evidence only; no stage or platform record changed, nothing pushed to the fork.

Trees (both with `git submodule update --init --recursive`, identical submodule SHAs: flashinfer `9f49803`, pybind `768cebe`, raft `1e4961e`, googletest, nvbench):

- port: `AMD-Ecosystem/Quest` at `6e7d8db768950aab224fece523f1f640d2f5b6c9` (= `head_sha`)
- base: `mit-han-lab/Quest` at `01c1623bf9395009520874e989e29f683203b357`, the merge base, which is also upstream `main` today

Toolchain: CUDA 12.8 (`/usr/local/cuda`, `nvcc` V12.8.93, build cuda_12.8.r12.8/compiler.35583870_0), CMake 3.31.6, Ninja, gcc/g++ 11.5.0 (Ubuntu `11.5.0-1ubuntu1~24.04.1` debs from `apt-get download`, unpacked with `dpkg -x` into the job directory; nothing installed on the host), PyTorch `2.5.0+cu124` (the `pyproject.toml` pin) in a job venv, Python 3.11.17. There is no NVIDIA GPU on the host, so this is compile and link only.

Recipe: upstream's own `quest/ops/setup.sh` (`cmake -GNinja` + plain `ninja`), with three changes forced by the host and applied the same way to both trees. (1) `CMakeLists.txt` hard-codes `/usr/bin/gcc-11`, which this host does not have, so a toolchain file points C, CXX and the CUDA host compiler at the unpacked gcc-11 (a toolchain file is read inside `project()`, so it overrides the earlier `set()`). (2) The arch is pinned to `-DCMAKE_CUDA_ARCHITECTURES=80` because `native` needs a GPU. (3) `TORCH_CUDA_ARCH_LIST=8.0`; without it, torch's CMake adds sm_50..sm_90a on a host with no GPU, where upstream's GPU machine would get only its own arch. Upstream has no CI.

```
cat > gcc11.cmake <<'X'
set(CMAKE_C_COMPILER "<job>/gcc11/root/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "<job>/gcc11/root/usr/bin/g++-11")
set(CMAKE_CUDA_HOST_COMPILER "<job>/gcc11/root/usr/bin/g++-11")
X
cd <tree>/quest/ops
TORCH_CUDA_ARCH_LIST=8.0 PATH=<venv-cuda>/bin:/usr/local/cuda/bin:$PATH cmake -GNinja \
  -DCMAKE_TOOLCHAIN_FILE=<job>/gcc11.cmake -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DPython_EXECUTABLE=<venv-cuda>/bin/python \
  -DCMAKE_PREFIX_PATH=$(<venv-cuda>/bin/python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
  -S . -B <job>/build-<tree>
ninja -C <job>/build-<tree> -j64 -k 0
```

### The RAFT `CUDA::nvToolsExt` wall does not reproduce on this toolkit

Unmodified upstream **configures and generates successfully** here. The wall recorded since August came from the `/opt/conda/envs/cuda-12.8` toolkit, which has no `libnvToolsExt.so`. `/usr/local/cuda` 12.8 on this host still ships the legacy `include/nvToolsExt.h` and `lib64/libnvToolsExt.so*`, so `FindCUDAToolkit` creates `CUDA::nvToolsExt`, and RAFT's `cpp/CMakeLists.txt:262` gets only a CMake dev deprecation warning ("nvToolsExt has been superseded by nvtx3"). The first base configure without the torch prefix stopped at `find_package(Torch)` after RAFT had configured cleanly. So the wall depends on how the toolkit was installed, not on its version; neither Quest source nor the command line had to supply the target. Step (1) of the requested fallback order therefore went straight to a full build of both trees.

### Results

| | base `01c1623` | port `6e7d8db` |
|---|---|---|
| configure + generate | OK | OK |
| ninja steps (all) | 162 of 179 succeeded, 15 failed, 2 not run | 162 of 179 succeeded, 15 failed, 2 not run |
| `_kernels` target | 8 of 8 (7 `.cu` + link), `_kernels.cpython-311-x86_64-linux-gnu.so` 18,681,432 B | 8 of 8, 18,714,200 B |
| wall (two builds in parallel, -j64 each) | 332 s | 331 s |

The 15 failures are the same set in both trees, all in RAFT's optional compiled library `raft_objs` (`libraft.so`/`libraft.a` are the 2 steps that never ran). Quest's module links only the header-only `raft::raft`, so these do not affect it. Files: `src/neighbors/detail/cagra/search_single_cta_{float,int8,uint8}_uint32_dim{128_t8,256_t16,512_t32,1024_t32}.cu` (12) and `src/neighbors/ivfpq_search_{float,int8_t,uint8_t}_int64_t.cu` (3). First error, identical on both:

```
_deps/raft-src/cpp/include/raft/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh(467): error #20281-D: in whole program compilation mode ("-rdc=false"), a __global__ function template instantiation or specialization ("raft::neighbors::cagra::detail::single_cta_search::search_kernel<8u, 64u, 64u, 0u, 128u, float, float, unsigned int, ::raft::neighbors::filtering::none_cagra_sample_filter>") will be required to have a definition in the current translation unit, when "-static-global-template-stub" will be set to "true" by default in t[he future]
```

This is RAFT branch-24.02 against CUDA 12.8 (`-static-global-template-stub` diagnostics turned into errors). It is upstream's problem and occurs on unmodified upstream. All 30 `warning` lines in each build log come from RAFT's dependency trees (`_deps/{raft,cccl,cuco,cutlass,rmm,fmt,spdlog,hnswlib}-src`); none come from Quest.

### Per-file comparison of Quest's own `.cu` files

To get each file's diagnostics on its own, each of the 7 `_kernels` compile lines from `ninja -t commands _kernels` was rerun separately in both trees. After the tree paths are normalized, the compile lines are identical between base and port: same defines (`-DBSK_TORCH_CHECK`, RAFT/thrust/torch defines, no `USE_HIP`), same include order, `-gencode arch=compute_80,code=sm_80`, `--expt-extended-lambda --expt-relaxed-constexpr`.

| file | base | port |
|---|---|---|
| `csrc/approx_attn.cu` | rc 0, 0 warnings | rc 0, 0 warnings |
| `csrc/batch_prefill.cu` (includes `prefill/prefill.cuh`, touched) | rc 0, 0 warnings | rc 0, 0 warnings |
| `csrc/bsk_ops.cu` | rc 0, 0 warnings | rc 0, 0 warnings |
| `csrc/estimate.cu` (decode headers, touched) | rc 0, 0 warnings | rc 0, 0 warnings |
| `csrc/page.cu` | rc 0, 0 warnings | rc 0, 0 warnings |
| `csrc/rms_norm.cu` (touched) | rc 0, 0 warnings | rc 0, 0 warnings |
| `csrc/topk.cu` (includes `topk/decode_select_k.cuh`, touched) | rc 0, 0 warnings | rc 0, 0 warnings |

Every nvcc output was empty. This covers every `.cu` file in the module, and so every `.cu` file the port touches or that includes a header the port touches (`decode_attn.cuh`, `decode_handler.cuh`, `decode_page.cuh`, `prefill.cuh`, `decode_select_k.cuh`, `rms_norm.cu`). The `kernels/include/hip_compat/` shims are not on the CUDA include path, and the compile lines confirm it. No file needed the direct-nvcc fallback, and none failed to compile. Both modules export the same 651 text symbols and the same undefined dynamic symbol list (`nm -D --undefined-only`, diffed). An import smoke test of the built modules was not run: the CUDA wheel venv lacked `typing_extensions`, and the fleet guard refused the `pip install` into it.

**Verdict: the CUDA path compiles and links with real nvcc 12.8 at `6e7d8db`, with the same result as upstream `01c1623`. There is no port regression.** The only failures are RAFT's optional compiled library, identical on unmodified upstream. This settles the "nvcc compile owed before the port goes to jeff" items from the 2026-11-19 review and validation sections, and it supersedes the standing `cuda-not-validated` RAFT-wall note for any host with a full `/usr/local/cuda` toolkit.

### Lesson for moat-porting: the RAFT `CUDA::nvToolsExt` configure wall is a toolkit-layout artifact

The conda `cuda-12.8` env that the fleet recipe names has no legacy `libnvToolsExt.so`, so `FindCUDAToolkit` creates no `CUDA::nvToolsExt`, and any RAPIDS 24.x dependency (RAFT, rmm) fails at configure. A `/usr/local/cuda` 12.8 install from NVIDIA's packages still ships it, and the same configure passes. Before recording that wall as `cuda-not-validated`, retry with a full system toolkit. Also, a hard-coded `CMAKE_C/CXX_COMPILER` such as `/usr/bin/gcc-11` can be met without root: `apt-get download` the gcc-11 debs, `dpkg -x` them into the job directory (gcc resolves its own libraries relative to its binary), and point a `-DCMAKE_TOOLCHAIN_FILE` at them; the toolchain file is read inside `project()` and wins over the earlier `set()`. (Quest)
## Validation 2026-11-19 (linux-gfx1201, at `6e7d8db`)

AMD Radeon AI PRO R9700, `gcnArchName` `gfx1201`, wave32, 32 CUs, 31.9 GiB VRAM; `HIP_VISIBLE_DEVICES=2` (pinned by the job). PyTorch 2.16.0a0+git6c3af02 (`torch.version.hip` 7.16.26385, hipcc "HIP version: 7.16.26385", AMD clang 24.0.0git) on the pip ROCm SDK `rocm`/`rocm-sdk-core` 10.1.0 (no `/opt/rocm`), Python 3.11 from the image venv exposed to the job venv through a `.pth` file. Same toolchain as the linux-gfx942 run. Fork cloned fresh at `6e7d8db` == `origin/moat-port` == `head_sha`; submodules `kernels/3rdparty/{flashinfer,pybind}` initialized with `git submodule update --init --depth 1` (a full flashinfer clone was too slow on this host's link). First RDNA / wave32 run of the HIP prefill and top-k paths.

### LDS attribute (asked for by the LDS lesson)

`hipDeviceGetAttribute` on gfx1201 (small host program, `agent_space/lds.hip`): `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor` = **131072** (128 KB), `hipDeviceAttributeMaxSharedMemoryPerBlock` = 65536, `hipDeviceAttributeSharedMemPerBlockOptin` = 65536. `torch.cuda.get_device_properties(0)` agrees (128 KB per multiprocessor, 64 KB per block). So on RDNA4 the per-multiprocessor value is twice what one workgroup may allocate (it is the WGP's LDS). The HIP dispatcher in `prefill.cuh` sizes tiles from the full per-CU value, i.e. 128 KB here; what keeps the launch legal is the `max_num_frags_z_reg` cap of 4, which makes the largest tile (head_dim 128, `num_frags_x` 2, 4 warps, z 4) exactly 64 KB = the per-block limit. That is the bound the 2026-11-19 review already relied on ("the z = 4 cap keeps the dynamic LDS at 64 KB or less"); this host confirms it holds with the per-CU value above the per-block limit. For the lesson: the "half the per-SM shared memory" heuristic gives 64 KB on gfx1201 (it would have fitted), but the per-SM attribute is not a per-block budget on RDNA4.

### Build

Clean, from `quest/ops`, job venv's Python first on `PATH`:

```
mkdir build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DPython_EXECUTABLE=$FLEET_VENV/bin/python \
  -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja -j 8     # wrapped: utils/timeit.sh Quest compile
ln -sf $PWD/_kernels*.so ../../
```

`CMAKE_BUILD_TYPE:STRING=Release` from the HIP default (none passed); all 8 compile lines carry `-O3` and `--offload-arch=gfx1201` only. 10m04s wall, 0 errors. Warnings: the pre-existing nodiscard-`hipError_t` ones (`decode_handler.cuh:197`, `prefill.cuh:986`, `approx_attn.cu:65`), and 1536 `prefill.cuh:694: loop not unrolled ... [-Wpass-failed=transform-warning]` (one per prefill instantiation, repeated at link by ld.lld). The gfx942 records do not say whether that warning appeared there; it is a missed-unroll diagnostic, not an error, and the prefill results below are correct.

### GPU tests (transformers 4.37.2, tokenizers 0.15.2, pytest 7.3.2)

```
PYTHONPATH=<fork> python -m pytest quest/tests -p no:cacheprovider -q     # wrapped: timeit.sh Quest test
```

**202 passed, 28 skipped, 0 failed** (17.2 s). Per suite: test_rope 64 passed; test_estimate 9; test_decode_attention 6; test_approx_attention 42; test_decode_launch_shape 3 (both decode launch shapes forced); test_topk 49 passed, 15 skipped; test_prefill_attention 29 passed, 13 skipped. The skips are only `test_topk.py:27: k should be less than kv_len` and `test_prefill_attention.py:51: qo_len > kv_len is not supported` (`-rs`), the tests' own conditions. Identical counts to the linux-gfx942 run at this head; nothing skipped that the 64-lane run ran.

### End-to-end (transformers 4.45.2, tokenizers 0.20.3, accelerate 1.0.1, sentencepiece 0.2.0 -- the pyproject pins)

The GPU holds the 7B model in fp16, so the same model as on gfx942: `NousResearch/Llama-2-7b-chat-hf@351844e75ed0bcbbe3f10671b3c808d2b83894ee` (safetensors) downloaded into the job directory; `do_sample: false` in its `generation_config.json` (original kept as `.orig`), and the unused `pytorch_model.bin.index.json` moved aside. `scripts/example_textgen.py` copy with only `MODEL_PATH` changed, each run wrapped in `timeit.sh Quest test`:

```
python example_textgen.py --method hg
python example_textgen.py --method quest --token_budget 1024
python example_textgen.py --method quest --token_budget 128
```

All exit 0 (15.3 / 14.6 / 14.7 s wall including model load), prompt 103 tokens, byte-identical text from "Input Sequence Length" on: md5 `b900af27eeb4170a5c47bc5940a2ca60`, the same digest as the linux-gfx942 run, so the output also matches across wave64 and wave32. `rocprofv3 --kernel-trace --stats` on the budget-128 run (its generated text identical again) shows the Quest path ran on this GPU: `BatchPrefillWithPagedKVCacheKernel<1,16,true,kNone,2,8,4,4,...,__half,float,__half,int>` 32 calls (433 us avg), `quest_radix_topk_one_block_kernel<__half,int,256>` 2130 calls (11.2 us), `MaxPossibleSampleWithPagedKVCacheKernel` 2130 (5.0 us), `BatchDecodeWithPagedKVCacheKernel` 3072 (13.0 us), `VariableLengthMergeStatesKernel` 3072, `AppendPagedKVCacheDecodeKernel` 3072, `AppendPagedKVCachePrefillKernel` 32 (19.3 us). Call counts equal the gfx942 trace at `661449c`.

### CUDA no-regression gate

`cuda-not-validated: no CUDA toolkit on this host` (no nvcc, no `/opt/conda`, no `/usr/local/cuda`). Still owed at this head on a host with nvcc, as the gfx942 record says.

### Content checks and integrity

`jargon.py --port Quest` clean. `audit-commits` reports only the known `6e7d8db` fenced-Test-Plan gap carried to PR preparation. ROCm build documented in `README.md` beside the CUDA build (followed as written, with the arch changed). `git -C projects/Quest/src status --porcelain` empty. No fork edit, no push to the fork.

State: linux-gfx1201 `completed` at `6e7d8db` (wave32 gate).

## Validation 2026-11-19 (windows-gfx1151, at `6e7d8db`)

AMD Radeon 8060S (gfx1151, RDNA3.5 APU, wave32, 20 CUs, shared system memory), Windows 11. `torch312_venv`: torch 2.12.0+rocm7.14.0a20260519, ROCm SDK 7.14 devel (`_rocm_sdk_devel`, HIP 7.13.26190, AMD clang 23.0.0), Python 3.12, `transformers==4.37.2`, `tokenizers==0.15.2`, `accelerate==1.0.1` (added for the end-to-end run). CMake 3.31.0, Ninja, VS 2022 BuildTools 14.44 (LIB/INCLUDE already in the environment). Fork cloned fresh into `projects/Quest/src` at `6e7d8db` == `head_sha`; submodules `flashinfer` and `pybind` only (`--depth 1`). `git status --porcelain` shows no modified tracked file (untracked: the five ROCm runtime DLLs staged in `quest/`, per the System32 `amdhip64_7.dll` workaround). No fork edit, no push. First Windows run of the HIP prefill and top-k code.

### Build: the README's ROCm command does not work as written on Windows -- FINDING for the porter

README command (`cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_PREFIX_PATH=... ..`, then `ninja`), venv Python first on PATH, ROCm SDK `bin` on PATH. Each attempt below changes one thing, run wrapped in `utils/timeit.sh Quest compile`:

1. README command as written (the host env set `CC`/`CXX` = clang-cl): configure fails at the HIP compiler check, `clang++: error: no such file or directory: '/machine:x64'` (CMake 3.31 gives the GNU-style clang++ MSVC-style link flags when C++ is clang-cl).
2. With `CC`/`CXX` unset and `-DCMAKE_HIP_COMPILER=<sdk>/lib/llvm/bin/amdclang++.exe`: CMake takes MSVC `cl.exe` for CXX and refuses: "The current configuration mixes Clang and MSVC ... not supported".
3. Both `-DCMAKE_CXX_COMPILER` and `-DCMAKE_HIP_COMPILER` = `amdclang++.exe`: configures, but (a) `find_package(Python)` selects the system Python 3.13 (registry) although the venv Python is first on PATH, and (b) `CMAKE_BUILD_TYPE` is `Debug`, not the `Release` the new `USE_HIP` default in `quest/ops/CMakeLists.txt` intends: `if(NOT CMAKE_BUILD_TYPE ...)` runs after `enable_language(HIP)`, and on Windows the HIP platform module has already seeded the cache with `Debug`, so the default never fires. (A Debug build links `python312_d.lib`, which a normal CPython does not ship, and compiles every HIP kernel at -O0.)
4. Working configure, no hand edit of any generated file:

```
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=<venv>/Scripts/python.exe \
  -DCMAKE_CXX_COMPILER=<sdk>/lib/llvm/bin/amdclang++.exe -DCMAKE_HIP_COMPILER=<sdk>/lib/llvm/bin/amdclang++.exe \
  -DCMAKE_PREFIX_PATH=<venv>/Lib/site-packages/torch/share/cmake ..      # CC/CXX unset
ninja -j 8       # 8/8 targets, 991 s (16m32s) wall, exit 0, no errors
```

The `-fuse-ld=lld-link` from CMake is still in `build.ninja` (1 occurrence) and was NOT stripped: it is harmless here (no `-fgpu-rdc`); the link prints only `clang: warning: argument unused during compilation: '--ld-path=...lld-link.exe'`. So the August hand edit is no longer needed. About 1900 warning lines (the known nodiscard-`hipError_t` ones plus the prefill unroll diagnostics), 0 errors.

What a Windows user must pass that the README does not say: both compilers as `amdclang++.exe`, `-DPython_EXECUTABLE`, `-DCMAKE_BUILD_TYPE=Release`. For the porter, in the project's build files (guarded to Windows/HIP, per the standing rule that the documented command must work): (i) on WIN32 with `USE_HIP`, choose amdclang++ for C++ and HIP, or document the Windows command; (ii) the unset-build-type default must beat the platform's seeded `Debug` (a seeded `Debug` is indistinguishable from a user choice after `enable_language`, so seed `CMAKE_BUILD_TYPE` before `project()`, or set `CMAKE_BUILD_TYPE_INIT`); (iii) `Python_EXECUTABLE`, if the project wants to stop picking an unrelated interpreter. Not fixed here.

### GPU tests (transformers 4.37.2), `utils/timeit.sh Quest test`, `ulimit -c 0`

```
PYTHONPATH=<fork> python -m pytest quest/tests -p no:cacheprovider -q -rs     # _kernels.cp312-win_amd64.pyd and runtime DLLs staged in quest/
```

**202 passed, 28 skipped, 0 failed** (18.7 s), identical to linux-gfx942 and linux-gfx1201. Skips are the tests' own: `test_topk.py:27` k >= kv_len (15) and `test_prefill_attention.py:51` qo_len > kv_len (13). Per suite (separate runs): test_rope 64 passed; test_estimate 9; test_decode_attention 6; test_approx_attention 42; test_decode_launch_shape 3 (both launch shapes forced); test_topk 49 passed, 15 skipped; test_prefill_attention 29 passed, 13 skipped. The five suites validated here in August (124) pass again; the two new suites pass. No CPU-only path exists.

### LDS attribute

Small HIP program (`hipDeviceGetAttribute`, built with amdclang++ `-x hip --offload-arch=gfx1151`): `hipDeviceAttributeMaxSharedMemoryPerMultiprocessor` = **65536**, `MaxSharedMemoryPerBlock` = 65536, `SharedMemPerBlockOptin` = 65536, multiprocessor count 20. torch reports the same (`shared_memory_per_multiprocessor` 65536). So on gfx1151 the per-CU value equals the per-block limit (gfx1201: 131072 per multiprocessor, 65536 per block); the HIP dispatcher's full per-CU value is 64 KB here, as on MI300X, and the z <= 4 cap keeps the largest tile at 64 KB = the block limit. The prefill tests (head_dim 64 and 128) pass.

### End-to-end (no 7B model on this host)

The Linux runs used Llama-2-7b-chat (13 GB fp16), too large for this shared-memory APU. The Quest path is instantiated for exactly 32 heads and head_dim 64/128, so the smallest model through it has hidden size 4096 and 32 heads; depth and vocabulary are what can shrink. Used instead a RANDOM-WEIGHT Llama-architecture checkpoint built for this check (`agent_space/quest_e2e/mk.py`): hidden 4096, 32 heads, 32 KV heads, 3 layers (Quest skips layers 0-1, so layer 2 is the one sparse layer), intermediate 512, vocab 32000, fp16, 0.96 GB, seed 0, with the tokenizer files of `NousResearch/Llama-2-7b-chat-hf@351844e7` (the same 103-token prompt). It is not a language model, so the output is noise and the Linux "byte-identical text" comparison does not transfer: greedy decoding on random weights has near-tied logits, and 1-ulp fp16 differences between the HF attention and Quest's kernels flip the argmax.

Script: stock `scripts/example_textgen.py` with only `MODEL_PATH` and `max_length` (2048 to 400, since a random model never emits EOS) changed, and `PYTHONIOENCODING=utf-8` (noise tokens do not print under cp1252). `--method hg`, `quest --token_budget 1024`, `quest --token_budget 128`: all three exit 0, 103-token prompt, generated texts NOT byte-identical (they share the prompt plus about 18 characters, then diverge).

Because text equality is the wrong test on this model, measured instead (`agent_space/quest_e2e/cmp.py`: HF model vs Quest model, same weights and prompt, 30 greedy new tokens, scores kept, budget 1024 so the dense path): step-0 (prefill) logits max|diff| 0.0098 at logit magnitude up to 5.95 (1 fp16 ulp); first token divergence at step 5, where the HF top1-top2 logit margin is 0.0039 (1 ulp, a true tie); the five steps before it match with max logit |diff| 0.0107. So prefill and dense decode agree to fp16 rounding. This is weaker evidence than the Linux runs' identical text from a real model, and is recorded as such.

The Quest path really ran (`AMD_LOG_LEVEL=3`, budget 128, 120 new tokens, `agent_space/quest_e2e/prof.py`): `BatchPrefillWithPagedKVCacheKernel` 3 launches (3 layers), `AppendPagedKVCachePrefillKernel` 3, `quest_radix_topk_one_block_kernel` 94 and `MaxPossibleSampleWithPagedKVCacheKernel` 94 (one sparse layer x 94 steps past the budget), `BatchDecodeWithPagedKVCacheKernel` 357, `VariableLengthMergeStates` 0 (single-block decode shape, as the 20-CU table predicts, same as August).

### Differences from the Linux runs

Build: three flags the README lacks (above); 16m32s vs 18m51s (gfx942) and 10m04s (gfx1201). End-to-end: random-weight 1 GB model, text not byte-identical (explained above). LDS: 64 KB per multiprocessor as gfx942, half of gfx1201. Test counts identical. Environmental: HuggingFace downloads needed `curl.exe` (schannel) because Python's TLS fails certificate verification on this network; `accelerate` was missing from the venv.

### CUDA gate, content checks

CUDA gate: not run (no CUDA toolkit on this host; recorded at this head by the linux-gfx950 host). `python utils/jargon.py --port Quest` clean (needs local `main` and `moat-port` branches in the clone). The README documents the ROCm build beside the CUDA build, but only the Linux command (see the finding above).

State: windows-gfx1151 `completed` at `6e7d8db` (windows gate), with the build finding above carried to the porter.

## Porter round 2026-11-19 (windows-gfx1151): the README's ROCm command works on Windows -- fork `a70f4d9`

Answers the build finding of the windows-gfx1151 validation at `6e7d8db`. One commit on top, nothing rewritten: `a70f4d9` `[ROCm] Make the documented ROCm build command work on Windows` (`quest/ops/CMakeLists.txt` +47/-5, `README.md` +10). CUDA path, kernels and tests untouched (every CMake change is in the `USE_HIP` branch; the Windows parts are under `CMAKE_HOST_WIN32`/`WIN32`).

Environment: the validation's (`torch312_venv`, torch 2.12.0+rocm7.14, ROCm SDK 7.14 devel, CMake 3.31.0, Ninja, VS 2022 BuildTools 14.44 with LIB/INCLUDE in the environment), `agent_space/quest_win_env.sh`.

### Reproduction at `6e7d8db` (README command as written, fresh build dir each, configure only)

1. Env `CC`/`CXX` = clang-cl: CXX identified "Clang 23.0.0 with MSVC-like command-line", HIP found on its own as GNU-like `clang++`, HIP compiler check broken: `clang++: error: no such file or directory: '/machine:x64'`. Root cause: `Platform/Windows-Clang.cmake` is include-guarded and takes MSVC-style rules for all languages from the first language (CXX via `project()`), so the GNU-style HIP link gets MSVC link flags.
2. `CC`/`CXX` unset: CMake takes MSVC `cl.exe` for CXX, then `enable_language(HIP)` dies "The current configuration mixes Clang and MSVC". Root cause: nothing tells CMake to use clang for C++; CMake's default Windows C++ compiler is cl.
3. Both compilers = ROCm `clang++.exe`: configures, `CMAKE_BUILD_TYPE=Debug`, Python = system 3.13 from the registry. Root cause (build type): `Platform/Windows-Clang.cmake:238` sets `CMAKE_BUILD_TYPE_INIT Debug`, which `project()` writes into the cache, so the port's `if(NOT CMAKE_BUILD_TYPE ...)` after `enable_language(HIP)` saw `Debug` and never fired (on Linux the INIT value is empty, which is why it worked there). Root cause (Python): FindPython on Windows searches the registry first and honours a virtual environment only through `VIRTUAL_ENV`/`CONDA_PREFIX` (an activated environment), not through PATH order; the validator's venv was on PATH but not activated. With the venv activated (`Scripts/activate`) FindPython picks `torch312_venv/Scripts/python.exe` with no project change.
4. Found while proving the fix (masked in the validation by `PYTORCH_ROCM_ARCH=gfx1151` in the env script): with `PYTORCH_ROCM_ARCH` unset, `-DCMAKE_HIP_ARCHITECTURES=gfx1151` alone fails in `find_package(Torch)`: `LoadHIP.cmake:70 No GPU arch specified for ROCm build`. torch's `torch_hip_get_arch_list` reads only `ENV{PYTORCH_ROCM_ARCH}` or `rocm_agent_enumerator`, which ROCm does not ship on Windows. With no architecture at all, CMake also silently caches `gfx906`.

### The change

- Build type: the Release default moves before `project()` (HIP branch of the existing `if(NOT USE_HIP)` block), condition `NOT CMAKE_BUILD_TYPE AND NOT GENERATOR_IS_MULTI_CONFIG` (the global property is set before `project()`; checked with Ninja = 0, Ninja Multi-Config = 1). Linux: same effect as before (cache entry Release when unset; `project()` does not overwrite an existing cache entry).
- C++ compiler on Windows, before `project()`, only when neither `CMAKE_CXX_COMPILER` nor `ENV{CXX}` is given: `CMAKE_HIP_COMPILER` if given, else `ENV{HIPCXX}`, else `<hipconfig --hipclangpath>/clang++.exe` (`find_program(hipconfig)` on PATH with `HIP_PATH`/`ROCM_PATH` hints; the pip SDK puts `hipconfig.exe` in the venv `Scripts`), and in that last case `CMAKE_HIP_COMPILER` too, so both are one driver. If nothing is found CMake falls back to cl and the check below fires.
- After `project()`, `WIN32` + `CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"`: FATAL_ERROR naming the fix (unset CXX or `-DCMAKE_CXX_COMPILER=<ROCm>/lib/llvm/bin/clang++.exe`, fresh build dir). Verified for env `CXX=clang-cl.exe` and `CXX=cl.exe` (MSVC sets the variant to MSVC too).
- `WIN32` and no architecture before `enable_language(HIP)`: FATAL_ERROR asking for `-DCMAKE_HIP_ARCHITECTURES` or `PYTORCH_ROCM_ARCH` (verified). After it, `WIN32` and `ENV{PYTORCH_ROCM_ARCH}` undefined: set it (configure-process environment only) to the resolved architecture list for `find_package(Torch)`.
- README: a Windows paragraph and PowerShell block under "Building on AMD GPUs (ROCm)": x64 build tools environment, activated Python environment (else `-DPython_EXECUTABLE=<path to python.exe>`), always name the architecture, leave `CXX` unset.

### Documented Windows command (as run, PowerShell, venv activated via `Activate.ps1`; CC, CXX, PYTORCH_ROCM_ARCH unset)

```
cd quest\ops
mkdir build; cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 `
  "-DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" ..
ninja
copy _kernels*.pyd ..\..\
```

Run as `agent_space/quest_porter/readme_build.ps1` through `utils/timeit.sh Quest compile` (the script adds only `$LASTEXITCODE` checks). Configure: CXX and HIP both `clang++.exe` (GNU-like), `CMAKE_BUILD_TYPE=Release`, Python `torch312_venv/Scripts/python.exe` 3.12.10, "Building PyTorch for GPU arch: gfx1151", `-O3` on all 8 compile lines, `--offload-arch=gfx1151`. Build 8/8, exit 0, 0 errors, 1520 s wall (a concurrent configure check ran during it; the validation's 991 s is the clean figure). No generated file edited; the `-fuse-ld=lld-link` link warning only, as before.

### Tests (`utils/timeit.sh Quest test`, `ulimit -c 0`, transformers 4.37.2)

```
PYTHONPATH=<fork> python -m pytest <fork>/quest/tests --rootdir <fork> -p no:cacheprovider -q
```

**202 passed, 28 skipped, 0 failed** (17.3 s), identical to the validation. Fresh `.pyd` from this build (the old one was moved aside first); the ROCm runtime DLLs staged in `quest/` per this host's System32 `amdhip64_7.dll` problem (environment, not documented in the README). End-to-end not repeated (unaffected).

`python utils/jargon.py --port Quest` clean; commit ASCII-only, LF. Linux hosts revalidate at `a70f4d9`: on Linux the only behavioural difference is where the Release default is set (same result). The old validation build was moved to `agent_space/tmp/quest_build_6e7d8db`.

## Lesson for cuda-to-rocm: a pip ROCm torch on Windows needs PYTORCH_ROCM_ARCH for find_package(Torch)

Promoted to cuda-to-rocm/references/windows.md in 37d13004 (build-type part: cuda-to-rocm/references/build-cmake-link.md in 35013557).

For `references/windows.md`, "CMake and compiler drivers" (beside the `rocm_agent_enumerator`/gfx906 fallback note). A CMake torch extension on Windows that names its target with `-DCMAKE_HIP_ARCHITECTURES` still fails in `find_package(Torch)` with `LoadHIP.cmake: No GPU arch specified for ROCm build`: torch's `torch_hip_get_arch_list` reads only the `PYTORCH_ROCM_ARCH` environment variable or `rocm_agent_enumerator`, which ROCm does not ship on Windows. A build environment that exports `PYTORCH_ROCM_ARCH` hides this from whoever writes the README. Fix in the project: on `WIN32`, when `ENV{PYTORCH_ROCM_ARCH}` is undefined, `set(ENV{PYTORCH_ROCM_ARCH} ...)` from the resolved `CMAKE_HIP_ARCHITECTURES` before `find_package(Torch)`, and stop with a message when no architecture is given at all. Also: the Release-default idiom must run before `project()` on Windows, because `Windows-Clang.cmake` sets `CMAKE_BUILD_TYPE_INIT Debug` and `project()` caches it. Before `project()` it must also check the `CMAKE_BUILD_TYPE` environment variable, which CMake (3.22+) applies only inside `project()`, or it forces Release over a build type the user gave there: `get_property(_mc GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)` then `if(NOT CMAKE_BUILD_TYPE AND "$ENV{CMAKE_BUILD_TYPE}" STREQUAL "" AND NOT _mc) set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)`; and FindPython on Windows follows an activated environment (`VIRTUAL_ENV`/`CONDA_PREFIX`), not PATH order. (Quest)

## Review 2026-11-19 pass 3 (linux-gfx942, delta 6e7d8db..a70f4d9) -- CHANGES REQUESTED

### Summary

`a70f4d9` makes the README's ROCm configure work on Windows: it chooses ROCm's clang++ for C++, moves the Release default before `project()`, stops on an MSVC-style C++ compiler or a missing architecture, and hands the architecture to `find_package(Torch)`. On Linux it changes nothing (checked below). One finding, in the Windows fallback the README documents.

### Build System

1. **`quest/ops/CMakeLists.txt:31` plus `README.md:73`: the README's fallback for a shell with no active environment ("outside one, add `-DPython_EXECUTABLE=<path to python.exe>`") does not work with the pip ROCm SDK, and the error it gives points the wrong way.** `find_program(_quest_hipconfig hipconfig HINTS ENV HIP_PATH ENV ROCM_PATH PATH_SUFFIXES bin)` finds `hipconfig.exe` only on PATH or under `HIP_PATH`/`ROCM_PATH`. The pip SDK installs it in the environment's `Scripts` directory, which is on PATH only when the environment is activated, and it sets neither variable. In the non-activated case the README describes, the lookup misses, CMake falls back to `cl.exe`, and the line 53 FATAL_ERROR fires telling the user to "Unset the CXX environment variable", which the README has already told them to leave unset. The `cuda-to-rocm` Windows reference (`references/windows.md`, the `hipconfig --hipclangpath` recipe) already names the fix: add the Python interpreter's directory as a hint, i.e. `get_filename_component(... "${Python_EXECUTABLE}" DIRECTORY)` when `Python_EXECUTABLE` is given (it is a cache entry before `project()` when passed with `-D`), and `$ENV{VIRTUAL_ENV}/Scripts` / `$ENV{CONDA_PREFIX}/Scripts`. The other option is a README that says the non-activated shell also needs the environment's `Scripts` directory on PATH. Either way, run the non-activated case on the Windows host to prove it. Evidence: code reading against CMake 3.31's `find_program` search order and the SDK layout recorded in notes (the porter round, "the pip SDK puts `hipconfig.exe` in the venv `Scripts`"); not run here (this is a Linux host). The documented activated-environment command is unaffected; the porter ran it.

### Checked on this host, no finding (evidence for the next round)

linux-gfx942 (MI300X), CMake 3.31.6 (pip), torch 2.16.0a0+git6c3af02 / HIP 7.16 on the pip ROCm SDK 10.1. Configure only, `-GNinja -DUSE_HIP=ON -DCMAKE_PREFIX_PATH=<torch cmake prefix>`, fresh build directory each, `--trace-expand --trace-source=quest/ops/CMakeLists.txt`:

| configure | `a70f4d9` | `6e7d8db` |
|---|---|---|
| `-DCMAKE_HIP_ARCHITECTURES=gfx942`, no build type, `PYTORCH_ROCM_ARCH` unset | Release | Release |
| same, `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | RelWithDebInfo | RelWithDebInfo |
| `-DCMAKE_BUILD_TYPE=Debug` (`a70f4d9`, env `PYTORCH_ROCM_ARCH` set) | Debug | -- |
| no architecture, no build type, `PYTORCH_ROCM_ARCH` unset (auto-detect) | Release, 8 lines `--offload-arch=gfx942` | same |
| `-G "Ninja Multi-Config"` | `CMAKE_BUILD_TYPE` not set, configs Debug;Release;RelWithDebInfo | -- |

- With identical options, `6e7d8db` and `a70f4d9` produce the same `CMakeCache.txt` (only the build-dir path differs) and the same `build.ninja` once source and build paths are normalized. The Linux build is unchanged.
- The trace shows each Windows guard evaluated false with its body not entered: line 25 `if(CMAKE_HOST_WIN32 AND ...)`, 53 `if(WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT ...)`, 74 `if(WIN32 AND NOT CMAKE_HIP_ARCHITECTURES)`, 85 `if(WIN32 AND NOT DEFINED ENV{PYTORCH_ROCM_ARCH})`. C++ stays `/usr/bin/c++`.
- CUDA path: the diff adds an `else()` branch to the existing `if(NOT USE_HIP)` and edits only inside `if(USE_HIP)`; no CUDA line changed. The upstream main tip is `01c1623`, the branch's merge base (0 behind).
- Claims in the commit and the lesson, checked against the source: `Windows-Clang.cmake:238` sets `CMAKE_BUILD_TYPE_INIT Debug` and `CMakeInitializeConfigs.cmake:22` caches it without FORCE, so a cache entry set before `project()` wins; `torch_hip_get_arch_list` (`Caffe2/public/utils.cmake:280`) reads only `ENV{PYTORCH_ROCM_ARCH}` or `rocm_agent_enumerator` and splits on spaces, which matches the commit's space join; FindPython honours `VIRTUAL_ENV`/`CONDA_PREFIX` (`FindPython/Support.cmake:1716`). The lesson's exact Windows error text (`LoadHIP.cmake:70` versus the macro's own "Could not detect ROCm arch" message) depends on the Windows torch 2.12 files, which this host does not have; not a finding.
- `jargon.py --port Quest` clean; `audit-commits` reports only the known `6e7d8db` fenced-Test-Plan gap. `a70f4d9`: `[ROCm]` title 62 chars, fenced Test Plan, "Authored with Claude.", no trailer.

### Recommendation

**Request Changes** (finding 1). The fix stays inside the Windows-guarded block or the README, so on Linux it should be a source-class delta to the configure checks above.
## CUDA no-regression compile 2026-11-19 (linux-gfx950 host, real nvcc, at `a70f4d9`) -- PASS, no port regression

Asked for by Jeff Daily before the port is offered upstream: show that the CUDA path still compiles with real nvcc at the current head, after the Windows build round. Evidence only; no stage or platform record changed, nothing pushed to the fork, no GitHub write. This repeats the `6e7d8db` run above at the new head, from fresh clones in this job's directory.

Trees (fresh `git clone` + `git submodule update --init --recursive`; identical submodule SHAs: flashinfer `9f49803b`, its googletest `40412d85` and nvbench `f57aa9c9`, googletest `6a593823`, nvbench `d8dced8a`, pybind `768cebe1`, raft `1e4961e2`):

- port: `AMD-Ecosystem/Quest` at `a70f4d93f08163844a874b5ed4feaee061cb9d9c` (= `head_sha`)
- base: `mit-han-lab/Quest` at `01c1623bf9395009520874e989e29f683203b357` (`git merge-base origin/main a70f4d9`, also upstream `main` today)

The port touches 18 files against the base. Since `6e7d8db` only `quest/ops/CMakeLists.txt` (+47/-5, all in the `USE_HIP` branch or under `CMAKE_HOST_WIN32`/`WIN32`) and `README.md` changed.

Toolchain: CUDA 12.8 at `/usr/local/cuda` (`nvcc` V12.8.93, build cuda_12.8.r12.8/compiler.35583870_0), CMake 3.31.6, Ninja 1.13.0, g++-11 11.5.0 (`11.5.0-1ubuntu1~24.04.1`, the unpacked debs from the earlier run copied into this job's directory), PyTorch `2.5.0+cu124` (the `pyproject.toml` pin, from `https://download.pytorch.org/whl/cu124`) in the job venv, Python 3.11.17. There is no NVIDIA GPU, so this is compile and link only. Upstream has no CI; the recipe is its `quest/ops/setup.sh` (`cmake -GNinja` + `ninja`) with the same three host-forced changes as before, applied to both trees: a toolchain file for the hard-coded `/usr/bin/gcc-11`, `-DCMAKE_CUDA_ARCHITECTURES=80` (`native` needs a GPU), `TORCH_CUDA_ARCH_LIST=8.0`.

```
cat > gcc11.cmake <<X
set(CMAKE_C_COMPILER "<job>/nvcc/gcc11/root/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "<job>/nvcc/gcc11/root/usr/bin/g++-11")
set(CMAKE_CUDA_HOST_COMPILER "<job>/nvcc/gcc11/root/usr/bin/g++-11")
X
cd <tree>/quest/ops
TORCH_CUDA_ARCH_LIST=8.0 PATH=<venv>/bin:/usr/local/cuda/bin:$PATH cmake -GNinja \
  -DCMAKE_TOOLCHAIN_FILE=<job>/nvcc/gcc11.cmake -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DPython_EXECUTABLE=<venv>/bin/python \
  -DCMAKE_PREFIX_PATH=$(<venv>/bin/python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
  -S . -B <job>/nvcc/build-<tree>
ninja -C <job>/nvcc/build-<tree> -j64 -k 0
```

### The RAFT `CUDA::nvToolsExt` wall: does not occur with this host's toolkit; reproduced on purpose and worked around from the command line

With `/usr/local/cuda` as installed, unmodified upstream **configures and generates** (rc 0): this toolkit ships `include/nvToolsExt.h` and `lib64/libnvToolsExt.so*`, so `FindCUDAToolkit` creates `CUDA::nvToolsExt`, and RAFT only prints the CMake deprecation warning ("nvToolsExt has been superseded by nvtx3"). This host has no `/opt/conda/envs/cuda-*` toolkit and no other CUDA version (`/usr/local/cuda -> cuda-12.8` only).

To check the recorded wall, I made `<job>/nvcc/cuda-nontx`: a symlink copy of `/usr/local/cuda` 12.8 without `libnvToolsExt*` and `nvToolsExt*.h`, keeping `include/nvtx3`. That reproduces the layout of a toolkit with no legacy nvToolsExt. Configuring the base with it (`-DCMAKE_CUDA_COMPILER=<job>/nvcc/cuda-nontx/bin/nvcc -DCUDAToolkit_ROOT=<job>/nvcc/cuda-nontx`, everything else as above) **reproduces the wall on unmodified upstream**: configure runs to the end, then generate fails, rc 1. There are two errors, not one. The first comes from torch, not RAFT: `torch/share/cmake/Caffe2/public/cuda.cmake:182 (set_property): The link interface of target "torch::nvtoolsext" contains: CUDA::nvToolsExt but the target was not found`. The second is `_deps/raft-src/cpp/CMakeLists.txt:262 (target_link_libraries): The link interface of target "raft" contains: CUDA::nvToolsExt ...`. No Quest source is reached. Both are checked only at generate time, so supplying the target from the command line before then is enough, with no Quest source touched. `<job>/nvcc/nvtx-shim.cmake`:

```
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
  target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
endif()
```

passed as `-DCMAKE_PROJECT__kernels_INCLUDE=<job>/nvcc/nvtx-shim.cmake` (run right after the top-level `project(_kernels ...)`). With it, base and port both configure and generate (rc 0) on the nvToolsExt-less toolkit, and `ninja -C <build> -j64 _kernels` builds 8 of 8 in both, rc 0, 0 warnings. The module sizes match the full builds byte for byte (18,681,432 / 18,714,200 B). So step (1) of the fallback order succeeded both ways: the real toolkit needs no workaround, and a toolkit without legacy nvToolsExt needs only the one-target shim.

### Results (real `/usr/local/cuda`, full build)

| | base `01c1623` | port `a70f4d9` |
|---|---|---|
| configure + generate | OK (rc 0), `CMAKE_BUILD_TYPE=Release` | OK (rc 0), `CMAKE_BUILD_TYPE=Release` |
| ninja steps (all) | 162 of 179 succeeded, 15 failed, 2 not run (rc 1) | 162 of 179 succeeded, 15 failed, 2 not run (rc 1) |
| `_kernels` target | 8 of 8 (7 `.cu` + link), 18,681,432 B | 8 of 8, 18,714,200 B |
| wall (both builds in parallel, -j64 each) | 333 s | 329 s |

The 15 failures are the same set in both trees (`diff` of the `FAILED:` lists is empty). All are in RAFT's optional compiled library `raft_objs` (`libraft.so`/`libraft.a` are the 2 steps that never ran), which Quest's module does not link: it uses the header-only `raft::raft`. The files are `src/neighbors/detail/cagra/search_single_cta_{float,int8,uint8}_uint32_dim{128_t8,256_t16,512_t32,1024_t32}.cu` (12) and `src/neighbors/ivfpq_search_{float,int8_t,uint8_t}_int64_t.cu` (3). The first error is identical in both trees:

```
_deps/raft-src/cpp/include/raft/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh(467): error #20281-D: in whole program compilation mode ("-rdc=false"), a __global__ function template instantiation or specialization ("raft::neighbors::cagra::detail::single_cta_search::search_kernel<(unsigned int)8u, ...
```

This is RAFT branch-24.02 against CUDA 12.8, built with `-Werror=all-warnings`, and it is upstream's failure. Each build log has 30 `warning` lines, all inside RAFT's dependency trees (`_deps/{cccl,cuco,fmt,hnswlib,nvidiacutlass,raft,rmm,spdlog}-src`) and all on the failed `raft_objs` lines; none come from `quest/ops/csrc` or `kernels/include`.

### Per-file comparison of Quest's `.cu` files

Each of the 7 `_kernels` compile lines from `ninja -t commands _kernels` was rerun separately in both trees, with output redirected to a scratch `.o`, to get each file's diagnostics on its own. With the tree and build paths replaced by placeholders, the 7 compile lines are byte-identical between base and port (`diff` empty; no `USE_HIP`, `-gencode arch=compute_80,code=sm_80`, `--expt-extended-lambda --expt-relaxed-constexpr`).

| file | base | port |
|---|---|---|
| `csrc/approx_attn.cu` | rc 0, no output | rc 0, no output |
| `csrc/batch_prefill.cu` (includes `prefill/prefill.cuh`, touched) | rc 0, no output | rc 0, no output |
| `csrc/bsk_ops.cu` | rc 0, no output | rc 0, no output |
| `csrc/estimate.cu` (decode headers, touched) | rc 0, no output | rc 0, no output |
| `csrc/page.cu` | rc 0, no output | rc 0, no output |
| `csrc/rms_norm.cu` (touched) | rc 0, no output | rc 0, no output |
| `csrc/topk.cu` (includes `topk/decode_select_k.cuh`, touched) | rc 0, no output | rc 0, no output |

These are all the `.cu` files in the module, so every `.cu` the port touches and every `.cu` that includes a header it touches was compiled through the real CMake build. No file needed a hand-assembled nvcc line, and none failed. The `hip_compat/` shims are not on the CUDA include path. Both modules export the same 651 text symbols, and their undefined dynamic symbol lists are identical (`nm -D`, diffed). This time an import smoke test ran: each `.so`, copied to its own directory, imports under `<venv>/bin/python -I` from `/tmp` after `import torch`, with the same 8 public names in both (`BatchDecodeWithPagedKVCachePyTorchWrapper`, `append_kv_cache_decode`, `append_kv_cache_prefill`, `apply_rope_in_place`, `estimate_attn_score`, `prefill_with_paged_kv_cache`, `rms_norm_forward`, `topk_filtering`). No kernel ran, because there is no GPU.

**Verdict: the CUDA path compiles and links with real nvcc 12.8 at `a70f4d9`, with the same result as upstream `01c1623`. There is no port regression.** The only failures are RAFT's optional compiled library, and they are identical on unmodified upstream. The RAFT/torch `CUDA::nvToolsExt` configure wall reproduces on unmodified upstream with a toolkit that has no legacy nvToolsExt, and a one-target `CMAKE_PROJECT__kernels_INCLUDE` shim clears it for both trees. Setup note: the fleet guard refused `pip install` into a second venv; the CUDA torch wheel went into the job's own `$FLEET_VENV`, which it allows.

### Linux build and tests at `a70f4d9` (linux-gfx942, after the review; evidence only, NOT a validation record)

The validation gate is closed (stage `changes-requested`), so no platform state was written. Recorded so the next round can judge whether a Windows-only fix needs a Linux rebuild. MI300X (`gfx942:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=2` (pinned by the job), torch 2.16.0a0+git6c3af02 / HIP 7.16 on the pip ROCm SDK 10.1, Python 3.11.17, transformers 4.37.2, tokenizers 0.15.2. Fork at `a70f4d9` == `origin/moat-port`, fresh clone, tree clean afterwards.

README Linux command, fresh build directory, `PYTORCH_ROCM_ARCH` unset, no build type passed:

```
cd quest/ops && mkdir build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
ninja      # timeit.sh Quest compile (with ninja -C <build>, since timeit.sh runs from the repository root)
ln -sf $PWD/_kernels*.so ../../
```

`CMAKE_BUILD_TYPE=Release` from the default; 8 compile lines with `-O3`; 8/8, exit 0, 0 errors, 1130 s. Tests, `ulimit -c 0`, wrapped in `timeit.sh Quest test` (`MOAT_PYTHON` set because the test venv on PATH lacks jsonschema):

```
PYTHONPATH=<fork> python -m pytest quest/tests -p no:cacheprovider -q -rs
```

**202 passed, 28 skipped, 0 failed** (21.7 s); skips are the tests' own (`test_prefill_attention.py:51` qo_len > kv_len, 13; `test_topk.py:27` k >= kv_len, 15). Per suite (separate runs): test_approx_attention 42; test_decode_attention 6; test_decode_launch_shape 3; test_estimate 9; test_prefill_attention 29 passed, 13 skipped; test_rope 64; test_topk 49 passed, 15 skipped. Identical to `6e7d8db` on this GPU. End-to-end Llama-2 run not repeated: the delta touches no kernel and no compiled code (the generated `build.ninja` is identical to `6e7d8db`'s, per the review above).

## Porter round 2026-11-19 (windows-gfx1151), answering review pass 3 -- fork `3c7754f`

One commit on top of `a70f4d9`, nothing rewritten: `3c7754f` `[ROCm] Find the ROCm SDK from Python_EXECUTABLE on Windows` (`quest/ops/CMakeLists.txt` +31/-2, `README.md` 1 line). Every CMake change is inside `CMAKE_HOST_WIN32`/`WIN32` guards in the `USE_HIP` branch; CUDA path, kernels, tests and Linux untouched. No validation recorded.

Finding 1 reproduced at `a70f4d9` first (clean shell: venv not activated and not on PATH; `CC`, `CXX`, `HIPCXX`, `PYTORCH_ROCM_ARCH`, `HIP_PATH`, `ROCM_PATH`, `VIRTUAL_ENV`, `CONDA_PREFIX` unset; README arguments plus `-DPython_EXECUTABLE=<venv>/Scripts/python.exe`, prefix path from the same python): rc 1 at `CMakeLists.txt:54`, "CMake was given .../cl.exe ... Unset the CXX environment variable", as the reviewer read it.

### The change

- hipconfig lookup (pre-`project()` Windows block): extra `find_program` hints: the directory of `Python_EXECUTABLE` and its `Scripts` subdirectory when `Python_EXECUTABLE` is given, and `$ENV{VIRTUAL_ENV}/Scripts`, `$ENV{CONDA_PREFIX}/Scripts`.
- A second wall the review did not see, found by running the case: with the compiler found, `find_package(Torch)` still printed "ROCM_PATH environment variable is not set and C:/opt/rocm does not exist. Building without ROCm support." and generation failed on `hip::amdhip64` (`c10_hip`, `torch_hip`, `torch_hip_library` link interfaces). torch's `LoadHIP.cmake` locates ROCm only through `ENV{ROCM_PATH}` or `find_program(rocm-sdk)` on PATH, and `rocm-sdk.exe` is also a venv `Scripts` console script. Fix: after `enable_language(HIP)`, on `WIN32` with `ENV{ROCM_PATH}` undefined, `set(ENV{ROCM_PATH} "${CMAKE_HIP_COMPILER_ROCM_ROOT}")` (the root CMake derived for the HIP compiler; configure-process environment only, re-derived on every re-configure since the compiler is cached). This also covers `-DCMAKE_HIP_COMPILER=...` in a shell with no environment.
- Error messages: when the lookup ran and found no clang++ and CMake fell back to an MSVC-style compiler, the message now says none was found and names the remedies (activate the environment, `-DPython_EXECUTABLE=<its python.exe>`, or `-DCMAKE_HIP_COMPILER=<ROCm>/lib/llvm/bin/clang++.exe`). The "CMake was given ... Unset the CXX" message remains for a compiler the user chose.
- README: the `-DPython_EXECUTABLE` sentence adds "and use that same `python.exe` in the `CMAKE_PREFIX_PATH` command; the ROCm SDK installed in that environment is then found from it" (the documented `$(python -c ...)` would otherwise run whichever python is on PATH).

### Evidence (torch312_venv, CMake 3.31.0, Ninja; scratch in `agent_space/quest_porter/`; out-of-tree `-S quest/ops -B <scratch>`)

Case 1, environment not activated and not on PATH, the variables above unset:

```
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 "-DCMAKE_PREFIX_PATH=<venv python: torch.utils.cmake_prefix_path>" -DPython_EXECUTABLE=<venv>/Scripts/python.exe -S quest/ops -B b1
ninja -C b1 -j2 -v CMakeFiles/_kernels.dir/csrc/rms_norm.cu.obj CMakeFiles/_kernels.dir/csrc/bsk_ops.cu.obj     # utils/timeit.sh Quest compile
```

Configure rc 0: `CMAKE_CXX_COMPILER` and `CMAKE_HIP_COMPILER` both `<venv>/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe` (GNU-like), `CMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`), Python `<venv>/Scripts/python.exe` 3.12.10, `_quest_hipconfig=<venv>/Scripts/hipconfig.exe`, gfx1151. The two objects compiled, 2/2, rc 0, `--offload-arch=gfx1151` on both, warnings only (the known `hipError_t` nodiscard ones). `build.ninja` equals the `a70f4d9` build's (`quest/ops/build`, the previous round) after replacing the build directory: the only differing lines are the 6 that spell the build directory with backslashes.

Case 2, environment activated (`Activate.ps1` from the same clean shell, `case2_configure.ps1`), README command as written (source directory given by path instead of `..`): rc 0. `CMakeCache.txt` against the `a70f4d9` build's, build directory normalized: 2 differing entries, `CMAKE_CACHEFILE_DIR` (the directory) and `_quest_hipconfig` (`<venv>/Scripts/hipconfig.exe` now; the previous round's shell also had the SDK `bin` on PATH and found `_rocm_sdk_devel/bin/hipconfig.exe`; both report the same clang path). Compilers, build type, flags, Python and every `*_DIR` identical. `build.ninja`: same 6 path-only lines. So **no full rebuild and no pytest rerun**: the generated build is the one `a70f4d9` built and tested (202 passed, 28 skipped), differing in paths only.

Messages: no environment and no `-DPython_EXECUTABLE` -> the new "none was found (no hipconfig on PATH or in the Python environment) ..." error; `CXX=cl` -> the "CMake was given ... (MSVC-style command line). Unset the CXX ..." error as before. `-DCMAKE_HIP_COMPILER=<sdk>/lib/llvm/bin/clang++.exe` with `-DPython_EXECUTABLE`, no environment: configures, both compilers that clang++.

`python utils/jargon.py --port Quest` clean; commit ASCII, no trailer; fork tree clean apart from the untracked runtime DLLs staged in `quest/` and the old `quest/ops/build`. Linux hosts: the delta is Windows-guarded configure logic plus one README sentence.

## Lesson for cuda-to-rocm: find_package(Torch) on Windows needs ROCM_PATH when the environment is not activated

Promoted to cuda-to-rocm/references/windows.md in 37d13004.

For `references/windows.md`, "CMake and compiler drivers", beside the `hipconfig --hipclangpath` recipe. Finding the HIP `clang++` through the Python interpreter's directory is not enough for a CMake torch extension configured with an unactivated pip ROCm environment (`-DPython_EXECUTABLE=<venv>/Scripts/python.exe`): torch's `LoadHIP.cmake` locates ROCm only through `ENV{ROCM_PATH}` or `find_program(rocm-sdk)` on PATH (`rocm-sdk.exe` is a venv `Scripts` console script too), prints "ROCM_PATH environment variable is not set and C:/opt/rocm does not exist. Building without ROCm support." as a STATUS line, and the configure only fails at generate time with `The link interface of target "c10_hip" contains: hip::amdhip64 but the target was not found`. Fix in the project, on `WIN32`, after `enable_language(HIP)` and before `find_package(Torch)`: when `ENV{ROCM_PATH}` is undefined, `set(ENV{ROCM_PATH} "${CMAKE_HIP_COMPILER_ROCM_ROOT}")`. Test the documented non-activated fallback in a shell that has neither the environment's `Scripts` nor the SDK `bin` on PATH; a build shell that exports them hides both this and the hipconfig miss. (Quest)

## Review 2026-11-19 pass 4 (linux-gfx942, delta 6e7d8db..3c7754f) -- CHANGES REQUESTED

### Summary

`a70f4d9` and `3c7754f` make the README's ROCm configure work on Windows, activated or through `-DPython_EXECUTABLE`. Pass 3's finding (hipconfig from `Python_EXECUTABLE`) is closed by the new hints at `quest/ops/CMakeLists.txt:33-43` plus the `ROCM_PATH` hand-off at `:115-117`. One new finding, on Linux, where the moved Release default overrides a build type given through the environment.

### Build System / Backward Compatibility

1. **`quest/ops/CMakeLists.txt:19`: the Release default, now run before `project()`, overrides the `CMAKE_BUILD_TYPE` environment variable on Linux (and Windows).** CMake (3.22+) documents that variable as "a default value for the `CMAKE_BUILD_TYPE` variable when there is no explicit configuration given on the first run". CMake applies it inside `project()`. At 6e7d8db the check ran after `project()`, saw the environment's value and left it alone. Now it runs first, sees no cache entry, and FORCEs Release. Reproduced on this host: fresh build directory, `CMAKE_BUILD_TYPE=Debug cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=<torch> -S quest/ops -B <dir>` caches **Debug at 6e7d8db and Release at 3c7754f**. This contradicts the claim in `a70f4d9` that Linux "behaves as before". The fix is to also require the variable to be empty: `if(NOT CMAKE_BUILD_TYPE AND "$ENV{CMAKE_BUILD_TYPE}" STREQUAL "" AND NOT _quest_multi_config)`. Checked in a scratch copy of 3c7754f with only that line changed (never committed): env Debug -> Debug; nothing given -> Release; env set but empty -> Release; `-DCMAKE_BUILD_TYPE=RelWithDebInfo` -> RelWithDebInfo; the default-case `build.ninja` is identical to 3c7754f's. On Windows the same condition lets an environment `CMAKE_BUILD_TYPE` win over `Windows-Clang.cmake`'s `CMAKE_BUILD_TYPE_INIT Debug`, as the user intends. The lesson "a pip ROCm torch on Windows needs PYTORCH_ROCM_ARCH ..." recommends the same idiom ("`if(NOT CMAKE_BUILD_TYPE)` ... must run before `project()`"), so correct it in the same round. Otherwise the skill would teach the regression.

### Checked on this host, no finding (evidence for the next round)

linux-gfx942 (MI300X), CMake 3.31.6, torch 2.16.0a0+git6c3af02 / HIP 7.16, pip ROCm SDK 10.1. Configure only, fresh build directory each, `-GNinja -DUSE_HIP=ON -DCMAKE_PREFIX_PATH=<torch>`, `PYTORCH_ROCM_ARCH` and `CMAKE_BUILD_TYPE` unset unless stated:

| configure | 6e7d8db | 3c7754f |
|---|---|---|
| `-DCMAKE_HIP_ARCHITECTURES=gfx942`, no build type | Release | Release |
| same, `-DCMAKE_BUILD_TYPE=` (empty) | Release | Release |
| same, `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | -- | RelWithDebInfo |
| same, `-DCMAKE_BUILD_TYPE=Debug` | -- | Debug |
| same, env `CMAKE_BUILD_TYPE=Debug` | **Debug** | **Release** (finding 1) |
| no architecture (auto-detect) | -- | Release, 8 lines `--offload-arch=gfx942` |
| `-G "Ninja Multi-Config"` | no `CMAKE_BUILD_TYPE`, Debug;Release;RelWithDebInfo | same |

- With no build type, `CMakeCache.txt` and `build.ninja` are identical between 6e7d8db and 3c7754f after normalizing the source and build paths. The default Linux build is unchanged, and the README Linux command still yields Release with `-O3` on all 8 compile lines.
- `--trace-expand --trace-source=quest/ops/CMakeLists.txt` at 3c7754f: every Windows guard evaluates false and no body is entered: `:25` (`CMAKE_HOST_WIN32 AND ...`, which contains all of the hipconfig/Python hint logic), `:70`/`:77` (MSVC-variant errors), `:98` (no-architecture error), `:109` (`PYTORCH_ROCM_ARCH`), `:115` (`ROCM_PATH`). C++ stays `/usr/bin/c++`.
- CUDA path: `git diff 6e7d8db..3c7754f` adds an `else()` to the existing `if(NOT USE_HIP)` before `project()` and otherwise edits only inside `if(USE_HIP)`. No CUDA line changed (the nvcc compile at a70f4d9 above agrees, and 3c7754f adds no CUDA-visible line).
- `jargon.py --port Quest` clean. `audit-commits` reports only the known `6e7d8db` fenced-Test-Plan gap carried to PR preparation. `a70f4d9` and `3c7754f`: `[ROCm]` titles of 62 and 58 chars, fenced Test Plans, AI disclosure, no trailer, ASCII.

### Recommendation

**Request Changes** (finding 1). The fix is one condition in the pre-`project()` block plus the lesson text. For the default Linux configure it produces the same `build.ninja` (checked above). The Linux build and tests at a70f4d9 (202 passed, 28 skipped), whose `build.ninja` also equals 6e7d8db's, therefore describe the compiled module, and the next Linux validation can judge whether to carry forward on that basis.

## Porter round 2026-11-19 (linux-gfx942), answering review pass 4 -- fork `2b938d8`

One commit on top of `3c7754f`, nothing rewritten: `2b938d8` `[ROCm] Keep a build type given in the environment` (`quest/ops/CMakeLists.txt` +3/-2: the comment and the one condition). The condition is the review's: `if(NOT CMAKE_BUILD_TYPE AND "$ENV{CMAKE_BUILD_TYPE}" STREQUAL "" AND NOT _quest_multi_config)`. Kept as is: an empty environment value behaves like an unset one in CMake (same cached result), `NOT CMAKE_BUILD_TYPE` still catches `-DCMAKE_BUILD_TYPE=` (empty), and the multi-config guard is unchanged. The Windows-guarded blocks are not touched. The pending lesson ("a pip ROCm torch on Windows needs PYTORCH_ROCM_ARCH ...") now spells the corrected idiom.

Host: linux-gfx942 (MI300X), CMake 3.31.6 (pip), torch 2.16.0a0+git6c3af02 / HIP 7.16 on the pip ROCm SDK, `/opt/python-3.11-venv`. Configure only, fresh build directory each, `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=<torch cmake prefix> -S <tree>/quest/ops`, `CMAKE_BUILD_TYPE` and `PYTORCH_ROCM_ARCH` unset unless stated. Trees: the fork at `2b938d8` and a worktree at `6e7d8db` (`agent_space/porter/cases.sh <tree> <outdir>`, both trees in parallel):

| configure | `6e7d8db` | `3c7754f` | `2b938d8` |
|---|---|---|---|
| `-GNinja`, nothing given | Release | (Release, review pass 4) | Release |
| env `CMAKE_BUILD_TYPE=Debug` | Debug | **Release** (rerun here) | Debug |
| env `CMAKE_BUILD_TYPE=` (empty) | Release | -- | Release |
| `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | RelWithDebInfo | -- | RelWithDebInfo |
| env Debug plus `-DCMAKE_BUILD_TYPE=MinSizeRel` | MinSizeRel | -- | MinSizeRel |
| `-G "Ninja Multi-Config"` | none; Debug;Release;RelWithDebInfo | -- | same |
| `-G "Ninja Multi-Config"`, env Debug | none; Debug;Release;RelWithDebInfo | -- | same |

All configures rc 0. For the default, env-Debug, RelWithDebInfo and multi-config cases, `CMakeCache.txt` and `build.ninja` are identical between `6e7d8db` and `2b938d8` once the build directory and then the source directory are replaced by placeholders. Default case: 8 compile lines with `-O3`, 8 with `--offload-arch=gfx942`. So the default generated build is the one `6e7d8db` produced (and `a70f4d9` was built and tested from: 202 passed, 28 skipped). Not run: Windows (whether an environment `CMAKE_BUILD_TYPE` beats `Windows-Clang.cmake`'s `CMAKE_BUILD_TYPE_INIT Debug` there is the review's reading, unverified on hardware; with nothing in the environment the Windows behaviour is unchanged from `3c7754f`).

`python3 utils/jargon.py --port Quest` clean; `utils/prose.py` on the commit message clean; title 48 chars, fenced Test Plan, AI disclosure, no trailer, ASCII. Pushed plainly (`3c7754f..2b938d8`); fork tree clean.

## Review 2026-11-19 pass 5 (linux-gfx942, delta 6e7d8db..2b938d8) -- PASS

### Summary

Delta `a70f4d9`, `3c7754f`, `2b938d8` on top of the last reviewed head `6e7d8db` (merge base with upstream main `01c1623`, 0 behind). Pass 4's finding is fixed: `quest/ops/CMakeLists.txt:20` now reads `if(NOT CMAKE_BUILD_TYPE AND "$ENV{CMAKE_BUILD_TYPE}" STREQUAL "" AND NOT _quest_multi_config)`, and the comment at `:15-18` says why. The pending lesson "a pip ROCm torch on Windows needs PYTORCH_ROCM_ARCH ..." now spells that same corrected idiom. Pass 3's finding stays closed (hints at `:33-43`, `ROCM_PATH` hand-off at `:116-118`). No new findings.

### Checked on this host, no finding

linux-gfx942 (MI300X), CMake 3.31.6, torch 2.16.0a0+git6c3af02 / HIP 7.16 (`/opt/python-3.11-venv`). I reran the porter's `agent_space/porter/cases.sh` myself into `agent_space/reviewer5/{old,new}`: the fork at `2b938d8` and the worktree at `6e7d8db`, configure only, fresh build directory each, `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_PREFIX_PATH=<torch>`. `CMAKE_BUILD_TYPE` and `PYTORCH_ROCM_ARCH` were unset unless stated. All runs rc 0:

| configure | `6e7d8db` | `2b938d8` |
|---|---|---|
| `-GNinja`, nothing given | Release | Release |
| env `CMAKE_BUILD_TYPE=Debug` | Debug | Debug (Release at `3c7754f`, pass 4) |
| env `CMAKE_BUILD_TYPE=` (empty) | Release | Release |
| `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | RelWithDebInfo | RelWithDebInfo |
| env Debug plus `-DCMAKE_BUILD_TYPE=MinSizeRel` | MinSizeRel | MinSizeRel |
| `-G "Ninja Multi-Config"`, with and without env Debug | none; Debug;Release;RelWithDebInfo | same |

- In the default, env-Debug, RelWithDebInfo and multi-config cases, `CMakeCache.txt` and `build.ninja` are identical between the two heads once the source and build paths are normalized. The default build has 8 compile lines with `-O3` and 8 with `--offload-arch=gfx942`, and C++ is `/usr/bin/c++`, so the default Linux build is unchanged.
- `--trace-expand --trace-source=quest/ops/CMakeLists.txt` at `2b938d8`: every Windows guard evaluates false and no body is entered: `:26` (`CMAKE_HOST_WIN32 ...`, which holds all the hipconfig/Python hint logic), `:71`/`:78` (MSVC-variant errors), `:99` (no-architecture error), `:110` (`PYTORCH_ROCM_ARCH`), `:116` (`ROCM_PATH`).
- Windows claim in `2b938d8` and in the lesson, that an environment `CMAKE_BUILD_TYPE` wins over `Windows-Clang.cmake:238`'s `CMAKE_BUILD_TYPE_INIT Debug`: this cannot run on Linux, so I checked it with a proxy. In a toy project, `CMAKE_USER_MAKE_RULES_OVERRIDE` set `CMAKE_BUILD_TYPE_INIT MinSizeRel`. With no environment value it gave MinSizeRel; with env `CMAKE_BUILD_TYPE=Debug` it gave Debug. The environment value is cached before `CMakeInitializeConfigs.cmake` makes its non-FORCE `set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE_INIT}" CACHE ...)`, which is the same path the Windows platform file's INIT takes. This is consistent with the claim but was not run on Windows.
- CUDA path: `git diff 6e7d8db..2b938d8` edits only the new `else()` (USE_HIP) branch before `project()` and the inside of `if(USE_HIP)`. No CUDA, rapids, raft or gcc-11 line changed. `2b938d8` touches 3 lines of the HIP pre-`project()` block that a CUDA configure never enters, so the real-nvcc compile at `a70f4d9` still describes the CUDA path.
- `jargon.py --port Quest` clean. `audit-commits Quest` reports only the known `6e7d8db` fenced-Test-Plan gap, which is carried to PR preparation. Commit titles are 49/58/61 chars (`2b938d8`/`3c7754f`/`a70f4d9`), all `[ROCm]` with HIP-specific added lines. Each has a fenced Test Plan and an AI disclosure, and none has a Co-Authored-By trailer. Fork tree clean (`git ls-files -m` empty).

### Recommendation

**Pass.** The Windows behaviour was validated by the windows-gfx1151 porter rounds (activated and `-DPython_EXECUTABLE` cases, build.ninja path-only equal to the `a70f4d9` build, which passed 202 / skipped 28). For Linux the delta is configure-only, with an identical generated build.

## Validation 2026-11-19 (linux-gfx942, at `2b938d8`)

AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, wave64, `HIP_VISIBLE_DEVICES=2` (pinned by the job). PyTorch 2.16.0a0+git6c3af02 (`torch.version.hip` 7.16.26385) on the pip ROCm SDK (no `/opt/rocm`), Python 3.11.17 (`/opt/python-3.11-venv`), CMake 3.31.6. Fork at `2b938d8` == `origin/moat-port` == `head_sha`. Review passed at this sha (pass 5). This is a full revalidation, not a carry-forward (`classify 6e7d8db 2b938d8`: `mixed`, `quest/ops/CMakeLists.txt`).

### Build

The README Linux command, with no build directory before it (`rm -rf build` first), `CMAKE_BUILD_TYPE` and `PYTORCH_ROCM_ARCH` unset, no `-DPython_EXECUTABLE`, wrapped `utils/timeit.sh Quest compile -- bash agent_space/porter/build.sh`:

```
cd quest/ops
mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

The script exited 0 and ninja built 8 of 8 in 1142 s. `CMAKE_BUILD_TYPE:STRING=Release` came from the default, and all 8 compile lines carry `-O3` and `--offload-arch=gfx942`. There were 0 errors. The warnings were 18 nodiscard-`hipError_t` warnings (`-Wunused-value`) and 1536 `-Wpass-failed=transform-warning` "loop not unrolled" remarks. Both come from the compile commands, which are unchanged from `6e7d8db`: the porter round and review pass 5 showed `build.ninja` is identical after path normalization. Earlier records counted only the source-level warnings.

### GPU tests (transformers 4.37.2, tokenizers 0.15.2, pytest 7.3.2)

transformers, tokenizers, huggingface-hub 0.24.7, safetensors and regex were installed with `--no-deps --target` into the job directory (`<job>/pylib-test`) through `$FLEET_VENV`'s pip. The fleet guard refuses pip for the shared `/opt` environment, including with `--target`. Wrapped `utils/timeit.sh Quest test`, `ulimit -c 0`:

```
PYTHONPATH=<fork>:<job>/pylib-test python -m pytest quest/tests -p no:cacheprovider -q -rs
```

**202 passed, 28 skipped, 0 failed** (24.5 s). Each suite was then run on its own:

| suite | result |
|---|---|
| test_approx_attention | 42 passed |
| test_decode_attention | 6 passed |
| test_decode_launch_shape | 3 passed |
| test_estimate | 9 passed |
| test_prefill_attention | 29 passed, 13 skipped |
| test_rope | 64 passed |
| test_topk | 49 passed, 15 skipped |

The skips are the tests' own (`test_prefill_attention.py:51` qo_len > kv_len, `test_topk.py:27` k >= kv_len). The results are identical to `6e7d8db` and `a70f4d9` on this GPU.

The end-to-end Llama-2 run was not repeated. No kernel or compiled source changed since `6e7d8db`, where it ran (byte-identical text across hg and quest at budgets 1024 and 128). The only change since then is configure logic, and the default configure generates the same `build.ninja`.

### CUDA no-regression gate

`cuda-not-validated: no CUDA toolkit on this host` (no nvcc, `/usr/local/cuda` or `/opt/conda`). Real nvcc compiled `a70f4d9` on the gfx950 host and found no port regression (above). `3c7754f` and `2b938d8` add lines only inside `if(USE_HIP)` and its pre-`project()` `else()` branch, so the CUDA configure sees no new line. This was checked by reading; this host did not recompile.

### Content checks and integrity

`jargon.py --port Quest` is clean. The ROCm build is documented in `README.md` beside the CUDA build, and the command above is that documentation verbatim apart from the architecture. `git -C projects/Quest/src status --porcelain` was empty at completion (the `_kernels` symlink and `build/` are ignored).

## CUDA no-regression compile 2026-11-19 (linux-gfx950 host, real nvcc, at `2b938d8`) -- PASS, no port regression

Asked for by Jeff Daily before the port is offered upstream: show that the CUDA path still compiles with real nvcc at the current `head_sha`. Evidence only; no stage or platform record changed, nothing pushed to the fork, no GitHub write. Same recipe as the `a70f4d9` run above, from fresh clones in this job's directory (`<job>/nvcc`).

Trees (fresh `git clone` + `git submodule update --init --recursive`; identical submodule SHAs in both: flashinfer `9f49803b`, its googletest `40412d85` and nvbench `f57aa9c9`, googletest `6a593823`, nvbench `d8dced8a`, pybind `768cebe1`, raft `1e4961e2`):

- port: `AMD-Ecosystem/Quest` at `2b938d8ab5eea6b19d522a53681a68819bf04dca` (= `head_sha`)
- base: `mit-han-lab/Quest` at `01c1623bf9395009520874e989e29f683203b357` (`git merge-base up/main 2b938d8`; also upstream `main` today, port 0 behind)

The port touches 18 files against the base. Since `a70f4d9` only `quest/ops/CMakeLists.txt` (+33/-3) and `README.md` (1 line) changed.

Toolchain: CUDA 12.8 at `/usr/local/cuda` (`nvcc` V12.8.93, build cuda_12.8.r12.8/compiler.35583870_0; the only toolkit on this host), CMake 3.31.6, Ninja 1.13.0, g++-11 11.5.0 (`11.5.0-1ubuntu1~24.04.1`, unpacked debs copied into the job directory), PyTorch `2.5.0+cu124` (the `pyproject.toml` pin, `pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124` into `$FLEET_VENV`), Python 3.11.17. No NVIDIA GPU: compile and link only. Upstream has no CI; the recipe is its `quest/ops/setup.sh` (`cmake -GNinja` + `ninja`) with the three host-forced changes, applied to both trees: a toolchain file for the hard-coded `/usr/bin/gcc-11`, `-DCMAKE_CUDA_ARCHITECTURES=80` (`native` needs a GPU), `TORCH_CUDA_ARCH_LIST=8.0`.

```
# <job>/nvcc/gcc11.cmake sets CMAKE_C_COMPILER, CMAKE_CXX_COMPILER, CMAKE_CUDA_HOST_COMPILER to <job>/nvcc/gcc11/root/usr/bin/{gcc-11,g++-11}
# run.sh <tree> <builddir> <cudaroot> [extra]:
cd <job>/nvcc/<tree>/quest/ops
TORCH_CUDA_ARCH_LIST=8.0 PATH=<venv>/bin:<cudaroot>/bin:$PATH cmake -GNinja \
  -DCMAKE_TOOLCHAIN_FILE=<job>/nvcc/gcc11.cmake -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=<cudaroot>/bin/nvcc -DCUDAToolkit_ROOT=<cudaroot> -DPython_EXECUTABLE=<venv>/bin/python \
  -DCMAKE_PREFIX_PATH=$(<venv>/bin/python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
  [extra] -S . -B <job>/nvcc/<builddir>
ninja -C <job>/nvcc/build-<tree> -j64 -k 0
```

### The RAFT `CUDA::nvToolsExt` wall

With the real `/usr/local/cuda`, unmodified upstream **configures and generates** (rc 0, `CMAKE_BUILD_TYPE=Release`): CUDA 12.8 here still ships `nvToolsExt.h` and `libnvToolsExt.so*`, so `CUDA::nvToolsExt` exists and RAFT only prints the deprecation warning. The wall therefore does not occur on this host's toolkit, and there is no older toolkit to try.

To reproduce it, `<job>/nvcc/cuda-nontx` is a symlink copy of `/usr/local/cuda-12.8` whose `targets/x86_64-linux/{lib,include}` (and so `lib64`, `include`) omit `libnvToolsExt*` and `nvToolsExt*.h`, keeping `include/nvtx3`. Note for whoever repeats this: a first copy that replaced only the top-level `include`/`lib64` leaked, because `FindCUDAToolkit` still found `targets/x86_64-linux/lib/libnvToolsExt.so` (only the torch error appeared); the `targets` directory has to be stripped too. With the correct copy (`run.sh <tree> build-wall-<tree> <job>/nvcc/cuda-nontx`), **both base and port** run configure to the end and fail at generate, rc 1, with the same two errors (diff of the error blocks empty after path normalization), no Quest source reached:

```
CMake Error at <venv>/.../torch/share/cmake/Caffe2/public/cuda.cmake:182 (set_property):
  The link interface of target "torch::nvtoolsext" contains: CUDA::nvToolsExt but the target was not found.
CMake Error at <build>/_deps/raft-src/cpp/CMakeLists.txt:262 (target_link_libraries):
  The link interface of target "raft" contains: CUDA::nvToolsExt ...
```

Step (1) of the fallback order works without touching Quest sources: `<job>/nvcc/nvtx-shim.cmake`

```
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED GLOBAL)
  target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
endif()
```

passed as `-DCMAKE_PROJECT__kernels_INCLUDE=<job>/nvcc/nvtx-shim.cmake`. Base and port then configure and generate on `cuda-nontx` (rc 0), and `ninja -C build-shim-<tree> -j32 _kernels` builds 8 of 8 in both, rc 0. The modules are byte-for-byte the same size as the real-toolkit full builds (18,681,432 / 18,714,200 B). The nvcc path step (2) was not needed.

### Results (real `/usr/local/cuda`, full build, all targets)

| | base `01c1623` | port `2b938d8` |
|---|---|---|
| configure + generate | OK (rc 0), Release | OK (rc 0), Release |
| ninja steps (all, 179 total) | 162 succeeded, 15 failed, 2 not run (rc 1) | 162 succeeded, 15 failed, 2 not run (rc 1) |
| `_kernels` target | 8 of 8 (7 `.cu` + link), 18,681,432 B | 8 of 8, 18,714,200 B |
| wall (both in parallel, -j64 each) | 344 s | 341 s |

The 15 failures are the same set in both trees (sorted `FAILED:` lists diff empty): RAFT's optional compiled `raft_objs` (`libraft.so`/`libraft.a` are the 2 that never ran), which Quest's module does not link (it uses header-only `raft::raft`): `src/neighbors/detail/cagra/search_single_cta_{float,int8,uint8}_uint32_dim{128_t8,256_t16,512_t32,1024_t32}.cu` (12) and `src/neighbors/ivfpq_search_{float,int8_t,uint8_t}_int64_t.cu` (3). First error, identical in both:

```
<build>/_deps/raft-src/cpp/include/raft/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh(467): error #20281-D: in whole program compilation mode ("-rdc=false"), a __global__ function template instantiation or specialization ("raft::neighbors::cagra::detail::single_cta_search::search_kernel<...
```

That is RAFT branch-24.02 under CUDA 12.8 with `-Werror=all-warnings`: upstream's failure. Each log has 30 warning lines, all from RAFT's dependency trees on the failed `raft_objs` lines; 0 warning or error lines name `quest/ops/csrc` or `kernels/include`.

### Per-file comparison of Quest's `.cu` files

The 7 compile lines of `ninja -t commands _kernels` are byte-identical between trees after replacing tree and build paths (diff empty; no `USE_HIP`; `-gencode arch=compute_80,code=sm_80`; `-O3` on all 7). Each was rerun on its own in both trees with `-o` redirected to a scratch object:

| file | base | port |
|---|---|---|
| `csrc/approx_attn.cu` | rc 0, no output, 2,453,208 B | rc 0, no output, 2,526,512 B |
| `csrc/batch_prefill.cu` | rc 0, no output | rc 0, no output |
| `csrc/bsk_ops.cu` | rc 0, no output | rc 0, no output |
| `csrc/estimate.cu` | rc 0, no output | rc 0, no output |
| `csrc/page.cu` | rc 0, no output | rc 0, no output |
| `csrc/rms_norm.cu` (touched) | rc 0, no output | rc 0, no output |
| `csrc/topk.cu` | rc 0, no output | rc 0, no output |

Every one of them includes `bsk_ops.h`, which includes the touched `decode/decode_handler.cuh`, `prefill/prefill.cuh` and `topk/decode_select_k.cuh`, so all 7 are "touched or include a touched header", and all were compiled through the real CMake build; no file needed a hand-assembled nvcc line and none could not be compiled. Only `approx_attn.cu`'s object differs in size (the decode-header changes); the other six objects are the same size in both trees. `hip_compat/` is not on the CUDA include path. Both modules export the same 651 text symbols and identical undefined dynamic symbol lists (`nm -D`, diffed). Import smoke test: each `.so` copied to its own directory imports under `<venv>/bin/python -I` from `/tmp` after `import torch`, exposing the same 8 public names (`BatchDecodeWithPagedKVCachePyTorchWrapper`, `append_kv_cache_decode`, `append_kv_cache_prefill`, `apply_rope_in_place`, `estimate_attn_score`, `prefill_with_paged_kv_cache`, `rms_norm_forward`, `topk_filtering`). No kernel ran (no GPU).

**Verdict: the CUDA path compiles and links with real nvcc 12.8 at `2b938d8`, with the same result as upstream `01c1623`. No port regression.** The only build failures are RAFT's optional compiled library, identical on unmodified upstream; the `CUDA::nvToolsExt` configure wall reproduces identically on both trees with a toolkit lacking legacy nvToolsExt and is cleared for both by the one-target shim.

## Validation 2026-11-19 (linux-gfx1201, at `2b938d8`)

AMD Radeon AI PRO R9700, `gcnArchName` `gfx1201`, wave32, 32 CUs, 31.9 GiB VRAM; `HIP_VISIBLE_DEVICES=1` (pinned by the job). PyTorch 2.16.0a0+git6c3af02 (`torch.version.hip` 7.16.26385) on the pip ROCm SDK (no `/opt/rocm`; HIP compiler `_rocm_sdk_devel/lib/llvm/bin/clang++`), Python 3.11.17 (`/opt/python-3.11-venv`), CMake 3.31.6. The job venv has no torch, so the image interpreter ran the build and tests (read-only). Fork cloned fresh at `2b938d8` == `origin/moat-port` == `head_sha`; submodules initialized with `git submodule update --init --depth 1` (same SHAs as the gfx950 nvcc record). Review passed at this sha (pass 5). Full revalidation, not a carry-forward: `classify 6e7d8db 2b938d8` is `mixed` (`quest/ops/CMakeLists.txt`, token count differs).

### Build

The README Linux command with the architecture changed, in a fresh build directory (`rm -rf build` first), `CMAKE_BUILD_TYPE` and `PYTORCH_ROCM_ARCH` unset, no `-DPython_EXECUTABLE`, image Python first on `PATH`, wrapped `utils/timeit.sh Quest compile -- bash agent_space/build.sh <fork>`:

```
cd quest/ops
mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
ninja -j 12
ln -sf $PWD/_kernels*.so ../../
```

Only change from the README is `-j 12` on ninja, because the fleet job caps `MAX_JOBS` at 12 on this shared host. rc 0, 8 of 8 in 615 s (6e7d8db on this GPU: 10m04s at `-j 8`). `CMAKE_BUILD_TYPE:STRING=Release` from the default; all 8 compile lines carry `-O3` and `--offload-arch=gfx1201` (no other arch); C++ is `/usr/bin/c++`. 0 errors. Warnings: 18 nodiscard-`hipError_t` (`-Wunused-value`) and 1536 `-Wpass-failed=transform-warning` "loop not unrolled" lines, the same counts as the linux-gfx942 build at this head.

### GPU tests (transformers 4.37.2, tokenizers 0.15.2, huggingface-hub 0.24.7, pytest 7.3.2)

Test dependencies installed with `$FLEET_VENV/bin/pip install --no-deps --target <job>/pylib-test` (the shared environment stays untouched). Wrapped `utils/timeit.sh Quest test`, `ulimit -c 0`:

```
PYTHONPATH=<fork>:<job>/pylib-test python -m pytest quest/tests -p no:cacheprovider -q -rs
```

**202 passed, 28 skipped, 0 failed** (18.9 s). Each suite on its own, beside this GPU's `6e7d8db` run:

| suite | `2b938d8` | `6e7d8db` (gfx1201) |
|---|---|---|
| test_approx_attention | 42 passed | 42 passed |
| test_decode_attention | 6 passed | 6 passed |
| test_decode_launch_shape | 3 passed | 3 passed |
| test_estimate | 9 passed | 9 passed |
| test_prefill_attention | 29 passed, 13 skipped | 29 passed, 13 skipped |
| test_rope | 64 passed | 64 passed |
| test_topk | 49 passed, 15 skipped | 49 passed, 15 skipped |

The skips are the tests' own conditions (`test_prefill_attention.py:51` qo_len > kv_len, `test_topk.py:27` k >= kv_len). Identical to `6e7d8db` here and to linux-gfx942 at `2b938d8`.

The end-to-end Llama-2 run was not repeated: no kernel or compiled source changed since `6e7d8db`, where it ran on this GPU (byte-identical text for hg and quest at budgets 1024 and 128, md5 `b900af27eeb4170a5c47bc5940a2ca60`, same as gfx942). The delta is configure logic only, and the default configure builds the same 8 compile lines (`-O3`, the one offload arch).

### CUDA no-regression gate

Not rerun: already recorded at this head (real nvcc 12.8 on the linux-gfx950 host, `2b938d8`, PASS, no port regression). This host has no CUDA toolkit anyway.

### Content checks and integrity

`jargon.py --port Quest` clean. `audit-commits Quest` reports only the known `6e7d8db` fenced-Test-Plan gap carried to PR preparation. ROCm build documented in `README.md` beside the CUDA build; the command above is that documentation verbatim apart from the architecture and the job cap. `git -C projects/Quest/src status --porcelain` empty (`build/` and the `_kernels` symlink are ignored). No fork edit, no push to the fork, no GitHub write.

State: linux-gfx1201 `completed` at `2b938d8` (wave32 gate).

## Validation 2026-11-19 (windows-gfx1151, at `2b938d8`)

AMD Radeon 8060S (gfx1151, RDNA3.5 APU, wave32, 20 CUs), Windows 11. `torch312_venv`: torch 2.12.0+rocm7.14.0a20260519 (`torch.version.hip` 7.13.26190), ROCm SDK 7.14 devel (`_rocm_sdk_devel`, HIP 7.13.26190-c2c4b25f, AMD clang 23.0.0), Python 3.12.10, transformers 4.37.2, tokenizers 0.15.2, pytest 9.1.1. CMake 3.31.0, Ninja 1.13.2, VS 2022 BuildTools 14.44 x64 environment loaded (LIB/INCLUDE/PATH). Fork at `2b938d8` == `origin/moat-port` == `head_sha` (review pass 5). Full revalidation, not a carry-forward (the delta `6e7d8db..2b938d8` is `mixed`, and Windows binaries have no binary-equivalence verdict). No fork edit, no push. Scripts and logs: `agent_space/quest_porter/val_2b938d8/`.

### Build: the README's Windows command as written, in-tree

The earlier Windows proof builds of this command used out-of-tree build directories or a shell that had the SDK `bin` on PATH; this run is the README's in-tree `quest\ops\build` from a shell with only the x64 build tools environment and the activated Python environment. Before it, the old `quest/ops/build` (the `a70f4d9` build) and the old `quest/_kernels.cp312-win_amd64.pyd` were moved out of the tree (`val_2b938d8/old/`), so the tests can only load the module built here. PowerShell (`powershell -NoProfile -File readme_build.ps1`, wrapped in `utils/timeit.sh Quest compile`); the script removes `CC`, `CXX`, `HIPCXX`, `PYTORCH_ROCM_ARCH`, `CMAKE_BUILD_TYPE` from the environment, runs the venv's `Activate.ps1`, and adds only `$LASTEXITCODE` checks to the README block. `HIP_PATH` and `ROCM_PATH` were also unset; neither the venv `Scripts` nor the SDK `bin` was on PATH before activation.

```
cd quest\ops
mkdir build; cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 `
  "-DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" ..
ninja
copy _kernels*.pyd ..\..\
```

rc 0, 8 of 8, **1093 s** (18m13s) for configure plus build, nothing else building on the host (`6e7d8db`: 991 s at `-j 8` with hand-given flags; `a70f4d9`: 1520 s with a concurrent configure). Selected: C++ and HIP compilers both `<venv>/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe` (Clang 23.0.0, GNU-like command line), `CMAKE_BUILD_TYPE=Release` from the default (`-O3` on all 8 compile lines), Python `<venv>/Scripts/python.exe` 3.12.10, `_quest_hipconfig=<venv>/Scripts/hipconfig.exe`, "Building PyTorch for GPU arch: gfx1151", `--offload-arch=gfx1151` on all 8 compile lines. 0 errors; 18 nodiscard-`hipError_t` warnings (`-Wunused-value`, the Linux count) and 1629 `-Wpass-failed` "loop not unrolled" lines; the link prints only the known unused `--ld-path=...lld-link.exe` warning. No generated file edited.

### GPU tests (`utils/timeit.sh Quest test`, `ulimit -c 0`)

```
PYTHONPATH=<fork> python -m pytest <fork>/quest/tests --rootdir <fork> -p no:cacheprovider -v -rs
```

**202 passed, 28 skipped, 0 failed** (16.5 s), one run; `quest._kernels` loaded from `quest/_kernels.cp312-win_amd64.pyd` copied by the README's last line (built 16:23 local, this build). Per suite, counted from that run's `-v` output, beside the earlier Windows runs:

| suite | `2b938d8` | `6e7d8db` and `a70f4d9` (windows-gfx1151) |
|---|---|---|
| test_rope | 64 passed | 64 passed |
| test_estimate | 9 passed | 9 passed |
| test_decode_attention | 6 passed | 6 passed |
| test_approx_attention | 42 passed | 42 passed |
| test_decode_launch_shape | 3 passed | 3 passed |
| test_topk | 49 passed, 15 skipped | 49 passed, 15 skipped |
| test_prefill_attention | 29 passed, 13 skipped | 29 passed, 13 skipped |
| total | 202 passed, 28 skipped | 202 passed, 28 skipped |

(The `a70f4d9` porter round recorded only the total.) Skips are the tests' own (`test_prefill_attention.py:51` qo_len > kv_len, `test_topk.py:27` k >= kv_len). Identical to linux-gfx942 and linux-gfx1201 at `2b938d8`. The five ROCm runtime DLLs stay staged in `quest/` (this host's System32 `amdhip64_7.dll` problem; environment, untracked). A first `timeit.sh` test invocation used the relative path `quest/tests` and collected nothing (exit 4, 0.5 s, "file or directory not found": `timeit.sh` runs from the repository root); it is in `stats.jsonl` and is a harness slip, not a test result.

The end-to-end model run was not repeated: no kernel or compiled source changed since `6e7d8db`, where it ran on this GPU; the delta is configure logic and README text.

### Configure-only checks (separate scratch directories, no build; `cfg_checks.ps1`)

Both start from a PowerShell with `CC`, `CXX`, `HIPCXX`, `PYTORCH_ROCM_ARCH`, `CMAKE_BUILD_TYPE`, `HIP_PATH`, `ROCM_PATH`, `VIRTUAL_ENV`, `CONDA_PREFIX` removed, source directory given by path.

| case | rc | C++ / HIP compiler | build type | Python | architecture |
|---|---|---|---|---|---|
| README fallback: environment not activated (`python` on PATH is the system 3.12), `-DPython_EXECUTABLE=<venv>\Scripts\python.exe` and that same `python.exe` in the `CMAKE_PREFIX_PATH` command | 0 | both `_rocm_sdk_devel/lib/llvm/bin/clang++.exe` (GNU-like), hipconfig found at `<venv>/Scripts/hipconfig.exe` | Release (`-O3 -DNDEBUG`, 8 lines) | `<venv>\Scripts\python.exe` 3.12.10 | gfx1151 (8 lines) |
| README command, environment activated, `$env:CMAKE_BUILD_TYPE='Debug'` | 0 | same two | **Debug** (`-O0 -g -D_DEBUG`, 8 lines, no `-O3`) | `<venv>/Scripts/python.exe` 3.12.10 | gfx1151 (8 lines) |

So on Windows an environment `CMAKE_BUILD_TYPE` is kept (review pass 5 could only show this by proxy on Linux), and the fallback finds the SDK and ROCm from `Python_EXECUTABLE` with nothing else in the environment. The Debug case was configured only; whether a Debug build links here (`python312_d.lib`) was not tried and is the user's choice of build type, not the default.

### CUDA gate, content checks, integrity

CUDA gate: not rerun, recorded at this head (real nvcc on the linux-gfx950 host, PASS). `python utils/jargon.py --port Quest` clean. The README documents the Windows ROCm build beside the Linux one, and the command above is that block verbatim. `git -C projects/Quest/src status --porcelain`: only the five untracked runtime DLLs in `quest/` (`quest/ops/build` and the `.pyd` are ignored); no modified tracked file.

State: windows-gfx1151 `completed` at `2b938d8` (windows gate).

## Ruling 2026-11-20 (jeffdaily): prefill on the matrix cores before the upstream pull request

jeffdaily ruled on 2026-11-20 that the deferral `quest-prefill-matrix-cores` is "now": the port's prefill attention must use AMD's matrix cores before the upstream pull request is sent. The pull request is held. The validated port is fork head `2b938d8`; commits go on top of `moat-port` as it is, plain pushes, never rewriting (a clean two-commit history of the same content exists on jeff's machine and will be rebuilt later). No upstream pull request is open. The design is in plan.md, "Plan 2026-11-20".

## Porter round 2026-11-20 (linux-gfx942): prefill on the matrix cores -- fork `6c06912`

One commit on `2b938d8`, plain push (`2b938d8..6c06912`), nothing rewritten: `6c06912` `[ROCm] Run prefill attention on AMD matrix instructions` (`kernels/include/prefill/prefill_matrix_core.cuh` new, 704 lines; `kernels/include/prefill/prefill.cuh` +104, all inside `#if defined(USE_HIP)` / `#if defined(QUEST_PREFILL_MMA)`; `kernels/include/hip_compat/README.md` +4). Design (a) of the plan, "Plan 2026-11-20". Host: whole MI300X (304 CUs, `gfx942:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=1`, PyTorch 2.16.0a0+git6c3af02 / HIP 7.16.26385 (pip ROCm SDK 10.1), Python 3.11.17 (`/opt/python-3.11-venv`), CMake 3.31.6.

### What the kernel does now on a matrix-core GPU

- `QUEST_PREFILL_MMA` is defined by the new header only in device passes: 1 for gfx90a/gfx94x/gfx950 (MFMA 16x16x16 f16 / bf16_1k), 11 for `__GFX11__` (WMMA `_w32`), 12 for `__GFX12__` (WMMA `_w32_gfx12`). Host pass and every other target (gfx908, gfx10xx, ...) keep the shuffle emulation in `hip_compat/flashinfer/mma.cuh` unchanged.
- S^T = K Q^T, so the D layout (lane = query column l % 16) is the B layout of O^T = V^T P^T: P goes from registers into the second product with no shuffles on MFMA and gfx12, one xor-16 exchange per tile on gfx11. Softmax row state per lane, reduced across the column's lane groups (xor 16, and 32 on wave64); the denominator is kept as per-lane partial sums and reduced once at the end. A wave64 holds two of the kernel's 32-lane warps (contiguous rows), so it works on 2 * num_frags_x query tiles.
- Q in registers when `num_q_tiles * num_frags_y * op_elems / 2 <= 64` (MFMA and gfx12; not gfx11, whose operands carry all 16 k per lane), output written straight from registers: shared memory then holds only K and V (32 KB at head_dim 128), two threadblocks per CU. The host reads `matrix_core::prefill_mma_shape`, a `static __device__` constant set per device pass, with `hipMemcpyFromSymbol` (cached per device) and sizes dynamic shared memory with the same `q_in_registers` function, so host and device cannot disagree and no arch list is duplicated on the host.
- V stored transposed in LDS (`vt_offset`, 128-bit chunks XOR-permuted by `d % (2 * num_frags_z)`), so the V^T operand is one 8/16-byte read.
- No `cp.async` on AMD: K/V tiles are loaded into registers one phase ahead (`page_load_kv` before the product, `page_store_kv` after the barrier). The paged pointers are always in bounds (`protective_get_*_ptr`), so loads are unconditional and only the LDS stores are predicated.
- Softmax exponent through `__builtin_amdgcn_exp2f` (`v_exp_f32`, the counterpart of PTX `ex2.approx`).

### How each step paid (standalone harness, 4096 x 4096 causal, 32 x 128, fp16)

| step | ms |
|---|---|
| `2b938d8` emulation | 100.6 |
| MFMA tile, S^T layout | 3.11 |
| + batched loads (no change: not the bottleneck) | 3.10 |
| + register prefetch (first try 3.66: HIP `uint4` kept the array in scratch, 128 B/lane; a native `ext_vector_type(4)` fixed it) | 2.44 |
| + V transposed in LDS, `v_exp_f32` | 1.97 |
| + Q in registers, LDS for K/V only (2 blocks/CU) | 1.06 |

Ablation that found the V read: with the first MFMA version, dropping the PV product saved 1.15 ms, dropping QK 0.3 ms; the transposed V operand was 4 `ds_read_u16` per MFMA.

Resource use (`-Rpass-analysis=kernel-resource-usage`): gfx942/gfx90a/gfx950 256 VGPR + 187-208 AGPR at num_frags_x=2, 0 scratch, all instantiations. gfx1100/gfx1151/gfx1201 (WMMA, wave32, 256 VGPR) spill 176-492 B/lane at num_frags_x=2 (twice the accumulator state per lane of wave64); gfx1201 num_frags_x=1 has none. Not measurable here; the Radeon validator should time it.

### Correctness evidence

Standalone harness (`agent_space/quest/harness/prefill_harness.cu`, not committed; builds one instantiation set in ~10 s with the extension's flags): fp64 CPU reference, shuffled pages, bottom-right causal rule, output and LSE checked. 100 cases: qo x kv in {1x17, 13x33, 16x16, 51x400, 63x700, 77x129, 128x129, 244x1110, 311x311, 502x1110} times {fp16 hd128 nfx1/nfx2 causal, fp16 hd128 non-causal, fp16 hd64, fp16 GQA group 4, bf16 hd128, bf16 hd64 group 4 non-causal, page size 1, page size 8, page size 32 group 8}. Tolerance 5e-3 + 5e-3 |ref| (fp16), 2e-2 + 3e-2 |ref| (bf16), LSE 2e-2 + 2e-3 |ref|. MFMA: 100/100. Emulated path at `2b938d8` with the same harness: 100/100 (validates the harness, LSE included). Max errors MFMA vs emulation: identical in 59 of the first 60 cases, smaller in one (output rounding dominates).

WMMA layouts tested on this GPU: a test-only header renames the four `__builtin_amdgcn_wmma_*` builtins to a shuffle emulation of their documented semantics (each 32-lane half of the wave64 as one wave32), and the harness is built with `-Xarch_device -DQUEST_PREFILL_MMA=11` or `=12`: 100/100 each (the gfx11 run also covers the Q-in-LDS branch). Negative control: the same test with the gfx11 D rows deliberately wrong fails 100/100. This checks the port's index logic against the documented layouts, not the hardware; the Radeon run is still owed.

Compile check, every target (`--cuda-device-only -S`): gfx90a emits `v_mfma_f32_16x16x16bf16_1k`/`_f16`, gfx942/gfx950 `v_mfma_f32_16x16x16_{bf16,f16}`, gfx1100/gfx1151/gfx1201 `v_wmma_f32_16x16x16_{bf16,f16}`, gfx908 and gfx1030 neither (emulation).

### Build (README Linux command, fresh build directory)

`agent_space/quest/build_readme.sh` = the README block verbatim with `-DCMAKE_HIP_ARCHITECTURES=gfx942` after `rm -rf build`, wrapped `utils/timeit.sh Quest compile`: rc 0, 8/8, 555 s wall (1160 s at `2b938d8`: the MFMA kernel bodies compile faster than the inlined-shuffle emulation), 0 errors, `CMAKE_BUILD_TYPE=Release` from the default.

### GPU tests (transformers 4.37.2, tokenizers 0.15.2, pytest 7.3.2 in `<job>/pylib-test`), `ulimit -c 0`

```
PYTHONPATH=<fork>:<job>/pylib-test python -m pytest <fork>/quest/tests -p no:cacheprovider -q -rs
```

**202 passed, 28 skipped, 0 failed** (32.8 s). Per suite: rope 64, estimate 9, decode_attention 6, approx_attention 42, decode_launch_shape 3, topk 49 + 15 skipped, prefill_attention 29 + 13 skipped. Same as `2b938d8`.

### Prefill time against SDPA, through the operator (`agent_space/quest/prefill_time.py`)

`quest.utils.prefill_forward` (what the model calls; includes the operator's own host work per call: index tensors, a device-to-host copy and two async mallocs, about 70 us) vs `F.scaled_dot_product_attention` with the same causal alignment (`is_causal` where qo = kv, `causal_lower_right` where qo < kv). Causal, 32 heads x 128, fp16, page size 16, mean of 20 (5 for the large shapes) after warmup.

| qo x kv | Quest prefill_forward | torch SDPA | ratio | max abs diff vs SDPA |
|---|---|---|---|---|
| 1 x 4096 | 0.459 ms | 0.177 ms | 2.60 | 6.1e-05 |
| 128 x 4096 | 0.583 ms | 0.200 ms | 2.91 | 1.2e-04 |
| 1024 x 1024 | 0.215 ms | 0.078 ms | 2.77 | 1.9e-03 |
| 1024 x 4096 | 0.594 ms | 0.264 ms | 2.25 | 2.4e-04 |
| 4096 x 4096 | 1.133 ms | 0.614 ms | 1.84 | 1.9e-03 |
| 8192 x 8192 | 3.094 ms | 2.007 ms | 1.54 | 1.9e-03 |
| 16384 x 16384 | 10.605 ms | 7.962 ms | 1.33 | 1.9e-03 |

The plan's target (within 2x of SDPA at 4096 x 4096) is met: 1.84x through the operator, 1.77x for the kernel alone (harness 1.06 ms). Long prompts, Quest's use case, are closer still; short ones are dominated by the operator's fixed host cost and by one tile row of work per head. Kernel time from the harness for the same shapes: 0.378, 0.517, 0.148, 0.527, 1.056, 3.04 ms.

### End-to-end (transformers 4.45.2, tokenizers 0.20.3, accelerate 1.0.1, sentencepiece 0.2.0 in `<job>/pylib-e2e`)

`NousResearch/Llama-2-7b-chat-hf` at `351844e75ed0bcbbe3f10671b3c808d2b83894ee` (downloaded again into `<job>/models`; `do_sample` set to false in `generation_config.json`, original kept as `.orig`), `scripts/example_textgen.py` with only `MODEL_PATH` changed (`agent_space/quest/e2e/`): `--method hg`, `--method quest --token_budget 1024`, `--method quest --token_budget 128`, all rc 0. The generated text is byte-identical across the three, apart from Quest's two banner lines, and the same continuation as recorded 2026-11-19 ("Reason: The rabbit is the only one who is not working hard. ... can also mean \"lazy.\""). `rocprofv3 --kernel-trace` of the budget-128 run: `BatchPrefillWithPagedKVCacheKernel<1,16,true,kNone,2,8,4,4,...>` 32 calls at 32.6 us (942 us on 2026-11-19, on a 38-CU partition), `quest_radix_topk_one_block_kernel` 2130 calls, `MaxPossibleSampleWithPagedKVCacheKernel` 2130, `BatchDecodeWithPagedKVCacheKernel` 3072, `VariableLengthMergeStatesKernel` 3072, `AppendPagedKVCachePrefillKernel` 32.

### CUDA path

Not compiled with nvcc here (no CUDA toolkit on this host). How it is known to be unchanged: every added line in `prefill.cuh` is inside `#if defined(USE_HIP)` (the include of the new header and the host shared-memory sizing) or `#if defined(QUEST_PREFILL_MMA)` (defined only by that header, only in AMD device passes), and the new header is reached only through the `USE_HIP` include. Checked mechanically: `prefill.cuh` at `2b938d8` and at `6c06912`, with `#include` lines removed, run through `cpp -P -undef -nostdinc` without `USE_HIP`: byte-identical output. The real-nvcc compile on the gfx950 host is still owed.

### Content checks

`jargon.py --port Quest` clean (after `git fetch origin main:main` in the clone); `prose.py` on the commit message clean; title 55 chars, `[ROCm]`, fenced Test Plan, AI-assistance sentence, no trailer, ASCII. `audit-commits` reports only the known `6e7d8db` gap. Fork tree clean (`git status --porcelain` empty).

### Owed next

1. Review of `2b938d8..6c06912` (project is `delta-ported`).
2. Validation on wave32 Radeon: linux-gfx1100 (RDNA3, WMMA gfx11 layout with the xor-16 P exchange and Q in LDS) and linux-gfx1201 (RDNA4, gfx12 layout, Q in registers); time prefill there with `prefill_time.py` too, since the WMMA paths spill (above).
3. windows-gfx1151 (gfx11 WMMA, Windows build).
4. linux-gfx90a (MFMA bf16_1k form) and the real-nvcc compile of `6c06912` on the gfx950 host.
5. Possible later speedups, not needed for the target: gfx950's 16x16x32 MFMA and `ds_read_b64_tr_b16`, direct global-to-LDS loads (`global_load_lds`), and removing the operator's per-call host synchronisation (upstream code).

## Lesson for cuda-to-rocm: a FlashAttention-style mma.sync kernel maps onto MFMA/WMMA by computing S^T

For `references/no-hip-equivalent.md` (the matrix-instruction part of the mma.sync/ldmatrix entry). Emulating `mma.sync` with shuffles is correct but leaves 100x on the table; a direct MFMA/WMMA port of an FA2 prefill loop needs only the tile functions rewritten. Compute S^T = K Q^T instead of S = Q K^T: the 16x16x16 instructions give D with lane l holding column l % 16, which is exactly the B-operand layout (column l % 16, consecutive k) of O^T = V^T P^T, so the probabilities feed the second product from registers (no shuffles on MFMA and RDNA4; RDNA3's B wants all 16 k per lane, one xor-16 exchange). Each lane owns whole query columns, so the softmax row max/sum reduce only across the column's 2 (wave32) or 4 (wave64) lane groups, and the denominator can stay a per-lane partial sum until the end. Then three AMD-specific costs dominate, in this order on MI300X (4096^2 causal, 32x128 fp16: 3.1 ms -> 1.06 ms, SDPA 0.60 ms): (1) LDS occupancy -- Q + K + V tiles filled 64 KB, one 2-wave block per CU; keeping Q as register operands (64 VGPR) let two blocks share a CU; (2) the V operand is strided in a kv-major tile; store V transposed into LDS from the registers that staged the global load; (3) no `cp.async`: issue the next tile's global loads into registers before the product and store after the barrier. Two traps: HIP's `uint4` (union wrapper) kept a register-staging array in scratch -- use `ext_vector_type(4)`; and the host must size dynamic shared memory for the path the device code took, which a `static __device__` constant read with `hipMemcpyFromSymbol` answers per GPU without duplicating the arch list. The WMMA layouts can be tested on a CDNA GPU by macro-renaming the builtins to a shuffle emulation (each 32-lane half of a wave64 as one wave32) and forcing the layout selector in the device pass. Select the RDNA4 path by target name (`__gfx1200__`, `__gfx1201__`, `__gfx12_generic__`), not `__GFX12__`: gfx1250 also defines `__GFX12__` and lacks the 16x16x16 `_w32_gfx12` WMMA, so the build breaks there ("needs target feature wmma-128b-insts"); `__GFX11__` is safe, every gfx11 target has the gfx11 WMMA. (Quest)

## Review 2026-11-20 (linux-gfx942, delta 2b938d8..6c06912) -- CHANGES REQUESTED

Reviewed `git diff 2b938d8..6c06912` (`prefill_matrix_core.cuh` new, `prefill.cuh` +104, `hip_compat/README.md` +4) with the pr-review skill, reading the unchanged kernel, dispatcher and FlashInfer `utils.cuh` switch macros around it. Host: whole MI300X, `HIP_VISIBLE_DEVICES=1`, SDK clang from the pip ROCm SDK 10.1. Scratch work in `agent_space/quest/` (harness copied from the porter's job).

### Port Correctness

1. `kernels/include/prefill/prefill_matrix_core.cuh:47` selects the RDNA4 WMMA path with `defined(__GFX12__)`, which the compiler also defines for gfx1250 (MI450). gfx1250 does not have the 16x16x16 `_w32_gfx12` WMMA, so any HIP build that lists gfx1250 now fails: `prefill_matrix_core.cuh:220:11: error: '__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12' needs target feature wmma-128b-insts,wavefrontsize32`. The same harness compiles for gfx1250 at `2b938d8` (rc 0, emulated path), so this is a regression of a released target. Fix: name the RDNA4 targets that have the instruction (`__gfx1200__`, `__gfx1201__`, `__gfx12_generic__`; macros checked with `clang -dM` per target), leaving gfx1250 on the emulation. On the same line, `gfx9-4-generic` (macros `__GFX9__ __gfx9_4_generic__`) also falls to the emulation although it has the MFMA forms used here: forcing `QUEST_PREFILL_MMA=1` for it compiles (2688 `v_mfma` in the assembly). Add `__gfx9_4_generic__` to the MFMA list. The lesson at the end of the porter round should carry the trap (`__GFX12__` includes gfx1250).

Reproduce: `agent_space/quest/harness/build.sh <fork> /tmp/q.s gfx1250 --cuda-device-only -S`.

### Checked, no finding (scope of jeff's questions)

- Operand layouts. MFMA 16x16x16 f16/bf16_1k: A row l%16, k 4(l/16)..+3; B column l%16, same k; D rows 4(l/16)+i. RDNA3: A/B all 16 k per lane, lanes 16-31 replicating 0-15; D rows 2i + l/16. RDNA4: k 8(l/16)..+7; D rows 8(l/16)+i. Cross-checked independently of llama.cpp against composable_kernel in `/var/lib/jenkins/pytorch/third_party/composable_kernel`: `ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp` gives gfx11 `kABKLane=1, kAK1PerLane=16, kRepeat=2, kCMLane=2, kCM0PerLane=8, kCM1PerLane=1` (row = 2i + l/16) and gfx12 `kABKLane=2, kK1PerLane=8, kCM0PerLane=1, kCM1PerLane=8` (row = 8(l/16)+i), and `test/wmma_op/wmma_op_util.hpp` stores gfx11 D with `r = ele * 2 + (lIdx / 16)`. The port's `acc_row`, `op_k0`, the gfx11 xor-16 P exchange (rows 4i..4i+3 from elements 2i, 2i+1 of the g=0 and g=1 lanes) and the replicated gfx11 A/B loads (k0 = 0 for all lanes) agree. What a real Radeon run must still confirm: that the hardware matches this documentation (no emulation can), the `_w32` builtins' behaviour when lanes 16-31 replicate, the xor-16 exchange on a real wave32, and the time with the spills below.
- Reference sweep re-run here at `6c06912` (fp64 reference, output and LSE, 100 cases: page sizes 1/8/16/32, GQA 4 and 8, fp16 and bf16, head 64/128, nfx 1/2, causal and not, qo < kv and qo = kv, odd lengths): native MFMA 100/100; RDNA3 layout via the porter's `wmma_emu.h` with `-Xarch_device -DQUEST_PREFILL_MMA=11 -DWMMA_EMU` 100/100; RDNA4 layout (`=12`) 100/100.
- Masking, paging, softmax: causal rule `kv_idx > kv_len + q_idx - qo_len` (bottom-right) and `mask_iteration`/`num_iterations` unchanged from the mma.sync path; every tile holding a row >= kv_len is at or past `mask_iteration`, so stale or uninitialised K rows are replaced by the select; V rows >= kv_len are zero-filled so 0 * garbage cannot reach O. Uninitialised K only reaches query columns past `qo_upper_bound`, which are never written. Page addressing uses the general `row / page_size` form valid for all of 1, 8, 16, 32; `page_iter_base` advances before both loads as before. Softmax: m reduced across the column's lane groups, d kept as per-lane partials scaled by the same `o_scale` and summed once in `normalize_d`; LSE = log2(d) + m from lanes 0-15. `vt_offset` XOR needs `2 * num_frags_z` to be a power of two: `SWITCH_NUM_FRAGS_Z` only instantiates 2 and 4.
- Fallback: gfx908, gfx1030 and gfx9-4-generic compile to the emulation (no `v_mfma`/`v_wmma`); the host reads `{0, 0}` from `prefill_mma_shape` there and keeps the original shared-memory size. gfx11-generic and gfx12-generic compile to WMMA.
- CUDA path: re-done here, `prefill.cuh` at `2b938d8` and `6c06912` with `#include` lines removed, `cpp -P -undef -nostdinc`: identical (970 lines), and identical again with `-D__CUDACC__ -D__CUDA_ARCH__=900`. No file outside `kernels/include/prefill` and the README changed.
- Header style: ASCII, no in-house vocabulary (`jargon.py --port Quest` clean), Apache header the same as on the copied FlashInfer shims (the file derives from FlashInfer's prefill tile functions), comments limited to layout facts.
- Commit `6c06912`: `[ROCm]`, 55 chars, AI sentence, fenced Test Plan, no trailer, ASCII.

### Register spills on the Radeon targets (jeff's question)

Not a correctness risk: the spills are compiler-generated scratch, private per lane, and the WMMA emulation runs above pass with the same code shape. They cost speed only. Measured per instantiation (`agent_space/quest/res2.sh`, `-Rpass-analysis=kernel-resource-usage`): gfx1201 spills only at nfx=2 (144-512 B/lane; nfx=1 is 0); gfx1100 spills at nfx=2 (340-448) and also at nfx=1 head 128 (72-192), which the porter round's "nfx=2" statement misses. The porter round also says Q stays in LDS on gfx11; `q_in_registers` keeps it in registers on gfx11 at head 128 nfx=1 and head 64 nfx<=2 (8 * nfx * nfy <= 64). Cheap reductions tried (compile only):
- Q in LDS on wave32 unless it fits 32 registers: gfx1201 nfx=2 nfz=2 goes to 0 spills (243-246 VGPR), nfx=2 nfz=4 to 252-280 B; gfx1100 nfx=1 nfz=4 gets worse (176 -> 468). The 64 KB dispatcher always picks nfz=4, so a zero-spill gfx12 needs nfz capped at 2 for wave32 nfx=2 as well (the dispatcher already caps nfz at 2 for its own register-heavy rope case).
- Building P one fz slice at a time: no change on either target (the compiler already overlaps it).
No cheap fix for gfx11. Whether either is worth doing should follow the Radeon timing; it is not part of this request.

### Recommendation

**Request Changes** (item 1).

## Porter round 2026-11-20 (linux-gfx942), answering the 2026-11-20 review -- fork `9cb8bee`

One commit on `6c06912`, plain push, nothing rewritten: `9cb8bee` `[ROCm] Keep gfx1250 on the emulated prefill path` (`prefill_matrix_core.cuh` selector, `hip_compat/README.md` wording).

- Item 1: the RDNA4 WMMA path is now selected by `__gfx1200__ || __gfx1201__ || __gfx12_generic__` instead of `__GFX12__`, so gfx1250 keeps the emulation; `__gfx9_4_generic__` joined the MFMA list. README now names the targets left on the emulation (gfx908, RDNA2, gfx1250).
- Device-only compile of the prefill harness per target (`agent_space/quest/harness/build.sh <fork> /tmp/f_<arch>.s <arch> --cuda-device-only -S`): rc 0 for all 13; `v_mfma` count 2688 on gfx90a, gfx942, gfx950, gfx9-4-generic; `v_wmma` 1344 on gfx1100, gfx1151, gfx11-generic, gfx1200, gfx1201, gfx12-generic; neither on gfx908, gfx1030, gfx1250.
- Whole extension for gfx1250 only (`agent_space/quest/build_gfx1250.sh`, the README command with `-DCMAKE_HIP_ARCHITECTURES=gfx1250`): rc 0, 8/8, 940 s, 0 errors. A first attempt with five targets in one build (`gfx1250;gfx9-4-generic;gfx12-generic;gfx908;gfx1030`) compiled every other source file for all five, then hit the one-hour job limit inside `batch_prefill.cu`'s per-target link-time code generation and was stopped; no error before that.
- README Linux build for gfx942 (`agent_space/quest/build_readme.sh`, fresh build directory, `utils/timeit.sh Quest compile`): rc 0, 8/8, 558 s, 0 errors.
- Tests, `ulimit -c 0`, `HIP_VISIBLE_DEVICES=1`: `env PYTHONPATH=<fork>:<job>/pylib-test python -m pytest <fork>/quest/tests -p no:cacheprovider -q -rs` -> 202 passed, 28 skipped (32.2 s).
- fp64 reference sweep (100 cases) at the fix: MFMA 100/100.
- `jargon.py --port Quest` clean; commit title 48 chars, `[ROCm]`, AI sentence, literal fenced Test Plan, no trailer, ASCII; fork tree clean.

## Review 2026-11-20 pass 2 (linux-gfx942, delta 6c06912..9cb8bee, whole delta 2b938d8..9cb8bee) -- PASS

Item 1 of the first pass is fixed by `9cb8bee`: the gfx12 WMMA branch names `__gfx1200__`, `__gfx1201__`, `__gfx12_generic__` (the RDNA4 macros `clang -dM` reports), so gfx1250 (`__GFX12__ __gfx1250__`) takes the emulation, and `__gfx9_4_generic__` joins the MFMA list. Checked independently of the porter's notes: the delta touches only the selector and the hip_compat README, `prefill.cuh` is unchanged from `6c06912` (whose CUDA view was verified identical to `2b938d8` in the first pass), and the per-target device compiles and the whole-extension gfx1250 build recorded in the porter round were run in this session (`agent_space/quest/build-gfx1250.log`, rc 0). `__GFX11__` still covers only targets with the gfx11 WMMA (gfx1100-1103, gfx1150-1153, gfx11-generic). Commit `9cb8bee`: `[ROCm]`, 48 chars, AI sentence, literal fenced Test Plan, no trailer, ASCII; `jargon.py --port Quest` and `prose.py` clean. No findings.

## Validation 2026-11-20 (linux-gfx942, at `9cb8bee`) -- PASS

Host: whole MI300X (`gfx942:sramecc+:xnack-`, 304 CUs, 192 GiB, idle before the runs), `HIP_VISIBLE_DEVICES=1`, PyTorch 2.16.0a0+git6c3af02 / HIP 7.16.26385 (pip ROCm SDK 10.1), Python 3.11 (`/opt/python-3.11-venv`), all GPU runs under `ulimit -c 0`. Fork clone at `9cb8bee`, submodules initialised, `git status --porcelain` empty.

- Build: `agent_space/quest/build_readme.sh` (README Linux block verbatim with `-DCMAKE_HIP_ARCHITECTURES=gfx942`, after `rm -rf build`, `ninja -j $MAX_JOBS`), wrapped `utils/timeit.sh Quest compile`: rc 0, 8/8, 555 s, 0 errors, `CMAKE_BUILD_TYPE=Release` from the default.
- Tests: `env PYTHONPATH=<fork>:<job>/pylib-test python -m pytest <fork>/quest/tests -p no:cacheprovider -v -rs` (transformers 4.37.2, tokenizers 0.15.2, pytest 7.3.2): **202 passed, 28 skipped, 0 failed** (32.3 s). rope 64, estimate 9, decode_attention 6, approx_attention 42, decode_launch_shape 3, topk 49 + 15 skipped (k >= kv_len), prefill_attention 29 + 13 skipped (qo_len > kv_len). Same as `2b938d8` and `6c06912`.
- End-to-end: `NousResearch/Llama-2-7b-chat-hf` at `351844e75ed0bcbbe3f10671b3c808d2b83894ee` (`do_sample` false), `scripts/example_textgen.py` with only `MODEL_PATH` changed (`agent_space/quest/e2e/example_textgen.py`), `PYTHONPATH=<fork>:<job>/pylib-e2e` (transformers 4.45.2): `--method hg`, `--method quest --token_budget 1024`, `--method quest --token_budget 128`, all rc 0. The generated text is byte-identical across the three (Quest's banner lines aside; md5 `fb7b22be...` of the filtered output for all three) and identical to the `6c06912` run ("Reason: The rabbit is the only one who is not working hard. ... can also mean \"lazy.\"").
- Prefill time through the operator against SDPA (`agent_space/quest/prefill_time.py`; causal, 32 heads x 128, fp16, page size 16; `is_causal` where qo = kv, `causal_lower_right` where qo < kv; mean of 20, 5 for the large shapes):

| qo x kv | Quest prefill_forward | torch SDPA | ratio | max abs diff vs SDPA |
|---|---|---|---|---|
| 1 x 4096 | 0.460 ms | 0.176 ms | 2.61 | 6.1e-05 |
| 128 x 4096 | 0.582 ms | 0.200 ms | 2.91 | 1.2e-04 |
| 1024 x 1024 | 0.215 ms | 0.077 ms | 2.77 | 2.0e-03 |
| 1024 x 4096 | 0.593 ms | 0.265 ms | 2.24 | 2.4e-04 |
| 4096 x 4096 | 1.135 ms | 0.614 ms | 1.85 | 2.0e-03 |
| 8192 x 8192 | 3.116 ms | 2.039 ms | 1.53 | 2.0e-03 |
| 16384 x 16384 | 10.656 ms | 7.954 ms | 1.34 | 2.0e-03 |

  Kernel alone (standalone harness built from the `9cb8bee` tree, `f_mfma time`): 0.364, 0.516, 0.148, 0.526, 1.061, 3.055 ms for the first six shapes. The plan's target (4096 x 4096 within 2x of SDPA, <= 1.2 ms) holds: 1.06 ms kernel, 1.135 ms through the operator. Within 1-2% of the `6c06912` figures, as expected: `9cb8bee` changes no gfx942 code.
- fp64 reference sweep at `9cb8bee`: MFMA 100/100 (porter round above); the RDNA3/RDNA4 layout emulations were run at `6c06912`, and `9cb8bee` does not touch the code they exercise.
- CUDA gate: not run here, no CUDA toolkit on this host (`/opt/conda/envs` absent, no `nvcc`). `cuda-not-validated: no toolkit on this host`; owed on the gfx950 host at `9cb8bee`. The CUDA view of `prefill.cuh` was shown identical to `2b938d8` by preprocessing (review 2026-11-20), and `9cb8bee` changes only the HIP-only header and a README.
- Content: `jargon.py --port Quest` clean; ROCm build documented in the README (Linux and Windows blocks); `audit-commits` reports only the known `6e7d8db` gap.

### Owed next (Quest at `9cb8bee`)

1. linux-gfx1201 (RDNA4, wave32): first real run of the gfx12 WMMA layout (Q in registers at nfx=2 head 128, Q in LDS never). Run the seven suites, the end-to-end comparison, and `prefill_time.py` for timing; the instantiations used at nfx=2 spill 144-512 B/lane (review 2026-11-20 has the per-instantiation table and a compile-measured zero-spill configuration if the time disappoints). The emulation verified the index logic against the documented layout; only hardware can confirm the hardware matches.
2. linux-gfx1100 (RDNA3, wave32): first real run of the gfx11 layout (replicated A/B halves, the xor-16 P exchange, Q in LDS at nfx=2 head 128). Same runs and timing; gfx1100 spills at both nfx=1 (72-192 B) and nfx=2 (340-448 B), and no cheap reduction was found.
3. windows-gfx1151 (gfx11 WMMA, Windows build with the README's PowerShell block): suites, end-to-end, timing.
4. linux-gfx90a (MFMA `bf16_1k` form, the only target using it): suites and timing.
5. Real-nvcc compile of `9cb8bee` on the gfx950 host (the CUDA gate).

## Validation 2026-11-23 (linux-gfx950, at `9cb8bee`) -- PASS

Host: AMD Instinct MI355X (`gfx950:sramecc+:xnack-`, 256 CUs, 288 GiB), wave64, `HIP_VISIBLE_DEVICES=0` (pinned by the job; idle before the runs). PyTorch 2.16.0a0+git6c3af02 / HIP 7.16.26385 (pip ROCm SDK), Python 3.11 (`/opt/python-3.11-venv`), all GPU runs under `ulimit -c 0`. Fork clone at `9cb8bee` == `origin/moat-port` == `head_sha`, submodules initialised. Requested by jeffdaily; this platform is additive evidence (wave64 is already satisfied by linux-gfx942 at this head). It is the first real-hardware run of the MFMA prefill code on gfx950: the earlier gfx950 records are compile-only. The MFMA path here is the same 16x16x16 code as on gfx942 (`QUEST_PREFILL_MMA` 1); the WMMA paths are not compiled for this target.

### Build

`agent_space/quest/build_readme.sh`, the README Linux block verbatim with `-DCMAKE_HIP_ARCHITECTURES=gfx950`, after `rm -rf build`, `ninja -j $MAX_JOBS`, wrapped `utils/timeit.sh Quest compile`:
## Validation 2026-11-23 (linux-gfx1201, at `9cb8bee`) -- PASS

AMD Radeon AI PRO R9700, `gcnArchName` `gfx1201`, wave32, 31.9 GiB, `HIP_VISIBLE_DEVICES=1` (pinned by the job; idle before every timed run, 0 % busy, 0.1 GiB used). PyTorch 2.16.0a0+git6c3af02 (`torch.version.hip` 7.16.26385) on the pip ROCm SDK (HIP compiler `_rocm_sdk_devel/lib/llvm/bin/clang++`), image Python 3.11.17 (`/opt/python-3.11-venv`, read-only; the job venv has no torch), CMake 3.31.6. All GPU runs under `ulimit -c 0`. Fork cloned fresh, checked out `9cb8bee` == `origin/moat-port` == `head_sha`, `git submodule update --init --depth 1`. Full revalidation (not a carry-forward): `classify 2b938d8 9cb8bee` is `mixed` and the delta adds the matrix-core prefill. This is the first run of the gfx12 WMMA tile code on real hardware.

### Build (README Linux command)

`agent_space/build.sh <fork>`, wrapped `utils/timeit.sh Quest compile -- bash agent_space/build.sh <fork>`: the README block with the architecture changed and ninja capped at the job's `MAX_JOBS` (24), after `rm -rf build`, `CMAKE_BUILD_TYPE` and `PYTORCH_ROCM_ARCH` unset, image Python first on `PATH`:

```
cd quest/ops
mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
ninja
ln -sf $PWD/_kernels*.so ../../
```

rc 0, 8/8, 348 s wall, 0 errors, 18 nodiscard-`hipError_t` warnings (the known ones), `CMAKE_BUILD_TYPE=Release` from the default, `--offload-arch=gfx950` on every compile line.

### GPU tests (transformers 4.37.2, tokenizers 0.15.2, huggingface-hub 0.24.7, pytest 7.3.2 in `<job>/pylib-test`)

```
env PYTHONPATH=<fork>:<job>/pylib-test python -m pytest <fork>/quest/tests -p no:cacheprovider -v -rs
```

**202 passed, 28 skipped, 0 failed** (21.9 s), the MI300X count. approx_attention 42, decode_attention 6, decode_launch_shape 3, estimate 9, prefill_attention 29 + 13 skipped (qo_len > kv_len), rope 64, topk 49 + 15 skipped (k >= kv_len).

### fp64 reference check of prefill (100 cases)

The porter's harness was not recorded in these notes (its container is gone), so the check was rebuilt from the 2026-11-20 description; the source, build script and timing script are now in `notes-prefill-harness.md` for the next validators. It calls `BatchPrefillWithPagedKVCacheDispatched` with a forced `num_frags_x` and checks output and LSE against an fp64 CPU reference, same 10 shapes, same 10 configurations, same tolerances as 2026-11-20. Built with the extension's flags (`bash build.sh <tree> bin/<t>_<n> <n>`), once against the fork at `9cb8bee` (MFMA) and once against `git archive 2b938d8 kernels/include` (the emulated path before the matrix-core change), and run on GPU 0:

| configuration | MFMA `9cb8bee` | max \|dO\| / \|dLSE\| | emulated `2b938d8` | max \|dO\| / \|dLSE\| |
|---|---|---|---|---|
| fp16 hd128 nfx1 causal p16 | 10/10 | 1.1e-03 / 5.3e-04 | 10/10 | 1.5e-03 / 1.6e-03 |
| fp16 hd128 nfx2 causal p16 | 10/10 | 1.1e-03 / 5.3e-04 | 10/10 | 1.5e-03 / 1.6e-03 |
| fp16 hd128 nfx2 non-causal p16 | 10/10 | 1.0e-03 / 3.5e-04 | 10/10 | 1.5e-03 / 1.6e-03 |
| fp16 hd64 nfx1 causal p16 | 10/10 | 1.1e-03 / 7.6e-04 | 10/10 | 1.2e-03 / 1.0e-03 |
| fp16 hd128 GQA4 nfx2 causal p16 | 10/10 | 1.4e-03 / 7.8e-04 | 10/10 | 1.6e-03 / 2.2e-03 |
| bf16 hd128 nfx2 causal p16 | 10/10 | 9.8e-03 / 6.5e-03 | 10/10 | 1.1e-02 / 1.4e-02 |
| bf16 hd64 GQA4 nfx1 non-causal p16 | 10/10 | 6.3e-03 / 4.4e-03 | 10/10 | 1.0e-02 / 1.2e-02 |
| fp16 hd128 nfx2 causal p1 | 10/10 | 1.2e-03 / 6.5e-04 | 10/10 | 1.3e-03 / 2.1e-03 |
| fp16 hd128 nfx1 causal p8 | 10/10 | 1.0e-03 / 9.0e-04 | 10/10 | 1.4e-03 / 2.1e-03 |
| fp16 hd128 GQA8 nfx2 causal p32 | 10/10 | 1.3e-03 / 7.8e-04 | 10/10 | 1.8e-03 / 2.0e-03 |

**MFMA 100/100, emulated 100/100.** Negative control (`-DNEGCTRL=true`, reference with the opposite causal flag, fp16 hd128 nfx2): 1/10 pass, the one being 1 x 17, where bottom-right causal and full attention are the same. The MFMA binaries report `prefill_mma_shape` {64, 4} from the device; the device assembly of the `9cb8bee` build has 384 `v_mfma` and no `v_wmma` for CFG 1, the `2b938d8` build neither. Resource use on gfx950 (`--cuda-device-only -S`, CFG 0/1/5/9): 190-256 VGPR + 0-205 AGPR, `ScratchSize: 0` in every prefill instantiation, so no spills on this target.

### End-to-end (transformers 4.45.2, tokenizers 0.20.3, accelerate 1.0.1, sentencepiece 0.2.0 in `<job>/pylib-e2e`)

The model fits (13 GB of 288 GB). `NousResearch/Llama-2-7b-chat-hf` at `351844e75ed0bcbbe3f10671b3c808d2b83894ee`, downloaded into `<job>/models` (safetensors, json, tokenizer), `do_sample` set to false in `generation_config.json` (original kept as `.orig`), `scripts/example_textgen.py` with only `MODEL_PATH` changed (`agent_space/quest/e2e/example_textgen.py`):
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
ninja -j 24
ln -sf $PWD/_kernels*.so ../../
```

rc 0, 8/8, 0 errors, **27m14s wall**. `CMAKE_BUILD_TYPE:STRING=Release` by default; all 8 compile lines `--offload-arch=gfx1201` only. Warnings: 18 nodiscard-`hipError_t`, and 0 `-Wpass-failed` (1536 at `2b938d8`: the emulated tile loops are gone). Build time from `.ninja_log`: `batch_prefill.cu.o` **1628 s** on its own, every other file 21-28 s. At `2b938d8` the whole build on this GPU took 615 s, and at `9cb8bee` on gfx942 555 s. So the gfx12 WMMA instantiations take about 3x longer to compile than the emulation they replace. This is consistent with the register pressure below (the allocator works hard on the 256-VGPR spilling kernels). It is not a failure. It does matter for anyone building several Radeon targets in one go: the porter round's five-target build already ran past an hour.

### GPU tests (transformers 4.37.2, tokenizers 0.15.2, huggingface-hub 0.24.7, pytest 7.3.2 in `<job>/pylib-test`, installed with `$FLEET_VENV/bin/pip install --no-deps --target`)

Wrapped `utils/timeit.sh Quest test -- bash agent_space/test.sh`:

```
cd <fork> && PYTHONPATH=<fork>:<job>/pylib-test python -m pytest quest/tests -p no:cacheprovider -q -rs
```

**202 passed, 28 skipped, 0 failed** (16.3 s), the same as the MI300X record at this head. Per suite (each run on its own): approx_attention 42, decode_attention 6, decode_launch_shape 3, estimate 9, prefill_attention 29 + 13 skipped (`qo_len > kv_len`), rope 64, topk 49 + 15 skipped (`k >= kv_len`). These counts are identical to `2b938d8` on this GPU.

### fp64 reference check of prefill (100 cases)

The porter's harness (`agent_space/quest/harness/prefill_harness.cu`) is not recorded in these notes, so I wrote an equivalent with the same grid. It is reproduced in full below so the Radeon validators that follow can run it. It calls `BatchPrefillWithPagedKVCacheDispatched` directly with `num_frags_x` forced, on the NHD layout with shuffled pages (3 spare pages), and checks the output and the LSE against an fp64 CPU reference (bottom-right causal rule). Tolerances are the porter's: fp16 5e-3 + 5e-3|ref|, bf16 2e-2 + 3e-2|ref|, LSE 2e-2 + 2e-3|ref|. It is needed because the Python operator dispatches fp16 only (`DISPATCH_PYTORCH_DTYPE_TO_CTYPE`), so the bf16 WMMA builtins are unreachable through the tests. The grid is 10 shapes (qo x kv in 1x17, 13x33, 16x16, 51x400, 63x700, 77x129, 128x129, 244x1110, 311x311, 502x1110) times 10 configurations: fp16 hd128 nfx1 causal; fp16 hd128 nfx2 causal; fp16 hd128 nfx2 non-causal; fp16 hd64 nfx2 causal; fp16 hd128 nfx2 causal GQA 4; bf16 hd128 nfx2 causal; bf16 hd64 nfx1 non-causal GQA 4; fp16 hd128 page size 1; fp16 hd128 nfx1 page size 8; bf16 hd128 nfx2 non-causal page size 32 GQA 8. All other cases use page size 16.

```
bash agent_space/harness/build.sh <fork> agent_space/harness/ref_9cb8bee     # 28 s
agent_space/harness/ref_9cb8bee check                                        # 13 s
```

| build | result |
|---|---|
| `9cb8bee` (gfx12 WMMA; the binary prints `prefill_mma_shape {32, 8}`) | **100/100** |
| `2b938d8` tree (`git archive 2b938d8 kernels/include`, emulated path), same harness | 100/100 (validates the harness on this GPU) |
| negative control: scratch copy of the `9cb8bee` header with the gfx12 `acc_row` replaced by the gfx11 formula `2 * i + lane / 16` | **0/100** (every case fails) |

Largest error per configuration over the 10 shapes:

| config | max dO WMMA | max dO emulated | max dLSE WMMA | max dLSE emulated |
|---|---|---|---|---|
| fp16 hd128 nfx1 causal ps16 | 1.08e-03 | 1.61e-03 | 5.24e-04 | 1.68e-03 |
| fp16 hd128 nfx2 causal ps16 | 8.04e-04 | 1.36e-03 | 5.48e-04 | 1.51e-03 |
| fp16 hd128 nfx2 noncausal ps16 | 6.75e-04 | 1.28e-03 | 4.06e-04 | 1.30e-03 |
| fp16 hd64 nfx2 causal ps16 | 1.09e-03 | 1.01e-03 | 5.90e-04 | 1.08e-03 |
| fp16 hd128 nfx2 causal ps16 g4 | 1.01e-03 | 1.64e-03 | 5.79e-04 | 1.94e-03 |
| bf16 hd128 nfx2 causal ps16 | 7.72e-03 | 8.82e-03 | 3.85e-03 | 1.17e-02 |
| bf16 hd64 nfx1 noncausal ps16 g4 | 5.87e-03 | 6.58e-03 | 3.59e-03 | 8.13e-03 |
| fp16 hd128 nfx2 causal ps1 | 1.07e-03 | 1.27e-03 | 7.46e-04 | 1.53e-03 |
| fp16 hd128 nfx1 causal ps8 | 1.06e-03 | 1.16e-03 | 6.37e-04 | 2.33e-03 |
| bf16 hd128 nfx2 noncausal ps32 g8 | 6.36e-03 | 9.63e-03 | 3.41e-03 | 1.43e-02 |

WMMA is at or below the emulation's error in 9 of 10 configurations, and within 8 % of it in the tenth (fp16 hd64). Only 1 of 100 cases has an identical max error (on MI300X it was 59 of 60). That is expected: the emulation rounds P to 16 bits and accumulates in a different order.

### The review's list for a real Radeon run (review 2026-11-20), answered for gfx1201

1. **Does the hardware match the documented gfx12 layout?** Yes. Real `v_wmma_f32_16x16x16_{f16,bf16}` (`_w32_gfx12`) give 100/100 for output and LSE across fp16 and bf16, head 64 and 128, GQA 1/4/8, page sizes 1/8/16/32, causal and not, nfx 1 and 2. The tests (202 passed) and the end-to-end text agree. The negative control fails all 100 cases, so the check does tell layouts apart. On gfx12 that means A/B k = 8(l/16)..+7 and D rows 8(l/16)+i.
2. **The `_w32` builtins when lanes 16-31 replicate:** not testable here. Replicated halves exist only in the gfx11 layout (`QUEST_PREFILL_MMA == 11`), and gfx1201 has neither those builtins nor that operand form. Still owed on linux-gfx1100 and windows-gfx1151.
3. **The xor-16 P exchange on a real wave32:** gfx11 only (gfx12 feeds P to the second product from registers with no exchange), so it was not exercised here and is still owed on gfx11. What gfx12 does exercise on a real wave32 is the xor-16 `column_max`/`column_sum` softmax reduction (one step at `kWaveSize` 32). It is correct.
4. **Speed with the spills:** see the next two sections. In short, the spilling configuration is still the fastest of the three measured, and capping `num_frags_z` at 2 on wave32 would gain about 15 % more.

Q in LDS is never taken on gfx12: `q_in_registers` holds for every instantiation (`nfx * nfy * 8 / 2 <= 64` for head 64/128, nfx <= 2). The harness's resource remarks show `LDS 0` static on all of them.

### Prefill time against SDPA, through the operator

`agent_space/prefill_time.py` (method of the MI300X table: `quest.utils.prefill_forward` against `F.scaled_dot_product_attention` with `is_causal` where qo = kv and `causal_lower_right` where qo < kv; causal, 32 heads x 128, fp16, page size 16, `InferenceController(1 layer, budget 1024, max_seq_len = kv)`; CUDA-event mean of 20 calls, 5 for qo >= 8192, after 3 warmup calls). Run twice; second run in brackets:

```
PYTHONPATH=<fork>:<job>/pylib-test python agent_space/prefill_time.py
```

| qo x kv | Quest prefill_forward | torch SDPA | ratio | max abs diff vs SDPA | MI300X ratio at `9cb8bee` |
|---|---|---|---|---|---|
| 1 x 4096 | 0.348 (0.362) ms | 0.128 (0.117) ms | 2.71 (3.09) | 6.1e-05 | 2.61 |
| 128 x 4096 | 0.540 (0.541) ms | 0.127 (0.126) ms | 4.25 (4.31) | 1.2e-04 | 2.91 |
| 1024 x 1024 | 0.349 (0.354) ms | 0.118 (0.118) ms | 2.95 (2.99) | 2.0e-03 | 2.77 |
| 1024 x 4096 | 1.129 (1.135) ms | 0.692 (0.689) ms | 1.63 (1.65) | 1.2e-04 | 2.24 |
| 4096 x 4096 | 2.428 (2.430) ms | 1.518 (1.522) ms | **1.60** | 2.0e-03 | 1.85 |
| 8192 x 8192 | 8.970 (8.988) ms | 6.849 (6.860) ms | 1.31 | 2.0e-03 | 1.53 |
| 16384 x 16384 | 35.567 (35.442) ms | 27.490 (27.482) ms | 1.29 | 2.0e-03 | 1.34 |

The plan's bar (4096 x 4096 within 2x of SDPA) holds on this Radeon too: 1.60x. Long prompts are closer than on MI300X. Short shapes are dominated by the operator's fixed host work per call (index tensors, a device-to-host copy, two async mallocs), and 128 x 4096 is the worst ratio here (4.3x).

Kernel alone (harness `time` mode, same shapes, dispatcher's nfx, hipEvent mean of 20/5): 0.254, 0.463, 0.294, 1.089, 2.439, 8.996, 34.895 ms (a repeat run was within 2 %).

**Old emulated path, one shape.** No build switch selects it at `9cb8bee` (`QUEST_PREFILL_MMA` is set per target inside `prefill_matrix_core.cuh`), so the same harness was built from the `2b938d8` tree: 4096 x 4096 causal **107.5 ms**, against **2.44 ms** for WMMA (44x).

### Spills: what they cost here

Resource use per instantiation (`bash agent_space/harness/build.sh <fork> /tmp/x.o --cuda-device-only -c -Rpass-analysis=kernel-resource-usage`; tuple G, page size, causal, nfx, nfy, nfz): every nfx=2 head-128 instantiation sits at 256 VGPR with scratch 144-540 B/lane (G1 PS16 causal nfz4: 512; nfz2: 176), occupancy 5. nfx=1 has 0 scratch (235-245 VGPR). bf16 is the same as fp16. This matches the review's table.

Times for the kernel alone, 4096 x 4096 causal (and 16384^2):

| configuration | spill (B/lane) | 4096^2 | 16384^2 |
|---|---|---|---|
| as shipped: nfx2, nfz4 (dispatcher's choice) | 512 | 2.44 ms | 34.9 ms |
| nfx1 forced (harness `time1`) | 0 | 2.98 ms | 41.8 ms |
| nfx2 with nfz capped at 2 (scratch copy of `prefill.cuh`: `max_num_frags_z_reg` = 2 whenever head 128 and nfx 2) | 176 | **2.07 ms** | **27.8 ms** |

The nfz=2 variant also passes 100/100 in the reference check. So the spills are not a correctness issue and do not make the shipped choice the slowest. A wave32-only cap of `num_frags_z` at 2 for head 128 nfx 2 would make prefill 15-20 % faster on gfx12 (and might cut the 27-minute compile). The scratch edit capped it for every target, which an upstream change must not do to CUDA or MFMA. This is a suggestion for the porter, not a finding: the bar is met as shipped.

### End-to-end (transformers 4.45.2, tokenizers 0.20.3, accelerate 1.0.1, sentencepiece 0.2.0, huggingface-hub 0.26.2 in `<job>/pylib-e2e`)

The model fits easily (13.5 GB fp16 on a 31.9 GiB card). `NousResearch/Llama-2-7b-chat-hf` at `351844e75ed0bcbbe3f10671b3c808d2b83894ee` was copied from the 2026-11-19 gfx1201 job on this host (`do_sample` false in `generation_config.json`, original kept as `.orig`). `scripts/example_textgen.py` with only `MODEL_PATH` changed, each wrapped in `utils/timeit.sh Quest test`:

```
PYTHONPATH=<fork>:<job>/pylib-e2e python example_textgen.py --method hg
PYTHONPATH=<fork>:<job>/pylib-e2e python example_textgen.py --method quest --token_budget 1024
PYTHONPATH=<fork>:<job>/pylib-e2e python example_textgen.py --method quest --token_budget 128
```

All rc 0. With Quest's two banner lines removed (`Quest allocates KV-Cache ...`, `Token budget is set to ...`), the three outputs are byte-identical (md5 `b900af27eeb4...` for all three), and the continuation is the one recorded on MI300X ("Reason: The rabbit is the only one who is not working hard. ... can also mean \"lazy.\""). `rocprofv3 --kernel-trace --stats` of the budget-128 run (output again identical): `BatchPrefillWithPagedKVCacheKernel<1,16,true,kNone,2,8,4,4,...>` 32 calls, min 35.0 us, average 229.7 us; the average is inflated by the GPU clocking up from idle between the short calls of a single generation (see the timing note below); MI300X recorded 32.6 us.

### Prefill time against SDPA (32 heads x 128, fp16, causal, page size 16)

GPU 0 showed no other process and 0 % activity between runs, but another container kept GPU 3 of this host at 100 % / ~830 W throughout. With the MI300X method (mean of 20 after 3 warmups, one process) the times moved by up to 10x between identical runs, for SDPA as much as for Quest, and the kernel-only harness moved too (4096 x 4096: 1.00 to 2.76 ms). Inside one process, min and median agreed, so the clock state moves on the scale of seconds. The GPU idles at ~94 MHz in deep sleep. The table is therefore the best of 5 separate processes, each with `MINMED=1 LOOP_MS=100 REPS=7 WARM_MS=1000` (1 s warmup, 7 timed loops of about 100 ms, per-process min). The range across the 5 processes is in parentheses.

```
MINMED=1 LOOP_MS=100 REPS=7 WARM_MS=1000 PYTHONPATH=<fork>:<job>/pylib-test python agent_space/quest/prefill_time.py
```

| qo x kv | Quest prefill_forward | torch SDPA | ratio | max abs diff vs SDPA |
|---|---|---|---|---|
| 1 x 4096 | 0.472 ms (0.47-1.40) | 0.109 ms (0.11-0.20) | 4.33 | 6.1e-05 |
| 128 x 4096 | 0.567 ms (0.57-2.29) | 0.146 ms (0.15-0.55) | 3.88 | 1.2e-04 |
| 1024 x 1024 | 0.259 ms (0.26-10.3) | 0.043 ms (0.04-0.14) | 6.02 | 2.0e-03 |
| 1024 x 4096 | 0.638 ms (0.64-0.91) | 0.126 ms (0.13-0.53) | 5.06 | 1.2e-04 |
| 4096 x 4096 | 1.204 ms (1.20-2.18) | 0.227 ms (0.23-0.80) | 5.30 | 2.0e-03 |
| 8192 x 8192 | 3.548 ms (3.55-4.02) | 0.805 ms (0.81-7.24) | 4.41 | 2.0e-03 |
| 16384 x 16384 | 10.534 ms (10.5-16.5) | 2.802 ms (2.80-6.91) | 3.76 | 2.0e-03 |

Kernel alone (`bin/mc_100` vs `bin/emu_100` from `notes-prefill-harness.md`: 32 heads, nfx 2 forced, mean of 20 launches; best of 5 processes for MFMA, of 3 for the emulation):

| qo x kv | MFMA `9cb8bee` | emulated `2b938d8` | speedup |
|---|---|---|---|
| 1 x 4096 | 0.599 ms | 24.40 ms | 41x |
| 128 x 4096 | 0.510 ms | 24.73 ms | 48x |
| 1024 x 1024 | 0.150 ms | 7.96 ms | 53x |
| 1024 x 4096 | 0.538 ms | 29.68 ms | 55x |
| 4096 x 4096 | 0.997 ms | 58.24 ms | 58x |
| 8192 x 8192 | 3.444 ms | 198.1 ms | 58x |

There is no build switch that brings back the emulated path on an MFMA target. `QUEST_PREFILL_MMA` is only an internal selector, and predefining it does not select the emulation, so the emulated figures come from the `2b938d8` headers built by the same harness.

Reading: Quest's prefill on MI355X runs at about the MI300X speed (4096 x 4096: 1.00 ms kernel, 1.20 ms operator; MI300X 1.06 / 1.135). SDPA is 2.7x faster here than on MI300X (0.227 vs 0.614 ms), so the ratio is ~5x where MI300X had 1.85x, and the plan's target ("within 2x of SDPA at 4096 x 4096") is not met on this GPU. This is a speed result, not a failure. The likely reason is that the port uses the 16x16x16 MFMA forms on every CDNA target, and gfx950's doubled matrix throughput comes from its 16x16x32 forms (and `ds_read_b64_tr_b16`), which the porter round lists as possible later steps (2026-11-20, "Owed next" item 5); MI355X also has 256 CUs to MI300X's 304. Not profiled further here.

### The review's list for a real Radeon run (review 2026-11-20, "Operand layouts")

This GPU is CDNA4 (wave64, MFMA), so none of the four items can be observed here, and each stays owed to linux-gfx1201, linux-gfx1100 and windows-gfx1151:
1. That the hardware matches the documented WMMA layouts: not observable (no WMMA instruction on gfx950). Observed instead: the MFMA 16x16x16 f16/bf16 layout on gfx950 hardware, 100/100 against fp64.
2. The `_w32` builtins with lanes 16-31 replicating 0-15 (gfx11): not observable.
3. The xor-16 P exchange on a real wave32: not observable (wave64 here).
4. Time with the WMMA spills: not applicable. gfx950 has no spills (`ScratchSize: 0`); its time is above. The harness in `notes-prefill-harness.md` builds for gfx1100 and gfx1201 with `ARCH=<gfx>` (checked here: 192 `v_wmma` each in the CFG 4 device assembly), so the Radeon validators can run the same 100 cases and the kernel timing.

### CUDA no-regression compile (real nvcc, at `9cb8bee`) -- PASS, no port regression

This is the CUDA gate owed at `9cb8bee`. The recipe is the one used for `2b938d8` (2026-11-19, above), from fresh clones in `<job>/nvcc`. Port: `AMD-Ecosystem/Quest` at `9cb8bee`. Base: `mit-han-lab/Quest` at `01c1623`. Submodule SHAs are identical in both (flashinfer `9f49803b`, its googletest `40412d85` and nvbench `f57aa9c9`, googletest `6a593823`, nvbench `d8dced8a`, pybind `768cebe1`, raft `1e4961e2`). Toolchain: CUDA 12.8 (`/usr/local/cuda`, nvcc V12.8.93), g++-11 11.5.0 (the unpacked debs from the 2026-11-19 job, copied), PyTorch `2.5.0+cu124` in `$FLEET_VENV`, `-DCMAKE_CUDA_ARCHITECTURES=80`, `TORCH_CUDA_ARCH_LIST=8.0`. No NVIDIA GPU, so this is compile and link only.

```
# run.sh <tree> <builddir> <cudaroot>, as 2026-11-19, then:
ninja -C <job>/nvcc/build-<tree> -j64 -k 0 _kernels
```

| | base `01c1623` | port `9cb8bee` |
|---|---|---|
| configure + generate | rc 0, Release | rc 0, Release |
| `_kernels` (7 `.cu` + link) | 8/8, rc 0, 18,681,432 B | 8/8, rc 0, 18,714,200 B |
| warnings in the log | 0 | 0 |

- The 7 nvcc compile lines are byte-identical between the trees after path normalisation.
- The port module is the same size as the `2b938d8` port module (18,714,200 B, 2026-11-19).
- Both modules export 651 text symbols, and each imports under `$FLEET_VENV/bin/python -I` from `/tmp` with the same 8 public names.
- The prefill header's CUDA branch is unchanged: `batch_prefill.cu`'s nvcc compile line, with `-c` replaced by `-E`, was run on the port tree at `9cb8bee` and on a worktree at `2b938d8`. With line markers and blank lines removed, the two outputs are identical (360,842 lines each, same md5); the raw outputs differ only in blank lines. No `matrix_core`, `QUEST_PREFILL_MMA`, `mfma` or `wmma` token reaches the CUDA translation unit.
- RAFT's optional `raft_objs` were not built this time (`_kernels` target only). Their 15 failures are upstream's and identical in both trees (2026-11-19).

### Content checks and integrity

- `jargon.py --port Quest` is clean (after `git fetch origin main:main moat-port:moat-port` in the clone).
- The ROCm build is documented in the README (Linux and Windows blocks), and the command above is that documentation.
- `audit-commits` reports only the known `6e7d8db` gap.
- `git -C projects/Quest/src status --porcelain` is empty: the `_kernels` link and `build/` are ignored.
All rc 0, prompt 103 tokens. The text from "Input Sequence Length" on is byte-identical across the three: md5 `b900af27eeb4170a5c47bc5940a2ca60`, the same digest as this GPU at `6e7d8db` and as gfx942 at `6e7d8db` ("Reason: The rabbit is the only one who is not working hard. ... can also mean \"lazy.\""). `rocprofv3 --kernel-trace --stats -d prof -o run -- ... --method quest --token_budget 128` (generated text identical again) read from `run_results.db`: `BatchPrefillWithPagedKVCacheKernel<1,16,true,kNone,2,8,4,4,...>` 32 calls at **24.4 us** (433 us at `6e7d8db` on this GPU, 18x), `quest_radix_topk_one_block_kernel` 2130 (11.2 us), `MaxPossibleSampleWithPagedKVCacheKernel` 2130 (4.9 us), `BatchDecodeWithPagedKVCacheKernel` 3072 (12.9 us), `VariableLengthMergeStatesKernel` 3072, `AppendPagedKVCacheDecodeKernel` 3072, `AppendPagedKVCachePrefillKernel` 32. The call counts equal the earlier traces.

### CUDA path

Not compiled with nvcc: this host has no CUDA toolkit (no `nvcc`, no `/usr/local/cuda`, no `/opt/conda/envs`); `cuda-not-validated: no toolkit on this host`. The real-nvcc compile of `9cb8bee` is still owed on the gfx950 host. Preprocessor check repeated here: `git show <sha>:kernels/include/prefill/prefill.cuh | grep -v '^\s*#\s*include' | cpp -P -undef -nostdinc [-D__CUDACC__ -D__CUDA_ARCH__=900] -`. Both `2b938d8` and `9cb8bee`, with and without the CUDA defines, give md5 `02d6a90c0d2eea0259c7a25b34a3f580` (970 lines), so the CUDA view of the prefill header is unchanged.

### Content checks and integrity

`jargon.py --port Quest` clean (after `git fetch origin main:main moat-port:moat-port` in the clone). `audit-commits Quest` reports only the known `6e7d8db` fenced-Test-Plan gap. The ROCm build is documented in `README.md` (Linux and Windows blocks). `git -C projects/Quest/src status --porcelain` is empty. No fork edit, no push to the fork, no GitHub write besides this project branch.

State: linux-gfx1201 `completed` at `9cb8bee` (wave32 gate).

### Owed next (Quest at `9cb8bee`)

1. linux-gfx1100 and windows-gfx1151: the gfx11 layout (replicated A/B halves, xor-16 P exchange, Q in LDS at head 128 nfx 2) still has had no hardware run. The harness below takes `--offload-arch=gfx1100` / `gfx1151` in `build.sh`. Its negative control for gfx11 is the gfx12 row formula.
2. linux-gfx90a (MFMA `bf16_1k` form) and the real-nvcc compile on the gfx950 host.
3. For the porter, optional: the wave32 `num_frags_z` cap (15-20 % on gfx12, measured above).

<details><summary>Harness used above: <code>agent_space/harness/build.sh</code> and <code>prefill_ref.cu</code></summary>

```
#!/bin/bash
# build.sh <fork tree> <out binary> [extra flags]: the extension's HIP flags, gfx1201
set -e
T=$1; OUT=$2; shift 2
SDK=/opt/python-3.11-venv/lib/python3.11/site-packages/_rocm_sdk_devel
CXX=$SDK/lib/llvm/bin/clang++
H=$T/kernels/include
exec $CXX -x hip -O3 -DNDEBUG -std=gnu++20 --offload-arch=gfx1201 -ffp-contract=on \
  -include $H/hip_compat/flashinfer/math.cuh -include $H/hip_compat/flashinfer/mma.cuh -include $H/hip_compat/flashinfer/vec_dtypes.cuh \
  -DUSE_HIP -D__HIP_PLATFORM_AMD__=1 -DCUDA_HAS_FP16=1 -DUSE_ROCM -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1 \
  -DHIP_ENABLE_WARP_SYNC_BUILTINS -Wno-unused-result -Wno-pass-failed \
  -I$H/hip_compat -I$H -I$T/kernels/3rdparty/flashinfer/include \
  --rocm-path=$SDK "$@" $(dirname $0)/prefill_ref.cu -o $OUT
```

```cpp
// fp64 reference check and timing of Quest's BatchPrefillWithPagedKVCacheDispatched.
// Validator harness (not part of the fork). Usage: prefill_ref check | time
#include <hip/hip_runtime.h>
#include "prefill/prefill.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace flashinfer;

#define CK(x)                                                                     \
  do {                                                                            \
    hipError_t e_ = (x);                                                          \
    if (e_ != hipSuccess) {                                                       \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      exit(2);                                                                    \
    }                                                                             \
  } while (0)

template <typename T> float to_f(T x);
template <> float to_f<half>(half x) { return __half2float(x); }
template <> float to_f<nv_bfloat16>(nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> T from_f(float x);
template <> half from_f<half>(float x) { return __float2half(x); }
template <> nv_bfloat16 from_f<nv_bfloat16>(float x) { return __float2bfloat16(x); }

struct Case {
  uint32_t qo, kv;
};

struct Result {
  double max_err_o, max_err_lse;
  size_t bad_o, bad_lse;
  float ms;
};

// Runs one configuration. num_kv_heads KV heads, GROUP query heads per KV head, NHD paged
// layout with shuffled pages (plus spare pages so the indices are not the identity).
template <typename DT, uint32_t NFX, uint32_t PS, uint32_t GROUP, uint32_t HD, bool CAUSAL>
Result run(uint32_t qo_len, uint32_t kv_len, uint32_t num_kv_heads, bool check, int iters,
           uint64_t seed) {
  constexpr QKVLayout L = QKVLayout::kNHD;
  const uint32_t nqh = num_kv_heads * GROUP;
  const uint32_t num_pages = (kv_len + PS - 1) / PS;
  const uint32_t total_pages = num_pages + 3;
  const uint32_t last_page_len = kv_len - (num_pages - 1) * PS;

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<DT> hq(size_t(qo_len) * nqh * HD);
  std::vector<DT> hkv(size_t(total_pages) * 2 * PS * num_kv_heads * HD, from_f<DT>(0.f));
  for (auto& x : hq) x = from_f<DT>(nd(rng));
  std::vector<int32_t> perm(total_pages);
  for (uint32_t i = 0; i < total_pages; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<int32_t> indices(perm.begin(), perm.begin() + num_pages);
  // K and V as dense [kv_len][heads][HD], scattered into pages.
  std::vector<float> K(size_t(kv_len) * num_kv_heads * HD), V(K.size());
  for (uint32_t t = 0; t < kv_len; ++t)
    for (uint32_t h = 0; h < num_kv_heads; ++h)
      for (uint32_t d = 0; d < HD; ++d) {
        DT k = from_f<DT>(nd(rng)), v = from_f<DT>(nd(rng));
        size_t i = (size_t(t) * num_kv_heads + h) * HD + d;
        K[i] = to_f(k);
        V[i] = to_f(v);
        uint32_t p = indices[t / PS], e = t % PS;
        hkv[((size_t(p) * 2 * PS + e) * num_kv_heads + h) * HD + d] = k;
        hkv[((size_t(p) * 2 * PS + PS + e) * num_kv_heads + h) * HD + d] = v;
      }

  DT *dq, *dkv, *dout;
  float* dlse;
  int32_t *dind, *dindptr, *dqoindptr, *dreq, *dtile;
  CK(hipMalloc(&dq, hq.size() * sizeof(DT)));
  CK(hipMalloc(&dkv, hkv.size() * sizeof(DT)));
  CK(hipMalloc(&dout, hq.size() * sizeof(DT)));
  CK(hipMalloc(&dlse, size_t(qo_len) * nqh * sizeof(float)));
  CK(hipMalloc(&dind, num_pages * sizeof(int32_t)));
  CK(hipMalloc(&dindptr, 2 * sizeof(int32_t)));
  CK(hipMalloc(&dqoindptr, 2 * sizeof(int32_t)));
  CK(hipMemcpy(dq, hq.data(), hq.size() * sizeof(DT), hipMemcpyHostToDevice));
  CK(hipMemcpy(dkv, hkv.data(), hkv.size() * sizeof(DT), hipMemcpyHostToDevice));
  CK(hipMemset(dout, 0xff, hq.size() * sizeof(DT)));
  CK(hipMemcpy(dind, indices.data(), num_pages * sizeof(int32_t), hipMemcpyHostToDevice));
  int32_t indptr[2] = {0, int32_t(num_pages)}, qoindptr[2] = {0, int32_t(qo_len)};
  CK(hipMemcpy(dindptr, indptr, sizeof(indptr), hipMemcpyHostToDevice));
  CK(hipMemcpy(dqoindptr, qoindptr, sizeof(qoindptr), hipMemcpyHostToDevice));
  // Same tiling as split_qo_indptr, with num_frags_x forced.
  const uint32_t rows_per_cta = NFX * 4 * 16;
  std::vector<int32_t> req, tile;
  for (uint32_t j = 0; j < qo_len * GROUP; j += rows_per_cta) {
    req.push_back(0);
    tile.push_back(j / rows_per_cta);
  }
  CK(hipMalloc(&dreq, req.size() * sizeof(int32_t)));
  CK(hipMalloc(&dtile, tile.size() * sizeof(int32_t)));
  CK(hipMemcpy(dreq, req.data(), req.size() * sizeof(int32_t), hipMemcpyHostToDevice));
  CK(hipMemcpy(dtile, tile.data(), tile.size() * sizeof(int32_t), hipMemcpyHostToDevice));

  paged_kv_t<PageStorage::kIndices, L, DT, int32_t> paged_kv(
      num_kv_heads, PS, HD, 1, 0, last_page_len, indices.back(), dkv, dind, dindptr);
  auto launch = [&]() {
    hipError_t st = BatchPrefillWithPagedKVCacheDispatched<PageStorage::kIndices, L, NFX, PS, GROUP, HD,
                                              RotaryMode::kNone, false, CAUSAL, DT, DT, int32_t>(
        dq, dreq, dtile, dqoindptr, paged_kv, dout, nullptr, dlse, uint32_t(req.size()), 1.f,
        1e4f, nullptr);
    CK(st);
  };
  launch();
  CK(hipDeviceSynchronize());

  Result r{0, 0, 0, 0, 0.f};
  if (check) {
    std::vector<DT> ho(hq.size());
    std::vector<float> hlse(size_t(qo_len) * nqh);
    CK(hipMemcpy(ho.data(), dout, ho.size() * sizeof(DT), hipMemcpyDeviceToHost));
    CK(hipMemcpy(hlse.data(), dlse, hlse.size() * sizeof(float), hipMemcpyDeviceToHost));
    const bool bf = std::is_same<DT, nv_bfloat16>::value;
    const double atol = bf ? 2e-2 : 5e-3, rtol = bf ? 3e-2 : 5e-3;
    const double scale = 1.0 / std::sqrt(double(HD));
    std::vector<double> s(kv_len), o(HD);
    for (uint32_t i = 0; i < qo_len; ++i)
      for (uint32_t qh = 0; qh < nqh; ++qh) {
        const uint32_t kh = qh / GROUP;
        uint32_t n = kv_len;
        if (CAUSAL) n = kv_len - qo_len + i + 1;  // bottom-right: j <= kv_len - qo_len + i
        double m = -INFINITY;
        for (uint32_t j = 0; j < n; ++j) {
          double acc = 0;
          for (uint32_t d = 0; d < HD; ++d)
            acc += double(to_f(hq[(size_t(i) * nqh + qh) * HD + d])) *
                   K[(size_t(j) * num_kv_heads + kh) * HD + d];
          s[j] = acc * scale;
          m = std::max(m, s[j]);
        }
        double den = 0;
        std::fill(o.begin(), o.end(), 0.0);
        for (uint32_t j = 0; j < n; ++j) {
          double p = std::exp(s[j] - m);
          den += p;
          for (uint32_t d = 0; d < HD; ++d) o[d] += p * V[(size_t(j) * num_kv_heads + kh) * HD + d];
        }
        for (uint32_t d = 0; d < HD; ++d) {
          double ref = o[d] / den;
          double got = to_f(ho[(size_t(i) * nqh + qh) * HD + d]);
          double err = std::fabs(got - ref);
          if (!(err <= atol + rtol * std::fabs(ref))) ++r.bad_o;
          if (!(err <= r.max_err_o)) r.max_err_o = std::isnan(err) ? INFINITY : std::max(r.max_err_o, err);
        }
        double lse_ref = (m + std::log(den)) / std::log(2.0);
        double lerr = std::fabs(hlse[size_t(i) * nqh + qh] - lse_ref);
        if (!(lerr <= 2e-2 + 2e-3 * std::fabs(lse_ref))) ++r.bad_lse;
        if (!(lerr <= r.max_err_lse)) r.max_err_lse = std::isnan(lerr) ? INFINITY : std::max(r.max_err_lse, lerr);
      }
  }
  if (iters > 0) {
    for (int w = 0; w < 3; ++w) launch();
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    CK(hipEventRecord(a));
    for (int it = 0; it < iters; ++it) launch();
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    CK(hipEventElapsedTime(&r.ms, a, b));
    r.ms /= iters;
  }
  hipFree(dq); hipFree(dkv); hipFree(dout); hipFree(dlse); hipFree(dind);
  hipFree(dindptr); hipFree(dqoindptr); hipFree(dreq); hipFree(dtile);
  return r;
}

static const Case kCases[] = {{1, 17},    {13, 33},   {16, 16},   {51, 400},   {63, 700},
                              {77, 129},  {128, 129}, {244, 1110}, {311, 311}, {502, 1110}};

template <typename F>
int sweep(const char* name, F f, int& pass, int& total) {
  int fails = 0;
  for (const Case& c : kCases) {
    Result r = f(c.qo, c.kv);
    bool ok = r.bad_o == 0 && r.bad_lse == 0;
    ++total;
    if (ok) ++pass; else ++fails;
    printf("%-34s qo=%4u kv=%5u  %s  max|dO|=%.3e max|dLSE|=%.3e bad_o=%zu bad_lse=%zu\n", name,
           c.qo, c.kv, ok ? "PASS" : "FAIL", r.max_err_o, r.max_err_lse, r.bad_o, r.bad_lse);
  }
  return fails;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "check";
  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, 0));
#ifdef FLASHINFER_PREFILL_MATRIX_CORE_CUH_
  matrix_core::mma_shape_t shape;
  CK(matrix_core::get_prefill_mma_shape(&shape));
  printf("device %s %s, prefill_mma_shape {%u, %u}\n", prop.name, prop.gcnArchName,
         shape.wave_size, shape.op_elems);
#else
  printf("device %s %s, no matrix-core header (emulated prefill)\n", prop.name, prop.gcnArchName);
#endif
  if (std::string(mode) == "check") {
    int pass = 0, total = 0;
    uint64_t seed = 1;
    // 10 configurations x 10 shapes, the grid of the porter's 2026-11-20 sweep.
    sweep("fp16 hd128 nfx1 causal ps16", [&](uint32_t q, uint32_t k) {
      return run<half, 1, 16, 1, 128, true>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("fp16 hd128 nfx2 causal ps16", [&](uint32_t q, uint32_t k) {
      return run<half, 2, 16, 1, 128, true>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("fp16 hd128 nfx2 noncausal ps16", [&](uint32_t q, uint32_t k) {
      return run<half, 2, 16, 1, 128, false>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("fp16 hd64 nfx2 causal ps16", [&](uint32_t q, uint32_t k) {
      return run<half, 2, 16, 1, 64, true>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("fp16 hd128 nfx2 causal ps16 g4", [&](uint32_t q, uint32_t k) {
      return run<half, 2, 16, 4, 128, true>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("bf16 hd128 nfx2 causal ps16", [&](uint32_t q, uint32_t k) {
      return run<nv_bfloat16, 2, 16, 1, 128, true>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("bf16 hd64 nfx1 noncausal ps16 g4", [&](uint32_t q, uint32_t k) {
      return run<nv_bfloat16, 1, 16, 4, 64, false>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("fp16 hd128 nfx2 causal ps1", [&](uint32_t q, uint32_t k) {
      return run<half, 2, 1, 1, 128, true>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("fp16 hd128 nfx1 causal ps8", [&](uint32_t q, uint32_t k) {
      return run<half, 1, 8, 1, 128, true>(q, k, 2, true, 0, seed++); }, pass, total);
    sweep("bf16 hd128 nfx2 noncausal ps32 g8", [&](uint32_t q, uint32_t k) {
      return run<nv_bfloat16, 2, 32, 8, 128, false>(q, k, 1, true, 0, seed++); }, pass, total);
    printf("TOTAL %d/%d passed\n", pass, total);
    return pass == total ? 0 : 1;
  }
  // time: 32 heads x 128, fp16, page 16, causal, num_frags_x as the dispatcher picks it.
  const Case shapes[] = {{1, 4096}, {128, 4096}, {1024, 1024}, {1024, 4096}, {4096, 4096},
                         {8192, 8192}, {16384, 16384}};
  int only = argc > 2 ? atoi(argv[2]) : -1;
  for (int i = 0; i < 7; ++i) {
    if (only >= 0 && i != only) continue;
    const Case c = shapes[i];
    int iters = c.qo >= 8192 ? 5 : 20;
    // "time1" forces num_frags_x = 1 for every shape.
    const bool nfx2 = c.qo > 64 && std::string(mode) != "time1";
    Result r = nfx2 ? run<half, 2, 16, 1, 128, true>(c.qo, c.kv, 32, false, iters, 7)
                    : run<half, 1, 16, 1, 128, true>(c.qo, c.kv, 32, false, iters, 7);
    printf("time qo=%5u kv=%5u nfx=%d  %.3f ms\n", c.qo, c.kv, nfx2 ? 2 : 1, r.ms);
  }
  return 0;
}
```

</details>
## Validation 2026-11-23 (linux-gfx1100, revalidate at `9cb8bee`) -- PASS

First run of the gfx11 WMMA prefill tiles on real RDNA3 hardware. Host: AMD Radeon Pro W7800 48GB (gfx1100, wave32, 70 CUs), `HIP_VISIBLE_DEVICES=1` (PCI 0x46; idle before every GPU run, `amd-smi metric --usage` 0 %; another job was busy on a different card), PyTorch 2.16.0a0+git6c3af02 / HIP 7.16.26385 (pip ROCm SDK), Python 3.11 (`/opt/python-3.11-venv`), all GPU runs under `ulimit -c 0`. Fork clone at `9cb8bee` (`moat-port`), submodules initialised, `git status --porcelain` empty before and after. `classify Quest c9828e2 9cb8bee` -> `mixed`, so a full run (this platform's prior `validated_sha` is `c9828e2`, before prefill and topk existed on HIP).

### Build (README Linux block, fresh build directory)

`agent_space/quest/build_readme.sh <fork>` = the README block verbatim with `-DCMAKE_HIP_ARCHITECTURES=gfx1100` after `rm -rf build`, `ninja -j $MAX_JOBS` (32):

```
cd quest/ops && rm -rf build && mkdir -p build && cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
ninja -j 32
ln -sf $PWD/_kernels*.so ../../
```

Wrapped `utils/timeit.sh Quest compile`: rc 0, 8/8, 0 errors, `CMAKE_BUILD_TYPE=Release` from the default, `--offload-arch=gfx1100` only. Wall **2840 s**, of which about 35 min is one single-threaded `ld.lld` device LTO step for `batch_prefill.cu` (MI300X: 555 s for the whole build). The spilling WMMA instantiations (below) are the likely cost in register allocation; not measured further. The old `find_package(Torch)` hazard on this host (recorded 2026-08-20) is gone: `torch.utils.cmake_prefix_path` now resolves into the wheel install.

### GPU tests (transformers 4.37.2, tokenizers 0.15.2, pytest 7.3.2 in `<job>/pylib-test`)

```
env PYTHONPATH=<fork>:<job>/pylib-test python3 -m pytest <fork>/quest/tests -p no:cacheprovider -v -rs
```

**202 passed, 28 skipped, 0 failed** (15.0 s). rope 64, estimate 9, decode_attention 6, approx_attention 42, decode_launch_shape 3, topk 49 + 15 skipped (k >= kv_len), prefill_attention 29 + 13 skipped (qo_len > kv_len). Same as MI300X at `9cb8bee`.

### fp64 reference check of prefill (gfx11 WMMA on hardware)

The porter's harness is not in the notes (it lives in another host's `agent_space`), so it was rewritten from its description: `agent_space/quest/harness/prefill_ref.cu` calls `BatchPrefillWithPagedKVCacheDispatched` directly (NHD, shuffled pages from a pool with 7 spare pages, output buffers pre-filled with 0xff), fp64 CPU reference with the bottom-right causal rule, output and base-2 LSE checked. Built with the extension's HIP flags (`harness/build.sh <tree> <out> gfx1100`: SDK clang, `-O3 -std=gnu++20 -ffp-contract=on`, the three `hip_compat` force-includes, `__HIP_NO_HALF_*`). Same 100-case matrix as the porter: qo x kv in {1x17, 13x33, 16x16, 51x400, 63x700, 77x129, 128x129, 244x1110, 311x311, 502x1110} x {fp16 hd128 nfx1 causal, fp16 hd128 nfx2 causal, fp16 hd128 non-causal, fp16 hd64, fp16 GQA 4, bf16 hd128, bf16 hd64 GQA 4 non-causal, page size 1, page size 8, page size 32 GQA 8} (nfx by the dispatcher's rule `qo * group > 64` except the two forced rows; 8 kv heads, fewer for GQA). Tolerances as the porter's: 5e-3 + 5e-3 |ref| (fp16), 2e-2 + 3e-2 |ref| (bf16), LSE 2e-2 + 2e-3 |ref|.

| build | result | max abs err |
|---|---|---|
| `9cb8bee` (gfx11 WMMA; device asm has 1872 `v_wmma_f32_16x16x16_f16` + 432 `_bf16`) | **100/100** | fp16 1.7e-3, bf16 1.2e-2, LSE 1.5e-2 |
| `2b938d8` (old shuffle emulation, same harness; validates the harness) | 100/100 | WMMA error smaller in 49 cases, equal in 48, larger in 3 (by <= 3.4e-4, bf16 16x16 case 6.5e-3 vs 6.1e-3) |
| negative control: copy of the `9cb8bee` header with gfx11 `acc_row` changed to `2 * i + 1 - lane / 16` (D row halves swapped) | 0/100 | smallest per-case max err 0.34 |

```
cd agent_space/quest/harness
./build.sh <fork> ./ref_9cb8bee gfx1100 && ./ref_9cb8bee check      # TOTAL 100/100 passed
```

The review's list of what a real Radeon run must confirm, answered:
1. The hardware matches the documented gfx11 layout (A/B row or column l % 16 with all 16 k, D rows `2i + l/16`): yes. 100/100 against fp64 with WMMA instructions in the code object, and swapping the D row halves fails every case, so the check discriminates the layout on this hardware.
2. The `_w32` builtins with lanes 16-31 replicating lanes 0-15 (k0 = 0 for all lanes): correct results in every case; all cases go through these operands.
3. The xor-16 P exchange on a real wave32: exercised by every case (P always feeds the second product through it); correct.
4. Q in registers vs Q in LDS on gfx11 (`nfx * nfy <= 8`): hd128 nfx1 and hd64 nfx1/2 keep Q in registers, hd128 nfx2 keeps it in LDS; both branches are in the matrix and pass. The end-to-end prefill instantiation is the Q-in-LDS one.
5. Time with the spills: below. The instantiation the model uses (`<1,16,true,kNone,2,8,4,4>` fp16) reports 256 VGPR and 348 B/lane scratch (`--cuda-device-only -S`, `; ScratchSize`), within the review's 340-448 B range; other instantiations 0-448 B.

### Prefill time against SDPA (`agent_space/quest/prefill_time.py`, same method as MI300X)

`quest.utils.prefill_forward` (includes the operator's per-call host work) vs `F.scaled_dot_product_attention` with the same alignment (`is_causal` where qo = kv, `causal_lower_right` where qo < kv, default backend selection). Causal, 32 heads x 128, fp16, page size 16, mean of 20 (5 for the large shapes) after 3 warmups, `utils/timeit.sh Quest test -- env PYTHONPATH=<fork>:<job>/pylib-test python3 agent_space/quest/prefill_time.py`:

| qo x kv | Quest prefill_forward | torch SDPA | ratio | max abs diff vs SDPA | kernel alone (harness) |
|---|---|---|---|---|---|
| 1 x 4096 | 0.450 ms | 0.174 ms | 2.58 | 6.1e-05 | 0.356 ms |
| 128 x 4096 | 0.680 ms | 0.879 ms | 0.77 | 1.2e-04 | 0.664 ms |
| 1024 x 1024 | 0.597 ms | 0.374 ms | 1.60 | 2.0e-03 | 0.462 ms |
| 1024 x 4096 | 2.614 ms | 2.260 ms | 1.16 | 2.4e-04 | 2.745 ms |
| 4096 x 4096 | 5.641 ms | 3.482 ms | 1.62 | 2.0e-03 | 6.175 ms |
| 8192 x 8192 | 21.947 ms | 13.946 ms | 1.57 | 2.0e-03 | 23.26 ms |
| 16384 x 16384 | 89.782 ms | 82.401 ms | 1.09 | 2.0e-03 | 93.27 ms |

Kernel alone: `./ref_9cb8bee time <qo> <kv> <iters>` (shuffled pages, which is why it can read slower than the operator's contiguous pages at large shapes). Old emulated path: there is no build switch for it in `9cb8bee` (the selector is by target), so the same harness was built from the `2b938d8` tree: **4096 x 4096 126.9 ms, 1024 x 1024 10.5 ms** (kernel alone) against 6.17 ms and 0.46 ms for WMMA: 20.5x and 22.8x faster. So with the spills the gfx1100 prefill is within 1.1-1.6x of SDPA on every shape from 1024 up, ahead of it at 128 x 4096, and 2.6x behind only at qo = 1 (operator host cost). Recorded as numbers, not a failure; whether the review's zero-spill nfz cap is worth it on gfx11 can follow from this table.

### End-to-end (transformers 4.45.2, tokenizers 0.20.3, accelerate 1.0.1, sentencepiece 0.2.0 in `<job>/pylib-e2e`)

The model fits (fp16 13.5 GB on a 48 GB card). `NousResearch/Llama-2-7b-chat-hf` at `351844e75ed0bcbbe3f10671b3c808d2b83894ee` (safetensors, downloaded into `<job>/models`; `do_sample` set to false in `generation_config.json`, original kept as `.orig`), `scripts/example_textgen.py` with only `MODEL_PATH` changed (`agent_space/quest/e2e/example_textgen.py`):

```
env PYTHONPATH=<fork>:<job>/pylib-e2e python3 agent_space/quest/e2e/example_textgen.py --method hg
env PYTHONPATH=<fork>:<job>/pylib-e2e python3 agent_space/quest/e2e/example_textgen.py --method quest --token_budget 1024
env PYTHONPATH=<fork>:<job>/pylib-e2e python3 agent_space/quest/e2e/example_textgen.py --method quest --token_budget 128
```

All rc 0. The output from `Input Sequence Length: 103` on is byte-identical across the three (md5 `b900af27...`) and is the same continuation as on MI300X ("Reason: The rabbit is the only one who is not working hard. ... can also mean \"lazy.\""). `rocprofv3 --kernel-trace --stats` of the budget-128 run: `BatchPrefillWithPagedKVCacheKernel<1,16,true,kNone,2,8,4,4,...>` 32 calls at 37.8 us (MI300X 32.6 us), `quest_radix_topk_one_block_kernel` 2130, `MaxPossibleSampleWithPagedKVCacheKernel` 2130, `BatchDecodeWithPagedKVCacheKernel` 3072, `VariableLengthMergeStatesKernel` 3072, `AppendPagedKVCachePrefillKernel` 32: the same counts as MI300X.

### CUDA gate

Not run here: this host has no CUDA toolkit now (`/opt/conda/envs` absent, no `nvcc`). `cuda-not-validated: no toolkit on this host`; the real-nvcc compile of `9cb8bee` is still owed on the gfx950 host. `prefill.cuh` is unchanged from `6c06912`, whose CUDA view was shown identical to `2b938d8` by preprocessing (review 2026-11-20).

### Content

`jargon.py --port Quest` clean (after `git fetch origin main:main` in the clone); the README's ROCm section documents the build used here; `audit-commits` reports only the known `6e7d8db` gap.

### Owed next (Quest at `9cb8bee`)

1. linux-gfx1201 (RDNA4 gfx12 WMMA layout, first hardware run), windows-gfx1151 (gfx11 WMMA, Windows build), linux-gfx90a (MFMA `bf16_1k`).
2. Real-nvcc compile of `9cb8bee` on the gfx950 host.
3. Possible, not required: the gfx1100 build time (one 35-minute device LTO step) may be worth a look alongside the spill reduction, since every gfx11 user pays it.

## Validation 2026-11-23 (windows-gfx1151, at `9cb8bee`) -- PASS

AMD Radeon 8060S (gfx1151, RDNA3.5 APU, wave32, 20 CUs, shared memory), Windows 11. `torch312_venv`: torch 2.12.0+rocm7.14.0a20260519, ROCm SDK 7.14 devel (HIP 7.13.26190, AMD clang 23.0.0), Python 3.12, transformers 4.37.2. CMake 3.31.0, Ninja, VS 2022 BuildTools x64 environment loaded. Fork at `9cb8bee` == `head_sha`, `git status --porcelain` shows only the five untracked runtime DLLs staged in `quest/`; no fork edit, no push. Full revalidation (`mixed` delta, Windows has no binary-equivalence verdict). First run of the gfx11 WMMA prefill on gfx1151 (RDNA3.5). Scripts and logs: `agent_space/quest_porter/val_9cb8bee/`. Laptop limits kept: one GPU process at a time, no timing shape above 4096 x 4096, no 7B model.

### Build (README Windows block as written, clean in-tree)

Old `quest/ops/build` and `_kernels*.pyd` moved aside first. `powershell -NoProfile -ExecutionPolicy Bypass -File readme_build.ps1` (process-scope bypass because this host's policy refuses unsigned local scripts; the script removes `CC CXX HIPCXX PYTORCH_ROCM_ARCH CMAKE_BUILD_TYPE`, runs the venv `Activate.ps1`, then the README block verbatim plus `$LASTEXITCODE` checks), wrapped `utils/timeit.sh Quest compile`; the only change from the README is `ninja -j 16` (16 physical cores):

```
cd quest\ops
mkdir build; cd build
cmake -GNinja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 `
  "-DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" ..
ninja -j 16
copy _kernels*.pyd ..\..\
```

rc 0, 8/8, **898 s** wall (15m). The expected 27-47 minute `batch_prefill.cu` wait did not occur here: no `-fgpu-rdc` on Windows, so no device LTO step (the Linux cost is the single-threaded `ld.lld` LTO). `CMAKE_BUILD_TYPE=Release`, `--offload-arch=gfx1151` on 8 of 8 compile lines, 0 errors, 18 nodiscard-`hipError_t` warnings (the known count), the link prints only the known unused `--ld-path` warning.

### GPU tests (`utils/timeit.sh Quest test`)

```
PYTHONPATH=<fork> python -m pytest <fork>\quest\tests --rootdir <fork> -p no:cacheprovider -v -rs
```

**202 passed, 28 skipped, 0 failed** (18.4 s). rope 64, estimate 9, decode_attention 6, approx_attention 42, decode_launch_shape 3, topk 49 + 15 skipped (k >= kv_len), prefill_attention 29 + 13 skipped (qo_len > kv_len). Same as `2b938d8` here and as the four Linux GPUs at `9cb8bee`. Test files read first: no repeat or stress loops (the largest run is a parametrized sweep; whole suite 18 s), nothing skipped on safety grounds.

### fp64 reference check of prefill (the harness of `notes-prefill-harness.md`)

`prefill_harness.cu` and `prefill_time.py` extracted unchanged from `notes-prefill-harness.md`; `build.sh` adapted to Windows (SDK `clang++.exe -x hip` with the extension's compile line flags `-D_DLL -D_MT -Xclang --dependent-lib=msvcrt -fms-runtime-lib=dll -fms-extensions`, link `amdhip64.lib`, the five ROCm runtime DLLs beside the executables). Built one binary per configuration (about 5 s each) and run one at a time:

```
bash build.sh <n>        # n = 0..9
./mc_<n>.exe             # prints the device mma shape: wave=32 op_elems=16
```

**100/100** cases pass (10 configurations x 10 shapes): fp16 hd128 nfx1 causal, nfx2 causal, nfx2 non-causal, hd64 nfx1, GQA4 nfx2, bf16 hd128 nfx2, bf16 hd64 GQA4 nfx1 non-causal, page size 1, 8, and 32 with GQA8. Largest errors seen: output 1.0e-3 (fp16, 16 x 16), LSE 5.5e-4 in the first configuration; every case within the harness tolerances. Negative control (`-DNEGCTRL=true`, reference uses the opposite causal flag, nfx2 causal): 1/10 cases pass, max |dO| 2.9, so the check discriminates. This is the same 100/100 as linux-gfx1100 (gfx11 WMMA layout), now on RDNA3.5.

### Prefill time against SDPA (`prefill_time.py`, SHAPES cut to 1024 x 1024 and 4096 x 4096)

`utils/timeit.sh Quest test -- powershell ... time.ps1`; causal, 32 heads x 128, fp16, page size 16, mean of 20 after 3 warmups; torch SDPA on this torch build falls to a slow path (Mem Efficient attention is experimental on this GPU and off):

| qo x kv | Quest prefill_forward | torch SDPA | ratio | max abs diff vs SDPA |
|---|---|---|---|---|
| 1024 x 1024 | 2.031 ms | 17.224 ms | 0.12 | 2.0e-03 |
| 4096 x 4096 | 17.761 ms | 270.908 ms | 0.07 | 2.0e-03 |

Quest is 8.5x and 15x faster than SDPA here; not comparable with the discrete-GPU ratios (SDPA is the slow side on gfx1151). Not run: the kernel-only CFG 100 timing (it includes an 8192 x 8192 shape), 8192 and 16384 shapes.

### End-to-end

Llama-2-7b-chat was not run: the earlier Windows sections show it was never run on this laptop (13.5 GB fp16 on the shared-memory APU); covered on four Linux GPUs. The random-weight 0.96 GB Llama-architecture check of 2026-11-19 (`agent_space/quest_e2e/cmp.py 1024 30`, HF vs Quest, same weights) was rerun: step-0 (prefill) logits max |diff| 0.0093 at logit range 5.95, first greedy divergence at step 5 where the HF top1-top2 margin is 0.0039 (a one-ulp tie), steps before it max |diff| 0.0098. Same numbers as at `6e7d8db` (0.0098, step 5, 0.0039): prefill and dense decode agree to fp16 rounding. Weaker evidence than a real-model text match, as recorded before.

### CUDA gate, content, integrity

CUDA gate: not run here (Windows host); recorded at `9cb8bee` by the linux-gfx950 host (real nvcc, PASS). `python utils/jargon.py --port Quest` clean. The README documents the Windows ROCm build and it is the block above. No machine instability (no driver reset, no device loss) during any GPU run.

State: windows-gfx1151 `completed` at `9cb8bee`.

## Ruling 2026-11-23 (jeffdaily): prefill within 2x of SDPA on every validated GPU before the pull request

Asked whether anything still blocks the upstream pull request, jeffdaily answered "Quest decision, do Option C": before the pull request, do both performance follow-ups the validators found, so that no validated GPU misses the plan's target of prefill within 2x of PyTorch's `scaled_dot_product_attention`. (1) gfx950: use the wider MFMA instructions (16x16x32) and bring the long shapes within 2x if the hardware allows, saying plainly what limits it if not; gfx942 and gfx90a keep their code path and results. (2) wave32 Radeons: cap `num_frags_z` (the linux-gfx1201 validator's scratch cap: 512 -> 176 B/lane of spill, 2.44 -> 2.07 ms at 4096^2), wave32 WMMA targets only, chosen by measurement, and measure whether it shortens the Radeon build of `batch_prefill.cu`. Changes stay inside the HIP-only prefill header and the `USE_HIP` blocks; the CUDA branch must stay byte-identical after preprocessing, proven with real nvcc in the same round.

## Porter round 2026-11-24 (linux-gfx950): gfx950 16x16x32 MFMA with direct loads, wave32 tile cap

Host: AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=5` (pinned by the job; devices 0, 2, 3 and 4 belong to colleagues' benchmarks and were never touched), PyTorch 2.16.0a0+git6c3af02 / HIP 7.16.26385 (pip ROCm SDK clang), all GPU runs under `ulimit -c 0`, builds at `MAX_JOBS`=25 or fewer. Third start of this job: the two earlier attempts (stopped for GPU conflicts) pushed nothing; their scratch (a 16x16x32 + `ds_read_b64_tr_b16` variant, a `global_load_lds` direct-load variant, the harness and profiling scripts) was copied into this job and reused. The work lock carried the second attempt's stamp; `set-state Quest linux-gfx950 porting` re-stamped it to this job.

**Status at the end of this round: the change is complete in the fork working tree and checked with the fp64 harness on gfx950 hardware, with the WMMA layout emulation, and by preprocessing for CUDA; it is NOT committed or pushed, and `head_sha` is still `9cb8bee`.** The host's root filesystem (3.5 TB overlay shared by every container) filled up during the round (58-100 MB free, consumed again within seconds after this job freed 13 GB of its own model copy and 113 MB of profiler output), which killed the README extension build at link time (`clang-linker-wrapper: error: 'clang' failed`), the Radeon compile-time runs (`llvm-offload-binary command failed`) and two harness builds (`LLVM ERROR: IO failure on output stream: No space left on device`). Without a full extension build at the final tree there is no test-suite, end-to-end or operator-timing evidence for it, so nothing was pushed (pushing would also have made every validated platform revalidate a head its own porter had not built). The full diff against `9cb8bee` is in `notes-prefill-round-2026-11-24.md`, so the next porter starts from it (`git apply`), not from scratch.

### What the change does (2 files, +341 -30, all HIP-only)

- `prefill_matrix_core.cuh`: gfx950 gets its own selector value (`QUEST_PREFILL_MMA` 2): `v_mfma_f32_16x16x32_{f16,bf16}` (k = 32, 8 operand elements per lane), V kept row-major in LDS and its V^T operand read with `ds_read_b64_tr_b16` (`load_v_tr`), and K/V tiles copied global -> LDS with `global_load_lds_dwordx4` into two static buffer pairs one tile ahead (`kv_tile_t`, `kv_tile_pages`, `kv_tile_copy`, one barrier per tile). When the page size is a multiple of the rows one copy instruction covers (4 rows at head 128, 8 at head 64: page sizes 8, 16, 32), every lane of an instruction reads the same page, so the page index is read into a scalar at the copy (`readfirstlane`) and only the last, partial tile keeps per-lane masking. gfx942/gfx90a (selector 1), gfx11 and gfx12 keep their code; the earlier attempt's direct-load design is kept, with the scalar-page change and the deferred page read added here.
- Two policy functions shared by host and device, the same pattern as `q_in_registers`: `max_num_frags_x(shape)` (1 on gfx950, else 2) and `kv_frags(num_frags_x, num_frags_y, num_frags_z, shape)`, the K/V fragments per tile the kernel uses: 2 on gfx950, and 2 on wave32 when the query tile is in registers and the fp32 accumulators (S^T, O^T) plus query operands at `num_frags_z` exceed 128 registers per lane; otherwise `num_frags_z`.
- `prefill.cuh`: the kernel's template parameter is named `kernel_num_frags_z` under `USE_HIP` and the body defines `num_frags_z = kv_frags(...)` in device passes, so every shared line (iteration counts, mask iteration, shared memory pointers, page step) sees the tile actually used, the kernel's mangled name is unchanged, and the CUDA text is untouched. The host sizes dynamic shared memory with the same function (0 on gfx950, whose tiles are static), and `split_qo_indptr` takes `max_num_frags_x` under `USE_HIP`. No change to the dispatcher's `SWITCH_*` lines.
- Header comment updated for gfx950. `hip_compat/README.md` and `README.md` stay as they are: their statements (emulation only on GPUs without the 16x16x16 instructions: gfx908, RDNA2, gfx1250) remain true.

### gfx950: where the time went, measured (kernel alone, standalone harness, 32 heads x 128, fp16, causal, page 16, best of 5 interleaved processes)

`agent_space/quest/harness/ab.sh <qo> <bins...>` runs every binary once per round for 5 rounds and keeps each one's best, so the load a colleague puts on other GPUs of this host affects all variants alike.

| step (4096 x 4096) | kernel ms |
|---|---|
| `9cb8bee` (16x16x16 MFMA, register-staged loads, nfx 2, nfz 4) | 0.990 |
| + 16x16x32 MFMA, V row-major + `ds_read_b64_tr_b16` (earlier attempt's `exp` tree) | 0.746 |
| + `global_load_lds` double-buffered direct loads (earlier attempt's `dl` tree) | 0.479 |
| + nfx 1, nfz 2: 192-206 VGPR, 0 AGPR, two wavefronts per SIMD instead of one (nfx 2 is 256 VGPR + 143-152 AGPR) | 0.446 |
| + scalar page per copy instruction, branch-free full tiles | 0.421 |
| + page index read into a scalar at the copy, not after its load (the compiler had put `s_waitcnt vmcnt(0)` right after the next tile's page loads, serialising the copy with the compute) | **0.413** |

Same steps at 8192^2 / 16384^2: `9cb8bee` 3.20 / 10.19 ms, final 1.19 / 4.33 ms. Short shapes: 1 x 4096 0.456 -> 0.146, 128 x 4096 0.464 -> 0.149, 1024 x 4096 0.451 -> 0.201 ms. nfx 2 / nfz 4 with all the copy changes is 0.460, nfx 2 / nfz 2 0.533: the occupancy step is what makes nfx 1 the choice.

Tried and measured without gain (kept out): skipping the O rescale when no column maximum moved, and packed `v_cvt_pk_f16_f32` for P (0.422 vs 0.413 at 4096^2, 4.30 vs 4.34 at 16384^2: noise or worse); three LDS XOR permutations for the K/V tiles (row % 8, (2r + 2r/16) % 16, row % 16: 0.410-0.413, although `SQ_LDS_BANK_CONFLICT` is ~44 % of `SQ_LDS_IDX_ACTIVE`); removing the barrier's `vmcnt` wait (no change, so global latency is covered); fixed trivial copy addresses (no change after the scalar-page step; 0.447 -> 0.348 before it, which is what pointed at the address math). Counter totals (`rocprofv3 --pmc`, `prof/pmc.sh`): VALU instructions about 8x the MFMA count; a cycle count (32 MFMA x 16 cycles and ~290 VALU x 4 cycles per wavefront iteration) puts pure issue at about half the measured ~3800 SIMD-cycles per wavefront iteration. What remains is dependency latency inside a wavefront (QK -> max -> exp -> PV) with only two wavefronts per SIMD to hide it; closing that needs a software-pipelined (ping-pong) loop of the kind hand-tuned attention kernels use, a rewrite beyond this port.

What SDPA runs here (`rocprofv3 --kernel-trace`, `prof/one.py 4096`): `flyc_attn_fwd`, 219 us at 4096^2, 512 workgroups of 512 threads (8 wavefronts), 128 VGPR + 128 AGPR, 68 KB LDS: a hand-tuned assembly kernel. Quest's prefill kernel inside the operator in the same trace: 497 us (the `dl` build, nfx 2).

**Kernel alone against SDPA (SDPA best of 2 processes, `MINMED=1 LOOP_MS=100 REPS=7 WARM_MS=1000`):** 4096^2 0.413 / 0.211 = 1.96x; 8192^2 1.19 / 0.722 = 1.65x; 16384^2 4.33 / 2.68 = 1.62x. So the kernel meets the target at the long shapes and only just at 4096^2.

**Through the operator it cannot, and the reason is upstream host code, not the kernel.** With the `dl` build (kernel 0.479 ms), `prefill_time.py` gave 0.769-0.778 ms at 4096^2 (SDPA 0.211-0.213, 3.6x), 1.95 ms at 8192^2 (2.7x), 5.43-5.45 ms at 16384^2 (2.0x): 0.29-0.36 ms per call on top of the kernel. `prefill_with_paged_kv_cache` (`quest/ops/csrc/batch_prefill.cu`) builds two index tensors with `torch::tensor` (host-to-device copies), calls `kv_indices[-1].item()` (a device-to-host sync), and `split_qo_indptr` copies `qo_indptr` back to the host; each sync drains the GPU, which then idles while the host prepares the launch (111 us median idle gap between consecutive prefill kernels under the tracer). On MI300X the same overhead was ~0.07 ms; on this host it is 4x that, plausibly the shared host's load. With the final kernel (0.413 ms) the operator would be about 0.70 ms at 4096^2, about 3.3x. Changing that host code is outside this round's boundary (shared CUDA code under `quest/ops/csrc`), so it is a decision for jeffdaily: keep the kernel-alone figure as the gfx950 evidence, or allow a change to the operator's host path (both syncs can be avoided for batch size 1, which is all Quest uses).

### wave32 tile cap: measured by spill (compile only; this host has no WMMA)

`agent_space/quest/harness/res.sh <tree> <arch> <cfg>` (harness device compile with `-Rpass-analysis=kernel-resource-usage`; `t9cb` = `git archive 9cb8bee`, `v2` = the change), fp16, causal, page 16, the instantiation the dispatcher launches (nfz 4) and its nfz 2 sibling, scratch B/lane / VGPR / occupancy:

| target, shape | `9cb8bee` nfz 4 | `9cb8bee` nfz 2 | change, launched instantiation |
|---|---|---|---|
| gfx1201 head 128 nfx 2 (Q in registers) | 512 / 256 / 5 | 176 / 256 / 5 | **176** / 256 / 5 (cap) |
| gfx1201 head 64 nfx 2 | 16 / 256 / 5 | 0 / 214 / 7 | **0** / 214 / 7 (cap) |
| gfx1201 head 128 nfx 1 | 0 / 242 / 5 | 0 / 235 / 6 | 0 / 242 / 5 (no cap) |
| gfx1201 head 64 nfx 1 | 0 / 225 / 6 | 0 / 146 / 9 | 0 / 225 / 6 (no cap) |
| gfx1100/gfx1151 head 128 nfx 2 (Q in LDS) | 348-396 / 256 / 5 | 424-448 / 256 / 5 | 348-396 (no cap: two fragments spill more) |
| gfx1100/gfx1151 head 64 nfx 2 | 260 / 256 / 5 | 0 / 254 / 5 | **0** / 254 / 5 (cap) |
| gfx1100/gfx1151 head 128 nfx 1 | 176 / 256 / 5 | 100 / 253 / 5 | **100** / 253 / 5 (cap) |
| gfx1100/gfx1151 head 64 nfx 1 | 0 / 215 / 7 | 0 / 218 / 6 | 0 / 215 / 7 (no cap) |

The rule in `kv_frags` reproduces exactly the "cap" column. The validator's measured case (gfx1201 head 128 nfx 2: 2.44 -> 2.07 ms, 34.9 -> 27.8 ms) is in it; the other capped cases are chosen on spill alone and need the Radeon validators' timing. gfx11 head 128 nfx 2, the operator's main prefill shape on gfx11, is deliberately not capped.

Radeon compile time of `batch_prefill.cu` before and after (`agent_space/quest/bp/bp.sh <tree> <arch> <out>`, the extension's exact compile line with the target swapped): **not obtained.** Both gfx11 "before" runs finished device code generation after 827 s and 831 s (three targets compiling in parallel), then failed writing the output (`llvm-offload-binary command failed`, the full disk); the "after" runs and the gfx1201 runs were still compiling when the disk filled. The structural expectation, not a measurement: the gfx12 head-128 nfx-2 nfz-4 instantiation now compiles as the nfz-2 body (176 instead of 512 B/lane of spill), so gfx12 should get faster to build; on gfx11 the most expensive instantiation (head 128 nfx 2) is unchanged.

### Correctness evidence for the change (all at the final working tree)

- fp64 reference harness on gfx950 hardware (`notes-prefill-harness.md` method; `harness/checkall.sh ../v2 fc`, 10 configurations x 10 shapes: fp16/bf16, head 64/128, nfx 1/2, GQA 4/8, page sizes 1/8/16/32, causal and not): **100/100**. Negative control (`-DNEGCTRL=true`, reference with the opposite causal flag), nfx 1 and nfx 2: 1/10 each (the 1 x 17 shape, where both rules agree).
- RDNA3 and RDNA4 index logic on this GPU: new test-only `harness/wmma_emu.h` (the four WMMA builtins renamed to a shuffle emulation of their documented layouts, each 32-lane half of a wave64 as one wave32), built with `-include wmma_emu.h -Xarch_device -DQUEST_PREFILL_MMA=11` (or `=12`); the binaries report `prefill_mma_shape` {32, 16} and {32, 8}, so the host applied the wave32 `kv_frags` rule: **100/100 each**. Negative control: gfx12 with the gfx11 accumulator row formula: 0/80 in the 8 configurations that built (2 failed to build: disk).
- gfx942 and gfx90a: no device-code change intended (`kv_frags` returns `num_frags_z`, `max_num_frags_x` 2, smem formula equal). An assembly diff of the harness for both was started but its output was lost to the full disk; owed.
- CUDA: `prefill.cuh` with `#include` lines removed through `cpp -P -undef -nostdinc`, at `9cb8bee` and at the change, with and without `-D__CUDACC__ -D__CUDA_ARCH__=900`: identical, md5 `02d6a90c0d2eea0259c7a25b34a3f580` (970 lines), the digest every earlier record has. `prefill_matrix_core.cuh` is reached only through the `USE_HIP` include. The real-nvcc compile was not run (no disk for a build directory); owed.
- An earlier build of the same gfx950 code (before the last wave32-only refinement of `kv_frags`) compiled `batch_prefill.cu` for gfx950 with the extension's exact line: rc 0, 333 s.

### Deferral `quest-prefill-matrix-cores`

Not closed (a person rules). Its subject, prefill on the matrix cores, was completed by `6c06912`/`9cb8bee`; this round is the performance follow-up of the 2026-11-23 ruling. When the change lands it completes item 2 and the kernel half of item 1; whether gfx950 meets the target through the operator depends on the host-code decision above.

### Owed next (Quest, change not yet pushed)

1. On a host with disk: `git apply` the patch from `notes-prefill-round-2026-11-24.md` on `9cb8bee`, build with the README command for gfx950, run the seven suites (expect 202 passed, 28 skipped), the fp64 harness, end-to-end (`hg`, Quest 1024 and 128, byte-identical text), operator timing (`prefill_time.py`, best of 5 processes), the gfx942/gfx90a assembly diff, the real-nvcc compile, and the Radeon `batch_prefill.cu` compile times; then commit (`[ROCm] Speed up prefill on gfx950 and the wave32 GPUs`), push, `advance-head`, `set-state ... ported`.
2. jeffdaily: the operator host-path decision (gfx950 through the operator is about 3.3x at 4096^2 because of ~0.3 ms of host synchronisation per call; kernel alone 1.96x).

### Addendum: Radeon compile times of `batch_prefill.cu` (2026-11-24 round), not conclusive

Six runs of `agent_space/quest/bp/bp.sh` (the extension's exact compile line, target swapped, own temporary directory each), before (`9cb8bee`) and after (the change) for each target, started within a minute of each other beside the gfx950 extension build, on a host whose disk was full. Wall seconds and how each ended: gfx1100 before 835 s (failed opening its output object: no space), after 1119.5 s (rc 0); gfx1151 before 831 s (first attempt, failed in `llvm-offload-binary`), after 510 s (failed at `amdgcn-link`: disk); gfx1201 before 698 s (failed at `amdgcn-link`: disk), after 788.5 s (rc 0). The spread has no consistent sign (gfx1151 39 % faster, gfx1100 34 % and gfx1201 13 % slower) and is larger than anything the change can do to these targets, so it reflects the host's varying load, not the change. A clean sequential pair on an idle host is owed; the Radeon validators' full builds will also show it.

## Porter round 2026-11-24 resumed (linux-gfx950, job roles6d): building and checking the saved change

Host: AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=5` (pinned by the job; devices 0, 2, 3 and 4 belong to colleagues and were never touched), PyTorch 2.16.0a0+git6c3af02 / HIP 7.16.26385, Python 3.11 (`/opt/python-3.11-venv`), all GPU runs under `ulimit -c 0`, `MAX_JOBS`=64 (exported). `set-state Quest linux-gfx950 porting` re-stamped the lock from the finished job `roles6c` (resume, expected). Fresh fork clone at `9cb8bee`; the patch extracted from `notes-prefill-round-2026-11-24.md` is byte-identical to the earlier job's `round.patch`, `git apply --check` clean, and the resulting two headers are byte-identical to that job's working tree. Free space on the job's filesystem at the start: 170 GB.

1. **Build** (free before: 169 GB): `agent_space/quest/build_readme.sh` (README Linux block verbatim, `-DCMAKE_HIP_ARCHITECTURES=gfx950`, after `rm -rf build`, `ninja -j $MAX_JOBS`), wrapped `utils/timeit.sh Quest compile`: rc 0, 8/8, 325 s, 0 errors; 18 warning lines, the same set (by file and count) as the `9cb8bee` gfx950 validation build: `ignoring return value of type 'hipError_t' declared with 'nodiscard'` (pre-existing host code). Free after: 169 GB.
2. **Test suites**: `env PYTHONPATH=<fork>:<job>/pylib-test python -m pytest <fork>/quest/tests -p no:cacheprovider -v -rs` (transformers 4.37.2, the same `pylib-test` as the earlier rounds), wrapped `timeit.sh Quest test`: **202 passed, 28 skipped, 0 failed** (19.0 s); the 28 skips are the usual 13 `qo_len > kv_len` (prefill) and 15 `k >= kv_len` (topk). Same as every record since `2b938d8`.
3. **fp64 reference harness** (`notes-prefill-harness.md` method, `harness/checkall.sh <fork> fc`, built against the fork tree; device reports `mma shape wave=64 op_elems=8`, the 16x16x32 path): **100/100** (10 configurations x 10 shapes). Negative control (`-DNEGCTRL=true`), CFG 0 (nfx 1) and CFG 1 (nfx 2): 1/10 each (the 1 x 17 shape, where both causal rules agree), as expected.
4. **End-to-end** (`NousResearch/Llama-2-7b-chat-hf` at `351844e7`, read in place from the 2026-11-23 gfx950 validation job's `models` directory, `do_sample` false; `agent_space/quest/e2e/example_textgen.py`, upstream's script with only `MODEL_PATH` changed; `env PYTHONPATH=<fork>:<job>/pylib-e2e python3 example_textgen.py --method hg`, `--method quest --token_budget 1024`, `--method quest --token_budget 128`, each wrapped `timeit.sh Quest test`): all rc 0. Input length 103, so the 103 x 103 prefill runs the new gfx950 kernel. Output with Quest's two banner lines removed (`grep -v '^Quest allocates\|^Token budget'`): md5 `b900af27eeb4170a5c47bc5940a2ca60` for all three, and the same digest as the earlier job's runs on this host ("Reason: The rabbit is the only one who is not working hard. ... can also mean \"lazy.\""). **Byte-identical.**
5. **CUDA, real nvcc** (the 2026-11-19 recipe: `<job>/nvcc/run.sh <tree> build-<tree> /usr/local/cuda`, then `ninja -C build-<tree> -j32 _kernels`; CUDA 12.8 `nvcc` V12.8.93, g++-11 11.5.0 toolchain file, `torch==2.5.0+cu124` in `$FLEET_VENV`, `-DCMAKE_CUDA_ARCHITECTURES=80`, no NVIDIA GPU so compile and link only). Trees: `<job>/nvcc/base` = fork at `9cb8bee`, `<job>/nvcc/v2` = `9cb8bee` + the change (copies of the fork clone with submodules). Both configure rc 0 and build `_kernels` **8/8, rc 0, 0 warning or error lines** (308 s / 311 s, in parallel); both modules 18,714,200 B. The 7 nvcc compile lines are identical after path normalization (`-gencode arch=compute_80,code=sm_80`, no `USE_HIP`).
   - Objects: the per-tree objects differ only through paths (`base` vs `v2` are different lengths, which moves embedded `__FILE__` strings) and nvcc's per-process names. To compare the code itself, the two header states were compiled at **one path** with the exact nvcc lines (`<job>/nvcc/cmp/same.sh base|v2`: copy that state's `prefill.cuh` and `prefill_matrix_core.cuh` into tree `s`, run the 7 lines from `ninja -t commands` with `-o`/`-MF` redirected). With nvcc's process-specific strings masked (`tmpxft_<pid>_<n>` temp names and the `_INTERNAL_<hash>_..._cu_<hash>_<pid>` anonymous-namespace id, same length; `cmp/norm2.py`), **all 7 objects are byte-identical** between `9cb8bee` and the change, including `batch_prefill.cu.o` (16.2 MB, host code and the sm_80 cubin).
   - Preprocessing: `prefill.cuh` with `#include` lines removed through `cpp -P -undef -nostdinc`, at `9cb8bee` and at the change, plain and with `-D__CUDACC__ -D__CUDA_ARCH__=900` and `=800`: identical, md5 `02d6a90c0d2eea0259c7a25b34a3f580` (970 lines), the digest of every earlier record. With `-DUSE_HIP` they differ (as intended). `prefill_matrix_core.cuh` is included only under `#if defined(USE_HIP)` (`prefill.cuh:45-47`).
   - **The CUDA view is unchanged, and so is nvcc's output.**
6. **gfx942 and gfx90a device code, before and after** (`agent_space/quest/bp/asm.sh <tree> <arch> <out.s>`: `batch_prefill.cu` compiled with the extension's exact HIP compile line, `-c` swapped for `-S --cuda-device-only`, one target; trees `<job>/nvcc/base` (`9cb8bee`) and `<job>/nvcc/v2`). All rc 0 (gfx942 271/272 s, gfx90a 424/429 s). The whole translation unit's device assembly (165.8 MB gfx942, 170.9 MB gfx90a; 1536 kernels and 165,888 `v_mfma` lines in both trees; every prefill instantiation the dispatcher can launch) differs, after replacing the tree path, in **14 lines per target and no instruction**: (a) the device constant `matrix_core::prefill_mma_shape` the host reads to size shared memory grew from 8 to 16 bytes with its two new fields, `k` = 16 and `static_kv` = 0 (`{64, 4}` -> `{64, 4, 16, 0}`); (b) the compiler's per-translation-unit id `__hip_cuid_<hash>` (a hash of the source, 5 lines). No kernel's code, register count, LDS size or metadata changed: gfx942 and gfx90a keep their code exactly. Diffs kept as `agent_space/quest/asm/diff_gfx9{42,0a}.txt`; the `.s` files were deleted after the diff.
   - Note on the earlier round's `bp/bp.sh` (also used for the Radeon compile times): its `sed -e "s#OUT#$O#"` replaced the first `OUT` in the template, which is inside `-DHIPBLASLT_OUTER_VEC`, not the `-o OUT` argument, so those runs wrote their object to a file named `OUT` in the temporary directory and defined a mangled macro (unused by Quest). The wall times are unaffected; fixed here (`s#-o OUT #-o $O #`).
7. **Prefill time** (32 heads x 128, fp16, causal, page size 16; GPU 5 only).
   - Kernel alone (standalone harness, `harness/ab.sh <qo> bin/b_100 bin/v2_101 bin/v2_100`: 5 rounds, each binary once per round, best per binary; `b_100` = `9cb8bee` at nfx 2, the dispatcher's choice there for these shapes; `v2_101` = the change at nfx 1, the dispatcher's choice on gfx950 now; `v2_100` = the change forced to nfx 2, which the dispatcher no longer launches on gfx950), ms:

| qo x kv | `9cb8bee` | change (nfx 1) | change forced nfx 2 |
|---|---|---|---|
| 1 x 4096 | 0.455 | **0.147** | 0.267 |
| 128 x 4096 | 0.463 | **0.149** | 0.274 |
| 1024 x 4096 | 0.451 | **0.202** | 0.273 |
| 4096 x 4096 | 0.991 | **0.414** | 0.544 |
| 8192 x 8192 | 3.218 | **1.188** | 1.857 |
| 16384 x 16384 | 10.240 | **4.322** | 5.788 |

     These reproduce the earlier round's figures (0.413 / 1.19 / 4.33 ms) to within 0.3 %.
   - Through the operator (`agent_space/quest/prefill_time.py`, best of 5 separate processes, each `MINMED=1 LOOP_MS=100 REPS=7 WARM_MS=1000`; range across the 5 processes in parentheses; `pt_v2_{1..5}.out`):

| qo x kv | Quest `prefill_forward` | torch SDPA | ratio | kernel alone | operator - kernel | kernel / SDPA | max abs diff vs SDPA |
|---|---|---|---|---|---|---|---|
| 1 x 4096 | 0.232 (0.232-0.240) ms | 0.106 (0.106-0.106) ms | 2.19 | 0.147 ms | 0.085 ms | 1.38 | 6.1e-05 |
| 128 x 4096 | 0.285 (0.285-0.307) ms | 0.140 (0.140-0.140) ms | 2.04 | 0.149 ms | 0.136 ms | 1.06 | 1.2e-04 |
| 1024 x 1024 | 0.245 (0.245-0.274) ms | 0.041 (0.041-0.041) ms | 5.98 | not measured | -- | -- | 2.0e-03 |
| 1024 x 4096 | 0.437 (0.437-0.474) ms | 0.122 (0.122-0.123) ms | 3.58 | 0.202 ms | 0.235 ms | 1.65 | 1.2e-04 |
| 4096 x 4096 | 0.676 (0.676-0.722) ms | 0.210 (0.210-0.213) ms | 3.22 | 0.414 ms | 0.263 ms | 1.97 | 2.0e-03 |
| 8192 x 8192 | 1.498 (1.498-1.537) ms | 0.717 (0.717-0.724) ms | 2.09 | 1.188 ms | 0.310 ms | 1.66 | 2.0e-03 |
| 16384 x 16384 | 4.770 (4.770-4.941) ms | 2.676 (2.676-2.681) ms | 1.78 | 4.322 ms | 0.448 ms | 1.62 | 2.0e-03 |

     At `9cb8bee` on this GPU (2026-11-23 validation, same method) the operator was 1.204 ms at 4096^2 (5.30x), 3.548 ms at 8192^2 (4.41x), 10.53 ms at 16384^2 (3.76x). Now 3.22x, 2.09x and 1.78x. The kernel alone is within 2x of SDPA at every shape it was measured at (1.97x at 4096^2, the closest); through the operator only 16384^2 is.
   - **Operator overhead at the new code, for the host-path decision.** `operator - kernel` above is 0.085-0.136 ms at qo <= 128, 0.26 ms at 4096^2, 0.31 at 8192^2, 0.45 at 16384^2. It grows with the shape, so it is not all host synchronisation. A kernel trace splits it (`rocprofv3 --kernel-trace`, `prof/one.py <n>`: 50 operator calls back to back, then 50 SDPA calls): at 4096^2 the prefill kernel inside the operator takes 405.7 us min / 434.8 us median, and the GPU sits idle 77.8 us min / 92.8 us median between consecutive prefill kernels (the only other kernel between them is one `__amd_rocclr_copyBuffer`, the host-to-device index copy); launch to launch 489 us min / 538 us median. At 16384^2: kernel 4346 / 4464 us, idle 105 / 125 us. So the host path (`torch::tensor` copies, `kv_indices[-1].item()`, `split_qo_indptr`) leaves the GPU idle about 0.08-0.13 ms per call on this host now (the earlier round's 111 us median gap is the same thing), and the kernel inside the operator runs a few percent slower than the harness's best (traced median 0.435 vs best-of-5 0.414 ms). Not explained: the traced launch-to-launch time (0.49-0.54 ms at 4096^2) is 0.14-0.19 ms below what `prefill_time.py` reports for the same call (0.676 ms best); the profiler run is one process right after the timing runs, so clock state or the event-timing loop may account for it, not checked further. Read either way: removing the syncs (batch size 1) would remove the ~0.09 ms idle gap and whatever host work it hides, which puts the operator at roughly 0.49-0.59 ms at 4096^2, 2.3-2.8x of SDPA. On this GPU the host path alone does not bring 4096^2 within 2x; the kernel is at 1.97x by itself. The host code was not touched (jeffdaily's open decision).
8. **Committed and pushed** (items 1 to 3 passed): fork commit `517c6a746eebc5e0b5a6cc23416e1be089d95885` on top of `9cb8bee` (no amend, plain push `9cb8bee..517c6a7 moat-port`; `pr-state` none, so `moat-port` is not frozen), title `[ROCm] Speed up prefill on gfx950 and the wave32 GPUs` (54 chars), body: problem, the three gfx950 changes, the wave32 cap with the measured and spill-chosen cases told apart, the shared policy functions, gfx90a/gfx942 unchanged (constant fields only), CUDA unchanged, AI-agent disclosure, Test Plan (gfx950 build and tests, fp64 reference, `example_textgen.py` three ways, the CUDA two-tree build). The same two files as the saved patch, `git status --porcelain` empty after the commit. `jargon.py --port Quest`: clean (after `git fetch origin main:main` in the fresh clone, which the check needs); `prose.py` on the body: clean. `advance-head Quest 517c6a7...` done; every platform validated at `9cb8bee` now reads `revalidate`.
   - What the reviewer should know: the change is HIP-only (shown above for CUDA by preprocessing and by nvcc object identity). gfx942/gfx90a device code is instruction-identical. gfx950 is checked on hardware (suites, fp64 harness, end-to-end). The gfx11/gfx12 code changed only by the `num_frags_z` cap; that path is checked on this host only through the WMMA layout emulation and by spill counts. The emulation was re-run here at `517c6a7` (`harness/checkall.sh <fork> e11|e12 -include harness/wmma_emu.h -Xarch_device -DQUEST_PREFILL_MMA=11|12`; devices report `wave=32 op_elems=16` and `op_elems=8`): **100/100 for each layout**, which covers capped instantiations (gfx12 head 128 nfx 2 in CFG 1, 2, 4, 5, 7, 9; gfx11 head 128 nfx 1 in CFG 0 and 8), so the Radeon validators' runs are its first hardware evidence, and the capped cases other than gfx1201 head 128 nfx 2 have no timing yet.
9. **Radeon compile time of `batch_prefill.cu`, gfx1201** (`agent_space/quest/bp/bp.sh <tree> gfx1201 <out.o>`: the extension's exact HIP compile line with the target swapped to gfx1201, `-o` fixed as described in item 6; own temporary directory each; trees `<job>/nvcc/base` = `9cb8bee`, `<job>/nvcc/v2` = the change, byte-identical to `517c6a7`). Two pairs, each pair before and after started together so both see the same load; the host was nearly idle (load average 8-16 on 256 cores; free disk 683 GB):

| run | `9cb8bee` | change | change / before |
|---|---|---|---|
| pair 1 | 1485.3 s, rc 0 | 789.1 s, rc 0 | 0.53 |
| pair 2 | 1476.3 s, rc 0 | 786.0 s, rc 0 | 0.53 |

   **The cap almost halves the gfx1201 build of `batch_prefill.cu` (about 24.7 -> 13.1 minutes, one translation unit for one target).** Most of the old build's time was the device link (`ld.lld` doing the device code generation was still running alone 10 minutes after the change's build had finished), consistent with the head-128 nfx-2 nfz-4 instantiation's 512 B/lane spill: the change compiles it as the nfz-2 body. Objects: 4,333,696 / 4,333,704 B before, 4,296,448 / 4,296,464 B after. gfx11 was not measured: there the most expensive instantiation (head 128 nfx 2, Q in shared memory) is deliberately not capped, so no comparable gain is expected.

### State at the end of this resumed round

Fork `moat-port` at `517c6a7`, `head_sha` advanced, project set to `ported` for the reviewer (lock released). Scratch deleted as each result was recorded (nvcc build trees, same-path object copies, assembly files, compile-time objects); the job directory keeps only the two small nvcc source trees and logs. Free disk at the end: 677 GB.

### Owed next (Quest at `517c6a7`)

1. Reviewer: review the delta `9cb8bee..517c6a7` (two HIP-only headers).
2. Validators: every platform validated at `9cb8bee` now reads `revalidate`. linux-gfx950 needs a fresh validator run at `517c6a7` (this porter round's evidence is above, but the validator role records it); linux-gfx942 and linux-gfx90a have instruction-identical kernels (item 6), but the device constant the host reads gained two fields (and the host code of the dispatcher changed under `USE_HIP`), so this is not a binary-identical delta; whether it qualifies for a carry-forward is for `carry-forward`'s own classification and the reviewer, otherwise they revalidate; linux-gfx1201, linux-gfx1100 and windows-gfx1151 run the capped wave32 code for the first time on hardware and owe full runs with `prefill_time.py`, which will also give the first timing of the spill-chosen capped cases.
3. jeffdaily: the operator host-path decision, now on current numbers (item 7): through the operator gfx950 is 3.22x of SDPA at 4096^2, 2.09x at 8192^2, 1.78x at 16384^2; the kernel alone is 1.97x / 1.66x / 1.62x. The host path leaves the GPU idle 0.08-0.13 ms per call; removing it would bring 4096^2 to roughly 2.3-2.8x, not within 2x.
4. Deferral `quest-prefill-matrix-cores`: not closed (a person rules).

## Review 2026-11-24 (linux-gfx942, delta `9cb8bee..517c6a7`) -- PASS

Host: AMD Instinct MI300X (`gfx942:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=0` (pinned by the job), PyTorch 2.16.0a0+git6c3af02 / HIP 7.16.26385 (pip ROCm SDK clang), GPU runs under `ulimit -c 0`. Fresh fork clone at `517c6a7` == `origin/moat-port` == `head_sha`. Reviewed with the pr-review skill. Every point below was checked in this session, not taken from the porter's notes. No findings.

Points jeffdaily asked to have checked independently:

- **CUDA branch untouched.** `prefill.cuh` with `#include` lines removed, through `cpp -P -undef -nostdinc`, at `9cb8bee` and `517c6a7`: identical with no defines, with `-D__CUDACC__ -D__CUDA_ARCH__=900` and with `=800`, md5 `02d6a90c0d2eea0259c7a25b34a3f580` (970 lines, the digest of every earlier record). With `-DUSE_HIP` the two differ, as intended. The delta touches only `prefill.cuh` and `prefill_matrix_core.cuh`, and the latter is included only under `#if defined(USE_HIP)` (`prefill.cuh:45-47`, the only include in the tree). The new kernel-body branch `#if QUEST_PREFILL_MMA == 2` evaluates to 0 under nvcc (the macro is undefined) and falls through to the unchanged `#else`. The porter's real-nvcc build of both trees (7 objects byte-identical once nvcc's per-process names are masked) was not repeated: this host has no CUDA toolkit.
- **gfx942 and gfx90a keep their code path: confirmed.** `batch_prefill.cu` compiled to device assembly with the extension's exact HIP compile line (taken from this build's `ninja -t commands`, `-c` replaced by `-S --cuda-device-only`, one target), with the `9cb8bee` and `517c6a7` headers at paths of the same length (`agent_space/quest/asm/asm.sh A|B <arch>`, 7.2 min gfx942, 11.7 min gfx90a). Both targets have 1536 kernels and 165,888 `v_mfma` lines in each tree. After the tree path is normalised, the diff is 14 lines per target and contains no instruction: (a) `matrix_core::prefill_mma_shape` grows from 8 to 16 bytes with `.long 16` (`k`) and `.long 0` (`static_kv`); (b) the per-translation-unit `__hip_cuid_<hash>` symbol (5 places). No kernel code, register count, LDS size or kernel metadata changed. The host side under `USE_HIP` was checked by reading it. The dynamic shared memory size is `(q_in_registers ? 0 : nfx * num_warps) + kv_frags(...) * 2` fragments; with `static_kv` 0 and wave 64, `kv_frags` returns `num_frags_z`, so the size equals the old value in both branches. `split_qo_indptr` takes `max_num_frags_x` = 2 for `k` 16, which is the old constant, and on a failed symbol read it falls back to `{}`, which also gives 2. The added symbol read shares the existing per-device cache in `get_prefill_mma_shape`. **For gfx942 and gfx90a the change is behavior-preserving**: the same instructions run with the same launch configuration. It is not binary-identical: the code object's constant data changed, and the host now reads 16 bytes instead of 8. A carry-forward would therefore be a judgment, not an identity. jeffdaily asked for a real run on this platform, and the validator below does one.
- **wave32 cap only on WMMA targets: confirmed.** `kv_frags` (`prefill_matrix_core.cuh:112-121`) caps when `static_kv` is set, which only the gfx950 selector value 2 does (`kShape` and the `{64, 8, 32, 1}` constant). It also caps when `wave_size == 32`, which only selector values 11 and 12 (gfx11 and gfx12 WMMA) report, when the query tile is in registers and the accumulators plus query operands exceed 128 registers. Wave64 MFMA (`{64, 4, 16, 0}`) and the emulated targets (`{0, 0, 0, 0}`, which also fails `q_in_registers`) return `num_frags_z` unchanged. For the shapes the dispatcher launches (`max_num_frags_z_reg` 4, since rotary is rejected on entry, and 64 KB LDS allows 4), the formula was evaluated by hand. It caps exactly four instantiations: gfx12 head 128 nfx 2 (256 > 128), gfx12 head 64 nfx 2 (160), gfx11 head 64 nfx 2 (224) and gfx11 head 128 nfx 1 (160). It does not cap gfx11 head 128 nfx 2 (Q in LDS, so `q_in_registers` is false), gfx12 head 128 nfx 1 (exactly 128) or head 64 nfx 1 on either. That matches the porter's spill table. The device (`kShape`) and the host (the constant read through `get_prefill_mma_shape`) evaluate the same function on the same values, so the kernel's tile and the dynamic shared memory size agree. Page sizes 1/8/16/32 all divide the capped 32-row tile, so the shared `page_iter_base += 16 * num_frags_z / page_size` step stays exact.
- **Each capped case is still correct, as far as an Instinct GPU can show.** The fp64 reference harness (`notes-prefill-harness.md`, built against the `517c6a7` tree with `ARCH=gfx942`) gave 100/100 natively (MFMA), 100/100 with the RDNA3 emulation and 100/100 with the RDNA4 emulation (`-include wmma_emu.h -Xarch_device -DQUEST_PREFILL_MMA=11|12`; the emulation header is the 2026-11-20 porter's, from that job's `agent_space`). The emulated binaries report `wave=32 op_elems=16` and `op_elems=8`, so the host applied the wave32 rule. Configurations 0-9 covered the capped gfx12 head 128 nfx 2 (CFG 1, 2, 4, 5, 7, 9) and gfx11 head 128 nfx 1 (CFG 0, 8) cases. **Head 64 with nfx 2, capped on both gfx11 and gfx12, was reached by no configuration and no pytest case.** `test_prefill_attention.py` uses head 128 only, and the harness had head 64 only at nfx 1. So this review added CFG 10 (fp16, head 64, nfx 2, causal, page 16) and CFG 11 (bf16, head 64, GQA 4, nfx 2, non-causal, page 8) to the recorded harness. Both give 10/10 natively, 10/10 with the RDNA3 emulation and 10/10 with the RDNA4 emulation, so 36 configurations x 10 shapes = 360/360. Negative controls: `-DNEGCTRL=true` on CFG 0, 1 and 10 gives 1/10 each (the 1 x 17 shape, where both causal rules agree). The emulation with the wrong gfx11 accumulator row (`wmma_emu_bad.h`, `QUEST_PREFILL_MMA=11`) gives 0/10 on CFG 10 and on CFG 11, so the emulation detects a wrong layout at the new configurations. An emulation checks index logic only, not the hardware's register allocation or spill code; that is what the Radeon runs below are for.
- **What the Radeon validators must run to cover every capped case and the uncapped one** (each at `517c6a7` on real hardware, besides the seven suites, end-to-end and `prefill_time.py`):
  - gfx1201 (gfx12 WMMA): harness CFG 1, 2, 4, 5, 7, 9 (capped head 128 nfx 2), CFG 10 and 11 (capped head 64 nfx 2), and CFG 0, 3, 6, 8 (uncapped nfx 1). Pytest's `qo_len` >= 77 cases launch the capped head-128 nfx-2 kernel through the operator, and `qo_len` 13/24/51 launch the uncapped head-128 nfx-1 kernel. Time CFG 100 (head 128 nfx 2 capped, against the 2.44 ms uncapped and 2.07 ms hand-capped records at 4096^2).
  - gfx1100 and windows-gfx1151 (gfx11 WMMA): harness CFG 0 and 8 (capped head 128 nfx 1), CFG 10 and 11 (capped head 64 nfx 2), and CFG 1, 2, 4, 5, 7, 9 (**deliberately uncapped head 128 nfx 2, Q in LDS**). Also CFG 3 and 6 (uncapped head 64 nfx 1). Pytest's `qo_len` 13/24/51 cases go through the capped head-128 nfx-1 kernel and `qo_len` >= 77 through the uncapped nfx-2 one. `prefill_time.py`'s 1 x 4096 row is the capped nfx-1 kernel through the operator, and its other rows are the uncapped nfx-2 one. Compare both with the `9cb8bee` gfx1100/gfx1151 timings: the capped nfx-1 kernel has no timing yet, and the nfx-2 one should not move.
  - Head-64 timing of the capped nfx-2 case is not covered by any script (CFG 100 is head 128). Quest's models are head 128, so correctness by CFG 10 and 11 is what matters there.
- **Commit message against the notes: the claims match.** 0.99 ms vs 0.21 ms SDPA at 4096^2 before (porter's table 0.990/0.991; SDPA 0.210-0.213); 0.41 ms after, 3.22 -> 1.19 ms and 10.2 -> 4.32 ms (table: 3.218 -> 1.188, 10.240 -> 4.322). The gfx1201 512 -> 176 B/lane spill matches the spill table. The R9700 2.44 -> 2.07 ms measurement (validation 2026-11-23, linux-gfx1201) was taken with `max_num_frags_z_reg` = 2, which instantiates nfz 2 and sizes shared memory for it. The commit's cap runs the nfz-2 body inside the nfz-4 instantiation with the same shared memory size, so "the same cap applied by hand" is accurate. The other capped cases are described as chosen by spill, which is true. "gfx90a and gfx942 keep their code ... apart from two new fields in the constant" is confirmed above. The title is 53 characters (the porter's notes say 54) and carries `[ROCm]`; the body has the AI-agent sentence, a literal fenced Test Plan, ASCII only, and no trailer. `jargon.py --port Quest` and `prose.py` on the message are clean.

Also read for defects, none found. The gfx950 double buffer: the barrier at the top of each step waits `vmcnt(0) lgkmcnt(0)` (`0x70` in the gfx9 encoding) before `s_barrier`, which covers read-after-write on the current buffer and write-after-read on the buffer the next copy fills. The prologue's `zero_v_tile` stores are ordered before the first copy by `block.sync()`. Rows at or past `kv_len` are never copied. Their scores are replaced (a select, not an add) with -5e4 by `mask_s`, which runs for every tile from `mask_iteration` on, and that covers the last partial tile in both the causal and non-causal formulas. Their stale or zero V rows therefore meet a probability of exactly 0. The page index uses `readfirstlane` only in the `full`-tile branch, where every lane of an instruction covers rows of one page (`page_size % rows_per_inst == 0`, and `first_chunk` is a multiple of 64). `kv_tile_t::num_chunks` divides exactly for every reachable shape (num_warps 4; head 64/128; effective nfz 2), and head 256 is not dispatched for prefill (`SWITCH_HEAD_DIM_PREFILL`).

## Validation 2026-11-24 (linux-gfx942, at `517c6a7`) -- PASS

Host: AMD Instinct MI300X (`gfx942:sramecc+:xnack-`), `HIP_VISIBLE_DEVICES=0` (pinned by the job; 0 % GFX activity before the runs), PyTorch 2.16.0a0+git6c3af02 / HIP 7.16.26385 (pip ROCm SDK), Python 3.11 (`/opt/python-3.11-venv`), all GPU runs under `ulimit -c 0`, builds within `MAX_JOBS`. Fork clone at `517c6a7` == `origin/moat-port` == `head_sha` == `reviewed_sha`, submodules initialised, `git status --porcelain` empty. This is a full real run, not a carry-forward (requested by jeffdaily), with the same method and counts as the `9cb8bee` validation on this platform. The review above found the delta instruction-identical for gfx942.

- **Build**: `agent_space/quest/build_readme.sh` (the README Linux block verbatim with `-DCMAKE_HIP_ARCHITECTURES=gfx942`, after `rm -rf build`, `ninja -j $MAX_JOBS`), wrapped `utils/timeit.sh Quest compile`: rc 0, 8/8, 573 s (four device-assembly compiles for the review ran beside it), 0 errors, 18 warning lines (pre-existing `nodiscard hipError_t` host code), `CMAKE_BUILD_TYPE=Release` from the default.
- **Tests**: `env PYTHONPATH=<fork>:<roles5 job>/pylib-test python -m pytest <fork>/quest/tests -p no:cacheprovider -v -rs` (the 2026-11-20 job's `pylib-test`, read in place), wrapped `timeit.sh Quest test`: **202 passed, 28 skipped, 0 failed** (33.6 s). rope 64, estimate 9, decode_attention 6, approx_attention 42, decode_launch_shape 3, topk 49 + 15 skipped (k >= kv_len), prefill_attention 29 + 13 skipped (qo_len > kv_len). Same as at `9cb8bee`.
- **fp64 reference harness** (`notes-prefill-harness.md`, built against this tree with `ARCH=gfx942`, run in the review above at the same commit): CFG 0-9 **100/100** native MFMA (device reports `wave=64 op_elems=4`); negative control (`-DNEGCTRL=true`) CFG 0 and 1: 1/10 each (the 1 x 17 shape, where both causal rules agree). The new CFG 10 and 11 were also 10/10 natively, and the CFG 10 negative control gave 1/10.
- **End-to-end**: `NousResearch/Llama-2-7b-chat-hf` at `351844e7` (read in place from the 2026-11-20 job's `models`), `do_sample` false, `agent_space/quest/e2e/example_textgen.py` (upstream's script with only `MODEL_PATH` changed), `PYTHONPATH=<fork>:<roles5 job>/pylib-e2e` (transformers 4.45.2): `--method hg`, `--method quest --token_budget 1024`, `--method quest --token_budget 128`, each wrapped `timeit.sh Quest test`, all rc 0, input length 103. Output with Quest's two banner lines removed: md5 `b900af27eeb4170a5c47bc5940a2ca60` for all three. That is also the digest of the `9cb8bee` outputs kept by the 2026-11-20 job on this host under the same filter ("Reason: The rabbit is the only one who is not working hard. ... can also mean \"lazy.\""). **Byte-identical.**
- **Prefill time through the operator against SDPA** (`agent_space/quest/prefill_time.py`, the 2026-11-20 script: causal, 32 heads x 128, fp16, page size 16; `is_causal` where qo = kv, `causal_lower_right` where qo < kv; mean of 20, 5 for the large shapes). Two runs, wrapped `timeit.sh Quest test`, first run shown (the second is within 1 %):

| qo x kv | Quest prefill_forward | torch SDPA | ratio | max abs diff vs SDPA | `9cb8bee` record (2026-11-20) |
|---|---|---|---|---|---|
| 1 x 4096 | 0.497 ms | 0.177 ms | 2.82 | 6.1e-05 | 0.460 ms |
| 128 x 4096 | 0.621 ms | 0.200 ms | 3.10 | 1.2e-04 | 0.582 ms |
| 1024 x 1024 | 0.254 ms | 0.079 ms | 3.22 | 2.0e-03 | 0.215 ms |
| 1024 x 4096 | 0.633 ms | 0.265 ms | 2.38 | 2.4e-04 | 0.593 ms |
| 4096 x 4096 | 1.183 ms | 0.626 ms | 1.89 | 2.0e-03 | 1.135 ms |
| 8192 x 8192 | 3.156 ms | 2.068 ms | 1.53 | 2.0e-03 | 3.116 ms |
| 16384 x 16384 | 10.819 ms | 8.002 ms | 1.35 | 2.0e-03 | 10.656 ms |

  Every shape is about 0.04 ms per call slower than the 2026-11-20 record, although the device code is identical. To tell host conditions apart from the change, `9cb8bee` was built here with the same README command (`agent_space/quest/base`, rc 0, 8/8). The two extensions were then timed interleaved, base / new three times each (`ab_{base,new}_{1,2,3}.out`; range over the three runs): 1 x 4096 0.494-0.498 / 0.494-0.500; 128 x 4096 0.616-0.620 / 0.616-0.624; 1024^2 0.247-0.253 / 0.249-0.253; 1024 x 4096 0.626-0.631 / 0.627-0.632; 4096^2 1.178-1.186 / 1.179-1.185; 8192^2 3.137-3.168 / 3.175-3.195; 16384^2 10.809-10.856 / 10.821-10.845 ms. **No difference between the two heads beyond noise**: the offset to the old record is host-side variation today, not the change. Kernel alone (recorded harness CFG 100, nfx 2 forced, best of 3 interleaved processes, `9cb8bee` / `517c6a7`): 128 x 4096 0.506 / 0.510, 1024^2 0.148 / 0.148, 1024 x 4096 0.517 / 0.517, 4096^2 1.063 / 1.067, 8192^2 3.005 / 3.004 ms. The plan's target holds: 4096 x 4096 is within 2x of SDPA (1.89x through the operator, 1.07 ms kernel).
- **CUDA gate**: no CUDA toolkit on this host. It is recorded at `517c6a7` by the porter round on the gfx950 host (real nvcc 12.8, sm_80, both trees 8/8, objects byte-identical with per-process names masked). This review's preprocessing check also gives the unchanged CUDA view (md5 `02d6a90c...`).
- **Content**: `jargon.py --port Quest` clean; the ROCm build is documented in the README (Linux and Windows blocks); `audit-commits` reports only the known `6e7d8db` gap; fork tree clean.

### Owed next (Quest at `517c6a7`)

1. linux-gfx1201, linux-gfx1100, windows-gfx1151: first hardware runs of the capped wave32 code. Run the seven suites, end-to-end and `prefill_time.py`, plus the fp64 harness configurations listed per target in the review above (CFG 10 and 11 included: the capped head-64 nfx-2 instantiation has no hardware evidence on any target).
2. linux-gfx950: a validator run at `517c6a7` (the porter round's evidence is not a validator record).
3. linux-gfx90a: instruction-identical device code (review above); a real run or a carry-forward with `--host-checked` per the review's host-code reading, a person's or the next validator's call.
4. jeffdaily: the gfx950 operator host-path decision (porter round 2026-11-24 resumed, item 7) is still open.
