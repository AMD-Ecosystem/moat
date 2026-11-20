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
