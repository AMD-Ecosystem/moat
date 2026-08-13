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
