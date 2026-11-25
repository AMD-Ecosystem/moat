# warp-1.15 notes

## Intake (2026-11-24)

Release line of NVIDIA/warp for upstream's `release-1.15`, depending on `warp-1.14`. Jeff Daily ruled on 2026-11-24 (warp-1.14 notes, morning decisions, decision 5 (a)): start the 1.15 planner now; start the 1.15 porter once warp-1.14 passes review and Linux revalidation at its final Linux head, without waiting for the 1.14 release. The take-up and base rulings of the development-line project (`git show origin/port/warp:projects/warp/notes.md`) bind this line as they bound 1.14.

Recorded (`--by jeffdaily`, from that ruling and the 1.14 pattern): `set-fork-url https://github.com/AMD-Ecosystem/warp`, `set-fork-branch release/1.15.0.amd0`, `set-delivery fork-release --tag-format v{version}+rocm10.2.{rev}` (tag `v1.15.0+rocm10.2.0`, as 1.14's `v1.14.0+rocm10.2.0`), `set-license-spdx Apache-2.0` (evidence LICENSE.md on `release-1.15`). `mirror-branch` created `release-1.15` and `release/1.15.0.amd0` in the fork, both at upstream's `ffa038b1adf3`.

For the planner:
- The port branch is currently an exact copy of upstream `release-1.15`. The 1.14 port lives on `release/1.14.0.amd0` (final Linux head `1c94e68b` at this writing; Windows round and a gfx1250 rerun still ahead). Plan how the 1.14 port is carried onto 1.15 (merging `release/1.14.0.amd0` into the 1.15 branch keeps the 1.14 history and lets later 1.14 commits arrive as further merges; say which, and why).
- Not yet done here: `utils/licenses.py scan-nvidia warp-1.15` (needs a fork clone; 1.14's found only licence notices) and the Git LFS hook handling from 1.14's intake notes (move the LFS `pre-push` aside, then `protect-fork warp-1.15`).
- Coordinator's upstream 1.14 -> 1.15 survey (what changed upstream that will need porter work) is in the coordinator's scratch and summarized in warp-1.14's discussion; the planner should do its own survey of `release-1.14..release-1.15` (native `.cu`/`.h` changes, new NVRTC/CUDA-only features, new libmathdx uses, build and test changes) and account for each in `surface.json`.
- Lessons from 1.14 for this line: run every added or changed test on a real NVIDIA GPU with libmathdx (the Slurm V100 nodes), and every fix round on a 64 KiB shared-memory device and a wave32 device, not only gfx950.

## Planning (2026-11-24, linux-gfx942 MI300X host, planner)

Plan written to `plan.md` from `release-1.15` at `ffa038b1adf3` and the warp-1.14 port at `1c94e68b`; no build or GPU run (planning only). Lock taken with `set-state warp-1.15 linux-gfx942 planning`. Stage left at `planning`: six questions for jeff are in the plan (Q1-Q6), and Q1 (how the 1.14 port is carried: two merges, `-s ours release-1.14` then `release/1.14.0.amd0`) blocks the porter's first step. The porter does not start before warp-1.14 passes Linux revalidation at its final head (jeff's ruling), whatever Q1's answer.

Owed items from intake, done here:
- `utils/licenses.py scan-nvidia warp-1.15` on the fork clone `projects/warp-1.15/src`: 4 hits, the same licence notices as 1.14 (`README.md`, `licenses/cuda-LICENSE.txt`, `licenses/libmathdx-LICENSE.txt`, `licenses/nanovdb-LICENSE.txt`), no proprietary source.
- Git LFS hook: this host has no `git-lfs`, so the clone had no LFS `pre-push` hook; `python3 utils/moatlib.py protect-fork warp-1.15` installed the fork guard ("fork pre-push hook installed"). Hosts with `git-lfs` follow 1.14's recipe (move LFS's hook aside first). 1.15 adds no LFS-tracked file.

Trial merge (scratch clone `agent_space/trial`, push URL invalid, never pushed): `git merge --no-commit origin/release/1.14.0.amd0` into `release/1.15.0.amd0` -> 45 conflicting files; merging pristine `release-1.14` alone -> 32 (all upstream-vs-upstream); `git merge -s ours origin/release-1.14` (tree identical to `release-1.15`, `git diff --stat` empty) then the port -> 20 conflicting files, each listed with its nature in the plan.

`surface.json`: `python3 utils/surface.py generate warp-1.15` (43 CUDA files, 19 components; the floor now includes 1.15's new top-level CMake build) extended to 79 components (1.14's ids kept, plus deterministic mode, clusters, lean launches, mipmaps, managed memory, the radix-sort bit range, the CMake HIP question, the release-tag workflows, the NVIDIA-libmathdx and 64 KiB test duties); 77 owed, 2 scoped out by binding 1.14 rulings (conditional graph nodes, Clang-CUDA JIT), the two cuBQL CMake targets removed from the floor with a reason; `surface.py check warp-1.15`: fully accounted for.

Survey method: the 1.14..1.15 diff was read in four parts (host runtime, device headers, AOT kernels, build/packaging/tests); every finding the plan relies on was re-read in source, and HIP API facts were checked against the ROCm 10.1 SDK headers on this host (to be re-checked on 10.2 by the porter).

## Rulings

### Ruling on plan Q1, how the 1.14 port is carried (jeff, 2026-11-24)

(a), two merges. First `git merge -s ours release-1.14` on `release/1.15.0.amd0`, then `git merge release/1.14.0.amd0`; later 1.14 commits arrive as further merges. Recorded in `plan.md` (Q1 marked ruled; the carry section and Phase 0 state the two merges as decided). Q2 to Q6 remain open; they block only the phases the plan names (Q2 Phase 1's cluster work, Q3 Phase 3's mipmap part, Q4 Phase 6, Q5 Phase 2's sort work) and the release (Q6), not the start of porting, so the stage moves to `planned`. The porter still waits for warp-1.14 to pass Linux revalidation at its final head.

## Ruling on plan Q2, thread block clusters (jeff, 2026-11-24)

(a): clusters are unsupported on HIP, decided by device capability, the way upstream treats pre-sm_90 NVIDIA GPUs. `cluster_dim` is reported ignored with upstream's warning, `get_cuda_max_cluster_dim()` returns 1, the native cluster calls are no-ops, and the launch uses `cluster_dim = 1`. `test_cluster_dim.py` selects devices by capability, not by `arch`, so its "ignored" cases run on HIP. Register a deferred item for gfx1250 cluster support.

## Ruling on plan Q3, mipmapped textures (jeff, 2026-11-24)

(a): hardware mipmapped arrays on Linux devices with image support; the GPU software sampler with mip levels on devices without it (gfx942, gfx950, and gfx90a 3D); on Windows, try `hipMipmappedArray` on gfx1151 and route to the software sampler if it fails. The six mipmap tests run on every AMD GPU against Warp's CPU result.

## Ruling on plan Q4, the optional CMake build (jeff, 2026-11-24)

(b): no HIP branch in the CMake build. On a host with ROCm and no CUDA Toolkit, CMake stops at configure with a message that ROCm builds use `build_lib.py`; the install docs say the same; a deferred item records CMake HIP support. Evidence checked before the ruling: through upstream `main` (changelog to 1.19.0, release branches to release-1.20), `build_lib.py` is still the release build and is not deprecated; the installation docs call CMake "the alternate CMake build path" and say to "Use `build_lib.py` for release builds that need Warp's full GPU architecture coverage". Revisit in a later line if upstream makes CMake the release build.
## Porting session 1 (2026-11-25, linux-gfx942 MI300X, porter): Phase 0 carry, first build and suite

Requested by jeff Daily. Lock taken with `set-state warp-1.15 linux-gfx942 porting` (from `planned`). Prerequisite checked: warp-1.14 is `review-passed` at `66ba4ffb`, validated there on every platform (gfx942, gfx950, gfx1100, gfx90a, gfx1201, gfx1151).

### Environment
- Job venv (Python 3.11.17) with `rocm[devel,device-gfx942]==10.2.0`, numpy 1.26.4, usd-core 26.11 from `https://stable.repo.amd.com/rocm/whl-next/`, then `rocm-sdk init` (`rocm-sdk version` 10.2.0, HIP 7.17). `clang-format==22.1.2` and `ruff==0.15.9` (the pre-commit pins) in the venv. `HIP_VISIBLE_DEVICES=1` (dispatcher pin), `ulimit -c 0` for every GPU run, private kernel caches under `agent_space/`.
- Fork clone `projects/warp-1.15/src` (`GIT_LFS_SKIP_SMUDGE=1 git clone`, then `git lfs install --local --skip-repo` and `git lfs pull`); this host has git-lfs, so LFS's `pre-push` was moved to `pre-push.lfs-original` and `protect-fork warp-1.15` installed the fork guard. Remote `upstream` = NVIDIA/warp; local `release-1.14` = `3601b2d0`, `release-1.15` = `ffa038b1`, `release/1.14.0.amd0` = `66ba4ffb`.

### Phase 0 carry (Q1 ruling (a))
1. `5bc97fee` "Record release-1.14 as superseded by release-1.15": `git merge -s ours --signoff release-1.14` on `release/1.15.0.amd0` (`ffa038b1`). `git diff release-1.15 HEAD | wc -c` = 0.
2. `e0a07e7c` "[ROCm] Merge the 1.14 HIP port into 1.15": `git merge --no-commit release/1.14.0.amd0` (1.14 head `66ba4ffb`), 20 conflicting files, the planner's list exactly. No 1.14 commit rewritten or squashed; both merges signed off, no trailer, "Authored with Claude." in the body. Resolutions (also listed in the merge commit body):

| File | Resolution |
|---|---|
| `VERSION.md`, `warp/config.py`, `warp/native/version.h` | `1.15.0+rocm10.2.0` |
| `warp/_src/build_dll.py` | port's HIP option block kept, 1.15's removal of the fastcall `Python.h` check and `fastcall.cpp` flags kept; Windows `cpp_cmd` uses the port's `host_compiler` (clang-cl under ROCm) in place of `args.host_compiler` |
| `warp/_src/build.py` | 1.15's `err =` on the Clang-CUDA compile, then the port's hipRTC branch (1.15 checks `err` after the if/else chain; the port's own HIP check stays) |
| `warp/_src/context.py`, `warp/native/warp.h`, `warp/native/warp.cpp` | both sides (1.15's managed-memory and memory-kind exports, the port's texture-hardware query) |
| `warp/_src/texture.py` | `cuda_array`: the port's hardware check first, then 1.15's mipmap limitation |
| `warp/native/texture.h` | 9 hunks: 1.15's `lod` logic (`tex*D` when `lod < 0`, else `tex*DLod`) under `WP_TEXTURE_HW_SAMPLING[_3D]` with `WP_TEXTURE_OBJECT(tex.tex)` |
| `warp/native/texture.cpp` | 1.15's mip-level validation, then (HIP only) mipmapped textures refused with "Mipmapped textures are not supported on AMD GPUs" (Q3 is open; this is the interim), then the software dispatch; 1.15's `wp_texture_get_mip_level_array_device` and the port's `texture_memory_type` both kept; object create passes `mip_filter_mode` to the software path |
| `warp/native/texture_software.{h,cpp}` (no conflict, changed in the merge because the merged tree did not compile otherwise) | descriptor built with 1.15's unified `Texture` constructor as one level (`num_mip_levels = 1`), `data` and `mip_data[0]` = device texels, `mip_filter_mode` passed through |
| `warp/native/cuda_util.cpp` | 1.15's three `cuMipmappedArray*_f` wrappers stay in the CUDA section before the port's `#endif // !WP_ENABLE_HIP` |
| `warp/native/warp.cu` | hunks 1-3 both sides; hunk 4 (launch grid): 1.15's code, with the HIP 2^32-1 threads-per-dimension cap applied to the grid-stride `cap` (`cap = int(std::min(unsigned(cap), 0xFFFFFFFFu / unsigned(block_dim)))` under `WP_ENABLE_HIP`); the lean 3D path keeps `grid.x * block_dim <= 2^24` already |
| `warp/native/bvh.h` | both declarations |
| `warp/native/bvh.cu` | 1.15's empty-tree guard, then the HIP depth-level refit; the old cuBQL glue deleted as 1.15 did (its refit-stream change is obsolete: 1.15's `bvh_cubql.cu` builds and frees on the current stream and refits through `bvh_refit_device`) |
| `warp/native/builtin.h` | 1.15's bit-cast `atomic_cas`/`atomic_exch` bodies under `defined(__CUDA_ARCH__) \|\| defined(__HIP_DEVICE_COMPILE__)` |
| `warp/native/tile_reduce.h` | ours (1.15), with every `warp_shuffle_down(..., int mask)` overload (the new half/bf16 ones and the vec/mat/pointer/shape/array/indexedarray ones) taking `tile_mask_t` |
| `warp/native/tile_radix_sort.h` | key with `WP_TILE_SYNC_MASK`, value through 1.15's `warp_shuffle_xor` |
| `warp/tests/cuda/test_texture.py` | 1.15's two new `cuda_array` tests registered on `interop_cuda_devices` (hardware textures), the port's `interop3d_cuda_devices` kept |
| `warp/tests/test_graph.py` | 1.15's renamed `get_test_devices_with_graph_capture_allocation_and_cuda_graph_module_load` plus the port's two helpers; no other port caller of the old name exists |

### Fixes after the merge (each its own commit)
- `b8bf0fa4` [ROCm] Compile Warp 1.15's new device and runtime code for HIP: `hip_compat.h` maps `CUmipmappedArray`, `CU_RESOURCE_TYPE_MIPMAPPED_ARRAY`, `CUlaunchConfig` (`HIP_LAUNCH_CONFIG`), `CUlaunchAttribute`, `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`, `CUmemoryPool` (`hipMemPool_t`), the two managed-memory device attributes, `CU_POINTER_ATTRIBUTE_IS_MANAGED`/`MEMORY_TYPE`, `CUDA_ERROR_INVALID_VALUE`, `cudaMemAttachGlobal`, `cudaThreadExchangeStreamCaptureMode`; HIP wrappers `cuMipmappedArrayCreate/Destroy/GetLevel_f`, `cuFuncGetAttribute_f` (through `hip_driver_result`), `cuOccupancyMaxActiveClusters_f` returns `hipErrorNotSupported` without a HIP call (the plan's interim pending Q2; HIP's function attribute 14 is NON_PORTABLE_CLUSTER_SIZE_ALLOWED as on CUDA, checked in `driver_types.h`); `deterministic.h` HIP branch for `is_global_store_target` (`!__builtin_amdgcn_is_shared && !__builtin_amdgcn_is_private` on the generic pointer) and the five other gates widened; `tile_radix_sort.h` 12 `__shfl_xor_sync(0xFFFFFFFFu, ...)` -> `WP_TILE_SYNC_MASK`, `FloatKeyToUint` gate widened; `codegen.py` per-struct `warp_shuffle_down` takes `tile_mask_t` (it took `int`, so on HIP struct tile reductions returned lane 0's value: `test_tile_struct` 4 failures, sum 1 vs 36).
- `b4145943` [ROCm] Port the BSR in-place compression to HIP (`sparse.cu`): 64-bit ballot/`__popcll`/prefix in `bsr_compress_select_sorted_runs` keyed on `__HIP_PLATFORM_AMD__` (clang checks the device function in the host pass, where `WP_TILE_SYNC_MASK` is the 32-bit CUDA value, so a `__HIP_DEVICE_COMPILE__` key fails the HIP static_assert); `cub/device/device_select.cuh` shim; `BlockIterator::operator-(difference_type)` (rocPRIM's partition computes `keys_input + block_offset - 1`); the two raw launches refused with a message above 2^32-1 threads and checked with `cudaGetLastError()` on HIP; the dynamic shared array declared `aligned(16)` on HIP.
- `aa1726b3` [ROCm] Keep deterministic mode's guarantees on HIP: raw run-to-run reduction (query and reduce) through `rocprim::deterministic_reduce_by_key` on HIP; capture-time buffers allocated with the memory pool off on HIP (`ScopedMempool(device, False)` in `det_buffer_allocation_scope`, `nullcontext()` on CUDA). Root cause of the second, measured with `agent_space/repro/relaxed{,2}.hip`: while a **blocking** stream is captured, `hipMallocAsync` on any other stream (blocking or non-blocking, never captured) returns `hipErrorStreamCaptureUnsupported` (900) for every begin mode (global, thread-local, relaxed) and thread mode (exchanged to relaxed or not); with a non-blocking capturing stream it succeeds; `hipMalloc` succeeds in all cases. CUDA allows the stream-ordered allocation there. A ROCm defect to register and file (deferred item below).
- `b64c4641` [ROCm] Refit cuBQL trees with the bottom-up walk on HIP: plan B.5 confirmed: `bvh_cubql.cu` allocates `node_counts` uninitialized and sets `num_leaf_nodes` to the node count; the HIP depth-level refit then read garbage depths for trees of >= 32768 nodes. Chose the plan's smaller option (skip the depth path for `BVH_CONSTRUCTOR_CUBQL`; cuBQL meshes never take the winding-number refit, `mesh.cu:360`). New `test_bvh_cubql_constructor_large` (20000 boxes, leaf size 1, ~40k nodes): passes with the guard; with the guard removed and rebuilt it fails (`AssertionError: 1 != 0`).
- `16ed353b` [ROCm] test_tile_oob: HIP reports the device assert as "unspecified launch failure" (error 719 after the HSA exception); Warp's own OOB text is still asserted. Capability-keyed (`device.is_hip`), not a skip.
- `55174181` Skip the capture_while fill test without conditional graph nodes (`test_graph_fill_drives_capture_while`, new in 1.15, uses `capture_while`; skips when `not wp.is_conditional_graph_supported()`, the 1.14 conditional-node ruling).

### Build
`utils/timeit.sh warp-1.15 compile -- bash -c "cd projects/warp-1.15/src && python build_lib.py --rocm-path $(rocm-sdk path --root) --hip-arch gfx942 -j 32"` -> "Warp build succeeded" (from the merge commit, five build iterations to get there: CUmipmappedArray/CUlaunchConfig mappings, the radix-sort xor masks, the DeviceSelect shim, the BlockIterator operator, the sparse host-pass mask). `deterministic.cu` (with rocThrust's transform iterator beside hipCUB) and `bvh_cubql.cu` compiled under hipcc without changes. The strict undefined-symbol check after the link passed.

### First full suite (at `e0a07e7c` + the uncommitted compile fixes, before the runtime fixes)
`agent_space/suite.sh full1` = `python -m warp.tests -s autodetect` from the clone, fresh kernel cache: **Ran 8103 tests in 773.2 s, FAILED (failures=9, errors=14, skipped=115)**. Each failure, and what became of it:
- `test_tile_struct` 4 (struct shuffle mask) -> fixed in `b8bf0fa4`.
- deterministic graph capture 7 errors + `test_record_cmd_deterministic_launch` 1 failure ("Failed to allocate", HIP error 900) -> fixed in `aa1726b3`.
- `test_tile_oob` 1 -> `16ed353b`. `test_graph_fill_drives_capture_while` 1 -> `55174181`.
- mipmap textures 5 errors (`test_texture1d_mipmap`, `2d_mipmap_full_chain`, `2d_mipmap_lod_selects_constant_level`, `2d_mipmap_rejects_copy`, `3d_mipmap`): the interim refusal; **Q3, not implemented**.
- `test_cluster_dim.test_aot_cluster_dim_below_sm90_errors` 1 error ("Compiling for an AMD GPU requires the target device": AOT compile by NVIDIA arch number with no device): **Q2, not implemented**.
- `test_async` 2 failures (`copy_c2f_h2d_..._PoolOff_..._NoGraph`, `copy_s2s_d2d_..._PoolOff_..._Graph`): the known pool-off copy flake (ROCm/rocm-systems#16355); rerun below.
- `test_unique_module` "Verify @wp.func(module="unique") registers in directly executed scripts" 1 failure (`ModuleNotFoundError: No module named 'warp'` in its child script): passes alone (28 OK); recheck in the next full run.

## Lesson for cuda-to-rocm: hipMallocAsync is refused on every stream while a blocking stream is captured
Reference file: `references/faults-memory-streams.md` (graph capture). Symptom: `hipMallocAsync` on a stream that is not being captured returns `hipErrorStreamCaptureUnsupported` (900, "operation not permitted when stream is capturing") while another stream is capturing a graph. Cause: on HIP 7.17 (ROCm 10.2, MI300X), whenever the capturing stream is a blocking stream (created without `hipStreamNonBlocking`), stream-ordered allocation is refused on every other stream, blocking or non-blocking, for every capture mode (global, thread-local, relaxed) and with the thread's mode exchanged to relaxed; with a non-blocking capturing stream it succeeds, and plain `hipMalloc` succeeds in all cases. CUDA allows the stream-ordered allocation on an uncaptured stream there, which is what a "relaxed mode, allocate on a side stream during capture" pattern relies on. Fix: allocate such buffers with `hipMalloc` (in a pooled-allocator design, turn the pool off around that scope) on HIP. Reproducer (begin a thread-local capture on a default-flag stream, then `hipMallocAsync` on a second stream) in the warp-1.15 port's notes. (warp-1.15, 2026-11-25)

## Lesson for cuda-to-rocm: rocPRIM's reduce_by_key is not run-to-run reproducible for floating point; hipCUB's ReduceByKey inherits that
Reference file: `references/libraries.md` (Thrust/CUB). hipCUB's `DeviceReduce::ReduceByKey` calls `rocprim::reduce_by_key`, whose documentation says floating-point results may differ from run to run and points to `rocprim::deterministic_reduce_by_key` (same arguments as `reduce_by_key`). Code that relies on CUB's ReduceByKey being reproducible (a "deterministic mode", a bit-exact test) must call the deterministic variant on HIP. Measured on MI300X (ROCm 10.2): 2^22 float64 values over 16 orders of magnitude reduced into 64 keys, 11 runs: 3 to 5 distinct results with `ReduceByKey`, 1 with `deterministic_reduce_by_key`, at 2-4% more time for the whole scatter-sort-reduce. A test with a few thousand values runs in one block and cannot see the difference. (warp-1.15)

## Lesson for cuda-to-rocm: HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE is not supported
Reference file: `references/platform-limits.md`. `hipPointerGetAttribute(..., HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE, p)` returns `hipErrorNotSupported` (801) for a `hipMallocAsync` pointer and for a `hipMalloc` pointer alike (HIP 7.17, ROCm 10.2), and leaves that error as the last error. Code that classifies pointers as "pool" versus "plain device" memory through this attribute (as CUDA's `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` allows) sees every device pointer as plain device memory; track pool membership yourself where it matters, and clear the last error after the failed query. `HIP_POINTER_ATTRIBUTE_IS_MANAGED` and `MEMORY_TYPE` work (managed memory reports type 1, host, with IS_MANAGED 1). (warp-1.15)
