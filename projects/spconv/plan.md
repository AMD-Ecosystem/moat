# spconv port plan

Planner: linux-gfx90a fleet job, 2026-11-26. Read-only analysis of upstream `263d6b474` (2.3.8, 2024-12-15, still the head), built on cumm's plan (`port/cumm` `1614d936`, `projects/cumm/plan.md` and `notes.md` there) and read against cumm `d971d02a0`, pccm `860790f` (PyPI 0.4.16) and ccimport `f5af43d` (PyPI 0.4.4). Nothing was built and the fork was not touched. Inputs: the intake screen in `notes.md`, jeff's sizing study (`~/.claude/plans/moat-cumm-hip-backend-study.md`, which built spconv's SIMT kernel set on gfx942 through a throwaway shim) and jeff's rulings of 2026-11-20 and 2026-11-23.

## Project

- Name: spconv. Upstream https://github.com/traveller59/spconv, default branch `master`. Fork https://github.com/AMD-Ecosystem/spconv, port branch `moat-port`.
- What it is: spatial sparse convolution for PyTorch (SubMConv/SparseConv/inverse conv/pooling in 1-4D, point-to-voxel, a GPU hash table, int8 quantized inference). Its CUDA is written as pccm meta-programs (`spconv/csrc/**`) and cumm GEMM/implicit-GEMM kernel descriptors (`spconv/core.py`), generated into C++/CUDA and compiled into one pybind module `spconv/core_cc` (prebuilt by `setup.py` with `SPCONV_DISABLE_JIT=1`, or JIT-built at import by `spconv/build.py` in an editable install). Descriptors not prebuilt are compiled at run time through cumm's NVRTC layer. The same code is also emitted as a plain C++ library, libspconv (`python -m spconv.gencode` plus CMake, `example/libspconv/`), which the separate MOAT project spconv_cpp consumes.
- Rulings that shape this plan: adopt (2026-11-20); proceed independently of `ZJLi2013/spconv_rocm` and never publish as `spconv-rocm`/`cumm-rocm`, use a per-ROCm-major name such as `spconv-rocm7` (2026-11-23); ccimport is consumed unmodified from PyPI (declined as already supported); scope "Option B": the port is complete only when the AMD matrix-instruction families (MFMA on CDNA, WMMA on RDNA) exist beside the plain kernels, no pull request, review page or wheels before then (2026-11-23).

## Existing AMD support

None upstream: no HIP/ROCm code, PR, branch or maintainer reply (issue #780 got only a third-party pointer). Other efforts, none authoritative, none a build of spconv as published:
- https://github.com/ZJLi2013/spconv_rocm (branch `rocm`): community FlyDSL rewrite that drops the pccm/cumm codegen, gfx942 inference, Point2Voxel stubbed. jeff ruled to proceed independently. Not inherited.
- https://github.com/HYGON-AI/spconv-das: Hygon DCU downstream of the above. Not relevant to AMD GPUs as a code base.
- https://github.com/L-Reichardt/spconv-triton: pure-Triton replacement (MI300X, no Windows). A replacement, not a port; it was the 2026-08-20 decline reason that jeff reversed.

Decision: port this code base on top of MOAT's cumm port. Delivery: upstream is dormant for outside contributions (last outside merge 2020, last commit 2024-12) but does not link platform forks, so per the standing rule the clean complete port is offered as a PR after approval; the fork's `moat-port` and `spconv-rocm<major>` wheels are the real deliverable.

Mechanical port versus AMD-native rewrite: spconv's own kernels (pair generation, hash, pooling, gather/scatter, point-to-voxel) are simple grid-stride loops and port mechanically. GEMM performance comes from cumm: NVIDIA tensor-op descriptors (Volta/Turing/Ampere) are not generated on HIP; spconv gains descriptor lists for cumm's new AMD-native MFMA and WMMA families (compiler builtins, cumm plan "AMD matrix-instruction families"). That is where this plan meets the Option B ruling.

## Build classification

pccm/ccimport code generation driven by setuptools, like cumm (`set-ext-type spconv pccm`). Evidence:
- `setup.py:166-235`: with `SPCONV_DISABLE_JIT=1` it builds `spconv/core_cc` from a list of pccm classes with `pccm.extension.PCCMExtension`/`PCCMBuild`; otherwise no extension at all.
- `spconv/build.py:23-76`: in an editable install, `pccm.builder.build_pybind(cus, PACKAGE_ROOT / "core_cc", ...)` builds the same module at import.
- `pyproject.toml`: build requires `pccm>=0.4.16` and `cumm>=0.7.11`; runtime requires `cumm-cuXXX>=0.7.11,<0.8.0` (`setup.py:37-42`). torch is a runtime-only dependency of `spconv.pytorch` (no `torch.utils.cpp_extension` anywhere), so this is not Strategy B.
- CUDA versus CPU-only is cumm's decision (`cumm.constants.CUMM_CPU_ONLY_BUILD`), read in `setup.py:186`, `spconv/csrc/sparse/all.py:88`, `inference.py:152`.
- Second build path: `spconv/gencode/__main__.py` calls `pccm.builder.pybind.gen_cmake`, which writes CMake with `find_package(CUDAToolkit)`, `add_library(<cls> OBJECT *.cu *.cc)` and nvcc flags under `$<COMPILE_LANGUAGE:CUDA>` (`pccm/core/__init__.py:2386-2490`); `example/libspconv/CMakeLists.txt` and `example/libspconv/spconv/CMakeLists.txt` declare `LANGUAGES CXX CUDA`.

## Port strategy

Strategy-A-shaped for a codegen build, exactly cumm's route, with spconv consuming cumm's HIP backend rather than re-deriving it. CUDA spelling stays in the generated sources; HIP-only code sits behind `__HIP_PLATFORM_AMD__` (C++) or cumm's HIP-build constant (Python); no build-time hipify. The CUDA path stays byte-identical in generated output.

1. **Build route: the same recipe as cumm, from cumm.** ccimport stays unmodified: `.cu` maps to the logical compiler `clang++` through `build_pybind(suffix_to_compiler={".cu": "clang++"})`, `clang++` is bound to `amdclang++` through `CCIMPORT_COMPILER_LINKER_MAP`, and HIP flags (`-x hip --offload-arch=<gfx> ...`) ride in cumm's `clang++` flag bucket of the classes spconv depends on (`TensorView`, `TensorViewKernel`, `GemmBasicHost`, `ThrustLib`). The cumm plan exposes this as one helper (for example `cumm.common.get_hip_build_kwargs()`) precisely so spconv uses the identical recipe. `PCCMExtension` cannot carry `suffix_to_compiler`, so on HIP `setup.py` uses a `build_ext` that calls `build_pybind` itself (reuse cumm's class if cumm exports one, otherwise a small spconv-owned copy), and `spconv/build.py` passes the same kwargs on the JIT path. On CUDA both files behave exactly as today.
2. **Kernel descriptor lists by backend.** `spconv/core.py` gains HIP lists: shuffle (Native) and implicit-GEMM (MaskImplicitGemm/MaskSplitImplicitGemm) descriptors for cumm's MFMA and WMMA families, fp16 in, fp32 accumulate, fp16/fp32 out, mirroring the coverage of the Turing f16 lists (fwd, dgrad, wgrad; access-per-vector 0 and 1; split-K; dynamic mask), plus SimtDP4A int8-inference forward descriptors (see "int8"). `ALL_NATIVE_PARAMS`/`ALL_IMPGEMM_PARAMS` become backend-selected aggregates: on HIP = SIMT + AMD + SimtDP4A int8; on CUDA = the four lists as today. `setup.py`, `spconv/build.py` and `spconv/gencode/__main__.py` stop concatenating the four NVIDIA lists by hand and use those aggregates (on CUDA the result is the same list in the same order, so the generated code is unchanged). The AMD tile sets come from the tuned per-family sets cumm exports.
3. **Algorithm selection by family, not compute capability.** AMD CC tuples collide with NVIDIA (gfx90a reports 9.0, gfx942 9.4, gfx11 11.x), and today every fp16 conv on gfx9 skips SIMT ("arch >= 7.0 has tensor cores") and selects Turing descriptors, the study's fp16 wall. The four filter sites get a HIP branch: `SimpleGemm.get_all_available` (`spconv/algo.py:304-364`), `SimpleConv.get_all_available` (`algo.py` conv counterpart), and the generated C++ `GemmTunerSimple::get_all_available` / `ConvTunerSimple::get_all_available` (`spconv/csrc/sparse/convops.py:537-595`, `1004-1128`, which is the default path, `SPCONV_CPP_GEMM = True`). On HIP: admit a descriptor when its family is available on the current device (cumm's gfx-keyed availability helper, in Python and C++), skip the Volta rule and the PTX logic, treat `is_nvrtc` as "compile with hipRTC", and skip SIMT fp16 only when at least one matrix-family descriptor survived the ld/mask checks for that problem (so an odd channel count still has a kernel). The `arch` tuple stays in every signature and cache key (API compatibility with the Python layer, libspconv and spconv_cpp). `get_available_algo_str_from_arch` (both copies) follows cumm's HIP availability helper.
4. **Library spellings.** rocThrust has no `thrust::cuda::par` (its `thrust/system/cuda` headers are CUDA-only; the HIP policy is `thrust::hip::par`, checked in the ROCm 10.1 SDK headers, `thrust/system/hip/detail/par.h:290,362-368`): branch the 7 live sites (`all.py:982,983,1049,1113,1338,1548`, `indices.py:1009`). `SimpleExternalSpconvMatmul` (`convops.py:78-372`, cuBLASLt, used by libspconv's example and the JIT build, not the release wheel) maps to hipBLASLt through cumm's compat header plus `hipblaslt` in `add_libraries` and the HIP include on HIP. `CustomThrustLib`'s nvcc-keyed `-Xcompiler -fno-gnu-unique -fvisibility=hidden` (`all.py:34-40`, a Thrust multi-module workaround) gets the clang equivalent (`-fvisibility=hidden`) in the `clang++` bucket on HIP.
5. **libspconv (gencode + CMake).** pccm is not ours to change and `gen_cmake` emits only CUDA-language CMake. On HIP, `spconv.gencode` additionally writes a small CMake fragment (HIP language on every generated `.cu` source via `set_source_files_properties(... TARGET_DIRECTORY <obj target> PROPERTIES LANGUAGE HIP)`, cumm's HIP compile options under `$<COMPILE_LANGUAGE:HIP>`, `hip::host` and the ROCm libraries), and `example/libspconv/CMakeLists.txt` / `spconv/CMakeLists.txt` select `LANGUAGES CXX HIP` and include it when built for ROCm, keeping the CUDA branch unchanged. This is the route spconv_cpp's plan should consume.
6. **Packaging.** `setup.py` on HIP: release name `spconv-rocm<major>` from cumm's ROCm-version variable, runtime dependency on the matching `cumm-rocm<major>` at the cumm port's version (no `<0.8.0` cap: cumm's 0.8.0 bump was a CI gcc/ABI change, "change version to 0.8.0 to avoid break spconv", `e6a95bd`, not an API change, and the cumm port is on 0.8.2+). The CUDA pins stay as they are. `pyproject.toml` stays unchanged; the README documents `--no-build-isolation` with the ported cumm installed (upstream already tells editable users to do the equivalent). A `tools/build-wheels-rocm.sh` script (fork Actions stay disabled).

### int8

Upstream's int8 quantized inference (`spconv.pytorch.quantization`, `int8_inference=True` descriptors) exists only as Turing/Ampere tensor-op descriptors, which HIP cannot use, and upstream's own GPU test `test/test_all_algo.py` runs only int8 (`dtypes = [(np.int8, np.int8)]`, `:336`) and asserts that descriptors exist (`assert avail_desps`, `:416`). Leaving int8 without kernels would fail upstream's test and cut a feature, so the plan keeps it: HIP descriptor lists of cumm's SimtDP4A family for the int8-inference forward conv (dtype strings as the Turing int8 lists: `s8,s8,s8,s32,f32`, `s8,s8,s8,s32,f16`, and the s8/f16/f32 output variants), prebuilt (not `is_nvrtc`). cumm keeps int8 on SIMT on HIP (its surface scopes out int8 matrix kernels pending jeff's answer to its question 1); cumm generates SimtDP4A conv kernels (`conv/algospec/all.py:28`) but its own SimtDP4A conv lists are commented out, so this combination is untested upstream and may need cumm-side fixes found here. If jeff extends cumm's AMD families to int8, spconv swaps these lists for AMD int8 descriptors (question in notes.md); either way the feature is covered.

tf32 descriptors (Ampere, opt-in through `SPCONV_ALLOW_TF32`, default off) fall back to SIMT f32 on HIP, which is what upstream does with tf32 disabled. bf16: upstream has no bf16 GEMM or conv descriptors on any backend (bf16 appears only in gather/pooling/inference dispatch, which the port covers); this plan adds none (question in notes.md, since the Option B ruling names bf16 for cumm's families).

## CUDA surface inventory

The regex census (`surface.json` `cuda`) sees one `.cu` file (`example/libspconv/main.cu`); all device code is generated from Python strings, so components are added by hand.

| area | where | what is CUDA-specific | HIP treatment |
|---|---|---|---|
| Build route | `setup.py:166-235`, `spconv/build.py:23-76`, `pyproject.toml` | `PCCMExtension`/`build_pybind` default nvcc route; `CUMM_CUDA_VERSION` naming and `c++14` for CUDA < 11 (`setup.py:204-216`, `all.py:135-145`) | cumm's HIP kwargs + `build_ext`; `c++17` on HIP; ROCm naming |
| Descriptor lists | `spconv/core.py` (`SHUFFLE_*_PARAMS:41-200`, `IMPLGEMM_*_PARAMS:205-1340`, aggregates `1341-1343`) | Volta/Turing/Ampere tensor-op descriptors (f16, tf32, int8) | not generated on HIP; AMD MFMA/WMMA f16 lists and SimtDP4A int8 lists added; SIMT f32/f16 lists shared |
| Selection | `spconv/algo.py:304-364` and conv twin, `convops.py:514-595`, `980-1128`; `spconv/pytorch/cppcore.py:102-109` (`get_arch` via `torch.cuda.get_device_capability`), `convops.py:1485-1500` (`get_compute_capability`) | CC-tuple `min_arch` comparisons, Volta skip, SIMT-f16 skip at CC >= 7.0, PTX compatibility, NVRTC fallback | family availability by gcnArchName on HIP (cumm helper), hipRTC for `is_nvrtc`; tuples kept for API |
| Runtime-compiled kernels | `algo.py:93-212,273-302` (`CummNVRTCModule`, `NVRTCMode.ConstantMemory`), `convops.py:650-712,1134-1208` (C++ `cached_get_nvrtc_params`), `constants.py:66` | NVRTC; `DynamicParallism` needs `libcudadevrt` | cumm's hipRTC path (ConstantMemory is spconv's default and is supported); DynamicParallism is never selected by spconv and cumm refuses it on HIP |
| Thrust | `all.py:34-40,953-1114,1336-1339,1547-1550`, `indices.py:1008-1011` | `thrust::cuda::par`, `.on(stream)`, custom allocator, `sort_by_key`, `stable_sort_by_key`, `unique`, tuple keys | `thrust::hip::par` (rocThrust), visibility flag in the `clang++` bucket |
| cuBLASLt | `convops.py:78-372` (`SimpleExternalSpconvMatmul`) | `cublasLt*` API, `cublasLt` link | hipBLASLt through cumm's compat header; `hipblaslt` link |
| Pair generation and masks | `indices.py` (21 kernels), `all.py` (`count_bits_kernel` `__popc`) | `atomicAggInc` (cumm), `atomicOr`/`atomicAdd` on 32-bit masks, `__popc` | no change (the 32 in mask math is the mask word width, not a lane count); relies on cumm's `warp_res` fix |
| Pooling, gather, inference, point-to-voxel | `maxpool.py` (7 kernels), `gather.py` (`tv::kernel_1d`), `inference.py` (3), `pointops.py` (5) | grid-stride loops, `tv::cuda::Launch`, max 512 threads, `half`/`bfloat16` dispatch | no change beyond cumm's runtime layer |
| Hash table | `spconv/csrc/hash/core.py` (`tv::hash::LinearHashTableSplit`, 64-bit keys) | 64-bit `atomicCAS` (cumm) | no change; cumm's study proved the path on gfx942 |
| Allocator | `alloc.py:384,419` (fill kernels), `pytorch/cppcore.py:112-175` (`TorchAllocator` zeros/empty) | | audit every `empty` buffer whose first access is a read-modify-write |
| libspconv | `spconv/gencode/__main__.py`, `example/libspconv/{CMakeLists.txt,spconv/CMakeLists.txt,run_build.sh,main.cu}` | CUDA-language CMake from pccm, `CUMM_CUDA_VERSION`/`CUMM_CUDA_ARCH_LIST`, `cudaStream_t`/`checkCudaErrors` in `main.cu` | HIP CMake fragment from gencode, HIP branch in the CMake files and script; `main.cu` compiles unchanged through cumm's compat header |
| Packaging | `setup.py:24-42`, `tools/build-wheels*.sh`, `tools/install_windows_cuda.ps1`, `.github/workflows/build.yaml`, README install/build sections | per-CUDA names and pins | `spconv-rocm<major>` + `cumm-rocm<major>`, ROCm wheel script, README ROCm section (Linux and Windows) |

Scoped out in `surface.json`, each with its reason: the NVIDIA tensor-op descriptor lists (replaced on HIP, unchanged on CUDA), the DynamicParallism NVRTC mode (never selected by spconv), `example/mnist/custom_fx2trt.py` (TensorRT, NVIDIA-only by purpose) and the assertion-free developer scripts under `test/debug`, `test/develop`, `test/apps`.

Python layer (`spconv/pytorch/**`, 26 `torch.cuda.synchronize`, `torch.cuda.current_stream`, `torch.cuda.amp`, `torch.cuda.device`) runs unchanged on ROCm PyTorch. Not CUDA work, must not regress: the CPU-only build and the CPU operators (`Point2VoxelCPU`, `SpconvIndicesCPU`, `GatherCPU`, `IndiceMaxPoolCPU`, `PointCloudCompress`, `BoxOps` with optional Boost).

## Risk list

- **Dependence on cumm's surface (highest).** Every GPU path needs cumm's HIP backend, its gfx-keyed availability helper in Python and C++, the AMD family descriptors, SimtDP4A int8 conv with the int8-inference epilogue, and hipRTC. The contract spconv needs is listed in notes.md ("Interface spconv needs from the cumm port") so the cumm porter can expose the names once; spconv builds against a recorded cumm port sha and notes it with every validation.
- **Selection logic silently picking nothing or the wrong family.** A wrong branch either returns an empty list (fp16 conv fails) or keeps SIMT everywhere (correct but slow, invisible to correctness tests). Tests assert the selected family for fp16 on each device class.
- **Wavefront width.** spconv's own kernels use no warp intrinsics; the 32-lane logic lives in cumm (`atomicAggInc`, kernel prologue) and the MFMA family is genuinely wave64. Validate on both widths.
- **wgrad tile pairing.** `ConvTunerSimple` requires `mask_width % tile_k == 0` for backward-weight descriptors, where `mask_width` is the forward descriptor's tile M. The AMD and SimtDP4A lists must offer wgrad tile K values that divide every forward tile M they pair with, or MaskSplitImplicitGemm backward finds no kernel.
- **f16 accumulation.** The Turing lists include `f16,f16,f16,f16,f16` (f16 accumulate) descriptors; the AMD families accumulate in fp32 only. That is more accurate and the `use_f32_as_accum` filter only removes descriptors; check no caller requires `dacc == f16`.
- **Fresh allocations not zeroed on ROCm.** `TorchAllocator.empty` returns uninitialised memory on both backends (the torch caching allocator), so exposure is shared with CUDA, but audit counters and masks (`IndiceNumPerLoc`, `PairMask*`, `TightUniqueCount`) for a read-modify-write first access; the study's poisoned-allocator run found none in the spconv path.
- **Shared memory and occupancy.** spconv's SIMT tiles are small; any AMD descriptor above 64 KiB LDS is dropped by cumm on HIP, so the lists must not rely on one.
- **hipRTC.** About half of the SIMT implicit-GEMM configs were NVRTC-only in the study, and any descriptor not prebuilt for the device goes through hipRTC; first-use compile latency is user-visible. Exercise `force_nvrtc` paths in tests; on Windows the hipRTC DLL discovery is cumm's.
- **hipBLASLt coverage.** Only the debug matmul class uses it; confirm hipBLASLt is present for each validated gfx (including the Windows SDK) and that a missing library degrades to a clear error, not an import failure of `core_cc`.
- **Torch-side reference on ROCm.** `test/test_conv.py` compares against dense `nn.Conv3d` on the same GPU, which runs MIOpen on ROCm. If a mismatch appears, rerun the reference on the CPU before blaming spconv (MIOpen solver nondeterminism is a known class).
- **FP contraction and tolerances.** clang's `-ffp-contract=fast` differs from nvcc builds; upstream tolerances (`atol=1e-4` fp32, norm-scaled for f16) are expected to hold; read error magnitudes, do not loosen CUDA tolerances (gate any change on HIP).
- **Build time and package size.** spconv prebuilds roughly 265 `.cu` files (study) per gfx target; the gfx list for wheels multiplies it. Cap ninja parallelism at the host's physical cores; the wheel gfx list is a person's decision.
- **Windows.** MSVC compiles `.cc`, amdclang++ compiles `.cu`; cumm's open `-fPIC` question applies here too, plus `/DNOMINMAX` (`all.py:134`), optional Boost, and DLL search through cumm's loader.

## File-by-file change list

Build and packaging: `setup.py` (HIP `build_ext`, aggregates, name and pins), `spconv/build.py` (HIP kwargs, aggregates), `spconv/constants.py` (only if a backend flag is needed beyond cumm's), `tools/build-wheels-rocm.sh` (new), `README.md` (ROCm build and install section, Linux and Windows, no device-selection variables in front of commands).

Descriptors and selection: `spconv/core.py` (HIP lists and backend-selected aggregates), `spconv/algo.py` (both `get_all_available`), `spconv/csrc/sparse/convops.py` (C++ tuners' `get_all_available`, `get_available_algo_str_from_arch`, `SimpleExternalSpconvMatmul` library and include on HIP, `get_compute_capability` unchanged), `spconv/pytorch/cppcore.py` (`get_arch` warning logic only if cumm's HIP `CompileInfo` needs a different call).

Kernels: `spconv/csrc/sparse/all.py` (thrust policy, `CustomThrustLib` HIP flag, `c++17`/`-w` only for CUDA), `spconv/csrc/sparse/indices.py` (thrust policy).

libspconv: `spconv/gencode/__main__.py` (aggregates, HIP CMake fragment), `example/libspconv/CMakeLists.txt`, `example/libspconv/spconv/CMakeLists.txt`, `example/libspconv/run_build.sh` (ROCm variables), `example/libspconv/README.md`.

Tests (shipped with the port, backend-neutral names, skip with no GPU): `test/test_all_algo.py` (a new test function running fp32 and fp16 through every available implicit-GEMM descriptor and the native GEMM path, `_test_native_conv_cuda`, which upstream leaves commented out, plus a `force_nvrtc` pass; the existing int8 test unchanged), `test/test_hip_selection.py` or an equivalent function (fp16 selection lands on the device's matrix family; int8 finds SimtDP4A descriptors), `test/test_point2voxel.py` (GPU against CPU), `test/test_hash.py` (insert/query/assign_arange against a Python dict, from `example/simple_hash.py`). No upstream test is weakened or skipped.

Expected size: 15 to 22 files, 700 to 1,400 lines (descriptor lists and tests dominate), 3 to 5 porter rounds after cumm reaches a usable head: (1) build route, aggregates, thrust, SIMT fp32 end to end on one width; (2) selection logic and AMD fp16 descriptors (MFMA then WMMA); (3) SimtDP4A int8 and hipRTC; (4) libspconv CMake route and hipBLASLt; (5) Windows. The study's 10 to 14 spconv files predates the matrix families and the libspconv route.

## Build commands

Linux, any ROCm host (TheRock pip SDK or `/opt/rocm`), ROCm PyTorch installed, the cumm port installed from its fork (`moat-port` at the recorded sha); `<gfx>` is the executing GPU's target. The HIP variable names are cumm's (its plan uses `CUMM_HIP_ARCH_LIST` as the working name):

```
python -m venv .venv && . .venv/bin/activate
pip install pccm ccimport pybind11 numpy fire pytest tqdm
# ROCm PyTorch for the device, then the ported cumm (editable or wheel), then:
CUMM_HIP_ARCH_LIST=<gfx> pip install -e . --no-build-isolation      # JIT core_cc at import
SPCONV_DISABLE_JIT=1 CUMM_HIP_ARCH_LIST=<gfx> pip wheel . -w dist --no-build-isolation --no-deps
```

Wrap with `utils/timeit.sh spconv compile -- ...`; cap `MAX_JOBS`/ninja jobs at physical cores. libspconv: `bash example/libspconv/run_build.sh` with the ROCm variables the porter documents, then run `build/main`. Windows: the same commands from a VS developer shell with the ROCm SDK on PATH, `<gfx>` = gfx1151.

CUDA no-regression (mi355 fleet host, real nvcc): build spconv against the cumm port's CUDA path (`CUMM_CUDA_VERSION=12.x`, `CUMM_CUDA_ARCH_LIST=8.0`) with `SPCONV_DISABLE_JIT=1`, import, and diff the generated sources (`python -m spconv.gencode` into two trees, upstream base vs port head, both with the CUDA backend): identical apart from intended HIP-guarded lines.

## Test plan

GPU tests, every platform (the validator runs all of them; `ulimit -c 0`):
1. `pytest test/test_conv.py`: fp32 SparseConv3d, SubMConv3d, SparseInverseConv/deconv, SparseMaxPool3d, global max pool, all three algorithms (Native, MaskImplicitGemm, MaskSplitImplicitGemm), forward and backward against dense torch.
2. `pytest test/test_multi_impl.py`: fp32 and fp16 networks, all algorithms cross-checked and against the stored `test/data/test_spconv.pkl`.
3. `pytest test/test_all_algo.py`: upstream's int8-inference sweep (subm and non-subm, needs the SimtDP4A int8 descriptors) and the port's fp32/fp16 sweep over every available descriptor (fwd, dgrad, wgrad, split-K, prebuilt and `force_nvrtc` through hipRTC) plus the native GEMM sweep.
4. The port's selection, point-to-voxel and hash tests (above). The selection test asserts fp16 picks the MFMA family on CDNA and the WMMA family on RDNA, so a green run cannot be all-SIMT.
5. Examples: `python example/mnist/mnist_sparse.py --fp16` (as `test_before_push.sh` runs it; record accuracy after the default epochs or a recorded short run), `python example/voxel_gen.py`, and libspconv: build with `run_build.sh` and run `main` on the bundled `benchmark-pc.jarr` (Linux; Windows build only if the route is documented there).
6. Performance record (evidence, not pass/fail): `python test/benchmark.py` subset, fp32 and fp16 SubMConv3d/SparseConv3d forward and backward, SIMT versus the device's matrix family. The matrix family must beat SIMT fp16 clearly; slower is a defect to fix before review.

Non-GPU regression set: CPU-only build (no ROCm, no CUDA) imports and `CPU_ONLY_BUILD` is true; CPU operators through `spconv.utils.Point2VoxelCPU*` and CPU index generation (`example/voxel_gen.py` CPU path); Python-side kernel generation for the HIP and CUDA lists without a GPU; the CUDA generated-source diff above.

Per gate:
- **wave64**: gfx90a or gfx942/gfx950, items 1-6 with the MFMA family. Both a gfx90a and a gfx942 run are wanted (the CC collision differs: 9.0 vs 9.4).
- **wave32**: gfx1100 and gfx1201, items 1-6 with the WMMA family. Both layouts (gfx11, gfx12) are different code in cumm, so the gfx12 run is owed even after a gfx11 run satisfies the gate.
- **windows**: windows-gfx1151, last, after both Linux widths pass at the same head (and with the cumm port validated there). Items 1-5 with the WMMA gfx11 layout (MSVC host, amdclang++ device, hipRTC DLL discovery); item 6 recorded only.
- **CUDA**: mi355 nvcc build and generated-source diff before any round is brought to jeff.

## Open questions

Recorded in notes.md for jeff; none blocks the porter and none is a scope cut:
1. int8 on HIP: SimtDP4A descriptors (planned, correct and slower) or AMD int8 matrix families (cumm question 1)?
2. bf16 GEMM/conv descriptors: upstream has none on CUDA; add HIP-only ones because the Option B ruling names bf16, or keep upstream parity (planned)?
3. Wheel name, venue and gfx list for `spconv-rocm7`, after the port is complete (same decision as cumm's).
