# metaeuk -- porting plan (linux-gfx90a lead)

## Project
- Name: metaeuk
- Upstream: https://github.com/soedinglab/metaeuk
- Default branch: main
- Role: eukaryotic gene prediction; uses MMseqs2 protein search framework (vendored under `lib/mmseqs`).

## DISPOSITION: CLEAN PORT (tractable). Strategy A. State -> planned. Dispatch a porter.
Effort class: MEDIUM-LOW. metaeuk vendors MMseqs2 + libmarv (the same GPU library already ported for MOAT's standalone MMseqs2 project). The port approach is identical to MMseqs2: apply the same `USE_HIP` gate + compat header + SIMD emulation to the vendored `lib/mmseqs/lib/libmarv`. No new GPU surface beyond what was analyzed in the MMseqs2 plan.

## Existing AMD support
- Grep in-repo docs: `grep -rniE 'amd|rocm|hip|gfx[0-9]' README*` yields only "AMD or Intel 64-bit system" referring to CPU architecture (SSE/AVX), NOT GPU.
- WebSearch ("metaeuk ROCm/AMD GPU HIP", "MMseqs2 ROCm/AMD"): NO AMD/ROCm port exists for metaeuk or its vendored MMseqs2. The Nature Methods 2025 MMseqs2-GPU paper and upstream README confirm NVIDIA CUDA only (Turing+ / Ampere+).
- GitHub forks: `gh api repos/soedinglab/metaeuk/forks` shows no ROCm/AMD/GPUOpen-org forks.
- Authoritative-vs-community judgment: N/A (nothing found). From-scratch HIP port our way.

Note: MOAT has already completed a HIP port of standalone MMseqs2 (soedinglab/MMseqs2 -> AMD-Ecosystem/MMseqs2 moat-port branch, validated on linux-gfx90a and linux-gfx1100). However, metaeuk VENDORS a frozen copy of MMseqs2 under `lib/mmseqs/` (not a git submodule), so the port edits must be applied to metaeuk's own vendored copy. The porting strategy is identical; we adapt the same changes.

Merge policy: Upstream metaeuk (and MMseqs2) accepts PRs; no link-to-forks policy. An upstream PR is the appropriate delivery vehicle once all platforms are validated.

## Build classification: pure CMake (Strategy A). Evidence
- `CMakeLists.txt` (top): `project(metaeuk CXX)`, `add_subdirectory(lib/mmseqs)`. No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py.
- `lib/mmseqs/CMakeLists.txt` lines 21, 250-254: `set(ENABLE_CUDA 0 CACHE BOOL "Enable CUDA")` ... `if (ENABLE_CUDA) ... add_subdirectory(lib/libmarv/src)`.
- `lib/mmseqs/lib/libmarv/src/CMakeLists.txt`: `project(... LANGUAGES CXX CUDA)`, `add_library(marv ...)`, NVCC_FLAGS `-rdc=true --extended-lambda --expt-relaxed-constexpr`, `CUDA_SEPARABLE_COMPILATION ON`.
=> Strategy A (compat header + `enable_language(HIP)`, mark `.cu` LANGUAGE HIP, gate on a HIP option).

## Port strategy: A (compat header + HIP language gate)
Identical to the standalone MMseqs2 port:
1. Add `USE_HIP` option to CMake; when set, `enable_language(HIP)` and mark libmarv `.cu` sources `LANGUAGE HIP`.
2. Add `cuda_to_hip.h` compat header aliasing cudaXxx -> hipXxx runtime symbols.
3. Emulate the SIMD intrinsics missing on HIP (`__vadd2`, `__vmaxs2`, `__viaddmax_s16x2[_relu]`, `__vimax3_s16x2`, `__vibmax_u16x2`, `__hmax2`, etc.) via scalar fallbacks in `mathops.cuh` / a new `marv_simd_amd.cuh`.
4. Add AMD config branch in `gapless_kernel_config.cuh` and `smithwaterman_kernel_config.cuh` (return a conservative config set).
5. Swap NVCC flags for HIP equivalents; exclude DPX-only instantiation TUs on HIP.
6. Keep NVIDIA path byte-identical.

The standalone MMseqs2 MOAT port (AMD-Ecosystem/MMseqs2@moat-port) serves as the reference implementation; adapt those exact changes to metaeuk's vendored copy.

## CUDA surface inventory (inherited from MMseqs2/libmarv)
(See MMseqs2 plan.md for full detail; summarized here.)
- 37 `.cu`/`.cuh` files in `lib/mmseqs/lib/libmarv/src/`.
- Kernels: gapless PSSM + Smith-Waterman alignment kernels, groupsize-strided (`cg::tiled_partition<4/8/16/32>`).
- SIMD intrinsics (~24 sites): `__vadd2`, `__vmaxs2`, `__viaddmax_s16x2[_relu]`, `__vimax3_s16x2`, `__vibmax_u16x2`, `__vmaxu4`, `__vadd4`, `__vaddus4/__vsubus4`, `__hmax2`, `__hadd2`. These DPX / packed-SIMD intrinsics have no HIP equivalents and must be emulated with scalar fallbacks.
- Warp shuffles: `__shfl_down_sync`/`__shfl_up_sync`/`__shfl_sync` with explicit width <= 32 (arch-agnostic logical warps).
- Cooperative groups: `cg::tiled_partition`, `cg::reduce` (rocPRIM-backed on HIP).
- Thrust: used in `dbbatching.cuh`, `cudasw4.cuh`, custom allocators -> rocThrust (1:1 drop-in).
- Streams/events/pinned memory: standard cuda* -> hip* runtime mappings.
- NO cuBLAS/cuFFT/cuRAND/cuSPARSE, NO textures/surfaces, NO WMMA/tensor-core, NO inline PTX in the hot path (the Blackwell int8 path in `ptx_wrappers.cuh` is guarded and never selected on pre-sm120).

## Risk list
1. **Wave size**: LOW. All tile widths are <=32 (explicit `groupsize` parameter); shuffle widths are explicit <=32. gfx90a wave64 executes 32-lane logical tiles within a single wavefront. No wave-geometry hardcoding.
2. **SIMD emulation correctness**: The DPX intrinsics have specific saturation and tie-break semantics (`__viaddmax_*_relu` = max(a+b, c, 0), `__vibmax_*` returns per-lane max + a >= b bool). Emulations must match exactly; validate against CPU reference.
3. **Dynamic shared memory**: gfx90a caps at 64 KB dynamic shared per block. Large-tile configs tuned for NVIDIA's 100+ KB must be pruned from the AMD config list.
4. **RDC + template bloat**: `-fgpu-rdc` with many half2/short2/float kernel instantiations may inflate objects; apply `--offload-compress` if needed.
5. **`__hmax2` emulation**: Must match CUDA NaN propagation (non-NaN operand wins).
6. **CC-based config selection**: `getOptimalKernelConfigs_*(deviceId)` uses compute capability; AMD reports a non-NVIDIA cc value. Add an AMD branch returning a conservative config (e.g. sm75-equivalent).

## File-by-file change list
(All paths relative to `lib/mmseqs/lib/libmarv/src/` within the metaeuk repo.)
- **NEW `cuda_to_hip.h`**: Compat header with cuda* -> hip* runtime aliases + `<cstring>/<cstdlib>` before HIP headers.
- **NEW (or inline) `marv_simd_amd.cuh`**: Scalar emulations of missing SIMD intrinsics.
- **`CMakeLists.txt`**: Add `USE_HIP` gate, `enable_language(HIP)`, mark ALIGN_SOURCES `LANGUAGE HIP`, `HIP_ARCHITECTURES` from `${CMAKE_HIP_ARCHITECTURES}`, swap NVCC_FLAGS, exclude `*_dpx*.cu` on HIP.
- **`mathops.cuh`**: Route intrinsics to emulations under `#if defined(USE_HIP)`.
- **`kernelhelpers.cuh`**: Same intrinsic routing; guard `__reduce_max_sync` with `defined(__CUDA_ARCH__) &&`.
- **`convert.cuh`**: Route single intrinsic site.
- **`gapless_kernel_config.cuh`, `smithwaterman_kernel_config.cuh`**: Add AMD cc branch with conservative configs.
- **`hpc_helpers/cuda_helpers.cuh`**: Replace inline PTX `laneid` with `__lane_id()` under HIP guard.
- **`ptx_wrappers.cuh`**: Ensure Blackwell int8 path is guarded out on HIP.

Parent CMake (`lib/mmseqs/CMakeLists.txt`, top-level `CMakeLists.txt`): Thread `USE_HIP` option down to libmarv.

## Build commands (gfx90a)
```bash
cmake -S projects/metaeuk/src -B projects/metaeuk/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DENABLE_CUDA=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build projects/metaeuk/build -j16
```
Follower platforms: change only `-DCMAKE_HIP_ARCHITECTURES=gfx1100` (or `gfx90a;gfx1100` for multi-arch fat binary validation).

## Test plan (real GPU correctness gate)
metaeuk's own test suite (`tests/run.sh`) tests CPU gene prediction workflows, not GPU search. The GPU path is exercised via the underlying MMseqs2 GPU search.

**GPU correctness gate** (same approach as MMseqs2): compare GPU search results against CPU reference.
```bash
# Create test database from metaeuk example data
./metaeuk createdb tests/multi_exon/contigs.fna testContigs --dbtype 2
./metaeuk createdb tests/multi_exon/proteins.faa testProteins --dbtype 1

# CPU reference (no GPU)
./metaeuk predictexons testContigs testProteins cpu_results tmp_cpu --gpu 0

# GPU run
./metaeuk predictexons testContigs testProteins gpu_results tmp_gpu --gpu 1

# Compare outputs
diff cpu_results gpu_results
```

If metaeuk does not expose `--gpu` directly, test via the underlying `mmseqs` commands:
```bash
# Build mmseqs from the vendored lib/mmseqs if needed
mmseqs createdb tests/multi_exon/proteins.faa protDB
mmseqs easy-search tests/multi_exon/proteins.faa protDB cpu.m8 tmp_cpu --gpu 0
mmseqs easy-search tests/multi_exon/proteins.faa protDB gpu.m8 tmp_gpu --gpu 1
diff cpu.m8 gpu.m8
```

**Non-GPU regression set**: Run the existing `tests/run.sh` to confirm CPU gene prediction workflows are unaffected by the HIP port (the GPU option defaults off; CPU path must not regress).

**Cross-arch consistency (followers)**: gfx1100/gfx1201 GPU results must match gfx90a GPU results (and CPU) on the same inputs.

## Open questions
1. Does metaeuk expose `--gpu` flag in its workflows, or only through the underlying MMseqs2 calls? (Porter to check `metaeuk --help` output.)
2. Exact metaeuk version of vendored MMseqs2 vs the ported standalone MMseqs2: git history shows metaeuk's copy is slightly behind the moat-port base. Confirm the HIP changes apply cleanly.
3. Whether to factor out the libmarv HIP port as a shared patch set vs duplicating the port across MMseqs2/foldseek/metaeuk. (Decision: for now, each vendor their own copy; can unify later via upstream PR.)

## Follower platforms
gfx1100/gfx1201 reuse this branch, validate first. If the port is wave-agnostic (expected: all tiles <=32 width explicit), validation should pass with only the arch flag change.
