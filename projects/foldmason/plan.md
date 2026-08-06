# foldmason -- porting plan (linux-gfx90a lead)

## Project
- Name: foldmason
- Upstream: https://github.com/steineggerlab/foldmason
- Default branch: main
- Role: Multiple protein structure alignment at scale; vendors foldseek (which vendors MMseqs2 + libmarv).

## Existing AMD support

### Upstream docs grep
```
grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/
```
Result: Only "amd" in "ARM64 amd SSE2" (typo for "and"). No AMD GPU support mentioned.

### Web search
Searched "foldmason ROCm", "foldmason AMD GPU", "foldmason HIP", "foldmason MI300/gfx9".
Result: No AMD/ROCm port found. No separate AMD project (no ROCm-DS-style). No community forks with HIP/ROCm.

### GitHub forks check
```
gh api repos/steineggerlab/foldmason/forks
```
Result: ~29 forks, none with rocm/hip/amd in name or under ROCm/AMD/GPUOpen orgs.

### Upstream merge policy
foldmason accepts PRs directly (standard GitHub model). No "links-to-forks" reference repo pattern.

### Relationship to foldseek (CRITICAL)
foldmason vendors foldseek as source files in `lib/foldseek/` (NOT a git submodule -- tracked directly in foldmason's git tree). foldseek vendors MMseqs2 at `lib/foldseek/lib/mmseqs/`, which vendors libmarv at `lib/foldseek/lib/mmseqs/lib/libmarv/`.

The MOAT foldseek project (steineggerlab/foldseek) is COMPLETED on linux-gfx90a and linux-gfx1100:
- Fork: https://github.com/AMD-Ecosystem/foldseek (branch moat-port)
- Port HEAD: e7471b4164e38cbac58b4f2c6c1b592e9bfac330
- Base: 718d42176d2f67d36a60866fedfb881f8d5a7ebf (upstream master)

foldmason's vendored foldseek/mmseqs/libmarv matches the upstream foldseek BASE commit (718d4217) that was ported. The HIP changes are NOT yet in foldmason.

### Decision: CLEAN PORT, reuse foldseek delta
- Authoritativeness: N/A (no existing AMD effort)
- Value: Applying the completed foldseek HIP port to foldmason's vendored copy brings GPU support (libmarv structure aligner for search workflows, potentially ProstT5 ggml-HIP for embeddings). foldmason's own C++ code has no CUDA.
- Approach: Apply the diff from AMD-Ecosystem/foldseek moat-port (718d4217..e7471b41) to foldmason's lib/foldseek/ subtree, then thread the USE_HIP option through foldmason's top-level CMake.

## Build classification: CMake (Strategy A). Evidence
- Top-level CMakeLists.txt: `project(foldseek C CXX)`, `add_subdirectory(lib/foldseek EXCLUDE_FROM_ALL)`. Standard CMake.
- No `find_package(Torch)`, no `torch.utils.cpp_extension`, no setup.py/pyproject.toml. NOT a pytorch extension.
- CUDA lives in the nested libmarv (`lib/foldseek/lib/mmseqs/lib/libmarv/src/`), same as foldseek.
- `ENABLE_CUDA` option exists at top-level and foldseek CMakeLists.txt but is OFF by default and primarily gates ProstT5 ggml-CUDA.

## Port strategy: A (compat header + enable_language(HIP) + mark .cu LANGUAGE HIP)

The foldseek HIP port (completed, validated) applies directly:
1. Apply the 28 changed files from AMD-Ecosystem/foldseek 718d4217..e7471b41 to foldmason's lib/foldseek/ subtree.
2. Thread `USE_HIP` from foldmason's top-level CMakeLists.txt down to foldseek -> mmseqs -> libmarv (matching the existing foldseek wiring).

All technical decisions are inherited from the foldseek port:
- `cuda_to_hip.h` compat header: cudaXxx->hipXxx runtime + Thrust/CUB + fp16 + DPX scalar emulations.
- `hip_compat/` shim directory: cooperative_groups reduce, cuda_fp16 forwarding.
- libmarv built SHARED with `-fgpu-rdc` + `--hip-link` device link.
- DPX path forced off on AMD (cc==9 collision trap).
- Wave-size handling: getPaddedQueryLength pinned at literal 32; 64-bit WARP_FULL_MASK; warpSize/2 reduction.
- Headers marked HEADER_FILE_ONLY to avoid duplicate __constant__ device globals.

## CUDA surface inventory (inherited from foldseek plan)
All CUDA is in libmarv at `lib/foldseek/lib/mmseqs/lib/libmarv/src/`. foldmason's own code (`src/`) has no CUDA.
- Kernels: gapless + Smith-Waterman PSSM kernels, cg::tiled_partition group-strided.
- Warp/lane: __shfl_*_sync with 32-bit masks, offset=16 reductions, __reduce_max_sync guarded.
- Libraries: Thrust + CUB -> rocThrust/hipCUB. NO cuBLAS/cuFFT/cuRAND/cuSPARSE.
- Memory: streams/events via RAII wrappers, pinned/managed memory. No textures/surfaces.

## Risk list (inherited from foldseek, already mitigated in the port)
1. CC==9 collision (gfx90a vs sm90 Hopper) -> DPX path forced off under __HIPCC__.
2. Wave size (gfx90a wave64) -> getPaddedQueryLength pinned at 32; 64-bit masks; warpSize/2 reductions.
3. __hmax2/__hmin2 missing on HIP -> pairwise emulation via scalar __hmax/__hmin.
4. cooperative_groups reduce missing -> butterfly shim in hip_compat/.
5. RDC + shared lib device link -> SHARED libmarv with --hip-link.

No NEW risks beyond the foldseek port. foldmason's own C++ is CUDA-free.

## File-by-file change list (lead, gfx90a)
Applying the foldseek diff to foldmason's lib/foldseek/ subtree:

### New files (from foldseek port)
- `lib/foldseek/lib/mmseqs/lib/libmarv/src/cuda_to_hip.h`
- `lib/foldseek/lib/mmseqs/lib/libmarv/src/hip_compat/cooperative_groups.h`
- `lib/foldseek/lib/mmseqs/lib/libmarv/src/hip_compat/cooperative_groups/reduce.h`
- `lib/foldseek/lib/mmseqs/lib/libmarv/src/hip_compat/cuda_fp16.h`

### Modified files (from foldseek port, paths adjusted for foldmason)
- `CMakeLists.txt` (foldmason top-level): thread USE_HIP to foldseek subdir
- `lib/foldseek/CMakeLists.txt`: thread USE_HIP to mmseqs
- `lib/foldseek/lib/mmseqs/CMakeLists.txt`: thread USE_HIP to libmarv
- `lib/foldseek/lib/mmseqs/lib/libmarv/src/CMakeLists.txt`: enable_language(HIP), SHARED lib, -fgpu-rdc
- `lib/foldseek/lib/mmseqs/src/CMakeLists.txt`: link libmarv under USE_HIP
- `lib/foldseek/src/CMakeLists.txt`: link libmarv under USE_HIP
- `lib/foldseek/lib/mmseqs/lib/libmarv/src/*.cuh` files (12 files): HIP guards, warp-size fixes, __HIPCC__ branches

## Build commands (gfx90a)
```
cmake -S projects/foldmason/src -B projects/foldmason/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build projects/foldmason/build --target foldmason -j16
```

Notes:
- `USE_HIP=ON` enables the libmarv HIP build (gates parallel to ENABLE_CUDA).
- ProstT5 ggml-CUDA/HIP is out of scope for initial port (disabled by default, ENABLE_PROSTT5=0 in foldmason CMakeLists.txt).
- libmarv.so will contain gfx90a device code objects.

## Test plan (real GPU correctness gate)

### Bundled regression tests (ZERO downloads, ~2MB data)
foldmason ships regression tests with 5 bundled structures in `regression/data/`:
```
regression/run_regression.sh ./build/foldmason ./tmp
```
Runs: run_easymsa.sh, run_msa2lddt.sh, run_refinemsa.sh, run_structuremsa.sh.

These use structuremsa/structurealign commands which are CPU-based MSA algorithms (NOT the GPU prefilter path). The tests validate that the build is functional and foldmason core algorithms work.

### GPU validation (inherited from foldseek -- the GPU code IS foldseek's)
The GPU code path is libmarv's ungappedprefilter for `search`/`easy-search` with `--gpu 1`. To validate the HIP port of the vendored libmarv, use the foldseek validation protocol on the foldmason binary (which can invoke the same commands since it's built on foldseek's framework):

```
# Using bundled foldseek example structures (already in lib/foldseek/example/)
foldmason createdb lib/foldseek/example/ exDB
foldmason search exDB exDB aln_cpu tmp_cpu --gpu 0 -e 10      # CPU oracle
foldmason makepaddedseqdb exDB exDB_pad                       # GPU needs padded DB
HIP_VISIBLE_DEVICES=X foldmason search exDB exDB_pad aln_gpu tmp_gpu --gpu 1 -e 10
foldmason convertalis exDB exDB aln_cpu cpu.m8
foldmason convertalis exDB exDB_pad aln_gpu gpu.m8
# Diff cpu.m8 vs gpu.m8 -- all CPU hits must appear in GPU with identical scores
```

Expected: GPU results are a strict superset of CPU results (GPU has higher sensitivity for borderline hits), with byte-identical alignment fields for shared (query,target) pairs.

### Non-GPU regression
The regression tests (structuremsa/structurealign/refinemsa) are CPU-based and must not regress.

## Open questions
1. Does foldmason expose the `--gpu` flag on any command, or is it purely foldseek/mmseqs-level? (Likely inherited from foldseek; check `foldmason --help`.)
2. Should the porter cherry-pick the foldseek diff or re-implement it fresh? (Cherry-pick/apply preferred to avoid drift; the libmarv is identical.)
3. ProstT5 ggml-HIP: out of scope for initial port but could be enabled later if ENABLE_PROSTT5=1 + GGML_HIP=ON. ggml has existing HIP support in llama.cpp lineage.

## Follower platforms
gfx1100/gfx1151/gfx1201 (wave32) reuse this branch and validate first. No code changes expected for wave32 -- the wave-size handling in the foldseek port is already wave-agnostic (getPaddedQueryLength pinned at 32, 64-bit masks auto-narrow, tiled_partition<N> with N<=32).
