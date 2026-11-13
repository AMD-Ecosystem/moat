# tilelang notes

## Intake screen (2026-11-05, gap-matrix seed)

Recommendation: **decline, `already-supported`.** Upstream tile-ai/tilelang has a mature, first-class ROCm/HIP backend; there is no CUDA-to-HIP port to do.

### Licence
- Top level: MIT (`LICENSE`, "Copyright (c) Tile-AI", with a preamble about Microsoft collaboration terms Dec 2024 - Mar 2025; the grant itself is plain MIT). GitHub reports "Other"/NOASSERTION because of that preamble. Recorded `MIT`.
- `licenses.py scan-nvidia agent_space/tilelang-screen` hit one file: `src/cuda/stubs/vendor/cuda.h` (a vendored copy of the CUDA driver API header carrying NVIDIA licence text). Relevant only if we ever edited or redistributed that file; moot for a decline.
- Submodules: `3rdparty/cutlass` (NVIDIA/cutlass, BSD-3-Clause), `3rdparty/tvm` (TileLang/tvm, Apache-2.0). Not recursed further since the recommendation is a decline.

### Existing AMD support (upstream, authoritative)
- `src/rocm/`, `src/tl_templates/hip/` (gemm, copy, atomic, reduce, scan, fp8, fp4 headers), `3rdparty/hip-headers`, and `testing/python/amd/` with about 25 HIP test files (MFMA intrinsic GEMM, MFMA preshuffle, gfx950 async copy, gfx950 MXFP4, ds_read_tr, attention/decoding/NSA/MLA kernels).
- README support table: "AMD ROCm/HIP | `hip` | Linux; CDNA and RDNA GPUs, including gfx942/gfx950 paths | Supported | Included in Linux wheels ... CI runs on a self-hosted gfx942 (MI300X) runner; gfx950 is not yet covered."
- Arch coverage per README news: CDNA4 MXFP4 for gfx950 (#2132), RDNA3/RDNA3.5 WMMA for gfx11 (#2044), RDNA4 (#1951), FlashMLA on MI300X. Open: gfx1200 linear shared layouts for WMMA (#3729), FP4/E8M0 HIP fixes (#3665).
- CI: `.github/workflows/ci.yml` has a `[self-hosted, amd, gfx942]` matrix entry that installs ROCm torch and runs `requirements-test-rocm.txt` plus ROCm example tests.
- Contributors: of about 40 merged ROCm/AMD/HIP PRs sampled, top authors are andyluo7 (14), LeiWang1999 (5, maintainer), zhangnju (4, GitHub company "AMD").
- Packaging: Linux wheels include the HIP target; `pip install tilelang` on top of ROCm torch.

### EmbeddedLLM/tilelang
- Fork of tile-ai/tilelang, last push 2026-06-01. Both its branches (`main`, `mxfp4gemm`) are 0 commits ahead of and 643 behind upstream main: it carries nothing upstream lacks (its MXFP4 work appears to have landed upstream as #2132). Not a live separate port.

### Remaining gap
- Windows ROCm: upstream's HIP target is Linux-only today, but open upstream PR #3425 (oraluben, opened 2026-10-04, last updated 2026-10-28, not draft), "[ROCm] Enable Windows HIP SDK builds and kernel execution", is already doing it. Per intake policy that is a contribute-to-it case, not a parallel port. Our Windows gfx1151 host could at most offer validation evidence on that PR, which is an upstream-visible write and a person's call.
- gfx950 CI coverage is missing upstream, but that is CI hardware, not porting work.

### Question for jeff
- Do you want MOAT to offer Windows gfx1151 validation on upstream tilelang PR #3425 (https://github.com/tile-ai/tilelang/pull/3425)? Otherwise this is a clean `already-supported` decline.
