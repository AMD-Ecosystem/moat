# Backward Compatibility Guidelines (for ports)

For a CUDA-to-ROCm/HIP port the backward-compatibility contract is: do not change the upstream project's existing behavior. ROCm support must be ADDITIVE and GUARDED. A port that quietly alters the CUDA or CPU code path is BC-breaking even if the ROCm path works.

As a reviewer, be paranoid about any change that touches a code path the upstream already ships. Your key role is to catch a "port" that is really a behavior change in disguise.

## What constitutes a BC break in a port

| Change | BC impact | Action |
|--------|-----------|--------|
| Edits CUDA-only or CPU-only code unconditionally | Breaking | Guard the change for ROCm, or justify why it is correct for all backends |
| Renames `.cu` files / moves sources | Potentially breaking (build, downstream includes) | Prefer `set_source_files_properties(... LANGUAGE HIP)`; keep filenames |
| Changes a default, flag, or public API of the project | Breaking | Keep defaults; add ROCm behind an option (USE_HIP), off by default |
| Changes numeric results on the existing CUDA path | Breaking | The CUDA path must be unchanged unless explicitly intended |
| Replaces a CUDA library call with a HIP one unconditionally | Breaking | Alias via the compat header on ROCm only; leave CUDA spelling on NVIDIA |
| Adds a hard dependency on ROCm for all builds | Breaking | ROCm deps gated behind the HIP build option |

## The additive-and-guarded principle

- New ROCm code lives behind `#if defined(USE_HIP)` / `__HIP_PLATFORM_AMD__`, a CMake `USE_HIP` option (default OFF), or the single compat header that is a no-op on NVIDIA.
- Shared code edited for ROCm (for example, replacing a hardcoded `32` with a warp_size abstraction) must remain correct on the original CUDA path. This is the one place a shared edit is acceptable: it must be a strict generalization, identical behavior on NVIDIA.
- Never delete or rewrite the CUDA path to make the HIP path simpler.

## Cross-platform regression (MOAT-specific)

All AMD targets share one fork branch. A change for one arch (wave32 on gfx1100) must not regress another already-validated arch (wave64 on gfx90a). Any fix to shared (non-arch-guarded) code must be arch-unified: correct on every wave size. A per-arch `#if` that flips behavior is a smell unless the divergence is genuinely hardware-specific. This is why already-completed platforms re-validate after any HEAD advance.

## Questions to ask

1. Does this line run on the CUDA or CPU path too? If yes, is its behavior identical there?
2. Is the ROCm code reachable on a non-ROCm build? It should not be.
3. Could a downstream consumer break (renamed file, changed default, new required dep)?
4. Does a shared-code fix hold on both wave32 and wave64?
5. If unsure, raise it in the review as a point to investigate.
