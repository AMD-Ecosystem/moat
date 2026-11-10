# koboldcpp notes

## Intake screen (2026-11-05, windows-gfx1151 host)

Source: third-party "AMD AI Software Gap Matrix" seed (verdict partial, AMD path upstream). Screened at upstream `concedo` @ 6d2fedf8624e5f05926af3191f9efdcb13c4133c (shallow clone in `agent_space/koboldcpp-screen`).

### Licence

- `python3 utils/licenses.py check LostRuins/koboldcpp` -> AGPL-3.0, tier 3 (strong copyleft; contributing needs a person's approval). LICENSE.md is the AGPL-3.0 text. Recorded with `set-license-spdx`.
- `python3 utils/licenses.py scan-nvidia agent_space/koboldcpp-screen` -> no NVIDIA proprietary licence text.
- Vendored ggml/llama.cpp and stable-diffusion.cpp are MIT upstream; not a blocker for a decline.

### CUDA surface

- All CUDA lives in the vendored ggml CUDA backend (`ggml/src/ggml-cuda/`, the llama.cpp backend with its mature HIP path) plus three koboldcpp-specific legacy files: `otherarch/ggml_v2-cuda.cu`, `otherarch/ggml_v2-cuda-legacy.cu`, `otherarch/ggml_v3-cuda.cu` (old GGML v2/v3 model formats).
- No other `.cu`/`.cuh` anywhere in the tree.

### Existing AMD support (upstream, first party)

- Makefile: `make LLAMA_HIPBLAS=1` builds `koboldcpp_hipblas` with `-DGGML_USE_HIP -DGGML_USE_CUDA -DGGML_HIP_NO_VMM`, auto-detects targets via `amdgpu-arch` (default list gfx803..gfx1201), links hipBLAS/rocBLAS, and has both `.so` and `.dll` build rules.
- CMakeLists.txt: `LLAMA_HIPBLAS` option builds ggml-rocm plus `ggml-v2-rocm`, `ggml-v3-rocm`, `ggml-v2-legacy-rocm` -- i.e. the koboldcpp-specific legacy `.cu` files are also compiled for HIP.
- README build table: "AMD ROCm | `make LLAMA_HIPBLAS=1`"; also links the AUR `koboldcpp-hipblas` package.
- CI: `.github/workflows/kcpp-build-release-linux-rocm.yaml` builds in `rocm/dev-ubuntu-22.04:7.14.0-full` and uploads `koboldcpp-linux-x64-rocm` to the `rocm-rolling` release (last asset update 2026-10-30, matching v1.124.2 of 2026-10-31).
- Merged upstream PRs maintaining it: #1823 "ROCm 7.1 CI" (2025-11), #2349 "AMD ROCm version bump for the binary" (2026-07). No open HIP/ROCm PRs.

### Community fork

- YellowRoseCx/koboldcpp-rocm (780 stars): last release v1.104.yr0-ROCm 2025-12-30, last push 2026-07-29; about 20 releases behind upstream v1.124.2. README itself says "check its maintenance status". Non-authoritative and now largely superseded by upstream's own ROCm build. Its historical value was Windows ROCm binaries.
- No ROCm or AMD-Ecosystem repo for koboldcpp (`gh search repos koboldcpp --owner ROCm --owner AMD-Ecosystem` -> none). The `amd` / `AMD-AGI` orgs were not searched (SAML SSO for this token), so a duplicate check there is still open, though irrelevant to a decline.

### Gap

Only packaging: upstream ships no Windows ROCm binary (Windows AMD users are pointed at Vulkan) and the README's GPU acceleration line still says `--usecuda` is "Nvidia Only" without mentioning the Linux ROCm rolling binary. The Makefile already has the Windows `.dll` HIP rule, and llama.cpp itself ships Windows HIP builds, so this is release-engineering for the maintainer rather than a CUDA-to-HIP port. There is no unported CUDA code.

### Recommendation

Decline, `SKIP_REASON` `already-supported`: upstream koboldcpp builds its whole CUDA surface (vendored ggml plus its own legacy v2/v3 `.cu` files) with HIP via `LLAMA_HIPBLAS`, and publishes a CI-built Linux ROCm binary at each release. The residual Windows-binary/docs gap is packaging, not porting, and on a tier-3 (AGPL-3.0) project.

## Ruling: validation project (Jeff Daily, 2026-11-10)

Jeff Daily overrode the intake decline: koboldcpp becomes a validation project. The port is not stale upstream, but the only ROCm fork is a stale community one (YellowRoseCx/koboldcpp-rocm) and AMD has no fork. Validate upstream's own HIP build (LLAMA_HIPBLAS) on every required gate (wave64, wave32, Windows) and fix what breaks. The fixes go on `moat-port` of an AMD-Ecosystem fork. A person creates that fork, never an agent; the project waits in `awaiting-fork` until it exists, then `release-forks` moves it to `screened` for the planner.

Licence: AGPL-3.0 is tier 3. Validation and fork-side fixes need no clearance. Anything offered upstream needs a licence clearance recorded by a person (`record-license-clearance`) first.
