# anari-visionaray GPU tests

The project has no in-tree test target, so the GPU gate is this headless ANARI harness. Every validator of this project runs these files unchanged, so results from different hosts can be compared number for number. Change them only in a commit on the MOAT branch, and record that change in `notes.md` so later runs are compared against the right version.

- `anari_validate.cpp` loads one or two ANARI libraries (`visionaray_hip`, then `visionaray` as the CPU reference), builds a scene, renders it `RUNS` times at 256x256 (`UFIXED8_RGBA_SRGB`), each time with a fresh world and frame, and prints one line per device: non-background pixel count, distinct colours, how many runs were pixel-identical to the first, and the wall time. With two devices it also prints the mean absolute per-channel difference (0..255) between their first images.
- `run_all.sh` compiles the harness and runs every scene with both renderers, one scene per process (a GPU fault leaves the process's HIP device unusable for later scenes), under `ulimit -c 0` and a per-process `timeout`. A timeout counts as a failure.

## Scenes

| name | content |
|---|---|
| `volume-nearest`, `volume-linear` | 32^3 `structuredRegular` radial blob, filter nearest / linear, transfer function blue-green-red, no light |
| `nanovdb-nearest`, `nanovdb-linear` | NanoVDB `createFogVolumeSphere(50)`, filter nearest / linear, no light |
| `gaussian-splat` | 216 Gaussian splats on a 6^3 grid, scale 0.05 and opacity 0.9 (the device takes both as linear values), colour from `f_dc`, no light. HIP only: the CPU device crashes on splats in upstream's own code (an over-aligned `dco::Gaussian` in `std::malloc` memory, see notes) |
| `surface` | two vertex-coloured triangles, matte material, one directional light |

Renderers: `default` (path tracing; `RENDERER=default`) and `raycast`. `run_all.sh` runs `surface` last in each renderer, so a case that hangs or wedges the GPU cannot cost the other results.

PASS for a case = every run of every device renders, the image has non-background pixels and more than one distinct colour, and the process exits within the timeout. Run-to-run identity and the HIP-vs-CPU difference are printed for comparison, not gated; compare them with the last record in `notes.md`.

## Running

Build ANARI-SDK, visionaray and the fork as in `notes.md` (the HIP build needs the visionaray `hip_bvh.h` fix recorded there), then:

```
bash projects/anari-visionaray/gpu_tests/run_all.sh <anari-install> <fork-build>/install <fork-src> <outdir>
```

Environment: `RUNS` (default 8), `TIMEOUT` seconds per process (default 600), `KILL_AFTER` seconds between the TERM and the KILL of a timed-out case (default 120), `SCENES` and `RENDERERS` to run a subset, `CXX`. Two switches change the scene for bisecting a failure and are not part of a validation run: `NO_LIGHT=1` drops the surface scene's directional light, and `MODE=<mode>` sets the renderer's `mode` parameter (for example `Ng`, which skips the light loop in the raycast renderer). Keep the dispatcher's GPU pin (`HIP_VISIBLE_DEVICES`). Each case writes `<outdir>/<renderer>-<scene>.log` and the first image per device as `<outdir>/<renderer>-<scene>-<device>.ppm`.

Do not kill a hung case with `kill -9` while another HIP process is running on the same GPU, and do not attach a debugger to a hung case; let the script's timeout end it. On 2026-11-23 a gfx1201 card that had survived two raycast-surface timeouts stopped running any HIP program (a 64-thread smoke kernel hangs in `hipStreamCreate`) after overlapping debug runs were killed that way, and it needed a host reset.
