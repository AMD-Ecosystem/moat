# flower port plan

Planned 2026-11-20 on linux-gfx1100 (planner role, no GPU used) against upstream `main` at b65f434 (2026-11-20).

**Two rulings from Jeff are needed before a porter starts (Q1 and Q2 under Open questions).** The rest of this plan holds whichever way they go; where an answer changes the work, the section says so.

## Project

- Upstream: https://github.com/flwrlabs/flower, default branch `main`, Apache-2.0.
- Fork: https://github.com/AMD-Ecosystem/flower. On 2026-11-20 its `main` and `moat-port` are both 311ff9b (an upstream mirror from 2026-11-12, 41 behind upstream). Write access was granted after intake (`permissions.push = true`), so the mirror can now be fast-forwarded.
- The repository is a monorepo: `framework/` (the `flwr` Python package, its C++ SDK in `framework/cc`, Kotlin and Swift mobile SDKs, docs, e2e tests), `examples/` (42), `baselines/` (31 paper reproductions), `datasets/` (`flwr-datasets`), `benchmarks/`, `agent/`, `hub/`, `model/`.

## Existing AMD support

Improvable. AMD's own upstream PR flwrlabs/flower#8383 (smahto957, head `AMD-Ecosystem/flower:smahto/flower_amd`, 5b30234 on 2026-11-19; state OPEN, `REVIEW_REQUIRED`, `mergeStateStatus` UNKNOWN, which is not "no conflict") is authoritative: AMD-authored, narrow, still being updated. It adds logging-only zero-GPU diagnostics to `framework/py/flwr/simulation/ray_transport/ray_actor.py`, a mocked `ray_actor_test.py`, and an AMD GPU simulation how-to plus edits to three docs pages. It claims AMD Instinct on Linux only. It contains no Radeon or Windows evidence and no discovery change.

Decision: do not re-port. MOAT's value is the evidence #8383 lacks (RDNA/wave32 Linux, Windows, and a reproducible ROCm-torch recipe on all three gates), plus whatever real defects that validation turns up. Jeff's ruling of 2026-11-11 stands: no PR competing with #8383. How the findings reach upstream is Q1.

Upstream does not link platform forks, so if a code or docs delta is wanted it would go upstream as a PR (or into #8383), not as a linked fork. There are no performance kernels; the mechanical-port versus AMD-native-rewrite question does not arise.

## Build classification

`python` (pure Python, uv-build). Evidence:

- `framework/pyproject.toml`: `[build-system] requires = ["uv-build>=0.12.20,<0.13"]`; no extension modules, no `setup.py`, no `torch.utils.cpp_extension`.
- `find . -name '*.cu' -o -name '*.cuh'` over the whole tree: 0 files. The only CMake in the tree is the CPU-only C++ client SDK (`framework/cc/flwr/CMakeLists.txt`, gRPC/protobuf, no CUDA) and its example (`examples/quickstart-cpp`).
- GPU compute comes entirely from the user's ML framework (PyTorch, TensorFlow, JAX) inside a ClientApp. Flower's own GPU touchpoint is the Simulation Runtime's Ray backend, which sizes the actor pool from Ray's `GPU` resource and lets Ray pin each actor to a device.

## Port strategy

Neither A nor B: a validation project. No HIP translation exists to do. The work is:

1. Fast-forward the fork's mirror to upstream `main` (`moatlib.py mirror-branch`) and cut `moat-port` from it.
2. Establish and record a repeatable ROCm environment recipe for Flower on each gate (ROCm torch wheel, `flwr[simulation]`, the `amdsmi` Python package matched to the host's `libamd_smi.so`).
3. Run Flower's CPU suite and the GPU-exercising e2e and example runs below on wave64, wave32 and Windows, and prove the ClientApps actually ran on the AMD GPU.
4. Fix only real defects found in Flower itself. What else, if anything, lands on `moat-port` is Q1.

## CUDA surface inventory

Empty census (no `.cu`/`.cuh`, no CUDA libraries). The GPU surface that matters is runtime, and the scan cannot see it, so it is added by hand to `surface.json`:

| surface | where | ROCm behaviour (as read from Ray 2.56.0 source, not yet run) | risk |
|---|---|---|---|
| Ray GPU discovery | `ray/_private/accelerators/amd_gpu.py` + vendored `ray/_private/thirdparty/pyamdsmi/pyamdsmi.py` in `ray==2.56.0` (Flower's `simulation` extra pins it) | Ray counts AMD GPUs by `import amdsmi`; `amdsmi_init(INIT_AMD_GPUS)`; `len(amdsmi_get_processor_handles())`. Any exception (including `ImportError`) is swallowed and the count is 0. | **High.** `amdsmi` is not a dependency of `ray` or `flwr[simulation]`. The PyPI `amdsmi` wheel (27.2.0, Linux x86_64 only) ships only the Python binding (`libamd_smi_python.so`) and needs the host's `libamd_smi.so` (from `/opt/rocm` or a matching ROCm). A pip-only ROCm torch environment therefore sees 0 GPUs, and any `num_gpus > 0` simulation stops with "ActorPool is empty". This is the gap #8383 diagnoses but does not fix. |
| Ray device pinning | `amd_gpu.py set_current_process_visible_accelerator_ids` | Ray writes `HIP_VISIBLE_DEVICES` per actor (or `CUDA_VISIBLE_DEVICES` if that alone is set; warns on `ROCR_VISIBLE_DEVICES`). ROCm torch honours it, so `cuda:0` in the ClientApp maps to the assigned GPU. | Medium: unverified with fractional `num_gpus` (several actors on one GPU) and with a host that already sets `HIP_VISIBLE_DEVICES`. |
| Ray accelerator type | `amd_gpu.py amd_product_dict` | Table holds Instinct MI2xx/MI3xx ids and old GCN "HD 7900" ids (0x6798...), no RDNA ids, so `accelerator_type` is None on Radeon. Flower never requests an accelerator type. | Low; record it, do not fix in Flower. |
| Ray amdsmi shutdown | `amd_gpu.py _get_amd_device_ids` | Calls `pyamdsmi.pyamdsmi_shutdown()`, which the vendored module does not define (it has `smi_shutdown`); the `AttributeError` is swallowed, leaving amd-smi initialised. | Low; a Ray defect. Validator confirms on hardware before anyone registers it. |
| Ray on Windows | PyPI has `ray-2.56.0` `win_amd64` wheels for cp310 to cp312 only; Flower's extra installs ray on win32 only for Python < 3.13 | No AMD discovery on Windows (no `amdsmi` wheel for Windows), so Ray reports 0 GPUs unless `ray.init(num_gpus=N)` is given; Flower exposes that as `init-args-num-gpus`. | Medium; how to offer it is Jeff's later decision (ruling of 2026-11-11). This plan only gathers evidence. |
| ClientApp device selection | every PyTorch example: `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` | Works unchanged on ROCm torch (HIP masquerades as `cuda`). | Low. Silent CPU fallback is the trap: a run that "passes" on CPU proves nothing (see Test plan). |
| TensorFlow GPU growth | `framework/py/flwr/simulation/ray_transport/utils.py enable_tf_gpu_growth` | Generic `tf.config` calls; e2e-tensorflow pins `tensorflow-cpu`. | Low; no ROCm TF in scope. |
| bitsandbytes (flowertune-llm examples) | `examples/flowertune-llm*/pyproject.toml` pin `bitsandbytes==0.49.1` | The Linux wheel ships `libbitsandbytes_rocm62..rocm71.so` only; a torch built on ROCm 7.2 or newer may find no matching binary. The Windows wheel is CUDA/CPU only. | Medium, but only if Q2 keeps these examples. |

No warp-width, texture, stream or memory-model surface exists. wave32 versus wave64 can only matter through Ray discovery and the ML framework, not Flower.

## Risk list

1. Ray sees 0 AMD GPUs in a pip-only environment (`amdsmi` missing or mismatched with `libamd_smi.so`). Expected on every host without a system ROCm; must be shown, and the working recipe recorded.
2. amd-smi enumerates GPUs the process cannot or should not use (other jobs' GPUs on a shared host, or an APU). Ray's count comes from amd-smi, not from `HIP_VISIBLE_DEVICES`; check that a pre-set `HIP_VISIBLE_DEVICES` limits what the actors get.
3. Silent CPU fallback in the ClientApp: every example falls back to CPU when `torch.cuda.is_available()` is false, so a green run is no evidence. Every GPU run must assert the device.
4. Windows: Ray 2.56 on Windows is upstream-beta and Python 3.11/3.12 only; ROCm torch for Windows (TheRock wheels) on 3.12 must coexist with it. Windows simulation results are evidence, not a gate, until Jeff rules.
5. Upstream moves fast (41 commits in 8 days). Validation describes one `head_sha`; keep the fork mirror current only at round boundaries.
6. Shared fleet hosts: the validator must pin a GPU (`HIP_VISIBLE_DEVICES`) and keep Ray from claiming all CPUs/GPUs (`init-args-num-cpus`, `init-args-num-gpus`).

## File-by-file change list

Default (Q1 option A): no change to Flower source. `moat-port` is the fast-forwarded mirror tip, and `head_sha` is that commit. Changes are made only for a defect that validation demonstrates, each a minimal fix with its own test, recorded in notes.md first.

Under Q1 option B the porter would add, and nothing else:
- `framework/e2e/e2e-pytorch/`: a GPU assertion that is inert on CPU-only CI (for example, when an opt-in environment variable is set, the client and server apps assert `torch.cuda.is_available()` and that the training tensors live on `cuda`), plus a `simulation_next.py` variant that requests `num_gpus > 0` and checks the pool size.
- No docs edits that duplicate #8383 (`how-to-run-simulations*.rst`, `simulate.rst`, the slurm how-to), and no edits to `ray_actor.py`, which #8383 rewrites.

Under option C or D the change list is whatever Jeff rules; the porter must not start them on this plan alone.

## Build commands

No compiled code. Per host, in a job-local venv (never the image's Python):

Linux (wave64 and wave32):
```
python3.12 -m venv "$VENV" && . "$VENV/bin/activate"
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm<N>   # N matching the host ROCm; or TheRock wheels
pip install -e "projects/flower/src/framework[simulation]"                          # editable flwr from the fork checkout
pip install "amdsmi==<the version matching the host's libamd_smi.so>"                # or the system ROCm's own amdsmi package
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
python -c "import ray; ray.init(num_cpus=4); print(ray.cluster_resources())"           # must show GPU >= 1
```
Record which `amdsmi` source worked (PyPI wheel plus `/opt/rocm/lib/libamd_smi.so`, or the system package), and the failure mode without it.

Windows (Python 3.12, TheRock ROCm torch with the `[device-<gfx>]` extra):
```
py -3.12 -m venv %VENV% && %VENV%\Scripts\activate
pip install <TheRock torch wheels for the card>
pip install -e "projects\flower\src\framework[simulation]"
```
Also probe Python 3.13 without the `simulation` extra (ray is not installed there by upstream design).

## Test plan

Flower has no GPU tests; its CI is CPU-only (`framework-test.yml`, `framework-e2e.yml` with `--extra-index-url https://download.pytorch.org/whl/cpu`). So the GPU evidence is authored harness runs on Flower's own e2e projects and examples, each proving GPU execution (a device assertion or a printed `torch.cuda.get_device_name` from inside the ClientApp, plus a `rocm-smi`/`amd-smi` utilisation sample or a kernel-dispatch count).

GPU runs, each gate:
1. Ray discovery: `ray.cluster_resources()["GPU"]` with and without `amdsmi`, and with a pre-set `HIP_VISIBLE_DEVICES`.
2. `framework/e2e/e2e-pytorch` on a ROCm torch: `python simulation.py`, `python simulation_next.py` (upstream CI's two simulation entry points), re-run with client resources `num_gpus = 0.5` and `1.0` (via `flwr federation simulation-config --client-resources-num-gpus` or the TOML), and `../test_superlink.sh e2e-pytorch` (deployment engine). Linux only for the Ray parts.
3. `examples/quickstart-pytorch` with `flwr run .` on the local simulation federation, num-gpus > 0.
4. `framework/e2e/e2e-pytorch-lightning` and `framework/e2e/e2e-opacus` simulation runs (the other torch e2e projects in upstream's matrix).
5. Deployment engine on Windows: `framework/e2e/test_windows.sh` (upstream's own Windows job, numpy app) unchanged, then the same SuperLink plus two SuperNodes flow with `e2e-pytorch` on the ROCm torch, asserting the GPU device.
6. Windows simulation (evidence only, not a gate): Python 3.12, `init-args-num-gpus 1`, `client-resources num-gpus 0.5`, e2e-pytorch `simulation_next.py`.

Non-GPU regression set (must not regress, run on Linux in the same venv): `cd framework && RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 python -m pytest py/flwr -n 8` (218 `*_test.py` files; includes the Ray tests `raybackend_test.py` and `ray_client_proxy_test.py`), and on Windows the same pytest run with failures compared against a CPU-torch baseline on the same host, since upstream CI never runs pytest on Windows.

Validator records for each run: exact command, Python/torch/HIP/Ray/amdsmi versions, pass/fail counts, the GPU-proof line, and wall time.

## Open questions

**Q1 (blocks the porter). What does `moat-port` carry, given Flower needs no code change and #8383 already covers docs and diagnostics?**
- A. Nothing beyond the fast-forwarded mirror. The pipeline validates unmodified upstream on all three gates; the evidence (Radeon, Windows, the `amdsmi` recipe) goes to Jeff, who decides whether and how it reaches #8383 or smahto957. Least overlap; no upstream PR from MOAT unless validation finds a defect. Planner's recommendation.
- B. A small GPU e2e harness only (the e2e-pytorch GPU assertion and `num_gpus > 0` simulation run above), offered upstream separately from #8383 since it touches none of its files. Gives upstream a reusable GPU check, but upstream CI has no GPU runner, so it would run only by hand.
- C. Build on #8383: cut `moat-port` from `smahto/flower_amd` and add Radeon/Windows doc lines and evidence there, delivered to smahto957 or into #8383. Needs Jeff's coordination with the author; inherits #8383's merge state.
- D. Fix the discovery gap in Flower itself (for example, when Ray reports 0 GPUs and the ML framework sees AMD GPUs, pass the count into `ray.init(num_gpus=...)`), or add `amdsmi` to an extra. A behaviour change in `ray_actor.py`/`raybackend.py`, which conflicts with #8383's edits there and is a design call for upstream.

**Q2 (blocks the porter only if Jeff wants wider coverage). Which parts of the monorepo are in the validation surface?** Proposed: the `flwr` framework package (CPU suite plus the GPU runs above), the torch e2e projects (e2e-pytorch, e2e-pytorch-lightning, e2e-opacus), `examples/quickstart-pytorch`, and Windows deployment. Proposed out:
- `framework/cc` C++ SDK and `examples/quickstart-cpp`: CPU-only gRPC client, no GPU code, nothing to port.
- Kotlin/Swift SDKs and the android/ios examples: mobile, no ROCm target.
- The other 37 examples and all 31 `baselines/`: user applications whose GPU behaviour is their ML framework's, not Flower's; 18 baselines and 18 examples call `torch.cuda`, the same pattern as quickstart-pytorch. Covering them would be a large sweep of third-party model stacks.
- The `flowertune-llm*` examples and `benchmarks/flowertune-llm` specifically: bitsandbytes 0.49.1 has ROCm binaries for ROCm 6.2 to 7.1 only and none for Windows, so they likely fail on current ROCm for reasons outside Flower.
- `datasets/` (`flwr-datasets`), `agent/`, `hub/`, `model/`: no GPU surface.
Keeping any of these (for example one LLM fine-tuning example to show a heavier workload) is Jeff's call.

**Q3 (does not block; already deferred by Jeff on 2026-11-11). How Flower should offer Ray simulation on Windows.** This plan only collects the evidence in Test plan item 6.
