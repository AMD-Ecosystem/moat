# openpi port plan

> **Porter: do not start until Jeff has ruled on D1, D2 and D3 below.** The rulings go in `notes.md` under "Planning (2026-11-20)". If they are not recorded there, stop and say so. This plan gives a recommendation for each one but does not assume the answer.

## Decisions Jeff must make before porting

**D1. Is JAX on the GPU in scope?** JAX is openpi's reference implementation. Every upstream unit test, `scripts/train.py`, `scripts/serve_policy.py` and all released checkpoints use JAX; PyTorch is a later second path (pi0/pi05 only, no pi0-FAST). The JAX pin is `jax[cuda12]==0.5.3`, and **no ROCm build of jax 0.5.3 exists**. ROCm JAX releases jump from 0.5.0 (`jax-rocm60-plugin`, ROCm 6.0, no gfx950 or ROCm 7 runtime) to 0.6.0 (`ROCm/rocm-jax`, the first `jax_rocm7` build; checked 2026-11-20 on PyPI and the rocm-jax releases page).
- **A (recommended): include JAX GPU on a newer jax.** Install `jax==0.7.1` with `jax-rocm7-plugin`/`jax-rocm7-pjrt==0.7.1` (on PyPI; 0.6.2 exists only as GitHub release wheels). jax 0.6 and later need `ml_dtypes>=0.5.0`, which conflicts with openpi's `override-dependencies = ["ml-dtypes==0.4.1", ...]`. jax 0.8 and later need `numpy>=2`, which conflicts with openpi's `numpy<2.0.0`, so 0.7.x is the newest jax that keeps openpi's numpy pin. flax 0.10.2, orbax-checkpoint 0.11.13 and chex 0.1.90 all accept jax 0.7 according to their metadata, but none of that is tested yet. The ROCm path then runs a different jax than upstream's CUDA path, and the evidence would show it. The CUDA pins stay unchanged. The ROCm environment is a documented constraints/override set, the same pattern #1035 uses for torch.
- **B: build jaxlib 0.5.3 and a ROCm plugin from source against ROCm 7.** ROCm/jax has only a `rocm-jaxlib-v0.5.0` branch, and XLA from early 2025 predates gfx950 support. This is a large toolchain project with uncertain results. Not recommended.
- **C: PyTorch-only GPU port, like #1035.** JAX stays on CPU. This is a partial port: it defers the JAX training, JAX serving and JAX test layer. It also leaves the port with **no upstream unit test that touches the GPU**, because every pytest test is JAX. The GPU gate would then be the PyTorch training and inference runs in the test plan below and nothing else.

**D2. How does the port relate to AMD PR #1035?** PR https://github.com/Physical-Intelligence/openpi/pull/1035 (Treemann, AMD; open, `mergeable` true, `mergeStateStatus` BLOCKED waiting on review; head `b39976dd`; base is upstream `main` `215abfb2`, which is also the fork's `main` and the planned commit). It is AMD-official and authoritative, but tested only on 8x MI308X (gfx942), through a Docker image only, with JAX on CPU.
- **a (recommended): stack on #1035.** Cut `moat-port` from upstream `main`, cherry-pick #1035's two commits with Treemann's authorship intact, then add our delta. Validation then also covers #1035 on wave64 (gfx950) and wave32. The upstream delivery has two parts: evidence offered on #1035 (an upstream comment, which needs Jeff's approval of the exact text), and our delta as a follow-up PR after #1035 merges, or offered to Treemann for inclusion.
- **b: independent branch from upstream `main` without #1035.** Our `docs/rocm.md` and Docker files would conflict with #1035, which amounts to competing with an AMD PR. Not recommended (`assess-existing-support`: contribute the differentiator, not a parallel PR).
- **c: evidence only.** Validate #1035 as it stands, report on the PR, and stage no code delta. This loses the `conftest.py` fix (R1), which upstream needs whatever happens.

**D3. Which examples are in scope?** Recommended: `examples/simple_client` (policy-serving smoke test, no robot) is covered, and so is `examples/aloha_sim` if D1=A (MuJoCo sim on the CPU plus a GPU policy server, no hardware). Scoped out: `aloha_real`, `droid` and `ur5`, which need physical robots and contain no GPU code beyond the policy server already covered. `libero` is proposed out too: its Docker image renders through the NVIDIA EGL ICD (`examples/libero/Dockerfile:60`), so making it work is a Mesa/EGL rendering task, not GPU compute. Cutting `libero` and the robot examples makes this a partial port, so it needs Jeff's ruling.

Assumption Jeff can override (not a blocking decision): wave32 Radeon parts have 16-32 GB, below the README's 22.5 GB for LoRA and 70 GB for full fine-tuning. On wave32 the GPU gate is therefore the `debug`/`debug_pi05` configs (dummy-size models), the pytest suite, and inference. The full-size `pi05_libero` fine-tune runs on wave64 (MI355, 288 GB) only.

## Project

- Name: openpi (Physical-Intelligence/openpi): pi0, pi0-FAST and pi0.5 vision-language-action models for robotics, in JAX/Flax with a PyTorch reimplementation of pi0/pi0.5.
- Upstream: https://github.com/Physical-Intelligence/openpi, default branch `main`, planned at `215abfb217db` ("docs(droid): fix config search instruction (#1023)", 2026-08-24; still the tip on 2026-11-20).
- Fork: https://github.com/AMD-Ecosystem/openpi. It has only `main` (= upstream `215abfb2`). The porter creates `moat-port`, and it must become the fork's GitHub default branch (a person's action).
- Windows gate: waived by Jeff on 2026-11-12 (upstream is Ubuntu-only, and ROCm JAX has no Windows build). Remaining gates: wave64 and wave32.
- Planned on linux-gfx950, with no GPU used. There is no lead platform.

## Existing AMD support

- **PR #1035** (AMD-official, authoritative, incomplete): `docs/rocm.md`, a README link, `docs/docker.md` note, `scripts/docker/train_rocm.Dockerfile` (base `rocm/primus:v26.4`, ROCm 7.14, PyTorch 2.12, gfx942), `scripts/docker/compose_rocm.yml`. PyTorch training only, JAX forced to CPU (`JAX_PLATFORMS=cpu`), `numpy` kept at 2.x against the pyproject pin, openpi/lerobot installed `--no-deps` under a constraints file. It documents a non-ROCm bug: `train_pytorch.py --overwrite` races `shutil.rmtree` across DDP ranks. One author ping, no maintainer response since 2026-09-09.
- PRs #858/#865 (sunxxuns, MI350 kernel benchmark): closed, not merged, no code to reuse.
- skariapaul/openpi-rocm: a community Radeon fork. Non-authoritative, a hint at most.
- Upstream does not link platform forks, so an upstream PR is a valid vehicle. Upstream's merge cadence is slow (last commit 2026-08-24), so the fork may hold the deliverable for a long time.

Decision: validate and extend #1035 (D2a). Our differentiators are wave32 and gfx950 coverage, JAX on the GPU (D1A), a non-Docker install recipe for hosts the primus image does not serve, and the shared-code bug R1.

There are no performance kernels. openpi has no CUDA, Triton or Pallas source. All GPU work goes through XLA (JAX) or PyTorch eager/Inductor ops. There is no mechanical-versus-native rewrite question. Performance is whatever ROCm JAX/XLA and PyTorch deliver, and the plan records throughput rather than tuning it.

## Build classification

`python-jax-torch` (set with `set-ext-type`). This is a pure Python package with a hatchling build backend and a uv workspace (`pyproject.toml`, `packages/openpi-client`). Evidence:
- No `.cu`/`.cuh`/`.cpp` outside `third_party/` (`find`), no `setup.py`, no `CMakeLists.txt`, no `torch.utils.cpp_extension`, no `cpp_extension.load`.
- GPU selection is entirely in dependency pins: `pyproject.toml:18` `jax[cuda12]==0.5.3`, `:29` `torch==2.7.1` (resolves to CUDA wheels), `:55` `pynvml>=12.0.0` (dev). `[tool.uv] override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74"]`.
- `uv.lock` pins the CUDA wheels, so `uv sync` cannot produce a ROCm environment. This is why #1035 is Docker plus constraints.

## Port strategy

Neither Strategy A nor B. Nothing is compiled, and the porting work is the environment plus the few CUDA assumptions in Python:
1. **Environment recipe:** a ROCm torch wheel (or base image) plus pip installs under a constraints file, exactly #1035's pattern. With D1A it also adds `jax==0.7.1` + `jax-rocm7-plugin`/`jax-rocm7-pjrt==0.7.1`, overrides `ml-dtypes>=0.5` and a matching `tensorstore`, and drops `JAX_PLATFORMS=cpu`. Upstream's `pyproject.toml`/`uv.lock` CUDA pins stay unchanged, so the CUDA path is untouched.
2. **Python fixes in shared code** for CUDA-only assumptions (R1, R2 below). Each one keeps CUDA behaviour identical.
3. **Docs:** extend #1035's `docs/rocm.md` with the JAX section, the non-Docker recipe and the wave32 notes.

torch version: upstream pins `torch==2.7.1`, whose ROCm wheels go up to `+rocm6.3` (download.pytorch.org, checked 2026-11-20) and predate gfx950 support. ROCm 7 wheels start at `torch 2.10.0+rocm7.0`. The ROCm recipe uses the newest ROCm torch the host's ROCm supports, as #1035 does (2.12). It does not change upstream's CUDA pin. `transformers==4.53.2` stays pinned, because `transformers_replace/` patches files of that exact version.

## Port surface

Machine-checked accounting: `projects/openpi/surface.json`. The scanner's floor is empty: there is no `.cu`, no CMake and no `setup.py` extension. That is the mumax3 case (a nearly empty census for a real port), so every entry below is planner-added. Entries whose fate depends on a decision are marked in their `note`. The porter moves each one to `covered`, `scoped_out` (with Jeff's reason) or `owed`.

| id | what | plan |
|---|---|---|
| `path:pytorch-training` | `scripts/train_pytorch.py`, `models_pytorch/` (pi0/pi05), single GPU and torchrun DDP over RCCL | cover (wave64 full-size, wave32 debug configs) |
| `path:pytorch-inference` | `policy_config.create_trained_policy` with a PyTorch checkpoint, `torch.compile(mode="max-autotune")` on `sample_actions` | cover |
| `tool:convert-jax-to-pytorch` | `examples/convert_jax_model_to_pytorch.py` | cover (needed to get PyTorch weights) |
| `path:jax-training` | `scripts/train.py` (pi0, pi0-FAST, pi05, LoRA), FSDP sharding over `jax.devices()` | D1: cover (A/B) or scoped out (C) |
| `path:jax-inference-serving` | `scripts/serve_policy.py`, `serving/websocket_policy_server.py`, JAX checkpoints | D1 |
| `test:pytest-suite` | the 11 `*_test.py` files (`pyproject` testpaths `src scripts packages`, `-m "not manual"`); all JAX except the client tests | D1A: on the GPU. D1C: CPU only, recorded as non-GPU evidence |
| `test:openpi-client` | `packages/openpi-client` `image_tools_test`, `msgpack_numpy_test` (CPU) | cover (no regression) |
| `fix:conftest-nvml-gpu-detect` | `src/openpi/conftest.py` decides "no GPU" from `pynvml.nvmlInit()` (R1) | cover (needed under every D1 option) |
| `build:rocm-env-recipe` | constraints/override set for ROCm torch (+ ROCm JAX under D1A), outside `uv.lock` | cover |
| `build:docker-train-rocm` | #1035's `train_rocm.Dockerfile` + `compose_rocm.yml` | cover (validate on gfx950; D2) |
| `build:docker-serve-policy` | `scripts/docker/serve_policy.Dockerfile` + `compose.yml` (`nvidia/cuda` base, `uv sync`, NVIDIA device reservation) | D1A: ROCm serve variant. D1C: scoped out |
| `doc:rocm` | `docs/rocm.md`, README and `docs/docker.md` links (from #1035) | cover and extend |
| `example:simple_client` | `examples/simple_client` (Dockerfile, compose with NVIDIA reservation, `main.py`) | cover (D3) |
| `example:aloha_sim` | `examples/aloha_sim` | D3 (recommended cover if D1A) |
| `example:libero` | `examples/libero` (NVIDIA EGL ICD in its Dockerfile) | D3 (proposed out) |
| `example:aloha_real`, `example:droid`, `example:ur5` | real-robot clients | D3 (proposed out: needs hardware) |
| `tool:compute-norm-stats` | `scripts/compute_norm_stats.py` (CPU data pass) | cover (prerequisite for training runs) |
| `ci:github-actions` | `.github/workflows/test.yml` (self-hosted `openpi-verylarge`), `pre-commit.yml` | scoped out: no fork CI (AGENTS.md Testing) |
| `vendored:transformers-replace` | `models_pytorch/transformers_replace/` copied over transformers 4.53.2 | cover (it is part of the PyTorch path; no CUDA in it) |
| `third_party:aloha`, `third_party:libero` | submodules, example-only | follow D3 |

## CUDA surface inventory

There are no kernels, intrinsics, textures or CUDA library calls in openpi source. The complete list of CUDA- or NVIDIA-specific code (grep for `cuda|nvidia|nvml|cudnn|nccl|tf32|XLA_` outside `third_party/` and `uv.lock`):

| site | what | ROCm |
|---|---|---|
| `pyproject.toml:18` | `jax[cuda12]==0.5.3` | no ROCm build at 0.5.3 (D1) |
| `pyproject.toml:29` | `torch==2.7.1` | CUDA wheel by default; ROCm recipe substitutes a ROCm wheel |
| `pyproject.toml:55`, `src/openpi/conftest.py:3-17` | `pynvml` used to detect a GPU | **R1**: on AMD, `nvmlInit` raises `NVMLError_LibraryNotFound`, so `JAX_PLATFORMS=cpu` is forced |
| `scripts/train_pytorch.py:98` | `backend="nccl" if torch.cuda.is_available()` | RCCL under the `nccl` name: works |
| `scripts/train_pytorch.py:106-122,212-305,388` | `torch.cuda.*` device, seeding, `empty_cache`, `memory_*` stats | HIP torch maps `torch.cuda`: works |
| `scripts/train_pytorch.py:269,429` | `PYTORCH_CUDA_ALLOC_CONF=...expandable_segments:True` | read by HIP torch; `expandable_segments` is supported on recent ROCm torch (check the warning) |
| `scripts/train_pytorch.py:425-427` | `cudnn.benchmark`, `allow_tf32` (8+ GPU branch only) | MIOpen benchmark; TF32 flags are inert on most AMD parts. Harmless |
| `src/openpi/models/pi0_config.py:35`, `models_pytorch/pi0_pytorch.py:112-113` | `torch.compile(mode="max-autotune")` default for inference | Inductor on ROCm; max-autotune tuning time and Triton-on-ROCm correctness are **R3** |
| `models_pytorch/pi0_pytorch.py:392,448` | forces `_attn_implementation="eager"` | eager attention: no flash-attn dependency |
| `transformers_replace/.../modeling_*.py` | `_supports_flash_attn_2/3`, `sdpa` | flags only; openpi selects eager |
| `src/openpi/training/data_loader.py:482-483` | `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `ALLOCATOR=platform` | honoured by the ROCm PJRT plugin |
| `scripts/train.py:203` | JAX compilation cache dir | portable |
| `scripts/docker/serve_policy.Dockerfile:10`, `scripts/docker/compose.yml:22-29`, `examples/*/compose.yml` | `nvidia/cuda` base image, `driver: nvidia` device reservation | ROCm: `/dev/kfd`, `/dev/dri`, `video`/`render` groups (as #1035's compose) |
| `examples/libero/Dockerfile:9,60` | CUDA base and NVIDIA EGL ICD | D3 |
| `examples/droid/compute_droid_nonidle_ranges.py:25` | `CUDA_VISIBLE_DEVICES=""` to force CPU | HIP honours `CUDA_VISIBLE_DEVICES` (also `HIP_VISIBLE_DEVICES`); for TF it hides GPUs either way. Harmless |
| `src/openpi/training/droid_rlds_dataset.py:59` | `tf.config.set_visible_devices([], "GPU")` | TF stays CPU (`tensorflow-cpu`). Harmless |

## Risk list

- **R1 (certain, shared code): NVML-based GPU detection in `conftest.py`.** On any non-NVIDIA host, `pynvml.nvmlInit()` fails and the test session sets `JAX_PLATFORMS=cpu`. With ROCm JAX installed, the whole suite would silently run on the CPU and still pass, which is false GPU evidence. Fix: decide "no GPU" only when no accelerator is present. For example, keep the NVML probe and fall back to checking whether the ROCm runtime sees a device (`/dev/kfd` exists and `amdsmi`/`rocm_smi` is importable), or ask jax for `gpu` devices after setting nothing. The CUDA path must behave exactly as before. The validator must show `jax.default_backend() == "gpu"` (or the ROCm platform name) inside a pytest run, not only from a shell.
- **R2 (likely): dependency pins under jax 0.7 (D1A).** The `ml-dtypes==0.4.1` and `tensorstore==0.1.74` overrides block jax >= 0.6. flax 0.10.2/orbax 0.11.13 removals or deprecations under jax 0.7 are untested, and `jax.experimental` use in `scripts/train.py` may break. First porter step: build the environment and run `pytest -m "not manual"` before changing any code. Keep any API fix version-neutral (it must still work on 0.5.3), and record versions exactly.
- **R3: `torch.compile(mode="max-autotune")` on ROCm.** Inductor autotuning on ROCm can take many minutes and has had Triton correctness issues on wave32. The fallback for validation is `pytorch_compile_mode=None` or `"default"` as a config override. Record both, and do not change upstream's default unless a ROCm-gated failure is proven.
- **R4: numerics.** bf16 training is a loss curve, so compare JAX-GPU against JAX-CPU (same jax build) on the `debug` configs, and PyTorch against JAX on a converted checkpoint (`convert_jax_model_to_pytorch.py`, compare `sample_actions` outputs with a fixed seed and noise). Tolerances belong to the test, not to a ROCm gate, unless a hardware difference is shown.
- **R5: XLA on ROCm (D1A).** XLA GPU codegen for gfx950/gfx1201 at jax 0.7.1 needs confirming. Check that the plugin's supported-arch list includes the host gfx before blaming openpi. FSDP sharding (`training/sharding.py`) on multiple GPUs goes over RCCL collectives inside XLA.
- **R6: numpy pin.** Upstream requires `numpy<2`. #1035 keeps numpy 2.x because the primus image's extensions need it. A non-Docker recipe with a pip ROCm torch wheel can keep numpy 1.26. Pick one per recipe and record it. jax 0.7.1 accepts both.
- **R7: primus base image arch coverage.** `rocm/primus:v26.4` is described in #1035 as gfx942 PyTorch. Confirm it carries gfx950 before running #1035's Docker on MI355, and record it if it does not (a finding for #1035, not a reason to drop gfx950). It is not expected to serve wave32 Radeon, which is why the non-Docker recipe exists.
- **R8 (not ROCm): the `--overwrite` race under DDP** that #1035 documents. Do not fix it in this port unless Jeff asks. It is a separate upstream bug, so record it as a candidate deferral.
- Wavefront, textures, pitch and rule-of-five: not applicable (no device code).

## File-by-file change list (assuming D1A, D2a, D3 recommended)

1. Cherry-pick #1035 `f766c2a801` and `b39976dd40` (authorship preserved).
2. `src/openpi/conftest.py`: GPU detection that works without NVML (R1). Shared code, no `[ROCm]` prefix if the fix is vendor-neutral.
3. `docs/rocm.md`: add a "JAX on AMD GPUs" section (versions, the override set and why), a non-Docker venv recipe (ROCm torch wheel index + constraints + jax-rocm plugin), wave32 memory guidance, and a tested-hardware line (gfx950, the wave32 gfx, ROCm/torch/jax versions).
4. `scripts/docker/train_rocm.Dockerfile` (from #1035): optionally install the ROCm JAX plugin instead of the CPU jaxlib and drop `JAX_PLATFORMS=cpu`. Alternatively leave #1035's file alone and add JAX in a separate serve image. The porter chooses whichever keeps #1035's diff intact. Preferring a separate file keeps #1035 reviewable as-is.
5. `scripts/docker/serve_policy_rocm.Dockerfile` + a `compose_rocm` service: the JAX policy server on ROCm (D1A).
6. `examples/simple_client/compose.yml` / `examples/aloha_sim/compose.yml`: no edit if the ROCm compose for the server lives under `scripts/docker/`. Document instead.
7. Any version-neutral API fix R2 turns up, each as its own commit.
Nothing in `pyproject.toml`/`uv.lock` changes.

## Build commands

There is nothing to compile. Environment setup (record the exact versions resolved):

```bash
git clone --recurse-submodules https://github.com/AMD-Ecosystem/openpi && cd openpi && git checkout moat-port
python3.11 -m venv .venv-rocm && . .venv-rocm/bin/activate
# ROCm torch matching the host ROCm (example: 7.x index)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1
pip freeze | grep -E '^(torch|torchvision|triton|numpy|pillow)==' > constraints.txt
# D1A: ROCm JAX
pip install -c constraints.txt "jax==0.7.1" "jaxlib==0.7.1" "jax-rocm7-plugin==0.7.1" "jax-rocm7-pjrt==0.7.1" "ml-dtypes>=0.5.0"
# remaining deps per docs/rocm.md, then:
GIT_LFS_SKIP_SMUDGE=1 pip install --no-deps -c constraints.txt -e ./packages/openpi-client -e . "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
cp -r src/openpi/models_pytorch/transformers_replace/* "$(python -c 'import os,transformers;print(os.path.dirname(transformers.__file__))')/"
python -c "import jax, torch; print(jax.devices(), torch.cuda.get_device_name(0))"
```

Docker (#1035) on wave64: `docker compose -f scripts/docker/compose_rocm.yml up -d --build`.

Fleet workers have no docker daemon of their own in some containers. If docker is unavailable, the venv recipe is the gate, and the Docker files are recorded as reviewed but not run on that host.

## Test plan

GPU gate, per platform (wrap each step in `utils/timeit.sh openpi test --`):
1. `pytest --strict-markers -m "not manual" src scripts packages`, with a check that JAX is on the GPU inside the session (R1). D1A: on the GPU. Record pass/fail/skip counts. `data_loader_test::test_with_real_dataset` downloads `lerobot/aloha_sim_transfer_cube_human` (HF access needed).
2. JAX training (D1A): `python scripts/train.py debug --exp-name=rocm --overwrite` and `debug_pi05`; then `debug_restore`. Record the loss trajectory against a JAX-CPU run of the same build.
3. PyTorch training: `python scripts/train_pytorch.py debug --exp_name rocm` and `debug_pi05`, single GPU; then `torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py debug --exp_name rocm_ddp` where two GPUs are free (pin with `HIP_VISIBLE_DEVICES`; this is shared hardware).
4. wave64 only: the `pi05_libero` PyTorch fine-tune as #1035 did (convert `pi05_base` with `examples/convert_jax_model_to_pytorch.py`, `compute_norm_stats.py`, then N steps). Under D1A, also the JAX `pi05_libero` for N steps. Record step time and peak memory.
5. Inference: `scripts/serve_policy.py --env ALOHA_SIM` (JAX checkpoint, D1A) and a PyTorch checkpoint, then `examples/simple_client/main.py --env ALOHA_SIM` against each. Under D3, also `examples/aloha_sim` end to end. Check JAX-vs-PyTorch action agreement on the converted checkpoint (R4).
6. `torch.compile` default (`max-autotune`) inference once, with time recorded. If it fails, fall back to `pytorch_compile_mode=None` and record the failure (R3).

Non-GPU regression set: the `openpi-client` tests, and `tokenizer_test`, `normalize_test`, `image_tools_test`, `transforms_test`, `download_test` (CPU, must pass unchanged). The CUDA path is not validated on NVIDIA here because no source is compiled. The conftest change must be read for an unchanged CUDA outcome (review), and the CUDA install path (`uv sync`) must still lock: `uv lock --check` on the branch.

## Open questions

D1, D2 and D3 above are the blocking ones. Non-blocking:
- Whether to offer the `conftest.py` fix (R1) upstream on its own ahead of #1035. It is vendor-neutral and small. That is Jeff's call at publication time.
- R8 (`--overwrite` race): deferral candidate; not registered by the planner.
