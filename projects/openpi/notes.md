# openpi notes

## Intake screen (2026-11-12)

Origin: Jeff Daily's 2026-11-12 ruling on a third-party "AMD AI software gap" list -- validation project (JAX and PyTorch, no custom kernels).

- Upstream: https://github.com/Physical-Intelligence/openpi, default branch `main`, pushed 2026-08-24, not archived.
- Licence: Apache-2.0 (LICENSE), tier 1. `scan-nvidia` on a depth-1 clone: clean. Submodules: `third_party/aloha` (Physical-Intelligence/aloha, MIT) and `third_party/libero` (Lifelong-Robot-Learning/LIBERO, MIT) -- both licensed, used only by examples.
- CUDA surface: no `.cu`/`.cuh`. JAX/Flax (`jax[cuda12]==0.5.3`, `flax==0.10.2`) and PyTorch (`torch==2.7.1`) model implementations of pi0 / pi0-FAST / pi0.5.
- Duplicate effort: **AMD PR #1035** (Treemann, AMD; open since 2026-09-02, mergeable_state `blocked` on review, one author ping, no maintainer response) "AMD Official Support: Add ROCm Docker support for PyTorch training on AMD GPUs": docs/rocm.md, README link, `scripts/docker/train_rocm.Dockerfile`, `compose_rocm.yml`; PyTorch training only, Docker only, tested on 8x MI308X; JAX GPU and policy serving explicitly out of scope. Back it with validation evidence rather than compete. Earlier AMD-side PRs #858/#865 (MI350 kernel benchmark) were closed. Community: skariapaul/openpi-rocm (Radeon, non-authoritative). No AMD-Ecosystem or ROCm fork.

## Validation scope

- Delivery route: back PR #1035 with independent validation evidence (wave64 and wave32), then any residual fixes as follow-up upstream PRs; the JAX path (out of #1035's scope) is the gap MOAT can add. Fork holds anything upstream will not take.
- GPU gate: upstream pytest suite (`src/openpi/models/{model,pi0,lora}_test.py`, `policies/policy_test.py`, `scripts/train_test.py`, plus the openpi-client tests; CI workflow `test.yml`), then `scripts/train_pytorch.py debug` and `pi05_libero` training (single GPU and torchrun DDP) as #1035 did, and `examples/simple_client` policy-serving inference.
- CUDA-only dependencies: `jax[cuda12]==0.5.3` pinned in `pyproject.toml` -- ROCm substitute is the `jax-rocm` plugin (jax-rocm7-plugin / jax-rocm-pjrt, built for matching jax versions); jax 0.5.3 pin vs available ROCm plugin versions is the main JAX risk. `torch==2.7.1` resolves to CUDA wheels; needs the ROCm torch index (this is why #1035 went Docker-only). `pynvml` in the dev group (NVML only; harmless if unused). `orbax-checkpoint`, `transformers==4.53.2` are portable.
- Windows: README says "tested with Ubuntu 22.04, we do not currently support other operating systems"; JAX has no ROCm Windows build. Suggest a windows waiver; only Jeff approves one.
- Existing AMD work to back: AMD PR #1035 (Docker + docs, PyTorch training).

## Planning (2026-11-20, planner on linux-gfx950, no GPU used)

**Jeff must rule on three questions before a porter starts.** The full options and evidence are in `plan.md`, "Decisions Jeff must make before porting". Porter: if no ruling is recorded below, stop.

- **D1. JAX on the GPU: in scope or not?** No ROCm build of the pinned `jax==0.5.3` exists. ROCm JAX goes from 0.5.0 (ROCm 6.0, no gfx950) straight to 0.6.0 (first `jax_rocm7`). Options: **A** (recommended) include it on `jax==0.7.1` + `jax-rocm7-plugin`/`-pjrt` 0.7.1 from PyPI. This needs the `ml-dtypes==0.4.1` override lifted to `>=0.5.0`; 0.7.x is the newest jax that keeps openpi's `numpy<2`. The CUDA pins stay untouched, and the ROCm path is a documented override set. **B** source-build jaxlib 0.5.3 + a ROCm plugin against ROCm 7 (not recommended). **C** PyTorch-only like #1035, with JAX on CPU. C is a partial port, and it leaves no upstream unit test on the GPU (every pytest test is JAX).
- **D2. Relationship to AMD PR #1035** (https://github.com/Physical-Intelligence/openpi/pull/1035, open, MERGEABLE, BLOCKED on review, head `b39976dd`, base = upstream/fork `main` `215abfb2`). Options: **a** (recommended) `moat-port` = upstream `main` + #1035's two commits (Treemann's authorship kept) + our delta. Evidence would be offered on #1035, which is an upstream comment needing Jeff's approval of the text, and the delta follows as a separate PR after #1035 merges. **b** an independent branch without #1035 (conflicts with an AMD PR). **c** evidence only, no code delta (loses the conftest fix).
- **D3. Examples.** Recommended: cover `simple_client`, and `aloha_sim` if D1=A. Scope out `libero` (NVIDIA EGL ICD rendering) and the real-robot `aloha_real`, `droid` and `ur5`. Cutting these makes it a partial port.
- Assumption Jeff can override: on wave32 (16-32 GB) the GPU gate is the debug configs, pytest and inference. The full-size `pi05_libero` fine-tune runs on wave64 only.

Rulings: (none recorded yet)

Findings (all read-only; commands below are reproducible):
- Upstream `main` is still `215abfb217db` (2026-08-24), the same as the fork's `main`. The fork has no `moat-port` yet.
- No CUDA, Triton or Pallas source. Build type recorded as `python-jax-torch`. `surface.py generate` finds an empty floor, so all 23 surface components are planner-added. One is already scoped out (fork CI).
- **conftest.py bug (shared code, needed under every option):** `src/openpi/conftest.py` sets `JAX_PLATFORMS=cpu` whenever `pynvml.nvmlInit()` fails. That always happens on AMD, so the pytest suite would run on the CPU and pass even with ROCm JAX installed. The validator must show the JAX backend is the GPU inside the pytest session.
- Version availability (checked 2026-11-20): PyPI `jax-rocm60-plugin` has {0.4.35, 0.5.0}. `jax-rocm7-plugin` starts at 0.7.1 (up to 0.12.1). ROCm/rocm-jax GitHub releases start at 0.6.0. ROCm/jax has a `rocm-jaxlib-v0.5.0` branch and no 0.5.3. jax metadata: 0.5.3 needs `ml_dtypes>=0.4.0`, 0.6.2/0.7.1 need `>=0.5.0`, and 0.8.2 needs `numpy>=2.0`. flax 0.10.2 and chex 0.1.90 accept `jax>=0.4.27`; orbax-checkpoint 0.11.13 accepts `jax>=0.5.0`.
- torch: `torch==2.7.1` ROCm wheels exist only as `+rocm6.2.4`/`+rocm6.3`. ROCm 7 wheels start at `2.10.0+rocm7.0` (download.pytorch.org index). The ROCm recipe uses a ROCm 7 torch, as #1035 does (2.12 from the `rocm/primus:v26.4` image). Upstream's CUDA pin is not changed.
- #1035 is tested on 8x MI308X (gfx942) only. Whether its `rocm/primus:v26.4` base carries gfx950 has not been checked yet (plan R7). It does not serve wave32 Radeon, which is the reason for a non-Docker recipe.
- PyTorch path CUDA-isms (`torch.cuda.*`, `backend="nccl"`, `PYTORCH_CUDA_ALLOC_CONF`, `cudnn.benchmark`/`allow_tf32`) all map under HIP torch. Attention is forced to `eager`, so there is no flash-attn dependency. `torch.compile(mode="max-autotune")` is the inference default (risk R3).
- #1035 also documents a non-ROCm `train_pytorch.py --overwrite` DDP race. It is a deferral candidate (R8) and not registered: the planner records it here only.

Commands used:
```bash
gh repo clone Physical-Intelligence/openpi projects/openpi/src -- --depth=1
gh pr view 1035 --repo Physical-Intelligence/openpi --json state,mergeStateStatus,mergeable,headRefOid,files,commits
gh pr diff 1035 --repo Physical-Intelligence/openpi
gh release list --repo ROCm/rocm-jax --limit 40
curl -s https://pypi.org/pypi/jax-rocm7-plugin/json   # and jax-rocm60-plugin, jax/<ver>, flax, orbax-checkpoint, chex
curl -s https://download.pytorch.org/whl/torch/ | grep -o 'torch-2\.7\.1+rocm[0-9.]*-cp311[^"#<]*'
grep -rn -i -E "cuda|nvidia|nvml|cudnn|nccl|tf32|XLA_" --include=*.py --include=*.toml --include=*.yml --include=Dockerfile* . | grep -v third_party
python3 utils/surface.py generate openpi && python3 utils/surface.py check openpi
```
