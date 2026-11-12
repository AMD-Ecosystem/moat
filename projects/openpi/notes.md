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
