# Isaac-GR00T notes

## Intake screen (2026-11-12)

Validation project (lane `validate`, seed `amd-gap-matrix`, NVIDIA-owned upstream), taken up per jeff's 2026-11-12 ruling on the third-party "AMD AI software gap" list.

- Upstream: https://github.com/NVIDIA/Isaac-GR00T, default branch `main`, very active (pushed 2026-11-10), not archived. Current model GR00T N1.7 (Qwen3-VL backbone + flow-matching action head).
- Licence: code Apache-2.0 (tier 1). Model weights (Hugging Face `nvidia/GR00T-N1.7-*`) under the NVIDIA Open Model License; the README calls it commercially usable. The weights are a runtime download, not part of the tree, and validation only uses them, so this bears on use, not on contributing.
- `utils/licenses.py scan-nvidia` on a depth-1 clone: one hit, `ATTRIBUTIONS.md`. Read it: it is the third-party attribution file, and the hit is the reproduced NVIDIA TensorRT Software License Agreement (TensorRT is a pip dependency). It licenses TensorRT, not any file in this repo; no source file carries proprietary text. Still a scan hit, so it needs a person's sign-off before anything is offered upstream.
- Submodules (simulation benchmarks for eval only): `external_dependencies/LIBERO` (MIT), `SimplerEnv` (squarefk fork, MIT), `robocasa` (squarefk fork, MIT licence text; GitHub reports NOASSERTION), `robocasa-gr1-tabletop-tasks` (MIT licence text, NOASSERTION on GitHub). Not cloned; licence files read via the API.
- CUDA surface: none of its own (no `.cu`, no `CUDAExtension`). PyTorch 2.9 + transformers 4.57; `gr00t/model/modules/qwen3_backbone.py` picks `flash_attention_2` when `flash_attn` imports and falls back to `sdpa` otherwise. DeepSpeed for fine-tuning. TensorRT only in `scripts/deployment` (ONNX export + TRT engine for the DiT / action head).
- Packaging is the real gap: `pyproject.toml` hard-depends on `flash-attn==2.8.3` (Linux, pinned cu12 wheels via `[tool.uv.sources]`), `tensorrt-cu12` + `tensorrt-cu12-libs` on every x86_64 platform, `deepspeed==0.17.6`, and torch from a `pytorch-cu128` index; `[tool.uv] required-environments` lists Linux x86_64 / aarch64 only. A ROCm install today means bypassing the lock file.
- Tests: pytest with markers `gpu`, `multigpu`, `edge_device`. 15 files carry `gpu`, e.g. `tests/gr00t/policy/test_gr00t_policy_gpu.py`, `tests/gr00t/experiment/test_experiment_run.py`, `tests/getting_started/test_finetune_new_embodiment_md.py`, `tests/examples/test_libero.py` / `test_so100.py` / `test_droid.py`, and TensorRT-only `tests/scripts/deployment/test_build_tensorrt_engine.py`, `test_trt_pipeline.py`.

## Duplicate effort

- No open or closed upstream PR or issue about ROCm/AMD (searched 2026-11-12; the two "amd" hits are unrelated CUDA install issues). No fork under ROCm or AMD-Ecosystem.
- No AMD/ROCm mention in upstream docs.

## Dependencies

- Hard: `flash-attention` (MOAT project, intake recommends decline `already-supported`). Recorded with `set-deps`. The code falls back to SDPA, so this is a packaging dependency more than a runtime one.
- Optional: DeepSpeed (ROCm/DeepSpeed exists, AMD-maintained), TensorRT (no ROCm equivalent), torchcodec (needs a ROCm-compatible FFmpeg 4-7 build; CPU decode is fine).

## Validation scope

- **Delivery route:** fork-only by default. NVIDIA-owned upstream (the `nvidia-org` case: no upstream PR expected); the fork's `moat-port` carries a ROCm install path (a ROCm pyproject / uv source or requirements, TensorRT and flash-attn made optional) and any runtime fixes. A small upstream PR making TensorRT optional on non-deployment installs is conceivable, but that is jeff's call.
- **GPU gate:** `pytest -m gpu` minus the TensorRT deployment tests: `tests/gr00t/policy/test_gr00t_policy_gpu.py` (policy load + inference against the released N1.7 checkpoint), `tests/gr00t/experiment/test_experiment_run.py` and `tests/getting_started/test_finetune_new_embodiment_md.py` (short fine-tune on `demo_data`), `tests/scripts/deployment/test_standalone_inference.py` (PyTorch path), plus `pytest -m "not gpu"` must stay green. Simulator example tests (LIBERO, SimplerEnv, RoboCasa) are optional evidence: they need MuJoCo / ManiSkill rendering stacks, not ROCm compute.
- **CUDA-only dependencies:** TensorRT (deployment engine, `test_build_tensorrt_engine.py`, `test_trt_pipeline.py`): gap, no ROCm substitute in tree; MIGraphX would be new work; exclude. flash-attn 2.8.3 cu12 wheel: substitute upstream flash-attn ROCm (CK/Triton) build or SDPA fallback. DeepSpeed 0.17.6: ROCm builds exist. Jetson / Spark / Thor scripts: NVIDIA hardware only, out of scope.
- **Windows:** no upstream Windows story (`required-environments` is Linux only; deepspeed, flash-attn and torchcodec markers are Linux-only). Inference through SDPA might work on Windows ROCm torch, but fine-tuning (16 GB+ inference, 40 GB+ fine-tune) and the sim stacks do not. Suggest a `windows` gate waiver; only jeff approves it.
- **Risks:** packaging (TensorRT hard dep on x86_64 even for training users); Qwen3-VL in bf16 under SDPA vs flash-attn numerics against the checkpoint-based test tolerances; torchcodec/FFmpeg on ROCm images; 16 GB+ inference fits Radeon 24 GB parts, fine-tune wants Instinct.
