# wenet port plan

Planner run 2026-11-20 on linux-gfx942 (fleet job, no GPU used, nothing built). Upstream read at `be6ad97fdb51` (main, 2026-10-26).

**Three decisions are open for jeff (Q1 to Q3 under Open questions). A porter must not start before they are ruled.** Everything below is written so that each answer only adds or removes a named item; the recommended answer is marked.

## Project

- Name: wenet
- Upstream: https://github.com/wenet-e2e/wenet (Apache-2.0), default branch `main`
- Fork: https://github.com/AMD-Ecosystem/wenet, port branch `moat-port`
- ext_type: `python-torch` (a pure-PyTorch Python package plus a C++ LibTorch runtime; no torch extension, no compiled CUDA in the package)

## Existing AMD support

None. Intake found no ROCm/HIP/AMD PR, branch, issue or fork upstream or in ROCm/AMD-Ecosystem; upstream carries vendor runtimes for Kunlun XPU, Horizon BPU, Intel IPEX/OpenVINO and Huawei NPU (`torch-npu` extra, `run_npu.sh`), but nothing for AMD. The README's Install section documents CUDA and NPU only. Decision: improvable -- the work is validating wenet on ROCm PyTorch and adding a documented ROCm path, in the same shape upstream already accepted for NPU. No performance kernel exists, so no port-vs-rewrite question arises.

Delivery vehicle: an ordinary upstream PR. Upstream merges vendor support in-tree (the NPU extra and runtimes above), so it does not prefer linked forks.

## Build classification

- Python package: `setup.py` with `install_requires` torch/torchaudio, `find_packages()`, no `ext_modules`, no `CUDAExtension`. Nothing to hipify; ROCm PyTorch maps `torch.cuda`, `--device cuda`, `nccl` (to RCCL) and `torch.cuda.amp` onto HIP at run time.
- C++ runtime `runtime/libtorch` (symlinks into `runtime/core`): CMake, `option(GPU ...)` in `runtime/libtorch/CMakeLists.txt:16`; `runtime/core/cmake/libtorch.cmake` downloads a pinned LibTorch 2.2.0 zip, the `cu118` one when `GPU=ON`, refuses `GPU` off Linux, and defines `USE_GPU`. Device code in `runtime/core/decoder/torch_asr_model.cc:41-253` is `torch::cuda::is_available()` and `at::kCUDA`, which a ROCm LibTorch serves unchanged. No `.cu` in the runtime.
- `runtime/gpu/`: Triton Inference Server deployment (Docker, model repositories, TensorRT, FasterTransformer, `riva-asrlib-decoder`, onnxruntime-gpu). The only CUDA source in the repo is here: `runtime/gpu/tensorrt/LayerNormPlugin/LayerNormPlugin.{cu,h}`.

## Port strategy

Neither A nor B in the usual sense: there is no CUDA to translate in the shipped package. The port is
1. validation of the Python package on ROCm PyTorch across both wavefront widths and Windows, fixing anything that breaks at run time (none is expected from reading the code);
2. a ROCm build path for the LibTorch runtime's GPU option (Q2);
3. documentation: a ROCm install section beside CUDA and NPU, and an explicit statement of which deployment paths are NVIDIA-only (Q1).

## CUDA surface inventory

| Item | Where | ROCm mapping / verdict |
|---|---|---|
| `torch.cuda.set_device/current_device`, `"cuda" in args.device` | `wenet/utils/train_utils.py:259,454`, `wenet/bin/{recognize,alignment,train}.py`, `wenet/cli/*` | works as is on ROCm torch |
| `dist_backend` default `nccl` | `train_utils.py:183` | RCCL behind the `nccl` name; works as is |
| `torch.cuda.amp.GradScaler/autocast` | `train_utils.py:595,684`, `transducer.py:519`, `recognize.py:278` | works (deprecation warning only, same as CUDA) |
| FSDP, DeepSpeed engines | `wenet/utils/fsdp_utils.py`, `train_utils.py`, `examples/*/whisper/conf/ds_stage*.json` | FSDP is core torch; DeepSpeed upstream supports ROCm (may JIT ops: build in the job venv only) |
| `F.scaled_dot_product_attention` (`use_sdpa`) | `wenet/models/transformer/attention.py`, `firered/attention.py` | ROCm SDPA backends (aotriton/CK flash, mem-efficient, math); numerics test needed |
| `torchaudio.functional.rnnt_loss` (default transducer loss) | `transducer.py:180,495` | GPU kernel inside the ROCm torchaudio wheel; risk R2 |
| k2 (optional: LF-MMI, HLG, pruned RNNT) | `wenet/models/k2/model.py`, `transducer.py:503` | k2 ROCm support merged upstream (k2-fsa/k2#1353, MOAT project `k2`, merged 2026-07-02); install k2 from source per its notes |
| `.cuda()`, `torch.cuda.empty_cache()`, `torch.svd_lowrank` on GPU | `wenet/models/finetune/lora/utils.py:152,266,299` | works as is; exercise it |
| onnxruntime `CUDAExecutionProvider` hard-coded | `wenet/bin/export_onnx_gpu.py:813,966,1089`, `recognize_onnx_gpu.py` | ROCm EP was removed from onnxruntime; the AMD provider is MIGraphX (plugin). Q1 |
| `USE_GPU` / `at::kCUDA` C++ | `runtime/core/decoder/torch_asr_model.cc` | works with a ROCm LibTorch; CMake only fetches CUDA LibTorch. Q2 |
| TensorRT `IPluginV2` LayerNorm (3 kernels, `cub::BlockReduce`) | `runtime/gpu/tensorrt/LayerNormPlugin/` | no ROCm TensorRT API; Q1 (recommended: out of scope) |
| Triton/TensorRT/FasterTransformer/Riva CUDA WFST decoder | `runtime/gpu/**` | NVIDIA-only stack; Q1 |

No warp intrinsics, `warpSize`, textures, raw allocation or streams exist outside `runtime/gpu`, so the wavefront and memory fault classes do not apply to wenet's own code; they can still surface inside torch, torchaudio or k2, which is why both widths are validated.

## Risk list

- R1 (environment, not ROCm): torchaudio 2.9 entered maintenance mode and removed APIs deprecated in 2.8; `load`/`save` now route to TorchCodec. wenet calls `torchaudio.load/save/info`, `torchaudio.sox_effects.apply_effects_tensor`, `torchaudio.compliance.kaldi`, `transforms.Resample` and `torchaudio.functional.forced_align` (`wenet/dataset/processor.py`, `wenet/cli/model.py`, `wenet/utils/ctc_utils.py:118` used by `alignment.py`, tools). forced_align is in the same keep-or-drop list as RNNT. The newest ROCm wheels on download.pytorch.org are torch/torchaudio 2.11 (`rocm7.1`). `requirements.txt` has no upper bound, so this hits upstream's CUDA CI equally. Porter: first check which of these exist in the host's torchaudio. If one is gone, validate on the newest ROCm torch/torchaudio pair that still carries it (e.g. a 2.8 ROCm wheel), state that pair in the docs, and register the API break with `utils/deferred.py add --project wenet` for jeff; do not fold a torchaudio migration into the ROCm PR.
- R2: `torchaudio.functional.rnnt_loss` was marked for removal in 2.9; upstream's 2.9 notes say they are working to keep RNNT. Confirm it exists and runs on GPU in the chosen torchaudio, and compare its loss and gradient against the CPU result. If absent, the transducer default path needs k2 (`enable_k2: true`) on ROCm or an older torchaudio (R1 rule).
- R3: SDPA numerics. `test_attention.py` asserts sdpa vs eager equivalence on CPU; on ROCm the flash/mem-efficient backends differ in rounding. If a GPU run of that comparison needs a looser tolerance, gate it on `torch.version.hip` (never loosen CUDA's) -- and only if the GPU variant is shipped at all (it is a validator harness by default).
- R4: DeepSpeed on ROCm may JIT-compile ops (FusedAdam etc.) at first use; build only inside the job venv, never into the shared image. If the wenet DeepSpeed configs request an op ROCm DeepSpeed lacks, record it, do not patch DeepSpeed.
- R5: multi-GPU DDP/FSDP needs 2 GPUs on one host at once; RCCL collectives take the whole host's GPUs (`cuda-to-rocm` libraries.md). Coordinate with other jobs (dispatcher pins `HIP_VISIBLE_DEVICES`); a single-GPU host records DDP as not run, not as failed.
- R6: the LibTorch runtime pins LibTorch 2.2.0. PyTorch's ROCm LibTorch zips for 2.2.0 are rocm5.x builds without gfx942/gfx1151 code objects, and TorchScript models exported by a newer torch may not load in an older LibTorch. Hence Q2's design uses a local, matching ROCm LibTorch (from the ROCm torch wheel or a LibTorch zip) instead of a new pinned download.
- R7: Windows gate. wenet supports Windows (upstream `windows.yml` builds the LibTorch runtime CPU-only; `setup.py` adds PySoundFile on Windows). On Windows the Python package runs on TheRock torch for gfx1151; torchaudio availability for TheRock Windows must be checked first. The LibTorch runtime's GPU option stays Linux-only (upstream's own `FATAL_ERROR`), so Windows validates the Python package plus the CPU runtime build only, unless jeff rules otherwise.
- R8: pretrained-model downloads (`wenet` CLI pulls from ModelScope/Hugging Face) need network; use a model cached under the job directory, or train a tiny model from `test/resources/dataset` (in tree, 9 MB) for decode tests.

## File-by-file change list

Expected changes, assuming the recommended rulings (Q1 a, Q2 a):

1. `runtime/core/cmake/libtorch.cmake`: when `GPU=ON` and the user supplies a local LibTorch (proposed: honour `Torch_DIR`/`CMAKE_PREFIX_PATH` or a `LIBTORCH_DIR` cache variable, matching how the file already calls `find_package(Torch ... PATHS ${libtorch_SOURCE_DIR})`), use it instead of FetchContent of the cu118 zip. This is vendor-neutral (it also lets CUDA users pick a newer LibTorch); no ROCm-specific branch should be needed because `find_package(Torch)` on a ROCm LibTorch brings the HIP libraries itself. Check the `-D__HIP_NO_HALF_OPERATORS__` flags that ROCm LibTorch's CMake package adds (`cuda-to-rocm` build-cmake-link.md) do not break the host C++, and RUNPATH to the ROCm libs for `decoder_main`. The CPU and CUDA download paths stay byte-identical in behaviour.
2. `runtime/libtorch/README.md` (+ `README_CN.md` if upstream keeps them in step): document the ROCm GPU build (`cmake -DGPU=ON -DCXX11_ABI=ON -DTorch_DIR=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')/Torch ..`, exact form decided by the porter after testing).
3. `README.md` Install section: a ROCm subsection parallel to the NPU one -- ROCm torch/torchaudio wheel index (Linux), TheRock wheels for Windows/gfx1151, the `torch.cuda.is_available()` check (true on ROCm), `--device cuda` usage note, optional k2 build pointer. `docs/python_package.md` GPU notes get one line that ROCm is driven by the same `--device cuda`.
4. Docs note (README deployment pointer or `runtime/gpu/README.md`): the Triton/TensorRT/FasterTransformer/Riva and onnxruntime-gpu paths are NVIDIA-only (Q1 a). Wording must not imply a promise of future AMD support.
5. Python source: none expected. Any change found necessary at run time must be device-generic (no `if hip` branches unless an AMD behaviour forces it) and must keep the CUDA and NPU paths unchanged.

Not changed: `runtime/gpu/**` (including `LayerNormPlugin.cu`), vendor runtimes, examples' `run.sh` scripts.

## Build commands

Python package (each platform, in the job venv, never the shared image):
```
python -m venv $FLEET_VENV   # or the host's job venv
$VENV/bin/python -m pip install --index-url https://download.pytorch.org/whl/rocm7.1 torch torchaudio   # Linux; Windows: TheRock wheels for the gfx target
$VENV/bin/python -m pip install -r requirements.txt   # with torch/torchaudio already satisfied; drop the deepspeed line only if it tries to replace torch
$VENV/bin/python -m pip install -e .
```
(If R1 forces an older pair, use that index, e.g. `rocm6.4` for 2.8.)

k2 (optional, for the k2 features): build from upstream k2 master (ROCm support merged) following `moatlib.py show k2` notes' `## Install as a dependency`, inside the job directory.

LibTorch runtime, Linux GPU (Q2 a):
```
cd runtime/libtorch && mkdir build && cd build
cmake -DGPU=ON -DCXX11_ABI=ON -DTorch_DIR=<ROCm LibTorch>/share/cmake/Torch -DBUILD_TESTING=ON ..
cmake --build . -j $MAX_JOBS
```
CPU runtime (regression, every OS): upstream's `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`, as `runtime.yml`/`windows.yml` do.

CUDA no-regression: the only source change expected is CMake; configure `runtime/libtorch` with `-DGPU=ON` and no local LibTorch on the nvcc fleet host to prove the cu118 download path is unchanged, plus a configure with a CUDA LibTorch given through the new variable. No nvcc compile of wenet code exists to run.

## Test plan

Non-GPU regression set (must stay green on the ROCm torch build, each platform): upstream's suite, `PYTHONPATH=$PWD pytest -q test/` (17 files: dataset/datapipes/processor, tokenizers, attention sdpa equivalence, grad checkpointing, init_model, paraformer, whisper). Record pass/fail/skip counts; network-dependent tests (hugging-face/whisper tokenizers) may need a cached model, say so rather than count them as failures. Runtime: `ctest` in the CPU and GPU runtime builds (`utils_test`, `feature_pipeline_test`, `post_processor_test`, `ctc_prefix_beam_search_test`, openfst tests).

GPU evidence (validator harness kept in notes; real GPU, count dispatches or check `torch.cuda` memory to prove it ran on the device):
1. pytest suite again with the attention/grad-ckpt/init_model tests forced onto the GPU (a small conftest fixture in `agent_space`, not shipped unless Q3 says otherwise): sdpa vs eager on HIP, every model family's forward/backward on GPU.
2. Training: `wenet/bin/train.py --device cuda` for a few hundred steps on `test/resources/dataset` (data.list and shards both) with a conformer and a u2++ config from `examples/aishell/s0/conf`, `--use_amp` fp16 and bf16, engines `torch_ddp` (1 GPU, and 2 GPUs via `torchrun` when two are free), `torch_fsdp`, `deepspeed` (the `ds_stage{1,2,3}.json` configs under `examples/aishell/whisper/conf/` and `examples/wenetspeech/whisper/conf/`). Loss must fall and match a CPU run of the first steps within tolerance.
3. Transducer: a transducer config with torchaudio `rnnt_loss` on GPU (loss/grad vs CPU), and with `enable_k2: true` on k2 ROCm.
4. Decoding: `wenet/bin/recognize.py --device cuda` all modes (ctc_greedy_search, ctc_prefix_beam_search, attention, attention_rescoring) on the trained checkpoint, same hyps as `--device cpu`; `wenet` CLI `transcribe --device cuda` on a pretrained model (cached) for `test/resources/*.wav`; `alignment.py --device cuda`; LoRA fine-tune path (`lora/utils.py`).
5. k2: `wenet/models/k2/model.py` LF-MMI loss and HLG decode on GPU, if a k2 ROCm build is available on the host.
6. LibTorch runtime (Q2 a, Linux): export a TorchScript model with `wenet/bin/export_jit.py`, run `decoder_main` with the GPU build on `test/resources` wavs, results identical to the CPU build; confirm the GPU was used.
7. Windows (gfx1151): items 1, 2 (single GPU, torch_ddp/fsdp only; DeepSpeed and RCCL are not on Windows), 3 (torchaudio path), 4; CPU runtime build and ctest.

Coverage: wave64 (CDNA, e.g. this gfx942 host), wave32 (RDNA Linux), windows (gfx1151). Wavefront width matters only through torch/torchaudio/k2 kernels, but the gates are not waived for that.

## Open questions (for jeff; a porter waits for these)

- **Q1: scope out the NVIDIA-only deployment layer?** `runtime/gpu/` (Triton server, TensorRT including the only `.cu`, FasterTransformer, Riva CUDA WFST decoder) and the onnxruntime-gpu scripts (`export_onnx_gpu.py`, `recognize_onnx_gpu.py`). Options:
  - (a) *recommended*: scope all of it out and say in the docs that these paths are NVIDIA-only. TensorRT, FasterTransformer and Riva have no ROCm API to port to.
  - (b) (a) plus make the two onnx-gpu scripts choose their onnxruntime provider (`MIGraphXExecutionProvider` on AMD, since onnxruntime dropped its ROCm provider) -- a small Python change, but validating it needs onnxruntime with the MIGraphX plugin on each host and the Triton serving half still stays NVIDIA-only.
  - (c) port the Triton deployment to AMD (Triton with an onnxruntime/MIGraphX backend, the TensorRT LayerNorm plugin replaced by a MIGraphX graph). Large and a new deployment stack, not a port; not recommended.
- **Q2: add a ROCm build of the LibTorch C++ runtime's GPU option?** Options:
  - (a) *recommended*: yes, Linux only like upstream's GPU option, via a local LibTorch (change list item 1). Small CMake change plus docs; adds a real C++ GPU deliverable and validation item 6.
  - (b) no: document only the Python package; the runtime components stay CPU paths, scoped out of GPU work with that reason.
  - (c) (a) plus lifting upstream's Linux-only `GPU` restriction for Windows ROCm LibTorch. Not recommended: upstream has no Windows GPU runtime for CUDA either, and ROCm LibTorch on Windows has known link gaps (`cuda-to-rocm` windows.md, the torch_hip.dll import entry).
- **Q3: is a port with no source change in the Python package worth an upstream PR?** If Q2 is (a) the PR is a CMake option plus docs; if (b) it is docs only. Options:
  - (a) *recommended*: proceed; the PR is the ROCm install docs (plus the CMake option), backed by full GPU validation on three platforms, the same value upstream accepted for NPU.
  - (b) proceed with validation but ship only if a run-time fix turns out to be needed; otherwise record the validation and send nothing.
  - (c) stop: a person declines as `not-a-target` (intake's fallback). This is a person's call.

Surface accounting: `surface.json` lists 76 components (the generated CMake floor plus planner-added Python, test, deploy and docs entries). The non-NVIDIA vendor runtimes (BPU, Kunlun XPU, OpenVINO, IPEX, Android ITN) are `scoped_out` with reasons. The Q1 items (`cuda:tensorrt-layernorm-plugin`, `deploy:triton-gpu-runtime`, `python:onnx-gpu-export`) and the Q2 items (runtime libraries, executables, options, ctest targets) are deliberately left unaccounted: the porter records them `covered`, `scoped_out` (with jeff's ruling as the reason) or `owed` once the rulings exist.
