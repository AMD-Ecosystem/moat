---
name: cuda-to-rocm
description: Port a CUDA project to ROCm/HIP for AMD GPUs and keep the port working. Covers strategy selection (CMake HIP language or PyTorch extension), the faults where AMD is strict and CUDA is lenient, wave32/wave64 wavefront portability, CUDA library substitutions and their gaps (cuBLAS, cuSOLVER, cuDNN, cuSPARSE, Thrust, NCCL, RAPIDS, nvJPEG, NVDEC), ROCm build and link problems on Linux and Windows (clang-cl, lld-link, DLL loading, the pip-installed ROCm SDK), and the validation bar. Use when planning, implementing, reviewing or validating a CUDA-to-HIP port, when a HIP build fails to compile or link, when it compiles but misbehaves at runtime, when a CUDA library has no obvious ROCm equivalent, or when syncing a ROCm port with upstream changes.
---

# Codex adapter

Before taking any task action, read `.claude/skills/cuda-to-rocm/SKILL.md` completely and follow it as the canonical skill. Resolve every relative reference against `.claude/skills/cuda-to-rocm/`, not this adapter directory. This file exists only for Codex repository discovery; do not duplicate canonical guidance here.
