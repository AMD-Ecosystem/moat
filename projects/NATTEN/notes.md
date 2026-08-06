# NATTEN notes

## 2026-06-04: Porter review confirms cant-port disposition

Porter reviewed the plan.md (planner disposition: SKIP, cant-port). The planner's analysis is correct:

1. libnatten is 100% CUTLASS/CuTe -- no naive/GEMM-free CUDA kernels remain (deleted in 0.21 line per upstream CHANGELOG).
2. CUTLASS does not port to ROCm per PORTING_GUIDE (would require ground-up reimplementation in Composable Kernel).
3. AMD is already supported via the Flex Attention backend (`torch.nn.attention.flex_attention` which lowers to Triton on ROCm). The project explicitly detects `torch.version.hip` and routes AMD devices to this path.

No porting work is possible or needed. All platforms marked blocked with reason cant-port. This project is terminal (all platforms non-viable) -- no upstream PR will be opened.
