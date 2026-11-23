# spconv-triton: opt-in SPCONV_TRITON_AUTOTUNE=fast (prepared contribution, not sent)

Prepared 2026-11-23 on a linux-gfx942 (MI300X) measurement job at jeff's request, under his 2026-11-23 ruling: offer the setting to spconv-triton (https://github.com/L-Reichardt/spconv-triton, Apache-2.0, default branch `main`) as a pull request he approves first, and hold Pointcept's own pull request until the setting is released. There is no fork of spconv-triton yet (creating one is jeff's act), so nothing was pushed or posted anywhere. Measurements and reasoning are in `notes.md`, section "spconv-triton SPCONV_TRITON_AUTOTUNE=fast prepared for upstream (2026-11-23, ...)".

## Upstream base

`L-Reichardt/spconv-triton` `main` = `63361695e891de7aff32841ebb86f2661a41b059` ("version 1.0.0", tag `1.0.0`, 2026-07-20). The 2026-11-20 draft was written against the 1.0.0 release, which is the same commit, so no rebase was needed. `git apply --check` of the patch below on that commit: clean.

## Draft pull-request title

Add opt-in SPCONV_TRITON_AUTOTUNE=fast for the conv GEMM kernels

## Draft pull-request body

The NVIDIA row is a placeholder for jeff's V100 run. Replace `NVIDIA_RESULT_PLACEHOLDER` with the measured cells (first step full, first step fast, steady full, steady fast, losses), or drop the row.

````markdown
## Problem

With an empty Triton cache, the first training step compiles every autotune candidate of `_gather_gemm_kernel` (37) and `_dweight_masked_kernel` (21) for every new conv shape. In fp32 on an AMD Radeon PRO W7800 (RDNA3, gfx1100) that made the first step of a SparseUNet take about 3.5 hours, and an OA-CNNs model 97 minutes. RDNA has no fp32 matrix instructions, so an IEEE fp32 `tl.dot` becomes an unrolled FMA block; the large tiles spill registers and each spends a minute or more in LLVM's register allocator. Those tiles never won on that GPU. On an MI300X the same SparseUNet first step takes about 19 minutes.

## Change

A new environment variable, `SPCONV_TRITON_AUTOTUNE`, read at import like the others and documented in `docs/ENVIRONMENT_VARIABLES.md`:

- `full` (default): today's candidate lists, unchanged.
- `fast`: the two conv GEMM kernels autotune over short subsets of the full lists (4 gather-GEMM and 3 weight-gradient candidates), chosen from full-sweep timings on the W7800.

The weight-gradient split estimate (`_DW_MAX_BLOCK_K/C`) still derives from the full list. Deriving it from the fast list changes the split factor, and with it the fp32 summation order: an earlier draft that did this changed training losses from the third step.

The subsets live in `_autotune.py`, not `gemm.py`, because Triton keys compiled kernels on their starting line number. Code added above the kernels in `gemm.py` would make every user recompile on upgrade, even with the default setting. As written, the default setting reuses a cache filled by 1.0.0: the full suite ran against such a cache with `SPCONV_TEST_EXPECT_WARM=1` and compiled nothing.

`scripts/verify_autotune_fast.py` checks that the default lists are the full ones, that the fast lists are non-empty subsets of them (so an edit to the full list cannot silently drop a fast entry), that the split estimate does not move, and that an unknown value is rejected. It is a script rather than a test because `tests/` is checksum-frozen; it can move into the suite if you prefer, with a manifest update.

## Results

fp32 SparseUNet training (Pointcept `semseg-spunet-v1m1-0-base`, 2 synthetic scenes x 20k voxels), empty `TRITON_CACHE_DIR` for the first step, steady step from warm reruns:

| GPU | first step, full | first step, fast | steady step, full | steady step, fast | losses |
|---|---|---|---|---|---|
| Radeon PRO W7800 (gfx1100) | about 12600 s | 222 s | 0.124 s | 0.126 s | identical at printed precision (20 steps) |
| Instinct MI300X (gfx942) | 1155 s | 139 s | 0.049-0.050 s | 0.053 s | bit-identical (40 steps) |
| NVIDIA V100 | NVIDIA_RESULT_PLACEHOLDER | | | | |

`fast` is a cold-start setting, not a speedup. On MI300X the subsets miss most full-sweep winners (they hold the winner for 12 of 29 gather-GEMM keys and 4 of 17 weight-gradient keys in this workload), so the steady step is about 6 to 8% slower there. The default stays `full` for that reason.

Test suite on MI300X (torch 2.16 dev build with ROCm 7.16 HIP, Triton 3.9.0): 274 passed, 1 failed, 1 skipped at main, with this change under the default setting, and with `SPCONV_TRITON_AUTOTUNE=fast` (cold cache, 447 s against 2719 s at main; the main run shared the GPU with other work part of the time). The one failure, `test_fx_proxyable_metaclass`, also fails at main and comes from the dev torch build.

## Not included

- Pruning candidates whose shared-memory need exceeds the device limit. On RDNA3 those candidates compile fully and then fail with `OutOfResources`, about a quarter of the cold time, and it would help every GPU under the default setting. But the estimate would have to match Triton's allocation on every backend and version, and a wrong estimate would drop a valid winner from the default sweep. It could be a follow-up.
- Compiling candidates in parallel. Triton has `AsyncCompileMode`, but its Autotuner compiles one candidate at a time; that belongs in Triton.

## Test plan

```
uv run pytest tests/ -q -p no:cacheprovider
SPCONV_TRITON_AUTOTUNE=fast uv run pytest tests/ -q -p no:cacheprovider
uv run python -m scripts.verify_autotune_fast
```

This pull request was prepared with the help of an AI coding agent.
````

## Reproducing the evidence

Environment on the MI300X: torch 2.16.0a0+git6c3af02 (HIP 7.16.26385), Triton 3.9.0+gitfc8d31a5, Python 3.11, Pointcept `moat-port` at 35af626 with its extensions built for gfx942 (see the 2026-11-19 gfx942 validation in `notes.md`). spconv-triton was the clone below, put on `sys.path` by a `.pth` file beside a copied `spconv_triton-1.0.0.dist-info` (hatchling was not installed, so `pip install -e` could not build); upstream main ran from a worktree through `PYTHONPATH`. Every GPU command ran under `ulimit -c 0` with `HIP_VISIBLE_DEVICES` pinned to one GPU.

```
git clone https://github.com/L-Reichardt/spconv-triton && cd spconv-triton    # main = 6336169
git worktree add ../spconv-triton-base 6336169
git am <this patch>

# Test suite: upstream main, the change with the default setting, the change with fast.
# The tox ROCm rows set HIPBLASLT_ALLOW_TF32=1 and PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; so did these runs.
cd ../spconv-triton-base && TRITON_CACHE_DIR=$PWD/../tc-base PYTHONPATH=$PWD HIPBLASLT_ALLOW_TF32=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q -p no:cacheprovider -rs
cp -a ../tc-base ../tc-default
cd ../spconv-triton && SPCONV_TEST_EXPECT_WARM=1 TRITON_CACHE_DIR=$PWD/../tc-default PYTHONPATH=$PWD HIPBLASLT_ALLOW_TF32=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q -p no:cacheprovider -rs
cp -a ../tc-base ../tc-default2
TRITON_CACHE_DIR=$PWD/../tc-default2 PYTHONPATH=$PWD HIPBLASLT_ALLOW_TF32=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q -p no:cacheprovider -rs
SPCONV_TRITON_AUTOTUNE=fast TRITON_CACHE_DIR=$(mktemp -d) PYTHONPATH=$PWD HIPBLASLT_ALLOW_TF32=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q -p no:cacheprovider -rs
python -m scripts.verify_autotune_fast
ruff check spconv_triton scripts/verify_autotune_fast.py && ruff format --check spconv_triton && mypy   # ruff 0.15.4, mypy 2.2.0 (the pinned dev versions)

# SparseUNet full vs fast: the gfx942 Pointcept driver (pointcept_sparse_conv_e2e.py, described in the
# 2026-11-19 gfx942 validation in notes.md; this copy also prints each loss with repr()), run from the
# Pointcept source root. Cold = empty cache; then two warm reruns of each on the same cache.
cd <Pointcept src at 35af626>
TRITON_CACHE_DIR=<empty dir A> PYTHONPATH=. python pointcept_sparse_conv_e2e.py --steps 40 --only spunet
SPCONV_TRITON_AUTOTUNE=fast TRITON_CACHE_DIR=<empty dir B> PYTHONPATH=. python pointcept_sparse_conv_e2e.py --steps 40 --only spunet
TRITON_CACHE_DIR=<dir A> PYTHONPATH=. python pointcept_sparse_conv_e2e.py --steps 40 --only spunet          # x2
SPCONV_TRITON_AUTOTUNE=fast TRITON_CACHE_DIR=<dir B> PYTHONPATH=. python pointcept_sparse_conv_e2e.py --steps 40 --only spunet   # x2

# Full-sweep winners against the fast lists: reads every *.autotune.json in a full-mode cache
# (each holds every candidate's median/p20/p80 time) and reports the winner per key and the best fast candidate.
python winners.py <dir A>
```

`winners.py` (host-local in `agent_space/runs/`, 70 lines) dedupes `_gather_gemm_kernel` and `_dweight_masked_kernel` records by (kernel, tuning key), skips records with no more candidates than the fast list, and per key prints the min-median candidate, whether it is in the fast list, and the best fast candidate's median over the winner's; the summary is the geomean and the worst of that ratio. `winners_ext.py` is the same with the three extra candidates discussed in `notes.md`.

For the V100 row: the same driver and commands on an NVIDIA GPU with a CUDA torch (Pointcept then imports spconv, not spconv-triton, so drive spconv-triton directly or force the import), or spconv-triton's own suite under both settings plus a SparseUNet-style loop built from `spconv_triton.pytorch` layers.

## Patch (`git format-patch -1 --stdout`, commit 1fca5d1 on 6336169)

````diff
From 1fca5d1515892cbadf301dcfcd2c033f549d1048 Mon Sep 17 00:00:00 2001
From: Jeff Daily <jeff.daily@amd.com>
Date: Mon, 23 Nov 2026 21:17:53 +0000
Subject: [PATCH] Add opt-in SPCONV_TRITON_AUTOTUNE=fast for the conv GEMM
 kernels

With an empty Triton cache, the first training step compiles every autotune candidate of `_gather_gemm_kernel` (37) and `_dweight_masked_kernel` (21) for every new conv shape. In fp32 on an AMD Radeon PRO W7800 (RDNA3, gfx1100) this made the first step of a SparseUNet take about 3.5 hours. Without fp32 matrix instructions an IEEE fp32 `tl.dot` becomes an unrolled FMA block, the large tiles spill registers, and each spends a minute or more in LLVM's register allocator. Those tiles never won on that GPU.

This adds `SPCONV_TRITON_AUTOTUNE`, read at import like the other variables. `full`, the default, keeps today's candidate lists. `fast` autotunes the two kernels over short subsets of the full lists (4 and 3 candidates) chosen from full-sweep timings on the W7800. The weight-gradient split estimate (`_DW_MAX_BLOCK_K/C`) still derives from the full list, because the split factor sets the fp32 summation order. The subsets and the helper live in `_autotune.py` rather than `gemm.py` because Triton keys compiled kernels on their starting line: code added above the kernels would invalidate every existing cache even under the default setting. With this change the default setting reuses a cache filled by 1.0.0 without recompiling anything.

Measured on fp32 SparseUNet training (20k voxels x 2 scenes, empty cache): on the W7800 the first step drops from about 3.5 hours to 222 s and the steady step goes from 0.124 to 0.126 s. On an AMD Instinct MI300X (gfx942) the first step drops from 1155 s to 139 s and the steady step goes from 0.049-0.050 to 0.053 s, because the subsets were chosen on RDNA3 and miss most MI300X winners. Training losses were identical between `full` and `fast`: bit-identical over 40 steps on the MI300X, and identical at the four printed decimals over 20 steps on the W7800.

`scripts/verify_autotune_fast.py` checks that the default lists are the full ones, that the fast lists are non-empty subsets of them, that the split estimate does not move, and that an unknown value is rejected.

This change was prepared with the help of an AI coding agent.

Test Plan:

```
uv run pytest tests/ -q -p no:cacheprovider
SPCONV_TRITON_AUTOTUNE=fast uv run pytest tests/ -q -p no:cacheprovider
uv run python -m scripts.verify_autotune_fast
```
---
 CHANGELOG.md                             |   2 +
 docs/ENVIRONMENT_VARIABLES.md            |   5 +-
 scripts/verify_autotune_fast.py          | 104 +++++++++++++++++++++++
 spconv_triton/constants.py               |   8 ++
 spconv_triton/pytorch/_impl/_autotune.py |  36 ++++++++
 spconv_triton/pytorch/_impl/gemm.py      |  10 +--
 6 files changed, 158 insertions(+), 7 deletions(-)
 create mode 100644 scripts/verify_autotune_fast.py

diff --git a/CHANGELOG.md b/CHANGELOG.md
index 82f499a..cd92247 100644
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -4,6 +4,8 @@ All notable changes to this project will be documented in this file.
 
 ## [Unreleased]
 
+- `SPCONV_TRITON_AUTOTUNE=fast` (opt-in) autotunes the conv GEMM kernels over a short candidate list, cutting first-step compile time where the full sweep takes hours (fp32 on AMD RDNA). Default unchanged.
+
 ## [0.1.0b1]
 
 First public beta.
diff --git a/docs/ENVIRONMENT_VARIABLES.md b/docs/ENVIRONMENT_VARIABLES.md
index 774d400..5721763 100644
--- a/docs/ENVIRONMENT_VARIABLES.md
+++ b/docs/ENVIRONMENT_VARIABLES.md
@@ -1,8 +1,8 @@
 # Environment variables
 
-spconv-Triton reads four environment variables.
+spconv-Triton reads five environment variables.
 
-One might be useful, the other three are kept for maximum compatibility.
+Two might be useful, the other three are kept for maximum compatibility.
 Suggestion: leave defaults.
 
 | Variable | Default | Effect |
@@ -10,6 +10,7 @@ Suggestion: leave defaults.
 | `SPCONV_ALLOW_FP16_ACCUM` | `0` | `1` = fp16 GEMMs accumulate in fp16 (matches spconv-cuda). Faster only on consumer Ampere/Ada, but with some accuracy loss, hence defaults to off. |
 | `SPCONV_DO_SORT` | `1` | `0` = skip the pair-mask sort in implicit-GEMM pair generation (inherited performance knob from spconv-cuda). |
 | `SPCONV_FX_TRACE_MODE` | `0` | `1` = `SparseConvTensor` skips shape asserts so models can be traced with `torch.fx` (inherited from spconv-cuda). |
+| `SPCONV_TRITON_AUTOTUNE` | `full` | `fast` = the conv GEMM kernels autotune over a short candidate list instead of the full sweep. For GPUs where the first step with an empty Triton cache compiles for tens of minutes to hours (fp32 on GPUs without fp32 matrix instructions, e.g. AMD RDNA). The short list was chosen on RDNA3 and can cost steady-state speed on other GPUs; leave `full` where the first-step time is acceptable. |
 | `SPCONV_SAVED_WEIGHT_LAYOUT` | unset | Layout (`KRSC`/`RSKC`/`RSCK`) of a checkpoint being loaded (same as spconv-cuda, including its bug. Loading non-KRSC layouts always fails). |
 
 ## Removed
diff --git a/scripts/verify_autotune_fast.py b/scripts/verify_autotune_fast.py
new file mode 100644
index 0000000..6f1cb53
--- /dev/null
+++ b/scripts/verify_autotune_fast.py
@@ -0,0 +1,104 @@
+"""``SPCONV_TRITON_AUTOTUNE``: the conv GEMM kernels' autotune candidate lists.
+
+The variable is read at import, so each setting is probed in a fresh
+interpreter. Checks:
+  - default: every conv GEMM kernel autotunes over its full candidate list;
+  - ``fast``: each list is a non-empty proper subset of the full list, and every
+    configured fast entry exists in it (an edit to the full list must not
+    silently drop one);
+  - ``fast`` leaves the weight-gradient split estimate (``_DW_MAX_BLOCK_K/C``)
+    unchanged: it sets the fp32 summation order, so results must not depend on
+    the candidate set;
+  - an unknown value is rejected at import.
+
+Run from the repository root: ``python -m scripts.verify_autotune_fast``.
+"""
+
+import json
+import os
+import subprocess
+import sys
+
+PROBE = """
+import json
+from spconv_triton import constants
+from spconv_triton.pytorch._impl import _autotune, gemm
+
+def keys(cs):
+    return [list(_autotune.config_key(c)) for c in cs]
+
+print(json.dumps({
+    "mode": constants.SPCONV_TRITON_AUTOTUNE,
+    "gemm_full": keys(gemm._gemm_configs()),
+    "gemm": keys(gemm._gather_gemm_kernel.configs),
+    "gemm_fast": [list(k) for k in _autotune.GEMM_FAST],
+    "dweight_full": keys(gemm._dweight_configs()),
+    "dweight": keys(gemm._dweight_masked_kernel.configs),
+    "dweight_fast": [list(k) for k in _autotune.DWEIGHT_FAST],
+    "dw_max": [gemm._DW_MAX_BLOCK_K, gemm._DW_MAX_BLOCK_C],
+}))
+"""
+
+
+def probe(mode):
+    env = dict(os.environ)
+    env.pop("SPCONV_TRITON_AUTOTUNE", None)
+    if mode is not None:
+        env["SPCONV_TRITON_AUTOTUNE"] = mode
+    r = subprocess.run(  # noqa: S603 (fixed argv: this interpreter + PROBE)
+        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True
+    )
+    if r.returncode != 0:
+        return None
+    return json.loads(r.stdout.strip().splitlines()[-1])
+
+
+def main():
+    checks = []
+    full = probe(None)
+    checks.append(("default mode is full", full["mode"] == "full"))
+    for kernel in ["gemm", "dweight"]:
+        checks.append(
+            (f"default {kernel}: full list", full[kernel] == full[f"{kernel}_full"])
+        )
+
+    fast = probe("fast")
+    checks.append(("fast mode", fast["mode"] == "fast"))
+    for kernel in ["gemm", "dweight"]:
+        cands, every = fast[kernel], fast[f"{kernel}_full"]
+        print(f"  {kernel}: {len(cands)} of {len(every)} candidates under fast")
+        checks.append(
+            (
+                f"fast {kernel}: non-empty proper subset",
+                0 < len(cands) < len(every) and all(c in every for c in cands),
+            )
+        )
+        checks.append(
+            (
+                f"fast {kernel}: every fast entry found",
+                sorted(cands) == sorted(fast[f"{kernel}_fast"]),
+            )
+        )
+    dw_max = [
+        max(c[1] for c in full["dweight_full"]),
+        max(c[2] for c in full["dweight_full"]),
+    ]
+    checks.append(
+        (
+            "fast: dW split estimate unchanged",
+            fast["dw_max"] == full["dw_max"] == dw_max,
+        )
+    )
+
+    checks.append(("unknown value rejected", probe("quick") is None))
+
+    ok = True
+    for name, passed in checks:
+        ok &= passed
+        print(f"  {'OK ' if passed else 'BAD'} {name}")
+    print("\n" + ("AUTOTUNE LISTS OK" if ok else "PROBLEM"))
+    raise SystemExit(0 if ok else 1)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/spconv_triton/constants.py b/spconv_triton/constants.py
index a6fead6..d8e6b10 100644
--- a/spconv_triton/constants.py
+++ b/spconv_triton/constants.py
@@ -55,6 +55,14 @@ SPCONV_ALLOW_TF32 = False
 # Faster ONLY on consumer Ampere/Ada.
 SPCONV_ALLOW_FP16_ACCUM = os.getenv("SPCONV_ALLOW_FP16_ACCUM", "0") == "1"
 
+# Autotune candidate set of the conv GEMM kernels, read at import. "full"
+# (default) sweeps every candidate. "fast" tries a short list, for GPUs where
+# compiling the full sweep makes the first step take tens of minutes to hours
+# (fp32 on GPUs without fp32 matrix instructions, e.g. AMD RDNA).
+SPCONV_TRITON_AUTOTUNE = os.getenv("SPCONV_TRITON_AUTOTUNE", "full")
+if SPCONV_TRITON_AUTOTUNE not in ["full", "fast"]:
+    raise AssertionError("please set SPCONV_TRITON_AUTOTUNE to full or fast")
+
 # Re-exported by spconv_triton.core, where the frozen suite pins it.
 NDIM_DONT_CARE = 3
 
diff --git a/spconv_triton/pytorch/_impl/_autotune.py b/spconv_triton/pytorch/_impl/_autotune.py
index ddceccf..68fe201 100644
--- a/spconv_triton/pytorch/_impl/_autotune.py
+++ b/spconv_triton/pytorch/_impl/_autotune.py
@@ -7,12 +7,19 @@
 
 Exposes ``AUTOTUNE_CACHE_KW`` to splat into ``@triton.autotune(...)``; enables
 disk-persisted config selection where supported, no-op otherwise.
+
+Also the ``SPCONV_TRITON_AUTOTUNE=fast`` candidate subsets of the conv GEMM
+kernels. They live here, not in gemm.py, because Triton keys compiled kernels on
+their source line: code added above the kernels would invalidate every existing
+cache, including under the default setting.
 """
 
 import inspect
 
 import triton
 
+from spconv_triton import constants
+
 # Splat into @triton.autotune(...): {"cache_results": True} on Triton >= 3.3.0,
 # else {} (the kwarg is absent and would raise TypeError).
 AUTOTUNE_CACHE_KW: dict[str, bool] = (
@@ -20,3 +27,32 @@ AUTOTUNE_CACHE_KW: dict[str, bool] = (
     if "cache_results" in inspect.signature(triton.autotune).parameters
     else {}
 )
+
+# SPCONV_TRITON_AUTOTUNE=fast: short candidate lists, each entry a config of the
+# full list as (BLOCK_M, BLOCK_N, BLOCK_R, GROUP_SIZE_M, num_warps, num_stages) for
+# the gather-GEMM and (BLOCK_P, BLOCK_K, BLOCK_C, num_warps, num_stages) for the
+# weight gradient. Chosen from full-sweep timings of fp32 training on RDNA3, where
+# the large fp32 tiles spill registers and take a minute or more each to compile.
+GEMM_FAST = [
+    (256, 16, 16, 8, 4, 3),
+    (128, 128, 32, 1, 8, 3),
+    (64, 16, 16, 8, 2, 4),
+    (64, 64, 32, 8, 4, 2),
+]
+DWEIGHT_FAST = [
+    (64, 32, 32, 4, 6),
+    (64, 32, 32, 2, 5),
+    (128, 32, 16, 4, 3),
+]
+
+
+def config_key(c):
+    return (*c.kwargs.values(), c.num_warps, c.num_stages)
+
+
+def autotune_configs(configs, fast):
+    """The candidates handed to ``triton.autotune``: all of ``configs``, or under
+    ``SPCONV_TRITON_AUTOTUNE=fast`` those listed in ``fast``."""
+    if constants.SPCONV_TRITON_AUTOTUNE != "fast":
+        return configs
+    return [c for c in configs if config_key(c) in fast]
diff --git a/spconv_triton/pytorch/_impl/gemm.py b/spconv_triton/pytorch/_impl/gemm.py
index 0a27cd3..6f89b62 100644
--- a/spconv_triton/pytorch/_impl/gemm.py
+++ b/spconv_triton/pytorch/_impl/gemm.py
@@ -37,7 +37,7 @@ import triton.language as tl
 
 from spconv_triton import constants
 
-from ._autotune import AUTOTUNE_CACHE_KW
+from ._autotune import AUTOTUNE_CACHE_KW, DWEIGHT_FAST, GEMM_FAST, autotune_configs
 
 # tl.dot's fp32-precision kwarg was renamed across Triton versions: >=3.0 takes
 # input_precision="ieee"/"tf32", 2.2/2.3 the older allow_tf32=False/True. Same
@@ -191,8 +191,8 @@ def _dweight_configs():
     ]
 
 
-# Largest dW tile, derived (not hardcoded) so the _dweight_splits occupancy
-# estimate can't drift when configs grow.
+# Largest dW tile, derived from the full list (not hardcoded, never the fast subset)
+# so the _dweight_splits estimate, and with it the fp32 sum order, can't drift.
 _DW_MAX_BLOCK_K = max(c.kwargs["BLOCK_K"] for c in _dweight_configs())
 _DW_MAX_BLOCK_C = max(c.kwargs["BLOCK_C"] for c in _dweight_configs())
 
@@ -204,7 +204,7 @@ def _or_combine(a, b):
 
 
 @triton.autotune(
-    configs=_gemm_configs(),
+    configs=autotune_configs(_gemm_configs(), GEMM_FAST),
     # HAS_BIAS is deliberately NOT a key: bias is a single post-reduction broadcast
     # add that does not shift the optimal tile/warp/stage, so keying on it would run
     # two identical sweeps (bias vs no-bias). It stays a constexpr compile specializer.
@@ -322,7 +322,7 @@ def _gather_gemm_kernel(
 
 
 @triton.autotune(
-    configs=_dweight_configs(),
+    configs=autotune_configs(_dweight_configs(), DWEIGHT_FAST),
     key=["kv", "K_dim", "C_dim", "IEEE", "ALLOW_TF32", "FP16_ACC"],
     **AUTOTUNE_CACHE_KW,
 )
-- 
2.55.0

````
