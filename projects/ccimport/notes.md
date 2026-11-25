# ccimport notes

## Intake -- 2026-11-20, linux-gfx942 (fleet job, headless)

Recommendation: **DECLINE, reason `already-supported`** -- not because someone else
ported it, but because there is nothing to port: ccimport works unmodified for a HIP
build of cumm and spconv. No fork is needed. Not recorded as a disposition (a person's
call, through the queue); `set-intake` carries the row. If the person agrees, the
`already-supported` disposition is also what clears cumm's `depends_on: [ccimport]`
(`DEP_SATISFIED_BY_DISPOSITION`), so cumm is not left waiting on a project nobody
will port. If the planner later shows a ccimport change is unavoidable, this flips to a
fork for a 1 to 2 file change.

Upstream: https://github.com/FindDefinition/ccimport, default branch `master`, screened
at `f5af43d` (2024-10-15, v0.4.4). 31 tracked files, 2,705 lines of Python, no
submodules, no binary files, no CUDA source.

### What it is

A pure-Python build helper by the cumm/spconv author: it writes ninja files for
C++/CUDA sources and builds pybind11 extensions. pccm (the author's code generator)
calls it, and cumm and spconv build their prebuilt modules through pccm. PyPI
`ccimport` 0.4.4 is a universal `py3-none-any` wheel; nothing in it is per-CUDA or
per-arch.

### Licence -- MIT, with one Apache-2.0 file; tier 1, cleared to contribute

    python3 utils/licenses.py check FindDefinition/ccimport             # MIT, tier 1
    git clone https://github.com/FindDefinition/ccimport agent_space/ccimport-screen
    python3 utils/licenses.py scan-nvidia agent_space/ccimport-screen   # clean

Recorded `MIT AND Apache-2.0` with evidence
https://github.com/FindDefinition/ccimport/blob/f5af43d/LICENSE (tier 1).

- `LICENSE`: MIT, "Copyright (c) 2020 FindDefinition". It is the only licence file.
  `setup.py` says `license='MIT'` and the MIT classifier.
- Per-file: 27 of 31 files carry no copyright or licence text; the other four are
  `LICENSE`, the two `setup.py` files (MIT classifier) and `constants.py`. Components
  whose terms differ:

| path | licence | origin | shipped? | port modifies? |
|---|---|---|---|---|
| `ccimport/constants.py` | Apache-2.0 header, "Copyright 2021 Yan Yan" (same author as the MIT `LICENSE`) | own | yes (Python) | only if a HIP compiler slot is added (not expected) |
| `ccimport/compat.py:100-140` (`_compiler_preprocessor_verbose`, `get_system_include_paths`) | MIT by origin ("copied from AndrewWalker/ccsyspath", whose repository licence is MIT); no notice carried | AndrewWalker/ccsyspath | yes | no |
| `ccimport/buildmeta.py:30`, `compat.py:142` | Stack Overflow snippets (CC BY-SA by origin), a few lines | Stack Overflow | yes | no |
| `test/test_source_iter.py` (test strings "adapted from cpython pytime.c") | PSF-2.0 by origin | CPython | no (test) | no |

Nothing needs a person's clearance. The Apache-2.0 file under an MIT top level is the
same author licensing one file differently; both are tier 1.

### Duplicate effort and upstream state

- No PR, issue or branch upstream mentions HIP/ROCm/AMD (`gh pr list`/`gh issue list
  --state all --search "hip OR rocm OR amd"`: none; branches `master`,
  `feature/debug`). Docs grep: no hits. No ccimport repository in AMD-Ecosystem or ROCm.
- ccimport already contains a vestigial HIP slot: `HIPCXX` and `hipcc` in
  `ALL_SUPPORTED_HIP_COMPILER` and the per-OS compiler sets, but `create_build_rule`
  has no `hipcc` branch (it raises `NotImplementedError`) and nothing uses it.
- Activity: last commit 2024-10-15. **No PR has ever been merged** (0 merged). Two
  outside PRs open and unanswered: #3 (2025-11-05, Python 3.14 / pyproject) and #4
  (2025-11-16, ninja package support). Dormant for outside contributions.

### Why no change is needed (the viability question, inverted)

- The `.cu` rule (`nvcc_build_setup`) is a plain gcc-style command:
  `$cc -MMD -MT $out -MF $out.d -I... $cflags -c $in -o $out` (`-MD` on Windows), and
  the `nvcc` link rule is `$ld @rsp -l "..." -L "..." $ldflags -o $out`. Every token is
  something hipcc/amdclang++ accepts; the nvcc-only flags (`-gencode`,
  `--expt-relaxed-constexpr`, `--extended-lambda`) are added by cumm, not ccimport.
- ccimport already lets the caller point the `nvcc` slot at another compiler without
  code changes: `CUDACXX` (constants.py; sets both compiler and linker for `nvcc`),
  `CCIMPORT_COMPILER_LINKER_MAP=nvcc:<path>`, and the `compiler_to_path` argument.
- jeff's HIP study (`~/.claude/plans/moat-cumm-hip-backend-study.md`) built spconv's
  whole SIMT kernel set (265 `.cu`, 168 `.cc`) for gfx942 through unmodified
  ccimport/pccm by putting a compiler named `nvcc` on PATH. The real port moves that
  selection into cumm (`common.py` flags and discovery), which is cumm's work.
- Its own estimate counted "ccimport 1 to 2 files"; that is the optional cleanup of
  giving HIP a first-class rule (wiring the dead `hipcc` slot, dropping the obsolete
  `__HIP_PLATFORM_HCC__`). It is not required for a working port, and a smallest
  complete port leaves it out.
- Not verified: Windows. The `nvcc` slot pointed at a clang-based HIP compiler on
  Windows uses gcc-style flags, which the clang driver accepts, but the cumm port's
  Windows round is where this gets proven.

### Dependencies

`depends_on: []`. ccimport needs only `pybind11`, `ninja` and `requests` from pip. It is a hard
dependency of cumm (through pccm) and of spconv (`ccimport>=0.4.4` in its `setup.py`).

### Recorded

    python3 utils/moatlib.py scaffold FindDefinition/ccimport --branch master
    python3 utils/moatlib.py set-license-spdx ccimport "MIT AND Apache-2.0" --evidence https://github.com/FindDefinition/ccimport/blob/f5af43d/LICENSE
    python3 utils/moatlib.py set-intake ccimport decline --reason already-supported ...

No `awaiting-fork`: the recommendation is that no fork exists. The intake queue issue
was not published from this job (a GitHub write outside its limits).

### Question for jeff

Agree that ccimport is consumed unmodified (record the `already-supported` skip, which
also clears cumm's dependency edge), or fork it anyway so a first-class HIP rule can be
offered upstream later?
