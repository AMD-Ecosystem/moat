#!/usr/bin/env python3
"""Enumerate what a port has to cover, and check that it all got accounted for.

The recurring failure this exists to stop: a port that claimed success while
covering a subset, caught only by a human saying "you didn't go far enough". That
was expensive and it happened repeatedly.

Two scans, because the two failure modes are different:

  * CUDA surface -- the call sites and kernels. Misses here look like an
    unported file.
  * Project structure -- what the project IS: libraries, executables, tests,
    benchmarks, examples. Misses here look like "the library was ported but its
    tests were not", which a CUDA-call census would never notice, and which is
    what "didn't go far enough" has usually actually meant.

The output is a FLOOR. The planner owns the result: it may ADD entries the tools
cannot see -- driver-API use, runtime-compiled PTX, non-C++ build paths -- and may
remove a generated one only with a recorded reason. mumax3 is the standing example
of why the floor is not the whole truth: Go + cgo + runtime PTX yields a nearly
empty CUDA census for a substantial port.

    python3 utils/surface.py generate <project>
    python3 utils/surface.py check <project>
    python3 utils/surface.py check --all
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

CUDA_MARKERS = {
    "kernel": [r"__global__\b"],
    "device_code": [r"__device__\b", r"__constant__\b", r"__shared__\b"],
    "launch": [r"<<<"],
    "warp_intrinsic": [r"__shfl\w*\s*\(", r"__ballot\w*\s*\(", r"\bwarpSize\b", r"__activemask\b"],
    "texture": [r"\bcudaTextureObject_t\b", r"\btex2D\w*\s*\(", r"\bcudaArray\b"],
    "driver_api": [r"\bcu(Module|Launch|Ctx|Device)[A-Z]\w*\s*\("],
    "runtime_ptx": [r"\bnvrtc\w*\s*\(", r"\.ptx\b"],
}
CUDA_LIBS = {
    "cublas": r"\bcublas\w*\b", "cufft": r"\bcufft\w*\b", "curand": r"\bcurand\w*\b",
    "cusparse": r"\bcusparse\w*\b", "cusolver": r"\bcusolver\w*\b", "cudnn": r"\bcudnn\w*\b",
    "thrust": r"\bthrust::", "cub": r"\bcub::", "cutlass": r"\bcutlass::",
    "nccl": r"\bnccl\w*\b", "cupti": r"\bcupti\w*\b",
}
SRC_EXT = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".c", ".inc"}
# Vendored dependency trees. `subprojects` is meson's, and missing it made lc0
# report 361 components from abseil/eigen/cutlass against 19 of its own. A port is
# responsible for the project, not for what the project vendors.
SKIP_DIRS = {".git", "third_party", "thirdparty", "external", "extern", "vendor",
             "subprojects", "_deps", "deps", "node_modules", ".github", "docs", "doc"}
# Build trees are matched by PREFIX, not exact name: a clone routinely carries
# build/, build-cuda/, build-omp/, cmake-build-debug/. Missing them scanned copied
# SDK headers and reported 196 cudaTextureObject_t hits in a project that uses none.
SKIP_DIR_PREFIXES = ("build", "cmake-build", "out", ".venv", "venv")

# CMake commands that declare something a port has to cover, and what kind it is.
CMAKE_TARGETS = [
    (r"add_library\s*\(\s*([A-Za-z0-9_.\-${}]+)", "library"),
    (r"add_executable\s*\(\s*([A-Za-z0-9_.\-${}]+)", "executable"),
    (r"add_test\s*\(\s*(?:NAME\s+)?([A-Za-z0-9_.\-${}]+)", "test"),
    (r"pybind11_add_module\s*\(\s*([A-Za-z0-9_.\-${}]+)", "python_module"),
]
# Optional components: a feature flag gates code that may need porting too, and is
# the classic thing a port silently leaves behind.
CMAKE_OPTIONS = r"option\s*\(\s*([A-Za-z0-9_]+)\s+\"([^\"]*)\"\s*(ON|OFF)?"


def _skip(parts):
    return any(d in SKIP_DIRS or d.startswith(SKIP_DIR_PREFIXES) for d in parts)


def _walk(root):
    for p in root.rglob("*"):
        if p.is_file() and not _skip(p.relative_to(root).parts):
            yield p


def scan_cuda(root):
    """Census of CUDA usage. Counts, not just presence: a file with 40 kernels is a
    different porting job from one with a single `cudaMalloc`."""
    files, markers, libs = [], {}, {}
    for p in _walk(root):
        if p.suffix.lower() not in SRC_EXT:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(root))
        hit = False
        for kind, pats in CUDA_MARKERS.items():
            n = sum(len(re.findall(pat, text)) for pat in pats)
            if n:
                markers[kind] = markers.get(kind, 0) + n
                hit = True
        for lib, pat in CUDA_LIBS.items():
            n = len(re.findall(pat, text))
            if n:
                libs[lib] = libs.get(lib, 0) + n
                hit = True
        if p.suffix.lower() in {".cu", ".cuh"} or hit:
            files.append(rel)
    return sorted(files), markers, libs


def scan_components(root):
    """What the project IS, from its build system. This is the half that catches
    'the library was ported but its tests were not'."""
    comps = []
    for p in _walk(root):
        if p.name not in {"CMakeLists.txt", "setup.py", "meson.build", "Makefile"} \
           and p.suffix != ".cmake":
            continue
        # Find modules declare imported targets for dependencies, not components.
        if p.name.startswith(("Find", "find")) and p.suffix == ".cmake":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(root))
        if p.name == "CMakeLists.txt" or p.suffix == ".cmake":
            for pat, kind in CMAKE_TARGETS:
                for m in re.finditer(pat, text, re.I):
                    name = m.group(1)
                    # ${VAR} names cannot be resolved without running CMake, and an
                    # IMPORTED or ::-qualified target is a dependency the build FINDS
                    # (ClangFormat::ClangFormat from a Find module), not a component
                    # this project ships and therefore has to port.
                    if name.startswith("${") or "::" in name:
                        continue
                    tail = text[m.end():m.end() + 200]
                    if re.search(r"\bIMPORTED\b|\bALIAS\b|\bINTERFACE\b", tail):
                        continue
                    k = kind
                    low = f"{rel} {name}".lower()
                    if kind == "executable":
                        if "bench" in low:
                            k = "benchmark"
                        elif "test" in low:
                            k = "test"
                        elif "example" in low or "sample" in low or "demo" in low:
                            k = "example"
                    comps.append({"id": f"{k}:{name}", "kind": k, "where": rel})
            for m in re.finditer(CMAKE_OPTIONS, text):
                comps.append({"id": f"option:{m.group(1)}", "kind": "option",
                              "where": rel, "default": m.group(3) or "unset",
                              "doc": m.group(2)[:80]})
        elif p.name == "setup.py":
            for m in re.finditer(r"(CUDAExtension|CppExtension)\s*\(\s*[\"']([^\"']+)", text):
                comps.append({"id": f"extension:{m.group(2)}", "kind": "python_extension",
                              "where": rel})
        elif p.name == "meson.build":
            for m in re.finditer(r"\b(executable|library|test)\s*\(\s*'([^']+)'", text):
                comps.append({"id": f"{m.group(1)}:{m.group(2)}", "kind": m.group(1),
                              "where": rel})
    seen, out = set(), []
    for c in comps:
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return sorted(out, key=lambda c: c["id"])


def generate(name, src=None):
    root = pathlib.Path(src) if src else REPO / "projects" / name / "src"
    if not root.is_dir():
        print(f"surface: no clone at {root}", file=sys.stderr)
        return None
    files, markers, libs = scan_cuda(root)
    comps = scan_components(root)
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()[:12]
    return {
        "generated_by": "utils/surface.py",
        "source_sha": sha,
        "note": ("A FLOOR, not the whole truth. The planner may add entries the scan "
                 "cannot see (driver API, runtime PTX, non-C++ builds) and may remove a "
                 "generated one only with a reason recorded in removed_from_floor."),
        "cuda": {"files": files, "markers": markers, "libraries": libs},
        "components": comps,
        "covered": [],
        "scoped_out": [],
        "removed_from_floor": [],
    }


def claims_success(name):
    """Has this project asserted that its port WORKS? Until it does, an uncovered
    component is work not yet done rather than an omission."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    obj, _where = moatlib.project_record(name)
    if obj is None:
        return False
    if obj.get("stage") in ("ported", "delta-ported", "review-passed"):
        return True
    return any(b.get("state") == "completed"
               for b in (obj.get("platforms") or {}).values())


def check(name):
    """Every component either covered or explicitly scoped out with a reason. The
    gate is ACCOUNTING, not coverage: a scoped-out component is a deliberate,
    reviewable decision, and the failure being eliminated is the silent omission.

    Judged only once a port CLAIMS SUCCESS, which is the failure the gate names: a
    port that said it worked while covering a subset. A project that is merely
    `planned` has covered nothing yet BY DEFINITION -- its surface.json is the
    inventory the porter is about to work through -- so judging it there fails every
    time, and because check.py is repo-wide that failure blocks pushes for every
    project on a shared repo rather than only the one being planned. colmap was the
    first project ever to carry a surface.json and wedged the trunk within minutes."""
    if not claims_success(name):
        return []
    p = REPO / "projects" / name / "surface.json"
    if not p.is_file():
        return [f"{name}: no surface.json (run: utils/surface.py generate {name})"]
    d = json.loads(p.read_text())
    covered = {c if isinstance(c, str) else c.get("id") for c in d.get("covered", [])}
    scoped = {}
    for s in d.get("scoped_out", []):
        if isinstance(s, str):
            scoped[s] = None
        else:
            scoped[s.get("id")] = s.get("reason")
    problems = []
    for c in d.get("components", []):
        cid = c["id"]
        if cid in covered:
            continue
        if cid in scoped:
            if not scoped[cid]:
                problems.append(f"{name}: {cid} scoped out with no reason")
            continue
        problems.append(f"{name}: {cid} ({c['kind']}, {c['where']}) is neither "
                        f"covered nor scoped out")
    for r in d.get("removed_from_floor", []):
        if isinstance(r, dict) and not r.get("reason"):
            problems.append(f"{name}: {r.get('id')} removed from the floor with no reason")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "check"])
    ap.add_argument("project", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--src", help="scan this directory instead of projects/<name>/src")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    a = ap.parse_args()

    if a.cmd == "generate":
        d = generate(a.project, a.src)
        if d is None:
            return 1
        if a.stdout:
            json.dump(d, sys.stdout, indent=2)
            print()
        else:
            out = REPO / "projects" / a.project / "surface.json"
            out.write_text(json.dumps(d, indent=2) + "\n")
            print(f"surface: wrote {out.relative_to(REPO)} -- "
                  f"{len(d['cuda']['files'])} CUDA files, {len(d['components'])} components")
        return 0

    names = ([p.parent.name for p in sorted((REPO / "projects").glob("*/surface.json"))]
             if a.all else [a.project])
    problems = [pr for n in names for pr in check(n)]
    for pr in problems:
        print(f"surface: {pr}")
    if not problems:
        print(f"surface: {len(names)} project(s) fully accounted for")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
