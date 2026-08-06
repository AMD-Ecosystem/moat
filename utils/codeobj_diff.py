#!/usr/bin/env python3
"""Binary-equivalence confirmer for the MOAT regression guard (validator-side).

Ground truth for "did the compiled program change on this arch?": compare the
GPU code objects and the exported symbol table of two builds. Used by the
validator to carry a platform's validation forward across a revalidate delta
WITHOUT re-running GPU tests, when the compiled output is provably unchanged.

What it compares, per matched binary (shared lib / executable) in the two builds:

  1. Exported dynamic defined symbols (llvm-nm -D --defined-only). These must be
     BYTE-IDENTICAL, never merely consistent-up-to-rename: an external consumer
     (a Python binding, another lib) references them by literal name, so renaming
     an exported symbol IS a behavior change and must force revalidation. This is
     the exported-rename hazard guard.
  2. Device code objects (extracted with roc-obj, disassembled with llvm-objdump),
     normalized to strip addresses/offsets, compared. This is the GPU ISA -- the
     actual instruction stream that runs on the device.

Verdict:
  identical      -- exported symbols and device ISA match on every binary. Safe
                    to carry forward on THIS arch.
  differ         -- something changed; full revalidation required.
  indeterminate  -- a binary was missing, or device-code extraction was not
                    possible; treat as differ (revalidate). Never a false pass.

Scope (v1): this is an EXACT device-ISA compare. It confirms the inert classes
the source classifier conservatively kicked to the validator -- reformatting and
comment reworks that shift __LINE__ but compile to identical codegen. Confirming
an internal-symbol RENAME (where the device ISA is identical only after
abstracting local symbol references under a consistent bijection) is a v2
enhancement; until then a rename-only delta revalidates on real GPU. The source
classifier (changeclass.py) is always the first gate, so this tool only ever runs
to confirm an already source-classified delta, never to discover one.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

ROCM = "/opt/rocm"
OBJDUMP = f"{ROCM}/llvm/bin/llvm-objdump"
NM = f"{ROCM}/llvm/bin/llvm-nm"
ROC_OBJ_LS = f"{ROCM}/bin/roc-obj-ls"

BIN_SUFFIXES = (".so", ".hsaco")


def _run(args):
    return subprocess.run(args, capture_output=True, text=True)


def _is_binary(path):
    if path.endswith(BIN_SUFFIXES):
        return True
    if os.path.basename(path).startswith("lib") and ".so" in path:
        return True
    return False


def _gather(root):
    """Map relative path -> absolute path for candidate binaries under root
    (or {basename: path} if root is a single file)."""
    out = {}
    if os.path.isfile(root):
        out[os.path.basename(root)] = os.path.abspath(root)
        return out
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            if _is_binary(full) and not os.path.islink(full):
                out[os.path.relpath(full, root)] = full
    return out


# Compiler/runtime-generated symbols that vary between otherwise-identical HIP
# builds (per-translation-unit GUID tags, fat-binary handles). They are not API
# and carry no behavior, so they are excluded from the exported-symbol compare;
# a real exported-symbol change (a renamed entrypoint) is still caught.
_VOLATILE_SYM_PREFIXES = ("__hip_cuid_", "__cuda_module_", "__hip_gpubin_handle",
                          "__hip_fatbin")


def _exported_symbols(binpath):
    r = _run([NM, "-D", "--defined-only", binpath])
    if r.returncode != 0:
        return None
    names = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            sym = parts[-1]
            if any(sym.startswith(p) for p in _VOLATILE_SYM_PREFIXES):
                continue
            names.append(sym)
    return sorted(names)


_ADDR_NORMALIZE = None


def _normalize_disasm(text):
    """Drop leading instruction addresses and trailing absolute-offset comments
    so two builds of identical code compare equal regardless of load layout."""
    global _ADDR_NORMALIZE
    if _ADDR_NORMALIZE is None:
        import re
        _ADDR_NORMALIZE = re.compile(r"^\s*[0-9a-f]+:\s*((?:[0-9a-f]{2} )+)\s*", re.I)
    lines = []
    for ln in text.splitlines():
        ln = ln.rstrip()
        if not ln or "file format" in ln:
            continue  # blank or the "<path>: file format ..." objdump header
        if ln.endswith(":") and " " not in ln.strip():
            continue  # section header
        m = _ADDR_NORMALIZE.match(ln)
        if m:
            ln = ln[m.end():]
        # strip "// 0x...." style absolute-address comments
        if "// " in ln:
            ln = ln.split("// ", 1)[0].rstrip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def _device_isa(binpath, workdir):
    """Disassemble every bundled gfx device code object and return a normalized
    string. Each code object is sliced out of the binary by the offset/size that
    roc-obj-ls reports (roc-obj-extract's URI form is deprecated/brittle), then
    disassembled with llvm-objdump. Returns None on tooling failure, or "" if the
    binary carries no device code (a pure host object)."""
    import re
    ls = _run([ROC_OBJ_LS, binpath])
    if ls.returncode != 0:
        return None
    slices = []  # (target, offset, size)
    for line in ls.stdout.splitlines():
        if "gfx" not in line or "size=" not in line:
            continue
        mo, ms = re.search(r"offset=(\d+)", line), re.search(r"size=(\d+)", line)
        if not (mo and ms):
            continue
        size = int(ms.group(1))
        if size <= 0:
            continue
        target = line.split()[1] if len(line.split()) > 1 else "gfx"
        slices.append((target, int(mo.group(1)), size))
    if not slices:
        return ""  # host-only binary; nothing on the device side to compare
    try:
        blob = open(binpath, "rb").read()
    except OSError:
        return None
    chunks = []
    for i, (target, off, size) in enumerate(sorted(slices)):
        co = os.path.join(workdir, f"co{i}.elf")
        with open(co, "wb") as fh:
            fh.write(blob[off:off + size])
        d = _run([OBJDUMP, "-d", co])
        if d.returncode != 0:
            return None
        chunks.append(f"### {target}\n{_normalize_disasm(d.stdout)}")
    return "\n".join(chunks)


def compare_binary(a, b):
    """Return (verdict, detail) for one pair of binaries."""
    sa, sb = _exported_symbols(a), _exported_symbols(b)
    if sa is None or sb is None:
        return "indeterminate", "nm failed"
    if sa != sb:
        added = sorted(set(sb) - set(sa))[:3]
        removed = sorted(set(sa) - set(sb))[:3]
        return "differ", f"exported symbols differ (+{added} -{removed})"
    with tempfile.TemporaryDirectory() as wa, tempfile.TemporaryDirectory() as wb:
        ia = _device_isa(a, wa)
        ib = _device_isa(b, wb)
    if ia is None or ib is None:
        return "indeterminate", "device-code extraction failed"
    if ia != ib:
        return "differ", "device ISA differs"
    return "identical", f"exported symbols + device ISA identical ({len(sa)} exports)"


def compare(old_build, new_build):
    """Compare two build trees (or two files). Returns (verdict, details)."""
    # Two explicit files: pair them directly regardless of basename.
    if os.path.isfile(old_build) and os.path.isfile(new_build):
        v, d = compare_binary(os.path.abspath(old_build), os.path.abspath(new_build))
        return v, [f"{os.path.basename(old_build)} vs {os.path.basename(new_build)}: {v} ({d})"]
    A, B = _gather(old_build), _gather(new_build)
    if not A or not B:
        return "indeterminate", ["no binaries found in one or both inputs"]
    keys = set(A) & set(B)
    if not keys:
        return "indeterminate", ["no binaries with matching names across builds"]
    if set(A) != set(B):
        only = sorted((set(A) ^ set(B)))[:5]
        return "differ", [f"binary set differs (e.g. {only})"]
    details, worst = [], "identical"
    rank = {"identical": 0, "indeterminate": 1, "differ": 2}
    for k in sorted(keys):
        v, d = compare_binary(A[k], B[k])
        details.append(f"{k}: {v} ({d})")
        if rank[v] > rank[worst]:
            worst = v
    return worst, details


def main(argv):
    if len(argv) != 3:
        print("usage: codeobj_diff.py <old_build_dir|file> <new_build_dir|file>", file=sys.stderr)
        return 2
    verdict, details = compare(argv[1], argv[2])
    print(f"verdict={verdict}")
    for d in details:
        print(f"  {d}")
    # exit 0 only on a positive identical (safe to carry forward); else nonzero
    return 0 if verdict == "identical" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
