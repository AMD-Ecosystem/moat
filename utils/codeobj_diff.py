#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
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
  2. Device code objects (cut out of the binary's `.hip_fatbin` offload bundles,
     disassembled with llvm-objdump), normalized to strip addresses/offsets,
     compared. This is the GPU ISA -- the actual instruction stream that runs on
     the device.

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

What it does NOT compare: host code, and data. A delta that changes only CPU-side
C++ (launch dimensions, allocation sizes) or constant data leaves the exported
symbols and the device instruction stream identical, and reads `identical` here.
That is why the verdict alone carries nothing: `moatlib.py carry-forward` classifies
the delta itself and refuses a `binary-equiv` claim over one the source classifier
does not call inert unless the caller states that the host side was checked.

ELF only. A Windows (PE/COFF) binary is answered `indeterminate`.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile

BIN_SUFFIXES = (".so", ".hsaco")
# Windows output. Nothing here reads PE/COFF, so these are gathered only to be
# answered `indeterminate` by name: a Windows build tree used to hold "no
# binaries", which reads as a wrong path rather than an unsupported format.
PE_SUFFIXES = (".dll", ".exe", ".pyd")
# CMake's compiler-identification and feature-probe executables. They are real ELF
# and identical across two builds, but they are not the project's output, and one
# stray probe that could not be read used to decide the verdict for the whole tree.
SKIP_DIRS = ("CMakeFiles",)
# Build intermediates. An object or a static archive is real ELF, so the magic test
# below would take them, and comparing thousands of them says nothing the linked
# output does not say better.
SKIP_SUFFIXES = (".o", ".obj", ".a", ".lo", ".d")


def _run(args):
    return subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


_TOOLS = {}


def _rocm_roots():
    """Where a ROCm install may keep its LLVM tools, most specific first: the
    environment's say, then the pip SDK (which has no /opt/rocm), then the
    package's fixed prefix."""
    roots = [os.environ.get(v) for v in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH")]
    try:
        r = _run([sys.executable, "-m", "rocm_sdk", "path", "--root"])
        if r.returncode == 0 and r.stdout.strip():
            roots.append(r.stdout.strip().splitlines()[-1])
    except OSError:
        pass
    roots.append("/opt/rocm")
    return [r for r in roots if r]


def _tool(name):
    """Absolute path of an LLVM tool from the ROCm install, or from PATH, or None.

    The ROCm copy comes first: it is the one that knows the amdgcn target, and a
    distribution's llvm-objdump on PATH may not."""
    if name not in _TOOLS:
        found = None
        for root in _rocm_roots():
            for sub in ("lib/llvm/bin", "llvm/bin", "bin"):
                cand = os.path.join(root, sub, name)
                if os.path.isfile(cand) and os.access(cand, os.X_OK):
                    found = cand
                    break
            if found:
                break
        _TOOLS[name] = found or shutil.which(name)
    return _TOOLS[name]


def _is_binary(path):
    if path.endswith(SKIP_SUFFIXES):
        return False
    if path.endswith(BIN_SUFFIXES) or path.lower().endswith(PE_SUFFIXES):
        return True
    if os.path.basename(path).startswith("lib") and ".so" in path:
        return True
    # Otherwise ask the file, not its name. A cmake project ships `libfoo.so` and was
    # found; a Makefile project ships `train_gpt2cu` and was not, so walking its build
    # tree found nothing and the tool answered `indeterminate` -- correct, and useless.
    # The validator then had to know to pass each executable as an explicit path.
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def _gather(root):
    """Map relative path -> absolute path for candidate binaries under root
    (or {basename: path} if root is a single file)."""
    out = {}
    if os.path.isfile(root):
        out[os.path.basename(root)] = os.path.abspath(root)
        return out
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
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
    nm = _tool("llvm-nm")
    if not nm:
        return None
    r = _run([nm, "-D", "--defined-only", binpath])
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


EM_AMDGPU = 224
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"


def _elf_sections(blob):
    """(e_machine, {section name: (offset, size)}) of a little-endian ELF64 image,
    or None when it is not one. Enough of ELF to find one section by name; the
    alternative is a tool dependency for a forty-byte table walk."""
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        return None
    try:
        machine = struct.unpack_from("<H", blob, 18)[0]
        shoff = struct.unpack_from("<Q", blob, 40)[0]
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 58)
        if not shoff or not shnum or shentsize < 64:
            return machine, {}
        heads = []
        for i in range(shnum):
            name, _typ, _flags, _addr, off, size = struct.unpack_from(
                "<IIQQQQ", blob, shoff + i * shentsize)
            heads.append((name, off, size))
        _n, stroff, strsize = heads[shstrndx]
        strtab = blob[stroff:stroff + strsize]
        out = {}
        for name, off, size in heads:
            label = strtab[name:strtab.index(b"\0", name)].decode("ascii", "replace")
            out[label] = (off, size)
        return machine, out
    except (struct.error, IndexError, ValueError):
        return None


def _bundle_slices(blob, sec_off, sec_size):
    """[(triple, file offset, size)] for every device code object in a `.hip_fatbin`
    section, or None when the section cannot be read.

    The section is one clang offload bundle per translation unit: the magic, a
    64-bit entry count, then per entry the code object's offset and size (relative
    to that bundle's start) and its target triple. Read directly, because the tool
    that lists them (roc-obj-ls) is absent from the pip SDK. A compressed bundle is
    not unpacked here and makes the answer None -- indeterminate, never a guess."""
    sec = blob[sec_off:sec_off + sec_size]
    out, done = [], 0
    while True:
        pos = sec.find(BUNDLE_MAGIC, done)
        # Everything between one bundle and the next must be padding. Bytes this
        # reader cannot account for (a compressed bundle, a layout it does not
        # know) are device code nobody compared, so they void the answer.
        gap = sec[done:pos if pos >= 0 else len(sec)]
        if gap.strip(b"\0"):
            return None
        if pos < 0:
            return out
        try:
            cur = pos + len(BUNDLE_MAGIC)
            (count,) = struct.unpack_from("<Q", sec, cur)
            cur += 8
            if count > 4096:
                return None
            end = cur
            for _ in range(count):
                off, size, tlen = struct.unpack_from("<QQQ", sec, cur)
                cur += 24
                triple = sec[cur:cur + tlen].decode("ascii", "replace")
                cur += tlen
                if tlen > 256 or pos + off + size > len(sec):
                    return None
                if size and "amdgcn" in triple:
                    out.append((triple, sec_off + pos + off, size))
                end = max(end, pos + off + size)
        except struct.error:
            return None
        # Resume after this bundle's last code object, so bytes that happen to
        # spell the magic inside a code object are never read as a header.
        done = max(end, cur)


def _device_isa(binpath, workdir):
    """Disassemble every bundled gfx device code object and return a normalized
    string. Returns None on tooling failure, or "" if the binary carries no device
    code (a pure host object) -- which is an answer, not a failure: a build tree
    always holds host-only binaries beside the GPU ones."""
    objdump = _tool("llvm-objdump")
    if not objdump:
        return None
    try:
        with open(binpath, "rb") as fh:
            blob = fh.read()
    except OSError:
        return None
    elf = _elf_sections(blob)
    if elf is None:
        return None
    machine, sections = elf
    if machine == EM_AMDGPU:
        slices = [("amdgcn", 0, len(blob))]      # a bare code object (.hsaco)
    elif ".hip_fatbin" not in sections:
        return ""
    else:
        slices = _bundle_slices(blob, *sections[".hip_fatbin"])
        if slices is None:
            return None
        if not slices:
            return ""
    chunks = []
    for i, (target, off, size) in enumerate(sorted(slices)):
        co = os.path.join(workdir, f"co{i}.elf")
        with open(co, "wb") as fh:
            fh.write(blob[off:off + size])
        d = _run([objdump, "-d", co])
        if d.returncode != 0:
            return None
        chunks.append(f"### {target}\n{_normalize_disasm(d.stdout)}")
    return "\n".join(chunks)


def _is_pe(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"MZ"
    except OSError:
        return False


def compare_binary(a, b):
    """Return (verdict, detail) for one pair of binaries."""
    if _is_pe(a) or _is_pe(b):
        return "indeterminate", ("PE/COFF binary: only ELF is compared here, so "
                                 "binary equivalence cannot be shown on Windows; "
                                 "revalidate on the GPU")
    if not _tool("llvm-nm") or not _tool("llvm-objdump"):
        return "indeterminate", ("llvm-nm/llvm-objdump not found; set ROCM_PATH to "
                                 "the ROCm install")
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
    import trunk  # noqa: E402 - run the trunk's copy of this tool, see trunk.py
    trunk.reexec(__file__)
    raise SystemExit(main(sys.argv))
