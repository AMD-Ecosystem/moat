#!/usr/bin/env python3
"""Textual-independence check for the cubvh BVH core rewrite.

Reproduces the provenance analysis method: strip comments, collapse
whitespace, then measure (a) whole-file difflib.SequenceMatcher ratio over
normalized line sequences and (b) the fraction of non-trivial normalized
lines appearing verbatim in the counterpart file, for each rewritten file
against its ancestors (the instant-ngp era snapshot and the pre-rewrite
cubvh files).

Usage:
  python similarity.py NEW_FILE REF_FILE [REF_FILE...]
  python similarity.py --matrix NEW_DIR_SPEC REF_DIR_SPEC
where a line >10 chars that is not an #include/#pragma and not pure
punctuation counts as non-trivial. Interface lines (pinned signatures) are
expected to match; everything else should be noise.
"""

import difflib
import re
import sys


def normalize(path):
    src = open(path, errors="replace").read()
    # strip block and line comments
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    lines = []
    for ln in src.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            lines.append(ln)
    return lines


def nontrivial(lines):
    out = []
    for ln in lines:
        if len(ln) <= 10:
            continue
        if ln.startswith("#include") or ln.startswith("#pragma"):
            continue
        if not re.search(r"[A-Za-z]", ln):
            continue
        out.append(ln)
    return out


def compare(new_path, ref_path):
    a, b = normalize(new_path), normalize(ref_path)
    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    nt = nontrivial(a)
    refset = set(b)
    matched = [ln for ln in nt if ln in refset]
    frac = len(matched) / max(1, len(nt))
    return ratio, frac, len(nt), matched


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    new = args[0]
    print(f"{'ref':50s} {'ratio':>6s} {'linefrac':>8s} {'nt':>5s}")
    worst = 0.0
    for ref in args[1:]:
        ratio, frac, n, matched = compare(new, ref)
        print(f"{ref[-50:]:50s} {ratio:6.3f} {frac:8.3f} {n:5d}")
        worst = max(worst, frac)
        if matched and "-v" in sys.argv:
            for ln in matched:
                print(f"    = {ln}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
