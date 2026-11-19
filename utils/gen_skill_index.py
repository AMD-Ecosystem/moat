#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Generate the entry index of the cuda-to-rocm skill, and each reference file's contents.

The skill's SKILL.md routes a reader to a reference file; the index under it lists every
entry of those files on one line, so a reader can scan for a symptom without opening
them. A hand-kept index drifted until half the entries were missing from it, so it is
generated from the entry titles and `check.py skills` fails when it is stale.

    python3 utils/gen_skill_index.py            # rewrite the marked regions
    python3 utils/gen_skill_index.py --check    # exit 1 if any region is stale

What an entry is, in a file listed in ENTRY_FILES:
  - a third-level heading (`### ...`), or
  - a paragraph that starts with a bold lead sentence (`**...**`) and comes before the
    first third-level heading of its `## ` section. A bold paragraph after such a
    heading is a sub-point of that heading's entry, not an entry.

The regions are delimited by HTML comments, which Markdown renderers hide:
`<!-- index:begin -->` .. `<!-- index:end -->` in SKILL.md, and
`<!-- toc:begin -->` .. `<!-- toc:end -->` near the top of each file in TOC_FILES.
"""

import argparse
import os
import pathlib
import re
import sys

TOOLS_ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILL_REL = pathlib.PurePosixPath(".claude/skills/cuda-to-rocm")

# The files whose entries are indexed, in the order the index lists them.
ENTRY_FILES = (
    "faults-wavefront.md",
    "faults-memory-streams.md",
    "faults-textures-interop.md",
    "faults-numerics.md",
    "build-headers-macros.md",
    "build-cmake-link.md",
    "windows.md",
    "libraries.md",
    "platform-limits.md",
    "diagnosis.md",
    "validation.md",
    "linux-host-setup.md",
)
# Files that carry a generated table of contents: the entry files and the long recipes.
TOC_FILES = ENTRY_FILES + ("strategy-a-cmake.md", "strategy-b-torch.md", "no-hip-equivalent.md")

INDEX_MARKS = ("<!-- index:begin -->", "<!-- index:end -->")
TOC_MARKS = ("<!-- toc:begin -->", "<!-- toc:end -->")
MAX_TITLE = 90


def repo_root():
    env = os.environ.get("MOAT_REPO")
    return pathlib.Path(env).resolve() if env else TOOLS_ROOT


def read(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def shorten(title):
    """One index line: whitespace collapsed, trailing punctuation dropped, cut at a word
    boundary when long. The full sentence is in the reference file."""
    title = re.sub(r"\s+", " ", title).strip().rstrip(".:")
    if len(title) <= MAX_TITLE:
        return title
    return title[:MAX_TITLE].rsplit(" ", 1)[0].rstrip(",;:(") + " ..."


def parse(text):
    """(file title, [(section heading or None, [entry titles])]) for one reference file."""
    lines = text.split("\n")
    title = ""
    sections = [(None, [])]
    seen_h3 = False
    fenced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
            seen_h3 = False
        elif line.startswith("### "):
            sections[-1][1].append(shorten(line[4:]))
            seen_h3 = True
        elif line.startswith("**") and not seen_h3 and (i == 0 or not lines[i - 1].strip()):
            lead = line[2:]
            j = i
            while "**" not in lead and j + 1 < len(lines) and lines[j + 1].strip():
                j += 1
                lead += " " + lines[j].strip()
            sections[-1][1].append(shorten(lead.split("**")[0]))
    return title, sections


def index_text(skill):
    out = []
    for name in ENTRY_FILES:
        title, sections = parse(read(skill / "references" / name))
        out.append(f"**`references/{name}`** -- {title}")
        out.append("")
        for _, entries in sections:
            out.extend(f"- {e}" for e in entries)
        out.append("")
    return "\n".join(out).rstrip("\n")


def toc_text(text):
    headings = [ln[3:].strip() for ln in text.split("\n") if ln.startswith("## ")]
    if len(headings) < 2:
        headings = [ln[4:].strip() for ln in text.split("\n") if ln.startswith("### ")]
    return "Contents:\n\n" + "\n".join(f"- {h}" for h in headings)


def replace_region(text, marks, body, where):
    begin, end = marks
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise SystemExit(f"{where}: needs exactly one {begin} .. {end} region")
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def render(repo):
    """{path: text it should have} for every file with a generated region."""
    skill = repo / pathlib.Path(*SKILL_REL.parts)
    out = {}
    for name in TOC_FILES:
        path = skill / "references" / name
        text = read(path)
        out[path] = replace_region(text, TOC_MARKS, toc_text(text), name)
    path = skill / "SKILL.md"
    out[path] = replace_region(read(path), INDEX_MARKS, index_text(skill), "SKILL.md")
    return out


def stale(repo=None):
    """Paths (relative to the repository) whose generated region is out of date."""
    repo = repo or repo_root()
    return [p.relative_to(repo).as_posix() for p, text in render(repo).items() if read(p) != text]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when a generated region differs from what is committed")
    a = ap.parse_args()
    repo = repo_root()
    if a.check:
        bad = stale(repo)
        for p in bad:
            print(f"{p}: generated region is stale (run utils/gen_skill_index.py)")
        return 1 if bad else 0
    changed = 0
    for path, text in render(repo).items():
        if read(path) != text:
            path.write_text(text, encoding="utf-8", newline="\n")
            changed += 1
            print(f"updated {path.relative_to(repo).as_posix()}")
    if not changed:
        print("index and contents are current")
    return 0


if __name__ == "__main__":
    import subprocess
    import trunk
    # The trunk export can only re-run a tool the trunk has. Until this file is there,
    # run the checkout's copy instead of failing on a missing script.
    if subprocess.run(["git", "cat-file", "-e", f"{trunk.TRUNK_REF}:utils/{pathlib.Path(__file__).name}"],
                      cwd=str(TOOLS_ROOT), capture_output=True).returncode == 0:
        trunk.reexec(__file__)
    sys.exit(main())
