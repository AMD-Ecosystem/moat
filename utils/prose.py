#!/usr/bin/env python3
"""Catch hard-wrapped, non-ASCII, or ROCm-miscased prose before it reaches GitHub.

GitHub reflows markdown to the reader's width. Text wrapped by hand at 80 columns
renders with the author's line breaks frozen in, which looks broken on a wide screen
and worse on a phone, and it makes every later edit reflow a whole paragraph so the
diff is unreadable.

AGENTS.md has said "do not manually line-wrap GitHub or markdown prose" for a long
time and agents kept doing it anyway -- including the run that added this file. A
rule that lives only in prose is one nobody checks, so this is the check.

    python3 utils/prose.py <file>...     # exit 1 if anything is hard-wrapped
    python3 utils/prose.py -             # read stdin

Scope is deliberately narrow: text WE author that GitHub renders -- pull request and
issue bodies. It is not run over the repo's own markdown, which is wrapped
throughout and is read in an editor as often as on the web.
"""

import re
import sys

# Below this, a line break is plausibly deliberate (a short heading, a table cell, a
# one-line list item). Above it, a break mid-paragraph means someone wrapped by hand.
WRAP_SUSPECT = 60

LIST_MARKERS = ("- ", "* ", "+ ", "> ", "| ")


def _is_new_block(line):
    """A line that legitimately starts its own rendered block."""
    t = line.lstrip()
    if not t:
        return True
    if t.startswith(LIST_MARKERS) or t.startswith("|") or t.startswith("#"):
        return True
    # ordered list: "1. ", "12) "
    head = t.split(" ", 1)[0]
    return head[:-1].isdigit() and head[-1:] in (".", ")")


def hard_wrapped(text):
    """[(line number, text)] for lines that continue a long previous line.

    The signal is a paragraph broken across lines: a line that is not the start of a
    new block, following a line long enough that the break was not a choice. Fenced
    code is exempt, since line breaks are the content there."""
    out = []
    fenced = False
    prev = ""
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            prev = ""
            continue
        if fenced:
            continue
        if line.strip() and prev.strip() and not _is_new_block(line) \
                and len(prev.rstrip()) >= WRAP_SUSPECT:
            out.append((i, line.strip()))
        prev = line
    return out


# Standalone rocm/Rocm/ROCM in prose; "ROCm" is the platform's casing. The
# lookarounds keep identifiers and paths out of it: USE_ROCM (preceded by _),
# rocm-smi and rocminfo (followed by - or letters), /opt/rocm and URLs
# (preceded by /), version dots. Inline code spans are stripped first and
# fenced blocks skipped -- code is content, not prose.
ROCM_MISCASED = re.compile(r"(?<![\w/.\-])([Rr][Oo][Cc][Mm])(?![\w.\-])")


def _prose_lines(text):
    """(line number, line with inline code removed) for non-fenced lines."""
    fenced = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            yield i, re.sub(r"`[^`]*`", "", line)


def non_ascii(text):
    """[(line number, [offending chars])] -- ASCII only, and no em-dash."""
    out = []
    for i, line in _prose_lines(text):
        bad = sorted({ch for ch in line if ord(ch) > 127})
        if bad:
            out.append((i, bad))
    return out


def miscased_rocm(text):
    """[(line number, word)] for rocm spelled any way but ROCm."""
    return [(i, m.group(1)) for i, line in _prose_lines(text)
            for m in ROCM_MISCASED.finditer(line) if m.group(1) != "ROCm"]


def check(text, label="text"):
    """[] or one-line problems, shaped for a gate's problem list."""
    problems = []
    hits = hard_wrapped(text)
    if hits:
        where = ", ".join(f"line {n}" for n, _ in hits[:4])
        problems.append(
            f"{label} is hard-wrapped ({where}) -- GitHub reflows markdown, so write "
            f"each paragraph as one line and let it wrap for the reader")
    na = non_ascii(text)
    if na:
        where = ", ".join(f"line {n} ({', '.join(repr(c) for c in chars[:3])})"
                          for n, chars in na[:4])
        problems.append(
            f"{label} has non-ASCII characters ({where}) -- ASCII only; write -- "
            f"rather than an em-dash")
    mc = miscased_rocm(text)
    if mc:
        where = ", ".join(f"line {n} ({w!r})" for n, w in mc[:4])
        problems.append(f"{label} miscases ROCm ({where}) -- the platform is "
                        f"written 'ROCm'")
    return problems


def main(argv=None):
    args = (argv if argv is not None else sys.argv[1:]) or ["-"]
    problems = []
    for a in args:
        text = sys.stdin.read() if a == "-" else open(a, encoding="utf-8").read()
        problems += check(text, a)
    for p in problems:
        print(p)
    if not problems:
        print("prose: clean")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
