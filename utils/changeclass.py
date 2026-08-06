#!/usr/bin/env python3
"""Source-class classifier for the MOAT regression guard.

Given two commits on a fork, decide whether the source delta is behavior-
preserving without building anything. The verdict drives cross-platform
carry-forward in moatlib.advance_head: a delta proven inert at the source level
need not re-trigger GPU validation on platforms that already passed.

Three classes are recognized, in increasing risk:

  doc-only     -- only documentation files changed (*.md, *.rst, docs/**, LICENSE).
  comment-only -- code files changed, but after dropping comments and whitespace
                  the token stream (identifiers included) is byte-identical.
  rename-only  -- token stream identical up to a consistent bijection on
                  identifier tokens (a project-wide symbol rename / reformat).

doc-only and comment-only are ARCH-INDEPENDENT inert: they cannot change any
compiled instruction stream on any target, so every passed platform carries
forward. (Exception: a comment edit that shifts line numbers in a file that
references __LINE__/__FILE__/assert is downgraded -- the embedded line number is
a real, if tiny, behavior surface; let the per-arch binary compare decide.)

rename-only is NOT treated as arch-independent here: a renamed *exported* symbol
(a pybind entry, an extern "C" name) changes behavior with an identical
instruction stream, so it must be confirmed by the binary compare on each arch
(which preserves the dynamic symbol table). The validator does that for arches it
can build; unbuildable arches revalidate on a rename.

Anything not cleanly in one of these classes is `mixed` -> full revalidation.
The classifier is deliberately conservative: any parse uncertainty yields `mixed`.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field

DOC_SUFFIXES = (".md", ".rst")
DOC_NAMES = ("LICENSE", "NOTICE", "COPYING", "AUTHORS", "CHANGELOG")

# VCS/repo-meta files: never a build input, so a change to them cannot alter any
# compiled output on any target. Treated as inert like documentation.
REPO_META_NAMES = (".gitignore", ".gitattributes", ".dockerignore",
                   ".editorconfig", ".mailmap", ".gitmodules")

# CI / continuous-integration config. MOAT validates by building and running the
# project directly, never through the project's CI, so a CI-config change (and in
# particular stripping a stray smoketest workflow before a PR) cannot affect the
# validated artifact. Treated as inert. Detected by path (.github/, .circleci/) or
# basename (top-level CI files).
CI_NAMES = (".travis.yml", ".gitlab-ci.yml", "appveyor.yml", "azure-pipelines.yml",
            ".cirrus.yml", "jenkinsfile")

C_FAMILY = (".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".hxx",
            ".hip", ".inl", ".inc")
CMAKE = ("cmakelists.txt",)  # by basename; plus .cmake by suffix
CMAKE_SUFFIX = (".cmake",)
PYTHON = (".py", ".pyx", ".pxd")

LINE_MACROS = ("__LINE__", "__FILE__", "assert(", "static_assert", "__PRETTY_FUNCTION__")

# Keywords kept distinct from renameable identifiers so a rename can never
# silently turn an identifier into a keyword (or vice versa). C-family superset
# (CUDA/HIP qualifiers included); harmless extras do not affect correctness.
C_KEYWORDS = frozenset("""
alignas alignof and and_eq asm auto bitand bitor bool break case catch char
char16_t char32_t class compl const constexpr const_cast continue decltype
default delete do double dynamic_cast else enum explicit export extern false
float for friend goto if inline int long mutable namespace new noexcept not
not_eq nullptr operator or or_eq private protected public register
reinterpret_cast return short signed sizeof static static_assert static_cast
struct switch template this thread_local throw true try typedef typeid typename
union unsigned using virtual void volatile wchar_t while xor xor_eq
__global__ __device__ __host__ __shared__ __constant__ __restrict__ __forceinline__
__launch_bounds__ __managed__ uint8_t uint16_t uint32_t uint64_t int8_t int16_t
int32_t int64_t size_t
""".split())

PY_KEYWORDS = frozenset("""
False None True and as assert async await break class continue def del elif else
except finally for from global if import in is lambda nonlocal not or pass raise
return try while with yield match case
""".split())


@dataclass
class Verdict:
    cls: str  # "doc-only" | "comment-only" | "rename-only" | "mixed"
    arch_independent: bool  # inert on every target, no build needed to prove
    detail: str
    files: list = field(default_factory=list)

    @property
    def inert(self) -> bool:
        return self.cls in ("doc-only", "comment-only", "rename-only")


def _run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True).stdout


def _blob(repo, sha, path):
    try:
        return _run(["git", "show", f"{sha}:{path}"], repo)
    except subprocess.CalledProcessError:
        return None


def _is_doc(path: str) -> bool:
    low = path.lower()
    base = path.rsplit("/", 1)[-1]
    if low.endswith(DOC_SUFFIXES):
        return True
    if path.startswith("docs/") or path.startswith("doc/") or "/docs/" in path:
        return True
    if base in REPO_META_NAMES:
        return True
    if path.startswith(".github/") or "/.github/" in path \
            or path.startswith(".circleci/") or "/.circleci/" in path:
        return True
    if base.lower() in CI_NAMES:
        return True
    return any(base.startswith(n) for n in DOC_NAMES)


def _lang(path: str):
    low = path.lower()
    base = low.rsplit("/", 1)[-1]
    if low.endswith(C_FAMILY):
        return "c"
    if base in CMAKE or low.endswith(CMAKE_SUFFIX):
        return "cmake"
    if low.endswith(PYTHON):
        return "py"
    return None


# --- tokenizers -------------------------------------------------------------
# Each returns a list of (kind, text) with whitespace dropped, or None if the
# input cannot be confidently tokenized (-> caller treats the file as mixed).
# kinds: "id", "kw", "num", "str", "punct", "comment".

def _tok_c(src: str):
    toks = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in " \t\r\n":
            i += 1
            continue
        two = src[i:i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            toks.append(("comment", src[i:j]))
            i = j
            continue
        if two == "/*":
            j = src.find("*/", i + 2)
            if j < 0:
                return None
            toks.append(("comment", src[i:j + 2]))
            i = j + 2
            continue
        if ch == '"' or ch == "'":
            # Raw string R"delim(...)delim" -- bail to be safe.
            if ch == '"' and i >= 1 and src[i - 1] in "Rr":
                return None
            q = ch
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == q:
                    break
                j += 1
            if j >= n:
                return None
            toks.append(("str", src[i:j + 1]))
            i = j + 1
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            toks.append(("kw" if word in C_KEYWORDS else "id", word))
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] in ".+-_'"):
                # consume exponent sign only right after e/E/p/P
                if src[j] in "+-" and src[j - 1] not in "eEpP":
                    break
                j += 1
            toks.append(("num", src[i:j]))
            i = j
            continue
        toks.append(("punct", ch))
        i += 1
    return toks


def _tok_line_hash(src: str, kws, multiline_str: bool = False):
    """Shared tokenizer for # line-comment languages (CMake, Python).

    Strings are treated atomically. multiline_str=True lets a double-quoted
    string span newlines (CMake quoted arguments do; Python single/double
    quotes do not -- only its triple-quotes, handled separately). Python
    triple-quotes and CMake bracket args/comments are not fully modeled; their
    delimiters are handled, and anything ambiguous returns None (-> mixed)."""
    toks = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == "#":
            j = src.find("\n", i)
            j = n if j < 0 else j
            toks.append(("comment", src[i:j]))
            i = j
            continue
        if ch in "\"'":
            triple = src[i:i + 3]
            if triple in ('"""', "'''"):
                j = src.find(triple, i + 3)
                if j < 0:
                    return None
                toks.append(("str", src[i:j + 3]))
                i = j + 3
                continue
            q = ch
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == q:
                    break
                if src[j] == "\n" and not multiline_str:
                    break
                j += 1
            if j >= n or src[j] != q:
                return None
            toks.append(("str", src[i:j + 1]))
            i = j + 1
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            toks.append(("kw" if word in kws else "id", word))
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "."):
                j += 1
            toks.append(("num", src[i:j]))
            i = j
            continue
        toks.append(("punct", ch))
        i += 1
    return toks


def _tokenize(lang: str, src: str):
    if lang == "c":
        return _tok_c(src)
    if lang == "cmake":
        # CMake commands are not reserved; quoted "..." args may span newlines.
        return _tok_line_hash(src, frozenset(), multiline_str=True)
    if lang == "py":
        return _tok_line_hash(src, PY_KEYWORDS)
    return None


def _strip_comments(toks):
    return [t for t in toks if t[0] != "comment"]


def _line_hazard(lang, old_src, new_src):
    """A comment/format change that alters line count in a C file referencing
    __LINE__/assert can perturb the binary -- not arch-independent."""
    if lang != "c":
        return False
    if old_src.count("\n") == new_src.count("\n"):
        return False
    return any(m in new_src or m in old_src for m in LINE_MACROS)


def _classify_file(lang, old_src, new_src):
    """Return ('comment-only'|'rename-only'|'mixed', arch_independent, detail)."""
    told = _tokenize(lang, old_src)
    tnew = _tokenize(lang, new_src)
    if told is None or tnew is None:
        return "mixed", False, "untokenizable"
    sold = _strip_comments(told)
    snew = _strip_comments(tnew)
    if len(sold) != len(snew):
        return "mixed", False, "token count differs"
    # Identical code token stream (identifiers included) -> comment/format only.
    if all(a == b for a, b in zip(sold, snew)):
        haz = _line_hazard(lang, old_src, new_src)
        return "comment-only", (not haz), ("line-shift hazard" if haz else "comments/format only")
    # Otherwise: identical up to a consistent identifier bijection -> rename.
    fwd, rev = {}, {}
    for (ka, ta), (kb, tb) in zip(sold, snew):
        if ka != kb:
            return "mixed", False, "token kind differs"
        if ka == "id":
            if fwd.setdefault(ta, tb) != tb or rev.setdefault(tb, ta) != ta:
                return "mixed", False, "inconsistent identifier mapping"
        elif ta != tb:
            return "mixed", False, "literal token differs"
    return "rename-only", False, f"consistent rename of {len(fwd)} identifier(s)"


def classify(repo: str, old_sha: str, new_sha: str) -> Verdict:
    """Classify the delta old_sha->new_sha in the git repo at `repo`."""
    try:
        names = _run(["git", "diff", "--name-status", "-M", old_sha, new_sha], repo)
    except subprocess.CalledProcessError as e:
        return Verdict("mixed", False, f"git diff failed: {e.stderr.strip()[:120]}")

    files = []
    worst = "doc-only"        # promote toward mixed
    arch_indep = True
    rank = {"doc-only": 0, "comment-only": 1, "rename-only": 2, "mixed": 3}
    details = []

    for line in names.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        path = parts[-1]
        files.append(path)

        if status.startswith("R") and len(parts) >= 3:
            # File rename: inert only if it is a doc file; otherwise the build
            # graph / includes can shift -> mixed.
            old_path = parts[1]
            if _is_doc(old_path) and _is_doc(path):
                fcls = "doc-only"
            else:
                fcls = "mixed"
            details.append(f"{path}: rename {fcls}")
        elif status in ("A", "D"):
            # Add/delete: inert only for doc files; any source add/delete is code.
            fcls = "doc-only" if _is_doc(path) else "mixed"
            details.append(f"{path}: {status} {fcls}")
        else:  # modified
            if _is_doc(path):
                fcls = "doc-only"
            else:
                lang = _lang(path)
                if lang is None:
                    fcls, ai, d = "mixed", False, "unknown file type"
                    details.append(f"{path}: {d}")
                else:
                    old_src = _blob(repo, old_sha, path) or ""
                    new_src = _blob(repo, new_sha, path) or ""
                    fcls, ai, d = _classify_file(lang, old_src, new_src)
                    details.append(f"{path}: {fcls} ({d})")
                    if fcls in ("comment-only", "rename-only") and not ai:
                        arch_indep = False
        if rank[fcls] > rank[worst]:
            worst = fcls

    # arch-independence holds only for the doc-only/comment-only classes with no
    # per-file hazard; rename-only and mixed are never arch-independent here.
    if worst in ("rename-only", "mixed"):
        arch_indep = False

    return Verdict(worst, arch_indep and worst in ("doc-only", "comment-only"),
                   "; ".join(details) if details else "no changes", files)


def main(argv):
    if len(argv) != 4:
        print("usage: changeclass.py <repo_dir> <old_sha> <new_sha>", file=sys.stderr)
        return 2
    v = classify(argv[1], argv[2], argv[3])
    print(f"class={v.cls} arch_independent={v.arch_independent} inert={v.inert}")
    print(v.detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
