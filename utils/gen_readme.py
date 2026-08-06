#!/usr/bin/env python3
"""Regenerate the MOAT project table in README.md between sentinel markers,
preserving all hand-written prose outside them. Reads every
projects/*/status.json. Idempotent: same data -> identical bytes.

Usage:
  python3 utils/gen_readme.py            # rewrite README.md
  python3 utils/gen_readme.py --check    # exit 1 if README is stale (CI)"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import moatlib  # noqa: E402

REPO_ROOT = moatlib.REPO_ROOT
README = REPO_ROOT / "README.md"
START = "<!-- MOAT:TABLE:START -->"
END = "<!-- MOAT:TABLE:END -->"
EMPTY = ("_No projects adopted yet. Run `python3 utils/discover.py` then adopt "
         "rows from `data/candidates.json`._")

# Per-platform state -> status glyph (legend is emitted by render_table).
EMOJI = {
    "completed": "✅",
    "validated": "✅",
    "port-ready": "🟡",       # ported at head_sha, not yet validated on this arch
    "revalidate": "🔄",       # was validated here; HEAD moved -> re-check
    "validating": "🔧",
    "review-passed": "🔧",
    "ported": "🔧",
    "delta-ported": "🔧",
    "porting": "🔧",
    "changes-requested": "🔧",
    "validation-failed": "🔧",
    "planned": "🔧",
    "unclaimed": "⬜",
    "awaiting-port": "⬜",          # no port exists yet to validate
    "screened": "⬜",               # passed intake, not yet planned
    "awaiting-fork": "⬜",          # waiting on an org admin to create the fork
    "awaiting-upstream": "⏸",       # viable but parked on an external event
}


def load_projects():
    out = []
    pdir = REPO_ROOT / "projects"
    if not pdir.exists():
        return out
    for d in sorted(pdir.iterdir()):
        sp = d / "status.json"
        if not sp.exists():
            continue
        try:
            rec = json.loads(sp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sys.stderr.write(f"gen_readme: skipping unparseable {sp}\n")
            continue
        # Delivery tracking: PR fields can be in status.json (new workflow) or
        # upstream.json (legacy). status.json takes precedence. pr_state (open/merged/closed)
        # drives the PR glyph; `outcome` records the terminal disposition for
        # projects whose success is NOT an upstream PR (e.g. we GPU-validated an
        # existing ROCm backend across archs). See outcome_cell() for the vocab.

        # Check status.json first (new PR tracking)
        if "pr_url" in rec:
            # PR fields already in rec from status.json
            pass
        else:
            # Fall back to upstream.json (legacy)
            up = d / "upstream.json"
            if up.exists():
                try:
                    u = json.loads(up.read_text(encoding="utf-8"))
                    rec["pr_url"] = u.get("pr_url")
                    rec["pr_number"] = u.get("pr_number")
                    rec["pr_state"] = u.get("pr_state")
                    rec["outcome"] = u.get("outcome")
                    rec["outcome_note"] = u.get("outcome_note")
                except json.JSONDecodeError:
                    pass
        out.append(rec)
    return out


# A gate is proven, or it is not. Ranked best-first: a project with one architecture
# proving a gate and another still working on it shows the proof.
GATE_RANK = ["proven", "stale", "working", "queued", "blocked", "none"]
GATE_GLYPH = {"proven": "✅", "stale": "🔄", "working": "🔧",
              "queued": "⬜", "blocked": "🚫", "none": "—"}
WORKING_STATES = {"porting", "ported", "delta-ported", "review-passed", "validating",
                  "changes-requested", "validation-failed", "planned", "screened"}
QUEUED_STATES = {"port-ready", "awaiting-port", "awaiting-fork", "awaiting-upstream",
                 "unclaimed"}


def gate_state(project, gate):
    """(verdict, arch) for one gate on one project.

    Reads the per-platform records the project already keeps and reports what they
    prove, which is the question the gate asks: has ANY architecture carrying this
    property actually run the tests at the current code? The architecture that did it
    comes back too, since "proven, and here is what proved it" is more use than a
    bare tick."""
    head = project.get("head_sha")
    plats = project.get("platforms") or {}
    if (project.get("waivers") or {}).get(gate, {}).get("approved_by"):
        return ("waived", None)
    best, best_arch = "none", None
    for arch in sorted(plats):          # deterministic pick when several qualify
        blk = plats[arch]
        if gate not in moatlib.gates_for(arch):
            continue
        state = blk.get("state")
        # A completed validation is evidence, and stays evidence even if the platform
        # was later marked blocked -- projects withdrawn on licence grounds carry a
        # block on every platform, and the port really did run. What that means for
        # the contribution is the Outcome column's job, not this one's.
        if state == "completed":
            verdict = "proven" if (not head or blk.get("validated_sha") == head) else "stale"
        elif blk.get("blocked"):
            verdict = "blocked"
        elif state == "revalidate":
            verdict = "stale"
        elif state in WORKING_STATES:
            verdict = "working"
        elif state in QUEUED_STATES:
            verdict = "queued"
        else:
            verdict = "none"
        if GATE_RANK.index(verdict) < GATE_RANK.index(best):
            best, best_arch = verdict, arch
    return (best, best_arch)


def gate_cell(project, gate):
    """Just the glyph. Which card proved a gate is in the project's status file; a
    column of architecture names is noise in a table 152 rows long, and the gate is
    the claim being made."""
    verdict, _ = gate_state(project, gate)
    return "⚖️" if verdict == "waived" else GATE_GLYPH[verdict]


def _validated_arch_count(p):
    """Number of platforms validated on real hardware (completed, not blocked)."""
    return sum(1 for b in p.get("platforms", {}).values()
               if b.get("state") == "completed" and not b.get("blocked"))


def outcome_cell(p):
    """The Outcome column: what this project actually delivered. An upstream PR
    (any state) is shown by its glyph + number. Projects without a PR carry an
    explicit `outcome` in upstream.json:
      validated  -- upstream already had a ROCm path; we GPU-validated it across
                    N archs (often extending coverage, e.g. first CDNA). 🔵
      fork       -- delivered as a working standalone fork; an upstream PR is not
                    appropriate (e.g. a kernel-experiments repo). 🍴
      superseded -- upstream/community already covers our archs; no value-add. ⚪
      blocked    -- non-viable. ⛔
      license-blocked -- the port works, but the upstream license (non-commercial,
                    no-derivative, or otherwise incompatible) bars contributing it.
                    Nothing we could do about it; the platform cells stay truthful
                    (the port was built/validated) and the outcome carries ⚖️.
    No PR and no outcome yet -> pending (—)."""
    if p.get("pr_url"):
        # pr_state is the authority: the PR lifecycle is one project-level fact, not
        # something an arch's validation record carries. pr_merged_at backs it up for
        # any record written before that was true.
        state = p.get("pr_state") or ("merged" if p.get("pr_merged_at") else "open")
        glyph = {"merged": "🟣", "closed": "🔴"}.get(state, "🟢")
        num = p.get("pr_number")
        if num is None:  # derive from the .../pull/<n> URL tail when not recorded
            tail = p["pr_url"].rstrip("/").rsplit("/", 1)[-1]
            num = tail if tail.isdigit() else "?"
        return f"{glyph} [#{num}]({p['pr_url']})"
    oc = p.get("outcome")
    if oc == "validated":
        n = _validated_arch_count(p)
        return f"🔵 validated ({n} arch)" if n else "🔵 validated"
    if oc == "fork":
        fu = p.get("fork_url")
        return f"🍴 [fork]({fu}/tree/{p.get('fork_branch') or moatlib.PORT_BRANCH})" if fu else "🍴 fork"
    if oc == "superseded":
        return "⚪ superseded"
    if oc == "blocked":
        return "⛔ blocked"
    if oc == "license-blocked":
        # First sentence only: the full reasoning lives in upstream.json, and a
        # paragraph inside a table cell wraps the row into unreadability.
        note = (p.get("outcome_note") or "").split(".")[0].strip()
        return f"⚖️ license-restricted -- {note}" if note else "⚖️ license-restricted"
    return "—"


def render_table(projects):
    if not projects:
        return EMPTY
    # Alphabetical by name (case-insensitive) so the table is a lookup-by-name
    # reference; the per-row glyphs still convey status. (Popularity/priority order
    # buried manually-adopted projects at priority 0 and told no progress story.)
    projects = sorted(projects, key=lambda p: p.get("name", "").lower())
    # Three short blocks beat one long sentence: what the columns mean, then a key
    # for each kind of cell. Someone scanning for "what is wave64" should find it on
    # its own line rather than in the middle of a paragraph.
    gates = list(moatlib.REQUIRED_GATES)
    legend = "\n".join([
        "**Coverage is proven per gate, not per machine.** A gate is met once *any* AMD GPU",
        "with that property has run the project's real test suite. Which specific card did it",
        "is recorded in each project's status file.",
        "",
        "| gate | what it covers |",
        "|---|---|",
        "| `wave64` | data-center cards -- Instinct MI200/MI300 class (CDNA), 64 threads per wavefront |",
        "| `wave32` | desktop and workstation cards -- Radeon RX / PRO (RDNA), 32 threads per wavefront |",
        "| `windows` | the port builds and runs on Windows, not only Linux |",
        "",
        "A wavefront is AMD's analogue of an NVIDIA warp, which is always 32 threads. Code that",
        "silently assumes 32 is the single most common way a CUDA port breaks on AMD, so the two",
        "widths are proven separately rather than assumed to follow from each other.",
        "",
        "| cell | meaning | | outcome | meaning |",
        "|---|---|---|---|---|",
        "| ✅ | proven on the current code | | 🟣 | contribution merged upstream |",
        "| 🔄 | proven earlier; the code has moved since | | 🟢 | pull request open |",
        "| 🔧 | in progress | | 🔴 | pull request closed |",
        "| ⬜ | not started | | 🔵 | upstream already supported AMD; we verified it |",
        "| 🚫 | blocked, with a reason recorded | | 🍴 | delivered as a fork; upstream PR not appropriate |",
        "| ⚖️ | not required for this project | | ⚖️ | licence bars contributing the port |",
        "| — | nothing recorded | | ⚪ | superseded by other work |",
        "",
        "The project name links upstream.",
    ])
    headers = ["Project"] + [f"`{g}`" for g in gates] + ["Outcome"]
    aligns = ["---"] + [":---:"] * len(gates) + ["---"]
    lines = [legend, "",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join(aligns) + " |"]
    for p in projects:
        name = p.get("name", "?")
        up = p.get("upstream_url")
        if p.get("fork_url"):
            proj = f"[{name}]({up}) ([fork]({p['fork_url']}/tree/{p.get('fork_branch') or moatlib.PORT_BRANCH}))"
        else:
            proj = f"[{name}]({up})"
        plats = p.get("platforms", {})
        cells = [gate_cell(p, g) for g in gates]
        lines.append("| " + " | ".join([proj] + cells + [outcome_cell(p)]) + " |")
    return "\n".join(lines)


def splice(readme_text, body):
    if START not in readme_text or END not in readme_text:
        raise SystemExit(f"gen_readme: README.md is missing {START} / {END} markers")
    head = readme_text[:readme_text.index(START) + len(START)]
    tail = readme_text[readme_text.index(END):]
    return f"{head}\n{body}\n{tail}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gen_readme")
    ap.add_argument("--check", action="store_true", help="exit 1 if README is stale")
    args = ap.parse_args(argv)
    current = README.read_text(encoding="utf-8")
    new = splice(current, render_table(load_projects()))
    if args.check:
        if new != current:
            sys.stderr.write("gen_readme: README.md is out of date (run gen_readme.py)\n")
            return 1
        print("README.md table is up to date")
        return 0
    README.write_text(new, encoding="utf-8")
    print("README.md table regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
