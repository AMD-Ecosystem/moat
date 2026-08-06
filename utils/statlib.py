#!/usr/bin/env python3
"""Aggregate projects/<name>/stats.jsonl into the numbers the README and a
future blog want. Compile/test wall-clock is accurate. Thinking time is a
residual (unattributed wall: model latency plus human-in-the-loop), not pure
model thinking. Tokens are approximate. Nothing here fabricates a number;
approximate values carry approx=True."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_records(project):
    p = REPO_ROOT / "projects" / project / "stats.jsonl"
    recs = []
    if not p.exists():
        return recs
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def wall_by_phase(recs):
    out = {"compile": 0.0, "test": 0.0, "misc": 0.0}
    for r in recs:
        if r.get("kind") == "phase":
            ph = r.get("phase", "misc")
            out[ph] = out.get(ph, 0.0) + float(r.get("seconds", 0))
    return out


def session_wall(recs):
    """Sum matched session start/end pairs using their epoch timestamps."""
    total = 0.0
    open_start = None
    sess = sorted((r for r in recs if r.get("kind") == "session"),
                  key=lambda r: r.get("ts", ""))
    for r in sess:
        if r.get("event") == "start":
            open_start = r.get("epoch")
        elif r.get("event") == "end" and open_start is not None:
            if r.get("epoch") is not None:
                total += max(0.0, float(r["epoch"]) - float(open_start))
            open_start = None
    return total


def tokens(recs):
    total, source = 0, None
    for r in recs:
        if r.get("kind") == "tokens":
            total += int(r.get("tokens", 0))
            source = r.get("source", source)
    return total, True, source  # always approximate


def aggregate(project):
    recs = read_records(project)
    phase = wall_by_phase(recs)
    swall = session_wall(recs)
    thinking = max(0.0, swall - sum(phase.values()))
    tok, approx, source = tokens(recs)
    return {
        "tokens_total": tok,
        "tokens_approx": approx,
        "tokens_source": source,
        "wall_seconds": {
            "thinking": round(thinking, 1),
            "compile": round(phase.get("compile", 0.0), 1),
            "test": round(phase.get("test", 0.0), 1),
            "misc": round(phase.get("misc", 0.0), 1),
        },
        "session_count": sum(1 for r in recs
                             if r.get("kind") == "session" and r.get("event") == "start"),
    }


def main(argv):
    if not argv:
        sys.stderr.write("usage: statlib.py <project>\n")
        return 2
    json.dump(aggregate(argv[0]), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
