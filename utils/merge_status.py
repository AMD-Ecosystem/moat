#!/usr/bin/env python3
"""git merge driver for projects/*/status.json (merge=moat-status in
.gitattributes). Deterministic so concurrent CLIs never hard-conflict.

git invokes: merge_status.py <ancestor O> <ours A> <theirs B> [path P]
We write the merged result into the OURS file (A), which git uses as the merge
result, and exit 0. Strategy: latest-writer-wins per platform block (by the
block's updated_at), max / most-advanced for top-level scalars. The ancestor is
not needed: per-platform state is monotonic, so latest-writer-wins converges."""

import json
import sys

PR_RANK = {None: 0, "none": 0, "approved": 1, "open": 2, "merged": 3}


def _load(p):
    with open(p) as f:
        return json.load(f)


def _b_is_later(a_ts, b_ts):
    # ISO-8601 Z strings sort lexicographically; ties resolve to B (deterministic).
    return (b_ts or "") >= (a_ts or "")


def merge(a, b):
    out = dict(a)  # start from ours
    a_top, b_top = a.get("updated_at") or "", b.get("updated_at") or ""
    newer = b if b_top > a_top else a
    out["updated_at"] = max(a_top, b_top)
    out["head_sha"] = newer.get("head_sha", out.get("head_sha"))
    if "pr_state" in a or "pr_state" in b:
        out["pr_state"] = max((a.get("pr_state"), b.get("pr_state")),
                              key=lambda s: PR_RANK.get(s, 0))
    if a.get("pr_url") or b.get("pr_url"):
        out["pr_url"] = newer.get("pr_url") or a.get("pr_url") or b.get("pr_url")
    pa, pb = a.get("platforms", {}), b.get("platforms", {})
    merged = {}
    for plat in set(pa) | set(pb):
        ba, bb = pa.get(plat), pb.get(plat)
        if ba is None:
            merged[plat] = bb
        elif bb is None:
            merged[plat] = ba
        else:
            merged[plat] = bb if _b_is_later(ba.get("updated_at"), bb.get("updated_at")) else ba
    out["platforms"] = merged
    return out


def main(argv):
    # argv: O A B [P]; we only need ours (A) and theirs (B).
    if len(argv) < 3:
        sys.stderr.write("merge_status: expected O A B [P]\n")
        return 2
    a_path, b_path = argv[1], argv[2]
    try:
        merged = merge(_load(a_path), _load(b_path))
    except Exception as e:  # fall back to a real conflict rather than guessing
        sys.stderr.write(f"merge_status: {e}; leaving conflict\n")
        return 1
    with open(a_path, "w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
