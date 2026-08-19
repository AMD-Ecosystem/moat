#!/usr/bin/env python3
"""git merge driver for projects/*/status.json (merge=moat-status in
.gitattributes). Deterministic so concurrent CLIs never hard-conflict.

git invokes: merge_status.py <ancestor O> <ours A> <theirs B> [path P]
We write the merged result into the OURS file (A), which git uses as the merge
result, and exit 0.

The default is LATEST-WRITER-WINS per field, decided by the record's top-level
updated_at. That default matters more than any of the rules below: this driver
used to start from `out = dict(a)` and then reconcile five named fields, so the
other twenty-five silently took OURS. A concurrent merge could drop an approval
snapshot, a licence clearance, a person's intake decision, the whole PR
lifecycle, or the fork-write lock -- and the lock is the one that fails OPEN,
letting a second arch start writing the fork. Recency is symmetric and drops
nothing on its own, and a field added to the schema later inherits it without
anyone remembering to come back here.

On top of that default, three kinds of field get a rule, because recency alone
would lose real information:

  STICKY   a value that RECORDS A DECISION -- an approval, a clearance, a PR
           number. A writer who simply never had it must not erase it, so a
           non-null beats a null regardless of which is newer. Deliberate
           removal is rare and is a human editing the file, not a merge.
  UNION    per-key maps (platforms, waivers) where two hosts legitimately write
           different keys at the same time. Merge key by key, latest writer wins
           within a key.
  RANKED   pr_state, which advances through a lifecycle. See _pr_rank.

  RESET    a field cleared by REMOVING it, where absence is the new value rather
           than a writer who never had it. `set-hold off` pops on_hold, so it is
           taken from the newer record whether or not it still has it. The
           fork-write lock used to release this way too; it now releases through
           an explicit `porting_released` marker instead -- see _merge_lock.

Everything else is recency: for those the current value is the point, and a key
missing from one side means that writer never had it rather than deleted it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from moatlib import PR_STATES
except Exception:  # a merge must still resolve if moatlib will not import
    PR_STATES = ("open", "merged", "closed")

# open -> {merged, closed}; both ends are terminal and mutually exclusive, so a
# tie between them resolves by recency rather than by an invented ordering. The
# old table here predated PR_STATES: it ranked a "approved" state that no longer
# exists and omitted "closed", which therefore ranked below "open" -- so a
# recorded close was undone by the next merge from a host that had not seen it.
_TERMINAL_PR = {"merged", "closed"}
_PR_RANKABLE = set(PR_STATES) == {"open"} | _TERMINAL_PR

STICKY = ("upstream_repo_id", "fork_url", "review_pr", "pr_approval",
          "license_clearance", "license_spdx", "pr_number", "pr_url",
          "pr_opened_at", "pr_merged_at", "pr_closed_at", "pr_closed_note",
          "adopted_at")
UNION = ("platforms", "waivers")
RESET = ("on_hold", "on_hold_reason")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _b_is_later(a_ts, b_ts):
    # ISO-8601 Z strings sort lexicographically; ties resolve to B (deterministic).
    return (b_ts or "") >= (a_ts or "")


def _pr_rank(s):
    return 0 if not s or s == "none" else (2 if s in _TERMINAL_PR else 1)


def _union(pa, pb):
    """Per-key merge of two maps, latest writer winning within a key. Blocks carry
    their own updated_at; anything without one resolves to B."""
    out = {}
    for k in set(pa) | set(pb):
        va, vb = pa.get(k), pb.get(k)
        if va is None or vb is None:
            out[k] = vb if va is None else va
        elif isinstance(va, dict) and isinstance(vb, dict):
            out[k] = vb if _b_is_later(va.get("updated_at"), vb.get("updated_at")) else va
        else:
            out[k] = vb
    return out


def _merge_lock(a, b):
    """Reconcile the fork-write lock. FIRST ACQUIRE WINS, deterministically.

    This is the one field where latest-writer-wins is actively wrong, and the
    reason is worth stating: two hosts that acquire concurrently do NOT get a
    push conflict. status.json carries `merge=moat-status` precisely so it never
    hard-conflicts, so the textual conflict git would otherwise raise on the same
    line is resolved by this driver instead -- and under recency the SECOND host
    to push silently takes a lock the first already held.

    So the tie is broken here, on the lock's own `since`: the earliest acquisition
    wins, ties broken on arch name so every host computes the same winner from the
    same pair. That makes the loser discoverable -- it re-reads after pushing, sees
    an arch that is not its own, and backs off -- which is what actually delivers
    mutual exclusion. Without it the guard in `actionable` only ever sees a lock
    that agrees with whoever asked last.

    One side holding no lock used to be read through timestamp inference -- a
    lock-less record newer than the lock's `since` counted as a release -- and a
    record written CONCURRENTLY with the acquisition satisfies that test without
    meaning it. On 2026-08-13 the inference erased two live locks (rmcl,
    LC-framework) and two hosts ran the same round. Releases now leave an
    explicit marker: `porting_released`, written by moatlib naming the ended
    acquisition's arch and `since`. A lock dies here only when a marker on
    EITHER side matches it exactly; an unmatched lock survives no matter how new
    the lock-less record is. A release written by tooling that predates the
    marker leaves the lock held until the holder's next transition or a person's
    `port-lock --release` -- failing closed, where the inference failed open."""
    rel = [r for r in (a.get("porting_released"), b.get("porting_released")) if r]

    def _live(lock):
        return lock and not any(r.get("arch") == lock.get("arch") and
                                r.get("since") == lock.get("since") for r in rel)

    la = a.get("porting") if _live(a.get("porting")) else None
    lb = b.get("porting") if _live(b.get("porting")) else None
    if la and lb:
        if la.get("arch") == lb.get("arch"):
            return lb if _b_is_later(la.get("since"), lb.get("since")) else la
        return min((la, lb), key=lambda l: (l.get("since") or "", l.get("arch") or ""))
    return la or lb


def _merge_intake(a, b, newer):
    """intake carries two different things: a recommendation an agent wrote, and
    `decided`, a person's answer to it. Take the newer recommendation, but keep
    whichever side actually has the decision -- a host still holding the
    pre-decision record would otherwise erase the answer."""
    ia, ib = a.get("intake"), b.get("intake")
    if not ia or not ib:
        return ib or ia
    out = dict(newer.get("intake") or {})
    decided = (ia.get("decided"), ib.get("decided"))
    if any(decided):
        da, db = decided
        out["decided"] = (db if da is None else
                          da if db is None else
                          db if _b_is_later(da.get("at"), db.get("at")) else da)
    return out


def merge(a, b):
    a_top, b_top = a.get("updated_at") or "", b.get("updated_at") or ""
    newer, older = (b, a) if b_top > a_top else (a, b)
    out = dict(older)
    out.update(newer)                       # default: latest writer wins per field
    out["updated_at"] = max(a_top, b_top)

    for k in STICKY:
        if out.get(k) is None:
            out[k] = a.get(k) if a.get(k) is not None else b.get(k)
        if k in out and out[k] is None:
            del out[k]

    for k in RESET:
        out.pop(k, None)
        if k in newer:
            out[k] = newer[k]

    if "porting" in a or "porting" in b:
        lock = _merge_lock(a, b)
        if lock is None:
            out.pop("porting", None)
        else:
            out["porting"] = lock
    # The newest release marker wins regardless of which record is newer overall:
    # a marker is an event, and the older record can carry the later release. The
    # loser only ever named an acquisition that is already gone from both sides.
    ra, rb = a.get("porting_released"), b.get("porting_released")
    if ra or rb:
        out["porting_released"] = \
            rb if not ra else ra if not rb else \
            rb if _b_is_later(ra.get("at"), rb.get("at")) else ra

    for k in UNION:
        if k in a or k in b:
            out[k] = _union(a.get(k) or {}, b.get(k) or {})

    if "intake" in a or "intake" in b:
        out["intake"] = _merge_intake(a, b, newer)

    if "pr_state" in a or "pr_state" in b:
        sa, sb = a.get("pr_state"), b.get("pr_state")
        if _PR_RANKABLE and _pr_rank(sa) != _pr_rank(sb):
            out["pr_state"] = sa if _pr_rank(sa) > _pr_rank(sb) else sb
        else:
            out["pr_state"] = newer.get("pr_state") or sa or sb
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
