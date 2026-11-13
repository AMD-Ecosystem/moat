#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""git merge driver for projects/*/status.json (merge=moat-status in
.gitattributes). Deterministic so concurrent CLIs never hard-conflict.

git invokes: merge_status.py <ancestor O> <ours A> <theirs B> [path P]
We write the merged result into the OURS file (A), which git uses as the merge
result, and exit 0.

A field only ONE side changed since the common ancestor is that side's (see
merge()). For a field both sides changed, and whenever git supplies no ancestor,
the default is LATEST-WRITER-WINS per field, decided by the record's top-level
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

One rule sits above all of those for the PR-lifecycle fields: `pr_round`. A
follow-up pull request (archive_pr in moatlib) moves a finished PR into
`pr_history` and starts the lifecycle over, and every per-field rule here would
resurrect the finished round from a host still carrying it -- STICKY would
restore the archived pr_merged_at over the new round's absence, and RANKED would
rank the archived "merged" above the new round's "open". So when the two sides
disagree on pr_round, the higher round supplies the ENTIRE PR block verbatim,
absences included, and the per-field rules apply only within a round. The loser
cannot lose information: a lower round's current PR is, by construction, an
entry in the higher round's pr_history.

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

# `maintained` is STICKY, not RESET like on_hold, though both are person-recorded
# standing rulings that a person clears: under RESET, any newer write from a host
# whose copy predates the ruling would silently erase it, and an erased ruling
# re-arms the upstream-PR machinery against a maintainer's explicit no. A
# resurrected clear (the STICKY failure mode) just needs re-clearing.
# `no_upstream` is STICKY for the same reason: erasing it re-lists a port for a
# pull request its archived upstream cannot take.
# `base_branch_ruling` is STICKY and, where it stands, decides fork_default_branch:
# a host whose copy predates the ruling would otherwise restore the old PR base
# just by writing something newer.
# `delivery` and `release` are STICKY: erasing the first would re-arm the upstream
# publish path against a person's ruling, and erasing the second would unfreeze a
# tagged release branch. `fork_branch_ruling` decides fork_branch exactly as
# base_branch_ruling decides fork_default_branch.
STICKY = ("upstream_repo_id", "fork_url", "review_pr", "pr_approval",
          "license_clearance", "license_spdx", "pr_number", "pr_url",
          "pr_opened_at", "pr_merged_at", "pr_closed_at", "pr_closed_note",
          "adopted_at", "maintained", "no_upstream", "fix_abandoned",
          "base_branch_ruling", "fork_url_ruling", "fork_branch_ruling",
          "delivery", "release")
UNION = ("platforms", "waivers")
RESET = ("on_hold", "on_hold_reason")
# Every field that describes the CURRENT upstream-PR round. When the two sides
# disagree on pr_round, the higher round supplies all of these verbatim (see the
# module docstring); within a round the usual per-field rules stand.
PR_ROUND_FIELDS = ("pr_state", "pr_url", "pr_number", "pr_opened_at",
                   "pr_merged_at", "pr_closed_at", "pr_closed_note",
                   "published_sha", "review_pr", "pr_approval", "fix",
                   "fix_merged_at", "fix_abandoned", "pr_history", "pr_round")


def _round_of(x):
    return x.get("pr_round") or 1


def _merge_history(a, b):
    """Union of the two sides' archives, keyed on the archived PR's number.

    Entries are write-once -- archive_pr records a finished PR exactly once -- so
    a key present on both sides carries the same content and either copy will do;
    ties resolve to B for determinism, like _union."""
    ha = {e.get("pr_number"): e for e in (a.get("pr_history") or [])}
    hb = {e.get("pr_number"): e for e in (b.get("pr_history") or [])}
    merged = {**ha, **hb}
    return sorted(merged.values(),
                  key=lambda e: (e.get("archived_at") or "", e.get("pr_number") or 0))


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _b_is_later(a_ts, b_ts):
    # ISO-8601 Z strings sort lexicographically; ties resolve to B (deterministic).
    return (b_ts or "") >= (a_ts or "")


def _pr_rank(s):
    return 0 if not s or s == "none" else (2 if s in _TERMINAL_PR else 1)


def _stamp(block):
    """When a block was last written. Platform blocks carry updated_at; a waiver
    carries only the times of its suggestion and of a person's ruling on it."""
    return max((block.get(k) or "") for k in
               ("updated_at", "at", "refused_at", "suggested_at"))


def _same_sha(x, y):
    return bool(x) and bool(y) and (x.startswith(y) or y.startswith(x))


def _failure_at_same_commit(va, vb):
    """The failing block when one side passed and the other failed THE SAME commit.

    Two hosts of one platform can reach different verdicts on one sha -- a flaky
    test, a host-specific fault. Recency would keep whichever wrote last, and a pass
    that happened to be recorded after a failure would then satisfy a gate over
    evidence that the commit does not pass there. The failure stands until someone
    looks at it."""
    for bad, good in ((va, vb), (vb, va)):
        if (bad.get("state") == "validation-failed" and good.get("state") == "completed"
                and _same_sha(bad.get("failed_sha"), good.get("validated_sha"))):
            return bad
    return None


def _union(pa, pb, po=None):
    """Per-key merge of two maps. With the ancestor `po`, a block only one side
    changed is that side's; where both changed it (or there is no ancestor) the
    later writer wins within the key, except that a failure and a pass recorded
    against the same commit resolve to the failure."""
    out = {}
    for k in set(pa) | set(pb):
        va, vb = pa.get(k), pb.get(k)
        vo = (po or {}).get(k)
        if va is None or vb is None:
            out[k] = vb if va is None else va
        elif va == vb:
            out[k] = vb
        elif vo is not None and va == vo:
            out[k] = vb
        elif vo is not None and vb == vo:
            out[k] = va
        elif isinstance(va, dict) and isinstance(vb, dict):
            out[k] = (_failure_at_same_commit(va, vb)
                      or (vb if _b_is_later(_stamp(va), _stamp(vb)) else va))
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


_ABSENT = object()

# Fields with a rule of their own further down; the ancestor pass leaves them alone.
_RULED = ("updated_at", "porting", "porting_released", "intake", "pr_state",
          "pr_history") + UNION


def _one_sided(k, a, b, o):
    """The side that alone changed field `k` since the ancestor, or None when
    neither or both did (or there is no ancestor to tell)."""
    if o is None:
        return None
    va, vb, vo = a.get(k, _ABSENT), b.get(k, _ABSENT), o.get(k, _ABSENT)
    if va == vb:
        return None
    if va == vo:
        return b
    if vb == vo:
        return a
    return None


def merge(a, b, o=None):
    """Merge two records. `o` is their common ancestor when git supplied one.

    Recency alone cannot tell a field a writer CHANGED from one it merely still
    carried. A host that read the record, lost a race, and wrote later would win
    every field on its stale copy: recording one platform's result put head_sha,
    the stage and published_sha back to what they were before another host
    advanced them, and a gate then read as satisfied at a commit the fork had left.
    So a field only one side changed since the ancestor is that side's, whichever
    record is newer; recency decides only what both sides changed."""
    a_top, b_top = a.get("updated_at") or "", b.get("updated_at") or ""
    newer, older = (b, a) if b_top > a_top else (a, b)
    out = dict(older)
    out.update(newer)                       # default: latest writer wins per field
    out["updated_at"] = max(a_top, b_top)
    settled = set()
    for k in (set(a) | set(b)) - set(_RULED):
        side = _one_sided(k, a, b, o)
        if side is None:
            continue
        settled.add(k)
        if k in side:
            out[k] = side[k]
        else:
            out.pop(k, None)

    for k in STICKY:
        if out.get(k) is None:
            out[k] = a.get(k) if a.get(k) is not None else b.get(k)
        if k in out and out[k] is None:
            del out[k]
    if out.get("base_branch_ruling"):
        out["fork_default_branch"] = out["base_branch_ruling"]["branch"]
    if out.get("fork_branch_ruling"):
        out["fork_branch"] = out["fork_branch_ruling"]["branch"]
    # A sticky delivery ruling keeps the release-line schema version with it, or a
    # newer write from a host whose copy predates the ruling would hand the record
    # back to tooling that would publish it (moatlib.RELEASE_SCHEMA_VERSION).
    if out.get("delivery"):
        out["schema_version"] = max(a.get("schema_version") or 0,
                                    b.get("schema_version") or 0)

    for k in RESET:
        if k in settled:
            continue        # one side set or cleared it; the ancestor pass took that
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
            out[k] = _union(a.get(k) or {}, b.get(k) or {}, (o or {}).get(k))

    if "intake" in a or "intake" in b:
        out["intake"] = _merge_intake(a, b, newer)

    if "pr_state" in a or "pr_state" in b:
        sa, sb = a.get("pr_state"), b.get("pr_state")
        if _PR_RANKABLE and _pr_rank(sa) != _pr_rank(sb):
            out["pr_state"] = sa if _pr_rank(sa) > _pr_rank(sb) else sb
        else:
            out["pr_state"] = newer.get("pr_state") or sa or sb

    if _round_of(a) != _round_of(b):
        # A follow-up round opened on one side. The higher round's PR block wins
        # wholesale -- overriding STICKY and the rank above, which would otherwise
        # resurrect the finished round -- and its pr_history already carries the
        # loser's current PR. Last, so no per-field rule can undo it.
        win = a if _round_of(a) > _round_of(b) else b
        for k in PR_ROUND_FIELDS:
            out.pop(k, None)
            if k in win and win[k] is not None:
                out[k] = win[k]
    elif "pr_history" in a or "pr_history" in b:
        out["pr_history"] = _merge_history(a, b)
    return out


def main(argv):
    # argv: O A B [P]
    if len(argv) < 3:
        sys.stderr.write("merge_status: expected O A B [P]\n")
        return 2
    o_path, a_path, b_path = argv[0], argv[1], argv[2]
    try:
        ancestor = _load(o_path)
        if not isinstance(ancestor, dict):
            ancestor = None
    except Exception:       # git hands over an empty file when both sides added it
        ancestor = None
    try:
        merged = merge(_load(a_path), _load(b_path), ancestor)
    except Exception as e:  # fall back to a real conflict rather than guessing
        sys.stderr.write(f"merge_status: {e}; leaving conflict\n")
        return 1
    with open(a_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    import trunk  # noqa: E402 - run the trunk's copy of this tool, see trunk.py
    trunk.reexec(__file__)
    sys.exit(main(sys.argv[1:]))
