#!/usr/bin/env python3
"""MOAT control-plane library: schema, per-platform state machine, cross-platform
gating + regression guard, validated status.json writes, and the single
git-sync write path. Also a small CLI used by orient.sh and the agents.

status.json is the source of truth. The three AMD targets share one fork branch,
so any HEAD advance re-validates the platforms that already passed (see
advance_head). State transitions are validated; illegal jumps raise."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = REPO_ROOT / "projects"
SCHEMA_VERSION = 3
# Records are migrated in place, one ref at a time, so a checkout must not refuse the
# ones the migration has not reached yet -- load_status validates, and a hard bump
# makes every record unreadable the moment it lands. The window is one version wide
# and closes when the migration finishes.
#
# 3 is a real break rather than an addition: `revalidate` became a `completed` block
# whose validated_sha is not the head, so a version-2 reader would see that block and
# conclude the arch is up to date on code it has never run.
READABLE_SCHEMA_VERSIONS = (2, 3)

def _load_arches():
    """config/arches.toml is the single source of truth for gates.

    It defines RULES, not a roster: which gates are required, which may be waived,
    and how a wavefront width follows from an architecture family."""
    with open(REPO_ROOT / "config" / "arches.toml", "rb") as f:
        cfg = tomllib.load(f)
    waves = {fam: gate for gate, fams in cfg["wave"].items() for fam in fams}
    return waves, cfg["gates"]["required"], set(cfg["gates"].get("waivable", []))


WAVE_BY_FAMILY, REQUIRED_GATES, WAIVABLE_GATES = _load_arches()

# A platform is an architecture on an operating system. Any host that reports a GPU
# has one, so this validates the FORM rather than checking membership of a list --
# a new GPU must not need a config edit, a schema change, or a release to be usable.
PLATFORM_RE = re.compile(r"^(linux|windows)-(gfx[0-9a-f]+)$")


def parse_platform(platform):
    """(os, gfx) for a well-formed platform, else (None, None)."""
    m = PLATFORM_RE.match(platform or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def wave_of(gfx):
    """The wave gate for an architecture, or None if its family is unknown.

    Longest prefix wins, so gfx11xx matches the RDNA family rather than the gfx1
    that is a prefix of it."""
    for fam in sorted(WAVE_BY_FAMILY, key=len, reverse=True):
        if gfx.startswith(fam):
            return WAVE_BY_FAMILY[fam]
    return None


def gates_for(platform):
    """The gates a platform satisfies: its wavefront width, plus its OS."""
    os_name, gfx = parse_platform(platform)
    if not os_name:
        return set()
    w = wave_of(gfx)
    return ({w} if w else set()) | {os_name}


def platform_problem(platform):
    """Why this platform string is unusable, or None if it is fine.

    An unknown architecture family is reported rather than assumed: guessing a
    wavefront width wrong under-allocates shared memory and corrupts device
    memory silently, which is far worse than refusing until someone adds a line."""
    os_name, gfx = parse_platform(platform)
    if not os_name:
        return (f"{platform!r} is not a platform: expected <os>-<gfx>, "
                f"e.g. linux-gfx90a or windows-gfx1201")
    if wave_of(gfx) is None:
        return (f"{platform}: no wavefront width known for {gfx}. Add its family to "
                f"[wave] in config/arches.toml -- do not guess, a wrong width "
                f"mis-sizes shared memory without failing")
    return None


def validations(obj):
    """Per-arch validation records -- status.json's `platforms` map."""
    return obj.get("platforms") or {}


PORT_BRANCH = "moat-port"  # the topic branch that holds the port on each fork

# Where the PORT is. One fork, one answer, so this is a property of the project and
# never of an architecture -- there is no such thing as "screened on gfx90a".
#
# The upstream PR is NOT in here. It is one fact about the project too, but an
# orthogonal one: opening it changes nothing any arch validated, and parking it on an
# arch's record overwrote that arch's real state (a merged PR rendered as an unknown
# status in the README because `upstream-landed` had displaced `completed`). It lives
# in `pr_state` -- see PR_STATES.
STAGE_TRANSITIONS = {
    # awaiting-fork is reachable from unclaimed because that is what intake does: it
    # screens an unadopted project and parks it for the fork decision, in one step.
    # Requiring screened first made the documented instruction illegal, and agents
    # compensated by transitioning twice -- which worked, and hid the contradiction.
    "unclaimed": {"screened", "awaiting-fork"},
    "screened": {"awaiting-fork", "planning"},
    # `planning` exists to be ACQUIRED. A planner writes plan.md, which is one shared
    # artifact on a shared branch, and two of them produce two different strategies
    # that no merge can reconcile -- plan.md has no merge driver, so the second push
    # hard-conflicts and one analysis is stranded. The porter had this solved and the
    # planner did not, on the reasoning that only the fork needs serialising.
    #
    # unclaimed and screened had direct `-> planned` edges from before the lock
    # existed, and they survived its introduction -- a planner taking either one wrote
    # plan.md having never entered the stage that serialises writing it. Removed:
    # any route that is about to WRITE a plan goes through `planning`. The stages
    # that RESTORE an already-written plan (awaiting-fork, awaiting-upstream,
    # not-portable below) keep their direct edge, because re-entering `planned`
    # there records a fact about an existing plan.md, not a new analysis.
    "planning": {"planned", "screened"},
    "awaiting-fork": {"screened", "planned", "porting"},
    "awaiting-upstream": {"planned", "porting", "unclaimed"},
    # The porter reaches awaiting-fork when it finds no fork to push to, which
    # porter.md has always instructed and the table has always refused.
    "planned": {"porting", "awaiting-upstream", "awaiting-fork"},
    # `delta-ported` is reached THROUGH `porting`, not around it. It used to be a
    # direct hop from changes-requested/validation-failed, which meant a fix could
    # be written to the fork without ever entering the one state that takes the
    # fork-write lock -- so two archs recovering from the same failed validation
    # could both write it. No project has ever been in delta-ported, so routing it
    # this way costs nothing and closes the hole.
    "porting": {"ported", "delta-ported"},
    # `reviewing` exists to be ACQUIRED, for the same reason `planning` does. A review
    # reads one shared branch and writes one shared verdict, and the verdict is the
    # part that races: `ported` has two legal exits and whichever reviewer writes
    # second wins, so a second reviewer finding nothing can overwrite a first one's
    # changes-requested and send a port with recorded findings on to validation.
    #
    # The sharper failure needs no race at all. The porter's lock only excludes other
    # LOCK HOLDERS, and a reviewer held nothing, so a porter could legally take the
    # lock and rewrite the branch mid-review -- which is exactly what happened to
    # HEonGPU: a review of 5d99b8f..4ceabb2 was overtaken by two porter commits and
    # its analysis was stale before it could be recorded. Reviewing under the same
    # lock the porter takes is what makes the branch hold still.
    #
    # Reviewing is the last project-scoped stage. Validation stays unlocked because it
    # is partitioned by platform -- each host writes its own arch's record and no two
    # want the same one -- while review has no such partition, which is why it is the
    # stage that collides.
    "ported": {"reviewing"},
    "delta-ported": {"reviewing"},
    # Both origins are exits, so releasing a review without a verdict puts the project
    # back where it came from rather than inventing one, the way planning -> screened
    # does. A review that reached a conclusion leaves by the conclusion.
    "reviewing": {"review-passed", "changes-requested", "ported", "delta-ported"},
    "changes-requested": {"porting"},
    # review-passed has no exit to `completed`: completing is an ARCH's fact now, and
    # a project stays review-passed while its architectures validate independently --
    # including when one of them FAILS. `validation-failed` used to be a stage here as
    # well as an arch state, and being in both machines is what broke it: set_state
    # resolves the collision by checking STAGE_STATES first, so a validator recording
    # one arch's failure moved the whole project out of review-passed and left the
    # arch's own record untouched. Leaving review-passed switches off the per-arch
    # derivation in arch_task, so every arch -- including ones completed at head --
    # routed to the porter, and the only edge back was through a port. A waiver being
    # approved or a sibling arch satisfying the gate could not move it, so four
    # projects sat advertising porter work that did not exist. It is an arch state
    # only now, and the porter is reached from review-passed directly.
    "review-passed": {"porting"},
    # A person may revive a project judged unportable -- ROCm gains a library, an
    # upstream rewrite lands. Nothing else leads out.
    "not-portable": {"planned", "porting"},
}
# Any stage may end here, which is why `not-portable` is not in the table above: the
# judgement can be reached from a planner's analysis, from a porter failing over and
# over, or from a validator. Only a person may record it (see set_not_portable).
STAGE_STATES = set(STAGE_TRANSITIONS) | {s for v in STAGE_TRANSITIONS.values() for s in v}

# What an ARCHITECTURE knows, which is only whether it ran the tests on this code.
# `blocked` is orthogonal and set separately: it means "this arch cannot run it, here
# is why", and after the split that is the only thing it may mean -- a verdict on the
# codebase is `not-portable` and a verdict on the OS is a gate waiver.
ARCH_TRANSITIONS = {
    None: {"completed", "validation-failed"},
    "completed": {"validation-failed"},
    "validation-failed": {"completed"},
}
ARCH_STATES = {s for s in ARCH_TRANSITIONS if s} | \
              {s for v in ARCH_TRANSITIONS.values() for s in v}

# Never stored. `port-ready` and `revalidate` were conclusions about (stage,
# validated_sha, head_sha) that a sweep wrote into the file, and a stored conclusion
# is a thing that goes stale; arch_task computes them instead. They remain in the
# vocabulary because agents and the selector still speak them.
DERIVED_ARCH_STATES = {"port-ready", "revalidate"}

STATES = STAGE_STATES | ARCH_STATES | DERIVED_ARCH_STATES

# Project-level upstream PR lifecycle, orthogonal to every arch's state. A port can
# be validated everywhere with no PR, or carry a merged PR while an arch is
# revalidating a later commit; neither fact constrains the other.
PR_STATES = ("open", "merged", "closed")

# Which agent handles each state, and selection priority (lower = sooner).
# Resume-before-start: drain work in flight before opening new fronts.
#
# Everything up to review is the PROJECT's work and one agent does it once. From
# review-passed on it is each architecture's, which is why the two derived states are
# here alongside the stages: what arch_task hands back is one or the other.
STAGE_FOR_STATE = {
    "unclaimed": "intake",
    "screened": "planner",
    "planning": "planner",
    "planned": "porter",
    "porting": "porter",
    "changes-requested": "porter",
    "validation-failed": "porter",
    "ported": "reviewer",
    "delta-ported": "reviewer",
    "reviewing": "reviewer",
    "review-passed": "validator",
    "port-ready": "validator",
    "revalidate": "validator",
}
# Nothing here covers the upstream PR: that work runs from `pr_state`, not from an arch
# state, and never through next_task. Shepherd work is EVENT-driven -- a maintainer
# comments, an upstream PR merges, a landed port goes stale -- so polling every open
# PR on every selection would be wrong. The moat-checkup skill covers it instead.
SELECT_RANK = {
    "revalidate": 0,
    "validation-failed": 1,
    "changes-requested": 2,
    "porting": 3,
    "delta-ported": 4,
    "planning": 4,
    "reviewing": 4,
    "planned": 5,
    "ported": 6,
    "review-passed": 7,
    "port-ready": 8,
    "screened": 9,
    "unclaimed": 10,
}
# Stages that take no agent action: gated on a human, or waiting on something outside
# our control. `awaiting-fork` waits on an org admin to create the fork -- creating one
# is a deliberate act by someone who can, so its existence carries the decision and
# nothing else needs to record one. `awaiting-upstream` waits on an external event (a
# third party's PR landing, say) and is viable-but-parked rather than dead.
#
# `not-portable` is the judgement that this codebase cannot be ported at all: reached
# when a planner's analysis says so, or a porter has failed repeatedly, and recorded
# only by a person. It is deliberately NOT a disposition -- a dispositioned project is
# one that left the pipeline before anyone worked it, and every `cant-port`
# disposition in this repo is a project that was never adopted. These have a folder, a
# plan, notes and often weeks of porter work, and a negative outcome is a deliverable.
# Stages whose work writes content the whole project shares, so exactly one
# architecture may hold them: the fork's port branch for `porting`, plan.md for
# `planning`, the review verdict for `reviewing`. Entering one acquires the lock and
# leaving releases it, rather than an agent being told to set a field by hand -- which
# is what the porter's lock was before it had a mechanism, and no project ever carried
# one.
#
# One lock, not three. The stages do not merely conflict with themselves, they conflict
# with each other: reviewing a branch a porter is rewriting is the collision that cost
# the most, and two separate locks would have permitted it.
EXCLUSIVE_STAGES = {"porting", "planning", "reviewing"}
EXCLUSIVE_AGENTS = {"porter", "planner", "reviewer"}

INERT_STAGES = {"awaiting-fork", "awaiting-upstream", "not-portable"}
INERT = INERT_STAGES | {"completed"}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def same_commit(a, b):
    """True when two recorded shas name the same commit.

    Shas arrive at whatever length whoever wrote them used -- `git rev-parse HEAD`
    gives 40, `--short` gives 7 or more, a person pasting from GitHub gives 8 -- and
    every staleness test used to be an equality. Five projects held a validated_sha
    that WAS the head commit at a different abbreviation and so read as stale
    forever; two of them blocked their upstream PR behind a line that said
    `linux-gfx90a=completed`, which is not a legible reason to be blocked.

    Compare on the shorter of the two, which is what git does for an abbreviated
    rev. Below 7 hex chars, or on anything that is not hex, fall back to equality
    rather than guessing -- a 4-character prefix is not evidence of identity."""
    if not a or not b:
        return False
    a, b = a.strip().lower(), b.strip().lower()
    n = min(len(a), len(b))
    if n < 7 or any(c not in "0123456789abcdef" for c in a[:n] + b[:n]):
        return a == b
    return a[:n] == b[:n]


def full_sha(sha, repo=None):
    """Expand an abbreviated sha to its full 40 using the fork clone, so the record
    stops accumulating mixed lengths. Returns sha unchanged when there is no clone
    to ask -- the record is still readable, because same_commit tolerates it."""
    if not sha or not repo or len(sha) >= 40:
        return sha
    r = subprocess.run(["git", "rev-parse", f"{sha}^{{commit}}"], cwd=str(repo),
                       capture_output=True, text=True)
    out = r.stdout.strip()
    return out if r.returncode == 0 and len(out) == 40 else sha


def _empty_stats():
    return {
        "tokens_total": 0,
        "tokens_approx": True,
        "wall_seconds": {"thinking": 0, "compile": 0, "test": 0, "misc": 0},
        "session_count": 0,
        "first_session_at": None,
        "last_session_at": None,
    }


def _platform_block(initial_state):
    """A fresh arch record. `state` is OMITTED rather than set to null when there is
    none: absent means "this architecture has recorded nothing", and null is a value
    the schema's enum has no member for. A stage transition creates the row to hang
    `last_agent` and timestamps on, and wrote a null into it that the schema gate then
    rejected -- on a project the transition had just successfully recorded."""
    block = {
        "state": initial_state,
        "blocked": False,
        "blocked_reason": None,
        "validated_sha": None,
        "failed_sha": None,
        "started_at": None,
        "completed_at": None,
        "updated_at": now_iso(),
        "stats": _empty_stats(),
    }
    if initial_state is None:
        del block["state"]
    return block


def status_path(name):
    return PROJECTS / name / "status.json"


def load_status(name):
    """A project's record. Reads the working tree first, then falls back to the refs.

    The fallback exists because 29 call sites take a project name and expect a
    record, and after the migration an in-flight project's folder is on its own
    branch rather than in this checkout. Making each caller resolve separately is how
    a few of them silently answer "not adopted" instead."""
    p = status_path(name)
    if p.exists():
        with open(p) as f:
            obj = json.load(f)
        validate_status(obj)
        return obj
    for ref in ("origin/main", f"origin/port/{name}"):
        raw = _ref_read(ref, f"projects/{name}/status.json")
        if raw:
            obj = json.loads(raw)
            validate_status(obj)
            return obj
    raise FileNotFoundError(str(p))


def adopted_repo_ids():
    """{repo id: project name} for every adopted project that records one.

    Adoption used to be matched on a name -- the basename in one place and the full
    owner/repo in another, so one conflated foo/bar with baz/bar and the other missed
    every transfer. Neither survives a repository moving, which happens: FlashRT went
    from LiangSu8899 to the flashrt-project org and came back through discovery as a
    fresh candidate."""
    out = {}
    for n in all_projects():
        obj, _ = project_record(n)
        rid = (obj or {}).get("upstream_repo_id")
        if rid:
            out[int(rid)] = n
    return out


def upstream_full_name(name):
    """The upstream repo as `owner/repo`, from the URL status.json already holds.
    This is the key dispositions.json is written under, so it is how a project record
    finds its own disposition."""
    obj, _where = project_record(name)
    if obj is None:
        return None          # not adopted anywhere; callers treat that as "no record"
    url = (obj.get("upstream_url") or "").rstrip("/")
    tail = url.replace("https://github.com/", "")
    return tail if tail.count("/") == 1 else None


def save_status(name, obj):
    # Writing a project whose record lives on another branch would create a second
    # copy here and diverge from the one being worked. Say where it lives instead.
    if not status_path(name).exists():
        for ref in ("origin/main", f"origin/port/{name}"):
            if _ref_read(ref, f"projects/{name}/status.json"):
                raise RuntimeError(
                    f"{name} is not in this checkout -- its record is on {ref}. "
                    f"Check out that branch to write it.")
    validate_status(obj)
    stale = check_against_trunk(obj)
    if stale:
        raise ValueError(
            f"{obj.get('name')}: this checkout would write {'; '.join(stale)}, which the "
            f"TRUNK's schema does not accept -- your tooling predates it. check.py judges "
            f"every ref, so writing it blocks pushes for every project from every host. "
            f"Run `python3 utils/moatlib.py branch-sync --apply`, then redo this.")
    obj["updated_at"] = now_iso()
    p = status_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


_TRUNK_VOCAB = None


def trunk_vocabulary(base_ref="origin/main"):
    """The value sets the TRUNK's schema accepts, or None if it cannot be read.

    Read from `schema/status.schema.json`, which is generated FROM moatlib, so it is
    the trunk's own answer rather than a guess parsed out of its source. Cached: this
    is consulted on every write and it is one git call.

    Two limits, both deliberate, and both worth knowing because they bound how much
    this guard is worth. `origin/main` is a LOCAL ref and nothing here fetches, so the
    check is only as current as the last fetch -- and a checkout stale enough to hold
    old tooling may hold an old trunk ref too, in which case this reads a schema that
    AGREES with the stale code and passes. Fetching from a path consulted on every
    write would cost more than it returns, so orient.sh's fetch is what keeps it
    honest. The cache then lives for the process, so a long session that outlives a
    trunk change keeps the old answer.

    Both fail in the same direction: a value the trunk has dropped can slip through,
    never a good one refused. That is the right direction for a guard that sits in
    front of every write, but it means this narrows the window rather than closing
    it -- starting from a synced worktree is still what actually prevents the case."""
    global _TRUNK_VOCAB
    if _TRUNK_VOCAB is None:
        raw = _ref_read(base_ref, "schema/status.schema.json")
        try:
            d = json.loads(raw) if raw else None
            _TRUNK_VOCAB = {
                "schema_version": d["properties"]["schema_version"].get("enum"),
                "stage": d["properties"]["stage"].get("enum"),
                "archstate": d["$defs"]["archstate"].get("enum"),
            } if d else False
        except (KeyError, TypeError, json.JSONDecodeError):
            _TRUNK_VOCAB = False
    return _TRUNK_VOCAB or None


def check_against_trunk(obj):
    """Reasons this record would be rejected by the TRUNK's schema, not just by ours.

    A worktree runs whatever tooling its branch last merged, so an agent can hold a
    vocabulary the trunk has moved past and write a value that no longer exists. That
    is not caught by validating against the local schema -- the local schema agrees
    with the local code, which is the problem. It surfaces later as a repo-wide gate
    failure, and because check.py judges every ref it blocks pushes for every project
    from every host, not just the one that wrote it. That happened today.

    Read-only and advisory about the TRUNK: it never rejects a value the trunk knows
    and we do not, since that direction is just a branch being behind on a value it is
    not using."""
    vocab = trunk_vocabulary()
    if not vocab:
        return []
    bad = []
    sv, stage = obj.get("schema_version"), obj.get("stage")
    if vocab["schema_version"] and sv is not None and sv not in vocab["schema_version"]:
        bad.append(f"schema_version {sv} (trunk accepts {vocab['schema_version']})")
    if vocab["stage"] and stage is not None and stage not in vocab["stage"]:
        bad.append(f"stage {stage!r}")
    if vocab["archstate"]:
        for plat, blk in (obj.get("platforms") or {}).items():
            st = blk.get("state")
            if st is not None and st not in vocab["archstate"]:
                bad.append(f"{plat} state {st!r}")
    return bad


def save_record(name, obj, message):
    """Persist a project's record wherever it lives -- this checkout, or its own branch.

    `save_status` refuses a record that is not in this tree, and that is right for
    anything that WORKS a project: a second copy here would diverge from the one being
    worked. A few callers do something else entirely -- they record a FACT about the
    project and never open its files: a fork appearing, an approval snapshot, the
    upstream PR opening or closing. Refusing those buys nothing and costs the fact.

    It cost the whole route upstream. Every publishable project is branch-resident by
    construction (`belongs_on_branch` is true while `pr_state` is unset), so
    `upstream.py --publish --apply` -- the one documented submission command -- could
    only ever run from the project's own branch, and anywhere else reported the refusal
    as a failure to record an approval and moved on.

    Same shape as `release_awaiting_fork`, which had this right first: advancing a
    record is safe to write to a branch, handing an agent a project whose files are
    absent is not. Returns the branch sha when it wrote to a branch, else None."""
    _cur, where = project_record(name)
    # Recording a fact must not smuggle in a stage move the transition table refuses.
    # set_state guards the table on the worked-checkout path; this is the only other
    # path a stage change can travel, so it holds the same line. `not-portable` is
    # deliberately reachable from anywhere (see STAGE_TRANSITIONS' trailing comment).
    if _cur is not None:
        old_stage, new_stage = _cur.get("stage"), obj.get("stage")
        if (new_stage != old_stage and new_stage != "not-portable"
                and new_stage not in STAGE_TRANSITIONS.get(old_stage or "unclaimed", set())):
            raise ValueError(f"{name}: illegal stage move "
                             f"{old_stage or 'unclaimed'} -> {new_stage}")
    if status_path(name).exists() and writable_here(name, where):
        save_status(name, obj)
        return None
    branch = port_branch_of(name)
    if branch is None:
        raise RuntimeError(
            f"{name}: its record is on the trunk and this checkout is on "
            f"{current_branch()}, which may not write it. `main` is protected, so that "
            f"record reaches it by pull request: check out `main` (or a branch off it) "
            f"and record it there.")
    obj["updated_at"] = now_iso()
    validate_status(obj)
    # The branch path skips save_status, so it would skip its trunk check too. A stale
    # checkout recording a fact onto someone else's branch is exactly the case that
    # check cannot afford to miss: the record it writes is judged by every ref sweep,
    # and validate_status above only asks whether THIS checkout's vocabulary is happy.
    stale = check_against_trunk(obj)
    if stale:
        raise ValueError(
            f"{name}: this checkout would write {'; '.join(stale)} to {branch}, which "
            f"the TRUNK's schema does not accept -- your tooling predates it. Run "
            f"`python3 utils/moatlib.py branch-sync --apply`, then redo this.")
    return commit_to_branch(
        branch, {f"projects/{name}/status.json": json.dumps(obj, indent=2) + "\n"},
        message)


def validate_status(obj):
    """Light hand-rolled validation (no jsonschema dependency). Raises ValueError."""
    for k in ("schema_version", "name", "upstream_url", "fork_default_branch",
              "priority", "ext_type", "platforms"):
        if k not in obj:
            raise ValueError(f"status.json missing required key: {k}")
    if obj["schema_version"] not in READABLE_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported schema_version: {obj['schema_version']}")
    if "stage" in obj and obj["stage"] not in STAGE_STATES:
        raise ValueError(f"invalid stage: {obj['stage']!r}")
    # A verdict nobody signed satisfies nothing, so the stage cannot stand without one.
    if obj.get("stage") == "not-portable" and not (obj.get("not_portable") or {}).get("by"):
        raise ValueError("stage is not-portable but not_portable.by is missing -- "
                         "an agent cannot self-certify a project unportable")
    unknown = {p for p in obj["platforms"] if platform_problem(p)}
    if unknown:
        raise ValueError(f"unknown arch(es) {sorted(unknown)}; add them to config/arches.toml")
    for plat, blk in obj["platforms"].items():
        # A migrated block may carry no state at all: what survives the split is what
        # an arch can know, and for an arch that never validated, that is only
        # `blocked` and its reason. Absent is a value here, not a missing field.
        if blk.get("state") is not None and blk["state"] not in STATES:
            raise ValueError(f"{plat}: invalid state {blk.get('state')!r}")
        if not isinstance(blk.get("blocked"), bool):
            raise ValueError(f"{plat}: blocked must be boolean")


# ---- state machine ---------------------------------------------------------

def set_state(name, platform, new_state, agent=None, save=True):
    """Validate and apply a transition with its side effects.

    One entry point for two machines, routed on which the state belongs to: a stage
    moves the PROJECT and every arch sees it, an arch state records what THIS GPU
    proved. Agents say `set-state <name> <arch> <state>` for both, and pass the arch
    either way -- for a stage it says who is doing the work, not whose fact it is.

    A platform's record is created on first use rather than pre-seeded, so a host
    whose GPU nothing has recorded before simply starts working and its record
    appears. The platform still has to be well-formed and its wavefront width
    known."""
    problem = platform_problem(platform)
    if problem:
        raise ValueError(problem)
    if new_state == "not-portable":
        raise ValueError(
            f"{name}: `not-portable` is a person's verdict on the codebase, not a "
            f"transition -- an agent may write the case but never the judgement. "
            f"`moatlib.py set-not-portable {name} --reason '<why>' --by <who>`")
    if new_state in DERIVED_ARCH_STATES:
        raise ValueError(
            f"{name}/{platform}: {new_state} is derived from (stage, validated_sha, "
            f"head_sha) and is never stored -- see arch_task")
    obj = load_status(name)
    is_stage = new_state in STAGE_STATES
    cur = project_stage(obj) or "unclaimed" if is_stage else \
        (obj["platforms"].get(platform) or {}).get("state")
    # Exclusivity is checked BEFORE the no-op short-circuit, because "the project is
    # already in this stage" does not mean "you are the one holding it". `cur` is the
    # PROJECT's stage now, shared by every arch, so a second host entering the stage a
    # first host already entered would short-circuit out and never reach the lock --
    # which is how the split silently reopened the hole the lock was built to close.
    if new_state in EXCLUSIVE_STAGES:
        held = obj.get("porting")
        if held and held.get("arch") != platform:
            raise ValueError(
                f"{name}: the work lock is held by {held['arch']} since "
                f"{held.get('since')}. Takeover is a person's decision, not a "
                f"timeout -- ask, then `moatlib.py port-lock {name} --take {platform}`")
    # A `completed` arch revalidating a NEWER head is the one same-state call that is
    # not a no-op. `revalidate` is DERIVED from validated_sha lagging head_sha (see
    # arch_task), so the stored word stays `completed` while the fact being recorded --
    # this GPU proved THIS code -- is new. Short-circuiting it sent both validators
    # that hit it off to write validated_sha their own way, one of them tagging a full
    # GPU rerun as a carry_forward, which is the opposite of what that field means.
    revalidated = (not is_stage and new_state == cur == "completed"
                   and not same_commit(
                       (obj["platforms"].get(platform) or {}).get("validated_sha"),
                       obj.get("head_sha")))
    # The other same-state call that is not a no-op: entering an exclusive stage the
    # project is ALREADY in while holding no lock. That is the ordinary case of picking
    # up work another host put down, and it is the half the exclusivity check above does
    # not close -- that refuses when someone ELSE holds the lock, and a FREE lock fell
    # through to the short-circuit and returned before the acquisition below, leaving
    # the project in an exclusive stage nobody held while reporting success. A GooFit
    # porter hit exactly that and reached for `port-lock --take` against a free lock.
    #
    # "Same value" is not "nothing happened", and that has now been wrong three distinct
    # ways: a second host entering a stage the first holds, an arch revalidating a newer
    # head, and this. Ask what is left to RECORD, not whether a token matches.
    acquires = (new_state in EXCLUSIVE_STAGES
                and (obj.get("porting") or {}).get("arch") != platform)
    same = new_state == cur
    if same and not (revalidated or acquires):
        return obj
    table = STAGE_TRANSITIONS if is_stage else ARCH_TRANSITIONS
    # Reached only when the state really changes, or when one of the cases above
    # deliberately fell through -- neither has a self-edge in the table to satisfy.
    if not same and new_state not in table.get(cur, set()):
        kind = "stage" if is_stage else f"{platform}"
        raise ValueError(f"{name}/{kind}: illegal transition {cur} -> {new_state}")
    if platform not in obj["platforms"]:
        obj["platforms"][platform] = _platform_block(None)
    blk = obj["platforms"][platform]
    # The fork-write lock, taken and released by the transition rather than by an
    # agent remembering to. porter.md has told porters to set `porting` by hand
    # since the field existed and no project has ever carried one, which is what a
    # protocol with no mechanism gets you.
    #
    # Only `porting` acquires. `validation-failed` and `changes-requested` are
    # porter-stage too, but ENTERING them is recording that a run failed, and two
    # archs can legitimately fail the same head at once -- refusing to record that
    # would be a worse bug than the one this prevents. Serialising the porter's
    # DISPATCH is `actionable`'s job, and its guard already reads this field.
    if new_state in EXCLUSIVE_STAGES:          # refused above if another arch holds it
        obj["porting"] = {"arch": platform, "since": now_iso()}
    elif cur in EXCLUSIVE_STAGES and obj.get("porting"):
        # Leaving an exclusive stage ends what the lock protected, so ANY arch
        # legally recording the exit releases it. Requiring the holder to be the
        # one leaving left the lock held forever when a different arch drove
        # porting -> ported, with only `port-lock --release` to clean it up.
        obj["porting"] = None
    ts = now_iso()
    if is_stage:
        obj["stage"] = new_state
    else:
        blk["state"] = new_state
    blk["updated_at"] = ts
    if agent:
        blk["last_agent"] = agent  # informational; not in strict schema
    if new_state in ("porting", "delta-ported") and not blk.get("started_at"):
        blk["started_at"] = ts
    if new_state == "completed":
        blk["completed_at"] = ts
        blk["validated_sha"] = obj.get("head_sha")
        # A real-GPU validation supersedes any prior carry-forward tag; drop the
        # stale annotation so the metadata reflects how this completion was reached.
        blk.pop("carry_forward", None)
        # Integrity backstop: completing while the fork has uncommitted source/build
        # edits means the validated content may not be in the branch. Warn loudly
        # (the validator must commit it first); pr_ready hard-blocks on the same.
        dirty = uncommitted_source_files(name)
        if dirty:
            sys.stderr.write(
                f"WARNING {name}: marking {platform} completed but the fork has "
                f"{len(dirty)} UNCOMMITTED source/build file(s) -- validated content "
                f"may not be in the branch (integrity gap). Commit or discard: "
                f"{', '.join(p for _, p in dirty[:6])}\n")
    if new_state == "validation-failed":
        # WHICH commit failed, so a failure reads the way a validation does: evidence
        # about one commit, not a permanent property of the arch. See failure_stands.
        blk["failed_sha"] = obj.get("head_sha")
    obj["platforms"][platform] = blk
    if save:
        save_status(name, obj)
    return obj


def set_not_portable(name, reason, by, clear=False):
    """Record, or lift, the judgement that this codebase cannot be ported.

    A PROJECT-level verdict, because that is the shape the evidence has: the reasons
    that reach it -- the compute core is CUTLASS/CuTe with no ROCm path, it needs
    NVSHMEM, it wants a ground-up Composable Kernel rewrite -- are facts about the
    source, true on every architecture at once. They were being recorded as a `blocked`
    flag on whichever arch happened to look, which reads as "this GPU cannot run it"
    and left every other arch free to be sent at the same wall.

    Two OTHER things look like this and are not:

      an OS that will not take the port -- ZhiLight's host runtime is POSIX to the
      bone -- is a GATE WAIVER on `windows`, which already exists and already needs
      maintainer approval;
      a toolchain or library defect on one platform -- a Triton codegen bug on
      gfx1100, rocBLAS picking a generic kernel on one Windows arch -- is genuinely
      per-arch and stays a `blocked` flag, with the report registered against that
      project (projects/<name>/deferred.json, via `deferred.py add --project`).

    `by` is required and never defaulted: an agent may assemble the case and must not
    return the verdict, exactly as with a licence clearance or a gate waiver."""
    obj = load_status(name)
    if clear:
        obj.pop("not_portable", None)
        obj["stage"] = "planned"
    else:
        if not (by or "").strip():
            raise ValueError(
                f"{name}: --by is required. Judging a project unportable is a person's "
                f"call; an agent recording its own verdict would satisfy nothing")
        if not (reason or "").strip():
            raise ValueError(f"{name}: --reason is required")
        obj["not_portable"] = {"reason": reason, "by": by, "at": now_iso()}
        obj["stage"] = "not-portable"
    obj["updated_at"] = now_iso()
    save_status(name, obj)
    return obj


def set_blocked(name, platform, blocked, reason=None):
    obj = load_status(name)
    if platform not in obj["platforms"]:
        # An absent record means this arch has recorded nothing, which is exactly the
        # arch most likely to discover it cannot run the project at all -- so blocking
        # one has to be able to create the row, the way a stage transition does. Only
        # blocking creates it: writing a row that says "not blocked" would record an
        # intention to validate somewhere, which is fleet state and not a fact about
        # the port.
        if not blocked:
            raise ValueError(f"{name}: {platform} has recorded nothing; nothing to clear")
        obj["platforms"][platform] = _platform_block(None)
    blk = obj["platforms"][platform]
    blk["blocked"] = bool(blocked)
    blk["blocked_reason"] = reason if blocked else None
    blk["updated_at"] = now_iso()
    # An arch that gave up is not writing the fork. Without this the lock outlives
    # the only state machine path that releases it, and the next arch needs a human
    # takeover to work a project nobody is working.
    if blocked and (obj.get("porting") or {}).get("arch") == platform:
        obj["porting"] = None
    save_status(name, obj)
    return obj


def port_lock(name, take=None, release=False):
    """Show, take over, or release the fork-write lock. Returns the lock or None.

    Takeover is deliberately a command a person runs and never a timeout: an agent
    that stops mid-port leaves a held lock, and the difference between that and an
    agent still working is not visible from here."""
    obj = load_status(name)
    if release or take:
        held = obj.get("porting")
        obj["porting"] = {"arch": take, "since": now_iso()} if take else None
        save_status(name, obj)
        what = f"taken by {take}" if take else "released"
        prev = f" (was {held['arch']} since {held.get('since')})" if held else ""
        sys.stderr.write(f"{name}: fork-write lock {what}{prev}\n")
    return obj.get("porting")


def set_hold(name, on_hold, reason=None):
    """Project-wide postponement. A held project is skipped by the selector on
    every platform (actionable() returns False) without touching any platform
    state, so the hold is reversible and leaves resume points intact. Used to
    park a whole stack that we are deliberately not working yet."""
    obj = load_status(name)
    if on_hold:
        obj["on_hold"] = True
        obj["on_hold_reason"] = reason
    else:
        obj.pop("on_hold", None)
        obj.pop("on_hold_reason", None)
    obj["updated_at"] = now_iso()
    save_status(name, obj)
    return obj


def _clean_pr_url(raw):
    """Extract the http(s) URL from a pr_url argument. Guards against a caller
    capturing extra stdout (e.g. a `Warning: N uncommitted changes` line) into
    the value via command substitution -- store only the URL, never the noise."""
    m = re.search(r"https?://\S+", raw or "")
    if not m:
        raise ValueError(f"set-pr-open: no http(s) URL found in pr_url {raw!r}")
    return m.group(0)


def _gh_json(*args):
    """`gh` returning parsed JSON, or None if the call fails. Callers decide what a
    failure means; a network problem must never read as 'not approved'."""
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _pr_ref(url):
    """(owner/repo, number) from a PR URL."""
    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", url or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def fetch_review_pr(url):
    """Everything the approval gate needs about the review PR on our own fork.

    `lastEditedAt` and each review's `commit_id` are the load-bearing fields, and
    neither is available from `gh pr view` alone -- see pr_approval_valid for why
    they, rather than anything we record ourselves, are what the gate rests on."""
    pr = _gh_json("pr", "view", url, "--json",
                  "title,body,headRefOid,state,reviewDecision,url")
    if pr is None:
        return None
    slug, num = _pr_ref(pr.get("url") or url)
    if not slug:
        return None
    raw = _gh_json("api", f"repos/{slug}/pulls/{num}/reviews")
    if raw is None:
        return None
    pr["review_list"] = [{"login": (r.get("user") or {}).get("login"),
                          "state": r.get("state"),
                          "at": r.get("submitted_at"),
                          "commit": r.get("commit_id"),
                          "body": r.get("body") or "",
                          "assoc": r.get("author_association"),
                          "id": r.get("id")} for r in raw]
    # Ordinary conversation comments count too, so the approver can just type in the
    # box GitHub puts in front of them. See _approving_review for what that costs.
    raw_c = _gh_json("api", f"repos/{slug}/issues/{num}/comments") or []
    pr["comment_list"] = [{"login": (c.get("user") or {}).get("login"),
                           "state": "ISSUE_COMMENT",
                           "at": c.get("created_at"),
                           "commit": None,
                           "body": c.get("body") or "",
                           "assoc": c.get("author_association")} for c in raw_c]
    # When an approval carries no commit of its own, this is what it is compared
    # against. GitHub does not expose push time, so the tip's COMMITTER date stands
    # in for it: a new or rebased commit trips the comparison, but an older commit
    # pushed after the approval slips past. The review form's commit binding has no
    # such gap, which is one more reason it is the preferred box.
    head = _gh_json("api", f"repos/{slug}/commits/{pr.get('headRefOid')}")
    pr["head_at"] = (((head or {}).get("commit") or {}).get("committer") or {}).get("date")
    pr["slug"], pr["number"] = slug, num
    owner, repo = slug.split("/", 1)
    # lastEditedAt tracks BODY edits only; a title rename is a RenamedTitleEvent on
    # the timeline and does not move it, so both are fetched.
    q = ('{repository(owner:"%s",name:"%s"){pullRequest(number:%s){lastEditedAt '
         'timelineItems(itemTypes:RENAMED_TITLE_EVENT,last:1){nodes{'
         '... on RenamedTitleEvent{createdAt}}}}}}'
         % (owner, repo, num))
    g = _gh_json("api", "graphql", "-f", f"query={q}")
    if g is None:
        # The REST fetches above refuse on failure and this must too: an edit clock
        # nobody could read is not "never edited". Returning the PR with both clocks
        # None would let approval_currency skip the edit checks and pass an approval
        # whose title or body may have been rewritten -- the one incident on record
        # is the GraphQL endpoint timing out while REST kept working.
        return None
    node = ((((g or {}).get("data") or {}).get("repository") or {})
            .get("pullRequest") or {})
    pr["lastEditedAt"] = node.get("lastEditedAt")
    renames = (node.get("timelineItems") or {}).get("nodes") or []
    pr["titleRenamedAt"] = (renames[0] or {}).get("createdAt") if renames else None
    return pr


# The approval command. GitHub will not let a pull request's author approve it, and
# MOAT is authored and reviewed by the same person: agents run on the maintainer's
# credentials, so every review PR is self-authored and the APPROVED button is greyed
# out. A separate bot identity would fix that and is not available.
#
# So the signal is a comment carrying this exact line, in either place GitHub offers:
# a review comment or the ordinary conversation box. The review form is stronger,
# because GitHub stamps it with `commit_id` and the gate can compare that against the
# branch tip directly. A conversation comment carries no commit, so it is judged by
# time instead -- an approval is stale if anything was pushed after it was written.
# Both are accepted because the box a reviewer is actually looking at is the
# conversation box, and a gate people have to be told twice about is a gate people
# route around. Requiring the line to stand alone keeps "/moat approve is premature
# here" from reading as consent.
APPROVE_COMMAND = "/moat approve"
# The author's Request Changes button is greyed out for the same reason Approve is,
# so an objection needs a command form exactly as consent does. It supersedes the
# same author's earlier approval and blocks anyone else's until they approve again.
CHANGES_COMMAND = "/moat changes-requested"
MOAT_COMMANDS = (APPROVE_COMMAND, CHANGES_COMMAND)
# Who may give it. Anyone can comment; these are the associations GitHub reports for
# someone with write access to the repository.
APPROVE_ASSOC = ("OWNER", "MEMBER", "COLLABORATOR")


def _moat_command_lines(body):
    """Stand-alone `/moat ...` lines in a body, with fenced code blocks skipped.

    The fence rule is load-bearing: every review PR opens with an instructions
    comment that QUOTES the approval line inside a fence, and before this rule the
    gate matched that quotation and read its own instructions as the maintainer's
    standing approval (marian-dev went upstream over an explicit rejection)."""
    fenced = False
    for ln in (body or "").splitlines():
        s = ln.strip()
        if s.startswith("```"):
            fenced = not fenced
            continue
        if not fenced and s.startswith("/moat"):
            yield s


def _command_of(review):
    """The decision this event's body carries: 'approve', 'changes-requested',
    None for chatter, or 'unknown' for a /moat line matching no known command.
    Unknown fails closed downstream -- a maintainer's typo must read as an
    unanswered question, never as chatter to publish over. A body carrying both
    commands is an objection: the ambiguity is theirs to resolve, not ours."""
    cmds = set(_moat_command_lines(review.get("body")))
    if not cmds:
        return None
    if cmds - set(MOAT_COMMANDS):
        return "unknown"
    if CHANGES_COMMAND in cmds:
        return "changes-requested"
    return "approve"


def _is_approval_comment(review):
    return (review.get("assoc") in APPROVE_ASSOC
            and _command_of(review) == "approve")


def _decision_events(pr):
    """The latest decision-carrying event per author, oldest input first.

    Chatter -- a comment or COMMENTED review with no /moat command -- neither is a
    decision nor undoes one; a /moat command in either box is a decision and
    supersedes the same author's earlier one."""
    latest = {}
    events = sorted((pr.get("review_list") or []) + (pr.get("comment_list") or []),
                    key=lambda r: r.get("at") or "")
    for r in events:
        if not r.get("login") or r.get("state") == "PENDING":
            continue
        if r.get("state") in ("COMMENTED", "ISSUE_COMMENT") \
                and _command_of(r) is None:
            continue          # ordinary chatter is not a decision, and does not undo one
        latest[r["login"]] = r
    return latest


def _approving_review(pr):
    """A standing approval on the review PR, or None.

    Only the latest decision per author counts: someone who approves and then
    requests changes -- by review or by `/moat changes-requested` -- has withdrawn
    the approval, and honouring the earlier event would publish over an objection.
    An outstanding objection from ANYONE with write access blocks, even alongside
    somebody else's approval, and so does an unrecognized /moat command from them:
    publishing while a reviewer may still be objecting is what this prevents."""
    latest = _decision_events(pr)
    if any(r.get("state") == "CHANGES_REQUESTED" for r in latest.values()):
        return None
    if pr.get("reviewDecision") == "CHANGES_REQUESTED":
        return None
    if any(_command_of(r) in ("changes-requested", "unknown")
           for r in latest.values() if r.get("assoc") in APPROVE_ASSOC):
        return None
    # An APPROVED review passes the same write-access test as the comment form:
    # on a public fork ANYONE can submit an approving review, and a drive-by
    # approval from an outsider must not open the gate.
    return next((r for r in latest.values()
                 if (r.get("state") == "APPROVED"
                     and r.get("assoc") in APPROVE_ASSOC)
                 or _is_approval_comment(r)), None)


def moat_command_audit(pr):
    """(blockers, notes) -- every /moat command on the review PR, judged.

    Run before anything opens upstream, so a command the gate did not understand
    surfaces to a person by name instead of being read as chatter. Blockers are
    standing objections and unrecognized commands from someone with write access,
    judged on each author's LATEST decision so a superseded typo does not block
    forever. Notes list commands from authors without write access -- they decide
    nothing, but a person should know someone tried."""
    blockers, notes = [], []
    for r in _decision_events(pr).values():
        cmd = _command_of(r)
        if cmd is None:
            continue
        who = f"{r.get('login')} at {r.get('at')}"
        if r.get("assoc") not in APPROVE_ASSOC:
            notes.append(f"{cmd!r} from {who}, who has no write access -- ignored")
        elif cmd == "unknown":
            bad = [c for c in _moat_command_lines(r.get("body"))
                   if c not in MOAT_COMMANDS]
            blockers.append(f"unrecognized command {', '.join(map(repr, bad))} from "
                            f"{who} -- known: {', '.join(MOAT_COMMANDS)}")
        elif cmd == "changes-requested":
            blockers.append(f"changes requested by {who} ({CHANGES_COMMAND}) -- "
                            f"standing until they post {APPROVE_COMMAND}")
    return blockers, notes


def _content_digest(pr):
    """Hash of exactly what a reviewer read: the title and the body together."""
    payload = f"{pr.get('title') or ''}\n\n{pr.get('body') or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_pr_approval(name, review_pr=None):
    """Snapshot the approval that already exists on the review PR.

    The approval is the maintainer's APPROVED review on our own fork's PR -- one
    action, in one place, on the page that showed them the code, the title and the
    body. Nothing here asks anyone to approve anything a second time; it records
    what they approved so an unattended job can later prove that what it is about to
    publish upstream is that same thing."""
    obj = load_status(name)
    # A fix round's approval lives on the fix review PR; the original review_pr
    # already did its job when the upstream PR opened.
    fix_url = (obj.get("fix") or {}).get("review_pr")
    url = review_pr or fix_url or obj.get("review_pr")
    if not url:
        raise ValueError(f"{name}: no review_pr recorded; pass --review-pr <url>")
    pr = fetch_review_pr(url)
    if pr is None:
        raise ValueError(f"{name}: could not read the review PR at {url}")
    if pr.get("state") and pr.get("state") != "OPEN":
        # Raise like the other refusals here: the old ("closed", msg) tuple return
        # crashed the CLI, which indexed it as a snapshot dict.
        raise ValueError(
            f"{name}: the review PR is {str(pr['state']).lower()} -- reopen it to "
            f"submit, or the approval on it is not a live decision")
    blockers, _notes = moat_command_audit(pr)
    if blockers:
        raise ValueError(f"{name}: " + "; ".join(blockers))
    review = _approving_review(pr)
    if review is None:
        raise ValueError(f"{name}: no standing approval on {url}")
    resolved = pr.get("url") or url
    if fix_url and resolved.rstrip("/") == fix_url.rstrip("/"):
        obj["fix"]["review_pr"] = resolved      # keep the original review_pr intact
    else:
        obj["review_pr"] = resolved
    obj["pr_approval"] = {
        "approved_by": review.get("login"),
        "at": review.get("at") or now_iso(),
        # The commit the review was actually attached to, not the branch tip: if
        # anything was pushed between the approval and this snapshot, those differ,
        # and the approved one is the truth.
        "head_sha": review.get("commit") or pr.get("headRefOid"),
        "content_sha256": _content_digest(pr),
        "review_pr": resolved,
    }
    save_record(name, obj,
                f"{name}: snapshot the approval standing on the review PR\n\n"
                f"Approved by {obj['pr_approval']['approved_by']} for "
                f"{(obj['pr_approval'].get('head_sha') or '?')[:12]}.")
    return obj["pr_approval"]


# Why an approval does not cover what we would publish. Callers act on these very
# differently -- `stale-commits` and `stale-content` mean the approval was overtaken
# and the reviewer should be asked again, while `withdrawn` means that already
# happened and nobody should be pinged a second time -- so the verdict is a code
# rather than prose to be pattern-matched.
APPROVAL_CODES = ("ok", "none", "withdrawn", "closed", "stale-commits",
                  "stale-content", "record-mismatch", "unverifiable", "unreachable",
                  "bad-command")


def pr_approval_valid(name, live=True):
    """Does a standing approval still cover what we would publish right now?

    The checks that matter compare two GitHub facts against each other and consult
    nothing we wrote: the commit a review was attached to versus the branch's current
    tip, and when the title/body was last edited versus when the approval was given.
    That ordering is deliberate. Our own `pr_approval` snapshot lives in status.json
    on the project branch, which agents write to freely -- so anything that trusted
    the snapshot alone could be defeated by editing the file to match whatever the
    port now says. It is provenance (who approved, and when), cross-checked last.

    What this does NOT stop: someone with write access to the FORK force-pushing a
    different tree under the same sha, or GitHub itself lying. Both are outside the
    threat this addresses, which is an ordinary push landing after an approval and
    quietly riding out with it.

    Returns (ok, reason); see pr_approval_status for the machine-readable verdict.
    `live=False` checks only the snapshot, for callers with no network; it is weaker
    by construction, so say so rather than implying otherwise. An agent may neither
    grant an approval nor repair a stale one."""
    code, why = pr_approval_status(name, live=live)
    return (code == "ok", why)


def approval_currency(pr):
    """Is the approval on this fetched PR current, judging by the PR alone?

    Deliberately needs nothing we recorded, so it answers for a port whose approval
    has only just been clicked and never snapshotted -- the case that matters, since
    the click is the entire signal that a port may be submitted."""
    if pr.get("state") and pr.get("state") != "OPEN":
        return ("closed", f"the review PR is {str(pr['state']).lower()} -- reopen it to "
                          f"submit, or the approval on it is not a live decision")
    blockers, _notes = moat_command_audit(pr)
    if blockers:
        # Say WHICH command blocks rather than the generic "no standing approval":
        # a typo'd /moat line and a standing objection both need a person, and they
        # need to be told what to look at.
        return ("bad-command", "; ".join(blockers))
    review = _approving_review(pr)
    if review is None:
        # Withdrawn, dismissed, or overridden by a changes-requested review. Whoever
        # needs to act has been told; do not ping them again.
        return ("withdrawn", "no standing approval")

    # Was anything pushed after the approval? A review carries the commit it was
    # given against; compare it to where the branch is NOW.
    tip = pr.get("headRefOid")
    if review.get("commit") and tip and review["commit"] != tip:
        return ("stale-commits", f"approved {review['commit'][:8]} but the branch is now "
                                 f"at {tip[:8]} -- commits landed after the approval")
    # An approval given in the conversation box carries no commit, so compare times:
    # anything pushed after it was written is content nobody approved.
    head_at, at = pr.get("head_at"), review.get("at")
    if not review.get("commit"):
        if not head_at or not at:
            # Cannot tell whether anything landed after the approval. That is not the
            # same as "nothing did", and the precedent here is that an unreachable
            # review PR refuses rather than passing.
            return ("unverifiable", "no commit on the approval and the branch tip's "
                                    "date could not be read -- cannot show that "
                                    "nothing landed after it")
        if head_at > at:
            return ("stale-commits", f"approved at {at} but the branch tip dates from "
                                     f"{head_at} -- commits landed after the approval")

    # Was the title or body rewritten after the approval? GitHub leaves an approval
    # standing through an edit, so this is the only thing that catches it. Body and
    # title move different clocks (lastEditedAt vs the rename event), so both are
    # checked -- either one edited after the approval is content nobody approved.
    edited, at = pr.get("lastEditedAt"), review.get("at")
    if edited and at and edited > at:
        return ("stale-content", f"the body was edited at {edited}, after the "
                                 f"approval at {at}")
    renamed = pr.get("titleRenamedAt")
    if renamed and at and renamed > at:
        return ("stale-content", f"the title was renamed at {renamed}, after the "
                                 f"approval at {at}")
    return ("ok", f"approved by {review['login']} at {at}")


def pr_approval_status(name, live=True):
    """(code, detail) -- see APPROVAL_CODES. The logic pr_approval_valid describes."""
    obj = load_status(name)
    a = obj.get("pr_approval")
    if not a:
        return ("none", "no recorded approval")
    if not a.get("approved_by"):
        return ("none", "approval record has no approved_by")
    if not live:
        head = obj.get("head_sha")
        if head and not same_commit(a.get("head_sha"), head):
            return ("stale-commits", f"approved {(a.get('head_sha') or '?')[:8]}, fork is "
                                     f"now at {head[:8]} -- the code changed")
        return ("ok", f"recorded approval by {a['approved_by']} at {a['at']} "
                      f"(snapshot only, not re-checked against GitHub)")

    url = a.get("review_pr") or obj.get("review_pr")
    if not url:
        return ("none", "approval snapshot names no review PR to re-check")
    pr = fetch_review_pr(url)
    if pr is None:
        # Unreachable is not disproof. Refuse, but say which it is, so nobody reads
        # a network outage as a withdrawn approval.
        return ("unreachable", f"could not reach the review PR at {url}")

    code, why = approval_currency(pr)
    if code != "ok":
        return (code, why)
    review = _approving_review(pr)

    # 3. Provenance: does what we recorded still describe reality? A mismatch here
    #    means the record was edited, not that the port changed.
    if _content_digest(pr) != a.get("content_sha256"):
        return ("record-mismatch", "the recorded approval does not match the review PR's "
                                   "current title/body")
    if review.get("commit") and not same_commit(a.get("head_sha"), review["commit"]):
        return ("record-mismatch",
                f"the recorded approval names {(a.get('head_sha') or '?')[:8]}, but the "
                f"approval on GitHub is against {review['commit'][:8]}")

    return ("ok", f"approved by {review['login']} at {review.get('at')} on {url}, "
                  f"still standing at {(pr.get('headRefOid') or '?')[:8]}")


def license_tier(name, obj=None):
    """The project's licence tier, from utils/licenses.py. Unknown is tier 4.

    Resolves the record branch-first via project_record, like every other "what do
    we know about X" question -- load_status prefers the working tree, which let a
    stale trunk-era copy shadow a licence recorded on the project's own branch.
    Pass `obj` when the caller already holds the record, so both judge the same one."""
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    import licenses
    if obj is None:
        obj, _where = project_record(name)
        if obj is None:
            raise FileNotFoundError(str(status_path(name)))
    return licenses.tier_of(obj.get("license_spdx"))


def license_gate(name, obj=None):
    """(ok, reason) -- may this port be offered upstream on licence grounds?

    Tiers 1 and 2 are cleared to contribute and pass. Tier 3 and tier 4 ALWAYS wait
    for a person, every time, and a clearance covers one project rather than setting a
    precedent for its tier. An agent may record a clearance only by carrying someone's
    decision into the file; it may not decide, which is why a record with no
    `approved_by` satisfies nothing -- the same rule as a gate waiver.

    A licence nobody recorded is reported separately from one that is genuinely
    restrictive. Both block -- publishing on an unverified licence is exactly the
    mistake worth preventing -- but the remedies are different, and conflating them
    would put a hundred unread licences in front of a person as though each were a
    judgement call. Reading a repo's licence establishes a FACT and any agent may do
    it; deciding to contribute under a restrictive one is a decision and never is."""
    if obj is None:
        obj, _where = project_record(name)
        if obj is None:
            raise FileNotFoundError(str(status_path(name)))
    spdx = obj.get("license_spdx")
    tier = license_tier(name, obj)
    if tier <= 2:
        return (True, f"tier {tier} ({spdx})")
    c = obj.get("license_clearance") or {}
    if c.get("approved_by"):
        return (True, f"tier {tier}, cleared by {c['approved_by']} at {c.get('at')}")
    if c:
        return (False, f"tier {tier}: a clearance is recorded but nobody approved it")
    if not spdx:
        return (False, "licence not recorded -- read the repo's licence and record it "
                       "in status.json.license_spdx; do not assume")
    return (False, f"tier {tier} ({spdx}): needs approval before it can be offered upstream")


# Intake's verdict, as data. The write-up in notes.md is the argument; this is the
# row in the queue a person reads. It is a RECOMMENDATION and never a decision --
# `verdict: decline` records what intake would choose, and only a person turns that
# into a disposition (see the autonomy boundary).
INTAKE_VERDICTS = ("fork", "decline")


def set_intake(name, verdict, summary, reason=None, duplicate=None, viable=None):
    """Record intake's recommendation on the project."""
    if verdict not in INTAKE_VERDICTS:
        raise ValueError(f"verdict must be one of {INTAKE_VERDICTS}")
    if verdict == "decline" and reason not in SKIP_REASONS:
        raise ValueError(f"a decline must name a reason from {SKIP_REASONS}")
    obj = load_status(name)
    obj["intake"] = {"verdict": verdict, "reason": reason,
                     "duplicate_effort": duplicate, "viable": viable,
                     "summary": summary, "at": now_iso()}
    save_status(name, obj)
    return obj["intake"]


def suggest_waiver(name, gate, reason):
    """Record an agent's CASE for waiving a gate. Satisfies nothing by itself.

    Written at the moment the evidence is in hand, which is the point of it. The
    obstacle a waiver answers is found mid-port, often by an unattended run with
    nobody to ask, and a finding with nowhere to go is one the next porter rediscovers
    -- ZhiLight and LichtFeld-Studio each had their Windows determination made once and
    then hand-copied onto the second Windows arch, "carried from windows-gfx1101
    determination", because this field had no writer.

    An unapproved waiver BLOCKS `pr_ready` rather than clearing anything, so recording
    a suggestion can never let a port out early. It only makes the case findable."""
    if gate not in WAIVABLE_GATES:
        raise ValueError(f"{gate!r} is not waivable (config/arches.toml waivable = "
                         f"{sorted(WAIVABLE_GATES)}). A gate nobody may waive is a gate "
                         f"that has to be satisfied or the port scoped around it")
    if not (reason or "").strip():
        raise ValueError(f"{name}: --reason is required; the case IS the record")
    obj = load_status(name)
    existing = (obj.get("waivers") or {}).get(gate) or {}
    if existing.get("approved_by"):
        raise ValueError(
            f"{name}: {gate} is already waived by {existing['approved_by']}. Overwriting "
            f"an approval with a suggestion would quietly un-approve it")
    obj.setdefault("waivers", {})[gate] = {"reason": reason, "suggested_at": now_iso()}
    save_status(name, obj)
    return obj["waivers"][gate]


def approve_waiver(name, gate, by, reason=None):
    """A maintainer's approval of a gate waiver, which is what makes it satisfy.

    Approves a specific CASE: without a suggestion on file, a reason must be given
    here, so the record always says what was waived and why rather than only that
    something was."""
    if gate not in WAIVABLE_GATES:
        raise ValueError(f"{gate!r} is not waivable (config/arches.toml waivable = "
                         f"{sorted(WAIVABLE_GATES)})")
    if not (by or "").strip():
        raise ValueError(
            f"{name}: --by is required. A waiver without it satisfies nothing, which is "
            f"what stops an agent certifying its own way past the one escapable gate")
    obj = load_status(name)
    w = dict((obj.get("waivers") or {}).get(gate) or {})
    if reason:
        w["reason"] = reason
    if not w.get("reason"):
        raise ValueError(f"{name}: no waiver suggested for {gate}; pass --reason to "
                         f"record what is being waived")
    w["approved_by"] = by
    w["at"] = now_iso()
    obj.setdefault("waivers", {})[gate] = w
    save_record(name, obj, f"{name}: {gate} waiver approved by {by}")
    return w


def refuse_waiver(name, gate, by, note):
    """A person declining a suggested waiver, with what they want done instead.

    The answer has to be recordable in BOTH directions or the queue only empties one
    way: an unanswered suggestion is reprinted by every orient forever, and the only
    way to silence it is approving it or hand-editing the file -- which is pressure in
    exactly the wrong direction, on the one gate that may be escaped at all.

    The refusal stays in the record rather than deleting the suggestion, because the
    next agent to hit the same wall needs to know it was asked and answered, and what
    to investigate instead. The gate stays unsatisfied either way."""
    if not (by or "").strip():
        raise ValueError(f"{name}: --by is required; a refusal is a person's answer too")
    if not (note or "").strip():
        raise ValueError(
            f"{name}: --note is required. A refusal without one leaves the next agent "
            f"exactly where the last one was, and it will suggest the same waiver again")
    obj = load_status(name)
    w = dict((obj.get("waivers") or {}).get(gate) or {})
    if not w:
        raise ValueError(f"{name}: no waiver suggested for {gate}")
    if w.get("approved_by"):
        raise ValueError(f"{name}: {gate} is already approved by {w['approved_by']}; "
                         f"withdrawing an approval is that person's call, not a refusal")
    w.update({"refused_by": by, "refused_at": now_iso(), "refused_note": note})
    obj.setdefault("waivers", {})[gate] = w
    save_record(name, obj, f"{name}: {gate} waiver refused by {by}")
    return w


def pending_waivers():
    """Every waiver awaiting an answer, across all refs. These BLOCK their project's
    PR, and the only thing that resolves one is a person. A refused one is answered and
    so is not here -- it still blocks, and the block is now a known quantity."""
    out = []
    for name, obj, _where in project_records():
        for gate, w in (obj.get("waivers") or {}).items():
            if not w.get("approved_by") and not w.get("refused_by"):
                out.append((name, gate, w.get("reason") or "", w.get("suggested_at") or ""))
    return sorted(out)


def record_license_clearance(name, approved_by, note=None):
    """Record a person's decision to allow a tier 3/4 project upstream."""
    obj = load_status(name)
    obj["license_clearance"] = {"approved_by": approved_by, "at": now_iso(),
                                "tier": license_tier(name, obj),
                                **({"note": note} if note else {})}
    save_record(name, obj, f"{name}: licence clearance recorded, approved by {approved_by}")
    return obj["license_clearance"]


def set_review_pr(name, url):
    """Record the review PR on our own fork -- where the port gets approved.

    REFUSED while a required gate is unsatisfied. Approving that PR is what opens the
    upstream one, so recording it for an unfinished port puts that decision in front of
    a person early, asserting the work is ready when it is not.

    `upstream.py --review --apply` already refuses to OPEN one before the gates pass.
    This refuses to RECORD one, which is the half that closes the route around it: the
    instruction not to reach for `gh pr create` is what three reviewers broke in a
    single session, each in a different way, and an instruction is the weakest thing to
    put in front of a behaviour that has already failed three times.

    Clearing is always allowed. Undoing a mistake must not require the gates to pass --
    the two PRs opened this way had to be retracted before anything could be fixed."""
    obj = load_status(name)
    if url:
        unmet = sorted(unsatisfied_gates(obj))
        if unmet:
            raise ValueError(
                f"{name}: cannot record a review PR while {', '.join(unmet)} "
                f"{'is' if len(unmet) == 1 else 'are'} unsatisfied. The review PR is "
                f"where a person approves the FINISHED port, and their approval on it "
                f"opens the upstream PR. Finish the gates, then "
                f"`upstream.py --review --apply --name {name}` opens and records it.")
    obj["review_pr"] = url or None
    save_record(name, obj, f"{name}: review PR "
                           + (f"recorded -- {url}" if url else "cleared"))
    return obj


# The PR lifecycle is recorded through save_record rather than save_status: each of
# these states a fact about a project without touching its files, and the project is
# branch-resident at exactly the moment they are called. Opening one from a session
# standing anywhere else used to fail, which is how the documented submission command
# could not submit.
def set_pr_open(name, pr_url, pr_number):
    """Record the upstream PR. Project-level: it changes nothing an arch validated."""
    obj = load_status(name)
    obj["pr_url"] = _clean_pr_url(pr_url)
    obj["pr_number"] = int(pr_number)
    obj["pr_opened_at"] = now_iso()
    obj["pr_state"] = "open"
    # What the open PR shows. From here on, head_sha may run ahead of this on a
    # staging branch (see fix_branch); the PR branch itself moves only through
    # `upstream.py --merge-fix`, which is what advances this field.
    obj["published_sha"] = obj.get("head_sha")
    save_record(name, obj, f"{name}: upstream PR opened -- {obj['pr_url']}")
    return obj


def set_pr_merged(name):
    """Record that the upstream PR merged."""
    obj = load_status(name)
    if "pr_url" not in obj:
        raise ValueError(f"{name}: no PR recorded, cannot mark as merged")
    obj["pr_merged_at"] = now_iso()
    obj["pr_state"] = "merged"
    save_record(name, obj, f"{name}: upstream PR merged -- {obj['pr_url']}")
    return obj


def set_pr_closed(name, note=None):
    """Record that the upstream PR closed without merging."""
    obj = load_status(name)
    if "pr_url" not in obj:
        raise ValueError(f"{name}: no PR recorded, cannot mark as closed")
    obj["pr_state"] = "closed"
    obj["pr_closed_at"] = now_iso()
    if note:
        obj["pr_closed_note"] = note
    save_record(name, obj, f"{name}: upstream PR closed without merging -- "
                           f"{note or obj['pr_url']}")
    return obj


def _fork_repo(name):
    return PROJECTS / name / "src"


# ---- fix rounds on an open upstream PR -------------------------------------
#
# Once an upstream PR is open, its head branch is upstream-visible: any push to it
# lands in front of the maintainer immediately, before review, revalidation, or a
# person's approval. So a maintainer-requested fix is staged on `moat-fix-<pr#>`,
# cut from the published tip, and the pipeline (porter -> reviewer -> validators)
# runs against that staging tip. head_sha follows the staging tip -- it keeps
# meaning "fork port tip under evaluation", so revalidate/pr-gate derivation is
# unchanged -- while `published_sha` records what the open PR shows. The one thing
# that may move the PR branch is `upstream.py --merge-fix --apply`, after a person
# approves the delta on a fork review PR (see set_fix_merged).

def record_writable_here(name):
    """Can this checkout write this project's record at all? (ok, why).

    save_record's own precedence, asked in advance. A trunk-resident record with no
    `port/<name>` branch is writable only from a checkout of the protected trunk, so
    every fix-round command would raise -- and the merge path would raise AFTER it
    had already moved the upstream PR. Asking first turns that into a refusal with a
    remedy."""
    obj, where = project_record(name)
    if obj is None:
        return (False, f"{name}: no record on any ref")
    if status_path(name).exists() and writable_here(name, where):
        return (True, "in this working tree")
    branch = port_branch_of(name)
    if branch:
        return (True, f"on {branch}")
    return (False,
            f"{name}: its record is on the trunk and this checkout is on "
            f"{current_branch()}, which may not write it. Work in flight belongs on "
            f"`port/{name}`: `python3 utils/moatlib.py fix-branch {name}` re-homes "
            f"the folder there before its first write")


def ensure_port_branch(name):
    """Put a trunk-resident project's folder back on `port/<name>`. (moved, message).

    A maintainer asking for a code change makes a finished project unfinished again,
    and belongs_on_branch says its folder has to go back -- but that predicate only
    flips once head_sha moves, which cannot happen until a fix round is recorded,
    which cannot be written while the record sits on the protected trunk. Something
    has to break the circle, and the fix round is the event that means "in flight
    again", so it breaks it: cut the claim from the trunk, where the folder already
    is, and every later write lands on the branch through save_record.

    Not a human decision -- the project was adopted long ago and this creates no
    fork, no PR and no upstream contact. Idempotent; a project that already has a
    branch is left exactly alone."""
    if port_branch_of(name):
        return (False, f"{name}: already on port/{name}")
    main_ref = _git("rev-parse", "--verify", "-q", "origin/main",
                    check=False).stdout.strip()
    if not main_ref:
        raise ValueError(f"{name}: cannot read origin/main to cut port/{name} from")
    if not _ref_read("origin/main", f"projects/{name}/status.json"):
        raise ValueError(f"{name}: its record is not on origin/main either -- there "
                         f"is nothing to re-home; find where it lives first")
    r = _git("push", "-q", "origin", f"{main_ref}:refs/heads/port/{name}", check=False)
    if r.returncode:
        raise ValueError(f"{name}: could not create port/{name}: "
                         f"{(r.stderr or r.stdout).strip()}")
    # Do not wait for a fetch to make the new branch visible: port_branch_of reads
    # remote-tracking refs, and save_record is about to ask.
    _git("update-ref", f"refs/remotes/origin/port/{name}", main_ref, check=False)
    _PORT_BRANCH_MAP.clear()
    return (True, f"{name}: re-homed onto port/{name} (cut from origin/main at "
                  f"{main_ref[:12]}) -- work in flight lives on its own branch")


def _verified_published_sha(obj, name):
    """What the open PR actually shows right now, for a record written before
    published_sha existed.

    The backfilled value becomes the ancestry baseline the merge fast-forward is
    checked against AND the baseline --dry-run calls a maintainer push, so it is the
    last field in this flow that should be assumed. head_sha is the right guess --
    nothing was supposed to have moved the PR branch -- but the entire point of this
    flow is that "supposed to" is not evidence, so ask GitHub and only accept the
    guess when it agrees."""
    repo, num = _pr_ref(obj.get("pr_url"))
    if not repo:
        raise ValueError(f"{name}: no usable pr_url to verify the published tip against")
    live = _gh_json("pr", "view", num, "--repo", repo, "--json", "headRefOid")
    if not live or not live.get("headRefOid"):
        raise ValueError(
            f"{name}: cannot read {repo}#{num} to confirm what the open PR shows, and "
            f"the published tip is the baseline every later check rests on -- retry "
            f"when GitHub is reachable rather than recording a guess")
    head = live["headRefOid"]
    if not same_commit(head, obj.get("head_sha")):
        raise ValueError(
            f"{name}: the open PR is at {head[:12]} but the record says head_sha "
            f"{(obj.get('head_sha') or '?')[:12]}. The published tip cannot be "
            f"inferred from a record that disagrees with the PR -- this is the "
            f"HEAD-MOVED case (`upstream.py --dry-run`): read what landed and let a "
            f"person rule on it before staging a fix round")
    return head


def fix_branch(name):
    """Establish (or report) the staging branch for a fix round.

    Record-only: the branch itself is the porter's git work; this names it so no
    porter invents a name, and pins the base so descent from the published tip can
    be checked at merge time. Idempotent for the recorded branch; refuses to open a
    second round while one is in flight. A record from before published_sha existed
    is backfilled from what the PR actually shows, confirmed against GitHub."""
    obj, _where = project_record(name)
    if obj is None:
        raise FileNotFoundError(str(status_path(name)))
    if obj.get("pr_state") != "open":
        raise ValueError(f"{name}: no open upstream PR -- fixes stage only while "
                         f"one is open; otherwise the port branch is still private "
                         f"and the porter pushes it directly")
    if not obj.get("pr_number"):
        raise ValueError(f"{name}: pr_state is open but no pr_number is recorded")
    branch = f"moat-fix-{obj['pr_number']}"
    fix = obj.get("fix")
    if fix:
        if fix.get("branch") != branch:
            raise ValueError(f"{name}: a fix round is already in flight on "
                             f"{fix.get('branch')!r} -- one staging branch at a "
                             f"time; merge or abandon it first")
        return fix
    # Before the first write, not after it fails: most open-PR records are on the
    # trunk, and every command in this flow writes the record.
    moved, why = ensure_port_branch(name)
    if moved:
        print(why, file=sys.stderr)
        # Write on top of exactly what the new branch holds, which is the trunk's
        # copy -- this checkout's copy of somebody else's folder may be older.
        obj, _where = project_record(name)
    ok, why = record_writable_here(name)
    if not ok:
        raise ValueError(why)
    if not obj.get("published_sha"):
        obj["published_sha"] = _verified_published_sha(obj, name)
    obj["fix"] = {"branch": branch, "base_sha": obj["published_sha"],
                  "review_pr": None, "opened_at": now_iso()}
    save_record(name, obj, f"{name}: fix round staged on {branch} "
                           f"(base {obj['published_sha'][:12]})")
    return obj["fix"]


def set_fix_review_pr(name, url):
    """Record the fork review PR where a person approves the staged delta.

    Mirrors set_review_pr's refusal: recording one asserts the delta is finished,
    so every required gate must hold at the staging tip first. Clearing is always
    allowed -- undoing a mistake must not require the gates to pass."""
    # project_record, not load_status: the freshest record rather than the nearest.
    # A port branch carries a copy of every folder the trunk had when it was cut, so
    # reading the working tree here would edit a stale copy and save_record would
    # then write it over the branch's real one.
    obj, _where = project_record(name)
    if obj is None:
        raise FileNotFoundError(str(status_path(name)))
    if not obj.get("fix"):
        raise ValueError(f"{name}: no fix round in flight (moatlib.py fix-branch "
                         f"establishes one)")
    if url:
        ready, blocking, _ = fix_ready(name)
        if not ready:
            listed = ", ".join(f"{p}={s}" for p, s in blocking)
            raise ValueError(
                f"{name}: cannot record a fix review PR while blocked: {listed}. "
                f"The fix review PR is where a person approves the FINISHED delta; "
                f"`upstream.py --fix-review --apply --name {name}` opens and "
                f"records it once the gates pass.")
    obj["fix"]["review_pr"] = url or None
    save_record(name, obj, f"{name}: fix review PR "
                           + (f"recorded -- {url}" if url else "cleared"))
    return obj


def set_fix_merged(name, new_published_sha):
    """The approved staging tip is now what the open PR shows.

    Called by the trusted merge path after the fast-forward push succeeds; the
    approval checks live there, not here. Clears the fix block -- the round is
    over, and the next one starts from the new published tip."""
    obj, _where = project_record(name)          # freshest, not nearest; see above
    if obj is None:
        raise FileNotFoundError(str(status_path(name)))
    fix = obj.get("fix")
    if not fix:
        raise ValueError(f"{name}: no fix round in flight")
    new_published_sha = full_sha(new_published_sha, _fork_repo(name))
    obj["published_sha"] = new_published_sha
    obj["fix"] = None
    obj["fix_merged_at"] = now_iso()
    save_record(name, obj,
                f"{name}: fix round merged -- {fix.get('branch')} fast-forwarded "
                f"the PR branch to {new_published_sha[:12]}")
    return obj


def set_published_sha(name, sha):
    """Stamp what the open upstream PR shows, on a record from before the fix flow.

    The caller (the upstream.py reconciler) has already verified `sha` against the
    live PR head; this only guards the record's own consistency: the PR must still
    be open, the value must agree with head_sha -- a record whose head disagrees
    with the PR is the HEAD-MOVED case and needs a person, not a stamp -- and an
    existing different value is never silently replaced (only the trusted merge
    path advances one). Idempotent on a matching stamp."""
    obj, _where = project_record(name)
    if obj is None:
        raise FileNotFoundError(str(status_path(name)))
    if obj.get("pr_state") != "open":
        raise ValueError(f"{name}: pr_state is {obj.get('pr_state')!r}, not open -- "
                         f"published_sha only describes an open PR")
    if not same_commit(sha, obj.get("head_sha")):
        raise ValueError(f"{name}: {sha[:12]} does not match head_sha "
                         f"{(obj.get('head_sha') or '?')[:12]} -- a record that "
                         f"disagrees with the PR is a person's to sort out")
    cur = obj.get("published_sha")
    if cur and not same_commit(cur, sha):
        raise ValueError(f"{name}: published_sha is already {cur[:12]}; only the "
                         f"trusted merge path may advance it")
    if cur:
        return obj
    obj["published_sha"] = sha
    save_record(name, obj, f"{name}: published_sha backfilled -- the open PR shows "
                           f"{sha[:12]} (verified against the live PR head)")
    return obj


def pr_state_of(name, refresh=False):
    """(pr_state, where) resolved from wherever the record lives, or (None, why).

    The distinction the fork pre-push hook is built on: None means the state could
    not be READ, which is not the same answer as "no PR". `refresh` re-fetches the
    project's refs first, because the question is about a write another host made --
    a remote-tracking ref that predates the PR opening would answer "not open" about
    a PR that is open."""
    if refresh:
        _git("fetch", "--quiet", "origin",
             f"+refs/heads/port/{name}:refs/remotes/origin/port/{name}", check=False)
        _git("fetch", "--quiet", "origin",
             "+refs/heads/main:refs/remotes/origin/main", check=False)
        _PORT_BRANCH_MAP.clear()
    obj, where = project_record(name)
    if obj is None:
        return (None, f"no record for {name} on any ref")
    return (obj.get("pr_state") or "none", where)


# Installed into a fork clone's .git/hooks/pre-push by protect_fork. It asks moatlib
# for the state rather than reading a path, because the record usually is NOT a file
# in the MOAT working tree: an in-flight project's folder lives on its own branch,
# and `projects/<name>/status.json` is absent from every checkout standing anywhere
# else. Reading the path directly made the hook exit 0 on exactly the hosts and
# exactly the projects it exists to protect. MOAT_PUBLISH=1 is set only by the
# trusted merge path after its approval checks pass.
FORK_HOOK_MARKER = "# moat-fork-hook"
_FORK_HOOK = """#!/usr/bin/env bash
{marker} v2
# Refuses pushes to the upstream PR's head branch while that PR is open.
# Installed by `moatlib.py protect-fork {name}`; see AGENTS.md on fix rounds.
set -u
[ "${{MOAT_PUBLISH:-}}" = "1" ] && exit 0

# Only the PR's head branch is guarded, so the staging branch and any scratch
# branch push normally -- and the lookup below is paid for only on the one push
# that could reach a maintainer.
targets=0
while read -r _local _lsha remote _rsha; do
  [ "$remote" = "refs/heads/{branch}" ] && targets=1
done
[ "$targets" = 1 ] || exit 0

# stdout only, so nothing a warning or a git message writes to stderr can be read
# as the answer; the exit status is what says whether there IS an answer.
state=$(python3 "{moatlib}" pr-state "{name}" --refresh 2>/dev/null)
if [ $? -ne 0 ]; then
  # A guard that cannot tell must not be the reason an unreviewed commit reached a
  # maintainer. Refuse, and say what to run to find out why.
  echo >&2 "moat: cannot tell whether {name} has an open upstream PR, so this push"
  echo >&2 "moat: to {branch} is refused rather than guessed at. Check with:"
  echo >&2 "moat:   python3 {moatlib} pr-state {name}"
  python3 "{moatlib}" pr-state "{name}" 2>&1 >/dev/null | sed >&2 's/^/moat:   /'
  exit 1
fi
if [ "$state" = "open" ]; then
  echo >&2 "moat: {branch} is the head of an OPEN upstream PR -- a push to it is"
  echo >&2 "moat: upstream-visible before anyone reviewed or approved it."
  echo >&2 "moat: Stage the fix instead: python3 utils/moatlib.py fix-branch {name}"
  echo >&2 "moat: (the approved merge runs through: utils/upstream.py --merge-fix)"
  exit 1
fi
exit 0
"""


def protect_fork(name):
    """Install the pre-push hook that keeps an open PR's branch from moving.

    Returns (level, message) where level is "ok", "skip" or "warn"; the CLI sends
    "warn" to stderr so an unprotected clone is not indistinguishable from a
    protected one in orient's output. Idempotent; refuses to clobber a hook that is
    not ours (say so, loudly, rather than silently replacing whatever someone
    installed) but does replace an older moat hook. A missing clone installs nothing
    and says so -- absence of a clone is absence of the risk."""
    repo = _fork_repo(name)
    git_dir = repo / ".git"
    if not git_dir.exists():
        return ("skip", f"{name}: no fork clone at {repo} -- nothing to protect")
    if not git_dir.is_dir():
        return ("warn", f"{name}: {repo} keeps its git dir elsewhere (a .git file), so "
                        f"the hook path cannot be derived here -- UNPROTECTED")
    obj, _where = project_record(name)
    if obj is None:
        return ("warn", f"{name}: a fork clone exists but no record does; cannot tell "
                        f"which branch to protect")
    text = _FORK_HOOK.format(marker=FORK_HOOK_MARKER, name=name,
                             moatlib=str(REPO_ROOT / "utils" / "moatlib.py"),
                             branch=obj.get("fork_branch") or PORT_BRANCH)
    hook = git_dir / "hooks" / "pre-push"
    if hook.exists():
        current = hook.read_text()
        if current == text:
            return ("ok", f"{name}: fork pre-push hook installed and current")
        if FORK_HOOK_MARKER not in current:
            return ("warn", f"{name}: a NON-moat pre-push hook is installed at {hook}; "
                            f"not touching it -- {obj.get('fork_branch') or PORT_BRANCH} "
                            f"is UNPROTECTED in this clone")
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(text)
    hook.chmod(hook.stat().st_mode | 0o111)
    return ("ok", f"{name}: fork pre-push hook installed")


# Tracked file kinds whose UNCOMMITTED modification in a fork is the integrity-gap
# fingerprint: a validation built against local source/build edits that were never
# committed, leaving the branch (and any PR off it) unbuildable. (Motivated by the
# baspacho/arrayfire 2026-06 gaps, incl. an arrayfire vcpkg.json -- so build
# MANIFESTS count, not just code.) Untracked files (build artifacts, scratch) do not.
_INTEGRITY_SUFFIXES = (
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".hip",
    ".inl", ".inc", ".py", ".go", ".rs", ".jl", ".java", ".f", ".f90", ".f95",
    ".cmake", ".toml", ".json", ".cfg", ".pri", ".pro",
)
_INTEGRITY_NAMES = (
    "cmakelists.txt", "makefile", "setup.py", "setup.cfg", "pyproject.toml",
    "meson.build", "conanfile.py", "conanfile.txt", "vcpkg.json", "package.json",
    "cargo.toml", "build.gradle",
)


def uncommitted_source_files(name):
    """Tracked source/build files modified-but-not-committed in a project's fork
    clone -- the integrity-gap fingerprint (a validation that built against local
    edits never committed, so the branch/PR will not build). Returns a list of
    (status_code, path); untracked files (build artifacts, scratch) and ignored
    files are excluded. Never raises: a missing clone or unavailable git yields []
    (absence of a clone is not an integrity claim)."""
    repo = _fork_repo(name)
    if not Path(repo).is_dir():
        return []
    try:
        r = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo),
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if code.strip() in ("??", "!!"):  # untracked / ignored -> not an integrity gap
            continue
        # The fingerprint is validated CONTENT not committed: modified/added/renamed/
        # copied. Pure deletions (a dirty checkout dropping dev/benchmark files) leave
        # the branch buildable -- not an integrity gap.
        if not any(c in code for c in ("M", "A", "R", "C")):
            continue
        p = path.split(" -> ")[-1].strip().strip('"')  # handle rename "old -> new"
        base = p.rsplit("/", 1)[-1].lower()
        if base in _INTEGRITY_NAMES or any(base.endswith(s) for s in _INTEGRITY_SUFFIXES):
            out.append((code.strip(), p))
    return out


def _classify_safe(repo, old_sha, new_sha):
    """Classify a fork delta, returning None on any failure so the caller falls
    back to revalidation. Conservative by construction: machinery missing, repo
    absent, or sha unreachable all yield None, never a false carry-forward."""
    if not old_sha or not Path(repo).is_dir():
        return None
    try:
        d = str(Path(__file__).resolve().parent)
        if d not in sys.path:
            sys.path.insert(0, d)
        import changeclass
        return changeclass.classify(str(repo), old_sha, new_sha)
    except Exception:
        return None


def advance_head(name, new_sha, repo=None):
    """A porter commit advanced the shared fork HEAD. Each platform that had
    passed at a different HEAD is re-examined against the source delta from its
    validated_sha to new_sha (the cross-platform regression guard):

      - arch-independent inert (documentation-only, or a comment/format change
        with no __LINE__ hazard) cannot alter any target's compiled output, so
        validation carries forward (validated_sha bumped, stays completed).
      - everything else flips to revalidate. Rename/refactor deltas are inert but
        not arch-independent (an exported-symbol rename changes behavior with an
        identical instruction stream), so the validator confirms them per-arch
        with a binary-equivalence check before re-running GPU tests; unbuildable
        arches simply revalidate.

    On any classification failure the platform revalidates -- the safe default.

    A platform that FAILED is re-examined the same way and for the same reason. Its
    failure is evidence about the commit it happened on, so a HEAD move normally
    retires it and sends the arch back to a validator (see failure_stands) -- but a
    delta that cannot change compiled output cannot be the fix, so the failure is
    carried forward to the new head instead."""
    obj = load_status(name)
    repo = repo or _fork_repo(name)
    new_sha = full_sha(new_sha, repo)
    prev_head = obj.get("head_sha")
    obj["head_sha"] = new_sha
    for plat in list(obj["platforms"]):
        blk = obj["platforms"][plat]
        state = blk.get("state")
        if state == "completed":
            old = blk.get("validated_sha")
            if same_commit(old, new_sha):
                continue
            verdict = _classify_safe(repo, old, new_sha)
            if verdict is not None and verdict.arch_independent:
                blk["validated_sha"] = new_sha
                blk["updated_at"] = now_iso()
                blk["carry_forward"] = {"from": old, "to": new_sha,
                                        "method": "source-class", "class": verdict.cls,
                                        "detail": verdict.detail[:200], "at": now_iso()}
            # No else. A block that cannot be carried forward keeps its `completed` and
            # its old validated_sha, which IS the record: this arch proved that commit
            # and has not proved this one. `revalidate` follows from the two shas
            # differing, so writing it down would only be a second copy that can go
            # stale.
        elif state == "validation-failed":
            # The same guard facing the other way. A HEAD move retires a failure (see
            # failure_stands), which is right when the commit was a fix and wrong when
            # it was a README edit -- a delta that cannot change any target's compiled
            # output cannot have fixed anything, so carry the FAILURE forward and let
            # the arch keep asking the porter for a real one.
            #
            # A block written before failures carried a sha is stamped with the head
            # being superseded, which is the head it failed against -- so a legacy
            # record heals itself the first time a porter advances the branch, rather
            # than needing five port branches migrated by hand.
            old = blk.get("failed_sha") or prev_head
            if not old or same_commit(old, new_sha):
                continue
            verdict = _classify_safe(repo, old, new_sha)
            # Inert: not the fix, so the failure moves up to the new head and goes on
            # standing. Anything else retires it -- and the sha is written down either
            # way, because a block that says only `validation-failed` cannot be judged
            # at all and would ask the porter for a fix it has already had.
            failed = (new_sha if verdict is not None and verdict.arch_independent
                      else old)
            if failed != blk.get("failed_sha"):
                blk["failed_sha"] = failed
                blk["updated_at"] = now_iso()
    save_status(name, obj)
    return obj


def carry_forward(name, platform, new_sha, method, detail):
    """Carry one platform's validation forward to new_sha without a GPU re-run,
    because the change was proven behavior-preserving. method is 'source-class'
    (doc/comment-only) or 'binary-equiv' (compiled code objects identical on this
    arch, dynamic symbol table included). The validator calls this on a revalidate
    delta whose compiled output is unchanged; advance_head handles the
    arch-independent source classes itself. Records provenance for audit."""
    obj = load_status(name)
    blk = obj["platforms"][platform]
    if blk.get("state") != "completed":
        raise ValueError(f"{name}/{platform}: carry_forward needs a completed validation "
                         f"to carry, not {blk.get('state')}")
    new_sha = full_sha(new_sha, _fork_repo(name))
    ts = now_iso()
    blk["state"] = "completed"
    blk["completed_at"] = ts
    blk["validated_sha"] = new_sha
    blk["updated_at"] = ts
    blk["carry_forward"] = {"to": new_sha, "method": method, "detail": detail[:200], "at": ts}
    save_status(name, obj)
    return obj


# States that count as "this port is validated" for opening validation to other
# archs and for satisfying dependents. Just the one now that the PR lifecycle is
# project-level: an arch that validated stays `completed` whatever its PR does.
PORT_DONE_STATES = ("completed",)


def port_done(obj):
    """Has any arch validated this port? With no lead, any one is sufficient."""
    return any(b.get("state") in PORT_DONE_STATES for b in validations(obj).values())


# ---- where a project's record lives ---------------------------------------
#
# A project is in exactly one of three places and the difference matters:
#   local   -- projects/<name>/ in the working tree, whatever branch that is
#   trunk   -- on origin/main: it reached a terminal state
#   branch  -- on origin/port/<name>: adopted and in flight
# Absent from all three means genuinely not adopted.
#
# Reading only the working tree conflates "in flight elsewhere" with "nobody has
# looked at it", and those lead opposite ways: the first must not be re-screened,
# the second must be. The PhoenixOS migration canary proved it -- with the folder
# moved to its branch, `port_request check` said "a request would be new" and
# `triage review` re-listed a planned project as an un-adopted candidate.

_REF_CACHE = {}
_BRANCH = []          # one-element cache: the branch cannot change mid-process


def current_branch():
    """The checked-out branch, resolved once. project_record asks per project, and
    spawning `git rev-parse` 156 times per scan was slow enough to time out a
    `triage review`."""
    if not _BRANCH:
        _BRANCH.append(_git("rev-parse", "--abbrev-ref", "HEAD",
                            check=False).stdout.strip())
    return _BRANCH[0]


def _ref_read(ref, path):
    """A file's contents at a git ref, or None. Local object store, no network."""
    key = (ref, path)
    if key not in _REF_CACHE:
        r = _git("show", f"{ref}:{path}", check=False)
        _REF_CACHE[key] = r.stdout if r.returncode == 0 else None
    return _REF_CACHE[key]


_PORT_BRANCH_MAP = []   # one-element cache, for the same reason as _BRANCH above


def port_branches():
    """{project name: ref} for every port/<name> branch the remote is known to have.

    Reads local remote-tracking refs, so it is only as fresh as the last fetch --
    orient.sh fetches before it asks. Resolved once per process: every project
    resolution asks for it, and a `for-each-ref` per project is what made a scan slow
    enough to time out."""
    if not _PORT_BRANCH_MAP:
        out = {}
        r = _git("for-each-ref", "--format=%(refname)", "refs/remotes/origin/port/",
                 check=False)
        for ref in r.stdout.splitlines():
            ref = ref.strip()
            if ref:
                out[ref.rsplit("/", 1)[-1]] = ref
        _PORT_BRANCH_MAP.append(out)
    return _PORT_BRANCH_MAP[0]


def port_branch_of(name):
    """`port/<name>` as the remote actually spells it, or None if there is none.

    The convention is exact, but a mismatch must fail loudly rather than silently drop
    the project: a branch cut as `port/hami-core` for the project `HAMi-core` resolved
    to nothing, and a finished screen went invisible to the queue and to every sweep."""
    branches = port_branches()
    if name in branches:
        return f"port/{name}"
    return next((f"port/{c}" for c in branches if c.lower() == name.lower()), None)


def project_record(name):
    """(status object, where) for a project, from wherever its record lives.

    `where` is "local", "trunk", "branch" or None.

    The project's OWN branch wins over the working tree, because an in-flight project
    is worked there and the trunk may still carry a copy that predates it. colmap hit
    this on the 2026-08-07 rerun: the screen was recorded on port/colmap while a
    scaffold stub of the same project sat on the trunk, and reading local-first made
    the queue see a project with no intake record. Being ON that branch is not a
    special case -- the working tree IS the branch then, so the local read is both
    correct and cheaper."""
    path = f"projects/{name}/status.json"
    on_branch = current_branch() == f"port/{name}"
    if not on_branch:
        raw = _ref_read(f"origin/{port_branch_of(name) or f'port/{name}'}", path)
        if raw:
            try:
                return (json.loads(raw), "branch")
            except json.JSONDecodeError:
                pass
    if status_path(name).exists():
        try:
            return (load_status(name), "local")
        except (ValueError, json.JSONDecodeError):
            return (None, None)
    for ref, where in (("origin/main", "trunk"),
                       (f"origin/port/{name}", "branch")):
        raw = _ref_read(ref, path)
        if raw:
            try:
                return (json.loads(raw), where)
            except json.JSONDecodeError:
                pass
    return (None, None)


def known_platforms():
    """Every platform any project records. Derived rather than configured: a platform
    exists because a host reported it, so the roster is whatever the records contain."""
    out = set()
    for name in all_projects():
        obj, _ = project_record(name)
        out |= set((obj or {}).get("platforms") or {})
    return out


def all_projects():
    """{name: where} for every project this repo knows about, across refs.

    `where` agrees with project_record: a project's own branch outranks a copy of it
    on the trunk or in this working tree, unless that branch is the one checked out.
    The two used to disagree for exactly one project -- colmap, the only one carrying
    both a branch and a trunk stub -- and the disagreement led opposite ways. The
    resolver said screened-on-a-branch; everything reading this map said
    unclaimed-and-local, so the "actionable elsewhere, go check it out" hint filtered
    it away while the selector offered to screen it a second time."""
    out = {}
    branches = port_branches()
    for ref in branches:
        out[ref] = "branch"

    def _shadowed(n):
        return n in branches and current_branch() != f"port/{n}"

    r = _git("ls-tree", "--name-only", "origin/main", "projects/", check=False)
    for line in r.stdout.splitlines():
        n = line.strip().rstrip("/").split("/")[-1]
        if n and n != "README.md" and not _shadowed(n):
            out[n] = "trunk"
    if PROJECTS.exists():
        for d in PROJECTS.iterdir():
            if (d / "status.json").exists() and not _shadowed(d.name):
                out[d.name] = "local"
    return out


def project_records():
    """(name, obj, where) for every project, resolved the way the rest of MOAT reads.

    The selector walked projects/ off disk while gen_readme, check.py, upstream.py and
    fleet all went through project_record, so the two answered differently for any
    project whose branch record differs from its copy on the trunk. Everything that
    asks "what is the state of every project" comes through here now."""
    for name, _where in sorted(all_projects().items()):
        obj, where = project_record(name)
        if obj is not None:
            yield name, obj, where


def writable_here(name, where):
    """May this checkout edit this project's record in place?

    A branch-resident project is READABLE from anywhere and writable only from its own
    branch -- you cannot edit files that are not in your tree. `commit_to_branch` is
    the exception and writes one without a checkout; the selector deliberately does
    not use it, because dispatching an agent at a project whose folder is absent would
    hand it a plan and notes it cannot open.

    A `port/<name>` branch owns exactly one project and may write only that one, even
    though it CARRIES every folder the trunk had when it was cut. Presence is not
    ownership: while the folder migration is in progress the trunk still holds dozens
    of in-flight projects, so a port branch is a full copy of them, every one reads
    `local`, and the selector would hand you somebody else's project and land its state
    on this branch. That is the canary bug that reverted the first migration attempt.
    Once the trunk holds only terminal projects the distinction stops mattering, since
    none of them is actionable -- but it matters for the whole of the migration."""
    branch = current_branch() or ""
    if branch.startswith("port/"):
        return branch == f"port/{name}"
    return where in ("local", "trunk")


def project_port_state(name):
    """Best per-arch state of another project, or None if it is not adopted.
    Resolved across refs, so an in-flight project on its own branch is not mistaken
    for one nobody has adopted."""
    obj, _where = project_record(name)
    if obj is None:
        return None
    try:
        states = {b.get("state") for b in validations(obj).values()}
    except (AttributeError, KeyError):
        return None
    for st in PORT_DONE_STATES:
        if st in states:
            return st
    return next(iter(states), None)


# What a dependency's disposition means for whatever depends on it. A disposition is
# not one answer: some of them say the dependency needs no port, and some say it will
# never have one. Treating them alike would release a dependent to fail at build time,
# which costs a whole porter attempt to rediscover what the disposition already
# recorded.
DEP_SATISFIED_BY_DISPOSITION = ("already-supported", "ported-elsewhere")
DEP_DOOMED_BY_DISPOSITION = ("cant-port", "license-blocked", "not-a-target",
                             "duplicate", "declined", "opted-out", "other")


def disposition_for_project(name):
    """A recorded decision about a project named by its MOAT short name.

    `depends_on` holds short names; dispositions are keyed by `owner/repo`. Normally
    the project's own status.json maps between them -- but a decided project has no
    folder on the trunk, which is exactly the case this has to answer, so fall back
    to matching the repo basename. An ambiguous match returns None rather than
    guessing: two owners can publish the same repo name, and silently picking one
    would apply somebody else's decision to this dependency."""
    full = upstream_full_name(name)
    if full:
        return get_disposition(full)
    hits = [v for k, v in load_dispositions().items()
            if k.rsplit("/", 1)[-1] == name.lower()]
    return hits[0] if len(hits) == 1 else None


def dep_status(dep):
    """Where a dependency stands, for a project that needs it built.

    Returns (verdict, detail). Verdicts:
      ok        -- usable now: a port validated, or upstream needs no port
      waiting   -- adopted and in the pipeline; will clear on its own
      doomed    -- dispositioned in a way that means it will never be portable, so
                   whatever depends on it cannot be built either
      unknown   -- not adopted and not dispositioned; nobody has looked at it yet,
                   so it needs an intake request before anything can proceed
    """
    state = project_port_state(dep)
    if state in PORT_DONE_STATES:
        return ("ok", state)
    disp = disposition_for_project(dep) or {}
    reason = disp.get("reason") if disp.get("disposition") == "skip" else None
    if reason in DEP_SATISFIED_BY_DISPOSITION:
        return ("ok", reason)
    if reason in DEP_DOOMED_BY_DISPOSITION:
        return ("doomed", reason)
    if state is not None:
        return ("waiting", state)
    # Adopted, but with no per-arch record yet -- a project re-opened for a second
    # screen has `platforms: {}` until a host touches it. That is squarely "in the
    # pipeline", and calling it unknown sent the caller to file an intake request that
    # port_request.py then correctly refused as already adopted. The two disagreed
    # about the same project (spconv), and only one of them can be right.
    if project_record(dep)[0] is not None:
        return ("waiting", "adopted, no arch has started")
    return ("unknown", "not adopted")


def unmet_deps(obj):
    """MOAT-internal projects this one depends_on that cannot be built against yet.
    A project is not portable until these clear: its build links/uses the ported
    dependency. See DEPENDENCIES.md."""
    return [d for d in obj.get("depends_on", []) if dep_status(d)[0] != "ok"]


def dep_report(obj):
    """Every unmet dependency with its verdict, for explaining a block to a person.
    `unmet_deps` answers "may this be selected"; this answers "why not"."""
    return [(d, *dep_status(d)) for d in obj.get("depends_on", [])
            if dep_status(d)[0] != "ok"]


INSTALL_DEP_HEADING = re.compile(r"^## Install as a dependency\s*$", re.M)


def dep_doc_gaps():
    """Dependency providers whose notes.md lacks '## Install as a dependency'.

    DEPENDENCIES.md makes the section a MUST for any MOAT project another target's
    depends_on names: it is the recipe the dependent's porter follows into _deps/.
    Nothing verified it, so providers shipped without one and a porter following the
    documented workflow found nothing there. A dependency satisfied by disposition
    (already-supported upstream) has no MOAT record and needs no section, so only
    providers with a record are judged. Returns (provider, [dependents]) rows."""
    dependents = {}
    for name, obj, _where in project_records():
        for dep in obj.get("depends_on") or []:
            dependents.setdefault(dep, []).append(name)
    rows = []
    for dep, users in sorted(dependents.items()):
        obj, _where = project_record(dep)
        if obj is None:
            continue
        p = PROJECTS / dep / "notes.md"
        notes = p.read_text() if p.exists() else None
        if notes is None:
            for ref in (f"origin/port/{dep}", "origin/main"):
                notes = _ref_read(ref, f"projects/{dep}/notes.md")
                if notes:
                    break
        if not notes or not INSTALL_DEP_HEADING.search(notes):
            rows.append((dep, sorted(users)))
    return rows


# States that describe the PROJECT and not an architecture. There is no such thing as
# "screened on gfx90a": a screen, a plan, and waiting on a fork are each one fact about
# one fork. They live in the per-arch map only because that is the only map there is,
# which is the last of the lead/follower model -- see `awaiting-port`, whose entire
# meaning is "a port exists and this arch has not been let in yet".
#
# That project-level field is `stage` (see STAGE_STATES), which is written but not yet
# read. Until the readers move to it, an arch with no record of its own reads the
# project's stage rather than defaulting to `unclaimed`. Without this, a project
# screened and parked on one arch was offered for a SECOND intake screen on every other
# arch -- cuda_voxelizer, h2o4gpu and tsne-cuda all did exactly that, and TornadoVM
# escaped only because someone hand-seeded five arch records, which is the wrong fix:
# copying a project-level fact N times is what you do when platforms are a fixed list.
#
# Ordered, unlike STAGE_STATES: this is read with max() to collapse N disagreeing arch
# records into the one answer, and furthest-along is the tie-break. That is the
# opposite of how the merge driver reconciles `stage` -- deliberately, because these
# are different questions. Here the copies are STALE, and the arch that got furthest
# is the one that was looked at last; there, two hosts wrote the same single field and
# the later write is the current one. `validation-failed` is absent because it is not
# lifted: it is evidence one arch produced, and stays in that arch's block.
STAGE_ORDER = ("unclaimed", "screened", "planned", "awaiting-fork", "awaiting-upstream",
               "porting", "ported", "delta-ported", "changes-requested", "review-passed")
assert set(STAGE_ORDER) <= STAGE_STATES, sorted(set(STAGE_ORDER) - STAGE_STATES)


def project_stage(obj):
    """The project's stage: the stored field, or the furthest along of the per-arch
    copies for a record the migration has not reached. Returns None if neither says.

    Furthest-along matters for the legacy path: one arch left at `unclaimed` must not
    drag a screened project backwards."""
    if obj.get("stage"):
        return obj["stage"]
    seen = [b.get("state") for b in validations(obj).values()
            if b.get("state") in STAGE_ORDER]
    return max(seen, key=STAGE_ORDER.index) if seen else None


def gate_satisfied(obj, gate):
    """Is this gate met -- by a validation of the CURRENT head on any arch carrying
    the attribute, or by a waiver a maintainer approved? The one definition; pr_ready
    asks the same question and must get the same answer."""
    # No recorded head means there is no current content for a validation to
    # prove, so a completed arch satisfies nothing -- only a waiver can.
    head = obj.get("head_sha")
    if head and any(gate in gates_for(a) and b.get("state") == "completed"
                    and same_commit(b.get("validated_sha"), head)
                    for a, b in validations(obj).items()):
        return True
    w = (obj.get("waivers") or {}).get(gate)
    return bool(w and w.get("approved_by") and gate in WAIVABLE_GATES)


def unsatisfied_gates(obj):
    return {g for g in REQUIRED_GATES if not gate_satisfied(obj, g)}


def settled(obj):
    """Nothing will be done with this project again, so nothing is owed.

    A `verify` disposition is NOT this: it flags a project for a closer look, which is
    the opposite of settled, and only a `skip` retires one. Reading any disposition as
    terminal put two projects on the wrong side of that."""
    disp = disposition_for_project(obj.get("name") or "")
    return bool((disp and disp.get("disposition") == "skip")
                or obj.get("stage") == "not-portable" or obj.get("on_hold"))


def outstanding(obj):
    """Work this project still owes, as a list of (arch, state). Empty means done.

    "Done" is not "an upstream PR exists". A PR opens once every gate is satisfied at
    the head of the day, and then the fork moves: a follow-up commit advances head_sha,
    the architectures that revalidate catch up, and any that do not are left holding
    evidence for code that is no longer there. Thirty projects on the trunk are in
    exactly that position, all but one of them missing wave64, because gfx90a validated
    before a later commit and nothing said so -- `revalidate` was a stored word that
    only a sweep wrote, so a stale validation read as `completed` to every reader.

    A merged PR is not done either, and that is the direction that would hurt: leaving a
    shipped port alone on the assumption it is finished, when a gate it claims is
    actually unproven at the code that shipped."""
    if settled(obj):
        return []
    out = []
    for arch in sorted(validations(obj)) or []:
        t = arch_task(obj, arch)
        if t:
            out.append((arch, t[1]))
    # A project nothing has recorded still owes whatever its stage asks for.
    if not validations(obj):
        t = arch_task(obj, "linux-gfx90a")
        if t:
            out.append(("(any)", t[1]))
    return out


def belongs_on_branch(obj):
    """Should this project's folder live on `port/<name>` rather than on the trunk?

    The trunk holds what is finished; work in flight lives where the work is. This is a
    FUNCTION of current state and not a one-way door, which is the whole point: a
    maintainer asking for a rewrite after the upstream PR merged makes a finished
    project unfinished again, and its folder has to go back. Same for a fork commit
    that stales an architecture's evidence.

    Finished takes BOTH halves. A port with every gate proven and no upstream PR is not
    done -- nobody has offered it to anyone, and thirty of those were sitting in the
    review backlog when this was written. A port with a PR but a stale architecture is
    not done either. Only a verdict ends it outright, because there is nothing left to
    prove or to offer."""
    if settled(obj):
        return False
    if not obj.get("pr_state"):
        return True
    return bool(outstanding(obj))


def misplaced_folders():
    """Projects whose folder is not where their state says it should be.

    Both directions. A folder on the trunk with work outstanding is the one that
    matters under branch protection -- every status write it attracts becomes a pull
    request against a protected trunk. A branch with nothing outstanding is the other
    half: its pull request should merge and the branch should go."""
    out = []
    for name, obj, where in project_records():
        want_branch = belongs_on_branch(obj)
        # Whether a branch EXISTS, not where this checkout happens to resolve the
        # record from. Standing on `port/<name>`, that project's folder is in the
        # working tree and reads `local` -- correctly -- so asking `where` reports
        # every branch as misplaced from its own branch, which is every orient run a
        # porter makes.
        on_branch = bool(_git("rev-parse", "--verify", "-q",
                              f"origin/port/{name}", check=False).stdout.strip())
        if want_branch and not on_branch:
            out.append((name, "trunk", "should be on port/%s" % name, outstanding(obj)))
        elif on_branch and not want_branch:
            out.append((name, "branch", "nothing outstanding; merge port/%s to main" % name, []))
    return sorted(out)


def stalled(obj):
    """Every architecture that has a record here has given up, before review.

    Nobody is working the project and the last host that tried stopped, so the next
    move is a person's: continue on other hardware, or record `not-portable`. Not
    auto-dispatched either way, because the reasons that reach this state are usually
    facts about the SOURCE -- the compute core is CUTLASS/CuTe with no ROCm path, it
    wants NVSHMEM -- which were recorded as a `blocked` flag on whichever arch looked.
    Sending three more architectures at the same wall is the failure this prevents.

    `awaiting-port` used to prevent it by accident: an arch with no record read as
    "waiting for a port" and so was never picked. That was never what it meant, and it
    only worked while every such project happened to carry those records. This says
    the same thing on purpose, and reports it (`moatlib.py stalled`) instead of leaving
    a project silently unpickable."""
    stage = project_stage(obj) or "unclaimed"
    # Answered, or deliberately parked. A recorded verdict is the person's move having
    # been made, so the project stops asking for one. `settled` covers the verdicts
    # recorded as dispositions -- an opt-out retirement leaves the arch records in
    # place, and without this check the project would reappear here asking a person
    # for a decision that was already given.
    if stage == "review-passed" or stage in INERT_STAGES or settled(obj):
        return False
    blocks = list(validations(obj).values())
    return bool(blocks) and all(b.get("blocked") for b in blocks)


def failure_stands(obj, blk):
    """Does this arch's recorded failure still describe the code on the branch?

    A `validation-failed` block is evidence about ONE commit, exactly as a `completed`
    block is, and it stops describing the port the moment a fix advances head_sha.

    Nothing used to say so, and the cycle never closed: the arch went to the porter,
    the porter's fix moved head, the reviewer passed it, and the arch went to the
    porter again -- forever, because the only thing that clears the stored word is a
    validator recording `completed`, and the selector never sent one. It stayed latent
    only because every arch that had failed was also `blocked`, which arch_task bails
    on first.

    A record with no `failed_sha` predates this and cannot be judged, so it stands:
    inventing staleness for it would claim a fix that may never have happened."""
    if blk.get("state") != "validation-failed":
        return False
    failed = blk.get("failed_sha")
    return not failed or same_commit(failed, obj.get("head_sha"))


def arch_task(obj, platform):
    """What this architecture should do now, as (agent, state), or None.

    The whole state machine in one place. Up to review the answer is the project's
    stage and one agent does that work once; from `review-passed` on, every arch
    validates independently and the answer is derived from its own evidence:

      validated this exact head -> nothing to do
      validated an older head   -> revalidate
      never validated           -> port-ready

    Those two used to be STORED, flipped in by a sweep that ran on every orient
    (`unblock_all_followers`) and by `advance_head`. A stored conclusion is one that
    can disagree with the facts it was drawn from, and keeping it current is what the
    sweep was for. Computed here, it cannot be stale and there is no sweep."""
    stage = project_stage(obj) or "unclaimed"
    if stage in INERT_STAGES:
        return None
    blk = validations(obj).get(platform) or {}
    if blk.get("blocked") or stalled(obj):
        return None
    if stage != "review-passed":
        agent = STAGE_FOR_STATE.get(stage)
        return (agent, stage) if agent else None
    # This arch tried and failed at the code that is still on the branch. Only IT is
    # sent to the porter: a wave32 fault does not invalidate a wave64 arch's evidence,
    # and what actually keeps a broken port from being submitted is pr_ready, which
    # needs a `completed` arch at head_sha for every required gate -- a failed arch
    # leaves its gate unsatisfied on its own. The porter's fix advances head_sha, which
    # makes every other arch stale and route to revalidate, so the rest of the fleet
    # catches up without the stage broadcasting.
    #
    # That same advance is what releases THIS arch: the failure it recorded is about a
    # commit no longer at the head, so it falls through to the un-validated case below
    # and a validator is sent to judge the fix. Nothing rewrites the block to make that
    # happen -- the record keeps saying what it saw, and staleness follows from the two
    # shas, which is the same reason `revalidate` is not stored either.
    if failure_stands(obj, blk):
        return ("porter", "validation-failed")
    if blk.get("state") == "completed":
        if same_commit(blk.get("validated_sha"), obj.get("head_sha")):
            return None                  # this arch has proved this code
        return ("validator", "revalidate")  # it proved an older one; refresh it
    # Nothing this arch has proved covers the current head: it never validated, or its
    # failure has been superseded by a fix. `port-ready` either way, which is the full
    # run -- an arch coming back from a failure has no standing claim to carry forward
    # from, even if an older `validated_sha` is still in its block.
    #
    # Offered only where a REQUIRED GATE still needs it. Coverage is gates, and an arch
    # beyond the one satisfying a gate is additive evidence that gates nothing --
    # welcome when someone asks for it, and not work the selector should invent.
    # Without this, every arch that has never touched any finished port becomes a
    # validation task: 315 of them here, ranked ahead of screening anything new.
    if gates_for(platform) & unsatisfied_gates(obj):
        return ("validator", "port-ready")
    return None


def platform_state(obj, platform):
    """The state this platform is IN, whether or not anything is to be done about it.

    `arch_task` answers "what now"; this answers "where is it", which the board needs
    for an arch that is finished or blocked. None means this architecture has recorded
    nothing and nothing is being asked of it -- which used to be spelled
    `awaiting-port`, a word that claimed a port was pending when often none was."""
    blk = validations(obj).get(platform) or {}
    if blk.get("state") == "completed":
        return ("completed" if same_commit(blk.get("validated_sha"), obj.get("head_sha"))
                else "revalidate")
    st = blk.get("state")
    # A failure a later commit has superseded is history, not where this arch is now.
    # Reported as whatever it is owed instead, so the board and the selector cannot
    # disagree -- one saying the arch is mid-fix while the other asks it to validate.
    if st == "validation-failed" and not failure_stands(obj, blk):
        st = None
    if st:
        return st
    task = arch_task(obj, platform)
    return task[1] if task else None


def actionable(obj, platform):
    """Is this platform pickable by an agent on this host right now?"""
    if obj.get("on_hold"):  # project-wide postponement of a whole dependency stack
        return False
    # A decided project is not work, whatever its per-arch records say. Without this
    # a dispositioned project whose folder is still on the trunk gets offered for
    # intake again -- re-screening something already declined.
    disp = disposition_for_project(obj.get("name") or "")
    if disp and disp.get("disposition") == "skip":
        return False
    # An ABSENT record means "no host has touched this platform yet", which is what
    # `scaffold` documents ("one appears when a host first works the project"). Bailing
    # on that deadlocked every newly adopted project: the record is created by working
    # the project, and working it required the record. opencv, rmagine and the two
    # diff-surfel repos sat forked and unoffered from June because of it.
    task = arch_task(obj, platform)
    if task is None:
        return False
    # Only one arch at a time may do work that writes SHARED content -- the fork's port
    # branch for a porter, plan.md for a planner, the verdict for a reviewer. Validation
    # is exempt because it is read-only on code and writes only its own arch's record,
    # which merges -- and because it is partitioned by platform, so two hosts picking it
    # up want different records. Review has no such partition: every host reviewing a
    # project reviews the same branch and writes the same verdict.
    lock = obj.get("porting")
    if lock and lock.get("arch") != platform and task[0] in EXCLUSIVE_AGENTS:
        return False
    if unmet_deps(obj):  # deps-first ordering: wait until depended-on ports complete
        return False
    return True


def dep_blocked(platform):
    """Projects this platform would otherwise work, held back only by a dependency.

    `next_task` returning NONE looks identical whether there is genuinely nothing to
    do or a project is waiting on a dependency nobody has adopted. That silence is
    the failure mode: deps-first ordering becomes deps-never and nothing says so."""
    out = []
    for name, obj, where in project_records():
        if obj.get("on_hold") or not writable_here(name, where):
            continue
        # Would it be pickable if the dependency cleared? Same test as actionable(),
        # minus the dependency check itself.
        disp = disposition_for_project(name)
        if disp and disp.get("disposition") == "skip":
            continue
        if arch_task(obj, platform) is None:
            continue
        report = dep_report(obj)
        if report:
            out.append((name, report))
    return out


def fleet(platform):
    """Actionable work across every ref, not just this checkout.

    After a project's folder moves to its own branch, `next_task` cannot see it --
    correctly, since you cannot work a project whose files are not in your tree. But
    then nothing answers "what is out there", and work becomes invisible rather than
    merely elsewhere. This scans the refs and says which branch to check out."""
    out = []
    for name, where in sorted(all_projects().items()):
        obj, _ = project_record(name)
        if obj is None or obj.get("on_hold"):
            continue
        disp = disposition_for_project(name)
        if disp and disp.get("disposition") == "skip":
            continue
        task = arch_task(obj, platform)
        if task is None or unmet_deps(obj):
            continue
        agent, state = task
        # The same exclusion `actionable` applies: porter/planner/reviewer work whose
        # lock another arch holds is not dispatchable anywhere, and naming its branch
        # here sends a checkout at a guaranteed `next: NONE`.
        lock = obj.get("porting")
        if lock and lock.get("arch") != platform and agent in EXCLUSIVE_AGENTS:
            continue
        # The branch as the remote actually spells it, so a caller can print a
        # checkout command that works even when the branch case-folds the name
        # (port/hami-core for HAMi-core). Reconstructing port/<project> does not.
        branch = (port_branch_of(name) or "") if where == "branch" else ""
        out.append({"project": name, "where": where, "state": state,
                    "stage": agent, "branch": branch,
                    "priority": float(obj.get("priority", 0))})
    out.sort(key=lambda r: (SELECT_RANK.get(r["state"], 99), -r["priority"], r["project"]))
    return out


def next_task(platform):
    """Pick the single next project for this platform. Returns dict or None.

    Resolved, and restricted to what this checkout can actually edit. Reading the
    working tree directly offered a project its own branch had already moved past --
    colmap was screened and decided on `port/colmap` while the trunk stub still said
    unclaimed, so the trunk offered to screen it a second time and `fleet` said
    planner. Anything a branch owns is reported by `fleet` instead, with the branch
    to check out."""
    cands = []
    for name, obj, where in project_records():
        if not writable_here(name, where) or not actionable(obj, platform):
            continue
        state = arch_task(obj, platform)[1]
        cands.append((SELECT_RANK.get(state, 99), -float(obj.get("priority", 0)),
                      name, state))
    if not cands:
        return None
    cands.sort()
    rank, negprio, name, state = cands[0]
    return {"project": name, "state": state, "stage": STAGE_FOR_STATE[state],
            "priority": -negprio, "rank": rank}


def release_awaiting_fork(org="AMD-Ecosystem", dry_run=False):
    """Advance projects whose fork has appeared.

    A project sits in `awaiting-fork` until someone with the rights creates the fork.
    That act IS the decision to take the project up -- nothing else records one, and
    nothing needs to: the fork either exists or it does not, which is checkable by
    anyone and cannot drift from whatever a document claims.

    Returns [(name, fork_url)] for the projects released.

    Resolved across refs and written across them too. Every project waiting on a fork
    now lives on its own branch, so walking the working tree reported "nothing waiting
    on a fork" while four waited -- a clean bill of health that was false, and the one
    report anyone would trust to tell them a fork had appeared. `save_record` writes the
    release without checking the branch out, which is safe here in a way it would not be
    for the selector: this advances a record, it does not hand an agent a project whose
    files are absent."""
    released = []
    for name, obj, _where in project_records():
        if project_stage(obj) != "awaiting-fork":
            continue
        fork = obj.get("fork_url") or f"https://github.com/{org}/{name}"
        slug = fork.replace("https://github.com/", "")
        r = subprocess.run(["gh", "api", f"repos/{slug}", "--jq", ".full_name"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode:
            continue                       # still no fork; leave it waiting
        if dry_run:
            released.append((name, slug))
            continue
        # One project, one release. This used to flip N arch records to `screened`,
        # which is the same project-level fact written once per arch.
        obj["stage"] = "screened"
        obj["fork_url"] = f"https://github.com/{slug}"
        obj["updated_at"] = now_iso()
        save_record(name, obj,
                    f"{name}: fork exists, releasing for planning\n\n"
                    f"{slug} was created, which is the decision to take this "
                    f"project up.")
        released.append((name, slug))
    return released


# ---- dispositions (candidates we will NOT port, and why) -------------------

DISPOSITIONS = REPO_ROOT / "data" / "dispositions.json"
SKIP_REASONS = ["already-supported", "ported-elsewhere",
                "cant-port", "not-a-target", "duplicate",
                # licence dispositions, decided at intake (see config/licenses.toml)
                "license-blocked",   # licence bars the work outright
                # A deliberate no with no reason given. Distinct from "other", which
                # reads as a catch-all: this says the decision was made and the
                # reasoning was intentionally not recorded. The repo is public, so a
                # written reason is permanent and quotable, and a project can be
                # reconsidered later without anything to walk back.
                "declined",
                # The maintainer asked not to receive our pull requests. Their decision,
                # not ours, and the only skip reason an agent may record on its own --
                # see data/optout.json, which is the owner-scoped record this retires a
                # single adopted project against.
                "opted-out",
                "other"]
# already-supported: this upstream repo already supports ROCm/HIP, by any means
#   (CUDA path ported to HIP, or a native/designed-in backend); provenance is
#   irrelevant, what matters is that it runs on AMD.
# ported-elsewhere: AMD's ROCm/HIP support for this project (or an equivalent)
#   lives in a SEPARATE repo, fork, or effort; only use it with a found reference.


def github_repo_id(full_name):
    """GitHub's numeric repo id, which survives renames and owner transfers.
    None if the repo is unreachable -- caller must not read that as "no such repo"."""
    r = subprocess.run(["gh", "api", f"repos/{full_name}", "--jq", ".id"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        return None
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def load_dispositions():
    if DISPOSITIONS.exists():
        try:
            return json.loads(DISPOSITIONS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_dispositions(d):
    DISPOSITIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(DISPOSITIONS, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)
        f.write("\n")


def get_disposition(full_name, repo_id=None):
    """A recorded decision about this repo, by name or by GitHub repo id.

    The id is what makes this survive a rename. `owner/repo` is not stable: lucebox
    was skipped in May as luce-org/lucebox-hub, was renamed, and came back through
    discovery under the new name because the key matched nothing. GitHub resolves an
    old name to the new one but not the reverse, so the only thing that closes it in
    both directions is the numeric id."""
    d = load_dispositions()
    hit = d.get(full_name.lower())
    if hit:
        return hit
    if repo_id is not None:
        for v in d.values():
            if v.get("repo_id") == repo_id:
                return v
    return None


def set_disposition(full_name, disposition, reason, note="", repo_id=None, by=None):
    if disposition == "skip" and reason not in SKIP_REASONS:
        raise ValueError(f"reason must be one of {SKIP_REASONS}")
    # Declining is a person's decision, exactly as creating the fork is. An agent
    # may carry that decision into the record, but a skip without a named decider
    # satisfies nothing -- the same rule set-not-portable and license clearances
    # already enforce.
    if disposition == "skip" and not by:
        raise ValueError(
            f"{full_name}: a skip needs --by <who decided>; an agent may write "
            f"the case but never the verdict")
    d = load_dispositions()
    if repo_id is None:
        repo_id = github_repo_id(full_name)      # best effort; None when offline
    d[full_name.lower()] = {"full_name": full_name, "disposition": disposition,
                            "reason": reason, "note": note, "decided": now_iso(),
                            "by": by, "repo_id": repo_id}
    save_dispositions(d)
    return d[full_name.lower()]


def clear_disposition(full_name):
    d = load_dispositions()
    if full_name.lower() in d:
        del d[full_name.lower()]
        save_dispositions(d)
        return True
    return False


# ---- opt-out (maintainers who asked not to receive our pull requests) -------

# Deliberately NOT a disposition, though both stop work on a repo. A disposition is
# our judgement about a project -- not a target, already supported, cannot be ported.
# An opt-out is somebody else's decision about us, and the two must not be filed under
# one word: a reader of dispositions.json is reading what we concluded, and an entry
# saying "declined" where a maintainer actually asked us to stop misreports whose
# decision it was. It is also scoped differently. A disposition is one repo, keyed by
# its numeric id; an opt-out is usually a whole owner and has to bind repos nobody has
# discovered yet, which an id cannot do.
OPTOUTS = REPO_ROOT / "data" / "optout.json"


def load_optouts():
    if OPTOUTS.exists():
        try:
            return json.loads(OPTOUTS.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_optouts(d):
    OPTOUTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTOUTS, "w") as f:
        json.dump(d, f, indent=2, sort_keys=True)
        f.write("\n")


def optout_for(full_name):
    """The opt-out covering `owner/repo`, or None.

    An owner-scoped entry covers every repo that owner has, including ones discovery
    has not reached, which is what "stop sending me these" actually means."""
    if not full_name:
        return None
    key = full_name.lower()
    d = load_optouts()
    return d.get(key) or d.get(key.split("/")[0])


def record_optout(who, source, note=""):
    """Record that `who` (an owner, or one owner/repo) asked not to receive our PRs.

    An agent MAY write this one, unlike a disposition or a licence clearance. Those
    are our judgement and need a person; this is carrying somebody else's decision
    into the record, and the only thing it can do is less work. `source` is required
    so the record says where the request was made and anyone can go read it."""
    if not source:
        raise ValueError("an opt-out needs a source: where the request was made")
    if who.count("/") > 1 or who.startswith("/") or who.endswith("/"):
        raise ValueError(f"expected an owner or owner/repo, got {who!r}")
    d = load_optouts()
    d[who.lower()] = {"who": who, "scope": "repo" if "/" in who else "owner",
                      "requested_at": now_iso(), "source": source, "note": note}
    save_optouts(d)
    return d[who.lower()]


def clear_optout(who, by):
    """Withdraw an opt-out. Requires `by`, and requires it HERE rather than only in
    optout.py, because this is the one direction that resumes contact with someone who
    asked us to stop. Every comparable control in this file refuses in the library --
    set_not_portable, approve_waiver, refuse_waiver -- so that a caller reaching past
    the command line cannot skip what the command line was enforcing. Recording an
    opt-out is the opposite and needs nobody: it can only ever cause less work."""
    if not (by or "").strip():
        raise ValueError(
            f"{who}: --by is required. Withdrawing an opt-out resumes contact with "
            f"someone who asked us to stop, so the record has to say who authorised it "
            f"-- an agent may carry that decision but never make it")
    d = load_optouts()
    if who.lower() in d:
        del d[who.lower()]
        save_optouts(d)
        return True
    return False


# ---- scaffolding -----------------------------------------------------------

def existing_claim(name):
    """Why this project is already taken, or None.

    The local working tree is not the whole picture: `port/<name>` on the remote IS
    the claim, and it is what another host has when it starts work. Checking only the
    local tree means two hosts handed the same candidate both do the full screen
    before either discovers the other.

    A remote we cannot reach returns a warning rather than None. An outage must not
    read as "nobody has this" -- that is the failure this exists to prevent."""
    if not _git("remote", check=False).stdout.strip():
        return None                       # local-only clone; nothing to contend with
    r = _git("ls-remote", "--heads", "origin", f"refs/heads/port/{name}", check=False)
    if r.returncode:
        return (f"UNVERIFIED -- could not reach the remote to check for port/{name}; "
                f"this is not a clean bill of health")
    if r.stdout.strip():
        return f"port/{name} exists on the remote"
    if _git("cat-file", "-e", f"origin/main:projects/{name}/status.json",
            check=False).returncode == 0:
        return f"projects/{name}/ already exists on the trunk"
    return None


def scaffold_project(full_name, upstream_url=None, default_branch="main",
                     ext_type="unknown", priority=0.0, force=False, depends_on=None):
    # No `force` on this one. Everything else here is our own decision and a person
    # may overrule it; this is the maintainer's, and there is nobody on our side who
    # can overrule it.
    opt = optout_for(full_name)
    if opt:
        raise ValueError(
            f"{opt['who']} asked not to receive pull requests from this effort "
            f"({opt['source']}). Recorded in data/optout.json; not adoptable.")
    disp = get_disposition(full_name, github_repo_id(full_name))
    if disp and disp.get("disposition") == "skip" and not force:
        raise ValueError(
            f"{full_name} is marked skip ({disp.get('reason')}): {disp.get('note', '')}. "
            f"Use force=True / --force to adopt anyway.")
    name = full_name.split("/")[-1]
    if not force:
        held = existing_claim(name)
        if held:
            raise ValueError(f"{name} is already claimed: {held}. "
                             f"Use force=True / --force to adopt anyway.")
    pdir = PROJECTS / name
    pdir.mkdir(parents=True, exist_ok=True)
    if status_path(name).exists():
        raise FileExistsError(f"{status_path(name)} already exists")
    upstream_url = upstream_url or f"https://github.com/{full_name}"
    obj = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "upstream_url": upstream_url,
        "fork_url": None,
        "fork_default_branch": default_branch,
        "priority": float(priority),
        "ext_type": ext_type,
        "stage": "unclaimed",
        "adopted_at": now_iso(),
        "updated_at": now_iso(),
        "head_sha": None,
        "depends_on": list(depends_on or []),
        "porting": None,
        "waivers": {},
        # No records up front: one appears when a host first works the project.
        # An absent record already means "never validated here".
        "platforms": {},
    }
    save_status(name, obj)
    (pdir / "notes.md").write_text(f"# {name} notes\n")
    return name


# ---- git sync --------------------------------------------------------------

def _git(*args, check=True, cwd=None):
    return subprocess.run(["git", *args], cwd=str(cwd or REPO_ROOT),
                          capture_output=True, text=True, check=check)


def ensure_git_config():
    """Register the semantic status.json merge driver (idempotent). A fresh
    clone needs this for .gitattributes merge=moat-status to take effect."""
    drv = f"python3 {REPO_ROOT / 'utils' / 'merge_status.py'} %O %A %B %P"
    _git("config", "merge.moat-status.name", "MOAT status.json semantic merge", check=False)
    _git("config", "merge.moat-status.driver", drv, check=False)


# Paths on the trunk whose change cannot alter how a port is done: the generated
# README, the discovery/disposition registries, and OTHER projects' records. A port
# branch may sit behind the trunk on all of these without any risk to the work.
#
# Deliberately a DENYLIST. Anything not listed here counts as substantive, so a path
# added to the repo later -- a new config file, a new skill directory -- defaults to
# "merge" rather than being silently skipped by a rule nobody remembered to update.
PORT_INERT = ("README.md", "data/")


def branch_drift(branch, base_ref="origin/main", cwd=None):
    """What has landed on the trunk that this port branch has not seen, split into
    the changes a port can feel and the ones it cannot.

    Returns (substantive, inert) as sorted path lists. Both empty means the branch
    already carries everything on the trunk."""
    base = _git("merge-base", "HEAD", base_ref, cwd=cwd, check=False).stdout.strip()
    if not base:
        return ([], [])
    out = _git("diff", "--name-only", base, base_ref, cwd=cwd, check=False).stdout
    substantive, inert = [], []
    for p in out.splitlines():
        p = p.strip()
        if not p:
            continue
        # The branch OWNS projects/<its-name>/ -- every host working this project
        # pushes to this branch, not to the trunk -- so a trunk change there is never
        # a reason to merge. It used to be classified substantive on the theory that
        # someone might push state via the trunk; the bam canary disproved that. Once
        # the project migrates, the trunk's version of that path is a DELETION, and
        # calling it substantive merged the deletion into the branch that owns it.
        if (p in PORT_INERT or p.startswith(PORT_INERT)
                or p.startswith("projects/")):
            inert.append(p)
        else:
            substantive.append(p)
    return (sorted(substantive), sorted(inert))


def branch_lessons(base_ref="origin/main"):
    """Global edits sitting on port branches: (name, paths, orphaned).

    A global edit is anything outside the branch's own `projects/<name>/` -- the
    `cuda-to-rocm` skill, an agent definition, a tool in utils/.

    On a LIVE port branch that is the CORRECT place for one. A lesson learned while
    porting is project-scoped until a person approves it, so it rides the branch and
    the port's own review is what publishes it. Lifting one to the trunk early is not
    a rescue, it is publishing an unreviewed claim to every agent: of four lessons
    written in one session, three were wrong in ways only review caught -- one
    reproduced verbatim the CMake defect it documented, one named a prebuilt rocPRIM
    library that does not exist, one stated the inverse of the rule it described.

    `orphaned` is the defect, and it is the only thing worth acting on. The port is
    FINISHED -- nothing outstanding, so its folder belongs back on the trunk and its
    branch is about to be deleted -- while a global edit on it is still absent from the
    trunk. No review is coming to carry it, so deleting the branch loses it.

    Superseded wording reads the same as new wording here, because both are lines the
    branch has and the trunk does not. Read the diff before concluding anything."""
    out = []
    for ref in _git("for-each-ref", "--format=%(refname:short)",
                    "refs/remotes/origin/port/", check=False).stdout.split():
        name = ref.split("port/", 1)[1]
        base = _git("merge-base", ref, base_ref, check=False).stdout.strip()
        if not base:
            continue
        changed = _git("diff", "--name-only", base, ref, check=False).stdout.splitlines()
        cands = [c.strip() for c in changed
                 if c.strip() and not c.startswith(f"projects/{name}/")
                 and c.strip() not in ("README.md",)]
        shared = []
        for c in cands:
            d = _git("diff", base_ref, ref, "--", c, check=False).stdout.splitlines()
            if any(l.startswith("+") and not l.startswith("+++") for l in d):
                shared.append(c)
        if not shared:
            continue
        obj, _where = project_record(name)
        orphaned = obj is not None and not belongs_on_branch(obj)
        out.append((name, sorted(shared), orphaned))
    return sorted(out)


def make_worktree(name, path=None, base_ref="origin/main"):
    """Create a worktree on `port/<name>` and bring it up to the trunk. Returns the path.

    One command because the sync is the part that gets skipped. A worktree cut from a
    port branch runs whatever tooling that branch last merged, which can be days old,
    and the agent then writes records with old code against today's schema. Both of
    today's bad records came from exactly that: one wrote a stage the trunk had just
    stopped recognising, which failed the repo-wide gates and blocked every push from
    every host; the other recorded a full GPU rerun as a carry-forward because its copy
    of `set_state` still no-opped on a same-state call.

    Advice would not have prevented either -- it was advice, and I was the one skipping
    it. So there is no unsynced way to get a worktree."""
    path = Path(path) if path else (REPO_ROOT / "agent_space" / f"wt-{name}")
    if path.exists():
        raise ValueError(f"{path} already exists -- remove it or pass another --path")
    ref = f"origin/port/{name}"
    if not _git("rev-parse", "--verify", "-q", ref, check=False).stdout.strip():
        raise ValueError(f"{ref} does not exist; this project has no port branch")
    # The `-B` below force-resets an existing local port/<name> onto the remote, which
    # is what makes a worktree reproducible and is silent when the local branch holds
    # work the remote does not. That is not hypothetical: commit_and_push gives up
    # after three failed pushes and says so, leaving the work committed locally.
    #
    # Count by PATCH-ID rather than ancestry. A remote that was rebased, squashed or
    # re-cut leaves the local ref on an old line whose shas are all absent from the
    # remote while every CHANGE on it is already there; ancestry calls that 37 commits
    # at risk when the honest answer is none, and a check that cries wolf that loudly
    # is one people learn to skip. `git cherry` marks those `-` and only genuinely new
    # work `+`, and when everything is `-` the reset discards nothing and should just
    # proceed -- which is the common case.
    local = f"port/{name}"
    if _git("rev-parse", "--verify", "-q", local, check=False).stdout.strip():
        only_here = [f"{sha[:9]} {subject[:60]}" for sha, _, subject in
                     (ln[2:].strip().partition(" ") for ln in
                      _git("cherry", "-v", ref, local, check=False).stdout.splitlines()
                      if ln.startswith("+"))]
        if only_here:
            raise ValueError(
                f"{name}: {len(only_here)} commit(s) exist only on the local {local}, and "
                f"creating the worktree would reset it onto {ref} and discard them: "
                f"{'; '.join(only_here[:3])}. Push them first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    r = _git("worktree", "add", "-q", "-B", local, str(path), ref, check=False)
    if r.returncode:
        raise ValueError(f"could not create the worktree: {(r.stderr or r.stdout).strip()}")
    # Sync with THIS copy of the code, pointed at the new worktree. Running the
    # worktree's own moatlib is circular: the branch that most needs a sync is the one
    # whose syncer is too old to perform it.
    action, detail = branch_sync(apply=True, base_ref=base_ref, cwd=path)
    detail = f"{action} -- {detail}"
    # A worktree that did not sync is the thing this command exists to prevent, so it
    # is not handed back. Returning the path with a warning attached would be worse
    # than the hand-rolled version it replaces: a caller gets something that looks
    # ready, and the warning is one line above the path they are about to paste.
    if action not in ("merged", "current", "inert"):
        # Say which of the two happened. If the removal fails, "it was not created" is
        # false, the directory is still there, and the next run reports "already
        # exists" -- a different problem than the one that occurred.
        removed = _git("worktree", "remove", "--force", str(path), check=False)
        if removed.returncode:
            raise ValueError(
                f"{name}: the worktree at {path} did not sync ({detail}), and removing "
                f"it failed: {(removed.stderr or removed.stdout).strip()}. It is still "
                f"on disk and is NOT synced -- do not use it; remove it by hand.")
        raise ValueError(
            f"{name}: the worktree was not synced, so it was not created -- {detail}. "
            f"Resolve it on the branch first: `git checkout port/{name}` in a checkout "
            f"you own, merge origin/main by hand, push, then run this again.")
    return (str(path), detail)


def branch_sync(apply=False, base_ref="origin/main", cwd=None):
    """Bring a port branch up to the trunk's tooling, but only when that is worth a
    merge commit. Returns (action, detail) for the caller to print.

    Merging on every trunk push would put a merge commit on every port branch for a
    README regeneration. Merging on none of them means a port runs whatever skills
    and agent definitions existed the day its branch was cut. So: look first, and
    merge only when something a port can actually feel has moved.

    `cwd` runs this against ANOTHER checkout -- a fresh worktree -- while the merging
    is still done by THIS copy of the code. That distinction is the whole point: a
    worktree cut from a port branch carries that branch's tooling, and a branch old
    enough to need syncing is old enough that its own `branch_sync` cannot do it. The
    ffpa-attn branch predates the rule that a branch keeps its own project folder
    across a trunk merge, so its copy hit that conflict and gave up, reporting four
    files that had merged cleanly. Syncing a branch with the branch's own syncer is
    circular exactly when it matters."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", check=False, cwd=cwd).stdout.strip()
    if not branch.startswith("port/"):
        return ("skip", "not a port branch")
    if _git("status", "--porcelain", check=False, cwd=cwd).stdout.strip():
        return ("dirty", "uncommitted changes -- not merging; commit or stash first")
    _git("fetch", "-q", "origin", base_ref.split("/", 1)[-1], check=False, cwd=cwd)
    substantive, inert = branch_drift(branch, base_ref, cwd=cwd)
    if not substantive:
        if not inert:
            return ("current", "up to date with the trunk")
        return ("inert", f"trunk moved, nothing a port can see ({len(inert)} path(s))")
    if not apply:
        return ("would-merge", ", ".join(substantive[:4]))
    ensure_git_config()
    project = branch[len("port/"):]
    pre = _git("rev-parse", "HEAD", check=False, cwd=cwd).stdout.strip()
    own = f"projects/{project}/"
    r = _git("merge", "--no-edit", base_ref, check=False, cwd=cwd)
    if r.returncode:
        conflicted = [c.strip() for c in
                      _git("diff", "--name-only", "--diff-filter=U",
                           check=False, cwd=cwd).stdout.splitlines() if c.strip()]
        # A conflict confined to this branch's OWN project folder has a settled answer
        # and does not need a person: the branch owns that path. It happens on every
        # sync now rather than rarely -- the trunk deleted the folder when the project
        # moved here, so any branch that has edited its own state since collides with
        # that deletion. Aborting on it left five branches unable to take a trunk
        # merge at all, running tooling old enough that it could not read the very
        # records it was holding, and silently offering another project's work.
        # README.md is GENERATED, so a conflict there is not a disagreement: both
        # sides ran gen_readme against different sets of records. The trunk's board is
        # the published one and gate_readme does not judge a port branch, so the
        # trunk's version wins and the branch's own regeneration is simply dropped.
        # Left unhandled it collides on every branch that ever regenerated the board,
        # which is all of them.
        settled = [c for c in conflicted if c.startswith(own) or c == "README.md"]
        if conflicted and len(settled) == len(conflicted):
            if any(c.startswith(own) for c in conflicted):
                _git("checkout", pre, "--", own, check=False, cwd=cwd)
                _git("add", "--", own, check=False, cwd=cwd)
            if "README.md" in conflicted:
                _git("checkout", base_ref, "--", "README.md", check=False, cwd=cwd)
                _git("add", "--", "README.md", check=False, cwd=cwd)
            _git("commit", "--no-edit", "-q", check=False, cwd=cwd)
        else:
            _git("merge", "--abort", check=False, cwd=cwd)
            return ("conflict", f"merging {base_ref} conflicts outside "
                                f"{own} and README.md -- resolve by hand: "
                                f"{', '.join(conflicted[:4] or substantive[:4])}")
    # The trunk does not carry an in-flight project's folder, and a branch with no
    # commits of its own fast-forwards straight onto that absence -- which is how the
    # bam canary lost its own state to a routine sync. Whatever the merge did to this
    # branch's project, the branch's version wins.
    if _git("cat-file", "-e", f"{pre}:projects/{project}/status.json",
            check=False, cwd=cwd).returncode == 0:
        _git("checkout", pre, "--", own, check=False, cwd=cwd)
        if _git("diff", "--cached", "--name-only", check=False, cwd=cwd).stdout.strip():
            _git("commit", "-q", "-m",
                 f"{project}: keep this branch's project state across the trunk merge", cwd=cwd)
    # Push so a sibling host reuses this merge instead of making its own; the branch
    # is shared, and two independent merges of the same trunk diverge for no reason.
    _git("push", "-q", "origin", branch, check=False, cwd=cwd)
    return ("merged", ", ".join(substantive[:4]))


def commit_to_branch(branch, files, message):
    """Commit files onto a branch WITHOUT checking it out, and push. Returns the sha.

    Plumbing rather than checkout because the branch may be held by a worktree, and
    because switching the working tree to write one file is a large side effect for a
    small edit. This is also the shape the migration needs generally: an in-flight
    project's record lives on its own branch, and a decision recorded on the trunk has
    to reach it somehow."""
    import os
    import tempfile
    base = _git("rev-parse", f"origin/{branch}", check=False).stdout.strip()
    if not base:
        raise ValueError(f"origin/{branch} does not exist")
    fd, idx = tempfile.mkstemp(prefix="moat-index-")
    os.close(fd)
    os.unlink(idx)
    env = {**os.environ, "GIT_INDEX_FILE": idx}

    def g(*args, stdin=None):
        r = subprocess.run(["git", *args], cwd=str(REPO_ROOT), env=env, input=stdin,
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    try:
        g("read-tree", base)
        for path, content in files.items():
            blob = g("hash-object", "-w", "--stdin", stdin=content)
            g("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        tree = g("write-tree")
        commit = g("commit-tree", tree, "-p", base, "-m", message)
        g("push", "-q", "origin", f"{commit}:refs/heads/{branch}")
        return commit
    finally:
        if os.path.exists(idx):
            os.unlink(idx)


def commit_and_push(paths, message, push=True, retries=3):
    """The single MOAT-repo write path: stage, commit-on-top, pull --rebase,
    push, bounded retry. Never amends, never force-pushes. No-op if nothing
    staged."""
    ensure_git_config()
    rels = [str(Path(p)) for p in (paths if isinstance(paths, (list, tuple)) else [paths])]
    _git("add", "--", *rels)
    staged = _git("diff", "--cached", "--name-only", check=False).stdout.strip()
    if not staged:
        return False
    _git("commit", "-m", message)
    if not push:
        return True
    # Name the remote and branch explicitly. A bare `git push` needs an upstream, and
    # a freshly created port/<name> has none -- so the first push of a new project
    # branch failed all three retries and returned False behind one stderr line, which
    # reads as a network problem rather than a branch that was never published.
    # Read fresh rather than via current_branch(), whose answer is cached for the
    # process and could predate a checkout. "HEAD" means detached, where there is no
    # branch to name and the bare push is the only sensible attempt.
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    if branch == "HEAD":
        branch = ""
    for _ in range(retries):
        # --autostash so a concurrent agent's unstaged files in the shared
        # working tree don't abort our rebase (multi-agent MOAT runs).
        _git("pull", "--rebase", "--autostash", check=False)
        r = (_git("push", "-u", "origin", branch, check=False) if branch
             else _git("push", check=False))
        if r.returncode == 0:
            return True
    sys.stderr.write(f"commit_and_push: push of {branch or 'HEAD'} failed after "
                     f"{retries} attempts; left committed locally\n")
    return False


def squash_carry_forward(name, new_sha, repo=None):
    """Advance head to a PR-prep squash, carrying every already-validated platform
    forward WITHOUT revalidation. Valid only when new_sha is a TREE-IDENTICAL
    collapse of the current validated head -- i.e. the squash combined
    already-validated commits and changed no content -- which is the case when the
    squash is done at PR-prep AFTER every platform is terminal (pr_ready). Then the
    squashed commit is known to work everywhere it already worked:
      - each platform `completed` AT THE OLD HEAD: validated_sha advanced to
        new_sha, stays completed. A completed platform whose validated_sha lags
        the old head validated some earlier tree, is owed a revalidation, and is
        reported as `stale` rather than carried;
      - each `blocked` (non-viable, e.g. Windows-unportable) platform: left UNTOUCHED
        -- never flipped from non-viable to passing;
      - an arch left un-validated because a sibling already satisfies every gate it
        could satisfy (see pr_ready): reported as `optional`, not a problem;
      - any other (actionable) state: left as-is (you should not be squashing yet).
    REFUSES if new_sha's tree differs from the current head's tree (then the squash
    introduced unvalidated content; validate it first / use advance_head). The
    carry-forward is recorded in the shared status.json, so other hosts see the new
    sha as already-validated and do not re-run -- the force-push history rewrite is
    irrelevant to them. Returns (ok, info)."""
    obj = load_status(name)
    repo = repo or _fork_repo(name)
    new_sha = full_sha(new_sha, repo)
    old_head = obj.get("head_sha")

    def _tree(sha):
        if not sha:
            return None
        r = subprocess.run(["git", "rev-parse", f"{sha}^{{tree}}"], cwd=str(repo),
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    t_old, t_new = _tree(old_head), _tree(new_sha)
    if not t_old or not t_new or t_old != t_new:
        return (False, f"not a tree-identical squash (old tree {str(t_old)[:8]} != new {str(t_new)[:8]}); "
                       f"validate the new content first / use advance_head")
    obj["head_sha"] = new_sha
    vals = validations(obj)
    # Only an arch validated at the OLD HEAD proved the tree the squash preserved.
    # A completed arch whose validated_sha lags it was owed a revalidation before
    # the squash and still is -- promoting it would mark content it never ran as
    # proven -- so it neither carries forward nor counts toward a satisfied gate.
    current = {a for a, b in vals.items()
               if b.get("state") == "completed" and not b.get("blocked")
               and same_commit(b.get("validated_sha"), old_head)}
    # A gate already satisfied by some carried arch needs nothing more; an arch
    # that could only have satisfied such a gate is optional, not a blocker.
    satisfied = {g for a in current for g in gates_for(a)}
    carried, kept_blocked, skipped, optional, stale = [], [], [], [], []
    for plat, blk in vals.items():
        if blk.get("blocked"):
            kept_blocked.append(plat)
        elif plat in current:
            blk["validated_sha"] = new_sha
            blk["updated_at"] = now_iso()
            carried.append(plat)
        elif blk.get("state") == "completed":
            stale.append((plat, (blk.get("validated_sha") or "?")[:8]))
        elif gates_for(plat) <= satisfied:
            optional.append((plat, blk.get("state")))
        else:
            skipped.append((plat, blk.get("state")))
    save_status(name, obj)
    return (True, {"carried": carried, "kept_blocked": kept_blocked,
                   "skipped": skipped, "optional": optional, "stale": stale})


def pr_ready(name):
    """Is a port ready for its single upstream PR?

    Readiness is expressed as GATES, not as a fixed platform list. Every gate in
    config/arches.toml `required` must be satisfied, and a gate is satisfied when
    ANY arch carrying that attribute is `completed` at the current head_sha. So
    gfx90a/Linux satisfies wave64, gfx1201/Windows satisfies wave32 and windows
    together, and validating gfx942 alongside gfx90a is additive evidence that
    gates nothing.

    A gate may also be satisfied by a recorded WAIVER, but only for gates listed
    `waivable` (in practice: windows) and only with maintainer approval. Agents may
    suggest a waiver; they cannot grant one, so a waiver missing `approved_by` does
    not satisfy anything.

    A gate with no completed arch and no waiver blocks. The blockers reported are
    the archs that could still satisfy it -- completing any ONE clears the gate.
    Archs documented non-viable (`blocked`) are reported separately so the PR body
    can scope its claim, and unscheduled archs (hardware gone) are never blockers.

    Returns (ready, blocking, nonviable)."""
    # The freshest record, not the nearest: load_status prefers the working tree,
    # so a checkout carrying a stale trunk-era copy of this project's folder would
    # have the gate judge that copy while the enumeration around it read the
    # project's branch. project_record puts the project's own branch first.
    obj, _where = project_record(name)
    if obj is None:
        raise FileNotFoundError(str(status_path(name)))

    # Checked here and not only at adoption, because an opt-out usually arrives after
    # we have already sent this owner a pull request -- that is what prompts it. This
    # is the last gate before anything reaches their repository and the one both routes
    # upstream pass through (the review PR and the publish step), so a request to stop
    # binds work already finished, not just work not yet started. `optout.py record`
    # also writes a disposition, which the next block would catch; this does not rely
    # on that having happened.
    opt = optout_for(upstream_full_name(name) or "")
    if opt:
        return (False, [("opted-out",
                         f"{opt['who']} asked not to receive pull requests from this "
                         f"effort ({opt['source']})")], [])

    # A recorded disposition settles the PR question: the project was delivered as a
    # validation-only record, or set aside as already-supported / non-viable /
    # licence-blocked. Either way it is not an upstream contribution. This replaces the
    # old upstream.json `outcome` field -- both said "settled", and one file saying it
    # is enough.
    disp = get_disposition(upstream_full_name(name) or "")
    if disp and disp.get("disposition") == "skip":
        return (False, [("dispositioned",
                         f"{disp.get('reason')} (recorded in dispositions.json; "
                         f"not a PR candidate)")], [])

    pr_state = obj.get("pr_state")
    if pr_state:
        return (False, [("pr-exists", f"the upstream PR is already {pr_state}")], [])
    if obj.get("pr_url"):
        return (False, [("pr-exists", "a PR is already recorded in status.json")], [])

    blocking, nonviable = _gate_blockers(name, obj)
    return (not blocking, blocking, nonviable)


def _gate_blockers(name, obj):
    """The gate core shared by pr_ready and fix_ready: coverage gates, licence,
    and fork integrity, judged at the record's current head_sha. Returns
    (blocking, nonviable) with blockers deduped."""
    vals = validations(obj)
    blocking, nonviable = [], []
    waivers = obj.get("waivers") or {}

    for gate in REQUIRED_GATES:
        archs = [a for a in vals if gate in gates_for(a)]
        w = waivers.get(gate)
        # Satisfied by evidence at the current head, or by an approved waiver. One
        # definition, shared with arch_task, so the gate that dispatches a validation
        # and the gate that clears the PR cannot disagree.
        if gate_satisfied(obj, gate):
            if w and w.get("approved_by"):
                nonviable.append(f"{gate} (waived by {w['approved_by']})")
            else:
                nonviable.extend(a for a in archs
                                 if vals[a].get("state") != "completed" and vals[a].get("blocked"))
            continue
        if w and w.get("refused_by"):
            blocking.append((gate, f"waiver REFUSED by {w['refused_by']} -- "
                                   f"{(w.get('refused_note') or '')[:120]}"))
            continue
        if w and not w.get("approved_by"):
            blocking.append((gate, "waiver suggested but not approved by a maintainer"))
            continue
        candidates = [(a, vals[a].get("state")) for a in archs
                      if not vals[a].get("blocked")]
        if candidates:
            blocking.extend(candidates)   # completing any ONE clears the gate
        elif archs:
            # Every arch that tried has a documented reason it cannot: no evidence
            # can exist, so the correct move is a waiver request or a scoped claim.
            blocking.append((gate, "no viable arch can satisfy this gate"))
            nonviable.extend(a for a in archs if vals[a].get("blocked"))
        else:
            # Nothing has TRIED. That is not the same as "cannot": the remedy is a
            # validation on a host carrying the attribute, not a waiver.
            blocking.append((gate, "no arch has attempted this gate yet -- validate "
                                   f"on a host with the {gate} attribute"))

    # Licence gate. Here rather than only in the publisher so that EVERY route to an
    # upstream PR passes it, whoever is doing the opening. Judged on the same record
    # as the gates above, not re-resolved through load_status's working-tree-first
    # order, so a stale local copy cannot shadow a clearance recorded on the branch.
    lic_ok, lic_why = license_gate(name, obj)
    if not lic_ok:
        blocking.append(("license", lic_why))

    # Integrity gate: the validated content must be COMMITTED. A fork with
    # uncommitted tracked source/build edits means a validation built against local
    # edits that are NOT in the branch -- the upstream PR would be unbuildable (the
    # baspacho/arrayfire 2026-06 gaps). Block the PR until the working tree is clean.
    dirty = uncommitted_source_files(name)
    if dirty:
        listed = ", ".join(p for _, p in dirty[:6]) + (" ..." if len(dirty) > 6 else "")
        blocking.append(("fork-uncommitted", f"{len(dirty)} uncommitted source/build file(s): {listed}"))

    # An arch that could satisfy several unsatisfied gates would otherwise be listed
    # once per gate; report each blocker once.
    seen, deduped = set(), []
    for item in blocking:
        if item not in seen:
            seen.add(item); deduped.append(item)
    return (deduped, sorted(set(nonviable)))


def fix_ready(name):
    """Is a staged fix round ready for its fork review PR and merge?

    The same bar as pr_ready -- every required gate satisfied at the current
    head_sha (the staging tip, under a fix round), licence standing, fork clean --
    with the PR-existence check inverted: an OPEN upstream PR is the precondition
    here, not a blocker. The opt-out check binds exactly as it does for a first
    submission: a fix push is a new arrival in someone's repository.

    Returns (ready, blocking, nonviable) like pr_ready."""
    obj, _where = project_record(name)
    if obj is None:
        raise FileNotFoundError(str(status_path(name)))
    opt = optout_for(upstream_full_name(name) or "")
    if opt:
        return (False, [("opted-out",
                         f"{opt['who']} asked not to receive pull requests from this "
                         f"effort ({opt['source']})")], [])
    if obj.get("pr_state") != "open":
        return (False, [("no-open-pr", "fix rounds exist only while an upstream PR "
                                       "is open")], [])
    fix = obj.get("fix")
    if not fix:
        return (False, [("no-fix-round", "no staging branch recorded "
                                         "(moatlib.py fix-branch)")], [])
    if not obj.get("published_sha"):
        return (False, [("no-published-sha", "the record does not say what the open "
                                             "PR shows")], [])
    if same_commit(obj.get("head_sha"), obj.get("published_sha")):
        return (False, [("no-delta", f"head_sha equals published_sha "
                                     f"({(obj.get('head_sha') or '?')[:12]}) -- "
                                     f"nothing is staged")], [])
    blocking, nonviable = _gate_blockers(name, obj)
    return (not blocking, blocking, nonviable)


def record_tokens(name, tokens, source=None):
    """Append a token-usage record to projects/<name>/stats.jsonl. `tokens` is an
    agent output-token count for a unit of work when its harness reports one;
    `source` labels the role and harness that produced it. statlib sums these as the
    project's token total. Approximate by nature (output tokens, not full context) --
    statlib always reports tokens as approx=True."""
    rec = {"kind": "tokens", "ts": now_iso(), "tokens": int(tokens)}
    if source:
        rec["source"] = source
    p = PROJECTS / name / "stats.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def commit_project(name, message, extra_paths=()):
    """Commit a project's control-plane artifacts together: status.json,
    notes.md, plan.md, stats.jsonl, surface.json and deferred.json (whichever
    exist), plus any extra_paths. surface.json is here because it is judged by a gate: left
    uncommitted it fails the check on the next push, from whichever host makes it.
    Agents call
    this for every state transition so the per-phase telemetry in stats.jsonl
    (compile/test wall-clock etc., written by timeit.sh -- provenance of the endeavor)
    is persisted WITH the transition and never accumulates uncommitted in the
    shared working tree. Prefer this over commit_and_push for project transitions."""
    paths = [f"projects/{name}/{fn}" for fn in
             ("status.json", "notes.md", "plan.md", "stats.jsonl", "surface.json",
              "deferred.json")
             if (PROJECTS / name / fn).exists()]
    paths.extend(str(p) for p in extra_paths)
    return commit_and_push(paths, message)


# ---- CLI -------------------------------------------------------------------

def _print_json(obj):
    json.dump(obj, sys.stdout, indent=2)
    sys.stdout.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="moatlib")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scaffold", help="create projects/<name>/{status.json,notes.md}")
    s.add_argument("full_name")
    s.add_argument("--url")
    s.add_argument("--branch", default="main")
    # Free-form, matching the schema. The three canonical values drive Strategy A vs B,
    # but the field records what the build IS, and reality includes pccm-codegen,
    # rust-cc-cuda, cmake-cupy-plugin and makefile -- all already in use. The schema was
    # widened for exactly this and the flag was not, so scaffolding spconv failed on the
    # value its own analysis had recorded.
    s.add_argument("--ext", default="unknown", metavar="TYPE")
    s.add_argument("--priority", type=float, default=0.0)
    s.add_argument("--force", action="store_true", help="adopt even if marked skip")
    s.add_argument("--deps", nargs="*", default=[], help="MOAT project name(s) this depends on")

    s = sub.add_parser("next-task", help="print next actionable project for a platform")
    s.add_argument("platform", help="<os>-<gfx>, e.g. linux-gfx90a")

    s = sub.add_parser("set-state")
    s.add_argument("name")
    s.add_argument("platform", help="<os>-<gfx>, e.g. linux-gfx90a")
    s.add_argument("new_state")
    s.add_argument("--agent")

    s = sub.add_parser("set-blocked",
                       help="record that an arch cannot run this, or clear that record")
    s.add_argument("name")
    s.add_argument("platform", help="<os>-<gfx>, e.g. linux-gfx90a")
    s.add_argument("reason", nargs="?")
    s.add_argument("--clear", action="store_true",
                   help="resume: this arch is not blocked after all")

    s = sub.add_parser("worktree",
                       help="create a worktree on a project's port branch, synced to the trunk")
    s.add_argument("name")
    s.add_argument("--path", help="where to put it (default agent_space/wt-<name>)")

    sub.add_parser("stalled", help="projects every architecture gave up on, before review")

    sub.add_parser("misplaced", help="projects whose folder is not where their state says")

    s = sub.add_parser("lessons",
                       help="global edits on port branches; only the orphaned ones need acting on")
    s.add_argument("--pending", action="store_true",
                   help="also list the ones correctly awaiting their port's review")

    sub.add_parser("waivers", help="gate waivers suggested but not yet approved")

    s = sub.add_parser("suggest-waiver",
                       help="record the case for waiving a gate (an agent may; it satisfies nothing)")
    s.add_argument("name")
    s.add_argument("gate")
    s.add_argument("--reason", required=True, help="the obstacle, in checkable detail")

    s = sub.add_parser("refuse-waiver",
                       help="a maintainer declining a suggested waiver, saying what to do instead")
    s.add_argument("name")
    s.add_argument("gate")
    s.add_argument("--by", required=True, help="who decided; never an agent")
    s.add_argument("--note", required=True, help="what to investigate instead")

    s = sub.add_parser("approve-waiver",
                       help="a maintainer approving a suggested waiver, which is what makes it count")
    s.add_argument("name")
    s.add_argument("gate")
    s.add_argument("--by", required=True, help="who approved; never an agent")
    s.add_argument("--reason", help="required only when nothing was suggested first")

    s = sub.add_parser("set-not-portable",
                       help="record a person's verdict that this codebase cannot be ported")
    s.add_argument("name")
    s.add_argument("--reason", help="why, in a sentence someone else can check")
    s.add_argument("--by", help="who decided; required, and never an agent")
    s.add_argument("--clear", action="store_true",
                   help="lift the verdict and return the project to planned")

    s = sub.add_parser("port-lock", help="show, take over, or release a project's fork-write lock")
    s.add_argument("name")
    s.add_argument("--take", metavar="ARCH", help="take the lock for ARCH (a person's decision)")
    s.add_argument("--release", action="store_true", help="release it (e.g. an agent stopped mid-port)")

    s = sub.add_parser("set-hold", help="postpone a whole project (selector skips it on every platform); off resumes it")
    s.add_argument("name")
    s.add_argument("on_off", choices=["on", "off"])
    s.add_argument("--reason")

    s = sub.add_parser("advance-head")
    s.add_argument("name")
    s.add_argument("sha")

    s = sub.add_parser("carry-forward",
                       help="carry a platform's validation forward across a behavior-preserving change")
    s.add_argument("name")
    s.add_argument("platform")
    s.add_argument("sha")
    s.add_argument("method", choices=["source-class", "binary-equiv"])
    s.add_argument("detail")

    s = sub.add_parser("classify",
                       help="classify a fork delta: doc-only/comment-only/rename-only/mixed")
    s.add_argument("name")
    s.add_argument("old_sha")
    s.add_argument("new_sha")

    s = sub.add_parser("commit-project",
                       help="commit a project's status/notes/plan/stats together (telemetry-safe)")
    s.add_argument("name")
    s.add_argument("message")

    s = sub.add_parser("record-tokens", help="append a token-usage record to a project's stats.jsonl")
    s.add_argument("name")
    s.add_argument("tokens", type=int)
    s.add_argument("source", nargs="?", default=None)

    s = sub.add_parser("pr-ready", help="check PR readiness: every required gate satisfied by a completed arch or an approved waiver")
    s.add_argument("name")

    si = sub.add_parser("set-intake", help="record intake's recommendation (not a decision)")
    si.add_argument("name")
    si.add_argument("verdict", choices=list(INTAKE_VERDICTS))
    si.add_argument("--summary", required=True, help="one line for the queue table")
    si.add_argument("--reason", choices=SKIP_REASONS, help="required when declining")
    si.add_argument("--duplicate", help="existing AMD/ROCm effort, or 'none'")
    si.add_argument("--viable", choices=["yes", "no", "unknown"])
    s = sub.add_parser("record-license-clearance",
                       help="record a person's decision to allow a tier 3/4 project upstream")
    s.add_argument("name")
    s.add_argument("--by", required=True, help="who approved it (never an agent)")
    s.add_argument("--note", help="scope or conditions of the approval")

    s = sub.add_parser("license-gate", help="may this port be offered upstream on licence grounds?")
    s.add_argument("name")

    s = sub.add_parser("set-review-pr", help="record the fork review PR where the port is approved")
    s.add_argument("name")
    s.add_argument("url", nargs="?")
    s.add_argument("--clear", action="store_true",
                   help="retract a recorded review PR; always allowed")

    s = sub.add_parser("record-pr-approval",
                       help="snapshot the approval standing on the fork review PR")
    s.add_argument("name")
    s.add_argument("--review-pr", help="review PR URL, if status.json has none yet")

    s = sub.add_parser("pr-approval", help="does a standing approval still cover what we would publish now?")
    s.add_argument("name")
    s.add_argument("--offline", action="store_true",
                   help="check the recorded snapshot only, without re-reading GitHub")

    s = sub.add_parser("pr-commands",
                       help="audit every /moat command on the review PR; nonzero if any blocks publishing")
    s.add_argument("name")

    sub.add_parser("pr-candidates",
                   help="list projects whose upstream PR is ready to open (honors recorded dispositions; "
                        "use this instead of scanning raw state==completed)")

    s = sub.add_parser("audit-clean", help="report forks with uncommitted tracked source/build edits (integrity-gap fingerprint)")
    s.add_argument("name", nargs="?", default=None, help="one project, or omit to scan every fork")

    s = sub.add_parser("set-pr-open", help="record PR metadata after creating the upstream PR")
    s.add_argument("name")
    s.add_argument("pr_url")
    s.add_argument("pr_number", type=int)

    s = sub.add_parser("fix-branch",
                       help="establish or report the staging branch for a fix round "
                            "on an open upstream PR")
    s.add_argument("name")

    s = sub.add_parser("set-fix-review-pr",
                       help="record the fork review PR where the staged fix delta is approved")
    s.add_argument("name")
    s.add_argument("url", nargs="?")
    s.add_argument("--clear", action="store_true",
                   help="retract a recorded fix review PR; always allowed")

    s = sub.add_parser("fix-ready",
                       help="check a staged fix round: every required gate satisfied at the staging tip")
    s.add_argument("name")

    s = sub.add_parser("protect-fork",
                       help="install the fork pre-push hook that refuses pushes to an open PR's branch")
    s.add_argument("name", nargs="?", default=None,
                   help="one project, or omit to protect every local fork clone")

    s = sub.add_parser("set-fix-merged",
                       help="record a fix round the trusted merge path already pushed "
                            "(recovery only; it authorises nothing)")
    s.add_argument("name")
    s.add_argument("new_published_sha")

    s = sub.add_parser("pr-state",
                       help="a project's recorded upstream PR state, from whichever ref holds it")
    s.add_argument("name")
    s.add_argument("--refresh", action="store_true",
                   help="fetch the project's refs first (the fork pre-push hook does)")

    s = sub.add_parser("set-pr-merged", help="record that the upstream PR merged")
    s.add_argument("name")

    s = sub.add_parser("set-pr-closed", help="record that the upstream PR closed without merging")
    s.add_argument("name")
    s.add_argument("--note", help="why it closed (withdrawn, rejected, superseded)")

    s = sub.add_parser("squash-carry-forward",
                       help="advance head to a tree-identical PR-prep squash, carrying validated platforms forward (no revalidation)")
    s.add_argument("name")
    s.add_argument("new_sha")

    rf = sub.add_parser("release-forks",
                        help="advance awaiting-fork projects whose fork now exists")
    rf.add_argument("--dry-run", action="store_true")
    fl = sub.add_parser("fleet", help="actionable work across all refs, not just this checkout")
    fl.add_argument("platform")
    db = sub.add_parser("dep-blocked",
                        help="projects held back only by a dependency, and why")
    db.add_argument("platform")
    bs = sub.add_parser("branch-sync",
                        help="merge the trunk into this port branch, but only if "
                             "something a port can feel has changed there")
    bs.add_argument("--apply", action="store_true")
    s = sub.add_parser("validate")
    s.add_argument("name")
    s = sub.add_parser("show")
    s.add_argument("name")
    sub.add_parser("projects",
                   help="every project MOAT knows and where its record lives "
                        "(a project's own branch outranks this checkout)")

    s = sub.add_parser("set-deps", help="record the MOAT projects a project depends on")
    s.add_argument("name")
    s.add_argument("deps", nargs="*")

    sub.add_parser("deps", help="print inter-project dependencies and what is blocked on them")
    sub.add_parser("dep-doc-gaps",
                   help="dependency providers whose notes.md lacks the required "
                        "'## Install as a dependency' section")

    args = ap.parse_args(argv)

    if args.cmd == "scaffold":
        name = scaffold_project(args.full_name, args.url, args.branch, args.ext, args.priority, args.force, args.deps)
        print(f"scaffolded projects/{name}" + (f" (depends_on={args.deps})" if args.deps else ""))
    elif args.cmd == "next-task":
        t = next_task(args.platform)
        if t is None:
            print("NONE")
            return 0
        _print_json(t)
    elif args.cmd == "set-state":
        set_state(args.name, args.platform, args.new_state, agent=args.agent)
        print(f"{args.name}/{args.platform} -> {args.new_state}")
    elif args.cmd == "set-blocked":
        if args.clear:
            # The reason is not carried forward: it described a state of affairs that
            # someone has just decided is over. Its home is the project's notes.md,
            # where the next porter reads it, and clearing here without recording it
            # there loses the diagnosis.
            set_blocked(args.name, args.platform, False)
            print(f"{args.name}/{args.platform} unblocked")
        elif not args.reason:
            print("set-blocked: a reason is required (or pass --clear)", file=sys.stderr)
            return 1
        else:
            set_blocked(args.name, args.platform, True, args.reason)
            print(f"{args.name}/{args.platform} blocked: {args.reason}")
    elif args.cmd == "worktree":
        path, detail = make_worktree(args.name, args.path)
        print(path)
        print(f"   trunk sync: {detail}", file=sys.stderr)
    elif args.cmd == "stalled":
        rows = [(n, o) for n, o, _w in project_records() if stalled(o)]
        for n, o in sorted(rows):
            why = next((b.get("blocked_reason") or "" for b in validations(o).values()
                        if b.get("blocked")), "")
            print(f"{n}\t{project_stage(o)}\t{why[:100]}")
        if rows:
            print(f"-- {len(rows)} project(s) waiting on a person: continue on other "
                  f"hardware, or `set-not-portable <name> --reason ... --by <who>`",
                  file=sys.stderr)
    elif args.cmd == "lessons":
        rows = branch_lessons()
        orphaned = [r for r in rows if r[2]]
        pending = [r for r in rows if not r[2]]
        for name, paths, _o in orphaned:
            print(f"{name}\tORPHANED\t{','.join(paths[:4])}")
        if args.pending:
            for name, paths, _o in pending:
                print(f"{name}\tpending-review\t{','.join(paths[:4])}")
        if orphaned:
            print(f"-- {len(orphaned)} finished port(s) carry a global edit the trunk "
                  f"does not have. Their branches are about to be deleted and nothing "
                  f"will carry it: read each diff, then land it with the port",
                  file=sys.stderr)
        elif args.pending and pending:
            print(f"-- {len(pending)} branch(es) carry a lesson awaiting their port's "
                  f"review, which is where it belongs. Do NOT lift these to the trunk: "
                  f"the review is what makes a lesson trustworthy",
                  file=sys.stderr)
    elif args.cmd == "misplaced":
        rows = misplaced_folders()
        for name, where, what, work in rows:
            todo = ",".join(f"{a}={st}" for a, st in work[:3])
            print(f"{name}\t{where}\t{what}\t{todo}")
        if rows:
            n = sum(1 for r in rows if r[1] == "trunk")
            print(f"-- {len(rows)} misplaced ({n} on the trunk with work outstanding, "
                  f"which under branch protection turns every status write into a PR)",
                  file=sys.stderr)
    elif args.cmd == "waivers":
        rows = pending_waivers()
        for name, gate, reason, at in rows:
            print(f"{name}\t{gate}\t{at}\t{reason[:120]}")
        if rows:
            print(f"-- {len(rows)} waiver(s) awaiting a maintainer; each BLOCKS its "
                  f"project's PR until approved: "
                  f"`moatlib.py approve-waiver <name> <gate> --by <who>`", file=sys.stderr)
    elif args.cmd == "suggest-waiver":
        w = suggest_waiver(args.name, args.gate, args.reason)
        print(f"{args.name}: {args.gate} waiver SUGGESTED -- {w['reason'][:100]}")
        print("   satisfies nothing until a maintainer approves it; it blocks pr-ready "
              "meanwhile", file=sys.stderr)
    elif args.cmd == "refuse-waiver":
        w = refuse_waiver(args.name, args.gate, args.by, args.note)
        print(f"{args.name}: {args.gate} waiver REFUSED by {w['refused_by']} -- {w['refused_note'][:100]}")
        print("   the gate stays unsatisfied; the refusal says what to do about it "
              "instead", file=sys.stderr)
    elif args.cmd == "approve-waiver":
        w = approve_waiver(args.name, args.gate, args.by, args.reason)
        print(f"{args.name}: {args.gate} waived by {w['approved_by']} -- {w['reason'][:100]}")
    elif args.cmd == "set-not-portable":
        obj = set_not_portable(args.name, args.reason, args.by, clear=args.clear)
        if args.clear:
            print(f"{args.name}: not-portable lifted; stage=planned")
        else:
            np = obj["not_portable"]
            print(f"{args.name}: not-portable, by {np['by']} -- {np['reason']}")
    elif args.cmd == "port-lock":
        lock = port_lock(args.name, take=args.take, release=args.release)
        print(f"{args.name}: fork-write lock held by {lock['arch']} since {lock['since']}"
              if lock else f"{args.name}: no fork-write lock held")
    elif args.cmd == "set-hold":
        on = args.on_off == "on"
        set_hold(args.name, on, args.reason)
        print(f"{args.name} on_hold: {on}" + (f" ({args.reason})" if on and args.reason else ""))
    elif args.cmd == "advance-head":
        advance_head(args.name, args.sha)
        print(f"{args.name} head_sha -> {args.sha}")
    elif args.cmd == "carry-forward":
        carry_forward(args.name, args.platform, args.sha, args.method, args.detail)
        print(f"{args.name}/{args.platform} carried forward -> {args.sha} ({args.method})")
    elif args.cmd == "classify":
        v = _classify_safe(_fork_repo(args.name), args.old_sha, args.new_sha)
        if v is None:
            print("class=unknown arch_independent=False (classification failed -> revalidate)")
        else:
            print(f"class={v.cls} arch_independent={v.arch_independent} inert={v.inert}")
            print(v.detail)
    elif args.cmd == "commit-project":
        ok = commit_project(args.name, args.message)
        print(f"committed projects/{args.name} (status/notes/plan/surface/deferred/stats)" if ok else "(nothing to commit)")
    elif args.cmd == "record-tokens":
        r = record_tokens(args.name, args.tokens, args.source)
        print(f"recorded {r['tokens']} tokens for {args.name}" + (f" ({args.source})" if args.source else ""))
    elif args.cmd == "squash-carry-forward":
        ok, info = squash_carry_forward(args.name, args.new_sha)
        if not ok:
            print(f"REFUSED: {info}")
        else:
            msg = f"{args.name} -> {args.new_sha[:8]}: carried {info['carried']}; kept-blocked {info['kept_blocked']}"
            if info.get("optional"):
                msg += f"; not blocking (gates already satisfied, or unscheduled) {info['optional']}"
            if info.get("stale"):
                msg += (f"; STALE completed {info['stale']} (validated an older head; "
                        f"owed a revalidation, not carried)")
            if info["skipped"]:
                msg += f"; SKIPPED actionable {info['skipped']} (should not squash yet)"
            print(msg)
    elif args.cmd == "pr-ready":
        ready, blocking, nonviable = pr_ready(args.name)
        print(f"{args.name}: PR-ready={ready}")
        if not _fork_repo(args.name).is_dir():
            # The cleanliness gate needs the clone: absent, there is nothing to
            # judge, and silence here read as "checked and clean" when it meant
            # "not checkable on this host".
            print("  note: no local fork clone, so fork cleanliness was NOT "
                  "re-checked here -- it was enforced on the hosts that validated")
        if blocking:
            print("  BLOCKING (every required gate needs ONE completed arch at head_sha, "
                  "or an approved waiver; fork must be clean): "
                  + ", ".join(f"{p}={s}" for p, s in blocking))
        if nonviable:
            print("  non-viable (does not block; scope the PR body): " + ", ".join(nonviable))
    elif args.cmd == "set-intake":
        viable = {"yes": True, "no": False, "unknown": None}.get(args.viable)
        _print_json(set_intake(args.name, args.verdict, args.summary,
                               reason=args.reason, duplicate=args.duplicate,
                               viable=viable))
    elif args.cmd == "record-license-clearance":
        c = record_license_clearance(args.name, args.by, args.note)
        print(f"{args.name}: tier {c['tier']} cleared upstream by {c['approved_by']}")
    elif args.cmd == "license-gate":
        ok, why = license_gate(args.name)
        print(f"{args.name}: license-ok={ok} ({why})")
        return 0 if ok else 1
    elif args.cmd == "set-review-pr":
        # Retracting has its own flag, so the bare form must not also retract. `url` is
        # optional, and omitting it -- a typo, a shell that ate the argument, or reading
        # `set-review-pr <name>` as "show me this project's review PR" -- used to take
        # the same path as `--clear`: it skipped the gate check, wrote null, printed
        # `review PR -> None` and exited 0. That erases the one field saying where the
        # port's approval lives, so the port drops out of `--publish` and comes back in
        # `--review` as needing a PR it already has.
        if not args.url and not args.clear:
            try:
                cur = load_status(args.name).get("review_pr") or "none recorded"
            except (FileNotFoundError, ValueError) as e:
                cur = f"unreadable -- {e}"
            print(f"set-review-pr: no URL given. Pass one to record it, or --clear to "
                  f"retract the recorded one. {args.name} currently: {cur}",
                  file=sys.stderr)
            return 2
        set_review_pr(args.name, None if args.clear else args.url)
        print(f"{args.name}: review PR -> {'(cleared)' if args.clear else args.url}")
    elif args.cmd == "record-pr-approval":
        a = record_pr_approval(args.name, args.review_pr)
        print(f"{args.name}: approved by {a['approved_by']} for "
              f"{(a.get('head_sha') or '?')[:8]} on {a['review_pr']}")
    elif args.cmd == "pr-approval":
        ok, why = pr_approval_valid(args.name, live=not args.offline)
        print(f"{args.name}: approval-valid={ok} ({why})")
        return 0 if ok else 1
    elif args.cmd == "pr-commands":
        obj, _where = project_record(args.name)
        url = ((obj or {}).get("pr_approval") or {}).get("review_pr") \
            or ((obj or {}).get("fix") or {}).get("review_pr") \
            or (obj or {}).get("review_pr")
        if not url:
            print(f"{args.name}: no review PR recorded")
            return 1
        pr = fetch_review_pr(url)
        if pr is None:
            print(f"{args.name}: could not reach the review PR at {url} -- "
                  f"an outage is not an answer")
            return 1
        latest = _decision_events(pr)
        for r in sorted(latest.values(), key=lambda e: e.get("at") or ""):
            what = _command_of(r) or str(r.get("state") or "").lower()
            print(f"  {r.get('at')}  {r.get('login')} ({r.get('assoc')}): {what}")
        if not latest:
            print("  no decisions on the review PR")
        blockers, notes = moat_command_audit(pr)
        for n in notes:
            print(f"  note: {n}")
        for b in blockers:
            print(f"  BLOCKS: {b}")
        print(f"{args.name}: {'BLOCKED' if blockers else 'clear'} on {url}")
        return 1 if blockers else 0
    elif args.cmd == "pr-candidates":
        names = [n for n, _o, _w in project_records()]
        ready_names = []
        for n in names:
            try:
                ready, _, nonviable = pr_ready(n)
            except Exception:
                continue
            if not ready:
                continue
            # A PR needs validated content: at least one platform actually completed.
            # An all-blocked project is "ready" only vacuously (everything non-viable).
            states = {b.get("state") for b in load_status(n)["platforms"].values()}
            if "completed" not in states:
                continue
            ready_names.append((n, nonviable))
        for n, nonviable in ready_names:
            scope = f"  (scope PR to exclude non-viable: {', '.join(nonviable)})" if nonviable else ""
            print(f"{n}{scope}")
        print(f"-- {len(ready_names)} PR-ready project(s); terminal-outcome, already-open, "
              "and all-blocked projects excluded")
        print("NOTE: pr_ready checks the licence gate and recorded dispositions itself; "
              "a project listed here has passed both. What it cannot judge is whether "
              "the change is worth sending -- read the diff before opening.")
    elif args.cmd == "audit-clean":
        names = ([args.name] if args.name
                 else [n for n, _o, _w in project_records()])
        real_gap = False
        judged = 0
        for n in names:
            # A missing clone is not evidence of cleanliness, so it must not count
            # toward the clean bill printed below -- same rule as pr-ready, which
            # says the check did not run rather than silently passing.
            if not _fork_repo(n).is_dir():
                continue
            judged += 1
            files = uncommitted_source_files(n)
            if not files:
                continue
            try:
                states = {b.get("state") for b in load_status(n)["platforms"].values()}
            except Exception:
                states = set()
            # A real integrity gap = uncommitted edits on a fork that already has a
            # validated platform resting on them. Otherwise it is just uncommitted
            # WIP on an unfinished port.
            terminal = states & set(PORT_DONE_STATES)
            tag = "INTEGRITY GAP (has a completed platform)" if terminal else "uncommitted WIP (no completed platform yet)"
            if terminal:
                real_gap = True
            print(f"{n}: {len(files)} uncommitted source/build file(s) -- {tag}:")
            for code, p in files:
                print(f"    {code:3} {p}")
        if real_gap:
            sys.exit(1)
        elif judged == 0:
            print("audit-clean: NO local fork clone to judge here"
                  + (f" ({args.name})" if args.name else "")
                  + " -- the check did not run; it binds on the hosts holding the clones")
        else:
            print(f"OK: no fork with a completed/pr platform has uncommitted source "
                  f"edits ({judged} local clone(s) judged"
                  + (f"; {len(names) - judged} project(s) have no clone here"
                     if len(names) > judged else "")
                  + (f" ({args.name})" if args.name else "") + ")")
    elif args.cmd == "set-pr-open":
        set_pr_open(args.name, args.pr_url, args.pr_number)
        print(f"{args.name}: PR opened -> {args.pr_url}")
    elif args.cmd == "fix-branch":
        fix = fix_branch(args.name)
        print(f"{args.name}: fix round on {fix['branch']} "
              f"(base {(fix.get('base_sha') or '?')[:12]}, review PR "
              f"{fix.get('review_pr') or 'not yet open'})")
    elif args.cmd == "set-fix-review-pr":
        # Same refusal shape as set-review-pr: the bare form must not retract.
        if not args.url and not args.clear:
            try:
                cur = ((load_status(args.name).get("fix") or {}).get("review_pr")
                       or "none recorded")
            except (FileNotFoundError, ValueError) as e:
                cur = f"unreadable -- {e}"
            print(f"set-fix-review-pr: no URL given. Pass one to record it, or "
                  f"--clear to retract the recorded one. {args.name} currently: "
                  f"{cur}", file=sys.stderr)
            return 2
        set_fix_review_pr(args.name, None if args.clear else args.url)
        print(f"{args.name}: fix review PR -> "
              f"{'(cleared)' if args.clear else args.url}")
    elif args.cmd == "fix-ready":
        ready, blocking, nonviable = fix_ready(args.name)
        print(f"{args.name}: fix-ready={ready}")
        if blocking:
            print("  BLOCKING (every required gate needs ONE completed arch at the "
                  "staging tip, or an approved waiver; fork must be clean): "
                  + ", ".join(f"{p}={s}" for p, s in blocking))
        if nonviable:
            print("  non-viable (does not block; scope the claim): "
                  + ", ".join(nonviable))
        return 0 if ready else 1
    elif args.cmd == "protect-fork":
        names = [args.name] if args.name else sorted(
            p.parent.name for p in PROJECTS.glob("*/src/.git"))
        if not names:
            print("protect-fork: no fork clones in this checkout")
        warned = 0
        for n in names:
            level, message = protect_fork(n)
            # orient.sh discards stdout; a clone it could NOT protect has to be the
            # one thing that still reaches the operator.
            print(message, file=sys.stderr if level == "warn" else sys.stdout)
            warned += level == "warn"
        if warned:
            print(f"protect-fork: {warned} fork clone(s) are UNPROTECTED (above)",
                  file=sys.stderr)
    elif args.cmd == "set-fix-merged":
        obj = set_fix_merged(args.name, args.new_published_sha)
        print(f"{args.name}: published_sha -> {obj['published_sha'][:12]}")
    elif args.cmd == "pr-state":
        state, why = pr_state_of(args.name, refresh=args.refresh)
        if state is None:
            print(why, file=sys.stderr)
            return 1
        print(state)
    elif args.cmd == "set-pr-merged":
        set_pr_merged(args.name)
        print(f"{args.name}: PR merged")
    elif args.cmd == "set-pr-closed":
        set_pr_closed(args.name, args.note)
        print(f"{args.name}: PR closed without merging"
              + (f" ({args.note})" if args.note else ""))
    elif args.cmd == "release-forks":
        rel = release_awaiting_fork(dry_run=args.dry_run)
        for name, slug in rel:
            print(f"{'would release' if args.dry_run else 'released'} {name} -> {slug}")
        # "Nothing released" and "nothing waiting" are different answers and this
        # printed the second for both, so four projects waiting on a fork nobody had
        # created yet read as a clean bill of health.
        waiting = sorted(n for n, o, _w in project_records()
                         if project_stage(o) == "awaiting-fork")
        still = [n for n in waiting if n not in {r[0] for r in rel}]
        if still:
            print(f"release-forks: {len(still)} still waiting on a fork that does not "
                  f"exist yet -- {', '.join(still)}")
        elif not rel:
            print("release-forks: nothing waiting on a fork")
        return 0
    elif args.cmd == "fleet":
        for r in fleet(args.platform):
            print(f"{r['project']}\t{r['where']}\t{r['state']}\t{r['stage']}"
                  f"\t{r['branch']}")
    elif args.cmd == "dep-blocked":
        rows = dep_blocked(args.platform)
        for name, report in rows:
            for dep, verdict, detail in report:
                print(f"{name}\t{dep}\t{verdict}\t{detail}")
        if not rows:
            print("", end="")
    elif args.cmd == "branch-sync":
        action, detail = branch_sync(apply=args.apply)
        print(f"branch-sync: {action} -- {detail}")
        return 1 if action == "conflict" else 0
    elif args.cmd == "validate":
        load_status(args.name)
        print(f"{args.name} status.json valid")
    elif args.cmd == "show":
        obj, where = project_record(args.name)
        if obj is None:
            print(f"{args.name}: no record on any ref", file=sys.stderr)
            return 1
        if where != "local":
            print(f"show: {args.name} read from its {where} record", file=sys.stderr)
        _print_json(obj)
    elif args.cmd == "projects":
        for n, where in sorted(all_projects().items()):
            print(f"{n}\t{where}")
    elif args.cmd == "set-deps":
        obj = load_status(args.name)
        obj["depends_on"] = list(args.deps)
        obj["updated_at"] = now_iso()
        save_status(args.name, obj)
        print(f"{args.name} depends_on = {args.deps}")
    elif args.cmd == "deps":
        any_dep = False
        for d_name, obj, _where in project_records():
            deps = obj.get("depends_on", [])
            if not deps:
                continue
            any_dep = True
            unmet = unmet_deps(obj)
            mark = "READY (deps complete)" if not unmet else ("WAITING on " + ", ".join(unmet))
            print(f"{d_name}: depends_on={deps} -> {mark}")
        if not any_dep:
            print("(no inter-project dependencies recorded)")
    elif args.cmd == "dep-doc-gaps":
        rows = dep_doc_gaps()
        for dep, users in rows:
            print(f"{dep}\t{','.join(users)}")
        if rows:
            print(f"-- {len(rows)} provider(s) lack the '## Install as a dependency' "
                  f"section DEPENDENCIES.md requires, so a dependent's porter has no "
                  f"recipe to follow. Write it in the provider's notes.md on its own "
                  f"branch", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
