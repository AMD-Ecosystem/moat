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
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS = REPO_ROOT / "projects"
SCHEMA_VERSION = 1

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
    """Per-arch validation records. Accepts the legacy `platforms` key so a
    status.json written before the gate model still reads."""
    return obj.get("validations") or obj.get("platforms") or {}


PORT_BRANCH = "moat-port"  # the topic branch that holds the port on each fork

# Per-arch pipeline. `awaiting-port` is where an arch waits until a port exists to
# validate (formerly blocked-needs-gfx90a, which named a lead that no longer exists).
# The `blocked` boolean (needs user input) is orthogonal and set separately.
#
# The upstream PR is NOT in here. It is one fact about the project, not about any
# arch: opening it changes nothing an arch validated, and parking it on one arch's
# record overwrote that arch's real state (a merged PR rendered as an unknown status
# in the README because `upstream-landed` had displaced `completed`). It lives in the
# project-level `pr_state` instead -- see PR_STATES.
ALLOWED = {
    "unclaimed": {"screened", "planned"},
    "screened": {"awaiting-fork", "planned"},
    "awaiting-fork": {"screened", "planned", "porting"},
    "awaiting-upstream": {"planned", "porting", "unclaimed"},
    "awaiting-port": {"port-ready"},
    "planned": {"porting", "awaiting-upstream"},
    "porting": {"ported"},
    "ported": {"review-passed", "changes-requested"},
    "changes-requested": {"porting", "delta-ported"},
    "review-passed": {"completed", "validation-failed"},
    "validation-failed": {"porting", "delta-ported"},
    "port-ready": {"completed", "validation-failed"},
    "delta-ported": {"review-passed", "changes-requested"},
    "revalidate": {"completed", "validation-failed"},
    "completed": {"revalidate"},
}
STATES = set(ALLOWED) | {s for v in ALLOWED.values() for s in v}

# Project-level upstream PR lifecycle, orthogonal to every arch's state. A port can
# be validated everywhere with no PR, or carry a merged PR while an arch is
# revalidating a later commit; neither fact constrains the other.
PR_STATES = ("open", "merged", "closed")

# Which agent handles each state, and selection priority (lower = sooner).
# Resume-before-start: drain work in flight before opening new fronts.
STAGE_FOR_STATE = {
    "unclaimed": "intake",
    "screened": "planner",
    "planned": "porter",
    "porting": "porter",
    "changes-requested": "porter",
    "validation-failed": "porter",
    "ported": "reviewer",
    "delta-ported": "reviewer",
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
    "planned": 5,
    "ported": 6,
    "review-passed": 7,
    "port-ready": 8,
    "screened": 9,
    "unclaimed": 10,
}
# States that take no agent action: terminal, gated on a human, or waiting on
# something outside our control. `awaiting-fork` waits on an org admin to create the
# fork; `awaiting-upstream` waits on an external event (a third party's PR landing,
# say) and is viable-but-parked rather than dead.
#
# `awaiting-fork` is where a project waits to be taken up. The fork appearing in the
# org is what releases it: creating one is a deliberate act by someone who can, so
# its existence carries the decision and nothing else needs to record one.
INERT = {"completed", "awaiting-port", "awaiting-fork", "awaiting-upstream"}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def claim_ttl_seconds():
    """Read claim TTL from config/moat.toml; default 30 min. A .claim file
    untouched for longer than this is stale (its CLI crashed) and reclaimable."""
    cfg = REPO_ROOT / "config" / "moat.toml"
    minutes = 30
    if cfg.exists():
        try:
            minutes = tomllib.loads(cfg.read_text()).get("claims", {}).get("claim_ttl_minutes", 30)
        except (tomllib.TOMLDecodeError, OSError):
            pass
    return float(minutes) * 60.0


def claim_live(name):
    """True if projects/<name>/.claim exists and was refreshed within the TTL.
    Same-host coordination via the shared filesystem; .claim is gitignored."""
    cf = PROJECTS / name / ".claim"
    if not cf.exists():
        return False
    return (time.time() - cf.stat().st_mtime) < claim_ttl_seconds()


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
    return {
        "state": initial_state,
        "blocked": False,
        "blocked_reason": None,
        "validated_sha": None,
        "started_at": None,
        "completed_at": None,
        "updated_at": now_iso(),
        "stats": _empty_stats(),
    }


def status_path(name):
    return PROJECTS / name / "status.json"


def load_status(name):
    with open(status_path(name)) as f:
        obj = json.load(f)
    validate_status(obj)
    return obj


def upstream_full_name(name):
    """The upstream repo as `owner/repo`, from the URL status.json already holds.
    This is the key dispositions.json is written under, so it is how a project record
    finds its own disposition."""
    try:
        url = (load_status(name).get("upstream_url") or "").rstrip("/")
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None          # not adopted here; callers treat that as "no record"
    tail = url.replace("https://github.com/", "")
    return tail if tail.count("/") == 1 else None


def save_status(name, obj):
    validate_status(obj)
    obj["updated_at"] = now_iso()
    p = status_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def validate_status(obj):
    """Light hand-rolled validation (no jsonschema dependency). Raises ValueError."""
    for k in ("schema_version", "name", "upstream_url", "fork_default_branch",
              "priority", "ext_type", "platforms"):
        if k not in obj:
            raise ValueError(f"status.json missing required key: {k}")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {obj['schema_version']}")
    unknown = {p for p in obj["platforms"] if platform_problem(p)}
    if unknown:
        raise ValueError(f"unknown arch(es) {sorted(unknown)}; add them to config/arches.toml")
    for plat, blk in obj["platforms"].items():
        if blk.get("state") not in STATES:
            raise ValueError(f"{plat}: invalid state {blk.get('state')!r}")
        if not isinstance(blk.get("blocked"), bool):
            raise ValueError(f"{plat}: blocked must be boolean")


# ---- state machine ---------------------------------------------------------

def set_state(name, platform, new_state, agent=None, save=True):
    """Validate and apply a transition with its side effects.

    A platform's record is created on first use rather than pre-seeded, so a host
    whose GPU nothing has recorded before simply starts working and its record
    appears. The platform still has to be well-formed and its wavefront width
    known."""
    problem = platform_problem(platform)
    if problem:
        raise ValueError(problem)
    obj = load_status(name)
    if platform not in obj["platforms"]:
        obj["platforms"][platform] = _platform_block(
            "awaiting-port" if obj.get("head_sha") else "unclaimed")
    blk = obj["platforms"][platform]
    cur = blk["state"]
    if new_state == cur:
        return obj
    if new_state not in ALLOWED.get(cur, set()):
        raise ValueError(f"{name}/{platform}: illegal transition {cur} -> {new_state}")
    blk["state"] = new_state
    ts = now_iso()
    blk["updated_at"] = ts
    if agent:
        blk["last_agent"] = agent  # informational; not in strict schema
    if new_state in ("porting", "port-ready", "delta-ported") and not blk.get("started_at"):
        blk["started_at"] = ts
    if new_state == "completed":
        blk["completed_at"] = ts
        blk["validated_sha"] = obj.get("head_sha")
        # A real-GPU validation supersedes any prior carry-forward tag; drop the
        # stale annotation so the metadata reflects how this completion was reached.
        blk.pop("carry_forward", None)
        _open_validation_season(obj)
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
    obj["platforms"][platform] = blk
    if save:
        save_status(name, obj)
    return obj


def set_blocked(name, platform, blocked, reason=None):
    obj = load_status(name)
    blk = obj["platforms"][platform]
    blk["blocked"] = bool(blocked)
    blk["blocked_reason"] = reason if blocked else None
    blk["updated_at"] = now_iso()
    save_status(name, obj)
    return obj


def set_hold(name, on_hold, reason=None):
    """Project-wide postponement. A held project is skipped by the selector on
    every platform (actionable() returns False) without touching any platform
    state, so the hold is reversible and leaves resume points intact. Used to
    park a whole stack (e.g. RAPIDS) that we are deliberately not working yet."""
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
                          "id": r.get("id")} for r in raw]
    pr["slug"], pr["number"] = slug, num
    owner, repo = slug.split("/", 1)
    q = ('{repository(owner:"%s",name:"%s"){pullRequest(number:%s){lastEditedAt}}}'
         % (owner, repo, num))
    g = _gh_json("api", "graphql", "-f", f"query={q}")
    pr["lastEditedAt"] = (((g or {}).get("data") or {}).get("repository") or {}
                          ).get("pullRequest", {}).get("lastEditedAt")
    return pr


def _approving_review(pr):
    """A standing approval on the review PR, or None.

    Only the latest review per author counts: someone who approves and then requests
    changes has withdrawn the approval, and honouring the earlier APPROVED event
    would publish over an objection. An outstanding CHANGES_REQUESTED from ANYONE
    blocks, even alongside somebody else's approval -- publishing while a reviewer is
    still objecting is the thing this is here to prevent."""
    latest = {}
    for r in pr.get("review_list") or []:
        if not r.get("login") or r.get("state") in ("COMMENTED", "PENDING"):
            continue
        latest[r["login"]] = r
    if any(r.get("state") == "CHANGES_REQUESTED" for r in latest.values()):
        return None
    if pr.get("reviewDecision") == "CHANGES_REQUESTED":
        return None
    return next((r for r in latest.values() if r.get("state") == "APPROVED"), None)


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
    url = review_pr or obj.get("review_pr")
    if not url:
        raise ValueError(f"{name}: no review_pr recorded; pass --review-pr <url>")
    pr = fetch_review_pr(url)
    if pr is None:
        raise ValueError(f"{name}: could not read the review PR at {url}")
    review = _approving_review(pr)
    if review is None:
        raise ValueError(f"{name}: no standing approval on {url}")
    obj["review_pr"] = pr.get("url") or url
    obj["pr_approval"] = {
        "approved_by": review.get("login"),
        "at": review.get("at") or now_iso(),
        # The commit the review was actually attached to, not the branch tip: if
        # anything was pushed between the approval and this snapshot, those differ,
        # and the approved one is the truth.
        "head_sha": review.get("commit") or pr.get("headRefOid"),
        "content_sha256": _content_digest(pr),
        "review_pr": obj["review_pr"],
    }
    save_status(name, obj)
    return obj["pr_approval"]


# Why an approval does not cover what we would publish. Callers act on these very
# differently -- `stale-commits` and `stale-content` mean the approval was overtaken
# and the reviewer should be asked again, while `withdrawn` means that already
# happened and nobody should be pinged a second time -- so the verdict is a code
# rather than prose to be pattern-matched.
APPROVAL_CODES = ("ok", "none", "withdrawn", "stale-commits", "stale-content",
                  "record-mismatch", "unreachable")


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

    # Was the title or body rewritten after the approval? GitHub leaves an approval
    # standing through an edit, so this is the only thing that catches it.
    edited, at = pr.get("lastEditedAt"), review.get("at")
    if edited and at and edited > at:
        return ("stale-content", f"the title or body was edited at {edited}, after the "
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
        if head and a.get("head_sha") != head:
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
    if review.get("commit") and a.get("head_sha") != review["commit"]:
        return ("record-mismatch",
                f"the recorded approval names {(a.get('head_sha') or '?')[:8]}, but the "
                f"approval on GitHub is against {review['commit'][:8]}")

    return ("ok", f"approved by {review['login']} at {at} on {url}, "
                  f"still standing at {(tip or '?')[:8]}")


def license_tier(name):
    """The project's licence tier, from utils/licenses.py. Unknown is tier 4."""
    sys.path.insert(0, str(REPO_ROOT / "utils"))
    import licenses
    return licenses.tier_of(load_status(name).get("license_spdx"))


def license_gate(name):
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
    obj = load_status(name)
    spdx = obj.get("license_spdx")
    tier = license_tier(name)
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


def record_license_clearance(name, approved_by, note=None):
    """Record a person's decision to allow a tier 3/4 project upstream."""
    obj = load_status(name)
    obj["license_clearance"] = {"approved_by": approved_by, "at": now_iso(),
                                "tier": license_tier(name),
                                **({"note": note} if note else {})}
    save_status(name, obj)
    return obj["license_clearance"]


def set_review_pr(name, url):
    """Record the review PR on our own fork -- where the port gets approved."""
    obj = load_status(name)
    obj["review_pr"] = url
    save_status(name, obj)
    return obj


def set_pr_open(name, pr_url, pr_number):
    """Record the upstream PR. Project-level: it changes nothing an arch validated."""
    obj = load_status(name)
    obj["pr_url"] = _clean_pr_url(pr_url)
    obj["pr_number"] = int(pr_number)
    obj["pr_opened_at"] = now_iso()
    obj["pr_state"] = "open"
    save_status(name, obj)
    return obj


def set_pr_merged(name):
    """Record that the upstream PR merged."""
    obj = load_status(name)
    if "pr_url" not in obj:
        raise ValueError(f"{name}: no PR recorded, cannot mark as merged")
    obj["pr_merged_at"] = now_iso()
    obj["pr_state"] = "merged"
    save_status(name, obj)
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
    save_status(name, obj)
    return obj


def _open_validation_season(obj):
    """A port exists at head_sha with no porting lock held, so every other arch may
    validate. There is no lead: this is not "the lead finished", it is "there is now
    something to validate", which any arch's completion establishes."""
    if not obj.get("head_sha"):
        return
    for plat, blk in validations(obj).items():
        if blk.get("blocked"):
            continue
        if blk["state"] == "awaiting-port":
            blk["state"] = "port-ready"
            blk["updated_at"] = now_iso()


def _fork_repo(name):
    return PROJECTS / name / "src"


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

    On any classification failure the platform revalidates -- the safe default."""
    obj = load_status(name)
    obj["head_sha"] = new_sha
    repo = repo or _fork_repo(name)
    for plat in list(obj["platforms"]):
        blk = obj["platforms"][plat]
        if blk["state"] != "completed" or blk.get("validated_sha") == new_sha:
            continue
        old = blk.get("validated_sha")
        verdict = _classify_safe(repo, old, new_sha)
        if verdict is not None and verdict.arch_independent:
            blk["validated_sha"] = new_sha
            blk["updated_at"] = now_iso()
            blk["carry_forward"] = {"from": old, "to": new_sha, "method": "source-class",
                                    "class": verdict.cls, "detail": verdict.detail[:200],
                                    "at": now_iso()}
        else:
            blk["state"] = "revalidate"
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
    if blk["state"] not in ("completed", "revalidate"):
        raise ValueError(f"{name}/{platform}: carry_forward needs completed/revalidate, not {blk['state']}")
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


def project_port_state(name):
    """Best per-arch state of another project, or None if it is not adopted."""
    try:
        states = {b.get("state") for b in validations(load_status(name)).values()}
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError):
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
                             "duplicate", "declined", "other")


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


def platform_state(obj, platform):
    """This platform's state, defaulting an absent record the way set_state does.
    A record that is not there means no host has touched this platform yet."""
    blk = validations(obj).get(platform)
    if blk:
        return blk["state"]
    return "awaiting-port" if obj.get("head_sha") else "unclaimed"


def actionable(obj, platform):
    """Is this platform pickable by an agent on this host right now?"""
    if obj.get("on_hold"):  # project-wide postponement (e.g. the RAPIDS stack)
        return False
    # A decided project is not work, whatever its per-arch records say. Without this
    # a dispositioned project whose folder is still on the trunk gets offered for
    # intake again -- re-screening something already declined.
    disp = disposition_for_project(obj.get("name") or "")
    if disp and disp.get("disposition") == "skip":
        return False
    vals = validations(obj)
    # An ABSENT record means "no host has touched this platform yet", which is what
    # `scaffold` documents ("one appears when a host first works the project") -- so
    # default it the same way set_state does rather than treating it as unselectable.
    # Bailing here deadlocked every newly adopted project: the record is created by
    # working the project, and working it required the record. opencv, rmagine and
    # the two diff-surfel repos sat forked and unoffered from June because of it.
    blk = vals.get(platform) or _platform_block(
        "awaiting-port" if obj.get("head_sha") else "unclaimed")
    if blk["blocked"]:
        return False
    if blk["state"] in INERT:
        return False
    # Only one arch may WRITE to the fork at a time. Validation is read-only on code
    # and writes only its own record, so it never contends.
    lock = obj.get("porting")
    if lock and lock.get("arch") != platform and STAGE_FOR_STATE.get(blk["state"]) == "porter":
        return False
    if unmet_deps(obj):  # deps-first ordering: wait until depended-on ports complete
        return False
    return blk["state"] in STAGE_FOR_STATE


def dep_blocked(platform):
    """Projects this platform would otherwise work, held back only by a dependency.

    `next_task` returning NONE looks identical whether there is genuinely nothing to
    do or a project is waiting on a dependency nobody has adopted. That silence is
    the failure mode: deps-first ordering becomes deps-never and nothing says so."""
    out = []
    if not PROJECTS.exists():
        return out
    for d in sorted(PROJECTS.iterdir()):
        if not (d / "status.json").exists():
            continue
        try:
            obj = load_status(d.name)
        except (ValueError, json.JSONDecodeError):
            continue
        if obj.get("on_hold"):
            continue
        # Would it be pickable if the dependency cleared? Same test as actionable(),
        # minus the dependency check itself.
        disp = disposition_for_project(obj.get("name") or "")
        if disp and disp.get("disposition") == "skip":
            continue
        blk = validations(obj).get(platform) or _platform_block(
            "awaiting-port" if obj.get("head_sha") else "unclaimed")
        if blk.get("blocked") or blk.get("state") in INERT:
            continue
        if blk.get("state") not in STAGE_FOR_STATE:
            continue
        report = dep_report(obj)
        if report:
            out.append((d.name, report))
    return out


def next_task(platform):
    """Pick the single next project for this platform. Returns dict or None."""
    cands = []
    if not PROJECTS.exists():
        return None
    for d in sorted(PROJECTS.iterdir()):
        sp = d / "status.json"
        if not sp.exists():
            continue
        try:
            obj = load_status(d.name)
        except (ValueError, json.JSONDecodeError):
            continue
        if not actionable(obj, platform):
            continue
        state = platform_state(obj, platform)
        cands.append((SELECT_RANK.get(state, 99), -float(obj.get("priority", 0)),
                      d.name, state))
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

    Returns [(name, fork_url)] for the projects released."""
    released = []
    if not PROJECTS.exists():
        return released
    for d in sorted(PROJECTS.iterdir()):
        if not (d / "status.json").exists():
            continue
        try:
            obj = load_status(d.name)
        except (ValueError, json.JSONDecodeError):
            continue
        vals = validations(obj)
        waiting = [a for a, b in vals.items() if b.get("state") == "awaiting-fork"]
        if not waiting:
            continue
        fork = obj.get("fork_url") or f"https://github.com/{org}/{d.name}"
        slug = fork.replace("https://github.com/", "")
        r = subprocess.run(["gh", "api", f"repos/{slug}", "--jq", ".full_name"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode:
            continue                       # still no fork; leave it waiting
        if dry_run:
            released.append((d.name, slug))
            continue
        for a in waiting:
            vals[a]["state"] = "screened"
            vals[a]["updated_at"] = now_iso()
        obj["fork_url"] = f"https://github.com/{slug}"
        save_status(d.name, obj)
        released.append((d.name, slug))
    return released


def unblock_all_followers():
    """Flip awaiting-port -> port-ready wherever a port now exists to validate.
    Called by orient.sh before selection so waiting archs become pickable."""
    changed = []
    if not PROJECTS.exists():
        return changed
    for d in sorted(PROJECTS.iterdir()):
        if not (d / "status.json").exists():
            continue
        try:
            obj = load_status(d.name)
        except (ValueError, json.JSONDecodeError):
            continue
        if not port_done(obj):
            continue
        touched = False
        for plat, blk in validations(obj).items():
            if blk.get("blocked"):
                continue
            if blk["state"] == "awaiting-port":
                blk["state"] = "port-ready"
                blk["updated_at"] = now_iso()
                touched = True
        if touched:
            save_status(d.name, obj)
            changed.append(d.name)
    return changed


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


def set_disposition(full_name, disposition, reason, note="", repo_id=None):
    if disposition == "skip" and reason not in SKIP_REASONS:
        raise ValueError(f"reason must be one of {SKIP_REASONS}")
    d = load_dispositions()
    if repo_id is None:
        repo_id = github_repo_id(full_name)      # best effort; None when offline
    d[full_name.lower()] = {"full_name": full_name, "disposition": disposition,
                            "reason": reason, "note": note, "decided": now_iso(),
                            "repo_id": repo_id}
    save_dispositions(d)
    return d[full_name.lower()]


def clear_disposition(full_name):
    d = load_dispositions()
    if full_name.lower() in d:
        del d[full_name.lower()]
        save_dispositions(d)
        return True
    return False


# ---- scaffolding -----------------------------------------------------------

def scaffold_project(full_name, upstream_url=None, default_branch="main",
                     ext_type="unknown", priority=0.0, force=False, depends_on=None):
    disp = get_disposition(full_name)
    if disp and disp.get("disposition") == "skip" and not force:
        raise ValueError(
            f"{full_name} is marked skip ({disp.get('reason')}): {disp.get('note', '')}. "
            f"Use force=True / --force to adopt anyway.")
    name = full_name.split("/")[-1]
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


def branch_drift(branch, base_ref="origin/main"):
    """What has landed on the trunk that this port branch has not seen, split into
    the changes a port can feel and the ones it cannot.

    Returns (substantive, inert) as sorted path lists. Both empty means the branch
    already carries everything on the trunk."""
    project = branch[len("port/"):] if branch.startswith("port/") else None
    base = _git("merge-base", "HEAD", base_ref, check=False).stdout.strip()
    if not base:
        return ([], [])
    out = _git("diff", "--name-only", base, base_ref, check=False).stdout
    substantive, inert = [], []
    for p in out.splitlines():
        p = p.strip()
        if not p:
            continue
        # This project's own record is substantive even though other projects' are
        # not: someone else may have pushed state for it.
        own = project and p.startswith(f"projects/{project}/")
        if not own and (p in PORT_INERT or p.startswith(PORT_INERT)
                        or p.startswith("projects/")):
            inert.append(p)
        else:
            substantive.append(p)
    return (sorted(substantive), sorted(inert))


def branch_sync(apply=False, base_ref="origin/main"):
    """Bring a port branch up to the trunk's tooling, but only when that is worth a
    merge commit. Returns (action, detail) for the caller to print.

    Merging on every trunk push would put a merge commit on every port branch for a
    README regeneration. Merging on none of them means a port runs whatever skills
    and agent definitions existed the day its branch was cut. So: look first, and
    merge only when something a port can actually feel has moved."""
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    if not branch.startswith("port/"):
        return ("skip", "not a port branch")
    if _git("status", "--porcelain", check=False).stdout.strip():
        return ("dirty", "uncommitted changes -- not merging; commit or stash first")
    _git("fetch", "-q", "origin", base_ref.split("/", 1)[-1], check=False)
    substantive, inert = branch_drift(branch, base_ref)
    if not substantive:
        if not inert:
            return ("current", "up to date with the trunk")
        return ("inert", f"trunk moved, nothing a port can see ({len(inert)} path(s))")
    if not apply:
        return ("would-merge", ", ".join(substantive[:4]))
    ensure_git_config()
    r = _git("merge", "--no-edit", base_ref, check=False)
    if r.returncode:
        _git("merge", "--abort", check=False)
        return ("conflict", f"merging {base_ref} conflicts -- resolve by hand: "
                            f"{', '.join(substantive[:4])}")
    # Push so a sibling host reuses this merge instead of making its own; the branch
    # is shared, and two independent merges of the same trunk diverge for no reason.
    _git("push", "-q", "origin", branch, check=False)
    return ("merged", ", ".join(substantive[:4]))


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
    for _ in range(retries):
        # --autostash so a concurrent agent's unstaged files in the shared
        # working tree don't abort our rebase (multi-agent MOAT runs).
        _git("pull", "--rebase", "--autostash", check=False)
        r = _git("push", check=False)
        if r.returncode == 0:
            return True
    sys.stderr.write("commit_and_push: push failed after retries; left committed locally\n")
    return False


def squash_carry_forward(name, new_sha, repo=None):
    """Advance head to a PR-prep squash, carrying every already-validated platform
    forward WITHOUT revalidation. Valid only when new_sha is a TREE-IDENTICAL
    collapse of the current validated head -- i.e. the squash combined
    already-validated commits and changed no content -- which is the case when the
    squash is done at PR-prep AFTER every platform is terminal (pr_ready). Then the
    squashed commit is known to work everywhere it already worked:
      - each `completed` platform: validated_sha advanced to new_sha, stays completed;
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
    # A gate already satisfied by some completed arch needs nothing more; an arch
    # that could only have satisfied such a gate is optional, not a blocker.
    satisfied = {g for a, b in vals.items() if b.get("state") == "completed"
                 and not b.get("blocked") for g in gates_for(a)}
    carried, kept_blocked, skipped, optional = [], [], [], []
    for plat, blk in vals.items():
        if blk.get("blocked"):
            kept_blocked.append(plat)
        elif blk.get("state") == "completed":
            blk["validated_sha"] = new_sha
            blk["updated_at"] = now_iso()
            carried.append(plat)
        elif gates_for(plat) <= satisfied:
            optional.append((plat, blk.get("state")))
        else:
            skipped.append((plat, blk.get("state")))
    save_status(name, obj)
    return (True, {"carried": carried, "kept_blocked": kept_blocked,
                   "skipped": skipped, "optional": optional})


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
    obj = load_status(name)

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

    vals = validations(obj)
    head = obj.get("head_sha")
    blocking, nonviable = [], []
    waivers = obj.get("waivers") or {}

    for gate in REQUIRED_GATES:
        archs = [a for a in vals if gate in gates_for(a)]
        # Satisfied by evidence: completed, and against the current head if we have
        # one -- a validation of superseded content proves nothing about this port.
        if any(vals[a].get("state") == "completed"
               and (not head or vals[a].get("validated_sha") == head) for a in archs):
            nonviable.extend(a for a in archs
                             if vals[a].get("state") != "completed" and vals[a].get("blocked"))
            continue
        # Satisfied by an approved waiver.
        w = waivers.get(gate)
        if w and w.get("approved_by") and gate in WAIVABLE_GATES:
            nonviable.append(f"{gate} (waived by {w['approved_by']})")
            continue
        if w and not w.get("approved_by"):
            blocking.append((gate, "waiver suggested but not approved by a maintainer"))
            continue
        candidates = [(a, vals[a].get("state")) for a in archs
                      if not vals[a].get("blocked")]
        if candidates:
            blocking.extend(candidates)   # completing any ONE clears the gate
        else:
            blocking.append((gate, "no viable arch can satisfy this gate"))
            nonviable.extend(a for a in archs if vals[a].get("blocked"))

    # Licence gate. Here rather than only in the publisher so that EVERY route to an
    # upstream PR passes it, whoever is doing the opening.
    lic_ok, lic_why = license_gate(name)
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
    return (not deduped, deduped, sorted(set(nonviable)))


def record_tokens(name, tokens, source=None):
    """Append a token-usage record to projects/<name>/stats.jsonl. `tokens` is an
    agent/subagent output-token count for a unit of work (e.g. from a task
    completion notification); `source` labels what produced it. statlib sums these
    as the project's token total. Approximate by nature (output tokens, not full
    context) -- statlib always reports tokens as approx=True."""
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
    notes.md, plan.md, and stats.jsonl (whichever exist), plus any extra_paths.
    Agents call
    this for every state transition so the per-phase telemetry in stats.jsonl
    (compile/test wall-clock etc., written by timeit.sh -- the README/blog metrics)
    is persisted WITH the transition and never accumulates uncommitted in the
    shared working tree. Prefer this over commit_and_push for project transitions."""
    paths = [f"projects/{name}/{fn}" for fn in
             ("status.json", "notes.md", "plan.md", "stats.jsonl")
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

    s = sub.add_parser("scaffold", help="create projects/<name>/{status,upstream}.json")
    s.add_argument("full_name")
    s.add_argument("--url")
    s.add_argument("--branch", default="main")
    s.add_argument("--ext", default="unknown", choices=["cmake", "torch-extension", "unknown"])
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

    s = sub.add_parser("set-blocked")
    s.add_argument("name")
    s.add_argument("platform", help="<os>-<gfx>, e.g. linux-gfx90a")
    s.add_argument("reason")

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

    s = sub.add_parser("record-license-clearance",
                       help="record a person's decision to allow a tier 3/4 project upstream")
    s.add_argument("name")
    s.add_argument("--by", required=True, help="who approved it (never an agent)")
    s.add_argument("--note", help="scope or conditions of the approval")

    s = sub.add_parser("license-gate", help="may this port be offered upstream on licence grounds?")
    s.add_argument("name")

    s = sub.add_parser("set-review-pr", help="record the fork review PR where the port is approved")
    s.add_argument("name")
    s.add_argument("url")

    s = sub.add_parser("record-pr-approval",
                       help="snapshot the approval standing on the fork review PR")
    s.add_argument("name")
    s.add_argument("--review-pr", help="review PR URL, if status.json has none yet")

    s = sub.add_parser("pr-approval", help="does a standing approval still cover what we would publish now?")
    s.add_argument("name")
    s.add_argument("--offline", action="store_true",
                   help="check the recorded snapshot only, without re-reading GitHub")

    sub.add_parser("pr-candidates",
                   help="list projects whose upstream PR is ready to open (honors recorded dispositions; "
                        "use this instead of scanning raw state==completed)")

    s = sub.add_parser("audit-clean", help="report forks with uncommitted tracked source/build edits (integrity-gap fingerprint)")
    s.add_argument("name", nargs="?", default=None, help="one project, or omit to scan every fork")

    s = sub.add_parser("set-pr-open", help="record PR metadata after creating the upstream PR")
    s.add_argument("name")
    s.add_argument("pr_url")
    s.add_argument("pr_number", type=int)

    s = sub.add_parser("set-pr-merged", help="record that the upstream PR merged")
    s.add_argument("name")

    s = sub.add_parser("set-pr-closed", help="record that the upstream PR closed without merging")
    s.add_argument("name")
    s.add_argument("--note", help="why it closed (withdrawn, rejected, superseded)")

    s = sub.add_parser("squash-carry-forward",
                       help="advance head to a tree-identical PR-prep squash, carrying validated platforms forward (no revalidation)")
    s.add_argument("name")
    s.add_argument("new_sha")

    sub.add_parser("unblock-followers")
    rf = sub.add_parser("release-forks",
                        help="advance awaiting-fork projects whose fork now exists")
    rf.add_argument("--dry-run", action="store_true")
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

    s = sub.add_parser("set-deps", help="record the MOAT projects a project depends on")
    s.add_argument("name")
    s.add_argument("deps", nargs="*")

    sub.add_parser("deps", help="print inter-project dependencies and what is blocked on them")

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
        set_blocked(args.name, args.platform, True, args.reason)
        print(f"{args.name}/{args.platform} blocked: {args.reason}")
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
        print(f"committed projects/{args.name} (status/notes/plan/stats)" if ok else "(nothing to commit)")
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
            if info["skipped"]:
                msg += f"; SKIPPED actionable {info['skipped']} (should not squash yet)"
            print(msg)
    elif args.cmd == "pr-ready":
        ready, blocking, nonviable = pr_ready(args.name)
        print(f"{args.name}: PR-ready={ready}")
        if blocking:
            print("  BLOCKING (every required gate needs ONE completed arch at head_sha, "
                  "or an approved waiver; fork must be clean): "
                  + ", ".join(f"{p}={s}" for p, s in blocking))
        if nonviable:
            print("  non-viable (does not block; scope the PR body): " + ", ".join(nonviable))
    elif args.cmd == "record-license-clearance":
        c = record_license_clearance(args.name, args.by, args.note)
        print(f"{args.name}: tier {c['tier']} cleared upstream by {c['approved_by']}")
    elif args.cmd == "license-gate":
        ok, why = license_gate(args.name)
        print(f"{args.name}: license-ok={ok} ({why})")
        return 0 if ok else 1
    elif args.cmd == "set-review-pr":
        set_review_pr(args.name, args.url)
        print(f"{args.name}: review PR -> {args.url}")
    elif args.cmd == "record-pr-approval":
        a = record_pr_approval(args.name, args.review_pr)
        print(f"{args.name}: approved by {a['approved_by']} for "
              f"{(a.get('head_sha') or '?')[:8]} on {a['review_pr']}")
    elif args.cmd == "pr-approval":
        ok, why = pr_approval_valid(args.name, live=not args.offline)
        print(f"{args.name}: approval-valid={ok} ({why})")
        return 0 if ok else 1
    elif args.cmd == "pr-candidates":
        names = [d.name for d in sorted(PROJECTS.iterdir()) if (d / "status.json").exists()]
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
        print("NOTE: license vetting (non-commercial/no-license = DO-NOT-PR) and RAPIDS "
              "ROCm-DS ownership are separate MANUAL gates this tool cannot see -- "
              "vet each before opening.")
    elif args.cmd == "audit-clean":
        names = [args.name] if args.name else [d.name for d in sorted(PROJECTS.iterdir())
                                               if (d / "status.json").exists()]
        real_gap = False
        for n in names:
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
        else:
            print("OK: no fork with a completed/pr platform has uncommitted source edits" +
                  (f" ({args.name})" if args.name else ""))
    elif args.cmd == "set-pr-open":
        set_pr_open(args.name, args.pr_url, args.pr_number)
        print(f"{args.name}: PR opened -> {args.pr_url}")
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
        if not rel:
            print("release-forks: nothing waiting on a fork")
        return 0
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
    elif args.cmd == "unblock-followers":
        changed = unblock_all_followers()
        print(" ".join(changed) if changed else "(none)")
    elif args.cmd == "validate":
        load_status(args.name)
        print(f"{args.name} status.json valid")
    elif args.cmd == "show":
        _print_json(load_status(args.name))
    elif args.cmd == "set-deps":
        obj = load_status(args.name)
        obj["depends_on"] = list(args.deps)
        obj["updated_at"] = now_iso()
        save_status(args.name, obj)
        print(f"{args.name} depends_on = {args.deps}")
    elif args.cmd == "deps":
        any_dep = False
        for d in sorted(PROJECTS.iterdir()):
            if not (d / "status.json").exists():
                continue
            try:
                obj = load_status(d.name)
            except (ValueError, json.JSONDecodeError):
                continue
            deps = obj.get("depends_on", [])
            if not deps:
                continue
            any_dep = True
            unmet = unmet_deps(obj)
            mark = "READY (deps complete)" if not unmet else ("WAITING on " + ", ".join(unmet))
            print(f"{d.name}: depends_on={deps} -> {mark}")
        if not any_dep:
            print("(no inter-project dependencies recorded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
