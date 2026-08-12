#!/usr/bin/env python3
"""Reconcile recorded upstream-PR state with reality.

A protected trunk creates a bookkeeping gap. A project's own PR merges when the port
is validated and the upstream PR is open; after that the branch is gone and nothing
is watching. When the upstream PR later merges, closes, or gets changes requested,
there is no longer any writable place to record it -- so the record rots and the
README keeps showing a PR as open weeks after it landed.

The drift is not hypothetical: on any given sweep a handful of records disagree with
GitHub, and the ones that matter are contributions that MERGED upstream without
anyone here noticing.

It PROPOSES, never applies. pytorch3d is why: its PR reads CLOSED with mergedAt
null because pytorchbot lands via Meta's internal flow and never sets the merged
flag, so the record saying `upstream-landed` is correct and GitHub's field is
misleading. A tool that auto-overwrote would silently downgrade a landed
contribution on the strength of a field that lies for that repo. So it writes a
branch and opens a PR, and a human decides -- the same path everything else takes.

    python3 utils/upstream.py --dry-run          # report PR drift, change nothing
    python3 utils/upstream.py --apply            # update the record-sync branch + PR
    python3 utils/upstream.py --forks            # report projects awaiting a fork
    python3 utils/upstream.py --forks --apply    # release the ones whose fork exists
    python3 utils/upstream.py --approvals        # report approvals overtaken by a push or edit
    python3 utils/upstream.py --approvals --apply # dismiss them and re-request review
    python3 utils/upstream.py --review           # report ports needing a review PR
    python3 utils/upstream.py --publish          # report approved ports ready to submit
    python3 utils/upstream.py --publish --apply  # open the upstream PRs
    python3 utils/upstream.py --fix-review       # staged fix rounds needing a review PR
    python3 utils/upstream.py --merge-fix        # approved fix rounds ready to merge
    python3 utils/upstream.py --merge-fix --apply # fast-forward the open PR to the approved tip

Record maintenance and one publishing step. None of it does any porting -- that needs a
GPU host and a session. These keep the record true, tell someone, and send an approved
port on its way.

Running it repeatedly is safe. There is ONE branch, regenerated from the trunk each
run and force-pushed, so an unmerged PR is updated in place rather than joined by a
second one carrying overlapping changes.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TODAY = subprocess.run(["date", "-u", "+%Y-%m-%d"],
                       capture_output=True, text=True).stdout.strip()
# One stable branch, deliberately not date-stamped: see publish().
# Named for what it carries, not for who pushes it: every run of this is a person
# in a session, so there is no bot identity for a reader to recognise.
BRANCH = "record-sync"

# What every pull request we send says about where it came from, appended to the body
# when the review PR is opened so it is part of what gets approved and part of what
# gets published verbatim. Three things a maintainer receiving an unsolicited PR is
# owed: that a machine wrote it, that a person read it before it was sent, and how to
# make it stop. It is added by the tool rather than left to whoever writes the body,
# because a disclosure that depends on remembering is one that eventually goes missing
# from the one PR where it mattered.
#
# SUBMISSION_MARKER is what the publish gate looks for. Keep it a stable substring of
# the note: editing the wording is fine, editing the marker orphans every review PR
# already approved.
#
# EDITING THE NOTE: the jargon checker cannot see the paragraph carrying the repo URL.
# `MOAT` is a jargon term, and config/jargon.toml allows any line matching
# `https?://\S*moat` or `github\.com/\S+` so ordinary links do not trip it -- but the
# allowance skips the WHOLE line, and a GitHub body is written one line per paragraph,
# so everything beside that URL goes unscanned. Dropping "the lead platform validated at
# head_sha" into it yields zero hits; the same words on their own line yield three. So
# check any wording change here by eye: this is the one paragraph the gate below cannot
# check for you.
SUBMISSION_MARKER = "prepared with the help of an AI assistant"
SUBMISSION_NOTE = (
    "---\n\n"
    "This pull request was prepared with the help of an AI assistant acting as a "
    "coding agent and was "
    "read and approved by a person before it was opened. It comes from an ongoing "
    "effort to add AMD GPU support to widely used CUDA projects, one repository at a "
    "time: https://github.com/AMD-Ecosystem/moat -- that repository describes how the "
    "work is done and what a person checks before anything is submitted.\n\n"
    "If you would rather not receive pull requests from this effort, say so here or "
    "open an issue at https://github.com/AMD-Ecosystem/moat/issues/new/choose and we "
    "will close this and stop. That can cover this repository alone or everything you "
    "own, whichever you prefer.")


def with_submission_note(body):
    """The body as it will be published. Idempotent, so a body that already carries
    the note (a maintainer-requested edit, a re-opened review PR) is left alone."""
    if SUBMISSION_MARKER in body:
        return body
    return body.rstrip() + "\n\n" + SUBMISSION_NOTE + "\n"


# GitHub PR state -> the project-level pr_state it implies. A closed PR says
# nothing about whether the port itself is good, so it is reported for a human
# rather than applied.
IMPLIES = {"MERGED": "merged", "OPEN": "open", "CLOSED": "closed"}

# Repos whose PR state cannot be trusted, with the reason. Being explicit beats a
# heuristic: the tool should say "I am ignoring this and here is why", not quietly
# special-case it.
UNTRUSTED = {
    "facebookresearch/pytorch3d":
        "pytorchbot lands via Meta's internal flow and never sets the merged flag, "
        "so a landed PR reads CLOSED with mergedAt null",
    "pytorch/pytorch": "same pytorchbot merge flow",
}


def all_records():
    """(name, record, where) for every project across refs, not just this checkout.

    Everything below used to glob the local tree. That was equivalent while every
    project's folder sat on the trunk, and stops being so the moment one moves to its
    own branch: a sweep run from any single branch would silently skip the rest, and a
    skipped project is indistinguishable from one with nothing to report."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    for name in sorted(moatlib.all_projects()):
        obj, where = moatlib.project_record(name)
        if obj is not None:
            yield name, obj, where


def gh_json(args):
    r = subprocess.run(["gh"] + args, capture_output=True, text=True, timeout=90)
    if r.returncode:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def recorded():
    """Every project with an upstream PR recorded, and what we think its state is."""
    out = []
    for name, d, _where in all_records():
        pr = d.get("pr_url") or next(
            (b.get("pr_url") for b in (d.get("platforms") or {}).values() if b.get("pr_url")), None)
        if not pr:
            continue
        m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", pr)
        if not m:
            continue
        ours = d.get("pr_state") or ("merged" if d.get("pr_merged_at") else None)
        out.append({"name": name, "repo": m.group(1), "num": m.group(2),
                    "url": pr, "ours": ours, "published": d.get("published_sha")})
    return out


def poll(rows):
    """Compare each record against GitHub.

    Returns (drift, unreviewed, headdrift, errors). headdrift is an OPEN PR whose
    head no longer matches the recorded published_sha: a push that did not come
    through the fix flow's merge. The usual cause is a maintainer editing our
    branch (allowed on PRs with maintainer-edit enabled), which was done on
    purpose by a person -- so it is never auto-absorbed: review the commit(s) and
    recommend a course of action for a human to decide."""
    drift, unreviewed, headdrift, errors = [], [], [], []
    for r in rows:
        d = gh_json(["pr", "view", r["num"], "--repo", r["repo"], "--json",
                     "state,mergedAt,reviewDecision,updatedAt,headRefOid"])
        if d is None:
            errors.append({**r, "why": "lookup failed"})
            continue
        r["real"] = d.get("state")
        r["merged_at"] = d.get("mergedAt")
        r["review"] = d.get("reviewDecision")
        if r["repo"] in UNTRUSTED:
            r["skipped"] = UNTRUSTED[r["repo"]]
            continue
        implied = IMPLIES.get(r["real"])
        if implied and implied != r["ours"]:
            drift.append({**r, "closed": True} if implied == "closed" else r)
        # An open PR with changes requested is work waiting on us, not drift.
        if r["real"] == "OPEN" and r["review"] == "CHANGES_REQUESTED":
            unreviewed.append(r)
        if (r["real"] == "OPEN" and r.get("published") and d.get("headRefOid")
                and d["headRefOid"] != r["published"]):
            headdrift.append({**r, "head": d["headRefOid"]})
    return drift, unreviewed, headdrift, errors


OURS = {"jeffdaily"}                     # accounts that speak for this project
# A CI bot having the last word means nobody is waiting on us. Matching the [bot]
# suffix catches most of them; the rest post under ordinary accounts and are listed.
BOTS = {"codacy-production", "codecov-commenter", "sonarcloud", "coderabbitai",
        "netlify", "vercel", "deepsource-autofix", "restyled-io"}


def _is_bot(login):
    return not login or login.endswith("[bot]") or login.lower() in BOTS


def attention(rows, quiet_weeks=6):
    """Open upstream PRs where something has changed that needs a person.

    Separate from the drift sweep on purpose. Drift is bookkeeping -- our record
    disagrees with GitHub and can be corrected mechanically. This is the opposite:
    GitHub is fine, and somebody is waiting on us. It reads and reports only; the
    reply is drafted and a person posts it.

    Three ways a PR ends up here, in descending order of how much it is owed:
    a maintainer asked for changes; a maintainer's comment is the last word in the
    thread; or nobody has said anything for weeks and it may need a nudge from a
    person who can find the right reviewer.
    """
    import datetime
    out = []
    for r in rows:
        d = gh_json(["pr", "view", r["num"], "--repo", r["repo"], "--json",
                     "state,reviewDecision,comments,updatedAt,title"])
        if d is None or d.get("state") != "OPEN":
            continue
        comments = d.get("comments") or []
        last = comments[-1] if comments else None
        last_by = ((last or {}).get("author") or {}).get("login")
        why = None
        if d.get("reviewDecision") == "CHANGES_REQUESTED":
            why = "changes requested"
        elif last_by and last_by not in OURS and not _is_bot(last_by):
            why = f"last word is {last_by}'s"
        else:
            updated = (d.get("updatedAt") or "")[:10]
            if updated:
                age = (datetime.date.fromisoformat(TODAY) - datetime.date.fromisoformat(updated)).days
                if age >= quiet_weeks * 7:
                    why = f"quiet for {age // 7} weeks"
        if why:
            out.append({**r, "why": why, "title": d.get("title", "")})
    return out


def report_attention(rows, today):
    found = attention(rows)
    print(f"upstream: {len(found)} of {len(rows)} recorded PR(s) need a person\n")
    # Most owed first: an explicit request outranks an unanswered comment, which
    # outranks silence.
    rank = {"changes requested": 0}
    def key(r):
        return (rank.get(r["why"], 1 if r["why"].startswith("last word") else 2), r["name"])
    for r in sorted(found, key=key):
        print(f"  {r['why']:24} {r['name']:26} {r['repo']}#{r['num']}")
        print(f"  {'':24} {r['url']}")
    if found:
        print("\n  Read the thread before replying. Draft it, show it, wait.")
    return 0


def apply_one(r):
    """Record a merge. Only MERGED is applied automatically: it is unambiguous and
    additive. A CLOSED PR needs a human -- withdrawn on licence grounds, rejected by
    a maintainer, and superseded are different outcomes that look identical here.

    Returns where the record landed -- "local" for a working-tree edit the record-sync
    PR below then carries to the protected trunk, "branch" for one written straight to
    the project's own port branch -- or False if there was nothing to apply. The two
    are not the same afterwards: a branch write is already published, and treating it
    as a pending working-tree edit is how `publish` came to report nothing to do while
    having just recorded several merges."""
    if r["real"] != "MERGED":
        return False
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    local = moatlib.status_path(r["name"]).exists()
    try:
        moatlib.set_pr_merged(r["name"])
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        print(f"  COULD NOT record {r['name']} as merged: {e}")
        return False
    return "local" if local else "branch"


def fork_poll(apply=False, stale_weeks=3):
    """Release projects whose fork has appeared, and flag ones waiting too long.

    A project sits in `awaiting-fork` until someone with the rights creates the fork;
    that act is the decision to take it up. Nobody should have to notice by hand, so
    this advances the state and prints what it found.

    Detection AND the write are `moatlib.release_awaiting_fork` -- the one
    implementation. This used to run its own fork-existence poll first and only
    delegated when its poll found something, so the two polls could disagree and the
    disagreement decided whether anything advanced. Now the same sweep that reports
    is the sweep that writes; only the ones it released are re-derived here for the
    WAITING report.

    State lives on the project's own `port/<name>` branch, not the trunk, so the
    release pushes directly. It does NOT start any work: porting needs a GPU host and
    a session. This only makes the project eligible and tells someone.
    """
    import datetime
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    released = [{"name": n, "slug": s}
                for n, s in moatlib.release_awaiting_fork(dry_run=not apply)]
    got = {r["name"] for r in released}
    waiting = []
    for name, d, _where in all_records():
        if name in got or moatlib.project_stage(d) != "awaiting-fork":
            continue
        slug = (d.get("fork_url") or f"https://github.com/AMD-Ecosystem/{name}") \
            .replace("https://github.com/", "")
        waiting.append({"name": name, "slug": slug, "since": d.get("updated_at") or ""})

    print(f"fork-poll: {len(released)} released, {len(waiting)} still waiting\n")
    for r in released:
        verb = "ADVANCED" if apply else "RELEASED"
        note = "-> screened" if apply else "fork exists:"
        print(f"  {verb:10} {r['name']:26} {note} {r['slug']}")
    now = datetime.datetime.now(datetime.timezone.utc)
    for w in waiting:
        old = ""
        if w["since"]:
            try:
                age = (now - datetime.datetime.fromisoformat(
                    w["since"].replace("Z", "+00:00"))).days
                if age >= stale_weeks * 7:
                    old = f"  <-- waiting {age} days"
            except ValueError:
                pass
        print(f"  WAITING    {w['name']:26} no fork at {w['slug']}{old}")
    return released, waiting


# An approval covers the code, title and body that were on screen when it was given.
# A push or an edit afterwards means the reviewer is now recorded as approving
# something they never saw -- and GitHub keeps showing "Approved" through both, so
# nothing surfaces it. Only `stale-*` is actionable: `withdrawn` means the approval is
# already gone (often because a previous run dismissed it) and re-asking would nag,
# and `record-mismatch` means our own file disagrees with GitHub, which is a human's
# problem and never a reason to touch someone's review.
ACTIONABLE_APPROVAL = {"stale-commits", "stale-content"}


def approval_drift():
    """Projects whose approval no longer covers their review PR.

    Looks for the approval on GitHub, not only for a snapshot in our files -- the
    same principle publishable() settled. The drift window this report exists for is
    an approval clicked and then a push landing before anyone submits, and
    record_pr_approval normally runs at publish time, so that window has no snapshot
    by construction. A project with a snapshot gets the full provenance check on top
    (pr_approval_status); one without is judged from the PR alone
    (approval_currency), and a port still awaiting its first approval is not drift
    and is not reported."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    rows = []
    for name, d, _where in all_records():
        if d.get("pr_state"):
            continue          # already published
        url = (d.get("pr_approval") or {}).get("review_pr") or d.get("review_pr")
        if not url:
            continue          # no review PR for an approval to drift from
        if d.get("pr_approval"):
            code, why = moatlib.pr_approval_status(name, live=True)
            if code != "ok":
                rows.append({"name": name, "code": code, "why": why, "url": url})
            continue
        pr = moatlib.fetch_review_pr(url)
        if pr is None:
            # An outage is not an answer; see publishable().
            rows.append({"name": name, "code": "unreachable",
                         "why": f"could not reach the review PR at {url}", "url": url})
            continue
        code, why = moatlib.approval_currency(pr)
        if code != "ok" and code != "withdrawn":
            rows.append({"name": name, "code": code, "why": why, "url": url})
    return rows


def refresh_approval(r):
    """Surface the overtaken approval on its review PR and ask for a fresh one.

    A plain APPROVED review is dismissed, so the PR stops displaying "Approved" for
    content nobody approved. An approval given as a `/moat approve` comment -- the
    form every self-authored review PR uses, since GitHub greys out the button for
    the author -- has no green state to mislead anyone and GitHub's dismissal
    endpoint takes only APPROVED review objects, so for those the stale notice is a
    comment on the PR. The submission gate refuses either way; this is about what a
    human reading the page is told. No review re-request: on a self-authored PR the
    approver IS the author, and GitHub refuses a review request naming the author.

    This is our own fork, inside the autonomy boundary -- unlike anything upstream,
    which always needs its own explicit yes."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    pr = moatlib.fetch_review_pr(r["url"])
    if pr is None:
        return False
    review = moatlib._approving_review(pr)
    if review is None:
        return False
    slug, num = pr["slug"], pr["number"]
    msg = (f"This approval is stale: {r['why']}.\n\n"
           "The approval covered the code, title and body as they stood when it was "
           "given, so it no longer describes what would be submitted upstream. "
           "Nothing is published while it is stale. Please re-approve if the change "
           "still looks right.")
    if review.get("id") and review.get("state") == "APPROVED":
        return gh_json(["api", "-X", "PUT",
                        f"repos/{slug}/pulls/{num}/reviews/{review['id']}/dismissals",
                        "-f", f"message={msg}", "-f", "event=DISMISS"]) is not None
    res = subprocess.run(["gh", "pr", "comment", r["url"], "--body", msg],
                         capture_output=True, text=True, timeout=90)
    return res.returncode == 0


def report_approvals(apply):
    rows = approval_drift()
    actionable = [r for r in rows if r["code"] in ACTIONABLE_APPROVAL]
    print(f"upstream: {len(rows)} approval(s) no longer current, "
          f"{len(actionable)} actionable\n")
    for r in rows:
        tag = "STALE" if r["code"] in ACTIONABLE_APPROVAL else r["code"].upper()
        print(f"  {tag:16} {r['name']:26} {r['why']}")
    if not rows:
        return 0
    if not apply:
        if actionable:
            print(f"\n  --apply marks {len(actionable)} overtaken approval(s) stale on "
                  f"their review PR (dismissed, or a comment where nothing is "
                  f"dismissible) and asks for a fresh one.")
        return 0
    for r in actionable:
        done = refresh_approval(r)
        print(f"  {'marked stale, fresh approval requested' if done else 'COULD NOT mark stale'} {r['name']}")
    return 0


def publishable():
    """Ports whose review PR carries a standing approval and are ready to submit.

    Returns (ready, unreachable). This is what makes clicking Approve mean something.
    It looks for the approval on GitHub rather than for a snapshot in our files,
    because the click is the whole signal -- a project nobody has run
    record-pr-approval on yet is exactly the case that needs finding.

    A review PR we cannot READ is returned separately rather than skipped, because
    those two answers are not the same and this printed the wrong one for both. "Not
    approved" is a fact about the port; "could not reach GitHub" is a fact about the
    network, and folding it into a count of zero says nothing is waiting when a
    finished, approved port may be. It happened on an ordinary day: the GraphQL
    endpoint timed out while REST kept working, `fetch_review_pr` opens with a
    `gh pr view --json` call, and `--publish` reported "0 approved port(s) awaiting
    submission" while marian-dev sat approved -- with orient.sh, which greps this
    output for READY, going quiet along with it.

    The rest of this file already settled the principle three times over: an
    unreachable review PR refuses rather than passing (pr_approval_status), an
    unreachable remote is a warning and not a clean bill (existing_claim), and
    "nothing released" is not "nothing waiting" (release-forks). This was the one
    place still reading an outage as an answer."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    out, unreachable, objected = [], [], []
    for name, d, _where in all_records():
        url = (d.get("pr_approval") or {}).get("review_pr") or d.get("review_pr")
        if not url or d.get("pr_state"):
            continue                      # no review PR, or already submitted
        pr = moatlib.fetch_review_pr(url)
        if pr is None:
            unreachable.append({"name": name, "url": url})
            continue
        if moatlib._approving_review(pr) is None:
            # Not approved is a real answer, but a standing /moat objection or a
            # command the gate did not recognize is also WORK -- route to the
            # porter, or fix the typo -- and skipping it silently hid both.
            blockers, _notes = moatlib.moat_command_audit(pr)
            if blockers:
                objected.append({"name": name, "url": url, "why": blockers})
            continue
        out.append({"name": name, "url": url, "pr": pr,
                    "title": pr.get("title") or "", "body": pr.get("body") or ""})
    return out, unreachable, objected


def review_candidates():
    """Ports whose gates are met and which have no review PR yet.

    The gap this fills: everything after a review PR existed was automated -- fetch
    it, snapshot the approval, verify it still covers the content, publish -- and
    nothing opened one. Twenty-eight ports sat PR-ready with none open."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    out = []
    for name in sorted(moatlib.all_projects()):
        d, _where = moatlib.project_record(name)
        if d is None or d.get("review_pr") or d.get("pr_state") or d.get("pr_url"):
            continue
        ready, blocking, _ = moatlib.pr_ready(name)
        if not ready:
            continue
        fork = (d.get("fork_url") or "").replace("https://github.com/", "")
        branch = d.get("fork_branch") or moatlib.PORT_BRANCH
        base = d.get("fork_default_branch") or "main"
        if not fork:
            continue
        # The port branch has to exist and differ from the base. pr_ready never
        # checked this, so a fork with no port at all could present as ready.
        cmp = gh_json(["api", f"repos/{fork}/compare/{base}...{branch}",
                       "--jq", "{commits:.total_commits,files:(.files|length)}"])
        if not cmp:
            out.append({"name": name, "fork": fork, "branch": branch, "base": base,
                        "problem": f"cannot compare {base}...{branch} on the fork"})
            continue
        if not cmp.get("commits"):
            out.append({"name": name, "fork": fork, "branch": branch, "base": base,
                        "problem": f"{branch} has no commits over {base}"})
            continue
        out.append({"name": name, "fork": fork, "branch": branch, "base": base,
                    "commits": cmp["commits"], "files": cmp["files"], "problem": None})
    return out


def open_review_pr(row, title, body, apply=False):
    """Open the review PR on our own fork. The title and body ARE the upstream PR's,
    so they are checked for in-house vocabulary before anyone sees them -- this is
    the last point where that is cheap to fix."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    import jargon
    import prose

    # Attached before the checks below, not after, so the note is scanned like the
    # rest of the body and shown in the --review preview exactly as it will publish.
    body = with_submission_note(body)

    terms, allow = jargon.load()
    hits = (jargon.scan_text(title, "title", terms, allow)
            + jargon.scan_text(body, "body", terms, allow))
    # The BRANCH too, not just the title and body. Every commit on it ships upstream
    # whichever round wrote it, and the porter's own check is an instruction nothing
    # verified: faster-gaussian-splatting carried "Strategy B (torch hipify)" in the
    # commit its branch starts from through a full review and a changes-requested
    # round, because each round scanned only what that round added. This is the last
    # point where fixing it is cheap -- after the review PR, a rewrite costs every
    # architecture its validation.
    try:
        repo, commits, diff = jargon.port_range(row["name"])
        hits += jargon.scan_commits(repo, commits, terms, allow)
        hits += jargon.scan_diff(repo, diff, terms, allow)
    except ValueError as e:
        # Never silently skip. A gate that cannot run is not a gate that passed.
        return ("jargon", f"cannot check the branch for in-house vocabulary: {e}")
    if hits:
        return ("jargon", "in-house vocabulary an external maintainer will not know: "
                + ", ".join(sorted({h[2] for h in hits})))
    wrapped = prose.check(title, "title") + prose.check(body, "body")
    if wrapped:
        return ("wrapped", wrapped[0])
    if row.get("problem"):
        return ("blocked", row["problem"])
    if not apply:
        return ("would-open",
                f"{row['fork']}: {row['branch']} -> {row['base']} "
                f"({row['commits']} commits, {row['files']} files)\n\n{title}\n\n{body}")
    r = subprocess.run(["gh", "pr", "create", "--repo", row["fork"],
                        "--head", row["branch"], "--base", row["base"],
                        "--title", title, "--body", body],
                       capture_output=True, text=True, timeout=90)
    if r.returncode:
        return ("error", (r.stderr or r.stdout).strip())
    url = r.stdout.strip().splitlines()[-1]
    moatlib.set_review_pr(row["name"], url)
    # How to approve goes in a COMMENT, never in the body: the body is republished
    # verbatim as the upstream pull request, and an external maintainer has no use for
    # our approval command.
    subprocess.run(
        ["gh", "pr", "comment", url, "--body",
         f"To approve this port, leave a comment containing this line by "
         f"itself:\n\n```\n{moatlib.APPROVE_COMMAND}\n```\n\n"
         f"To send it back to the porter instead:\n\n"
         f"```\n{moatlib.CHANGES_COMMAND}\n```\n\n"
         f"Both are commands because GitHub greys out the Approve and Request Changes "
         f"buttons for a pull request's author, and this PR was opened on your "
         f"credentials. Your latest command is the one that stands, and a command "
         f"quoted in a code fence -- like the two above -- is ignored.\n\n"
         f"Either box works: a review comment (*Review changes -> Comment*, or "
         f"`gh pr review {url} --comment --body '{moatlib.APPROVE_COMMAND}'`) or an "
         f"ordinary conversation comment. Prefer the review form -- it records which "
         f"commit you were looking at, which is what proves the approval covers this "
         f"code and not an earlier push; a conversation comment counts too, judged by "
         f"its time against the branch tip.\n\n"
         f"The title and body above are what gets opened upstream, verbatim, so approving "
         f"here approves all three: the code, the title and the body. Anything pushed "
         f"afterwards, or any edit to the title or body, voids it and needs a fresh one."],
        capture_output=True, text=True, timeout=90)
    return ("opened", url)


def publish_blockers(name, row):
    """Everything that must hold before a port is submitted upstream, as a list of
    reasons it must not be. Checked at publish time rather than trusted from earlier:
    a gate that passed when the reviewer looked is not evidence about now."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    import jargon
    import prose

    bad = []
    # The approval must still cover this exact content. Judged from the PR itself,
    # not from our snapshot: a port approved five minutes ago has no snapshot yet,
    # and that is the case this whole path exists to catch.
    code, why = moatlib.approval_currency(row["pr"])
    if code != "ok":
        bad.append(f"approval {code}: {why}")
    # Every /moat command on the review PR, judged again at the last gate before
    # anything opens upstream. approval_currency already refuses on these; this
    # repeats the check independently so a drift between the two implementations
    # fails closed rather than publishing over an objection nobody re-read.
    blockers, _notes = moatlib.moat_command_audit(row["pr"])
    bad += blockers
    # Gates, clean fork, no terminal outcome, no PR already open.
    ready, blocking, _ = moatlib.pr_ready(name)
    if not ready:
        bad.append("not PR-ready: " + ", ".join(f"{p}={s}" for p, s in blocking))
    # Licence is checked inside pr_ready, so it is already covered above. Nothing to
    # repeat here: one implementation, every route upstream.
    # The title and body are about to become upstream-visible.
    terms, allow = jargon.load()
    hits = (jargon.scan_text(row["title"], "title", terms, allow)
            + jargon.scan_text(row["body"], "body", terms, allow))
    if hits:
        bad.append("in-house vocabulary in the title/body: "
                   + ", ".join(sorted({h[2] for h in hits})[:4]))
    # The BRANCH again, not just the title and body, and read from the PR itself
    # rather than a local clone so it runs on submission hosts that never built the
    # port. The open-time scan cannot see a commit that landed between the review PR
    # opening and the approval -- the approval is given against the new tip, so the
    # staleness check will not catch it either -- and this is the last gate before
    # that commit ships.
    pr = row["pr"]
    msgs = subprocess.run(
        ["gh", "api", "--paginate",
         f"repos/{pr['slug']}/pulls/{pr['number']}/commits",
         "--jq", ".[].commit.message"],
        capture_output=True, text=True, timeout=90)
    if msgs.returncode:
        bad.append("cannot re-check the branch's commit messages for in-house "
                   "vocabulary (gh api failed) -- a gate that cannot run is not a "
                   "gate that passed")
    else:
        bhits = jargon.scan_text(msgs.stdout, "branch commits", terms, allow)
        diff = subprocess.run(["gh", "pr", "diff", row["url"]],
                              capture_output=True, text=True, timeout=180)
        if diff.returncode:
            bad.append("cannot re-check the branch's added lines for in-house "
                       "vocabulary (gh pr diff failed)")
        else:
            added = "\n".join(l[1:] for l in diff.stdout.splitlines()
                              if l.startswith("+") and not l.startswith("+++"))
            bhits += jargon.scan_text(added, "branch added lines", terms, allow)
        if bhits:
            bad.append("in-house vocabulary on the branch: "
                       + ", ".join(sorted({h[2] for h in bhits})[:4]))
    # The approved title and body are about to be republished verbatim upstream, so
    # they are checked here too rather than trusted from when the review PR was
    # opened -- a maintainer may have asked for an edit, and an edit is where
    # hand-wrapping (and a stray em-dash or miscased ROCm) creeps back in.
    bad += prose.check(row["title"], "title")
    bad += prose.check(row["body"], "body")
    # The disclosure has to survive to the thing that actually gets opened. It is
    # added when the review PR is opened, so its absence here means the body was
    # edited afterwards -- which voids the approval anyway, and this says why.
    if SUBMISSION_MARKER not in (row["body"] or ""):
        bad.append("the body no longer says the change was AI-prepared and "
                   "human-approved, or where to opt out; restore the note from "
                   "upstream.py SUBMISSION_NOTE and get a fresh approval")
    return bad


def open_upstream(name, row):
    """Open the upstream PR with the approved title and body, verbatim.

    The record is resolved across refs like everything else here: a project with a
    standing approval and no PR yet lives on its own `port/<name>` branch by
    construction, so reading the working tree found nothing unless the session
    happened to be standing on that branch."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    d, _where = moatlib.project_record(name)
    if d is None:
        return (None, f"no record for {name} in this checkout or on the refs")
    slug = d["upstream_url"].split("github.com/", 1)[-1]
    fork_owner = d["fork_url"].split("github.com/", 1)[-1].split("/")[0]
    branch = d.get("fork_branch") or moatlib.PORT_BRANCH
    # The fork's default branch is kept a clean mirror of upstream's, so it IS the
    # upstream base. Formerly upstream.json.default_branch, which disagreed with this
    # field on 10 projects until GitHub was asked which was right.
    base = d.get("fork_default_branch") or "main"

    # The one write to a repo we do not own, and the only one MOAT makes without a
    # person at the keyboard. It goes to the gh binary directly, past the PATH guard
    # that refuses foreign writes from a shell: every check that earns this call --
    # the recorded approval re-read from GitHub, the licence gate, every required
    # coverage gate, the jargon scan -- has already run above. Trusted code proves its
    # own case; it does not ask the guard to recognise it. (The guard's first version
    # tried to recognise it by process ancestry, which any enclosing shell's command
    # line could satisfy, and two comments reached a live upstream issue as a result.)
    import gh_guard
    real = gh_guard.real_gh()
    if real is None:
        return (None, "gh is not installed")
    r = subprocess.run(
        [real, "pr", "create", "--repo", slug, "--base", base,
         "--head", f"{fork_owner}:{branch}", "--title", row["title"],
         "--body", row["body"]],
        capture_output=True, text=True, cwd=str(REPO))
    if r.returncode:
        return (None, (r.stderr or r.stdout).strip().splitlines()[-1][:200])
    url = next((l.strip() for l in r.stdout.splitlines()
                if "github.com" in l and "/pull/" in l), "")
    num = url.rstrip("/").rsplit("/", 1)[-1]
    # Called rather than shelled out to, so a record that will not write is reported
    # here instead of vanishing into a subprocess nobody read the exit code of. The PR
    # is already open at this point, so failing to record it is worth saying loudly.
    try:
        moatlib.set_pr_open(name, url, num)
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        return (url, f"opened {url} but could NOT record it: {e}")

    # The review PR has done its job -- the change it was reviewing is now in front of
    # the maintainers. Close it pointing at where the conversation continues, rather
    # than leaving an approved PR open forever looking like outstanding work. Closing
    # keeps the review thread readable; only the branch it reviewed matters now, and
    # that branch is the upstream PR's head.
    if url:
        subprocess.run(
            ["gh", "pr", "close", row["url"], "--comment",
             f"Submitted upstream as {url} with the title and body approved here. "
             f"Closing this review PR; the discussion continues on the upstream one."],
            capture_output=True, text=True, cwd=str(REPO))
    return (url, None)


def report_publish(apply):
    rows, unreachable, objected = publishable()
    print(f"upstream: {len(rows)} approved port(s) awaiting submission"
          + (f", {len(objected)} with a standing objection" if objected else "")
          + (f", {len(unreachable)} review PR(s) UNREACHABLE" if unreachable else "")
          + "\n")
    for r in objected:
        print(f"  OBJECTED   {r['name']:26} {r['why'][0][:78]}")
        for b in r["why"][1:]:
            print(f"             {'':26} {b[:78]}")
    if objected:
        print(f"  a standing objection routes the port back to the porter; publish "
              f"stays closed until the objector posts {'/moat approve'!r}\n")
    ready, held = [], []
    for r in rows:
        bad = publish_blockers(r["name"], r)
        (held if bad else ready).append({**r, "blockers": bad})
    clone_less = [r for r in ready
                  if not (REPO / "projects" / r["name"] / "src").is_dir()]
    for r in ready:
        print(f"  READY      {r['name']:26} \"{r['title'][:58]}\"")
    if clone_less:
        # pr_ready's cleanliness gate has nothing to judge without the clone, and
        # saying nothing read as "checked and clean" on submission hosts that had
        # never built the port. The check binds where the validations ran.
        print(f"  note: no local fork clone for "
              f"{', '.join(r['name'] for r in clone_less)} -- fork cleanliness was "
              f"not re-checked here, only on the hosts that validated")
    for r in held:
        print(f"  HELD       {r['name']:26} {r['blockers'][0][:78]}")
        for b in r["blockers"][1:]:
            print(f"             {'':26} {b[:78]}")
    # Printed even with --apply, and before anything is opened: a run that could not
    # read some review PRs has not seen the whole picture, and saying so is the point.
    for r in unreachable:
        print(f"  UNREACHABLE {r['name']:25} could not read {r['url']}")
    if unreachable:
        print(f"\n  {len(unreachable)} review PR(s) could not be read, so their approval "
              f"state is UNKNOWN -- that is not the same as nothing to submit. Re-run "
              f"when GitHub is reachable.")
    if not apply:
        if ready:
            print(f"\n  --apply opens {len(ready)} upstream PR(s) with the approved "
                  f"title and body.")
        return 0
    for r in ready:
        # Snapshot the approval BEFORE publishing, so the record of who authorised
        # this exists even if the PR creation then fails.
        try:
            moatlib_record(r["name"])
        except Exception as e:                      # noqa: BLE001 - reported, not raised
            print(f"  FAILED to record the approval for {r['name']}: {e}")
            continue
        url, err = open_upstream(r["name"], r)
        if url and err:          # the PR is open but the record did not take
            print(f"  PARTIAL  {r['name']}: {err}")
        else:
            print(f"  {'opened ' + url if url else 'FAILED: ' + (err or '?')}  ({r['name']})")
    return 0


def moatlib_record(name):
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    return moatlib.record_pr_approval(name)


# ---- fix rounds: review and merge ------------------------------------------
#
# While an upstream PR is open its head branch is upstream-visible, so fixes stage
# on `moat-fix-<pr#>` (moatlib.fix_branch) and reach the PR only here: a person
# approves the delta on a fork review PR, and --merge-fix fast-forwards the PR
# branch to exactly the approved tip. The same contract as --publish: the one
# write is mechanical because a person approved exactly that content, and every
# check re-runs live at merge time.

# The fix review PR's body section that becomes the upstream reply, posted
# verbatim as a comment on the upstream PR after the merge. Everything from this
# heading to the end of the body is the reply; approving the PR approves it along
# with the code. No section means no comment is posted.
REPLY_HEADING = "## Upstream reply"


def fix_reply_of(body):
    """The reply text a fix review PR's body carries, or None."""
    lines = (body or "").splitlines()
    for i, line in enumerate(lines):
        if line.strip() == REPLY_HEADING:
            return "\n".join(lines[i + 1:]).strip() or None
    return None


def _fix_delta_hits(fork, base, head):
    """Jargon hits in the staged delta's commit messages and added lines, read
    from the fork over the API so the scan does not need a local clone. Raises
    ValueError when the comparison cannot be read: a gate that cannot run is not
    a gate that passed."""
    sys.path.insert(0, str(REPO / "utils"))
    import jargon
    terms, allow = jargon.load()
    cmp = gh_json(["api", f"repos/{fork}/compare/{base}...{head}",
                   "--jq", "{commits: [.commits[].commit.message], "
                           "patches: [.files[].patch // \"\"]}"])
    if cmp is None:
        raise ValueError(f"cannot read {fork} compare {base[:12]}...{head[:12]}")
    hits = []
    for msg in cmp.get("commits") or []:
        hits += jargon.scan_text(msg, "delta commit", terms, allow)
    added = "\n".join(l[1:] for p in (cmp.get("patches") or [])
                      for l in p.splitlines()
                      if l.startswith("+") and not l.startswith("+++"))
    hits += jargon.scan_text(added, "delta added lines", terms, allow)
    return hits


def fix_review_rows():
    """Staged fix rounds whose gates are met and which have no review PR yet."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    out = []
    for name, d, _where in all_records():
        fix = d.get("fix")
        if not fix or fix.get("review_pr") or d.get("pr_state") != "open":
            continue
        fork = (d.get("fork_url") or "").replace("https://github.com/", "")
        if not fork:
            continue
        ready, blocking, _ = moatlib.fix_ready(name)
        if not ready:
            out.append({"name": name, "fork": fork, "fix": fix,
                        "problem": "not fix-ready: "
                                   + ", ".join(f"{p}={s}" for p, s in blocking)})
            continue
        base = fix.get("base_sha")
        tip = gh_json(["api", f"repos/{fork}/git/ref/heads/{fix['branch']}",
                       "--jq", "{sha: .object.sha}"])
        if not tip or not tip.get("sha"):
            out.append({"name": name, "fork": fork, "fix": fix,
                        "problem": f"{fix['branch']} does not exist on the fork"})
            continue
        out.append({"name": name, "fork": fork, "fix": fix, "base": base,
                    "tip": tip["sha"], "problem": None,
                    "branch": fix["branch"],
                    "target": d.get("fork_branch") or moatlib.PORT_BRANCH})
    return out


def open_fix_review_pr(row, title, body, apply=False):
    """Open the fork review PR for a staged fix delta.

    Unlike open_review_pr, the title and body are NOT republished upstream -- the
    upstream-visible content is the delta's commits, scanned here, plus the
    optional reply section, scanned as the upstream prose it is about to become."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    import jargon
    import prose

    if row.get("problem"):
        return ("blocked", row["problem"])
    try:
        hits = _fix_delta_hits(row["fork"], row["base"], row["tip"])
    except ValueError as e:
        return ("jargon", f"cannot check the delta for in-house vocabulary: {e}")
    if hits:
        return ("jargon", "in-house vocabulary in the staged delta: "
                + ", ".join(sorted({h[2] for h in hits})))
    reply = fix_reply_of(body)
    if REPLY_HEADING in (body or "") and not reply:
        return ("reply", f"the body carries {REPLY_HEADING!r} with nothing under "
                         f"it -- drop the heading or write the reply")
    if reply:
        terms, allow = jargon.load()
        rhits = jargon.scan_text(reply, "upstream reply", terms, allow)
        if rhits:
            return ("jargon", "in-house vocabulary in the upstream reply: "
                    + ", ".join(sorted({h[2] for h in rhits})))
        wrapped = prose.check(reply, "upstream reply")
        if wrapped:
            return ("wrapped", wrapped[0])
    if not apply:
        return ("would-open",
                f"{row['fork']}: {row['branch']} -> {row['target']} "
                f"(base {row['base'][:12]}, tip {row['tip'][:12]}"
                + (", carries an upstream reply" if reply else "")
                + f")\n\n{title}\n\n{body}")
    r = subprocess.run(["gh", "pr", "create", "--repo", row["fork"],
                        "--head", row["branch"], "--base", row["target"],
                        "--title", title, "--body", body],
                       capture_output=True, text=True, timeout=90)
    if r.returncode:
        return ("error", (r.stderr or r.stdout).strip())
    url = r.stdout.strip().splitlines()[-1]
    moatlib.set_fix_review_pr(row["name"], url)
    subprocess.run(
        ["gh", "pr", "comment", url, "--body",
         f"To approve this fix round, leave a comment containing this line by "
         f"itself:\n\n```\n{moatlib.APPROVE_COMMAND}\n```\n\n"
         f"To send it back to the porter instead:\n\n"
         f"```\n{moatlib.CHANGES_COMMAND}\n```\n\n"
         f"Approving covers the commits on this branch"
         + (f" and the section under {REPLY_HEADING!r} in the body, which is "
            f"posted verbatim as a comment on the upstream pull request after "
            f"the merge" if reply else "")
         + ". `utils/upstream.py --merge-fix --apply` then fast-forwards the "
           "open upstream PR's branch to exactly the approved tip. Anything "
           "pushed afterwards, or any edit to the body, voids the approval and "
           "needs a fresh one."],
        capture_output=True, text=True, timeout=90)
    return ("opened", url)


def merge_fix_rows():
    """Fix rounds with a recorded review PR, ready for the merge gate."""
    for name, d, _where in all_records():
        fix = d.get("fix")
        if fix and fix.get("review_pr") and d.get("pr_state") == "open":
            yield name, d, fix


def merge_fix_blockers(name, d, fix, pr):
    """Everything that must hold before the PR branch moves, re-checked live."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    import prose
    import jargon

    bad = []
    code, why = moatlib.approval_currency(pr)
    if code != "ok":
        bad.append(f"approval {code}: {why}")
    blockers, _notes = moatlib.moat_command_audit(pr)
    bad += blockers
    ready, blocking, _ = moatlib.fix_ready(name)
    if not ready:
        bad.append("not fix-ready: " + ", ".join(f"{p}={s}" for p, s in blocking))

    fork = (d.get("fork_url") or "").replace("https://github.com/", "")
    tip = pr.get("headRefOid")
    pub = d.get("published_sha")
    if not tip:
        bad.append("cannot read the approved tip from the fix review PR")
    if tip and not moatlib.same_commit(tip, d.get("head_sha")):
        bad.append(f"the fix review PR's head {tip[:12]} is not the recorded "
                   f"head_sha {(d.get('head_sha') or '?')[:12]} -- the record and "
                   f"the approval describe different commits")
    try:
        hits = _fix_delta_hits(fork, fix["base_sha"], tip or fix["branch"])
        if hits:
            bad.append("in-house vocabulary in the staged delta: "
                       + ", ".join(sorted({h[2] for h in hits})[:4]))
    except ValueError as e:
        bad.append(str(e))
    reply = fix_reply_of(pr.get("body"))
    if REPLY_HEADING in (pr.get("body") or "") and not reply:
        bad.append(f"the body carries {REPLY_HEADING!r} with nothing under it")
    if reply:
        terms, allow = jargon.load()
        rhits = jargon.scan_text(reply, "upstream reply", terms, allow)
        if rhits:
            bad.append("in-house vocabulary in the upstream reply: "
                       + ", ".join(sorted({h[2] for h in rhits})[:4]))
        bad += prose.check(reply, "upstream reply")

    # The merge is a git push, so it needs the clone and needs it to agree with
    # GitHub about what is being fast-forwarded from where.
    clone = REPO / "projects" / name / "src"
    if not (clone / ".git").exists():
        bad.append(f"no fork clone at {clone} -- the merge push runs from a host "
                   f"that has one")
        return bad
    fork_url = d.get("fork_url")
    branch = d.get("fork_branch") or "moat-port"
    ls = subprocess.run(["git", "-C", str(clone), "ls-remote", fork_url,
                         f"refs/heads/{branch}"],
                        capture_output=True, text=True, timeout=60)
    remote_tip = (ls.stdout.split() or [""])[0]
    if ls.returncode or not remote_tip:
        bad.append(f"cannot read {branch} on the fork ({fork_url})")
    elif not pub or not moatlib.same_commit(remote_tip, pub):
        bad.append(f"the fork's {branch} is at {remote_tip[:12]}, not the "
                   f"published {(pub or '?')[:12]} -- the PR branch moved outside "
                   f"the fix flow; a person sorts that out first")
    if tip:
        f = subprocess.run(["git", "-C", str(clone), "fetch", fork_url,
                            f"+refs/heads/{fix['branch']}:refs/moat/fix"],
                           capture_output=True, text=True, timeout=120)
        have = subprocess.run(["git", "-C", str(clone), "rev-parse", "--verify",
                               "--quiet", "refs/moat/fix"],
                              capture_output=True, text=True)
        if f.returncode or not moatlib.same_commit(have.stdout.strip(), tip):
            bad.append(f"the fork's {fix['branch']} tip does not match the "
                       f"approved {tip[:12]} -- fetch failed or the branch moved")
        elif pub:
            anc = subprocess.run(["git", "-C", str(clone), "merge-base",
                                  "--is-ancestor", pub, tip],
                                 capture_output=True, text=True)
            if anc.returncode:
                bad.append(f"{fix['branch']} is not a descendant of the published "
                           f"{pub[:12]} -- the staging branch was rebased; the "
                           f"merge must be a fast-forward")
    return bad


def do_merge_fix(name, d, fix, pr):
    """The pre-authorized write: fast-forward the open PR's branch to the approved
    tip, post the approved reply (if the body carries one), delete the staging
    branch, and record all of it. Every check has already run in
    merge_fix_blockers; like open_upstream, trusted code proves its own case."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    clone = REPO / "projects" / name / "src"
    fork_url = d.get("fork_url")
    branch = d.get("fork_branch") or "moat-port"
    tip = pr["headRefOid"]
    env = {**os.environ, "MOAT_PUBLISH": "1"}
    push = subprocess.run(["git", "-C", str(clone), "push", fork_url,
                           f"{tip}:refs/heads/{branch}"],
                          capture_output=True, text=True, timeout=120, env=env)
    if push.returncode:
        return (False, f"push failed: {(push.stderr or push.stdout).strip()[:200]}")

    reply = fix_reply_of(pr.get("body"))
    posted = None
    if reply:
        import gh_guard
        real = gh_guard.real_gh()
        if real is None:
            posted = "gh is not installed; the approved reply was NOT posted"
        else:
            c = subprocess.run([real, "pr", "comment", d["pr_url"],
                                "--body", reply],
                               capture_output=True, text=True, timeout=90)
            posted = ("posted the approved reply" if c.returncode == 0 else
                      f"could NOT post the approved reply: "
                      f"{(c.stderr or c.stdout).strip()[:160]}")

    # The branch's job is done and its commits are on the PR branch; a person
    # ruled that staging branches are deleted on merge so the next round can
    # reuse the name.
    subprocess.run(["git", "-C", str(clone), "push", fork_url,
                    f":refs/heads/{fix['branch']}"],
                   capture_output=True, text=True, timeout=60, env=env)
    try:
        moatlib.set_fix_merged(name, tip)
    except Exception as e:                      # noqa: BLE001 - reported, not raised
        return (True, f"merged to {tip[:12]} but could NOT record it: {e}"
                      + (f"; {posted}" if posted else ""))
    return (True, f"fast-forwarded {branch} to {tip[:12]}"
                  + (f"; {posted}" if posted else ""))


def report_merge_fix(apply, only=None):
    rows = [(n, d, f) for n, d, f in merge_fix_rows() if only in (None, n)]
    print(f"upstream: {len(rows)} fix round(s) with a review PR recorded\n")
    ret = 0
    for name, d, fix in rows:
        sys.path.insert(0, str(REPO / "utils"))
        import moatlib
        pr = moatlib.fetch_review_pr(fix["review_pr"])
        if pr is None:
            print(f"  UNREACHABLE {name:25} could not read {fix['review_pr']} -- "
                  f"an outage is not a withdrawn approval")
            ret = 1
            continue
        bad = merge_fix_blockers(name, d, fix, pr)
        if bad:
            print(f"  HELD       {name:26} {bad[0][:78]}")
            for b in bad[1:]:
                print(f"             {'':26} {b[:78]}")
            continue
        reply = fix_reply_of(pr.get("body"))
        print(f"  READY      {name:26} {fix['branch']} -> "
              f"{d.get('fork_branch') or 'moat-port'} at "
              f"{(pr.get('headRefOid') or '?')[:12]}"
              + (" (+ upstream reply)" if reply else ""))
        if not apply:
            continue
        try:
            moatlib_record(name)     # who authorised this, before anything moves
        except Exception as e:                  # noqa: BLE001 - reported, not raised
            print(f"  FAILED to record the approval for {name}: {e}")
            ret = 1
            continue
        ok, detail = do_merge_fix(name, d, fix, pr)
        print(f"  {'MERGED' if ok else 'FAILED':10} {name:26} {detail}")
        if not ok:
            ret = 1
    if rows and not apply:
        print("\n  --merge-fix --apply performs the fast-forward push (and posts "
              "the approved reply, where the body carries one).")
    return ret


RECONCILED = REPO / "data" / "reconciled.json"


def stamp_reconciled(n_records, n_drift):
    """Record that the upstream sweep ran. Written on every full poll, applied or not:
    the question this answers is "has anyone LOOKED", not "did anything change"."""
    RECONCILED.parent.mkdir(parents=True, exist_ok=True)
    RECONCILED.write_text(json.dumps(
        {"at": TODAY, "records": n_records, "drifted": n_drift}, indent=2) + "\n")


def reconciled_age_days(today):
    """Days since the last full sweep, or None if it has never run. The stamp is
    committed rather than local: whether the record has been checked is a fact about
    the project, not about one clone."""
    import datetime
    try:
        at = json.loads(RECONCILED.read_text())["at"]
    except (OSError, ValueError, KeyError):
        return None
    try:
        return (datetime.date.fromisoformat(today) - datetime.date.fromisoformat(at)).days
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True)
    g.add_argument("--apply", action="store_true")
    ap.add_argument("--review", action="store_true",
                    help="ports whose gates are met with no review PR open yet")
    ap.add_argument("--name", help="with --review --apply: which project to open")
    ap.add_argument("--title", help="with --review --apply: the upstream PR title")
    ap.add_argument("--body-file", help="with --review --apply: file holding the body")
    ap.add_argument("--forks", action="store_true",
                    help="poll for forks of awaiting-fork projects instead of PR state")
    ap.add_argument("--approvals", action="store_true",
                    help="find port approvals overtaken by a later push or edit")
    ap.add_argument("--publish", action="store_true",
                    help="submit approved ports upstream with their approved title and body")
    ap.add_argument("--fix-review", action="store_true",
                    help="staged fix rounds whose gates are met with no review PR open yet")
    ap.add_argument("--merge-fix", action="store_true",
                    help="fast-forward an open upstream PR to an approved fix round's tip")
    ap.add_argument("--attention", action="store_true",
                    help="open upstream PRs where a maintainer is waiting on us")
    a = ap.parse_args()

    if a.forks:
        fork_poll(apply=a.apply)
        return 0
    if a.approvals:
        return report_approvals(apply=a.apply)
    if a.review:
        rows = review_candidates()
        if not a.apply and not a.name:
            for r in rows:
                if r["problem"]:
                    print(f"  BLOCKED  {r['name']:22} {r['problem']}")
                else:
                    print(f"  READY    {r['name']:22} {r['branch']} -> {r['base']} "
                          f"({r['commits']} commits, {r['files']} files)")
            print(f"-- {sum(1 for r in rows if not r['problem'])} port(s) need a review PR; "
                  f"{sum(1 for r in rows if r['problem'])} blocked")
            print("   open one: --review --apply --name <p> --title '<t>' --body-file <f>")
            print("   (same command without --apply previews it and runs the gates)")
            return 0
        if not (a.name and a.title and a.body_file):
            print("--review --name needs --title and --body-file "
                  "(add --apply to open the PR rather than preview it)", file=sys.stderr)
            return 2
        row = next((r for r in rows if r["name"] == a.name), None)
        if row is None:
            print(f"{a.name} is not awaiting a review PR (already has one, or not "
                  f"PR-ready)", file=sys.stderr)
            return 2
        body = pathlib.Path(a.body_file).read_text()
        action, detail = open_review_pr(row, a.title, body, apply=a.apply)
        print(f"review-pr: {action} -- {detail}")
        return 0 if action in ("opened", "would-open") else 1
    if a.publish:
        return report_publish(apply=a.apply)
    if a.fix_review:
        rows = fix_review_rows()
        if not a.apply and not a.name:
            for r in rows:
                if r["problem"]:
                    print(f"  BLOCKED  {r['name']:22} {r['problem']}")
                else:
                    print(f"  READY    {r['name']:22} {r['branch']} -> {r['target']} "
                          f"(base {r['base'][:12]}, tip {r['tip'][:12]})")
            print(f"-- {sum(1 for r in rows if not r['problem'])} fix round(s) need "
                  f"a review PR; {sum(1 for r in rows if r['problem'])} blocked")
            print("   open one: --fix-review --apply --name <p> --title '<t>' "
                  "--body-file <f>")
            print(f"   (a body section headed {REPLY_HEADING!r} is posted verbatim "
                  f"on the upstream PR after the merge)")
            return 0
        if not (a.name and a.title and a.body_file):
            print("--fix-review --name needs --title and --body-file "
                  "(add --apply to open the PR rather than preview it)",
                  file=sys.stderr)
            return 2
        row = next((r for r in rows if r["name"] == a.name), None)
        if row is None:
            print(f"{a.name} has no fix round awaiting a review PR", file=sys.stderr)
            return 2
        body = pathlib.Path(a.body_file).read_text()
        action, detail = open_fix_review_pr(row, a.title, body, apply=a.apply)
        print(f"fix-review-pr: {action} -- {detail}")
        return 0 if action in ("opened", "would-open") else 1
    if a.merge_fix:
        return report_merge_fix(apply=a.apply, only=a.name)
    if a.attention:
        return report_attention(recorded(), TODAY)

    rows = recorded()
    drift, unreviewed, headdrift, errors = poll(rows)
    skipped = [r for r in rows if r.get("skipped")]
    # The sweep just happened, so stamp it whether or not anything is applied.
    # Nothing runs on a schedule any more, so this timestamp is the only thing that
    # can tell anyone the record has gone unchecked; orient.sh reads it every run.
    if not errors:
        stamp_reconciled(len(rows), len(drift))

    print(f"upstream: {len(rows)} recorded PRs, {len(drift)} drifted, "
          f"{len(unreviewed)} awaiting our response, {len(headdrift)} head-moved, "
          f"{len(skipped)} skipped, {len(errors)} lookup errors\n")
    for r in drift:
        print(f"  DRIFT      {r['name']:26} we say {str(r['ours']):16} "
              f"GitHub says {r['real']:8} {r['repo']}#{r['num']}")
    for r in unreviewed:
        print(f"  CHANGES    {r['name']:26} maintainer requested changes           "
              f"{r['repo']}#{r['num']}")
    for r in headdrift:
        print(f"  HEAD-MOVED {r['name']:26} PR head {r['head'][:12]} != published "
              f"{r['published'][:12]} {r['repo']}#{r['num']}")
    if headdrift:
        # Deliberately never applied: the usual cause is a maintainer pushing to
        # our branch, which a person did on purpose. The move is to READ what
        # landed and put a recommendation in front of a human, not to absorb or
        # revert it.
        print("\n  a moved head on an open PR is a push outside the fix flow -- "
              "usually a maintainer edit. Review the commit(s) between the two "
              "shas and recommend a course of action; a person decides. If the "
              "content is accepted, a fresh fix round from the new tip re-enters "
              "the flow.")
    for r in skipped:
        print(f"  SKIPPED    {r['name']:26} {r['skipped']}")
    for r in errors:
        print(f"  ERROR      {r['name']:26} {r['why']}")

    if not a.apply:
        if drift:
            print(f"\n  --apply records the {sum(1 for r in drift if r['real'] == 'MERGED')} "
                  f"merge(s) and opens a PR; closures need a human decision.")
        return 0

    landed = [(r, apply_one(r)) for r in drift]
    on_branch = [r for r, where in landed if where == "branch"]
    local = [r for r, where in landed if where == "local"]
    if not (on_branch or local):
        print("\nupstream: nothing to apply automatically")
        return 0
    for r in on_branch:
        print(f"\nupstream: recorded {r['name']} as merged on its own port branch "
              f"(already pushed; not part of the record-sync PR)")
    if not local:
        return 0
    subprocess.run([sys.executable, "utils/gen_readme.py"], cwd=str(REPO),
                   capture_output=True)
    return publish(local)


def publish(applied):
    """Put the corrections on ONE stable branch and keep ONE PR current.

    A date-stamped branch would open a second PR every run while the first was still
    unmerged, and they would accumulate with overlapping changes. Instead the branch
    is recomputed from scratch each time against the current trunk and force-pushed:
    an open PR follows its head automatically, so it always shows the drift as of the
    latest run rather than a stale snapshot plus corrections.

    Force-pushing is safe here precisely because nobody else edits this branch -- it
    is bot-owned and fully regenerated, never appended to."""
    def git(*args, check=True):
        r = subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True)
        if check and r.returncode:
            print(f"upstream: git {' '.join(args)}: {r.stderr.strip()}", file=sys.stderr)
        return r

    trunk = git("symbolic-ref", "--short", "refs/remotes/origin/HEAD",
                check=False).stdout.strip().split("/")[-1] or "main"
    # Hold the edits apply_one() made as a commit object, then CLEAN the tree before
    # switching branches: `checkout -B` with a dirty tree either refuses or carries
    # changes across unpredictably, depending on whether they conflict with the target.
    stash = git("stash", "create").stdout.strip()
    if not stash:
        print("upstream: no changes to publish")
        return 0
    started_on = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    git("checkout", "-q", "--", ".")                   # tree now matches HEAD
    git("fetch", "-q", "origin", trunk)
    git("checkout", "-q", "-B", BRANCH, f"origin/{trunk}")
    git("checkout", stash, "--", ".")                  # replay the edits onto the trunk
    git("add", "-A")

    if not git("diff", "--cached", "--quiet", check=False).returncode:
        print("upstream: branch matches the trunk; nothing to publish")
        git("reset", "-q", "--hard", f"origin/{trunk}")
        git("checkout", "-q", started_on)
        return 0

    names = ", ".join(r["name"] for r in applied)
    subject = f"records: upstream merges ({names})"
    git("commit", "-q", "-m", subject)
    git("push", "-q", "--force-with-lease", "-u", "origin", BRANCH)

    existing = gh_json(["pr", "list", "--head", BRANCH, "--state", "open",
                        "--json", "number,url"]) or []
    body = ("Recorded upstream merges that happened after each project's PR merged here.\n\n"
            + "\n".join(f"- **{r['name']}** -- {r['repo']}#{r['num']} merged "
                         f"{(r['merged_at'] or '')[:10]}" for r in applied)
            + "\n\nOpened by `utils/upstream.py`. A protected trunk cannot be written "
              "directly, so this arrives as a PR like everything else.\n\n"
              "This branch is regenerated from the trunk on every run and force-pushed, so "
              "the PR always reflects current drift rather than accumulating. Closures are "
              "never applied automatically -- withdrawn, rejected and superseded look "
              "identical to the API and mean different things.\n")
    if existing:
        subprocess.run(["gh", "pr", "edit", str(existing[0]["number"]), "--body", body],
                       cwd=str(REPO), capture_output=True, text=True)
        print(f"\nupstream: updated existing PR {existing[0]['url']} "
              f"({len(applied)} merge(s))")
    else:
        # Check our own title before opening: a tool that enforces a convention on
        # everyone else and exempts itself teaches people the convention is optional.
        chk = subprocess.run([sys.executable, "utils/pr_intent.py", "--check-title",
                              "--branch", BRANCH, "--title", subject],
                             cwd=str(REPO), capture_output=True, text=True)
        if chk.returncode:
            print(chk.stdout.strip(), file=sys.stderr)
            print("upstream: refusing to open a pull request with that title",
                  file=sys.stderr)
            return 1
        subprocess.run(["gh", "pr", "create", "--fill-first", "--body", body],
                       cwd=str(REPO), capture_output=True, text=True)
        print(f"\nupstream: opened a PR with {len(applied)} merge(s)")
    git("checkout", "-q", started_on)                  # leave the caller where it was
    return 0


if __name__ == "__main__":
    sys.exit(main())
