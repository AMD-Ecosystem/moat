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
    python3 utils/upstream.py --apply            # update the moatbot-sync branch + PR
    python3 utils/upstream.py --forks            # report projects awaiting a fork
    python3 utils/upstream.py --forks --apply    # release the ones whose fork exists
    python3 utils/upstream.py --approvals        # report approvals overtaken by a push or edit
    python3 utils/upstream.py --approvals --apply # dismiss them and re-request review
    python3 utils/upstream.py --publish          # report approved ports ready to submit
    python3 utils/upstream.py --publish --apply  # open the upstream PRs

Record maintenance and one publishing step. None of it does any porting -- that needs a
GPU host and a session. These keep the record true, tell someone, and send an approved
port on its way.

Running it repeatedly is safe. There is ONE branch, regenerated from the trunk each
run and force-pushed, so an unmerged PR is updated in place rather than joined by a
second one carrying overlapping changes.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
TODAY = subprocess.run(["date", "-u", "+%Y-%m-%d"],
                       capture_output=True, text=True).stdout.strip()
# One stable branch, deliberately not date-stamped: see publish().
# Named for the identity that pushes it, which is what a reader sees.
BRANCH = "moatbot-sync"

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
    for sp in sorted((REPO / "projects").glob("*/status.json")):
        d = json.loads(sp.read_text())
        pr = d.get("pr_url") or next(
            (b.get("pr_url") for b in (d.get("platforms") or {}).values() if b.get("pr_url")), None)
        if not pr:
            continue
        m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", pr)
        if not m:
            continue
        ours = d.get("pr_state") or ("merged" if d.get("pr_merged_at") else None)
        out.append({"name": sp.parent.name, "repo": m.group(1), "num": m.group(2),
                    "url": pr, "ours": ours})
    return out


def poll(rows):
    """Compare each record against GitHub. Returns (drift, unreviewed, errors)."""
    drift, unreviewed, errors = [], [], []
    for r in rows:
        d = gh_json(["pr", "view", r["num"], "--repo", r["repo"], "--json",
                     "state,mergedAt,reviewDecision,updatedAt"])
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
    return drift, unreviewed, errors


OURS = {"jeffdaily", "moatbot"}          # accounts that speak for this project
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
    a maintainer, and superseded are different outcomes that look identical here."""
    if r["real"] != "MERGED":
        return False
    subprocess.run([sys.executable, "utils/moatlib.py", "set-pr-merged", r["name"]],
                   cwd=str(REPO), capture_output=True, text=True)
    return True


# A decline is expressed as a label on the project's draft PR. Approval can live
# outside MOAT -- a fork in the org IS the record -- but a decline cannot: it has to
# reach data/dispositions.json on the trunk, or the project simply gets proposed
# again and the work repeats. So a decline MERGES the draft PR rather than closing it.
#
# `declined` on its own is deliberate. The repo is public, so a written reason is
# permanent and quotable, and a project can be reconsidered later without anything to
# walk back. It records that a decision was made, not why.
DECLINE_LABELS = {
    "declined:license": "license-blocked",
    "declined:already-supported": "already-supported",
    "declined": "declined",
}


def _draft_pr(branch):
    prs = gh_json(["pr", "list", "--head", branch, "--state", "open",
                   "--json", "number,labels,url"]) or []
    return prs[0] if prs else None


def _decline_reason(pr):
    """The reason a decline label maps to, or None. A more specific label wins over
    the bare one, so `declined` plus `declined:license` records the licence."""
    names = {l["name"] for l in pr.get("labels", [])}
    specific = [DECLINE_LABELS[n] for n in names
                if n in DECLINE_LABELS and n != "declined"]
    if specific:
        return specific[0]
    return DECLINE_LABELS["declined"] if "declined" in names else None


def fork_poll(apply=False, stale_weeks=3):
    """Release projects whose fork has appeared, and flag ones waiting too long.

    A project sits in `awaiting-fork` until someone with the rights creates the fork;
    that act is the decision to take it up. Nobody should have to notice by hand, and
    -- more to the point -- whoever created the fork gets no confirmation unless
    something says so. This closes both ends: it advances the state and comments on
    the project's draft PR.

    State lives on the project's own `port/<name>` branch, not the trunk, so this
    pushes directly. It does NOT start any work: porting needs a GPU host and a
    session. This only makes the project eligible and tells someone.
    """
    import datetime
    released, waiting, declined, conflicts = [], [], [], []
    for sp in sorted((REPO / "projects").glob("*/status.json")):
        d = json.loads(sp.read_text())
        name = sp.parent.name
        blocks = {a: b for a, b in (d.get("platforms") or {}).items()
                  if b.get("state") == "awaiting-fork"}
        if not blocks:
            continue
        slug = (d.get("fork_url") or f"https://github.com/AMD-Ecosystem/{name}") \
            .replace("https://github.com/", "")
        exists = gh_json(["api", f"repos/{slug}", "--jq", ".full_name"]) is not None or \
            subprocess.run(["gh", "api", f"repos/{slug}"], capture_output=True).returncode == 0
        if not exists:
            since = min((b.get("updated_at") or "") for b in blocks.values())
            waiting.append({"name": name, "slug": slug, "since": since})
            continue
        released.append({"name": name, "slug": slug, "archs": sorted(blocks)})

    print(f"fork-poll: {len(released)} released, {len(declined)} declined, "
          f"{len(waiting)} still waiting, {len(conflicts)} conflicted\n")
    for r in released:
        print(f"  RELEASED   {r['name']:26} fork exists: {r['slug']}")
    for r in declined:
        print(f"  DECLINED   {r['name']:26} reason={r['reason']:18} PR #{r['pr']}")
    for c in conflicts:
        print(f"  CONFLICT   {c['name']:26} declined ({c['reason']}) but "
              f"{c['slug']} exists -- resolve by hand")
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

    if not apply or not (released or declined):
        return released, waiting, declined, conflicts

    for r in released:
        branch = f"port/{r['name']}"
        def git(*a):
            return subprocess.run(["git", *a], cwd=str(REPO), capture_output=True, text=True)
        if git("fetch", "-q", "origin", branch).returncode:
            print(f"  {r['name']}: no {branch} on the remote; state left alone")
            continue
        git("checkout", "-q", "-B", branch, f"origin/{branch}")
        obj = json.loads((REPO / "projects" / r["name"] / "status.json").read_text())
        for a in r["archs"]:
            obj["platforms"][a]["state"] = "screened"
        (REPO / "projects" / r["name"] / "status.json").write_text(
            json.dumps(obj, indent=2) + "\n")
        git("add", "-A")
        git("commit", "-q", "-m",
            f"{r['name']}: fork exists, releasing for planning\n\n"
            f"{r['slug']} was created, which is the decision to take this project up.")
        git("push", "-q", "origin", branch)
        pr = gh_json(["pr", "list", "--head", branch, "--state", "open",
                      "--json", "number"]) or []
        if pr:
            subprocess.run(
                ["gh", "pr", "comment", str(pr[0]["number"]), "--body",
                 f"`{r['slug']}` now exists, so this project is released for planning. "
                 f"State advanced to `screened`; the next session on a suitable host "
                 f"will pick it up."],
                cwd=str(REPO), capture_output=True, text=True)

    for c in declined:
        branch = f"port/{c['name']}"
        def git(*a):
            return subprocess.run(["git", *a], cwd=str(REPO), capture_output=True, text=True)
        if git("fetch", "-q", "origin", branch).returncode:
            print(f"  {c['name']}: no {branch} on the remote; disposition not written")
            continue
        git("checkout", "-q", "-B", branch, f"origin/{branch}")
        # The disposition on the trunk is what makes the decline durable: scaffold
        # refuses a skipped project, so nobody re-proposes it by accident.
        repo_slug = (c["upstream"] or "").replace("https://github.com/", "")
        r = subprocess.run([sys.executable, "utils/triage.py", "skip", repo_slug,
                            "--reason", c["reason"], "--note", "declined via PR label"],
                           cwd=str(REPO), capture_output=True, text=True)
        if r.returncode:
            print(f"  {c['name']}: triage skip failed: {r.stderr.strip()[:120]}")
            continue
        git("add", "-A")
        git("commit", "-q", "-m",
            f"{c['name']}: declined ({c['reason']})\n\n"
            f"Recorded so the project is not proposed again. Merging this is what "
            f"makes the decision durable; closing the PR would lose it.")
        git("push", "-q", "origin", branch)
        subprocess.run(["gh", "pr", "ready", str(c["pr"])], cwd=str(REPO),
                       capture_output=True, text=True)
        subprocess.run(
            ["gh", "pr", "comment", str(c["pr"]), "--body",
             f"Declined (`{c['reason']}`). The disposition is recorded on this branch; "
             f"merging makes it durable, so the project will not be proposed again. "
             f"It can be revisited later with `scaffold --force`."],
            cwd=str(REPO), capture_output=True, text=True)
    return released, waiting, declined, conflicts


# An approval covers the code, title and body that were on screen when it was given.
# A push or an edit afterwards means the reviewer is now recorded as approving
# something they never saw -- and GitHub keeps showing "Approved" through both, so
# nothing surfaces it. Only `stale-*` is actionable: `withdrawn` means the approval is
# already gone (often because a previous run dismissed it) and re-asking would nag,
# and `record-mismatch` means our own file disagrees with GitHub, which is a human's
# problem and never a reason to touch someone's review.
ACTIONABLE_APPROVAL = {"stale-commits", "stale-content"}


def approval_drift():
    """Projects whose recorded approval no longer covers their review PR."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    rows = []
    for sp in sorted((REPO / "projects").glob("*/status.json")):
        d = json.loads(sp.read_text())
        if not d.get("pr_approval") or d.get("pr_state"):
            continue          # never approved, or already published
        name = sp.parent.name
        code, why = moatlib.pr_approval_status(name, live=True)
        if code != "ok":
            rows.append({"name": name, "code": code, "why": why,
                         "url": (d["pr_approval"].get("review_pr") or d.get("review_pr"))})
    return rows


def refresh_approval(r):
    """Dismiss the overtaken approval and ask the same reviewer to look again.

    Dismissing rather than only commenting is the point: the PR must stop displaying
    "Approved" for content nobody approved, and the submission gate refuses either
    way, so leaving the green check up only misleads a human reading the page.

    This is our own fork, inside the autonomy boundary -- unlike anything upstream,
    which always needs its own explicit yes."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    pr = moatlib.fetch_review_pr(r["url"])
    if pr is None:
        return False
    review = moatlib._approving_review(pr)
    if review is None or not review.get("id"):
        return False
    slug, num = pr["slug"], pr["number"]
    msg = (f"Dismissing this approval automatically: {r['why']}.\n\n"
           "The approval covered the code, title and body as they stood when it was "
           "given, so it no longer describes what would be submitted upstream. "
           "Nothing is published while it is stale. Please re-approve if the change "
           "still looks right.")
    ok = gh_json(["api", "-X", "PUT",
                  f"repos/{slug}/pulls/{num}/reviews/{review['id']}/dismissals",
                  "-f", f"message={msg}", "-f", "event=DISMISS"]) is not None
    if not ok:
        return False
    # Re-request so it lands in their review queue rather than waiting to be noticed.
    gh_json(["api", "-X", "POST", f"repos/{slug}/pulls/{num}/requested_reviewers",
             "-f", f"reviewers[]={review['login']}"])
    return True


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
            print(f"\n  --apply dismisses {len(actionable)} overtaken approval(s) "
                  f"and re-requests review.")
        return 0
    for r in actionable:
        done = refresh_approval(r)
        print(f"  {'dismissed + re-requested' if done else 'COULD NOT dismiss'} {r['name']}")
    return 0


def publishable():
    """Ports whose review PR carries a standing approval and are ready to submit.

    This is what makes clicking Approve mean something. It looks for the approval on
    GitHub rather than for a snapshot in our files, because the click is the whole
    signal -- a project nobody has run record-pr-approval on yet is exactly the case
    that needs finding."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    out = []
    for sp in sorted((REPO / "projects").glob("*/status.json")):
        d = json.loads(sp.read_text())
        url = (d.get("pr_approval") or {}).get("review_pr") or d.get("review_pr")
        if not url or d.get("pr_state"):
            continue                      # no review PR, or already submitted
        name = sp.parent.name
        pr = moatlib.fetch_review_pr(url)
        if pr is None or moatlib._approving_review(pr) is None:
            continue                      # not approved (yet), or unreachable
        out.append({"name": name, "url": url, "pr": pr,
                    "title": pr.get("title") or "", "body": pr.get("body") or ""})
    return out


def publish_blockers(name, row):
    """Everything that must hold before a port is submitted upstream, as a list of
    reasons it must not be. Checked at publish time rather than trusted from earlier:
    a gate that passed when the reviewer looked is not evidence about now."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    import jargon

    bad = []
    # The approval must still cover this exact content. Judged from the PR itself,
    # not from our snapshot: a port approved five minutes ago has no snapshot yet,
    # and that is the case this whole path exists to catch.
    code, why = moatlib.approval_currency(row["pr"])
    if code != "ok":
        bad.append(f"approval {code}: {why}")
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
        bad.append(f"in-house vocabulary in the title/body: "
                   + ", ".join(sorted({h[2] for h in hits})[:4]))
    return bad


def open_upstream(name, row):
    """Open the upstream PR with the approved title and body, verbatim."""
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib

    d = json.loads((REPO / "projects" / name / "status.json").read_text())
    up = json.loads((REPO / "projects" / name / "upstream.json").read_text())
    slug = up.get("full_name") or d["upstream_url"].split("github.com/", 1)[-1]
    fork_owner = d["fork_url"].split("github.com/", 1)[-1].split("/")[0]
    branch = d.get("fork_branch") or moatlib.PORT_BRANCH
    base = up.get("default_branch") or "main"

    r = subprocess.run(
        ["gh", "pr", "create", "--repo", slug, "--base", base,
         "--head", f"{fork_owner}:{branch}", "--title", row["title"],
         "--body", row["body"]],
        capture_output=True, text=True, cwd=str(REPO))
    if r.returncode:
        return (None, (r.stderr or r.stdout).strip().splitlines()[-1][:200])
    url = next((l.strip() for l in r.stdout.splitlines()
                if "github.com" in l and "/pull/" in l), "")
    num = url.rstrip("/").rsplit("/", 1)[-1]
    subprocess.run([sys.executable, "utils/moatlib.py", "set-pr-open", name, url, num],
                   cwd=str(REPO), capture_output=True, text=True)

    # The review PR has done its job -- the change it was reviewing is now in front of
    # the maintainers. Close it pointing at where the conversation continues, rather
    # than leaving an approved PR open forever looking like outstanding work. Closing
    # keeps the review thread readable; only the branch it reviewed matters now, and
    # that branch is the upstream PR's head.
    if url:
        subprocess.run(
            ["gh", "pr", "close", d["review_pr"], "--comment",
             f"Submitted upstream as {url} with the title and body approved here. "
             f"Closing this review PR; the discussion continues on the upstream one."],
            capture_output=True, text=True, cwd=str(REPO))
    return (url, None)


def report_publish(apply):
    rows = publishable()
    print(f"upstream: {len(rows)} approved port(s) awaiting submission\n")
    ready, held = [], []
    for r in rows:
        bad = publish_blockers(r["name"], r)
        (held if bad else ready).append({**r, "blockers": bad})
    for r in ready:
        print(f"  READY      {r['name']:26} \"{r['title'][:58]}\"")
    for r in held:
        print(f"  HELD       {r['name']:26} {r['blockers'][0][:78]}")
        for b in r["blockers"][1:]:
            print(f"             {'':26} {b[:78]}")
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
        print(f"  {'opened ' + url if url else 'FAILED: ' + (err or '?')}  ({r['name']})")
    return 0


def moatlib_record(name):
    sys.path.insert(0, str(REPO / "utils"))
    import moatlib
    return moatlib.record_pr_approval(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True)
    g.add_argument("--apply", action="store_true")
    ap.add_argument("--forks", action="store_true",
                    help="poll for forks of awaiting-fork projects instead of PR state")
    ap.add_argument("--approvals", action="store_true",
                    help="find port approvals overtaken by a later push or edit")
    ap.add_argument("--publish", action="store_true",
                    help="submit approved ports upstream with their approved title and body")
    ap.add_argument("--attention", action="store_true",
                    help="open upstream PRs where a maintainer is waiting on us")
    a = ap.parse_args()

    if a.forks:
        fork_poll(apply=a.apply)
        return 0
    if a.approvals:
        return report_approvals(apply=a.apply)
    if a.publish:
        return report_publish(apply=a.apply)
    if a.attention:
        return report_attention(recorded(), TODAY)

    rows = recorded()
    drift, unreviewed, errors = poll(rows)
    skipped = [r for r in rows if r.get("skipped")]

    print(f"upstream: {len(rows)} recorded PRs, {len(drift)} drifted, "
          f"{len(unreviewed)} awaiting our response, {len(skipped)} skipped, "
          f"{len(errors)} lookup errors\n")
    for r in drift:
        print(f"  DRIFT      {r['name']:26} we say {str(r['ours']):16} "
              f"GitHub says {r['real']:8} {r['repo']}#{r['num']}")
    for r in unreviewed:
        print(f"  CHANGES    {r['name']:26} maintainer requested changes           "
              f"{r['repo']}#{r['num']}")
    for r in skipped:
        print(f"  SKIPPED    {r['name']:26} {r['skipped']}")
    for r in errors:
        print(f"  ERROR      {r['name']:26} {r['why']}")

    if not a.apply:
        if drift:
            print(f"\n  --apply records the {sum(1 for r in drift if r['real'] == 'MERGED')} "
                  f"merge(s) and opens a PR; closures need a human decision.")
        return 0

    applied = [r for r in drift if apply_one(r)]
    if not applied:
        print("\nupstream: nothing to apply automatically")
        return 0
    subprocess.run([sys.executable, "utils/gen_readme.py"], cwd=str(REPO),
                   capture_output=True)
    return publish(applied)


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
    git("commit", "-q", "-m", f"moatbot: record upstream merges ({names})")
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
