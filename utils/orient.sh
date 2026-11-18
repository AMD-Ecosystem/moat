#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# MOAT entrypoint. Pull the latest MOAT state, detect this host's AMD arch, pick
# the single next project + stage for this platform, and print a dispatch
# summary. This is NOT a read-only command: before it selects anything it installs
# the git hooks and the gh guard launcher, registers the merge driver, installs the
# fork pre-push hooks, pulls, merges the trunk into a port branch and pushes that,
# and releases projects whose fork has appeared. Only the selection at the end
# reads without writing. Run this (or /port-next) when starting a CLI in the MOAT repo.
set -uo pipefail
cd "$(dirname "$0")/.."
. utils/lib.sh

# One interpreter for everything below, chosen by what it can import (see lib.sh).
if ! moat_resolve_py; then
  moat_no_py orient
  exit 3
fi

bash utils/setup_git.sh >/dev/null 2>&1 \
  || echo "orient: WARNING -- utils/setup_git.sh failed; the status merge driver may not be registered" >&2
# The hooks and the guard are the trunk's code, run from origin/main as this clone
# last saw it. Fetch first, or what gets installed is as old as the last fetch.
git fetch --quiet origin main >/dev/null 2>&1 || true
if ! hooks_out=$("$PY" utils/install_hooks.py 2>&1); then
  echo "orient: WARNING -- utils/install_hooks.py failed; the pre-push gates or the gh guard may be missing:" >&2
  printf '%s\n' "$hooks_out" | tail -n 4 | sed 's/^/           /' >&2
fi
# A guard that PATH never reaches reads exactly like one that is working. Stop here,
# before any of the upkeep below runs gh, and say what to do about it.
if ! moat_gh_guard_problem >&2; then
  exit 3
fi
# Fork clones get their own pre-push hook: while a project's upstream PR is open,
# its port branch may not move except through `upstream.py --merge-fix`. Only stdout
# is discarded: a clone this could NOT protect reports on stderr, and an unprotected
# clone that looks exactly like a protected one is the failure this guards against.
"$PY" utils/moatlib.py protect-fork >/dev/null \
  || echo "orient: moatlib.py protect-fork FAILED (exit $?) -- fork clones may be UNPROTECTED" >&2

# Refuse to work on the trunk. Project state belongs on a shared port/<name> branch,
# and that branch existing on the remote is what tells another host the project is
# claimed. MOAT_ALLOW_TRUNK=1 overrides, for control-plane work that is genuinely not
# a port (tooling, docs) but still wants orient's platform detection.
_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ -z "${MOAT_ALLOW_TRUNK:-}" ]; then
  case "$_branch" in
    main|master)
      echo "orient: on '$_branch' -- MOAT work happens on a port/<name> branch." >&2
      echo "        git checkout -b port/<project>    (or check out an existing one)" >&2
      echo "        MOAT_ALLOW_TRUNK=1 to override for non-port work." >&2
      exit 2 ;;
  esac
fi

# Sync the latest MOAT state. Best-effort: offline or local-only is fine.
# This pulls the CURRENT branch -- for a port branch that is the shared per-project
# state another host may have pushed, which is what the selection below reads.
if git remote 2>/dev/null | grep -q .; then
  # --rebase=merges keeps an unpushed trunk merge a merge; a plain rebase replays
  # every trunk commit onto the port branch. A rebase that stops on a conflict is
  # aborted, so the checkout is never left mid-rebase for the next command.
  if ! git pull --rebase=merges --autostash >/dev/null 2>&1; then
    if [ -d "$(git rev-parse --git-path rebase-merge)" ] \
        || [ -d "$(git rev-parse --git-path rebase-apply)" ]; then
      git rebase --abort >/dev/null 2>&1
      echo "orient: pull --rebase hit conflicts and was ABORTED -- this checkout is behind origin; merge it by hand" >&2
    else
      echo "orient: pull --rebase skipped (offline, or nothing to pull from)" >&2
    fi
  fi
fi

# The trunk carries the tooling, the agent definitions and the porting knowledge, and
# a port branch keeps whatever those looked like the day it was cut. Merging on every
# trunk push would put a merge commit on every branch for a README regeneration, so
# look first and merge only when something a port can actually feel has moved. No-op
# on the trunk itself.
sync_err=$(mktemp)
SYNC=$("$PY" utils/moatlib.py branch-sync --apply 2>"$sync_err")
case "$SYNC" in
  *"skip --"*|*"current --"*) ;;
  *"conflict --"*) echo "${SYNC/branch-sync: /branch   : }" >&2 ;;
  "")
    # No verdict at all means the tool itself failed, which is not "in sync".
    echo "orient: WARNING -- branch-sync produced no verdict; this branch may be behind the trunk:" >&2
    tail -n 3 "$sync_err" | sed 's/^/           /' >&2 ;;
  *) echo "${SYNC/branch-sync: /branch   : }" ;;
esac
rm -f "$sync_err"

echo "== MOAT orient =="

# Standing upkeep and the nags below need no GPU, so they run BEFORE platform
# detection: a host whose GPU vanished (or that never had one) still releases
# forks and still sees approved ports waiting on publish, suggested waivers and
# unruled deferrals. Exiting on detection failure used to skip all of it, which
# made a broken ROCm install read exactly like "nothing waiting".
#
# Every tool below runs through moat_run: its stdout is what gets formatted, and a
# non-zero exit prints one `orient: <tool> FAILED` line instead of nothing.

# Releases projects whose fork has appeared, wherever the record lives. A write to
# a project branch, so it belongs in a session rather than a scheduled job.
# Declines are NOT recorded here -- labels record nothing; intake_queue.py apply
# --decline carries a person's answer and is the only route.
# Not silenced: this is the only thing that says a fork appeared for a project whose
# branch nobody has checked out, and discarding it made "no one has to notice by hand"
# false. Only the routine "nothing to do" lines are dropped.
FORKS=$(moat_run "upstream.py --forks --apply" "$PY" utils/upstream.py --forks --apply)
printf '%s\n' "$FORKS" | grep -E "RELEASED|ADVANCED|WAITING" | sed 's/^/forks    : /' || true

# Publishing is a project-level step and belongs to nobody's architecture, so the
# per-arch selector below will never surface it. It comes first because it is cheap,
# finished work waiting on one command -- and because a port sitting approved but
# unsubmitted is the most wasteful state MOAT has.
PUBLISH=$(moat_run "upstream.py --publish" "$PY" utils/upstream.py --publish)
READY=$(printf '%s\n' "$PUBLISH" | sed -n 's/^  READY *\([^ ]*\).*/\1/p' | tr '\n' ' ')
if [ -n "${READY// /}" ]; then
  echo "approved : ${READY}-- submit upstream: $PY utils/upstream.py --publish --apply --name <project>"
  echo "           (opens the PR with the approved title and body, then closes the review PR)"
fi
# A review PR that could not be READ is not a port with nothing to submit, and this
# line is the only place anyone would notice the difference: the grep above finds no
# READY either way, so an outage used to look exactly like an empty queue.
UNREACHABLE=$(printf '%s\n' "$PUBLISH" | sed -n 's/^  UNREACHABLE *\([^ ]*\).*/\1/p' | tr '\n' ' ')
if [ -n "${UNREACHABLE// /}" ]; then
  echo "unknown  : ${UNREACHABLE}-- review PR(s) unreadable, approval state UNKNOWN (not 'nothing to submit')"
fi

# A release line approved on its in-fork review PR is finished work waiting on one
# command, exactly like an approved port, so it is surfaced the same way.
FINISH=$(moat_run "upstream.py --finish-release" "$PY" utils/upstream.py --finish-release)
RELEASE=$(printf '%s\n' "$FINISH" | sed -n 's/^  \(READY\|RESUME\) *\([^ ]*\).*/\2/p' | tr '\n' ' ')
if [ -n "${RELEASE// /}" ]; then
  echo "approved : ${RELEASE}-- finish the release: $PY utils/upstream.py --finish-release --apply"
  echo "           (tags the approved commit, freezes the branch, closes the review PR unmerged)"
fi

# A suggested waiver blocks its project's PR and only a person can clear it, so it has
# to be visible from a session that is not the one that suggested it. The porter that
# found the obstacle may have run unattended hours ago; without this the case sits in
# the file and the port sits finished-but-unsubmittable, which is the state MOAT is
# worst at noticing.
WAIVERS=$(moat_run "moatlib.py waivers" "$PY" utils/moatlib.py waivers)
printf '%s\n' "$WAIVERS" \
  | awk -F'\t' 'NF>1 {printf "waiver   :   AWAITING   %-26s %s -- %.70s\n", $1, $2, $4}' || true

# Lessons are written in project notes (a port branch may not edit a skill) and promoted
# to the skill on the trunk after the port's review. Only the count is shown here;
# `lessons` lists them. ORPHANED is older: a skill edit made on a now-finished port
# branch before branches were limited to their own folder.
LESSONS=$(moat_run "moatlib.py lessons" "$PY" utils/moatlib.py lessons)
NL=$(printf '%s\n' "$LESSONS" | awk -F'\t' '$2=="CANDIDATE"' | grep -c . || true)
[ "$NL" -gt 0 ] && echo "lesson   :   $NL in project notes await promotion: $PY utils/moatlib.py lessons"
printf '%s\n' "$LESSONS" \
  | awk -F'\t' '$2=="ORPHANED" {printf "lesson   :   ORPHANED   %-24s %.60s\n", $1, $3}' || true

# A folder in the wrong place is invisible until it bites, and it bites differently at
# each end: one on the trunk with work outstanding turns every status write into a pull
# request once the trunk is protected, and one on a branch with nothing left owed is a
# branch nobody will ever delete. Placement follows state and state moves, so this is a
# standing check rather than a migration step -- a maintainer asking for a rewrite after
# the PR merged puts a finished project back in flight.
MISPLACED=$(moat_run "moatlib.py misplaced" "$PY" utils/moatlib.py misplaced)
printf '%s\n' "$MISPLACED" \
  | awk -F'\t' 'NF>1 {printf "misplaced:   %-26s %s\n", $1, $3}' || true

# A dependent's porter builds each dep from the provider's "## Install as a
# dependency" recipe (DEPENDENCIES.md makes the section a MUST). Nothing verified it
# existed, so a missing one surfaced only at the moment a porter needed it.
DEPDOC=$(moat_run "moatlib.py dep-doc-gaps" "$PY" utils/moatlib.py dep-doc-gaps)
printf '%s\n' "$DEPDOC" \
  | awk -F'\t' 'NF>1 {printf "dep-doc  :   MISSING    %-26s needed by %s\n", $1, $2}' || true

# Work somebody set aside without anybody deciding it should be. A deferral is cheap to
# record and easy to forget, and the failure is silent: "we will get to it" becomes
# nobody ever looked. Only a person can rule on one, so the list has to reach a person.
#
# Print the ID, because that is the argument `decide` takes. The first cut printed the
# project, the kind and the summary -- everything except the one field you need to act
# -- so the nag named a command you then had to go look up the argument for. The count
# is here for the same reason: the list is sorted, so a bare `head` shows the same five
# forever and gives no sign that seventy others are behind them.
DEFERRED=$(moat_run "deferred.py pending" "$PY" utils/deferred.py pending)
if [ -n "$DEFERRED" ]; then
  echo "$DEFERRED" | awk -F'\t' 'NF>2 {printf "deferred :   UNRULED    %-34s %.48s\n", $3, $4}' \
    | head -n 5
  N=$(printf '%s\n' "$DEFERRED" | grep -c . || true)
  [ "$N" -gt 5 ] && echo "deferred :   ...and $((N - 5)) more: $PY utils/deferred.py pending"
  echo "deferred :   rule it:   $PY utils/deferred.py decide <id> --choice defer|now --by <who>"
fi

# Nothing reconciles the record on a schedule, so the only thing that can say it has
# gone unchecked is how long since someone swept. Reads a stored date, costs no API
# call, and names the command rather than making anyone remember it.
STALE=$(moat_run "the reconciled-age check" "$PY" -c '
import sys; sys.path.insert(0, "utils")
import upstream
d = upstream.reconciled_age_days(upstream.TODAY)
if d is None:
    print("never reconciled -- run /moat-checkup")
elif d >= 14:
    print(f"last reconciled {d} days ago -- run /moat-checkup")
')
[ -n "$STALE" ] && echo "records  : $STALE"

# Selection below is per-arch and needs a platform. Everything above already ran,
# so a host with no detectable GPU still did the upkeep and showed the nags.
# detect_arch's stderr is reprinted rather than discarded. Swallowing it made
# every failure read as "this machine has no AMD GPU", including the ones where
# the GPU was fine and only the tool that reports it could not be found -- a
# healthy host looked dead, and the message named no way to tell them apart.
detect_err=$(mktemp)
if ! arch_out=$(bash utils/detect_arch.sh 2>"$detect_err"); then
  echo "platform : UNKNOWN (arch detection failed)"
  sed 's/^/           /' "$detect_err"
  rm -f "$detect_err"
  echo "next     : NONE -- per-arch dispatch needs a platform"
  echo "hint     : GPU-independent stages (intake, planning, review) can still be"
  echo "           worked by setting MOAT_PLATFORM=<os>-<gfx> to bypass detection;"
  echo "           building and validating still need real hardware"
  exit 0
fi
rm -f "$detect_err"
eval "$arch_out"   # sets GFX_ARCH GFX_TRIPLE PLATFORM

echo "platform : $PLATFORM (gfx=$GFX_ARCH)"
# Any AMD GPU is a platform. What it needs is a known wavefront width, since that
# is the one property the name does not carry; moatlib says so and names the fix.
# A check that could not run is an error, not a platform with nothing wrong.
if ! PROBLEM=$(moat_run "the platform check" "$PY" -c 'import sys; sys.path.insert(0, "utils"); import moatlib; print(moatlib.platform_problem(sys.argv[1]) or "", end="")' "$PLATFORM"); then
  echo "next     : ERROR -- could not check the platform; this is NOT 'nothing to do'"
  exit 1
fi
if [ -n "$PROBLEM" ]; then
  echo "next     : NONE -- $PROBLEM"
  exit 0
fi

# Nothing here serializes two CLIs on one host. A lock file used to be taken at this
# point, but it was a no-op wherever `flock` is missing (Git Bash) and was released
# when this script exited, long before either CLI wrote a claim. What actually keeps
# two sessions off one project is the shared work lock moatlib takes when a role
# enters its stage: the second one to try is told who holds it and stops.

# Control-plane mode never dispatches a port. The trunk only carries projects in
# terminal states plus stubs a branch may have moved past, so a dispatch line
# printed here offers work that is stale or already claimed -- the override is for
# tooling and docs sessions that want the platform detection, nothing more.
case "$_branch" in
  main|master)
    echo "next     : NONE -- control-plane mode on '$_branch'; ports dispatch from a port/<name> branch"
    FLEET=$(moat_run "moatlib.py fleet" "$PY" utils/moatlib.py fleet "$PLATFORM")
    printf '%s\n' "$FLEET" | moat_fleet_checkouts
    exit 0 ;;
esac

# The selector failing is not the selector finding nothing. It prints NONE and
# exits 0 when there is no work; any other exit is a broken tool, and reporting
# that as "nothing to do" sends a host home with work waiting.
next_err=$(mktemp)
if ! NEXT=$("$PY" utils/moatlib.py next-task "$PLATFORM" 2>"$next_err") || [ -z "$NEXT" ]; then
  echo "next     : ERROR -- the selector failed; this is NOT 'nothing to do'"
  tail -n 5 "$next_err" | sed 's/^/           /'
  rm -f "$next_err"
  exit 1
fi
# The selector succeeded, and may still have said something on stderr: the work it
# offers is under a lock another host or job of this platform took.
LOCKWARN=$(moat_warnings "$next_err")
rm -f "$next_err"
if [ "$NEXT" = "NONE" ]; then
  echo "next     : NONE actionable on $PLATFORM"
  # "Nothing to do" and "waiting on a dependency nobody adopted" print the same way
  # otherwise, and the second one never resolves by itself.
  BLOCKED=$(moat_run "moatlib.py dep-blocked" "$PY" utils/moatlib.py dep-blocked "$PLATFORM")
  if [ -n "$BLOCKED" ]; then
    echo "blocked  : held only by a dependency --"
    printf '%s\n' "$BLOCKED" | while IFS=$'\t' read -r proj dep verdict detail; do
      case "$verdict" in
        unknown) fix="needs intake: $PY utils/port_request.py file <owner/repo> --blocks $proj" ;;
        doomed)  fix="$dep will not be ported ($detail); scope $proj around it or disposition it" ;;
        *)       fix="$dep is $detail; it clears on its own" ;;
      esac
      echo "           $proj <- $dep ($verdict) -- $fix"
    done
  fi
  # A project whose folder lives on its own branch is invisible to next-task here,
  # and correctly so -- you cannot work files that are not in your tree. But that
  # must not read as "there is nothing to do", so name it and say where it is.
  FLEET=$(moat_run "moatlib.py fleet" "$PY" utils/moatlib.py fleet "$PLATFORM")
  ELSEWHERE=$(printf '%s\n' "$FLEET" | moat_fleet_checkouts)
  if [ -n "$ELSEWHERE" ]; then
    echo "elsewhere: actionable on another branch --"
    printf '%s\n' "$ELSEWHERE"
  fi
  echo "hint     : adopt a project from data/candidates.json:"
  echo "           $PY utils/moatlib.py scaffold <owner/repo>"
  exit 0
fi

read -r PROJECT STATE STAGE < <(echo "$NEXT" | "$PY" -c \
  'import sys,json;d=json.load(sys.stdin);print(d["project"],d["state"],d["stage"])')

echo "next     : projects/$PROJECT  state=$STATE  -> dispatch: $STAGE"
if [ -n "$LOCKWARN" ]; then
  printf '%s\n' "$LOCKWARN" | sed 's/^/lock     : WARNING -- /'
fi
echo "triple   : $GFX_TRIPLE"
echo "action   : Use the $STAGE role on projects/$PROJECT for $PLATFORM"
echo "           (read AGENTS.md Pipeline + Human decisions first)"
