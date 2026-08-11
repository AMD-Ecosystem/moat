#!/usr/bin/env bash
# MOAT entrypoint. Pull the latest MOAT state, detect this host's AMD arch, pick
# the single next project + stage for this platform, and print a dispatch
# summary. Standing upkeep on the way writes and pushes: hook/merge-driver
# registration, the trunk-into-branch sync merge, and fork releases. Selection
# itself is read-only. Run this (or /port-next) when starting a CLI in the MOAT repo.
set -uo pipefail
cd "$(dirname "$0")/.."

bash utils/setup_git.sh >/dev/null 2>&1 || true
python3 utils/install_hooks.py >/dev/null 2>&1 || true

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
  git pull --rebase --autostash >/dev/null 2>&1 \
    || echo "orient: pull --rebase skipped (offline or local conflicts)" >&2
fi

# The trunk carries the tooling, the agent definitions and the porting knowledge, and
# a port branch keeps whatever those looked like the day it was cut. Merging on every
# trunk push would put a merge commit on every branch for a README regeneration, so
# look first and merge only when something a port can actually feel has moved. No-op
# on the trunk itself.
SYNC=$(python3 utils/moatlib.py branch-sync --apply 2>/dev/null)
case "$SYNC" in
  *"skip --"*|*"current --"*) ;;
  *"conflict --"*) echo "${SYNC/branch-sync: /branch   : }" >&2 ;;
  "") ;;
  *) echo "${SYNC/branch-sync: /branch   : }" ;;
esac

echo "== MOAT orient =="

# Standing upkeep and the nags below need no GPU, so they run BEFORE platform
# detection: a host whose GPU vanished (or that never had one) still releases
# forks and still sees approved ports waiting on publish, suggested waivers and
# unruled deferrals. Exiting on detection failure used to skip all of it, which
# made a broken ROCm install read exactly like "nothing waiting".

# Releases projects whose fork has appeared, wherever the record lives. A write to
# a project branch, so it belongs in a session rather than a scheduled job.
# Declines are NOT recorded here -- labels record nothing; intake_queue.py apply
# --decline carries a person's answer and is the only route.
# Not silenced: this is the only thing that says a fork appeared for a project whose
# branch nobody has checked out, and discarding it made "no one has to notice by hand"
# false. Only the routine "nothing to do" lines are dropped.
python3 utils/upstream.py --forks --apply 2>/dev/null \
  | grep -E "RELEASED|ADVANCED|WAITING" | sed 's/^/forks    : /' || true

# Publishing is a project-level step and belongs to nobody's architecture, so the
# per-arch selector below will never surface it. It comes first because it is cheap,
# finished work waiting on one command -- and because a port sitting approved but
# unsubmitted is the most wasteful state MOAT has.
PUBLISH=$(python3 utils/upstream.py --publish 2>/dev/null)
READY=$(printf '%s\n' "$PUBLISH" | sed -n 's/^  READY *\([^ ]*\).*/\1/p' | tr '\n' ' ')
if [ -n "${READY// /}" ]; then
  echo "approved : ${READY}-- submit upstream: python3 utils/upstream.py --publish --apply"
  echo "           (opens the PR with the approved title and body, then closes the review PR)"
fi
# A review PR that could not be READ is not a port with nothing to submit, and this
# line is the only place anyone would notice the difference: the grep above finds no
# READY either way, so an outage used to look exactly like an empty queue.
UNREACHABLE=$(printf '%s\n' "$PUBLISH" | sed -n 's/^  UNREACHABLE *\([^ ]*\).*/\1/p' | tr '\n' ' ')
if [ -n "${UNREACHABLE// /}" ]; then
  echo "unknown  : ${UNREACHABLE}-- review PR(s) unreadable, approval state UNKNOWN (not 'nothing to submit')"
fi

# A suggested waiver blocks its project's PR and only a person can clear it, so it has
# to be visible from a session that is not the one that suggested it. The porter that
# found the obstacle may have run unattended hours ago; without this the case sits in
# the file and the port sits finished-but-unsubmittable, which is the state MOAT is
# worst at noticing.
python3 utils/moatlib.py waivers 2>/dev/null \
  | awk -F'\t' 'NF>1 {printf "waiver   :   AWAITING   %-26s %s -- %.70s\n", $1, $2, $4}' || true

# A lesson on a live port branch is where it belongs -- the port's own review is what
# publishes it -- so this reports only the case nothing will ever carry: a FINISHED
# port whose branch still holds a global edit the trunk lacks. That branch is about to
# be deleted. `lessons --pending` shows the rest, and they are not to be lifted.
python3 utils/moatlib.py lessons 2>/dev/null \
  | awk -F'\t' '$2=="ORPHANED" {printf "lesson   :   ORPHANED   %-24s %.60s\n", $1, $3}' || true

# A folder in the wrong place is invisible until it bites, and it bites differently at
# each end: one on the trunk with work outstanding turns every status write into a pull
# request once the trunk is protected, and one on a branch with nothing left owed is a
# branch nobody will ever delete. Placement follows state and state moves, so this is a
# standing check rather than a migration step -- a maintainer asking for a rewrite after
# the PR merged puts a finished project back in flight.
python3 utils/moatlib.py misplaced 2>/dev/null \
  | awk -F'\t' 'NF>1 {printf "misplaced:   %-26s %s\n", $1, $3}' || true

# A dependent's porter builds each dep from the provider's "## Install as a
# dependency" recipe (DEPENDENCIES.md makes the section a MUST). Nothing verified it
# existed, so a missing one surfaced only at the moment a porter needed it.
python3 utils/moatlib.py dep-doc-gaps 2>/dev/null \
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
DEFERRED=$(python3 utils/deferred.py pending 2>/dev/null || true)
if [ -n "$DEFERRED" ]; then
  echo "$DEFERRED" | awk -F'\t' 'NF>2 {printf "deferred :   UNRULED    %-34s %.48s\n", $3, $4}' \
    | head -5
  N=$(printf '%s\n' "$DEFERRED" | grep -c . || true)
  [ "$N" -gt 5 ] && echo "deferred :   ...and $((N - 5)) more: python3 utils/deferred.py pending"
  echo "deferred :   rule it:   python3 utils/deferred.py decide <id> --choice defer|now --by <who>"
fi

# Nothing reconciles the record on a schedule, so the only thing that can say it has
# gone unchecked is how long since someone swept. Reads a stored date, costs no API
# call, and names the command rather than making anyone remember it.
STALE=$(python3 -c '
import sys; sys.path.insert(0, "utils")
import upstream
d = upstream.reconciled_age_days(upstream.TODAY)
if d is None:
    print("never reconciled -- run /moat-checkup")
elif d >= 14:
    print(f"last reconciled {d} days ago -- run /moat-checkup")
' 2>/dev/null)
[ -n "$STALE" ] && echo "records  : $STALE"

# Selection below is per-arch and needs a platform. Everything above already ran,
# so a host with no detectable GPU still did the upkeep and showed the nags.
if ! arch_out=$(bash utils/detect_arch.sh 2>/dev/null); then
  echo "platform : UNKNOWN (no AMD GPU detected)"
  echo "next     : NONE -- per-arch dispatch needs a platform"
  echo "hint     : GPU-independent stages (intake, planning, review) can still be"
  echo "           worked by setting MOAT_PLATFORM=<os>-<gfx> to bypass detection;"
  echo "           building and validating still need real hardware"
  exit 0
fi
eval "$arch_out"   # sets GFX_ARCH GFX_TRIPLE PLATFORM

echo "platform : $PLATFORM (gfx=$GFX_ARCH)"
# Any AMD GPU is a platform. What it needs is a known wavefront width, since that
# is the one property the name does not carry; moatlib says so and names the fix.
if PROBLEM=$(python3 -c 'import sys; sys.path.insert(0, "utils"); import moatlib; print(moatlib.platform_problem(sys.argv[1]) or "", end="")' "$PLATFORM") && [ -n "$PROBLEM" ]; then
  echo "next     : NONE -- $PROBLEM"
  exit 0
fi

# Serialize select+claim so two same-host CLIs never grab the same project.
exec 9>"projects/.selection.lock"
if command -v flock >/dev/null 2>&1; then flock -w 10 9 || true; fi

# Control-plane mode never dispatches a port. The trunk only carries projects in
# terminal states plus stubs a branch may have moved past, so a dispatch line
# printed here offers work that is stale or already claimed -- the override is for
# tooling and docs sessions that want the platform detection, nothing more.
case "$_branch" in
  main|master)
    echo "next     : NONE -- control-plane mode on '$_branch'; ports dispatch from a port/<name> branch"
    python3 utils/moatlib.py fleet "$PLATFORM" 2>/dev/null | awk -F'\t' '$2=="branch"' \
      | while IFS=$'\t' read -r proj where state stage branch; do
          echo "           $proj ($state -> $stage): git checkout ${branch:-port/$proj}"
        done
    exit 0 ;;
esac

NEXT=$(python3 utils/moatlib.py next-task "$PLATFORM" 2>/dev/null || echo NONE)
if [ "$NEXT" = "NONE" ] || [ -z "$NEXT" ]; then
  echo "next     : NONE actionable on $PLATFORM"
  # "Nothing to do" and "waiting on a dependency nobody adopted" print the same way
  # otherwise, and the second one never resolves by itself.
  BLOCKED=$(python3 utils/moatlib.py dep-blocked "$PLATFORM" 2>/dev/null)
  if [ -n "$BLOCKED" ]; then
    echo "blocked  : held only by a dependency --"
    printf '%s\n' "$BLOCKED" | while IFS=$'\t' read -r proj dep verdict detail; do
      case "$verdict" in
        unknown) fix="needs intake: python3 utils/port_request.py file <owner/repo> --blocks $proj" ;;
        doomed)  fix="$dep will not be ported ($detail); scope $proj around it or disposition it" ;;
        *)       fix="$dep is $detail; it clears on its own" ;;
      esac
      echo "           $proj <- $dep ($verdict) -- $fix"
    done
  fi
  # A project whose folder lives on its own branch is invisible to next-task here,
  # and correctly so -- you cannot work files that are not in your tree. But that
  # must not read as "there is nothing to do", so name it and say where it is.
  ELSEWHERE=$(python3 utils/moatlib.py fleet "$PLATFORM" 2>/dev/null | awk -F'\t' '$2=="branch"')
  if [ -n "$ELSEWHERE" ]; then
    echo "elsewhere: actionable on another branch --"
    # The 5th field is the branch as the remote spells it (port/hami-core for
    # HAMi-core); reconstructing port/$proj here printed checkouts that failed.
    printf '%s\n' "$ELSEWHERE" | while IFS=$'\t' read -r proj where state stage branch; do
      echo "           $proj ($state -> $stage): git checkout ${branch:-port/$proj}"
    done
  fi
  echo "hint     : adopt a project from data/candidates.json:"
  echo "           python3 utils/moatlib.py scaffold <owner/repo>"
  exit 0
fi

read -r PROJECT STATE STAGE < <(echo "$NEXT" | python3 -c \
  'import sys,json;d=json.load(sys.stdin);print(d["project"],d["state"],d["stage"])')

echo "next     : projects/$PROJECT  state=$STATE  -> dispatch: $STAGE"
echo "triple   : $GFX_TRIPLE"
echo "action   : Use the $STAGE role on projects/$PROJECT for $PLATFORM"
echo "           (read AGENTS.md Pipeline + Human decisions first)"
