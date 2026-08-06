#!/usr/bin/env bash
# MOAT entrypoint. Pull the latest MOAT state, detect this host's AMD arch, pick
# the single next project + stage for this platform, and print a dispatch
# summary. Read-only on state except an advisory claim and follower-unblock
# bookkeeping. Run this (or /port-next) when starting a CLI in the MOAT repo.
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
if git remote 2>/dev/null | grep -q .; then
  git pull --rebase --autostash >/dev/null 2>&1 \
    || echo "orient: pull --rebase skipped (offline or local conflicts)" >&2
fi

if ! arch_out=$(bash utils/detect_arch.sh 2>/dev/null); then
  echo "== MOAT orient =="
  echo "platform : UNKNOWN (no AMD GPU detected)"
  echo "next     : NONE"
  exit 0
fi
eval "$arch_out"   # sets GFX_ARCH GFX_TRIPLE PLATFORM

echo "== MOAT orient =="
echo "platform : $PLATFORM (gfx=$GFX_ARCH)"
# Any AMD GPU is a platform. What it needs is a known wavefront width, since that
# is the one property the name does not carry; moatlib says so and names the fix.
if PROBLEM=$(python3 -c 'import sys; sys.path.insert(0, "utils"); import moatlib; print(moatlib.platform_problem(sys.argv[1]) or "", end="")' "$PLATFORM") && [ -n "$PROBLEM" ]; then
  echo "next     : NONE -- $PROBLEM"
  exit 0
fi

# Releases projects whose fork has appeared AND records any decline label, which
# `release-forks` alone does not. Both are writes to a project branch, so they
# belong in a session rather than in a scheduled job.
python3 utils/upstream.py --forks --apply >/dev/null 2>&1 || true
python3 utils/moatlib.py unblock-followers >/dev/null 2>&1 || true

# Serialize select+claim so two same-host CLIs never grab the same project.
exec 9>"projects/.selection.lock"
if command -v flock >/dev/null 2>&1; then flock -w 10 9 || true; fi

# Publishing is a project-level step and belongs to nobody's architecture, so the
# per-arch selector below will never surface it. It comes first because it is cheap,
# finished work waiting on one command -- and because a port sitting approved but
# unsubmitted is the most wasteful state MOAT has.
READY=$(python3 utils/upstream.py --publish 2>/dev/null | sed -n 's/^  READY *\([^ ]*\).*/\1/p' | tr '\n' ' ')
if [ -n "${READY// /}" ]; then
  echo "approved : ${READY}-- submit upstream: python3 utils/upstream.py --publish --apply"
  echo "           (opens the PR with the approved title and body, then closes the review PR)"
fi

NEXT=$(python3 utils/moatlib.py next-task "$PLATFORM" 2>/dev/null || echo NONE)
if [ "$NEXT" = "NONE" ] || [ -z "$NEXT" ]; then
  echo "next     : NONE actionable on $PLATFORM"
  echo "hint     : adopt a project from data/candidates.json:"
  echo "           python3 utils/moatlib.py scaffold <owner/repo>"
  exit 0
fi

read -r PROJECT STATE STAGE < <(echo "$NEXT" | python3 -c \
  'import sys,json;d=json.load(sys.stdin);print(d["project"],d["state"],d["stage"])')

# Advisory claim: we hold the selection lock and next-task already excluded
# live-claimed projects. The dispatched CLI should refresh this .claim while
# working so it stays live (heartbeat); a stale .claim is reclaimable.
printf '{"host":"%s","pid":%s,"platform":"%s","started":"%s"}\n' \
  "$(hostname)" "$$" "$PLATFORM" "$(date -u +%FT%TZ)" > "projects/$PROJECT/.claim"

echo "next     : projects/$PROJECT  state=$STATE  -> dispatch: $STAGE"
echo "triple   : $GFX_TRIPLE"
echo "action   : Use the $STAGE subagent on projects/$PROJECT"
echo "           (read CLAUDE.md Pipeline + Autonomy boundary first)"
