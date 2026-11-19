#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Mark session boundaries in a project's stats.jsonl.
# Usage:
#   utils/session.sh <project> <platform> start|end
#
# start/end pairs give session wall-clock, which statlib.py sums.
#
# Tokens are deliberately NOT recorded here. Only the orchestrator may receive a
# child agent's token count when its harness reports one, so
# `moatlib.py record-tokens` owns that record kind. This script used to write
# {kind:"tokens"} as well, which left two different shapes in the same file.
#
# Needs nothing but bash and the Python the rest of MOAT runs under (it used to
# need `jq` and GNU `date +%s.%N`, and wrote nothing without them).
set -uo pipefail
cd "$(dirname "$0")/.."
. utils/lib.sh
proj="${1:?project}"; platform="${2:?platform}"; event="${3:?start|end}"
out="projects/${proj}/stats.jsonl"
case "$event" in
  start|end) ;;
  *) echo "session.sh: unknown event $event (expected start|end)" >&2; exit 2;;
esac
# A project's folder is created by `scaffold`, never here (see timeit.sh).
if [ ! -d "projects/${proj}" ]; then
  echo "session.sh: no projects/${proj} in this checkout (mistyped name, or its branch is not checked out)" >&2
  exit 2
fi
if ! moat_resolve_py; then
  moat_no_py session.sh
  exit 2
fi
# MSYS2_ARG_CONV_EXCL: Git Bash rewrites arguments that look like POSIX paths when it
# starts a native program, which would rewrite the recorded command line.
if ! MSYS2_ARG_CONV_EXCL='*' "$PY" - "$out" "$event" "$platform" <<'PYEOF'
import datetime, json, sys, time
out, event, platform = sys.argv[1:4]
rec = {"kind": "session",
       "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "epoch": round(time.time(), 6), "event": event, "platform": platform}
with open(out, "a", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
PYEOF
then
  echo "session.sh: could NOT write the $event record to $out" >&2
  exit 1
fi
