#!/usr/bin/env bash
# Mark session boundaries in a project's stats.jsonl.
# Usage:
#   utils/session.sh <project> <platform> start|end
#
# start/end pairs give session wall-clock, which statlib.py sums.
#
# Tokens are deliberately NOT recorded here. Only the orchestrator can see a
# subagent's token count (it arrives in the completion notification), so
# `moatlib.py record-tokens` owns that record kind. This script used to write
# {kind:"tokens"} as well, which left two different shapes in the same file.
set -uo pipefail
cd "$(dirname "$0")/.."
proj="${1:?project}"; platform="${2:?platform}"; event="${3:?start|end}"
out="projects/${proj}/stats.jsonl"; mkdir -p "projects/${proj}"
ts=$(date -u +%FT%TZ); epoch=$(date +%s.%N)
case "$event" in
  start|end)
    jq -cn --arg ts "$ts" --argjson epoch "$epoch" --arg ev "$event" --arg p "$platform" \
       '{kind:"session",ts:$ts,epoch:$epoch,event:$ev,platform:$p}' >> "$out" ;;
  *) echo "session.sh: unknown event $event (expected start|end)" >&2; exit 2;;
esac
