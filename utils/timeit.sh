#!/usr/bin/env bash
# Time one phase of work and append a record to projects/<project>/stats.jsonl.
# Usage: utils/timeit.sh <project> <phase> -- <command...>   (phase: compile|test|misc)
# Exits with the wrapped command's exit code so it is transparent in scripts/CI.
set -uo pipefail
cd "$(dirname "$0")/.."
proj="${1:?project}"; phase="${2:?phase}"; shift 2
[ "${1:-}" = "--" ] && shift
out="projects/${proj}/stats.jsonl"
mkdir -p "projects/${proj}"
start=$(date +%s.%N)
"$@"; code=$?
end=$(date +%s.%N)
secs=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')
# The command is recorded for provenance, not for replay, into a file this repo
# publishes -- so no home directory goes in it. Callers pass whole environments through
# here (`timeit.sh p compile -- env PATH=... cmake ...`), and a Windows PATH carries the
# account name in every entry; that is how a real username reached 121 places in this
# repo before anyone looked. Redacted at the point of writing rather than trusted to
# every caller, because the caller that forgets is the one nobody reviews.
jq -cn --arg ts "$(date -u +%FT%TZ)" --arg phase "$phase" \
   --argjson secs "$secs" --argjson exit "$code" --arg cmd "$*" \
   '{kind:"phase",ts:$ts,phase:$phase,seconds:$secs,exit:$exit,
     cmd:($cmd
          | gsub("(?<u>[Uu]sers)[\\\\/]+(?<n>[^\\\\/\\s\"]+)"; "\(.u)/<user>")
          | gsub("/home/(?<n>[^/\\s\"]+)"; "/home/<user>"))}' >> "$out"
exit "$code"
