#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Time one phase of work and append a record to projects/<project>/stats.jsonl.
# Usage: utils/timeit.sh <project> <phase> -- <command...>
# Phase is conventionally compile|test|misc; any label is kept and reported
# (validators use cuda-compile for the CUDA-baseline build, for example).
# Exits with the wrapped command's exit code so it is transparent in scripts/CI.
#
# Needs nothing but bash and the Python the rest of MOAT runs under. It used to need
# `jq` and GNU `date +%s.%N`; on a host without jq the command ran and the record
# was silently dropped.
set -uo pipefail
cd "$(dirname "$0")/.."
. utils/lib.sh
proj="${1:?project}"; phase="${2:?phase}"; shift 2
[ "${1:-}" = "--" ] && shift
out="projects/${proj}/stats.jsonl"
# A project's folder is created by `scaffold`, never here: `mkdir -p` turned a
# mistyped name into a new projects/<typo>/ that then looked like a project.
if [ ! -d "projects/${proj}" ]; then
  echo "timeit: no projects/${proj} in this checkout (mistyped name, or its branch is not checked out); nothing was run" >&2
  exit 2
fi
if ! moat_resolve_py; then
  moat_no_py timeit
  exit 2
fi
start=$(moat_now)
"$@"; code=$?
end=$(moat_now)
# The command is recorded for provenance, not for replay, into a file this repo
# publishes -- so no home directory goes in it. Callers pass whole environments through
# here (`timeit.sh p compile -- env PATH=... cmake ...`), and a Windows PATH carries the
# account name in every entry; that is how a real username reached 121 places in this
# repo before anyone looked. Redacted at the point of writing rather than trusted to
# every caller, because the caller that forgets is the one nobody reviews.
# MSYS2_ARG_CONV_EXCL: Git Bash rewrites arguments that look like POSIX paths when it
# starts a native program, which would rewrite the recorded command line.
if ! MSYS2_ARG_CONV_EXCL='*' "$PY" - "$out" "$phase" "$start" "$end" "$code" "$*" <<'PYEOF'
import datetime, json, re, sys
out, phase, start, end, code, cmd = sys.argv[1:7]
cmd = re.sub(r'([Uu]sers)[\\/]+[^\\/\s"]+', r"\1/<user>", cmd)
cmd = re.sub(r'/home/[^/\s"]+', "/home/<user>", cmd)
rec = {"kind": "phase",
       "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "phase": phase, "seconds": round(float(end) - float(start), 3),
       "exit": int(code), "cmd": cmd}
with open(out, "a", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
PYEOF
then
  echo "timeit: the command ran (exit $code) but its record could NOT be written to $out" >&2
fi
exit "$code"
