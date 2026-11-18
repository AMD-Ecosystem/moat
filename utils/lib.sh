#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Shared by the shell entry points (orient.sh, setup_git.sh, timeit.sh, session.sh).
# Source it; it defines functions and runs nothing.

# The one Python every shell entry point uses, in PY.
#
# The name `python3` is not MOAT's interpreter on every host. On one Windows host it
# is an unrelated virtual environment with none of the tooling's dependencies, and
# every tool orient ran under it crashed with its error discarded, so orient printed
# almost nothing and read as "nothing waiting". So the interpreter is chosen once,
# by what it can do rather than by its name: the first of python3, python that can
# import the tooling's dependencies. MOAT_PYTHON names one explicitly.
moat_resolve_py() {
  local c
  if [ -n "${MOAT_PYTHON:-}" ]; then
    PY=$MOAT_PYTHON
    return 0
  fi
  for c in python3 python; do
    if "$c" -c 'import jsonschema, tomllib' >/dev/null 2>&1; then
      PY=$c
      return 0
    fi
  done
  PY=""
  return 1
}

moat_no_py() {
  echo "$1: no Python that can run MOAT's tools (needs 3.11+ and jsonschema)." >&2
  echo "$1: tried python3 and python on PATH; set MOAT_PYTHON to the interpreter" >&2
  echo "$1: MOAT uses on this host." >&2
}

# moat_run <label> <command...>
# Run a tool with its stdout passed through. A non-zero exit is reported in one line
# on stderr, with the last thing the tool said. Callers used to discard stderr and the
# exit status together (`2>/dev/null`, `|| true`), which made a tool that crashed
# indistinguishable from one that had nothing to report.
moat_run() {
  local label=$1 err rc
  shift
  err=$(mktemp)
  "$@" 2>"$err"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "orient: $label FAILED (exit $rc): $(grep -v '^[[:space:]]*$' "$err" | tail -n 1)" >&2
  fi
  rm -f "$err"
  return "$rc"
}

# moat_gh_guard_problem
# Silent and 0 when `gh` in this shell is the guard launcher, or when there is no gh
# at all (a validation-only host never calls it). Otherwise prints why and how to fix
# it, and returns 1. The guard is a PATH shim, so a real gh that sorts ahead of it
# leaves every check it makes unrun -- and nothing said so until a push-time gate.
moat_gh_guard_problem() {
  local gh
  gh=$(command -v gh 2>/dev/null) || return 0
  if [ -f "$gh" ] && grep -q "moat-gh-guard v" "$gh" 2>/dev/null; then
    return 0
  fi
  echo "orient: \`gh\` resolves to $gh, which is NOT the MOAT guard launcher."
  echo "        The guard is inert in this shell: a gh write made from here reaches"
  echo "        GitHub without any of its checks. Put the launcher first and re-run:"
  echo "            export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo "        (add that line to the shell profile; if no launcher is installed there,"
  echo "        run: ${PY:-python} utils/install_hooks.py)"
  return 1
}

# moat_now: seconds since the epoch with a fraction, without GNU date's %N.
moat_now() {
  if [ -n "${EPOCHREALTIME:-}" ]; then
    printf '%s\n' "${EPOCHREALTIME/,/.}"
  else
    "$PY" -c 'import time; print(f"{time.time():.6f}")'
  fi
}
