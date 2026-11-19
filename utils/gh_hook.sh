#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The command .claude/settings.json runs as a PreToolUse hook for shell tools. It
# hands the tool event (JSON on standard input) to the gh guard's hook launcher,
# which judges the command text and exits 2, with the reason on stderr, for a
# GitHub write the guard refuses (see gh_guard.py --hook).
#
# The launcher is installed per host by utils/install_hooks.py. When it is missing
# this cannot fail closed on everything: the hook runs before EVERY shell command,
# so refusing them all would also refuse the command that installs the launcher,
# and a host that had not installed it yet would be locked out the moment this
# landed. So without the launcher only a command that mentions `gh` is refused;
# anything else runs, with a line saying the guard is not installed.

launcher="${HOME}/.local/bin/moat-gh-hook"
if [ -x "$launcher" ]; then
  exec "$launcher"
fi

event=$(cat)
if printf '%s' "$event" | grep -Eq '(^|[^A-Za-z0-9_-])gh(\.exe)?([^A-Za-z0-9_-]|$)'; then
  echo "moat: this command mentions gh and the gh guard hook is not installed on this host, so it was not run. Install it first: python3 utils/install_hooks.py (python on Windows)." >&2
  exit 2
fi
echo "moat: the gh guard hook is not installed on this host (run utils/install_hooks.py); this command does not mention gh and was allowed." >&2
exit 0
