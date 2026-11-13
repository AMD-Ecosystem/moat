#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Register the semantic status.json merge driver for this clone (idempotent).
# Needed so .gitattributes 'merge=moat-status' takes effect on pull/merge.
set -euo pipefail
cd "$(dirname "$0")/.."
# The driver command lives in moatlib (MERGE_DRIVER) so there is one copy of it. It
# names no checkout path: this config is shared by every worktree of the clone.
py=python3; command -v "$py" >/dev/null 2>&1 || py=python
"$py" -c 'import sys; sys.path.insert(0, "utils"); import moatlib; moatlib.ensure_git_config()'
# Every port/<name> starts life with no upstream, and this clone creates them
# routinely. Without this a plain `git push` on a new branch errors out instead of
# publishing it -- and the branch existing on the remote IS a project's claim, so a
# push that quietly does not happen is a claim nobody else can see.
git config push.autoSetupRemote true
echo "registered merge.moat-status driver and push.autoSetupRemote"
