#!/usr/bin/env bash
# Register the semantic status.json merge driver for this clone (idempotent).
# Needed so .gitattributes 'merge=moat-status' takes effect on pull/merge.
set -euo pipefail
cd "$(dirname "$0")/.."
git config merge.moat-status.name "MOAT status.json semantic merge"
git config merge.moat-status.driver "python3 $(pwd)/utils/merge_status.py %O %A %B %P"
# Every port/<name> starts life with no upstream, and this clone creates them
# routinely. Without this a plain `git push` on a new branch errors out instead of
# publishing it -- and the branch existing on the remote IS a project's claim, so a
# push that quietly does not happen is a claim nobody else can see.
git config push.autoSetupRemote true
echo "registered merge.moat-status driver and push.autoSetupRemote"
