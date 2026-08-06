#!/usr/bin/env bash
# Register the semantic status.json merge driver for this clone (idempotent).
# Needed so .gitattributes 'merge=moat-status' takes effect on pull/merge.
set -euo pipefail
cd "$(dirname "$0")/.."
git config merge.moat-status.name "MOAT status.json semantic merge"
git config merge.moat-status.driver "python3 $(pwd)/utils/merge_status.py %O %A %B %P"
echo "registered merge.moat-status driver"
