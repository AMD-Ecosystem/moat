#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""What a decision record can honestly say about who wrote it.

Several records exist to say that a PERSON decided something: a skip, an opt-out
withdrawn, a fork accepted. The only thing that said so was a `--by` argument, which
is whatever the caller typed -- and agents run on the maintainer's credentials, so an
agent that types the maintainer's name produces a record identical to one the
maintainer made. Nothing here can tell those apart, and it does not pretend to.

What it does is stop the record claiming more than is known. Beside `by` (a claim)
each record now carries `recorded_as`: the GitHub account `gh` was actually
authenticated as when the record was written, and on which GitHub host. A reader then
sees both facts -- "by X, written under account Y" -- instead of one unverifiable
name. Decisions that DELETE a record (un-skipping a project, withdrawing an opt-out)
left nothing behind at all; they now append a line to data/decisions.jsonl with the
entry they removed.

The host recorded is the GitHub host (github.com, or GH_HOST), never this machine's
name: these files are published.
"""

import datetime
import json
import os
import pathlib
import subprocess

_CACHE = []


def recorded_as(run=subprocess.run):
    """{"login": ..., "gh_host": ...} for the authenticated gh account, or None when
    gh is missing, logged out or unreachable. Asked once per process."""
    if _CACHE:
        return _CACHE[0]
    who = None
    try:
        r = run(["gh", "api", "user", "--jq", ".login"], capture_output=True,
                text=True, timeout=30)
        login = r.stdout.strip() if r.returncode == 0 else ""
        if login:
            who = {"login": login, "gh_host": os.environ.get("GH_HOST") or "github.com"}
    except (OSError, subprocess.SubprocessError):
        who = None
    _CACHE.append(who)
    return who


def stamp(entry):
    """Add `recorded_as` to a decision record that is about to be saved."""
    who = recorded_as()
    if who:
        entry["recorded_as"] = who
    return entry


def log_path(repo_root):
    return pathlib.Path(repo_root) / "data" / "decisions.jsonl"


def log_removal(repo_root, kind, subject, by, was):
    """Append a tombstone for a decision record that is being deleted."""
    line = {"at": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": kind, "subject": subject, "by": by,
            "recorded_as": recorded_as(), "was": was}
    path = log_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
    return line
