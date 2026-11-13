#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run MOAT's tooling from the trunk, whatever branch the checkout is on.

A port branch carries a copy of `utils/` from the last time it merged the trunk, and
that copy can be weeks old. Running it is how an old `upstream.py --publish` that
ignored `--name` opened ten upstream pull requests from a branch cut for one, and how a
record gets written in a state the trunk no longer recognises. So the code always comes
from `origin/main`; only the data comes from the checkout.

Every entry point calls `reexec(__file__)` before it does anything else. When the
checkout's `utils/` is not the trunk's, it re-runs the same script from a read-only
export of `origin/main` (kept under the clone's git directory), with `MOAT_REPO` naming
the checkout so records, projects and git commands still act on it. The export holds
`utils/`, `config/` and `schema/`: the code and the inputs that define what the code
means. Everything else is read from the checkout.

Local tooling still runs when it is at least as new as the trunk: a checkout whose HEAD
contains `origin/main` (tooling work on the trunk or ahead of it) is left alone, and so
is one whose `utils/` matches. Nothing here fetches; orient and the publishing tools do.

    python3 utils/trunk.py run check.py --fast   # run a trunk script against this checkout
    python3 utils/trunk.py path                  # print the export directory

The pre-push hook runs this file straight out of `origin/main` (`python3 -c "$(git show
origin/main:utils/trunk.py)" run check.py`), so even a branch too old to contain it is
gated by the trunk's checks.
"""

import io
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile

TRUNK_REF = os.environ.get("MOAT_TRUNK_REF", "origin/main")   # override for tests only
EXPORT_PATHS = ("utils", "config", "schema")
ENV_REPO = "MOAT_REPO"
ENV_ACTIVE = "MOAT_TRUNK_TOOLS"
KEEP = 3


def repo_root(file):
    """The checkout the tools act on: `MOAT_REPO` when the trunk export runs, else the
    checkout that holds `file`."""
    env = os.environ.get(ENV_REPO)
    return pathlib.Path(env).resolve() if env else pathlib.Path(file).resolve().parents[1]


def tools_root(file):
    """Where the running code and its config live: the export, or the checkout."""
    return pathlib.Path(file).resolve().parents[1]


def _git(repo, *args):
    r = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def _toplevel(start):
    rc, out = _git(start, "rev-parse", "--show-toplevel")
    return pathlib.Path(out) if rc == 0 and out else None


def trunk_state(repo):
    """(trunk sha, export base dir, local tooling is current) for a checkout, or None
    when there is no trunk ref to run from (offline fresh clone, not a git checkout)."""
    rc, out = _git(repo, "rev-parse", "--git-common-dir", TRUNK_REF, f"{TRUNK_REF}:utils")
    lines = out.splitlines()
    if rc or len(lines) != 3:
        return None
    common, sha, trunk_utils = lines
    common = pathlib.Path(common)
    if not common.is_absolute():
        common = (pathlib.Path(repo) / common).resolve()
    rc, head_utils = _git(repo, "rev-parse", "HEAD:utils")
    current = rc == 0 and head_utils == trunk_utils
    if not current:
        # HEAD already contains the trunk: tooling work on or ahead of it runs as is.
        current = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=str(repo),
            capture_output=True).returncode == 0
    return sha, common / "moat-tools", current


def export(repo, sha, base):
    """The directory holding `EXPORT_PATHS` of `sha`, created on first use."""
    dest = base / sha[:12]
    if (dest / "utils" / "trunk.py").exists():
        return dest
    base.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f".{sha[:12]}-", dir=str(base)))
    try:
        p = subprocess.run(["git", "archive", "--format=tar", sha, *EXPORT_PATHS],
                           cwd=str(repo), capture_output=True, check=True)
        with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(str(tmp), filter="data")
            else:
                tf.extractall(str(tmp))
        try:
            os.replace(str(tmp), str(dest))
        except OSError:
            # Another process exported the same commit first; theirs is identical.
            shutil.rmtree(str(tmp), ignore_errors=True)
    except Exception:
        shutil.rmtree(str(tmp), ignore_errors=True)
        raise
    _prune(base, keep=dest)
    return dest


def _prune(base, keep):
    """Keep the newest few exports; a running process may still be using one."""
    dirs = sorted((d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")),
                  key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs[KEEP:]:
        if d != keep:
            shutil.rmtree(str(d), ignore_errors=True)


def _run_from(dest, repo, script, args):
    env = {**os.environ, ENV_REPO: str(repo), ENV_ACTIVE: dest.name}
    target = dest / "utils" / script
    return subprocess.run([sys.executable, str(target), *args], env=env).returncode


def reexec(file):
    """Re-run the calling script from the trunk export when the checkout's tooling is
    not the trunk's. Returns only when the caller should run itself."""
    if os.environ.get(ENV_ACTIVE) or os.environ.get("MOAT_LOCAL_TOOLS"):
        return
    repo = pathlib.Path(file).resolve().parents[1]
    try:
        state = trunk_state(repo)
        if state is None or state[2]:
            return
        sha, base, _ = state
        dest = export(repo, sha, base)
    except Exception as e:                       # never strand a command on the launcher
        sys.stderr.write(f"moat: could not run the trunk tooling ({e}); using this "
                         f"checkout's copy\n")
        return
    sys.exit(_run_from(dest, repo, pathlib.Path(file).name, sys.argv[1:]))


def main(argv):
    if len(argv) >= 1 and argv[0] == "path":
        repo = _toplevel(os.getcwd())
        state = trunk_state(repo) if repo else None
        if state is None:
            print("trunk: no origin/main here", file=sys.stderr)
            return 1
        print(export(repo, state[0], state[1]))
        return 0
    if len(argv) >= 2 and argv[0] == "run":
        repo = _toplevel(os.getcwd())
        if repo is None:
            print("trunk: not inside a git checkout", file=sys.stderr)
            return 1
        state = trunk_state(repo)
        if state is None or state[2]:
            # No trunk ref (offline, or a clone with no origin): the local copy is all
            # there is, and refusing would only block work. A checkout whose tooling is
            # the trunk's, or newer, runs its own.
            local = repo / "utils" / argv[1]
            env = {**os.environ, ENV_ACTIVE: "local"}
            return subprocess.run([sys.executable, str(local), *argv[2:]],
                                  env=env).returncode
        return _run_from(export(repo, state[0], state[1]), repo, argv[1], argv[2:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
