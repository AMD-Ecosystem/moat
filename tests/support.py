# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Helpers shared by the tests that run shell scripts."""
import os
import pathlib
import shutil
import subprocess
import sys


def bash():
    """A POSIX bash to run hook and orient scripts under, or None.

    On Windows a bare `bash` can be the WSL launcher in System32, which is not the
    shell git runs hooks with. The one that ships beside git is."""
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            root = pathlib.Path(git).resolve().parent.parent
            for cand in (root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe",
                         root.parent / "bin" / "bash.exe"):
                if cand.exists():
                    return str(cand)
        return None
    return shutil.which("bash")


def run_bash(script, *args, cwd=None, stdin="", env=None, path_first=None):
    """Run a script file under bash; returns the CompletedProcess.

    `path_first` is a directory to put at the very front of PATH. It is done inside
    the shell, not through `env`: Git for Windows' bash puts its own directories and
    $HOME/bin ahead of whatever PATH it was started with."""
    argv = [bash(), str(script), *args]
    if path_first:
        argv = [bash(), "-c",
                'd=$1; shift; command -v cygpath >/dev/null 2>&1 && d=$(cygpath -u "$d"); '
                'PATH="$d:$PATH"; export PATH; exec bash "$@"',
                "bash", posix(path_first), posix(script), *args]
    return subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True, text=True,
                          env=env, timeout=120)


def posix(path):
    return pathlib.Path(path).as_posix()


def write_script(path, text):
    """An executable script with LF endings, whatever the platform."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(text)
    path.chmod(0o755)
    return path


PYTHON = pathlib.Path(sys.executable).as_posix()
