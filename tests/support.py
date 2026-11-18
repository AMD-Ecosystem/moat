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


def run_bash(script, *args, cwd=None, stdin="", env=None):
    """Run a script file under bash; returns the CompletedProcess."""
    return subprocess.run([bash(), str(script), *args], cwd=cwd, input=stdin,
                          capture_output=True, text=True, env=env, timeout=120)


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
