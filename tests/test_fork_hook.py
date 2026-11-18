# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""moatlib's fork pre-push hook, rendered and run under bash against a scratch clone."""
import os
import pathlib
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py
from tests import support

ZERO = "0" * 40
A, B = "a" * 40, "b" * 40


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@unittest.skipUnless(support.bash(), "needs bash")
class ForkHook(unittest.TestCase):
    """moatlib's fork pre-push hook: the freeze, and the one-shot publish token."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        self.repo = root / "src"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        # Stands in for `moatlib.py pr-state <name> --refresh`.
        self.tools = root / "tools"
        self.state = self.tools / "state.txt"
        support.write_script(self.tools / "utils" / "moatlib.py", (
            "import pathlib, sys\n"
            "s = (pathlib.Path(__file__).parents[1] / 'state.txt').read_text().strip()\n"
            "if s == 'FAIL':\n    sys.exit('cannot refresh')\n"
            "print(s)\n"))
        self.state.write_text("open")
        self.obj = {"fork_url": "https://github.com/AMD-Ecosystem/proj"}
        with mock.patch.object(moatlib, "REPO_ROOT", self.tools), \
                mock.patch.object(moatlib, "project_record",
                                  lambda n: (self.obj, "branch")):
            self.level, self.msg = moatlib.protect_fork("proj", repo=self.repo)
        self.hook = self.repo / ".git" / "hooks" / "pre-push"

    def tearDown(self):
        self.tmp.cleanup()

    def push(self, ref=f"refs/heads/{moatlib.PORT_BRANCH}", sha=A, old=B, env=None):
        return support.run_bash(self.hook, cwd=str(self.repo),
                                stdin=f"refs/heads/local {sha} {ref} {old}\n", env=env)

    def token(self):
        return self.repo / ".git" / moatlib.PUBLISH_TOKEN

    def test_installed(self):
        self.assertEqual(self.level, "ok", self.msg)
        text = self.hook.read_text()
        self.assertIn(moatlib.FORK_HOOK_MARKER, text)
        self.assertIn(support.PYTHON, text)
        self.assertNotIn("MOAT_PUBLISH", text)
        self.assertNotIn("\r", self.hook.read_bytes().decode())

    def test_open_pr_freezes_the_port_branch_only(self):
        r = self.push()
        self.assertEqual(r.returncode, 1)
        self.assertIn("OPEN upstream PR", r.stderr)
        self.assertEqual(self.push(ref="refs/heads/moat-fix-7").returncode, 0)

    def test_private_states_let_the_push_through(self):
        for state in ("none", "merged", "closed"):
            self.state.write_text(state)
            self.assertEqual(self.push().returncode, 0, state)
        for state in ("maintained", "delivered", "some-new-state"):
            self.state.write_text(state)
            self.assertEqual(self.push().returncode, 1, state)

    def test_a_state_nobody_could_read_refuses(self):
        self.state.write_text("FAIL")
        r = self.push()
        self.assertEqual(r.returncode, 1)
        self.assertIn("cannot tell", r.stderr)

    def test_the_environment_variable_no_longer_opens_it(self):
        r = self.push(env={**os.environ, "MOAT_PUBLISH": "1"})
        self.assertEqual(r.returncode, 1)

    def test_token_lets_exactly_the_named_update_through_once(self):
        ref = f"refs/heads/{moatlib.PORT_BRANCH}"
        path = moatlib.write_publish_token(self.repo, [(A, ref)])
        self.assertEqual(pathlib.Path(path).resolve(), self.token().resolve())
        self.assertEqual(self.push().returncode, 0)
        self.assertFalse(self.token().exists())
        self.assertEqual(self.push().returncode, 1)        # one push, not a window

    def test_token_for_another_commit_or_ref_does_not(self):
        ref = f"refs/heads/{moatlib.PORT_BRANCH}"
        moatlib.write_publish_token(self.repo, [(B, ref)])
        self.assertEqual(self.push(sha=A).returncode, 1)
        self.assertFalse(self.token().exists())            # spent either way
        moatlib.write_publish_token(self.repo, [(A, "refs/heads/moat-fix-7")])
        self.assertEqual(self.push(sha=A).returncode, 1)

    def test_expired_or_malformed_token_does_not(self):
        ref = f"refs/heads/{moatlib.PORT_BRANCH}"
        for text in (f"expires {int(time.time()) - 5}\n{A} {ref}\n",
                     f"{A} {ref}\n", f"expires soon\n{A} {ref}\n"):
            with open(self.token(), "w", newline="\n") as f:
                f.write(text)
            self.assertEqual(self.push().returncode, 1, text)
            self.assertFalse(self.token().exists())

    def test_token_names_a_full_sha_and_ref(self):
        for sha, ref in ((A[:12], "refs/heads/x"), (A, "moat-port"), ("", "refs/x")):
            with self.assertRaises(ValueError):
                moatlib.write_publish_token(self.repo, [(sha, ref)])

    def test_reinstall_is_idempotent_and_spares_a_foreign_hook(self):
        with mock.patch.object(moatlib, "REPO_ROOT", self.tools), \
                mock.patch.object(moatlib, "project_record",
                                  lambda n: (self.obj, "branch")):
            self.assertIn("current", moatlib.protect_fork("proj", repo=self.repo)[1])
            self.hook.write_text("#!/bin/sh\nexit 0\n")
            level, msg = moatlib.protect_fork("proj", repo=self.repo)
        self.assertEqual(level, "warn")
        self.assertIn("UNPROTECTED", msg)


if __name__ == "__main__":
    unittest.main()
