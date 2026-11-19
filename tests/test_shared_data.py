# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock


import moatlib


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "commit.gpgsign=false", *args],
                   cwd=str(repo), check=True, capture_output=True)


class SharedData(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self._tmp.name).resolve()
        (self.repo / "data").mkdir()
        self.optouts = self.repo / "data" / "optout.json"
        patches = [mock.patch.object(moatlib, "REPO_ROOT", self.repo),
                   mock.patch.object(moatlib, "OPTOUTS", self.optouts),
                   mock.patch.object(moatlib, "_HEAD_HAS_TRUNK", [])]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    def write(self, obj):
        self.optouts.write_text(json.dumps(obj), encoding="utf-8")

    def test_missing_file_is_empty(self):
        self.assertEqual(moatlib.load_optouts(), {})

    def test_malformed_file_raises(self):
        self.optouts.write_text('<<<<<<< HEAD\n{"a": 1}\n=======\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            moatlib.load_optouts()

    def test_stale_checkout_reads_the_trunk(self):
        git(self.repo, "init", "-q", "-b", "port/x")
        self.write({})
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "old")
        self.write({"someowner": {"who": "someowner", "source": "issue"}})
        git(self.repo, "commit", "-q", "-am", "opt-out recorded on the trunk")
        git(self.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        git(self.repo, "reset", "-q", "--hard", "HEAD~1")
        self.assertEqual(json.loads(self.optouts.read_text()), {})
        self.assertEqual(moatlib.optout_for("SomeOwner/repo")["who"], "someowner")

    def test_checkout_ahead_of_trunk_reads_its_own_file(self):
        git(self.repo, "init", "-q", "-b", "main")
        self.write({})
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        git(self.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        self.write({"newowner": {"who": "newowner", "source": "email"}})
        self.assertIsNotNone(moatlib.optout_for("newowner/anything"))


if __name__ == "__main__":
    unittest.main()
