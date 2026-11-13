# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()


class CommitToBranch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name).resolve()
        git(root, "init", "-q", "--bare", "origin.git")
        git(root, "clone", "-q", "origin.git", "work")
        self.repo = root / "work"
        git(self.repo, "config", "user.name", "t")
        git(self.repo, "config", "user.email", "t@example.com")
        (self.repo / "rec.json").write_text('{"n": 0}\n', encoding="utf-8")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        git(self.repo, "branch", "-M", "main")
        git(self.repo, "push", "-q", "origin", "main")
        git(self.repo, "push", "-q", "origin", "main:refs/heads/port/x")
        git(self.repo, "fetch", "-q", "origin")
        for p in (mock.patch.object(moatlib, "REPO_ROOT", self.repo),
                  mock.patch.object(moatlib, "_REF_CACHE", {})):
            p.start()
            self.addCleanup(p.stop)

    def test_a_second_read_sees_the_first_write(self):
        self.assertEqual(moatlib._ref_read("origin/port/x", "rec.json"), '{"n": 0}\n')
        moatlib.commit_to_branch("port/x", {"rec.json": '{"n": 1}\n'}, "first")
        self.assertEqual(moatlib._ref_read("origin/port/x", "rec.json"), '{"n": 1}\n')

    def test_two_writes_stack(self):
        moatlib.commit_to_branch("port/x", {"a.txt": "a\n"}, "first")
        moatlib.commit_to_branch("port/x", {"b.txt": "b\n"}, "second")
        names = git(self.repo, "ls-tree", "--name-only", "origin/port/x").split()
        self.assertEqual(sorted(names), ["a.txt", "b.txt", "rec.json"])
        self.assertEqual(git(self.repo, "rev-list", "--count", "origin/port/x"), "3")

    def test_the_checkout_is_left_alone(self):
        before = git(self.repo, "rev-parse", "HEAD")
        moatlib.commit_to_branch("port/x", {"a.txt": "a\n"}, "first")
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), before)
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")


if __name__ == "__main__":
    unittest.main()
