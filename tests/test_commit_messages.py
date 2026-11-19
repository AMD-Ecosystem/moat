# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py

GOOD = ("[ROCm] {t}\n\nWhy.\n\nAuthored with an AI coding agent.\n\n"
        "Test Plan:\n```\nmake test\n```\n")
BAD = "[ROCm] {t}\n\nWhy.\n\nAuthored with an AI coding agent.\n"


class PublishedCommitsAreSkipped(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q", "-b", "main")
        self.commit("base", "upstream base\n")
        self.git("checkout", "-q", "-b", "moat-port")
        patcher = mock.patch.object(moatlib, "_fork_repo", lambda name: self.repo)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def git(self, *args):
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
        r = subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=str(self.repo),
                           capture_output=True, text=True, env=env, check=True)
        return r.stdout.strip()

    def commit(self, fname, message):
        (self.repo / (fname + ".c")).write_text("int " + fname + ";\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def record(self, **extra):
        return dict({"name": "proj", "fork_default_branch": "main"}, **extra)

    def test_unpublished_branch_is_judged_whole(self):
        sha = self.commit("a", BAD.format(t="first"))
        found = moatlib.commit_message_problems("proj", self.record())
        self.assertEqual(found, [f"proj {sha[:7]}: body has no Test Plan"])

    def test_commits_behind_the_published_tip_are_skipped(self):
        tip = self.commit("a", BAD.format(t="first"))
        self.assertEqual(
            moatlib.commit_message_problems("proj", self.record(published_sha=tip)), [])

    def test_commits_past_the_published_tip_are_still_judged(self):
        tip = self.commit("a", BAD.format(t="first"))
        self.commit("b", GOOD.format(t="second"))
        later = self.commit("c", BAD.format(t="third"))
        found = moatlib.commit_message_problems("proj", self.record(published_sha=tip))
        self.assertEqual(found, [f"proj {later[:7]}: body has no Test Plan"])

    def test_an_archived_pull_request_tip_counts_too(self):
        tip = self.commit("a", BAD.format(t="first"))
        self.commit("b", GOOD.format(t="second"))
        obj = self.record(pr_history=[{"published_sha": tip}])
        self.assertEqual(moatlib.commit_message_problems("proj", obj), [])

    def test_a_tip_the_clone_lacks_skips_nothing(self):
        sha = self.commit("a", BAD.format(t="first"))
        found = moatlib.commit_message_problems(
            "proj", self.record(published_sha="c" * 40))
        self.assertEqual(found, [f"proj {sha[:7]}: body has no Test Plan"])


if __name__ == "__main__":
    unittest.main()
