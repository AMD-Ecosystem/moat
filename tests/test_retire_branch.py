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


class RetireBranch(unittest.TestCase):
    """A finished project's branch is deleted only when it holds nothing main lacks."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name).resolve()
        git(root, "init", "-q", "--bare", "origin.git")
        git(root, "clone", "-q", "origin.git", "work")
        self.repo = root / "work"
        git(self.repo, "config", "user.name", "t")
        git(self.repo, "config", "user.email", "t@example.com")
        self.write("shared.md", "one\n")
        self.commit("base")
        git(self.repo, "branch", "-M", "main")
        git(self.repo, "push", "-q", "origin", "main")
        git(self.repo, "checkout", "-q", "-b", "port/x")
        self.write("projects/x/status.json", '{"name": "x"}\n')
        self.commit("record")
        git(self.repo, "push", "-q", "origin", "port/x")
        git(self.repo, "checkout", "-q", "main")
        self.finished = True
        self.record = {"name": "x"}
        for p in (mock.patch.object(moatlib, "REPO_ROOT", self.repo),
                  mock.patch.object(moatlib, "_REF_CACHE", {}),
                  mock.patch.object(moatlib, "project_record",
                                    lambda name: (self.record, "branch")),
                  mock.patch.object(moatlib, "belongs_on_branch",
                                    lambda obj: not self.finished)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(moatlib._cat_file_close)

    def write(self, rel, text):
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")

    def commit(self, msg):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", msg)

    def remote_has_branch(self):
        return bool(git(self.repo, "ls-remote", "origin", "refs/heads/port/x"))

    def merge_to_main(self, squash=False):
        if squash:
            git(self.repo, "merge", "-q", "--squash", "origin/port/x")
            self.commit("squashed")
        else:
            git(self.repo, "merge", "-q", "--no-ff", "-m", "merge", "origin/port/x")
        git(self.repo, "push", "-q", "origin", "main")

    def refused(self, needle):
        with self.assertRaises(ValueError) as cm:
            moatlib.retire_branch("x", apply=True)
        self.assertIn(needle, str(cm.exception))
        self.assertTrue(self.remote_has_branch())

    def test_an_unmerged_branch_is_kept(self):
        self.refused("merge port/x first")

    def test_a_project_with_work_outstanding_keeps_its_claim(self):
        self.merge_to_main()
        self.finished = False
        self.refused("work outstanding")

    def test_a_held_project_is_left_alone(self):
        self.merge_to_main()
        self.record = {"name": "x", "on_hold": True}
        self.refused("on hold")

    def test_a_merged_branch_is_reported_then_deleted(self):
        self.merge_to_main()
        self.assertIn("--apply", moatlib.retire_branch("x"))
        self.assertTrue(self.remote_has_branch())
        self.assertIn("deleted", moatlib.retire_branch("x", apply=True))
        self.assertFalse(self.remote_has_branch())
        self.assertEqual(moatlib.branch_retirable("x")[0], "")

    def test_a_squash_merged_branch_is_deleted(self):
        self.merge_to_main(squash=True)
        moatlib.retire_branch("x", apply=True)
        self.assertFalse(self.remote_has_branch())

    def test_a_squash_merge_that_left_a_shared_edit_behind_is_kept(self):
        git(self.repo, "checkout", "-q", "port/x")
        self.write("shared.md", "one\na lesson\n")
        self.commit("lesson")
        git(self.repo, "push", "-q", "origin", "port/x")
        git(self.repo, "checkout", "-q", "main")
        git(self.repo, "checkout", "-q", "origin/port/x", "--", "projects/x")
        self.commit("record only")
        git(self.repo, "push", "-q", "origin", "main")
        self.refused("shared files")

    def test_a_record_that_moved_on_the_branch_after_the_merge_is_kept(self):
        self.merge_to_main(squash=True)
        git(self.repo, "checkout", "-q", "port/x")
        self.write("projects/x/status.json", '{"name": "x", "n": 2}\n')
        self.commit("later write")
        git(self.repo, "push", "-q", "origin", "port/x")
        git(self.repo, "checkout", "-q", "main")
        self.refused("differs")

    def test_misplaced_names_the_delete_command_once_merged(self):
        rows = lambda: [r[2] for r in moatlib.misplaced_folders()]
        with mock.patch.object(moatlib, "project_records",
                               lambda: iter([("x", self.record, "branch")])):
            self.assertIn("merge port/x to main", rows()[0])
        self.merge_to_main()
        git(self.repo, "fetch", "-q", "origin")
        with mock.patch.object(moatlib, "project_records",
                               lambda: iter([("x", self.record, "branch")])):
            self.assertIn("retire-branch x --apply", rows()[0])


if __name__ == "__main__":
    unittest.main()
