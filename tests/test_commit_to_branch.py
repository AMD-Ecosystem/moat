# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import json
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
        # The batch reader runs inside the temporary repo; Windows cannot delete it
        # while that process is alive.
        self.addCleanup(moatlib._cat_file_close)

    def test_a_second_read_sees_the_first_write(self):
        self.assertEqual(moatlib._ref_read("origin/port/x", "rec.json"), '{"n": 0}\n')
        moatlib.commit_to_branch("port/x", {"rec.json": '{"n": 1}\n'}, "first")
        self.assertEqual(moatlib._ref_read("origin/port/x", "rec.json"), '{"n": 1}\n')

    def test_a_missing_file_or_ref_reads_as_none(self):
        self.assertIsNone(moatlib._ref_read("origin/port/x", "absent.json"))
        self.assertIsNone(moatlib._ref_read("origin/port/nope", "rec.json"))
        self.assertIsNone(moatlib._ref_read("origin/port/x", ""))   # a tree, not a file
        self.assertEqual(moatlib._ref_read("origin/main", "rec.json"), '{"n": 0}\n')

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


class SharedRepo(unittest.TestCase):
    """A bare origin, this host's clone, and another host's clone of it."""

    STATUS = "projects/x/status.json"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name).resolve()
        git(self.root, "init", "-q", "--bare", "origin.git")
        git(self.root, "clone", "-q", "origin.git", "work")
        self.repo = self.root / "work"
        self.identify(self.repo)
        self.write(self.repo, self.STATUS, self.status())
        self.write(self.repo, "projects/x/notes.md", "# x\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        git(self.repo, "branch", "-M", "main")
        git(self.repo, "push", "-q", "origin", "main")
        git(self.repo, "push", "-q", "origin", "main:refs/heads/port/x")
        git(self.repo, "fetch", "-q", "origin")
        git(self.root, "clone", "-q", "origin.git", "other")
        self.other = self.root / "other"
        self.identify(self.other)
        git(self.other, "checkout", "-q", "port/x")
        for p in (mock.patch.object(moatlib, "REPO_ROOT", self.repo),
                  mock.patch.object(moatlib, "_REF_CACHE", {}),
                  mock.patch.object(moatlib, "ensure_git_config", lambda: None)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(moatlib._cat_file_close)

    @staticmethod
    def identify(repo):
        git(repo, "config", "user.name", "t")
        git(repo, "config", "user.email", "t@example.com")
        git(repo, "config", "core.autocrlf", "false")

    @staticmethod
    def write(repo, rel, text):
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")

    @staticmethod
    def status(**fields):
        rec = {"name": "x", "updated_at": "2026-01-01T00:00:00Z", "head_sha": "a" * 40,
               "platforms": {}}
        rec.update(fields)
        return json.dumps(rec, indent=2) + "\n"

    def other_pushes(self, rel, text, branch="port/x"):
        self.write(self.other, rel, text)
        git(self.other, "add", "-A")
        git(self.other, "commit", "-q", "-m", f"other host edits {rel}")
        git(self.other, "push", "-q", "origin", branch)

    def on_origin(self, rel, branch="port/x"):
        return git(self.root / "origin.git", "show", f"{branch}:{rel}")


class CommitToMovedBranch(SharedRepo):
    def test_a_record_written_from_a_stale_ref_merges_with_the_other_hosts(self):
        self.other_pushes(self.STATUS, self.status(
            updated_at="2026-01-02T00:00:00Z",
            platforms={"linux-gfx90a": {"state": "completed", "validated_sha": "a" * 40,
                                        "updated_at": "2026-01-02T00:00:00Z"}}))
        ours = self.status(updated_at="2026-01-03T00:00:00Z", pr_state="open",
                           pr_number=7)
        moatlib.commit_to_branch("port/x", {self.STATUS: ours}, "x: PR opened")
        merged = json.loads(self.on_origin(self.STATUS))
        self.assertEqual(merged["pr_state"], "open")
        self.assertEqual(merged["platforms"]["linux-gfx90a"]["state"], "completed")
        self.assertEqual(git(self.root / "origin.git", "rev-list", "--count", "port/x"),
                         "3")

    def test_another_file_that_moved_underneath_is_not_overwritten(self):
        self.other_pushes("projects/x/notes.md", "# x\nfrom the other host\n")
        with self.assertRaisesRegex(ValueError, "changed this file since it was read"):
            moatlib.commit_to_branch("port/x", {"projects/x/notes.md": "# x\nours\n"},
                                     "x: notes")
        self.assertIn("from the other host", self.on_origin("projects/x/notes.md"))

    def test_an_untouched_file_is_written_onto_the_new_tip(self):
        self.other_pushes("projects/x/notes.md", "# x\nfrom the other host\n")
        moatlib.commit_to_branch("port/x", {self.STATUS: self.status(pr_state="open")},
                                 "x: PR opened")
        self.assertEqual(json.loads(self.on_origin(self.STATUS))["pr_state"], "open")
        self.assertIn("from the other host", self.on_origin("projects/x/notes.md"))

    def test_a_push_that_cannot_land_is_a_refusal_that_says_nothing_was_recorded(self):
        git(self.repo, "remote", "set-url", "--push", "origin",
            str(self.root / "no-such-remote.git"))
        with self.assertRaisesRegex(ValueError, "NOTHING was recorded"):
            moatlib.commit_to_branch("port/x", {self.STATUS: self.status(n=1)}, "x: n",
                                     retries=2)


class CommitAndPushScope(SharedRepo):
    def test_only_the_named_paths_are_committed(self):
        self.write(self.repo, self.STATUS, self.status(stage="porting"))
        self.write(self.repo, "someone-elses.txt", "staged by another agent\n")
        git(self.repo, "add", "someone-elses.txt")
        self.assertTrue(moatlib.commit_and_push(self.STATUS, "x takes the work lock",
                                                push=False))
        self.assertEqual(git(self.repo, "show", "--name-only", "--format=", "HEAD"),
                         self.STATUS)
        self.assertEqual(git(self.repo, "diff", "--cached", "--name-only"),
                         "someone-elses.txt")

    def test_nothing_to_commit_when_only_other_paths_are_staged(self):
        self.write(self.repo, "someone-elses.txt", "staged by another agent\n")
        git(self.repo, "add", "someone-elses.txt")
        before = git(self.repo, "rev-parse", "HEAD")
        self.assertFalse(moatlib.commit_and_push(self.STATUS, "nothing", push=False))
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), before)


class BranchSyncAllKeepsLocalWork(SharedRepo):
    def sweep(self):
        return {b: (action, detail) for b, action, detail in
                moatlib.branch_sync_all(apply=False, worktree=self.root / "wt-sweep")}

    def test_a_local_branch_with_unpushed_commits_is_not_reset(self):
        git(self.repo, "branch", "port/x", "origin/port/x")
        git(self.repo, "checkout", "-q", "port/x")
        self.write(self.repo, "projects/x/notes.md", "# x\nunpushed work\n")
        git(self.repo, "commit", "-q", "-am", "work the push did not carry")
        local = git(self.repo, "rev-parse", "port/x")
        git(self.repo, "checkout", "-q", "main")
        action, detail = self.sweep()["port/x"]
        self.assertEqual(action, "local-work")
        self.assertIn("work the push did not carry", detail)
        self.assertEqual(git(self.repo, "rev-parse", "port/x"), local)

    def test_a_local_branch_with_nothing_of_its_own_is_swept(self):
        git(self.repo, "branch", "port/x", "origin/port/x")
        self.assertNotEqual(self.sweep()["port/x"][0], "local-work")


if __name__ == "__main__":
    unittest.main()
