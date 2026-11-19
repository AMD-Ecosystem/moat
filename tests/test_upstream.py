# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""upstream.py: the publish race and the approved merge."""
import contextlib
import io
import json
import sys
import unittest
from unittest import mock

import gh_guard  # utils/ is put on sys.path by tests/__init__.py
import moatlib
import upstream

A, B = "a" * 40, "b" * 40
UP = "https://github.com/up/proj/pull/7"
REVIEW = "https://github.com/AMD-Ecosystem/proj/pull/3"


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def record():
    return {"name": "proj", "upstream_url": "https://github.com/up/proj",
            "fork_url": "https://github.com/AMD-Ecosystem/proj",
            "fork_default_branch": "main", "head_sha": A}


class PublishRace(unittest.TestCase):
    """The branch is read from the fork right before the create, and the new pull
    request's head is read back after it."""

    def open(self, tip=A, shown=A, ls_rc=0):
        calls, opened = [], []

        def run(argv, **kw):
            calls.append(argv)
            if argv[:2] == ["git", "ls-remote"]:
                return Result(ls_rc, f"{tip}\trefs/heads/moat-port\n" if tip else "")
            if argv[1:3] == ["pr", "create"]:
                return Result(0, UP + "\n")
            if argv[1:3] == ["pr", "view"]:
                return (Result(0, json.dumps({"headRefOid": shown})) if shown
                        else Result(1, "", "timeout"))
            if argv[1:3] == ["pr", "close"]:
                return Result(0)
            raise AssertionError(f"unexpected command {argv}")

        def set_pr_open(name, url, num, published_sha=None):
            opened.append((url, str(num), published_sha))

        row = {"url": REVIEW, "title": "t", "body": "b", "pr": {"headRefOid": A}}
        with mock.patch.object(upstream, "_run", run), \
                mock.patch.object(moatlib, "project_record",
                                  lambda n: (record(), "branch")), \
                mock.patch.object(moatlib, "set_pr_open", set_pr_open), \
                mock.patch.object(upstream, "_land_local_record", lambda *a: None), \
                mock.patch.object(gh_guard, "real_gh", lambda: "real-gh"):
            url, err = upstream.open_upstream("proj", row, approved=A)
        verbs = [" ".join(c[1:3]) for c in calls if c[0] in ("real-gh", "gh")]
        return url, err, verbs, opened

    def test_branch_moved_before_the_create_opens_nothing(self):
        url, err, verbs, opened = self.open(tip=B)
        self.assertIsNone(url)
        self.assertIn("nothing was opened", err)
        self.assertEqual((verbs, opened), ([], []))

    def test_unreadable_branch_opens_nothing(self):
        for kw in ({"ls_rc": 128}, {"tip": ""}):
            url, err, verbs, opened = self.open(**kw)
            self.assertIsNone(url)
            self.assertEqual((verbs, opened), ([], []))

    def test_approved_commit_is_opened_recorded_and_review_closed(self):
        url, err, verbs, opened = self.open()
        self.assertEqual((url, err), (UP, None))
        self.assertEqual(verbs, ["pr create", "pr view", "pr close"])
        self.assertEqual(opened, [(UP, "7", None)])

    def test_head_that_moved_during_the_create_is_not_recorded_as_approved(self):
        url, err, verbs, opened = self.open(shown=B)
        self.assertEqual(url, UP)
        self.assertIn("UNAPPROVED", err)
        self.assertIn("A person must", err)
        # Recorded open (the branch freezes) with the commit it really shows, and
        # neither the review PR nor the upstream PR is closed by the tool.
        self.assertEqual(opened, [(UP, "7", B)])
        self.assertNotIn("pr close", verbs)

    def test_unreadable_new_head_is_reported_and_review_left_open(self):
        url, err, verbs, opened = self.open(shown=None)
        self.assertEqual(url, UP)
        self.assertIn("could not be read back", err)
        self.assertNotIn("pr close", verbs)

    def test_set_pr_open_records_the_head_it_was_given(self):
        saved = []
        obj = record()
        with mock.patch.object(moatlib, "load_status", lambda n: obj), \
                mock.patch.object(moatlib, "save_record",
                                  lambda n, o, m: saved.append(m)):
            moatlib.set_pr_open("proj", UP, 7, published_sha=B)
            self.assertEqual(obj["published_sha"], B)
            self.assertIn("NOT", saved[-1])
            obj2 = record()
            with mock.patch.object(moatlib, "load_status", lambda n: obj2):
                moatlib.set_pr_open("proj", UP, 7)
            self.assertEqual(obj2["published_sha"], A)
            self.assertNotIn("NOT", saved[-1])


class Instructions(unittest.TestCase):
    def test_the_posted_instructions_name_the_commit(self):
        text = upstream.approval_instructions(A, "this port", REVIEW)
        self.assertIn(f"{moatlib.APPROVE_COMMAND} {A[:12]}", text)
        # The quoted form is itself a valid, sha-bound approval when pasted, and is
        # inert where it stands because it sits inside a fence.
        self.assertEqual(list(moatlib._moat_command_lines(text)), [])
        line = f"{moatlib.APPROVE_COMMAND} {A[:12]}"
        self.assertEqual(moatlib._moat_command(line), (moatlib.APPROVE_COMMAND, A[:12]))

    def test_unknown_head_prints_a_placeholder(self):
        self.assertIn("<the head commit's sha>",
                      upstream.approval_instructions(None, "this port"))


class Arguments(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["upstream.py", *argv]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return upstream.main(), out.getvalue(), err.getvalue()

    def test_merge_fix_apply_needs_a_name(self):
        called = []
        with mock.patch.object(upstream, "report_merge_fix",
                               lambda **kw: called.append(kw) or 0):
            rc, _out, err = self.run_main("--merge-fix", "--apply")
            self.assertEqual((rc, called), (2, []))
            self.assertIn("--name", err)
            self.assertEqual(self.run_main("--merge-fix", "--apply",
                                           "--name", "proj")[0], 0)
            self.assertEqual(self.run_main("--merge-fix")[0], 0)
        self.assertEqual(called, [{"apply": True, "only": "proj"},
                                  {"apply": False, "only": None}])

    def test_finish_release_apply_needs_a_name(self):
        called = []
        with mock.patch.object(upstream, "report_finish_release",
                               lambda **kw: called.append(kw) or 0):
            rc, _out, err = self.run_main("--finish-release", "--apply")
            self.assertEqual((rc, called), (2, []))
            self.assertIn("--name", err)
            self.assertEqual(self.run_main("--finish-release", "--apply",
                                           "--name", "warp-1.14")[0], 0)
            self.assertEqual(self.run_main("--finish-release")[0], 0)
        self.assertEqual(called, [{"apply": True, "only": "warp-1.14"},
                                  {"apply": False, "only": None}])


if __name__ == "__main__":
    unittest.main()
