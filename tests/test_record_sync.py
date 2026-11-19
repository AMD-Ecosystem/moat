# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""upstream.py: what a sweep writes, and the record-sync branch."""
import contextlib
import io
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import upstream  # utils/ is put on sys.path by tests/__init__.py

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


class Sweep(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["upstream.py", *argv]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return upstream.main(), out.getvalue(), err.getvalue()

    def test_a_sweep_made_to_look_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            stamp = pathlib.Path(d) / "data" / "reconciled.json"
            with mock.patch.object(upstream, "RECONCILED", stamp), \
                    mock.patch.object(upstream, "recorded", lambda: []):
                for argv in ((), ("--dry-run",), ("--name", "proj")):
                    rc, out, _err = self.run_main(*argv)
                    self.assertEqual(rc, 0)
                    self.assertFalse(stamp.exists(), argv)
                    self.assertIn("nothing was written", out)
                rc, out, _err = self.run_main("--stamp")
                self.assertTrue(stamp.exists())
                self.assertEqual(json.loads(stamp.read_text())["at"], upstream.TODAY)

    def test_today_needs_no_date_program(self):
        self.assertRegex(upstream.TODAY, r"^\d{4}-\d{2}-\d{2}$")
        src = pathlib.Path(upstream.__file__).read_text(encoding="utf-8")
        self.assertIsNone(re.search(r'_run\(\["date"', src))


def git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode:
        raise AssertionError(f"git {' '.join(args)}: {r.stderr}")
    return r.stdout.strip()


class RecordSync(unittest.TestCase):
    """publish(): the record-sync branch, against a scratch clone and bare origin."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        self.origin, self.repo = root / "origin.git", root / "work"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)],
                       check=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.repo)], check=True)
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "t")
        git(self.repo, "config", "core.autocrlf", "false")
        for rel, text in (("projects/p/status.json", "{}\n"), ("README.md", "r\n"),
                          ("data/reconciled.json", "{}\n"), ("notes.txt", "n\n")):
            f = self.repo / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-q", "-m", "base")
        git(self.repo, "remote", "add", "origin", str(self.origin))
        git(self.repo, "push", "-q", "-u", "origin", "main")
        git(self.repo, "checkout", "-q", "-b", "work")

    def tearDown(self):
        self.tmp.cleanup()

    def publish(self, fail=None):
        real, gh_calls = upstream._run, []

        def run(argv, **kw):
            if argv[0] == "git":
                if fail and argv[1:1 + len(fail)] == list(fail):
                    return Result(1, "", "simulated failure")
                return real(argv, **kw)
            gh_calls.append(argv)
            return Result(0, "")

        applied = [{"name": "p", "repo": "up/p", "num": "1", "merged_at": "2026-01-01"}]
        err = io.StringIO()
        with mock.patch.object(upstream, "REPO", self.repo), \
                mock.patch.object(upstream, "_run", run), \
                mock.patch.object(upstream, "gh_json", lambda a: []), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err):
            return upstream.publish(applied), err.getvalue(), gh_calls

    def dirty(self, *rels):
        for rel in rels:
            (self.repo / rel).write_text("changed\n")

    def test_the_sweeps_own_stamp_and_readme_are_published_not_refused(self):
        self.dirty("projects/p/status.json", "README.md", "data/reconciled.json")
        rc, err, gh_calls = self.publish()
        self.assertEqual((rc, err), (0, ""))
        changed = git(self.origin, "diff", "--name-only", "main", upstream.BRANCH)
        self.assertEqual(sorted(changed.split()),
                         ["README.md", "data/reconciled.json", "projects/p/status.json"])
        self.assertEqual(git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"), "work")
        self.assertTrue(any(c[1:3] == ["pr", "create"] for c in gh_calls))

    def test_an_unrelated_modified_file_still_refuses(self):
        self.dirty("projects/p/status.json", "notes.txt")
        rc, err, gh_calls = self.publish()
        self.assertEqual(rc, 1)
        self.assertIn("notes.txt", err)
        self.assertEqual((self.repo / "notes.txt").read_text(), "changed\n")
        self.assertEqual(gh_calls, [])

    def test_a_failed_step_stops_and_restores_the_checkout(self):
        for fail in (("checkout", "-q", "-B"), ("fetch",), ("push",)):
            with self.subTest(fail=fail):
                self.dirty("projects/p/status.json")
                before = git(self.repo, "rev-parse", "work")
                rc, err, gh_calls = self.publish(fail=fail)
                self.assertEqual(rc, 1)
                self.assertIn("simulated failure", err)
                # Back where it started, corrections still in the tree, nothing
                # committed onto the starting branch and no pull request touched.
                self.assertEqual(git(self.repo, "rev-parse", "--abbrev-ref", "HEAD"),
                                 "work")
                self.assertEqual(git(self.repo, "rev-parse", "work"), before)
                self.assertEqual((self.repo / "projects/p/status.json").read_text(),
                                 "changed\n")
                self.assertEqual(gh_calls, [])
                self.assertNotIn(upstream.BRANCH,
                                 git(self.origin, "branch", "--list"))


class SiblingTools(unittest.TestCase):
    def test_sibling_tools_run_from_the_tools_root(self):
        self.assertEqual(pathlib.Path(upstream._tool("gen_readme.py")),
                         upstream.TOOLS_ROOT / "utils" / "gen_readme.py")
        self.assertEqual(upstream._tool_env()["MOAT_REPO"], str(upstream.REPO))
        src = pathlib.Path(upstream.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"utils/gen_readme.py"', src)
        self.assertNotIn('"utils/pr_intent.py"', src)


if __name__ == "__main__":
    unittest.main()
