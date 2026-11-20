# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


import gh_guard  # utils/ is put on sys.path by tests/__init__.py

FORK = "AMD-Ecosystem/fork"


def git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)


class Classify(unittest.TestCase):
    def test_cases_table(self):
        for argv, want in gh_guard.CASES:
            with self.subTest(argv=" ".join(argv)):
                self.assertEqual(gh_guard.classify(argv).allow, want)

    def test_outside_a_checkout_table(self):
        for argv, want in gh_guard.OUTSIDE_CASES:
            with self.subTest(argv=" ".join(argv)):
                self.assertEqual(gh_guard.classify(argv, strict=False).allow, want)

    def test_moat_command_in_body_file_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "body.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("Reviewed.\n\n/moat approve\n")
            for argv in (
                ["pr", "comment", "3", "--repo", FORK, "--body-file", path],
                ["pr", "comment", "3", "--repo", FORK, f"--body-file={path}"],
                ["pr", "comment", "3", "--repo", FORK, f"-F{path}"],
                ["api", f"repos/{FORK}/issues/3/comments", "-F", f"body=@{path}"],
                ["api", f"repos/{FORK}/issues/3/comments", f"-Fbody=@{path}"],
                ["api", f"repos/{FORK}/issues/3/comments", f"--field=body=@{path}"],
            ):
                for strict in (True, False):
                    with self.subTest(argv=argv[-1], strict=strict):
                        self.assertFalse(gh_guard.classify(argv, strict=strict).allow)

    def test_ordinary_body_file_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "body.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("Addressed in abc1234.\n")
            argv = ["pr", "comment", "3", "--repo", FORK, "--body-file", path]
            self.assertTrue(gh_guard.classify(argv).allow)

    def test_json_input_is_decoded_before_the_scan(self):
        # Raw, this file is one line that does not start with /moat; GitHub stores
        # a comment whose second line is the command.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "payload.json")
            for payload in ({"body": "Looks right.\n/moat approve"},
                            {"event": "COMMENT",
                             "comments": [{"body": "/moat approve"}]}):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                with open(path, encoding="utf-8") as f:
                    self.assertFalse(gh_guard._has_moat_command(f.read()))
                for argv in (
                    ["api", f"repos/{FORK}/issues/3/comments", "--input", path],
                    ["api", f"repos/{FORK}/issues/3/comments", f"--input={path}"],
                ):
                    for strict in (True, False):
                        with self.subTest(argv=argv[-1], strict=strict):
                            self.assertFalse(
                                gh_guard.classify(argv, strict=strict).allow)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"body": "The docs say `/moat approve` is a person's"}, f)
            self.assertTrue(gh_guard.classify(
                ["api", f"repos/{FORK}/issues/3/comments", "--input", path]).allow)

    def test_unreadable_body_file_is_refused(self):
        missing = os.path.join(tempfile.gettempdir(), "moat-no-such-body-file.md")
        for argv in (
            ["pr", "comment", "3", "--repo", FORK, "--body-file", missing],
            ["api", f"repos/{FORK}/issues/3/comments", "--input", missing],
            ["api", f"repos/{FORK}/issues/3/comments", "-F", f"body=@{missing}"],
        ):
            for strict in (True, False):
                with self.subTest(argv=argv[-1], strict=strict):
                    d = gh_guard.classify(argv, strict=strict)
                    self.assertFalse(d.allow)
                    self.assertIn("could not be read", d.reason)
        # Somebody else's repository, from outside a checkout: not ours to judge.
        self.assertTrue(gh_guard.classify(
            ["pr", "comment", "3", "--repo", "torvalds/linux", "--body-file", missing],
            strict=False).allow)

    def test_a_plain_api_field_is_not_a_file(self):
        argv = ["api", "-X", "PATCH", f"repos/{FORK}/pulls/3", "-F", "draft=false",
                "-F", "title=no-such-file"]
        self.assertTrue(gh_guard.classify(argv).allow)
        self.assertEqual(gh_guard.body_files("api", argv), [])

    def test_stdin_body_is_judged_when_captured(self):
        argv = ["pr", "comment", "3", "--repo", FORK, "--body-file", "-"]
        self.assertFalse(gh_guard.classify(argv).allow)
        self.assertTrue(gh_guard.classify(argv, stdin_text="fixed\n").allow)
        self.assertFalse(
            gh_guard.classify(argv, stdin_text="ok\n/moat approve\n").allow)

    def test_repo_flag_spellings_and_last_one_wins(self):
        for argv, want in (
            (["--repo", "a/b", "--repo", "c/d"], "c/d"),
            (["-R", "a/b", "-Rc/d"], "c/d"),
            (["-R=c/d"], "c/d"),
            (["--repo=a/b", "-R", "c/d"], "c/d"),
            (["--body", "-R", "--repo", "c/d"], "c/d"),
            (["--body", "--repo", "9"], None),
            (["review", "--comment", "--repo", "c/d"], "c/d"),
            (["--repo", "what is this"], gh_guard.BAD_REPO),
            (["9"], None),
        ):
            with self.subTest(argv=argv):
                self.assertEqual(gh_guard.explicit_repo(argv), want)

    def test_gh_repo_env_names_the_target(self):
        argv = ["pr", "comment", "9", "--body", "hi"]
        with mock.patch.dict(os.environ, {"GH_REPO": "torvalds/linux"}):
            self.assertFalse(gh_guard.classify(argv).allow)
        with mock.patch.dict(os.environ, {"GH_REPO": "AMD-Ecosystem/moat"}):
            self.assertTrue(gh_guard.classify(argv).allow)

    def test_api_method_spellings(self):
        for rest, want in (
            (["-X", "POST", "x"], "POST"), (["-XPOST", "x"], "POST"),
            (["-X=patch", "x"], "PATCH"), (["--method", "delete", "x"], "DELETE"),
            (["--method=PUT", "x"], "PUT"), (["x"], None), (["x", "-X"], ""),
        ):
            with self.subTest(rest=rest):
                self.assertEqual(gh_guard.api_method(rest), want)


class WorkingDirectoryTarget(unittest.TestCase):
    """Which repository a command with no --repo acts on is gh's rule."""

    def repo(self, d, **remotes):
        git(d, "init", "-q")
        for name, slug in remotes.items():
            git(d, "remote", "add", name, f"https://github.com/{slug}.git")
        return d

    def test_upstream_remote_outranks_origin(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d, origin=FORK, upstream="torvalds/linux")
            self.assertEqual(gh_guard.cwd_repo(d), "torvalds/linux")
            argv = ["pr", "create", "--title", "t", "--body", "b"]
            with mock.patch.dict(os.environ, {"GH_REPO": ""}):
                self.assertFalse(gh_guard.classify(argv, cwd=d).allow)

    def test_a_set_default_remote_wins(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d, origin=FORK, upstream="torvalds/linux")
            git(d, "config", "remote.origin.gh-resolved", "base")
            self.assertEqual(gh_guard.cwd_repo(d), FORK)
            git(d, "config", "remote.origin.gh-resolved", "someone/else")
            self.assertEqual(gh_guard.cwd_repo(d), "someone/else")

    def test_mixed_remotes_need_an_explicit_repo(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d, origin=FORK, mirror="torvalds/linux")
            argv = ["pr", "comment", "3", "--body", "hi"]
            with mock.patch.dict(os.environ, {"GH_REPO": ""}):
                refused = gh_guard.classify(argv, cwd=d)
                self.assertFalse(refused.allow)
                self.assertIn("--repo", refused.reason)
                self.assertTrue(gh_guard.classify(
                    argv + ["--repo", FORK], cwd=d).allow)

    def test_our_own_clone_resolves(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d, origin="AMD-Ecosystem/moat")
            with mock.patch.dict(os.environ, {"GH_REPO": ""}):
                self.assertTrue(gh_guard.classify(
                    ["issue", "create", "--title", "t", "--body", "b"], cwd=d).allow)

    def test_a_missing_directory_is_unresolved_not_a_crash(self):
        gone = os.path.join(tempfile.gettempdir(), "moat-no-such-dir", "x")
        with mock.patch.dict(os.environ, {"GH_REPO": ""}):
            self.assertFalse(gh_guard.classify(
                ["pr", "comment", "3", "--body", "hi"], cwd=gone).allow)


class CommandLines(unittest.TestCase):
    def test_command_table(self):
        for text, want in gh_guard.COMMAND_CASES:
            with self.subTest(text=text):
                self.assertEqual(not gh_guard.check_command(text), want)

    def test_prose_is_not_an_invocation(self):
        for text in (
            "cat > plan.md <<'EOF'\nThen run gh pr create --repo torvalds/linux\nEOF",
            "tee -a notes.md <<EOF\nnever run gh repo fork torvalds/linux\nEOF",
            'git commit -m "docs: say when gh pr merge is allowed"',
            "git commit -F msg.txt  # gh pr comment 3 --repo torvalds/linux",
            "echo 'gh pr comment 3 --repo torvalds/linux --body \"/moat approve\"'",
            'printf "%s\\n" "gh api -X DELETE repos/torvalds/linux"',
            "python fleet.py dispatch --prompt 'use gh pr view, never gh pr comment'",
            "$tool = 'gh pr create --repo torvalds/linux'",
            "Set-Clipboard @'\ngh pr comment 3 --repo torvalds/linux -b hi\n'@",
        ):
            with self.subTest(text=text):
                self.assertEqual(gh_guard.gh_invocations(text), [])
                self.assertEqual(gh_guard.check_command(text), [])

    def test_invocations_are_found_where_a_shell_would_run_them(self):
        for text in (
            "ls; gh pr comment 3 --repo torvalds/linux -b hi",
            "true && gh pr comment 3 --repo torvalds/linux -b hi",
            "(cd x; gh pr comment 3 --repo torvalds/linux -b hi)",
            "sudo env FOO=1 gh pr comment 3 --repo torvalds/linux -b hi",
            "timeout 30 gh pr comment 3 --repo torvalds/linux -b hi",
            "out=$(gh pr comment 3 --repo torvalds/linux -b hi)",
            "out=`gh pr comment 3 --repo torvalds/linux -b hi`",
            "sh -c 'gh pr comment 3 --repo torvalds/linux -b hi'",
            "bash <<'EOF'\ngh pr comment 3 --repo torvalds/linux -b hi\nEOF",
            "cat <<EOF\n$(gh pr comment 3 --repo torvalds/linux -b hi)\nEOF",
            "/usr/bin/gh pr comment 3 --repo torvalds/linux -b hi",
            "gh.exe pr comment 3 --repo torvalds/linux -b hi",
            "export GH_REPO=torvalds/linux\ngh pr comment 3 -b hi",
            "gh pr comment 3 \\\n  --repo torvalds/linux -b hi",
            "gh pr comment 3 --repo torvalds/linux -b hi 2>&1 | tail -3",
        ):
            with self.subTest(text=text):
                self.assertEqual(len(gh_guard.check_command(text)), 1)

    def test_a_write_that_depends_on_an_expansion_is_refused(self):
        for text in (
            "gh $CMD --repo AMD-Ecosystem/moat",
            "gh pr $VERB 3 --repo AMD-Ecosystem/moat",
            "gh pr comment 3 --repo \"$R\" --body-file README.md",
            "gh pr comment 3 --repo AMD-Ecosystem/moat --body \"$(cat b.md)\"",
            "gh api repos/AMD-Ecosystem/moat/issues/3/comments -f \"body=$B\"",
            "gh api repos/AMD-Ecosystem/moat/issues $FLAGS",
            "gh pr comment 3 --repo AMD-Ecosystem/moat $EXTRA --body-file README.md",
        ):
            with self.subTest(text=text):
                self.assertEqual(len(gh_guard.check_command(text)), 1)
        # Reads stay usable, and so does a literal target with a variable number.
        for text in (
            "gh pr view \"$URL\" --json state",
            "gh api \"repos/$slug/pulls/$n/commits\" --paginate",
            "gh pr edit \"$n\" --repo AMD-Ecosystem/moat --title \"records: $what\"",
        ):
            with self.subTest(text=text):
                self.assertEqual(gh_guard.check_command(text), [])

    def test_hook_event(self):
        def run(event):
            err = io.StringIO()
            with mock.patch.object(sys, "stderr", err):
                return gh_guard.hook_main(event), err.getvalue()

        bad = json.dumps({"tool_name": "Bash", "tool_input": {
            "command": "gh pr comment 3 --repo torvalds/linux --body hi"}})
        rc, err = run(bad)
        self.assertEqual(rc, 2)
        self.assertIn("torvalds/linux", err)
        self.assertEqual(
            run(json.dumps({"tool_input": {"command": "git status"}}))[0], 0)
        self.assertEqual(run(json.dumps({"tool_input": {"file_path": "x"}}))[0], 0)
        self.assertEqual(run("not json")[0], 2)


class Launcher(unittest.TestCase):
    """main() as the PATH shim: what runs, and what never reaches gh."""

    def run_shim(self, argv, inside, stdin=b""):
        ran = []

        def delegate(args, stdin_bytes=None):
            ran.append((list(args), stdin_bytes))
            return 0
        fake_in = mock.Mock()
        fake_in.buffer = io.BytesIO(stdin)
        with mock.patch.object(sys, "argv", ["gh_guard.py", "--shim", "--", *argv]), \
                mock.patch.object(gh_guard, "under_repo", lambda: inside), \
                mock.patch.object(gh_guard, "delegate", delegate), \
                mock.patch.object(gh_guard, "real_gh", lambda: "gh"), \
                mock.patch.object(sys, "stdin", fake_in), \
                mock.patch.object(sys, "stderr", io.StringIO()):
            return gh_guard.main(), ran

    def test_moat_line_is_refused_from_outside_a_checkout(self):
        rc, ran = self.run_shim(
            ["pr", "comment", f"https://github.com/{FORK}/pull/3",
             "--body", "/moat approve"], inside=False)
        self.assertEqual((rc, ran), (13, []))

    def test_the_refusal_names_the_interpreter_that_is_running(self):
        argv = ["pr", "comment", "9", "--repo", "pytorch/pytorch", "--body", "hi"]
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["gh_guard.py", "--shim", "--", *argv]), \
                mock.patch.object(gh_guard, "under_repo", lambda: True), \
                mock.patch.object(gh_guard, "real_gh", lambda: "gh"), \
                mock.patch.object(gh_guard, "PY", "python-on-this-host"), \
                mock.patch.object(sys, "stderr", err):
            self.assertEqual(gh_guard.main(), 13)
        self.assertIn("    python-on-this-host utils/upstream.py --publish --apply "
                      "--name <project>", err.getvalue())
        self.assertNotIn("python3", err.getvalue())

    def test_unrelated_write_outside_a_checkout_runs(self):
        argv = ["pr", "comment", "9", "--repo", "pytorch/pytorch", "--body", "hi"]
        rc, ran = self.run_shim(argv, inside=False)
        self.assertEqual((rc, ran), (0, [(argv, None)]))
        rc, ran = self.run_shim(argv, inside=True)
        self.assertEqual((rc, ran), (13, []))

    def test_stdin_body_is_inspected_and_passed_on(self):
        argv = ["pr", "comment", "3", "--repo", FORK, "--body-file", "-"]
        rc, ran = self.run_shim(argv, inside=True, stdin=b"fixed in abc1234\n")
        self.assertEqual((rc, ran), (0, [(argv, b"fixed in abc1234\n")]))
        for inside in (True, False):
            rc, ran = self.run_shim(argv, inside=inside,
                                    stdin=b"ok\n/moat approve\n")
            self.assertEqual((rc, ran), (13, []))

    def test_under_repo(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(gh_guard.under_repo(d))
            os.makedirs(os.path.join(d, "utils"))
            os.makedirs(os.path.join(d, ".git"))
            os.makedirs(os.path.join(d, "projects", "x", "src"))
            for rel in ("AGENTS.md", os.path.join("utils", "gh_guard.py")):
                open(os.path.join(d, rel), "w").close()
            self.assertTrue(gh_guard.under_repo(d))
            self.assertTrue(
                gh_guard.under_repo(os.path.join(d, "projects", "x", "src")))


if __name__ == "__main__":
    unittest.main()
