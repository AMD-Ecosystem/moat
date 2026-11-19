# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The hook and launcher texts, rendered and run under bash against scratch repos."""
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import gh_guard  # utils/ is put on sys.path by tests/__init__.py
import install_hooks
from tests import support

ZERO = "0" * 40
A, B = "a" * 40, "b" * 40


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@unittest.skipUnless(support.bash(), "needs bash")
class PrePush(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        # Stands in for check.py: says which interpreter ran it, exits as told.
        support.write_script(self.repo / "utils" / "check.py", (
            "import os, sys\n"
            "open('ran.txt', 'w').write(sys.executable + '\\n' + ' '.join(sys.argv[1:])"
            " + '\\n' + os.environ.get('MOAT_PUSH_UPDATES', ''))\n"
            "sys.exit(int(open('exit.txt').read()) if os.path.exists('exit.txt') else 0)\n"))

    def tearDown(self):
        self.tmp.cleanup()

    def push(self, python=None, stdin=None, path_first=None):
        hook = support.write_script(self.repo / "hook.sh",
                                    install_hooks.pre_push_text(python))
        stdin = stdin or f"refs/heads/x {A} refs/heads/x {ZERO}\n"
        return support.run_bash(hook, cwd=str(self.repo), stdin=stdin,
                                path_first=path_first)

    def ran(self):
        f = self.repo / "ran.txt"
        return f.read_text().splitlines() if f.exists() else None

    def test_gates_run_under_the_installing_interpreter(self):
        r = self.push()
        self.assertEqual(r.returncode, 0, r.stderr)
        ran = self.ran()
        self.assertEqual(pathlib.Path(ran[0]), pathlib.Path(sys.executable))
        self.assertEqual(ran[1], "--fast")
        self.assertIn(f"{A}:refs/heads/x:{ZERO}", ran[2])

    def test_failed_gates_refuse_and_do_not_advertise_a_bypass(self):
        (self.repo / "exit.txt").write_text("1")
        r = self.push()
        self.assertEqual(r.returncode, 1)
        self.assertIn("pre-push gates failed", r.stderr)
        self.assertNotIn("if genuinely urgent", r.stderr)
        self.assertIn("never an agent's", r.stderr)

    def test_a_push_that_only_deletes_runs_no_gate(self):
        r = self.push(stdin=f"(delete) {ZERO} refs/heads/gone {A}\n")
        self.assertEqual(r.returncode, 0)
        self.assertIsNone(self.ran())

    def fake_path(self, python3_works, python_works):
        """A directory whose `python3` and `python` either run the real interpreter
        or cannot import anything, to go at the front of PATH."""
        bindir = pathlib.Path(self.tmp.name) / "bin"
        for name, works in (("python3", python3_works), ("python", python_works)):
            support.write_script(bindir / name, "#!/bin/sh\n" + (
                f'exec "{support.PYTHON}" "$@"\n' if works else "exit 1\n"))
        return bindir

    def test_a_vanished_interpreter_falls_back_to_one_that_can_run_the_gates(self):
        gone = support.posix(pathlib.Path(self.tmp.name) / "venv" / "python")
        # `python3` here is the wrong Python (it cannot import the gates' packages),
        # which is exactly the host this was written for; `python` is the right one.
        r = self.push(python=gone, path_first=self.fake_path(False, True))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.ran()[1], "--fast")

    def test_no_usable_interpreter_refuses_and_says_why(self):
        gone = support.posix(pathlib.Path(self.tmp.name) / "venv" / "python")
        r = self.push(python=gone, path_first=self.fake_path(False, False))
        self.assertEqual(r.returncode, 1)
        self.assertIn("no Python that can run the pre-push gates", r.stderr)
        self.assertIn("install_hooks.py", r.stderr)
        self.assertIsNone(self.ran())

    def test_same_script_ignores_which_interpreter_but_not_a_missing_one(self):
        t = install_hooks.PRE_PUSH_TEMPLATE
        self.assertTrue(install_hooks._same_script(install_hooks.pre_push_text(), t))
        # Installed under another interpreter that still exists: current.
        other = support.posix(support.bash())
        self.assertTrue(install_hooks._same_script(
            install_hooks.pre_push_text(other), t))
        self.assertFalse(install_hooks._same_script(
            install_hooks.pre_push_text("/no/such/python"), t))
        self.assertFalse(install_hooks._same_script(
            install_hooks.pre_push_text() + "echo extra\n", t))
        old = t.replace('"$MOAT_PY" utils/check.py', "python3 utils/check.py")
        self.assertFalse(install_hooks._same_script(
            old.replace(install_hooks.PYTHON_SLOT, support.PYTHON), t))


@unittest.skipUnless(support.bash(), "needs bash")
class Launchers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        # Stands in for the guard: prints its arguments, one per line.
        self.guard = support.write_script(self.dir / "share dir" / "gh_guard.py",
                                          "import sys\nprint('\\n'.join(sys.argv[1:]))\n")

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, table, kind, python=support.PYTHON):
        name, body = table[kind]
        text = body.format(marker=gh_guard.SHIM_MARKER,
                           guard=self.guard.as_posix()).replace(
                               install_hooks.PYTHON_SLOT, python)
        return support.write_script(self.dir / name, text)

    def test_cases_the_gate_runs(self):
        self.assertEqual(install_hooks.launcher_case_problems(), [])

    def test_sh_launcher_passes_arguments_through(self):
        sh = self.render(install_hooks.GH_LAUNCHERS, "posix")
        r = support.run_bash(sh, "pr", "comment", "--body", "two words", "k=v w")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines(),
                         ["--shim", "--", "pr", "comment", "--body", "two words", "k=v w"])

    def test_sh_hook_launcher_runs_hook_mode(self):
        sh = self.render(install_hooks.HOOK_LAUNCHERS, "posix")
        r = support.run_bash(sh)
        self.assertEqual(r.stdout.splitlines(), ["--hook"])

    def test_sh_launcher_survives_a_vanished_interpreter(self):
        sh = self.render(install_hooks.GH_LAUNCHERS, "posix", python="/no/such/python")
        bindir = self.dir / "bin"
        for name in ("python3", "python"):
            support.write_script(bindir / name,
                                 f'#!/bin/sh\nexec "{support.PYTHON}" "$@"\n')
        r = support.run_bash(sh, "version", path_first=bindir)
        self.assertEqual(r.stdout.splitlines(), ["--shim", "--", "version"])

    @unittest.skipUnless(os.name == "nt", "cmd launcher")
    def test_cmd_launcher_passes_arguments_through(self):
        for table, want in ((install_hooks.GH_LAUNCHERS, ["--shim", "--", "pr", "list"]),
                            (install_hooks.HOOK_LAUNCHERS, ["--hook", "pr", "list"])):
            cmd = self.render(table, "nt")
            r = subprocess.run(["cmd", "/c", str(cmd), "pr", "list"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.splitlines(), want)

    def test_installed_launcher_names_the_guard_for_old_and_new_readers(self):
        for path, text in install_hooks.gh_launchers():
            self.assertEqual(install_hooks.gh_launcher_guard(text),
                             install_hooks.guard_copy_path())
            self.assertIn(support.PYTHON, text)
        for path, text in install_hooks.hook_launchers():
            self.assertIn("--hook", text)
            self.assertIsNone(install_hooks.gh_launcher_guard(text))

    def test_path_problem_says_when_the_real_gh_sorts_first(self):
        shim = self.dir / "bin" / "gh"
        support.write_script(shim, f"#!/bin/sh\n# {gh_guard.SHIM_MARKER}\n")
        real = support.write_script(self.dir / "real" / "gh", "#!/bin/sh\n")
        err = io.StringIO()
        with mock.patch.object(install_hooks.shutil, "which", lambda n: str(shim)):
            self.assertEqual(install_hooks.path_problem(shim), 0)
        with mock.patch.object(install_hooks.shutil, "which", lambda n: str(real)), \
                mock.patch.object(sys, "stderr", err):
            self.assertEqual(install_hooks.path_problem(shim), 1)
        self.assertIn("INERT", err.getvalue())
        self.assertIn("export PATH=", err.getvalue())


if __name__ == "__main__":
    unittest.main()
