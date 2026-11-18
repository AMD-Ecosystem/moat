# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The shell entry points: lib.sh's functions, timeit.sh, session.sh, orient.sh."""
import json
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

import gh_guard  # utils/ is put on sys.path by tests/__init__.py
from tests import support

UTILS = pathlib.Path(__file__).resolve().parents[1] / "utils"


@unittest.skipUnless(support.bash(), "needs bash")
class Lib(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.bin = self.dir / "bin"
        self.bin.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_lib(self, body, env=None):
        script = support.write_script(
            self.dir / "t.sh", f'. "{support.posix(UTILS / "lib.sh")}"\n{body}\n')
        e = {k: v for k, v in os.environ.items() if k != "MOAT_PYTHON"}
        e.update(env or {})
        return support.run_bash(script, env=e, path_first=self.bin)

    def fake_python(self, name, works):
        support.write_script(self.bin / name, "#!/bin/sh\n" + (
            f'exec "{support.PYTHON}" "$@"\n' if works else "exit 1\n"))

    def test_resolve_py_picks_by_what_it_can_import_not_by_name(self):
        # python3 is the wrong interpreter here; python is the one with the packages.
        self.fake_python("python3", False)
        self.fake_python("python", True)
        r = self.run_lib('moat_resolve_py; echo "rc=$? PY=$PY"')
        self.assertEqual(r.stdout.strip(), "rc=0 PY=python")
        self.fake_python("python3", True)
        r = self.run_lib('moat_resolve_py; echo "rc=$? PY=$PY"')
        self.assertEqual(r.stdout.strip(), "rc=0 PY=python3")

    def test_resolve_py_fails_when_nothing_can_run_the_tools(self):
        self.fake_python("python3", False)
        self.fake_python("python", False)
        r = self.run_lib('moat_resolve_py || moat_no_py orient; echo "rc=$? PY=[$PY]"')
        self.assertIn("PY=[]", r.stdout)
        self.assertIn("orient: no Python that can run MOAT's tools", r.stderr)
        self.assertIn("MOAT_PYTHON", r.stderr)

    def test_resolve_py_honours_an_explicit_interpreter(self):
        self.fake_python("python3", False)
        r = self.run_lib('moat_resolve_py; echo "PY=$PY"', env={"MOAT_PYTHON": "/x/py"})
        self.assertEqual(r.stdout.strip(), "PY=/x/py")

    def test_run_reports_a_failed_tool_in_one_line(self):
        tool = support.write_script(
            self.dir / "tool.sh", 'echo row1; echo "Traceback" >&2; echo >&2; '
            'echo "KeyError: boom" >&2; exit 3\n')
        r = self.run_lib(f'OUT=$(moat_run "tool.py --x" bash "{support.posix(tool)}"); '
                         f'echo "rc=$? out=$OUT"')
        self.assertEqual(r.stdout.strip(), "rc=3 out=row1")
        self.assertEqual(r.stderr.strip(),
                         "orient: tool.py --x FAILED (exit 3): KeyError: boom")

    def test_run_is_silent_when_the_tool_succeeds(self):
        tool = support.write_script(self.dir / "tool.sh", 'echo row1; echo warn >&2\n')
        r = self.run_lib(f'OUT=$(moat_run "tool" bash "{support.posix(tool)}"); '
                         f'echo "rc=$? out=$OUT"')
        self.assertEqual((r.stdout.strip(), r.stderr), ("rc=0 out=row1", ""))

    def guard_check(self, gh_text=None):
        if gh_text is not None:
            support.write_script(self.bin / "gh", gh_text)
        # PATH is only this directory plus wherever grep lives, so a real gh
        # installed on the test host is not what `command -v gh` finds.
        script = support.write_script(self.dir / "g.sh", (
            f'. "{support.posix(UTILS / "lib.sh")}"\n'
            'here=$(cd "$(dirname "$0")" && pwd)\n'
            'PATH="$here/bin:$(dirname "$(command -v grep)")"\n'
            'PY=python\nmoat_gh_guard_problem; echo "rc=$?"\n'))
        return support.run_bash(script)

    def test_guard_check_passes_on_the_launcher_and_on_no_gh(self):
        self.assertEqual(self.guard_check().stdout.strip(), "rc=0")
        r = self.guard_check(f"#!/bin/sh\n# {gh_guard.SHIM_MARKER}\n")
        self.assertEqual(r.stdout.strip(), "rc=0")

    def test_guard_check_fails_loudly_on_the_real_binary(self):
        r = self.guard_check("#!/bin/sh\necho real gh\n")
        self.assertTrue(r.stdout.strip().endswith("rc=1"), r.stdout)
        self.assertIn("NOT the MOAT guard launcher", r.stdout)
        self.assertIn('export PATH="$HOME/.local/bin:$PATH"', r.stdout)
        self.assertIn("install_hooks.py", r.stdout)

    def test_fleet_rows_with_and_without_a_lock_note(self):
        rows = self.dir / "rows.txt"
        with open(rows, "w", newline="\n") as f:
            f.write("HAMi-core\tbranch\tported\treviewer\tport/hami-core\n"
                    "colmap\tbranch\tporting\tporter\tport/colmap\t"
                    "work lock taken 2026-11-10 by another host or job on this platform\n"
                    "local\ttree\tplanned\tporter\tport/local\n"
                    "nobranch\tbranch\tscreened\tplanner\t\n")
        r = self.run_lib(f'moat_fleet_checkouts < "{support.posix(rows)}"')
        self.assertEqual(r.stdout.splitlines(), [
            "           HAMi-core (ported -> reviewer): git checkout port/hami-core",
            "           colmap (porting -> porter): git checkout port/colmap",
            "             WARNING -- work lock taken 2026-11-10 by another host or job "
            "on this platform",
            "           nobranch (screened -> planner): git checkout port/nobranch"])

    def test_warnings_are_picked_out_of_a_tools_stderr(self):
        err = self.dir / "err.txt"
        with open(err, "w", newline="\n") as f:
            f.write("moat: MOAT_LOCAL_TOOLS is set\nWARNING colmap: work lock taken\n")
        r = self.run_lib(f'moat_warnings "{support.posix(err)}"; '
                         f'moat_warnings "{support.posix(err)}.gone"; echo "rc=$?"')
        self.assertEqual(r.stdout.splitlines(), ["colmap: work lock taken", "rc=0"])

    def test_now_is_a_number(self):
        r = self.run_lib(f'PY="{support.PYTHON}"; moat_now; EPOCHREALTIME=; '
                         f'unset EPOCHREALTIME; moat_now')
        for line in r.stdout.split():
            self.assertGreater(float(line), 1e9)


@unittest.skipUnless(support.bash(), "needs bash")
class Telemetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "utils").mkdir()
        for name in ("lib.sh", "timeit.sh", "session.sh"):
            support.write_script(self.root / "utils" / name,
                                 (UTILS / name).read_text(encoding="utf-8"))
        (self.root / "projects" / "proj").mkdir(parents=True)
        self.stats = self.root / "projects" / "proj" / "stats.jsonl"
        self.env = {**os.environ, "MOAT_PYTHON": support.PYTHON}

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, name, *args):
        return support.run_bash(self.root / "utils" / name, *args, env=self.env)

    def records(self):
        return [json.loads(line) for line in self.stats.read_text().splitlines()]

    def test_timeit_records_and_passes_the_exit_code_through(self):
        r = self.run_tool("timeit.sh", "proj", "test", "--", "sh", "-c",
                          "echo ran; exit 7")
        self.assertEqual((r.returncode, r.stdout.strip()), (7, "ran"), r.stderr)
        (rec,) = self.records()
        self.assertEqual((rec["kind"], rec["phase"], rec["exit"]), ("phase", "test", 7))
        self.assertRegex(rec["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertGreaterEqual(rec["seconds"], 0)
        self.assertLess(rec["seconds"], 60)
        self.assertIn("exit 7", rec["cmd"])

    def test_timeit_redacts_home_directories(self):
        self.run_tool("timeit.sh", "proj", "misc", "--", "true",
                      "/home/alice/src", "C:\\Users\\bob\\build", "/c/Users/carol/x")
        cmd = self.records()[0]["cmd"]
        for name in ("alice", "bob", "carol"):
            self.assertNotIn(name, cmd)
        self.assertIn("/home/<user>/src", cmd)
        self.assertEqual(cmd.count("<user>"), 3)

    def test_a_mistyped_project_runs_nothing_and_creates_nothing(self):
        marker = self.root / "ran"
        r = self.run_tool("timeit.sh", "porj", "test", "--", "touch",
                          support.posix(marker))
        self.assertEqual(r.returncode, 2)
        self.assertIn("no projects/porj", r.stderr)
        self.assertFalse(marker.exists())
        self.assertFalse((self.root / "projects" / "porj").exists())
        r = self.run_tool("session.sh", "porj", "linux-gfx90a", "start")
        self.assertEqual(r.returncode, 2)
        self.assertFalse((self.root / "projects" / "porj").exists())

    def test_session_records_both_events_and_refuses_others(self):
        for event in ("start", "end"):
            r = self.run_tool("session.sh", "proj", "linux-gfx90a", event)
            self.assertEqual(r.returncode, 0, r.stderr)
        a, b = self.records()
        self.assertEqual([a["event"], b["event"]], ["start", "end"])
        self.assertEqual((a["kind"], a["platform"]), ("session", "linux-gfx90a"))
        self.assertGreaterEqual(b["epoch"], a["epoch"])
        self.assertEqual(self.run_tool("session.sh", "proj", "p", "pause").returncode, 2)

    def test_neither_needs_jq_or_gnu_date(self):
        for name in ("timeit.sh", "session.sh"):
            code = "\n".join(line for line in
                             (UTILS / name).read_text(encoding="utf-8").splitlines()
                             if not line.lstrip().startswith("#"))
            self.assertNotIn("jq ", code)
            self.assertNotIn("%N", code)


class OrientText(unittest.TestCase):
    """orient.sh pulls, merges and pushes, so it is not run here; these hold the
    properties the audit found missing in its text."""

    def code(self, name):
        return [line for line in (UTILS / name).read_text(encoding="utf-8").splitlines()
                if not line.lstrip().startswith("#")]

    def test_no_script_names_python3_as_a_command(self):
        for name in ("orient.sh", "setup_git.sh", "timeit.sh", "session.sh"):
            for line in self.code(name):
                self.assertIsNone(re.search(r"(^|[|;&(]\s*|\$\()\s*python3?\s", line),
                                  f"{name}: {line}")

    def test_orient_resolves_the_interpreter_and_checks_the_guard_first(self):
        text = "\n".join(self.code("orient.sh"))
        self.assertLess(text.index("moat_resolve_py"), text.index("install_hooks.py"))
        self.assertLess(text.index("moat_gh_guard_problem"),
                        text.index("upstream.py --forks"))
        self.assertLess(text.index("git fetch --quiet origin main"),
                        text.index("install_hooks.py"))

    def test_orient_discards_no_tool_error(self):
        for line in self.code("orient.sh"):
            if '"$PY"' in line:
                self.assertNotIn("2>/dev/null", line, line)
                self.assertNotIn("|| true", line, line)

    def test_orient_prints_the_selectors_lock_warning(self):
        text = "\n".join(self.code("orient.sh"))
        self.assertLess(text.index('LOCKWARN=$(moat_warnings "$next_err")'),
                        text.index('rm -f "$next_err"\nif [ "$NEXT" = "NONE" ]'))
        self.assertIn('printf \'%s\\n\' "$LOCKWARN"', text)
        # Fleet rows are parsed in one place, which knows the sixth column.
        self.assertNotIn("read -r proj where state stage branch", text)

    @unittest.skipUnless(support.bash(), "needs bash")
    def test_scripts_parse(self):
        for name in ("lib.sh", "orient.sh", "setup_git.sh", "timeit.sh", "session.sh",
                     "detect_arch.sh"):
            script = support.posix(UTILS / name)
            r = subprocess.run([support.bash(), "-n", script], capture_output=True,
                               text=True)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")


if __name__ == "__main__":
    unittest.main()
