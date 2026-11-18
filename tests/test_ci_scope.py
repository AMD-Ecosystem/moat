# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CI and the pre-push hook hand the gates the same description of a push.

The gates scope themselves from MOAT_PUSH_UPDATES and MOAT_PUSH_REFS. When only the
hook exported them, CI ran the trunk-owned gates against a port branch's inherited
files and never ran branch-scope, so the two disagreed about the same push."""
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import check  # utils/ is put on sys.path by tests/__init__.py

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA = "a" * 40
OLD = "b" * 40


def pushing(ref, new=SHA, old=OLD):
    return mock.patch.dict(os.environ, {"MOAT_PUSH_REFS": ref,
                                        "MOAT_PUSH_UPDATES": f"{new}:{ref}:{old}"})


def not_pushing():
    env = {k: v for k, v in os.environ.items()
           if k not in ("MOAT_PUSH_REFS", "MOAT_PUSH_UPDATES")}
    return mock.patch.dict(os.environ, env, clear=True)


class Workflow(unittest.TestCase):
    """ci.yml is not parsed as YAML (no parser in the standard library); the gate
    step's text is what has to carry the export."""

    def setUp(self):
        text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.text = text
        start = text.index("- name: Run gates")
        self.step = text[start:text.index("- name:", start + 1)]

    def test_push_event_is_replayed_through_the_hook_interface(self):
        self.assertIn('export MOAT_PUSH_REFS="$CI_REF"', self.step)
        self.assertIn('export MOAT_PUSH_UPDATES="$CI_AFTER:$CI_REF:$CI_BEFORE"', self.step)
        self.assertIn("github.event.before", self.step)
        self.assertIn("check.py --fast", self.step)

    def test_event_values_reach_the_shell_through_env_only(self):
        run = self.step[self.step.index("run: |"):]
        self.assertNotIn("${{", run)

    def test_no_step_interpolates_an_expression_into_its_script(self):
        # Every `run:` in the file, single-line or block: a ${{ }} there is pasted
        # into bash before bash parses it.
        lines = self.text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("run:"):
                continue
            indent = len(line) - len(line.lstrip())
            body = [stripped]
            for more in lines[i + 1:]:
                if more.strip() and len(more) - len(more.lstrip()) <= indent:
                    break
                body.append(more)
            self.assertNotIn("${{", "\n".join(body), line)
        self.assertIn('--base "origin/$PR_BASE"', self.text)
        self.assertIn("PR_BASE: ${{ github.base_ref }}", self.text)

    def test_check_dependencies_are_pinned_to_exact_versions(self):
        line = next(ln for ln in self.text.splitlines() if "pip install" in ln)
        packages = [w for w in line.split("pip install", 1)[1].split()
                    if not w.startswith("-")]
        self.assertEqual(sorted(p.split("==")[0] for p in packages),
                         ["jsonschema", "pyflakes"])
        for p in packages:
            self.assertRegex(p, r"^[a-z]+==\d+\.\d+\.\d+$")


class PushShape(unittest.TestCase):
    def test_ci_shaped_value_parses_like_the_hooks(self):
        with pushing("refs/heads/port/HEonGPU"):
            self.assertEqual(check.push_updates(),
                             [(SHA, "refs/heads/port/HEonGPU", OLD)])

    def test_new_branch_has_an_all_zero_before(self):
        with pushing("refs/heads/port/new", old=check.ZERO):
            self.assertEqual(check.push_updates()[0][2], check.ZERO)

    def test_port_only(self):
        with pushing("refs/heads/port/HEonGPU"):
            self.assertTrue(check.port_only_push())
        with pushing("refs/heads/main"):
            self.assertFalse(check.port_only_push())
        with not_pushing():
            self.assertFalse(check.port_only_push())


class TrunkOwnedGates(unittest.TestCase):
    """A port-only push is not judged on files the trunk owns. Each gate must return
    before it reads anything: the branch's copy is whatever it inherited."""

    def test_skipped_on_a_port_push_without_touching_the_tree(self):
        boom = mock.Mock(side_effect=AssertionError("gate read the tree"))
        with pushing("refs/heads/port/HEonGPU"), \
                mock.patch.object(check, "_run", boom), \
                mock.patch.object(check, "STANDALONE_SKILL", pathlib.Path("/nonexistent")), \
                mock.patch.object(check, "REPO", pathlib.Path("/nonexistent")):
            for gate in (check.gate_skills, check.gate_code, check.gate_licenses,
                         check.gate_harness_adapters):
                self.assertEqual(gate(), [], gate.__name__)

    def test_skills_gate_still_judges_a_trunk_push(self):
        with tempfile.TemporaryDirectory() as d:
            repo = pathlib.Path(d)
            skill = repo / ".claude" / "skills" / "cuda-to-rocm"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("run moatlib here\n", encoding="utf-8")
            where = (mock.patch.object(check, "REPO", repo),
                     mock.patch.object(check, "STANDALONE_SKILL", skill))
            with where[0], where[1]:
                with pushing("refs/heads/main"):
                    self.assertEqual(len(check.gate_skills()), 1)
                with pushing("refs/heads/port/x"):
                    self.assertEqual(check.gate_skills(), [])


class Readme(unittest.TestCase):
    def test_skipped_in_ci_even_when_the_trunk_is_pushed(self):
        boom = mock.Mock(side_effect=AssertionError("readme gate ran in CI"))
        with pushing("refs/heads/main"), \
                mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}), \
                mock.patch.object(check, "_run", boom):
            self.assertEqual(check.gate_readme(), [])

    def test_not_in_play_for_a_port_push_outside_ci(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return mock.Mock(stdout="main\n", returncode=0)

        env = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"}
        env.update({"MOAT_PUSH_REFS": "refs/heads/port/x"})
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(check, "_run", run):
            self.assertEqual(check.gate_readme(), [])
        self.assertFalse(any("gen_readme.py" in " ".join(map(str, c)) for c in calls))


class BranchScope(unittest.TestCase):
    def test_runs_when_ci_supplies_the_push(self):
        seen = []

        def run(cmd, **kw):
            seen.append(cmd)
            if cmd[:2] == ["git", "cat-file"]:
                return mock.Mock(returncode=0, stdout="")
            if cmd[:2] == ["git", "rev-list"]:
                return mock.Mock(returncode=0, stdout="c1\n")
            if cmd[:2] == ["git", "diff-tree"]:
                return mock.Mock(returncode=0, stdout="utils/moatlib.py\nprojects/x/notes.md\n")
            return mock.Mock(returncode=0, stdout="edit the tooling\n")

        with pushing("refs/heads/port/x"), mock.patch.object(check, "_run", run):
            problems = check.gate_branch_scope()
        self.assertEqual(len(problems), 1)
        self.assertIn("utils/moatlib.py", problems[0])
        self.assertIn(["git", "rev-list", "--no-merges", SHA, "^origin/main", f"^{OLD}"], seen)

    def test_judges_nothing_without_a_push(self):
        with not_pushing():
            self.assertEqual(check.gate_branch_scope(), [])


if __name__ == "__main__":
    unittest.main()
