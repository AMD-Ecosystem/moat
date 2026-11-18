# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""When a checkout runs its own tooling and when it runs the trunk's."""
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import trunk  # utils/ is put on sys.path by tests/__init__.py


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=str(cwd), check=True, capture_output=True, text=True).stdout.strip()


class TrunkState(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = pathlib.Path(self._tmp.name).resolve()
        git(self.repo, "init", "-q")
        for rel, text in (("utils/tool.py", "print(1)\n"),
                          ("config/arches.toml", "[gates]\n"),
                          ("schema/status.schema.json", "{}\n"),
                          ("projects/x/status.json", "{}\n")):
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base")
        git(self.repo, "branch", "-M", "trunk")
        git(self.repo, "checkout", "-q", "-b", "port/x")
        p = mock.patch.object(trunk, "TRUNK_REF", "trunk")
        p.start()
        self.addCleanup(p.stop)

    def advance_trunk(self, rel, text):
        git(self.repo, "checkout", "-q", "trunk")
        (self.repo / rel).write_text(text, encoding="utf-8")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", f"change {rel}")
        git(self.repo, "checkout", "-q", "port/x")

    def current(self):
        return trunk.trunk_state(self.repo)[2]

    def test_a_branch_with_the_trunks_tooling_runs_its_own(self):
        self.assertTrue(self.current())

    def test_a_record_only_trunk_change_leaves_the_branch_current(self):
        self.advance_trunk("projects/x/status.json", '{"n": 1}\n')
        self.assertTrue(self.current())

    def test_a_code_change_on_the_trunk_makes_the_branch_stale(self):
        self.advance_trunk("utils/tool.py", "print(2)\n")
        self.assertFalse(self.current())

    def test_a_config_only_change_on_the_trunk_makes_the_branch_stale(self):
        self.advance_trunk("config/arches.toml", "[gates]\nrequired = []\n")
        self.assertFalse(self.current())

    def test_a_schema_only_change_on_the_trunk_makes_the_branch_stale(self):
        self.advance_trunk("schema/status.schema.json", '{"a": 1}\n')
        self.assertFalse(self.current())

    def test_a_branch_that_merged_the_trunk_is_current_again(self):
        self.advance_trunk("config/arches.toml", "[gates]\nrequired = []\n")
        git(self.repo, "merge", "-q", "--no-edit", "trunk")
        self.assertTrue(self.current())

    def test_no_trunk_ref_means_no_answer(self):
        with mock.patch.object(trunk, "TRUNK_REF", "no-such-ref"):
            self.assertIsNone(trunk.trunk_state(self.repo))
