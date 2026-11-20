# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Which copy of a project's record is the record."""
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py


class TrunkResidentRecordSeenFromAnotherPortBranch(unittest.TestCase):
    """A port branch's copy of another project's folder is a stale trunk snapshot."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.stale = pathlib.Path(self._tmp.name) / "status.json"
        self.stale.write_text(json.dumps({"name": "x", "copy": "stale"}))
        self.refs = {"origin/main": json.dumps({"name": "x", "copy": "trunk"})}
        for p in (mock.patch.object(moatlib, "status_path", lambda n: self.stale),
                  mock.patch.object(moatlib, "load_status",
                                    lambda n: json.loads(self.stale.read_text())),
                  mock.patch.object(moatlib, "port_branch_of", lambda n: None),
                  mock.patch.object(moatlib, "_ref_read",
                                    lambda ref, path: self.refs.get(ref))):
            p.start()
            self.addCleanup(p.stop)

    def read(self, branch):
        with mock.patch.object(moatlib, "current_branch", lambda: branch):
            return moatlib.project_record("x")

    def test_another_port_branch_reads_the_trunk(self):
        obj, where = self.read("port/other")
        self.assertEqual((obj["copy"], where), ("trunk", "trunk"))

    def test_the_trunk_checkout_reads_its_working_tree(self):
        obj, where = self.read("main")
        self.assertEqual((obj["copy"], where), ("stale", "local"))

    def test_a_tooling_branch_reads_its_working_tree(self):
        obj, where = self.read("tooling/something")
        self.assertEqual((obj["copy"], where), ("stale", "local"))

    def test_the_projects_own_branch_reads_its_working_tree(self):
        obj, where = self.read("port/x")
        self.assertEqual((obj["copy"], where), ("stale", "local"))

    def test_no_trunk_copy_falls_back_to_the_working_tree(self):
        self.refs.clear()
        obj, where = self.read("port/other")
        self.assertEqual((obj["copy"], where), ("stale", "local"))


if __name__ == "__main__":
    unittest.main()
