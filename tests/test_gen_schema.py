# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The schema generator writes only when asked to."""
import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import gen_schema  # utils/ is put on sys.path by tests/__init__.py


class Arguments(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = pathlib.Path(self._tmp.name) / "schema" / "status.schema.json"
        for p in (mock.patch.object(gen_schema, "OUT", self.out),
                  mock.patch.object(gen_schema.m, "REPO_ROOT",
                                    pathlib.Path(self._tmp.name))):
            p.start()
            self.addCleanup(p.stop)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = gen_schema.main(list(argv))
            except SystemExit as e:
                code = e.code
        return code, out.getvalue() + err.getvalue()

    def test_help_does_not_write(self):
        code, text = self.run_main("--help")
        self.assertEqual(code, 0)
        self.assertIn("--check", text)
        self.assertFalse(self.out.exists())

    def test_an_unknown_argument_does_not_write(self):
        code, _ = self.run_main("--chekc")
        self.assertEqual(code, 2)
        self.assertFalse(self.out.exists())

    def test_check_does_not_write_and_reports_a_missing_schema(self):
        code, _ = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertFalse(self.out.exists())

    def test_bare_command_writes_lf_and_then_checks_clean(self):
        code, _ = self.run_main()
        self.assertEqual(code, 0)
        raw = self.out.read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertEqual(json.loads(raw), gen_schema.build())
        self.assertEqual(self.run_main("--check")[0], 0)
