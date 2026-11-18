# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Guards added after the 2026-11 health audit of the state machine."""
import contextlib
import copy
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py

HEAD = "a" * 40
OLD = "b" * 40
THIRD = "c" * 40


def record(**fields):
    out = {"schema_version": 3, "name": "proj",
           "upstream_url": "https://github.com/o/proj", "fork_default_branch": "main",
           "priority": 0, "ext_type": "cmake", "head_sha": HEAD,
           "stage": "review-passed", "platforms": {}}
    out.update(fields)
    return out


def done(sha=HEAD, **extra):
    return {"state": "completed", "blocked": False, "validated_sha": sha, **extra}


def failed(sha=HEAD, **extra):
    return {"state": "validation-failed", "blocked": False, "failed_sha": sha, **extra}


class Store:
    """An in-memory record standing in for load_status/save_status, so a test can
    drive the real transition code without a repository."""

    def __init__(self, case, obj, **patches):
        self.obj = obj
        self.saved = []
        self.err = io.StringIO()
        defaults = {
            "load_status": lambda name: copy.deepcopy(self.obj),
            "project_record": lambda name: (copy.deepcopy(self.obj), "local"),
            "save_status": self._save,
            "save_record": lambda name, o, message: self._save(name, o),
            "_clone_head": lambda name: None,
            "uncommitted_source_files": lambda name: [],
            "_publish_lock": lambda name, arch: copy.deepcopy(self.obj),
            "commit_and_push": lambda *a, **k: True,
            "full_sha": lambda sha, repo=None: sha,
        }
        defaults.update(patches)
        for attr, value in defaults.items():
            p = mock.patch.object(moatlib, attr, value)
            p.start()
            case.addCleanup(p.stop)
        r = contextlib.redirect_stderr(self.err)
        r.__enter__()
        case.addCleanup(r.__exit__, None, None, None)

    def _save(self, name, obj):
        self.obj = copy.deepcopy(obj)
        self.saved.append(self.obj)


class CliRefusal(unittest.TestCase):
    def run_main(self, argv, env=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.dict(os.environ, env or {}):
            os.environ.pop("MOAT_TRACEBACK", None) if not env else None
            code = moatlib.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_a_refusal_is_one_line_and_a_nonzero_exit(self):
        code, out, err = self.run_main(["set-state", "proj", "not-a-platform", "porting"])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertTrue(err.startswith("moatlib: "))
        self.assertNotIn("Traceback", err)

    def test_the_traceback_is_available_on_request(self):
        with self.assertRaises(ValueError):
            self.run_main(["set-state", "proj", "not-a-platform", "porting"],
                          env={"MOAT_TRACEBACK": "1"})

    def test_a_missing_record_is_a_refusal_too(self):
        def missing(name):
            raise FileNotFoundError(2, "No such file", f"projects/{name}/status.json")
        with mock.patch.object(moatlib, "load_status", missing):
            code, _out, err = self.run_main(["validate", "nope"])
        self.assertEqual(code, 1)
        self.assertIn("projects/nope/status.json", err)
        self.assertEqual(len(err.strip().splitlines()), 1)


class PortLockTake(unittest.TestCase):
    def test_a_malformed_platform_cannot_take_the_lock(self):
        s = Store(self, record(stage="porting"))
        for bad in ("gfx942", "linux-gfx-942", "linux-gfx999999z"):
            with self.subTest(arch=bad):
                with self.assertRaisesRegex(ValueError, "cannot take the lock"):
                    moatlib.port_lock("proj", take=bad)
        self.assertEqual(s.saved, [])

    def test_a_real_platform_takes_it(self):
        s = Store(self, record(stage="porting"))
        lock = moatlib.port_lock("proj", take="linux-gfx942")
        self.assertEqual(lock["arch"], "linux-gfx942")
        self.assertEqual(len(s.saved), 1)


class NotPortable(unittest.TestCase):
    def test_clear_is_refused_unless_the_stage_is_not_portable(self):
        s = Store(self, record(stage="review-passed"))
        with self.assertRaisesRegex(ValueError, "no verdict to lift"):
            moatlib.set_not_portable("proj", None, None, clear=True)
        self.assertEqual(s.saved, [])
        self.assertEqual(s.obj["stage"], "review-passed")

    def test_clear_lifts_a_standing_verdict(self):
        s = Store(self, record(stage="not-portable",
                               not_portable={"reason": "r", "by": "someone", "at": "t"}))
        moatlib.set_not_portable("proj", None, None, clear=True)
        self.assertEqual(s.obj["stage"], "planned")
        self.assertNotIn("not_portable", s.obj)

    def test_the_verdict_releases_a_held_lock(self):
        lock = {"arch": "linux-gfx942", "since": "2026-01-01T00:00:00Z", "holder": "h"}
        s = Store(self, record(stage="porting", porting=lock))
        moatlib.set_not_portable("proj", "no ROCm path", "someone")
        self.assertIsNone(s.obj["porting"])
        self.assertEqual(s.obj["porting_released"]["since"], lock["since"])
        self.assertEqual(s.obj["stage"], "not-portable")


class DepDocGaps(unittest.TestCase):
    def test_notes_are_read_as_utf8_whatever_the_locale(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = pathlib.Path(tmp) / "dep" / "notes.md"
            notes.parent.mkdir()
            notes.write_bytes("# dep “notes”\n\n## Install as a dependency\n"
                              .encode("utf-8"))
            user = record(name="user", depends_on=["dep"])
            with mock.patch.object(moatlib, "PROJECTS", pathlib.Path(tmp)), \
                    mock.patch.object(moatlib, "project_records",
                                      lambda: iter([("user", user, "local")])), \
                    mock.patch.object(moatlib, "project_record",
                                      lambda name: (record(name=name), "local")):
                # cp1252 cannot decode byte 0x9d, which is in the closing quote.
                real = pathlib.Path.read_text

                def strict(self, encoding=None, errors=None):
                    return real(self, encoding=encoding or "cp1252", errors=errors)
                with mock.patch.object(pathlib.Path, "read_text", strict):
                    self.assertEqual(moatlib.dep_doc_gaps(), [])


if __name__ == "__main__":
    unittest.main()
