# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py
import deferred


class Deferred(unittest.TestCase):
    """One global store and one project, `proj`, whose branch is checked out, so
    every read and write stays in the temporary directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = pathlib.Path(self._tmp.name).resolve()
        (root / "data").mkdir()
        (root / "projects" / "proj").mkdir(parents=True)
        self.registry = root / "data" / "deferred.json"
        self.project = root / "projects" / "proj" / "deferred.json"
        for p in (mock.patch.object(deferred, "REGISTRY", self.registry),
                  mock.patch.object(deferred, "FINDINGS", root / "findings"),
                  mock.patch.object(moatlib, "PROJECTS", root / "projects"),
                  mock.patch.object(moatlib, "all_projects",
                                    lambda: {"proj": "local"}),
                  mock.patch.object(moatlib, "current_branch", lambda: "port/proj"),
                  mock.patch.object(moatlib, "port_branches", lambda: [])):
            p.start()
            self.addCleanup(p.stop)
        deferred._branches_by_lowercase.cache_clear()
        self.addCleanup(deferred._branches_by_lowercase.cache_clear)
        self.run_cli("add", "other", "--id", "g1", "--summary", "global one")
        self.run_cli("add", "feature-port", "--id", "p1", "--project", "proj",
                     "--summary", "project one")

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            deferred.main(list(argv))
        return out.getvalue(), err.getvalue()

    def refused(self, *argv):
        """The refusal's text. argparse and sys.exit(message) both raise SystemExit;
        only the latter carries its message as the exit code."""
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as caught:
            deferred.main(list(argv))
        self.assertNotIn(caught.exception.code, (0, None))
        return f"{caught.exception.code} {err.getvalue()}"

    def item(self, item_id):
        path = self.registry if item_id.startswith("g") else self.project
        return deferred._find(json.loads(path.read_text()), item_id)

    def test_close_global_as_superseded(self):
        self.run_cli("close", "g1", "--as", "superseded", "--by", "someone",
                     "--reason", "fixed upstream", "--superseded-by", "g2")
        it = self.item("g1")
        self.assertEqual(it["status"], "superseded")
        self.assertEqual(
            {k: it["closed"][k] for k in ("as", "reason", "by", "superseded_by")},
            {"as": "superseded", "reason": "fixed upstream", "by": "someone",
             "superseded_by": "g2"})
        self.assertTrue(it["closed"]["at"])

    def test_close_project_as_not_applicable(self):
        self.run_cli("close", "p1", "--as", "not-applicable", "--by", "someone",
                     "--reason", "premise was wrong")
        it = self.item("p1")
        self.assertEqual(it["status"], "not-applicable")
        self.assertNotIn("superseded_by", it["closed"])
        # The other store is untouched.
        self.assertEqual(self.item("g1")["status"], "open")

    def test_closed_item_leaves_pending_and_list(self):
        self.assertIn("p1", self.run_cli("pending")[0])
        self.run_cli("close", "p1", "--as", "superseded", "--by", "someone",
                     "--reason", "fixed upstream")
        out, _ = self.run_cli("pending")
        self.assertNotIn("p1", out)
        self.assertIn("g1", out)
        self.assertNotIn("p1", self.run_cli("list")[0])
        shown = self.run_cli("list", "--all")[0]
        self.assertIn("(superseded) feature-port [proj] p1", shown)
        self.assertIn("closed (someone): fixed upstream", shown)

    def test_closed_now_deferral_is_no_longer_owed(self):
        self.run_cli("decide", "p1", "--choice", "now", "--by", "someone")
        self.assertEqual(moatlib.now_deferrals("proj"), ["p1"])
        self.run_cli("close", "p1", "--as", "not-applicable", "--by", "someone",
                     "--reason", "premise was wrong")
        self.assertEqual(moatlib.now_deferrals("proj"), [])
        # The ruling it was closed under stays on the record.
        self.assertEqual(self.item("p1")["decided"]["choice"], "now")

    def test_close_refuses_closed_or_done(self):
        self.run_cli("close", "g1", "--as", "superseded", "--by", "someone",
                     "--reason", "fixed upstream")
        self.assertIn("already superseded", self.refused(
            "close", "g1", "--as", "not-applicable", "--by", "other",
            "--reason", "again"))
        self.assertEqual(self.item("g1")["closed"]["by"], "someone")
        self.run_cli("set-status", "p1", "done")
        self.assertIn("already done", self.refused(
            "close", "p1", "--as", "superseded", "--by", "someone",
            "--reason", "late"))
        self.assertNotIn("closed", self.item("p1"))

    def test_close_accepts_a_filed_item(self):
        self.run_cli("set-status", "g1", "filed", "--upstream", "https://example.com/1")
        self.run_cli("close", "g1", "--as", "superseded", "--by", "someone",
                     "--reason", "duplicate of an earlier report")
        self.assertEqual(self.item("g1")["status"], "superseded")
        self.assertEqual(self.item("g1")["upstream_issue"], "https://example.com/1")

    def test_close_requires_by_reason_and_a_known_way(self):
        base = ("close", "g1", "--as", "superseded")
        self.assertIn("--by", self.refused(*base, "--reason", "r"))
        self.assertIn("--reason", self.refused(*base, "--by", "someone"))
        self.assertIn("--by is required",
                      self.refused(*base, "--by", " ", "--reason", "r"))
        self.assertIn("--reason is required",
                      self.refused(*base, "--by", "someone", "--reason", ""))
        self.assertIn("invalid choice", self.refused(
            "close", "g1", "--as", "done", "--by", "someone", "--reason", "r"))
        self.assertIn("unknown id", self.refused(
            "close", "nope", "--as", "superseded", "--by", "someone", "--reason", "r"))
        self.assertEqual(self.item("g1")["status"], "open")

    def test_set_status_cannot_close_or_reopen(self):
        self.assertIn("invalid choice", self.refused("set-status", "g1", "superseded"))
        self.run_cli("close", "g1", "--as", "superseded", "--by", "someone",
                     "--reason", "fixed upstream")
        self.assertIn("does not reopen", self.refused("set-status", "g1", "open"))
        self.assertEqual(self.item("g1")["status"], "superseded")

    def test_amend_keeps_history(self):
        self.run_cli("amend", "p1", "--summary", "project one, corrected",
                     "--by", "a", "--reason", "wrong cost estimate")
        self.run_cli("amend", "p1", "--summary", "project one, corrected twice",
                     "--by", "b", "--reason", "wrong option set")
        it = self.item("p1")
        self.assertEqual(it["summary"], "project one, corrected twice")
        self.assertEqual(
            [(h["summary"], h["by"], h["reason"]) for h in it["summary_history"]],
            [("project one", "a", "wrong cost estimate"),
             ("project one, corrected", "b", "wrong option set")])
        self.assertTrue(all(h["at"] for h in it["summary_history"]))
        self.assertEqual(it["status"], "open")
        self.assertIn("summary amended 2x, last by b: wrong option set",
                      self.run_cli("list")[0])

    def test_amend_global_leaves_a_ruling_and_says_so(self):
        self.run_cli("decide", "g1", "--choice", "defer", "--by", "someone")
        _, err = self.run_cli("amend", "g1", "--summary", "global one, corrected",
                              "--by", "a", "--reason", "premise was wrong")
        self.assertIn("ruled `defer` by someone", err)
        self.assertEqual(self.item("g1")["decided"]["choice"], "defer")
        self.assertEqual(self.item("g1")["summary_history"][0]["summary"], "global one")

    def test_amend_requires_by_reason_and_a_change(self):
        base = ("amend", "g1", "--summary", "new text")
        self.assertIn("--by", self.refused(*base, "--reason", "r"))
        self.assertIn("--reason", self.refused(*base, "--by", "a"))
        self.assertIn("--by is required",
                      self.refused(*base, "--by", "", "--reason", "r"))
        self.assertIn("--reason is required",
                      self.refused(*base, "--by", "a", "--reason", " "))
        self.assertIn("already has that summary", self.refused(
            "amend", "g1", "--summary", "global one", "--by", "a", "--reason", "r"))
        self.assertIn("unknown id", self.refused(
            "amend", "nope", "--summary", "x", "--by", "a", "--reason", "r"))
        self.assertEqual(self.item("g1")["summary"], "global one")
        self.assertNotIn("summary_history", self.item("g1"))


if __name__ == "__main__":
    unittest.main()
