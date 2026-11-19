# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""pr-candidates --freshness: reasons a gate-ready port cannot be sent as it stands."""
import unittest
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py


def record(**kw):
    return dict({"name": "proj", "upstream_url": "https://github.com/up/proj",
                 "fork_url": "https://github.com/ours/proj",
                 "fork_default_branch": "main"}, **kw)


class Freshness(unittest.TestCase):
    def ask(self, obj=None, info="default", cmp="default", theirs=(), deps=None):
        if info == "default":
            info = {"archived": False}
        if cmp == "default":
            cmp = {"ahead_by": 2, "behind_by": 0, "files": [{"filename": "a.cu"}],
                   "merge_base_commit": {"sha": "m" * 40}}

        def gh(*a):
            path = a[1]
            if path == "repos/up/proj":
                return info
            if "ours:" in path:
                return cmp
            return {"files": [{"filename": f} for f in theirs]}
        with mock.patch.object(moatlib, "_gh_json", gh), \
                mock.patch.object(moatlib, "project_record",
                                  lambda n: ((deps or {}).get(n), "branch")), \
                mock.patch.object(moatlib, "disposition_for_project", lambda n: None):
            return moatlib.upstream_freshness(obj or record())

    def behind(self, n):
        return {"ahead_by": 2, "behind_by": n, "files": [{"filename": "a.cu"}],
                "merge_base_commit": {"sha": "m" * 40}}

    def test_a_level_port_has_nothing_to_report(self):
        self.assertEqual(self.ask(), [])

    def test_upstream_moving_elsewhere_is_not_a_reason_to_hold(self):
        self.assertEqual(self.ask(cmp=self.behind(3), theirs=["docs.md"]), [])

    def test_upstream_changing_the_ports_files_is(self):
        (why,) = self.ask(cmp=self.behind(3), theirs=["a.cu", "b.cu"])
        self.assertIn("3 commit(s) past", why)
        self.assertIn("changed 1 of the port's 1 file(s) (a.cu)", why)

    def test_a_change_list_too_long_to_compare_is_reported(self):
        (why,) = self.ask(cmp=self.behind(900),
                          theirs=[f"f{i}" for i in range(300)])
        self.assertIn("too many to compare", why)

    def test_an_archived_upstream_and_an_empty_port_are_reported(self):
        why = self.ask(info={"archived": True},
                       cmp={"ahead_by": 0, "behind_by": 0, "files": []})
        self.assertEqual(len(why), 2)
        self.assertIn("archived", why[0])
        self.assertIn("nothing upstream lacks", why[1])

    def test_an_unmerged_dependency_port_is_reported(self):
        obj = record(depends_on=["dep", "done", "unknown"])
        deps = {"dep": {"pr_state": "open"}, "done": {"pr_state": "merged"}}
        (why,) = self.ask(obj=obj, deps=deps)
        self.assertIn("depends on dep", why)

    def test_an_unreadable_github_is_none_not_fresh(self):
        self.assertIsNone(self.ask(cmp=None))
        self.assertIsNone(moatlib.upstream_freshness({"name": "x"}))


if __name__ == "__main__":
    unittest.main()
