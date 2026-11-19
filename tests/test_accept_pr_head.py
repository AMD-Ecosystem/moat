# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""accept-pr-head records a person's ruling on a maintainer's push to our PR branch."""
import copy
import unittest
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py

OLD, NEW = "a" * 40, "b" * 40


def record(**kw):
    return dict({"name": "proj", "pr_url": "https://github.com/up/proj/pull/7",
                 "pr_number": 7, "pr_state": "open", "head_sha": OLD,
                 "published_sha": OLD,
                 "fork_url": "https://github.com/ours/proj"}, **kw)


class AcceptPrHead(unittest.TestCase):
    def run_it(self, obj=None, live="default", status="ahead", by="someone", **kw):
        obj = obj or record()
        saved, asked = [], []
        if live == "default":
            live = {"headRefOid": NEW, "state": "OPEN"}

        def gh(*a):
            asked.append(a)
            return live if a[0] == "pr" else {"status": status}
        with mock.patch.object(moatlib, "project_record", lambda n: (obj, "branch")), \
                mock.patch.object(moatlib, "_gh_json", gh), \
                mock.patch.object(moatlib, "ensure_port_branch",
                                  lambda n: (False, "already")), \
                mock.patch.object(moatlib, "record_writable_here",
                                  lambda n: (True, "here")), \
                mock.patch.object(moatlib, "save_record",
                                  lambda n, ob, msg: saved.append(copy.deepcopy(ob))):
            out = moatlib.accept_pr_head("proj", by, **kw)
        return out, saved, asked

    def test_the_new_head_becomes_the_published_tip(self):
        out, saved, asked = self.run_it(note="his merge of main")
        self.assertEqual(out["published_sha"], NEW)
        self.assertEqual(out["head_sha"], OLD)
        acc = out["pr_head_accepted"]
        self.assertEqual((acc["from"], acc["to"], acc["by"], acc["note"]),
                         (OLD, NEW, "someone", "his merge of main"))
        self.assertEqual(len(saved), 1)
        self.assertIn(f"repos/ours/proj/compare/{OLD}...{NEW}", asked[-1])

    def refused(self, pattern, **kw):
        obj = kw.pop("obj", None) or record()
        with self.assertRaisesRegex(ValueError, pattern):
            self.run_it(obj=obj, **kw)
        self.assertEqual(obj.get("published_sha"), record().get("published_sha")
                         if "published_sha" in obj else None)
        self.assertNotIn("pr_head_accepted", obj)

    def test_refusals(self):
        self.refused("--by is required", by=None)
        self.refused("no open upstream PR", obj=record(pr_state="merged"))
        self.refused("a round is in flight", obj=record(fix={"branch": "moat-fix-7"}))
        self.refused("cannot read", live=None)
        self.refused("not open", live={"headRefOid": NEW, "state": "CLOSED"})
        self.refused("nothing moved", live={"headRefOid": OLD, "state": "OPEN"})
        self.refused("was rewritten", status="diverged")
        obj = record()
        del obj["published_sha"]
        self.refused("no published_sha", obj=obj)


if __name__ == "__main__":
    unittest.main()
