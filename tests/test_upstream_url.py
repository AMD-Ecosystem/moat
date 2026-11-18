"""set-upstream-url follows a rename of the recorded repository and nothing else."""
import copy
import unittest
from unittest import mock

import merge_status  # utils/ is put on sys.path by tests/__init__.py
import moatlib

OLD = "https://github.com/old-owner/proj"
NEW = "https://github.com/new-org/proj"


def record():
    return {"name": "proj", "upstream_url": OLD, "upstream_repo_id": 42}


class SetUpstreamUrl(unittest.TestCase):
    def run_it(self, obj, live, url=NEW, by="someone", writable=(True, "here")):
        saved = []
        with mock.patch.object(moatlib, "project_record", lambda n: (obj, "main")), \
                mock.patch.object(moatlib, "_gh_json", lambda *a: live), \
                mock.patch.object(moatlib, "record_writable_here",
                                  lambda n: writable), \
                mock.patch.object(moatlib, "save_record",
                                  lambda n, o, m: saved.append((copy.deepcopy(o), m))):
            out = moatlib.set_upstream_url("proj", url, by, reason="moved")
        return out, saved

    def test_a_rename_of_the_same_repository_is_recorded(self):
        out, saved = self.run_it(record(), {"id": 42, "full_name": "new-org/proj"})
        self.assertEqual(out["upstream_url"], NEW)
        r = out["upstream_url_ruling"]
        self.assertEqual((r["url"], r["previous"], r["by"], r["reason"]),
                         (NEW, OLD, "someone", "moved"))
        self.assertEqual(len(saved), 1)

    def test_canonical_case_is_what_gets_written(self):
        out, _ = self.run_it(record(), {"id": 42, "full_name": "New-Org/Proj"})
        self.assertEqual(out["upstream_url"], "https://github.com/New-Org/Proj")

    def refused(self, obj, live, pattern, **kw):
        with self.assertRaisesRegex(ValueError, pattern):
            self.run_it(obj, live, **kw)
        self.assertEqual(obj.get("upstream_url"), OLD)
        self.assertNotIn("upstream_url_ruling", obj)

    def test_a_different_repository_is_refused(self):
        self.refused(record(), {"id": 7, "full_name": "new-org/proj"},
                     "different repository")

    def test_an_unreachable_github_is_refused(self):
        self.refused(record(), None, "cannot read")

    def test_a_record_with_no_id_is_refused(self):
        obj = record()
        del obj["upstream_repo_id"]
        self.refused(obj, {"id": 42, "full_name": "new-org/proj"},
                     "no upstream_repo_id")

    def test_a_redirecting_name_is_refused(self):
        self.refused(record(), {"id": 42, "full_name": "new-org/proj"},
                     "redirects to new-org/proj",
                     url="https://github.com/other-old/proj")

    def test_no_by_is_refused(self):
        self.refused(record(), {"id": 42, "full_name": "new-org/proj"},
                     "--by is required", by=None)

    def test_the_same_url_is_refused(self):
        self.refused(record(), {"id": 42, "full_name": "old-owner/proj"},
                     "already", url=OLD)

    def test_a_malformed_url_is_refused(self):
        self.refused(record(), {"id": 42, "full_name": "new-org/proj"},
                     "expected https://github.com", url="new-org/proj")

    def test_an_unwritable_record_is_refused(self):
        self.refused(record(), {"id": 42, "full_name": "new-org/proj"},
                     "elsewhere", writable=(False, "proj: lives elsewhere"))


class MergeKeepsTheRename(unittest.TestCase):
    def test_a_stale_writer_cannot_restore_the_old_owner(self):
        base = {"schema_version": 3, "name": "proj", "upstream_url": OLD,
                "upstream_repo_id": 42, "platforms": {}, "waivers": {},
                "updated_at": "2026-01-01T00:00:00Z"}
        ruled = dict(base, upstream_url=NEW, updated_at="2026-01-02T00:00:00Z",
                     upstream_url_ruling={"url": NEW, "previous": OLD,
                                          "by": "someone",
                                          "at": "2026-01-02T00:00:00Z"})
        stale = dict(base, updated_at="2026-01-03T00:00:00Z")
        for ours, theirs in ((ruled, stale), (stale, ruled)):
            m = merge_status.merge(copy.deepcopy(ours), copy.deepcopy(theirs),
                                   copy.deepcopy(base))
            self.assertEqual(m["upstream_url"], NEW)
            self.assertEqual(m["upstream_url_ruling"]["by"], "someone")


if __name__ == "__main__":
    unittest.main()
