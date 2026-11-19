"""set-pr-landed-via records a PR that closed because its work merged through another."""
import copy
import unittest
from unittest import mock

import gen_readme  # utils/ is put on sys.path by tests/__init__.py
import merge_status
import moatlib
import upstream

OURS = "https://github.com/up/proj/pull/10"
THEIRS = "https://github.com/up/proj/pull/12"
WHEN = "2026-10-01T11:45:32Z"


def record():
    return {"name": "proj", "pr_url": OURS, "pr_number": 10, "pr_state": "merged",
            "pr_merged_at": "2026-10-09T00:00:00Z"}


def mine(**kw):
    return dict({"state": "CLOSED", "mergedAt": None, "url": OURS,
                 "commits": [{"oid": "a" * 40}, {"oid": "b" * 40}]}, **kw)


def other(**kw):
    return dict({"state": "MERGED", "mergedAt": WHEN, "url": THEIRS,
                 "mergeCommit": {"oid": "f" * 40},
                 "commits": [{"oid": "a" * 40}, {"oid": "b" * 40},
                             {"oid": "c" * 40}]}, **kw)


class SetPrLandedVia(unittest.TestCase):
    def run_it(self, obj, m, o, url=THEIRS, by="someone", writable=(True, "here"),
               **kw):
        saved = []
        live = {"10": m, "12": o}
        with mock.patch.object(moatlib, "project_record", lambda n: (obj, "main")), \
                mock.patch.object(moatlib, "_gh_json", lambda *a: live.get(a[2])), \
                mock.patch.object(moatlib, "record_writable_here",
                                  lambda n: writable), \
                mock.patch.object(moatlib, "save_record",
                                  lambda n, ob, msg: saved.append(copy.deepcopy(ob))):
            out = moatlib.set_pr_landed_via("proj", url, by, **kw)
        return out, saved

    def test_carried_commits_are_recorded(self):
        out, saved = self.run_it(record(), mine(), other(), note="superseded")
        lv = out["landed_via"]
        self.assertEqual((lv["url"], lv["number"], lv["merged_at"], lv["carried"],
                          lv["merge_commit"], lv["by"], lv["note"]),
                         (THEIRS, 12, WHEN, "commits", "f" * 40, "someone",
                          "superseded"))
        self.assertEqual((out["pr_url"], out["pr_state"], out["pr_merged_at"]),
                         (OURS, "merged", WHEN))
        self.assertEqual(len(saved), 1)

    def refused(self, pattern, m=None, o=None, obj=None, **kw):
        obj = obj or record()
        with self.assertRaisesRegex(ValueError, pattern):
            self.run_it(obj, m or mine(), o or other(), **kw)
        self.assertNotIn("landed_via", obj)

    def test_a_rewrite_needs_a_person_to_say_so(self):
        o = other(commits=[{"oid": "a" * 40}])
        self.refused("does not carry 1 commit", o=o)
        out, _ = self.run_it(record(), mine(), o, content_differs=True)
        self.assertEqual(out["landed_via"]["carried"], "content-differs")

    def test_our_pr_still_open_is_refused(self):
        self.refused("not closed unmerged", m=mine(state="OPEN"))

    def test_our_pr_merged_itself_is_refused(self):
        self.refused("not closed unmerged", m=mine(state="MERGED", mergedAt=WHEN))

    def test_an_unmerged_superseder_is_refused(self):
        self.refused("not merged", o=other(state="OPEN", mergedAt=None))

    def test_another_repository_is_refused(self):
        self.refused("is not in up/proj", url="https://github.com/else/proj/pull/12")

    def test_the_same_pr_is_refused(self):
        self.refused("recorded PR itself", url=OURS)

    def test_an_unreachable_github_is_refused(self):
        self.refused("cannot read", url="https://github.com/up/proj/pull/99")

    def test_no_recorded_pr_is_refused(self):
        self.refused("no upstream PR recorded", obj={"name": "proj"})

    def test_no_by_is_refused(self):
        self.refused("--by is required", by=None)

    def test_a_malformed_url_is_refused(self):
        self.refused("expected https://github.com", url="up/proj#12")

    def test_an_unwritable_record_is_refused(self):
        self.refused("elsewhere", writable=(False, "proj: lives elsewhere"))


class LandedViaACommit(unittest.TestCase):
    """The maintainer took the work in a commit of their own, with no pull request."""
    SHA = "c" * 40
    URL = "https://github.com/up/proj/commit/" + "c" * 40

    def run_it(self, pr=None, commit="default", status="ahead", url=None, **kw):
        obj = record()
        if commit == "default":
            commit = {"sha": self.SHA, "html_url": self.URL,
                      "commit": {"committer": {"date": WHEN}}}

        def gh(*a):
            if a[0] == "pr":
                return pr or {"state": "CLOSED", "mergedAt": None}
            if "/compare/" in a[1]:
                return {"status": status}
            if "/commits/" in a[1]:
                return commit
            return {"default_branch": "master"}
        kw.setdefault("content_differs", True)
        with mock.patch.object(moatlib, "project_record", lambda n: (obj, "main")), \
                mock.patch.object(moatlib, "_gh_json", gh), \
                mock.patch.object(moatlib, "record_writable_here",
                                  lambda n: (True, "here")), \
                mock.patch.object(moatlib, "save_record", lambda n, ob, msg: None):
            return moatlib.set_pr_landed_via("proj", url or self.URL, "someone", **kw)

    def test_it_is_recorded_with_the_commit_and_its_date(self):
        out = self.run_it(note="synced")
        lv = out["landed_via"]
        self.assertEqual((lv["url"], lv["commit"], lv["merged_at"], lv["carried"],
                          lv["note"]),
                         (self.URL, self.SHA, WHEN, "content-differs", "synced"))
        self.assertNotIn("number", lv)
        self.assertEqual((out["pr_state"], out["pr_merged_at"]), ("merged", WHEN))

    def test_a_person_must_say_the_content_differs(self):
        with self.assertRaisesRegex(ValueError, "--content-differs"):
            self.run_it(content_differs=False)

    def test_a_commit_off_the_default_branch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not on up/proj's default branch"):
            self.run_it(status="diverged")

    def test_an_open_pr_another_repository_and_an_unreadable_commit_are_refused(self):
        with self.assertRaisesRegex(ValueError, "not closed unmerged"):
            self.run_it(pr={"state": "OPEN", "mergedAt": None})
        with self.assertRaisesRegex(ValueError, "is not in up/proj"):
            self.run_it(url="https://github.com/else/proj/commit/" + self.SHA)
        with self.assertRaisesRegex(ValueError, "cannot read"):
            self.run_it(commit=None)


class MergeKeepsTheRuling(unittest.TestCase):
    def test_a_stale_closed_cannot_replace_it(self):
        base = {"schema_version": 3, "name": "proj", "pr_url": OURS, "pr_number": 10,
                "pr_state": "merged", "pr_merged_at": "2026-10-09T00:00:00Z",
                "platforms": {}, "waivers": {}, "updated_at": "2026-01-01T00:00:00Z"}
        ruled = dict(base, pr_merged_at=WHEN, updated_at="2026-01-02T00:00:00Z",
                     landed_via={"url": THEIRS, "number": 12, "merged_at": WHEN,
                                 "carried": "commits", "by": "someone",
                                 "at": "2026-01-02T00:00:00Z"})
        stale = dict(base, pr_state="closed", updated_at="2026-01-03T00:00:00Z")
        for a, b in ((ruled, stale), (stale, ruled)):
            m = merge_status.merge(copy.deepcopy(a), copy.deepcopy(b),
                                   copy.deepcopy(base))
            self.assertEqual((m["pr_state"], m["pr_merged_at"]), ("merged", WHEN))
            self.assertEqual(m["landed_via"]["number"], 12)


class SweepAndBoard(unittest.TestCase):
    def poll(self, row, state):
        live = {"state": state, "mergedAt": None, "reviewDecision": None,
                "updatedAt": WHEN, "headRefOid": "a" * 40}
        with mock.patch.object(upstream, "gh_json", lambda args: live):
            return upstream.poll([row])[0]

    def row(self, **kw):
        return dict({"name": "proj", "repo": "up/proj", "num": "10", "url": OURS,
                     "ours": "merged", "published": "a" * 40, "head": "a" * 40}, **kw)

    def test_a_ruled_closed_pr_is_not_drift(self):
        self.assertEqual(self.poll(self.row(landed_via=THEIRS), "CLOSED"), [])

    def test_without_the_ruling_it_is(self):
        self.assertEqual(len(self.poll(self.row(), "CLOSED")), 1)

    def test_a_reopened_pr_is_drift_again(self):
        self.assertEqual(len(self.poll(self.row(landed_via=THEIRS), "OPEN")), 1)

    def test_the_board_links_a_commit(self):
        url = "https://github.com/up/proj/commit/" + "c" * 40
        p = dict(record(), landed_via={"url": url, "commit": "c" * 40})
        self.assertEqual(gen_readme.outcome_cell(p),
                         f"\U0001F7E3 [#10]({OURS}) via [ccccccccc]({url})")

    def test_the_board_links_both(self):
        p = dict(record(), landed_via={"url": THEIRS, "number": 12})
        self.assertEqual(gen_readme.outcome_cell(p),
                         f"\U0001F7E3 [#10]({OURS}) via [#12]({THEIRS})")


if __name__ == "__main__":
    unittest.main()
