# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import copy
import json
import unittest
from unittest import mock


import moatlib

OWNER = "OWNER"


def ev(login, body, at, state="ISSUE_COMMENT", assoc=OWNER, commit=None):
    return {"login": login, "state": state, "at": at, "commit": commit,
            "body": body, "assoc": assoc}


def pr(reviews=(), comments=(), **kw):
    base = {"state": "OPEN", "headRefOid": "a" * 40, "head_at": "2026-01-01T00:00:00Z",
            "review_list": list(reviews), "comment_list": list(comments),
            "lastEditedAt": None, "titleRenamedAt": None}
    base.update(kw)
    return base


class Pages(unittest.TestCase):
    def test_concatenated_pages(self):
        text = json.dumps([{"id": 1}]) + json.dumps([{"id": 2}, {"id": 3}]) + "\n"
        self.assertEqual([x["id"] for x in moatlib._parse_json_pages(text)], [1, 2, 3])

    def test_single_and_empty(self):
        self.assertEqual(moatlib._parse_json_pages("[]"), [])
        self.assertEqual(moatlib._parse_json_pages(""), [])

    def test_garbage_and_non_list_refuse(self):
        self.assertIsNone(moatlib._parse_json_pages("[1][2"))
        self.assertIsNone(moatlib._parse_json_pages('{"message": "Not Found"}'))


class Decisions(unittest.TestCase):
    def test_comment_approval_counts(self):
        p = pr(comments=[ev("m", "/moat approve", "2026-01-02T00:00:00Z")])
        self.assertEqual(moatlib.approval_currency(p)[0], "ok")

    def test_later_objection_withdraws(self):
        p = pr(comments=[ev("m", "/moat approve", "2026-01-02T00:00:00Z"),
                         ev("m", "/moat changes-requested", "2026-01-03T00:00:00Z")])
        self.assertIsNone(moatlib._approving_review(p))
        self.assertEqual(moatlib.approval_currency(p)[0], "bad-command")

    def test_objection_on_a_late_page_is_seen(self):
        # 40 chatter reviews after the approval, then the objection: what an
        # unpaginated fetch of 30 used to cut off.
        chatter = [ev("porter", f"addressed in c{i}", f"2026-01-02T00:{i:02d}:00Z",
                      state="COMMENTED") for i in range(40)]
        p = pr(reviews=[ev("m", "/moat approve", "2026-01-01T12:00:00Z",
                           state="COMMENTED", commit="a" * 40)] + chatter
               + [ev("m", "/moat changes-requested", "2026-01-03T00:00:00Z",
                     state="COMMENTED", commit="a" * 40)])
        self.assertIsNone(moatlib._approving_review(p))

    def test_chatter_does_not_undo_an_approval(self):
        p = pr(comments=[ev("m", "/moat approve", "2026-01-02T00:00:00Z"),
                         ev("m", "thanks", "2026-01-03T00:00:00Z")])
        self.assertIsNotNone(moatlib._approving_review(p))

    def test_fenced_quote_is_not_a_command(self):
        p = pr(comments=[ev("m", "Approve with:\n```\n/moat approve\n```\n",
                            "2026-01-02T00:00:00Z")])
        self.assertIsNone(moatlib._approving_review(p))

    def test_outsider_cannot_approve(self):
        p = pr(comments=[ev("x", "/moat approve", "2026-01-02T00:00:00Z",
                            assoc="NONE")])
        self.assertIsNone(moatlib._approving_review(p))

    def test_unknown_command_blocks(self):
        p = pr(comments=[ev("m", "/moat aprove", "2026-01-02T00:00:00Z")])
        self.assertEqual(moatlib.approval_currency(p)[0], "bad-command")

    def test_dismissed_review_no_longer_approves(self):
        p = pr(reviews=[ev("m", "/moat approve", "2026-01-02T00:00:00Z",
                           state="DISMISSED", commit="a" * 40)])
        self.assertIsNone(moatlib._approving_review(p))

    def test_push_after_review_approval_is_stale(self):
        p = pr(reviews=[ev("m", "/moat approve", "2026-01-02T00:00:00Z",
                           state="COMMENTED", commit="b" * 40)])
        self.assertEqual(moatlib.approval_currency(p)[0], "stale-commits")

    def test_commit_dated_after_comment_approval_is_stale(self):
        p = pr(comments=[ev("m", "/moat approve", "2026-01-02T00:00:00Z")],
               head_at="2026-01-05T00:00:00Z")
        self.assertEqual(moatlib.approval_currency(p)[0], "stale-commits")

    def test_body_edit_after_approval_is_stale(self):
        p = pr(comments=[ev("m", "/moat approve", "2026-01-02T00:00:00Z")],
               lastEditedAt="2026-01-04T00:00:00Z")
        self.assertEqual(moatlib.approval_currency(p)[0], "stale-content")

    def test_closed_review_pr(self):
        p = pr(comments=[ev("m", "/moat approve", "2026-01-02T00:00:00Z")],
               state="CLOSED")
        self.assertEqual(moatlib.approval_currency(p)[0], "closed")


NEW = "2026-12-01T00:00:00Z"        # after moatlib.SHA_REQUIRED_FROM
TIP = "a" * 40


class ShaBoundApproval(unittest.TestCase):
    """A conversation comment carries no commit, so it names one."""

    def test_cutoff_is_before_the_test_dates(self):
        self.assertLess(moatlib.SHA_REQUIRED_FROM, NEW)
        self.assertGreater(moatlib.SHA_REQUIRED_FROM, "2026-01-02T00:00:00Z")

    def test_comment_naming_the_head_is_current(self):
        for sha in (TIP, TIP[:7], TIP[:12].upper()):
            with self.subTest(sha=sha):
                p = pr(comments=[ev("m", f"/moat approve {sha}", NEW)])
                self.assertEqual(moatlib.approval_currency(p)[0], "ok")

    def test_commit_made_before_the_approval_and_pushed_after_it(self):
        # The gap the time comparison cannot see: the new tip's committer date is
        # OLDER than the approval, so by dates nothing landed after it.
        p = pr(comments=[ev("m", f"/moat approve {TIP[:12]}", NEW)],
               headRefOid="b" * 40, head_at="2026-11-25T00:00:00Z")
        code, why = moatlib.approval_currency(p)
        self.assertEqual(code, "stale-commits")
        self.assertIn("names a different commit", why)

    def test_new_comment_without_a_sha_is_unbound(self):
        p = pr(comments=[ev("m", "/moat approve", NEW)],
               head_at="2026-11-25T00:00:00Z")
        code, why = moatlib.approval_currency(p)
        self.assertEqual(code, "unbound")
        self.assertIn(f"/moat approve {TIP[:12]}", why)
        self.assertIn("unbound", moatlib.APPROVAL_CODES)

    def test_comment_from_before_the_rule_is_judged_as_it_was(self):
        old = "2026-11-01T00:00:00Z"
        p = pr(comments=[ev("m", "/moat approve", old)])
        self.assertEqual(moatlib.approval_currency(p)[0], "ok")
        p = pr(comments=[ev("m", "/moat approve", old)], head_at=NEW)
        self.assertEqual(moatlib.approval_currency(p)[0], "stale-commits")

    def test_review_form_needs_no_sha_and_honours_one(self):
        rev = dict(state="COMMENTED", commit=TIP)
        p = pr(reviews=[ev("m", "/moat approve", NEW, **rev)])
        self.assertEqual(moatlib.approval_currency(p)[0], "ok")
        p = pr(reviews=[ev("m", f"/moat approve {TIP[:8]}", NEW, **rev)])
        self.assertEqual(moatlib.approval_currency(p)[0], "ok")
        p = pr(reviews=[ev("m", f"/moat approve {'c' * 8}", NEW, **rev)])
        self.assertEqual(moatlib.approval_currency(p)[0], "stale-commits")
        p = pr(reviews=[ev("m", "", NEW, state="APPROVED", commit=TIP)])
        self.assertEqual(moatlib.approval_currency(p)[0], "ok")

    def test_malformed_sha_is_an_unknown_command(self):
        for body in ("/moat approve abc12", "/moat approve nothex1",
                     f"/moat approve {TIP[:8]} please"):
            with self.subTest(body=body):
                p = pr(comments=[ev("m", body, NEW)])
                self.assertEqual(moatlib.approval_currency(p)[0], "bad-command")

    def test_sha_form_is_still_an_approval_everywhere_else(self):
        e = ev("m", f"looks right\n/moat approve {TIP[:10]}\n", NEW)
        self.assertEqual(moatlib._command_of(e), "approve")
        self.assertEqual(moatlib.approved_sha(e), TIP[:10])
        self.assertIsNotNone(moatlib._approving_review(pr(comments=[e])))
        self.assertEqual(moatlib.moat_command_audit(pr(comments=[e]))[0], [])
        two = ev("m", f"/moat approve {TIP[:10]}\n/moat approve {'b' * 10}", NEW)
        self.assertIsNone(moatlib.approved_sha(two))

    def test_later_objection_withdraws_a_sha_bound_approval(self):
        p = pr(comments=[ev("m", f"/moat approve {TIP[:10]}", NEW),
                         ev("m", "/moat changes-requested", "2026-12-02T00:00:00Z")])
        self.assertIsNone(moatlib._approving_review(p))

    def test_edited_comment_is_judged_as_it_now_reads(self):
        # GitHub keeps created_at through an edit, and the gate reads the current
        # body: an approval edited to name another commit stops covering the head.
        p = pr(comments=[ev("m", f"/moat approve {'b' * 12}", NEW)])
        self.assertEqual(moatlib.approval_currency(p)[0], "stale-commits")


class RecordApproval(unittest.TestCase):
    """record_pr_approval applies the currency check the publish path uses."""

    def record(self, fetched):
        obj = {"review_pr": "https://github.com/AMD-Ecosystem/proj/pull/3",
               "head_sha": TIP}
        saved = []
        with mock.patch.object(moatlib, "load_status", lambda n: obj), \
                mock.patch.object(moatlib, "fetch_review_pr", lambda u: fetched), \
                mock.patch.object(moatlib, "save_record",
                                  lambda n, o, m: saved.append(o)):
            return moatlib.record_pr_approval("proj"), saved

    def test_current_approval_is_snapshotted(self):
        p = pr(comments=[ev("m", f"/moat approve {TIP[:12]}", NEW)], title="t", body="b")
        snap, saved = self.record(p)
        self.assertEqual((snap["approved_by"], snap["head_sha"]), ("m", TIP))
        self.assertEqual(len(saved), 1)

    def test_overtaken_approval_is_not_recorded(self):
        for p in (
            pr(reviews=[ev("m", "/moat approve", NEW, state="COMMENTED",
                           commit="b" * 40)]),
            pr(comments=[ev("m", f"/moat approve {'b' * 12}", NEW)]),
            pr(comments=[ev("m", "/moat approve", NEW)]),
            pr(comments=[ev("m", f"/moat approve {TIP[:12]}", NEW)],
               lastEditedAt="2026-12-05T00:00:00Z"),
        ):
            with self.assertRaises(ValueError) as cm:
                self.record(p)
            self.assertIn("does not cover", str(cm.exception))


class FixReviewBinding(unittest.TestCase):
    REC = {"fork_url": "https://github.com/AMD-Ecosystem/proj.git",
           "fix": {"branch": "moat-fix-7", "base_sha": "c" * 40}}

    def good(self, **kw):
        p = {"url": "https://github.com/AMD-Ecosystem/proj/pull/9",
             "headRefName": "moat-fix-7", "baseRefName": moatlib.PORT_BRANCH}
        p.update(kw)
        return p

    def test_the_rounds_own_review_binds(self):
        self.assertEqual(moatlib.fix_review_binding_problems(self.REC, self.good()), [])
        self.assertEqual(moatlib.fix_review_binding_problems(
            self.REC, self.good(slug="amd-ecosystem/Proj", url=None)), [])

    def test_another_fork_or_branch_does_not(self):
        for kw, want in (
            ({"url": "https://github.com/AMD-Ecosystem/other/pull/9"}, "not this"),
            ({"url": "https://github.com/up/proj/pull/9"}, "not this"),
            ({"headRefName": "scratch"}, "not the staging branch"),
            ({"headRefName": moatlib.PORT_BRANCH}, "not the staging branch"),
            ({"baseRefName": "main"}, "not the port branch"),
        ):
            with self.subTest(kw=kw):
                probs = moatlib.fix_review_binding_problems(self.REC, self.good(**kw))
                self.assertEqual(len(probs), 1)
                self.assertIn(want, probs[0])

    def test_recorded_fork_branch_is_the_base(self):
        rec = {**self.REC, "fork_branch": "release/1.0.amd0"}
        self.assertEqual(moatlib.fix_review_binding_problems(
            rec, self.good(baseRefName="release/1.0.amd0")), [])

    def set(self, url, live, ready=True):
        obj = copy.deepcopy(self.REC)
        obj["fix"]["review_pr"] = None
        saved = []
        with mock.patch.object(moatlib, "project_record", lambda n: (obj, "branch")), \
                mock.patch.object(moatlib, "fix_ready",
                                  lambda n: (ready, [("linux-gfx90a", "revalidate")], None)), \
                mock.patch.object(moatlib, "_gh_json", lambda *a: live), \
                mock.patch.object(moatlib, "save_record",
                                  lambda n, o, m: saved.append(o)):
            return moatlib.set_fix_review_pr("proj", url), saved

    def test_set_fix_review_pr_records_only_the_rounds_own_review(self):
        url = "https://github.com/AMD-Ecosystem/proj/pull/9"
        obj, saved = self.set(url, self.good())
        self.assertEqual((obj["fix"]["review_pr"], len(saved)), (url, 1))
        for bad_url, live, want in (
            ("https://example.com/whatever", self.good(), "not a GitHub pull request"),
            (url, None, "could not read"),
            (url, self.good(headRefName="scratch"), "not the staging branch"),
            ("https://github.com/AMD-Ecosystem/other/pull/2",
             self.good(url="https://github.com/AMD-Ecosystem/other/pull/2"),
             "not this project's fork"),
        ):
            with self.subTest(want=want):
                with self.assertRaises(ValueError) as cm:
                    self.set(bad_url, live)
                self.assertIn(want, str(cm.exception))

    def test_clearing_needs_no_checks(self):
        obj, saved = self.set(None, None, ready=False)
        self.assertIsNone(obj["fix"]["review_pr"])
        self.assertEqual(len(saved), 1)


class Binding(unittest.TestCase):
    REC = {"fork_url": "https://github.com/AMD-Ecosystem/proj",
           "fork_default_branch": "master", "head_sha": "a" * 40}

    def good(self, **kw):
        p = {"slug": "AMD-Ecosystem/proj", "headRefName": moatlib.PORT_BRANCH,
             "baseRefName": "master", "headRefOid": "a" * 40}
        p.update(kw)
        return p

    def test_matching_pr_binds(self):
        self.assertEqual(moatlib.review_pr_binding_problems(self.REC, self.good()), [])

    def test_slug_case_and_short_head(self):
        rec = {**self.REC, "head_sha": "a" * 12}
        p = self.good(slug="amd-ecosystem/Proj")
        self.assertEqual(moatlib.review_pr_binding_problems(rec, p), [])

    def test_another_projects_review_pr(self):
        probs = moatlib.review_pr_binding_problems(
            self.REC, self.good(slug="AMD-Ecosystem/other"))
        self.assertEqual(len(probs), 1)
        self.assertIn("not this project's fork", probs[0])

    def test_tip_moved_past_validated_head(self):
        probs = moatlib.review_pr_binding_problems(
            self.REC, self.good(headRefOid="b" * 40))
        self.assertEqual(len(probs), 1)
        self.assertIn("not the recorded head_sha", probs[0])

    def test_wrong_branches(self):
        probs = moatlib.review_pr_binding_problems(
            self.REC, self.good(headRefName="scratch", baseRefName="main"))
        self.assertEqual(len(probs), 2)

    def test_recorded_fork_branch_is_honoured(self):
        rec = {**self.REC, "fork_branch": "release/1.0.amd0"}
        self.assertEqual(moatlib.review_pr_binding_problems(
            rec, self.good(headRefName="release/1.0.amd0")), [])

    def test_missing_head_refuses(self):
        rec = {k: v for k, v in self.REC.items() if k != "head_sha"}
        self.assertTrue(moatlib.review_pr_binding_problems(rec, self.good()))


if __name__ == "__main__":
    unittest.main()
