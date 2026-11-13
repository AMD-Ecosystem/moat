# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import json
import unittest


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
