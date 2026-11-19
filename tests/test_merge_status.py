# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import copy
import unittest

import merge_status  # utils/ is put on sys.path by tests/__init__.py
from merge_status import merge


def base():
    return {
        "name": "proj", "updated_at": "2026-01-01T00:00:00Z",
        "head_sha": "x" * 40, "stage": "review-passed",
        "published_sha": "p" * 40, "on_hold": True, "on_hold_reason": "parked",
        "platforms": {
            "linux-gfx90a": {"state": "completed", "validated_sha": "x" * 40,
                             "updated_at": "2026-01-01T00:00:00Z"},
        },
    }


def edit(rec, at, **fields):
    out = copy.deepcopy(rec)
    out.update(fields)
    out["updated_at"] = at
    return out


class StaleWriter(unittest.TestCase):
    """A host that read the record, lost a race and wrote later."""

    def setUp(self):
        self.o = base()
        # The porter advances the head and resumes the project at T1.
        self.porter = edit(self.o, "2026-01-02T00:00:00Z",
                           head_sha="y" * 40, stage="ported", published_sha="q" * 40)
        self.porter.pop("on_hold")
        self.porter.pop("on_hold_reason")
        # A validator records its platform at T2 from the copy it read before T1.
        self.validator = edit(self.o, "2026-01-03T00:00:00Z")
        self.validator["platforms"]["windows-gfx1151"] = {
            "state": "completed", "validated_sha": "x" * 40,
            "updated_at": "2026-01-03T00:00:00Z"}

    def check(self, m):
        self.assertEqual(m["head_sha"], "y" * 40)
        self.assertEqual(m["stage"], "ported")
        self.assertEqual(m["published_sha"], "q" * 40)
        self.assertNotIn("on_hold", m)
        self.assertNotIn("on_hold_reason", m)
        self.assertIn("windows-gfx1151", m["platforms"])
        self.assertIn("linux-gfx90a", m["platforms"])
        self.assertEqual(m["updated_at"], "2026-01-03T00:00:00Z")

    def test_stale_side_is_theirs(self):
        self.check(merge(self.porter, self.validator, self.o))

    def test_stale_side_is_ours(self):
        self.check(merge(self.validator, self.porter, self.o))

    def test_without_an_ancestor_recency_still_decides(self):
        # The old behaviour, kept for a merge git cannot give an ancestor for.
        m = merge(self.porter, self.validator)
        self.assertEqual(m["head_sha"], "x" * 40)


class BothChanged(unittest.TestCase):
    def test_later_writer_wins_a_field_both_changed(self):
        o = base()
        a = edit(o, "2026-01-02T00:00:00Z", head_sha="a" * 40)
        b = edit(o, "2026-01-03T00:00:00Z", head_sha="b" * 40)
        self.assertEqual(merge(a, b, o)["head_sha"], "b" * 40)
        self.assertEqual(merge(b, a, o)["head_sha"], "b" * 40)

    def test_hold_set_on_one_side_survives_a_newer_unrelated_write(self):
        o = base()
        o.pop("on_hold")
        o.pop("on_hold_reason")
        held = edit(o, "2026-01-02T00:00:00Z", on_hold=True, on_hold_reason="why")
        other = edit(o, "2026-01-03T00:00:00Z", stage="porting")
        m = merge(held, other, o)
        self.assertTrue(m["on_hold"])
        self.assertEqual(m["stage"], "porting")


class Platforms(unittest.TestCase):
    def blocks(self, **kw):
        o = base()
        return o, copy.deepcopy(o), copy.deepcopy(o)

    def test_failure_stands_over_a_later_pass_at_the_same_commit(self):
        o, a, b = self.blocks()
        a["platforms"]["linux-gfx1100"] = {
            "state": "validation-failed", "failed_sha": "x" * 40,
            "updated_at": "2026-01-02T00:00:00Z"}
        b["platforms"]["linux-gfx1100"] = {
            "state": "completed", "validated_sha": "x" * 40,
            "updated_at": "2026-01-03T00:00:00Z"}
        for m in (merge(a, b, o), merge(b, a, o), merge(a, b)):
            self.assertEqual(m["platforms"]["linux-gfx1100"]["state"],
                             "validation-failed")

    def test_pass_at_a_newer_commit_replaces_an_old_failure(self):
        o, a, b = self.blocks()
        a["platforms"]["linux-gfx1100"] = {
            "state": "validation-failed", "failed_sha": "x" * 40,
            "updated_at": "2026-01-02T00:00:00Z"}
        b["platforms"]["linux-gfx1100"] = {
            "state": "completed", "validated_sha": "y" * 40,
            "updated_at": "2026-01-03T00:00:00Z"}
        self.assertEqual(merge(a, b, o)["platforms"]["linux-gfx1100"]["state"],
                         "completed")

    def test_unchanged_block_does_not_beat_a_changed_one(self):
        o, a, b = self.blocks()
        a["platforms"]["linux-gfx90a"] = {
            "state": "revalidate", "updated_at": "2026-01-02T00:00:00Z"}
        # b still carries the ancestor's block untouched.
        self.assertEqual(merge(a, b, o)["platforms"]["linux-gfx90a"]["state"],
                         "revalidate")
        self.assertEqual(merge(b, a, o)["platforms"]["linux-gfx90a"]["state"],
                         "revalidate")


class Waivers(unittest.TestCase):
    SUGGESTED = {"reason": "no Windows build", "suggested_at": "2026-01-01T00:00:00Z"}

    def test_stale_suggestion_does_not_erase_an_approval(self):
        o = base()
        o["waivers"] = {"windows": dict(self.SUGGESTED)}
        approved = copy.deepcopy(o)
        approved["waivers"]["windows"].update(
            approved_by="someone", at="2026-01-02T00:00:00Z")
        stale = edit(o, "2026-01-05T00:00:00Z")
        for m in (merge(approved, stale, o), merge(stale, approved, o),
                  merge(approved, stale), merge(stale, approved)):
            self.assertEqual(m["waivers"]["windows"].get("approved_by"), "someone")

    def test_refusal_after_approval_wins(self):
        o = base()
        o["waivers"] = {"windows": dict(self.SUGGESTED)}
        a, b = copy.deepcopy(o), copy.deepcopy(o)
        a["waivers"]["windows"].update(approved_by="p", at="2026-01-02T00:00:00Z")
        b["waivers"]["windows"].update(refused_by="q", refused_at="2026-01-03T00:00:00Z")
        self.assertEqual(merge(a, b, o)["waivers"]["windows"].get("refused_by"), "q")
        self.assertEqual(merge(b, a, o)["waivers"]["windows"].get("refused_by"), "q")


class ExistingRules(unittest.TestCase):
    def test_lock_first_acquire_wins(self):
        o = base()
        a = edit(o, "2026-01-02T00:00:00Z",
                 porting={"arch": "linux-gfx90a", "since": "2026-01-02T00:00:00Z"})
        b = edit(o, "2026-01-02T00:00:05Z",
                 porting={"arch": "windows-gfx1151", "since": "2026-01-02T00:00:05Z"})
        self.assertEqual(merge(a, b, o)["porting"]["arch"], "linux-gfx90a")
        self.assertEqual(merge(b, a, o)["porting"]["arch"], "linux-gfx90a")

    def test_release_marker_frees_the_lock(self):
        lock = {"arch": "linux-gfx90a", "since": "2026-01-02T00:00:00Z"}
        o = edit(base(), "2026-01-02T00:00:00Z", porting=dict(lock))
        released = edit(o, "2026-01-03T00:00:00Z",
                        porting_released={**lock, "at": "2026-01-03T00:00:00Z"})
        released.pop("porting")
        still = edit(o, "2026-01-04T00:00:00Z", stage="porting")
        self.assertNotIn("porting", merge(released, still, o))
        self.assertNotIn("porting", merge(still, released, o))

    def test_sticky_decision_survives_a_writer_who_never_had_it(self):
        o = base()
        a = edit(o, "2026-01-02T00:00:00Z", review_pr="https://example.invalid/pull/1")
        b = edit(o, "2026-01-03T00:00:00Z", stage="porting")
        for m in (merge(a, b, o), merge(b, a, o), merge(a, b), merge(b, a)):
            self.assertEqual(m["review_pr"], "https://example.invalid/pull/1")

    def test_terminal_pr_state_outranks_open(self):
        o = edit(base(), "2026-01-01T00:00:00Z", pr_state="open")
        a = edit(o, "2026-01-02T00:00:00Z", pr_state="merged")
        b = edit(o, "2026-01-03T00:00:00Z", stage="porting")
        self.assertEqual(merge(a, b, o)["pr_state"], "merged")
        self.assertEqual(merge(b, a, o)["pr_state"], "merged")

    def test_merge_is_symmetric_for_disjoint_edits(self):
        o = base()
        a = edit(o, "2026-01-02T00:00:00Z", stage="porting")
        b = edit(o, "2026-01-03T00:00:00Z", head_sha="z" * 40)
        self.assertEqual(merge(a, b, o), merge(b, a, o))


class Stamp(unittest.TestCase):
    def test_stamp_picks_the_latest_known_time(self):
        self.assertEqual(merge_status._stamp(
            {"suggested_at": "1", "at": "3", "refused_at": "2"}), "3")
        self.assertEqual(merge_status._stamp({}), "")


if __name__ == "__main__":
    unittest.main()


class LockAfterTwoReleases(unittest.TestCase):
    """A host still carrying a lock that has since been released twice over."""

    L1 = {"arch": "linux-gfx942", "since": "2026-01-01T01:00:00Z", "holder": "aaaa"}
    L2 = {"arch": "linux-gfx90a", "since": "2026-01-01T03:00:00Z", "holder": "bbbb"}

    def test_a_lock_only_carried_from_the_ancestor_is_not_written_back(self):
        o = edit(base(), "2026-01-01T01:00:00Z", stage="porting", porting=self.L1)
        # Upstream: L1 released (ported), L2 taken (reviewing), L2 released -- the
        # one marker left names L2, so nothing matches the L1 the validator carries.
        upstream = edit(o, "2026-01-01T04:00:00Z", stage="review-passed", porting=None,
                        porting_released={"arch": "linux-gfx90a",
                                          "since": self.L2["since"],
                                          "at": "2026-01-01T04:00:00Z"})
        validator = edit(o, "2026-01-01T05:00:00Z")
        for a, b in ((validator, upstream), (upstream, validator)):
            self.assertIsNone(merge(a, b, o).get("porting"))

    def test_the_other_sides_new_lock_wins_over_the_carried_one(self):
        o = edit(base(), "2026-01-01T01:00:00Z", porting=self.L1)
        upstream = edit(o, "2026-01-01T03:00:00Z", porting=self.L2,
                        porting_released={"arch": "linux-gfx942",
                                          "since": self.L1["since"],
                                          "at": "2026-01-01T02:00:00Z"})
        stale = edit(o, "2026-01-01T05:00:00Z")
        for a, b in ((stale, upstream), (upstream, stale)):
            self.assertEqual(merge(a, b, o)["porting"], self.L2)

    def test_a_concurrent_acquisition_still_survives_a_lockless_writer(self):
        o = edit(base(), "2026-01-01T01:00:00Z")
        holder = edit(o, "2026-01-01T02:00:00Z", porting=self.L1)
        other = edit(o, "2026-01-01T03:00:00Z")
        for a, b in ((holder, other), (other, holder)):
            self.assertEqual(merge(a, b, o)["porting"], self.L1)

    def test_without_an_ancestor_an_unmatched_lock_stays_held(self):
        holder = edit(base(), "2026-01-01T02:00:00Z", porting=self.L1)
        other = edit(base(), "2026-01-01T03:00:00Z")
        self.assertEqual(merge(holder, other)["porting"], self.L1)


class DeliberateClear(unittest.TestCase):
    """A STICKY field one side removed on purpose, proven by the ancestor."""

    RULING = {"evidence": "https://example.invalid/readme", "by": "someone",
              "at": "2026-01-01T00:30:00Z"}

    def test_a_one_sided_clear_stands(self):
        for key, value in (("review_pr", "https://example.invalid/pull/1"),
                           ("maintained", self.RULING), ("no_upstream", self.RULING)):
            with self.subTest(key=key):
                o = edit(base(), "2026-01-01T01:00:00Z", **{key: value})
                cleared = edit(o, "2026-01-01T02:00:00Z")
                del cleared[key]
                other = edit(o, "2026-01-01T03:00:00Z", head_sha="y" * 40)
                for a, b in ((cleared, other), (other, cleared)):
                    self.assertNotIn(key, merge(a, b, o))

    def test_a_host_that_never_had_it_does_not_erase_it(self):
        o = edit(base(), "2026-01-01T01:00:00Z")
        ruled = edit(o, "2026-01-01T02:00:00Z", maintained=self.RULING)
        stale = edit(o, "2026-01-01T03:00:00Z", head_sha="y" * 40)
        for a, b in ((ruled, stale), (stale, ruled)):
            self.assertEqual(merge(a, b, o)["maintained"], self.RULING)

    def test_without_an_ancestor_the_value_is_kept(self):
        ruled = edit(base(), "2026-01-01T02:00:00Z", maintained=self.RULING)
        without = edit(base(), "2026-01-01T03:00:00Z")
        self.assertEqual(merge(ruled, without)["maintained"], self.RULING)

    def test_a_cleared_delivery_does_not_keep_the_release_schema_version(self):
        o = edit(base(), "2026-01-01T01:00:00Z", schema_version=4,
                 delivery={"mode": "fork-release", "by": "someone"})
        cleared = edit(o, "2026-01-01T02:00:00Z", schema_version=3)
        del cleared["delivery"]
        other = edit(o, "2026-01-01T03:00:00Z", head_sha="y" * 40)
        out = merge(cleared, other, o)
        self.assertNotIn("delivery", out)
        self.assertEqual(out["schema_version"], 3)


class ReopenedPr(unittest.TestCase):
    def test_a_reopen_is_not_undone_by_a_host_still_saying_closed(self):
        o = edit(base(), "2026-01-01T01:00:00Z", pr_state="closed", pr_number=7)
        reopened = edit(o, "2026-01-01T02:00:00Z", pr_state="open")
        stale = edit(o, "2026-01-01T03:00:00Z", head_sha="y" * 40)
        for a, b in ((reopened, stale), (stale, reopened)):
            self.assertEqual(merge(a, b, o)["pr_state"], "open")

    def test_a_close_still_beats_a_host_that_has_not_seen_it(self):
        o = edit(base(), "2026-01-01T01:00:00Z", pr_state="open", pr_number=7)
        closed = edit(o, "2026-01-01T02:00:00Z", pr_state="closed")
        stale = edit(o, "2026-01-01T03:00:00Z", head_sha="y" * 40)
        for a, b in ((closed, stale), (stale, closed)):
            self.assertEqual(merge(a, b, o)["pr_state"], "closed")

    def test_without_an_ancestor_the_terminal_state_outranks_open(self):
        closed = edit(base(), "2026-01-01T02:00:00Z", pr_state="closed")
        opened = edit(base(), "2026-01-01T03:00:00Z", pr_state="open")
        self.assertEqual(merge(closed, opened)["pr_state"], "closed")


class Ties(unittest.TestCase):
    def test_top_level_and_block_ties_break_the_same_way(self):
        at = "2026-01-01T02:00:00Z"
        a = edit(base(), at, head_sha="a" * 40)
        b = edit(base(), at, head_sha="b" * 40)
        for rec, sha in ((a, "a" * 40), (b, "b" * 40)):
            rec["platforms"]["linux-gfx90a"] = {"state": "completed",
                                                "validated_sha": sha, "updated_at": at}
        out = merge(a, b)
        self.assertEqual(out["head_sha"], "b" * 40)
        self.assertEqual(out["platforms"]["linux-gfx90a"]["validated_sha"], "b" * 40)
