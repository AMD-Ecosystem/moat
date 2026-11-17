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
