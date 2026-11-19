# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import unittest

import moatlib  # utils/ is put on sys.path by tests/__init__.py

HEAD = "a" * 40
OLD = "b" * 40


def rec(platforms=None, waivers=None, head=HEAD):
    out = {"name": "proj", "platforms": platforms or {}, "waivers": waivers or {}}
    if head:
        out["head_sha"] = head
    return out


def done(sha=HEAD):
    return {"state": "completed", "validated_sha": sha}


class SameCommit(unittest.TestCase):
    def test_abbreviations_of_one_commit(self):
        self.assertTrue(moatlib.same_commit(HEAD, HEAD[:8]))
        self.assertTrue(moatlib.same_commit(HEAD[:7].upper(), HEAD))

    def test_different_commits(self):
        self.assertFalse(moatlib.same_commit(HEAD, OLD))

    def test_too_short_or_missing_is_not_identity(self):
        self.assertFalse(moatlib.same_commit("aaaa", HEAD))
        self.assertFalse(moatlib.same_commit(None, HEAD))
        self.assertFalse(moatlib.same_commit("", ""))

    def test_non_hex_falls_back_to_equality(self):
        self.assertTrue(moatlib.same_commit("not-a-sha", "not-a-sha"))
        self.assertFalse(moatlib.same_commit("not-a-sha", "not-a-shb"))


class Platforms(unittest.TestCase):
    def test_gates_follow_from_the_platform_name(self):
        self.assertEqual(moatlib.gates_for("linux-gfx90a"), {"wave64", "linux"})
        self.assertEqual(moatlib.gates_for("windows-gfx1151"), {"wave32", "windows"})
        self.assertEqual(moatlib.gates_for("linux-gfx1100"), {"wave32", "linux"})

    def test_malformed_platform_carries_no_gate(self):
        for p in ("gfx90a", "macos-gfx90a", "linux-GFX90A", "", None):
            with self.subTest(platform=p):
                self.assertEqual(moatlib.gates_for(p), set())
                self.assertIsNotNone(moatlib.platform_problem(p))

    def test_every_configured_family_resolves_to_a_required_wave_gate(self):
        for fam, gate in moatlib.WAVE_BY_FAMILY.items():
            with self.subTest(family=fam):
                self.assertIn(gate, moatlib.REQUIRED_GATES)
                self.assertEqual(moatlib.wave_of(fam), gate)

    def test_waivable_gates_are_required_gates(self):
        self.assertLessEqual(moatlib.WAIVABLE_GATES, set(moatlib.REQUIRED_GATES))


class Gates(unittest.TestCase):
    def test_nothing_validated(self):
        self.assertEqual(moatlib.unsatisfied_gates(rec()), set(moatlib.REQUIRED_GATES))

    def test_full_coverage_at_head(self):
        r = rec({"linux-gfx90a": done(), "windows-gfx1151": done()})
        self.assertEqual(moatlib.unsatisfied_gates(r), set())

    def test_one_windows_platform_carries_two_gates(self):
        r = rec({"windows-gfx1151": done()})
        self.assertEqual(moatlib.unsatisfied_gates(r), {"wave64"})

    def test_validation_of_an_older_commit_satisfies_nothing(self):
        r = rec({"linux-gfx90a": done(OLD), "windows-gfx1151": done()})
        self.assertEqual(moatlib.unsatisfied_gates(r), {"wave64"})

    def test_abbreviated_validated_sha_still_counts(self):
        r = rec({"linux-gfx90a": done(HEAD[:10])})
        self.assertTrue(moatlib.gate_satisfied(r, "wave64"))

    def test_only_completed_counts(self):
        for state in ("revalidate", "validation-failed", "port-ready", "blocked"):
            with self.subTest(state=state):
                r = rec({"linux-gfx90a": {"state": state, "validated_sha": HEAD}})
                self.assertFalse(moatlib.gate_satisfied(r, "wave64"))

    def test_no_recorded_head_means_no_validation_counts(self):
        r = rec({"linux-gfx90a": done()}, head=None)
        self.assertFalse(moatlib.gate_satisfied(r, "wave64"))

    def test_suggested_waiver_satisfies_nothing(self):
        r = rec(waivers={"windows": {"reason": "no Windows build"}})
        self.assertFalse(moatlib.gate_satisfied(r, "windows"))

    def test_approved_waiver_satisfies_a_waivable_gate(self):
        gate = sorted(moatlib.WAIVABLE_GATES)[0]
        r = rec(waivers={gate: {"reason": "r", "approved_by": "someone"}})
        self.assertTrue(moatlib.gate_satisfied(r, gate))

    def test_approved_waiver_on_a_non_waivable_gate_satisfies_nothing(self):
        fixed = set(moatlib.REQUIRED_GATES) - moatlib.WAIVABLE_GATES
        for gate in fixed:
            with self.subTest(gate=gate):
                r = rec(waivers={gate: {"reason": "r", "approved_by": "someone"}})
                self.assertFalse(moatlib.gate_satisfied(r, gate))

    def test_offline_platform_does_not_satisfy_a_gate_by_being_offline(self):
        for p in moatlib.OFFLINE_PLATFORMS:
            with self.subTest(platform=p):
                r = rec({p: {"state": "revalidate", "validated_sha": OLD}})
                for gate in moatlib.gates_for(p):
                    self.assertFalse(moatlib.gate_satisfied(r, gate))


if __name__ == "__main__":
    unittest.main()
