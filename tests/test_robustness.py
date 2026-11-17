# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Guards added after the tooling audit: what a verdict, a head move, a lock and a
fork release are allowed to rest on."""
import contextlib
import io
import unittest
from unittest import mock

import merge_status  # utils/ is put on sys.path by tests/__init__.py
import moatlib

HEAD = "a" * 40
OLD = "b" * 40


class VerdictNamesTheCommitBuilt(unittest.TestCase):
    def problem(self, clone_head, sha=None, head=HEAD):
        with mock.patch.object(moatlib, "_clone_head", lambda name: clone_head):
            return moatlib.verdict_sha_problem(
                "proj", {"head_sha": head}, "linux-gfx90a", "completed", sha)

    def test_clone_at_the_head_passes(self):
        self.assertIsNone(self.problem(HEAD))

    def test_clone_at_another_commit_is_refused(self):
        self.assertIn("not the one built", self.problem(OLD))

    def test_no_clone_and_no_sha_has_nothing_to_compare(self):
        self.assertIsNone(self.problem(None))

    def test_named_sha_overrides_the_clone(self):
        self.assertIsNone(self.problem(OLD, sha=HEAD[:12]))

    def test_named_sha_that_is_not_the_head_is_refused(self):
        self.assertIn("--sha", self.problem(HEAD, sha=OLD))

    def test_a_record_with_no_head_cannot_take_a_verdict(self):
        self.assertIn("no head_sha", self.problem(HEAD, head=None))


class CarryForwardGoesToTheHead(unittest.TestCase):
    def run_carry(self, sha):
        obj = {"head_sha": HEAD,
               "platforms": {"linux-gfx90a": {"state": "completed",
                                              "validated_sha": OLD}}}
        saved = []
        with mock.patch.object(moatlib, "load_status", lambda name: obj), \
                mock.patch.object(moatlib, "save_status",
                                  lambda name, o: saved.append(o)), \
                mock.patch.object(moatlib, "full_sha", lambda sha, repo=None: sha):
            moatlib.carry_forward("proj", "linux-gfx90a", sha, "binary-equiv", "same")
        return obj, saved

    def test_carried_to_the_head(self):
        obj, saved = self.run_carry(HEAD)
        self.assertEqual(obj["platforms"]["linux-gfx90a"]["validated_sha"], HEAD)
        self.assertEqual(len(saved), 1)

    def test_any_other_commit_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not the recorded head_sha"):
            self.run_carry("c" * 40)


class AdvanceHeadWithoutAClone(unittest.TestCase):
    def test_abbreviated_sha_is_refused_when_nothing_can_resolve_it(self):
        with mock.patch.object(moatlib, "ensure_port_branch",
                               lambda name: (False, "")), \
                mock.patch.object(moatlib, "load_status",
                                  lambda name: {"head_sha": OLD, "platforms": {}}), \
                mock.patch.object(moatlib, "_fork_repo",
                                  lambda name: moatlib.REPO_ROOT / "no-such-clone"):
            with self.assertRaisesRegex(ValueError, "full 40-character"):
                moatlib.advance_head("proj", "abc1234")


class ForkLineage(unittest.TestCase):
    OBJ = {"upstream_url": "https://github.com/Owner/Repo"}

    def test_a_fork_of_the_upstream_counts(self):
        self.assertIsNone(moatlib.fork_lineage_problem(
            self.OBJ, "AMD-Ecosystem/Repo", "AMD-Ecosystem/Repo\towner/repo\towner/repo\n"))

    def test_a_fork_of_a_fork_counts_through_its_source(self):
        self.assertIsNone(moatlib.fork_lineage_problem(
            self.OBJ, "AMD-Ecosystem/Repo", "AMD-Ecosystem/Repo\tmid/Repo\tOwner/Repo\n"))

    def test_a_same_named_repository_that_is_not_a_fork_does_not(self):
        self.assertIn("not a GitHub fork", moatlib.fork_lineage_problem(
            self.OBJ, "AMD-Ecosystem/Repo", "AMD-Ecosystem/Repo\t\t\n"))

    def test_a_fork_of_something_else_does_not(self):
        self.assertIn("not of owner/repo", moatlib.fork_lineage_problem(
            self.OBJ, "AMD-Ecosystem/Repo", "AMD-Ecosystem/Repo\tother/repo\tother/repo\n"))


class TipClock(unittest.TestCase):
    def test_later_of(self):
        a, b = "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"
        self.assertEqual(moatlib.later_of(a, b), b)
        self.assertEqual(moatlib.later_of(b, a), b)
        self.assertEqual(moatlib.later_of(a, None), a)
        self.assertIsNone(moatlib.later_of(None, None))


class LockHolders(unittest.TestCase):
    def lock(self, since, holder=None, arch="linux-gfx942"):
        out = {"arch": arch, "since": since}
        if holder:
            out["holder"] = holder
        return {"porting": out}

    def test_two_hosts_of_one_platform_first_acquire_wins(self):
        first = self.lock("2026-01-01T00:00:00Z", "aaaa")
        second = self.lock("2026-01-01T00:00:05Z", "bbbb")
        for a, b in ((first, second), (second, first)):
            self.assertEqual(merge_status._merge_lock(a, b)["holder"], "aaaa")

    def test_one_holder_reacquiring_keeps_the_later_lock(self):
        early = self.lock("2026-01-01T00:00:00Z", "aaaa")
        late = self.lock("2026-01-02T00:00:00Z", "aaaa")
        self.assertEqual(merge_status._merge_lock(early, late)["since"],
                         "2026-01-02T00:00:00Z")

    def test_locks_from_before_holders_behave_as_before(self):
        early = self.lock("2026-01-01T00:00:00Z")
        late = self.lock("2026-01-02T00:00:00Z", "aaaa")
        self.assertEqual(merge_status._merge_lock(early, late)["since"],
                         "2026-01-02T00:00:00Z")

    def test_holder_is_opaque_and_stable(self):
        h = moatlib.lock_holder()
        self.assertRegex(h, r"^[0-9a-f]{12}$")
        self.assertEqual(h, moatlib.lock_holder())


class UnreadableRecord(unittest.TestCase):
    def test_reported_once(self):
        moatlib._UNREADABLE.discard(("proj-x", "this checkout"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            moatlib._unreadable("proj-x", "this checkout", ValueError("bad json"))
            moatlib._unreadable("proj-x", "this checkout", ValueError("bad json"))
        self.assertEqual(err.getvalue().count("UNREADABLE"), 1)
        self.assertIn("proj-x", err.getvalue())


if __name__ == "__main__":
    unittest.main()
