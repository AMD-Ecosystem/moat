# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Guards added after the 2026-11 health audit of the state machine."""
import contextlib
import copy
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import changeclass  # utils/ is put on sys.path by tests/__init__.py
import moatlib

HEAD = "a" * 40
OLD = "b" * 40
THIRD = "c" * 40


def record(**fields):
    out = {"schema_version": 3, "name": "proj",
           "upstream_url": "https://github.com/o/proj", "fork_default_branch": "main",
           "priority": 0, "ext_type": "cmake", "head_sha": HEAD,
           "stage": "review-passed", "platforms": {}}
    out.update(fields)
    return out


def done(sha=HEAD, **extra):
    return {"state": "completed", "blocked": False, "validated_sha": sha, **extra}


def failed(sha=HEAD, **extra):
    return {"state": "validation-failed", "blocked": False, "failed_sha": sha, **extra}


def classified(cls):
    """Patch the source classifier to return one class for any delta (None: the
    delta could not be classified)."""
    verdict = None if cls is None else changeclass.Verdict(
        cls, cls in ("doc-only", "comment-only"), f"x.cpp: {cls}")
    return mock.patch.object(moatlib, "_classify_safe", lambda repo, a, b: verdict)


class Store:
    """An in-memory record standing in for load_status/save_status, so a test can
    drive the real transition code without a repository."""

    def __init__(self, case, obj, **patches):
        self.obj = obj
        self.saved = []
        self.err = io.StringIO()
        defaults = {
            "load_status": lambda name: copy.deepcopy(self.obj),
            "project_record": lambda name: (copy.deepcopy(self.obj), "local"),
            "save_status": self._save,
            "save_record": lambda name, o, message: self._save(name, o),
            "_clone_head": lambda name: None,
            "uncommitted_source_files": lambda name: [],
            "_publish_lock": lambda name, arch: copy.deepcopy(self.obj),
            "commit_and_push": lambda *a, **k: True,
            "full_sha": lambda sha, repo=None: sha,
        }
        defaults.update(patches)
        for attr, value in defaults.items():
            p = mock.patch.object(moatlib, attr, value)
            p.start()
            case.addCleanup(p.stop)
        r = contextlib.redirect_stderr(self.err)
        r.__enter__()
        case.addCleanup(r.__exit__, None, None, None)

    def _save(self, name, obj):
        self.obj = copy.deepcopy(obj)
        self.saved.append(self.obj)


class CliRefusal(unittest.TestCase):
    def run_main(self, argv, env=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.dict(os.environ, env or {}):
            os.environ.pop("MOAT_TRACEBACK", None) if not env else None
            code = moatlib.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_a_refusal_is_one_line_and_a_nonzero_exit(self):
        code, out, err = self.run_main(["set-state", "proj", "not-a-platform", "porting"])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertTrue(err.startswith("moatlib: "))
        self.assertNotIn("Traceback", err)

    def test_the_traceback_is_available_on_request(self):
        with self.assertRaises(ValueError):
            self.run_main(["set-state", "proj", "not-a-platform", "porting"],
                          env={"MOAT_TRACEBACK": "1"})

    def test_a_missing_record_is_a_refusal_too(self):
        def missing(name):
            raise FileNotFoundError(2, "No such file", f"projects/{name}/status.json")
        with mock.patch.object(moatlib, "load_status", missing):
            code, _out, err = self.run_main(["validate", "nope"])
        self.assertEqual(code, 1)
        self.assertIn("projects/nope/status.json", err)
        self.assertEqual(len(err.strip().splitlines()), 1)


class PortLockTake(unittest.TestCase):
    def test_a_malformed_platform_cannot_take_the_lock(self):
        s = Store(self, record(stage="porting"))
        for bad in ("gfx942", "linux-gfx-942", "linux-gfx999999z"):
            with self.subTest(arch=bad):
                with self.assertRaisesRegex(ValueError, "cannot take the lock"):
                    moatlib.port_lock("proj", take=bad)
        self.assertEqual(s.saved, [])

    def test_a_real_platform_takes_it(self):
        s = Store(self, record(stage="porting"))
        lock = moatlib.port_lock("proj", take="linux-gfx942")
        self.assertEqual(lock["arch"], "linux-gfx942")
        self.assertEqual(len(s.saved), 1)


class NotPortable(unittest.TestCase):
    def test_clear_is_refused_unless_the_stage_is_not_portable(self):
        s = Store(self, record(stage="review-passed"))
        with self.assertRaisesRegex(ValueError, "no verdict to lift"):
            moatlib.set_not_portable("proj", None, None, clear=True)
        self.assertEqual(s.saved, [])
        self.assertEqual(s.obj["stage"], "review-passed")

    def test_clear_lifts_a_standing_verdict(self):
        s = Store(self, record(stage="not-portable",
                               not_portable={"reason": "r", "by": "someone", "at": "t"}))
        moatlib.set_not_portable("proj", None, None, clear=True)
        self.assertEqual(s.obj["stage"], "planned")
        self.assertNotIn("not_portable", s.obj)

    def test_the_verdict_releases_a_held_lock(self):
        lock = {"arch": "linux-gfx942", "since": "2026-01-01T00:00:00Z", "holder": "h"}
        s = Store(self, record(stage="porting", porting=lock))
        moatlib.set_not_portable("proj", "no ROCm path", "someone")
        self.assertIsNone(s.obj["porting"])
        self.assertEqual(s.obj["porting_released"]["since"], lock["since"])
        self.assertEqual(s.obj["stage"], "not-portable")


class DepDocGaps(unittest.TestCase):
    def test_notes_are_read_as_utf8_whatever_the_locale(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = pathlib.Path(tmp) / "dep" / "notes.md"
            notes.parent.mkdir()
            notes.write_bytes("# dep “notes”\n\n## Install as a dependency\n"
                              .encode("utf-8"))
            user = record(name="user", depends_on=["dep"])
            with mock.patch.object(moatlib, "PROJECTS", pathlib.Path(tmp)), \
                    mock.patch.object(moatlib, "project_records",
                                      lambda: iter([("user", user, "local")])), \
                    mock.patch.object(moatlib, "project_record",
                                      lambda name: (record(name=name), "local")):
                # cp1252 cannot decode byte 0x9d, which is in the closing quote.
                real = pathlib.Path.read_text

                def strict(self, encoding=None, errors=None):
                    return real(self, encoding=encoding or "cp1252", errors=errors)
                with mock.patch.object(pathlib.Path, "read_text", strict):
                    self.assertEqual(moatlib.dep_doc_gaps(), [])


class PlatformStateVocabulary(unittest.TestCase):
    def test_arch_states_are_accepted(self):
        moatlib.validate_status(record(platforms={
            "linux-gfx90a": done(), "linux-gfx1100": failed(),
            "windows-gfx1151": {"blocked": True}}))

    def test_stage_and_derived_words_are_refused_in_a_platform_block(self):
        for state in ("porting", "review-passed", "revalidate", "port-ready"):
            with self.subTest(state=state):
                with self.assertRaisesRegex(ValueError, "invalid state"):
                    moatlib.validate_status(record(platforms={
                        "linux-gfx90a": {"state": state, "blocked": False}}))


class NoDisposition(unittest.TestCase):
    """Derivation tests: no dispositions, no dependencies, nothing read from git."""

    def setUp(self):
        for attr, value in (("disposition_for_project", lambda name: None),
                            ("now_deferrals", lambda name: [])):
            p = mock.patch.object(moatlib, attr, value)
            p.start()
            self.addCleanup(p.stop)


class HoldTouchesNoState(NoDisposition):
    def held(self, **fields):
        return record(on_hold=True, on_hold_reason="parked", **fields)

    def test_a_held_project_still_owes_what_it_owed(self):
        obj = self.held(platforms={"linux-gfx90a": done(OLD)})
        self.assertFalse(moatlib.settled(obj))
        self.assertEqual(moatlib.outstanding(obj), [("linux-gfx90a", "revalidate")])
        self.assertTrue(moatlib.belongs_on_branch(obj))

    def test_the_selector_still_skips_it(self):
        obj = self.held(platforms={"linux-gfx90a": done(OLD)})
        with mock.patch.object(moatlib, "unmet_deps", lambda o, platform=None: []):
            self.assertFalse(moatlib.actionable(obj, "linux-gfx90a"))

    def test_a_held_project_is_parked_not_stalled(self):
        obj = self.held(stage="porting",
                        platforms={"linux-gfx90a": {"blocked": True}})
        self.assertFalse(moatlib.stalled(obj))


class FailedArchAfterItsGateIsMet(NoDisposition):
    WAIVER = {"windows": {"reason": "r", "approved_by": "someone"}}

    def test_a_standing_failure_asks_for_a_porter_while_its_gate_is_open(self):
        obj = record(platforms={"windows-gfx1151": failed(), "linux-gfx90a": done()})
        self.assertEqual(moatlib.arch_task(obj, "windows-gfx1151"),
                         ("porter", "validation-failed"))

    def test_an_approved_waiver_ends_the_demand(self):
        obj = record(platforms={"windows-gfx1151": failed(), "linux-gfx90a": done(),
                                "linux-gfx1100": done()}, waivers=self.WAIVER)
        self.assertIsNone(moatlib.arch_task(obj, "windows-gfx1151"))
        self.assertEqual(moatlib.outstanding(obj), [])

    def test_a_sibling_satisfying_the_gate_ends_it_too(self):
        obj = record(platforms={"linux-gfx1100": failed(), "linux-gfx90a": done(),
                                "windows-gfx1151": done()})
        self.assertIsNone(moatlib.arch_task(obj, "linux-gfx1100"))

    def test_a_waiver_for_one_gate_does_not_excuse_another(self):
        # windows-gfx1151 also carries wave32, which nothing else satisfies here.
        obj = record(platforms={"windows-gfx1151": failed(), "linux-gfx90a": done()},
                     waivers=self.WAIVER)
        self.assertEqual(moatlib.arch_task(obj, "windows-gfx1151"),
                         ("porter", "validation-failed"))


class GateGaps(NoDisposition):
    ROSTER = {"linux-gfx90a", "linux-gfx1100", "windows-gfx1151"}

    def test_a_gate_whose_only_platform_is_blocked_is_reported(self):
        obj = record(platforms={"linux-gfx90a": done(), "linux-gfx1100": done(),
                                "windows-gfx1151": {"blocked": True}})
        gaps = moatlib.gate_gaps(obj, self.ROSTER)
        self.assertEqual([g for g, _ in gaps], ["windows"])
        self.assertIn("blocked: windows-gfx1151", gaps[0][1])
        self.assertIsNone(moatlib.arch_task(obj, "windows-gfx1151"))

    def test_an_offline_platform_cannot_be_sent_either(self):
        obj = record(platforms={"linux-gfx90a": done(), "linux-gfx1100": done()})
        with mock.patch.object(moatlib, "OFFLINE_PLATFORMS", {"windows-gfx1151"}):
            gaps = moatlib.gate_gaps(obj, self.ROSTER)
        self.assertEqual(gaps, [("windows", "offline: windows-gfx1151")])

    def test_a_dispatchable_platform_means_no_gap(self):
        obj = record(platforms={"linux-gfx90a": done(), "linux-gfx1100": done()})
        with mock.patch.object(moatlib, "OFFLINE_PLATFORMS", set()):
            self.assertEqual(moatlib.gate_gaps(obj, self.ROSTER), [])

    def test_a_pending_waiver_is_named(self):
        obj = record(platforms={"linux-gfx90a": done(), "linux-gfx1100": done(),
                                "windows-gfx1151": {"blocked": True}},
                     waivers={"windows": {"reason": "r"}})
        self.assertIn("awaiting a maintainer", moatlib.gate_gaps(obj, self.ROSTER)[0][1])

    def test_only_reviewed_unheld_projects_are_judged(self):
        blocked = {"windows-gfx1151": {"blocked": True}}
        self.assertEqual(moatlib.gate_gaps(record(stage="porting", platforms=blocked),
                                           self.ROSTER), [])
        self.assertEqual(moatlib.gate_gaps(record(on_hold=True, platforms=blocked),
                                           self.ROSTER), [])


class BlockedNeverCounts(NoDisposition):
    def test_a_blocked_block_completed_at_head_satisfies_no_gate(self):
        obj = record(platforms={"linux-gfx90a": done(blocked=True, blocked_reason="r")})
        self.assertFalse(moatlib.gate_satisfied(obj, "wave64"))
        obj["platforms"]["linux-gfx90a"]["blocked"] = False
        self.assertTrue(moatlib.gate_satisfied(obj, "wave64"))

    def test_completed_cannot_be_recorded_on_a_blocked_platform(self):
        s = Store(self, record(platforms={
            "linux-gfx90a": {"blocked": True, "blocked_reason": "no GL context"}}))
        with self.assertRaisesRegex(ValueError, "recorded BLOCKED"):
            moatlib.set_state("proj", "linux-gfx90a", "completed")
        self.assertEqual(s.saved, [])

    def test_a_failure_can_still_be_recorded_on_a_blocked_platform(self):
        s = Store(self, record(platforms={
            "linux-gfx90a": {"blocked": True, "blocked_reason": "r"}}))
        moatlib.set_state("proj", "linux-gfx90a", "validation-failed")
        self.assertEqual(s.obj["platforms"]["linux-gfx90a"]["failed_sha"], HEAD)

    def test_a_blocked_validation_is_not_carried_forward(self):
        Store(self, record(platforms={"linux-gfx90a": done(OLD, blocked=True)}))
        with self.assertRaisesRegex(ValueError, "recorded BLOCKED"):
            moatlib.carry_forward("proj", "linux-gfx90a", HEAD, "binary-equiv", "same")


class SquashCarry(NoDisposition):
    def squash(self, platforms):
        s = Store(self, record(head_sha=OLD, platforms=platforms))
        fake = mock.Mock(returncode=0, stdout="tree-id\n")
        with mock.patch.object(moatlib.subprocess, "run", lambda *a, **k: fake), \
                mock.patch.object(moatlib, "_fork_repo", lambda name: "repo"):
            ok, info = moatlib.squash_carry_forward("proj", HEAD)
        self.assertTrue(ok)
        return s.obj, info

    def test_a_standing_failure_moves_with_the_squash(self):
        obj, info = self.squash({"linux-gfx90a": done(OLD),
                                 "windows-gfx1151": failed(OLD)})
        self.assertEqual(info["failed"], ["windows-gfx1151"])
        self.assertEqual(obj["platforms"]["windows-gfx1151"]["failed_sha"], HEAD)
        self.assertEqual(moatlib.arch_task(obj, "windows-gfx1151"),
                         ("porter", "validation-failed"))

    def test_a_failure_already_superseded_is_left_behind(self):
        obj, info = self.squash({"linux-gfx90a": done(OLD),
                                 "windows-gfx1151": failed(THIRD)})
        self.assertEqual(info["failed"], [])
        self.assertEqual(obj["platforms"]["windows-gfx1151"]["failed_sha"], THIRD)

    def test_a_carried_platform_records_how_it_got_there(self):
        obj, info = self.squash({"linux-gfx90a": done(OLD)})
        cf = obj["platforms"]["linux-gfx90a"]["carry_forward"]
        self.assertEqual((cf["method"], cf["from"], cf["to"]), ("squash", OLD, HEAD))
        self.assertEqual(info["carried"], ["linux-gfx90a"])

    def test_a_blocked_platform_counts_neither_before_nor_after(self):
        before = record(head_sha=OLD, platforms={"linux-gfx90a": done(OLD, blocked=True)})
        self.assertFalse(moatlib.gate_satisfied(before, "wave64"))
        obj, info = self.squash({"linux-gfx90a": done(OLD, blocked=True)})
        self.assertEqual(info["kept_blocked"], ["linux-gfx90a"])
        self.assertEqual(obj["platforms"]["linux-gfx90a"]["validated_sha"], OLD)
        self.assertFalse(moatlib.gate_satisfied(obj, "wave64"))


class DependencyPerOperatingSystem(NoDisposition):
    def setUp(self):
        super().setUp()
        dep = record(name="dep", platforms={"linux-gfx90a": done()})
        p = mock.patch.object(moatlib, "project_record",
                              lambda name: (dep, "branch") if name == "dep"
                              else (None, None))
        p.start()
        self.addCleanup(p.stop)
        self.dep = dep
        self.user = record(name="user", stage="planned", depends_on=["dep"])

    def test_a_linux_validation_releases_linux_dependents(self):
        self.assertEqual(moatlib.dep_status("dep", "linux-gfx1100"), ("ok", "completed"))
        self.assertTrue(moatlib.actionable(self.user, "linux-gfx1100"))

    def test_it_does_not_release_a_windows_dependent(self):
        verdict, detail = moatlib.dep_status("dep", "windows-gfx1151")
        self.assertEqual(verdict, "waiting")
        self.assertIn("windows", detail)
        self.assertFalse(moatlib.actionable(self.user, "windows-gfx1151"))
        self.assertEqual(moatlib.dep_report(self.user, "windows-gfx1151"),
                         [("dep", "waiting", detail)])

    def test_a_windows_validation_of_the_dependency_does(self):
        self.dep["platforms"]["windows-gfx1201"] = done()
        self.assertTrue(moatlib.actionable(self.user, "windows-gfx1151"))

    def test_without_a_platform_any_validation_counts(self):
        self.assertEqual(moatlib.dep_status("dep"), ("ok", "completed"))
        self.assertEqual(moatlib.unmet_deps(self.user), [])


class PrLifecycleGuards(unittest.TestCase):
    PR = {"pr_url": "https://github.com/o/proj/pull/7", "pr_number": 7}
    FIX = {"branch": "moat-fix-7", "base_sha": OLD, "review_pr": None}

    def test_recording_a_merge_twice_keeps_the_first_time(self):
        s = Store(self, record(pr_state="merged", pr_merged_at="2026-01-01T00:00:00Z",
                               **self.PR))
        moatlib.set_pr_merged("proj")
        self.assertEqual(s.saved, [])

    def test_a_merged_pr_cannot_be_recorded_closed(self):
        s = Store(self, record(pr_state="merged", pr_merged_at="2026-01-01T00:00:00Z",
                               **self.PR))
        with self.assertRaisesRegex(ValueError, "recorded MERGED"):
            moatlib.set_pr_closed("proj", "withdrawn")
        self.assertEqual(s.saved, [])

    def test_recording_a_close_twice_keeps_the_first_time(self):
        s = Store(self, record(pr_state="closed", pr_closed_at="2026-01-01T00:00:00Z",
                               **self.PR))
        moatlib.set_pr_closed("proj")
        self.assertEqual(s.saved, [])
        moatlib.set_pr_closed("proj", "superseded")
        self.assertEqual(s.obj["pr_closed_at"], "2026-01-01T00:00:00Z")
        self.assertEqual(s.obj["pr_closed_note"], "superseded")

    def test_a_finish_with_a_round_staged_is_recorded_and_says_so(self):
        for finish, state in ((moatlib.set_pr_merged, "merged"),
                              (moatlib.set_pr_closed, "closed")):
            with self.subTest(state=state):
                s = Store(self, record(pr_state="open", fix=dict(self.FIX), **self.PR))
                finish("proj")
                self.assertEqual(s.obj["pr_state"], state)
                self.assertIn("abandon-fix", s.err.getvalue())

    def test_fix_merged_must_name_the_staging_tip(self):
        s = Store(self, record(pr_state="open", fix=dict(self.FIX),
                               published_sha=OLD, **self.PR),
                  _fork_repo=lambda name: pathlib.Path("no-such-clone"))
        with self.assertRaisesRegex(ValueError, "not the recorded head_sha"):
            moatlib.set_fix_merged("proj", THIRD)
        self.assertEqual(s.saved, [])
        moatlib.set_fix_merged("proj", HEAD[:12])
        self.assertEqual(s.obj["published_sha"], HEAD[:12])
        self.assertIsNone(s.obj["fix"])


class SyncRoundInFlight(unittest.TestCase):
    def test_a_staged_sync_round_is_reported_after_upstream_moved(self):
        fix = {"branch": "moat-sync-1111111", "base_sha": OLD, "review_pr": None,
               "kind": "sync", "upstream_sha": "1" * 40}
        s = Store(self, record(maintained={"evidence": "e", "by": "someone"}, fix=fix,
                               published_sha=OLD))

        def no_network(*a, **k):
            raise AssertionError("an in-flight round must not need the upstream tip")
        with mock.patch.object(moatlib.subprocess, "run", no_network):
            self.assertEqual(moatlib.sync_branch("proj"), fix)
        self.assertEqual(s.saved, [])

    def test_a_fix_round_in_flight_still_refuses_a_sync_round(self):
        Store(self, record(maintained={"evidence": "e", "by": "someone"},
                           fix={"branch": "moat-fix-7", "base_sha": OLD},
                           published_sha=OLD))
        ls = mock.Mock(returncode=0, stdout="2" * 40 + "\trefs/heads/main\n")
        with mock.patch.object(moatlib.subprocess, "run", lambda *a, **k: ls):
            with self.assertRaisesRegex(ValueError, "already in flight"):
                moatlib.sync_branch("proj")


class WritersUseLf(unittest.TestCase):
    """What the tools write has LF line endings on every host."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        for attr, value in (("PROJECTS", self.root / "projects"),
                            ("DISPOSITIONS", self.root / "data" / "dispositions.json"),
                            ("OPTOUTS", self.root / "data" / "optout.json")):
            p = mock.patch.object(moatlib, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def assert_lf(self, path):
        raw = path.read_bytes()
        self.assertIn(b"\n", raw)
        self.assertNotIn(b"\r", raw)

    def test_status_json(self):
        with mock.patch.object(moatlib, "current_branch", lambda: "port/proj"), \
                mock.patch.object(moatlib, "_ref_read", lambda ref, path: None), \
                mock.patch.object(moatlib, "check_against_trunk", lambda obj: []):
            moatlib.save_status("proj", record())
        self.assert_lf(self.root / "projects" / "proj" / "status.json")

    def test_token_telemetry(self):
        moatlib.record_tokens("proj", 10, "validator:test")
        moatlib.record_tokens("proj", 20)
        self.assert_lf(self.root / "projects" / "proj" / "stats.jsonl")

    def test_shared_registries(self):
        moatlib.save_dispositions({"o/r": {"disposition": "skip"}})
        moatlib.save_optouts({"o": {"who": "o"}})
        self.assert_lf(moatlib.DISPOSITIONS)
        self.assert_lf(moatlib.OPTOUTS)


class ProjectFolderAcrossRefs(unittest.TestCase):
    """A file in a project's folder is read from where the folder lives."""

    SURFACE = '{"owed": [{"id": "test:gpu_tests", "reason": "windows is last"}]}\n'

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.projects = pathlib.Path(self._tmp.name) / "projects"
        self.refs = {}
        for attr, value in (
                ("PROJECTS", self.projects),
                ("current_branch", lambda: "main"),
                ("port_branch_of", lambda name: "port/proj"),
                ("_ref_read", lambda ref, path: self.refs.get((ref, path)))):
            p = mock.patch.object(moatlib, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def local(self, text):
        d = self.projects / "proj"
        d.mkdir(parents=True, exist_ok=True)
        (d / "surface.json").write_text(text, encoding="utf-8")

    def test_the_branch_copy_is_read_from_a_checkout_without_the_folder(self):
        self.refs[("origin/port/proj", "projects/proj/surface.json")] = self.SURFACE
        self.assertEqual(moatlib.project_file("proj", "surface.json"), self.SURFACE)

    def test_the_branch_copy_outranks_a_stale_local_one(self):
        self.refs[("origin/port/proj", "projects/proj/surface.json")] = self.SURFACE
        self.local('{"owed": []}\n')
        self.assertEqual(moatlib.project_file("proj", "surface.json"), self.SURFACE)

    def test_on_the_projects_own_branch_the_working_tree_is_read(self):
        self.refs[("origin/port/proj", "projects/proj/surface.json")] = '{"owed": []}\n'
        self.local(self.SURFACE)
        with mock.patch.object(moatlib, "current_branch", lambda: "port/proj"):
            self.assertEqual(moatlib.project_file("proj", "surface.json"), self.SURFACE)

    def test_the_trunk_is_the_last_resort_and_nowhere_is_none(self):
        self.assertIsNone(moatlib.project_file("proj", "surface.json"))
        self.refs[("origin/main", "projects/proj/surface.json")] = self.SURFACE
        self.assertEqual(moatlib.project_file("proj", "surface.json"), self.SURFACE)

    def test_owed_surface_blocks_the_pr_from_a_trunk_checkout(self):
        self.refs[("origin/port/proj", "projects/proj/surface.json")] = self.SURFACE
        obj = record(reviewed_sha=HEAD, platforms={"linux-gfx90a": done(),
                                                   "windows-gfx1151": done()})
        with mock.patch.object(moatlib, "license_gate", lambda name, o: (True, "ok")), \
                mock.patch.object(moatlib, "uncommitted_source_files", lambda name: []):
            blocking, _nonviable = moatlib._gate_blockers("proj", obj)
        self.assertEqual([b[0] for b in blocking], ["surface-owed"])
        self.assertIn("test:gpu_tests", blocking[0][1])

    def test_surface_owed_reads_the_same_place(self):
        import surface
        self.refs[("origin/port/proj", "projects/proj/surface.json")] = self.SURFACE
        self.assertEqual(surface.owed("proj"), {"test:gpu_tests": "windows is last"})
        self.assertEqual(surface.owed("proj", root=self._tmp.name), {})


class CommitProjectStaging(unittest.TestCase):
    def test_archived_notes_are_staged_with_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "proj"
            d.mkdir()
            for fn in ("status.json", "notes.md", "notes-archive.md", "notes-2026.md",
                       "scratch.txt"):
                (d / fn).write_text("x\n", encoding="utf-8")
            staged = []
            with mock.patch.object(moatlib, "PROJECTS", pathlib.Path(tmp)), \
                    mock.patch.object(moatlib, "commit_and_push",
                                      lambda paths, message: staged.extend(paths) or True):
                moatlib.commit_project("proj", "proj: archive old rounds")
        self.assertEqual(sorted(staged), [
            "projects/proj/notes-2026.md", "projects/proj/notes-archive.md",
            "projects/proj/notes.md", "projects/proj/status.json"])


class SetExtType(unittest.TestCase):
    def test_the_planner_records_the_build_type(self):
        s = Store(self, record(ext_type="unknown"))
        moatlib.set_ext_type("proj", "torch-extension")
        self.assertEqual(s.obj["ext_type"], "torch-extension")
        moatlib.set_ext_type("proj", "torch-extension")
        self.assertEqual(len(s.saved), 1)

    def test_an_empty_or_multi_word_value_is_refused(self):
        s = Store(self, record(ext_type="unknown"))
        for bad in ("", "  ", "cmake and meson"):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "one non-empty word"):
                    moatlib.set_ext_type("proj", bad)
        self.assertEqual(s.saved, [])


class WorkLockHolders(NoDisposition):
    """Two hosts of one platform are two holders."""

    P = "linux-gfx942"

    def setUp(self):
        super().setUp()
        p = mock.patch.object(moatlib, "lock_holder", lambda: "me")
        p.start()
        self.addCleanup(p.stop)
        self.published = []

    def store(self, stage, lock):
        fields = {"porting": lock} if lock else {}
        s = Store(self, record(stage=stage, **fields))
        p = mock.patch.object(
            moatlib, "_publish_lock",
            lambda name, arch: self.published.append(arch) or copy.deepcopy(s.obj))
        p.start()
        self.addCleanup(p.stop)
        return s

    def lock(self, holder="other", arch=None):
        out = {"arch": arch or self.P, "since": "2026-01-01T00:00:00Z"}
        if holder:
            out["holder"] = holder
        return out

    def test_a_second_host_of_the_platform_does_not_silently_join(self):
        s = self.store("porting", self.lock("other"))
        moatlib.set_state("proj", self.P, "porting")
        self.assertEqual(s.obj["porting"]["holder"], "me")
        self.assertNotEqual(s.obj["porting"]["since"], "2026-01-01T00:00:00Z")
        self.assertIn("DIFFERENT host or job", s.err.getvalue())
        self.assertEqual(self.published, [self.P])

    def test_the_holder_re_entering_its_own_stage_is_a_no_op(self):
        s = self.store("porting", self.lock("me"))
        moatlib.set_state("proj", self.P, "porting")
        self.assertEqual(s.saved, [])
        self.assertEqual(self.published, [])
        self.assertEqual(s.err.getvalue(), "")

    def test_a_lock_from_before_holders_is_re_stamped_and_said(self):
        s = self.store("planning", self.lock(None))
        moatlib.set_state("proj", self.P, "planning")
        self.assertEqual(s.obj["porting"]["holder"], "me")
        self.assertIn("no holder id", s.err.getvalue())
        self.assertEqual(self.published, [self.P])

    def test_another_platform_is_still_refused_outright(self):
        s = self.store("porting", self.lock("other", arch="linux-gfx90a"))
        with self.assertRaisesRegex(ValueError, "work lock is held by linux-gfx90a"):
            moatlib.set_state("proj", self.P, "porting")
        self.assertEqual(s.saved, [])

    def test_a_displaced_holder_cannot_release_the_new_holders_lock(self):
        s = self.store("porting", self.lock("other"))
        with self.assertRaisesRegex(ValueError, "another host or job has held"):
            moatlib.set_state("proj", self.P, "ported")
        self.assertEqual(s.saved, [])
        self.assertEqual(s.obj["stage"], "porting")

    def test_another_platform_cannot_record_the_exit(self):
        s = self.store("reviewing", self.lock("other", arch="linux-gfx90a"))
        with self.assertRaisesRegex(ValueError, "Only the holder leaves the stage"):
            moatlib.set_state("proj", self.P, "review-passed")
        self.assertEqual(s.saved, [])

    def test_the_holder_leaves_and_the_lock_is_released(self):
        s = self.store("porting", self.lock("me"))
        moatlib.set_state("proj", self.P, "ported")
        self.assertIsNone(s.obj["porting"])
        self.assertEqual(s.obj["porting_released"]["arch"], self.P)
        self.assertEqual(s.obj["stage"], "ported")

    def test_a_resumer_enters_then_leaves(self):
        s = self.store("porting", self.lock("other"))
        moatlib.set_state("proj", self.P, "porting")
        moatlib.set_state("proj", self.P, "ported")
        self.assertEqual(s.obj["stage"], "ported")
        self.assertIsNone(s.obj["porting"])

    def test_a_holderless_lock_belongs_to_its_platform_on_exit(self):
        s = self.store("planning", self.lock(None))
        moatlib.set_state("proj", self.P, "planned")
        self.assertEqual(s.obj["stage"], "planned")

    def test_the_selector_says_whose_round_it_is_offering(self):
        obj = record(stage="porting", porting=self.lock("other"))
        self.assertIn("another host or job", moatlib.lock_note(obj, self.P, "porter"))
        self.assertIn("2026-01-01T00:00:00Z", moatlib.lock_note(obj, self.P, "porter"))
        self.assertIsNone(moatlib.lock_note(obj, self.P, "validator"))
        mine = record(stage="porting", porting=self.lock("me"))
        self.assertIsNone(moatlib.lock_note(mine, self.P, "porter"))
        self.assertIsNone(moatlib.lock_note(record(stage="planned"), self.P, "porter"))

    def test_next_task_and_fleet_carry_the_note(self):
        obj = record(stage="porting", porting=self.lock("other"))
        with mock.patch.object(moatlib, "project_records",
                               lambda: iter([("proj", obj, "local")])), \
                mock.patch.object(moatlib, "all_projects", lambda: {"proj": "local"}), \
                mock.patch.object(moatlib, "project_record", lambda name: (obj, "local")), \
                mock.patch.object(moatlib, "writable_here", lambda name, where: True), \
                mock.patch.object(moatlib, "_git",
                                  lambda *a, **k: mock.Mock(returncode=0, stdout="")):
            task = moatlib.next_task(self.P)
            rows = moatlib.fleet(self.P)
        self.assertEqual(task["project"], "proj")
        self.assertIn("another host or job", task["lock"])
        self.assertIn("another host or job", rows[0]["lock"])


class ReviewGate(NoDisposition):
    """Ready means reviewed, at this head."""

    FULL = {"linux-gfx90a": done(), "windows-gfx1151": done()}

    def setUp(self):
        super().setUp()
        for attr, value in (("license_gate", lambda name, o=None: (True, "ok")),
                            ("uncommitted_source_files", lambda name: []),
                            ("project_file", lambda name, fn: None),
                            ("lock_holder", lambda: "me")):
            p = mock.patch.object(moatlib, attr, value)
            p.start()
            self.addCleanup(p.stop)

    def blockers(self, obj):
        return moatlib._gate_blockers("proj", obj)[0]

    def classify_as(self, arch_independent):
        return classified("doc-only" if arch_independent else "mixed")

    def test_full_coverage_is_not_ready_before_the_review_passes(self):
        for stage in ("ported", "reviewing", "changes-requested", "porting"):
            with self.subTest(stage=stage):
                found = self.blockers(record(stage=stage, platforms=dict(self.FULL)))
                self.assertEqual([b[0] for b in found], ["review"])
                self.assertIn(stage, found[0][1])

    def test_reviewed_at_head_with_full_coverage_is_ready(self):
        obj = record(reviewed_sha=HEAD, platforms=dict(self.FULL))
        self.assertEqual(self.blockers(obj), [])

    def test_a_record_from_before_the_field_is_not_made_unready(self):
        self.assertEqual(self.blockers(record(platforms=dict(self.FULL))), [])

    def test_a_review_of_an_older_commit_blocks(self):
        obj = record(reviewed_sha=OLD, platforms=dict(self.FULL))
        found = self.blockers(obj)
        self.assertEqual([b[0] for b in found], ["review"])
        self.assertIn(OLD[:12], found[0][1])

    def test_completed_is_refused_until_the_review_has_passed(self):
        for stage in ("ported", "reviewing", "changes-requested"):
            with self.subTest(stage=stage):
                s = Store(self, record(stage=stage))
                with self.assertRaisesRegex(ValueError, "once the review has passed"):
                    moatlib.set_state("proj", "linux-gfx90a", "completed")
                self.assertEqual(s.saved, [])

    def test_a_failure_is_recorded_in_any_stage(self):
        s = Store(self, record(stage="reviewing"))
        moatlib.set_state("proj", "linux-gfx90a", "validation-failed")
        self.assertEqual(s.obj["platforms"]["linux-gfx90a"]["failed_sha"], HEAD)

    def test_passing_the_review_stamps_the_commit_reviewed(self):
        lock = {"arch": "linux-gfx90a", "since": "t", "holder": "me"}
        s = Store(self, record(stage="reviewing", porting=lock))
        moatlib.set_state("proj", "linux-gfx90a", "review-passed")
        self.assertEqual(s.obj["reviewed_sha"], HEAD)

    def test_a_head_move_by_a_real_change_leaves_the_review_behind(self):
        obj = record(head_sha=OLD, reviewed_sha=OLD,
                     platforms={"linux-gfx90a": done(OLD), "windows-gfx1151": done(OLD)})
        with self.classify_as(False):
            moatlib._move_head(obj, HEAD, "repo")
        self.assertEqual(obj["reviewed_sha"], OLD)
        self.assertTrue(moatlib.review_stale(obj))
        self.assertEqual(moatlib.arch_task(obj, "linux-gfx90a"),
                         ("reviewer", "delta-ported"))
        # Even once every arch has validated the new head, it is not ready.
        for blk in obj["platforms"].values():
            blk["validated_sha"] = HEAD
        self.assertEqual([b[0] for b in self.blockers(obj)], ["review"])

    def test_an_inert_head_move_carries_the_review_like_a_validation(self):
        obj = record(head_sha=OLD, reviewed_sha=OLD,
                     platforms={"linux-gfx90a": done(OLD), "windows-gfx1151": done(OLD)})
        with self.classify_as(True):
            moatlib._move_head(obj, HEAD, "repo")
        self.assertEqual(obj["reviewed_sha"], HEAD)
        self.assertFalse(moatlib.review_stale(obj))
        self.assertEqual(self.blockers(obj), [])

    def test_a_legacy_record_heals_on_its_first_head_move(self):
        obj = record(head_sha=OLD, platforms={"linux-gfx90a": done(OLD)})
        with self.classify_as(False):
            moatlib._move_head(obj, HEAD, "repo")
        self.assertEqual(obj["reviewed_sha"], OLD)
        self.assertTrue(moatlib.review_stale(obj))

    def test_a_head_move_before_review_stamps_nothing(self):
        obj = record(stage="porting", head_sha=OLD)
        with self.classify_as(False):
            moatlib._move_head(obj, HEAD, "repo")
        self.assertNotIn("reviewed_sha", obj)

    def test_a_stale_review_is_re_entered_and_re_stamped(self):
        s = Store(self, record(reviewed_sha=OLD, platforms=dict(self.FULL)))
        moatlib.set_state("proj", "linux-gfx90a", "reviewing")
        self.assertEqual(s.obj["stage"], "reviewing")
        moatlib.set_state("proj", "linux-gfx90a", "review-passed")
        self.assertEqual(s.obj["reviewed_sha"], HEAD)
        self.assertEqual(self.blockers(s.obj), [])

    def test_a_tree_identical_squash_keeps_the_review(self):
        s = Store(self, record(head_sha=OLD, reviewed_sha=OLD,
                               platforms={"linux-gfx90a": done(OLD)}))
        fake = mock.Mock(returncode=0, stdout="tree-id\n")
        with mock.patch.object(moatlib.subprocess, "run", lambda *a, **k: fake), \
                mock.patch.object(moatlib, "_fork_repo", lambda name: "repo"):
            moatlib.squash_carry_forward("proj", HEAD)
        self.assertEqual(s.obj["reviewed_sha"], HEAD)

    def test_abandoning_a_round_returns_the_review_to_the_published_tip(self):
        fix = {"branch": "moat-fix-7", "base_sha": OLD, "review_pr": None}
        s = Store(self, record(reviewed_sha=HEAD, fix=fix, pr_state="open",
                               published_sha=OLD, platforms={"linux-gfx90a": done()}),
                  record_writable_here=lambda name: (True, "here"),
                  _fork_repo=lambda name: "repo")
        with self.classify_as(False):
            moatlib.abandon_fix("proj", "someone", "upstream fixed it")
        self.assertEqual(s.obj["head_sha"], OLD)
        self.assertEqual(s.obj["reviewed_sha"], OLD)


class CarryForwardIsClassified(unittest.TestCase):
    """A carry-forward claim is checked against the source classifier."""

    P = "linux-gfx90a"

    def carry(self, cls, method, **kw):
        s = Store(self, record(platforms={self.P: done(OLD)}),
                  _fork_repo=lambda name: "repo")
        with classified(cls):
            moatlib.carry_forward("proj", self.P, HEAD, method, "why", **kw)
        return s.obj["platforms"][self.P]

    def refused(self, cls, method, pattern, **kw):
        s = Store(self, record(platforms={self.P: done(OLD)}),
                  _fork_repo=lambda name: "repo")
        with classified(cls), self.assertRaisesRegex(ValueError, pattern):
            moatlib.carry_forward("proj", self.P, HEAD, method, "why", **kw)
        self.assertEqual(s.saved, [])

    def test_source_class_needs_an_arch_independent_delta(self):
        for cls in ("doc-only", "comment-only"):
            with self.subTest(cls=cls):
                blk = self.carry(cls, "source-class")
                self.assertEqual(blk["validated_sha"], HEAD)
                self.assertEqual(blk["carry_forward"]["class"], cls)
                self.assertEqual(blk["carry_forward"]["from"], OLD)

    def test_source_class_is_not_an_override_of_the_classifier(self):
        for cls in ("mixed", "rename-only", None):
            with self.subTest(cls=cls):
                self.refused(cls, "source-class", "cannot be carried forward as source-class")

    def test_binary_equiv_confirms_an_inert_delta(self):
        for cls in ("comment-only", "rename-only", "doc-only"):
            with self.subTest(cls=cls):
                blk = self.carry(cls, "binary-equiv")
                self.assertEqual(blk["validated_sha"], HEAD)
                self.assertNotIn("host_checked", blk["carry_forward"])

    def test_binary_equiv_over_a_code_change_needs_the_host_side_checked(self):
        for cls in ("mixed", None):
            with self.subTest(cls=cls):
                self.refused(cls, "binary-equiv", "invisible to it")

    def test_the_stated_host_check_is_recorded(self):
        blk = self.carry("mixed", "binary-equiv", host_checked=True)
        self.assertEqual(blk["validated_sha"], HEAD)
        self.assertEqual(blk["carry_forward"]["class"], "mixed")
        self.assertIs(blk["carry_forward"]["host_checked"], True)

    def test_a_host_check_does_not_rescue_a_source_class_claim(self):
        self.refused("mixed", "source-class", "source-class", host_checked=True)

    def test_a_real_classification_of_a_host_only_change(self):
        """End to end with the real classifier: device code untouched, host launch
        dimensions changed -- the delta codeobj_diff would call identical."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            def git(*args):
                return subprocess.run(
                    ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
                     "-c", "commit.gpgsign=false", *args], cwd=tmp, check=True,
                    capture_output=True, text=True).stdout.strip()
            git("init", "-q")
            src = pathlib.Path(tmp) / "launch.cpp"
            src.write_text("void run() { launch(kernel, 64, 64); }\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-q", "-m", "base")
            old = git("rev-parse", "HEAD")
            src.write_text("void run() { launch(kernel, 32, 64); }\n", encoding="utf-8")
            git("commit", "-q", "-am", "host only")
            new = git("rev-parse", "HEAD")
            s = Store(self, record(head_sha=new, platforms={self.P: done(old)}),
                      _fork_repo=lambda name: pathlib.Path(tmp))
            with self.assertRaisesRegex(ValueError, "`mixed`"):
                moatlib.carry_forward("proj", self.P, new, "binary-equiv", "identical")
            self.assertEqual(s.saved, [])


class ContentGateFailure(NoDisposition):
    """A failure that was about the documentation or the jargon check is retired by
    the documentation-only fix that resolves it."""

    P = "windows-gfx1151"

    def moved(self, blk, cls):
        obj = record(head_sha=OLD, reviewed_sha=OLD,
                     platforms={self.P: blk, "linux-gfx90a": done(OLD)})
        with classified(cls):
            moatlib._move_head(obj, HEAD, "repo")
        return obj

    def test_the_gate_is_recorded_with_the_failure(self):
        s = Store(self, record())
        moatlib.set_state("proj", self.P, "validation-failed", gate="docs")
        blk = s.obj["platforms"][self.P]
        self.assertEqual((blk["state"], blk["failed_sha"], blk["failed_gate"]),
                         ("validation-failed", HEAD, "docs"))
        moatlib.validate_status(s.obj)

    def test_a_gate_goes_with_a_failure_only(self):
        s = Store(self, record())
        with self.assertRaisesRegex(ValueError, "goes with validation-failed only"):
            moatlib.set_state("proj", self.P, "completed", gate="docs")
        with self.assertRaisesRegex(ValueError, "goes with validation-failed only"):
            moatlib.set_state("proj", self.P, "validation-failed", gate="tests")
        self.assertEqual(s.saved, [])

    def test_a_doc_only_fix_retires_a_content_gate_failure(self):
        for gate in moatlib.CONTENT_GATES:
            with self.subTest(gate=gate):
                obj = self.moved(failed(OLD, failed_gate=gate), "doc-only")
                self.assertEqual(obj["platforms"][self.P]["failed_sha"], OLD)
                self.assertFalse(moatlib.failure_stands(obj, obj["platforms"][self.P]))
                # A validator is sent, not the porter again.
                self.assertEqual(moatlib.arch_task(obj, self.P),
                                 ("validator", "port-ready"))

    def test_a_plain_failure_is_still_carried_across_a_doc_only_delta(self):
        obj = self.moved(failed(OLD), "doc-only")
        self.assertEqual(obj["platforms"][self.P]["failed_sha"], HEAD)
        self.assertEqual(moatlib.arch_task(obj, self.P), ("porter", "validation-failed"))

    def test_any_failure_is_retired_by_a_code_change(self):
        for blk in (failed(OLD), failed(OLD, failed_gate="jargon")):
            obj = self.moved(blk, "mixed")
            self.assertEqual(obj["platforms"][self.P]["failed_sha"], OLD)

    def test_a_later_plain_failure_or_a_pass_clears_the_gate(self):
        s = Store(self, record(platforms={self.P: failed(HEAD, failed_gate="docs")}))
        moatlib.set_state("proj", self.P, "validation-failed")
        self.assertNotIn("failed_gate", s.obj["platforms"][self.P])
        moatlib.set_state("proj", self.P, "validation-failed", gate="jargon")
        self.assertEqual(s.obj["platforms"][self.P]["failed_gate"], "jargon")
        moatlib.set_state("proj", self.P, "completed")
        self.assertNotIn("failed_gate", s.obj["platforms"][self.P])

    def test_the_cli_takes_the_gate(self):
        s = Store(self, record())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = moatlib.main(["set-state", "proj", self.P, "validation-failed",
                                 "--gate", "jargon"])
        self.assertEqual(code, 0)
        self.assertEqual(s.obj["platforms"][self.P]["failed_gate"], "jargon")


class EvidenceForAnEarlierHead(NoDisposition):
    """A platform's validation of a commit counts again when the head returns."""

    P = "linux-gfx90a"

    def test_revalidating_remembers_the_commit_left_behind(self):
        s = Store(self, record(platforms={self.P: done(OLD)}))
        moatlib.set_state("proj", self.P, "completed")
        blk = s.obj["platforms"][self.P]
        self.assertEqual((blk["validated_sha"], blk["validated_prior"]), (HEAD, [OLD]))
        moatlib.validate_status(s.obj)

    def test_the_history_is_short_and_has_no_duplicates(self):
        blk = done("1" * 40)
        for n in "234567123":
            moatlib._set_validated(blk, n * 40)
        self.assertEqual(blk["validated_sha"], "3" * 40)
        self.assertEqual(len(blk["validated_prior"]), moatlib.VALIDATED_PRIOR_KEPT)
        self.assertEqual(len({p for p in blk["validated_prior"]}), len(blk["validated_prior"]))
        self.assertNotIn("3" * 40, blk["validated_prior"])

    def test_abandoning_a_round_restores_the_validation_of_its_base(self):
        fix = {"branch": "moat-fix-7", "base_sha": OLD, "review_pr": None}
        s = Store(self, record(fix=fix, pr_state="open", published_sha=OLD,
                               platforms={self.P: done(HEAD, validated_prior=[OLD]),
                                          "windows-gfx1151": done(OLD)}),
                  record_writable_here=lambda name: (True, "here"),
                  _fork_repo=lambda name: "repo")
        with classified("mixed"):
            moatlib.abandon_fix("proj", "someone", "upstream fixed it")
        blk = s.obj["platforms"][self.P]
        self.assertEqual(blk["validated_sha"], OLD)
        self.assertEqual(blk["carry_forward"]["method"], "prior-validation")
        self.assertIn(HEAD, blk["validated_prior"])
        self.assertIsNone(moatlib.arch_task(s.obj, self.P))
        self.assertEqual(moatlib.unsatisfied_gates(s.obj), set())

    def test_a_failure_at_the_abandoned_tip_gives_way_to_the_pass_at_the_base(self):
        blk = failed(HEAD, validated_sha=OLD)
        obj = record(platforms={self.P: blk})
        with classified("mixed"):
            moatlib._move_head(obj, OLD, "repo")
        self.assertEqual((blk["state"], blk["validated_sha"], blk["failed_sha"]),
                         ("completed", OLD, HEAD))

    def test_a_failure_at_the_commit_returned_to_still_stands(self):
        blk = failed(OLD, validated_sha=OLD)
        obj = record(platforms={self.P: blk})
        with classified("mixed"):
            moatlib._move_head(obj, OLD, "repo")
        self.assertEqual(blk["state"], "validation-failed")
        self.assertTrue(moatlib.failure_stands(obj, blk))

    def test_a_commit_never_validated_is_still_owed(self):
        blk = done(OLD)
        obj = record(head_sha=OLD, platforms={self.P: blk})
        with classified("mixed"):
            moatlib._move_head(obj, THIRD, "repo")
        self.assertEqual(blk["validated_sha"], OLD)
        self.assertEqual(moatlib.arch_task(obj, self.P), ("reviewer", "delta-ported"))


class HeldFoldersStayPut(NoDisposition):
    def misplaced(self, obj, has_branch):
        ref = mock.Mock(stdout="abc\n" if has_branch else "")
        with mock.patch.object(moatlib, "project_records",
                               lambda: iter([("proj", obj, "branch")])), \
                mock.patch.object(moatlib, "_git", lambda *a, **k: ref):
            return moatlib.misplaced_folders()

    def test_a_held_project_is_asked_to_move_in_neither_direction(self):
        held = record(on_hold=True, platforms={"linux-gfx90a": done(OLD)})
        self.assertEqual(self.misplaced(held, has_branch=True), [])
        self.assertEqual(self.misplaced(held, has_branch=False), [])

    def test_releasing_the_hold_brings_it_back_into_the_report(self):
        obj = record(platforms={"linux-gfx90a": done(OLD)})
        rows = self.misplaced(obj, has_branch=False)
        self.assertEqual([(r[0], r[1]) for r in rows], [("proj", "trunk")])


if __name__ == "__main__":
    unittest.main()
