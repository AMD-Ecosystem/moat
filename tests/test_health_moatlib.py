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

import moatlib  # utils/ is put on sys.path by tests/__init__.py

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


if __name__ == "__main__":
    unittest.main()
