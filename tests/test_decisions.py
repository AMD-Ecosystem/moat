# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decision records: who they say decided, what a run made to look may write, and
where a tool leaves the checkout."""
import contextlib
import io
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import identity  # utils/ is put on sys.path by tests/__init__.py
import intake_queue
import moatlib
import optout
import port_request
import triage
import upstream

ACCOUNT = {"login": "shared-account", "gh_host": "github.com"}


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def quiet(fn, *args, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(*args, **kw)
    return rc, out.getvalue(), err.getvalue()


class Identity(unittest.TestCase):
    def setUp(self):
        identity._CACHE.clear()
        self.addCleanup(identity._CACHE.clear)

    def test_recorded_as_is_the_authenticated_account_and_github_host(self):
        calls = []

        def run(argv, **kw):
            calls.append(argv)
            return Result(0, "shared-account\n")
        with mock.patch.dict("os.environ", {"GH_HOST": ""}):
            self.assertEqual(identity.recorded_as(run), ACCOUNT)
        self.assertEqual(identity.recorded_as(run), ACCOUNT)       # asked once
        self.assertEqual(calls, [["gh", "api", "user", "--jq", ".login"]])

    def test_unavailable_gh_records_nothing_rather_than_a_guess(self):
        self.assertIsNone(identity.recorded_as(lambda *a, **k: Result(1, "", "no auth")))
        identity._CACHE.clear()

        def boom(*a, **k):
            raise OSError("gh not found")
        self.assertIsNone(identity.recorded_as(boom))
        self.assertEqual(identity.stamp({"by": "x"}), {"by": "x"})

    def test_removal_log_keeps_what_was_removed_and_names_no_machine(self):
        identity._CACHE.append(ACCOUNT)
        with tempfile.TemporaryDirectory() as d:
            identity.log_removal(d, "unskip", "o/r", "someone", {"reason": "declined"})
            identity.log_removal(d, "unskip", "o/r2", "someone", None)
            lines = [json.loads(l) for l in
                     identity.log_path(d).read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["was"], {"reason": "declined"})
        self.assertEqual(sorted(lines[0]),
                         ["at", "by", "kind", "recorded_as", "subject", "was"])
        self.assertEqual(lines[0]["recorded_as"], ACCOUNT)


class Triage(unittest.TestCase):
    def setUp(self):
        identity._CACHE.clear()
        identity._CACHE.append(ACCOUNT)
        self.addCleanup(identity._CACHE.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.disp = {"old/name": {"full_name": "old/name", "disposition": "skip",
                                  "reason": "declined", "note": "", "by": "p",
                                  "repo_id": 5}}

        def set_disposition(full, disposition, reason, note="", repo_id=None, by=None):
            self.disp[full.lower()] = {"full_name": full, "disposition": disposition,
                                       "reason": reason, "note": note, "by": by}
            return self.disp[full.lower()]

        def clear(full):
            return self.disp.pop(full.lower(), None) is not None
        for name, fake in (
            ("REPO_ROOT", self.root), ("load_dispositions", lambda: self.disp),
            ("save_dispositions", lambda d: None), ("set_disposition", set_disposition),
            ("clear_disposition", clear),
            ("get_disposition", lambda f, i=None: self.disp.get(f.lower())),
            ("all_projects", lambda: {}), ("adopted_repo_ids", lambda: {}),
            ("optout_for", lambda f: None), ("github_repo_id", lambda f: 7),
        ):
            p = mock.patch.object(moatlib, name, fake)
            p.start()
            self.addCleanup(p.stop)

    def test_review_writes_nothing_unless_asked(self):
        cands = self.root / "candidates.json"
        text = json.dumps([{"full_name": "new/repo", "priority": 1, "stars": 9}])
        cands.write_text(text)
        with mock.patch.object(triage, "CANDIDATES", cands):
            rc, out, _err = quiet(triage.main, ["review"])
            self.assertEqual(rc, 0)
            self.assertEqual(cands.read_text(), text)
            self.assertIn("--cache-ids", out)
            self.assertIn("new/repo", out)
            quiet(triage.main, ["review", "--cache-ids"])
            self.assertEqual(json.loads(cands.read_text())[0]["repo_id"], 7)

    def test_skip_records_the_account_beside_the_named_decider(self):
        quiet(triage.main, ["skip", "o/r", "--reason", "declined", "--by", "someone"])
        self.assertEqual(self.disp["o/r"]["by"], "someone")
        self.assertEqual(self.disp["o/r"]["recorded_as"], ACCOUNT)

    def test_unskip_needs_a_decider_and_leaves_a_tombstone(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            triage.main(["unskip", "old/name"])
        self.assertIn("old/name", self.disp)
        rc, out, _err = quiet(triage.main, ["unskip", "old/name", "--by", "someone"])
        self.assertEqual(rc, 0)
        self.assertNotIn("old/name", self.disp)
        (line,) = [json.loads(l) for l in
                   identity.log_path(self.root).read_text().splitlines()]
        self.assertEqual((line["kind"], line["subject"], line["by"]),
                         ("unskip", "old/name", "someone"))
        self.assertEqual(line["was"]["reason"], "declined")
        self.assertEqual(line["recorded_as"], ACCOUNT)

    def test_unskip_of_nothing_logs_nothing(self):
        quiet(triage.main, ["unskip", "no/such", "--by", "someone"])
        self.assertFalse(identity.log_path(self.root).exists())


def git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode:
        raise AssertionError(f"git {' '.join(args)}: {r.stderr}")
    return r.stdout.strip()


class Scratch(unittest.TestCase):
    """A bare origin with a trunk, and clones of it."""

    def setUp(self):
        identity._CACHE.clear()
        identity._CACHE.append(ACCOUNT)
        self.addCleanup(identity._CACHE.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.origin = self.root / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)],
                       check=True)
        self.a = self.clone("a", seed=True)

    def clone(self, name, seed=False):
        repo = self.root / name
        if seed:
            subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
            git(repo, "remote", "add", "origin", str(self.origin))
        else:
            subprocess.run(["git", "clone", "-q", str(self.origin), str(repo)],
                           check=True)
        git(repo, "config", "user.email", "t@example.com")
        git(repo, "config", "user.name", "t")
        git(repo, "config", "core.autocrlf", "false")
        if seed:
            for rel, text in (("data/optout.json", "{}\n"),
                              ("data/dispositions.json", "{}\n"), ("README.md", "r\n")):
                f = repo / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(text)
            git(repo, "add", ".")
            git(repo, "commit", "-q", "-m", "base")
            git(repo, "push", "-q", "-u", "origin", "main")
        return repo


class IntakeQueue(Scratch):
    def apply(self, declines, fail=False, by="someone"):
        disp = self.a / "data" / "dispositions.json"

        def set_disposition(full, disposition, reason, note="", repo_id=None, by=None):
            if fail:
                raise RuntimeError("simulated failure")
            d = json.loads(disp.read_text())
            d[full.lower()] = {"full_name": full, "disposition": disposition,
                               "reason": reason, "note": note, "by": by}
            disp.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
            return d[full.lower()]

        def save(d):
            disp.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
        real_run = subprocess.run

        def run(argv, *a, **k):
            # git runs for real in the scratch clone; the README generator does not.
            return real_run(argv, *a, **k) if argv[0] == "git" else Result(0)
        with mock.patch.object(moatlib, "REPO_ROOT", self.a), \
                mock.patch.object(moatlib, "set_disposition", set_disposition), \
                mock.patch.object(moatlib, "load_dispositions",
                                  lambda: json.loads(disp.read_text())), \
                mock.patch.object(moatlib, "save_dispositions", save), \
                mock.patch.object(intake_queue, "find_issue", lambda: None), \
                mock.patch.object(intake_queue, "gh",
                                  lambda a, check=False: Result(0, "https://pr/1\n")), \
                mock.patch.object(intake_queue.subprocess, "run", run):
            return intake_queue.apply_decisions(declines, ["why"], by, apply=True)

    def test_the_checkout_is_returned_to_the_branch_it_started_on(self):
        git(self.a, "checkout", "-q", "-b", "port/x")
        action, _detail = self.apply(["o/r:declined"])
        self.assertEqual(action, "pr")
        self.assertEqual(git(self.a, "rev-parse", "--abbrev-ref", "HEAD"), "port/x")
        self.assertEqual(git(self.a, "status", "--porcelain"), "")
        pushed = git(self.origin, "show", f"{intake_queue.BRANCH}:data/dispositions.json")
        entry = json.loads(pushed)["o/r"]
        self.assertEqual((entry["by"], entry["recorded_as"]), ("someone", ACCOUNT))

    def test_a_failure_part_way_still_returns_it(self):
        git(self.a, "checkout", "-q", "-b", "port/x")
        with self.assertRaises(RuntimeError):
            self.apply(["o/r:declined"], fail=True)
        self.assertEqual(git(self.a, "rev-parse", "--abbrev-ref", "HEAD"), "port/x")
        self.assertEqual(git(self.a, "status", "--porcelain"), "")

    def test_a_dirty_checkout_is_refused_before_anything_moves(self):
        git(self.a, "checkout", "-q", "-b", "port/x")
        (self.a / "README.md").write_text("someone else's edit\n")
        with self.assertRaises(RuntimeError) as cm:
            self.apply(["o/r:declined"])
        self.assertIn("README.md", str(cm.exception))
        self.assertEqual(git(self.a, "rev-parse", "--abbrev-ref", "HEAD"), "port/x")
        self.assertEqual((self.a / "README.md").read_text(), "someone else's edit\n")

    def test_apply_needs_a_named_decider(self):
        asked = []
        with mock.patch.object(intake_queue, "gh",
                               lambda a, check=False: asked.append(a) or Result(0, "me")):
            rc, _out, err = quiet(intake_queue.main, ["apply", "--accept", "o/r"])
        self.assertEqual((rc, asked), (2, []))
        self.assertIn("--by", err)

    def test_an_accept_records_the_account_beside_the_decider(self):
        obj = {"intake": {"verdict": "fork", "summary": "s"}}
        saved = []
        with mock.patch.object(moatlib, "project_record", lambda n: (obj, "branch")), \
                mock.patch.object(moatlib, "save_record",
                                  lambda n, o, m: saved.append(o) or "a" * 40):
            intake_queue.record_accepts(["o/r"], "someone", apply=True)
        decided = saved[0]["intake"]["decided"]
        self.assertEqual((decided["by"], decided["recorded_as"]), ("someone", ACCOUNT))


class OptOutKeepsAMaintainedFork(unittest.TestCase):
    """A request to stop ends pull requests, not a branch we keep on our own fork."""
    OPTED = ["opted-out: Owner asked not to receive pull requests"]

    def test_the_merge_gate_blocks_an_ordinary_round(self):
        d = {"pr_state": "open"}
        self.assertEqual(upstream.optout_merge_blockers(d, {"body": ""}, self.OPTED),
                         self.OPTED)
        d = {"pr_state": "open", "maintained": {"by": "p"}}
        self.assertEqual(upstream.optout_merge_blockers(d, {"body": ""}, self.OPTED),
                         self.OPTED)

    def test_the_merge_gate_lets_a_maintained_fork_merge_and_sends_no_reply(self):
        d = {"pr_state": "closed", "maintained": {"by": "p"}}
        self.assertEqual(upstream.optout_merge_blockers(d, {"body": "x"}, self.OPTED), [])
        body = f"x\n\n{upstream.REPLY_HEADING}\n\nhello\n"
        (why,) = upstream.optout_merge_blockers(d, {"body": body}, self.OPTED)
        self.assertIn("sent no reply", why)
        self.assertEqual(upstream.optout_merge_blockers(d, {"body": body}, []), [])

    def test_recording_retires_a_project_but_not_a_maintained_fork(self):
        rows = [{"name": "kept", "full_name": "Owner/kept", "stage": "review-passed",
                 "maintained": True, "fork": None, "pr": None, "pr_state": "closed",
                 "review_pr": None},
                {"name": "gone", "full_name": "Owner/gone", "stage": "porting",
                 "maintained": False, "fork": None, "pr": None, "pr_state": None,
                 "review_pr": None}]
        retired = []
        rec = {"who": "Owner", "scope": "owner", "source": "https://x/1", "note": ""}
        patches = [
            mock.patch.object(moatlib, "record_optout", lambda *a: rec),
            mock.patch.object(moatlib, "set_disposition",
                              lambda full, *a, **k: retired.append(full)),
            mock.patch.object(optout, "stamp_optout", lambda who: None),
            mock.patch.object(optout, "publish_pending", lambda **k: (True, "ok")),
            mock.patch.object(optout, "affected", lambda who: rows),
            mock.patch.object(identity, "recorded_as", lambda: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        _rc, out, _err = quiet(optout.cmd_record,
                               mock.Mock(who="Owner", source="https://x/1", note=""))
        self.assertEqual(retired, ["Owner/gone"])
        self.assertIn("kept (Owner/kept) is a maintained fork", out)


class OptOutPending(Scratch):
    REC = {"who": "Owner", "scope": "owner", "source": "https://x/issue/1", "note": ""}

    def test_a_record_binds_a_host_that_has_not_merged_it(self):
        b = self.clone("b")
        self.assertEqual(optout.pending_for("Owner/Repo", repo=b), (None, None))
        head = git(self.a, "rev-parse", "HEAD")
        ok, detail = optout.publish_pending(add=self.REC, repo=self.a)
        self.assertTrue(ok, detail)
        # Nothing about the recording checkout moved: no branch switch, no index or
        # working-tree change.
        self.assertEqual(git(self.a, "rev-parse", "HEAD"), head)
        self.assertEqual(git(self.a, "status", "--porcelain"), "")
        # The trunk does not have it; the other host is bound anyway.
        self.assertEqual(json.loads(git(self.origin, "show", "main:data/optout.json")), {})
        entry, problem = optout.pending_for("owner/anything", repo=b)
        self.assertEqual((entry["who"], problem), ("Owner", None))
        self.assertEqual(optout.pending_for("other/repo", repo=b), (None, None))
        with mock.patch.object(upstream, "REPO", b), \
                mock.patch.object(moatlib, "upstream_full_name", lambda n: "Owner/Repo"):
            (why,) = upstream.pending_optout_problems("proj")
        self.assertIn("opted-out: Owner", why)
        self.assertIn("not yet on the trunk", why)

    def test_records_accumulate_and_a_removal_unbinds(self):
        b = self.clone("b")
        optout.publish_pending(add=self.REC, repo=self.a)
        one = {**self.REC, "who": "x/y", "scope": "repo"}
        self.assertTrue(optout.publish_pending(add=one, repo=b)[0])
        self.assertEqual(sorted(optout.read_pending(self.a)[0]), ["owner", "x/y"])
        self.assertEqual(optout.publish_pending(add=one, repo=b),
                         (True, f"the {optout.PENDING_BRANCH} branch already says so"))
        self.assertTrue(optout.publish_pending(remove="Owner", repo=self.a)[0])
        self.assertEqual(optout.pending_for("owner/repo", repo=b), (None, None))
        self.assertIsNotNone(optout.pending_for("x/y", repo=b)[0])

    def test_an_unreadable_branch_is_a_problem_not_an_empty_answer(self):
        b = self.clone("b")
        git(b, "remote", "set-url", "origin", str(self.root / "gone.git"))
        entry, problem = optout.pending_for("owner/repo", repo=b)
        self.assertIsNone(entry)
        self.assertIn("could not fetch", problem)
        with mock.patch.object(upstream, "REPO", b), \
                mock.patch.object(moatlib, "upstream_full_name", lambda n: "Owner/Repo"):
            (why,) = upstream.pending_optout_problems("proj")
        self.assertIn("could not fetch", why)
        ok, detail = optout.publish_pending(add=self.REC, repo=b)
        self.assertFalse(ok)

    def test_no_pending_branch_blocks_nothing(self):
        with mock.patch.object(upstream, "REPO", self.a), \
                mock.patch.object(moatlib, "upstream_full_name", lambda n: "Owner/Repo"):
            self.assertEqual(upstream.pending_optout_problems("proj"), [])


class PortRequest(unittest.TestCase):
    def known(self, asked, canon=None, repo_id=None, ids=None):
        with mock.patch.object(port_request, "canonical",
                               lambda f: (canon or f, repo_id)), \
                mock.patch.object(port_request, "open_requests", lambda: []), \
                mock.patch.object(moatlib, "all_projects",
                                  lambda: {"bar": "branch", "renamed": "trunk"}), \
                mock.patch.object(moatlib, "upstream_full_name",
                                  lambda n: {"bar": "baz/bar",
                                             "renamed": "Some/Thing"}[n]), \
                mock.patch.object(moatlib, "adopted_repo_ids", lambda: ids or {}), \
                mock.patch.object(moatlib, "get_disposition", lambda n, i=None: None):
            return port_request.known(asked)

    def test_a_different_owner_with_the_same_repo_name_is_not_adopted(self):
        self.assertIsNone(self.known("foo/bar"))

    def test_the_adopted_repo_is_found_by_full_name_whatever_its_folder(self):
        self.assertIn("projects/bar (in flight on port/bar)", self.known("baz/bar"))
        self.assertIn("projects/renamed (on the trunk)", self.known("some/thing"))

    def test_a_renamed_repo_is_found_by_id(self):
        self.assertIn("projects/bar", self.known("new/name", repo_id=42, ids={42: "bar"}))
        self.assertIsNone(self.known("new/name", repo_id=43, ids={42: "bar"}))


if __name__ == "__main__":
    unittest.main()
