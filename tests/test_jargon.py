# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""jargon.py: what the vocabulary scan reads, and what it must not flag."""
import base64
import pathlib
import subprocess
import tempfile
import unittest

import jargon  # utils/ is put on sys.path by tests/__init__.py
import upstream

TERMS, ALLOW = jargon.load()
FENCE = "```"


def found(text):
    return sorted({h[2].lower() for h in jargon.scan_text(text, "t", TERMS, ALLOW)})


class Urls(unittest.TestCase):
    def test_a_url_is_masked_and_the_rest_of_its_line_is_read(self):
        line = ("The lead platform validated at head_sha, see "
                "https://github.com/AMD-Ecosystem/moat for how.")
        self.assertEqual(found(line), ["head_sha", "lead platform"])
        self.assertEqual(found("See github.com/AMD-Ecosystem/moat/issues/1 for it."), [])
        self.assertEqual(found("Branch: https://github.com/x/y/tree/moat-port"), [])

    def test_the_note_every_pull_request_carries_is_scanned_and_clean(self):
        self.assertIn("github.com/AMD-Ecosystem/moat", upstream.SUBMISSION_NOTE)
        self.assertEqual(found(upstream.SUBMISSION_NOTE), [])
        dirty = upstream.SUBMISSION_NOTE.replace(
            "one repository at a time", "one repository at a time on the lead platform")
        self.assertEqual(found(dirty), ["lead platform"])


class Fences(unittest.TestCase):
    def test_an_unclosed_fence_does_not_hide_what_follows(self):
        text = f"Summary.\n\n{FENCE}\nmake test\n\nThe lead platform passed.\n"
        self.assertEqual(found(text), ["lead platform"])

    def test_a_closed_fence_still_shields_prose_terms(self):
        text = f"Summary.\n\n{FENCE}\necho revalidate the lead platform\n{FENCE}\n"
        self.assertEqual(found(text), [])

    def test_one_units_stray_fence_does_not_reach_the_next_unit(self):
        units = [("c1", f"Fix.\n\n{FENCE}\nmake\n"),
                 ("c2", f"Tests.\n\n{FENCE}\nctest\n{FENCE}\n\nThe followers passed.\n"),
                 ("c3", "Port-ready on gfx90a.\n")]
        hits = jargon.scan_units(units, "commit", TERMS, ALLOW)
        self.assertEqual(sorted((h[0], h[2].lower()) for h in hits),
                         [("commit c2", "the followers"), ("commit c3", "port-ready")])
        # Joined, the way callers used to pass them, the stray fence in c1 flips
        # every block after it; scanning them apart is what keeps that from mattering.
        joined = "\n".join(t for _, t in units)
        self.assertNotEqual(len(jargon.scan_text(joined, "x", TERMS, ALLOW)), 0)


class FencedVocabulary(unittest.TestCase):
    def block(self, *lines):
        return f"Test Plan:\n\n{FENCE}\n" + "\n".join(lines) + f"\n{FENCE}\n"

    def test_our_tools_records_and_paths_are_wrong_in_a_command_too(self):
        for line, want in (
            ("cd /var/lib/jenkins/work && ctest", "/var/lib/jenkins"),
            ("cd /home/alice/src/build && ctest", "/home/alice/"),
            ("cmake -S C:\\Users\\bob\\src -B build", "c:\\users\\bob"),
            ("python3 ../../utils/prose.py README.md", "utils/prose.py"),
            ("python3 utils/moatlib.py show proj", "moatlib"),
            ("cd projects/proj/src && make", "projects/proj/src"),
            ("ls agent_space/out", "agent_space"),
            ("git push origin moat-fix-12", "moat"),
            ("MOAT_PLATFORM=linux-gfx90a ctest", None),
        ):
            with self.subTest(line=line):
                got = found(self.block(line))
                if want is None:
                    self.assertTrue(got)
                else:
                    self.assertIn(want, got)

    def test_ordinary_commands_and_prose_terms_in_a_block_are_not_flagged(self):
        self.assertEqual(found(self.block(
            "cmake -S . -B build -DUSE_HIP=ON",
            "git fetch origin moat-port && git checkout moat-port",
            "ctest --test-dir build   # revalidate after the follower fix",
            "./build/tests --gtest_filter='Strategy.*'",
            "ls ~/src /opt/rocm /home",
        )), [])

    def test_the_vocabulary_check_itself_is_not_jargon(self):
        self.assertEqual(found(self.block(
            "grep -rniE '\\bmoat\\b|head_sha' . || echo clean",
            "$ grep -c moatlib README.md",
            "python3 utils/jargon.py --port proj")), [])
        self.assertEqual(found("    grep -niE '\\bmoat\\b' -r src/\n"), [])

    def test_an_indented_command_line_is_judged_like_a_block(self):
        self.assertEqual(found("    git checkout moat-port\n"), [])
        self.assertIn("moatlib", found("    python3 utils/moatlib.py show x\n"))


class Terms(unittest.TestCase):
    def test_terms_the_skill_gate_had_and_this_did_not(self):
        for text, want in (
            ("See plan.md for the scope.", "plan.md"),
            ("Recorded in status.json.", "status.json"),
            ("Matches published_sha.", "published_sha"),
            ("Tested across the fleet.", "the fleet"),
            ("Ran on a fleet host.", "fleet host"),
            ("State is cuda-not-validated.", "cuda-not-validated"),
            ("Scratch output is in agent_space.", "agent_space"),
            ("The porter agent added this.", "porter agent"),
            ("Found by the validator role.", "validator role"),
            ("Written by moatlib.", "moatlib"),
            ("Needs advance-head first.", "advance-head"),
            ("Marked cant-port.", "cant-port"),
            ("See config/arches.toml.", "config/arches.toml"),
            ("Set MOAT_PLATFORM first.", "moat_platform"),
            ("Cleared by OSRB.", "osrb"),
        ):
            with self.subTest(text=text):
                self.assertIn(want, found(text))

    def test_ordinary_wording_is_not_flagged(self):
        for text in (
            "A strategy based on a compatibility header.",
            "The strategy and its trade-offs are described below.",
            "Strategy Based Dispatch is the upstream class name.",
            "The MPPI path follower now runs on AMD GPUs.",
            "Followers of this issue asked for HIP support.",
            "The input validator rejects negative sizes.",
            "Thanks to the reviewer for the suggestion.",
            "The query planner picks the GPU path.",
            "A fleet of drones is simulated per batch.",
            "Uses the project's own utils/ helpers and _deps/ directory.",
            "Each target depends_on the generated header.",
            "The remoat option is unrelated.",
        ):
            with self.subTest(text=text):
                self.assertEqual(found(text), [])

    def test_the_in_house_spellings_still_are(self):
        for text, want in (
            ("Uses Strategy B (torch hipify).", "strategy b"),
            ("Strategy A was rejected.", "strategy a"),
            ("The followers validate next.", "the followers"),
            ("Both followers passed.", "both followers"),
            ("The follower platforms re-ran it.", "follower platforms"),
            ("Built on the lead platform.", "lead platform"),
        ):
            with self.subTest(text=text):
                self.assertIn(want, found(text))

    def test_an_embedded_blob_is_not_prose(self):
        # 64 KiB of base64 spells most four-letter words somewhere.
        blob = base64.b64encode(bytes(range(256)) * 256).decode()
        seeded = blob[:400] + "MOaT" + blob[400:800] + "OsrB" + blob[800:]
        self.assertEqual(found(f'<image href="data:image/png;base64,{seeded}"/>'), [])

    def test_every_pattern_compiles_and_has_a_fix(self):
        import re
        for table in (TERMS, TERMS.fenced):
            for pat, fix in table.items():
                re.compile(pat)
                self.assertTrue(fix.strip(), pat)
        self.assertTrue(TERMS.fenced)


class Git(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")
        self.git("config", "core.autocrlf", "false")
        self.commit("base", {"README.md": "base\n"})
        self.git("checkout", "-q", "-b", "port")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        subprocess.run(["git", "-C", self.repo, *args], check=True, capture_output=True)

    def commit(self, message, files):
        for rel, text in files.items():
            p = pathlib.Path(self.repo) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        self.git("add", ".")
        self.git("commit", "-q", "-m", message)

    def test_each_commit_and_each_file_is_scanned_by_itself(self):
        self.commit(f"Add a.\n\n{FENCE}\nmake a\n", {"a.md": f"Build:\n\n{FENCE}\nmake\n"})
        self.commit("Add b.\n\nThe lead platform passed.\n",
                    {"b.md": "Port-ready on gfx90a.\n"})
        commits = jargon.scan_commits(self.repo, "main..port", TERMS, ALLOW)
        self.assertEqual([h[2] for h in commits], ["lead platform"])
        diff = jargon.scan_diff(self.repo, "main...port", TERMS, ALLOW)
        self.assertEqual([(h[0], h[2].lower()) for h in diff],
                         [("added lines b.md", "port-ready")])

    def test_added_by_file(self):
        diff = ("diff --git a/x.c b/x.c\n--- a/x.c\n+++ b/x.c\n@@ -1 +1,2 @@\n one\n"
                "+two\n-gone\ndiff --git a/y.md b/y.md\n--- /dev/null\n+++ b/y.md\n"
                "@@ -0,0 +1 @@\n+why\n")
        self.assertEqual(jargon.added_by_file(diff), [("x.c", "two"), ("y.md", "why")])
        self.assertEqual(jargon.added_by_file(""), [])


if __name__ == "__main__":
    unittest.main()
