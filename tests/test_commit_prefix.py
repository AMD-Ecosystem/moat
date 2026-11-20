"""The [ROCm] title prefix is required only on a ROCm-specific commit."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import moatlib  # utils/ is put on sys.path by tests/__init__.py

BODY = ("\n\nWhy.\n\nAuthored with an AI coding agent.\n\n"
        "Test Plan:\n```\nmake test\n```\n")


class PrefixOnlyOnRocmSpecificCommits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q", "-b", "main")
        self.change("base.c", "int base;\n", "upstream base\n")
        self.git("checkout", "-q", "-b", "moat-port")
        patcher = mock.patch.object(moatlib, "_fork_repo", lambda name: self.repo)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def git(self, *args):
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
        r = subprocess.run(["git", "-c", "core.autocrlf=false", *args],
                           cwd=str(self.repo), capture_output=True, text=True,
                           env=env, check=True)
        return r.stdout.strip()

    def change(self, fname, text, message):
        (self.repo / fname).write_text(text)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def problems(self):
        return moatlib.commit_message_problems(
            "proj", {"name": "proj", "fork_default_branch": "main"})

    def test_a_shared_code_fix_needs_no_prefix(self):
        self.change("a.h", "int a;\n", "[ROCm] port" + BODY)
        self.change("sort.h", "#include <cstddef>\nunsigned int n; // ship it\n",
                    "Use unsigned int" + BODY)
        self.assertEqual(self.problems(), [])

    def test_a_rocm_specific_change_without_the_prefix_is_an_offence(self):
        self.change("a.h", "int a;\n", "[ROCm] port" + BODY)
        sha = self.change("k.h", "#ifdef __HIPCC__\n#include <hip/hip_runtime.h>\n"
                                 "#endif\n", "Guard header" + BODY)
        found = self.problems()
        self.assertEqual(len(found), 1)
        self.assertIn(f"proj {sha[:7]}: title does not start with [ROCm] but the "
                      f"change is ROCm-specific (2 added line(s)", found[0])

    def test_removed_rocm_lines_do_not_count(self):
        self.change("k.h", "int x; // HIP_PLATFORM\n", "[ROCm] add" + BODY)
        self.change("k.h", "int x;\n", "Drop a comment" + BODY)
        self.assertEqual(self.problems(), [])

    def test_the_other_rules_still_apply_without_the_prefix(self):
        self.change("a.h", "int a;\n", "[ROCm] port" + BODY)
        sha = self.change("b.h", "int b;\n", "Fix b\n\nWhy.\n")
        self.assertEqual(self.problems(),
                         [f"proj {sha[:7]}: body does not disclose AI-agent assistance",
                          f"proj {sha[:7]}: body has no Test Plan"])

    def test_unprefixed_commits_of_ours_do_not_read_as_a_stale_mirror(self):
        for n in "abc":
            self.change(n + ".h", "int " + n + ";\n", "Fix " + n + BODY)
        self.assertEqual(self.problems(), [])

    def test_upstream_commits_still_read_as_a_stale_mirror(self):
        for n in "abc":
            self.change(n + ".h", "int " + n + ";\n", "upstream change " + n + "\n")
        self.assertIn("are not ours", self.problems()[0])

    def test_plain_words_are_not_rocm_specific(self):
        for text in ("relationship", "ship", "chip", "amdahl", "process", "CUDA_HOME"):
            self.assertIsNone(moatlib._ROCM_SPECIFIC.search(text), text)
        for text in ("hipcc", "<hip/hip_runtime.h>", "hipMalloc", "USE_ROCM", "rocPRIM",
                     "__HIP_PLATFORM_AMD__", "gfx942", "ROCm", "find_package(hip)"):
            self.assertIsNotNone(moatlib._ROCM_SPECIFIC.search(text), text)


if __name__ == "__main__":
    unittest.main()
