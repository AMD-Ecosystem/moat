# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import os
import tempfile
import unittest
from unittest import mock


import gh_guard  # utils/ is put on sys.path by tests/__init__.py


class Classify(unittest.TestCase):
    def test_cases_table(self):
        for argv, want in gh_guard.CASES:
            with self.subTest(argv=" ".join(argv)):
                self.assertEqual(gh_guard.classify(argv).allow, want)

    def test_moat_command_in_body_file_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "body.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("Reviewed.\n\n/moat approve\n")
            for argv in (
                ["pr", "comment", "3", "--repo", "AMD-Ecosystem/fork",
                 "--body-file", path],
                ["pr", "comment", "3", "--repo", "AMD-Ecosystem/fork",
                 f"--body-file={path}"],
                ["api", "repos/AMD-Ecosystem/fork/issues/3/comments",
                 "-F", f"body=@{path}"],
                ["api", "repos/AMD-Ecosystem/fork/issues/3/comments",
                 f"-Fbody=@{path}"],
            ):
                with self.subTest(argv=argv[-1]):
                    self.assertFalse(gh_guard.classify(argv).allow)

    def test_ordinary_body_file_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "body.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("Addressed in abc1234.\n")
            argv = ["pr", "comment", "3", "--repo", "AMD-Ecosystem/fork",
                    "--body-file", path]
            self.assertTrue(gh_guard.classify(argv).allow)

    def test_gh_repo_env_names_the_target(self):
        argv = ["pr", "comment", "9", "--body", "hi"]
        with mock.patch.dict(os.environ, {"GH_REPO": "torvalds/linux"}):
            self.assertFalse(gh_guard.classify(argv).allow)
        with mock.patch.dict(os.environ, {"GH_REPO": "AMD-Ecosystem/moat"}):
            self.assertTrue(gh_guard.classify(argv).allow)

    def test_api_method_spellings(self):
        for rest, want in (
            (["-X", "POST", "x"], "POST"), (["-XPOST", "x"], "POST"),
            (["-X=patch", "x"], "PATCH"), (["--method", "delete", "x"], "DELETE"),
            (["--method=PUT", "x"], "PUT"), (["x"], None), (["x", "-X"], ""),
        ):
            with self.subTest(rest=rest):
                self.assertEqual(gh_guard.api_method(rest), want)


if __name__ == "__main__":
    unittest.main()
