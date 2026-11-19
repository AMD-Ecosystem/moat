# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""prose.py: its command line, and what counts as a hand-wrapped paragraph."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import prose  # utils/ is put on sys.path by tests/__init__.py

LONG = "A first line of a paragraph that runs well past sixty characters in length."


class CommandLine(unittest.TestCase):
    def run_main(self, argv, stdin=None, tty=False):
        fake = io.StringIO(stdin or "")
        fake.isatty = lambda: tty
        read = []
        real_read = fake.read
        fake.read = lambda *a: read.append(1) or real_read(*a)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "stdin", fake), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = prose.main(argv)
        return rc, out.getvalue(), err.getvalue(), bool(read)

    def test_help_prints_usage_and_reads_nothing(self):
        for flag in ("--help", "-h"):
            rc, out, _err, read = self.run_main([flag])
            self.assertEqual((rc, read), (0, False))
            self.assertIn("usage: prose.py", out)

    def test_no_arguments_on_a_terminal_prints_usage_instead_of_waiting(self):
        rc, _out, err, read = self.run_main([], tty=True)
        self.assertEqual((rc, read), (2, False))
        self.assertIn("usage: prose.py", err)

    def test_a_pipe_is_still_read_with_or_without_the_dash(self):
        for argv in ([], ["-"]):
            rc, out, _err, read = self.run_main(argv, stdin="One line.\n")
            self.assertEqual((rc, read), (0, True))
            self.assertIn("prose: clean", out)
            rc, out, _err, _read = self.run_main(argv, stdin=f"{LONG}\nwrapped here.\n")
            self.assertEqual(rc, 1)

    def test_a_missing_file_is_a_message_not_a_traceback(self):
        missing = os.path.join(tempfile.gettempdir(), "moat-no-such-body.md")
        rc, _out, err, _read = self.run_main([missing])
        self.assertEqual(rc, 2)
        self.assertIn("cannot read", err)

    def test_files_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "body.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{LONG}\nwrapped here.\n")
            rc, out, _err, _read = self.run_main([path])
            self.assertEqual(rc, 1)
            self.assertIn("hard-wrapped (line 2)", out)


class HardWrapped(unittest.TestCase):
    def test_a_wrapped_paragraph_is_flagged(self):
        self.assertEqual([n for n, _ in prose.hard_wrapped(f"{LONG}\nand continues.\n")],
                         [2])

    def test_html_blocks_and_rules_start_their_own_block(self):
        for follower in ("<details>", "<summary>Logs</summary>", "</details>", "<br>",
                         '<img src="x.png" width="400">', "<!-- note -->"[:0] or "<p>",
                         "---", "***"):
            with self.subTest(follower=follower):
                self.assertEqual(prose.hard_wrapped(f"{LONG}\n{follower}\n"), [])

    def test_text_after_a_long_html_line_is_not_a_wrap(self):
        text = ('<summary>A summary line that is itself longer than sixty characters, '
                'easily</summary>\nBody text.\n')
        self.assertEqual(prose.hard_wrapped(text), [])
        text = f'<details open class="{"x" * 70}">\nBody text.\n'
        self.assertEqual(prose.hard_wrapped(text), [])

    def test_a_less_than_sign_in_prose_is_not_a_tag(self):
        self.assertEqual([n for n, _ in prose.hard_wrapped(f"{LONG}\n< 5 ms per frame.\n")],
                         [2])


if __name__ == "__main__":
    unittest.main()
