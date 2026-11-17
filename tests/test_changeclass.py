# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import pathlib
import subprocess
import tempfile
import unittest

import changeclass  # utils/ is put on sys.path by tests/__init__.py
from changeclass import _classify_file, _is_doc


def cls(lang, old, new):
    return _classify_file(lang, old, new)[0]


class CFamily(unittest.TestCase):
    def test_comment_and_whitespace_only(self):
        old = "int f(int a) {\n  return a + 1;\n}\n"
        new = "// adds one\nint f(int a)\n{\n    return a + 1; /* done */\n}\n"
        self.assertEqual(_classify_file("c", old, new)[:2], ("comment-only", True))

    def test_literal_change_is_mixed(self):
        self.assertEqual(cls("c", "int x = 32;\n", "int x = 64;\n"), "mixed")

    def test_consistent_rename(self):
        old = "int warp(int a) { return a; }\nint g() { return warp(1); }\n"
        new = "int wave(int a) { return a; }\nint g() { return wave(1); }\n"
        kind, arch_independent, _ = _classify_file("c", old, new)
        self.assertEqual((kind, arch_independent), ("rename-only", False))

    def test_inconsistent_rename_is_mixed(self):
        old = "int a(int x) { return x; }\nint g() { return a(1); }\n"
        new = "int b(int x) { return x; }\nint g() { return c(1); }\n"
        self.assertEqual(cls("c", old, new), "mixed")

    def test_identifier_to_keyword_is_mixed(self):
        self.assertEqual(cls("c", "foo x;\n", "int x;\n"), "mixed")

    def test_line_shift_with_assert_is_not_arch_independent(self):
        old = "void f(int a) {\n  assert(a);\n}\n"
        new = "// note\nvoid f(int a) {\n  assert(a);\n}\n"
        self.assertEqual(_classify_file("c", old, new)[:2], ("comment-only", False))

    def test_raw_string_bails(self):
        src = 'const char* s = R"x(a)x";\n'
        self.assertEqual(cls("c", src, src + "// c\n"), "mixed")

    def test_token_moved_out_of_a_define_is_mixed(self):
        old = "#define A\nB\nint x;\n"
        new = "#define A B\nint x;\n"
        self.assertEqual(cls("c", old, new), "mixed")

    def test_token_moved_out_of_an_if_is_mixed(self):
        old = "#if defined(X) && \\\n    defined(Y)\nint x;\n#endif\n"
        same = "#if defined(X) && defined(Y)\nint x;\n#endif\n"
        moved = "#if defined(X) &&\ndefined(Y)\nint x;\n#endif\n"
        self.assertEqual(cls("c", old, same), "comment-only")
        self.assertEqual(cls("c", old, moved), "mixed")

    def test_comment_after_a_directive_is_comment_only(self):
        old = "#define N 4\nint a[N];\n"
        new = "#define N 4  // lanes\nint a[N];\n"
        self.assertEqual(cls("c", old, new), "comment-only")


class Python(unittest.TestCase):
    def test_comment_only(self):
        old = "def f(a):\n    return a + 1\n"
        new = "# adds one\ndef f(a):\n    return a + 1  # done\n\n"
        self.assertEqual(cls("py", old, new), "comment-only")

    def test_statement_moved_out_of_an_if_is_mixed(self):
        old = "if ok:\n    a()\n    b()\n"
        new = "if ok:\n    a()\nb()\n"
        self.assertEqual(cls("py", old, new), "mixed")

    def test_joined_lines_are_mixed(self):
        self.assertEqual(cls("py", "a = 1\nb\n", "a = 1; b\n"), "mixed")

    def test_cython_directive_comment_is_code(self):
        old = "# cython: boundscheck=True\ncdef int f(): return 1\n"
        new = "# cython: boundscheck=False\ncdef int f(): return 1\n"
        self.assertEqual(cls("py", old, new), "mixed")

    def test_distutils_comment_added_is_code(self):
        old = "cdef int f(): return 1\n"
        new = "# distutils: sources = extra.c\ncdef int f(): return 1\n"
        self.assertEqual(cls("py", old, new), "mixed")

    def test_rename_keeps_layout(self):
        old = "def f(a):\n    return a\n"
        new = "def g(b):\n    return b\n"
        self.assertEqual(cls("py", old, new), "rename-only")


class CMake(unittest.TestCase):
    def test_reflowed_arguments_are_comment_only(self):
        old = "add_library(x a.cu b.cu)\n"
        new = "# sources\nadd_library(x\n  a.cu\n  b.cu)\n"
        self.assertEqual(cls("cmake", old, new), "comment-only")

    def test_changed_argument_is_mixed(self):
        self.assertEqual(cls("cmake", "set(X 1)\n", "set(X 2)\n"), "mixed")


class DocPaths(unittest.TestCase):
    def test_documents(self):
        for p in ("README.md", "docs/build.rst", "doc/conf.py", "LICENSE",
                  "src/docs/notes.md", ".github/workflows/ci.yml", ".gitignore"):
            with self.subTest(path=p):
                self.assertTrue(_is_doc(p))

    def test_source_under_a_nested_docs_directory_is_not_a_document(self):
        for p in ("src/docs/render.cu", "python/pkg/docs/__init__.py",
                  "docs_src/kernel.cu", "src/kernel.cu", "CMakeLists.txt"):
            with self.subTest(path=p):
                self.assertFalse(_is_doc(p))


class Classify(unittest.TestCase):
    """classify() end to end, against a scratch repository."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = pathlib.Path(self._tmp.name)
        self.git("init", "-q")

    def git(self, *args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
             "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
            cwd=str(self.repo), check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, files):
        for path, text in files.items():
            f = self.repo / path
            f.parent.mkdir(parents=True, exist_ok=True)
            if text is None:
                f.unlink()
            else:
                f.write_text(text, encoding="utf-8", newline="\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "c")
        return self.git("rev-parse", "HEAD")

    def verdict(self, before, after):
        a = self.commit(before)
        b = self.commit(after)
        return changeclass.classify(str(self.repo), a, b)

    def test_doc_only(self):
        v = self.verdict({"README.md": "a\n", "k.cu": "int x;\n"}, {"README.md": "b\n"})
        self.assertEqual((v.cls, v.arch_independent), ("doc-only", True))

    def test_comment_only_source(self):
        v = self.verdict({"k.cu": "int x;\n"}, {"k.cu": "int x; // lanes\n"})
        self.assertEqual((v.cls, v.arch_independent), ("comment-only", True))

    def test_code_change_is_mixed(self):
        v = self.verdict({"k.cu": "int x = 1;\n"}, {"k.cu": "int x = 2;\n"})
        self.assertEqual((v.cls, v.arch_independent), ("mixed", False))

    def test_added_source_file_is_mixed(self):
        v = self.verdict({"k.cu": "int x;\n"}, {"n.cu": "int y;\n"})
        self.assertEqual(v.cls, "mixed")

    def test_unknown_file_type_is_mixed(self):
        v = self.verdict({"data.bin": "a\n"}, {"data.bin": "b\n"})
        self.assertEqual(v.cls, "mixed")

    def test_one_code_change_outweighs_any_number_of_documents(self):
        v = self.verdict({"README.md": "a\n", "k.cu": "int x = 1;\n"},
                         {"README.md": "b\n", "k.cu": "int x = 2;\n"})
        self.assertEqual(v.cls, "mixed")

    def test_source_in_a_nested_docs_directory_is_judged_as_source(self):
        v = self.verdict({"src/docs/k.cu": "int x = 1;\n"},
                         {"src/docs/k.cu": "int x = 2;\n"})
        self.assertEqual(v.cls, "mixed")

    def test_bad_sha_is_mixed(self):
        a = self.commit({"k.cu": "int x;\n"})
        v = changeclass.classify(str(self.repo), a, "0" * 40)
        self.assertEqual((v.cls, v.arch_independent), ("mixed", False))


if __name__ == "__main__":
    unittest.main()
