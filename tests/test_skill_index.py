# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The cuda-to-rocm entry index: what an entry is, and that the committed index is current."""
import unittest

import gen_skill_index  # utils/ is put on sys.path by tests/__init__.py

SAMPLE = """# Faults: sample

Intro line.

## First section

**A bold lead is an entry.** Body text
continues here.

**A lead that wraps over
two lines is one entry.** More body.

Not bold, so not an entry.

### A heading entry

**A bold paragraph after a heading is a sub-point.** It belongs to the entry above.

## Second section

```
**bold inside a code fence is ignored**
```

**The rule resets in each section.**
"""


class Parse(unittest.TestCase):
    def test_entries(self):
        title, sections = gen_skill_index.parse(SAMPLE)
        self.assertEqual(title, "Faults: sample")
        self.assertEqual([(h, e) for h, e in sections if e], [
            ("First section", ["A bold lead is an entry",
                               "A lead that wraps over two lines is one entry",
                               "A heading entry"]),
            ("Second section", ["The rule resets in each section"]),
        ])

    def test_long_titles_are_cut_at_a_word(self):
        cut = gen_skill_index.shorten("word " * 60)
        self.assertLessEqual(len(cut), gen_skill_index.MAX_TITLE + 4)
        self.assertTrue(cut.endswith(" ..."))
        self.assertEqual(gen_skill_index.shorten("Short title."), "Short title")

    def test_toc_lists_sections(self):
        self.assertEqual(gen_skill_index.toc_text(SAMPLE),
                         "Contents:\n\n- First section\n- Second section")

    def test_region_must_exist_once(self):
        with self.assertRaises(SystemExit):
            gen_skill_index.replace_region("no markers", gen_skill_index.TOC_MARKS, "x", "f")
        self.assertEqual(
            gen_skill_index.replace_region("a\n<!-- toc:begin -->\nold\n<!-- toc:end -->\nb",
                                           gen_skill_index.TOC_MARKS, "new", "f"),
            "a\n<!-- toc:begin -->\nnew\n<!-- toc:end -->\nb")


class Committed(unittest.TestCase):
    def test_index_and_contents_are_current(self):
        self.assertEqual(gen_skill_index.stale(gen_skill_index.TOOLS_ROOT), [])

    def test_every_reference_file_is_listed(self):
        """A reference file with entries that the generator does not know would never be
        indexed; every Markdown file under references/ is either an entry file or one of
        the hand-indexed recipes."""
        skill = gen_skill_index.TOOLS_ROOT.joinpath(*gen_skill_index.SKILL_REL.parts)
        on_disk = {p.name for p in (skill / "references").glob("*.md")}
        known = set(gen_skill_index.TOC_FILES) | {"assess-existing-support.md", "naming.md",
                                                 "upstream-sync.md"}
        self.assertEqual(on_disk, known)


if __name__ == "__main__":
    unittest.main()
