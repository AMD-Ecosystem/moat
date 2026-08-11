@AGENTS.md

<!--
The rules live in AGENTS.md, which both harnesses read: Codex loads it natively, and
the line above imports it into Claude Code's context. One file, one copy, no drift.

A real file rather than a symlink, because git materialises a committed symlink as a
text file on Windows unless core.symlinks is on, and MOAT validates on Windows. A
Windows host would have found a one-line file here reading "AGENTS.md" and no rules at
all.

Do not add rules to this file. Anything written below the import is invisible to Codex.
-->
