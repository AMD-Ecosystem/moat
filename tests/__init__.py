# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the tools in utils/.

The tools re-run themselves from the trunk export only when executed as scripts
(`trunk.reexec` sits under `__main__`), so importing them tests this checkout's
code. Run from the repository root:

    python3 -m unittest discover -s tests -t .
"""

import pathlib
import sys

UTILS = pathlib.Path(__file__).resolve().parents[1] / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))
