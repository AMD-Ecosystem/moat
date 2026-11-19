# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The standalone-skill gate: what counts as MOAT vocabulary in cuda-to-rocm.

The skill is published outside this repository, so the gate has to catch every class
of home-repository term without tripping on the technical text around it.
"""
import unittest

import check  # utils/ is put on sys.path by tests/__init__.py

LEAKS = [
    # the repository, its tools and records
    "Run MOAT's selector first.",
    "use moatlib to record it",
    "run `python3 utils/jargon.py` before pushing",
    "record the nvcc command in notes.md",
    "the planner writes plan.md",
    "read status.json first",
    "registered with deferred.py against the project",
    "run it on a fleet host that has nvcc",
    "the current head_sha",
    "its validated_sha is orphaned",
    "record cuda-not-validated instead",
    "a codeobj_diff binary-equivalence check",
    "consumed through the _deps/ mechanism",
    "a report in findings/ is not publication",
    "a prior session's `upstream-issues` drafts",
    # a project folder in the home repository
    "the harness lives in projects/<name>/probe",
    "TornadoVM's fragment harness, `projects/TornadoVM/probe/mma/`",
    "see projects/oneflow for the round",
    # the fork organization
    "Known consumers: AMD-Ecosystem/alien (origin of this file)",
    # branches, states, dispositions
    "push it to moat-port",
    "stage it on moat-fix-12",
    "every arch has to revalidate",
    "a carry-forward records that",
    "a `validation-failed` recorded against this class",
    "once the platform is port-ready",
    "the stage is review-passed",
    "recorded cant-port on CUTLASS grounds",
    "the already-supported disposition",
    # deferral ids
    "(LichtFeld-Studio; deferral `rocjpeg-no-scan-jpeg-hangs-vcn`)",
    "the deferrals above",
    # roles and workflow
    "Send it back to the porter with the symbol list.",
    "plvs lost three porter sessions on two hosts",
    "Use the same toolchain as the Linux validators",
    "inside the validator role",
    "not only at intake",
    "that defeats the integrity gate",
    "inside a single 30-minute retry budget",
    "it is not part of the fork commit",
    # one host described as a fixture
    "the runtime is 7.12 on the gfx1151 host",
]

CLEAN = [
    "A real-GPU test run is the only validation that counts.",
    "Validate wave-size work on BOTH widths.",
    "NEVER amend a commit another arch has already validated.",
    "the commit that was validated is no longer the one that was tested",
    "the real Linux build stays the gate",
    "- Documentation: https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/",
    "sparse-checkout set projects/hipcub projects/rocprim projects/rocthrust` pulls a",
    "point `-I` at `projects/hipcub/hipcub/include`, `projects/rocprim/rocprim/include`,",
    "and `projects/rocthrust` (its `thrust/` subdirectory is the include root).",
    "hipMM (https://github.com/AMD-Ecosystem/hipMM, the ROCm-DS port of RMM)",
    "the reviewer's first question is why the other one was not used",
    "two review rounds and a full 20/20 GPU validation passed over the inlined header",
    "A project the port depends on may already have a ROCm build.",
    "copied from another host's notes, pointing at a path that does not exist",
    "this is porting work, not a rewrite",
    "the importer rewrites the folder name",
    "defer the decision to the maintainer",
    "On a host with four identical cards the order differed.",
    "measured on a gfx1151 APU with 72 GB of unified memory",
    "a `port/` prefix in a URL path such as https://example.com/port/x is not a branch",
    "the helpers sit in src/utils/ of the upstream project",
    "This is not portable to CDNA, which has no graphics pipeline.",
    "the library is already supported upstream",
    "build_deps/ holds the fetched sources",
]


class Vocabulary(unittest.TestCase):
    def test_leaks_are_caught(self):
        for line in LEAKS:
            with self.subTest(line=line):
                self.assertIsNotNone(check.MOAT_VOCABULARY.search(line))

    def test_technical_text_passes(self):
        for line in CLEAN:
            with self.subTest(line=line):
                m = check.MOAT_VOCABULARY.search(line)
                self.assertIsNone(m, m and m.group(0))


if __name__ == "__main__":
    unittest.main()
