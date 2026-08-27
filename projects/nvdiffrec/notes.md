# nvdiffrec notes

## Intake screen (2026-08-27)

Upstream: https://github.com/NVlabs/nvdiffrec (2296 stars, not archived, last push
2026-08-12). "Extracting Triangular 3D Models, Materials, and Lighting From Images" --
inverse rendering that jointly optimises a triangle mesh, spatially-varying materials
and environment lighting from multi-view images.

**Recommendation: decline, `license-blocked`.** The licence decides this and the screen
stopped there, as the role intends. The dependency chain says the same thing
independently.

### Licence -- the decider

GitHub reports `NOASSERTION` ("Other"), so the field was not trusted and the files were
read directly.

    python3 utils/licenses.py check NVlabs/nvdiffrec
    # NVlabs/nvdiffrec: license=UNPARSED (GitHub returned NOASSERTION)

Screened against a scratch shallow clone (`git clone --depth 1 --recurse-submodules`),
since there is no fork clone at intake:

    python3 utils/licenses.py scan-nvidia agent_space/nvdiffrec-screen
    # scan-nvidia: 49 file(s) ... carry NVIDIA proprietary licence text

`LICENSE.txt` at the root is the **NVIDIA Source Code License for nvdiffrec**. Its
section 3.3 Use Limitation:

> The Work and any derivative works thereof only may be used or intended for use
> non-commercially. Notwithstanding the foregoing, NVIDIA and its affiliates may use
> the Work and any derivative works commercially. As used herein, "non-commercially"
> means for research or evaluation purposes only.

Recorded as `status.json.license_spdx = LicenseRef-NVIDIA-Source-Code-License`; there
is no registered SPDX identifier for this licence. `licenses.py tier` puts it at
**tier 4** -- not open source, contributing needs a person's approval and the
licensor's, by email rather than a PR.

This is the identical finding to the sibling project, already dispositioned:

    NVlabs/nvdiffrast -> skip, license-blocked, 2026-08-07
    "Wholly under the NVIDIA Source Code License, proprietary non-commercial."

**Every code file carries a second, stricter per-file header** -- all 49 hits, including
each of the five `.cu` files and `torch_bindings.cpp`:

    Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
    property and proprietary rights in and to this material, related
    documentation and any modifications thereto. Any use, reproduction,
    disclosure or distribution of this material and related documentation
    without an express license agreement from NVIDIA CORPORATION or
    its affiliates is strictly prohibited.

Note the internal discrepancy: `LICENSE.txt` grants a non-commercial right to prepare
and distribute derivative works, while the per-file headers say any use or distribution
without an express agreement is strictly prohibited. Only NVIDIA can resolve which
governs. It is **not** an unresolved-licensing case in the sense that would require
stopping to ask, because both readings are restrictive and neither permits an ordinary
upstream contribution -- the discrepancy changes nothing about the recommendation. It
would only matter if someone later sought NVIDIA's written permission, in which case
both instruments need naming.

Mixed licensing, checked: `.gitmodules` does not exist -- no submodules, nothing
vendored. The only other licence file is `data/spot/LICENSE.txt`, a public-domain 3D
model (Spot, released by Keenan Crane), which is a data asset and not a code-
contribution question.

**Scope reminder.** Tier 4 speaks to CONTRIBUTING upstream. It is separately relevant
here that this licence also bars commercial USE, so nvdiffrec could not be depended on
or shipped either -- but that is a different question from the one this screen answers.

### Doomed hard dependency

`nvdiffrast` is imported at module scope in six files (`train.py`, `render/render.py`,
`render/light.py`, `render/texture.py`, `render/util.py`, `render/regularizer.py`), so
it is unconditional, not optional. Recorded in `status.json.depends_on`. It carries the
recorded `license-blocked` skip disposition above, so it can never satisfy the selector's
dependency gate. Even if nvdiffrec's own licence were somehow cleared, the port would
have no usable base: nvdiffrast owns the differentiable rasteriser this project is built
on, and porting nvdiffrec without it is not meaningful.

`tinycudann` (NVlabs/tiny-cuda-nn) is the second external CUDA dependency, imported at
module scope in `render/mlptexture.py`, which `train.py:36` imports unconditionally --
so it too is import-time hard rather than config-optional. Its licence is **not** a
problem: GitHub also reports `NOASSERTION`, but the file is plainly BSD-3-Clause with an
NVIDIA copyright, i.e. tier 1. It is recorded here rather than in `depends_on` because
it is not a MOAT project and would need its own screen if ever pursued.

### Duplicate effort -- none

Run for completeness before the licence conclusion was written up; all negative.

- `gh search repos nvdiffrec --owner AMD-Ecosystem --owner ROCm` -> `[]`
- No fork of `NVlabs/nvdiffrec` exists under any AMD- or ROCm-named owner.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README.md` -> no hits; no "notable forks" section.
- `gh pr list --repo NVlabs/nvdiffrec --state open ...` -> `[]`. No open or draft port PR.
- Branches: `main`, `slang`, `gh-pages`. None port-related. (`slang` is upstream's own
  2023 rewrite of renderutils in slangpy to get autodiff without hand-written CUDA
  backward passes -- an alternative implementation, not an AMD effort.)

### Portability, for the record only

Not pursued in depth, because the licence stops the screen. What was observed in
passing, so a future reader need not re-derive it: the project's own CUDA surface is
small and ordinary -- five `.cu` files plus `torch_bindings.cpp` under
`render/renderutils/c_src/` (~110 KB total), built as a JIT `torch.utils.cpp_extension`
plugin named `renderutils_plugin`. No OptiX anywhere. In isolation that would be a
routine port; it is the licence and the nvdiffrast dependency, not the code, that make
this a decline.

### What a person is being asked

Confirm the decline with reason `license-blocked`, matching the recorded nvdiffrast
disposition. The alternative -- pursuing NVIDIA for written permission covering both
nvdiffrec and nvdiffrast -- is a person's call and outside what an agent may initiate.

No fork was created, no upstream write was made, and no disposition was recorded; the
scratch clone lives in gitignored `agent_space/`.
