# MOAT project dependencies

Some target projects build on top of other targets -- e.g. `barney` builds on `cuBQL`, `anari-visionaray` on `visionaray`, `plvs` on `opencv_contrib`. MOAT models this so projects port in the right order and a porter knows how to consume an already-ported dependency instead of re-porting it.

## The model

- Each project's `status.json` carries `depends_on: [<MOAT project name>, ...]` -- the OTHER MOAT targets its build links or uses.
- The selector (`next_task` / `orient.sh`) will NOT pick a project until every entry in `depends_on` clears. There is no lead platform: ANY arch at `completed` satisfies a dependency, because a validated port is a validated port whoever ran it. `moatlib.py deps` shows the graph; `moatlib.py dep-blocked <platform>` shows what is waiting and why.
- A dependency clears four different ways, and they are NOT the same answer:
  - **ok** -- some arch reached `completed`, OR the dependency is dispositioned `already-supported` / `ported-elsewhere`, meaning it needs no port from us. Build against it.
  - **waiting** -- adopted and in the pipeline. It clears on its own.
  - **doomed** -- dispositioned as anything short of a port existing: `cant-port`, `license-blocked`, `declined`, `duplicate`, `not-a-target`, `opted-out` or `other`. It will not be ported, so neither is anything that links it. Scope the dependent around that feature or recommend a disposition for it too; do not proceed and find out at build time.
  - **unknown** -- nobody has looked at it. File an intake request with `python3 utils/port_request.py file <owner/repo> --blocks <project> --why "..." --apply`, record the edge with `set-deps`, and stop. A person decides whether to fork it.
- `depends_on` is for HARD build dependencies of the project's CORE. A *module-level* optional dependency (only one extra feature needs another project) is documented in the project's `notes.md`, not added to `depends_on`, so it does not gate the whole port.

## Recording dependencies

- At adoption: `python3 utils/moatlib.py scaffold <owner/repo> --ext cmake --priority <p> --deps <depA> <depB>`
- Later: `python3 utils/moatlib.py set-deps <name> <depA> <depB>`
- View: `python3 utils/moatlib.py deps`

## Porting a project that has dependencies

When `orient.sh` hands you a project P, check `depends_on` (it is in `status.json`, or run `moatlib.py deps`). Because P only became actionable once each dep cleared, a dependency D is either already ported to a fork, or dispositioned `already-supported`/`ported-elsewhere` -- in that case build against the upstream package and skip the rest of this section. To build P against a ported D:

1. Clone the ported dependency (the `moat-port` branch is the deliverable):
   `git clone -b moat-port https://github.com/AMD-Ecosystem/<D> _deps/<D>/src`
   (`_deps/` at the repo root is gitignored -- it is a local build/install area, never committed.)
2. Build + install D following the **`## Install as a dependency`** section of `projects/<D>/notes.md` -- typically `cmake -S _deps/<D>/src -B _deps/<D>/build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=<arch> ... && cmake --build _deps/<D>/build --target install` with `-DCMAKE_INSTALL_PREFIX=/var/lib/jenkins/moat/_deps/<D>/install`.
3. Point P's build at it: `-DCMAKE_PREFIX_PATH=/var/lib/jenkins/moat/_deps/<D>/install` (so P's `find_package(<D>)` resolves the ROCm build), or the include/lib paths that D's notes document.
4. If a dependency is somehow not actually usable when you need it, `set-blocked` P with a concrete "needs <D>" reason and move on -- do NOT port the dependency inline unless it is trivial.

## The "Install as a dependency" convention (for ported base libraries)

If a project is (or is likely to be) a dependency of another MOAT target, its `notes.md` MUST include a `## Install as a dependency` section giving: the exact configure + build + install commands (with the HIP flags and arch-from-`CMAKE_HIP_ARCHITECTURES`), the install-prefix layout, and what a dependent sets to consume it (the `find_package` package name and/or the include + link flags). A base library that several targets consume especially needs it.

## Known dependency graphs

Module-level (not hard `depends_on`):
- `cupoch`'s deferred `imageproc` module needs `libSGM` (CUDA semi-global-matching stereo). Porting `libSGM` would unblock that cupoch module; the cupoch core does not need it.
- `RXMesh`'s matrix/solver/diff module needs the low-level cusolverSp csrqr API (ROCm/hipSOLVER#443, filed) and a cuDSS-class GPU direct solver (maps to STRUMPACK, completed). It is header-only; the delivered RXMesh core + mesh-query + dynamic-editing port does not need it.
