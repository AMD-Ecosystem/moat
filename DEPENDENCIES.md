# MOAT project dependencies

Some target projects build on top of other targets -- e.g. most of RAPIDS builds on `rmm`, and `cuml`/`cugraph` build on `raft`/`cudf`. MOAT models this so projects port in the right order and a porter knows how to consume an already-ported dependency instead of re-porting it.

## The model

- Each project's `status.json` carries `depends_on: [<MOAT project name>, ...]` -- the OTHER MOAT targets its build links or uses.
- The selector (`next_task` / `orient.sh`) will NOT pick a project until every entry in `depends_on` has its LEAD platform (`linux-gfx90a`) at state `completed`. That is the deps-first ordering: base libraries port first; dependents wait. `moatlib.py deps` shows the graph and what is waiting on what.
- `depends_on` is for HARD build dependencies of the project's CORE. A *module-level* optional dependency (only one extra feature needs another project) is documented in the project's `notes.md`, not added to `depends_on`, so it does not gate the whole port.

## Recording dependencies

- At adoption: `python3 utils/moatlib.py scaffold <owner/repo> --ext cmake --priority <p> --deps <depA> <depB>`
- Later: `python3 utils/moatlib.py set-deps <name> <depA> <depB>`
- View: `python3 utils/moatlib.py deps`

## Porting a project that has dependencies

When `orient.sh` hands you a project P, check `depends_on` (it is in `status.json`, or run `moatlib.py deps`). Because P only became actionable once its deps reached `completed`, each dependency D is already ported to a fork. To build P against D:

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
