# Port Review Checklist

Use with SKILL.md. Spawn sub-agents to verify items against the actual code. Report only violations, each with a file:line.

## 1. Port strategy
- [ ] Strategy matches the build type (`cuda-to-rocm` build classification): pure CMake -> Strategy A (compat header + LANGUAGE HIP); pytorch extension -> Strategy B (torch hipify). A correct implementation of the wrong strategy is Request Changes.
- [ ] Strategy A: the aliases live in one place -- a single cuda_to_hip.h compat header, or the shadow-header directory when the port uses the shim-header method -- and are a no-op on NVIDIA; there is no ad-hoc HIP-aware header beside them.
- [ ] Strategy A: .cu files are marked LANGUAGE HIP, not renamed to .hip (unless renaming is genuinely required).
- [ ] Strategy B: no hand-renamed symbols and no compat header; only fixes hipify cannot do (e.g. warp size) are in source, guarded by USE_ROCM.

## 2. Fault classes (AMD strict where CUDA lenient)
The list of fault classes is the index at the end of the `cuda-to-rocm` skill's `SKILL.md`: one line per entry, grouped by reference file, generated from the files so it is complete. Do not review against a remembered subset.
- [ ] Every group of that index was walked against the diff and the code around it, and each got a hit or a clean pass: `faults-wavefront.md`, `faults-memory-streams.md`, `faults-textures-interop.md`, `faults-numerics.md`, `build-headers-macros.md`, `build-cmake-link.md`, `libraries.md`, `platform-limits.md`, and `windows.md` when the project builds there. For a hit, the fix is the one in the entry.
- [ ] The three that every port meets were checked by reading code, not by grep alone: no hardcoded warp or lane count of 32 (device code uses a per-arch constant, host code queries `warpSize`, and a static array uses the bound that matches the direction the width enters its size); ballot and active-mask results are stored in 64 bits and `*_sync` masks are 64-bit on HIP; no allocation is read, or read-modify-written, before it is initialized.
- [ ] Library substitutions follow the table in `libraries.md`, including its "not 1:1" column: a cuDNN call layer is rewritten for MIOpen, not renamed, and a routine or mode the ROCm library lacks has a named fallback.

## 3. Minimal footprint
- [ ] Host (non-.cu) C++ is untouched except where genuinely required; the diff is small.
- [ ] `#if defined(USE_HIP)` guards are rare and only where behavior truly diverges.
- [ ] Plain CUDA spelling preserved outside the compat header (Strategy A).
- [ ] The NVIDIA/CUDA build still works; the change is additive and the CUDA path is unchanged (see bc-guidelines.md).

## 4. Arch-unified (shared branch)
- [ ] Fixes to shared (non-arch-guarded) code are correct on both wave32 and wave64; no per-arch hack that would regress the other AMD target.
- [ ] No change silently alters the CUDA-path numerics.

## 5. Build system
- [ ] enable_language(HIP) used (Strategy A); HIP gated behind a USE_HIP option (default OFF).
- [ ] No gfx arch is pinned in the build files: targets read `${CMAKE_HIP_ARCHITECTURES}` (user override, else host auto-detect), and the auto-detect path was configured once with no `-DCMAKE_HIP_ARCHITECTURES`.
- [ ] The HIP build really contains device code: the built library or executable carries a code object for the target arch (`llvm-objdump --offloading`), not only host objects that happen to link.
- [ ] ROCm dependencies are gated behind the HIP build; no new hard dependency on the CUDA/CPU build.
- [ ] CMake finds the HIP compiler (or documents CMAKE_HIP_COMPILER).

## 6. Testing
- [ ] The plan's real GPU test suite is actually run on the target arch. A port without a real-GPU run is Request Changes.
- [ ] Non-GPU tests are not regressed versus upstream.
- [ ] The CUDA path was compiled and linked with real nvcc (not run) when the port claims the NVIDIA build is unchanged, and always when it adds a CUDA/HIP-split source file; "the CUDA path is preserved" in the notes is not a compile.
- [ ] Both wavefront widths and Windows (when the project builds there) have a real-GPU run of the same branch, or the gap is stated.
- [ ] Any CPU-only docker smoketest is treated as a compile-only tripwire, never the validation gate.

## 7. No in-house vocabulary in upstream-visible text
- [ ] Commit messages, PR title/body, code comments and docs contain no MOAT jargon: "lead"/"follower", "Strategy A/B", "head_sha", "validated_sha", "revalidate", "curated commit", "moat-port", "port-ready", or "MOAT" itself. External maintainers do not know these terms, and their appearance has repeatedly caused review churn. Run `python3 utils/jargon.py --port <name>`; it must be clean. It scans the WHOLE branch every time, never just the round you are reviewing: a commit that passed an earlier round is still in the branch and still ships. faster-gaussian-splatting carried "Strategy B (torch hipify)" in its base commit through a full review because the check was scoped to the newest commit. If you find in-house vocabulary the checker missed, add it to `config/jargon.toml` in the same change. Keep the technical rationale, drop the in-house label -- say "a compatibility header", not "Strategy A"; name the GPU, not "the lead platform".

## 8. Commit hygiene
- [ ] Title <= 72 chars, prefixed [ROCm] when the commit is ROCm-specific. A commit that only fixes shared code (nothing HIP, ROCm or AMD in it) carries no prefix; the tooling cannot tell a wrongly prefixed commit, so check that each prefixed commit earns it and that shared-code fixes are not folded into ROCm commits.
- [ ] Body explains the change and discloses assistance from an AI coding agent; NO Co-Authored-By noreply trailer; has a Test Plan section.
- [ ] New commits go on top: a commit any platform has validated is never amended or force-pushed over (the amended commit is not the one that was tested). No ghstack. Multi-commit history on the port branch is fine -- the single-curated-commit rule was retired.
- [ ] The fork the PR is opened from matches `status.json.fork_url`.
- [ ] The port branch is not behind the fork's default branch. Read that branch from
      `status.json.fork_default_branch` rather than assuming `main`: 20 of the trunk's 50
      projects use something else (16 `master`, plus `develop`, `dev`, `v0`, `AdaLovelace`),
      and in a fork that carries both refs a hardcoded `origin/main` returns a plausible
      wrong number instead of failing. Fetch as part of the check: `origin/<default>` is
      otherwise only whatever that clone last saw, so a reviewer on a clone that has not
      fetched gets 0 and reads it as clean -- the exact false negative this item exists to
      prevent. This must print 0:
      `SRC=projects/<name>/src; DEF=$(python3 -c "import json;print(json.load(open('projects/<name>/status.json'))['fork_default_branch'])"); git -C $SRC fetch -q origin && git -C $SRC rev-list --count moat-port..origin/$DEF`
      No DIFF will reveal this. `git diff origin/<default>...HEAD` is merge-base relative, so it
      shows only what the port added and stays clean no matter how far behind the branch is,
      and `git merge-tree` reports no conflict whenever the hunks do not textually collide.
      It matters most in exactly the case where it is least visible: upstream changing a
      function the port also edits, a few lines away. On faster-gaussian-splatting the branch
      sat two commits behind for three reviews while upstream's tile-culling fix rewrote the
      same function the port renamed a call in, ten lines above the port's own hunk. Catching
      up is nearly free before a validation or an approval lands on the head and expensive
      after: it moves the head sha, so every arch revalidates and `approval_currency` refuses
      with `stale-commits`. Ask for it at review time, as a MERGE of the default branch (never
      a rebase, which rewrites commits that were already tested; see upstream-sync.md in
      `cuda-to-rocm`), then have the validator run once on the merged sha.
- [ ] A test added to an AMD-specific file exercises AMD-specific behaviour. A device-generic
      test parked in a HIP-only file never runs on the CUDA path, and a regression test for a
      wave64 fault that does not actually exercise wave64 proves nothing -- it just runs on an
      AMD GPU. Both directions are wrong: put device-generic tests where both backends run
      them, and make an arch-specific test exercise the arch.
- [ ] No em-dash; ASCII only; "ROCm" casing in prose.

## 9. Generalizable lessons promoted
- [ ] If this port produced knowledge that would help someone port a DIFFERENT project -- a new fault class, a strategy improvement, a diagnostic method, or a correction to an existing entry -- it is in the `cuda-to-rocm` skill's `references/` (the topic file its reader would open, per the skill's "Where to look" table), with the source project named and no MOAT tool, record, or state in it (MOAT-workflow lessons go to the `moat-porting` skill). A lesson left only in `projects/<name>/notes.md` is invisible to the next porter, and the same bug then gets paid for twice. A correction counts as much as an addition.
- [ ] **Read the lesson as carefully as the code, because merging this branch publishes it to every agent.** It rides the port branch deliberately so that you are the gate; there is no second review later. Check the claim against the source it describes, not against the porter's summary, and check that any code block in it is the FIXED form -- one entry reproduced verbatim the CMake defect it was documenting, which would have made that bug the recipe for every future port of that kind. A wrong lesson is a Request Changes on its own: the whole point of it reaching `main` is that others will follow it.

## 10. General code quality
- [ ] Matches the project's existing style and patterns (read the surrounding code).
- [ ] No dead code, debug prints, or commented-out blocks left behind.
- [ ] Comments carry non-obvious context, not restatement; ASCII only in new comments.
- [ ] Error handling on HIP API calls is not silently dropped.
