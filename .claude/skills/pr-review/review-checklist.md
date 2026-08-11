# Port Review Checklist

Use with SKILL.md. Spawn sub-agents to verify items against the actual code. Report only violations, each with a file:line.

## 1. Port strategy
- [ ] Strategy matches the build type (`cuda-to-rocm` build classification): pure CMake -> Strategy A (compat header + LANGUAGE HIP); pytorch extension -> Strategy B (torch hipify). A correct implementation of the wrong strategy is Request Changes.
- [ ] Strategy A: a single cuda_to_hip.h compat header carries all aliases; it is a no-op on NVIDIA; there is no second HIP-aware header.
- [ ] Strategy A: .cu files are marked LANGUAGE HIP, not renamed to .hip (unless renaming is genuinely required).
- [ ] Strategy B: no hand-renamed symbols and no compat header; only fixes hipify cannot do (e.g. warp size) are in source, guarded by USE_ROCM.

## 2. Fault classes (AMD strict where CUDA lenient)
- [ ] No hardcoded warp/lane count of 32. Device code uses a per-arch constant (64 on __GFX9__, else 32); host code queries warpSize at runtime. Static shared-memory arrays use a compile-time upper bound or 64.
- [ ] Lane masks for __shfl*/__ballot/__activemask are 64-bit where the API takes a mask; no uint32 mask assumes wave32.
- [ ] Texture/stream/event RAII wrappers have rule-of-five: default-init handle to 0, move-only, guarded destructor. No double-free of a default handle.
- [ ] Kernels reading neighbor indices (+/-1, +/-width, stencils) clamp at edges; no read one-past-end relied upon.
- [ ] Pitched 2D texture binds respect 256-byte row pitch on AMD; point-sampled fetches prefer a linear bind to avoid pitch.
- [ ] Library calls swapped correctly (cuBLAS->hipBLAS, cuFFT->hipFFT, cuRAND->hipRAND, cuSPARSE->hipSPARSE, cuDNN->MIOpen, Thrust/CUB->rocThrust/hipCUB); handle types and v2-enum signature differences handled.

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
- [ ] HIP_ARCHITECTURES / arch flags set; the target builds for the intended gfx arch.
- [ ] ROCm dependencies are gated behind the HIP build; no new hard dependency on the CUDA/CPU build.
- [ ] CMake finds the HIP compiler (or documents CMAKE_HIP_COMPILER).

## 6. Testing
- [ ] The plan's real GPU test suite is actually run on the target arch. A port without a real-GPU run is Request Changes.
- [ ] Non-GPU tests are not regressed versus upstream.
- [ ] Any CPU-only docker smoketest is treated as a compile-only tripwire, never the validation gate.

## 7. No in-house vocabulary in upstream-visible text
- [ ] Commit messages, PR title/body, code comments and docs contain no MOAT jargon: "lead"/"follower", "Strategy A/B", "head_sha", "validated_sha", "revalidate", "curated commit", "moat-port", "port-ready", or "MOAT" itself. External maintainers do not know these terms, and their appearance has repeatedly caused review churn. Run `python3 utils/jargon.py --port <name>`; it must be clean. It scans the WHOLE branch every time, never just the round you are reviewing: a commit that passed an earlier round is still in the branch and still ships. faster-gaussian-splatting carried "Strategy B (torch hipify)" in its base commit through a full review because the check was scoped to the newest commit. If you find in-house vocabulary the checker missed, add it to `config/jargon.toml` in the same change. Keep the technical rationale, drop the in-house label -- say "a compatibility header", not "Strategy A"; name the GPU, not "the lead platform".

## 8. Commit hygiene
- [ ] Title prefixed [ROCm], <= 72 chars.
- [ ] Body explains the change and discloses assistance from an AI coding agent; NO Co-Authored-By noreply trailer; has a Test Plan section.
- [ ] History pushed with --force-with-lease; no bare --force; no ghstack. Multi-commit history on the port branch is fine -- the single-curated-commit rule was retired.
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
      same function the port renamed a call in, ten lines above the port's own hunk. Rebasing
      is nearly free before a validation or an approval lands on the head and expensive after:
      it moves the head sha, so every arch revalidates and `approval_currency` refuses with
      `stale-commits`. Ask for the rebase at review time, then have the validator run once on
      the rebased sha.
- [ ] A test added to an AMD-specific file exercises AMD-specific behaviour. A device-generic
      test parked in a HIP-only file never runs on the CUDA path, and a regression test for a
      wave64 fault that does not actually exercise wave64 proves nothing -- it just runs on an
      AMD GPU. Both directions are wrong: put device-generic tests where both backends run
      them, and make an arch-specific test exercise the arch.
- [ ] No em-dash; ASCII only; "ROCm" casing in prose.

## 9. Generalizable lessons promoted
- [ ] If this port produced knowledge that would help someone port a DIFFERENT project -- a new fault class, a strategy improvement, a diagnostic method, or a correction to an existing entry -- it is in the `cuda-to-rocm` skill's `references/`, with the source project named. A lesson left only in `projects/<name>/notes.md` is invisible to the next porter, and the same bug then gets paid for twice. A correction counts as much as an addition.
- [ ] **Read the lesson as carefully as the code, because merging this branch publishes it to every agent.** It rides the port branch deliberately so that you are the gate; there is no second review later. Check the claim against the source it describes, not against the porter's summary, and check that any code block in it is the FIXED form -- one entry reproduced verbatim the CMake defect it was documenting, which would have made that bug the recipe for every future port of that kind. A wrong lesson is a Request Changes on its own: the whole point of it reaching `main` is that others will follow it.

## 10. General code quality
- [ ] Matches the project's existing style and patterns (read the surrounding code).
- [ ] No dead code, debug prints, or commented-out blocks left behind.
- [ ] Comments carry non-obvious context, not restatement; ASCII only in new comments.
- [ ] Error handling on HIP API calls is not silently dropped.
