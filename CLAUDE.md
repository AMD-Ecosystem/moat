# MOAT

MOAT (Moat Obliteration via Automated Translation) is a Claude-driven effort to port popular CUDA GitHub projects to ROCm/HIP, one repo at a time. Coverage is expressed as gates -- wave64, wave32, windows (config/arches.toml) -- not as a fixed list of architectures. Each port lives on a fork in the AMD-Ecosystem org. This repo is the control plane: it tracks progress, holds the porting knowledge, and lets any Claude CLI resume where the last one stopped.

# On startup, do this

1. Run `bash utils/orient.sh` (or `/port-next`). It pulls the latest MOAT state, detects this host's AMD arch, and prints the single next project to work on, the stage (planner, porter, reviewer, or validator), and a ready-to-paste dispatch line.
2. Dispatch the named subagent scoped to that project. State in `projects/<name>/status.json` gates every handoff; never skip a state.
3. Operate in auto mode within the Autonomy boundary below. Stop only at the two halts: the upstream-PR gate, and genuine blockers.

# Pipeline

`unclaimed -> intake -> planner -> porter -> reviewer -> validator -> [maintainer approval] -> submit upstream`.

Five agents. The reviewer can bounce back to the porter (changes-requested); the validator can bounce back to the porter (validation-failed). Review comes before validation deliberately: it is the cheap filter ahead of hours of GPU build.

**intake** decides viability -- licence first and gating, then duplicate effort and portability -- before any analysis effort is spent. **planner** decides scope and strategy. **porter** implements. **reviewer** checks the diff on a review-only PR inside the fork. **validator** proves it on real GPU. What follows approval -- submitting the PR, maintainer rounds, merge, and post-merge staleness -- is not an agent but the `moat-checkup` skill, because every step of it ends at a person.

Either intake or the planner may terminate a project; both terminations are recorded as a disposition and merged, so a negative outcome is still a deliverable.

Intake does everything except the fork: the branch, the project folder, the write-up, and a typed recommendation (`moatlib.py set-intake`). Then it sets `awaiting-fork` and stops. Screens collect into ONE issue -- `python3 utils/intake_queue.py publish --apply`, the `intake-queue` skill -- so a batch is one decision rather than one per project; a person answers in prose, an agent round-trips that reading as a small PR recording only the declines, and their approval of THAT is the record. Accepts need no PR at all. **The fork appearing in the org is what releases the project** -- agents cannot create one, so its existence is a deliberate act by someone who can, and that carries the decision without anything needing to model who made it. `moatlib.py release-forks` advances waiting projects; orient.sh runs `upstream.py --forks --apply` before every selection and prints what it found, and that path delegates to the same function so there is one implementation rather than two that drift. A project whose folder lives on its own branch is advanced too, through `commit_to_branch`, which writes a record to a branch without checking it out -- safe here because it advances a record rather than handing an agent a project whose files are absent. The selector deliberately does NOT do this: it only offers what this checkout can edit. Declines are not recorded by labels: `intake_queue.py apply --decline` is the only route.

Coverage is expressed as GATES, not a platform list: `wave64`, `wave32`, `windows` (config/arches.toml). Platforms are NOT enumerated anywhere -- a platform is `<os>-<gfx>`, whatever a host reports, and its gates follow from its name, so a machine with a new GPU works with no config change. Only an unknown wavefront family is refused, since guessing that wrong corrupts memory silently. A gate is satisfied by ANY arch carrying that attribute, so gfx90a/Linux covers wave64 while gfx1201/Windows covers wave32 and windows together -- two hosts suffice. Linux is not gated because wave64 is only satisfiable there. Extra archs are additive evidence and gate nothing. `windows` is the one waivable gate, and **every waiver needs maintainer approval** -- agents may only suggest one.

There is no lead platform. Any arch may start the work; a `{arch, since}` porting lock in status.json covers writes to the fork only, and validation runs unlocked and in parallel. Takeover of a held lock is a human decision, not a timeout (`moatlib.py port-lock <name> --take <arch>`). The lock is taken and released by the state transition itself -- entering `porting` acquires it, leaving releases it -- never by hand-editing the field. Recording a FAILURE never blocks: `validation-failed` and `changes-requested` are reachable on several archs at once, and it is the porter's dispatch that is serialised, not the record. Concurrent acquires do not conflict on push (status.json merges semantically and by design never hard-conflicts), so the merge driver resolves them by earliest acquisition -- first acquire wins, both hosts converge on the same winner, and the loser re-reads after pushing and backs off.

The AMD targets share one unified ROCm port, so a functional change re-validates the archs that already passed. Behavior-preserving changes do NOT: the regression guard (`moatlib.advance_head`) carries validation forward for documentation-only or comment/format-only deltas, and the validator may confirm a rename or refactor per-arch with a binary-equivalence check (`utils/codeobj_diff.py`) and carry forward without re-running GPU tests. Any classification uncertainty defaults to full revalidation.

Strategies and fault classes live in the `cuda-to-rocm` skill.

# Branches: two different things both called "the port branch"

Keep these apart; they are in different repositories and do different jobs.

- **`port/<name>` in THIS repo** holds a project's control-plane state -- its `projects/<name>/` folder. `main` is protected, so state reaches it by pull request, and the branch existing on the remote IS the claim that a project is in flight. It is shared: every host working that project pushes to it. Deleted when its PR merges.
- **`moat-port` on the FORK** (`AMD-Ecosystem/<project>`) holds the actual code port. This is what gets reviewed and what the upstream PR is opened from.

Work a project by checking out its `port/<name>`. The trunk carries projects that reached a terminal state; a project in flight lives only on its own branch, so `next-task` on the trunk cannot offer it -- you cannot edit files that are not in your tree. `python3 utils/moatlib.py fleet <platform>` lists actionable work across every ref and names the branch to check out, and orient prints it when nothing local is actionable.

Anything asking "what do we know about project X" must go through `moatlib.project_record` / `all_projects`, which resolve across the trunk and the port branches. Reading `projects/` directly answers "not adopted" for a project that is merely elsewhere, and those lead opposite ways: one must not be re-screened, the other must be.

`orient.sh` merges the trunk into your port branch, but only when something a port can feel changed there -- the skill, the agents, the tooling -- not for a README regeneration. A branch owns its own `projects/<name>/` and never takes the trunk's version of it.

# Autonomy boundary

Auto mode is maximal within these bounds.

Allowed without asking: edit the working tree and the fork clone in `projects/<name>/src/`; build, hipify, compile, run tests including on GPU; install missing build dependencies via apt or conda when a standard package exists; inside the fork clone do git branch/add/commit/amend/rebase and `git push --force-with-lease` to the fork; `gh` reads; update status.json, plan.md, notes.md, the `cuda-to-rocm` skill and push them to this MOAT repo.

Requires explicit approval: recording a DISPOSITION (`triage.py skip`) -- declining a project is a person's decision exactly as creating its fork is, and an agent may write the case but never the verdict. The same holds for judging an ADOPTED project unportable (`moatlib.py set-not-portable <name> --reason ... --by <who>`), which is where a project goes when the planner's analysis or repeated porter failure says the codebase cannot be ported at all; a record without `by` satisfies nothing. That verdict is about the SOURCE and so is project-wide. An operating system that will not take an otherwise working port is a gate WAIVER on `windows` instead, and a toolchain or library defect on one card stays a per-arch `blocked` flag with the report filed in `data/deferred.json`. `moatlib.py stalled` lists the projects where every arch that tried has given up, which is the queue those verdicts come from. Also requires explicit approval (show the draft, wait for yes): any GitHub-visible action against an UPSTREAM repo -- a PR or issue comment, a review, an edited PR body -- or anything else visible outside our own forks. Each post is its own act and needs its own yes; an approval never carries to the next one.

Opening the upstream PR is the ONE exception, because its content was already approved wholesale. The review PR on our own fork carries the code, the title and the body -- the entire upstream PR -- on one page, and approving it there approves all three at once. Nobody is asked to read the same thing twice somewhere else, and publishing it reveals nothing unread.

Approval is given as a REVIEW comment whose body carries the line `/moat approve` by itself, from someone with write access. GitHub will not let a pull request's author approve it, and MOAT is authored and reviewed by the same person -- agents run on the maintainer's credentials, so the APPROVED button is greyed out on every review PR. A review comment is allowed there AND records `commit_id`, which an ordinary issue comment does not; that binding is what proves an approval covers this code rather than an earlier push. A plain `APPROVED` review still counts wherever one is possible, ordinary review chatter never does, and an outstanding changes-requested from anyone still blocks. An agent may post the instructions and read the answer; it may never write the line.

An agent then SNAPSHOTS that approval with `moatlib.py record-pr-approval <name>`, which reads the review PR and stores who approved, the exact commit their review was attached to, and a hash of the title and body. Before opening upstream, the agent doing it runs `moatlib.py pr-approval <name>`, which re-reads the review PR and refuses if the approval was withdrawn or anyone has changes outstanding, if any commit landed after the approval, or if the title or body was edited after it. An unreachable review PR also refuses, and says so: a network outage must never read as a withdrawn approval.

That gate compares GitHub against GitHub -- the commit a review was given against versus the branch tip now, the body's last-edit time versus the approval's -- deliberately, because the snapshot lives in status.json on a project branch that agents write to. Editing the record cannot revive a stale approval. **An agent may neither grant an approval nor repair one**; both need the person who approved the first time. What this does not defend against is someone with write access to the fork rewriting history under the same sha -- out of scope, and a different problem from a push landing after approval and riding out with it.

A finished port needs a review PR on its own fork before any of this: `python3 utils/upstream.py --review` lists the ports whose gates pass and have none open, and `--review --apply --name <p> --title '<t>' --body-file <f>` opens it. That title and body ARE the upstream PR's, so they are checked for in-house vocabulary and hand-wrapping before anyone sees them.

Submission runs in a SESSION, never on a schedule: `python3 utils/upstream.py --publish --apply` re-checks the approval and every gate, opens the PR with the approved title and body verbatim, records it, and closes the review PR. It is local because opening a PR on a repository we do not own needs access no unattended job should hold, and `orient.sh` names any project waiting on it. If you open one by hand instead, run `python3 utils/moatlib.py set-pr-open <name> <pr_url> <pr_number>`; then `set-pr-merged <name>` when it merges, or `set-pr-closed <name> --note "<why>"` if it closes without merging. These are project-level (`pr_state`): the PR is one fact about the project, and opening it changes nothing any arch validated.

Licence gate on every route upstream: tiers 1 and 2 are cleared to contribute; **tier 3 and tier 4 ALWAYS block until a person approves that specific project** (`moatlib.py record-license-clearance <name> --by <who>`), and a clearance covers one project rather than its tier. An agent may carry someone's decision into the record but may never make it, so a clearance without `approved_by` satisfies nothing. A licence nobody recorded blocks too, but the remedy differs: reading a repo's licence and writing it to `status.json.license_spdx` is establishing a fact and any agent may do it. Check with `moatlib.py license-gate <name>`.

Upstream-PR readiness gate: every gate in `config/arches.toml` `required` must be satisfied before the single upstream PR opens. A gate is satisfied when ANY arch carrying that attribute is `completed` at the current head_sha -- a validation of superseded content proves nothing about this port -- or by a waiver, which only applies to a `waivable` gate and only with a maintainer's `approved_by`. An agent-suggested waiver satisfies nothing and blocks. An unsatisfied gate reports every arch that could still satisfy it, and completing any ONE clears it, so hardware that is gone or non-viable never wedges the PR as long as a sibling arch carries the same attribute. A gate whose candidate archs are ALL `blocked` genuinely blocks: no evidence exists for that attribute, and the correct move is a waiver request or a scoped claim, not proceeding. Archs beyond the ones satisfying a gate are additive evidence and gate nothing, so a host that later gains a GPU may validate purely to strengthen the record (a validation changes no head_sha, so it disturbs neither the open PR nor any other arch). Check with `python3 utils/moatlib.py pr-ready <name>`.

There is no PR-prep phase. A port is finished when it is finished, which includes the jargon scrub and documenting the ROCm build -- both are porting work (porter.md), not a step bolted on afterwards, and the validator confirms both before marking an arch `completed`. Squashing is optional: a `moat-port` branch may carry natural multi-commit history, and if you do collapse it to something tree-identical to validated content, `moatlib.py squash-carry-forward <name> <sha>` carries every completed arch forward so nobody re-runs.

Make progress without asking; ask only when truly unavoidable (install missing build deps yourself via apt/conda; do not ask for those). If you are genuinely stuck on a project after a real attempt (an unresolvable dependency, a porting-strategy decision with no clear answer, missing hardware or access, or repeated validation failure with unclear cause after 3 porter attempts), set the arch `blocked=true` with a concrete reason and MOVE ON to the next project rather than waiting; the blocked projects get summarized on request.

# Standing rules

- Forks live in the **AMD-Ecosystem** org and the upstream PRs are opened from them. Agents CANNOT create forks there -- creation is admin-only, so a project with no fork sits in `awaiting-fork` until someone creates one. Projects another team owns, or that fork nothing, are not tracked here at all; `data/dispositions.json` records why each was set aside.
- Never ghstack.
- The MOAT repo itself takes commits on top and is never force-pushed. Subproject forks may carry natural multi-commit history on the `moat-port` branch (the old single-curated-commit rule is retired -- it was overly restrictive); `git push --force-with-lease` is still allowed for rebases/cleanup, and bare `--force` without lease is forbidden everywhere. Squash to a tidy single commit only if/when you want it, e.g. right before opening the upstream PR. IMPORTANT: do not `git commit --amend` away a commit a platform has already validated (its `validated_sha`); amending orphans that commit on the remote and forces every passed platform to revalidate. Put follow-up edits in a NEW commit on top so `validated_sha` stays a reachable ancestor and the regression guard can classify the delta (see "MOAT repo synchronization" / cosmetic carry-forward). The port lives on a `moat-port` topic branch (the fork's default branch stays a clean upstream mirror); the single upstream PR is `moat-port` -> upstream default.
- Validation means exercising the change on real GPU. Lint is not validation. A CPU-only docker build smoketest proves compilation only, never GPU correctness, so it is never the sole validation gate.
- Commit titles: prefix `[ROCm]`, <= 72 chars. Body mentions Claude by name; no `Co-Authored-By: noreply` trailer.
- Prose: "ROCm" casing (code identifiers like USE_ROCM and arch names like gfx90a stay as-is). ASCII only, no em-dash (use -- or ; or parentheses). Do not manually line-wrap anything GitHub renders from a text box -- pull request bodies, issue bodies, review comments. GitHub reflows those to the reader's width, so hand-wrapped text freezes the author's line breaks in and makes every later edit reflow a whole paragraph into an unreadable diff. Write each paragraph as one line. Checked by `python3 utils/prose.py <file>`, which the review-PR and publish gates run on every body -- the rule was here in prose for a long time and agents kept breaking it anyway. This repo's OWN markdown (README, CONTRIBUTING, this file, agents and skills) is exempt and stays wrapped: it is read in an editor as often as on the web, and prose.py deliberately does not cover it. No sycophancy.
- ROCm vs HIP (use the right word; they name different facets -- see the `cuda-to-rocm` skill): HIP is the programming model -- the kernel-language dialect plus the `hipXxx` runtime API (the analogue of CUDA C++ and the CUDA runtime). ROCm is the platform/toolkit -- the compiler, runtime, driver, and the roc*/hip* domain libraries (the analogue of the CUDA Toolkit). The CODE port is "to HIP" (hipify, the cuda_to_hip.h shim, runtime symbols); the TARGET, build flag, and libraries are "ROCm" (USE_HIP/USE_ROCM, "ROCm 7.2.1", cuFFT->hipFFT). A pure language+runtime port is most precisely "a HIP port targeting ROCm." Do not call the platform "HIP" or the kernel dialect "ROCm". Commit prefix stays `[ROCm]` as the umbrella ("adds AMD support"); name a specific roc*/hip* library only when actually substituting it.
- No MOAT vocabulary in upstream-visible text (commit messages, code comments, PR titles/bodies). Keep the technical rationale, drop the in-house labels -- this text goes to external maintainers. The terms and their replacements live in `config/jargon.toml`; check with `python3 utils/jargon.py --commits <upstream-default>..moat-port -C <fork>` before pushing, and add any term the checker missed in the same change that found it. The range is the WHOLE branch, never just the commits you added: a message that passed an earlier round is still in the branch and still ships with it.
- Copyright and authorship in ported source files: default to NO added copyright or author lines; maintainers have pushed back on them more than once. Add `Copyright (c) <year> Advanced Micro Devices, Inc.` and an author credit (`Jeff Daily`) ONLY when the project's house style clearly does this for outside contributions (e.g. upstream files already carry per-company parallel copyright lines) AND there is no doubt it is welcome. When in doubt, or on any maintainer request, leave them out or remove them -- the project license (e.g. Apache-2.0 section 5) governs contribution terms regardless of in-file notices. Never add per-author/company lines under an ASF header. Trivial config/build-flag edits never need attribution.

# MOAT repo synchronization

This repo is shared by every CLI, so keep it fresh: pull before deciding, push often. orient.sh runs `git pull --rebase` before selecting. Route every status.json transition and artifact write through `moatlib.commit_and_push` (commit on top, pull --rebase, merge, push, bounded retry); for a project transition prefer `moatlib.commit_project(name, msg)` (or `moatlib.py commit-project`), which also stages that project's `stats.jsonl` so the per-phase telemetry timeit.sh writes is persisted and never accumulates uncommitted. status.json conflicts resolve via the `merge=moat-status` driver; notes.md and stats.jsonl use `merge=union`.

Telemetry: agents wrap build/test phases in `utils/timeit.sh` (wall-clock) and bracket runs with `utils/session.sh` (session wall). Tokens can only be recorded by the ORCHESTRATOR: when a dispatched subagent task completes, its notification reports `subagent_tokens` -- record it with `python3 utils/moatlib.py record-tokens <name> <tokens> "<agent role>"` so token cost is captured for the README/blog metrics (the subagent cannot self-report; the count exists only in the parent's completion notification). statlib.py aggregates compile/test wall, session/thinking wall, and tokens (always approx=True).

# Stop discipline (every agent)

Canonical here so no agent restates it. Budget roughly 60 minutes wall-clock / 300k tokens per attempt; as you approach it, stop and report partial state.

- Never re-run an IDENTICAL failing command more than twice. A third identical retry is forbidden -- the next action must be a different hypothesis or a stop. Re-running a broken build hoping it changes is the single biggest token sink.
- Triage the error CLASS before grinding. On Windows an exit 127, a "DLL"/"cannot load"/"image not found" message, or `hipErrorLaunchFailure (719)` on first launch is a runtime-environment problem, not a GPU or port fault. Fix the environment once; do NOT rebuild in response to a DLL-load error.
- Always leave partial value. Even when stopping, record what BUILT, which suites PASSED, and the verbatim blocking error with magnitudes, so the next run resumes from there rather than from zero. A crisp diagnosis beats an hour of grinding.
- After `max_attempts` (3; config/moat.toml states it, no code enforces it) failed cycles with unclear root cause, set `blocked` with a concrete reason and move on. Never thrash.

# Integrity gate (every agent that touches a fork)

Every source or build edit needed to build or run a validation MUST be committed to the fork's port branch BEFORE marking `completed`. Validating against uncommitted local edits leaves the branch -- and its upstream PR -- unbuildable; this is the exact gap that stranded baspacho and arrayfire. Check `git -C projects/<name>/src status --porcelain` first: untracked build artifacts are fine, modified tracked source/build files are not. `moatlib pr_ready` hard-blocks a dirty fork and `moatlib audit-clean` scans for it, but the agent at the keyboard is the first line.

# Telemetry and committing (every agent)

Wrap every build/test phase in `utils/timeit.sh <name> <phase> -- <cmd>` so wall-clock is recorded. Push each transition with `python3 utils/moatlib.py commit-project <name> "<msg>"`, which stages status.json, notes.md, plan.md and stats.jsonl together so telemetry is committed with the transition and never accumulates uncommitted in the shared tree.

# Where a lesson goes

Three destinations existed and the one wired into agents' instructions won, so state it once:

**Capture in `projects/<name>/notes.md`.** Every agent already does this and it is correct -- 93% of what lands there is validation provenance, maintainer correspondence and project build recipes, all of which belong exactly where they are.

**Promote to the `cuda-to-rocm` skill** when the lesson passes one test: *would this help someone porting a DIFFERENT project?* If yes it is a fault class, a strategy, or a diagnostic method, and it goes in the skill's `references/` with the source project named. If no, it stays in notes.md. If a tool should read it, it belongs in a typed field, not prose.

**File it by who needs it, not by where you found it.** The activity that surfaces a lesson is rarely the question its reader will be asking. A torch hipify-generation rule found during a Windows build belongs with Strategy B, not under OS-keyed guards; an nvcc no-regression check found during PR prep belongs with validation, not with the fault classes. Ask which file someone would open with this problem, and if two could hold it, put the rule in one and a pointer in the other.

**Promote at the moment of learning, not in a later sweep.** A separate lessons file that no agent was instructed to open went unused from the day it was created; the destination has to be one agents are actually told to write to, with something checking it happened.

The reviewer checks that promotion happened (see the pr-review checklist). Correcting an existing entry counts: SCAMP proved the warp-count sizing rule in the skill was wrong in one direction, and that correction was worth more than any addition.

# Scratch space

Use `agent_space/` (gitignored, at repo root) for temporary scripts and throwaway experiments. Do not commit files from this directory.

# PR review

When asked to review work (the reviewer agent or otherwise), always use the /pr-review skill.

# Build

Check local memory for build configuration (env vars, incremental-build shortcuts) before building, and apply what you find. If nothing applies, search the project for build docs or analyze its build files (CMake, setup.py). Create repeatable build scripts as needed and record them in the project's notes.md.

Do not record which machines exist, what is installed in them, or which are retired -- that is fleet state, it goes stale silently, and MOAT is a record of ports. A platform exists because a host reported it; a platform nobody runs simply never appears. Anything durable a machine taught us -- the Windows TheRock-versus-HIP-SDK trap, the multi-GPU runtime crash, how to pick a device index -- belongs in the `cuda-to-rocm` skill's `references/validation.md`, where the next port can find it.

# Testing

Find the project's automated tests from its docs or build files. Focus on GPU tests, but do not regress non-GPU tests. Full validation is required to mark success. Do NOT add GitHub Actions smoketest workflows to ports: a CPU-only GHA build cannot observe any GPU fault (so it is not a real gate), and every yml edit changes the fork HEAD sha, which trips the cross-platform regression guard and forces all platforms to revalidate -- not worth the churn or the failing-run email noise. Leave upstream workflows as they are; disable Actions on a newly created fork so neither our changes nor inherited upstream CI run and email on it: `gh api -X PUT repos/AMD-Ecosystem/<fork>/actions/permissions -F enabled=false`. A local CPU-only docker build (image `rocm/dev-ubuntu-24.04:7.2.4-complete`) is fine as a manual compile check, never wired into the fork's Actions.

# Commit messages

Do not bullet-list individual changes. If the change is large, explain the order to review it; if short, omit the list. Include a Test Plan section with the literal commands run in fenced code blocks. If fixing a bug, explain the root cause and how the fix works. Disclose that the work was authored with an AI assistant. When amending, check that the message still describes the change.

# Coding style

- Minimize comments; code should be self-documenting. Comments carry non-obvious global context, not restatement of the code.
- No trivial 1-2 line single-use helpers unless they clearly improve readability.
- Prefer clear abstractions and explicit state. No dynamic setattr/getattr field juggling.
- Match the existing style of the project being modified.
- Assume the reader knows the project's domain but not this specific code.
- ASCII only in new comments. Leave preexisting Unicode in untouched comments alone.

If uncertain, choose the simpler, more concise implementation.

# Where things live

- `.claude/skills/cuda-to-rocm/` -- the porting knowledge: SKILL.md is the always-loaded index, references/ hold the detail.
- `.claude/skills/intake-queue/` -- publishing the batch queue and carrying a decision back into state.
- projects/<name>/ -- plan.md, notes.md, status.json, stats.jsonl per project.
- utils/ -- orient.sh (entrypoint), moatlib.py (state machine + sync), discover.py, gen_readme.py.
- .claude/agents/ -- intake, planner, porter, reviewer, validator.
- data/candidates.json -- ranked discovery output.
- findings/<slug>/ -- prepared ROCm-component bug reports and reproducers. Gitignored, so it is local to a working copy and not published: an unfiled report in a porting repo is one the component owners will never see. `data/deferred.json` is the durable record of what was found and whether it was filed; file the report properly rather than leaving it here.
- data/deferred.json -- the deferred-work registry (what we postponed and where to resume). Ask MOAT "what did we defer?" with `python3 utils/deferred.py list`; record a deferral with `utils/deferred.py add` (kinds: rocm-bug-report, feature-port, other). When you scope a sub-feature out of a port or prepare a findings/ bug report you do not file, register it here so it is not lost.

# How to add a project

Review candidates with `python3 utils/triage.py review`. **Declining is a person's decision and an agent may only recommend one** -- the same rule as creating the fork. A person marks a project not-to-port with `python3 utils/triage.py skip <owner/repo> --reason <already-supported|ported-elsewhere|cant-port|not-a-target|duplicate|license-blocked|declined|other> --note "..."` (or `triage.py verify <owner/repo>` to flag one for investigation without skipping); decisions persist in data/dispositions.json, keyed by GitHub repo id so a rename cannot slip a decided project back into discovery. `scaffold` refuses a skipped project, and refuses a name already claimed by `port/<name>` on the remote or by a folder on the trunk. Adopt a remaining row with `python3 utils/moatlib.py scaffold <owner/repo>` (writes projects/<name>/{status.json,notes.md}); orient.sh then picks it up.

# Project dependencies

Some targets build on other targets (`barney` on `cuBQL`; `anari-visionaray` on `visionaray`; `plvs` on `opencv_contrib`). A project's status.json `depends_on` lists the MOAT projects its build needs, and the selector will not pick a project until each one clears (`moatlib.py deps` shows the graph). A dep clears four different ways and they are NOT the same answer -- `moatlib.py dep-blocked <platform>` prints the verdict and the fix:

- **ok** -- a port validated on some arch, or the dep is dispositioned `already-supported`/`ported-elsewhere` so it needs no port. Build against it.
- **waiting** -- adopted and in the pipeline. It clears on its own; nothing to do.
- **doomed** -- dispositioned `cant-port`/`license-blocked`. It will never be portable, so neither is anything that links it: scope the dependent around that feature, or recommend a disposition for it too. Do not proceed and discover this at build time.
- **unknown** -- nobody has looked at it. File an intake request: `python3 utils/port_request.py file <owner/repo> --blocks <project> --why "..." --apply`. That opens a `port-request` issue, the same queue a community suggestion lands in, and a person decides whether to fork it. Record the edge with `set-deps` so the block is mechanical rather than remembered. When porting a project that has deps, clone + build + install each ported dep (its `fork_url` @ moat-port) per its notes.md "## Install as a dependency" section into `_deps/<dep>/` (gitignored at the repo root) and point your build at it (e.g. `-DCMAKE_PREFIX_PATH=.../_deps/<dep>/install`). Record deps with `scaffold --deps ...` or `set-deps <name> <deps...>`. A base library other targets consume MUST document an "## Install as a dependency" section in its notes.md. Full workflow: DEPENDENCIES.md.
