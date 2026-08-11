# MOAT

MOAT (Migration Orchestration via Automated Translation) is an AI-agent-driven effort to port popular CUDA GitHub projects to ROCm/HIP, one repository at a time. This repository is the control plane: Python, JSON, git, and shell tooling own the workflow and let either Claude Code or Codex resume it.

# Start here

1. Work on `port/<name>` in this repository, never `main`. The branch holds `projects/<name>/` while work is in flight and its existence is the project claim. `MOAT_ALLOW_TRUNK=1` is only for control-plane work such as tooling or documentation.
2. Run `bash utils/orient.sh` or invoke the `port-next` skill. It synchronizes state, detects the host platform, performs standing upkeep, and prints the next project and role. If no project is local, `python3 utils/moatlib.py fleet <platform>` names actionable branches.
3. Dispatch exactly the role reported by the selector: intake, planner, porter, reviewer, or validator. The role must read its canonical definition in `.claude/agents/<role>.md`, this file, the project records, and the `cuda-to-rocm` skill. State transitions gate every handoff; never skip one.
4. Work autonomously inside the boundary below. Stop at a human decision or a genuine blocker.

# Pipeline and durable state

`unclaimed -> intake -> planner -> porter -> reviewer -> validator -> [maintainer approval] -> submit upstream`.

- **intake** establishes the licence, checks duplicate effort and viability, scaffolds the project, and writes a typed recommendation. It never decides whether to adopt or decline.
- **planner** determines scope and strategy and writes `plan.md` plus `surface.json`. It never edits the fork.
- **porter** implements and builds the port on the fork's `moat-port` branch.
- **reviewer** reviews the fork diff with the `pr-review` skill and records problems in `notes.md`. It opens no PR.
- **validator** builds and runs real tests on one AMD platform and records evidence tied to the exact fork commit.
- Work after validation belongs to the `moat-checkup` skill because each maintainer interaction ends at a person.

`projects/<name>/status.json` is the source of truth. Use `utils/moatlib.py`; do not hand-edit state, locks, waivers, approvals, or dispositions. A failure and a pass each describe one commit. When `head_sha` advances, `moatlib` derives whether prior evidence still applies and dispatches revalidation when needed.

Coverage is expressed as the required gates in `config/arches.toml`, currently `wave64`, `wave32`, and `windows`. Any `<os>-<gfx>` platform with a known wavefront family may work. A gate is satisfied by any platform carrying that property that completed validation at the current `head_sha`. Extra platforms are additive evidence. Only a configured waivable gate may be waived, and only after a maintainer approves it; an agent may only suggest one.

There is no lead platform. The shared `porting` record serializes the exclusive `planning` and `porting` stages. Entering the stage acquires the lock and leaving releases it. Validation is unlocked and may run in parallel. If another platform holds the lock, stop; takeover is a person's decision, never a timeout.

# Two repositories, two port branches

- `port/<name>` in this MOAT repository holds the project's control-plane records while work is outstanding. Completed records reach `main` through review.
- `moat-port` in `AMD-Ecosystem/<project>` holds the actual source port. The fork default branch remains an unmodified upstream mirror.

Use `moatlib.project_record`, `all_projects`, `fleet`, or the corresponding CLI commands when asking what MOAT knows. A project folder may live on another ref, so scanning the checked-out `projects/` directory alone can produce the opposite answer.

# Human decisions and external writes

Agents may edit MOAT records and fork source, build and test locally, push MOAT project branches and `moat-port` branches, and perform read-only GitHub queries.

The following require a person's explicit decision:

- adopting a project: a person creates the fork; an agent must never run `gh repo fork` or create the organization repository;
- declining a candidate or declaring an adopted project not portable;
- approving a licence clearance, gate waiver, deferral ruling, or work-lock takeover;
- granting or repairing `/moat approve` or `/moat changes-requested` on a review PR;
- any GitHub-visible write to an upstream repository, including comments, reviews, issues, edited bodies, and maintainer replies.

An upstream maintainer asking MOAT to stop is the one decision an agent may record directly with `utils/optout.py record`, because it is their decision and can only reduce work. Never argue or ask for a reason.

The one pre-authorized upstream write is `python3 utils/upstream.py --publish --apply`: it may open the upstream PR only after the review PR on our fork contains the exact code, title, and body a person approved and the tool rechecks the live approval and all gates. Do not reproduce that write with an ad-hoc `gh` command.

`utils/gh_guard.py`, installed by `utils/install_hooks.py`, blocks obvious forbidden GitHub writes from either harness. It is defense in depth, not a substitute for scoped credentials. Do not bypass it, call the real `gh` binary, use `curl`, or otherwise route around a refusal. The trusted publisher resolves the real binary itself only after its approval checks pass.

For the full public-contribution and approval model, read `CONTRIBUTING.md` and the `moat-checkup` skill before preparing or publishing a PR.

# Standing rules

- Build the smallest complete port. Preserve upstream structure and the CUDA path; do not refactor unrelated code.
- Use the `cuda-to-rocm` skill for strategy, fault classes, terminology, and validation methods. Promote a lesson there only when it helps a different project; project-specific facts stay in `notes.md`.
- Before touching code, read the project `status.json`, `plan.md`, and `notes.md`. Resume recorded work instead of rediscovering it.
- Record exact commands, versions, pass/fail counts, and errors in `notes.md`. Do not record mutable fleet inventory.

# MOAT repository synchronization

This repository is shared by every host and harness. Pull before deciding and push each state transition promptly. `orient.sh` pulls the current project branch and merges relevant control-plane changes from the trunk. Route state and artifact writes through `moatlib`; prefer `python3 utils/moatlib.py commit-project <name> "<message>"`, which stages the project's status, plan, notes, surface, deferrals, and telemetry together and uses the synchronized commit/pull/push path. Do not implement a second synchronization path in a harness adapter.

`status.json` has the `moat-status` semantic merge driver; notes and telemetry use union merging. Run `bash utils/setup_git.sh` in a fresh clone to register them. If a push races, let `moatlib` retry its bounded synchronization. Do not resolve shared state by discarding another host's evidence.

# Telemetry and committing

Wrap build and test commands with `utils/timeit.sh <name> <phase> -- <command>` and sessions with `utils/session.sh <name> <platform> start|end`. If the harness exposes a reliable token count, the orchestrator may record it with `moatlib.py record-tokens <name> <tokens> "<role>:<harness>"`. Token telemetry is approximate, provider-dependent, and never gates work. Do not invent a count when the harness exposes none.

# Where a lesson goes

Capture project-specific evidence, recipes, and correspondence in `projects/<name>/notes.md`. Promote a lesson to the canonical `cuda-to-rocm` skill only when it would help someone porting a different project, and place it where a reader with that problem would look. Put the skill edit on the project branch that produced it so the same review judges the code and the lesson. Never lift an unreviewed lesson straight to `main`.

# Scratch space

Use `agent_space/` for temporary scripts and experiments. It is gitignored; do not commit it.

# Stop discipline

Budget roughly 60 minutes or 300k tokens per attempt. Never run an identical failing command more than twice; a third attempt must test a different hypothesis or stop. Classify the error before rebuilding. On Windows, DLL-load failures, exit 127, and a first-launch `hipErrorLaunchFailure (719)` are usually environment failures, not evidence that the port is wrong.

Always leave partial value: record what built, what passed, the exact failure, and relevant magnitudes. After the configured maximum attempts, set a concrete per-platform block and move on rather than thrashing. A source-wide `not-portable` verdict still belongs to a person.

# Integrity gate

Every source or build edit needed for validation must be committed to the fork before marking a platform completed. Check `git -C projects/<name>/src status --porcelain`; untracked build output is acceptable, modified tracked source/build files are not. `pr_ready` and `audit-clean` backstop this only on hosts carrying a local clone.

# PR review

Always invoke the `pr-review` skill when reviewing a port. Review the local fork branch rather than creating a mid-pipeline PR, and independently fact-check every reported finding. If the current harness cannot dispatch child agents from the reviewer role, perform the same investigation inline; parallelism is optional, verification is not.

# Build

Find the real build procedure in the project documentation and build files. Use standard package managers for missing build dependencies when appropriate, make the procedure repeatable, and record it in `notes.md`. Do not record mutable fleet inventory; reusable host or toolchain diagnostics belong in the validation reference of the `cuda-to-rocm` skill.

# Testing

Find the project's automated tests in its documentation and build files. Focus on GPU tests without regressing non-GPU tests. A real AMD GPU pass is required for completion. Do not add fork GitHub Actions smoke tests: CPU-only CI cannot observe GPU correctness and inherited fork Actions should remain disabled. A local CPU-only container compile check is evidence, never the GPU gate.

# Commit messages and upstream-visible text

Commit titles start with `[ROCm]` and are at most 72 characters. Commit bodies explain the rationale, disclose assistance from an AI coding agent, and include a Test Plan with literal commands in fenced blocks. Do not add a `Co-Authored-By` trailer for an agent.

No MOAT vocabulary may appear in upstream-visible commit messages, comments, code, PR titles, or PR bodies. Run `python3 utils/jargon.py --port <name>` before pushing. Write GitHub-rendered prose as one line per paragraph; run `utils/prose.py` on drafted bodies. Use ASCII in new comments and upstream prose.

Use HIP for the programming model and runtime API, and ROCm for the platform, toolkit, build target, and libraries. A code change is a HIP port targeting ROCm; `[ROCm]` remains the umbrella commit prefix.

# Coding style

Default to no new copyright or author lines. Add them only when the upstream house style clearly expects parallel attribution from external contributors. Match the target project's style, minimize comments, prefer explicit state and simple abstractions, and assume the reader understands the domain but not this change.

# Harness adapters

The workflow has one canonical copy and thin discovery adapters:

- This `AGENTS.md` is the canonical repository instruction file. Codex reads it natively; `CLAUDE.md` imports it for Claude Code. Add shared rules here.
- `.claude/skills/` contains the canonical skills because Claude Code discovers that location. `.agents/skills/` contains checked-in Codex adapters that load those same files. Each canonical skill must have exactly one adapter.
- `.claude/agents/` contains the canonical role definitions. `.codex/agents/` contains checked-in Codex definitions that tell the role to load its canonical Markdown definition. Each canonical role must have exactly one adapter.
- `.codex/config.toml` contains only portable checked-in Codex settings. No generated absolute paths or restart-after-bootstrap step is allowed.
- Claude-specific tool and model metadata stays in `.claude`; Codex-specific sandbox and model metadata stays in `.codex`. Shared role behavior belongs in the canonical Markdown body.

`python3 utils/check.py` verifies adapter coverage, configuration syntax, the Claude import, and this file's size. Never fix drift by copying the canonical body into an adapter.

# Project intake and dependencies

Review discovery candidates with `utils/triage.py review`. An agent may recommend a disposition but only a person's recorded decision may write one. Adopt with `utils/moatlib.py scaffold <owner/repo>` after the branch claim is established; `scaffold` refuses existing claims, dispositions, and opt-outs.

Hard build dependencies live in `status.json.depends_on`; optional module dependencies belong in `notes.md`. The selector waits until every hard dependency is completed or dispositioned as already supported/ported elsewhere. An unknown dependency needs an intake request; a doomed one requires rescoping or a human disposition rather than a doomed build attempt. Use `moatlib.py deps`, `dep-blocked`, and `DEPENDENCIES.md`. A dependency provider must document `## Install as a dependency` in its notes.

Deferred work belongs with the project that deferred it: use `utils/deferred.py add --project <name>` so it lands in `projects/<name>/deferred.json` and is reviewed with that port. The global `data/deferred.json` is only for genuinely unscoped work or a removed project. A person, never an agent, rules `defer` versus `now`; `utils/deferred.py pending` lists cases awaiting that decision. A local report in `findings/` is not durable publication—register it and bring it to the responsible person rather than leaving it invisible in a checkout.

# Where to look

- `utils/moatlib.py`: state machine, selector, gates, locks, synchronization, and project writes.
- `config/arches.toml`: required gates and wavefront-family mapping.
- `schema/status.schema.json`: generated status schema.
- `.claude/agents/`: canonical stage roles.
- `.claude/skills/cuda-to-rocm/`: porting strategies, fault classes, and validation knowledge.
- `.claude/skills/moat-checkup/`: approval, publication, maintainer rounds, and reconciliation.
- `projects/<name>/`: status, plan, notes, surface accounting, deferrals, and telemetry.
- `data/candidates.json`, `data/dispositions.json`, and `data/retired_stats.jsonl`: discovery decisions and retained telemetry.
- `CONTRIBUTING.md`, `DEPENDENCIES.md`, and `VISUAL.md`: detailed workflow rationale.

When uncertain, choose the safer, simpler action that preserves evidence and leaves every human decision with a person.
