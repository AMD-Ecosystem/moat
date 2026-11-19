---
name: port-next
description: Resume MOAT. Detect this host's AMD platform, find the next actionable project, and dispatch the right role. Use when starting work in the MOAT repo or when asked to continue porting or "port next".
---

# port-next

Run the MOAT pipeline for the next available project on this host.

## Steps
0. Be on the right branch. MOAT work happens on `port/<name>` in this repo, not on `main` -- orient refuses the trunk (exit 2). An in-flight project's folder lives only on its own branch, so check out the branch for the project you intend to work (`git checkout port/<name>`), or create it when adopting one. Not to be confused with `moat-port`, which is the branch on the FORK that holds the code. To find that branch from `main`, run `MOAT_ALLOW_TRUNK=1 bash utils/orient.sh`: it does the standing upkeep, prints the platform, and lists the actionable branches with the checkout command for each, dispatching nothing. `python3 utils/moatlib.py fleet <platform>` is the same list as tab-separated rows, and `bash utils/detect_arch.sh` prints `<platform>`.
1. Run `bash utils/orient.sh`. It pulls the latest state, detects the AMD arch, and prints the next project, its state, and the stage (intake / planner / porter / reviewer / validator), or NONE. It is not a read-only command (it installs hooks, merges the trunk into the port branch and pushes that, and releases forks). Exit 3 means the host is not set up: `gh` on PATH is not the guard launcher, or no Python can run the tools; the message names the fix. A `lock     : WARNING` line under `next` means the round on offer was started by another host or job of this platform: entering the stage takes its lock over, so dispatch only if that one has stopped.
2. If it names a project + stage, dispatch the named role adapter scoped to the project and platform: "Use the <stage> role on projects/<name> for <platform>". Claude Code discovers the roles in `.claude/agents/`; Codex discovers the matching adapters in `.codex/agents/`; both execute the canonical role body in `.claude/agents/<stage>.md`. Stages are intake, planner, porter, reviewer and validator. Nothing after approval is a stage: submitting upstream and the maintainer rounds that follow are the `moat-checkup` skill, run in this session, because each step ends at a person. The child agent reads AGENTS.md (Pipeline, Human decisions and external writes, Standing rules, Stop discipline), plan.md, notes.md, and the `cuda-to-rocm` and `moat-porting` skills.
3. Operate in auto mode within the boundary AGENTS.md draws ("Human decisions and external writes"). Stop only at the upstream-PR gate or a genuine blocker (record it with `python3 utils/moatlib.py set-blocked <name> <platform> '<reason>'`, ask a specific question).
4. When the role finishes a stage (state advances), re-run this skill to pick up the next stage or project. Honor the pipeline depth in config/moat.toml: at most one heavy build/test stage at a time per host; a planner may run ahead for the next project.

## Notes
- Forks live in the AMD-Ecosystem org. Never open or comment on an upstream PR without explicit approval.
- See AGENTS.md for the full pipeline and rules.
