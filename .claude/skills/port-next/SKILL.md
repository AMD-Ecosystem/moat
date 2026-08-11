---
name: port-next
description: Resume MOAT. Detect this host's AMD platform, find the next actionable project, and dispatch the right role. Use when starting work in the MOAT repo or when asked to continue porting or "port next".
---

# port-next

Run the MOAT pipeline for the next available project on this host.

## Steps
0. Be on the right branch. MOAT work happens on `port/<name>` in this repo, not on `main` -- orient refuses the trunk. An in-flight project's folder lives only on its own branch, so check out the branch for the project you intend to work (`git checkout port/<name>`), or create it when adopting one. Not to be confused with `moat-port`, which is the branch on the FORK that holds the code. If nothing is actionable here, `python3 utils/moatlib.py fleet <platform>` names work on other branches.
1. Run `bash utils/orient.sh`. It pulls the latest state, detects the AMD arch, and prints the next project, its state, and the stage (intake / planner / porter / reviewer / validator), or NONE.
2. If it names a project + stage, dispatch the named role adapter scoped to the project and platform: "Use the <stage> role on projects/<name> for <platform>". Claude Code discovers the roles in `.claude/agents/`; Codex discovers the matching adapters in `.codex/agents/`; both execute the canonical role body in `.claude/agents/<stage>.md`. Stages are intake, planner, porter, reviewer and validator. Nothing after approval is a stage: submitting upstream and the maintainer rounds that follow are the `moat-checkup` skill, run in this session, because each step ends at a person. The child agent reads AGENTS.md (Pipeline, Human decisions and external writes, Standing rules, Stop discipline), plan.md, notes.md, and the `cuda-to-rocm` skill.
3. Operate in auto mode within the Autonomy boundary. Stop only at the upstream-PR gate or a genuine blocker (set `blocked`, ask a specific question).
4. When the role finishes a stage (state advances), re-run this skill to pick up the next stage or project. Honor the pipeline depth in config/moat.toml: at most one heavy build/test stage at a time per host; a planner may run ahead for the next project.

## Notes
- Forks live in the AMD-Ecosystem org. Never open or comment on an upstream PR without explicit approval.
- See AGENTS.md for the full pipeline and rules.
