---
name: pr-review
description: Review a CUDA-to-ROCm/HIP port for correctness, minimal footprint, the AMD fault classes, build-system correctness, test adequacy, and not breaking upstream CUDA/CPU behavior. Use when reviewing a port branch, when asked to review code changes, or when the user mentions "review PR", "code review", or "check this port".
---

# MOAT PR Review Skill

Review a CUDA-to-ROCm/HIP port focusing on what CI cannot check: port correctness, minimal footprint, the AMD-strict-where-CUDA-lenient fault classes, build-system correctness, test adequacy, and whether upstream CUDA/CPU behavior is preserved.

## Usage Modes

### No Argument

If invoked with no arguments, do not review. Ask:

> What would you like me to review?
> - A local branch (e.g., `/pr-review branch`) -- the usual MOAT case (the fork branch)
> - A PR number or URL (e.g., `/pr-review 123`)

### Local Branch Mode (the usual MOAT case)

Review the port on the fork branch in `projects/<name>/src/` against its base:

```bash
git -C projects/<name>/src branch --show-current
git -C projects/<name>/src diff <base>...HEAD --stat
git -C projects/<name>/src diff <base>...HEAD
git -C projects/<name>/src log <base>..HEAD --oneline
```

The base is the upstream default branch the fork started from (`fork_default_branch` in `projects/<name>/status.json`). Use the branch name in the review header.

### PR Number / URL Mode

If a PR exists (an upstream PR or a fork PR), fetch it with `gh`:

```bash
gh pr view <PR> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits
gh pr view <PR> --json comments,reviews
gh pr diff <PR>
```

Fetch the comments and reviews, not just the diff. On a PR that has been round-tripped
with a maintainer, what they already asked for IS the review context, and repeating a
point they have answered wastes their patience rather than yours.

Add `detailed` for line-by-line specific comments.

### GitHub Actions Mode

If the prompt carries `<formatted_context>`, `<pr_or_issue_body>` or `<comments>` tags, you
were invoked from a GitHub workflow rather than a terminal. The PR body, comments and
review state are already in those tags -- do not re-fetch them, and do not reach for `gh`,
which is not available there. Use git alone:

```bash
git diff <base>...HEAD
git log <base>..HEAD
```

## Review Philosophy

A single line can have deep cross-cutting implications: a hardcoded `32` silently corrupts results on a wave64 GPU; a missing index clamp faults on AMD where CUDA tolerated it; a per-arch hack that fixes wave32 regresses wave64. Treat every line as potentially load-bearing.

1. Only report problems. The output contains only issues, concerns, and actionable suggestions. Do not praise, do not explain why something is fine, do not write "looks good". Omit empty sections.
2. Investigate, do not guess. When unsure whether a check applies, spawn a sub-agent to read the relevant code. A reviewer who guesses wrong provides negative value.
3. Review the strategy, not just the diff. A correct implementation of the wrong porting strategy is still wrong (compat-header where torch-hipify belongs, or vice versa; renaming files when marking LANGUAGE HIP would do).
4. Focus on what CI cannot check. Skip formatting and lint. Focus on port correctness, the fault classes, minimal footprint, build correctness, test adequacy, and upstream-behavior preservation.
5. Everything is a must-fix. No nits. If it is worth mentioning, it is worth fixing.
6. Be specific and actionable. Reference file:line. Name the function/file/idiom the author should use.
7. Match the project's existing style and the `cuda-to-rocm` strategy.
8. Assume competence. The author knows the project's domain; explain only non-obvious ROCm context.
9. No repetition. Each observation appears in exactly one section.

### Using sub-agents

The checklist is large and the fault classes need code reading. Spawn sub-agents to investigate whether checks apply (read the kernel, the CMake, the surrounding code, the tests that should exist). Spawn them in parallel for independent areas. A typical port should spawn 3-8 sub-agents.

## Review Workflow

### Step 1: Understand context
Read `projects/<name>/plan.md` (the intended strategy and risks), `notes.md`, and invoke the `cuda-to-rocm` skill. Group the diff by type (compat header, CMake, kernel fixes, tests, CI). Note scope.

### Step 2: Deep review
Go through every changed line against [review-checklist.md](review-checklist.md). For any
file changed substantially, read the surrounding UNCHANGED code too: a diff shows what
moved, not what it has to remain consistent with, and the fault classes are mostly about
interactions with code the diff never touches.

### Step 3: Upstream-behavior compatibility
Evaluate per [bc-guidelines.md](bc-guidelines.md): the port must not change the upstream project's existing CUDA or CPU behavior; ROCm support must be additive and guarded.

### Step 4: Formulate
Organize findings by the categories below. Every finding is traceable to a file:line.

### Step 5: Fact-check
Spawn a sub-agent per reported issue (in parallel) to independently verify by re-reading the code. Each returns valid / invalid / needs rewording. Drop invalid issues; reword the rest; keep low-confidence ones flagged as such.

## Output Format

Omit sections with no problems. Do not write affirmative commentary. The Summary is the one exception: one sentence on what the port does, then the verdict.

```markdown
## Port Review: <project> (<branch> vs <base>)

### Summary
What the port does (1 sentence), then the verdict.

### Port Correctness
[Problems only -- wrong strategy, mis-hipified symbols, broken kernels]

### Fault Classes
[Problems only -- warpSize/32, rule-of-five, OOB reads, texture pitch, lane masks, library swaps]

### Minimal Footprint
[Problems only -- host C++ touched needlessly, files renamed instead of LANGUAGE HIP, unguarded divergence, NVIDIA build broken]

### Build System
[Problems only -- enable_language(HIP), arch flags, USE_HIP gating, CUDA build intact]

### Testing
[Problems only -- missing GPU test run, non-GPU regression, CPU smoketest treated as the gate]

### Backward Compatibility (upstream)
[Problems only -- changed CUDA/CPU behavior, non-additive change]

### Commit Hygiene
[Problems only -- title not [ROCm]/over 72 chars, noreply trailer, ghstack, bare --force, AMD-internal account references]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

Missing a real-GPU test run, or a per-arch hack in shared code, always means **Request Changes**.
```

### Specific Comments (Detailed Review Only)
Only when the user requests a "detailed" review. File-specific feedback with line references, no repetition of the sections above.

## Files to Reference

Read these for context rather than relying on memory:
- `CLAUDE.md` -- standing rules, autonomy boundary, coding style
- the `cuda-to-rocm` skill -- porting strategies and fault classes the port must follow
- `projects/<name>/plan.md` -- the intended strategy, risks, and test plan for this project
- `projects/<name>/notes.md` -- project-specific gotchas and prior review/validation records
