---
name: claude-worker
description: Delegate a bounded project analysis or implementation task to the locally installed Claude Code CLI when the user explicitly asks to use Claude, TokenPlan, or the external executor, or when an applicable project AGENTS.md grants standing authorization. Keep planning and final review in Codex. Do not use without authorization for an external model call.
---

# Claude Worker

Use the bundled PowerShell executor to send a narrowly scoped task to Claude Code while Codex remains responsible for decisions and acceptance.

## Authorization boundary

- Invoke this workflow only after the user explicitly asks to use Claude, TokenPlan, or the external executor for the current task, or when an applicable project `AGENTS.md` explicitly grants standing authorization for the matching task category.
- A standing authorization never expands the user's current request. A plan-only or read-only request does not authorize implementation, and an explicit instruction not to use Claude overrides standing authorization.
- Creating or configuring this skill alone does not authorize a live Claude call; the authorization must come from the current user request or an applicable project policy written on the user's behalf.
- Briefly notify the user before each live invocation because it consumes an external model plan.
- Never expose API keys, login caches, environment dumps, or other secrets in a task file or response.
- Do not modify Codex, Claude Code, or CC-Switch global configuration as part of delegation.

## Workflow

1. Inspect the project and decide what outcome, file scope, constraints, and validation are needed.
2. Read [references/task-template.md](references/task-template.md), then create a concrete task file under this skill's `runs/` directory. Do not create delegation files elsewhere.
3. Choose a mode:
   - `Analyze` for read-only exploration, review, or planning.
   - `Implement` only when the user has authorized project changes.
4. Run `scripts/invoke-claude-worker.ps1` from the target project directory. Pass the task file, project working directory, mode, timeout, and only the narrowly scoped Bash test tools the task requires.
5. Parse the wrapper's JSON result. Stop and report when the status is `rejected`, `failed`, or `timed_out`; do not silently fall back to another provider.
6. Inspect the actual changed files. When Git is available, inspect `git status --short` and `git diff`. Independently rerun the key validation needed for acceptance.
7. Summarize the external worker's changes, Codex's verification, and any remaining risks. The worker's self-report is evidence, not proof.

Do not run more than one write-capable external worker at a time in the same working directory. Retry at most once, only when the failure is transient or the correction stays within the original scope.

## Executor examples

Read-only analysis:

```powershell
pwsh -NoProfile -File .agents\skills\claude-worker\scripts\invoke-claude-worker.ps1 `
  -TaskFile .agents\skills\claude-worker\runs\pending-analysis.md `
  -WorkingDirectory . `
  -Mode Analyze
```

Implementation with one test command family:

```powershell
pwsh -NoProfile -File .agents\skills\claude-worker\scripts\invoke-claude-worker.ps1 `
  -TaskFile .agents\skills\claude-worker\runs\pending-implementation.md `
  -WorkingDirectory . `
  -Mode Implement `
  -AllowedBashTools "Bash(python -m pytest *)"
```

Never pass `Bash(*)` or a general `Bash` allowance. The executor rejects unscoped or destructive Bash rules and always denies web tools plus Git commit, push, reset, clean, checkout, switch, and common deletion commands.
