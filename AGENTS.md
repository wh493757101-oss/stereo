# Project Agent Policy

## Operating model

Use Codex as the principal agent and Claude Code as an external execution agent through the project-local `$claude-worker` skill.

- Codex owns requirement interpretation, architecture, planning, task boundaries, risk decisions, and final acceptance.
- Claude Code performs bounded execution work such as repository exploration, local information gathering, implementation, test creation, test execution, and failure diagnosis.
- Treat Claude Code's report as evidence. Codex must inspect the actual result and make the final decision.

## Standing authorization

The user authorizes Codex to invoke `$claude-worker` automatically for matching tasks in this project without asking for confirmation before every invocation. This standing authorization permits use of the configured Claude Code and CC-Switch/TokenPlan path, but it does not expand the scope of the user's current request.

The user's current instruction always takes precedence. Do not invoke Claude Code when the user says to plan only, not to execute, not to use Claude, or not to spend external tokens. Briefly announce each real Claude invocation before starting it.

## Delegate to Claude Code when

Use `$claude-worker` by default for substantial execution work, including:

- searching or reading multiple project files to map behavior, dependencies, or call paths;
- collecting and summarizing information already available in the local project;
- implementing features, bug fixes, refactors, migrations, or documentation changes;
- writing or updating tests and running narrowly scoped validation commands;
- investigating build or test failures;
- performing a bounded implementation review when an independent execution pass is useful.

Codex may perform the minimum local inspection needed to understand the request, define a safe task, and verify the result. Codex should keep architectural choices, tradeoff decisions, prioritization, acceptance criteria, and final review for itself.

Do not delegate when the request can be answered from the conversation alone, when only a decision or plan is requested, or when delegation would add no meaningful execution value.

## Delegation procedure

For every invocation:

1. Codex first defines one observable outcome, allowed read and write scope, constraints, prohibited actions, and exact validation commands.
2. Codex creates the task file only under `.agents/skills/claude-worker/runs/` using the skill's task template.
3. Use `Analyze` for read-only exploration, local information collection, diagnosis, or review. Use `Implement` only when the user's current request authorizes project changes.
4. Grant only the narrowly scoped Bash command families required for validation. Never grant general `Bash` or `Bash(*)` access.
5. Run no more than one write-capable Claude worker at a time in this working directory.
6. After Claude returns, Codex inspects the actual files and diff when Git is available, then independently reruns the key acceptance checks.
7. If correction is needed, Codex may issue one bounded retry within the original scope. Otherwise, report the blocker instead of silently completing the work itself through another provider.
8. The final response distinguishes Claude's work from Codex's verification and lists remaining risks.

## Boundaries

- The current worker may inspect local project information but may not use web search or external networks.
- Do not expose credentials, environment dumps, login state, API keys, or TokenPlan configuration to task files or model output.
- Do not let Claude Code modify Codex, Claude Code, or CC-Switch global configuration.
- Do not let Claude Code install dependencies or commit, push, reset, clean, checkout, or switch Git state.
- Do not let Claude Code modify files outside the scope authorized by the user's current request.

