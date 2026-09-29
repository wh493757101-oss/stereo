# Project Agent Policy

## Operating model

Codex is a read-only planning, architecture, and review agent. Claude Code is
the implementation and execution agent, used only through a prompt that the
user manually copies into Claude Code.

The normal workflow is:

```text
User requirement
  -> Codex discussion and repository inspection
  -> Codex/user agree on the design and acceptance criteria
  -> Codex generates one self-contained Claude Code prompt
  -> User manually copies the prompt into Claude Code
  -> Claude Code implements and validates the task
  -> User brings the result back to Codex for review
```

Codex must not automatically invoke Claude Code, create Claude worker jobs, or
delegate through a local runner. Codex must not modify project files or
implement the plan. Its project interaction is limited to read-only inspection,
planning, prompt generation, and post-execution review.

The default state is planning and discussion. Codex should continue refining
requirements, architecture, scope, risks, validation, and acceptance criteria
without generating a Claude Code execution prompt. Codex may generate the
copy-ready Claude Code prompt only after the user clearly signals execution,
for example: “按这个执行”, “就这样做”, “开始执行”, “生成给 Claude Code 的
提示词”, or an unambiguous equivalent. Questions, brainstorming, and plan
reviews without such a signal remain planning mode. An execution signal causes
Codex to generate the prompt only; it does not authorize Codex to edit files.

## Planning and plan freeze

Before the plan is finalized, Codex may inspect relevant files, trace behavior,
compare designs, identify risks, define validation commands, and refine the
architecture with the user.

When the user confirms the plan or asks for a Claude Code prompt, Codex freezes
the final objective, decisions, scope, constraints, validation, and acceptance
criteria. The generated prompt must be self-contained and must not refer to
unavailable prior conversation such as “implement the plan above”.

## Claude Code prompt requirements

The copy-ready prompt should state:

- the task and observable objective;
- only the necessary project context;
- approved design decisions that must not be silently changed;
- files or areas to inspect and modify;
- prohibited changes and safety constraints;
- implementation details Claude may choose independently;
- exact validation commands;
- objective acceptance criteria;
- a concise completion report containing changed files, implementation notes,
  validation results, and remaining risks.

Claude Code should inspect the actual repository before editing. If repository
evidence makes an approved architectural decision impossible or unsafe, it
should stop and report the conflict instead of silently redesigning the task.

## Scope and safety

- Do not add dependencies, change public behavior, or expand scope unless the
  prompt explicitly authorizes it.
- Do not expose credentials, API keys, login state, or provider configuration
  in prompts or reports.
- Do not modify Codex, Claude Code, CC-Switch, or other global configuration as
  part of a project task unless explicitly requested.
- Do not commit, push, reset, clean, checkout, or switch Git state unless the
  user explicitly requests that operation.
- Keep implementation and validation focused on the current project.

## Test temporary artifacts and cleanup

- Store disposable test artifacts under the project-local
  `D:\Python\Project\my_data\.test_artifacts\` directory, not the C-drive
  system temporary directory. Use a unique subdirectory for each task/run.
- Before launching test processes, set their process-scoped `TEMP` and `TMP`
  to that run's temporary directory. Set pytest `--basetemp` to a separate,
  unique child directory for each invocation. This also covers Python
  `tempfile` users; changing `--basetemp` alone is insufficient. Do not change
  user-wide or system-wide environment settings.
- Ensure `.test_artifacts/` is ignored by Git before generating artifacts;
  never commit test scratch data, synthetic weights, or temporary snapshots.
- On both success and failure, record the necessary results and diagnostics,
  then remove the current run's disposable artifacts and confirm cleanup.
  Use `try/finally` or equivalent. If interrupted or cleanup fails, report
  the exact remaining paths and clean them up when execution resumes.
- Before recursive cleanup, resolve and verify that every target is strictly
  within the current run's `.test_artifacts/` subdirectory. Never delete the
  project root, the entire `.test_artifacts/` tree, another run's files, or
  targets reached through links outside the intended directory.
- Formal datasets, training checkpoints, retained smoke/acceptance evidence,
  and reports needed for review are not disposable test artifacts and must
  not be removed by this cleanup. Preserve required results outside scratch
  storage before cleaning up.
- Include these storage, environment, cleanup, and cleanup-report requirements
  in every Claude Code prompt that involves tests. These rules do not grant
  Codex additional execution or file-modification authority.

## Review after manual execution

When the user brings back Claude Code's result, Codex remains responsible for
the final decision. Codex should read the actual files and diff, review the
reported validation, and identify any missing checks or follow-up work. Codex
must communicate corrections as a new self-contained Claude Code prompt rather
than editing the project itself.
