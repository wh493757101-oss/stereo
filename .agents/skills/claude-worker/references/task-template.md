# External worker task

Copy this template into a new task file under the skill's `runs/` directory. Replace every placeholder before invoking the worker.

## Objective

State one observable outcome.

## Allowed scope

- List files or directories the worker may inspect.
- List files the worker may modify, or write `none` for analysis.

## Requirements

- State required behavior and compatibility constraints.
- State relevant project conventions.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.

## Validation

- List the exact checks or tests to run.
- If a command is not explicitly allowed by the caller, report it instead of running it.

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- files changed
- validation commands and results
- unresolved risks or assumptions
