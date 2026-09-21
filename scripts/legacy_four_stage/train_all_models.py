"""ARCHIVED: sequential four-stage launcher (archived 2026-09-21).

Not part of the current three-stage mainline (scripts/train_pipeline.py).
Kept explicitly runnable for historical reproduction only. See README.md
in this directory.

Each stage runs in its own Python process so GPU memory and Ultralytics state
are released between stages. Existing selected model directories are rejected
before any child process starts.

Usage (archived):
    python scripts/legacy_four_stage/train_all_models.py \
        --run-id run_20260914_manual_labels --device 0

Resume selected unfinished stages (archived):
    python scripts/legacy_four_stage/train_all_models.py \
        --run-id run_20260914_manual_labels --device 0 \
        --stages b-gray b-polar
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.legacy_four_stage.train_models as training

ALL_STAGES = ("baseline", "a", "b-gray", "b-polar")
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "legacy_four_stage" / "train_models.py"


class ExistingRunOutputError(RuntimeError):
    """Raised when a selected model directory already exists."""


def _model_directory(project: str, run_id: str, stage: str) -> Path:
    return Path(training.resolve_project(project)) / run_id / f"model_{stage}"


def _validate_stages(stages: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(stages)
    if not selected:
        raise ValueError("At least one training stage is required.")
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate training stages are not allowed.")
    unknown = [stage for stage in selected if stage not in ALL_STAGES]
    if unknown:
        raise ValueError(f"Unknown training stages: {unknown}")
    return selected


def _ensure_outputs_absent(
    stages: Sequence[str], *, run_id: str, project: str
) -> None:
    existing = [
        _model_directory(project, run_id, stage)
        for stage in stages
        if _model_directory(project, run_id, stage).exists()
    ]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise ExistingRunOutputError(
            "Selected model output already exists; choose only unfinished "
            f"stages or use a new run id: {paths}"
        )


def _stage_command(
    stage: str,
    *,
    run_id: str,
    device: str,
    project: str,
    workers: int,
    seed: int,
    patience: int,
) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--stage",
        stage,
        "--run-id",
        run_id,
        "--device",
        device,
        "--project",
        project,
        "--workers",
        str(workers),
        "--seed",
        str(seed),
        "--patience",
        str(patience),
    ]


def run_stages(
    stages: Sequence[str],
    *,
    run_id: str,
    device: str = "0",
    project: str = training.DEFAULT_PROJECT,
    workers: int = training.DEFAULT_WORKERS,
    seed: int = training.DEFAULT_SEED,
    patience: int = training.DEFAULT_PATIENCE,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    """Run selected stages sequentially, stopping at the first failure."""
    selected = _validate_stages(stages)
    training.validate_run_id(run_id)
    _ensure_outputs_absent(selected, run_id=run_id, project=project)

    print(f"Run id: {run_id}")
    print(f"Stages: {', '.join(selected)}")
    for index, stage in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] Starting stage: {stage}", flush=True)
        result = runner(
            _stage_command(
                stage,
                run_id=run_id,
                device=device,
                project=project,
                workers=workers,
                seed=seed,
                patience=patience,
            ),
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"ERROR: stage {stage} failed with exit code "
                f"{result.returncode}; remaining stages were not started.",
                file=sys.stderr,
            )
            return result.returncode if result.returncode > 0 else 1
        print(f"[{index}/{len(selected)}] Completed stage: {stage}", flush=True)

    print(f"\nAll selected stages completed under run id: {run_id}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=training.DEFAULT_PROJECT)
    parser.add_argument("--workers", type=int, default=training.DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=training.DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=training.DEFAULT_PATIENCE)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=list(ALL_STAGES),
        help="Stages to run in order; defaults to all four stages.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    args = parse_args(argv)
    try:
        return run_stages(
            args.stages,
            run_id=args.run_id,
            device=args.device,
            project=args.project,
            workers=args.workers,
            seed=args.seed,
            patience=args.patience,
            runner=runner,
        )
    except (training.InvalidRunIdError, ExistingRunOutputError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; remaining stages were not started.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
