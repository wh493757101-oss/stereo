"""Tests for the archived sequential four-stage launcher
(scripts.legacy_four_stage.train_all_models). The active three-stage
pipeline is covered by tests/test_train_pipeline.py."""

from pathlib import Path
from types import SimpleNamespace

import scripts.legacy_four_stage.train_all_models as launcher


RUN_ID = "run_20260914_manual_labels"


def _fake_runner(return_codes, calls):
    codes = iter(return_codes)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=next(codes))

    return run


def test_runs_all_stages_in_order_as_separate_python_processes(tmp_path):
    calls = []
    result = launcher.run_stages(
        launcher.ALL_STAGES,
        run_id=RUN_ID,
        device="0",
        project=str(tmp_path),
        workers=0,
        runner=_fake_runner([0, 0, 0, 0], calls),
    )

    assert result == 0
    assert [call[0][call[0].index("--stage") + 1] for call in calls] == [
        "baseline",
        "a",
        "b-gray",
        "b-polar",
    ]
    for command, kwargs in calls:
        assert command[0] == launcher.sys.executable
        assert command[1] == str(launcher.TRAIN_SCRIPT)
        assert command[command.index("--run-id") + 1] == RUN_ID
        assert command[command.index("--device") + 1] == "0"
        assert command[command.index("--project") + 1] == str(tmp_path)
        assert kwargs == {"cwd": launcher.PROJECT_ROOT, "check": False}


def test_stops_immediately_when_one_stage_fails(tmp_path):
    calls = []
    result = launcher.run_stages(
        launcher.ALL_STAGES,
        run_id=RUN_ID,
        device="0",
        project=str(tmp_path),
        workers=0,
        runner=_fake_runner([0, 7], calls),
    )

    assert result == 7
    assert len(calls) == 2


def test_selected_stages_support_resuming_remaining_work(tmp_path):
    calls = []
    result = launcher.run_stages(
        ("b-gray", "b-polar"),
        run_id=RUN_ID,
        device="0",
        project=str(tmp_path),
        workers=0,
        runner=_fake_runner([0, 0], calls),
    )

    assert result == 0
    assert [call[0][call[0].index("--stage") + 1] for call in calls] == [
        "b-gray",
        "b-polar",
    ]


def test_existing_selected_model_directory_blocks_before_launch(tmp_path, capsys):
    existing = tmp_path / RUN_ID / "model_a"
    existing.mkdir(parents=True)
    calls = []

    result = launcher.main(
        [
            "--run-id",
            RUN_ID,
            "--project",
            str(tmp_path),
            "--stages",
            "a",
            "b-gray",
        ],
        runner=_fake_runner([], calls),
    )

    assert result == 2
    assert calls == []
    assert str(existing) in capsys.readouterr().err


def test_invalid_run_id_blocks_before_launch(tmp_path):
    calls = []
    result = launcher.main(
        ["--run-id", "../escape", "--project", str(tmp_path)],
        runner=_fake_runner([], calls),
    )

    assert result == 2
    assert calls == []


def test_duplicate_stage_is_rejected(tmp_path):
    calls = []
    result = launcher.main(
        [
            "--run-id",
            RUN_ID,
            "--project",
            str(tmp_path),
            "--stages",
            "a",
            "a",
        ],
        runner=_fake_runner([], calls),
    )

    assert result == 2
    assert calls == []

