"""Tests for the run_material_experiments orchestrator (plan/preflight/run/collect).

Mode separation is the core contract: ``plan``, ``preflight`` and
``collect`` must never start training or create an optimizer, ``run``
records every attempt and stops on the first failure, and ``collect``
excludes smoke artifacts and never touches the test split.

Every test that touches run directories uses the ``isolated_runs`` fixture,
which redirects the orchestrator's run root and task matrix into a scratch
directory. No test may read or write the real ``runs/train`` matrix.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .conftest import (
    ARTIFACTS_ROOT,
    ROOT,
    assert_owned,
    build_test_protocol_spec,
    build_v4_dataset,
    fabricate_completed_run,
    install_test_protocol,
    material_run_dir,
    new_scratch,
    remove_owned_tree,
    session_artifacts_root,
    train_run_root,
)

from scripts import run_material_experiments as orchestrator
from scripts.run_material_experiments import (
    EXPECTED_PROTOCOL,
    V1_SPEC,
    ExperimentTask,
    build_tasks,
    check_device_available,
    cmd_collect,
    cmd_plan,
    cmd_preflight,
    cmd_run,
    detailed_metrics,
    parse_args,
    parse_task_list,
    paired_flip_stats,
    bootstrap_macro_f1_delta,
    protocol_check,
    resolve_device,
    task_subset,
    validate_completed_task,
)

SEED_2026_TASKS = [
    "gray_s2026", "gray_gray_valid_s2026", "gray_valid_s2026", "gray_polar_s2026",
]
SEED_2027_TASKS = [
    "gray_s2027", "gray_gray_valid_s2027", "gray_valid_s2027", "gray_polar_s2027",
]
SEED_2028_TASKS = [
    "gray_s2028", "gray_gray_valid_s2028", "gray_valid_s2028", "gray_polar_s2028",
]
SEED_TASK_GROUPS = (SEED_2026_TASKS, SEED_2027_TASKS, SEED_2028_TASKS)


def _dataset_fingerprint(data_root: Path) -> str:
    from scripts.train_polar_fusion import validate_dataset

    return validate_dataset(data_root)["audit_fingerprint"]


def _matching_v1_spec(scratch_dir: Path, data_root: Path):
    """A v1 protocol spec whose identity is this fixture's dataset.

    The production validator binds a run's dataset and pretrained identity to
    the *chosen* protocol, so a synthetic fixture is validated against a spec
    built for it (``conftest.build_test_protocol_spec``) rather than against a
    relaxed rule set. Production defaults are untouched.
    """
    from scripts.run_material_experiments import V1_SPEC

    return build_test_protocol_spec(
        scratch_dir, V1_SPEC,
        data_fingerprint=_dataset_fingerprint(data_root),
        pretrained_sha256="a" * 64,
    )


def _manifest(scratch_dir: Path) -> dict:
    return json.loads((scratch_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _attempts(manifest: dict, task_id: str) -> list[dict]:
    return manifest["tasks"][task_id]["attempts"]


# ---------------------------------------------------------------------------
# Task matrix


def test_matrix_exactly_12_unique_tasks():
    tasks = build_tasks()
    assert len(tasks) == 12
    assert len({task.task_id for task in tasks}) == 12
    assert len({task.run_id for task in tasks}) == 12
    variants = {task.variant for task in tasks}
    assert variants == {"gray", "gray_gray_valid", "gray_valid", "gray_polar"}
    seeds = {task.seed for task in tasks}
    assert seeds == {2026, 2027, 2028}
    for variant in variants:
        for seed in seeds:
            task_id = f"{variant}_s{seed}"
            assert any(task.task_id == task_id for task in tasks), task_id


def test_matrix_seed_major_order_puts_seed_2026_first():
    tasks = build_tasks()
    assert [task.task_id for task in tasks[:4]] == SEED_2026_TASKS


def test_matrix_rejects_duplicates_and_unknown_variants():
    with pytest.raises(ValueError, match="duplicate matrix entry"):
        build_tasks(variants=["gray", "gray"])
    with pytest.raises(ValueError, match="unknown material-fusion variant"):
        build_tasks(variants=["gray_polar_rgb"])


def test_matrix_run_dirs_follow_the_configured_run_root(isolated_runs):
    for task in isolated_runs.tasks:
        assert Path(task.run_dir).is_relative_to(isolated_runs.run_root)
        assert task.run_id.startswith("cmx_v1_test_")


def test_task_subset_selects_and_rejects():
    tasks = build_tasks()
    selected = task_subset(tasks, ["gray_polar_s2026", "gray_s2026"])
    assert [task.task_id for task in selected] == ["gray_polar_s2026", "gray_s2026"]
    with pytest.raises(ValueError, match="unknown task"):
        task_subset(tasks, ["gray_s2026", "bogus_task"])


def test_delivered_task_commands_select_exactly_four_tasks_per_seed():
    """The three documented ``--tasks`` commands resolve to their seed's 4 tasks.

    Parsing and selection only — nothing here may start a training process.
    """
    tasks = build_tasks()
    for group in SEED_TASK_GROUPS:
        args = parse_args(["run", "--tasks", ",".join(group), "--device", "cpu"])
        assert args.tasks == ",".join(group)
        selected = task_subset(tasks, parse_task_list(args.tasks))
        assert [task.task_id for task in selected] == group
        assert len(selected) == 4
        seed = selected[0].seed
        assert {task.seed for task in selected} == {seed}
        assert len({task.variant for task in selected}) == 4


def test_task_selection_never_starts_a_subprocess(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("task selection must never start a subprocess")

    monkeypatch.setattr("scripts.run_material_experiments.subprocess.run", _fail)
    tasks = build_tasks()
    for group in SEED_TASK_GROUPS:
        args = parse_args(["run", "--tasks", ",".join(group), "--device", "cpu"])
        assert len(task_subset(tasks, parse_task_list(args.tasks))) == 4


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],                                              # --tasks is required
        ["run", "--tasks", "gray_s2026", "--seed", "2026"],    # no --seed on run
        ["run", "--seed-batch", "2026"],                       # no --seed-batch
    ],
)
def test_run_parser_rejects_unsupported_arguments(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_empty_or_colon_task_lists_are_refused_before_any_run():
    tasks = build_tasks()
    args = parse_args(["run", "--tasks", "  ,  ", "--device", "cpu"])
    with pytest.raises(ValueError, match="at least one task id"):
        task_subset(tasks, parse_task_list(args.tasks))
    # the colon form is not a task id; it must be refused, not silently matched
    with pytest.raises(ValueError, match="unknown task"):
        task_subset(tasks, ["gray:2026"])
    with pytest.raises(ValueError, match="unknown task"):
        task_subset(tasks, ["gray_polar:2026"])


# ---------------------------------------------------------------------------
# plan / preflight never train


def test_plan_writes_matrix_and_never_trains(scratch_dir, isolated_runs, monkeypatch):
    monkeypatch.setattr("scripts.run_material_experiments.PLAN_PATH",
                        scratch_dir / "plan.json")

    def _fail(*_args, **_kwargs):
        raise AssertionError("plan mode must never start a subprocess")

    monkeypatch.setattr("scripts.run_material_experiments.subprocess.run", _fail)
    exit_code = cmd_plan(SimpleNamespace(device="cpu"))
    assert exit_code == 0
    plan = json.loads((scratch_dir / "plan.json").read_text(encoding="utf-8"))
    assert plan["matrix"]["tasks"] == 12
    assert len(plan["tasks"]) == 12
    for entry in plan["tasks"]:
        assert "train_material_fusion.py" in entry["command"]
        assert f"--seed {entry['seed']}" in entry["command"]
    basis = plan["resource_basis"]
    assert basis["epochs"] == 30
    assert basis["effective_batch"] == 16
    assert basis["optimizer_steps_per_run"] > 0
    assert plan["auxiliary_tasks"]["note"]


def test_preflight_fails_when_the_config_pins_no_pretrained_sha(
    scratch_dir, isolated_runs, monkeypatch
):
    """An unset config SHA must fail the v1 gate and be logged as unset.

    The spec keeps the protocol's real pretrained SHA, so the previous
    ``or``-backfill would have hidden the empty config value and passed.
    """
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    spec = build_test_protocol_spec(
        scratch_dir, V1_SPEC,
        data_fingerprint=_dataset_fingerprint(data_root),
        pretrained_sha256="", data=str(data_root), label="empty_sha",
    )
    spec = dataclasses.replace(spec, pretrained_sha256=V1_SPEC.pretrained_sha256)
    install_test_protocol(monkeypatch, spec)
    output = scratch_dir / "preflight.json"
    code = cmd_preflight(
        SimpleNamespace(device="cpu", protocol="v1", output=str(output))
    )
    assert code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    checks = {check["check"]: check for check in report["checks"]}
    assert checks["pretrained_weight"]["ok"] is False
    assert "config=''" in checks["pretrained_weight"]["detail"]
    assert report["all_ok"] is False


def test_protocol_check_reports_every_difference():
    config = dict(EXPECTED_PROTOCOL)
    config["epochs"] = 10
    ok, differences = protocol_check(config)
    assert not ok
    assert any("epochs" in diff for diff in differences)
    ok, differences = protocol_check(dict(EXPECTED_PROTOCOL))
    assert ok and differences == []


def test_preflight_read_only_checks(scratch_dir, isolated_runs, monkeypatch):
    monkeypatch.setattr("scripts.run_material_experiments.EXPERIMENTS_DIR", scratch_dir)
    monkeypatch.setattr(
        "core.material_fusion_dataset.admit_material_dataset",
        lambda root, expected="": {
            "data_root": str(root), "audit_fingerprint": "f" * 64,
            "verified_files": 1, "class_names": ["metal_submarine"], "splits": {},
        },
    )
    exit_code = cmd_preflight(SimpleNamespace(device="cpu"))
    assert exit_code == 0
    report = json.loads((scratch_dir / "preflight.json").read_text(encoding="utf-8"))
    assert report["all_ok"] is True
    assert report["device"] == {
        "requested": "cpu", "resolved": "cpu", "available": True,
        "detail": report["device"]["detail"],
    }
    checks = {check["check"]: check for check in report["checks"]}
    assert checks["protocol_matches_fixed_constants"]["ok"] is True
    assert checks["dataset_admitted"]["ok"] is True
    assert checks["pretrained_weight"]["ok"] is True
    assert checks["output_dir_state"]["ok"] is True


@pytest.fixture()
def fake_cuda(monkeypatch):
    """A deterministic 2-device CUDA runtime, independent of this machine."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda index: f"Fake GPU {index}")


def test_device_check_agrees_with_the_execution_resolution(fake_cuda):
    """The gate must accept exactly what the execution path normalizes."""
    for requested in ("cpu", "0", "1", "cuda", "cuda:0", "cuda:1"):
        ok, detail = check_device_available(requested)
        assert ok is True, (requested, detail)
        assert resolve_device(requested)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("0") == "cuda:0"
    assert resolve_device("cuda") == "cuda:0"
    assert resolve_device("cuda:1") == "cuda:1"
    assert check_device_available("cuda:1")[1].startswith("cuda:1 ")


@pytest.mark.parametrize(
    "requested",
    ["2", "cuda:2", "-1", "cuda:-1", "gpu", "", "   ", "cuda:", "cuda:x", "bogus", "0.5"],
)
def test_device_check_refuses_out_of_range_and_unsupported_devices(fake_cuda, requested):
    ok, detail = check_device_available(requested)
    assert ok is False, requested
    assert detail and requested in detail


def test_device_check_refuses_cuda_when_the_runtime_has_none(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    for requested in ("0", "cuda", "cuda:0"):
        ok, detail = check_device_available(requested)
        assert ok is False and "CUDA is not available" in detail
    assert check_device_available("cpu")[0] is True


def test_device_check_never_raises(monkeypatch):
    for available in (True, False):
        monkeypatch.setattr("torch.cuda.is_available", lambda: available)
        monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
        for requested in ("cpu", "0", "cuda:0", "gpu", "-1", "", "cuda:x", "cuda:"):
            ok, detail = check_device_available(requested)
            assert isinstance(ok, bool) and isinstance(detail, str)


def test_preflight_fails_when_the_requested_device_is_unavailable(
    scratch_dir, isolated_runs, monkeypatch
):
    monkeypatch.setattr("scripts.run_material_experiments.EXPERIMENTS_DIR", scratch_dir)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "core.material_fusion_dataset.admit_material_dataset",
        lambda root, expected="": {
            "data_root": str(root), "audit_fingerprint": "f" * 64,
            "verified_files": 1, "class_names": ["metal_submarine"], "splits": {},
        },
    )
    assert cmd_preflight(SimpleNamespace(device="0")) == 1
    report = json.loads((scratch_dir / "preflight.json").read_text(encoding="utf-8"))
    checks = {check["check"]: check for check in report["checks"]}
    assert checks["device_available"]["ok"] is False
    assert report["device"]["requested"] == "0"
    assert report["device"]["resolved"] == "cuda:0"
    assert report["device"]["available"] is False


def test_preflight_reports_an_unsupported_device_without_raising(
    scratch_dir, isolated_runs, monkeypatch
):
    monkeypatch.setattr("scripts.run_material_experiments.EXPERIMENTS_DIR", scratch_dir)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "core.material_fusion_dataset.admit_material_dataset",
        lambda root, expected="": {
            "data_root": str(root), "audit_fingerprint": "f" * 64,
            "verified_files": 1, "class_names": ["metal_submarine"], "splits": {},
        },
    )
    assert cmd_preflight(SimpleNamespace(device="gpu")) == 1
    report = json.loads((scratch_dir / "preflight.json").read_text(encoding="utf-8"))
    assert report["device"] == {
        "requested": "gpu", "resolved": "", "available": False,
        "detail": report["device"]["detail"],
    }
    checks = {check["check"]: check for check in report["checks"]}
    assert checks["device_available"]["ok"] is False


# ---------------------------------------------------------------------------
# completion validation


def test_validate_completed_task_requires_full_artifacts(scratch_dir, isolated_runs):
    task = isolated_runs.by_task["gray_s2026"]
    ok, reason = validate_completed_task(task)
    assert not ok and "does not exist" in reason

    Path(task.run_dir).mkdir(parents=True, exist_ok=False)
    ok, reason = validate_completed_task(task)
    assert not ok and "missing" in reason
    shutil.rmtree(task.run_dir)  # scratch-owned; fabrication needs a free path

    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(task, data_root, _dataset_fingerprint(data_root))
    ok, reason = validate_completed_task(task, _matching_v1_spec(scratch_dir, data_root))
    assert ok, reason


def test_validate_completed_task_refuses_smoke(scratch_dir, isolated_runs):
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(
        task, data_root, _dataset_fingerprint(data_root), smoke=True, limit_batches=2,
    )
    ok, reason = validate_completed_task(task)
    assert not ok and "smoke" in reason


def test_validate_completed_task_refuses_wrong_variant_or_seed(scratch_dir, isolated_runs):
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(
        task, data_root, _dataset_fingerprint(data_root), seed=999,
    )
    ok, reason = validate_completed_task(task)
    assert not ok and "seed" in reason


def test_validate_completed_task_refuses_protocol_drift(scratch_dir, isolated_runs):
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(
        task, data_root, _dataset_fingerprint(data_root),
        config_overrides={"weight_decay": 0.02},
    )
    ok, reason = validate_completed_task(task)
    assert not ok and "fixed protocol" in reason and "weight_decay" in reason


def test_validate_completed_task_refuses_checkpoint_config_drift(scratch_dir, isolated_runs):
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(
        task, data_root, _dataset_fingerprint(data_root),
        checkpoint_fingerprint="b" * 64,
    )
    ok, reason = validate_completed_task(task)
    assert not ok and "data_fingerprint" in reason


def test_validate_completed_task_refuses_a_best_epoch_that_is_not_the_selection(
    scratch_dir, isolated_runs
):
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(
        task, data_root, _dataset_fingerprint(data_root), best_epoch=5,
    )
    ok, reason = validate_completed_task(task, _matching_v1_spec(scratch_dir, data_root))
    assert not ok and "the history selects epoch" in reason


def test_validate_completed_task_refuses_a_truncated_history(scratch_dir, isolated_runs):
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(
        task, data_root, _dataset_fingerprint(data_root), history_epochs=5,
    )
    ok, reason = validate_completed_task(task, _matching_v1_spec(scratch_dir, data_root))
    assert not ok and "expected one per epoch" in reason


# ---------------------------------------------------------------------------
# run mode


def _stub_subprocess(monkeypatch, returncode):
    calls = []

    def _run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("scripts.run_material_experiments.subprocess.run", _run)
    return calls


def test_run_refuses_an_unusable_device_before_spawning(
    scratch_dir, isolated_runs, monkeypatch
):
    """An unusable device must fail once, not as twelve failed attempts."""
    monkeypatch.setattr("scripts.run_material_experiments.RUN_MANIFEST_PATH",
                        scratch_dir / "run_manifest.json")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    calls = _stub_subprocess(monkeypatch, returncode=0)
    args = SimpleNamespace(tasks="gray_s2026,gray_polar_s2026", device="0",
                           skip_completed=False)
    assert cmd_run(args) == 2
    assert calls == []
    assert not (scratch_dir / "run_manifest.json").exists()


def test_run_stops_on_first_failure_and_records_attempt(scratch_dir, isolated_runs,
                                                        monkeypatch):
    monkeypatch.setattr("scripts.run_material_experiments.RUN_MANIFEST_PATH",
                        scratch_dir / "run_manifest.json")
    _stub_subprocess(monkeypatch, returncode=1)
    args = SimpleNamespace(tasks="gray_s2026,gray_polar_s2026", device="cpu",
                           skip_completed=False)
    exit_code = cmd_run(args)
    assert exit_code == 1
    manifest = _manifest(scratch_dir)
    attempts = _attempts(manifest, "gray_s2026")
    assert len(attempts) == 1
    failed = attempts[0]
    assert failed["status"] == "failed"
    assert failed["exit_code"] == 1
    assert failed["started_at"] and failed["ended_at"]
    assert failed["command"][1].endswith("train_material_fusion.py")
    assert "gray_polar_s2026" not in manifest["tasks"]


def test_run_refuses_existing_directory_without_skip(scratch_dir, isolated_runs,
                                                     monkeypatch):
    monkeypatch.setattr("scripts.run_material_experiments.RUN_MANIFEST_PATH",
                        scratch_dir / "run_manifest.json")
    calls = _stub_subprocess(monkeypatch, returncode=0)
    task = isolated_runs.by_task["gray_s2026"]
    Path(task.run_dir).mkdir(parents=True)
    args = SimpleNamespace(tasks=f"{task.task_id},gray_polar_s2026", device="cpu",
                           skip_completed=False)
    exit_code = cmd_run(args)
    assert exit_code == 2
    assert calls == []  # nothing was ever started
    manifest = _manifest(scratch_dir)
    assert _attempts(manifest, task.task_id)[0]["status"] == "refused_existing"
    assert "gray_polar_s2026" not in manifest["tasks"]


def test_run_skip_completed_requires_full_validation(scratch_dir, isolated_runs,
                                                     monkeypatch):
    monkeypatch.setattr("scripts.run_material_experiments.RUN_MANIFEST_PATH",
                        scratch_dir / "run_manifest.json")
    calls = _stub_subprocess(monkeypatch, returncode=0)
    task = isolated_runs.by_task["gray_s2026"]
    Path(task.run_dir).mkdir(parents=True)  # exists but empty -> not skippable
    args = SimpleNamespace(tasks=task.task_id, device="cpu", skip_completed=True)
    assert cmd_run(args) == 2
    assert calls == []
    manifest = _manifest(scratch_dir)
    assert _attempts(manifest, task.task_id)[0]["status"] == "refused_existing"


def test_run_skip_completed_skips_only_validated_runs_and_validates_after_run(
    scratch_dir, isolated_runs, monkeypatch
):
    monkeypatch.setattr("scripts.run_material_experiments.RUN_MANIFEST_PATH",
                        scratch_dir / "run_manifest.json")
    monkeypatch.setattr(orchestrator, "run_admission",
                        lambda spec, device: (True, "ok"))
    calls = _stub_subprocess(monkeypatch, returncode=0)
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(task, data_root, _dataset_fingerprint(data_root))
    install_test_protocol(monkeypatch, _matching_v1_spec(scratch_dir, data_root))
    # Second task "runs" successfully per the stub but produces no artifacts,
    # so post-run validation must mark it failed instead of completed.
    args = SimpleNamespace(tasks=f"{task.task_id},gray_polar_s2026", device="cpu",
                           skip_completed=True)
    exit_code = cmd_run(args)
    assert exit_code == 1
    manifest = _manifest(scratch_dir)
    assert _attempts(manifest, task.task_id)[0]["status"] == "skipped_completed"
    second = _attempts(manifest, "gray_polar_s2026")[0]
    assert second["status"] == "failed"
    assert "does not exist" in second["detail"]
    assert len(calls) == 1  # only the second task reached the subprocess


def test_run_manifest_appends_attempts_and_preserves_failed_history(
    scratch_dir, isolated_runs, monkeypatch
):
    monkeypatch.setattr("scripts.run_material_experiments.RUN_MANIFEST_PATH",
                        scratch_dir / "run_manifest.json")
    task = isolated_runs.by_task["gray_s2026"]
    _stub_subprocess(monkeypatch, returncode=1)
    args = SimpleNamespace(tasks=task.task_id, device="cpu", skip_completed=False)
    assert cmd_run(args) == 1

    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(task, data_root, _dataset_fingerprint(data_root))
    install_test_protocol(monkeypatch, _matching_v1_spec(scratch_dir, data_root))
    monkeypatch.setattr(orchestrator, "run_admission",
                        lambda spec, device: (True, "ok"))
    _stub_subprocess(monkeypatch, returncode=0)
    assert cmd_run(SimpleNamespace(tasks=task.task_id, device="cpu",
                                   skip_completed=True)) == 0

    attempts = _attempts(_manifest(scratch_dir), task.task_id)
    assert [attempt["status"] for attempt in attempts] == ["failed", "skipped_completed"]
    assert [attempt["attempt"] for attempt in attempts] == [1, 2]
    # the earlier failure survives the later pass instead of being overwritten
    assert attempts[0]["exit_code"] == 1
    assert attempts[0]["started_at"] and attempts[0]["ended_at"]


# ---------------------------------------------------------------------------
# collect mode


class _StubGray(torch.nn.Module):
    """Deterministic stand-in for the historical gray Model B."""

    def forward(self, rgb):
        mean = rgb.mean(dim=(2, 3))
        return torch.cat([mean, mean, mean, mean], dim=1)


def _fabricate_seed_2026(scratch_dir, isolated_runs, seed: int = 0):
    data_root = build_v4_dataset(
        scratch_dir / "v4", counts=(8, 4, 4), seed=seed,
        invalid_indices={"val": {1}},
    )
    fingerprint = _dataset_fingerprint(data_root)
    for task in isolated_runs.tasks:
        if task.seed == 2026:
            fabricate_completed_run(task, data_root, fingerprint)
    return data_root


def test_collect_aggregates_without_optimizer_or_training(
    scratch_dir, isolated_runs, monkeypatch
):
    data_root = _fabricate_seed_2026(scratch_dir, isolated_runs)
    install_test_protocol(monkeypatch, _matching_v1_spec(scratch_dir, data_root))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    for name in ("Adam", "AdamW", "SGD"):

        def _fail(*_args, _name=name, **_kwargs):
            raise AssertionError(f"collect must never create torch.optim.{_name}")

        monkeypatch.setattr(f"torch.optim.{name}", _fail)
    output = scratch_dir / "collect_out"
    args = SimpleNamespace(device="cpu", batch=4, data=str(data_root),
                           output=str(output))
    exit_code = cmd_collect(args)
    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))

    assert report["split"] == "val"
    assert report["test_split_used"] is False
    assert set(report["tasks"]) == set(SEED_2026_TASKS)
    assert len(report["excluded"]) == 8
    for task_id, entry in report["tasks"].items():
        assert entry["pure"]["samples"] == 4
        assert entry["param_count"] > 0
        assert entry["elapsed_seconds"] == 12.5
        assert entry["data_fingerprint"] == report["data_fingerprint"]
    gray_entry = report["tasks"]["gray_s2026"]
    assert gray_entry["routed"] is None
    dual_entry = report["tasks"]["gray_polar_s2026"]
    assert dual_entry["routed"] is not None
    assert dual_entry["routed"]["samples"] == 4
    assert "invalid_subset_pure" in dual_entry
    assert dual_entry["invalid_subset_pure"]["samples"] == 1

    assert report["n_val_groups"] == 4
    assert "4 capture groups" in report["group_uncertainty_note"]
    assert "seed_2026_gray_polar_vs_gray" in report["paired_flips"]
    assert "seed_2026_gray_polar_vs_gray" in report["cluster_bootstrap"]
    for seed in (2027, 2028):
        assert report["paired_flips"][f"seed_{seed}"]["status"] == "incomplete"
    for variant in report["per_variant_seeds"]:
        assert "mean_std" not in report["per_variant_seeds"][variant]
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "no test used" in summary


def test_collect_excludes_smoke_artifacts(scratch_dir, isolated_runs, monkeypatch):
    data_root = _fabricate_seed_2026(scratch_dir, isolated_runs)
    install_test_protocol(monkeypatch, _matching_v1_spec(scratch_dir, data_root))
    smoke_task = isolated_runs.by_task["gray_s2027"]
    fabricate_completed_run(
        smoke_task, data_root, _dataset_fingerprint(data_root),
        smoke=True, limit_batches=2,
    )
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "collect_out"
    args = SimpleNamespace(device="cpu", batch=4, data=str(data_root),
                           output=str(output))
    assert cmd_collect(args) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert smoke_task.task_id not in report["tasks"]
    assert "smoke" in report["excluded"][smoke_task.task_id]
    # a smoke artifact must never leak into the aggregated seed results
    for variant, seeds_result in report["per_variant_seeds"].items():
        assert "s2027" not in seeds_result


def test_collect_refuses_evaluation_data_that_is_not_the_training_dataset(
    scratch_dir, isolated_runs, monkeypatch
):
    """`--data` relocates the eval data; it must still be the pinned dataset."""
    train_data = build_v4_dataset(scratch_dir / "v4_train", counts=(8, 4, 4), seed=1)
    other_data = build_v4_dataset(scratch_dir / "v4_other", counts=(8, 4, 4), seed=2)
    train_fingerprint = _dataset_fingerprint(train_data)
    assert train_fingerprint != _dataset_fingerprint(other_data)

    task = isolated_runs.by_task["gray_s2026"]
    fabricate_completed_run(task, train_data, train_fingerprint)
    install_test_protocol(monkeypatch, _matching_v1_spec(scratch_dir, train_data))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "collect_out"
    args = SimpleNamespace(device="cpu", batch=4, data=str(other_data),
                           output=str(output))
    # the relocation is refused outright: a different dataset state never
    # becomes this protocol's collection, and no report is written
    assert cmd_collect(args) == 2
    assert not output.exists()


def test_collect_refuses_existing_output(scratch_dir, isolated_runs):
    data_root = _fabricate_seed_2026(scratch_dir, isolated_runs)
    output = scratch_dir / "collect_out"
    output.mkdir()
    args = SimpleNamespace(device="cpu", batch=4, data=str(data_root),
                           output=str(output))
    assert cmd_collect(args) == 2


def test_collect_refuses_an_unusable_device(scratch_dir, isolated_runs, monkeypatch):
    data_root = _fabricate_seed_2026(scratch_dir, isolated_runs)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    output = scratch_dir / "collect_out"
    args = SimpleNamespace(device="0", batch=4, data=str(data_root),
                           output=str(output))
    assert cmd_collect(args) == 2
    assert not output.exists()


# ---------------------------------------------------------------------------
# output ownership and cleanup safety
#
# Real, protected locations. They are only ever passed *as arguments* to the
# guards below — never created, never deleted, never planted with a canary.


def _protected_targets() -> dict[str, Path]:
    import tempfile

    return {
        "a formal run id": ROOT / "runs" / "train" / "cmx_v1_gray_s2026",
        "a formal run id in another case": ROOT / "runs" / "train" / "CMX_V1_GRAY_S2026",
        "a dotdot alias into a formal run": (
            ROOT / "runs" / "train" / "decoy" / ".." / "cmx_v1_gray_s2026"
        ),
        "a non-cmx formal run": ROOT / "runs" / "train" / "yolo26_formal_001",
        "an unregistered run directory": ROOT / "runs" / "train" / "not_registered_here",
        "the real runs root": ROOT / "runs" / "train",
        "the project root": ROOT,
        "an absolute path outside the project": Path(tempfile.gettempdir()) / "guard_probe",
        "another artifacts run": ARTIFACTS_ROOT / "19700101T000000Z_0" / "artifacts",
    }


@pytest.mark.parametrize("label", sorted(_protected_targets()))
def test_ownership_guard_refuses_every_foreign_path(label, scratch_dir):
    target = _protected_targets()[label]
    with pytest.raises(AssertionError):
        assert_owned(target, scratch_dir)
    with pytest.raises(AssertionError):
        remove_owned_tree(target, scratch_dir, missing_ok=True)


def test_ownership_guard_refuses_the_owned_root_itself(scratch_dir):
    with pytest.raises(AssertionError, match="not strictly inside"):
        assert_owned(scratch_dir, scratch_dir)
    with pytest.raises(AssertionError, match="not strictly inside"):
        remove_owned_tree(scratch_dir, scratch_dir, missing_ok=True)
    assert scratch_dir.is_dir()  # the refusal must not have removed anything


def test_ownership_guard_refuses_another_tests_scratch(artifacts, scratch_dir):
    """A sibling test's directory is inside the run root but not this test's."""
    other = new_scratch(artifacts, "tests")
    try:
        with pytest.raises(AssertionError, match="not strictly inside"):
            remove_owned_tree(other, scratch_dir, missing_ok=True)
        assert other.is_dir()
    finally:
        remove_owned_tree(other, artifacts, missing_ok=True)


def test_cleanup_removes_only_the_owned_subtree(artifacts, scratch_dir):
    nested = scratch_dir / "runs" / "train" / "cmx_v1_test_gray_s2026" / "material_fusion"
    nested.mkdir(parents=True)
    (nested / "best.pt").write_bytes(b"scratch")
    sibling = new_scratch(artifacts, "tests")
    try:
        remove_owned_tree(nested, scratch_dir)
        assert not nested.exists()
        assert scratch_dir.is_dir() and sibling.is_dir()
    finally:
        remove_owned_tree(sibling, artifacts, missing_ok=True)


def test_cleanup_reports_a_missing_target_instead_of_succeeding_silently(scratch_dir):
    with pytest.raises(FileNotFoundError, match="nothing to remove"):
        remove_owned_tree(scratch_dir / "absent", scratch_dir)
    # explicit opt-in keeps the "already gone" case quiet
    remove_owned_tree(scratch_dir / "absent", scratch_dir, missing_ok=True)


def test_cleanup_failure_reports_the_target_and_the_error(scratch_dir, monkeypatch):
    from . import conftest as conftest_module

    target = scratch_dir / "tree"
    target.mkdir()
    (target / "file").write_bytes(b"x")

    def _boom(path):
        raise OSError(13, "Access is denied")

    monkeypatch.setattr(conftest_module.shutil, "rmtree", _boom)
    with pytest.raises(RuntimeError) as excinfo:
        remove_owned_tree(target, scratch_dir)
    message = str(excinfo.value)
    assert str(target) in message and "Access is denied" in message
    assert target.is_dir()  # nothing was actually removed


def _make_dir_link(link: Path, target: Path) -> str:
    """Create a directory link (symlink, else junction); '' if unsupported.

    Windows only grants ``os.symlink`` with the symlink privilege, but a
    junction needs no privilege — and ``Path.is_symlink`` reports False for
    a junction, which is exactly why the guard cannot rely on it.
    """
    try:
        import os

        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return "junction"
    except (ImportError, OSError, AttributeError):
        return ""


def _remove_link(link: Path) -> None:
    """Remove a directory link itself, never the target it points at.

    ``remove_owned_tree`` deliberately refuses a tree containing a link, so
    the link tests must undo their own link before the scratch teardown.
    """
    import os

    if not (link.exists() or link.is_symlink()):
        return
    try:
        os.rmdir(link)
    except OSError:
        os.unlink(link)


def test_link_inside_the_artifacts_tree_is_detected(artifacts, scratch_dir):
    """The detector must catch a junction, which ``is_symlink`` misses."""
    from .conftest import is_link_or_reparse_point

    outside = new_scratch(artifacts, "outside")
    link = scratch_dir / "link"
    kind = _make_dir_link(link, outside)
    if not kind:
        remove_owned_tree(outside, artifacts, missing_ok=True)
        pytest.skip("this session cannot create a directory link")
    try:
        assert is_link_or_reparse_point(link) is True
        assert link.is_symlink() is (kind == "symlink")
    finally:
        _remove_link(link)
        remove_owned_tree(outside, artifacts, missing_ok=True)


def test_cleanup_refuses_a_link_that_escapes_the_owned_root(artifacts, scratch_dir):
    """A link out of the run tree must be refused, and its target untouched."""
    outside = new_scratch(artifacts, "outside")
    marker = outside / "precious.txt"
    marker.write_bytes(b"do-not-delete")
    link = scratch_dir / "escape"
    kind = _make_dir_link(link, outside)
    if not kind:
        remove_owned_tree(outside, artifacts, missing_ok=True)
        pytest.skip("this session cannot create a directory link")
    try:
        with pytest.raises(AssertionError, match="symlink or reparse point"):
            remove_owned_tree(link, scratch_dir)
        assert marker.read_bytes() == b"do-not-delete"
        assert outside.is_dir()
    finally:
        _remove_link(link)
        remove_owned_tree(outside, artifacts, missing_ok=True)


def test_cleanup_refuses_a_nested_link_inside_the_tree(artifacts, scratch_dir):
    """The whole tree is refused when any part of it is a link, not traversed."""
    outside = new_scratch(artifacts, "outside")
    marker = outside / "precious.txt"
    marker.write_bytes(b"do-not-delete")
    tree = scratch_dir / "tree"
    tree.mkdir()
    (tree / "own.txt").write_bytes(b"own")
    kind = _make_dir_link(tree / "nested", outside)
    if not kind:
        remove_owned_tree(outside, artifacts, missing_ok=True)
        pytest.skip("this session cannot create a directory link")
    try:
        with pytest.raises(AssertionError, match="inside the tree"):
            remove_owned_tree(tree, scratch_dir)
        assert marker.read_bytes() == b"do-not-delete"
        assert (tree / "own.txt").read_bytes() == b"own"
    finally:
        _remove_link(tree / "nested")
        remove_owned_tree(outside, artifacts, missing_ok=True)


def test_cleanup_refuses_a_link_on_the_path_to_the_target(artifacts, scratch_dir):
    """A junction in the middle of the path must not be walked through."""
    outside = new_scratch(artifacts, "outside")
    (outside / "deeper").mkdir()
    marker = outside / "deeper" / "precious.txt"
    marker.write_bytes(b"do-not-delete")
    hop = scratch_dir / "hop"
    kind = _make_dir_link(hop, outside)
    if not kind:
        remove_owned_tree(outside, artifacts, missing_ok=True)
        pytest.skip("this session cannot create a directory link")
    try:
        with pytest.raises(AssertionError):
            remove_owned_tree(hop / "deeper", scratch_dir)
        assert marker.read_bytes() == b"do-not-delete"
    finally:
        _remove_link(hop)
        remove_owned_tree(outside, artifacts, missing_ok=True)


def test_scratch_cleanup_happens_on_success_failure_and_assertion_error(artifacts):
    """The scratch finalizer must run for every outcome, and report failures."""
    outcomes = []

    def _run(outcome):
        scratch = new_scratch(artifacts, "tests")
        try:
            if outcome == "raise":
                raise RuntimeError("boom")
            if outcome == "assert":
                assert False, "mid-test assertion"
            outcomes.append("ok")
        except (RuntimeError, AssertionError):
            outcomes.append("caught")
        finally:
            remove_owned_tree(scratch, artifacts, missing_ok=True)
        return scratch

    for outcome in ("ok", "raise", "assert"):
        removed = _run(outcome)
        assert not removed.exists()
    assert outcomes == ["ok", "caught", "caught"]


def test_fabricate_refuses_a_non_test_run_id(scratch_dir):
    formal = ExperimentTask(
        task_id="gray_s2026", variant="gray", seed=2026,
        run_id="cmx_v1_gray_s2026",
        run_dir=str(ROOT / "runs" / "train" / "cmx_v1_gray_s2026" / "material_fusion"),
    )
    with pytest.raises(AssertionError, match="not strictly inside"):
        fabricate_completed_run(formal, scratch_dir / "v4", "f" * 64)


def test_fabricate_refuses_a_case_variant_of_a_formal_run_id(scratch_dir):
    formal = ExperimentTask(
        task_id="gray_s2026", variant="gray", seed=2026,
        run_id="CMX_V1_GRAY_S2026",
        run_dir=str(ROOT / "runs" / "train" / "CMX_V1_GRAY_S2026" / "material_fusion"),
    )
    with pytest.raises(AssertionError, match="not strictly inside"):
        fabricate_completed_run(formal, scratch_dir / "v4", "f" * 64)


def test_run_production_training_refuses_a_foreign_scratch(monkeypatch):
    """A caller must not be able to point training output at the real runs tree."""
    from .conftest import run_production_training

    with pytest.raises(AssertionError, match="not strictly inside"):
        run_production_training(ROOT / "runs" / "train", "cmx_v1_test_escape")


def test_run_production_training_refuses_a_reused_run_id(scratch_dir):
    from .conftest import run_production_training

    first = run_production_training(scratch_dir, "cmx_v1_test_reuse")
    assert first["exit_code"] == 0
    with pytest.raises(FileExistsError, match="already exists"):
        run_production_training(scratch_dir, "cmx_v1_test_reuse")


# ---------------------------------------------------------------------------
# metric helpers


def test_detailed_metrics_matches_hand_computed():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    preds = [0, 1, 1, 1, 2, 2, 3, 0]
    metrics = detailed_metrics(labels, preds)
    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["confusion_matrix"] == [
        [1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [1, 0, 0, 1],
    ]
    # class 0: tp=1 fp=1 fn=1 -> f1=0.5; class 1: tp=2 fp=1 fn=0 -> 0.8;
    # class 2: 1.0; class 3: tp=1 fp=0 fn=1 -> ~0.667
    assert metrics["per_class"]["metal_submarine"]["f1"] == pytest.approx(0.5)
    assert metrics["per_class"]["plastic_submarine"]["f1"] == pytest.approx(0.8)
    assert metrics["per_class"]["plastic_fish"]["f1"] == pytest.approx(1.0)
    assert metrics["per_class"]["real_fish"]["f1"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["plastic_fish"]["support"] == 2


def test_paired_flip_stats_counts_each_cell():
    labels = [0, 0, 1, 1]
    preds_a = [0, 1, 1, 0]  # correct, wrong, correct, wrong
    preds_b = [0, 0, 0, 1]  # correct, correct, wrong, wrong
    stats = paired_flip_stats(preds_a, preds_b, labels)
    assert stats == {
        "samples": 4, "both_correct": 1, "both_wrong": 0,
        "a_fixes_b_errors": 1, "a_breaks_b_correct": 2,
    }


def test_bootstrap_reports_group_count_and_ci():
    labels = [0, 1] * 20
    preds_a = [0 if i % 2 == 0 else 1 for i in range(40)]
    preds_b = [0] * 40
    groups = [f"g{i // 4}" for i in range(40)]
    stats = bootstrap_macro_f1_delta(preds_a, preds_b, labels, groups, n_boot=100)
    assert stats["n_groups"] == 10
    assert stats["n_boot"] == 100
    assert stats["ci95"][0] <= stats["point_estimate"] <= stats["ci95"][1]
