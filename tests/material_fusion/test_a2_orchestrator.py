"""A2 acceptance tests: the v2 mask protocol in the orchestrator, and the
closed empty-fingerprint admission gap.

Two contracts are covered here:

1. ``scripts/run_material_experiments.py`` selects its protocol explicitly
   (``--protocol {v1,v2_mask}``, default v1) and carries it through
   plan/preflight/run/collect. The two protocols never share a task matrix,
   a run manifest, a management directory or a completion verdict: a v1 task
   refuses a v2 directory and vice versa, even though both matrices use the
   same ``<variant>_s<seed>`` task ids.
2. A v2 run must name the concrete V5 dataset it was trained on. An empty,
   whitespace, malformed or non-string fingerprint is refused at the config,
   the checkpoint, the protocol-admission layer and every entry point, and it
   is refused *before* a model is built, before a training run directory
   exists and before a report is written. The generic read-only dataset audit
   keeps its "no expectation" use, but that use never masquerades as a
   model/data binding.

Nothing here trains on real data: ``run`` mode is exercised with a mock
executor that records the exact argv, and every training-shaped fixture is a
bounded CPU run (or a fabricated run directory) inside ``.test_artifacts``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from .conftest import (
    ROOT,
    build_alternate_v5,
    build_synthetic_source,
    build_synthetic_v4,
    build_synthetic_v5,
    build_test_protocol_spec,
    build_v4_dataset,
    fabricate_completed_mask_run,
    fabricate_completed_run,
    install_test_protocol,
    material_run_dir,
    train_run_root,
    v4_dataset_fingerprint,
    write_mask_experiment_config,
    write_synthetic_pretrained,
    write_v2_checkpoint_payload,
)

from core.fusion_dataset import dataset_fingerprint
from core.material_fusion_dataset import MaterialFusionDataset
from core.material_mask_dataset import (
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    POOLING_INSTANCE_MASK,
    admit_dataset_for_protocol,
    admit_mask_dataset,
    mask_dataset_fingerprint,
    require_fingerprint,
)
from models.material_fusion import (
    MATERIAL_FUSION_MASK_VERSION,
    MaterialFusionModel,
    load_material_checkpoint,
    read_material_checkpoint_metadata,
    validate_material_config,
)
from scripts import run_material_experiments as orchestrator
from scripts.run_material_experiments import (
    DEFAULT_PROTOCOL,
    V1_SPEC,
    V2_DATA_FINGERPRINT,
    V2_MASK_SPEC,
    ExperimentTask,
    build_tasks,
    cmd_collect,
    cmd_plan,
    cmd_preflight,
    cmd_run,
    detailed_metrics,
    management_dir,
    parse_args,
    protocol_spec,
    run_manifest_path,
    train_command,
    validate_completed_task,
)

SEED_2026_TASKS = [
    "gray_s2026", "gray_gray_valid_s2026", "gray_valid_s2026", "gray_polar_s2026",
]

# Values that must never be accepted as a dataset fingerprint. ``True``/``0``
# are included because ``bool`` is an ``int`` subclass and Python would happily
# coerce them in a laxer implementation.
INVALID_FINGERPRINTS = [
    None, "", "   ", "\t\n", 0, 1, True, False, 1.5, [], {}, b"a" * 64,
    "a" * 63, "a" * 65, "A" * 64, "g" * 64, "z" * 64, "a" * 63 + "-", " " * 64,
]


@pytest.fixture(scope="module")
def mask_chain(module_scratch):
    """A synthetic source -> V4 -> V5 chain, built by the production code."""
    workspace = module_scratch.resolve()
    source = build_synthetic_source(module_scratch / "source", counts=(4, 4, 4), size=64)
    v4 = build_synthetic_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)
    return {
        "workspace": workspace,
        "source": source,
        "v4": v4,
        "v5": v5,
        "parent": dataset_fingerprint(v4),
        "fingerprint": mask_dataset_fingerprint(v5),
    }


def _v4_fingerprint(data_root: Path) -> str:
    from scripts.train_polar_fusion import validate_dataset

    return validate_dataset(data_root)["audit_fingerprint"]


def _stub_subprocess(monkeypatch, returncode):
    """Replace the executor; the recorded argv proves nothing real started."""
    calls = []

    def _run(command, **_kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(orchestrator.subprocess, "run", _run)
    return calls


def _argv_contains(argv, text: str) -> bool:
    """True when any argv element contains ``text`` (paths are single elements)."""
    return any(text in str(part) for part in argv)


class _StubGray(torch.nn.Module):
    """Deterministic stand-in for the historical gray Model B."""

    def __init__(self, fixed_class: int = 3):
        super().__init__()
        self.fixed_class = int(fixed_class)

    def forward(self, rgb):
        logits = torch.zeros(rgb.shape[0], 4, device=rgb.device)
        logits[:, self.fixed_class] = 1.0
        return logits


def _patch_config(tmp_path: Path, spec, **overrides):
    """A copy of a protocol's config file with fields overridden."""
    import yaml

    config = yaml.safe_load((ROOT / spec.config).read_text(encoding="utf-8"))
    config.update(overrides)
    path = tmp_path / f"{spec.name}_patched.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return dataclasses.replace(spec, config=str(path))


# ---------------------------------------------------------------------------
# Protocol configuration: v1 stays the default, v2 is explicit


def test_default_protocol_is_v1_and_v1_paths_are_unchanged():
    for argv in (["plan"], ["preflight"], ["run", "--tasks", "gray_s2026"], ["collect"]):
        assert parse_args(argv).protocol == DEFAULT_PROTOCOL == "v1"
    tasks = build_tasks()
    assert {task.protocol for task in tasks} == {"v1"}
    assert all(task.run_id.startswith("cmx_v1_") for task in tasks)
    assert management_dir(V1_SPEC) == orchestrator.EXPERIMENTS_DIR
    assert run_manifest_path(V1_SPEC) == orchestrator.RUN_MANIFEST_PATH
    assert orchestrator.EXPECTED_PROTOCOL == dict(V1_SPEC.expected)


def test_v2_matrix_is_twelve_unique_isolated_tasks():
    tasks = build_tasks(protocol="v2_mask")
    assert len(tasks) == 12
    assert len({task.task_id for task in tasks}) == 12
    assert len({task.run_id for task in tasks}) == 12
    assert {task.protocol for task in tasks} == {"v2_mask"}
    assert all(task.run_id.startswith("cmx_v2_mask_") for task in tasks)
    assert [task.task_id for task in tasks[:4]] == SEED_2026_TASKS
    assert {task.variant for task in tasks} == {
        "gray", "gray_gray_valid", "gray_valid", "gray_polar",
    }
    assert {task.seed for task in tasks} == {2026, 2027, 2028}
    for task in tasks:
        argv = train_command(task, "0", V2_MASK_SPEC)
        assert argv[1].endswith("train_material_fusion.py")
        assert _argv_contains(argv, "material_fusion_v2_mask.yaml")
        assert task.run_id in argv
        assert "--init-from" not in argv
        assert "--limit-batches" not in argv and "--epochs" not in argv
        assert Path(task.run_dir).is_relative_to(orchestrator.RUNS_ROOT)
        assert Path(task.run_dir).name == "material_fusion"
    # the two matrices never share a directory
    v1_dirs = {task.run_dir for task in build_tasks(protocol="v1")}
    v2_dirs = {task.run_dir for task in tasks}
    assert not (v1_dirs & v2_dirs)
    assert management_dir(V2_MASK_SPEC) != management_dir(V1_SPEC)
    assert run_manifest_path(V2_MASK_SPEC) != run_manifest_path(V1_SPEC)


def test_alternating_protocol_calls_do_not_contaminate_each_other(scratch_dir, monkeypatch):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    v1_first = [task.run_id for task in build_tasks()]
    v2_first = [task.run_id for task in build_tasks(protocol="v2_mask")]
    v1_second = [task.run_id for task in build_tasks()]
    v2_second = [task.run_id for task in build_tasks(protocol="v2_mask")]
    assert v1_first == v1_second
    assert v2_first == v2_second
    assert set(v1_first).isdisjoint(v2_first)
    assert management_dir(V1_SPEC).name == "material_experiments"
    assert management_dir(V2_MASK_SPEC).name == "material_experiments_v2_mask"


def test_unknown_protocol_is_refused():
    with pytest.raises(ValueError, match="unknown protocol"):
        protocol_spec("v3")
    with pytest.raises(SystemExit):
        parse_args(["plan", "--protocol", "v3"])


# ---------------------------------------------------------------------------
# plan


def test_plan_writes_per_protocol_files_and_never_overwrites(
    scratch_dir, monkeypatch, capsys
):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    monkeypatch.setattr(
        orchestrator, "PLAN_PATH", scratch_dir / "material_experiments" / "plan.json"
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("plan must never start a subprocess")

    monkeypatch.setattr(orchestrator.subprocess, "run", _fail)
    assert cmd_plan(SimpleNamespace(device="cpu", protocol="v1")) == 0
    assert cmd_plan(SimpleNamespace(device="cpu", protocol="v2_mask")) == 0
    capsys.readouterr()

    v1 = json.loads((scratch_dir / "material_experiments" / "plan.json").read_text("utf-8"))
    v2 = json.loads(
        (scratch_dir / "material_experiments_v2_mask" / "plan.json").read_text("utf-8")
    )
    assert v1["protocol"] == "v1" and v1["experiment"] == "material_fusion_v1"
    assert v1["data"]["root"].endswith("underwater_cls_fusion_v4_band")
    assert v2["protocol"] == "v2_mask"
    assert v2["experiment"] == MATERIAL_FUSION_MASK_VERSION
    assert v2["data"]["root"].endswith("underwater_cls_fusion_v5_instance_mask")
    assert v2["data"]["data_fingerprint"] == V2_DATA_FINGERPRINT
    assert v2["data"]["parent_data_fingerprint"] == V2_MASK_SPEC.parent_data_fingerprint
    assert v2["pooling"] == {
        "pooling_mode": POOLING_INSTANCE_MASK,
        "mask_schema": MASK_SCHEMA_VERSION,
        "mask_resize_version": MASK_RESIZE_VERSION,
    }
    assert v2["fixed_budget"]["epochs"] == 30
    assert v2["fixed_budget"]["effective_batch"] == 16
    assert v2["fixed_budget"]["smoke"] is False
    assert v2["fixed_budget"]["limit_batches"] == 0
    assert v2["resource_basis"]["epochs"] == 30
    assert v2["resource_basis"]["optimizer_steps_per_run"] > 0
    assert len(v2["tasks"]) == 12
    for entry in v2["tasks"]:
        assert entry["protocol"] == "v2_mask"
        assert entry["experiment"] == MATERIAL_FUSION_MASK_VERSION
        assert entry["run_id"].startswith("cmx_v2_mask_")
        assert "material_fusion_v2_mask.yaml" in entry["command"]
        assert "--init-from" not in entry["command"]
        assert "--limit-batches" not in entry["command"]
        assert "--smoke" not in entry["command"]
    # a second plan must refuse to overwrite the first
    assert orchestrator.main(["plan", "--protocol", "v2_mask"]) == 2
    assert json.loads(
        (scratch_dir / "material_experiments_v2_mask" / "plan.json").read_text("utf-8")
    ) == v2


# ---------------------------------------------------------------------------
# preflight


def test_v2_preflight_passes_read_only_and_creates_no_run_directory(
    scratch_dir, monkeypatch, capsys
):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    run_root = scratch_dir / "runs" / "train"
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", run_root)

    def _fail(*_args, **_kwargs):
        raise AssertionError("preflight must never start a subprocess")

    monkeypatch.setattr(orchestrator.subprocess, "run", _fail)
    output = scratch_dir / "preflight_v2.json"
    code = cmd_preflight(
        SimpleNamespace(device="cpu", protocol="v2_mask", output=str(output))
    )
    capsys.readouterr()
    assert code == 0
    report = json.loads(output.read_text("utf-8"))
    assert report["protocol"] == "v2_mask"
    assert report["experiment"] == MATERIAL_FUSION_MASK_VERSION
    assert report["all_ok"] is True
    checks = {check["check"]: check for check in report["checks"]}
    assert checks["protocol_matches_fixed_constants"]["ok"] is True
    assert checks["protocol_data_identity"]["ok"] is True
    assert checks["dataset_admitted"]["ok"] is True
    assert V2_DATA_FINGERPRINT in checks["dataset_admitted"]["detail"]
    assert f"mask_schema={MASK_SCHEMA_VERSION}" in checks["dataset_admitted"]["detail"]
    assert checks["pretrained_weight"]["ok"] is True
    assert checks["fallback_weights_present"]["ok"] is True
    assert checks["output_dir_state"]["ok"] is True
    assert "all 12 v2_mask run directories are free" in checks["output_dir_state"]["detail"]
    assert not run_root.exists() or not any(run_root.iterdir())


def test_failed_v2_preflight_creates_no_run_directory_but_reports_the_reason(
    scratch_dir, monkeypatch, capsys
):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    run_root = scratch_dir / "runs" / "train"
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", run_root)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    output = scratch_dir / "preflight_fail.json"
    code = cmd_preflight(
        SimpleNamespace(device="0", protocol="v2_mask", output=str(output))
    )
    capsys.readouterr()
    assert code == 1
    report = json.loads(output.read_text("utf-8"))
    assert report["all_ok"] is False
    checks = {check["check"]: check for check in report["checks"]}
    assert checks["device_available"]["ok"] is False
    assert "CUDA is not available" in checks["device_available"]["detail"]
    assert not run_root.exists() or not any(run_root.iterdir())


def test_preflight_reports_a_protocol_config_conflict(scratch_dir, monkeypatch):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    conflicting = dataclasses.replace(V1_SPEC, config=V2_MASK_SPEC.config)
    checks = orchestrator.admission_checks(conflicting, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    assert by_name["protocol_matches_fixed_constants"]["ok"] is False
    assert "not the 'v1' protocol" in by_name["protocol_matches_fixed_constants"]["detail"]

    conflicting_reverse = dataclasses.replace(V2_MASK_SPEC, config=V1_SPEC.config)
    checks = orchestrator.admission_checks(
        conflicting_reverse, "cpu", include_output_state=False
    )
    by_name = {check["check"]: check for check in checks}
    assert by_name["protocol_matches_fixed_constants"]["ok"] is False
    assert "not the 'v2_mask' protocol" in by_name["protocol_matches_fixed_constants"]["detail"]


def test_v2_preflight_refuses_a_wrong_data_fingerprint(scratch_dir, monkeypatch):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    spec = _patch_config(scratch_dir, V2_MASK_SPEC, data_fingerprint="a" * 64)
    checks = orchestrator.admission_checks(spec, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    # the config's identity is not the protocol's: preflight must not pass
    assert by_name["protocol_data_identity"]["ok"] is False
    assert not all(check["ok"] for check in checks)


def test_v2_preflight_refuses_a_wrong_parent_fingerprint(scratch_dir, monkeypatch):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    spec = _patch_config(scratch_dir, V2_MASK_SPEC, parent_data_fingerprint="b" * 64)
    checks = orchestrator.admission_checks(spec, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    assert by_name["protocol_data_identity"]["ok"] is False
    assert not all(check["ok"] for check in checks)


def test_v2_preflight_refuses_a_wrong_pretrained_sha(scratch_dir, monkeypatch):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    spec = _patch_config(scratch_dir, V2_MASK_SPEC, pretrained_sha256="c" * 64)
    checks = orchestrator.admission_checks(spec, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    assert by_name["pretrained_weight"]["ok"] is False


# ---------------------------------------------------------------------------
# run: mock executor only, never a real training process


def test_run_stops_on_first_failure_and_records_the_full_argv(
    scratch_dir, isolated_mask_runs, monkeypatch
):
    monkeypatch.setattr(orchestrator, "run_admission", lambda spec, device: (True, "ok"))
    calls = _stub_subprocess(monkeypatch, returncode=1)
    args = SimpleNamespace(
        tasks="gray_s2026,gray_polar_s2026", device="cpu",
        skip_completed=False, protocol="v2_mask",
    )
    assert cmd_run(args) == 1
    assert len(calls) == 1  # the second task was never dispatched
    argv = calls[0]
    assert argv[1].endswith("train_material_fusion.py")
    assert _argv_contains(argv, "material_fusion_v2_mask.yaml")
    assert "cmx_v2_mask_test_gray_s2026" in argv
    assert "--variant" in argv and "gray" in argv
    assert "--init-from" not in argv and "--limit-batches" not in argv
    manifest = json.loads(
        (management_dir(V2_MASK_SPEC) / "run_manifest.json").read_text("utf-8")
    )
    assert manifest["protocol"] == "v2_mask"
    assert manifest["experiment"] == MATERIAL_FUSION_MASK_VERSION
    attempt = manifest["tasks"]["gray_s2026"]["attempts"][0]
    assert attempt["status"] == "failed" and attempt["exit_code"] == 1
    assert attempt["command"] == argv
    assert "gray_polar_s2026" not in manifest["tasks"]


def test_run_refuses_before_spawning_when_admission_fails(
    scratch_dir, isolated_mask_runs, monkeypatch
):
    calls = _stub_subprocess(monkeypatch, returncode=0)
    monkeypatch.setattr(
        orchestrator, "run_admission",
        lambda spec, device: (False, "dataset_admitted: fingerprint mismatch"),
    )
    args = SimpleNamespace(
        tasks="gray_s2026", device="cpu", skip_completed=False, protocol="v2_mask",
    )
    assert cmd_run(args) == 2
    assert calls == []
    assert not (management_dir(V2_MASK_SPEC) / "run_manifest.json").exists()


def test_run_refuses_a_manifest_from_another_protocol(
    scratch_dir, isolated_mask_runs, monkeypatch
):
    calls = _stub_subprocess(monkeypatch, returncode=0)
    monkeypatch.setattr(orchestrator, "run_admission", lambda spec, device: (True, "ok"))
    path = management_dir(V2_MASK_SPEC) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"protocol": "v1", "tasks": {}}), encoding="utf-8")
    args = SimpleNamespace(
        tasks="gray_s2026", device="cpu", skip_completed=False, protocol="v2_mask",
    )
    assert cmd_run(args) == 2
    assert calls == []
    assert json.loads(path.read_text("utf-8"))["protocol"] == "v1"  # untouched


def test_skip_completed_never_skips_an_invalid_v2_run(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch
):
    monkeypatch.setattr(orchestrator, "run_admission", lambda spec, device: (True, "ok"))
    calls = _stub_subprocess(monkeypatch, returncode=0)
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"],
        smoke=True, limit_batches=3,
    )
    args = SimpleNamespace(
        tasks=task.task_id, device="cpu", skip_completed=True, protocol="v2_mask",
    )
    assert cmd_run(args) == 2
    assert calls == []
    manifest = json.loads(
        (management_dir(V2_MASK_SPEC) / "run_manifest.json").read_text("utf-8")
    )
    attempt = manifest["tasks"][task.task_id]["attempts"][0]
    assert attempt["status"] == "refused_existing"


# ---------------------------------------------------------------------------
# completion validation


def test_v1_completion_validation_still_passes_and_rejects_v2_tasks(
    scratch_dir, isolated_runs
):
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fabricate_completed_run(task, data_root, _v4_fingerprint(data_root))
    ok, reason = validate_completed_task(task, _v1_test_spec(scratch_dir, _v4_fingerprint(data_root)))
    assert ok, reason
    # the same directory is not evidence for the other protocol
    v2_task = dataclasses.replace(
        task, protocol="v2_mask", run_id="cmx_v2_mask_test_gray_s2026"
    )
    ok, reason = validate_completed_task(v2_task, V2_MASK_SPEC)
    assert not ok and "not this task's protocol" in reason


def test_v2_completion_validation_passes_and_rejects_v1_tasks(
    scratch_dir, isolated_mask_runs, mask_chain
):
    task = isolated_mask_runs.by_task["gray_polar_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    ok, reason = validate_completed_task(task, _mask_test_spec(scratch_dir, mask_chain))
    assert ok, reason
    v1_task = dataclasses.replace(
        task, protocol="v1", run_id="cmx_v1_test_gray_polar_s2026"
    )
    ok, reason = validate_completed_task(v1_task, V1_SPEC)
    assert not ok and "not this task's protocol" in reason


def _v2_negative_cases():
    return {
        "smoke": ({"smoke": True, "limit_batches": 3}, "smoke"),
        "truncated_history": ({"history_epochs": 5}, "expected one per epoch"),
        "wrong_best_epoch": ({"best_epoch": 5}, "the history selects epoch"),
        "init_from": ({"init_from": "runs/train/other/material_fusion/best.pt"}, "init_from"),
        "checkpoint_fingerprint": ({"checkpoint_fingerprint": "b" * 64}, "data_fingerprint"),
        "checkpoint_parent": ({"checkpoint_parent_fingerprint": "c" * 64}, "parent_data_fingerprint"),
    }


@pytest.mark.parametrize("case", sorted(_v2_negative_cases()))
def test_v2_completion_validation_refuses_invalid_runs(
    case, scratch_dir, isolated_mask_runs, mask_chain
):
    overrides, expected_fragment = _v2_negative_cases()[case]
    task = isolated_mask_runs.by_task[f"gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"],
        **overrides,
    )
    # the spec matches the fixture, so the refusal must be the injected defect
    ok, reason = validate_completed_task(task, _mask_test_spec(scratch_dir, mask_chain))
    assert not ok
    assert expected_fragment in reason, reason


def test_v2_completion_validation_uses_the_expected_mask_protocol(
    scratch_dir, isolated_mask_runs, mask_chain
):
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    spec = _mask_test_spec(scratch_dir, mask_chain)
    drifted = dataclasses.replace(spec, mask_schema="instance_mask_polygon_crop_v2")
    ok, reason = validate_completed_task(task, drifted)
    assert not ok and "mask_schema" in reason


def test_real_v1_runs_still_validate_under_the_v1_protocol():
    tasks = build_tasks(protocol="v1")
    if not all(Path(task.run_dir).is_dir() for task in tasks):
        pytest.skip("the real v1 matrix is not present in this checkout")
    for task in tasks:
        ok, reason = validate_completed_task(task, V1_SPEC)
        assert ok, (task.task_id, reason)


def test_real_v2_smoke_runs_are_refused_on_protocol_grounds_not_on_a_v1_constant():
    """The formal/smoke separation is proven with the *v2* validator.

    A smoke directory is refused because its own recorded budget (2 epochs,
    limit_batches 3) is not the fixed v2 protocol — not merely because a v1
    constant was applied to it.
    """
    run_dirs = sorted((ROOT / "runs" / "train").glob("mask_a1fix_smoke_*/material_fusion"))
    if not run_dirs:
        pytest.skip("no recorded v2 smoke runs are present in this checkout")
    for run_dir in run_dirs:
        run_config = json.loads((run_dir / "run_config.json").read_text("utf-8"))
        config = run_config["config"]
        task = ExperimentTask(
            task_id=f"{config['variant']}_s{config['seed']}",
            variant=config["variant"],
            seed=int(config["seed"]),
            run_id=run_config["run_id"],
            run_dir=str(run_dir),
            protocol="v2_mask",
        )
        ok, reason = validate_completed_task(task, V2_MASK_SPEC)
        assert not ok, run_dir
        assert "fixed protocol" in reason, reason
        assert "epochs" in reason and "limit_batches" in reason, reason
        assert "not this task's protocol" not in reason, reason


# ---------------------------------------------------------------------------
# collect


def _fabricate_v2_seed_2026(isolated_mask_runs, mask_chain):
    for task in isolated_mask_runs.tasks:
        if task.seed == 2026:
            fabricate_completed_mask_run(
                task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
            )


def test_v2_collect_passes_the_instance_mask_into_the_model(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    _fabricate_v2_seed_2026(isolated_mask_runs, mask_chain)
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    observed = []
    real_forward = MaterialFusionModel.forward

    def _spy(self, rgb, aux=None, instance_mask=None):
        observed.append(
            None if instance_mask is None
            else (tuple(instance_mask.shape), bool(torch.isfinite(instance_mask).all()))
        )
        return real_forward(self, rgb, aux, instance_mask)

    monkeypatch.setattr(MaterialFusionModel, "forward", _spy)

    def _no_v4_gate(*_args, **_kwargs):
        raise AssertionError("a v2 collect must never use the V4 admission gate")

    monkeypatch.setattr("core.material_fusion_dataset.admit_material_dataset", _no_v4_gate)

    output = scratch_dir / "collect_v2"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()

    assert observed, "the model was never called"
    assert all(entry is not None for entry in observed), observed
    assert all(
        len(shape) == 3 and shape[1] == shape[2] == 224 and 0 < shape[0] <= 3
        for shape, _ in observed
    ), observed
    assert all(finite for _, finite in observed)

    report = json.loads((output / "report.json").read_text("utf-8"))
    assert report["protocol"] == "v2_mask"
    assert report["experiment"] == MATERIAL_FUSION_MASK_VERSION
    assert report["pooling_mode"] == POOLING_INSTANCE_MASK
    assert report["split"] == "val" and report["test_split_used"] is False
    assert report["eval_data_identity_bound"] is True  # bound to the spec identity
    assert set(report["tasks"]) == set(SEED_2026_TASKS)
    assert report["complete_matrix"] is False  # only one seed is present
    for task_id, entry in report["tasks"].items():
        assert entry["protocol"] == "v2_mask"
        assert entry["pooling_mode"] == POOLING_INSTANCE_MASK
        assert entry["data_fingerprint"] == mask_chain["fingerprint"]
        assert entry["parent_data_fingerprint"] == mask_chain["parent"]
        assert len(entry["checkpoint_sha256"]) == 64
        assert entry["per_sample_count"] == entry["pure"]["samples"]


def test_v2_collect_per_sample_records_recompute_the_aggregates_and_keep_the_tail(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    _fabricate_v2_seed_2026(isolated_mask_runs, mask_chain)
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "collect_v2_tail"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text("utf-8"))

    records = [
        json.loads(line)
        for line in (output / "per_sample_predictions.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert report["per_sample_predictions"]["records"] == len(records)

    manifest_rows = orchestrator.manifest_split_rows(mask_chain["v5"], "val")
    expected_names = [row["sample_name"] for row in manifest_rows]
    by_task: dict[str, list[dict]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], []).append(record)
    for task_id, task_records in by_task.items():
        # every val sample exactly once, in manifest order, tail batch included
        names = [record["sample_name"] for record in task_records]
        assert names == expected_names
        assert len(set(names)) == len(names)
        labels = [record["label"] for record in task_records]
        pure = detailed_metrics(labels, [r["raw_prediction"] for r in task_records])
        assert pure == report["tasks"][task_id]["pure"]
        assert len(task_records) == report["tasks"][task_id]["per_sample_count"]


def test_v2_collect_routing_is_separate_and_never_fabricated(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    _fabricate_v2_seed_2026(isolated_mask_runs, mask_chain)
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(fixed_class=3), "stub gray identity"),
    )
    # The pure path is pinned to class 1 so the raw and routed prediction
    # vectors are guaranteed to differ; the real forward (with the mask) still
    # runs underneath.
    real_forward = MaterialFusionModel.forward

    def _fixed(self, rgb, aux=None, instance_mask=None):
        real_forward(self, rgb, aux, instance_mask)
        logits = torch.zeros(rgb.shape[0], 4, device=rgb.device)
        logits[:, 1] = 1.0
        return logits

    monkeypatch.setattr(MaterialFusionModel, "forward", _fixed)

    class _InvalidQualityDataset(MaterialFusionDataset):
        """Forces val sample 0's quality valid_ratio to 0 (routing condition)."""

        def __getitem__(self, index):
            item = list(super().__getitem__(index))
            if index == 0:
                position = -3 if self.with_instance_mask else -2
                item[position] = torch.zeros_like(item[position])
            return tuple(item)

    monkeypatch.setattr(
        "core.material_fusion_dataset.MaterialFusionDataset", _InvalidQualityDataset
    )
    output = scratch_dir / "collect_v2_routed"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text("utf-8"))
    dual = report["tasks"]["gray_polar_s2026"]
    assert dual["invalid_subset_pure"]["samples"] == 1
    assert dual["routed"] is not None
    assert report["tasks"]["gray_s2026"]["routed"] is None  # E1 has no routing

    records = [
        json.loads(line)
        for line in (output / "per_sample_predictions.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    dual_records = [r for r in records if r["task_id"] == "gray_polar_s2026"]
    fallback_rows = [r for r in dual_records if r["used_fallback"]]
    assert len(fallback_rows) == 1
    assert fallback_rows[0]["routed_prediction"] == 3  # the stub's fixed class
    assert fallback_rows[0]["raw_prediction"] == 1
    assert fallback_rows[0]["polar_valid"] is False
    # raw and routed are stored separately and each aggregate recomputes
    labels = [r["label"] for r in dual_records]
    assert detailed_metrics(labels, [r["raw_prediction"] for r in dual_records]) == dual["pure"]
    assert detailed_metrics(
        labels, [r["routed_prediction"] for r in dual_records]
    ) == dual["routed"]
    assert dual["routed"] != dual["pure"]


def test_v2_collect_reports_a_missing_fallback_instead_of_fabricating_routing(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    _fabricate_v2_seed_2026(isolated_mask_runs, mask_chain)
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))

    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("stub: the historical gray fallback is absent")

    monkeypatch.setattr("scripts.eval_material_fusion.load_fallback_gray", _missing)
    output = scratch_dir / "collect_v2_nofb"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text("utf-8"))
    dual = report["tasks"]["gray_polar_s2026"]
    assert dual["routed"] is None
    assert "fallback unavailable" in dual["routed_note"]
    records = [
        json.loads(line)
        for line in (output / "per_sample_predictions.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert all(record["routed_prediction"] is None for record in records)
    assert all(record["used_fallback"] is False for record in records)


def test_v2_collect_excludes_a_checkpoint_trained_on_a_different_v5(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    """A checkpoint of another V5 derivation is never part of this collection.

    The refusal now happens at the protocol-identity stage (the checkpoint's
    recorded dataset is not the protocol's), which is strictly earlier than the
    per-checkpoint data binding the collection also keeps.
    """
    alternate = build_alternate_v5(
        mask_chain["v5"], scratch_dir / "v5_alt", mask_chain["v4"], mask_chain["parent"]
    )
    alternate_fingerprint = mask_dataset_fingerprint(alternate)
    assert alternate_fingerprint != mask_chain["fingerprint"]
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(task, alternate, alternate_fingerprint, mask_chain["parent"])
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "collect_v2_wrong_v5"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert task.task_id not in report["tasks"]
    reason = report["excluded"][task.task_id]
    assert "data_fingerprint" in reason
    assert alternate_fingerprint in reason


def test_v2_collect_refuses_an_existing_output(
    scratch_dir, isolated_mask_runs, mask_chain
):
    existing = scratch_dir / "collect_exists"
    existing.mkdir()
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(existing), protocol="v2_mask",
    )
    assert cmd_collect(args) == 2


def test_v2_collect_refuses_a_config_from_another_protocol(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch
):
    conflicting = dataclasses.replace(V2_MASK_SPEC, config=V1_SPEC.config)
    monkeypatch.setitem(orchestrator.PROTOCOLS, "v2_mask", conflicting)
    output = scratch_dir / "collect_conflict"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 2
    assert not output.exists()


def test_v1_collect_never_uses_the_v5_gate(scratch_dir, isolated_runs, monkeypatch, capsys):
    data_root = build_v4_dataset(
        scratch_dir / "v4", counts=(8, 4, 4), invalid_indices={"val": {1}}
    )
    fingerprint = _v4_fingerprint(data_root)
    for task in isolated_runs.tasks:
        if task.seed == 2026:
            fabricate_completed_run(task, data_root, fingerprint)
    install_test_protocol(monkeypatch, _v1_test_spec(scratch_dir, fingerprint))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )

    def _no_v5_gate(*_args, **_kwargs):
        raise AssertionError("a v1 collect must never use the V5 admission gate")

    monkeypatch.setattr("core.material_mask_dataset.admit_mask_dataset", _no_v5_gate)
    output = scratch_dir / "collect_v1"
    args = SimpleNamespace(
        device="cpu", batch=4, data=str(data_root), output=str(output), protocol="v1",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert report["protocol"] == "v1"
    assert report["pooling_mode"] == "gap"
    assert set(report["tasks"]) == set(SEED_2026_TASKS)
    assert report["tasks"]["gray_s2026"]["routed"] is None
    assert report["tasks"]["gray_polar_s2026"]["routed"] is not None
    assert "Protocol: v1" in (output / "summary.md").read_text("utf-8")


# ---------------------------------------------------------------------------
# The empty-fingerprint admission gap


@pytest.mark.parametrize("value", INVALID_FINGERPRINTS)
def test_require_fingerprint_refuses_every_invalid_value(value):
    with pytest.raises(ValueError) as excinfo:
        require_fingerprint(value, "data_fingerprint")
    assert "data_fingerprint" in str(excinfo.value)


def test_require_fingerprint_accepts_a_well_formed_digest():
    assert require_fingerprint("a" * 64, "fp") == "a" * 64
    assert require_fingerprint(V2_DATA_FINGERPRINT, "fp") == V2_DATA_FINGERPRINT


def _mask_config_dict(mask_chain, **overrides):
    config = {
        "experiment": MATERIAL_FUSION_MASK_VERSION,
        "variant": "gray_polar",
        "data": str(mask_chain["v5"]),
        "data_fingerprint": mask_chain["fingerprint"],
        "pretrained": "weights/pretrained/mit_b0.pth",
        "pretrained_sha256": "",
        "upstream_commit": "e251d860aebc2f583a6c4919877e6bebe7f1aff3",
        "preprocessing_version": "v4_band_224_area_linear_v1",
        "pooling_mode": POOLING_INSTANCE_MASK,
        "mask_schema": MASK_SCHEMA_VERSION,
        "mask_resize_version": MASK_RESIZE_VERSION,
        "parent_data_fingerprint": mask_chain["parent"],
        "imgsz": 224,
        "batch": 8,
        "accum_steps": 2,
        "effective_batch": 16,
        "epochs": 30,
        "warmup_epochs": 2,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "lr": 1e-4,
        "encoder_lr": 1e-5,
        "seed": 2026,
        "workers": 0,
        "fallback_weights": "",
        "smoke": False,
        "limit_batches": 0,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("value", INVALID_FINGERPRINTS)
def test_v2_config_requires_a_concrete_dataset_fingerprint(value, mask_chain):
    with pytest.raises(ValueError):
        validate_material_config(
            _mask_config_dict(mask_chain, data_fingerprint=value)
        )


def test_v2_config_accepts_a_concrete_fingerprint(mask_chain):
    config = validate_material_config(_mask_config_dict(mask_chain))
    assert config["data_fingerprint"] == mask_chain["fingerprint"]


@pytest.mark.parametrize("value", INVALID_FINGERPRINTS)
def test_v2_checkpoint_requires_a_concrete_dataset_fingerprint(value, scratch_dir):
    path = write_v2_checkpoint_payload(
        scratch_dir / "ckpt.pt", data_fingerprint=value
    )
    with pytest.raises(ValueError):
        read_material_checkpoint_metadata(path)


def test_v2_checkpoint_accepts_a_concrete_fingerprint(scratch_dir):
    path = write_v2_checkpoint_payload(scratch_dir / "ckpt_ok.pt", data_fingerprint="c" * 64)
    metadata = read_material_checkpoint_metadata(path)
    assert metadata.data_fingerprint == "c" * 64
    assert metadata.pooling_mode == POOLING_INSTANCE_MASK


def test_v1_legacy_compatibility_is_unchanged():
    """A v1 config may still leave the fingerprint unset (template/legacy path)."""
    v1_config = {
        "experiment": "material_fusion_v1",
        "variant": "gray_polar",
        "data": "datasets/underwater_cls_fusion_v4_band",
        "data_fingerprint": "",
        "pretrained": "weights/pretrained/mit_b0.pth",
        "pretrained_sha256": "",
        "upstream_commit": "e251d860aebc2f583a6c4919877e6bebe7f1aff3",
        "preprocessing_version": "v4_band_224_area_linear_v1",
        "imgsz": 224, "batch": 8, "accum_steps": 2, "effective_batch": 16,
        "epochs": 30, "warmup_epochs": 2, "optimizer": "adamw",
        "weight_decay": 0.01, "lr": 1e-4, "encoder_lr": 1e-5, "seed": 2026,
        "workers": 0, "fallback_weights": "", "smoke": False, "limit_batches": 0,
    }
    normalized = validate_material_config(v1_config)
    assert normalized["data_fingerprint"] == ""
    assert normalized["pooling_mode"] == "gap"


@pytest.mark.parametrize("value", ["", "   ", None, "a" * 63, "A" * 64])
def test_v2_protocol_admission_refuses_an_empty_or_invalid_expectation(value, mask_chain):
    with pytest.raises(ValueError) as excinfo:
        admit_dataset_for_protocol(
            mask_chain["v5"],
            pooling_mode=POOLING_INSTANCE_MASK,
            expected_fingerprint=value,
            expected_parent_fingerprint=mask_chain["parent"],
        )
    assert "expected_fingerprint" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        admit_dataset_for_protocol(
            mask_chain["v5"],
            pooling_mode=POOLING_INSTANCE_MASK,
            expected_fingerprint=mask_chain["fingerprint"],
            expected_parent_fingerprint=value,
        )
    assert "expected_parent_fingerprint" in str(excinfo.value)


def test_v2_protocol_admission_accepts_the_concrete_identity(mask_chain):
    report = admit_dataset_for_protocol(
        mask_chain["v5"],
        pooling_mode=POOLING_INSTANCE_MASK,
        expected_fingerprint=mask_chain["fingerprint"],
        expected_parent_fingerprint=mask_chain["parent"],
        expected_mask_schema=MASK_SCHEMA_VERSION,
        expected_mask_resize_version=MASK_RESIZE_VERSION,
    )
    assert report["audit_fingerprint"] == mask_chain["fingerprint"]
    assert report["v4_parent_fingerprint"] == mask_chain["parent"]


def test_generic_read_only_audit_may_still_run_without_an_expectation(mask_chain):
    """The audit keeps its no-expectation use; it just never claims a binding."""
    report = admit_mask_dataset(mask_chain["v5"], expected_parent_fingerprint="")
    assert report["audit_fingerprint"] == mask_chain["fingerprint"]
    assert "binding" not in report


def test_v2_protocol_admission_refuses_a_wrong_but_wellformed_fingerprint(mask_chain):
    with pytest.raises(ValueError, match="does not match the expected"):
        admit_dataset_for_protocol(
            mask_chain["v5"],
            pooling_mode=POOLING_INSTANCE_MASK,
            expected_fingerprint="d" * 64,
            expected_parent_fingerprint=mask_chain["parent"],
        )


def test_same_parent_but_a_different_v5_is_refused(scratch_dir, mask_chain):
    alternate = build_alternate_v5(
        mask_chain["v5"], scratch_dir / "v5_alt", mask_chain["v4"], mask_chain["parent"]
    )
    alternate_fingerprint = mask_dataset_fingerprint(alternate)
    assert alternate_fingerprint != mask_chain["fingerprint"]
    # it admits against its own fingerprint and the shared V4 parent ...
    report = admit_dataset_for_protocol(
        alternate,
        pooling_mode=POOLING_INSTANCE_MASK,
        expected_fingerprint=alternate_fingerprint,
        expected_parent_fingerprint=mask_chain["parent"],
    )
    assert report["v4_parent_fingerprint"] == mask_chain["parent"]
    # ... and is refused against the other V5 derivation of the same V4 state
    with pytest.raises(ValueError, match="does not match the expected"):
        admit_dataset_for_protocol(
            alternate,
            pooling_mode=POOLING_INSTANCE_MASK,
            expected_fingerprint=mask_chain["fingerprint"],
            expected_parent_fingerprint=mask_chain["parent"],
        )


def test_empty_fingerprint_is_refused_before_training_builds_a_model(
    scratch_dir, mask_chain, monkeypatch
):
    import scripts.train_material_fusion as train_module
    from scripts.train_material_fusion import main as train_main

    pretrained = write_synthetic_pretrained(scratch_dir / "mit.pth")
    config_path = write_mask_experiment_config(
        scratch_dir / "bad_fp.yaml", mask_chain["v5"], pretrained, "",
        mask_chain["parent"], variant="gray_polar", imgsz=64,
    )
    built = []

    def _no_model(*_args, **_kwargs):
        built.append(1)
        raise AssertionError("no model may be built for an invalid fingerprint")

    monkeypatch.setattr(train_module, "build_material_model", _no_model)
    run_root = train_run_root(scratch_dir)
    with mock.patch.object(train_module, "RUNS_ROOT", run_root):
        exit_code = train_main([
            "--run-id", "a2_bad_fp", "--config", str(config_path), "--device", "cpu",
        ])
    assert exit_code == 2
    assert built == []
    assert not material_run_dir(scratch_dir, "a2_bad_fp").exists()


def test_empty_fingerprint_is_refused_before_a_model_is_built(scratch_dir, mask_chain):
    import models.material_fusion as model_module

    payload = write_v2_checkpoint_payload(scratch_dir / "bad_ckpt.pt", data_fingerprint="")
    built = []
    with mock.patch.object(
        model_module, "build_material_model",
        lambda *a, **k: built.append(1),
    ):
        with pytest.raises(ValueError):
            load_material_checkpoint(payload, variant="gray_polar", imgsz=224)
    assert built == []


def test_empty_fingerprint_is_refused_before_any_report_is_written(
    scratch_dir, mask_chain
):
    import scripts.diagnose_material_mask as diagnose
    import scripts.eval_material_fusion as evaluate

    payload = write_v2_checkpoint_payload(scratch_dir / "bad_ckpt.pt", data_fingerprint="")
    diag_output = scratch_dir / "diagnose_out"
    assert diagnose.main([
        "--mode", "controlled", "--data", str(mask_chain["v5"]),
        "--checkpoint", str(payload), "--split", "val", "--device", "cpu",
        "--output", str(diag_output),
    ]) == 2
    assert not diag_output.exists()

    eval_report = scratch_dir / "eval_out.json"
    assert evaluate.main([
        "--checkpoint", str(payload), "--split", "val", "--device", "cpu",
        "--no-fallback", "--output", str(eval_report),
    ]) == 2
    assert not eval_report.exists()


def test_diagnose_admission_refuses_an_empty_checkpoint_identity(scratch_dir, mask_chain):
    import scripts.diagnose_material_mask as diagnose

    good = write_v2_checkpoint_payload(
        scratch_dir / "good_ckpt.pt",
        data_fingerprint=mask_chain["fingerprint"],
        parent_data_fingerprint=mask_chain["parent"],
    )
    metadata = read_material_checkpoint_metadata(good)
    args = diagnose.parse_args([
        "--mode", "controlled", "--data", str(mask_chain["v5"]),
        "--checkpoint", str(good), "--split", "val",
        "--output", str(scratch_dir / "diagnose_bind"),
    ])
    with pytest.raises(ValueError, match="checkpoint data_fingerprint"):
        diagnose._admit(args, dataclasses.replace(metadata, data_fingerprint=""))
    with pytest.raises(ValueError, match="checkpoint parent_data_fingerprint"):
        diagnose._admit(args, dataclasses.replace(metadata, parent_data_fingerprint=""))
    # a valid identity is bound only after the fingerprint really was compared
    report = diagnose._admit(args, metadata)
    assert report["binding"] == "checkpoint_v5_fingerprint_and_v4_parent"
    assert report["audit_fingerprint"] == mask_chain["fingerprint"]
    # a well-formed but different V5 is refused, never labelled bound
    wrong = write_v2_checkpoint_payload(
        scratch_dir / "wrong_ckpt.pt",
        data_fingerprint="e" * 64,
        parent_data_fingerprint=mask_chain["parent"],
    )
    with pytest.raises(ValueError, match="does not match the expected"):
        diagnose._admit(args, read_material_checkpoint_metadata(wrong))


# ---------------------------------------------------------------------------
# The chosen protocol is the identity authority
#
# A complete, internally consistent run of a *different* dataset or a
# different pretrained weight is not this experiment's evidence. These tests
# deliberately run against a spec whose identity is the fixture's, so the only
# thing that can refuse is the spec binding itself.


def _mask_test_spec(scratch_dir: Path, mask_chain):
    """A v2 protocol spec whose identity is the synthetic V5 chain's."""
    return build_test_protocol_spec(
        scratch_dir,
        V2_MASK_SPEC,
        data_fingerprint=mask_chain["fingerprint"],
        parent_data_fingerprint=mask_chain["parent"],
        pretrained_sha256="a" * 64,
        data=mask_chain["v5"],
    )


def _v1_test_spec(scratch_dir: Path, fingerprint: str):
    """A v1 protocol spec whose identity is the synthetic V4 fixture's."""
    return build_test_protocol_spec(
        scratch_dir, V1_SPEC, data_fingerprint=fingerprint,
        pretrained_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    "field, foreign, fragment",
    [
        ("data_fingerprint", "c" * 64, "data_fingerprint"),
        ("parent_data_fingerprint", "9" * 64, "parent_data_fingerprint"),
        ("pretrained_sha256", "b" * 64, "pretrained_sha256"),
    ],
)
def test_v2_completed_run_must_match_the_protocol_identity(
    field, foreign, fragment, scratch_dir, isolated_mask_runs, mask_chain
):
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    spec = _mask_test_spec(scratch_dir, mask_chain)
    # the aligned spec accepts the fixture (this is the injected test protocol)
    ok, reason = validate_completed_task(task, spec)
    assert ok, reason
    # only the one identity field differs from the protocol the run claims
    drifted = dataclasses.replace(spec, **{field: foreign})
    ok, reason = validate_completed_task(task, drifted)
    assert not ok, f"{field} was not bound to the protocol"
    assert fragment in reason, reason


@pytest.mark.parametrize(
    "field, foreign, fragment",
    [
        ("data_fingerprint", "c" * 64, "data_fingerprint"),
        ("pretrained_sha256", "b" * 64, "pretrained_sha256"),
    ],
)
def test_v1_completed_run_must_match_the_protocol_identity(
    field, foreign, fragment, scratch_dir, isolated_runs
):
    task = isolated_runs.by_task["gray_s2026"]
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    fingerprint = v4_dataset_fingerprint(data_root)
    fabricate_completed_run(task, data_root, fingerprint)
    spec = _v1_test_spec(scratch_dir, fingerprint)
    ok, reason = validate_completed_task(task, spec)
    assert ok, reason
    drifted = dataclasses.replace(spec, **{field: foreign})
    ok, reason = validate_completed_task(task, drifted)
    assert not ok, f"{field} was not bound to the protocol"
    assert fragment in reason, reason


def test_a_foreign_identity_run_is_refused_by_the_production_spec(
    scratch_dir, isolated_mask_runs, mask_chain
):
    """The fixture is self-consistent, but it is not the production v2 dataset."""
    task = isolated_mask_runs.by_task["gray_s2026"]
    run_dir = fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    # prove internal consistency independently of the validator
    run_config = json.loads((run_dir / "run_config.json").read_text("utf-8"))
    assert run_config["config"]["data_fingerprint"] == run_config["dataset_fingerprint"]
    for role in ("best", "last"):
        meta = read_material_checkpoint_metadata(run_dir / f"{role}.pt")
        assert meta.data_fingerprint == run_config["config"]["data_fingerprint"]
        assert meta.parent_data_fingerprint == run_config["config"]["parent_data_fingerprint"]
    assert run_config["config"]["data_fingerprint"] != V2_MASK_SPEC.data_fingerprint
    assert run_config["config"]["parent_data_fingerprint"] != V2_MASK_SPEC.parent_data_fingerprint
    ok, reason = validate_completed_task(task, V2_MASK_SPEC)
    assert not ok
    assert "data_fingerprint" in reason


def test_skip_completed_refuses_a_foreign_identity_run(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch
):
    monkeypatch.setattr(orchestrator, "run_admission", lambda spec, device: (True, "ok"))
    calls = _stub_subprocess(monkeypatch, returncode=0)
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    args = SimpleNamespace(
        tasks=task.task_id, device="cpu", skip_completed=True, protocol="v2_mask",
    )
    assert cmd_run(args) == 2
    assert calls == []
    manifest = json.loads(
        (management_dir(V2_MASK_SPEC) / "run_manifest.json").read_text("utf-8")
    )
    attempt = manifest["tasks"][task.task_id]["attempts"][0]
    assert attempt["status"] == "refused_existing"


def test_v2_preflight_refuses_a_replaced_pretrained_file_even_when_the_config_agrees(
    scratch_dir, monkeypatch
):
    """A config SHA that merely matches a swapped file is not the official weight."""
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    replacement = scratch_dir / "fake_mit_b0.pth"
    replacement.write_bytes(b"not the official MiT-B0 weights")
    sha = hashlib.sha256(replacement.read_bytes()).hexdigest()
    assert sha != V2_MASK_SPEC.pretrained_sha256
    spec = _patch_config(
        scratch_dir, V2_MASK_SPEC, pretrained=str(replacement), pretrained_sha256=sha
    )
    checks = orchestrator.admission_checks(spec, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    assert by_name["pretrained_weight"]["ok"] is False
    assert V2_MASK_SPEC.pretrained_sha256 in by_name["pretrained_weight"]["detail"]


def test_v2_preflight_refuses_a_config_identity_that_is_not_the_spec(
    scratch_dir, monkeypatch
):
    monkeypatch.setattr(orchestrator, "EXPERIMENTS_DIR", scratch_dir / "material_experiments")
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", scratch_dir / "runs" / "train")
    spec = _patch_config(scratch_dir, V2_MASK_SPEC, data_fingerprint="a" * 64)
    checks = orchestrator.admission_checks(spec, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    assert by_name["protocol_data_identity"]["ok"] is False
    assert spec.data_fingerprint in by_name["protocol_data_identity"]["detail"]


# ---------------------------------------------------------------------------
# collect binds the data to the chosen protocol


def test_v2_collect_refuses_relocated_data_that_is_a_different_v5(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    """Internally consistent (data + checkpoint), but not this protocol's V5."""
    alternate = build_alternate_v5(
        mask_chain["v5"], scratch_dir / "v5_alt", mask_chain["v4"], mask_chain["parent"]
    )
    alternate_fingerprint = mask_dataset_fingerprint(alternate)
    assert alternate_fingerprint != mask_chain["fingerprint"]
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(task, alternate, alternate_fingerprint, mask_chain["parent"])
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "collect_bypass"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(alternate), output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 2
    assert not output.exists()
    capsys.readouterr()


def test_v2_collect_accepts_a_relocated_copy_with_the_same_identity(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    relocated = scratch_dir / "v5_relocated"
    shutil.copytree(mask_chain["v5"], relocated)
    assert mask_dataset_fingerprint(relocated) == mask_chain["fingerprint"]
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "collect_relocated"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(relocated), output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert task.task_id in report["tasks"]
    assert report["data_fingerprint"] == mask_chain["fingerprint"]


def test_v2_collect_default_path_uses_the_spec_bound_dataset(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    """No ``--data``: the config's dataset is admitted with the spec identity."""
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "collect_default"
    args = SimpleNamespace(
        device="cpu", batch=3, data=None, output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert task.task_id in report["tasks"]
    assert report["data_fingerprint"] == mask_chain["fingerprint"]
    assert report["eval_data_identity_bound"] is True


def test_v2_collect_refuses_a_config_identity_that_is_not_the_spec(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    drifted = dataclasses.replace(
        _mask_test_spec(scratch_dir, mask_chain), data_fingerprint="f" * 64
    )
    install_test_protocol(monkeypatch, drifted)
    output = scratch_dir / "collect_identity_drift"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 2
    assert not output.exists()
    capsys.readouterr()


# ---------------------------------------------------------------------------
# The per-sample report digest must be the digest of the bytes on disk


def test_per_sample_report_sha_matches_the_written_bytes(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch, capsys
):
    task = isolated_mask_runs.by_task["gray_s2026"]
    fabricate_completed_mask_run(
        task, mask_chain["v5"], mask_chain["fingerprint"], mask_chain["parent"]
    )
    install_test_protocol(monkeypatch, _mask_test_spec(scratch_dir, mask_chain))
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    real_rows = orchestrator.manifest_split_rows

    def _rows_with_odd_group_names(data_root, split):
        rows = real_rows(data_root, split)
        for index, row in enumerate(rows):
            # non-ASCII plus an embedded newline: JSONL must escape it and the
            # file must keep LF line endings
            row["group_name"] = f"组{index}·鱼类\nsecond line"
        return rows

    monkeypatch.setattr(orchestrator, "manifest_split_rows", _rows_with_odd_group_names)

    output = scratch_dir / "collect_jsonl"
    args = SimpleNamespace(
        device="cpu", batch=3, data=str(mask_chain["v5"]),
        output=str(output), protocol="v2_mask",
    )
    assert cmd_collect(args) == 0
    capsys.readouterr()
    report = json.loads((output / "report.json").read_text("utf-8"))
    path = Path(report["per_sample_predictions"]["path"])
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == report["per_sample_predictions"]["sha256"]
    assert b"\r\n" not in payload, "the JSONL must be written with LF endings"
    assert "组0".encode("utf-8") in payload, "non-ASCII must be written as UTF-8"
    assert payload.count(b"\n") == report["per_sample_predictions"]["records"]
    # every line is still a complete JSON record
    for line in payload.decode("utf-8").splitlines():
        json.loads(line)


# ---------------------------------------------------------------------------
# An unset pretrained SHA must not be backfilled with the protocol's
#
# ``admission_checks`` used to write ``config["pretrained_sha256"] or
# spec.pretrained_sha256``, so a config that pinned no pretrained weight passed
# the three-way check with the protocol's own digest and the log showed the
# backfilled value as if the config had recorded it. The completion validator
# meanwhile required the config's SHA to equal the spec's exactly, so the two
# gates disagreed. These tests pin the config's raw value as the only accepted
# one.


def _empty_sha_spec(scratch_dir: Path, mask_chain):
    """A v2 spec whose *config* pins an empty pretrained SHA.

    The spec keeps the protocol's real pretrained SHA: the config is what is
    under test, and the previous ``or``-backfill would have hidden its empty
    value behind the protocol's digest.
    """
    written = build_test_protocol_spec(
        scratch_dir, V2_MASK_SPEC,
        data_fingerprint=mask_chain["fingerprint"],
        parent_data_fingerprint=mask_chain["parent"],
        pretrained_sha256="",
        data=mask_chain["v5"],
        label="empty_sha",
    )
    return dataclasses.replace(written, pretrained_sha256=V2_MASK_SPEC.pretrained_sha256)


def _empty_sha_v1_spec(scratch_dir: Path, data_root: Path):
    """The same construction for the fixed v1 protocol."""
    written = build_test_protocol_spec(
        scratch_dir, V1_SPEC,
        data_fingerprint=v4_dataset_fingerprint(data_root),
        pretrained_sha256="",
        data=str(data_root),
        label="empty_sha",
    )
    return dataclasses.replace(written, pretrained_sha256=V1_SPEC.pretrained_sha256)


def _write_identity_pretrained(scratch_dir: Path, name: str = "identity_mit.pth"):
    """A scratch file used purely as a pretrained identity (its own sha256)."""
    path = Path(scratch_dir) / name
    path.write_bytes(b"identity-only pretrained placeholder")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_preflight_refuses_an_empty_config_pretrained_sha(
    scratch_dir, mask_chain
):
    """The file matches the spec, but the config pins nothing: refused."""
    spec = _empty_sha_spec(scratch_dir, mask_chain)
    checks = orchestrator.admission_checks(spec, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    assert by_name["pretrained_weight"]["ok"] is False
    # the log reports the config's own value, never a backfilled protocol SHA
    detail = by_name["pretrained_weight"]["detail"]
    assert "config=''" in detail, detail
    assert f"protocol={spec.pretrained_sha256}" in detail, detail
    assert not all(check["ok"] for check in checks)

    admitted, reason = orchestrator.run_admission(spec, "cpu")
    assert admitted is False
    assert "pretrained_weight" in reason, reason


def test_v1_preflight_refuses_an_empty_config_pretrained_sha(scratch_dir):
    """The same rule applies to the fixed v1 protocol."""
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    spec = _empty_sha_v1_spec(scratch_dir, data_root)
    checks = orchestrator.admission_checks(spec, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    assert by_name["pretrained_weight"]["ok"] is False
    detail = by_name["pretrained_weight"]["detail"]
    assert "config=''" in detail, detail
    assert f"protocol={spec.pretrained_sha256}" in detail, detail
    assert not all(check["ok"] for check in checks)

    admitted, reason = orchestrator.run_admission(spec, "cpu")
    assert admitted is False
    assert "pretrained_weight" in reason, reason


def test_v1_preflight_accepts_a_three_way_matching_pretrained_sha(scratch_dir):
    """File == config == protocol, all non-empty: the legitimate case passes."""
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    pretrained, sha = _write_identity_pretrained(scratch_dir)
    spec = build_test_protocol_spec(
        scratch_dir, V1_SPEC,
        data_fingerprint=v4_dataset_fingerprint(data_root),
        pretrained_sha256=sha, pretrained=str(pretrained), data=str(data_root),
        label="identity_ok",
    )
    checks = orchestrator.admission_checks(spec, "cpu", include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    assert by_name["pretrained_weight"]["ok"] is True
    detail = by_name["pretrained_weight"]["detail"]
    assert f"config={sha!r}" in detail and f"protocol={sha}" in detail, detail
    assert all(check["ok"] for check in checks), checks


def test_v2_run_refuses_an_empty_pretrained_sha_before_spawning(
    scratch_dir, isolated_mask_runs, mask_chain, monkeypatch
):
    """A config that pins no pretrained weight never reaches a subprocess."""
    calls = _stub_subprocess(monkeypatch, returncode=0)
    spec = _empty_sha_spec(scratch_dir, mask_chain)
    install_test_protocol(monkeypatch, spec)
    manifest_path = management_dir(spec) / "run_manifest.json"

    for skip_completed in (False, True):
        args = SimpleNamespace(
            tasks="gray_s2026", device="cpu",
            skip_completed=skip_completed, protocol="v2_mask",
        )
        assert cmd_run(args) == 2, f"skip_completed={skip_completed}"

    assert calls == [], "no training subprocess may be started"
    assert not manifest_path.exists(), "no run manifest may be created"
    assert not any(isolated_mask_runs.run_root.iterdir()), "no run directory"


def test_v1_run_refuses_an_empty_pretrained_sha_before_spawning(
    scratch_dir, isolated_runs, monkeypatch
):
    calls = _stub_subprocess(monkeypatch, returncode=0)
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4))
    spec = _empty_sha_v1_spec(scratch_dir, data_root)
    install_test_protocol(monkeypatch, spec)
    # v1's manifest path is the module-level one; redirect it into this scratch
    # so the assertion can never read or touch the real matrix's manifest.
    manifest_path = scratch_dir / "run_manifest.json"
    monkeypatch.setattr(orchestrator, "RUN_MANIFEST_PATH", manifest_path)

    for skip_completed in (False, True):
        args = SimpleNamespace(
            tasks="gray_s2026", device="cpu",
            skip_completed=skip_completed, protocol="v1",
        )
        assert cmd_run(args) == 2, f"skip_completed={skip_completed}"

    assert calls == []
    assert not manifest_path.exists()
    assert not any(isolated_runs.run_root.iterdir())
