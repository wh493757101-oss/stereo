"""Shared fixtures for the CMX material-fusion experiment tests.

Every disposable artifact a test creates — synthetic datasets, synthetic
pretrained weights, bounded CPU training run directories, stage snapshots
and reports — is written under a unique per-session directory inside the
project-local ``.test_artifacts/`` tree (see AGENTS.md). Nothing is written
to the real ``runs/train``, and cleanup is decided by resolved path
ownership, never by directory name or prefix.

Scratch directories are created with ``tempfile``-style uniqueness rather
than pytest's ``tmp_path``: the repo's default basetemp directory can stay
locked on Windows (WinError 5) and break ``tmp_path``-based tests, and
``tmp_path`` would land outside ``.test_artifacts`` when pytest is invoked
without an explicit ``--basetemp``.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish", "real_fish")

# The only tree this package may write to. A test path outside it is not
# test-owned, whatever it is named: the real ``runs/train`` matrix, a
# ``yolo26_*`` run, another package's scratch and a differently-cased or
# ``..``-aliased spelling of any of them all resolve outside this root.
ARTIFACTS_ROOT = ROOT / ".test_artifacts"

# Names the current run's temporary directory (the same directory the
# harness points TEMP/TMP and pytest --basetemp at). Every disposable test
# artifact is written under ``<that dir>/artifacts``; the variable only ever
# relocates test scratch and is validated to stay inside ARTIFACTS_ROOT.
ARTIFACTS_ENV = "MATERIAL_FUSION_TEST_ARTIFACTS"

# Test-owned run ids are prefixed so the isolated matrix is recognisable in
# reports; this is a naming convention, not the ownership check.
TEST_RUN_ID_PREFIX = "cmx_v1_test_"

_REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SESSION_LABEL = (
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
)
_SESSION_ROOT: Path | None = None


# ---------------------------------------------------------------------------
# Ownership-checked scratch space


def is_link_or_reparse_point(path: Path) -> bool:
    """True for a symlink, junction or any other reparse point.

    ``Path.is_symlink`` misses Windows junctions, which are reparse points
    but not symlinks; ``lstat`` plus the reparse-point file attribute
    catches both.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT_FLAG)


def session_artifacts_root() -> Path:
    """This session's unique disposable root, created on first use.

    The root is ``<run temp dir>/artifacts`` when the harness names one via
    ``ARTIFACTS_ENV``, otherwise ``.test_artifacts/<utc stamp>_<pid>/artifacts``.
    Either way it is created exclusively inside :data:`ARTIFACTS_ROOT`: a
    reused or non-empty directory is refused rather than cleared.
    """
    global _SESSION_ROOT
    if _SESSION_ROOT is None:
        override = os.environ.get(ARTIFACTS_ENV, "").strip()
        base = Path(override) if override else ARTIFACTS_ROOT
        _SESSION_ROOT = _prepare_artifacts_root(
            base / f"run_{_SESSION_LABEL}" / "artifacts"
        )
    return _SESSION_ROOT


def _prepare_artifacts_root(root: Path) -> Path:
    root = Path(root)
    if is_link_or_reparse_point(root.parent) or is_link_or_reparse_point(root):
        raise AssertionError(
            f"refusing to use {root}: it or its parent is a symlink or reparse point"
        )
    project_artifacts = ARTIFACTS_ROOT.resolve()
    resolved_parent = root.parent.resolve()
    if (
        resolved_parent != project_artifacts
        and project_artifacts not in resolved_parent.parents
    ):
        raise AssertionError(
            f"refusing to use {root}: {ARTIFACTS_ENV} must name a directory "
            f"inside {ARTIFACTS_ROOT}"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        if not root.is_dir():
            raise AssertionError(f"refusing to use {root}: not a directory")
        if any(root.iterdir()):
            raise AssertionError(
                f"refusing to reuse the non-empty artifacts root {root}; each run "
                "uses a fresh unique directory"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
    return root


def new_scratch(parent: Path, label: str) -> Path:
    """A fresh, unique, empty directory ``parent/label/<uuid>``.

    Creation is exclusive: an existing directory is never reused or
    cleared, so a collision is reported instead of silently sharing state.
    """
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / label / uuid4().hex
    target.mkdir(parents=True, exist_ok=False)
    return target


def assert_owned(path: Path | str, owned_root: Path | str) -> Path:
    """Resolve ``path`` and prove it is strictly inside ``owned_root``.

    Ownership is decided by resolved path relationships, never by name,
    prefix or "is somewhere under runs/train". A ``..`` alias, a
    differently-cased spelling, an absolute path elsewhere on the disk, the
    owned root itself and a symlink/junction/reparse-point escape all fail
    the containment test. Every existing component of the raw path is also
    checked for link/reparse markers so the refusal names the exact reason.
    """
    owned = Path(owned_root)
    if is_link_or_reparse_point(owned):
        raise AssertionError(
            f"refusing to use {owned}: the owned root is itself a symlink or "
            "reparse point"
        )
    owned = owned.resolve(strict=True)

    target = Path(path)
    probe = target if target.is_absolute() else Path.cwd() / target
    cursor = probe
    while True:
        if is_link_or_reparse_point(cursor):
            raise AssertionError(
                f"refusing to touch {probe}: {cursor} is a symlink or reparse "
                "point, so the target can escape the owned root"
            )
        if cursor == owned or cursor.parent == cursor:
            break
        cursor = cursor.parent

    resolved = probe.resolve()
    if resolved == owned or owned not in resolved.parents:
        raise AssertionError(
            f"refusing to touch {resolved}: not strictly inside the owned root "
            f"{owned}"
        )
    return resolved


def _first_reparse_point(root: Path) -> Path | None:
    """The first symlink/junction/reparse point at or below ``root``."""
    stack = [root]
    while stack:
        current = stack.pop()
        if is_link_or_reparse_point(current):
            return current
        if not current.is_dir():
            continue
        for entry in os.scandir(current):
            stack.append(Path(entry.path))
    return None


def remove_owned_tree(
    path: Path | str, owned_root: Path | str, *, missing_ok: bool = False
) -> None:
    """Remove a disposable tree after proving it is inside ``owned_root``.

    A link anywhere inside the tree is refused rather than traversed: the
    removal must not be able to reach content outside the owned root through
    a junction or symlink, whatever ``shutil.rmtree`` happens to do with it.
    Failures are never swallowed: a removal error is re-raised with the exact
    target path and the underlying exception, so a cleanup that did not
    happen can never be reported as success.
    """
    target = assert_owned(path, owned_root)
    if not target.exists():
        if missing_ok:
            return
        raise FileNotFoundError(f"nothing to remove at {target}")
    escape = _first_reparse_point(target)
    if escape is not None:
        raise AssertionError(
            f"refusing to remove {target}: {escape} is a symlink or reparse "
            "point inside the tree, so the removal could escape the owned root"
        )
    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise RuntimeError(
            f"failed to remove test output {target}: {exc!r}"
        ) from exc


def unique_path(parent: Path, label: str) -> Path:
    """A unique, not-yet-existing path under ``parent``.

    For outputs the production code must create exclusively itself (the
    stage-snapshot target refuses an existing directory), so this returns a
    path that does not exist rather than creating one.
    """
    target = Path(parent) / label / uuid4().hex
    if target.exists():
        raise FileExistsError(f"generated path already exists: {target}")
    return target


def train_run_root(scratch: Path) -> Path:
    """The scratch-owned ``runs/train`` holding a test's training output."""
    return Path(scratch) / "runs" / "train"


def material_run_dir(scratch: Path, run_id: str) -> Path:
    """The scratch-owned run directory for ``run_id``."""
    return train_run_root(scratch) / run_id / "material_fusion"


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(scope="session")
def artifacts():
    """The session's unique ``.test_artifacts`` run directory."""
    root = session_artifacts_root()
    yield root
    remove_owned_tree(root, ARTIFACTS_ROOT, missing_ok=True)


@pytest.fixture()
def scratch_dir(artifacts):
    """Isolated scratch directory, removed when the test ends.

    The finalizer runs on success, on failure and after a mid-test
    assertion error, and reports a removal failure instead of ignoring it.
    """
    path = new_scratch(artifacts, "tests")
    yield path
    remove_owned_tree(path, artifacts, missing_ok=True)


@pytest.fixture(scope="module")
def module_scratch(artifacts):
    """Isolated scratch directory shared by one module, removed at its end."""
    path = new_scratch(artifacts, "modules")
    yield path
    remove_owned_tree(path, artifacts, missing_ok=True)


@pytest.fixture()
def train_runs_root(scratch_dir):
    """Redirect the training entry's output root into this test's scratch.

    ``scripts.train_material_fusion.RUNS_ROOT`` is the production default
    (``<project>/runs/train``) resolved at call time; patching it is the
    only isolation hook, and the production CLI gains no relaxation flag.
    """
    import scripts.train_material_fusion as train_module

    run_root = train_run_root(scratch_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    with mock.patch.object(train_module, "RUNS_ROOT", run_root):
        yield run_root


@pytest.fixture(scope="module")
def module_train_runs_root(module_scratch):
    """Module-scoped ``train_runs_root`` for shared training fixtures."""
    import scripts.train_material_fusion as train_module

    run_root = train_run_root(module_scratch)
    run_root.mkdir(parents=True, exist_ok=True)
    with mock.patch.object(train_module, "RUNS_ROOT", run_root):
        yield run_root


def _tree_inventory(root: Path) -> dict[str, tuple]:
    if not root.is_dir():
        return {}
    inventory: dict[str, tuple] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            inventory[relative] = ("dir",)
        else:
            try:
                inventory[relative] = ("file", path.stat().st_size)
            except OSError:
                inventory[relative] = ("file", -1)
    return inventory


@pytest.fixture(scope="session", autouse=True)
def real_runs_tree_untouched():
    """Fail the session if the tests created or removed real training output.

    Read-only observation of ``runs/train`` before and after the session;
    the guard plants nothing there. A concurrent formal run would also
    change the tree, and the failure message says so.
    """
    root = ROOT / "runs" / "train"
    before = _tree_inventory(root)
    yield
    after = _tree_inventory(root)
    if before != after:
        created = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(
            name for name in set(before) & set(after) if before[name] != after[name]
        )
        raise AssertionError(
            "the real runs/train tree changed during the test session — tests "
            f"must write only under {ARTIFACTS_ROOT}. created={created} "
            f"removed={removed} changed={changed}. If a formal training was "
            "running concurrently, that is the cause."
        )


# ---------------------------------------------------------------------------
# Synthetic dataset / checkpoint builders


def build_v4_dataset(
    root: Path,
    counts: tuple[int, int, int] = (8, 4, 4),
    size: int = 16,
    seed: int = 0,
    invalid_indices: dict[str, set[int]] | None = None,
) -> Path:
    """Write a small but fully audited V4-format dataset.

    ``invalid_indices`` maps split -> sample indices that get an all-invalid
    polarization measurement (valid all zero, quality valid_ratio 0.0), the
    routing condition of the material-fusion fallback.
    """
    from core.fusion_dataset import file_digest, save_fusion_sample

    invalid_indices = invalid_indices or {}
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    digests: dict[str, str] = {}
    for split, count in zip(("train", "val", "test"), counts):
        for index in range(count):
            class_id = index % len(CLASS_NAMES)
            sample_name = f"{split}_sample_{index:03d}"
            gray = rng.integers(20, 235, size=(size, size), dtype=np.uint8)
            if index in invalid_indices.get(split, set()):
                valid = np.zeros((size, size), dtype=np.uint8)
                signed_q = np.zeros((size, size), dtype=np.float32)
            else:
                valid = (rng.random((size, size)) > 0.3).astype(np.uint8)
                signed_q = (rng.random((size, size)).astype(np.float32) - 0.5)
                signed_q[valid == 0] = 0.0
            abs_q = np.abs(signed_q).astype(np.float32)
            valid_ratio = float(valid.mean())
            mean_abs_q = (
                float(abs_q[valid > 0].mean()) if valid.any() else 0.0
            )
            quality = np.asarray(
                [valid_ratio, 1.0, 1.0, mean_abs_q], dtype=np.float32
            )
            relative = f"{split}/{CLASS_NAMES[class_id]}/{sample_name}.npz"
            save_fusion_sample(
                root / relative, gray, signed_q, abs_q, valid, quality, class_id
            )
            digests[relative] = file_digest(root / relative)
            rows.append(
                {
                    "sample_name": sample_name,
                    "split": split,
                    "group_name": f"{split}_group_{index}",
                    "class_id": str(class_id),
                    "class_name": CLASS_NAMES[class_id],
                    "source_frame": sample_name,
                    "object_index": "0",
                    "stereo_valid": "true",
                    "stereo_reason": "ok",
                    "stereo_valid_ratio": f"{valid_ratio:.6f}",
                    "disparity": "100.0",
                    "polar_valid_ratio": f"{valid_ratio:.6f}",
                    "quality_valid_ratio": f"{valid_ratio:.6f}",
                    "quality_in_bounds_ratio": "1.000000",
                    "quality_brightness_valid_ratio": "1.000000",
                    "quality_mean_abs_q": f"{mean_abs_q:.6f}",
                    "npz_path": relative,
                }
            )

    manifest_path = root / "dataset_manifest.csv"
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = root / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "class_names": list(CLASS_NAMES),
                "npz_fields": ["gray", "signed_q", "abs_q", "valid", "quality", "class_id"],
                "splits": dict(zip(("train", "val", "test"), counts)),
                "total_samples": sum(counts),
                "generation": {"synthetic_test_fixture": True},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    audit_path = root / "dataset_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "output_root": str(root),
                "class_names": list(CLASS_NAMES),
                "expected_splits": dict(zip(("train", "val", "test"), counts)),
                "manifest_sha256": file_digest(manifest_path),
                "summary_sha256": file_digest(summary_path),
                "file_digests": digests,
                "audit_passed": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def build_synthetic_mit_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    """A correct-shape MiT-B0 backbone state dict in original naming.

    Built from a randomly initialized single-stream encoder so every tensor
    has the exact shape the loader must accept, without needing the real
    pretrained file.
    """
    from models.cmx.dual_segformer import MiTEncoder

    torch.manual_seed(seed)
    encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
    return {key: value.detach().clone() for key, value in encoder.state_dict().items()}


def write_synthetic_pretrained(path: Path, seed: int = 0) -> Path:
    """Save a synthetic backbone state dict as a checkpoint file."""
    torch.save(build_synthetic_mit_state_dict(seed), path)
    return path


def write_experiment_config(
    path: Path,
    data_root: Path,
    pretrained: Path,
    variant: str = "gray_polar",
    effective_batch: int = 4,
    epochs: int = 2,
    warmup_epochs: int = 1,
    imgsz: int = 64,
) -> Path:
    """Minimal valid experiment config for fast CPU training tests."""
    import yaml

    config = {
        "experiment": "material_fusion_v1",
        "variant": variant,
        "data": str(data_root),
        "data_fingerprint": "",
        "pretrained": str(pretrained),
        "pretrained_sha256": "",
        "upstream_commit": "e251d860aebc2f583a6c4919877e6bebe7f1aff3",
        "preprocessing_version": "v4_band_224_area_linear_v1",
        "imgsz": imgsz,
        "batch": effective_batch // 2,
        "accum_steps": 2,
        "effective_batch": effective_batch,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "lr": 1e-4,
        "encoder_lr": 1e-5,
        "seed": 2026,
        "workers": 0,
        "fallback_weights": "",
        "fallback_identity": "",
        "smoke": False,
        "limit_batches": 0,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_production_training(
    scratch_dir: Path,
    run_id: str,
    variant: str = "gray_polar",
    epochs: int = 2,
    warmup_epochs: int = 1,
    limit_batches: int = 1,
    snapshot_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """Run the real training entry on a tiny synthetic dataset.

    Bounded CPU unit-test training of the production path; the dataset, the
    synthetic pretrained file, the config, the run directory and any stage
    snapshots all live inside ``scratch_dir``. The production output root is
    redirected through the training entry's injectable ``RUNS_ROOT`` for the
    duration of the call, so nothing lands in the real ``runs/train``.

    The run id must not already exist: a test uses a unique id instead of
    clearing a directory left by an earlier attempt.
    """
    import scripts.train_material_fusion as train_module
    from scripts.train_material_fusion import main as train_main

    scratch = Path(scratch_dir)
    assert_owned(scratch, session_artifacts_root())
    run_root = train_run_root(scratch)
    run_dir = material_run_dir(scratch, run_id)
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} already exists; use a unique run id for each call "
            "instead of deleting a previous directory"
        )
    data_root = build_v4_dataset(scratch / f"v4_{run_id}", counts=(4, 4, 4))
    pretrained = write_synthetic_pretrained(scratch / f"mit_{run_id}.pth")
    config_path = write_experiment_config(
        scratch / f"config_{run_id}.yaml", data_root, pretrained,
        variant=variant, effective_batch=4, epochs=epochs,
        warmup_epochs=warmup_epochs, imgsz=64,
    )
    args = [
        "--run-id", run_id,
        "--config", str(config_path),
        "--device", "cpu",
        "--epochs", str(epochs),
        "--warmup-epochs", str(warmup_epochs),
        "--limit-batches", str(limit_batches),
    ]
    if snapshot_dir is not None:
        args += ["--stage-snapshot-dir", str(snapshot_dir)]
    args += extra_args or []
    with mock.patch.object(train_module, "RUNS_ROOT", run_root):
        exit_code = train_main(args)
    return {
        "exit_code": exit_code,
        "run_id": run_id,
        "run_dir": run_dir,
        "data_root": data_root,
        "config_path": config_path,
        "pretrained": pretrained,
        "snapshot_dir": Path(snapshot_dir) if snapshot_dir is not None else None,
    }


# ---------------------------------------------------------------------------
# Orchestrator isolation


def isolated_tasks(run_root: Path) -> list:
    """The 12-task matrix with test-owned run ids under ``run_root``."""
    from scripts.run_material_experiments import VARIANTS, ExperimentTask

    tasks = []
    for seed in (2026, 2027, 2028):
        for variant in VARIANTS:
            run_id = f"{TEST_RUN_ID_PREFIX}{variant}_s{seed}"
            tasks.append(
                ExperimentTask(
                    task_id=f"{variant}_s{seed}",
                    variant=variant,
                    seed=seed,
                    run_id=run_id,
                    run_dir=str(run_root / run_id / "material_fusion"),
                )
            )
    return tasks


# Test-owned run ids for the v2 mask matrix, prefixed so the isolated matrix is
# recognisable; this is a naming convention, not the ownership check.
MASK_TEST_RUN_ID_PREFIX = "cmx_v2_mask_test_"


def isolated_mask_tasks(run_root: Path) -> list:
    """The 12-task v2 matrix with test-owned run ids under ``run_root``."""
    from scripts.run_material_experiments import VARIANTS, ExperimentTask

    tasks = []
    for seed in (2026, 2027, 2028):
        for variant in VARIANTS:
            run_id = f"{MASK_TEST_RUN_ID_PREFIX}{variant}_s{seed}"
            tasks.append(
                ExperimentTask(
                    task_id=f"{variant}_s{seed}",
                    variant=variant,
                    seed=seed,
                    run_id=run_id,
                    run_dir=str(run_root / run_id / "material_fusion"),
                    protocol="v2_mask",
                )
            )
    return tasks


@pytest.fixture()
def isolated_runs(scratch_dir, monkeypatch):
    """Redirect every orchestrator run directory into the test scratch dir.

    Returns an object with ``run_root`` (scratch-owned ``runs/train``),
    ``tasks`` and ``by_task``. Nothing the orchestrator writes during the
    test can land in the real ``runs/train``.
    """
    from scripts import run_material_experiments as orchestrator

    run_root = train_run_root(scratch_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    tasks = isolated_tasks(run_root)
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", run_root)
    # Accept and ignore the protocol argument: the fixture replaces the whole
    # matrix, so every mode resolves to these scratch-owned v1 tasks.
    monkeypatch.setattr(orchestrator, "build_tasks", lambda *a, **k: list(tasks))
    return SimpleNamespace(
        run_root=run_root, tasks=tasks, by_task={task.task_id: task for task in tasks},
    )


@pytest.fixture()
def isolated_mask_runs(scratch_dir, monkeypatch):
    """The v2 equivalent of ``isolated_runs``.

    Redirects the run root, the task matrix *and* the v2 management directory
    (``EXPERIMENTS_DIR.parent / material_experiments_v2_mask``) into this
    test's scratch, so a v2 plan/preflight/run/collect can never write to the
    real ``runs/train`` or ``analysis/runs`` trees.
    """
    from scripts import run_material_experiments as orchestrator

    run_root = train_run_root(scratch_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    tasks = isolated_mask_tasks(run_root)
    monkeypatch.setattr(orchestrator, "RUNS_ROOT", run_root)
    monkeypatch.setattr(orchestrator, "build_tasks", lambda *a, **k: list(tasks))
    monkeypatch.setattr(
        orchestrator, "EXPERIMENTS_DIR",
        scratch_dir / "analysis" / "runs" / "material_experiments",
    )
    return SimpleNamespace(
        run_root=run_root, tasks=tasks, by_task={task.task_id: task for task in tasks},
    )


def _protocol_history(epochs: int, warmup_epochs: int) -> list[dict]:
    """Deterministic metrics history whose best epoch is not the last one."""
    peak = max(1, epochs - 10)
    history = []
    for epoch in range(1, epochs + 1):
        if epoch <= peak:
            macro_f1 = 0.30 + 0.01 * epoch
        else:
            macro_f1 = 0.30 + 0.01 * peak - 0.001 * (epoch - peak)
        history.append(
            {
                "epoch": epoch,
                "stage": "warmup" if epoch <= warmup_epochs else "finetune",
                "val_macro_f1": round(macro_f1, 6),
                "val_accuracy": round(macro_f1 + 0.05, 6),
            }
        )
    return history


def fabricate_completed_run(
    task,
    data_root: Path,
    fingerprint: str,
    *,
    seed: int | None = None,
    smoke: bool = False,
    limit_batches: int = 0,
    config_overrides: dict | None = None,
    checkpoint_fingerprint: str | None = None,
    best_epoch: int | None = None,
    history_epochs: int | None = None,
    pretrained_sha256: str = "a" * 64,
) -> Path:
    """Fabricate a completed formal run directory for read-path tests.

    Writes a config that matches the fixed protocol, a metrics history with
    one entry per epoch, and best/last checkpoints carrying strictly valid
    metadata consistent with that config — enough to pass
    ``validate_completed_task``. The run directory must be inside this
    session's ``.test_artifacts`` tree, which proves test ownership by
    resolved path rather than by name; a real run directory is never touched.

    The keyword overrides exist to build deliberately *invalid* runs for the
    admission tests (protocol drift, config/checkpoint drift, a best epoch
    that is not the history's selection, a truncated history).
    """
    import json as json_module

    from models.material_fusion import (
        PREPROCESSING_VERSION,
        UPSTREAM_COMMIT,
        build_material_model,
        material_fusion_metadata,
        save_material_checkpoint,
    )

    seed = task.seed if seed is None else seed
    run_dir = assert_owned(Path(task.run_dir), session_artifacts_root())
    if run_dir.exists():
        raise FileExistsError(f"test run dir already exists: {run_dir}")

    config = {
        "experiment": "material_fusion_v1",
        "variant": task.variant,
        "data": str(data_root),
        "data_fingerprint": fingerprint,
        "pretrained": str(data_root / "mit_b0.pth"),
        "pretrained_sha256": pretrained_sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "preprocessing_version": PREPROCESSING_VERSION,
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
        "seed": seed,
        "workers": 0,
        "fallback_weights": "",
        "smoke": smoke,
        "limit_batches": limit_batches,
    }
    config.update(config_overrides or {})

    history = _protocol_history(int(config["epochs"]), int(config["warmup_epochs"]))
    if history_epochs is not None:
        history = history[:history_epochs]
    selected_epoch, selected_entry = history[-1]["epoch"], history[-1]
    for entry in history:
        key = (entry["val_macro_f1"], entry["val_accuracy"])
        if key > (selected_entry["val_macro_f1"], selected_entry["val_accuracy"]):
            selected_epoch, selected_entry = entry["epoch"], entry
    best_meta_epoch = selected_epoch if best_epoch is None else best_epoch
    best_meta_entry = next(
        (entry for entry in history if entry["epoch"] == best_meta_epoch), selected_entry
    )
    last_entry = history[-1]
    checkpoint_metadata_fingerprint = (
        fingerprint if checkpoint_fingerprint is None else checkpoint_fingerprint
    )

    torch.manual_seed(0)
    model = build_material_model(task.variant, imgsz=int(config["imgsz"]))
    common = dict(
        variant=task.variant, imgsz=int(config["imgsz"]),
        data_fingerprint=checkpoint_metadata_fingerprint,
        pretrained_sha256=config["pretrained_sha256"], seed=seed,
        smoke=bool(config["smoke"]), limit_batches=int(config["limit_batches"]),
        physical_batch=int(config["batch"]), accum_steps=int(config["accum_steps"]),
        effective_batch=int(config["effective_batch"]), drop_path_rate=0.0,
    )
    save_material_checkpoint(
        run_dir / "best.pt", model,
        material_fusion_metadata(
            epoch=best_meta_epoch, stage=best_meta_entry["stage"],
            val_macro_f1=best_meta_entry["val_macro_f1"],
            val_accuracy=best_meta_entry["val_accuracy"], **common,
        ),
    )
    save_material_checkpoint(
        run_dir / "last.pt", model,
        material_fusion_metadata(
            epoch=last_entry["epoch"], stage=last_entry["stage"],
            val_macro_f1=last_entry["val_macro_f1"],
            val_accuracy=last_entry["val_accuracy"], **common,
        ),
    )
    del model
    (run_dir / "metrics_history.jsonl").write_text(
        "".join(json_module.dumps(entry) + "\n" for entry in history), encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json_module.dumps(
            {
                "run_id": task.run_id,
                "config": config,
                "variant": task.variant,
                "dataset_fingerprint": fingerprint,
                "elapsed_seconds": 12.5,
                "peak_vram_mb": 0.0,
                "best": {
                    "epoch": best_meta_epoch, "stage": best_meta_entry["stage"],
                    "val_macro_f1": best_meta_entry["val_macro_f1"],
                    "val_accuracy": best_meta_entry["val_accuracy"],
                },
                "last": {
                    "epoch": last_entry["epoch"], "stage": last_entry["stage"],
                    "val_macro_f1": last_entry["val_macro_f1"],
                    "val_accuracy": last_entry["val_accuracy"],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


# ---------------------------------------------------------------------------
# Synthetic source / V4 / V5 chain
#
# These builders produce a tiny but *real* provenance chain: PNG frames with
# YOLO-seg labels feed the real V4 generation helpers, and the real V5
# generator then derives the instance masks from those same annotations. A
# test can therefore break any link (label, object index, crop, mask) and
# check that the pipeline refuses it.


def build_synthetic_source(
    root: Path,
    counts: tuple[int, int, int] = (4, 4, 4),
    size: int = 64,
    seed: int = 0,
    objects_per_frame: int = 2,
) -> Path:
    """A tiny four-class YOLO-seg source dataset with a pair manifest."""
    import cv2

    rng = np.random.default_rng(seed)
    root = Path(root)
    rows = []
    for split, count in zip(("train", "val", "test"), counts):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "images_right" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"{split}_frame_{index:03d}"
            left = rng.integers(20, 235, size=(size, size), dtype=np.uint8)
            right = np.clip(left.astype(np.int16) + rng.integers(-30, 30, size=left.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(str(root / "images" / split / f"{stem}.png"), left)
            cv2.imwrite(str(root / "images_right" / split / f"{stem}.png"), right)
            lines = []
            for obj in range(objects_per_frame):
                class_id = (index + obj) % len(CLASS_NAMES)
                # distinct, non-overlapping rectangles in normalised coords
                x1 = 0.05 + 0.30 * obj
                y1 = 0.10 + 0.05 * obj
                w = 0.20 - 0.02 * obj
                h = 0.35 - 0.05 * obj
                polygon = (
                    (x1, y1), (x1 + w, y1), (x1 + w, y1 + h), (x1, y1 + h),
                )
                coords = " ".join(f"{v:.6f}" for point in polygon for v in point)
                lines.append(f"{class_id} {coords}")
            (root / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            rows.append(
                {
                    "output_stem": stem,
                    "split": split,
                    "group_name": f"{split}/group_{index}",
                    "left_path": (root / "images" / split / f"{stem}.png").as_posix(),
                    "right_path": (root / "images_right" / split / f"{stem}.png").as_posix(),
                }
            )
    import csv

    with (root / "pair_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["output_stem", "split", "group_name", "left_path", "right_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    (root / "data.yaml").write_text(
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    return root


def build_synthetic_v4(
    source_root: Path,
    v4_root: Path,
    workspace: Path,
    crop_pad: int = 10,
    seed: int = 0,
) -> Path:
    """Build a real V4 dataset (npz + manifest + summary + audit) from a source."""
    import cv2

    from core.fusion_dataset import build_quality_vector, file_digest, save_fusion_sample
    from scripts.make_polar_dataset import make_crop_window, read_yolo_polygons
    from scripts.prepare_cls_fusion_dataset import (
        FusionSampleRecord,
        audit_fusion_dataset,
        build_object_mask,
        load_gray,
        read_class_names,
        read_source_manifest,
    )

    rng = np.random.default_rng(seed)
    source_root = Path(source_root)
    v4_root = Path(v4_root)
    v4_root.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(source_root)
    records = []
    for row in read_source_manifest(source_root, workspace):
        label_path = source_root / "labels" / row.split / f"{row.output_stem}.txt"
        left_gray = load_gray(row.left_path)
        height, width = left_gray.shape[:2]
        polygons = read_yolo_polygons(label_path, width, height)
        for object_index, annotation in enumerate(polygons):
            mask = build_object_mask(annotation.polygon, width, height)
            window = make_crop_window(annotation.polygon, width, height, crop_pad)
            crop_gray = left_gray[window.y1:window.y2, window.x1:window.x2]
            crop_mask = mask[window.y1:window.y2, window.x1:window.x2]
            # valid is a strict subset of the instance mask (a 1-pixel border
            # of the polygon is left unmeasured), so valid_ratio < 1.
            inner = cv2.erode(crop_mask, np.ones((3, 3), np.uint8), iterations=1)
            valid = inner.astype(np.uint8)
            signed = np.zeros(crop_gray.shape, dtype=np.float32)
            values = (rng.random(crop_gray.shape).astype(np.float32) - 0.5)
            signed[valid > 0] = values[valid > 0]
            abs_q = np.abs(signed).astype(np.float32)
            total = int(crop_mask.sum())
            valid_count = int(valid.sum())
            valid_ratio = valid_count / total if total else 0.0
            mean_abs_q = float(abs_q[valid > 0].mean()) if valid_count else 0.0
            quality = build_quality_vector(
                valid_ratio=valid_ratio, in_bounds_ratio=1.0,
                brightness_valid_ratio=1.0, mean_abs_q=mean_abs_q,
            )
            class_name = class_names[annotation.class_id]
            sample_name = f"{row.output_stem}_obj{object_index:03d}"
            relative = Path(row.split) / class_name / f"{sample_name}.npz"
            save_fusion_sample(
                v4_root / relative, gray=crop_gray, signed_q=signed, abs_q=abs_q,
                valid=valid, quality=quality, class_id=annotation.class_id,
            )
            records.append(
                FusionSampleRecord(
                    sample_name=sample_name, split=row.split, group_name=row.group_name,
                    class_id=annotation.class_id, class_name=class_name,
                    source_frame=row.output_stem, object_index=object_index,
                    stereo_valid=True, stereo_reason="ok", stereo_valid_ratio=1.0,
                    disparity=100.0, polar_valid_ratio=valid_ratio,
                    quality=tuple(float(v) for v in quality), npz_path=relative.as_posix(),
                )
            )
    from scripts.prepare_cls_fusion_dataset import write_manifest

    write_manifest(
        v4_root, class_names, records,
        generation={"matching": "bands", "band_margin": 20,
                    "full_image_threshold": 0.9, "crop_pad": crop_pad},
    )
    expected = {
        split: sum(1 for r in records if r.split == split)
        for split in ("train", "val", "test")
    }
    audit_fusion_dataset(
        output_root=v4_root, records=records, class_names=class_names,
        audit_root=v4_root / "audit", expected_splits=expected,
    )
    return v4_root


def build_synthetic_v5(
    source_root: Path,
    v4_root: Path,
    v5_root: Path,
    workspace: Path,
    crop_pad: int = 10,
) -> Path:
    """Run the real V5 generator over a synthetic source + V4 pair."""
    from core.fusion_dataset import dataset_fingerprint
    from scripts.prepare_material_mask_dataset import build_mask_dataset

    build_mask_dataset(
        source_root=source_root, v4_root=v4_root, output_root=v5_root,
        crop_pad=crop_pad, workspace=workspace,
        expected_parent_fingerprint=dataset_fingerprint(v4_root),
    )
    return v5_root


def write_mask_experiment_config(
    path: Path,
    v5_root: Path,
    pretrained: Path,
    fingerprint: str,
    parent_fingerprint: str,
    variant: str = "gray_polar",
    effective_batch: int = 4,
    epochs: int = 2,
    warmup_epochs: int = 1,
    imgsz: int = 64,
) -> Path:
    """A minimal valid ``material_fusion_v2_mask`` config for CPU tests."""
    import yaml

    from core.material_mask_dataset import (
        MASK_RESIZE_VERSION,
        MASK_SCHEMA_VERSION,
    )

    config = {
        "experiment": "material_fusion_v2_mask",
        "variant": variant,
        "data": str(v5_root),
        "data_fingerprint": fingerprint,
        "pretrained": str(pretrained),
        "pretrained_sha256": "",
        "upstream_commit": "e251d860aebc2f583a6c4919877e6bebe7f1aff3",
        "preprocessing_version": "v4_band_224_area_linear_v1",
        "pooling_mode": "instance_mask",
        "mask_schema": MASK_SCHEMA_VERSION,
        "mask_resize_version": MASK_RESIZE_VERSION,
        "parent_data_fingerprint": parent_fingerprint,
        "imgsz": imgsz,
        "batch": effective_batch // 2,
        "accum_steps": 2,
        "effective_batch": effective_batch,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "lr": 1e-4,
        "encoder_lr": 1e-5,
        "seed": 2026,
        "workers": 0,
        "fallback_weights": "",
        "fallback_identity": "",
        "smoke": False,
        "limit_batches": 0,
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_synthetic_mask_training(
    scratch_dir: Path,
    run_id: str,
    v5_root: Path,
    fingerprint: str,
    parent_fingerprint: str,
    variant: str = "gray_polar",
    epochs: int = 2,
    warmup_epochs: int = 1,
    limit_batches: int = 1,
    pretrained: Path | None = None,
) -> dict:
    """Train a real v2 mask-pooling checkpoint on a synthetic V5 dataset.

    Bounded CPU training through the production entry, with the run root
    redirected into the test scratch. Used by tests that need a legitimate
    mask-pooling checkpoint without the real dataset.
    """
    import scripts.train_material_fusion as train_module
    from scripts.train_material_fusion import main as train_main

    scratch = Path(scratch_dir)
    assert_owned(scratch, session_artifacts_root())
    if pretrained is None:
        pretrained = write_synthetic_pretrained(scratch / f"mit_{run_id}.pth")
    config_path = write_mask_experiment_config(
        scratch / f"config_{run_id}.yaml", v5_root, pretrained, fingerprint,
        parent_fingerprint, variant=variant, effective_batch=4, epochs=epochs,
        warmup_epochs=warmup_epochs, imgsz=64,
    )
    run_root = train_run_root(scratch)
    run_dir = material_run_dir(scratch, run_id)
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; use a unique run id")
    with mock.patch.object(train_module, "RUNS_ROOT", run_root):
        exit_code = train_main([
            "--run-id", run_id, "--config", str(config_path),
            "--variant", variant, "--device", "cpu",
            "--epochs", str(epochs), "--warmup-epochs", str(warmup_epochs),
            "--limit-batches", str(limit_batches),
        ])
    return {
        "exit_code": exit_code,
        "run_id": run_id,
        "run_dir": run_dir,
        "config_path": config_path,
        "v5_root": Path(v5_root),
        "fingerprint": fingerprint,
        "parent_fingerprint": parent_fingerprint,
    }


def build_alternate_v5(
    source_v5: Path,
    dst: Path,
    v4_root: Path,
    parent_fingerprint: str,
    crop_pad: int = 10,
) -> Path:
    """A second V5 derivation of the *same* V4 state.

    The mask is changed by moving one pixel *outside* the polar-valid region,
    so the mask pixel count, the valid pixel count and the quality vector are
    all unchanged — the six V4 fields stay element-for-element equal to V4 —
    while the mask content (and therefore the V5 fingerprint) differs. Used to
    prove a run is bound to the concrete V5 dataset, not merely to "some mask
    extension of the same V4".
    """
    import csv as csv_module
    import shutil

    from core.material_mask_dataset import load_mask_sample, save_mask_sample
    from scripts.prepare_material_mask_dataset import (
        audit_mask_dataset,
        records_from_manifest,
    )

    source_v5 = Path(source_v5)
    dst = Path(dst)
    shutil.copytree(source_v5, dst)
    manifest_path = dst / "dataset_manifest.csv"
    with manifest_path.open(encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    row = rows[0]
    path = dst / row["npz_path"]
    sample = load_mask_sample(path)
    mask = sample.instance_mask.copy()
    valid = sample.valid > 0
    removable = np.argwhere((mask > 0) & ~valid)
    addable = np.argwhere((mask == 0) & ~valid)
    if removable.size == 0 or addable.size == 0:
        raise ValueError(
            "the synthetic crop has no mask pixel that can be moved outside the "
            "valid region; a different derivation cannot be built"
        )
    mask[tuple(removable[0])] = 0
    mask[tuple(addable[0])] = 1
    assert int(mask.sum()) == int(sample.instance_mask.sum())
    save_mask_sample(
        path, gray=sample.gray, signed_q=sample.signed_q, abs_q=sample.abs_q,
        valid=sample.valid, quality=sample.quality, class_id=sample.class_id,
        instance_mask=mask,
    )

    records = records_from_manifest(dst)
    expected_splits = {
        split: sum(1 for r in records if r.split == split)
        for split in ("train", "val", "test")
    }
    audit_mask_dataset(
        output_root=dst, records=records, class_names=list(CLASS_NAMES),
        v4_root=Path(v4_root), expected_splits=expected_splits, crop_pad=crop_pad,
        workspace=Path(v4_root).resolve().parent.parent,
        expected_parent_fingerprint=parent_fingerprint,
    )
    return dst


# ---------------------------------------------------------------------------
# v2 (material_fusion_v2_mask) completed-run fabrication


def fabricate_completed_mask_run(
    task,
    v5_root: Path,
    fingerprint: str,
    parent_fingerprint: str,
    *,
    seed: int | None = None,
    smoke: bool = False,
    limit_batches: int = 0,
    config_overrides: dict | None = None,
    checkpoint_fingerprint: str | None = None,
    checkpoint_parent_fingerprint: str | None = None,
    best_epoch: int | None = None,
    history_epochs: int | None = None,
    init_from: str = "",
    mask_schema: str | None = None,
    mask_resize_version: str | None = None,
    pretrained_sha256: str = "a" * 64,
) -> Path:
    """Fabricate a completed *v2 mask-pooling* run directory for read-path tests.

    Writes a config that matches the fixed v2 protocol, a 30-epoch metrics
    history, and best/last checkpoints carrying strictly valid
    instance-mask-pooling metadata consistent with that config. The run
    directory must be inside this session's ``.test_artifacts`` tree; a real
    run directory is never touched.

    The keyword overrides build deliberately *invalid* runs (protocol drift,
    config/checkpoint drift, a wrong best epoch, a truncated history, a
    non-empty ``init_from``).
    """
    import json as json_module

    from core.material_mask_dataset import MASK_RESIZE_VERSION, MASK_SCHEMA_VERSION
    from models.material_fusion import (
        MATERIAL_FUSION_MASK_VERSION,
        POOLING_INSTANCE_MASK,
        PREPROCESSING_VERSION,
        UPSTREAM_COMMIT,
        build_material_model,
        material_fusion_metadata,
        save_material_checkpoint,
    )

    seed = task.seed if seed is None else seed
    run_dir = assert_owned(Path(task.run_dir), session_artifacts_root())
    if run_dir.exists():
        raise FileExistsError(f"test run dir already exists: {run_dir}")

    config = {
        "experiment": MATERIAL_FUSION_MASK_VERSION,
        "variant": task.variant,
        "data": str(v5_root),
        "data_fingerprint": fingerprint,
        "pretrained": str(Path(v5_root) / "mit_b0.pth"),
        "pretrained_sha256": pretrained_sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "preprocessing_version": PREPROCESSING_VERSION,
        "pooling_mode": POOLING_INSTANCE_MASK,
        "mask_schema": MASK_SCHEMA_VERSION if mask_schema is None else mask_schema,
        "mask_resize_version": (
            MASK_RESIZE_VERSION if mask_resize_version is None else mask_resize_version
        ),
        "parent_data_fingerprint": parent_fingerprint,
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
        "seed": seed,
        "workers": 0,
        "fallback_weights": "",
        "smoke": smoke,
        "limit_batches": limit_batches,
    }
    config.update(config_overrides or {})

    history = _protocol_history(int(config["epochs"]), int(config["warmup_epochs"]))
    if history_epochs is not None:
        history = history[:history_epochs]
    selected_epoch, selected_entry = history[-1]["epoch"], history[-1]
    for entry in history:
        key = (entry["val_macro_f1"], entry["val_accuracy"])
        if key > (selected_entry["val_macro_f1"], selected_entry["val_accuracy"]):
            selected_epoch, selected_entry = entry["epoch"], entry
    best_meta_epoch = selected_epoch if best_epoch is None else best_epoch
    best_meta_entry = next(
        (entry for entry in history if entry["epoch"] == best_meta_epoch), selected_entry
    )
    last_entry = history[-1]

    torch.manual_seed(0)
    model = build_material_model(
        task.variant, imgsz=int(config["imgsz"]), pooling_mode=POOLING_INSTANCE_MASK
    )
    common = dict(
        version=MATERIAL_FUSION_MASK_VERSION,
        variant=task.variant,
        imgsz=int(config["imgsz"]),
        data_fingerprint=(
            fingerprint if checkpoint_fingerprint is None else checkpoint_fingerprint
        ),
        pretrained_sha256=config["pretrained_sha256"],
        seed=seed,
        smoke=bool(config["smoke"]),
        limit_batches=int(config["limit_batches"]),
        physical_batch=int(config["batch"]),
        accum_steps=int(config["accum_steps"]),
        effective_batch=int(config["effective_batch"]),
        drop_path_rate=0.0,
        init_from=init_from,
        pooling_mode=POOLING_INSTANCE_MASK,
        mask_schema=config["mask_schema"],
        mask_resize_version=config["mask_resize_version"],
        parent_data_fingerprint=(
            config["parent_data_fingerprint"]
            if checkpoint_parent_fingerprint is None
            else checkpoint_parent_fingerprint
        ),
    )
    save_material_checkpoint(
        run_dir / "best.pt", model,
        material_fusion_metadata(
            epoch=best_meta_epoch, stage=best_meta_entry["stage"],
            val_macro_f1=best_meta_entry["val_macro_f1"],
            val_accuracy=best_meta_entry["val_accuracy"], **common,
        ),
    )
    save_material_checkpoint(
        run_dir / "last.pt", model,
        material_fusion_metadata(
            epoch=last_entry["epoch"], stage=last_entry["stage"],
            val_macro_f1=last_entry["val_macro_f1"],
            val_accuracy=last_entry["val_accuracy"], **common,
        ),
    )
    del model
    (run_dir / "metrics_history.jsonl").write_text(
        "".join(json_module.dumps(entry) + "\n" for entry in history), encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json_module.dumps(
            {
                "run_id": task.run_id,
                "config": config,
                "variant": task.variant,
                "dataset_fingerprint": fingerprint,
                "pooling_mode": POOLING_INSTANCE_MASK,
                "mask_schema": config["mask_schema"],
                "mask_resize_version": config["mask_resize_version"],
                "parent_data_fingerprint": config["parent_data_fingerprint"],
                "init_from": init_from,
                "version": MATERIAL_FUSION_MASK_VERSION,
                "elapsed_seconds": 12.5,
                "peak_vram_mb": 0.0,
                "best": {
                    "epoch": best_meta_epoch, "stage": best_meta_entry["stage"],
                    "val_macro_f1": best_meta_entry["val_macro_f1"],
                    "val_accuracy": best_meta_entry["val_accuracy"],
                },
                "last": {
                    "epoch": last_entry["epoch"], "stage": last_entry["stage"],
                    "val_macro_f1": last_entry["val_macro_f1"],
                    "val_accuracy": last_entry["val_accuracy"],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def write_v2_checkpoint_payload(
    path: Path,
    *,
    variant: str = "gray_polar",
    data_fingerprint: str = "",
    pooling_mode: str = "instance_mask",
    parent_data_fingerprint: str = "8" * 64,
) -> Path:
    """Write a raw v2 checkpoint payload, bypassing the strict save path.

    Used to build the deliberately-invalid artifacts the admission tests need
    (an empty or malformed fingerprint, a contradictory pooling mode) that
    ``save_material_checkpoint`` now refuses to produce.
    """
    from core.material_mask_dataset import MASK_RESIZE_VERSION, MASK_SCHEMA_VERSION
    from models.material_fusion import (
        MATERIAL_FUSION_MASK_VERSION,
        PREPROCESSING_VERSION,
        UPSTREAM_COMMIT,
        build_material_model,
    )

    torch.manual_seed(0)
    model = build_material_model(variant, imgsz=224, pooling_mode=pooling_mode)
    payload = {
        "metadata": {
            "version": MATERIAL_FUSION_MASK_VERSION,
            "variant": variant,
            "class_names": list(CLASS_NAMES),
            "imgsz": 224,
            "preprocessing_version": PREPROCESSING_VERSION,
            "data_fingerprint": data_fingerprint,
            "upstream_commit": UPSTREAM_COMMIT,
            "pretrained_sha256": "a" * 64,
            "seed": 2026,
            "epoch": 1,
            "stage": "warmup",
            "smoke": False,
            "limit_batches": 0,
            "val_macro_f1": 0.5,
            "val_accuracy": 0.5,
            "physical_batch": 8,
            "accum_steps": 2,
            "effective_batch": 16,
            "drop_path_rate": 0.0,
            "init_from": "",
            "fallback_weights": "",
            "fallback_identity": "",
            "pooling_mode": pooling_mode,
            "mask_schema": MASK_SCHEMA_VERSION,
            "mask_resize_version": MASK_RESIZE_VERSION,
            "parent_data_fingerprint": parent_data_fingerprint,
        },
        "state_dict": model.state_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


# ---------------------------------------------------------------------------
# Test protocol specs
#
# The production admission rules compare a run against the *chosen protocol's*
# identity (dataset, V4 parent, pretrained weight). A synthetic fixture
# therefore has to be validated against a protocol whose identity matches it —
# never against a relaxed rule set. These helpers derive such a spec from a
# real one (inheriting the fixed budget, the pooling/mask contract and the
# run-id prefix) and point only the dataset/pretrained identity at the
# fixture. Production defaults are never modified and there is no CLI bypass.


def v4_dataset_fingerprint(data_root: Path) -> str:
    """The audited V4 fingerprint of a synthetic dataset."""
    from scripts.train_polar_fusion import validate_dataset

    return validate_dataset(data_root)["audit_fingerprint"]


def build_test_protocol_spec(
    scratch: Path,
    base_spec,
    *,
    data_fingerprint: str,
    parent_data_fingerprint: str | None = None,
    pretrained_sha256: str | None = None,
    pretrained: str | None = None,
    data: str | None = None,
    variant: str | None = None,
    label: str = "",
):
    """A ProtocolSpec whose identity matches a synthetic fixture."""
    import dataclasses

    import yaml

    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / base_spec.config).read_text(encoding="utf-8"))
    if variant is not None:
        config["variant"] = variant
    if data is not None:
        config["data"] = str(data)
    config["data_fingerprint"] = data_fingerprint
    if pretrained is not None:
        config["pretrained"] = str(pretrained)
    if pretrained_sha256 is not None:
        config["pretrained_sha256"] = pretrained_sha256
    if parent_data_fingerprint is not None and "parent_data_fingerprint" in config:
        config["parent_data_fingerprint"] = parent_data_fingerprint
    suffix = f"_{label}" if label else ""
    path = scratch / f"test_protocol_{base_spec.name}{suffix}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return dataclasses.replace(
        base_spec,
        config=str(path),
        data=str(config["data"]),
        data_fingerprint=str(config["data_fingerprint"]),
        parent_data_fingerprint=str(config.get("parent_data_fingerprint", "")),
        pretrained=str(config["pretrained"]),
        pretrained_sha256=str(config["pretrained_sha256"]),
    )


def install_test_protocol(monkeypatch, spec) -> None:
    """Make ``protocol_spec(spec.name)`` resolve to ``spec`` inside one test."""
    from scripts import run_material_experiments as orchestrator

    monkeypatch.setitem(orchestrator.PROTOCOLS, spec.name, spec)
