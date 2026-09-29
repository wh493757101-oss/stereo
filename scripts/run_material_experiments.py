"""Orchestrator for the CMX material-fusion fair-comparison matrix.

Two experiment protocols share this entry point, selected explicitly with
``--protocol`` (default ``v1``, so every old command keeps its meaning):

- ``v1``      — ``material_fusion_v1``, the V4 dataset, GAP head, run ids
                ``cmx_v1_<variant>_s<seed>``, management artifacts under
                ``analysis/runs/material_experiments/``.
- ``v2_mask`` — ``material_fusion_v2_mask``, the V5 instance-mask dataset,
                instance-mask pooling, run ids
                ``cmx_v2_mask_<variant>_s<seed>``, management artifacts under
                ``analysis/runs/material_experiments_v2_mask/``.

The two protocols never share a task matrix, a run manifest or a management
directory: a run id, a manifest entry or a completion verdict from one is
never accepted as evidence for the other. The protocol is carried through
every function as an explicit :class:`ProtocolSpec`; no mode mutates a module
global to switch protocols.

Four modes with strictly separated responsibilities:

- ``plan``      — print/write the 12-task matrix (4 variants x 3 seeds),
                  commands, output dirs and the resource basis. Read-only.
- ``preflight`` — read-only admission checks (protocol expectations,
                  dataset fingerprint, pretrained SHA, fallback weights,
                  output-dir collisions). Never trains.
- ``run``       — explicitly start a chosen task subset, serially, via
                  ``scripts/train_material_fusion.py``. Re-runs the essential
                  admission before the first spawn (a saved preflight report
                  is never trusted as a substitute), stops on the first
                  failure and records every attempt (command, exit code,
                  start/end time). Existing directories are refused unless
                  ``--skip-completed`` is given, and even then only runs that
                  pass full completion validation for *this* protocol are
                  skipped.
- ``collect``   — read-only aggregation of completed runs plus the
                  authorized train/val inference report. Smoke artifacts
                  are excluded, the test split is never read, no optimizer
                  or parameter update ever happens.

Auxiliary bias baselines (``scripts/fit_bias_baseline.py``) and the
shuffled-polarization ablation (``scripts/ablate_material_shuffle.py``)
are separate, explicitly-invoked scripts; they are listed by ``plan`` but
never started by any mode here.

Usage:
    python scripts/run_material_experiments.py plan
    python scripts/run_material_experiments.py plan --protocol v2_mask
    python scripts/run_material_experiments.py preflight --device 0
    python scripts/run_material_experiments.py preflight --protocol v2_mask \\
        --device 0 --output analysis/runs/material_experiments_v2_mask/preflight.json
    python scripts/run_material_experiments.py run --device 0 \\
        --tasks gray_s2026,gray_gray_valid_s2026,gray_valid_s2026,gray_polar_s2026
    python scripts/run_material_experiments.py collect --device 0

``run`` takes an explicit ``--tasks`` list of ``<variant>_s<seed>`` ids; it
has no seed shortcut, and the task ids use underscores, not colons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_fusion_dataset import CLASS_NAMES, VARIANTS, validate_variant
from core.material_mask_dataset import (
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    POOLING_GAP,
    POOLING_INSTANCE_MASK,
    V4_PARENT_FINGERPRINT,
    require_fingerprint,
)
from models.material_fusion import (
    MATERIAL_FUSION_MASK_VERSION,
    MATERIAL_FUSION_VERSION,
    check_run_config_checkpoint_consistency,
    expected_stage_for_epoch,
    read_material_checkpoint_metadata,
    require_str,
    validate_material_config,
)
from models.polar_fusion import read_manifest_split, torch_device_name

TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_material_fusion.py"
RUNS_ROOT = PROJECT_ROOT / "runs" / "train"

# v1 management artifacts. Both names stay module-level (and are read at call
# time) because the test isolation hooks redirect exactly these two paths.
EXPERIMENTS_DIR = PROJECT_ROOT / "analysis" / "runs" / "material_experiments"
PLAN_PATH = EXPERIMENTS_DIR / "plan.json"
RUN_MANIFEST_PATH = EXPERIMENTS_DIR / "run_manifest.json"

SEEDS = (2026, 2027, 2028)
RUN_SUBDIR = "material_fusion"

DEFAULT_PROTOCOL = "v1"
PROTOCOL_CHOICES = ("v1", "v2_mask")

# The fixed training protocol every formal run must use (task-defined
# constants). A config that differs from these values is reported, never
# silently used. Identical for both protocols except for the experiment name
# and the pooling mode, so a v2 run can never be validated with v1 rules.
_COMMON_PROTOCOL = {
    "imgsz": 224,
    "batch": 8,
    "accum_steps": 2,
    "effective_batch": 16,
    "epochs": 30,
    "warmup_epochs": 2,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "lr": 0.0001,
    "encoder_lr": 0.00001,
    "workers": 0,
    "smoke": False,
    "limit_batches": 0,
}

V1_DATA_FINGERPRINT = (
    "8db8998d7a4c280064626961e4d51c92c531878ea2e92f0d1480df46379bd334"
)
V2_DATA_FINGERPRINT = (
    "bb9e3661fb1e71e94056893f6dfc50b9e3dee1c604f430ac0727d75362d32982"
)
PRETRAINED_SHA256 = (
    "4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc"
)


@dataclass(frozen=True)
class ProtocolSpec:
    """Everything that differs between the two experiment protocols.

    Carried explicitly into plan/preflight/run/collect so no mode switches
    protocol by mutating a module global, and so a task id that happens to be
    spelled the same in both matrices can never be cross-read.
    """

    name: str
    experiment: str
    config: str
    run_id_prefix: str
    management_dir: str
    pooling_mode: str
    mask_schema: str
    mask_resize_version: str
    data: str
    data_fingerprint: str
    parent_data_fingerprint: str
    pretrained: str
    pretrained_sha256: str
    expected: Mapping

    @property
    def run_subdir(self) -> str:
        return RUN_SUBDIR


def _spec(
    *,
    name: str,
    experiment: str,
    config: str,
    run_id_prefix: str,
    management_dir: str,
    pooling_mode: str,
    data: str,
    data_fingerprint: str,
    parent_data_fingerprint: str,
    mask_schema: str = "",
    mask_resize_version: str = "",
) -> ProtocolSpec:
    return ProtocolSpec(
        name=name,
        experiment=experiment,
        config=config,
        run_id_prefix=run_id_prefix,
        management_dir=management_dir,
        pooling_mode=pooling_mode,
        mask_schema=mask_schema,
        mask_resize_version=mask_resize_version,
        data=data,
        data_fingerprint=data_fingerprint,
        parent_data_fingerprint=parent_data_fingerprint,
        pretrained="weights/pretrained/mit_b0.pth",
        pretrained_sha256=PRETRAINED_SHA256,
        expected=MappingProxyType(
            {"experiment": experiment, "pooling_mode": pooling_mode, **_COMMON_PROTOCOL}
        ),
    )


V1_SPEC = _spec(
    name="v1",
    experiment=MATERIAL_FUSION_VERSION,
    config="configs/experiments/material_fusion_v1.yaml",
    run_id_prefix="cmx_v1",
    management_dir="material_experiments",
    pooling_mode=POOLING_GAP,
    data="datasets/underwater_cls_fusion_v4_band",
    data_fingerprint=V1_DATA_FINGERPRINT,
    parent_data_fingerprint="",
)

V2_MASK_SPEC = _spec(
    name="v2_mask",
    experiment=MATERIAL_FUSION_MASK_VERSION,
    config="configs/experiments/material_fusion_v2_mask.yaml",
    run_id_prefix="cmx_v2_mask",
    management_dir="material_experiments_v2_mask",
    pooling_mode=POOLING_INSTANCE_MASK,
    data="datasets/underwater_cls_fusion_v5_instance_mask",
    data_fingerprint=V2_DATA_FINGERPRINT,
    parent_data_fingerprint=V4_PARENT_FINGERPRINT,
    mask_schema=MASK_SCHEMA_VERSION,
    mask_resize_version=MASK_RESIZE_VERSION,
)

PROTOCOLS: dict[str, ProtocolSpec] = {
    spec.name: spec for spec in (V1_SPEC, V2_MASK_SPEC)
}

# Backwards-compatible alias: the v1 fixed protocol constants, as before.
EXPECTED_PROTOCOL: dict = dict(V1_SPEC.expected)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2026


def protocol_spec(name: str) -> ProtocolSpec:
    """Resolve a ``--protocol`` value to its spec (unknown values refused)."""
    if not isinstance(name, str) or name not in PROTOCOLS:
        raise ValueError(
            f"unknown protocol {name!r}; expected one of {sorted(PROTOCOLS)}"
        )
    return PROTOCOLS[name]


def management_dir(spec: ProtocolSpec) -> Path:
    """This protocol's management-artifact directory.

    v1 keeps the original module-level path (so existing test redirects of
    ``EXPERIMENTS_DIR`` keep working); v2 gets its own sibling directory, and
    neither is ever reused for the other protocol.
    """
    if spec.name == DEFAULT_PROTOCOL:
        return Path(EXPERIMENTS_DIR)
    return Path(EXPERIMENTS_DIR).parent / spec.management_dir


def _resolve_report_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def plan_path(spec: ProtocolSpec, override: str | None = None) -> Path:
    if override is not None:
        return _resolve_report_path(override, PROJECT_ROOT / "plan.json")
    if spec.name == DEFAULT_PROTOCOL:
        return Path(PLAN_PATH)
    return management_dir(spec) / "plan.json"


def preflight_path(spec: ProtocolSpec, override: str | None = None) -> Path:
    if override is not None:
        return _resolve_report_path(override, PROJECT_ROOT / "preflight.json")
    return management_dir(spec) / "preflight.json"


def run_manifest_path(spec: ProtocolSpec) -> Path:
    if spec.name == DEFAULT_PROTOCOL:
        return Path(RUN_MANIFEST_PATH)
    return management_dir(spec) / "run_manifest.json"


# ---------------------------------------------------------------------------
# Task matrix


@dataclass(frozen=True)
class ExperimentTask:
    task_id: str
    variant: str
    seed: int
    run_id: str
    run_dir: str
    protocol: str = DEFAULT_PROTOCOL


def build_tasks(
    variants: Sequence[str] = VARIANTS,
    seeds: Sequence[int] = SEEDS,
    protocol: str = DEFAULT_PROTOCOL,
) -> list[ExperimentTask]:
    """The full matrix, seed-major so the seed-2026 sanity block comes first.

    Run directories are derived from the module-level ``RUNS_ROOT`` at call
    time, so tests can redirect every task into a scratch directory without
    touching ``runs/train``. The run id carries the protocol prefix, so a v1
    and a v2 task with the same ``task_id`` can never share a directory.
    """
    spec = protocol_spec(protocol)
    seen: set[tuple[str, int]] = set()
    tasks: list[ExperimentTask] = []
    for seed in seeds:
        if type(seed) is not int:
            raise ValueError(f"seed must be an int, got {seed!r}")
        for variant in variants:
            validate_variant(variant)
            key = (variant, seed)
            if key in seen:
                raise ValueError(f"duplicate matrix entry: {key}")
            seen.add(key)
            run_id = f"{spec.run_id_prefix}_{variant}_s{seed}"
            tasks.append(
                ExperimentTask(
                    task_id=f"{variant}_s{seed}",
                    variant=variant,
                    seed=seed,
                    run_id=run_id,
                    run_dir=str(RUNS_ROOT / run_id / RUN_SUBDIR),
                    protocol=spec.name,
                )
            )
    if len({task.run_id for task in tasks}) != len(tasks):
        raise ValueError("run ids are not unique across the matrix")
    return tasks


def task_subset(tasks: list[ExperimentTask], requested: Sequence[str]) -> list[ExperimentTask]:
    available = {task.task_id: task for task in tasks}
    selected: list[ExperimentTask] = []
    for item in requested:
        if item not in available:
            raise ValueError(
                f"unknown task {item!r}; valid tasks: {sorted(available)}"
            )
        if available[item] not in selected:
            selected.append(available[item])
    return selected


def parse_task_list(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError(
            "--tasks must name at least one task id of the form "
            "<variant>_s<seed>, e.g. gray_s2026"
        )
    return items


def train_command(
    task: ExperimentTask, device: str, spec: ProtocolSpec | None = None
) -> list[str]:
    """The exact subprocess command for one task, in its own protocol.

    The command never carries ``--init-from`` (both protocols initialise from
    the official pretrained weights) and never carries a smoke truncation
    flag: formal runs use the config's own budget.
    """
    spec = spec or protocol_spec(task.protocol)
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--run-id", task.run_id,
        "--config", spec.config,
        "--variant", task.variant,
        "--seed", str(task.seed),
        "--device", device,
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_exclusive(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; move or delete it explicitly instead of "
            "letting the orchestrator overwrite recorded history"
        )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Completion validation (used by run --skip-completed and collect)


def select_best_epoch(history: Sequence[dict]) -> tuple[int, dict]:
    """The protocol's best-epoch selection over a metrics history.

    Mirrors ``scripts.train_material_fusion`` exactly (val Macro-F1 primary,
    accuracy tiebreak, an exact tie keeps the earlier epoch) so a recorded
    best epoch can be checked against the history that produced it instead
    of being trusted.
    """
    from scripts.train_material_fusion import selection_key, should_replace_best

    best_key = (-1.0, -1.0)
    best_epoch, best_entry = 0, {}
    for entry in history:
        key = selection_key({"macro_f1": entry["val_macro_f1"],
                             "accuracy": entry["val_accuracy"]})
        if should_replace_best(best_key, key):
            best_key, best_epoch, best_entry = key, int(entry["epoch"]), entry
    return best_epoch, best_entry


def validate_completed_task(
    task: ExperimentTask, spec: ProtocolSpec | None = None
) -> tuple[bool, str]:
    """Full validation that a run directory is a completed formal run.

    A matching directory or file name is not success. The run must be
    reproducible from its own recorded evidence:

    - the run belongs to *this* task's protocol (the expected protocol, never
      the artifact's own claim): the recorded experiment, the run-id prefix,
      the pooling mode and the mask schema/resize versions all match;
    - the resolved config matches the fixed protocol field by field;
    - the run's identity is the chosen protocol's identity: the config's
      ``data_fingerprint``, ``parent_data_fingerprint`` and
      ``pretrained_sha256`` each equal the :class:`ProtocolSpec`'s, so an
      internally consistent run of another dataset state or another
      pretrained file is refused;
    - ``run_config.json`` records the same dataset fingerprint as the config;
    - the metrics history is contiguous from epoch 1 to the configured epoch
      count, with the protocol stage at every epoch;
    - both checkpoints agree with the config on variant, image size,
      preprocessing version, upstream commit, seed, batch geometry, pretrained
      SHA, dataset fingerprint, pooling mode and the mask protocol;
    - a v2 run carries no ``init_from`` (it must not inherit another run's
      training history);
    - the recorded best epoch *is* the history's selection, and the best/last
      metrics match the history entry they claim to come from;
    - ``run_config.json``'s best/last records match the checkpoint metadata.

    v1 and v2 are never each other's evidence: the rules come from ``spec``,
    so a v2 directory is refused by a v1 task and vice versa even though the
    two matrices share task ids.
    """
    spec = spec or protocol_spec(task.protocol)
    run_dir = Path(task.run_dir)
    if not run_dir.is_dir():
        return False, "run directory does not exist"
    for name in ("run_config.json", "metrics_history.jsonl", "best.pt", "last.pt"):
        if not (run_dir / name).is_file():
            return False, f"missing {name}"
    try:
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        config = validate_material_config(run_config["config"])
        if config["variant"] != task.variant:
            return False, f"run config variant {config['variant']!r} != task {task.variant!r}"
        if config["seed"] != task.seed:
            return False, f"run config seed {config['seed']} != task {task.seed}"
        if config["experiment"] != spec.experiment:
            return False, (
                f"run config experiment {config['experiment']!r} is not this "
                f"task's protocol {spec.experiment!r}; a run of one protocol is "
                "never evidence for the other"
            )
        protocol_ok, differences = protocol_check(config, spec.expected)
        if not protocol_ok:
            return False, (
                "run config does not match the fixed protocol: " + "; ".join(differences)
            )
        if not task.run_id.startswith(f"{spec.run_id_prefix}_"):
            return False, (
                f"task run id {task.run_id!r} does not carry this protocol's "
                f"{spec.run_id_prefix!r} prefix"
            )
        recorded_run_id = run_config.get("run_id")
        if recorded_run_id != task.run_id:
            return False, (
                f"run_config run_id {recorded_run_id!r} does not match the task's "
                f"run id {task.run_id!r}"
            )
        # The run's identity must be the *chosen protocol's* identity, not merely
        # internally consistent. A complete, self-consistent run of another
        # dataset state or another pretrained file is not this experiment's
        # evidence, and the artifact's own claim never decides which identities
        # are acceptable.
        if config["data_fingerprint"] != spec.data_fingerprint:
            return False, (
                f"run config data_fingerprint {config['data_fingerprint']!r} is not "
                f"the {spec.name!r} protocol's dataset {spec.data_fingerprint!r}"
            )
        if config["pretrained_sha256"] != spec.pretrained_sha256:
            return False, (
                f"run config pretrained_sha256 {config['pretrained_sha256']!r} is not "
                f"the {spec.name!r} protocol's pretrained weight "
                f"{spec.pretrained_sha256!r}"
            )
        if config["parent_data_fingerprint"] != spec.parent_data_fingerprint:
            return False, (
                f"run config parent_data_fingerprint "
                f"{config['parent_data_fingerprint']!r} is not the {spec.name!r} "
                f"protocol's parent {spec.parent_data_fingerprint!r}"
            )
        recorded_fingerprint = run_config.get("dataset_fingerprint")
        if recorded_fingerprint != config["data_fingerprint"]:
            return False, (
                f"run_config dataset_fingerprint {recorded_fingerprint!r} does not "
                f"match the config data_fingerprint {config['data_fingerprint']!r}"
            )
        if config["mask_schema"] != spec.mask_schema:
            return False, (
                f"run config mask_schema {config['mask_schema']!r} is not this "
                f"protocol's {spec.mask_schema!r}"
            )
        if config["mask_resize_version"] != spec.mask_resize_version:
            return False, (
                f"run config mask_resize_version "
                f"{config['mask_resize_version']!r} is not this protocol's "
                f"{spec.mask_resize_version!r}"
            )
        recorded_parent = run_config.get("parent_data_fingerprint", "")
        if config["parent_data_fingerprint"] != recorded_parent:
            return False, (
                f"run_config parent_data_fingerprint {recorded_parent!r} does not "
                "match the config parent_data_fingerprint "
                f"{config['parent_data_fingerprint']!r}"
            )
        if spec.pooling_mode == POOLING_INSTANCE_MASK:
            # The identities that bind a v2 run to its concrete V5 dataset must
            # be present and well-formed, not merely absent from a comparison.
            require_fingerprint(config["data_fingerprint"], "run config data_fingerprint")
            require_fingerprint(
                config["parent_data_fingerprint"], "run config parent_data_fingerprint"
            )
            require_fingerprint(
                recorded_fingerprint, "run_config dataset_fingerprint"
            )
            if run_config.get("init_from"):
                return False, (
                    "a material_fusion_v2_mask run must not continue from another "
                    "run's weights (init_from must be empty), got "
                    f"{run_config['init_from']!r}"
                )

        history = [
            json.loads(line)
            for line in (run_dir / "metrics_history.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not history:
            return False, "metrics history is empty"
        epochs = config["epochs"]
        if len(history) != epochs:
            return False, (
                f"metrics history has {len(history)} entries, expected one per "
                f"epoch ({epochs})"
            )
        for index, entry in enumerate(history):
            if entry.get("epoch") != index + 1:
                return False, "metrics history epochs are not contiguous from 1"
            expected_stage = expected_stage_for_epoch(entry["epoch"], config["warmup_epochs"])
            if entry.get("stage") != expected_stage:
                return False, (
                    f"metrics history epoch {entry['epoch']} stage "
                    f"{entry.get('stage')!r} does not match the protocol stage "
                    f"{expected_stage!r}"
                )

        metadata = {}
        for role in ("best", "last"):
            meta = read_material_checkpoint_metadata(run_dir / f"{role}.pt")
            check_run_config_checkpoint_consistency(config, meta)
            if spec.pooling_mode == POOLING_INSTANCE_MASK and meta.init_from:
                return False, (
                    f"{role}.pt records init_from {meta.init_from!r}; a v2 run "
                    "must not continue from another run's weights"
                )
            metadata[role] = meta

        best_epoch, best_entry = select_best_epoch(history)
        if metadata["best"].epoch != best_epoch:
            return False, (
                f"best.pt records epoch {metadata['best'].epoch} but the history "
                f"selects epoch {best_epoch}"
            )
        for role, entry in (("best", best_entry), ("last", history[-1])):
            meta = metadata[role]
            if (meta.val_macro_f1, meta.val_accuracy) != (
                entry["val_macro_f1"], entry["val_accuracy"],
            ):
                return False, (
                    f"{role}.pt metrics ({meta.val_macro_f1}, {meta.val_accuracy}) "
                    f"do not match metrics history epoch {entry['epoch']} "
                    f"({entry['val_macro_f1']}, {entry['val_accuracy']})"
                )
        if metadata["last"].epoch != epochs:
            return False, (
                f"last.pt records epoch {metadata['last'].epoch}, expected the "
                f"run's final epoch {epochs}"
            )
        for role in ("best", "last"):
            recorded = run_config.get(role)
            if not isinstance(recorded, dict):
                return False, f"run_config.json is missing the {role} record"
            meta = metadata[role]
            if recorded.get("epoch") != meta.epoch or recorded.get("stage") != meta.stage:
                return False, (
                    f"run_config {role} record (epoch {recorded.get('epoch')}, stage "
                    f"{recorded.get('stage')!r}) does not match {role}.pt "
                    f"(epoch {meta.epoch}, stage {meta.stage!r})"
                )
            if (float(recorded.get("val_macro_f1", float("nan"))),
                    float(recorded.get("val_accuracy", float("nan")))) != (
                        meta.val_macro_f1, meta.val_accuracy):
                return False, f"run_config {role} metrics do not match {role}.pt"
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        return False, f"validation failed: {exc}"
    return True, "ok"


# ---------------------------------------------------------------------------
# plan


def dataset_resource_basis(spec: ProtocolSpec) -> dict:
    """Sample/batch/step counts from this protocol's own dataset manifest."""
    data_root = PROJECT_ROOT / spec.data
    train_paths, _, _ = read_manifest_split(data_root, "train")
    val_paths, _, _ = read_manifest_split(data_root, "val")
    expected = spec.expected
    train_batches = math.ceil(len(train_paths) / expected["batch"])
    steps_per_epoch = math.ceil(train_batches / expected["accum_steps"])
    return {
        "data_root": spec.data,
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
        "train_batches_per_epoch": train_batches,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "optimizer_steps_per_run": steps_per_epoch * expected["epochs"],
        "epochs": expected["epochs"],
        "effective_batch": expected["effective_batch"],
    }


def smoke_time_reference(spec: ProtocolSpec) -> list[dict]:
    """Elapsed seconds of recorded smoke runs of *this* protocol (planning basis).

    These are observed wall-clock times of truncated smoke runs. They are not
    a formal-run duration estimate and they are not a deployment FPS claim.
    """
    references = []
    for run_config_path in sorted(PROJECT_ROOT.glob("runs/train/*/material_fusion/run_config.json")):
        try:
            payload = json.loads(run_config_path.read_text(encoding="utf-8"))
            config = payload.get("config", {})
            if config.get("smoke") and config.get("experiment") == spec.experiment:
                references.append(
                    {
                        "run_id": payload.get("run_id", ""),
                        "variant": config.get("variant", ""),
                        "limit_batches": config.get("limit_batches"),
                        "epochs": config.get("epochs"),
                        "elapsed_seconds": payload.get("elapsed_seconds"),
                    }
                )
        except (json.JSONDecodeError, OSError):
            continue
    return references


def build_plan_payload(
    tasks: list[ExperimentTask], device: str, spec: ProtocolSpec
) -> dict:
    expected = spec.expected
    return {
        "generated_at": _utc_now(),
        "protocol": spec.name,
        "experiment": spec.experiment,
        "config": spec.config,
        "data": {
            "root": spec.data,
            "data_fingerprint": spec.data_fingerprint,
            "parent_data_fingerprint": spec.parent_data_fingerprint,
        },
        "pooling": {
            "pooling_mode": spec.pooling_mode,
            "mask_schema": spec.mask_schema,
            "mask_resize_version": spec.mask_resize_version,
        },
        "pretrained": {
            "path": spec.pretrained,
            "sha256": spec.pretrained_sha256,
        },
        "fixed_budget": {
            "imgsz": expected["imgsz"],
            "batch": expected["batch"],
            "accum_steps": expected["accum_steps"],
            "effective_batch": expected["effective_batch"],
            "epochs": expected["epochs"],
            "warmup_epochs": expected["warmup_epochs"],
            "optimizer": expected["optimizer"],
            "weight_decay": expected["weight_decay"],
            "lr": expected["lr"],
            "encoder_lr": expected["encoder_lr"],
            "workers": expected["workers"],
            "smoke": expected["smoke"],
            "limit_batches": expected["limit_batches"],
        },
        "management_dir": str(management_dir(spec)),
        "matrix": {"variants": list(VARIANTS), "seeds": list(SEEDS), "tasks": len(tasks)},
        "tasks": [
            {
                "task_id": task.task_id,
                "protocol": task.protocol,
                "experiment": spec.experiment,
                "variant": task.variant,
                "seed": task.seed,
                "run_id": task.run_id,
                "run_dir": task.run_dir,
                "command": " ".join(train_command(task, device, spec)),
            }
            for task in tasks
        ],
        "resource_basis": dataset_resource_basis(spec),
        "smoke_time_reference": smoke_time_reference(spec),
        "smoke_time_reference_note": (
            "Observed wall-clock times of truncated smoke runs. They are not a "
            "formal-run duration estimate and not a deployment FPS claim."
        ),
        "auxiliary_tasks": {
            "bias_baseline": (
                "python scripts/fit_bias_baseline.py --mode both --device <dev> "
                "--output analysis/runs/material_bias_baseline/report.json"
            ),
            "shuffle_ablation": (
                "python scripts/ablate_material_shuffle.py --checkpoint <best.pt> "
                "--split val --device <dev> --output analysis/runs/material_shuffle_<name>/report.json"
            ),
            "note": (
                "Auxiliary tasks are optional diagnostics started manually; "
                "no orchestrator mode launches them and collect never fits them."
            ),
        },
    }


def cmd_plan(args: argparse.Namespace) -> int:
    spec = protocol_spec(getattr(args, "protocol", DEFAULT_PROTOCOL))
    tasks = build_tasks(protocol=spec.name)
    payload = build_plan_payload(tasks, args.device, spec)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    target = plan_path(spec, getattr(args, "output", None))
    _write_json_exclusive(target, payload)
    print(f"plan written: {target}")
    return 0


# ---------------------------------------------------------------------------
# preflight


def protocol_check(
    config: Mapping, expected: Mapping | None = None
) -> tuple[bool, list[str]]:
    """Compare a validated config against a protocol's fixed constants."""
    expected = EXPECTED_PROTOCOL if expected is None else expected
    differences = [
        f"{field}: config={config.get(field)!r} expected={expected_value!r}"
        for field, expected_value in expected.items()
        if config.get(field) != expected_value
    ]
    return (not differences), differences


def resolve_device(requested: str) -> str:
    """``torch_device_name`` without raising, for reporting.

    Returns the resolved torch device name, or an empty string when the
    request is not a device the execution path accepts.
    """
    try:
        return torch_device_name(requested)
    except ValueError:
        return ""


def check_device_available(requested: str) -> tuple[bool, str]:
    """Verify the requested device really exists, not just that it parses.

    Normalization is exactly ``torch_device_name`` — the helper every
    execution path (training, evaluation, collect) uses — so a device the
    preflight accepts can never be one a later run rejects: ``cpu``, a bare
    index, ``cuda`` and ``cuda:N`` are supported, and anything else (``gpu``,
    an empty string, a malformed name) is refused up front. That helper maps
    ``"0"`` to ``"cuda:0"`` without checking the machine has CUDA, so the
    resolved name is checked against the runtime here: a formal run must
    fail at preflight, not after the first batch.
    """
    resolved = resolve_device(requested)
    if not resolved:
        return False, (
            f"unrecognized device {requested!r}: expected 'cpu', 'cuda', 'cuda:N' "
            "or a CUDA device index"
        )
    if resolved == "cpu":
        return True, f"cpu (torch {torch.__version__})"
    if not torch.cuda.is_available():
        return False, (
            f"device {requested!r} resolves to {resolved!r} but CUDA is not "
            f"available (torch {torch.__version__})"
        )
    count = torch.cuda.device_count()
    try:
        index = int(resolved.split(":", 1)[1])
    except (IndexError, ValueError):
        return False, (
            f"device {requested!r} resolves to the malformed device name {resolved!r}"
        )
    if not 0 <= index < count:
        return False, (
            f"device {requested!r} resolves to {resolved!r}, which is out of "
            f"range (device_count={count})"
        )
    return True, (
        f"{resolved} {torch.cuda.get_device_name(index)} (torch {torch.__version__})"
    )


# The checks a run must re-confirm immediately before its first spawn. A
# saved preflight report is never trusted as a substitute: the config, the
# data identity, the dataset admission and the device are all re-derived.
ESSENTIAL_CHECKS = (
    "config_valid",
    "protocol_matches_fixed_constants",
    "protocol_data_identity",
    "dataset_admitted",
    "pretrained_weight",
    "device_available",
)


def admission_checks(
    spec: ProtocolSpec, device: str, *, include_output_state: bool = True
) -> list[dict]:
    """The read-only admission checks for one protocol and one device.

    Everything here is a read of the config, the dataset audit, the
    pretrained file and the run directories. Nothing is written, no training
    process exists, and no optimizer is constructed. Both ``preflight`` and
    the pre-spawn gate of ``run`` call this same function, so the two can
    never drift into different admission rules.
    """
    from core.material_mask_dataset import admit_dataset_for_protocol

    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    import yaml

    config = None
    try:
        template = yaml.safe_load(
            (PROJECT_ROOT / spec.config).read_text(encoding="utf-8")
        )
        config = validate_material_config(template)
        record("config_valid", True, spec.config)
    except (ValueError, FileNotFoundError, OSError) as exc:
        record("config_valid", False, str(exc))

    if config is not None:
        if config["experiment"] != spec.experiment:
            record(
                "protocol_matches_fixed_constants", False,
                f"config experiment {config['experiment']!r} is not the "
                f"{spec.name!r} protocol's {spec.experiment!r}",
            )
        else:
            ok, differences = protocol_check(config, spec.expected)
            record(
                "protocol_matches_fixed_constants", ok,
                "config matches the fixed protocol" if ok
                else "config differs from the fixed protocol; confirm before running: "
                     + "; ".join(differences),
            )

        # The protocol pins one concrete dataset. A config that names a
        # different one is reported here and refused, never silently used.
        if config["data_fingerprint"] != spec.data_fingerprint:
            record(
                "protocol_data_identity", False,
                f"config data_fingerprint {config['data_fingerprint']!r} is not the "
                f"{spec.name!r} protocol's {spec.data_fingerprint!r}",
            )
        elif config["parent_data_fingerprint"] != spec.parent_data_fingerprint:
            record(
                "protocol_data_identity", False,
                "config parent_data_fingerprint "
                f"{config['parent_data_fingerprint']!r} is not the {spec.name!r} "
                f"protocol's {spec.parent_data_fingerprint!r}",
            )
        else:
            record(
                "protocol_data_identity", True,
                f"data_fingerprint={config['data_fingerprint']} "
                f"parent_data_fingerprint={config['parent_data_fingerprint'] or 'n/a'}",
            )

        # The admission is bound to the protocol's identity (never to whatever
        # the config happens to say), so a config that disagrees cannot widen it.
        data_root = PROJECT_ROOT / (config["data"] or spec.data)
        try:
            report = admit_dataset_for_protocol(
                data_root,
                pooling_mode=spec.pooling_mode,
                expected_fingerprint=spec.data_fingerprint,
                expected_parent_fingerprint=spec.parent_data_fingerprint,
                expected_mask_schema=spec.mask_schema,
                expected_mask_resize_version=spec.mask_resize_version,
            )
            record(
                "dataset_admitted", True,
                f"root={report['data_root']} fingerprint={report['audit_fingerprint']} "
                f"splits={report['splits']} classes={report['class_names']}"
                + (
                    f" mask_schema={report['mask_schema_version']}"
                    f" mask_resize={report['mask_resize_version']}"
                    if "mask_schema_version" in report else ""
                ),
            )
        except (ValueError, FileNotFoundError) as exc:
            record("dataset_admitted", False, str(exc))

        pretrained = PROJECT_ROOT / (config["pretrained"] or spec.pretrained)
        if pretrained.is_file():
            import hashlib

            file_sha = hashlib.sha256(pretrained.read_bytes()).hexdigest()
            # The config's own value, never backfilled: an empty
            # ``pretrained_sha256`` means the run pinned no pretrained weight,
            # and must fail here instead of being silently repaired with the
            # protocol's digest (which would also make the log claim a value
            # the config never recorded).
            config_sha = config["pretrained_sha256"]
            # Three-way agreement: the file, the config and the protocol must
            # all name the same weight. A config SHA that merely matches a
            # swapped file is not the official pretrained weight.
            matches = file_sha == config_sha == spec.pretrained_sha256
            record(
                "pretrained_weight", matches,
                f"file={file_sha} config={config_sha!r} "
                f"protocol={spec.pretrained_sha256}"
                + ("" if matches else
                   " — the pretrained identity is not the "
                   f"{spec.name!r} protocol's"),
            )
        else:
            record("pretrained_weight", False, f"missing file: {pretrained}")

        fallback = PROJECT_ROOT / config["fallback_weights"]
        record(
            "fallback_weights_present",
            fallback.is_file(),
            str(fallback) + (" (present)" if fallback.is_file() else " (missing)"),
        )

    device_ok, device_detail = check_device_available(device)
    record("device_available", device_ok, device_detail)

    if include_output_state:
        tasks = build_tasks(protocol=spec.name)
        collisions = []
        for task in tasks:
            completed, reason = validate_completed_task(task, spec)
            if Path(task.run_dir).exists() and not completed:
                collisions.append(f"{task.task_id}: {reason}")
            elif completed:
                collisions.append(f"{task.task_id}: completed run already present (will be "
                                  "refused unless --skip-completed is passed to run)")
        record(
            "output_dir_state", not collisions,
            "; ".join(collisions) if collisions
            else f"all {len(tasks)} {spec.name} run directories are free",
        )

    return checks


def run_admission(spec: ProtocolSpec, device: str) -> tuple[bool, str]:
    """The essential admission gate re-run before the first training spawn."""
    checks = admission_checks(spec, device, include_output_state=False)
    by_name = {check["check"]: check for check in checks}
    for name in ESSENTIAL_CHECKS:
        check = by_name.get(name)
        if check is None:
            return False, f"admission check {name!r} did not run"
        if not check["ok"]:
            return False, f"{name}: {check['detail']}"
    return True, "admission ok"


def cmd_preflight(args: argparse.Namespace) -> int:
    spec = protocol_spec(getattr(args, "protocol", DEFAULT_PROTOCOL))
    checks = admission_checks(spec, args.device)
    device_ok, device_detail = check_device_available(args.device)
    payload = {
        "generated_at": _utc_now(),
        "protocol": spec.name,
        "experiment": spec.experiment,
        "config": spec.config,
        "data": {
            "root": spec.data,
            "data_fingerprint": spec.data_fingerprint,
            "parent_data_fingerprint": spec.parent_data_fingerprint,
        },
        "pooling": {
            "pooling_mode": spec.pooling_mode,
            "mask_schema": spec.mask_schema,
            "mask_resize_version": spec.mask_resize_version,
        },
        "management_dir": str(management_dir(spec)),
        "device": {
            "requested": args.device,
            "resolved": resolve_device(args.device),
            "available": device_ok,
            "detail": device_detail,
        },
        "checks": checks,
        "all_ok": all(check["ok"] for check in checks),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    target = preflight_path(spec, getattr(args, "output", None))
    _write_json_exclusive(target, payload)
    print(f"preflight written: {target}")
    return 0 if payload["all_ok"] else 1


# ---------------------------------------------------------------------------
# run


def _load_run_manifest(spec: ProtocolSpec) -> dict:
    """The run manifest of one protocol, refusing a foreign one.

    Each protocol has its own manifest file. A manifest whose recorded
    protocol is not this one is refused rather than adopted or merged: a v1
    attempt log is never evidence about a v2 task.
    """
    path = run_manifest_path(spec)
    if not path.is_file():
        return {"created_at": _utc_now(), "protocol": spec.name,
                "experiment": spec.experiment, "tasks": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.get("protocol", DEFAULT_PROTOCOL)
    if recorded != spec.name:
        raise ValueError(
            f"{path} records protocol {recorded!r} but this run is "
            f"{spec.name!r}; refusing to take over another protocol's manifest"
        )
    return manifest


def _save_run_manifest(spec: ProtocolSpec, manifest: dict) -> None:
    path = run_manifest_path(spec)
    manifest.setdefault("protocol", spec.name)
    manifest.setdefault("experiment", spec.experiment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _task_record(manifest: dict, task: ExperimentTask) -> dict:
    """The append-only record for one task; existing attempts are preserved."""
    return manifest["tasks"].setdefault(
        task.task_id,
        {
            "task_id": task.task_id,
            "protocol": task.protocol,
            "variant": task.variant,
            "seed": task.seed,
            "run_id": task.run_id,
            "run_dir": task.run_dir,
            "attempts": [],
        },
    )


def _record_attempt(record: dict, entry: dict) -> dict:
    """Append one attempt instead of overwriting the previous one.

    A retry after a failure, or a later ``--skip-completed`` pass, must not
    erase the earlier start/end times, exit code or failure detail.
    """
    entry["attempt"] = len(record["attempts"]) + 1
    entry["updated_at"] = _utc_now()
    record["attempts"].append(entry)
    return entry


def _last_attempt(record: dict) -> dict:
    attempts = record.get("attempts") or []
    return attempts[-1] if attempts else {}


def cmd_run(args: argparse.Namespace) -> int:
    spec = protocol_spec(getattr(args, "protocol", DEFAULT_PROTOCOL))
    tasks = build_tasks(protocol=spec.name)
    selected = task_subset(tasks, parse_task_list(args.tasks))
    # The same gate preflight reports, re-derived here before the first spawn:
    # an unusable device, a drifted config, a mismatched dataset or a missing
    # pretrained file must not produce twelve failed attempts, and a saved
    # preflight.json is never trusted in place of this re-check.
    admission_ok, admission_detail = run_admission(spec, args.device)
    if not admission_ok:
        print(f"run refused: {admission_detail}", file=sys.stderr)
        return 2
    try:
        manifest = _load_run_manifest(spec)
    except ValueError as exc:
        print(f"run refused: {exc}", file=sys.stderr)
        return 2

    for task in selected:
        record = _task_record(manifest, task)
        if args.skip_completed:
            completed, reason = validate_completed_task(task, spec)
            if completed:
                _record_attempt(record, {"status": "skipped_completed", "detail": reason})
                _save_run_manifest(spec, manifest)
                print(f"[skip-completed] {task.task_id}: {reason}")
                continue
        if Path(task.run_dir).exists():
            _record_attempt(record, {
                "status": "refused_existing",
                "detail": "run directory already exists; --skip-completed only "
                          "skips a run that passes full completion validation, and "
                          "this one does not. The orchestrator never deletes or "
                          "overwrites it — remove it explicitly to retry.",
            })
            _save_run_manifest(spec, manifest)
            _report_stop(manifest, selected, task, "refused existing directory")
            return 2
        entry = {
            "status": "running",
            "command": train_command(task, args.device, spec),
            "started_at": _utc_now(),
        }
        _record_attempt(record, entry)
        _save_run_manifest(spec, manifest)
        print(f"[run] {task.task_id}: {' '.join(entry['command'])}", flush=True)
        completed_process = subprocess.run(entry["command"], cwd=str(PROJECT_ROOT))
        entry["ended_at"] = _utc_now()
        entry["exit_code"] = completed_process.returncode
        if completed_process.returncode != 0:
            entry["status"] = "failed"
            _save_run_manifest(spec, manifest)
            _report_stop(manifest, selected, task, f"exit code {completed_process.returncode}")
            return 1
        completed_ok, reason = validate_completed_task(task, spec)
        entry["status"] = "completed" if completed_ok else "failed"
        entry["detail"] = reason
        _save_run_manifest(spec, manifest)
        if not completed_ok:
            _report_stop(manifest, selected, task, f"post-run validation failed: {reason}")
            return 1
        print(f"[done] {task.task_id}: {reason}")
    print(f"all {len(selected)} selected tasks completed")
    return 0


def _report_stop(
    manifest: dict, selected: list[ExperimentTask], failed: ExperimentTask, reason: str
) -> None:
    stopped = [
        task.task_id
        for task in selected
        if task.task_id != failed.task_id
        and _last_attempt(manifest["tasks"].get(task.task_id, {})).get("status")
        in (None, "running")
    ]
    print(
        f"STOPPED at {failed.task_id} ({reason}); not started: {stopped}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# collect


def detailed_metrics(
    labels: Sequence[int], preds: Sequence[int], num_classes: int = len(CLASS_NAMES)
) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1/support, confusion."""
    labels = list(labels)
    preds = list(preds)
    total = len(labels)
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for label, pred in zip(labels, preds):
        confusion[label][pred] += 1
    per_class = {}
    f1s = []
    for cls in range(num_classes):
        tp = confusion[cls][cls]
        fp = sum(confusion[row][cls] for row in range(num_classes)) - tp
        fn = sum(confusion[cls]) - tp
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        f1s.append(f1)
        per_class[CLASS_NAMES[cls]] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
    return {
        "samples": total,
        "accuracy": sum(1 for l, p in zip(labels, preds) if l == p) / max(total, 1),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_rows": "true class", "confusion_cols": "predicted class",
    }


def _macro_f1_from(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = int(np.sum((labels == cls) & (preds == cls)))
        fp = int(np.sum((labels != cls) & (preds == cls)))
        fn = int(np.sum((labels == cls) & (preds != cls)))
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def bootstrap_macro_f1_delta(
    preds_a: list[int],
    preds_b: list[int],
    labels: list[int],
    groups: list[str],
    num_classes: int = len(CLASS_NAMES),
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Cluster bootstrap of the macro-F1 delta (a - b), resampling groups."""
    labels_a = np.asarray(labels)
    preds_a_arr = np.asarray(preds_a)
    preds_b_arr = np.asarray(preds_b)
    unique_groups = sorted(set(groups))
    group_index = {name: i for i, name in enumerate(unique_groups)}
    member = [[] for _ in unique_groups]
    for idx, group in enumerate(groups):
        member[group_index[group]].append(idx)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique_groups), size=len(unique_groups))
        idx = np.concatenate([member[i] for i in chosen]) if len(chosen) else np.array([], int)
        if idx.size == 0:
            continue
        deltas.append(
            _macro_f1_from(labels_a[idx], preds_a_arr[idx], num_classes)
            - _macro_f1_from(labels_a[idx], preds_b_arr[idx], num_classes)
        )
    deltas_arr = np.asarray(deltas)
    lo, hi = np.percentile(deltas_arr, [2.5, 97.5])
    return {
        "point_estimate": _macro_f1_from(labels_a, preds_a_arr, num_classes)
        - _macro_f1_from(labels_a, preds_b_arr, num_classes),
        "ci95": [float(lo), float(hi)],
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "n_groups": len(unique_groups),
    }


def paired_flip_stats(
    preds_a: list[int], preds_b: list[int], labels: list[int]
) -> dict:
    """Paired same-seed same-sample flip counts (a relative to b)."""
    fixed = sum(1 for pa, pb, l in zip(preds_a, preds_b, labels) if pb != l and pa == l)
    broken = sum(1 for pa, pb, l in zip(preds_a, preds_b, labels) if pb == l and pa != l)
    both_correct = sum(1 for pa, pb, l in zip(preds_a, preds_b, labels) if pa == l and pb == l)
    both_wrong = sum(1 for pa, pb, l in zip(preds_a, preds_b, labels) if pa != l and pb != l)
    return {
        "samples": len(labels),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_fixes_b_errors": fixed,
        "a_breaks_b_correct": broken,
    }


@torch.no_grad()
def collect_task_predictions(model, loader, device: str) -> dict:
    """Pure-model val predictions in manifest order (no optimizer exists).

    A mask-pooling model receives the instance mask together with its batch:
    the batch is unpacked through the shared helper, so the mask the model
    pools with is never dropped and a v2 model is never silently evaluated
    with global average pooling.
    """
    from core.material_fusion_dataset import unpack_batch

    with_mask = bool(getattr(model, "uses_instance_mask", False))
    model.eval()
    labels_all: list[int] = []
    preds: list[int] = []
    logits_all: list[list[float]] = []
    valid_flags: list[bool] = []
    for batch in loader:
        rgb, aux, quality, labels, mask = unpack_batch(batch, model.is_dual, with_mask)
        rgb = rgb.to(device)
        aux = None if aux is None else aux.to(device)
        if mask is not None:
            logits = model(rgb, aux, mask.to(device))
        else:
            logits = model(rgb, aux) if model.is_dual else model(rgb)
        labels_all.extend(labels.tolist())
        logits_all.extend(logits.cpu().tolist())
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        valid_flags.extend(bool(v) for v in (quality[:, 0] > 0).tolist())
    return {
        "labels": labels_all,
        "preds": preds,
        "valid": valid_flags,
        "logits": logits_all,
    }


def _load_completed_task(
    task: ExperimentTask, spec: ProtocolSpec
) -> tuple[dict | None, str]:
    """The recorded evidence of one completed run of *this* protocol."""
    from models.material_fusion import sha256_of_file

    completed, reason = validate_completed_task(task, spec)
    if not completed:
        return None, reason
    run_dir = Path(task.run_dir)
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    best_meta = read_material_checkpoint_metadata(run_dir / "best.pt")
    return {
        "run_config": run_config,
        "best_metadata": dict(best_meta.__dict__),
        "best_meta": best_meta,
        "run_dir": str(run_dir),
        "best_path": str(run_dir / "best.pt"),
        "checkpoint_sha256": sha256_of_file(run_dir / "best.pt"),
    }, reason


def _collect_paired_and_bootstrap(
    task_results: dict, groups: list[str]
) -> tuple[dict, dict]:
    paired: dict = {}
    bootstrap: dict = {}
    for seed in SEEDS:
        candidates = {
            variant: task_results.get(f"{variant}_s{seed}", {}).get("predictions")
            for variant in VARIANTS
        }
        if any(value is None for value in candidates.values()):
            paired[f"seed_{seed}"] = {"status": "incomplete", "reason": "one or more "
                                      "variants of this seed are not completed"}
            continue
        e4 = candidates["gray_polar"]
        for other in ("gray", "gray_gray_valid", "gray_valid"):
            pair_id = f"seed_{seed}_gray_polar_vs_{other}"
            paired[pair_id] = paired_flip_stats(
                e4["preds"], candidates[other]["preds"], e4["labels"]
            )
            if other == "gray":
                bootstrap[pair_id] = bootstrap_macro_f1_delta(
                    e4["preds"], candidates[other]["preds"], e4["labels"], groups
                )
    return paired, bootstrap


def manifest_split_rows(data_root: Path, split: str) -> list[dict]:
    """Manifest rows of one split, in manifest (== loader) order."""
    with (Path(data_root) / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def split_group_names(data_root: Path, split: str) -> list[str]:
    """Capture-group names for one split, in manifest (== loader) order."""
    return [row["group_name"] for row in manifest_split_rows(data_root, split)]


def cmd_collect(args: argparse.Namespace) -> int:
    from core.material_fusion_dataset import MaterialFusionDataset, unpack_batch
    from core.material_mask_dataset import admit_dataset_for_protocol
    from models.material_fusion import load_material_checkpoint, variant_is_dual
    from scripts.eval_material_fusion import load_fallback_gray

    spec = protocol_spec(getattr(args, "protocol", DEFAULT_PROTOCOL))

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists():
        print(f"collect refused: output {output_dir} already exists", file=sys.stderr)
        return 2

    device_ok, device_detail = check_device_available(args.device)
    if not device_ok:
        print(f"collect refused: {device_detail}", file=sys.stderr)
        return 2
    device = resolve_device(args.device)

    import yaml

    template = validate_material_config(
        yaml.safe_load((PROJECT_ROOT / spec.config).read_text(encoding="utf-8"))
    )
    if template["experiment"] != spec.experiment:
        print(
            f"collect refused: {spec.config} is not the {spec.name!r} protocol "
            f"config ({template['experiment']!r})",
            file=sys.stderr,
        )
        return 2
    # The config must claim the chosen protocol's identity. A config that
    # describes a different dataset state would otherwise decide which data the
    # collection is bound to.
    for field, expected in (
        ("data_fingerprint", spec.data_fingerprint),
        ("parent_data_fingerprint", spec.parent_data_fingerprint),
    ):
        if template[field] != expected:
            print(
                f"collect refused: {spec.config} records {field} "
                f"{template[field]!r}, not the {spec.name!r} protocol's {expected!r}",
                file=sys.stderr,
            )
            return 2

    # The data is bound to the chosen protocol's identity whether it is read
    # from the config's default path or from ``--data``: ``--data`` relocates
    # the read path (a byte-identical copy has the same fingerprint), it never
    # switches to another dataset version. The per-checkpoint binding in the
    # task loop below still runs on top of this.
    if args.data is not None:
        data_root = Path(args.data)
        if not data_root.is_absolute():
            data_root = PROJECT_ROOT / data_root
    else:
        data_root = PROJECT_ROOT / (template["data"] or spec.data)
    try:
        data_report = admit_dataset_for_protocol(
            data_root,
            pooling_mode=spec.pooling_mode,
            expected_fingerprint=spec.data_fingerprint,
            expected_parent_fingerprint=spec.parent_data_fingerprint,
            expected_mask_schema=spec.mask_schema,
            expected_mask_resize_version=spec.mask_resize_version,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(
            f"collect refused: the evaluation data at {data_root} is not the "
            f"{spec.name!r} protocol's dataset: {exc}",
            file=sys.stderr,
        )
        return 2
    val_rows = manifest_split_rows(data_root, "val")
    val_paths, _, _ = read_manifest_split(data_root, "val")
    if len(val_rows) != len(val_paths):
        print(
            "collect refused: the manifest and the loader disagree on the val "
            f"sample count ({len(val_rows)} vs {len(val_paths)})",
            file=sys.stderr,
        )
        return 2
    groups = [row["group_name"] for row in val_rows]

    admissions: dict[tuple, dict] = {}

    def admitted_for(meta) -> dict:
        """Admit the eval data against one checkpoint's recorded identity."""
        key = (
            meta.pooling_mode, meta.data_fingerprint, meta.parent_data_fingerprint,
            meta.mask_schema, meta.mask_resize_version,
        )
        if key not in admissions:
            admissions[key] = admit_dataset_for_protocol(
                data_root,
                pooling_mode=meta.pooling_mode,
                expected_fingerprint=meta.data_fingerprint,
                expected_parent_fingerprint=meta.parent_data_fingerprint,
                expected_mask_schema=meta.mask_schema,
                expected_mask_resize_version=meta.mask_resize_version,
            )
        return admissions[key]

    fallback = None
    fallback_identity = ""
    fallback_error = ""
    fallback_file = PROJECT_ROOT / template["fallback_weights"]
    try:
        fallback, fallback_identity = load_fallback_gray(fallback_file, device, 224)
    except (FileNotFoundError, ValueError) as exc:
        fallback_error = str(exc)

    tasks = build_tasks(protocol=spec.name)
    task_results: dict = {}
    excluded: dict = {}
    per_sample_records: list[dict] = []
    for task in tasks:
        payload, reason = _load_completed_task(task, spec)
        if payload is None:
            excluded[task.task_id] = reason
            continue
        best_meta = payload["best_meta"]
        if best_meta.pooling_mode != spec.pooling_mode:
            excluded[task.task_id] = (
                f"checkpoint pooling_mode {best_meta.pooling_mode!r} is not this "
                f"protocol's {spec.pooling_mode!r}"
            )
            continue
        try:
            admitted_for(best_meta)
        except (ValueError, FileNotFoundError) as exc:
            excluded[task.task_id] = (
                f"evaluation data is not the dataset this checkpoint was trained "
                f"on: {exc}"
            )
            continue
        model, best_meta, _ = load_material_checkpoint(
            payload["best_path"], variant=task.variant, pooling_mode=spec.pooling_mode
        )
        model = model.to(device)
        with_mask = bool(getattr(model, "uses_instance_mask", False))
        loader = torch.utils.data.DataLoader(
            MaterialFusionDataset(
                val_paths, variant=task.variant, imgsz=best_meta.imgsz,
                with_instance_mask=with_mask,
            ),
            batch_size=args.batch, shuffle=False, num_workers=0,
        )
        predictions = collect_task_predictions(model, loader, device)
        if len(predictions["labels"]) != len(val_rows):
            excluded[task.task_id] = (
                f"the loader produced {len(predictions['labels'])} predictions for "
                f"{len(val_rows)} val samples; refusing a partial aggregate"
            )
            continue
        entry: dict = {
            "run_dir": payload["run_dir"],
            "protocol": spec.name,
            "pooling_mode": spec.pooling_mode,
            "data_fingerprint": best_meta.data_fingerprint,
            "parent_data_fingerprint": best_meta.parent_data_fingerprint,
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "best_epoch": payload["best_metadata"].get("epoch"),
            "train_reported": {
                "val_macro_f1": payload["best_metadata"].get("val_macro_f1"),
                "val_accuracy": payload["best_metadata"].get("val_accuracy"),
            },
            "param_count": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": payload["run_config"].get("elapsed_seconds"),
            "peak_vram_mb": payload["run_config"].get("peak_vram_mb"),
            "pure": detailed_metrics(predictions["labels"], predictions["preds"]),
        }
        valid_idx = [i for i, ok in enumerate(predictions["valid"]) if ok]
        invalid_idx = [i for i, ok in enumerate(predictions["valid"]) if not ok]
        entry["valid_subset"] = detailed_metrics(
            [predictions["labels"][i] for i in valid_idx],
            [predictions["preds"][i] for i in valid_idx],
        )
        entry["invalid_subset_pure"] = detailed_metrics(
            [predictions["labels"][i] for i in invalid_idx],
            [predictions["preds"][i] for i in invalid_idx],
        )
        routed: list[int] | None = None
        if variant_is_dual(task.variant):
            routed = list(predictions["preds"])
            if fallback is None:
                # A missing fallback never fabricates a routed result.
                routed = None
                entry["routed"] = None
                entry["routed_note"] = f"fallback unavailable: {fallback_error}"
            elif invalid_idx:
                sub_loader = torch.utils.data.DataLoader(
                    torch.utils.data.Subset(
                        MaterialFusionDataset(
                            val_paths, variant=task.variant, imgsz=best_meta.imgsz,
                            with_instance_mask=with_mask,
                        ),
                        invalid_idx,
                    ),
                    batch_size=args.batch, shuffle=False, num_workers=0,
                )
                offset = 0
                for batch in sub_loader:
                    rgb, _, _, _, _ = unpack_batch(batch, model.is_dual, with_mask)
                    preds_fb = fallback(rgb.to(device)).argmax(dim=1).cpu().tolist()
                    for local, pred in enumerate(preds_fb):
                        routed[invalid_idx[offset + local]] = pred
                    offset += len(preds_fb)
                entry["routed"] = detailed_metrics(predictions["labels"], routed)
            else:
                entry["routed"] = detailed_metrics(predictions["labels"], routed)
        else:
            entry["routed"] = None
        # One record per val sample, in loader order, so the aggregate above
        # can be recomputed from these records alone (including the tail batch).
        for index, row in enumerate(val_rows):
            per_sample_records.append(
                {
                    "task_id": task.task_id,
                    "protocol": spec.name,
                    "variant": task.variant,
                    "seed": task.seed,
                    "run_id": task.run_id,
                    "sample_name": row["sample_name"],
                    "split": "val",
                    "group": row["group_name"],
                    "label": int(row["class_id"]),
                    "raw_logits": predictions["logits"][index],
                    "raw_prediction": predictions["preds"][index],
                    "polar_valid": bool(predictions["valid"][index]),
                    "routed_prediction": None if routed is None else routed[index],
                    "used_fallback": bool(
                        routed is not None and not predictions["valid"][index]
                    ),
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "data_fingerprint": best_meta.data_fingerprint,
                }
            )
        entry["per_sample_count"] = len(val_rows)
        entry["predictions"] = predictions
        task_results[task.task_id] = entry

    paired, bootstrap = _collect_paired_and_bootstrap(task_results, groups)
    per_variant = {}
    for variant in VARIANTS:
        seeds_result = {
            f"s{seed}": {
                "best_epoch": task_results[f"{variant}_s{seed}"]["best_epoch"],
                "macro_f1": task_results[f"{variant}_s{seed}"]["pure"]["macro_f1"],
                "accuracy": task_results[f"{variant}_s{seed}"]["pure"]["accuracy"],
            }
            for seed in SEEDS if f"{variant}_s{seed}" in task_results
        }
        if len(seeds_result) == len(SEEDS):
            f1s = [v["macro_f1"] for v in seeds_result.values()]
            accs = [v["accuracy"] for v in seeds_result.values()]
            seeds_result["mean_std"] = {
                "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            }
        per_variant[variant] = seeds_result

    for entry in task_results.values():
        entry.pop("predictions")

    group_warning = (
        f"only {len(set(groups))} capture groups in val; the cluster bootstrap "
        "widely underestimates certainty — treat CIs as indicative only and do "
        "not treat consecutive frames as independent evidence"
        if len(set(groups)) < 30 else "group count sufficient"
    )
    report = {
        "generated_at": _utc_now(),
        "protocol": spec.name,
        "experiment": spec.experiment,
        "pooling_mode": spec.pooling_mode,
        "split": "val",
        "test_split_used": False,
        "class_names": list(CLASS_NAMES),
        "data_fingerprint": data_report["audit_fingerprint"],
        "eval_data_identity_bound": True,
        "eval_data_identity": {
            "data_fingerprint": spec.data_fingerprint,
            "parent_data_fingerprint": spec.parent_data_fingerprint,
            "root": str(data_root),
        },
        "n_val_groups": len(set(groups)),
        "group_uncertainty_note": group_warning,
        "fallback_identity": fallback_identity,
        "fallback_error": fallback_error,
        "tasks": task_results,
        "excluded": excluded,
        "per_variant_seeds": per_variant,
        "paired_flips": paired,
        "cluster_bootstrap": bootstrap,
        "complete_matrix": len(task_results) == len(tasks),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    per_sample_path = output_dir / "per_sample_predictions.jsonl"
    # Written as explicit UTF-8/LF bytes: a text-mode write would translate the
    # line endings on Windows, and the report's digest must identify the bytes
    # that actually landed on disk (it is taken from the file afterwards).
    per_sample_bytes = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in per_sample_records
    ).encode("utf-8")
    per_sample_path.write_bytes(per_sample_bytes)
    report["per_sample_predictions"] = {
        "path": str(per_sample_path),
        "records": len(per_sample_records),
        "sha256": hashlib.sha256(per_sample_path.read_bytes()).hexdigest(),
        "encoding": "utf-8",
        "line_endings": "lf",
        "note": (
            "One record per completed task per val sample, in loader order; the "
            "aggregates in report.json are recomputable from these records. The "
            "digest is the sha256 of the written file's bytes."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
    print(f"collect report: {output_dir / 'report.json'}")
    print(f"per-sample predictions: {per_sample_path}")
    print(f"summary: {output_dir / 'summary.md'}")
    return 0


def render_summary(report: dict) -> str:
    lines = [
        "# CMX material-fusion comparison (val split, no test used)",
        "",
        f"Protocol: {report['protocol']} ({report['experiment']}, "
        f"pooling={report['pooling_mode']})",
        f"Generated: {report['generated_at']}",
        f"Val capture groups: {report['n_val_groups']} — {report['group_uncertainty_note']}",
        "",
        "## Per-task best-epoch results (val)",
        "",
        "| task | best epoch | accuracy | macro-F1 |",
        "|---|---|---|---|",
    ]
    for task_id, entry in report["tasks"].items():
        lines.append(
            f"| {task_id} | {entry['best_epoch']} "
            f"| {entry['pure']['accuracy']:.4f} | {entry['pure']['macro_f1']:.4f} |"
        )
    lines += ["", "## Per-variant across seeds", ""]
    for variant, seeds_result in report["per_variant_seeds"].items():
        stats = seeds_result.get("mean_std")
        if stats:
            lines.append(
                f"- {variant}: macro-F1 {stats['macro_f1_mean']:.4f} "
                f"± {stats['macro_f1_std']:.4f}"
            )
        else:
            lines.append(f"- {variant}: incomplete ({len(seeds_result)} of 3 seeds)")
    lines += ["", "## Paired flips (gray_polar vs others, same seed)", ""]
    for pair_id, stats in report["paired_flips"].items():
        if stats.get("status") == "incomplete":
            lines.append(f"- {pair_id}: incomplete")
            continue
        lines.append(
            f"- {pair_id}: fixes {stats['a_fixes_b_errors']}, "
            f"breaks {stats['a_breaks_b_correct']}, "
            f"both correct {stats['both_correct']}, both wrong {stats['both_wrong']}"
        )
    lines += ["", "## Cluster bootstrap (macro-F1 delta, gray_polar - gray)", ""]
    for pair_id, stats in report["cluster_bootstrap"].items():
        lines.append(
            f"- {pair_id}: {stats['point_estimate']:+.4f} "
            f"[{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}] "
            f"({stats['n_groups']} groups, {stats['n_boot']} resamples)"
        )
    if report["excluded"]:
        lines += ["", "## Excluded tasks", ""]
        for task_id, reason in report["excluded"].items():
            lines.append(f"- {task_id}: {reason}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI


def _add_protocol_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protocol", choices=PROTOCOL_CHOICES, default=DEFAULT_PROTOCOL,
        help="Experiment protocol (default: %(default)s). 'v2_mask' is the V5 "
             "instance-mask protocol with its own run ids, management directory "
             "and admission rules; it is never mixed with v1.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    plan_p = sub.add_parser("plan", help="write the task matrix and commands (read-only)")
    plan_p.add_argument("--device", default="0")
    _add_protocol_argument(plan_p)
    plan_p.add_argument("--output", default=None,
                        help="plan file (default: this protocol's management dir). "
                             "An existing file is never overwritten.")
    plan_p.set_defaults(func=cmd_plan)

    pre_p = sub.add_parser("preflight", help="read-only admission checks; never trains")
    pre_p.add_argument("--device", default="0")
    _add_protocol_argument(pre_p)
    pre_p.add_argument("--output", default=None,
                       help="preflight file (default: this protocol's management "
                            "dir). An existing file is never overwritten.")
    pre_p.set_defaults(func=cmd_preflight)

    run_p = sub.add_parser("run", help="explicitly start a task subset (trains)")
    run_p.add_argument("--tasks", required=True,
                       help="comma list of task ids of the form <variant>_s<seed>, "
                            "e.g. gray_s2026,gray_polar_s2026 (see `plan` for the "
                            "full 12-task matrix)")
    run_p.add_argument("--device", default="0")
    _add_protocol_argument(run_p)
    run_p.add_argument("--skip-completed", action="store_true",
                       help="skip tasks whose run directory passes full completion "
                            "validation (never a mere directory-name check)")
    run_p.set_defaults(func=cmd_run)

    col_p = sub.add_parser("collect", help="aggregate completed runs (read-only + val inference)")
    col_p.add_argument("--device", default="0")
    col_p.add_argument("--batch", type=int, default=32)
    col_p.add_argument("--data", default=None,
                       help="Dataset root override (default: the protocol config's "
                            "own dataset, admitted with the pinned fingerprint). "
                            "Relocates the path only; the identity is not relaxed.")
    col_p.add_argument("--output", default=None,
                       help="report directory (default: this protocol's management "
                            "dir / collect_<utc-stamp>)")
    _add_protocol_argument(col_p)
    col_p.set_defaults(func=cmd_collect_default)

    return parser.parse_args(argv)


def cmd_collect_default(args: argparse.Namespace) -> int:
    spec = protocol_spec(getattr(args, "protocol", DEFAULT_PROTOCOL))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output = args.output or str(management_dir(spec) / f"collect_{stamp}")
    return cmd_collect(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
