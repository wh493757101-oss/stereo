"""CMX material-fusion classification model and checkpoint contract.

Architecture (fixed by the experiment definition):

    appearance stream -> MiT-B0 (single) or CMX dual-stream encoder
    four levels of FRM/FFM-fused features (dims 32/64/160/256)
    per-level 1x1 projection to 128 channels + global average pooling
    concatenated 512-dim vector -> LayerNorm -> Dropout(0.1) -> Linear(4)

Variants (fixed):

    gray             single-stream MiT-B0, appearance only
    gray_gray_valid  CMX, auxiliary [G, G, V]
    gray_valid       CMX, auxiliary [0, 0, V]
    gray_polar       CMX, auxiliary [signed_q, abs_q, V]

The classifier returns raw logits only. Polar-invalid routing (bypassing
the CMX model and using the independent historical gray Model B) is
deliberately kept outside this module: the checkpoint's raw output and the
post-routing output are separate, separately reported quantities.

Checkpoints are ``{"metadata": {...}, "state_dict": {...}}``; the metadata
is validated field-by-field with strict types (``smoke`` must be a real
bool, ``limit_batches`` a real int) before any model is rebuilt.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.material_fusion_dataset import (
    AUX_CHANNEL_MEANING,
    CLASS_NAMES,
    PREPROCESSING_VERSION,
    VARIANTS,
    validate_variant,
    variant_is_dual,
)
from core.material_mask_dataset import (
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    POOLING_GAP,
    POOLING_INSTANCE_MASK,
    POOLING_MODES,
    V4_PARENT_FINGERPRINT,
    validate_pooling_mode,
)
from models.cmx.dual_segformer import (
    MIT_B0_SPEC,
    MiTEncoder,
    UPSTREAM_COMMIT,
    load_mit_b0_checkpoint,
    mit_b0,
)

MATERIAL_FUSION_VERSION = "material_fusion_v1"
# Second-round protocol version: identical training protocol and identical
# inputs, with the classification head's global average pooling replaced by
# instance-mask weighted pooling on the V5 mask dataset.
MATERIAL_FUSION_MASK_VERSION = "material_fusion_v2_mask"
SUPPORTED_EXPERIMENT_VERSIONS = (MATERIAL_FUSION_VERSION, MATERIAL_FUSION_MASK_VERSION)

NUM_CLASSES = len(CLASS_NAMES)

HEAD_PROJ_DIM = 128
HEAD_DROPOUT = 0.1
ENCODER_LEVEL_DIMS = tuple(MIT_B0_SPEC["embed_dims"])

STAGES = ("warmup", "finetune")

# Stage-snapshot evidence format (v2). The evidence records the identity of
# the run it belongs to, per-file digests of everything it verifies, and
# the checkpoint files it is associated with. Digests cover the evidence
# files themselves; checkpoints never reference the evidence, so there is
# no circular digest dependency. This protects against wrong mixing,
# corruption and missing evidence — it does not claim to resist an
# attacker who rewrites every local file consistently.
STAGE_SNAPSHOT_FORMAT = "material_fusion_stage_snapshots"
STAGE_SNAPSHOT_FORMAT_VERSION = 2
SNAPSHOT_ROLES = ("init", "post_warmup", "post_finetune")
SNAPSHOT_FILENAMES = {
    "init": "init.pt",
    "post_warmup": "post_warmup.pt",
    "post_finetune": "post_finetune.pt",
}
SNAPSHOT_STAGES = {
    "init": "init",
    "post_warmup": "warmup",
    "post_finetune": "finetune",
}

# Top-level encoder module prefixes. The pretrained MiT backbone streams
# are frozen during warmup; the newly initialized FRM/FFM fusion layers are
# not part of the pretrained weights and train in both stages.
_BACKBONE_MODULE_PREFIXES = (
    "patch_embed", "extra_patch_embed", "block", "extra_block", "norm", "extra_norm",
)
_FUSION_MODULE_PREFIXES = ("FRMs", "FFMs")

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


# ---------------------------------------------------------------------------
# Strict type helpers (``bool`` is a subclass of ``int``; both are refused
# wherever the contract requires the other).


def require_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(
            f"{field} must be a bool, got {type(value).__name__}: {value!r}"
        )
    return value


def require_int(value: Any, field: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(
            f"{field} must be an int, got {type(value).__name__}: {value!r}"
        )
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return value


def require_float(value: Any, field: str, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(
            f"{field} must be a number, got {type(value).__name__}: {value!r}"
        )
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {result}")
    return result


def require_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a string, got {type(value).__name__}: {value!r}"
        )
    return value


def _require_class_names(value: Any, field: str = "class_names") -> tuple[str, ...]:
    names = tuple(str(name) for name in value) if isinstance(value, (list, tuple)) else None
    if names != CLASS_NAMES:
        raise ValueError(
            f"{field} must be exactly {CLASS_NAMES} (fixed project class order), "
            f"got {value!r}"
        )
    return names


def _require_sha256(value: Any, field: str, allow_empty: bool = False) -> str:
    value = require_str(value, field)
    if not value and allow_empty:
        return value
    if _SHA256_PATTERN.match(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hex chars, got {value!r}")
    return value


def expected_stage_for_epoch(epoch: int, warmup_epochs: int) -> str:
    """Protocol stage of a 1-based epoch number."""
    require_int(epoch, "epoch", minimum=1)
    require_int(warmup_epochs, "warmup_epochs", minimum=0)
    return "warmup" if epoch <= warmup_epochs else "finetune"


def sha256_of_file(path: str | Path) -> str:
    """sha256 hex digest of a file's bytes (evidence-integrity helper)."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_stage_snapshot_metadata(payload: Mapping[str, Any]) -> dict:
    """Strictly validate stage-snapshot evidence metadata (format v2).

    Types are checked strictly (no ``int()``/``bool()`` coercion); the
    required snapshot roles follow the run protocol (``post_warmup`` is
    required exactly when ``warmup_epochs > 0``, ``post_finetune`` exactly
    when a finetune epoch exists). Returns the normalized mapping.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("stage snapshot metadata must be a mapping")
    required = (
        "format", "format_version", "run_id", "variant", "seed",
        "data_fingerprint", "imgsz", "preprocessing_version", "upstream_commit",
        "pretrained_sha256", "epochs", "warmup_epochs", "smoke", "limit_batches",
        "physical_batch", "accum_steps", "effective_batch", "snapshots", "checkpoints",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"stage snapshot metadata is missing fields: {missing}")

    if payload["format"] != STAGE_SNAPSHOT_FORMAT:
        raise ValueError(
            f"stage snapshot format must be {STAGE_SNAPSHOT_FORMAT!r}, "
            f"got {payload['format']!r}"
        )
    format_version = require_int(payload["format_version"], "format_version", minimum=1)
    if format_version != STAGE_SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported stage snapshot format_version {format_version}; "
            f"expected {STAGE_SNAPSHOT_FORMAT_VERSION}"
        )

    normalized: dict[str, Any] = {
        "format": STAGE_SNAPSHOT_FORMAT,
        "format_version": format_version,
        "run_id": require_str(payload["run_id"], "run_id"),
        "variant": validate_variant(payload["variant"]),
        "seed": require_int(payload["seed"], "seed"),
        "data_fingerprint": require_str(payload["data_fingerprint"], "data_fingerprint"),
        "imgsz": require_int(payload["imgsz"], "imgsz", minimum=1),
        "preprocessing_version": require_str(
            payload["preprocessing_version"], "preprocessing_version"
        ),
        "upstream_commit": require_str(payload["upstream_commit"], "upstream_commit"),
        "pretrained_sha256": _require_sha256(
            payload["pretrained_sha256"], "pretrained_sha256", allow_empty=True
        ),
        "epochs": require_int(payload["epochs"], "epochs", minimum=1),
        "warmup_epochs": require_int(payload["warmup_epochs"], "warmup_epochs", minimum=0),
        "smoke": require_bool(payload["smoke"], "smoke"),
        "limit_batches": require_int(payload["limit_batches"], "limit_batches", minimum=0),
        "physical_batch": require_int(payload["physical_batch"], "physical_batch", minimum=1),
        "accum_steps": require_int(payload["accum_steps"], "accum_steps", minimum=1),
        "effective_batch": require_int(payload["effective_batch"], "effective_batch", minimum=1),
    }
    if not normalized["run_id"]:
        raise ValueError("run_id must not be empty")
    if normalized["warmup_epochs"] >= normalized["epochs"]:
        raise ValueError(
            f"warmup_epochs ({normalized['warmup_epochs']}) must be smaller than "
            f"epochs ({normalized['epochs']})"
        )
    if normalized["smoke"] != (normalized["limit_batches"] > 0):
        raise ValueError(
            "smoke must be True exactly when limit_batches > 0, got "
            f"smoke={normalized['smoke']}, limit_batches={normalized['limit_batches']}"
        )
    if normalized["physical_batch"] * normalized["accum_steps"] != normalized["effective_batch"]:
        raise ValueError(
            "effective_batch must equal physical_batch * accum_steps, got "
            f"{normalized['physical_batch']} * {normalized['accum_steps']} != "
            f"{normalized['effective_batch']}"
        )

    snapshots = payload["snapshots"]
    if not isinstance(snapshots, Mapping):
        raise ValueError("stage snapshot 'snapshots' must be a mapping")
    required_roles = ["init"]
    if normalized["warmup_epochs"] > 0:
        required_roles.append("post_warmup")
    if normalized["epochs"] > normalized["warmup_epochs"]:
        required_roles.append("post_finetune")
    unknown_roles = sorted(set(snapshots) - set(SNAPSHOT_ROLES))
    if unknown_roles:
        raise ValueError(f"unknown snapshot roles: {unknown_roles}")
    missing_roles = [role for role in required_roles if role not in snapshots]
    if missing_roles:
        raise ValueError(
            f"stage snapshot evidence is missing required roles {missing_roles} "
            f"(warmup_epochs={normalized['warmup_epochs']}, epochs={normalized['epochs']})"
        )
    normalized_snapshots: dict[str, dict] = {}
    for role in required_roles:
        entry = snapshots[role]
        if not isinstance(entry, Mapping):
            raise ValueError(f"snapshot entry {role!r} must be a mapping")
        for field in ("file", "epoch", "stage", "sha256"):
            if field not in entry:
                raise ValueError(f"snapshot entry {role!r} is missing {field!r}")
        if entry["file"] != SNAPSHOT_FILENAMES[role]:
            raise ValueError(
                f"snapshot {role!r} file must be {SNAPSHOT_FILENAMES[role]!r}, "
                f"got {entry['file']!r}"
            )
        stage = require_str(entry["stage"], f"snapshots.{role}.stage")
        if stage != SNAPSHOT_STAGES[role]:
            raise ValueError(
                f"snapshot {role!r} stage must be {SNAPSHOT_STAGES[role]!r}, got {stage!r}"
            )
        epoch = require_int(entry["epoch"], f"snapshots.{role}.epoch", minimum=0)
        expected_epoch = {
            "init": 0,
            "post_warmup": normalized["warmup_epochs"],
            "post_finetune": normalized["epochs"],
        }[role]
        if epoch != expected_epoch:
            raise ValueError(
                f"snapshot {role!r} epoch must be {expected_epoch}, got {epoch}"
            )
        normalized_snapshots[role] = {
            "file": SNAPSHOT_FILENAMES[role],
            "epoch": epoch,
            "stage": stage,
            "sha256": _require_sha256(entry["sha256"], f"snapshots.{role}.sha256"),
        }
    normalized["snapshots"] = normalized_snapshots

    checkpoints = payload["checkpoints"]
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"last", "best"}:
        raise ValueError("stage snapshot 'checkpoints' must contain 'last' and 'best'")
    normalized_checkpoints: dict[str, dict] = {}
    for role in ("last", "best"):
        entry = checkpoints[role]
        if not isinstance(entry, Mapping):
            raise ValueError(f"checkpoint entry {role!r} must be a mapping")
        for field in ("file", "epoch", "stage", "sha256"):
            if field not in entry:
                raise ValueError(f"checkpoint entry {role!r} is missing {field!r}")
        stage = require_str(entry["stage"], f"checkpoints.{role}.stage")
        if stage not in STAGES:
            raise ValueError(
                f"checkpoint {role!r} stage must be one of {STAGES}, got {stage!r}"
            )
        epoch = require_int(entry["epoch"], f"checkpoints.{role}.epoch", minimum=1)
        if epoch > normalized["epochs"]:
            raise ValueError(
                f"checkpoint {role!r} epoch {epoch} exceeds the run's epochs "
                f"{normalized['epochs']}"
            )
        expected_stage = expected_stage_for_epoch(epoch, normalized["warmup_epochs"])
        if stage != expected_stage:
            raise ValueError(
                f"checkpoint {role!r} stage {stage!r} does not match the protocol "
                f"stage {expected_stage!r} for epoch {epoch}"
            )
        normalized_checkpoints[role] = {
            "file": require_str(entry["file"], f"checkpoints.{role}.file"),
            "epoch": epoch,
            "stage": stage,
            "sha256": _require_sha256(entry["sha256"], f"checkpoints.{role}.sha256"),
        }
    normalized["checkpoints"] = normalized_checkpoints
    if normalized_checkpoints["last"]["epoch"] != normalized["epochs"]:
        raise ValueError(
            "the recorded last checkpoint must be the run's final epoch "
            f"({normalized['epochs']}), got {normalized_checkpoints['last']['epoch']}"
        )
    return normalized


def check_run_config_checkpoint_consistency(
    config: Mapping[str, Any],
    metadata: Any,
    assert_epoch_stage: bool = True,
) -> dict:
    """Strict contract between a resolved run config and checkpoint metadata.

    Every compared field is type-checked first (strict types: no bool/int
    or float/int coercion), then compared by value, with explicit field
    mapping (``config.batch`` <-> ``metadata.physical_batch``). A config
    value that is empty (legacy/template sources cannot assert it) is
    recorded as unasserted rather than silently masked with ``or``.
    ``assert_epoch_stage=False`` (legacy sources without a recorded warmup
    schedule) records the epoch/stage boundary checks as unasserted
    instead of asserting them against template defaults.
    Returns ``{"asserted": {...}, "unasserted": [...]}``.
    """
    config = validate_material_config(config)
    meta = (
        dict(metadata) if isinstance(metadata, Mapping) else dataclasses.asdict(metadata)
    )

    def mismatch(field: str, config_value: Any, meta_value: Any) -> None:
        raise ValueError(
            f"run config {field} {config_value!r} does not match the checkpoint "
            f"{field} {meta_value!r}"
        )

    asserted: dict[str, Any] = {}
    pairs = (
        ("variant", config["variant"], validate_variant(meta["variant"])),
        ("imgsz", config["imgsz"], require_int(meta["imgsz"], "metadata.imgsz", minimum=1)),
        ("preprocessing_version", config["preprocessing_version"],
         require_str(meta["preprocessing_version"], "metadata.preprocessing_version")),
        ("upstream_commit", config["upstream_commit"],
         require_str(meta["upstream_commit"], "metadata.upstream_commit")),
        ("seed", config["seed"], require_int(meta["seed"], "metadata.seed")),
        ("batch", config["batch"],
         require_int(meta["physical_batch"], "metadata.physical_batch", minimum=1)),
        ("accum_steps", config["accum_steps"],
         require_int(meta["accum_steps"], "metadata.accum_steps", minimum=1)),
        ("effective_batch", config["effective_batch"],
         require_int(meta["effective_batch"], "metadata.effective_batch", minimum=1)),
        ("smoke", config["smoke"], require_bool(meta["smoke"], "metadata.smoke")),
        ("limit_batches", config["limit_batches"],
         require_int(meta["limit_batches"], "metadata.limit_batches", minimum=0)),
    )
    for field, config_value, meta_value in pairs:
        if config_value != meta_value:
            mismatch(field, config_value, meta_value)
        asserted[field] = config_value

    unasserted: list[str] = []
    # A mask-pooling run is defined by the concrete V5 dataset it trained on:
    # both sides must name it, so an empty value on either side is a mismatch
    # rather than an "unasserted" field.
    if config["pooling_mode"] == POOLING_INSTANCE_MASK:
        if not config["data_fingerprint"]:
            raise ValueError(
                "a mask-pooling run config must pin the V5 dataset fingerprint "
                "(data_fingerprint); an empty value would accept any V5 dataset"
            )
        if not require_str(meta["data_fingerprint"], "metadata.data_fingerprint"):
            raise ValueError(
                "a mask-pooling checkpoint must record the V5 dataset fingerprint "
                "in data_fingerprint; an empty value cannot bind the weights to "
                "their data"
            )
    for field in ("pretrained_sha256", "data_fingerprint"):
        config_value = config[field]
        meta_value = require_str(meta[field], f"metadata.{field}")
        if not config_value:
            unasserted.append(field)
            continue
        if config_value != meta_value:
            mismatch(field, config_value, meta_value)
        asserted[field] = config_value

    # The pooling contract must agree in both directions: a config that
    # declares mask pooling may never be paired with a GAP checkpoint (and
    # vice versa), whatever the parameter shapes look like.
    meta_pooling = validate_pooling_mode(
        require_str(meta.get("pooling_mode", POOLING_GAP), "metadata.pooling_mode")
    )
    if config["pooling_mode"] != meta_pooling:
        mismatch("pooling_mode", config["pooling_mode"], meta_pooling)
    asserted["pooling_mode"] = config["pooling_mode"]
    for field in ("mask_schema", "mask_resize_version", "parent_data_fingerprint"):
        config_value = config[field]
        meta_value = require_str(meta.get(field, ""), f"metadata.{field}")
        if not config_value:
            unasserted.append(field)
            continue
        if config_value != meta_value:
            mismatch(field, config_value, meta_value)
        asserted[field] = config_value

    epoch = require_int(meta["epoch"], "metadata.epoch", minimum=1)
    stage = require_str(meta["stage"], "metadata.stage")
    if stage not in STAGES:
        raise ValueError(f"metadata.stage must be one of {STAGES}, got {stage!r}")
    if not assert_epoch_stage:
        unasserted.extend(["epoch", "stage", "epochs", "warmup_epochs"])
        return {"asserted": asserted, "unasserted": unasserted}
    if epoch > config["epochs"]:
        raise ValueError(
            f"checkpoint epoch {epoch} exceeds the run config's epochs "
            f"{config['epochs']}"
        )
    expected_stage = expected_stage_for_epoch(epoch, config["warmup_epochs"])
    if stage != expected_stage:
        raise ValueError(
            f"checkpoint stage {stage!r} does not match the run config's protocol "
            f"stage {expected_stage!r} for epoch {epoch} "
            f"(warmup_epochs={config['warmup_epochs']})"
        )
    asserted["epoch"] = epoch
    asserted["stage"] = stage
    return {"asserted": asserted, "unasserted": unasserted}


# ---------------------------------------------------------------------------
# Model


def mask_level_weights(instance_mask: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Area-resample an instance mask to one feature level's spatial size.

    ``instance_mask`` is ``(B, H, W)`` or ``(B, 1, H, W)`` with values in
    [0, 1]. Resampling is pure area averaging (``mode="area"``, i.e. the
    mean of the source region mapped to each output cell), so it never
    thresholds: a target smaller than one output cell still yields a
    positive weight and can never be silently zeroed away. The result is
    fractional coverage in [0, 1] per output cell.

    The mask must be at least as large as the level; upsampling a mask
    would be a different (and unrecorded) algorithm, so it is refused.
    """
    if instance_mask.dim() == 3:
        instance_mask = instance_mask.unsqueeze(1)
    if instance_mask.dim() != 4 or instance_mask.shape[1] != 1:
        raise ValueError(
            "instance_mask must have shape (B, H, W) or (B, 1, H, W), got "
            f"{tuple(instance_mask.shape)}"
        )
    if not torch.isfinite(instance_mask).all():
        raise ValueError("instance_mask contains NaN/Inf")
    if bool((instance_mask < 0).any()) or bool((instance_mask > 1).any()):
        raise ValueError("instance_mask values must lie in [0, 1]")
    target = (int(size[0]), int(size[1]))
    source = (int(instance_mask.shape[-2]), int(instance_mask.shape[-1]))
    if source[0] < target[0] or source[1] < target[1]:
        raise ValueError(
            f"instance_mask {source} is smaller than the level {target}; area "
            "resampling is defined for downscaling only"
        )
    return F.interpolate(instance_mask.to(torch.float32), size=target, mode="area")


class MultiScaleClassificationHead(nn.Module):
    """Per-level 1x1 projection, pooling, concat, LayerNorm/Dropout/Linear.

    ``pooling_mode="gap"`` is the v1 head: global average pooling per level.
    ``pooling_mode="instance_mask"`` replaces it with the instance-mask
    weighted spatial mean

        pooled[c] = sum(F[c,h,w] * M[h,w]) / sum(M[h,w])

    where ``M`` is the instance mask area-resampled to that level. ``M``
    comes only from the annotated instance mask — never from the polar
    valid channel or from q — and every variant uses the same ``M``. A
    level whose weight sum is zero is refused instead of silently falling
    back to GAP.

    The encoder's convolutions and attention already mix spatial
    information, so instance pooling cannot guarantee background isolation;
    it only removes the *pooling* operator's own background contribution.
    """

    def __init__(
        self,
        in_dims: Sequence[int],
        num_classes: int = NUM_CLASSES,
        proj_dim: int = HEAD_PROJ_DIM,
        dropout: float = HEAD_DROPOUT,
        pooling_mode: str = POOLING_GAP,
    ):
        super().__init__()
        if len(in_dims) != 4:
            raise ValueError(f"expected four encoder levels, got {len(in_dims)}")
        self.in_dims = tuple(int(dim) for dim in in_dims)
        self.num_classes = int(num_classes)
        self.proj_dim = int(proj_dim)
        self.pooling_mode = validate_pooling_mode(pooling_mode)
        self.proj = nn.ModuleList(
            [nn.Conv2d(dim, self.proj_dim, kernel_size=1) for dim in self.in_dims]
        )
        self.norm = nn.LayerNorm(self.proj_dim * len(self.in_dims))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.proj_dim * len(self.in_dims), self.num_classes)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        instance_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(features) != len(self.proj):
            raise ValueError(
                f"head expects {len(self.proj)} feature levels, got {len(features)}"
            )
        if self.pooling_mode == POOLING_INSTANCE_MASK:
            if instance_mask is None:
                raise ValueError(
                    "this head pools with the instance mask; an instance_mask "
                    "tensor is required (refusing to fall back to GAP)"
                )
            pooled = [
                self._masked_pool(proj(feature), instance_mask)
                for proj, feature in zip(self.proj, features)
            ]
        else:
            if instance_mask is not None:
                raise ValueError(
                    "this head pools with global average pooling; an "
                    "instance_mask was passed, which would be silently ignored"
                )
            pooled = [
                F.adaptive_avg_pool2d(proj(feature), 1).flatten(1)
                for proj, feature in zip(self.proj, features)
            ]
        return self.fc(self.drop(self.norm(torch.cat(pooled, dim=1))))

    @staticmethod
    def _masked_pool(projected: torch.Tensor, instance_mask: torch.Tensor) -> torch.Tensor:
        """Instance-mask weighted spatial mean of one projected level."""
        weights = mask_level_weights(instance_mask, projected.shape[-2:]).to(
            dtype=projected.dtype
        )
        total = weights.sum(dim=(2, 3))
        empty = total <= 0
        if bool(empty.any()):
            rows = empty.nonzero(as_tuple=False).flatten().tolist()[:10]
            raise ValueError(
                "instance-mask pooling has zero weight at level "
                f"{tuple(projected.shape[-2:])} for batch rows {rows}; an empty "
                "mask is refused rather than silently pooled with GAP"
            )
        return (projected * weights).sum(dim=(2, 3)) / total


class MaterialFusionModel(nn.Module):
    """Four-level encoder plus the shared classification head.

    ``forward(rgb)`` for the single-stream ``gray`` variant and
    ``forward(rgb, aux)`` for the dual-stream CMX variants; returns raw
    logits ``(B, num_classes)``. With ``pooling_mode="instance_mask"`` the
    call additionally requires ``instance_mask`` (``(B, H, W)`` in [0, 1]).
    """

    def __init__(
        self,
        variant: str,
        num_classes: int = NUM_CLASSES,
        imgsz: int = 224,
        drop_path_rate: float = MIT_B0_SPEC["drop_path_rate"],
        pooling_mode: str = POOLING_GAP,
    ):
        super().__init__()
        self.variant = validate_variant(variant)
        self.num_classes = int(num_classes)
        self.imgsz = int(imgsz)
        self.drop_path_rate = float(drop_path_rate)
        self.pooling_mode = validate_pooling_mode(pooling_mode)
        if self.variant == "gray":
            self.encoder = MiTEncoder(img_size=self.imgsz, drop_path_rate=self.drop_path_rate)
        else:
            self.encoder = mit_b0(drop_path_rate=self.drop_path_rate)
        self.head = MultiScaleClassificationHead(
            ENCODER_LEVEL_DIMS, self.num_classes, pooling_mode=self.pooling_mode
        )

    @property
    def is_dual(self) -> bool:
        return variant_is_dual(self.variant)

    @property
    def uses_instance_mask(self) -> bool:
        return self.pooling_mode == POOLING_INSTANCE_MASK

    def forward(
        self,
        rgb: torch.Tensor,
        aux: torch.Tensor | None = None,
        instance_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_dual and aux is None:
            raise ValueError(
                f"variant {self.variant!r} needs the auxiliary input tensor "
                "(aux=None received)"
            )
        if not self.is_dual and aux is not None:
            raise ValueError(
                "the single-stream gray variant does not accept an auxiliary input"
            )
        features = self.encoder(rgb, aux) if self.is_dual else self.encoder(rgb)
        return self.head(features, instance_mask)

    def encoder_parameters(self) -> list[nn.Parameter]:
        """All encoder parameters (pretrained backbone + fusion layers)."""
        return list(self.encoder.parameters())

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """The three protocol groups; every parameter appears exactly once.

        ``backbone`` — the pretrained MiT streams (frozen during warmup,
        ``encoder_lr`` after unfreezing).
        ``fusion``   — the newly initialized FRM/FFM layers (empty for the
        single-stream ``gray`` variant; ``lr`` in both stages).
        ``head``     — the classification head (``lr`` in both stages).
        """
        groups: dict[str, list[nn.Parameter]] = {"backbone": [], "fusion": [], "head": []}
        for name, param in self.encoder.named_parameters():
            top = name.split(".")[0]
            if top.startswith(_FUSION_MODULE_PREFIXES):
                groups["fusion"].append(param)
            elif top.startswith(_BACKBONE_MODULE_PREFIXES):
                groups["backbone"].append(param)
            else:
                raise ValueError(f"unclassified encoder parameter: {name}")
        groups["head"] = list(self.head.parameters())
        return groups

    def backbone_parameters(self) -> list[nn.Parameter]:
        return self.parameter_groups()["backbone"]

    def fusion_parameters(self) -> list[nn.Parameter]:
        return self.parameter_groups()["fusion"]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Toggle the pretrained backbone only.

        Freezing also puts the backbone modules in eval mode (DropPath must
        be identity for a frozen feature extractor); the fusion layers and
        the head are never touched here, and the graph through the backbone
        stays connected so gradients still reach the fusion layers.
        """
        for name, module in self.encoder.named_children():
            top = name.split(".")[0]
            if not top.startswith(_BACKBONE_MODULE_PREFIXES):
                continue
            for param in module.parameters():
                param.requires_grad = not frozen
            module.train(not frozen)

    def set_stage(self, stage: str) -> None:
        """Apply the warmup/finetune training protocol.

        warmup:   pretrained backbone frozen and in eval mode; FRM/FFM and
                  the head trainable and in train mode (their BatchNorm
                  statistics keep updating).
        finetune: everything trainable and in train mode.
        """
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        self.train()
        self.set_backbone_frozen(stage == "warmup")


def build_material_model(
    variant: str,
    num_classes: int = NUM_CLASSES,
    imgsz: int = 224,
    drop_path_rate: float = MIT_B0_SPEC["drop_path_rate"],
    pooling_mode: str = POOLING_GAP,
) -> MaterialFusionModel:
    return MaterialFusionModel(
        variant,
        num_classes=num_classes,
        imgsz=imgsz,
        drop_path_rate=drop_path_rate,
        pooling_mode=pooling_mode,
    )


def load_pretrained_into_model(
    model: MaterialFusionModel,
    path: str | Path,
    expected_sha256: str = "",
) -> dict:
    """Load the MiT-B0 pretrained encoder into ``model`` (strict)."""
    report = load_mit_b0_checkpoint(model.encoder, path, expected_sha256=expected_sha256)
    report["variant"] = model.variant
    return report


# ---------------------------------------------------------------------------
# Checkpoint metadata


@dataclasses.dataclass(frozen=True)
class MaterialFusionMetadata:
    """Audit metadata stored with every material-fusion checkpoint.

    The last four fields describe the second-round mask-pooling protocol.
    They carry defaults so v1 checkpoints (which never wrote them) keep
    loading with their original GAP behaviour; a ``material_fusion_v2_mask``
    checkpoint must carry them explicitly, which
    :func:`validate_checkpoint_metadata` enforces.
    """

    version: str
    variant: str
    class_names: tuple[str, ...]
    imgsz: int
    preprocessing_version: str
    data_fingerprint: str
    upstream_commit: str
    pretrained_sha256: str
    seed: int
    epoch: int
    stage: str
    smoke: bool
    limit_batches: int
    val_macro_f1: float
    val_accuracy: float
    physical_batch: int
    accum_steps: int
    effective_batch: int
    drop_path_rate: float
    init_from: str
    fallback_weights: str
    fallback_identity: str
    pooling_mode: str = POOLING_GAP
    mask_schema: str = ""
    mask_resize_version: str = ""
    parent_data_fingerprint: str = ""


def material_fusion_metadata(
    *,
    version: str = MATERIAL_FUSION_VERSION,
    variant: str,
    class_names: Sequence[str] = CLASS_NAMES,
    imgsz: int,
    preprocessing_version: str = PREPROCESSING_VERSION,
    data_fingerprint: str,
    upstream_commit: str = UPSTREAM_COMMIT,
    pretrained_sha256: str,
    seed: int,
    epoch: int,
    stage: str,
    smoke: bool,
    limit_batches: int,
    val_macro_f1: float,
    val_accuracy: float,
    physical_batch: int,
    accum_steps: int,
    effective_batch: int,
    drop_path_rate: float,
    init_from: str = "",
    fallback_weights: str = "",
    fallback_identity: str = "",
    pooling_mode: str = POOLING_GAP,
    mask_schema: str = "",
    mask_resize_version: str = "",
    parent_data_fingerprint: str = "",
) -> MaterialFusionMetadata:
    return MaterialFusionMetadata(
        version=str(version),
        variant=validate_variant(variant),
        class_names=tuple(str(name) for name in class_names),
        imgsz=int(imgsz),
        preprocessing_version=str(preprocessing_version),
        data_fingerprint=str(data_fingerprint),
        upstream_commit=str(upstream_commit),
        pretrained_sha256=str(pretrained_sha256),
        seed=int(seed),
        epoch=int(epoch),
        stage=str(stage),
        smoke=smoke,
        limit_batches=limit_batches,
        val_macro_f1=float(val_macro_f1),
        val_accuracy=float(val_accuracy),
        physical_batch=int(physical_batch),
        accum_steps=int(accum_steps),
        effective_batch=int(effective_batch),
        drop_path_rate=float(drop_path_rate),
        init_from=str(init_from),
        fallback_weights=str(fallback_weights),
        fallback_identity=str(fallback_identity),
        pooling_mode=validate_pooling_mode(str(pooling_mode)),
        mask_schema=str(mask_schema),
        mask_resize_version=str(mask_resize_version),
        parent_data_fingerprint=str(parent_data_fingerprint),
    )


def validate_checkpoint_metadata(payload: Mapping[str, Any]) -> dict:
    """Strictly validate one checkpoint payload's metadata.

    Field-by-field checks with strict types; ``smoke`` must be a real bool
    and ``limit_batches`` a real int (bools and floats are refused even
    though Python's bool subclasses int). ``smoke`` must agree with
    ``limit_batches`` (formal runs use 0, smoke runs > 0).
    """
    if not isinstance(payload, Mapping) or "metadata" not in payload or "state_dict" not in payload:
        raise ValueError(
            "checkpoint payload must be a mapping with 'metadata' and 'state_dict'"
        )
    meta = payload["metadata"]
    if not isinstance(meta, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    required = (
        "version", "variant", "class_names", "imgsz", "preprocessing_version",
        "data_fingerprint", "upstream_commit", "pretrained_sha256", "seed",
        "epoch", "stage", "smoke", "limit_batches", "val_macro_f1",
        "val_accuracy", "physical_batch", "accum_steps", "effective_batch",
        "drop_path_rate", "init_from", "fallback_weights", "fallback_identity",
    )
    missing = [field for field in required if field not in meta]
    if missing:
        raise ValueError(f"checkpoint metadata is missing fields: {missing}")

    version = require_str(meta["version"], "version")
    if version not in SUPPORTED_EXPERIMENT_VERSIONS:
        raise ValueError(
            f"version must be one of {SUPPORTED_EXPERIMENT_VERSIONS}, got {version!r}"
        )
    # Strict protocol <-> pooling mapping, both directions:
    #   v1  -> GAP only. The v1 format predates the field, so a missing
    #          pooling_mode is GAP by that format's definition; an explicit
    #          instance_mask on a v1 checkpoint is refused.
    #   v2  -> instance_mask only, declared explicitly. A v2 checkpoint that
    #          omits pooling_mode, or declares GAP, is refused rather than
    #          guessed.
    if "pooling_mode" in meta:
        declared_pooling = validate_pooling_mode(
            require_str(meta["pooling_mode"], "pooling_mode")
        )
    elif version == MATERIAL_FUSION_MASK_VERSION:
        raise ValueError(
            f"a {MATERIAL_FUSION_MASK_VERSION} checkpoint must record pooling_mode; "
            "refusing to guess how its head pools"
        )
    else:
        declared_pooling = POOLING_GAP
    if version == MATERIAL_FUSION_MASK_VERSION and declared_pooling != POOLING_INSTANCE_MASK:
        raise ValueError(
            f"a {MATERIAL_FUSION_MASK_VERSION} checkpoint must record pooling_mode "
            f"{POOLING_INSTANCE_MASK!r}, got {declared_pooling!r}; the v2 protocol "
            "is defined by instance-mask pooling"
        )
    if version == MATERIAL_FUSION_VERSION and declared_pooling != POOLING_GAP:
        raise ValueError(
            f"a {MATERIAL_FUSION_VERSION} checkpoint must be a {POOLING_GAP!r} "
            f"(global average pooling) checkpoint, got pooling_mode "
            f"{declared_pooling!r}; refusing a contradictory protocol declaration"
        )
    # A v2 checkpoint is bound to one concrete V5 dataset. The fingerprint is
    # the only field that names it, so an empty (or malformed) value would
    # make the checkpoint acceptable on any V5 derivation; it is refused here
    # instead of being repaired by str()/strip()/zero-padding.
    if version == MATERIAL_FUSION_MASK_VERSION:
        fingerprint = require_str(meta["data_fingerprint"], "data_fingerprint")
        if _SHA256_PATTERN.match(fingerprint) is None:
            raise ValueError(
                f"a {MATERIAL_FUSION_MASK_VERSION} checkpoint must record the "
                "64 lowercase hex sha256 of the V5 dataset it was trained on in "
                f"data_fingerprint, got {fingerprint!r}; an unset fingerprint "
                "would not bind the weights to their data"
            )

    normalized: dict[str, Any] = {
        "version": version,
        "variant": validate_variant(meta["variant"]),
        "class_names": _require_class_names(meta["class_names"]),
        "imgsz": require_int(meta["imgsz"], "imgsz", minimum=1),
        "preprocessing_version": require_str(
            meta["preprocessing_version"], "preprocessing_version"
        ),
        "data_fingerprint": require_str(meta["data_fingerprint"], "data_fingerprint"),
        "upstream_commit": require_str(meta["upstream_commit"], "upstream_commit"),
        "pretrained_sha256": require_str(meta["pretrained_sha256"], "pretrained_sha256"),
        "seed": require_int(meta["seed"], "seed"),
        "epoch": require_int(meta["epoch"], "epoch", minimum=1),
        "stage": require_str(meta["stage"], "stage"),
        "smoke": require_bool(meta["smoke"], "smoke"),
        "limit_batches": require_int(meta["limit_batches"], "limit_batches", minimum=0),
        "val_macro_f1": require_float(meta["val_macro_f1"], "val_macro_f1"),
        "val_accuracy": require_float(meta["val_accuracy"], "val_accuracy"),
        "physical_batch": require_int(meta["physical_batch"], "physical_batch", minimum=1),
        "accum_steps": require_int(meta["accum_steps"], "accum_steps", minimum=1),
        "effective_batch": require_int(meta["effective_batch"], "effective_batch", minimum=1),
        "drop_path_rate": require_float(meta["drop_path_rate"], "drop_path_rate", minimum=0.0),
        "init_from": require_str(meta["init_from"], "init_from"),
        "fallback_weights": require_str(meta["fallback_weights"], "fallback_weights"),
        "fallback_identity": require_str(meta["fallback_identity"], "fallback_identity"),
        "pooling_mode": declared_pooling,
        "mask_schema": require_str(meta.get("mask_schema", ""), "mask_schema"),
        "mask_resize_version": require_str(
            meta.get("mask_resize_version", ""), "mask_resize_version"
        ),
        "parent_data_fingerprint": require_str(
            meta.get("parent_data_fingerprint", ""), "parent_data_fingerprint"
        ),
    }
    if normalized["stage"] not in STAGES:
        raise ValueError(
            f"stage must be one of {STAGES}, got {normalized['stage']!r}"
        )
    if normalized["pooling_mode"] == POOLING_INSTANCE_MASK:
        if normalized["mask_schema"] != MASK_SCHEMA_VERSION:
            raise ValueError(
                f"a mask-pooling checkpoint must record mask_schema "
                f"{MASK_SCHEMA_VERSION!r}, got {normalized['mask_schema']!r}"
            )
        if normalized["mask_resize_version"] != MASK_RESIZE_VERSION:
            raise ValueError(
                f"a mask-pooling checkpoint must record mask_resize_version "
                f"{MASK_RESIZE_VERSION!r}, got "
                f"{normalized['mask_resize_version']!r}"
            )
        if _SHA256_PATTERN.match(normalized["parent_data_fingerprint"]) is None:
            raise ValueError(
                "a mask-pooling checkpoint must record the 64-hex V4 parent "
                f"fingerprint, got {normalized['parent_data_fingerprint']!r}"
            )
    if normalized["physical_batch"] * normalized["accum_steps"] != normalized["effective_batch"]:
        raise ValueError(
            "effective_batch must equal physical_batch * accum_steps, got "
            f"{normalized['physical_batch']} * {normalized['accum_steps']} != "
            f"{normalized['effective_batch']}"
        )
    if normalized["smoke"] != (normalized["limit_batches"] > 0):
        raise ValueError(
            "smoke must be True exactly when limit_batches > 0 (formal runs "
            f"use limit_batches=0); got smoke={normalized['smoke']}, "
            f"limit_batches={normalized['limit_batches']}"
        )
    return normalized


def _metadata_to_payload_dict(metadata: MaterialFusionMetadata) -> dict:
    return dict(dataclasses.asdict(metadata))


def save_material_checkpoint(
    path: str | Path,
    model: MaterialFusionModel,
    metadata: MaterialFusionMetadata,
) -> Path:
    """Save ``{"metadata", "state_dict"}``; never touches the live model.

    The payload is validated with the same strict validator used on load
    before it is written, so a checkpoint whose protocol version and pooling
    mode contradict each other (or whose metadata disagrees with the model
    that is actually being saved) can never reach disk.
    """
    if tuple(metadata.class_names) != CLASS_NAMES:
        raise ValueError(
            f"class_names must be exactly {CLASS_NAMES}, got {metadata.class_names}"
        )
    if len(metadata.class_names) != model.num_classes:
        raise ValueError(
            f"class_names ({len(metadata.class_names)}) != num_classes "
            f"({model.num_classes})"
        )
    if metadata.variant != model.variant:
        raise ValueError(
            f"metadata variant {metadata.variant!r} != model variant {model.variant!r}"
        )
    model_pooling = validate_pooling_mode(
        require_str(model.pooling_mode, "model.pooling_mode")
    )
    if metadata.pooling_mode != model_pooling:
        raise ValueError(
            f"metadata pooling_mode {metadata.pooling_mode!r} != the model's actual "
            f"pooling_mode {model_pooling!r}; refusing to save a checkpoint whose "
            "metadata misdescribes how it pools"
        )
    payload = {
        "metadata": _metadata_to_payload_dict(metadata),
        "state_dict": model.state_dict(),
    }
    # Round-trip through the loader's validator: identical parameter shapes do
    # not make two protocols interchangeable, and this is where a version /
    # pooling contradiction is caught on the write side too.
    validate_checkpoint_metadata(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def read_material_checkpoint_metadata(path: str | Path) -> MaterialFusionMetadata:
    """Read and strictly validate the metadata of a checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    normalized = validate_checkpoint_metadata(payload)
    return MaterialFusionMetadata(**normalized)


def load_material_checkpoint(
    path: str | Path,
    variant: str,
    imgsz: int | None = None,
    num_classes: int = NUM_CLASSES,
    pooling_mode: str | None = None,
) -> tuple[MaterialFusionModel, MaterialFusionMetadata, dict]:
    """Rebuild a model from a checkpoint, verifying the requested contract.

    ``variant`` (and ``imgsz`` when given) must match the recorded metadata
    exactly; a mismatch is refused instead of silently rebuilding something
    else. ``pooling_mode`` (when given) must match the recorded mode too, so
    a v1 GAP checkpoint cannot be loaded as — or re-interpreted as — a
    mask-pooling model.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    normalized = validate_checkpoint_metadata(payload)
    metadata = MaterialFusionMetadata(**normalized)
    validate_variant(variant)
    if metadata.variant != variant:
        raise ValueError(
            f"checkpoint {path} records variant {metadata.variant!r}, "
            f"requested {variant!r}"
        )
    if imgsz is not None and int(imgsz) != metadata.imgsz:
        raise ValueError(
            f"checkpoint {path} records imgsz {metadata.imgsz}, requested {int(imgsz)}"
        )
    if pooling_mode is not None:
        validate_pooling_mode(pooling_mode)
        if pooling_mode != metadata.pooling_mode:
            raise ValueError(
                f"checkpoint {path} records pooling_mode "
                f"{metadata.pooling_mode!r}, requested {pooling_mode!r}"
            )
    model = build_material_model(
        metadata.variant,
        num_classes=num_classes,
        imgsz=metadata.imgsz,
        drop_path_rate=metadata.drop_path_rate,
        pooling_mode=metadata.pooling_mode,
    )
    model.load_state_dict(payload["state_dict"])
    return model, metadata, payload


# ---------------------------------------------------------------------------
# Experiment config validation and config/checkpoint consistency


def validate_material_config(config: Mapping[str, Any]) -> dict:
    """Strictly validate an experiment config mapping (e.g. loaded YAML)."""
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    required = (
        "experiment", "variant", "data", "data_fingerprint", "pretrained",
        "pretrained_sha256", "upstream_commit", "preprocessing_version",
        "imgsz", "batch", "accum_steps", "effective_batch", "epochs",
        "warmup_epochs", "optimizer", "weight_decay", "lr", "encoder_lr",
        "seed", "workers", "fallback_weights", "smoke", "limit_batches",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"experiment config is missing fields: {missing}")

    normalized = {
        "experiment": require_str(config["experiment"], "experiment"),
        "variant": validate_variant(config["variant"]),
        "data": require_str(config["data"], "data"),
        "data_fingerprint": require_str(config["data_fingerprint"], "data_fingerprint"),
        "pretrained": require_str(config["pretrained"], "pretrained"),
        "pretrained_sha256": require_str(config["pretrained_sha256"], "pretrained_sha256"),
        "upstream_commit": require_str(config["upstream_commit"], "upstream_commit"),
        "preprocessing_version": require_str(
            config["preprocessing_version"], "preprocessing_version"
        ),
        "imgsz": require_int(config["imgsz"], "imgsz", minimum=1),
        "batch": require_int(config["batch"], "batch", minimum=1),
        "accum_steps": require_int(config["accum_steps"], "accum_steps", minimum=1),
        "effective_batch": require_int(config["effective_batch"], "effective_batch", minimum=1),
        "epochs": require_int(config["epochs"], "epochs", minimum=1),
        "warmup_epochs": require_int(config["warmup_epochs"], "warmup_epochs", minimum=0),
        "optimizer": require_str(config["optimizer"], "optimizer"),
        "weight_decay": require_float(config["weight_decay"], "weight_decay", minimum=0.0),
        "lr": require_float(config["lr"], "lr", minimum=0.0),
        "encoder_lr": require_float(config["encoder_lr"], "encoder_lr", minimum=0.0),
        "seed": require_int(config["seed"], "seed"),
        "workers": require_int(config["workers"], "workers", minimum=0),
        "fallback_weights": require_str(config["fallback_weights"], "fallback_weights"),
        "smoke": require_bool(config["smoke"], "smoke"),
        "limit_batches": require_int(config["limit_batches"], "limit_batches", minimum=0),
        # Second-round fields. Optional so v1 configs keep validating; a v2
        # config is required to declare them explicitly below.
        "pooling_mode": validate_pooling_mode(
            require_str(config.get("pooling_mode", POOLING_GAP), "pooling_mode")
        ),
        "mask_schema": require_str(config.get("mask_schema", ""), "mask_schema"),
        "mask_resize_version": require_str(
            config.get("mask_resize_version", ""), "mask_resize_version"
        ),
        "parent_data_fingerprint": require_str(
            config.get("parent_data_fingerprint", ""), "parent_data_fingerprint"
        ),
    }
    if normalized["experiment"] not in SUPPORTED_EXPERIMENT_VERSIONS:
        raise ValueError(
            f"experiment must be one of {SUPPORTED_EXPERIMENT_VERSIONS}, got "
            f"{normalized['experiment']!r}"
        )
    if normalized["experiment"] == MATERIAL_FUSION_MASK_VERSION:
        if normalized["pooling_mode"] != POOLING_INSTANCE_MASK:
            raise ValueError(
                f"a {MATERIAL_FUSION_MASK_VERSION} config must set pooling_mode "
                f"{POOLING_INSTANCE_MASK!r}, got {normalized['pooling_mode']!r}"
            )
        if normalized["mask_schema"] != MASK_SCHEMA_VERSION:
            raise ValueError(
                f"a {MATERIAL_FUSION_MASK_VERSION} config must set mask_schema "
                f"{MASK_SCHEMA_VERSION!r}, got {normalized['mask_schema']!r}"
            )
        if normalized["mask_resize_version"] != MASK_RESIZE_VERSION:
            raise ValueError(
                f"a {MATERIAL_FUSION_MASK_VERSION} config must set "
                f"mask_resize_version {MASK_RESIZE_VERSION!r}, got "
                f"{normalized['mask_resize_version']!r}"
            )
        if _SHA256_PATTERN.match(normalized["parent_data_fingerprint"]) is None:
            raise ValueError(
                "a mask-pooling config must pin the 64-hex V4 parent fingerprint "
                f"in parent_data_fingerprint, got "
                f"{normalized['parent_data_fingerprint']!r}"
            )
        if _SHA256_PATTERN.match(normalized["data_fingerprint"]) is None:
            raise ValueError(
                "a mask-pooling config must pin the 64-hex fingerprint of the V5 "
                f"dataset in data_fingerprint, got "
                f"{normalized['data_fingerprint']!r}; an empty fingerprint would "
                "admit any V5 derivation"
            )
    elif normalized["pooling_mode"] != POOLING_GAP:
        raise ValueError(
            f"experiment {normalized['experiment']!r} is the GAP protocol; "
            f"pooling_mode must be {POOLING_GAP!r}, got "
            f"{normalized['pooling_mode']!r}"
        )
    if normalized["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError(
            f"upstream_commit must pin {UPSTREAM_COMMIT}, got "
            f"{normalized['upstream_commit']!r}"
        )
    if normalized["preprocessing_version"] != PREPROCESSING_VERSION:
        raise ValueError(
            f"preprocessing_version must be {PREPROCESSING_VERSION!r}, got "
            f"{normalized['preprocessing_version']!r}"
        )
    if normalized["optimizer"].lower() != "adamw":
        raise ValueError(
            f"optimizer must be 'adamw' (fixed protocol), got {normalized['optimizer']!r}"
        )
    if normalized["imgsz"] % 32 != 0:
        raise ValueError(
            f"imgsz must be a multiple of 32 (MiT stride), got {normalized['imgsz']}"
        )
    if normalized["batch"] * normalized["accum_steps"] != normalized["effective_batch"]:
        raise ValueError(
            "effective_batch must equal batch * accum_steps, got "
            f"{normalized['batch']} * {normalized['accum_steps']} != "
            f"{normalized['effective_batch']}"
        )
    if normalized["warmup_epochs"] >= normalized["epochs"]:
        raise ValueError(
            f"warmup_epochs ({normalized['warmup_epochs']}) must be smaller than "
            f"epochs ({normalized['epochs']})"
        )
    if normalized["smoke"] != (normalized["limit_batches"] > 0):
        raise ValueError(
            "smoke must be True exactly when limit_batches > 0 (formal runs "
            f"use limit_batches=0); got smoke={normalized['smoke']}, "
            f"limit_batches={normalized['limit_batches']}"
        )
    sha = normalized["pretrained_sha256"]
    if sha and _SHA256_PATTERN.match(sha) is None:
        raise ValueError(
            f"pretrained_sha256 must be empty or 64 lowercase hex chars, got {sha!r}"
        )
    return normalized


def check_config_checkpoint_consistency(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Refuse a config/checkpoint pair that describes different runs."""
    checks = (
        ("variant", config.get("variant"), metadata.get("variant")),
        ("imgsz", config.get("imgsz"), metadata.get("imgsz")),
        (
            "preprocessing_version",
            config.get("preprocessing_version"),
            metadata.get("preprocessing_version"),
        ),
        ("upstream_commit", config.get("upstream_commit"), metadata.get("upstream_commit")),
    )
    for field, config_value, meta_value in checks:
        if config_value is None:
            continue
        if config_value != meta_value:
            raise ValueError(
                f"config {field} {config_value!r} does not match checkpoint "
                f"{field} {meta_value!r}"
            )
    if config.get("batch") is not None and config.get("accum_steps") is not None:
        effective = int(config["batch"]) * int(config["accum_steps"])
        if effective != metadata.get("effective_batch"):
            raise ValueError(
                f"config effective_batch {effective} (batch {config['batch']} x "
                f"accum {config['accum_steps']}) does not match checkpoint "
                f"effective_batch {metadata.get('effective_batch')}"
            )
    fingerprint = config.get("data_fingerprint")
    if fingerprint and fingerprint != metadata.get("data_fingerprint"):
        raise ValueError(
            f"config data_fingerprint {fingerprint!r} does not match checkpoint "
            f"data_fingerprint {metadata.get('data_fingerprint')!r}"
        )
    sha = config.get("pretrained_sha256")
    if sha and sha != metadata.get("pretrained_sha256"):
        raise ValueError(
            f"config pretrained_sha256 {sha!r} does not match checkpoint "
            f"pretrained_sha256 {metadata.get('pretrained_sha256')!r}"
        )
    config_pooling = config.get("pooling_mode")
    if config_pooling is not None:
        meta_pooling = validate_pooling_mode(
            str(metadata.get("pooling_mode", POOLING_GAP))
        )
        if validate_pooling_mode(str(config_pooling)) != meta_pooling:
            raise ValueError(
                f"config pooling_mode {config_pooling!r} does not match checkpoint "
                f"pooling_mode {meta_pooling!r}; a GAP checkpoint and a "
                "mask-pooling model are not interchangeable"
            )


__all__ = [
    "AUX_CHANNEL_MEANING",
    "CLASS_NAMES",
    "ENCODER_LEVEL_DIMS",
    "HEAD_DROPOUT",
    "HEAD_PROJ_DIM",
    "MATERIAL_FUSION_MASK_VERSION",
    "MATERIAL_FUSION_VERSION",
    "MASK_RESIZE_VERSION",
    "MASK_SCHEMA_VERSION",
    "MaterialFusionMetadata",
    "MaterialFusionModel",
    "MultiScaleClassificationHead",
    "NUM_CLASSES",
    "POOLING_GAP",
    "POOLING_INSTANCE_MASK",
    "POOLING_MODES",
    "PREPROCESSING_VERSION",
    "STAGES",
    "SUPPORTED_EXPERIMENT_VERSIONS",
    "UPSTREAM_COMMIT",
    "V4_PARENT_FINGERPRINT",
    "VARIANTS",
    "build_material_model",
    "check_config_checkpoint_consistency",
    "load_material_checkpoint",
    "load_pretrained_into_model",
    "mask_level_weights",
    "material_fusion_metadata",
    "read_material_checkpoint_metadata",
    "require_bool",
    "require_float",
    "require_int",
    "require_str",
    "save_material_checkpoint",
    "validate_checkpoint_metadata",
    "validate_material_config",
    "validate_pooling_mode",
    "validate_variant",
]
