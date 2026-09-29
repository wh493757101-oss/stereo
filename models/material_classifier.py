"""Runtime adapter for the frozen CMX v2_mask research candidates.

This module wires the ``material_fusion_v2_mask`` classification candidates
(``runs/train/cmx_v2_mask_<variant>_s<seed>/material_fusion/best.pt``) into
the independent ``material_fusion`` input mode of the inference engine.

It deliberately contains **no** new model mathematics: preprocessing reuses
the offline V4/V5 tensor path (:func:`models.polar_fusion._fusion_tensors`
and :func:`core.material_fusion_dataset.instance_mask_tensor`), the model
rebuild goes through the strict checkpoint loader
(:func:`models.material_fusion.load_material_checkpoint`), and the
polar-invalid routing reuses the production routing function
(:func:`scripts.eval_material_fusion.route_batch`).

Three regions stay distinct exactly as in the V5 dataset:

- ``instance_mask``  the target region (at runtime: the Model A prediction);
- ``polar_valid``    the measurement-valid region;
- their intersection the target-interior measurable region.

The independent gray fallback (the historical YOLO26 gray Model B) is used
**only** for polar-invalid instances of the dual-stream variants and is a
separate model identity in every result. The single-stream ``gray`` variant
(E1) is the pure control: it never loads or calls a fallback.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from core.material_fusion_dataset import (
    CLASS_NAMES,
    instance_mask_tensor,
    variant_inputs,
)
from models.material_fusion import (
    MATERIAL_FUSION_MASK_VERSION,
    MaterialFusionModel,
    load_material_checkpoint,
    read_material_checkpoint_metadata,
)
from models.polar_fusion import (
    QUALITY_VECTOR_LENGTH,
    VALID_RATIO_INDEX,
    torch_device_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where the instance mask of a runtime input came from. The live pipeline can
# only ever produce the predicted source; ground-truth masks are allowed in
# clearly labelled verification inputs only and never mix with the predicted
# path.
MASK_SOURCE_PREDICTED = "predicted_instance_mask"
MASK_SOURCE_GROUND_TRUTH = "ground_truth_annotation"
MASK_SOURCE_NPZ = "npz_offline_identity"
MASK_SOURCES = (MASK_SOURCE_PREDICTED, MASK_SOURCE_GROUND_TRUTH, MASK_SOURCE_NPZ)

REASON_VALID_RATIO_ZERO = "valid_ratio_zero"
REASON_NON_FINITE_POLAR = "non_finite_polar"


@dataclasses.dataclass(frozen=True)
class MaterialInstanceInput:
    """One instance's material-classification input at runtime.

    Carries the native-resolution crops (no resizing happens before this
    record exists) plus the stable identity fields needed to associate
    results back after batching: ``frame_id`` and ``instance_id``.
    ``invalid_reason`` explains a polar-invalid instance (sync skew
    exceeded, stereo matching failed, non-finite polar values, ...).
    """

    frame_id: str
    instance_id: int
    bbox: tuple[int, int, int, int]
    gray: np.ndarray
    signed_q: np.ndarray
    abs_q: np.ndarray
    polar_valid: np.ndarray
    instance_mask: np.ndarray
    quality: np.ndarray
    polar_invalid: bool
    invalid_reason: str
    mask_source: str = MASK_SOURCE_PREDICTED

    def __post_init__(self) -> None:
        if self.mask_source not in MASK_SOURCES:
            raise ValueError(
                f"mask_source must be one of {MASK_SOURCES}, got "
                f"{self.mask_source!r}; unlabelled mask provenance is refused"
            )
        if self.quality.shape != (QUALITY_VECTOR_LENGTH,):
            raise ValueError(
                f"quality must have shape ({QUALITY_VECTOR_LENGTH},), got "
                f"{self.quality.shape}"
            )

    @property
    def valid(self) -> np.ndarray:
        """The V4 measurement-valid channel under its dataset name."""
        return self.polar_valid


@dataclasses.dataclass(frozen=True)
class MaterialClassResult:
    """One material-classification outcome for one instance.

    ``valid=False`` marks an instance failure (no fabricated class label);
    ``used_fallback`` marks exactly the instances that went through the
    independent gray fallback, with ``fallback_reason`` stating why.
    """

    class_id: int
    class_name: str
    confidence: float
    valid: bool
    used_fallback: bool
    fallback_reason: str
    model_identity: str
    instance_id: int
    frame_id: str
    invalid_reason: str
    # The raw four-class logits of the model that actually produced this
    # result, at full float precision (per-sample evidence for evaluation;
    # empty for failures).
    raw_logits: tuple[float, ...] = ()
    # Whether this instance's polarization was marked unavailable. Reported
    # so a pure (no-routing) diagnostic can state its denominators.
    polar_invalid: bool = False

    @classmethod
    def failure(
        cls,
        sample: "MaterialInstanceInput | None",
        reason: str,
    ) -> "MaterialClassResult":
        instance_id = sample.instance_id if sample is not None else -1
        frame_id = sample.frame_id if sample is not None else ""
        return cls(
            class_id=-1,
            class_name="",
            confidence=0.0,
            valid=False,
            used_fallback=False,
            fallback_reason="",
            model_identity="",
            instance_id=instance_id,
            frame_id=frame_id,
            invalid_reason=reason,
        )


def material_input_tensors(
    sample: MaterialInstanceInput,
    imgsz: int,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Offline-identical preprocessing of one runtime instance input.

    Returns ``(rgb, aux, mask, quality)`` — exactly the tensors the offline
    :class:`core.material_fusion_dataset.MaterialFusionDataset` yields for
    the same pixel data, computed through the same shared code path.
    """
    import cv2

    from models.polar_fusion import _fusion_tensors

    gray_t, polar_t, quality_t = _fusion_tensors(sample, imgsz, cv2)
    mask_t = instance_mask_tensor(sample.instance_mask, imgsz, cv2)
    rgb, aux = variant_inputs(variant, gray_t, polar_t)
    return rgb, aux, mask_t, quality_t


class MaterialFusionClassifier:
    """Batch inference wrapper over one frozen v2_mask checkpoint.

    Loads the checkpoint through the strict material-fusion loader, refuses
    any contract conflict (protocol version, pooling mode, class order,
    preprocessing/mask versions, smoke markers, expected file sha256) at
    construction, and exposes a ``predict_batch`` interface over
    :class:`MaterialInstanceInput` records for the inference engine.

    Dual-stream variants require an explicit independent gray fallback
    (``fallback_weights``); the single-stream ``gray`` variant refuses one,
    because E1 is the pure control and must never silently swap to another
    gray model.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        *,
        expected_sha256: str = "",
        fallback_weights: str | Path = "",
        fallback: Any | None = None,
        require_formal: bool = True,
    ):
        self.device = torch_device_name(device)
        path = Path(checkpoint)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"material checkpoint not found: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(
                f"material checkpoint sha256 mismatch: {path} is {digest}, "
                f"expected {expected_sha256}"
            )
        self.checkpoint_path = path
        self.checkpoint_sha256 = digest

        self.metadata = read_material_checkpoint_metadata(path)
        meta = self.metadata
        if meta.version != MATERIAL_FUSION_MASK_VERSION:
            raise ValueError(
                f"the material_fusion runtime mode runs the v2_mask protocol, "
                f"got checkpoint version {meta.version!r}"
            )
        if require_formal and (meta.smoke or meta.limit_batches):
            raise ValueError(
                "a smoke checkpoint cannot serve the runtime mode: "
                f"smoke={meta.smoke}, limit_batches={meta.limit_batches}"
            )
        if tuple(meta.class_names) != CLASS_NAMES:
            raise ValueError(
                f"checkpoint class_names {meta.class_names} != the fixed project "
                f"order {CLASS_NAMES}"
            )

        self.model, self.metadata, _ = load_material_checkpoint(
            path, variant=meta.variant, imgsz=meta.imgsz
        )
        self.model.to(self.device).eval()
        self.variant = self.metadata.variant
        self.is_dual = self.model.is_dual
        self.uses_instance_mask = self.model.uses_instance_mask
        self.imgsz = int(self.metadata.imgsz)
        self.class_names = list(self.metadata.class_names)
        self.model_identity = (
            f"material_fusion_v2_mask variant={self.variant} "
            f"seed={self.metadata.seed} sha256={digest[:12]}"
        )

        self.fallback = None
        self.fallback_identity = ""
        if self.is_dual:
            if fallback is not None:
                # An injected fallback module (tests / preloaded wrappers);
                # it must behave like the gray adapter (callable on a
                # normalized gray batch) and is recorded as injected.
                self.fallback = fallback
                self.fallback_identity = "fallback:injected"
            elif fallback_weights:
                from scripts.eval_material_fusion import load_fallback_gray

                fallback_path = Path(fallback_weights)
                if not fallback_path.is_absolute():
                    fallback_path = PROJECT_ROOT / fallback_path
                self.fallback, self.fallback_identity = load_fallback_gray(
                    fallback_path, self.device, self.imgsz
                )
                self.fallback_identity = f"fallback:{self.fallback_identity}"
            else:
                raise ValueError(
                    f"the dual-stream variant {self.variant!r} routes "
                    "polar-invalid instances to the independent gray fallback; "
                    "an explicit fallback_weights path is required"
                )
        elif fallback_weights or fallback is not None:
            raise ValueError(
                "the single-stream gray variant is the pure control and never "
                "uses a fallback; refusing to attach one"
            )

    # ------------------------------------------------------------------
    # Batch prediction

    def predict_batch(
        self, samples: Sequence[MaterialInstanceInput | None]
    ) -> list[MaterialClassResult]:
        """Classify a batch of instance inputs, preserving order and count.

        Invalid structures (``None``, empty gray crop, empty or wrong-size
        mask) yield explicit failures and reach neither model. Polar-invalid
        rows of a dual variant reach only the independent gray fallback;
        valid rows only the CMX model. The output length is checked against
        the input length — a shorter result is an error, never a silent
        ``zip`` truncation.
        """
        outcomes = self._predict_rows(list(samples))
        if len(outcomes) != len(samples):
            raise ValueError(
                f"classification returned {len(outcomes)} results for "
                f"{len(samples)} inputs; refusing a silent zip truncation"
            )
        return outcomes

    def predict_pure_batch(
        self, samples: Sequence[MaterialInstanceInput | None]
    ) -> list[MaterialClassResult]:
        """Pure (no-routing) diagnostic over the same instances.

        Every structurally valid row is forwarded through the CMX model and
        the independent gray fallback is never used, so a pure score is not
        a production result — it is the research condition used to separate
        "the classifier's own behaviour" from "the routing policy". A row
        whose polarization arrays are not finite is **not evaluable** here
        (``valid=False``, ``invalid_reason='non_finite_polar_not_evaluable'``):
        it is never silently zeroed and then reported as a legal pure
        inference, and the caller must report it in the denominator.
        """
        outcomes = self._predict_rows(list(samples), routing=False)
        if len(outcomes) != len(samples):
            raise ValueError(
                f"pure classification returned {len(outcomes)} results for "
                f"{len(samples)} inputs; refusing a silent zip truncation"
            )
        return outcomes

    def _predict_rows(
        self,
        samples: list[MaterialInstanceInput | None],
        *,
        routing: bool = True,
    ) -> list[MaterialClassResult]:
        rows: list[MaterialClassResult] = [
            MaterialClassResult.failure(sample, "none") for sample in samples
        ]
        # Every decision about a row is made *before* anything is appended,
        # so the forward tensors, the routing flags and the original row
        # indices stay one-to-one. A filtered row can therefore never shift
        # another row's result.
        forward_indices: list[int] = []
        for index, sample in enumerate(samples):
            reason = _structure_problem(sample)
            if reason:
                rows[index] = MaterialClassResult.failure(sample, reason)
                continue
            assert sample is not None
            if not routing and _non_finite_polar(sample):
                # Pure diagnostic: a non-finite polarization input cannot be
                # scored at all. It is recorded as not evaluable, never
                # silently zeroed and passed off as a legal pure inference.
                rows[index] = MaterialClassResult.failure(
                    sample, "non_finite_polar_not_evaluable"
                )
                continue
            forward_indices.append(index)
        if not forward_indices:
            return rows

        rgbs: list[torch.Tensor] = []
        auxs: list[torch.Tensor | None] = []
        masks: list[torch.Tensor] = []
        invalid_flags: list[bool] = []
        fallback_reasons: list[str] = []
        for index in forward_indices:
            sample = samples[index]
            assert sample is not None
            rgb, aux, mask, _quality = material_input_tensors(
                sample, self.imgsz, self.variant
            )
            rgbs.append(rgb)
            auxs.append(aux)
            masks.append(mask)
            if _non_finite_polar(sample):
                invalid_flags.append(True)
                fallback_reasons.append(REASON_NON_FINITE_POLAR)
            elif bool(sample.polar_invalid):
                invalid_flags.append(True)
                fallback_reasons.append(
                    sample.invalid_reason or "polar_invalid"
                )
            elif float(sample.quality[VALID_RATIO_INDEX]) <= 0.0:
                invalid_flags.append(True)
                fallback_reasons.append(REASON_VALID_RATIO_ZERO)
            else:
                invalid_flags.append(False)
                fallback_reasons.append("")

        rgb = torch.stack(rgbs).to(self.device)
        aux = torch.stack(auxs).to(self.device) if auxs[0] is not None else None
        mask = torch.stack(masks).to(self.device)
        invalid = torch.tensor(invalid_flags, device=self.device)

        from scripts.eval_material_fusion import route_batch

        if not routing:
            # Pure condition: everything goes through the CMX model; the
            # fallback is not attached to this call at all.
            never = torch.zeros_like(invalid)
            _preds, logits, used_fallback = route_batch(
                self.model, None, rgb, aux if self.is_dual else None, never,
                mask,
            )
        elif self.is_dual:
            assert self.fallback is not None
            _preds, logits, used_fallback = route_batch(
                self.model, self.fallback, rgb, aux, invalid, mask
            )
        else:
            # The pure control: every row goes through the CMX model, no
            # routing and no fallback exists.
            never = torch.zeros_like(invalid)
            _preds, logits, used_fallback = route_batch(
                self.model, None, rgb, None, never, mask
            )

        probs = torch.softmax(logits, dim=-1)
        confidences, class_ids = probs.max(dim=-1)
        for pos, index in enumerate(forward_indices):
            sample = samples[index]
            assert sample is not None
            class_id = int(class_ids[pos])
            used = bool(used_fallback[pos])
            identity = (
                f"{self.fallback_identity}" if used else self.model_identity
            )
            rows[index] = MaterialClassResult(
                class_id=class_id,
                class_name=self.class_names[class_id],
                confidence=float(confidences[pos]),
                valid=True,
                used_fallback=used,
                fallback_reason=fallback_reasons[pos] if used else "",
                model_identity=identity,
                instance_id=sample.instance_id,
                frame_id=sample.frame_id,
                invalid_reason=sample.invalid_reason,
                raw_logits=tuple(
                    float(x) for x in logits[pos].cpu().tolist()
                ),
                polar_invalid=bool(sample.polar_invalid),
            )
        return rows


def _non_finite_polar(sample: MaterialInstanceInput) -> bool:
    """True when either polarization array holds NaN/Inf."""
    return not (
        bool(np.isfinite(sample.signed_q).all())
        and bool(np.isfinite(sample.abs_q).all())
    )


def _structure_problem(sample: MaterialInstanceInput | None) -> str:
    """The instance-structure problem, or "" when the row is well-formed."""
    if sample is None:
        return "none"
    if sample.gray.ndim != 2 or sample.gray.size == 0:
        return "invalid_gray_crop"
    mask = np.asarray(sample.instance_mask)
    if mask.shape != sample.gray.shape:
        return "mask_shape_mismatch"
    if mask.dtype != np.uint8:
        return "mask_dtype_not_uint8"
    if not bool(np.isin(mask, (0, 1)).all()):
        return "mask_not_binary"
    if not bool((mask > 0).any()):
        return "empty_instance_mask"
    for name, array in (("signed_q", sample.signed_q), ("abs_q", sample.abs_q),
                        ("polar_valid", sample.polar_valid)):
        if np.asarray(array).shape != sample.gray.shape:
            return f"{name}_shape_mismatch"
    return ""


# ---------------------------------------------------------------------------
# Independent experiment runtime configuration.


@dataclasses.dataclass(frozen=True)
class MaterialCandidate:
    """One frozen candidate checkpoint in the 12-run matrix."""

    variant: str
    seed: int
    checkpoint: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class MaterialRuntimeConfig:
    """Validated view of ``configs/experiments/material_runtime_v2_mask.yaml``.

    Training-data identity and runtime-data identity are separate fields and
    are never forced equal: the training fingerprint pins the V5 state the
    candidates were trained on, the runtime fingerprint is filled per
    verification run for whatever data actually flowed through the pipeline.
    """

    model_a_path: str
    model_a_sha256: str
    model_a_conf: float
    model_a_iou: float
    model_a_imgsz: int
    model_a_data_independence: str
    fallback_path: str
    fallback_sha256: str
    fallback_identity: str
    candidates: tuple[MaterialCandidate, ...]
    engineering_seed: int
    cmx_imgsz: int
    crop_padding: int
    preprocessing_version: str
    pooling_mode: str
    mask_schema: str
    mask_resize_version: str
    upstream_commit: str
    calibration_file: str
    r_convention: str
    baseline: float
    focal_length: float
    rectify_enabled: bool
    stereo: dict[str, Any]
    polar: dict[str, Any]
    device: str
    precision: str
    classification_batch: int
    fallback_strategy: str
    class_names: tuple[str, ...]
    training_data_root: str
    data_fingerprint_v5: str
    parent_data_fingerprint_v4: str
    runtime_data_root: str
    runtime_data_fingerprint: str


_RUNTIME_CONFIG_REQUIRED = (
    "model_a", "fallback", "candidates", "engineering_seed", "material_model",
    "calibration", "stereo", "polarization", "runtime", "data_identity",
)


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a config path against the project root, never the cwd."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_candidate(
    config: MaterialRuntimeConfig,
    *,
    variant: str,
    seed: int,
) -> MaterialCandidate:
    """The single candidate the caller explicitly asked for.

    ``variant`` and ``seed`` must both be given; there is no "latest",
    "best-scoring" or "first that loads" fallback. A missing candidate or an
    ambiguous (duplicated) variant/seed pair is refused instead of guessed.
    """
    matches = [
        candidate
        for candidate in config.candidates
        if candidate.variant == variant and candidate.seed == int(seed)
    ]
    if not matches:
        available = sorted({c.variant for c in config.candidates})
        raise ValueError(
            f"no candidate for variant {variant!r} seed {int(seed)} in the "
            f"runtime config; available variants: {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"duplicate candidate entries for variant {variant!r} seed "
            f"{int(seed)}; the selection is ambiguous and is refused"
        )
    return matches[0]


def material_checkpoint_consistency(
    config: MaterialRuntimeConfig,
    candidate: MaterialCandidate,
    metadata,
) -> dict:
    """Cross-check the runtime config against the selected checkpoint.

    The config declares how the model must be run; the checkpoint records
    what it was trained as. A disagreement is refused rather than resolved
    in favour of either side.
    """
    problems: list[str] = []
    if metadata.variant != candidate.variant:
        problems.append(
            f"checkpoint variant {metadata.variant!r} != requested "
            f"{candidate.variant!r}"
        )
    if int(metadata.seed) != int(candidate.seed):
        problems.append(
            f"checkpoint seed {metadata.seed} != requested {candidate.seed}"
        )
    if int(metadata.imgsz) != int(config.cmx_imgsz):
        problems.append(
            f"config material_model.imgsz {config.cmx_imgsz} != checkpoint imgsz "
            f"{metadata.imgsz}"
        )
    for field, config_value, meta_value in (
        ("preprocessing_version", config.preprocessing_version,
         metadata.preprocessing_version),
        ("pooling_mode", config.pooling_mode, metadata.pooling_mode),
        ("mask_schema", config.mask_schema, metadata.mask_schema),
        ("mask_resize_version", config.mask_resize_version,
         metadata.mask_resize_version),
        ("upstream_commit", config.upstream_commit, metadata.upstream_commit),
        ("data_fingerprint", config.data_fingerprint_v5, metadata.data_fingerprint),
        ("parent_data_fingerprint", config.parent_data_fingerprint_v4,
         metadata.parent_data_fingerprint),
    ):
        if str(config_value) != str(meta_value):
            problems.append(
                f"config {field} {config_value!r} != checkpoint {field} "
                f"{meta_value!r}"
            )
    if tuple(metadata.class_names) != tuple(config.class_names):
        problems.append(
            f"checkpoint class_names {tuple(metadata.class_names)} != config "
            f"class_names {tuple(config.class_names)}"
        )
    if problems:
        raise ValueError(
            "the runtime config does not describe the selected checkpoint: "
            + "; ".join(problems)
        )
    return {
        "variant": metadata.variant,
        "seed": int(metadata.seed),
        "imgsz": int(metadata.imgsz),
        "pooling_mode": metadata.pooling_mode,
        "mask_schema": metadata.mask_schema,
        "mask_resize_version": metadata.mask_resize_version,
        "data_fingerprint": metadata.data_fingerprint,
        "parent_data_fingerprint": metadata.parent_data_fingerprint,
        "class_names": list(metadata.class_names),
    }


def build_material_classifier_for_candidate(
    config: MaterialRuntimeConfig,
    candidate: MaterialCandidate,
    *,
    device: str,
) -> MaterialFusionClassifier:
    """Build the classifier for one selected candidate, identities verified.

    The checkpoint file must match the digest the config records, and the
    checkpoint's own metadata must describe the same run the config does.
    Dual-stream variants are bound to the config's independent gray
    fallback (digest-verified); the single-stream ``gray`` variant binds
    none.
    """
    checkpoint = resolve_project_path(candidate.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"candidate checkpoint not found: {checkpoint}"
        )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != candidate.sha256:
        raise ValueError(
            f"candidate {candidate.variant} s{candidate.seed} sha256 mismatch: "
            f"{checkpoint} is {digest}, the config records {candidate.sha256}"
        )
    metadata = read_material_checkpoint_metadata(checkpoint)
    material_checkpoint_consistency(config, candidate, metadata)

    is_dual = candidate.variant != "gray"
    fallback_path = ""
    if is_dual:
        fallback_file = resolve_project_path(config.fallback_path)
        if not fallback_file.is_file():
            raise FileNotFoundError(
                f"the independent gray fallback is missing: {fallback_file}"
            )
        fallback_digest = hashlib.sha256(fallback_file.read_bytes()).hexdigest()
        if fallback_digest != config.fallback_sha256:
            raise ValueError(
                f"fallback sha256 mismatch: {fallback_file} is {fallback_digest}, "
                f"the config records {config.fallback_sha256}"
            )
        fallback_path = fallback_file
    return MaterialFusionClassifier(
        checkpoint,
        device=device,
        expected_sha256=candidate.sha256,
        fallback_weights=fallback_path,
    )


def verify_material_runtime_fixed_values(config: MaterialRuntimeConfig) -> dict:
    """Verify the config's fixed values against what the engine actually does.

    Some runtime constants are not engine parameters (the polarization
    formula constants, the band full-image fallback threshold, the numeric
    precision). They must either agree with the engine's behaviour or be
    refused — silently ignoring a config value is not an option.
    """
    from core.polar_compute import POLAR_EPSILON
    from core.stereo_matching import StereoMatcher

    problems: list[str] = []
    if config.precision != "float32":
        problems.append(
            f"runtime.precision {config.precision!r} is not supported by the "
            "float32 inference path"
        )
    expected_polar = {
        "gain": 1.0,
        "epsilon": float(POLAR_EPSILON),
        "min_intensity_sum": 10.0,
    }
    for field, expected in expected_polar.items():
        declared = config.polar.get(field)
        if declared is None or float(declared) != expected:
            problems.append(
                f"polarization.{field} {declared!r} != the engine's "
                f"{expected!r}; the engine does not take this as a parameter"
            )
    engine_full_image_threshold = inspect.signature(
        StereoMatcher.compute_bands
    ).parameters["full_image_threshold"].default
    declared_threshold = config.stereo.get("full_image_threshold")
    if declared_threshold is None or float(declared_threshold) != float(
        engine_full_image_threshold
    ):
        problems.append(
            f"stereo.full_image_threshold {declared_threshold!r} != the "
            f"engine's {engine_full_image_threshold!r} (the engine calls "
            "compute_bands with its default)"
        )
    if int(config.classification_batch) <= 0:
        problems.append("runtime.classification_batch must be positive")
    if problems:
        raise ValueError(
            "the runtime config declares values the engine cannot honour: "
            + "; ".join(problems)
        )
    return {
        "precision": config.precision,
        "polar": expected_polar,
        "full_image_threshold": engine_full_image_threshold,
        "classification_batch": int(config.classification_batch),
    }


def load_material_runtime_config(path: str | Path) -> MaterialRuntimeConfig:
    """Load and validate the independent material runtime config."""
    import re
    import yaml

    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config {path} must be a mapping")
    missing = [field for field in _RUNTIME_CONFIG_REQUIRED if field not in raw]
    if missing:
        raise ValueError(f"runtime config is missing sections/fields: {missing}")

    sha_pattern = re.compile(r"[0-9a-f]{64}\Z")

    def require_sha(value: Any, field: str) -> str:
        value = str(value)
        if sha_pattern.match(value) is None:
            raise ValueError(
                f"{field} must be 64 lowercase hex chars, got {value!r}"
            )
        return value

    model_a = raw["model_a"]
    fallback = raw["fallback"]
    material = raw["material_model"]
    calibration = raw["calibration"]
    stereo = dict(raw["stereo"])
    polar = dict(raw["polarization"])
    runtime = raw["runtime"]
    data_identity = raw["data_identity"]

    candidates: list[MaterialCandidate] = []
    for entry in raw["candidates"]:
        candidates.append(
            MaterialCandidate(
                variant=str(entry["variant"]),
                seed=int(entry["seed"]),
                checkpoint=str(entry["checkpoint"]),
                sha256=require_sha(entry["sha256"], "candidate sha256"),
            )
        )
    if len(candidates) != 12:
        raise ValueError(
            f"expected the 12 frozen candidates, got {len(candidates)}"
        )

    for field in (
        "imgsz", "crop_padding", "preprocessing_version", "pooling_mode",
        "mask_schema", "mask_resize_version", "upstream_commit",
    ):
        if field not in material:
            raise ValueError(f"material_model is missing {field!r}")
    for field in ("file", "r_convention", "baseline", "focal_length", "rectification"):
        if field not in calibration:
            raise ValueError(f"calibration is missing {field!r}")
    for field in ("device", "precision", "classification_batch", "fallback_strategy", "class_names"):
        if field not in runtime:
            raise ValueError(f"runtime is missing {field!r}")
    for field in ("training_data_root", "training_data_fingerprint",
                  "training_parent_data_fingerprint", "runtime_data_root",
                  "runtime_data_fingerprint"):
        if field not in data_identity:
            raise ValueError(f"data_identity is missing {field!r}")

    return MaterialRuntimeConfig(
        model_a_path=str(model_a["path"]),
        model_a_sha256=require_sha(model_a["sha256"], "model_a.sha256"),
        model_a_conf=float(model_a.get("conf_threshold", 0.25)),
        model_a_iou=float(model_a.get("iou_threshold", 0.45)),
        model_a_imgsz=int(model_a.get("imgsz", 640)),
        model_a_data_independence=str(model_a.get("data_independence", "unknown")),
        fallback_path=str(fallback["path"]),
        fallback_sha256=require_sha(fallback["sha256"], "fallback.sha256"),
        fallback_identity=str(fallback.get("identity", "")),
        candidates=tuple(candidates),
        engineering_seed=int(raw["engineering_seed"]),
        cmx_imgsz=int(material["imgsz"]),
        crop_padding=int(material["crop_padding"]),
        preprocessing_version=str(material["preprocessing_version"]),
        pooling_mode=str(material["pooling_mode"]),
        mask_schema=str(material["mask_schema"]),
        mask_resize_version=str(material["mask_resize_version"]),
        upstream_commit=str(material["upstream_commit"]),
        calibration_file=str(calibration["file"]),
        r_convention=str(calibration["r_convention"]),
        baseline=float(calibration["baseline"]),
        focal_length=float(calibration["focal_length"]),
        rectify_enabled=bool(calibration["rectification"].get("enabled", False)),
        stereo=stereo,
        polar=polar,
        device=str(runtime["device"]),
        precision=str(runtime["precision"]),
        classification_batch=int(runtime["classification_batch"]),
        fallback_strategy=str(runtime["fallback_strategy"]),
        class_names=tuple(str(name) for name in runtime["class_names"]),
        training_data_root=str(data_identity["training_data_root"]),
        data_fingerprint_v5=require_sha(
            data_identity["training_data_fingerprint"], "training_data_fingerprint"
        ),
        parent_data_fingerprint_v4=require_sha(
            data_identity["training_parent_data_fingerprint"],
            "training_parent_data_fingerprint",
        ),
        runtime_data_root=str(data_identity["runtime_data_root"]),
        runtime_data_fingerprint=str(data_identity["runtime_data_fingerprint"]),
    )


__all__ = [
    "MASK_SOURCES",
    "MASK_SOURCE_GROUND_TRUTH",
    "MASK_SOURCE_NPZ",
    "MASK_SOURCE_PREDICTED",
    "MaterialCandidate",
    "MaterialClassResult",
    "MaterialFusionClassifier",
    "MaterialInstanceInput",
    "MaterialRuntimeConfig",
    "build_material_classifier_for_candidate",
    "instance_mask_tensor",
    "load_material_runtime_config",
    "material_checkpoint_consistency",
    "material_input_tensors",
    "resolve_project_path",
    "select_candidate",
    "verify_material_runtime_fixed_values",
]
