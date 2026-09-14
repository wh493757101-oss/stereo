"""
Dual-stage inference engine.

Stage 1: YOLO-A (standard 3ch grayscale copy) -> instance masks + bboxes
Stage 2: one band-restricted dense SGBM computation per synchronized
         rectified frame (bands merged from the Model A bboxes) ->
         robust per-instance median disparity for depth, plus reliable
         per-pixel disparity for the polarization differential computed
         from the original rectified intensities
Stage 3: YOLO-B classification (batched when supported) -> material class.
         The crop channels depend on ``model_b_input_mode``: ``"polar"``
         classifies [gray, polar, gray]; ``"gray"`` classifies
         [gray, gray, gray].

The engine is configuration-driven (see ``configs/default.yaml``) and
supports dependency injection of Model A, Model B, the stereo matcher and
the rectifier so it can be exercised without weights or a display.

Outputs instance segmentation + material class + depth, per-stage timings
and per-instance polarization quality.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import yaml

from core.polar_compute import (
    PolarFeatureResult,
    build_polar_yolo_image,
    compute_polar_features,
)
from core.rectification import StereoRectifier
from core.stereo_matching import (
    StereoMatcher,
    StereoMatcherConfig,
    build_horizontal_bands,
    disparity_to_depth,
    to_gray_u8,
)
from models.classification import ClassificationModel, ClassificationResult
from models.segmentation import Instance, SegmentationModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

DEFAULT_BASELINE_M = 0.09890970798524992
DEFAULT_FOCAL_PX = 3643.5231322766995

MODEL_B_INPUT_MODES = ("gray", "polar")

STAGE_TIMING_KEYS = (
    "model_a_s",
    "stereo_s",
    "depth_s",
    "polar_s",
    "model_b_s",
    "total_s",
)


@dataclasses.dataclass
class DetailedInferenceResult:
    """Full per-frame inference state for the GUI.

    ``left_gray``/``right_gray`` are the normalized (and, when enabled,
    rectified) images actually used for matching; overlays must be drawn on
    these, not on the raw camera frames. ``sync_skew_ms`` echoes the input:
    None means no skew measurement was available.

    ``timings`` holds per-stage wall-clock seconds (keys in
    ``STAGE_TIMING_KEYS``; absent stages are simply not present). They are
    measured diagnostics, not a throughput claim.

    ``polar_quality`` carries one record per detected instance with the
    in-mask polarization validity ratios; ``valid_ratio`` is the fraction of
    instance pixels with a trustworthy polarization measurement (the "polar
    gate": 0 whenever sync was exceeded, matching failed or too few pixels
    were valid).
    """

    left_gray: np.ndarray
    right_gray: np.ndarray
    instances: list[Instance] = dataclasses.field(default_factory=list)
    depths: list[dict] = dataclasses.field(default_factory=list)
    polar_map: np.ndarray | None = None
    sync_skew_ms: float | None = None
    already_rectified: bool = False
    timings: dict[str, float] = dataclasses.field(default_factory=dict)
    polar_quality: list[dict] = dataclasses.field(default_factory=list)


def resolve_device(requested: str = "auto") -> str:
    """Resolve an inference device using ``torch.cuda.is_available()``.

    ``"auto"`` (or empty) selects device ``"0"`` when torch CUDA is usable,
    otherwise ``"cpu"``. ``"cpu"`` is always valid. Numeric IDs (``"0"``,
    ``"1"``), ``"cuda"``/``"gpu"`` and ``"cuda:N"`` are explicit CUDA
    requests and fail with a clear error when torch reports no CUDA.
    """
    requested = str(requested).strip().lower()
    if requested in {"", "auto"}:
        return "0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    is_cuda_id = (
        requested in {"cuda", "gpu"}
        or requested.startswith("cuda:")
        or requested.isdigit()
    )
    if is_cuda_id:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA inference requested (device={requested!r}) but "
                "torch.cuda.is_available() is False; use device='auto' or 'cpu'."
            )
        if requested in {"cuda", "gpu"}:
            return "0"
        return requested
    return requested


class SegmentationModelProtocol(Protocol):
    """Structural type of the injected Model A."""

    def predict(self, image: np.ndarray, **kwargs: Any) -> list[Instance]: ...


class ClassificationModelProtocol(Protocol):
    """Structural type of the injected Model B."""

    def predict(self, image: np.ndarray) -> ClassificationResult: ...


class DualStageInferenceEngine:
    """Dual-stage inference: grayscale segmentation then polar-aided classification.

    Models, matcher and rectifier may be injected directly (``model_a``,
    ``model_b``, ``matcher``, ``rectifier``) for testing; otherwise they are
    constructed from the checkpoint paths. Use :meth:`from_config` to build
    the engine from ``configs/default.yaml``.
    """

    def __init__(
        self,
        model_a_path: str | Path | None = None,
        model_b_path: str | Path | None = None,
        baseline: float = DEFAULT_BASELINE_M,
        focal_length: float = DEFAULT_FOCAL_PX,
        device: str = "auto",
        conf_threshold: float = 0.25,
        *,
        model_a: SegmentationModelProtocol | None = None,
        model_b: ClassificationModelProtocol | None = None,
        matcher: StereoMatcher | None = None,
        rectifier: StereoRectifier | None = None,
        rectify_enabled: bool = False,
        stereo_config: StereoMatcherConfig | None = None,
        crop_padding: int = 10,
        max_sync_skew_ms: float = 2.0,
        band_vertical_margin: int = 20,
        model_b_imgsz: int | None = None,
        model_b_input_mode: str = "polar",
        model_a_iou_threshold: float = 0.45,
        model_a_imgsz: int | None = None,
        # Legacy NCC-era parameters; accepted for caller compatibility.
        max_disp: int | None = None,
        window_size: int | None = None,
        min_ncc: float | None = None,
    ):
        del min_ncc  # SGBM has no per-window NCC threshold.
        if model_a is None and model_a_path is None:
            raise ValueError("either model_a or model_a_path is required")

        input_mode = str(model_b_input_mode)
        if input_mode not in MODEL_B_INPUT_MODES:
            raise ValueError(
                f"model_b_input_mode must be one of {MODEL_B_INPUT_MODES}, "
                f"got {model_b_input_mode!r}"
            )
        self.model_b_input_mode = input_mode

        self.device = resolve_device(device)
        self.baseline = float(baseline)
        self.focal_length = float(focal_length)
        self.crop_padding = int(crop_padding)
        self.max_sync_skew_ms = float(max_sync_skew_ms)
        self.band_vertical_margin = int(band_vertical_margin)
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        if self.max_sync_skew_ms < 0:
            raise ValueError("max_sync_skew_ms must be non-negative")
        if self.band_vertical_margin < 0:
            raise ValueError("band_vertical_margin must be non-negative")

        stereo_cfg = stereo_config or StereoMatcherConfig()
        if max_disp is not None or window_size is not None:
            stereo_cfg = dataclasses.replace(
                stereo_cfg,
                max_disparity=int(max_disp) if max_disp is not None else stereo_cfg.max_disparity,
                block_size=int(window_size) if window_size is not None else stereo_cfg.block_size,
            )
        self.matcher = matcher if matcher is not None else StereoMatcher(stereo_cfg)

        self.rectifier = rectifier
        self.rectify_enabled = bool(rectify_enabled)
        if self.rectify_enabled and self.rectifier is None:
            raise ValueError("rectify_enabled=True requires a rectifier")

        self.model_a = model_a if model_a is not None else SegmentationModel(
            model_path=model_a_path,
            in_channels=3,
            device=self.device,
            conf_threshold=conf_threshold,
            iou_threshold=float(model_a_iou_threshold),
        )
        self.model_a_imgsz = int(model_a_imgsz) if model_a_imgsz is not None else None

        if model_b is not None:
            self.model_b = model_b
        elif model_b_path is not None:
            self.model_b = ClassificationModel(
                model_path=model_b_path,
                device=self.device,
                imgsz=model_b_imgsz,
            )
        else:
            self.model_b = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        **overrides: Any,
    ) -> "DualStageInferenceEngine":
        """Build the engine from a YAML config (default ``configs/default.yaml``).

        Relative paths in the config (model checkpoints, calibration file)
        are resolved against the project root deterministically. Keyword
        ``overrides`` replace any constructor argument.
        """
        path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        def resolve(rel: str | Path) -> Path:
            candidate = Path(rel)
            return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

        model_a_cfg = cfg.get("model_a") or {}
        model_b_cfg = cfg.get("model_b") or {}
        calibration_cfg = cfg.get("calibration") or {}
        rectification_cfg = cfg.get("rectification") or {}
        stereo_cfg = cfg.get("stereo") or {}
        runtime_cfg = cfg.get("runtime") or {}

        stereo_config = StereoMatcherConfig(
            matcher=str(stereo_cfg.get("matcher", "sgbm")),
            mode=str(stereo_cfg.get("mode", "3way")),
            max_disparity=int(stereo_cfg.get("max_disp", 768)),
            scale=float(stereo_cfg.get("scale", 0.25)),
            block_size=int(stereo_cfg.get("block_size", 7)),
            uniqueness_ratio=int(stereo_cfg.get("uniqueness_ratio", 10)),
            speckle_window=int(stereo_cfg.get("speckle_window", 100)),
            speckle_range=int(stereo_cfg.get("speckle_range", 32)),
            texture_threshold=float(stereo_cfg.get("texture_threshold", 10.0)),
            lr_check_threshold_px=float(stereo_cfg.get("lr_check_threshold_px", 2.0)),
            min_valid_ratio=float(stereo_cfg.get("min_valid_ratio", 0.05)),
        )

        rectify_enabled = bool(rectification_cfg.get("enabled", False))
        rectifier = None
        if rectify_enabled:
            rectifier = StereoRectifier(
                resolve(calibration_cfg["file"]),
                r_convention=str(calibration_cfg.get("r_convention", "matlab")),
                alpha=float(rectification_cfg.get("alpha", 0.0)),
            )

        kwargs: dict[str, Any] = dict(
            model_a_path=resolve(model_a_cfg["path"]),
            model_b_path=resolve(model_b_cfg["path"]),
            baseline=float(calibration_cfg.get("baseline", DEFAULT_BASELINE_M)),
            focal_length=float(calibration_cfg.get("focal_length", DEFAULT_FOCAL_PX)),
            device=runtime_cfg.get("device", "auto"),
            conf_threshold=float(model_a_cfg.get("conf_threshold", 0.25)),
            model_a_iou_threshold=float(model_a_cfg.get("iou_threshold", 0.45)),
            model_a_imgsz=model_a_cfg.get("imgsz"),
            rectifier=rectifier,
            rectify_enabled=rectify_enabled,
            stereo_config=stereo_config,
            crop_padding=int(runtime_cfg.get("crop_padding", 10)),
            max_sync_skew_ms=float(runtime_cfg.get("sync_skew_ms", 2.0)),
            band_vertical_margin=int(stereo_cfg.get("band_margin", 20)),
            model_b_imgsz=model_b_cfg.get("imgsz"),
            model_b_input_mode=str(model_b_cfg.get("input_mode", "polar")),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sync_skew_ms: float | None = None,
        already_rectified: bool = False,
    ) -> tuple[list[Instance], list[dict], np.ndarray | None]:
        """Process one stereo frame through the dual-stage pipeline.

        ``left``/``right`` may be ``(H, W)``, ``(H, W, 1)``, BGR or BGRA;
        they are normalized through the shared ``to_gray_u8`` boundary and,
        when enabled, rectified online before inference.

        ``sync_skew_ms`` is the measured per-call left/right timestamp skew;
        ``None`` means no measurement is available (the pair is processed,
        but the caller must not report sync as verified). When a measured
        skew exceeds ``max_sync_skew_ms`` Model A (and optional gray /
        zero-polar Model B) still run, but stereo matching is skipped:
        every depth result is invalid with reason ``"sync_skew_exceeded"``
        and the polar map contains no pseudo-correspondence signal.

        ``already_rectified`` marks pre-rectified input (e.g. playback of
        rectified datasets); online rectification is skipped so the pair is
        not remapped twice.

        Returns:
            (instances, depth_results, polar_map)
        """
        result = self.process_frame_detailed(
            left,
            right,
            sync_skew_ms=sync_skew_ms,
            already_rectified=already_rectified,
        )
        return result.instances, result.depths, result.polar_map

    def process_frame_detailed(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sync_skew_ms: float | None = None,
        already_rectified: bool = False,
    ) -> "DetailedInferenceResult":
        """Like :meth:`process_frame` but returns the full intermediate state.

        Pipeline: Model A -> merged horizontal bands -> band-restricted dense
        matching -> robust per-instance median disparity (depth) -> reliable
        per-pixel disparity (polar) -> Model B batch classification.

        The result carries the normalized (and, when enabled, rectified)
        left/right grayscale images actually used for matching, so the GUI
        can draw instances on the same geometry the depth was computed in.
        ``sync_skew_ms`` echoes the input, preserving the None-means-
        unavailable distinction.
        """
        frame_started = time.perf_counter()
        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        if self.rectify_enabled and not already_rectified:
            left_gray, right_gray = self.rectifier.rectify(left_gray, right_gray)

        timings: dict[str, float] = {}

        started = time.perf_counter()
        instances = self.model_a.predict(
            _gray_bgr_copy(left_gray), **_predict_kwargs(self.model_a_imgsz)
        )
        timings["model_a_s"] = time.perf_counter() - started
        if not instances:
            timings["total_s"] = time.perf_counter() - frame_started
            return DetailedInferenceResult(
                left_gray=left_gray,
                right_gray=right_gray,
                instances=[],
                depths=[],
                polar_map=None,
                sync_skew_ms=sync_skew_ms,
                already_rectified=bool(already_rectified),
                timings=timings,
            )

        polar_map = np.zeros(left_gray.shape, dtype=np.float32)

        measured = None if sync_skew_ms is None else abs(float(sync_skew_ms))
        if measured is not None and measured > self.max_sync_skew_ms:
            depth_results = [
                _invalid_depth(inst.id, "sync_skew_exceeded") for inst in instances
            ]
            polar_quality = [
                _polar_quality_record(inst.id, reason="sync_skew_exceeded")
                for inst in instances
            ]
            if self.model_b is not None:
                started = time.perf_counter()
                instances = self._classify(instances, left_gray, polar_map)
                timings["model_b_s"] = time.perf_counter() - started
            timings["total_s"] = time.perf_counter() - frame_started
            return DetailedInferenceResult(
                left_gray=left_gray,
                right_gray=right_gray,
                instances=instances,
                depths=depth_results,
                polar_map=polar_map,
                sync_skew_ms=sync_skew_ms,
                already_rectified=bool(already_rectified),
                timings=timings,
                polar_quality=polar_quality,
            )

        # One band-restricted dense computation per frame; every instance
        # reuses the result. Bands come from the Model A bboxes.
        started = time.perf_counter()
        bands = build_horizontal_bands(
            (inst.bbox for inst in instances),
            image_height=left_gray.shape[0],
            vertical_margin=self.band_vertical_margin,
        )
        if hasattr(self.matcher, "compute_bands"):
            stereo_result = self.matcher.compute_bands(left_gray, right_gray, bands)
        else:
            stereo_result = self.matcher.compute(left_gray, right_gray)
        timings["stereo_s"] = time.perf_counter() - started

        started = time.perf_counter()
        depth_results = []
        polar_mask = np.zeros(left_gray.shape, dtype=bool)
        stats_by_id: dict[int, Any] = {}
        for inst in instances:
            stats = self.matcher.instance_stats(
                stereo_result, inst.mask, instance_id=inst.id
            )
            stats_by_id[inst.id] = stats
            # Only a fully valid stats record may produce a depth value or a
            # nonzero polar contribution (low_valid_ratio is invalid, matching
            # the generated training data).
            depth_m: float | None = None
            if stats.valid:
                depth_m = disparity_to_depth(
                    stats.disparity, self.baseline, self.focal_length
                )
                if depth_m <= 0:
                    depth_m = None
            valid = stats.valid and depth_m is not None
            depth_results.append(
                {
                    "instance_id": inst.id,
                    "disparity": round(stats.disparity, 2),
                    "valid_ratio": round(stats.valid_ratio, 3),
                    "confidence": round(stats.valid_ratio, 3),
                    "depth": round(depth_m, 3) if depth_m is not None else None,
                    "valid": valid,
                    "reason": stats.reason,
                }
            )

            if stats.valid:
                polar_mask |= inst.mask > 0
        timings["depth_s"] = time.perf_counter() - started

        # Polar uses the reliable per-pixel disparity, not a constant
        # object-level fill: pixels whose disparity failed the matcher's
        # left-right consistency (or fall outside the image) get no
        # polarization value instead of a pseudo-correspondence.
        started = time.perf_counter()
        polar_result: PolarFeatureResult | None = None
        if np.any(polar_mask):
            polar_result = compute_polar_features(
                left_gray,
                right_gray,
                stereo_result.disparity,
                object_mask=polar_mask,
                disparity_valid=stereo_result.valid,
            )
            polar_map = polar_result.abs_q
        polar_quality = [
            _instance_polar_quality(inst, polar_result, stats_by_id[inst.id].reason)
            for inst in instances
        ]
        timings["polar_s"] = time.perf_counter() - started

        if self.model_b is not None:
            started = time.perf_counter()
            instances = self._classify(instances, left_gray, polar_map)
            timings["model_b_s"] = time.perf_counter() - started

        timings["total_s"] = time.perf_counter() - frame_started
        return DetailedInferenceResult(
            left_gray=left_gray,
            right_gray=right_gray,
            instances=instances,
            depths=depth_results,
            polar_map=polar_map,
            sync_skew_ms=sync_skew_ms,
            already_rectified=bool(already_rectified),
            timings=timings,
            polar_quality=polar_quality,
        )

    def _classify(
        self,
        instances: list[Instance],
        left_gray: np.ndarray,
        polar_map: np.ndarray,
    ) -> list[Instance]:
        """Run Model B crops for material classification.

        ``polar`` mode classifies [gray, polar, gray]; ``gray`` mode
        classifies [gray, gray, gray]. Crop geometry is identical in both
        modes. When the injected Model B exposes ``predict_batch`` all crops
        are classified in one call; otherwise the legacy per-crop interface
        is used.
        """
        height, width = left_gray.shape
        crops: list[np.ndarray | None] = []
        for inst in instances:
            x1, y1, x2, y2 = inst.bbox
            x1p = max(0, x1 - self.crop_padding)
            y1p = max(0, y1 - self.crop_padding)
            x2p = min(width, x2 + self.crop_padding)
            y2p = min(height, y2 + self.crop_padding)

            crop_gray = left_gray[y1p:y2p, x1p:x2p]
            if crop_gray.size == 0:
                crops.append(None)
                continue
            crop_polar = polar_map[y1p:y2p, x1p:x2p]
            if self.model_b_input_mode == "gray":
                crops.append(_gray_bgr_copy(crop_gray))
            else:
                crops.append(build_polar_yolo_image(crop_gray, crop_polar))

        if not any(crop is not None for crop in crops):
            return list(instances)

        batchable = [crop for crop in crops if crop is not None]
        if hasattr(self.model_b, "predict_batch"):
            batch_results = self.model_b.predict_batch(batchable)
        else:
            batch_results = [self.model_b.predict(crop) for crop in batchable]

        results = iter(batch_results)
        updated = []
        for inst, crop in zip(instances, crops):
            result = next(results) if crop is not None else None
            if result is not None and result.valid:
                inst.class_id = int(result.top1_id)
                inst.class_name = result.top1_name or f"class_{inst.class_id}"
                inst.confidence = round(float(result.top1_conf), 3)
            updated.append(inst)
        return updated


def _gray_bgr_copy(gray: np.ndarray) -> np.ndarray:
    """3-channel copy of a grayscale image: Model A's standard input."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _predict_kwargs(imgsz: int | None) -> dict[str, Any]:
    """Forward Model A's imgsz only when explicitly configured."""
    return {"imgsz": imgsz} if imgsz is not None else {}


def _invalid_depth(instance_id: int, reason: str) -> dict:
    return {
        "instance_id": instance_id,
        "disparity": 0.0,
        "valid_ratio": 0.0,
        "confidence": 0.0,
        "depth": None,
        "valid": False,
        "reason": reason,
    }


def _polar_quality_record(
    instance_id: int,
    reason: str = "ok",
    valid_ratio: float = 0.0,
    in_bounds_ratio: float = 0.0,
    brightness_valid_ratio: float = 0.0,
) -> dict:
    return {
        "instance_id": instance_id,
        "valid_ratio": round(float(valid_ratio), 4),
        "in_bounds_ratio": round(float(in_bounds_ratio), 4),
        "brightness_valid_ratio": round(float(brightness_valid_ratio), 4),
        "reason": reason,
    }


def _instance_polar_quality(
    inst: Instance,
    polar_result: PolarFeatureResult | None,
    reason: str,
) -> dict:
    """Per-instance polarization quality from the frame-level polar result.

    Instances whose stereo stats were invalid (or when no polar computation
    ran at all) get an all-zero record with the failure reason; the polar
    gate is 0 for them.
    """
    mask_bool = np.asarray(inst.mask) > 0
    total = int(mask_bool.sum())
    if polar_result is None or total == 0:
        return _polar_quality_record(inst.id, reason=reason)

    valid = polar_result.valid_mask & mask_bool
    return _polar_quality_record(
        inst.id,
        reason=reason,
        valid_ratio=valid.sum() / total,
        in_bounds_ratio=(polar_result.in_bounds_mask & mask_bool).sum() / total,
        brightness_valid_ratio=(
            polar_result.brightness_valid_mask & mask_bool
        ).sum() / total,
    )
