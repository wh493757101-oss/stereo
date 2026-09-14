"""
Dual-stage inference engine.

Stage 1: YOLO-A (standard 3ch grayscale copy) -> instance masks + bboxes
Stage 2: one dense SGBM computation per synchronized rectified frame ->
         robust per-instance disparity -> polar feature computed from the
         original rectified intensities, only inside each instance mask
Stage 3: YOLO-B classification -> material class. The crop channels
         depend on ``model_b_input_mode``: ``"polar"`` classifies
         [gray, polar, gray]; ``"gray"`` classifies [gray, gray, gray].

The engine is configuration-driven (see ``configs/default.yaml``) and
supports dependency injection of Model A, Model B, the stereo matcher and
the rectifier so it can be exercised without weights or a display.

Outputs instance segmentation + material class + depth.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import yaml

from core.polar_compute import build_polar_yolo_image, compute_polar_feature
from core.rectification import StereoRectifier
from core.stereo_matching import (
    StereoMatcher,
    StereoMatcherConfig,
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


@dataclasses.dataclass
class DetailedInferenceResult:
    """Full per-frame inference state for the GUI.

    ``left_gray``/``right_gray`` are the normalized (and, when enabled,
    rectified) images actually used for matching; overlays must be drawn on
    these, not on the raw camera frames. ``sync_skew_ms`` echoes the input:
    None means no skew measurement was available.
    """

    left_gray: np.ndarray
    right_gray: np.ndarray
    instances: list[Instance] = dataclasses.field(default_factory=list)
    depths: list[dict] = dataclasses.field(default_factory=list)
    polar_map: np.ndarray | None = None
    sync_skew_ms: float | None = None
    already_rectified: bool = False


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
        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        if self.max_sync_skew_ms < 0:
            raise ValueError("max_sync_skew_ms must be non-negative")

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

        The result carries the normalized (and, when enabled, rectified)
        left/right grayscale images actually used for matching, so the GUI
        can draw instances on the same geometry the depth was computed in.
        ``sync_skew_ms`` echoes the input, preserving the None-means-
        unavailable distinction.
        """
        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        if self.rectify_enabled and not already_rectified:
            left_gray, right_gray = self.rectifier.rectify(left_gray, right_gray)

        instances = self.model_a.predict(_gray_bgr_copy(left_gray), **_predict_kwargs(self.model_a_imgsz))
        if not instances:
            return DetailedInferenceResult(
                left_gray=left_gray,
                right_gray=right_gray,
                instances=[],
                depths=[],
                polar_map=None,
                sync_skew_ms=sync_skew_ms,
                already_rectified=bool(already_rectified),
            )

        polar_map = np.zeros(left_gray.shape, dtype=np.float32)

        measured = None if sync_skew_ms is None else abs(float(sync_skew_ms))
        if measured is not None and measured > self.max_sync_skew_ms:
            depth_results = [
                _invalid_depth(inst.id, "sync_skew_exceeded") for inst in instances
            ]
            if self.model_b is not None:
                instances = self._classify(instances, left_gray, polar_map)
            return DetailedInferenceResult(
                left_gray=left_gray,
                right_gray=right_gray,
                instances=instances,
                depths=depth_results,
                polar_map=polar_map,
                sync_skew_ms=sync_skew_ms,
                already_rectified=bool(already_rectified),
            )

        # One dense computation per frame; every instance reuses the result.
        stereo_result = self.matcher.compute(left_gray, right_gray)

        depth_results = []
        object_disparity = np.zeros(left_gray.shape, dtype=np.float32)
        polar_mask = np.zeros(left_gray.shape, dtype=bool)
        for inst in instances:
            stats = self.matcher.instance_stats(
                stereo_result, inst.mask, instance_id=inst.id
            )
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
                # Same robust object disparity drives both the reported depth
                # and the polar warp. Build one piecewise-constant disparity
                # image so the full-frame remap runs once regardless of how
                # many instances were detected. Later instances retain the
                # previous overlap behavior by overwriting earlier values.
                mask_bool = inst.mask > 0
                object_disparity[mask_bool] = stats.disparity
                polar_mask |= mask_bool

        if np.any(polar_mask):
            polar_map = compute_polar_feature(
                left_gray,
                right_gray,
                object_disparity,
                mask=polar_mask,
            )

        if self.model_b is not None:
            instances = self._classify(instances, left_gray, polar_map)

        return DetailedInferenceResult(
            left_gray=left_gray,
            right_gray=right_gray,
            instances=instances,
            depths=depth_results,
            polar_map=polar_map,
            sync_skew_ms=sync_skew_ms,
            already_rectified=bool(already_rectified),
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
        modes.
        """
        updated = []
        height, width = left_gray.shape
        for inst in instances:
            x1, y1, x2, y2 = inst.bbox
            x1p = max(0, x1 - self.crop_padding)
            y1p = max(0, y1 - self.crop_padding)
            x2p = min(width, x2 + self.crop_padding)
            y2p = min(height, y2 + self.crop_padding)

            crop_gray = left_gray[y1p:y2p, x1p:x2p]
            crop_polar = polar_map[y1p:y2p, x1p:x2p]
            if crop_gray.size == 0:
                updated.append(inst)
                continue

            if self.model_b_input_mode == "gray":
                crop = _gray_bgr_copy(crop_gray)
            else:
                crop = build_polar_yolo_image(crop_gray, crop_polar)
            result = self.model_b.predict(crop)
            if result.valid:
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
