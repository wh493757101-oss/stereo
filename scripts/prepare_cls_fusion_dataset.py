"""Build the V3 polar fusion classification dataset (npz samples + manifest).

Consumes the leakage-free four-class v2 segmentation dataset (pair manifest
plus YOLO-seg labels) and emits numeric-only ``.npz`` samples under
``datasets/underwater_cls_fusion_v3``:

    <output>/<split>/<class_name>/<output_stem>_obj<index>.npz

Each sample stores gray / signed_q / abs_q / valid / quality / class_id
(see :mod:`core.fusion_dataset`); all string provenance goes to
``dataset_manifest.csv``. Per stereo pair exactly one dense SGBM pass runs;
the polarization differential uses the reliable per-pixel disparity inside
each object mask (no constant object-level fill, no out-of-bounds
saturation). Invalid stereo never drops a sample: arrays are written zeroed
and the manifest records the failure reason.

After generation the script audits every file (decode, dtype/shape, no
NaN/Inf, group/split isolation, no saturated invalid bands) and writes
``analysis/data/fusion_v3_audit.json`` plus preview PNGs under
``analysis/data/fusion_v3_preview/``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from core.fusion_dataset import (
    NPZ_FIELDS,
    QUALITY_VECTOR_KEYS,
    build_quality_vector,
    file_digest,
    load_fusion_sample,
    quality_vector_from_result,
    save_fusion_sample,
)
from core.polar_compute import compute_polar_features
from core.stereo_matching import (
    StereoMatcher,
    StereoMatcherConfig,
    build_horizontal_bands,
    to_gray_u8,
)
from scripts.make_polar_dataset import make_crop_window, read_yolo_polygons

SPLIT_NAMES = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = {"train": 2078, "val": 368, "test": 362}
EXPECTED_TOTAL = 2808
PREVIEW_PER_CLASS = 2
MATCHING_MODES = ("full", "bands")


@dataclass(frozen=True)
class FusionSampleRecord:
    sample_name: str
    split: str
    group_name: str
    class_id: int
    class_name: str
    source_frame: str
    object_index: int
    stereo_valid: bool
    stereo_reason: str
    stereo_valid_ratio: float
    disparity: float
    polar_valid_ratio: float
    quality: tuple[float, ...]
    npz_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V3 polar fusion classification dataset.",
    )
    parser.add_argument(
        "--source",
        default="datasets/underwater_seg_v2",
        help="Source four-class YOLO-seg dataset root (pair manifest + labels).",
    )
    parser.add_argument(
        "--output",
        default="datasets/underwater_cls_fusion_v3",
        help="Output root for the fusion npz dataset.",
    )
    parser.add_argument(
        "--audit-root",
        default="analysis/data",
        help="Directory receiving fusion_v3_audit.json and fusion_v3_preview/.",
    )
    parser.add_argument(
        "--crop-pad",
        type=int,
        default=10,
        help="Padding in pixels around each instance polygon.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace existing non-empty output directories.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Write samples without running the post-generation audit.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Skip generation; audit the existing output from its manifest.",
    )
    parser.add_argument(
        "--audit-stem",
        default="fusion_v3",
        help="Stem for the audit report/preview names under --audit-root "
        "(writes <stem>_audit.json and <stem>_preview/).",
    )
    parser.add_argument(
        "--matching",
        choices=MATCHING_MODES,
        default="full",
        help="full: one dense pass per pair (V3). bands: band-restricted "
        "matching identical to online inference (V4).",
    )
    parser.add_argument(
        "--band-margin",
        type=int,
        default=20,
        help="Vertical margin around instance bboxes when building bands "
        "(--matching bands, mirrors stereo.band_margin in configs/default.yaml).",
    )
    parser.add_argument(
        "--full-image-threshold",
        type=float,
        default=0.9,
        help="Band coverage ratio above which a full-image pass is used "
        "(--matching bands, mirrors the inference engine).",
    )
    parser.add_argument(
        "--sgbm-mode",
        choices=["sgbm", "hh", "3way"],
        default="3way",
        help="OpenCV StereoSGBM dynamic-programming mode",
    )
    parser.add_argument(
        "--max-disp",
        type=int,
        default=768,
        help="Full-resolution maximum disparity in pixels",
    )
    parser.add_argument(
        "--block-size",
        "--window",
        dest="block_size",
        type=int,
        default=7,
        help="SGBM block size in full-resolution pixels (odd)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Matching scale relative to full resolution",
    )
    parser.add_argument(
        "--lr-check-threshold",
        type=float,
        default=2.0,
        help="Left-right consistency threshold in full-resolution pixels",
    )
    parser.add_argument(
        "--uniqueness-ratio",
        type=int,
        default=10,
        help="SGBM uniqueness ratio (percent)",
    )
    parser.add_argument(
        "--speckle-window",
        type=int,
        default=100,
        help="SGBM speckle window size",
    )
    parser.add_argument(
        "--speckle-range",
        type=int,
        default=32,
        help="SGBM speckle range",
    )
    parser.add_argument(
        "--texture-threshold",
        type=float,
        default=10.0,
        help="Minimum local horizontal-gradient texture for valid disparity",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.05,
        help="Minimum global valid-pixel ratio; below it the disparity map is discarded",
    )
    args = parser.parse_args()
    if args.crop_pad < 0:
        parser.error("--crop-pad must be non-negative")
    if args.band_margin < 0:
        parser.error("--band-margin must be non-negative")
    return args


def build_matcher(args: argparse.Namespace) -> StereoMatcher:
    return StereoMatcher(
        StereoMatcherConfig(
            matcher="sgbm",
            mode=args.sgbm_mode,
            max_disparity=args.max_disp,
            scale=args.scale,
            block_size=args.block_size,
            uniqueness_ratio=args.uniqueness_ratio,
            speckle_window=args.speckle_window,
            speckle_range=args.speckle_range,
            texture_threshold=args.texture_threshold,
            lr_check_threshold_px=args.lr_check_threshold,
            min_valid_ratio=args.min_valid_ratio,
        )
    )


def _resolve_manifest_path(value: str, workspace: Path) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else workspace / path


def read_source_manifest(source_root: Path, workspace: Path) -> list:
    """Same manifest contract as prepare_cls_paired_datasets."""
    from scripts.prepare_cls_paired_datasets import ManifestRow

    manifest_path = source_root / "pair_manifest.csv"
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"output_stem", "split", "group_name", "left_path", "right_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(
                f"{manifest_path} is missing required columns: {', '.join(missing)}"
            )
        for line_index, row in enumerate(reader, start=2):
            output_stem = str(row["output_stem"]).strip()
            split = str(row["split"]).strip()
            if not output_stem or split not in SPLIT_NAMES:
                raise ValueError(
                    f"invalid output_stem/split at {manifest_path}:{line_index}"
                )
            if output_stem in seen:
                raise ValueError(
                    f"duplicate output_stem {output_stem!r} in {manifest_path}"
                )
            seen.add(output_stem)
            rows.append(
                ManifestRow(
                    output_stem=output_stem,
                    split=split,
                    group_name=str(row["group_name"]).strip(),
                    left_path=_resolve_manifest_path(row["left_path"], workspace),
                    right_path=_resolve_manifest_path(row["right_path"], workspace),
                )
            )
    if not rows:
        raise ValueError(f"pair manifest is empty: {manifest_path}")
    return rows


def read_class_names(source_root: Path) -> list[str]:
    with (source_root / "data.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"invalid class names in {source_root / 'data.yaml'}")
    return [str(name) for name in names]


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return to_gray_u8(image)


def build_object_mask(polygon, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def prepare_output_root(output_root: Path, clean: bool) -> None:
    resolved = output_root.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not clean:
            raise RuntimeError(
                f"output directory is not empty: {resolved}; pass --clean to replace it"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def build_fusion_dataset(
    source_root: Path,
    output_root: Path,
    matcher: StereoMatcher,
    crop_pad: int = 10,
    clean: bool = False,
    workspace: Path | None = None,
    matching: str = "full",
    band_margin: int = 20,
    full_image_threshold: float = 0.9,
) -> list[FusionSampleRecord]:
    """Generate all fusion npz samples; returns one record per sample.

    ``matching="full"`` runs one dense SGBM pass per pair (V3).
    ``matching="bands"`` builds horizontal bands from all ground-truth
    bboxes of the frame exactly like online inference
    (:func:`core.stereo_matching.build_horizontal_bands` ->
    :meth:`StereoMatcher.compute_bands`) so training and inference see the
    same disparity distribution (V4).
    """
    if matching not in MATCHING_MODES:
        raise ValueError(f"unknown matching mode: {matching!r}")
    workspace = workspace if workspace is not None else Path.cwd()
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("output root must be outside the source root")

    class_names = read_class_names(source_root)
    rows = read_source_manifest(source_root, workspace)
    prepare_output_root(output_root, clean)
    for split in SPLIT_NAMES:
        for class_name in class_names:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)

    records: list[FusionSampleRecord] = []
    for row in rows:
        label_path = source_root / "labels" / row.split / f"{row.output_stem}.txt"
        if not label_path.is_file():
            continue
        left_gray = load_gray(row.left_path)
        height, width = left_gray.shape[:2]
        polygons = read_yolo_polygons(label_path, width, height)
        if not polygons:
            continue
        right_gray = load_gray(row.right_path)
        if matching == "bands":
            bboxes = []
            for annotation in polygons:
                xs = [x for x, _ in annotation.polygon]
                ys = [y for _, y in annotation.polygon]
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            bands = build_horizontal_bands(bboxes, height, vertical_margin=band_margin)
            # One band-restricted pass per pair; empty bands are skipped
            # inside compute_bands without running SGBM.
            result = matcher.compute_bands(
                left_gray, right_gray, bands, full_image_threshold=full_image_threshold
            )
        else:
            # Exactly one dense matching pass per stereo pair; every object
            # below reuses this single per-pixel disparity map.
            result = matcher.compute(left_gray, right_gray)

        for object_index, annotation in enumerate(polygons):
            if not 0 <= annotation.class_id < len(class_names):
                raise ValueError(
                    f"unknown class id {annotation.class_id} in {label_path}; "
                    f"supported ids: 0..{len(class_names) - 1}"
                )
            class_name = class_names[annotation.class_id]
            mask = build_object_mask(annotation.polygon, width, height)
            stats = matcher.instance_stats(result, mask, object_index)

            if stats.valid:
                polar_result = compute_polar_features(
                    left_gray,
                    right_gray,
                    result.disparity,
                    object_mask=mask,
                    disparity_valid=result.valid,
                )
                quality, components = quality_vector_from_result(polar_result, mask)
            else:
                # Without a valid correspondence the differential is not a
                # physical polarization measurement: write zeros and record
                # the failure reason in the manifest.
                polar_result = None
                components = {key: 0.0 for key in QUALITY_VECTOR_KEYS}
                quality = build_quality_vector(**components)

            window = make_crop_window(annotation.polygon, width, height, crop_pad)
            crop_gray = left_gray[window.y1 : window.y2, window.x1 : window.x2]
            if polar_result is not None:
                crop_signed = polar_result.signed_q[
                    window.y1 : window.y2, window.x1 : window.x2
                ]
                crop_abs = polar_result.abs_q[
                    window.y1 : window.y2, window.x1 : window.x2
                ]
                crop_valid = polar_result.valid_mask[
                    window.y1 : window.y2, window.x1 : window.x2
                ].astype(np.uint8)
            else:
                crop_signed = np.zeros(crop_gray.shape, dtype=np.float32)
                crop_abs = np.zeros(crop_gray.shape, dtype=np.float32)
                crop_valid = np.zeros(crop_gray.shape, dtype=np.uint8)

            # valid_mask is mask-restricted (object_mask above) and the crop
            # window covers the whole polygon bbox, so every valid pixel
            # must lie inside the crop. If a crop configuration cuts them
            # the recorded mean_abs_q can no longer be verified from the
            # crop; refuse explicitly instead of writing unverifiable data.
            if polar_result is not None and int(crop_valid.sum()) != int(
                polar_result.valid_mask.sum()
            ):
                raise ValueError(
                    f"crop window cuts valid polar pixels for "
                    f"{row.output_stem}_obj{object_index:03d}: "
                    f"{int(crop_valid.sum())} of "
                    f"{int(polar_result.valid_mask.sum())} valid pixels are "
                    "inside the window; the mean_abs_q audit cannot verify "
                    "such samples (increase --crop-pad)"
                )

            sample_name = f"{row.output_stem}_obj{object_index:03d}"
            relative = Path(row.split) / class_name / f"{sample_name}.npz"
            save_fusion_sample(
                output_root / relative,
                gray=crop_gray,
                signed_q=crop_signed,
                abs_q=crop_abs,
                valid=crop_valid,
                quality=quality,
                class_id=annotation.class_id,
            )
            records.append(
                FusionSampleRecord(
                    sample_name=sample_name,
                    split=row.split,
                    group_name=row.group_name,
                    class_id=annotation.class_id,
                    class_name=class_name,
                    source_frame=row.output_stem,
                    object_index=object_index,
                    stereo_valid=stats.valid,
                    stereo_reason=stats.reason,
                    stereo_valid_ratio=round(stats.valid_ratio, 6),
                    disparity=round(stats.disparity, 4),
                    polar_valid_ratio=round(components["valid_ratio"], 6),
                    quality=tuple(float(v) for v in quality),
                    npz_path=relative.as_posix(),
                )
            )

    write_manifest(
        output_root,
        class_names,
        records,
        generation={
            "matching": matching,
            "band_margin": band_margin if matching == "bands" else None,
            "full_image_threshold": full_image_threshold if matching == "bands" else None,
            "crop_pad": crop_pad,
        },
    )
    return records


MANIFEST_FIELDS = (
    "sample_name",
    "split",
    "group_name",
    "class_id",
    "class_name",
    "source_frame",
    "object_index",
    "stereo_valid",
    "stereo_reason",
    "stereo_valid_ratio",
    "disparity",
    "polar_valid_ratio",
    *(f"quality_{key}" for key in QUALITY_VECTOR_KEYS),
    "npz_path",
)


def record_to_row(record: FusionSampleRecord) -> dict[str, str]:
    row = {
        "sample_name": record.sample_name,
        "split": record.split,
        "group_name": record.group_name,
        "class_id": str(record.class_id),
        "class_name": record.class_name,
        "source_frame": record.source_frame,
        "object_index": str(record.object_index),
        "stereo_valid": "true" if record.stereo_valid else "false",
        "stereo_reason": record.stereo_reason,
        "stereo_valid_ratio": f"{record.stereo_valid_ratio:.6f}",
        "disparity": f"{record.disparity:.4f}",
        "polar_valid_ratio": f"{record.polar_valid_ratio:.6f}",
        "npz_path": record.npz_path,
    }
    for key, value in zip(QUALITY_VECTOR_KEYS, record.quality):
        row[f"quality_{key}"] = f"{value:.6f}"
    return row


def write_manifest(
    output_root: Path,
    class_names: list[str],
    records: list[FusionSampleRecord],
    generation: dict | None = None,
) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_FIELDS))
    writer.writeheader()
    for record in records:
        writer.writerow(record_to_row(record))
    (output_root / "dataset_manifest.csv").write_text(
        buffer.getvalue(), encoding="utf-8"
    )
    summary = {
        "class_names": class_names,
        "npz_fields": list(NPZ_FIELDS),
        "quality_vector_keys": list(QUALITY_VECTOR_KEYS),
        "splits": {
            split: sum(1 for record in records if record.split == split)
            for split in SPLIT_NAMES
        },
        "total_samples": len(records),
        "generation": generation or {},
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _has_saturated_left_band(
    sample, region: float = 0.25, fraction: float = 0.9, min_column_valid: float = 0.1
) -> bool:
    """Detect the out-of-bounds saturation signature in one sample.

    The legacy bug produced polar=1.0 for every crop column left of the
    object disparity (right-view sample outside the image). Signature: a
    column in the left ``region`` of the crop where at least ``fraction``
    of the valid pixels are saturated (abs_q >= 0.999). Columns must hold
    at least ``max(4, min_column_valid * height)`` valid pixels: sparse
    occlusion-boundary columns can be 100% saturated with only a handful
    of valid pixels (physically real extreme ratios), while the bug
    saturated whole well-populated columns.
    """
    valid = sample.valid > 0
    saturated = valid & (sample.abs_q >= 0.999)
    height, width = sample.valid.shape
    edge_columns = max(1, int(width * region))
    density_threshold = max(4, int(height * min_column_valid))
    region_valid = valid[:, :edge_columns]
    region_saturated = saturated[:, :edge_columns]
    column_valid = region_valid.sum(axis=0)
    column_saturated = region_saturated.sum(axis=0)
    dense = column_valid >= density_threshold
    bad_columns = dense & (column_saturated >= fraction * column_valid)
    return bool(bad_columns.any())


def records_from_manifest(output_root: Path) -> list[FusionSampleRecord]:
    """Rebuild sample records from a written dataset_manifest.csv."""
    output_root = Path(output_root)
    with (output_root / "dataset_manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    records = []
    for row in rows:
        quality = tuple(
            float(row[f"quality_{key}"]) for key in QUALITY_VECTOR_KEYS
        )
        records.append(
            FusionSampleRecord(
                sample_name=row["sample_name"],
                split=row["split"],
                group_name=row["group_name"],
                class_id=int(row["class_id"]),
                class_name=row["class_name"],
                source_frame=row["source_frame"],
                object_index=int(row["object_index"]),
                stereo_valid=row["stereo_valid"] == "true",
                stereo_reason=row["stereo_reason"],
                stereo_valid_ratio=float(row["stereo_valid_ratio"]),
                disparity=float(row["disparity"]),
                polar_valid_ratio=float(row["polar_valid_ratio"]),
                quality=quality,
                npz_path=row["npz_path"],
            )
        )
    return records


def _check_sample_values(sample) -> str | None:
    """Numeric invariants of one decoded sample; None when all hold."""
    if not bool(np.isin(sample.valid, (0, 1)).all()):
        return "valid mask values are not strictly 0/1"
    if sample.signed_q.min() < -1.0 or sample.signed_q.max() > 1.0:
        return "signed_q outside [-1, 1]"
    if sample.abs_q.min() < 0.0 or sample.abs_q.max() > 1.0:
        return "abs_q outside [0, 1]"
    valid = sample.valid > 0
    if valid.any() and not np.allclose(
        sample.abs_q[valid], np.abs(sample.signed_q[valid]), atol=1e-5
    ):
        return "abs_q != |signed_q| on valid pixels"
    return None


def _check_manifest_consistency(
    output_root: Path,
    records: list[FusionSampleRecord],
    class_names: list[str],
) -> dict[str, list[str]]:
    """Manifest-level integrity checks (ids, duplicates, files, layout)."""
    failures: dict[str, list[str]] = {
        "class_order_mismatches": [],
        "duplicate_manifest_rows": [],
        "unknown_splits": [],
        "path_layout_mismatches": [],
        "missing_files": [],
        "quality_consistency_failures": [],
    }
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for record in records:
        label = f"{record.sample_name}(id={record.class_id},name={record.class_name})"
        if not 0 <= record.class_id < len(class_names):
            failures["class_order_mismatches"].append(label)
        elif class_names[record.class_id] != record.class_name:
            failures["class_order_mismatches"].append(label)
        if record.sample_name in seen_names or record.npz_path in seen_paths:
            failures["duplicate_manifest_rows"].append(record.sample_name)
        seen_names.add(record.sample_name)
        seen_paths.add(record.npz_path)
        if record.split not in SPLIT_NAMES:
            failures["unknown_splits"].append(record.npz_path)
        expected_layout = f"{record.split}/{record.class_name}/{record.sample_name}.npz"
        if record.npz_path != expected_layout:
            failures["path_layout_mismatches"].append(record.npz_path)
        if not (output_root / record.npz_path).is_file():
            failures["missing_files"].append(record.npz_path)
        quality = record.quality
        # Tolerance 1.5e-6: the manifest stores 6-decimal text while the
        # npz quality vector is float32; a value ending in 5 at the 7th
        # digit can round the two representations apart by one unit.
        if (
            len(quality) != len(QUALITY_VECTOR_KEYS)
            or any(not 0.0 <= v <= 1.0 for v in quality)
            or abs(quality[0] - record.polar_valid_ratio) > 1.5e-6
        ):
            failures["quality_consistency_failures"].append(record.npz_path)
    return {key: value[:50] for key, value in failures.items()}


def _check_npz_quality(sample, record: FusionSampleRecord) -> str | None:
    """Cross-check the npz quality vector (the training gate input) against
    the manifest record and the sample arrays themselves.

    Checks, in order:

    1. npz quality == manifest quality (6-decimal text tolerance).
    2. All four entries lie within [0, 1].
    3. Zero valid pixels in the crop: valid_ratio and mean_abs_q must both
       be exactly 0 (valid pixels are mask-restricted and the crop window
       covers the whole mask, so an empty crop-valid set means an empty
       mask-valid set). in_bounds_ratio and brightness_valid_ratio are
       independent per-pixel measurements and are validated by their own
       definitions ([0, 1] range), NOT forced to zero: an all-black pair
       is fully in bounds with zero brightness-valid pixels, i.e.
       quality = [0, 1, 0, 0] is legitimate.
    4. Valid pixels present: valid_ratio must be positive, and the
       containment relations implied by the valid-pixel definition must
       hold on the stored float32 values (exact comparison, equality is
       legal): valid_ratio <= in_bounds_ratio and
       valid_ratio <= brightness_valid_ratio, because a valid pixel is
       necessarily in bounds and bright enough to measure. The instance
       mask is not stored in the npz, so the three ratios cannot be
       recomputed exactly from the crop; these relations are the strongest
       ratio check available without the mask.
    5. Valid pixels present: mean_abs_q is recomputed from the crop with
       rtol=0. Offline valid is mask-restricted and the crop window always
       contains the full instance mask, so the crop's valid pixels are
       exactly the pixels that entered the recorded mask-level mean;
       1e-6 covers float32 accumulation differences, 1.5e-6 additionally
       covers the manifest's 6-decimal text rounding.
    """
    if not np.allclose(
        sample.quality,
        np.asarray(record.quality, dtype=np.float32),
        atol=1.5e-6,
    ):
        return "npz quality != manifest quality"

    quality = np.asarray(sample.quality, dtype=np.float64)
    if np.any(quality < 0.0) or np.any(quality > 1.0):
        return f"quality entries outside [0, 1]: {sample.quality.tolist()}"
    valid_ratio, in_bounds_ratio, brightness_valid_ratio, mean_abs_q = quality

    valid = sample.valid > 0
    if not valid.any():
        if valid_ratio > 0.0:
            return f"zero valid pixels but valid_ratio {valid_ratio:.6f} != 0"
        if mean_abs_q > 0.0:
            return f"zero valid pixels but mean_abs_q {mean_abs_q:.6f} != 0"
        return None

    # Containment relations on the stored float32 values (exact comparison;
    # equality is legal). A valid pixel is by definition in bounds and
    # bright enough, so valid_ratio can never exceed either bound, and a
    # crop holding valid pixels must record a positive valid_ratio.
    if valid_ratio <= 0.0:
        return (
            f"valid_ratio {valid_ratio:.6f} must be positive when the crop "
            f"has {int(valid.sum())} valid pixels"
        )
    if valid_ratio > in_bounds_ratio:
        return (
            f"valid_ratio {valid_ratio:.6f} exceeds in_bounds_ratio "
            f"{in_bounds_ratio:.6f}"
        )
    if valid_ratio > brightness_valid_ratio:
        return (
            f"valid_ratio {valid_ratio:.6f} exceeds brightness_valid_ratio "
            f"{brightness_valid_ratio:.6f}"
        )

    recomputed = float(sample.abs_q[valid].mean())
    if not np.isclose(recomputed, mean_abs_q, rtol=0.0, atol=1e-6):
        return (
            f"mean_abs_q {mean_abs_q:.6f} != recomputed crop mean "
            f"{recomputed:.6f} over {int(valid.sum())} valid pixels"
        )
    if not np.isclose(recomputed, float(record.quality[3]), rtol=0.0, atol=1.5e-6):
        return (
            f"manifest mean_abs_q {float(record.quality[3]):.6f} != recomputed "
            f"crop mean {recomputed:.6f}"
        )
    return None


def audit_fusion_dataset(
    output_root: Path,
    records: list[FusionSampleRecord],
    class_names: list[str],
    audit_root: Path,
    expected_splits: dict[str, int] | None = EXPECTED_SPLIT_COUNTS,
    audit_stem: str = "fusion_v3",
) -> dict:
    """Decode-check every sample and verify dataset invariants.

    Returns the audit dict; also writes ``<stem>_audit.json`` under
    ``audit_root`` and a copy as ``dataset_audit.json`` inside the dataset
    root (the file the training gate requires), plus preview PNGs.
    ``expected_splits`` is the required per-split sample count (None skips
    the count gate).
    """
    output_root = Path(output_root)
    audit_root = Path(audit_root)
    manifest_path = output_root / "dataset_manifest.csv"
    summary_path = output_root / "dataset_summary.json"
    audit: dict = {
        "output_root": output_root.resolve().as_posix(),
        "npz_fields": list(NPZ_FIELDS),
        "quality_vector_keys": list(QUALITY_VECTOR_KEYS),
        "class_names": class_names,
        "expected_splits": expected_splits,
        # Bind the verdict to the exact manifest/summary bytes so training
        # can refuse a dataset whose manifest rows or generation parameters
        # changed after this audit ran.
        "manifest_sha256": file_digest(manifest_path) if manifest_path.is_file() else "",
        "summary_sha256": file_digest(summary_path) if summary_path.is_file() else "",
    }

    total = len(records)
    split_counts = {
        split: sum(1 for r in records if r.split == split) for split in SPLIT_NAMES
    }
    audit["counts"] = {"total": total, **split_counts}
    audit["count_match_expected"] = expected_splits is None or all(
        split_counts[split] == expected for split, expected in expected_splits.items()
    )

    per_class: dict[str, dict[str, int]] = {
        name: {split: 0 for split in SPLIT_NAMES} for name in class_names
    }
    group_to_splits: dict[str, set[str]] = {}
    for record in records:
        per_class[record.class_name][record.split] += 1
        group_to_splits.setdefault(record.group_name, set()).add(record.split)
    audit["per_class"] = per_class
    leakage = {
        group: sorted(splits)
        for group, splits in group_to_splits.items()
        if len(splits) > 1
    }
    audit["group_split_leakage"] = leakage

    manifest_failures = _check_manifest_consistency(output_root, records, class_names)
    audit.update(manifest_failures)

    # Any npz on disk that the manifest does not reference.
    referenced = {r.npz_path for r in records}
    audit["orphan_npz_files"] = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*.npz")
        if path.relative_to(output_root).as_posix() not in referenced
    )[:50]

    decode_failures: list[str] = []
    invalid_nonzero_failures: list[str] = []
    saturated_band_failures: list[str] = []
    value_range_failures: list[str] = []
    npz_quality_failures: list[str] = []
    polar_valid_ratios: list[float] = []
    mean_abs_q: list[float] = []
    saturated_valid_pixels = 0
    previews_written = 0
    file_digests: dict[str, str] = {}

    preview_dir = audit_root / f"{audit_stem}_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_budget = {name: PREVIEW_PER_CLASS for name in class_names}

    for record in records:
        path = output_root / record.npz_path
        # Content digest at audit time: the training gate re-verifies these
        # so later tampering/loss of samples is caught even though the
        # historic audit verdict is unchanged.
        file_digests[record.npz_path] = (
            file_digest(path) if path.is_file() else ""
        )
        try:
            sample = load_fusion_sample(path)
        except Exception as exc:  # decode/dtype/shape/NaN validation
            decode_failures.append(f"{record.npz_path}: {exc}")
            continue

        if int(sample.class_id) != record.class_id:
            decode_failures.append(
                f"{record.npz_path}: class_id {sample.class_id} != {record.class_id}"
            )
            continue

        failure = _check_sample_values(sample)
        if failure is not None:
            value_range_failures.append(f"{record.npz_path}: {failure}")
            continue

        quality_failure = _check_npz_quality(sample, record)
        if quality_failure is not None:
            npz_quality_failures.append(f"{record.npz_path}: {quality_failure}")
            continue

        valid = sample.valid > 0
        polar_valid_ratios.append(float(valid.mean()))
        if valid.any():
            mean_abs_q.append(float(sample.abs_q[valid].mean()))

        # Invalid pixels must carry a zero differential: any out-of-bounds
        # right-view sample that saturated towards polar=1 would show up here.
        if np.any(sample.abs_q[~valid] != 0.0) or np.any(
            sample.signed_q[~valid] != 0.0
        ):
            invalid_nonzero_failures.append(record.npz_path)
        # Isolated saturated valid pixels are physically real extreme ratios
        # (bright left view, near-black right view). The out-of-bounds bug
        # signature is a full column band of saturated valid pixels at the
        # left edge of the crop; only that pattern is a failure.
        saturated_valid_pixels += int((valid & (sample.abs_q >= 0.999)).sum())
        if _has_saturated_left_band(sample):
            saturated_band_failures.append(record.npz_path)

        if preview_budget.get(record.class_name, 0) > 0:
            preview_budget[record.class_name] -= 1
            _write_preview(preview_dir, record, sample)
            previews_written += 1

    audit["decoded_samples"] = (
        total - len(decode_failures) - len(value_range_failures) - len(npz_quality_failures)
    )
    audit["decode_failures"] = decode_failures[:50]
    audit["invalid_pixel_nonzero_failures"] = invalid_nonzero_failures[:50]
    audit["saturated_left_band_failures"] = saturated_band_failures[:50]
    audit["value_range_failures"] = value_range_failures[:50]
    audit["npz_quality_failures"] = npz_quality_failures[:50]
    audit["saturated_valid_pixels"] = saturated_valid_pixels
    audit["file_digests"] = file_digests
    audit["polar_valid_ratio"] = {
        "mean": float(np.mean(polar_valid_ratios)) if polar_valid_ratios else 0.0,
        "min": float(np.min(polar_valid_ratios)) if polar_valid_ratios else 0.0,
        "max": float(np.max(polar_valid_ratios)) if polar_valid_ratios else 0.0,
    }
    audit["mean_abs_q_valid_pixels"] = {
        "mean": float(np.mean(mean_abs_q)) if mean_abs_q else 0.0,
    }
    audit["previews_written"] = previews_written
    audit["stereo_valid_samples"] = sum(1 for r in records if r.stereo_valid)
    audit["audit_passed"] = (
        not decode_failures
        and not invalid_nonzero_failures
        and not saturated_band_failures
        and not value_range_failures
        and not npz_quality_failures
        and not any(manifest_failures.values())
        and not audit["orphan_npz_files"]
        and not leakage
        and audit["count_match_expected"]
        # A formal dataset (expected_splits given) must keep the manifest
        # and generation summary it was audited with; the training gate
        # refuses reports without their digests.
        and (
            expected_splits is None
            or (bool(audit["manifest_sha256"]) and bool(audit["summary_sha256"]))
        )
    )

    audit_path = audit_root / f"{audit_stem}_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    audit_path.write_text(audit_text, encoding="utf-8")
    # The training gate reads the audit from inside the dataset root, so a
    # dataset always travels with its own verification verdict.
    (output_root / "dataset_audit.json").write_text(audit_text, encoding="utf-8")
    return audit


def _write_preview(preview_dir: Path, record: FusionSampleRecord, sample) -> None:
    h, w = sample.gray.shape
    panel = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    panel[:, :w] = cv2.cvtColor(sample.gray, cv2.COLOR_GRAY2BGR)
    abs_u8 = np.rint(np.clip(sample.abs_q, 0.0, 1.0) * 255).astype(np.uint8)
    panel[:, w + 10 : 2 * w + 10] = cv2.applyColorMap(abs_u8, cv2.COLORMAP_VIRIDIS)
    valid_u8 = (sample.valid > 0) * np.uint8(255)
    panel[:, 2 * w + 20 :] = cv2.cvtColor(valid_u8, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(
        str(preview_dir / f"{record.sample_name}_preview.png"),
        cv2.resize(panel, (min(1200, panel.shape[1]), h), interpolation=cv2.INTER_AREA)
        if panel.shape[1] > 1200
        else panel,
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    if args.audit_only:
        records = records_from_manifest(output_root)
        # Class names must follow the manifest's class-id order (data.yaml
        # order), not alphabetical, or the class-order audit check fails.
        by_id: dict[int, str] = {}
        for record in records:
            by_id.setdefault(record.class_id, record.class_name)
        audit = audit_fusion_dataset(
            output_root=output_root,
            records=records,
            class_names=[by_id[key] for key in sorted(by_id)],
            audit_root=Path(args.audit_root),
            audit_stem=args.audit_stem,
        )
        print(f"Samples: {len(records)}")
        print(f"Audit passed: {audit['audit_passed']}")
        print(f"Audit report: {(Path(args.audit_root) / f'{args.audit_stem}_audit.json').resolve()}")
        if not audit["audit_passed"]:
            sys.exit(1)
        return

    matcher = build_matcher(args)
    records = build_fusion_dataset(
        source_root=Path(args.source),
        output_root=output_root,
        matcher=matcher,
        crop_pad=args.crop_pad,
        clean=args.clean,
        matching=args.matching,
        band_margin=args.band_margin,
        full_image_threshold=args.full_image_threshold,
    )
    print(f"Fusion dataset: {output_root.resolve()}")
    print(f"Samples: {len(records)}")
    if not args.skip_audit:
        audit = audit_fusion_dataset(
            output_root=output_root,
            records=records,
            class_names=read_class_names(Path(args.source)),
            audit_root=Path(args.audit_root),
            audit_stem=args.audit_stem,
        )
        print(f"Audit passed: {audit['audit_passed']}")
        print(f"Audit report: {(Path(args.audit_root) / f'{args.audit_stem}_audit.json').resolve()}")
        if not audit["audit_passed"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
