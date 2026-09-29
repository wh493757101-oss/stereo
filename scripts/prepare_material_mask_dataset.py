"""Build the V5 instance-mask extension of the V4 fusion dataset.

Read-only correspondence check and generation in one place, because the
mask is only trustworthy if its provenance chain is checkable:

    sample_name
      -> split / group_name / source_frame          (V4 manifest)
      -> source image (left_path) + YOLO label file (pair manifest)
      -> object_index                                (label line order)
      -> polygon                                     (normalised -> pixels)
      -> full-image rasterised mask                  (same function as V4)
      -> the identical V4 crop window                (crop_pad, clipped)
      -> cropped instance mask
      -> per-scale pooling weights                   (area resampling)

The six V4 arrays are copied from the V4 npz unchanged; this script never
re-runs stereo matching, never re-estimates q or the gain and never changes
the split, the class ids or the object indices. ``valid`` keeps exactly its
V4 meaning: it was already restricted to the target interior by the V4
generation, so V5 does not restore measurement validity outside the target.

Usage:
    # read-only correspondence check over the whole dataset
    python scripts/prepare_material_mask_dataset.py --check-only

    # build the V5 dataset (refuses an existing output directory)
    python scripts/prepare_material_mask_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fusion_dataset import (
    QUALITY_VECTOR_KEYS,
    dataset_fingerprint,
    file_digest,
    load_fusion_sample,
)
from core.material_mask_dataset import (
    CLASS_NAMES,
    MASK_NPZ_FIELDS,
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    V4_PARENT_FINGERPRINT,
    label_source_digest,
    mask_content_digest,
    resolve_source_path,
    sample_mapping_digest,
    save_mask_sample,
)
from scripts.make_polar_dataset import make_crop_window, read_yolo_polygons
from scripts.prepare_cls_fusion_dataset import build_object_mask, load_gray

SPLIT_NAMES = ("train", "val", "test")

DEFAULT_SOURCE = "datasets/underwater_seg_v2"
DEFAULT_V4 = "datasets/underwater_cls_fusion_v4_band"
DEFAULT_OUTPUT = "datasets/underwater_cls_fusion_v5_instance_mask"
DEFAULT_CROP_PAD = 10

# V4 manifest columns copied through unchanged, so the V5 manifest carries
# the same provenance a V4 consumer expects.
V4_COPIED_FIELDS = (
    "stereo_valid",
    "stereo_reason",
    "stereo_valid_ratio",
    "disparity",
    "polar_valid_ratio",
    *(f"quality_{key}" for key in QUALITY_VECTOR_KEYS),
)

MANIFEST_FIELDS = (
    "sample_name",
    "split",
    "group_name",
    "class_id",
    "class_name",
    "source_frame",
    "object_index",
    *V4_COPIED_FIELDS,
    "npz_path",
    "left_path",
    "label_path",
    "image_width",
    "image_height",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "mask_pixels",
    "crop_area",
    "mask_area_ratio",
)


@dataclass(frozen=True)
class MaskSampleRecord:
    """One V5 sample: the copied V4 provenance plus the mask provenance."""

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
    left_path: str
    label_path: str
    image_width: int
    image_height: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    mask_pixels: int
    crop_area: int
    mask_area_ratio: float

    def row(self) -> dict[str, str]:
        data = {
            "sample_name": self.sample_name,
            "split": self.split,
            "group_name": self.group_name,
            "class_id": str(self.class_id),
            "class_name": self.class_name,
            "source_frame": self.source_frame,
            "object_index": str(self.object_index),
            "stereo_valid": "true" if self.stereo_valid else "false",
            "stereo_reason": self.stereo_reason,
            "stereo_valid_ratio": f"{self.stereo_valid_ratio:.6f}",
            "disparity": f"{self.disparity:.4f}",
            "polar_valid_ratio": f"{self.polar_valid_ratio:.6f}",
            "npz_path": self.npz_path,
            "left_path": self.left_path,
            "label_path": self.label_path,
            "image_width": str(self.image_width),
            "image_height": str(self.image_height),
            "crop_x1": str(self.crop_x1),
            "crop_y1": str(self.crop_y1),
            "crop_x2": str(self.crop_x2),
            "crop_y2": str(self.crop_y2),
            "mask_pixels": str(self.mask_pixels),
            "crop_area": str(self.crop_area),
            "mask_area_ratio": f"{self.mask_area_ratio:.6f}",
        }
        for key, value in zip(QUALITY_VECTOR_KEYS, self.quality):
            data[f"quality_{key}"] = f"{value:.6f}"
        return data

    def mapping_entry(self) -> tuple[str, str]:
        return (
            self.sample_name,
            "|".join(
                str(part)
                for part in (
                    self.source_frame,
                    self.object_index,
                    self.crop_x1,
                    self.crop_y1,
                    self.crop_x2,
                    self.crop_y2,
                    self.image_width,
                    self.image_height,
                )
            ),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Source four-class YOLO-seg dataset (pair manifest + labels).")
    parser.add_argument("--v4", default=DEFAULT_V4,
                        help="V4 fusion dataset the masks are derived from.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="V5 output root; must not exist.")
    parser.add_argument("--crop-pad", type=int, default=DEFAULT_CROP_PAD,
                        help="Padding of the V4 crop window (must match V4 generation).")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the read-only provenance check and write nothing.")
    parser.add_argument("--report", default="",
                        help="Optional path for the provenance report JSON.")
    args = parser.parse_args(argv)
    if args.crop_pad < 0:
        parser.error("--crop-pad must be non-negative")
    return args


def _project_relative(path: Path) -> str:
    """Record a source path relative to the project root.

    Admission re-verifies the annotation sources by resolving the recorded
    paths against the project root, so the provenance must be
    project-anchored; a source outside the project root cannot be re-checked
    and is refused at generation time instead of producing unverifiable data.
    """
    resolved = Path(path).resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"source {resolved} is outside the project root {root}; V5 provenance "
            "must be project-anchored so it can be re-verified at admission"
        )
    return resolved.relative_to(root).as_posix()


def _resolve(value: str | Path, workspace: Path) -> Path:
    """Absolute path for a project-relative or absolute value."""
    path = Path(str(value).strip())
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def read_pair_manifest(source_root: Path, workspace: Path) -> dict[str, dict]:
    """``output_stem -> {split, group_name, left_path}`` from the source."""
    manifest_path = Path(source_root) / "pair_manifest.csv"
    rows: dict[str, dict] = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"output_stem", "split", "group_name", "left_path", "right_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(
                f"{manifest_path} is missing required columns: {', '.join(missing)}"
            )
        for line_index, row in enumerate(reader, start=2):
            stem = str(row["output_stem"]).strip()
            if not stem:
                raise ValueError(f"empty output_stem at {manifest_path}:{line_index}")
            if stem in rows:
                raise ValueError(f"duplicate output_stem {stem!r} in {manifest_path}")
            rows[stem] = {
                "split": str(row["split"]).strip(),
                "group_name": str(row["group_name"]).strip(),
                "left_path": _resolve(row["left_path"], workspace),
            }
    if not rows:
        raise ValueError(f"pair manifest is empty: {manifest_path}")
    return rows


def read_v4_manifest(v4_root: Path) -> list[dict]:
    with (Path(v4_root) / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty V4 manifest: {v4_root / 'dataset_manifest.csv'}")
    return rows


def read_class_names(source_root: Path) -> list[str]:
    import yaml

    with (Path(source_root) / "data.yaml").open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if not names:
        raise ValueError(f"invalid class names in {source_root / 'data.yaml'}")
    return [str(name) for name in names]


class FrameResolutionError(RuntimeError):
    """A source frame could not be resolved to polygons unambiguously."""


def resolve_frame(
    stem: str,
    rows: list[dict],
    pair: dict,
    class_names: list[str],
    source_root: Path,
    v4_root: Path,
    crop_pad: int,
    workspace: Path,
) -> tuple[list[MaskSampleRecord], list[np.ndarray], list[str]]:
    """Resolve every V4 object of one source frame to its cropped mask.

    Read-only with respect to both datasets. Returns the records, the masks
    (in the same order) and the list of failure reasons (empty when the
    frame resolves cleanly).
    """
    failures: list[str] = []
    left = Path(pair["left_path"])
    if not left.is_absolute():
        left = workspace / left
    label_path = Path(source_root) / "labels" / pair["split"] / f"{stem}.txt"
    if not left.is_file():
        raise FrameResolutionError(f"{stem}: source image missing: {left}")
    if not label_path.is_file():
        raise FrameResolutionError(f"{stem}: label file missing: {label_path}")

    gray = load_gray(left)
    height, width = gray.shape[:2]
    polygons = read_yolo_polygons(label_path, width, height)
    if not polygons:
        raise FrameResolutionError(f"{stem}: label file has no polygons: {label_path}")

    records: list[MaskSampleRecord] = []
    masks: list[np.ndarray] = []
    for row in rows:
        sample_name = row["sample_name"]
        object_index = int(row["object_index"])
        if not 0 <= object_index < len(polygons):
            failures.append(
                f"{sample_name}: object_index {object_index} is out of range for "
                f"{label_path} ({len(polygons)} polygons); the sample cannot be "
                "matched to an annotation"
            )
            continue
        annotation = polygons[object_index]
        class_id = int(row["class_id"])
        if annotation.class_id != class_id:
            failures.append(
                f"{sample_name}: label class {annotation.class_id} != manifest class "
                f"{class_id}; refusing to guess which object this sample is"
            )
            continue
        if not 0 <= class_id < len(class_names):
            failures.append(
                f"{sample_name}: class id {class_id} outside the source class list"
            )
            continue
        if class_names[class_id] != row["class_name"]:
            failures.append(
                f"{sample_name}: source class name {class_names[class_id]!r} != "
                f"manifest {row['class_name']!r}"
            )
            continue

        v4_path = Path(v4_root) / row["npz_path"]
        if not v4_path.is_file():
            failures.append(f"{sample_name}: V4 sample missing: {v4_path}")
            continue
        v4_sample = load_fusion_sample(v4_path)
        try:
            window = make_crop_window(annotation.polygon, width, height, crop_pad)
        except ValueError as exc:
            failures.append(f"{sample_name}: {exc}")
            continue
        crop_shape = (window.y2 - window.y1, window.x2 - window.x1)
        if crop_shape != v4_sample.gray.shape:
            failures.append(
                f"{sample_name}: reconstructed crop {crop_shape} != V4 crop "
                f"{v4_sample.gray.shape}; the crop window is not reproducible"
            )
            continue
        crop_gray = gray[window.y1 : window.y2, window.x1 : window.x2]
        if not np.array_equal(crop_gray, v4_sample.gray):
            failures.append(
                f"{sample_name}: reconstructed gray crop differs from the V4 "
                "sample; the source frame or the crop window is not the one used "
                "to generate V4"
            )
            continue

        full_mask = build_object_mask(annotation.polygon, width, height)
        mask = full_mask[window.y1 : window.y2, window.x1 : window.x2]
        if int(mask.sum()) == 0:
            failures.append(f"{sample_name}: instance mask is empty inside the crop")
            continue
        outside = int(((v4_sample.valid > 0) & (mask == 0)).sum())
        if outside:
            failures.append(
                f"{sample_name}: {outside} V4-valid pixels lie outside the "
                "instance polygon; the mask does not describe the measurement "
                "support"
            )
            continue

        mask_pixels = int(mask.sum())
        crop_area = int(mask.size)
        records.append(
            MaskSampleRecord(
                sample_name=sample_name,
                split=row["split"],
                group_name=row["group_name"],
                class_id=class_id,
                class_name=row["class_name"],
                source_frame=stem,
                object_index=object_index,
                stereo_valid=row["stereo_valid"] == "true",
                stereo_reason=row["stereo_reason"],
                stereo_valid_ratio=float(row["stereo_valid_ratio"]),
                disparity=float(row["disparity"]),
                polar_valid_ratio=float(row["polar_valid_ratio"]),
                quality=tuple(
                    float(row[f"quality_{key}"]) for key in QUALITY_VECTOR_KEYS
                ),
                npz_path=row["npz_path"],
                left_path=_project_relative(left),
                label_path=_project_relative(label_path),
                image_width=int(width),
                image_height=int(height),
                crop_x1=int(window.x1),
                crop_y1=int(window.y1),
                crop_x2=int(window.x2),
                crop_y2=int(window.y2),
                mask_pixels=mask_pixels,
                crop_area=crop_area,
                mask_area_ratio=mask_pixels / crop_area,
            )
        )
        masks.append(mask)
    return records, masks, failures


def group_by_frame(v4_rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group V4 rows by source frame, preserving the V4 manifest order."""
    order: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for row in v4_rows:
        stem = row["source_frame"]
        if stem not in grouped:
            grouped[stem] = []
            order.append(stem)
        grouped[stem].append(row)
    return [(stem, grouped[stem]) for stem in order]


def run_provenance_check(
    source_root: Path,
    v4_root: Path,
    crop_pad: int,
    workspace: Path,
) -> dict:
    """Read-only correspondence check over the whole V4 dataset."""
    source_root = _resolve(source_root, workspace)
    v4_root = _resolve(v4_root, workspace)
    class_names = read_class_names(source_root)
    pairs = read_pair_manifest(source_root, workspace)
    v4_rows = read_v4_manifest(v4_root)

    resolved: list[MaskSampleRecord] = []
    failures: list[str] = []
    unresolved_frames: list[str] = []
    for stem, rows in group_by_frame(v4_rows):
        pair = pairs.get(stem)
        if pair is None:
            unresolved_frames.append(f"{stem}: not present in the source pair manifest")
            continue
        try:
            records, _, frame_failures = resolve_frame(
                stem, rows, pair, class_names, source_root, v4_root, crop_pad,
                workspace,
            )
        except FrameResolutionError as exc:
            unresolved_frames.append(str(exc))
            continue
        resolved.extend(records)
        failures.extend(frame_failures)

    expected = {split: 0 for split in SPLIT_NAMES}
    for row in v4_rows:
        expected[row["split"]] = expected.get(row["split"], 0) + 1
    return {
        "v4_root": str(Path(v4_root)),
        "source_root": str(Path(source_root)),
        "crop_pad": int(crop_pad),
        "v4_samples": len(v4_rows),
        "resolved_samples": len(resolved),
        "v4_expected_splits": expected,
        "resolved_splits": {
            split: sum(1 for r in resolved if r.split == split) for split in SPLIT_NAMES
        },
        "unresolved_frames": unresolved_frames[:50],
        "unresolved_frame_count": len(unresolved_frames),
        "failures": failures[:50],
        "failure_count": len(failures),
        "class_names": class_names,
        "records": resolved,
    }


def _prepare_output_root(output_root: Path) -> None:
    """Refuse an existing output directory; there is no --clean."""
    resolved = Path(output_root).resolve()
    if resolved.exists():
        raise FileExistsError(
            f"V5 output directory already exists: {resolved}; this generator "
            "never overwrites an existing dataset — remove it explicitly or "
            "choose another --output"
        )
    resolved.mkdir(parents=True, exist_ok=False)


def records_from_manifest(output_root: Path) -> list[MaskSampleRecord]:
    """Rebuild the sample records from a written V5 ``dataset_manifest.csv``.

    Used to re-audit an existing V5 dataset (and by tests that build a second
    derivation of the same V4 state) without re-running the provenance chain.
    """
    output_root = Path(output_root)
    with (output_root / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        records.append(
            MaskSampleRecord(
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
                quality=tuple(
                    float(row[f"quality_{key}"]) for key in QUALITY_VECTOR_KEYS
                ),
                npz_path=row["npz_path"],
                left_path=row["left_path"],
                label_path=row["label_path"],
                image_width=int(row["image_width"]),
                image_height=int(row["image_height"]),
                crop_x1=int(row["crop_x1"]),
                crop_y1=int(row["crop_y1"]),
                crop_x2=int(row["crop_x2"]),
                crop_y2=int(row["crop_y2"]),
                mask_pixels=int(row["mask_pixels"]),
                crop_area=int(row["crop_area"]),
                mask_area_ratio=float(row["mask_area_ratio"]),
            )
        )
    return records


def write_manifest(output_root: Path, records: list[MaskSampleRecord], summary: dict) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_FIELDS))
    writer.writeheader()
    for record in records:
        writer.writerow(record.row())
    (output_root / "dataset_manifest.csv").write_text(buffer.getvalue(), encoding="utf-8")
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_mask_dataset(
    source_root: Path,
    v4_root: Path,
    output_root: Path,
    crop_pad: int,
    workspace: Path,
    expected_parent_fingerprint: str = V4_PARENT_FINGERPRINT,
) -> dict:
    """Generate the V5 dataset, then audit it. Raises on any failure.

    ``expected_parent_fingerprint`` is the V4 state the masks must extend; it
    defaults to the formal fair-comparison V4 dataset and is only overridden
    when building a synthetic dataset in tests.
    """
    check = run_provenance_check(source_root, v4_root, crop_pad, workspace)
    if check["failure_count"] or check["unresolved_frame_count"]:
        raise ValueError(
            "the provenance check failed; refusing to build an incomplete dataset: "
            f"{check['unresolved_frame_count']} unresolved frames, "
            f"{check['failure_count']} sample failures; first entries: "
            f"{(check['unresolved_frames'] + check['failures'])[:5]}"
        )

    source_root = _resolve(source_root, workspace)
    v4_root = _resolve(v4_root, workspace)
    class_names = check["class_names"]
    v4_summary = json.loads((Path(v4_root) / "dataset_summary.json").read_text(encoding="utf-8"))
    v4_generation = v4_summary.get("generation", {})
    if v4_generation.get("crop_pad") != crop_pad:
        raise ValueError(
            f"V4 was generated with crop_pad {v4_generation.get('crop_pad')!r} but "
            f"this run uses {crop_pad}; the crop window would not be the V4 window"
        )
    expected_splits = {
        split: int(count) for split, count in v4_summary["splits"].items()
    }

    _prepare_output_root(output_root)
    for split in SPLIT_NAMES:
        for class_name in class_names:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)

    records: list[MaskSampleRecord] = []
    pairs = read_pair_manifest(source_root, workspace)
    v4_rows = read_v4_manifest(v4_root)
    for stem, rows in group_by_frame(v4_rows):
        pair = pairs[stem]
        frame_records, masks, frame_failures = resolve_frame(
            stem, rows, pair, class_names, source_root, v4_root, crop_pad, workspace
        )
        if frame_failures:
            raise ValueError(
                f"frame {stem} failed after the check passed: {frame_failures[:3]}"
            )
        for record, mask in zip(frame_records, masks):
            v4_sample = load_fusion_sample(Path(v4_root) / record.npz_path)
            save_mask_sample(
                output_root / record.npz_path,
                gray=v4_sample.gray,
                signed_q=v4_sample.signed_q,
                abs_q=v4_sample.abs_q,
                valid=v4_sample.valid,
                quality=v4_sample.quality,
                class_id=v4_sample.class_id,
                instance_mask=mask,
            )
            records.append(record)

    if len(records) != len(v4_rows):
        raise ValueError(
            f"V5 has {len(records)} samples but V4 has {len(v4_rows)}; the sample "
            "set must be preserved exactly"
        )

    summary = {
        "class_names": class_names,
        "npz_fields": list(MASK_NPZ_FIELDS),
        "quality_vector_keys": list(QUALITY_VECTOR_KEYS),
        "splits": {
            split: sum(1 for r in records if r.split == split) for split in SPLIT_NAMES
        },
        "total_samples": len(records),
        "generation": {
            "v4_root": str(Path(v4_root)),
            "source_root": str(Path(source_root)),
            "crop_pad": int(crop_pad),
            "mask_schema_version": MASK_SCHEMA_VERSION,
            "mask_resize_version": MASK_RESIZE_VERSION,
            "v4_generation": v4_generation,
            "instance_mask": (
                "annotation polygon rasterised with "
                "scripts.prepare_cls_fusion_dataset.build_object_mask, cropped "
                "with the identical V4 crop window"
            ),
            "regions": {
                "instance_mask": "annotated target region",
                "polar_valid": "V4 measurement-valid region (already "
                               "target-interior in V4)",
                "instance_valid": "instance_mask & polar_valid",
            },
            "note": (
                "V5 does not restore measurement validity outside the target and "
                "does not recompute SGBM, q or the gain"
            ),
        },
    }
    write_manifest(output_root, records, summary)

    audit = audit_mask_dataset(
        output_root=output_root,
        records=records,
        class_names=class_names,
        v4_root=Path(v4_root),
        expected_splits=expected_splits,
        crop_pad=crop_pad,
        workspace=workspace,
        expected_parent_fingerprint=expected_parent_fingerprint,
    )
    return {"records": records, "summary": summary, "audit": audit}


def audit_mask_dataset(
    output_root: Path,
    records: list[MaskSampleRecord],
    class_names: list[str],
    v4_root: Path,
    expected_splits: dict[str, int],
    crop_pad: int,
    workspace: Path,
    expected_parent_fingerprint: str = V4_PARENT_FINGERPRINT,
) -> dict:
    """Decode-check every V5 sample and bind the mask to its sources."""
    from core.material_mask_dataset import load_mask_sample

    output_root = Path(output_root)
    manifest_path = output_root / "dataset_manifest.csv"
    summary_path = output_root / "dataset_summary.json"
    v4_manifest = v4_root / "dataset_manifest.csv"

    audit: dict = {
        "output_root": output_root.resolve().as_posix(),
        "v4_root": v4_root.resolve().as_posix(),
        "npz_fields": list(MASK_NPZ_FIELDS),
        "quality_vector_keys": list(QUALITY_VECTOR_KEYS),
        "class_names": class_names,
        "expected_splits": expected_splits,
        "manifest_sha256": file_digest(manifest_path),
        "summary_sha256": file_digest(summary_path),
        "v4_manifest_sha256": file_digest(v4_manifest),
        "v4_parent_fingerprint": dataset_fingerprint(v4_root),
        "mask_schema_version": MASK_SCHEMA_VERSION,
        "mask_resize_version": MASK_RESIZE_VERSION,
        "crop_pad": int(crop_pad),
    }
    if expected_parent_fingerprint and (
        audit["v4_parent_fingerprint"] != expected_parent_fingerprint
    ):
        raise ValueError(
            f"the V4 dataset at {v4_root} fingerprints as "
            f"{audit['v4_parent_fingerprint']}, which is not the expected V4 state "
            f"{expected_parent_fingerprint}; refusing to derive V5 from another "
            "V4 state"
        )

    counts = {split: sum(1 for r in records if r.split == split) for split in SPLIT_NAMES}
    audit["counts"] = {"total": len(records), **counts}
    audit["count_match_expected"] = counts == expected_splits
    per_class = {
        name: {split: 0 for split in SPLIT_NAMES} for name in class_names
    }
    group_to_splits: dict[str, set[str]] = {}
    for record in records:
        per_class[record.class_name][record.split] += 1
        group_to_splits.setdefault(record.group_name, set()).add(record.split)
    audit["per_class"] = per_class
    audit["group_split_leakage"] = {
        group: sorted(splits)
        for group, splits in group_to_splits.items()
        if len(splits) > 1
    }

    decode_failures: list[str] = []
    mask_failures: list[str] = []
    v4_mismatch_failures: list[str] = []
    quality_recompute_failures: list[str] = []
    valid_outside_failures: list[str] = []
    file_digests: dict[str, str] = {}
    mask_entries: list[tuple[str, np.ndarray]] = []
    area_ratios: list[float] = []
    valid_ratios: list[float] = []

    for record in records:
        path = output_root / record.npz_path
        file_digests[record.npz_path] = file_digest(path)
        try:
            sample = load_mask_sample(path)
        except Exception as exc:  # decode / dtype / shape / mask invariants
            decode_failures.append(f"{record.npz_path}: {exc}")
            continue
        if sample.class_id != record.class_id:
            decode_failures.append(
                f"{record.npz_path}: class_id {sample.class_id} != {record.class_id}"
            )
            continue

        v4_sample = load_fusion_sample(v4_root / record.npz_path)
        # Requirement: the six V4 arrays are unchanged element for element.
        if (
            not np.array_equal(sample.gray, v4_sample.gray)
            or not np.array_equal(sample.signed_q, v4_sample.signed_q)
            or not np.array_equal(sample.abs_q, v4_sample.abs_q)
            or not np.array_equal(sample.valid, v4_sample.valid)
            or not np.array_equal(sample.quality, v4_sample.quality)
            or sample.class_id != v4_sample.class_id
        ):
            v4_mismatch_failures.append(record.npz_path)
            continue

        mask = sample.instance_mask
        if not bool(np.isin(mask, (0, 1)).all()):
            mask_failures.append(f"{record.npz_path}: mask is not binary")
            continue
        if mask.shape != sample.gray.shape:
            mask_failures.append(f"{record.npz_path}: mask shape != crop shape")
            continue
        mask_bool = mask > 0
        if int(mask_bool.sum()) != record.mask_pixels:
            mask_failures.append(
                f"{record.npz_path}: mask pixel count {int(mask_bool.sum())} != "
                f"manifest {record.mask_pixels}"
            )
            continue
        outside = int(((sample.valid > 0) & ~mask_bool).sum())
        if outside:
            valid_outside_failures.append(f"{record.npz_path}: {outside} pixels")
            continue

        # Quality components: ``valid_ratio`` and ``mean_abs_q`` are exactly
        # recomputable now that the mask is stored; the in-bounds and
        # brightness ratios are NOT (their per-pixel masks are not stored),
        # so only their containment bounds can be checked.
        total = int(mask_bool.sum())
        valid_count = int((sample.valid > 0).sum())
        recomputed_valid_ratio = valid_count / total
        quality = np.asarray(sample.quality, dtype=np.float64)
        if abs(quality[0] - recomputed_valid_ratio) > 1.5e-6:
            quality_recompute_failures.append(
                f"{record.npz_path}: quality valid_ratio {quality[0]:.6f} != "
                f"recomputed {recomputed_valid_ratio:.6f}"
            )
            continue
        if quality[0] > quality[1] + 1.5e-6 or quality[0] > quality[2] + 1.5e-6:
            quality_recompute_failures.append(
                f"{record.npz_path}: valid_ratio {quality[0]:.6f} exceeds the "
                f"in-bounds/brightness bounds ({quality[1]:.6f}, {quality[2]:.6f})"
            )
            continue
        if valid_count:
            recomputed_mean = float(sample.abs_q[sample.valid > 0].mean())
            if abs(quality[3] - recomputed_mean) > 1.5e-6:
                quality_recompute_failures.append(
                    f"{record.npz_path}: mean_abs_q {quality[3]:.6f} != recomputed "
                    f"{recomputed_mean:.6f}"
                )
                continue

        mask_entries.append((record.sample_name, mask))
        area_ratios.append(record.mask_area_ratio)
        valid_ratios.append(recomputed_valid_ratio)

    audit["decoded_samples"] = len(mask_entries)
    audit["decode_failures"] = decode_failures[:50]
    audit["mask_failures"] = mask_failures[:50]
    audit["v4_elementwise_mismatches"] = v4_mismatch_failures[:50]
    audit["quality_recompute_failures"] = quality_recompute_failures[:50]
    audit["valid_outside_mask_failures"] = valid_outside_failures[:50]
    audit["file_digests"] = file_digests
    audit["mask_content_sha256"] = mask_content_digest(mask_entries)
    audit["label_source_sha256"] = label_source_digest(
        (record.label_path, file_digest(resolve_source_path(record.label_path)))
        for record in records
    )
    audit["sample_mapping_sha256"] = sample_mapping_digest(
        record.mapping_entry() for record in records
    )
    audit["mask_area_ratio"] = {
        "mean": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "min": float(np.min(area_ratios)) if area_ratios else 0.0,
        "max": float(np.max(area_ratios)) if area_ratios else 0.0,
    }
    audit["polar_valid_ratio_inside_mask"] = {
        "mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "min": float(np.min(valid_ratios)) if valid_ratios else 0.0,
        "max": float(np.max(valid_ratios)) if valid_ratios else 0.0,
    }
    audit["quality_recomputable_components"] = ["valid_ratio", "mean_abs_q"]
    audit["quality_bound_only_components"] = [
        "in_bounds_ratio", "brightness_valid_ratio"
    ]
    audit["quality_note"] = (
        "instance_mask makes valid_ratio exactly recomputable (valid pixels / "
        "instance-mask pixels). in_bounds_ratio and brightness_valid_ratio use "
        "per-pixel masks that are not stored, so only their containment bounds "
        "are verified; this audit does not claim they are exactly recomputable."
    )
    audit["regions"] = {
        "instance_mask": "annotated target region (this dataset adds it)",
        "polar_valid": "V4 measurement-valid region (target-interior in V4)",
        "instance_valid": "instance_mask & polar_valid",
        "note": "V5 does not restore measurement validity outside the target",
    }

    referenced = {r.npz_path for r in records}
    audit["orphan_npz_files"] = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*.npz")
        if path.relative_to(output_root).as_posix() not in referenced
    )[:50]

    audit["audit_passed"] = (
        not decode_failures
        and not mask_failures
        and not v4_mismatch_failures
        and not quality_recompute_failures
        and not valid_outside_failures
        and not audit["orphan_npz_files"]
        and not audit["group_split_leakage"]
        and audit["count_match_expected"]
        and bool(audit["manifest_sha256"])
        and bool(audit["summary_sha256"])
        and bool(audit["mask_content_sha256"])
    )
    text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    (output_root / "dataset_audit.json").write_text(text, encoding="utf-8")
    return audit


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = PROJECT_ROOT
    source_root = _resolve(args.source, workspace)
    v4_root = _resolve(args.v4, workspace)

    if args.check_only:
        check = run_provenance_check(
            source_root, v4_root, args.crop_pad, workspace
        )
        check.pop("records", None)
        report = json.dumps(check, ensure_ascii=False, indent=2) + "\n"
        print(report)
        if args.report:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
        return 0 if not check["failure_count"] and not check["unresolved_frame_count"] else 1

    try:
        result = build_mask_dataset(
            source_root=source_root,
            v4_root=v4_root,
            output_root=_resolve(args.output, workspace),
            crop_pad=args.crop_pad,
            workspace=workspace,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"V5 generation refused: {exc}", file=sys.stderr)
        return 2
    audit = result["audit"]
    print(f"V5 dataset: {_resolve(args.output, workspace)}")
    print(f"Samples: {audit['counts']['total']} {audit['counts']}")
    print(f"Audit passed: {audit['audit_passed']}")
    print(f"Mask content: {audit['mask_content_sha256']}")
    if not audit["audit_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
