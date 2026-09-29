"""V5 instance-mask dataset contract (extension of the V4 fusion npz format).

V5 is the V4 dataset plus one array per sample:

    instance_mask  uint8  (h, w)   1 inside the source instance polygon

The six original fields (``gray``, ``signed_q``, ``abs_q``, ``valid``,
``quality``, ``class_id``) are copied from V4 element for element and are
held to exactly the same contract (:func:`core.fusion_dataset.validate_fusion_arrays`).
No stereo matching, no polarization re-estimation and no split change is
performed by the V5 generation; only the polygon-derived mask and its
provenance are added.

Three regions are kept distinct, because conflating them is the mistake this
batch exists to avoid:

- ``instance_mask``      — the annotated target region (this batch adds it).
- ``polar_valid``        — the V4 measurement-valid region, which V4 already
                           restricted to the target interior.
- ``instance_mask & polar_valid`` — the target-interior measurable region.

V5 does *not* restore measurement validity outside the target; the V4
``valid`` channel keeps exactly its previous meaning.

Schema/version strings are recorded in the dataset audit and in every
checkpoint so a mask-pooling model can never be confused with a v1 GAP model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.fusion_dataset import (
    NPZ_FIELDS,
    QUALITY_VECTOR_LENGTH,
    dataset_fingerprint,
    file_digest,
    validate_fusion_arrays,
    verify_dataset_integrity,
)

MASK_FIELD = "instance_mask"
MASK_NPZ_FIELDS = (*NPZ_FIELDS, MASK_FIELD)

# The pooling protocols a checkpoint can record. Defined here (the protocol
# module) so dataset admission, the model and the training entry all compare
# against one set of literals instead of repeating the strings.
POOLING_GAP = "gap"
POOLING_INSTANCE_MASK = "instance_mask"
POOLING_MODES = (POOLING_GAP, POOLING_INSTANCE_MASK)


def validate_pooling_mode(pooling_mode: str) -> str:
    """Return the pooling mode after checking it is one of the two."""
    if pooling_mode not in POOLING_MODES:
        raise ValueError(
            f"unknown pooling_mode {pooling_mode!r}; expected one of {POOLING_MODES}"
        )
    return pooling_mode


# A dataset fingerprint is a 64-character lowercase hex sha256. Anything else
# is refused rather than coerced, because an unset fingerprint would otherwise
# silently degrade to "any dataset matches".
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def require_fingerprint(value, field: str) -> str:
    """Return a validated 64-hex lowercase sha256 dataset fingerprint.

    ``None``, booleans, numbers, empty or whitespace strings, wrong lengths
    and non-lowercase-hex characters are all refused. There is deliberately
    no ``str()``/``strip()``/zero-padding coercion: a caller that cannot name
    the concrete dataset must fail, not be repaired into passing.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field} must be a 64-char lowercase hex sha256 string, got "
            f"{type(value).__name__}: {value!r}"
        )
    if _FINGERPRINT_PATTERN.match(value) is None:
        raise ValueError(
            f"{field} must be exactly 64 lowercase hex characters (a sha256 "
            f"dataset fingerprint), got {value!r}"
        )
    return value

# How the mask was produced from the annotations (polygon rasterisation plus
# the V4 crop window), and how it is turned into per-scale pooling weights.
MASK_SCHEMA_VERSION = "instance_mask_polygon_crop_v1"
MASK_RESIZE_VERSION = "area_crop_to_imgsz_then_area_to_levels_v1"

# The V4 dataset this extension was derived from (the formal fair-comparison
# state; the audit records it so a V5 dataset is always bound to its parent).
V4_PARENT_FINGERPRINT = (
    "8db8998d7a4c280064626961e4d51c92c531878ea2e92f0d1480df46379bd334"
)

CLASS_NAMES = (
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
)

# Fields the V5 audit must carry; a report without them cannot bind the mask
# data or the parent dataset and is refused by the training gate.
REQUIRED_AUDIT_FIELDS = (
    "audit_passed",
    "manifest_sha256",
    "summary_sha256",
    "file_digests",
    "mask_content_sha256",
    "v4_parent_fingerprint",
    "v4_manifest_sha256",
    "label_source_sha256",
    "sample_mapping_sha256",
    "mask_schema_version",
    "mask_resize_version",
    "crop_pad",
)


@dataclass(frozen=True)
class MaskFusionSampleArrays:
    """One V5 sample: the V4 arrays plus the instance mask."""

    gray: np.ndarray
    signed_q: np.ndarray
    abs_q: np.ndarray
    valid: np.ndarray
    quality: np.ndarray
    class_id: int
    instance_mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.gray.shape

    @property
    def polar_valid(self) -> np.ndarray:
        """The V4 measurement-valid region (target-interior, V4 definition)."""
        return self.valid > 0

    @property
    def instance_valid(self) -> np.ndarray:
        """Target-interior measurable region: ``instance_mask & polar_valid``."""
        return (self.instance_mask > 0) & (self.valid > 0)


def validate_instance_mask(
    path: str | Path,
    instance_mask: np.ndarray,
    valid: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Enforce the instance-mask invariants for one sample.

    Binary values, non-empty, same spatial shape as the crop, and the V4
    measurement-valid region may not extend beyond the annotated target.
    """
    path = Path(path)
    if instance_mask.dtype != np.uint8:
        raise ValueError(
            f"{path}: {MASK_FIELD} dtype {instance_mask.dtype} != uint8"
        )
    if instance_mask.shape != tuple(shape):
        raise ValueError(
            f"{path}: {MASK_FIELD} shape {instance_mask.shape} != crop shape "
            f"{tuple(shape)}"
        )
    if not bool(np.isin(instance_mask, (0, 1)).all()):
        raise ValueError(f"{path}: {MASK_FIELD} is not strictly binary")
    mask_bool = instance_mask > 0
    if not mask_bool.any():
        raise ValueError(f"{path}: {MASK_FIELD} is empty")
    outside = int(((valid > 0) & ~mask_bool).sum())
    if outside:
        raise ValueError(
            f"{path}: {outside} V4-valid pixels lie outside the instance mask; "
            "the mask does not describe the measurement support"
        )
    return instance_mask


def save_mask_sample(
    path: str | Path,
    *,
    gray: np.ndarray,
    signed_q: np.ndarray,
    abs_q: np.ndarray,
    valid: np.ndarray,
    quality: np.ndarray,
    class_id: int,
    instance_mask: np.ndarray,
) -> Path:
    """Write one V5 sample; the V4 arrays are stored exactly as given."""
    path = Path(path)
    gray = np.asarray(gray, dtype=np.uint8)
    signed_q = np.asarray(signed_q, dtype=np.float32)
    abs_q = np.asarray(abs_q, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.uint8)
    quality = np.asarray(quality, dtype=np.float32)
    instance_mask = np.asarray(instance_mask, dtype=np.uint8)
    if quality.shape != (QUALITY_VECTOR_LENGTH,):
        raise ValueError(
            f"quality must have shape ({QUALITY_VECTOR_LENGTH},), got {quality.shape}"
        )
    validate_instance_mask(path, instance_mask, valid, gray.shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gray=gray,
        signed_q=signed_q,
        abs_q=abs_q,
        valid=valid,
        quality=quality,
        class_id=np.asarray(int(class_id), dtype=np.int64),
        instance_mask=instance_mask,
    )
    return path


def load_mask_sample(path: str | Path) -> MaskFusionSampleArrays:
    """Load and validate one V5 sample (V4 contract plus the mask contract)."""
    path = Path(path)
    with np.load(path) as data:
        missing = [field for field in MASK_NPZ_FIELDS if field not in data.files]
        if missing:
            raise ValueError(
                f"{path} is missing fields: {', '.join(missing)}"
            )
        base = validate_fusion_arrays(
            path,
            gray=data["gray"],
            signed_q=data["signed_q"],
            abs_q=data["abs_q"],
            valid=data["valid"],
            quality=data["quality"],
            class_id=data["class_id"],
        )
        instance_mask = data[MASK_FIELD]
    validate_instance_mask(path, instance_mask, base.valid, base.gray.shape)
    return MaskFusionSampleArrays(
        gray=base.gray,
        signed_q=base.signed_q,
        abs_q=base.abs_q,
        valid=base.valid,
        quality=base.quality,
        class_id=base.class_id,
        instance_mask=instance_mask,
    )


def mask_content_digest(entries) -> str:
    """Order-independent digest of the mask arrays of a dataset.

    ``entries`` is an iterable of ``(sample_name, instance_mask)``. Each
    entry is folded in sorted order so the digest identifies the mask
    content of the whole dataset regardless of file iteration order.
    """
    digest = hashlib.sha256()
    digest.update(b"mask_schema\0")
    digest.update(MASK_SCHEMA_VERSION.encode("ascii"))
    digest.update(b"\0resize\0")
    digest.update(MASK_RESIZE_VERSION.encode("ascii"))
    for name, mask in sorted(entries, key=lambda item: item[0]):
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        digest.update(b"\0sample\0")
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0shape\0")
        digest.update(f"{mask.shape[0]}x{mask.shape[1]}".encode("ascii"))
        digest.update(b"\0pixels\0")
        digest.update(int(mask.sum()).to_bytes(8, "big"))
        digest.update(b"\0bytes\0")
        digest.update(hashlib.sha256(mask.tobytes()).hexdigest().encode("ascii"))
    return digest.hexdigest()


def _read_audit(data_root: Path) -> dict:
    audit_path = Path(data_root) / "dataset_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"missing dataset audit report: {audit_path}; run "
            "scripts/prepare_material_mask_dataset.py to build and audit V5"
        )
    return json.loads(audit_path.read_text(encoding="utf-8"))


def _digest_of_pairs(prefix: bytes, pairs) -> str:
    """Order-independent sha256 over ``(key, value)`` text pairs."""
    digest = hashlib.sha256()
    digest.update(prefix)
    for key, value in sorted(pairs, key=lambda item: str(item[0])):
        digest.update(b"\0")
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def label_source_digest(entries) -> str:
    """Digest of the annotation files the masks were rasterised from."""
    return _digest_of_pairs(b"label_source\0", entries)


def sample_mapping_digest(entries) -> str:
    """Digest of the sample -> frame/object/crop-coordinate mapping."""
    return _digest_of_pairs(b"sample_mapping\0", entries)


# Project root, used to resolve the manifest's project-relative source paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The manifest columns that make up one sample-mapping entry, in the same
# order the V5 generator used.
MAPPING_COLUMNS = (
    "source_frame",
    "object_index",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "image_width",
    "image_height",
)


def resolve_source_path(value: str, workspace: Path | None = None) -> Path:
    """Resolve a manifest-recorded source path, refusing ambiguous escapes.

    Source paths are recorded project-relative by the generator. An absolute
    path, an empty value or anything that resolves outside the workspace is
    refused rather than guessed at.
    """
    workspace = Path(workspace) if workspace is not None else PROJECT_ROOT
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty source path in the manifest: {value!r}")
    raw = value.strip()
    path = Path(raw)
    if path.is_absolute() or path.drive or raw.startswith(("/", "\\")):
        raise ValueError(
            f"source path {raw!r} is absolute; the manifest records "
            "project-relative source paths"
        )
    workspace_resolved = workspace.resolve()
    resolved = (workspace_resolved / path).resolve()
    if resolved != workspace_resolved and workspace_resolved not in resolved.parents:
        raise ValueError(
            f"source path {raw!r} resolves outside the project root "
            f"({resolved}); refusing an escaping source reference"
        )
    return resolved


def verify_mask_sources(
    rows,
    *,
    expected_label_source_sha256: str,
    expected_sample_mapping_sha256: str,
    workspace: Path | None = None,
) -> dict:
    """Recompute the annotation-source and sample-mapping digests from the manifest.

    The audit records both digests, but a recorded digest only proves the
    data was consistent *when the audit ran*. This re-reads the annotation
    files named by the (already integrity-checked) manifest, re-hashes them
    and recomputes both digests with the same algorithm, so editing or
    deleting a source annotation after generation is caught at admission.

    File digests are cached per recorded path: several samples share one
    frame's label file, and the digest folds one entry per manifest row (so
    duplicate references and the sorted-by-key folding are preserved exactly
    as the generator produced them).
    """
    cache: dict[str, str] = {}
    label_entries: list[tuple[str, str]] = []
    mapping_entries: list[tuple[str, str]] = []
    missing: list[str] = []
    for row in rows:
        label_path = row.get("label_path", "")
        resolved = resolve_source_path(label_path, workspace)
        if label_path not in cache:
            if not resolved.is_file():
                missing.append(label_path)
                continue
            cache[label_path] = file_digest(resolved)
        label_entries.append((label_path, cache[label_path]))
        mapping_entries.append(
            (
                row["sample_name"],
                "|".join(str(row[column]) for column in MAPPING_COLUMNS),
            )
        )
    if missing:
        raise ValueError(
            f"{len(missing)} annotation source files recorded in the manifest are "
            f"missing, first entries: {sorted(set(missing))[:5]}; the mask "
            "provenance cannot be verified against its sources"
        )
    if len(label_entries) != len(rows):
        raise ValueError(
            f"only {len(label_entries)} of {len(rows)} manifest rows resolved to a "
            "source annotation"
        )

    recomputed_label = label_source_digest(label_entries)
    if recomputed_label != expected_label_source_sha256:
        changed = sorted(
            path for path, digest in cache.items()
        )
        raise ValueError(
            "the annotation sources changed since the audit passed: label source "
            f"digest {recomputed_label} != recorded "
            f"{expected_label_source_sha256}. The masks were rasterised from these "
            f"files; re-audit or regenerate V5 rather than editing the sources. "
            f"({len(changed)} distinct label files, first: {changed[:3]})"
        )
    recomputed_mapping = sample_mapping_digest(mapping_entries)
    if recomputed_mapping != expected_sample_mapping_sha256:
        raise ValueError(
            "the sample -> frame/object/crop mapping changed since the audit "
            f"passed: mapping digest {recomputed_mapping} != recorded "
            f"{expected_sample_mapping_sha256}"
        )
    return {
        "label_files": len(cache),
        "manifest_rows": len(rows),
        "label_source_sha256": recomputed_label,
        "sample_mapping_sha256": recomputed_mapping,
    }


def admit_dataset_for_protocol(
    data_root: str | Path,
    *,
    pooling_mode: str,
    expected_fingerprint: str = "",
    expected_parent_fingerprint: str = "",
    expected_mask_schema: str = "",
    expected_mask_resize_version: str = "",
) -> dict:
    """Admission gate selected by the run's *validated* pooling protocol.

    ``instance_mask`` runs go through the V5 gate (digest re-verification,
    mask decode, mask content digest, annotation source, V4 parent, class
    order); ``gap`` runs keep the original V4 gate. The pooling mode is
    validated first, so an unknown or empty protocol is refused instead of
    silently falling back to one gate. There is no path that skips the
    fingerprint comparison.

    A ``gap`` admission may be run without an expected fingerprint (the
    generic read-only audit; the caller then binds nothing). An
    ``instance_mask`` admission may not: a v2 model is defined by *which* V5
    dataset it was trained on, so an empty fingerprint or parent is refused
    here rather than being read as "any V5 will do". Both must be concrete
    64-hex digests.

    ``expected_parent_fingerprint`` is the V4 state the masks must extend;
    for a v2 checkpoint it comes from the checkpoint's own recorded parent,
    so the dataset and the weights are bound to the same V4 state.
    """
    pooling_mode = validate_pooling_mode(pooling_mode)
    if pooling_mode == POOLING_INSTANCE_MASK:
        require_fingerprint(expected_fingerprint, "expected_fingerprint")
        require_fingerprint(
            expected_parent_fingerprint, "expected_parent_fingerprint"
        )
        report = admit_mask_dataset(
            data_root,
            expected_fingerprint=expected_fingerprint,
            expected_parent_fingerprint=expected_parent_fingerprint,
        )
        for field, expected, actual in (
            ("mask_schema", expected_mask_schema, report["mask_schema_version"]),
            (
                "mask_resize_version",
                expected_mask_resize_version,
                report["mask_resize_version"],
            ),
        ):
            if expected and expected != actual:
                raise ValueError(
                    f"the dataset's {field} is {actual!r} but the run records "
                    f"{expected!r}; the mask derivation is not the one this run "
                    "was trained with"
                )
        return report
    from core.material_fusion_dataset import admit_material_dataset

    return admit_material_dataset(data_root, expected_fingerprint)


def mask_dataset_fingerprint(data_root: str | Path) -> str:
    """Fingerprint of the audited V5 state.

    Binds the V5 manifest/summary/per-file digests (through
    :func:`core.fusion_dataset.dataset_fingerprint`) to the mask content, the
    annotation-source summary, the sample/crop mapping, the mask
    schema/resize versions, the crop padding and the V4 parent fingerprint,
    so a V5 dataset can never be confused with another V4 state or another
    mask derivation.
    """
    data_root = Path(data_root)
    audit = _read_audit(data_root)
    missing = [field for field in REQUIRED_AUDIT_FIELDS if field not in audit]
    if missing:
        raise ValueError(
            f"{data_root / 'dataset_audit.json'} is missing V5 audit fields: "
            f"{missing}; re-run the V5 generation/audit"
        )
    parent = dataset_fingerprint(data_root)
    digest = hashlib.sha256()
    digest.update(b"v5\0")
    digest.update(parent.encode("ascii"))
    for field in (
        "mask_content_sha256",
        "v4_parent_fingerprint",
        "v4_manifest_sha256",
        "label_source_sha256",
        "sample_mapping_sha256",
        "mask_schema_version",
        "mask_resize_version",
    ):
        digest.update(b"\0")
        digest.update(field.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(audit[field]).encode("utf-8"))
    digest.update(b"\0crop_pad\0")
    digest.update(str(audit["crop_pad"]).encode("ascii"))
    return digest.hexdigest()


def admit_mask_dataset(
    data_root: str | Path,
    expected_fingerprint: str = "",
    expected_parent_fingerprint: str = V4_PARENT_FINGERPRINT,
) -> dict:
    """Read-only admission gate for the V5 dataset.

    Re-verifies the recorded manifest/summary/per-file digests, then decodes
    every sample to re-check the mask invariants and to recompute the mask
    content digest, and finally compares the V5 fingerprint and the V4 parent
    fingerprint. Nothing is written. Raises on any mismatch.
    """
    import csv

    data_root = Path(data_root)
    audit = _read_audit(data_root)
    missing = [field for field in REQUIRED_AUDIT_FIELDS if field not in audit]
    if missing:
        raise ValueError(
            f"{data_root / 'dataset_audit.json'} is missing V5 audit fields: "
            f"{missing}; a V5 dataset must be audited before training"
        )
    if not audit["audit_passed"]:
        raise ValueError(
            f"V5 dataset audit did not pass ({data_root / 'dataset_audit.json'})"
        )
    integrity = verify_dataset_integrity(data_root)

    with (data_root / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty V5 manifest: {data_root / 'dataset_manifest.csv'}")
    names = sorted({row["class_name"] for row in rows})
    by_id = {int(row["class_id"]): row["class_name"] for row in rows}
    ordered = [by_id[key] for key in sorted(by_id)]
    if ordered != list(CLASS_NAMES):
        raise ValueError(
            f"V5 dataset class order {ordered} does not match the fixed project "
            f"order {list(CLASS_NAMES)}"
        )

    # The recorded digests only prove the sources were consistent when the
    # audit ran; re-read the annotations named by the (integrity-checked)
    # manifest and recompute both source digests, so a later edit or deletion
    # of a source annotation is caught here.
    source_report = verify_mask_sources(
        rows,
        expected_label_source_sha256=str(audit["label_source_sha256"]),
        expected_sample_mapping_sha256=str(audit["sample_mapping_sha256"]),
    )

    entries = []
    counts: dict[str, int] = {}
    for row in rows:
        sample = load_mask_sample(data_root / row["npz_path"])
        if sample.class_id != int(row["class_id"]):
            raise ValueError(
                f"{row['npz_path']}: class_id {sample.class_id} != manifest "
                f"{row['class_id']}"
            )
        entries.append((row["sample_name"], sample.instance_mask))
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    recomputed = mask_content_digest(entries)
    if recomputed != audit["mask_content_sha256"]:
        raise ValueError(
            "V5 mask content changed since the audit passed: mask digest "
            f"{recomputed} != recorded {audit['mask_content_sha256']}; re-run the "
            "V5 generation/audit"
        )

    fingerprint = mask_dataset_fingerprint(data_root)
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError(
            f"V5 dataset fingerprint {fingerprint} does not match the expected "
            f"{expected_fingerprint}; refusing to train on a different dataset state"
        )
    parent = str(audit["v4_parent_fingerprint"])
    if expected_parent_fingerprint and parent != expected_parent_fingerprint:
        raise ValueError(
            f"V5 dataset records V4 parent fingerprint {parent}, expected "
            f"{expected_parent_fingerprint}; this mask extension was not derived "
            "from the formal V4 dataset state"
        )
    return {
        "data_root": str(data_root),
        "audit_fingerprint": fingerprint,
        "v4_parent_fingerprint": parent,
        "verified_files": integrity["verified_files"],
        "mask_content_sha256": recomputed,
        "label_source_sha256": source_report["label_source_sha256"],
        "sample_mapping_sha256": source_report["sample_mapping_sha256"],
        "source_label_files": source_report["label_files"],
        "class_names": ordered,
        "splits": counts,
        "mask_schema_version": audit["mask_schema_version"],
        "mask_resize_version": audit["mask_resize_version"],
    }
