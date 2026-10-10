"""Frozen-feature preparation and polarization quality statistics (Part 2B).

Read-only over the datasets. For every dev-pool sample (old train+val,
bound to the protocol members) this entry

1. computes the polarization quality statistics (interior vs contour-band
   validity, q distributions on ``instance_mask & polar_valid``,
   brightness correlation, metadata availability) — statistics only,
   nothing is fitted;
2. extracts the frozen gray feature **G** with the OFFICIAL MiT-B0
   encoder (last stage, instance-mask weighted pooling) under
   ``eval()`` + ``inference_mode``, verifying the encoder's parameters
   and buffers are byte-identical before and after every forward pass;
3. builds the control vector **C** (the first three quality entries,
   in-mask gray statistics, explicit S-region flag) and the polarization
   statistics **S** of the ORIGINAL crop NPZ;
4. writes the RAW feature cache (no standardization, no PCA — the
   standardizer is fitted per fold inside the training entry) with full
   provenance binding.

The cache never contains training results or classification gains.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_mask_dataset import (  # noqa: E402
    admit_mask_dataset,
    load_mask_sample,
)
from core.material_probe import (  # noqa: E402
    CONTROL_FEATURE_KEYS,
    FEATURE_DEFINITION_VERSION,
    POLAR_STAT_KEYS,
    QUANTILE_METHOD,
    build_cache_binding,
    build_control_features,
    build_polar_stats,
    compare_module_states,
    snapshot_module_state,
)
from core.existing_data_protocol import (  # noqa: E402
    BudgetExceeded,
    BudgetWatch,
    protocol_digest,
    read_manifest_rows,
)

EXIT_OK = 0
EXIT_REFUSED = 1

# Interior = a single 3x3 erosion of the instance mask; the contour band is
# the rest of the mask. Frozen in the experiment config before any result
# was seen; never redefined afterwards.
INTERIOR_DEFINITION = "erode3x3_instance_mask_v1"
BAND_DEFINITION = "instance_mask_minus_interior_v1"


def load_protocol(path: str | Path) -> dict:
    protocol_path = Path(path)
    if protocol_path.is_dir():
        protocol_path = protocol_path / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"protocol not found: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    recorded = protocol.get("protocol_sha256", "")
    if not recorded or protocol_digest(protocol) != recorded:
        raise ValueError(
            f"protocol {protocol_path} failed its own digest check; the "
            "feature cache cannot bind to an unverifiable protocol"
        )
    return protocol


def fold_roles(protocol: dict) -> dict[str, dict[str, str]]:
    """``sample_name -> {fold_id: role}`` from the protocol members."""
    roles: dict[str, dict[str, str]] = {}
    for fold in protocol["folds"]["folds"]:
        fold_id = fold["fold_id"]
        for part in ("outer_eval", "inner_train", "inner_val"):
            for sample in fold[f"{part}_members"]["samples"]:
                roles.setdefault(sample["sample_name"], {})[fold_id] = part
    return roles


def verify_sample_set(protocol: dict, manifest_rows: list[dict]) -> list[str]:
    """The protocol members and the manifest dev rows must be the same set.

    A dev sample legitimately appears once per fold (one role in each),
    so the universes are compared on unique names, and every protocol
    sample must additionally hold exactly one role per fold.
    """
    role_entries = [
        (sample["sample_name"], fold["fold_id"], part)
        for fold in protocol["folds"]["folds"]
        for part in ("outer_eval", "inner_train", "inner_val")
        for sample in fold[f"{part}_members"]["samples"]
    ]
    fold_count = len(protocol["folds"]["folds"])
    from collections import Counter

    appearance_counts = Counter(name for name, _fold, _part in role_entries)
    protocol_names = sorted(appearance_counts)
    manifest_names = sorted(row["sample_name"] for row in manifest_rows)
    if protocol_names != manifest_names:
        missing = sorted(set(protocol_names) - set(manifest_names))[:10]
        extra = sorted(set(manifest_names) - set(protocol_names))[:10]
        raise ValueError(
            "protocol members and the dev-pool manifest disagree "
            f"(protocol-only={missing}, manifest-only={extra})"
        )
    wrong_slots = {
        name: count
        for name, count in appearance_counts.items()
        if count != fold_count
    }
    if wrong_slots:
        raise ValueError(
            "every dev sample must hold exactly one role per fold; "
            f"violations (first 10): {dict(sorted(wrong_slots.items())[:10])}"
        )
    return manifest_names


def interior_and_band_masks(instance_mask: np.ndarray):
    """The frozen interior / contour-band definitions (erode3x3_v1)."""
    import cv2

    mask = (np.asarray(instance_mask) > 0).astype(np.uint8)
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    band = mask & (1 - interior)
    return interior, band


def polarization_quality_statistics(
    rows: list[dict], data_root: Path
) -> tuple[dict, list[dict]]:
    """Per-sample and aggregated polarization quality statistics.

    Nothing here is fitted: plain counts, ratios and distributions over
    the dev pool, aggregated by original split, class and capture group.
    """
    per_sample: list[dict] = []
    by_split: dict[str, dict] = defaultdict(
        lambda: {"samples": 0, "interior_valid_ratio_sum": 0.0,
                 "band_valid_ratio_sum": 0.0, "brightness_sum": 0.0})
    by_class: dict = defaultdict(dict)
    by_group: dict = defaultdict(dict)

    for row in rows:
        sample = load_mask_sample(data_root / row["npz_path"])
        interior, band = interior_and_band_masks(sample.instance_mask)
        region = sample.instance_valid  # instance_mask & polar_valid

        def _ratio(region_mask: np.ndarray) -> float:
            total = int(region_mask.sum())
            if total == 0:
                return 0.0
            return float((sample.valid[region_mask] > 0).sum() / total)

        interior_ratio = _ratio(interior > 0)
        band_ratio = _ratio(band > 0)
        brightness = (
            float(sample.gray[sample.instance_mask > 0].astype(np.float64).mean())
            / 255.0 if int(sample.instance_mask.sum()) else 0.0
        )
        record = {
            "sample_name": row["sample_name"],
            "split": row["split"],
            "group": row["group_name"],
            "class_id": int(row["class_id"]),
            "interior_valid_ratio": interior_ratio,
            "band_valid_ratio": band_ratio,
            "valid_ratio_quality": float(sample.quality[0]),
            "brightness_0_1": brightness,
            "interior_pixels": int(interior.sum()),
            "band_pixels": int(band.sum()),
            "s_region_pixels": int(region.sum()),
            "has_valid_s": bool(region.any()),
        }
        if region.any():
            values = sample.signed_q[region]
            record["signed_q_mean"] = float(values.mean())
            record["signed_q_std_ddof0"] = float(values.std(ddof=0))
            record["abs_q_mean"] = float(np.abs(values).mean())
        per_sample.append(record)

        split_entry = by_split[row["split"]]
        split_entry["samples"] += 1
        split_entry["interior_valid_ratio_sum"] += interior_ratio
        split_entry["band_valid_ratio_sum"] += band_ratio
        split_entry["brightness_sum"] += brightness
        by_class[row["class_name"]][row["sample_name"]] = record
        by_group[row["group_name"]][row["sample_name"]] = record

    def _finish(entries: dict) -> dict:
        finished = {}
        for name, entry in entries.items():
            samples = entry["samples"]
            finished[name] = {
                "samples": samples,
                "interior_valid_ratio_mean": (
                    entry["interior_valid_ratio_sum"] / samples
                    if samples else 0.0),
                "band_valid_ratio_mean": (
                    entry["band_valid_ratio_sum"] / samples if samples else 0.0),
                "brightness_mean": (
                    entry["brightness_sum"] / samples if samples else 0.0),
            }
        return finished

    brightness = np.asarray([r["brightness_0_1"] for r in per_sample])
    valid_ratio = np.asarray([r["valid_ratio_quality"] for r in per_sample])
    correlation = {"pearson": None, "spearman": None}
    if len(per_sample) > 2 and np.std(brightness) > 0 and np.std(valid_ratio) > 0:
        correlation["pearson"] = float(np.corrcoef(brightness, valid_ratio)[0, 1])
        from scipy.stats import spearmanr

        correlation["spearman"] = float(spearmanr(brightness, valid_ratio).statistic)

    aggregate = {
        "interior_definition": INTERIOR_DEFINITION,
        "band_definition": BAND_DEFINITION,
        "q_region": "instance_mask_and_polar_valid",
        "note": (
            "correlation is not causation; no left/right misalignment or "
            "sync failure may be inferred from the q maps alone"
        ),
        "by_split": _finish(by_split),
        "by_class": {
            name: {
                "samples": len(entries),
                "has_valid_s": sum(1 for r in entries.values() if r["has_valid_s"]),
                "interior_valid_ratio_mean": float(np.mean(
                    [r["interior_valid_ratio"] for r in entries.values()]))
                if entries else 0.0,
                "band_valid_ratio_mean": float(np.mean(
                    [r["band_valid_ratio"] for r in entries.values()]))
                if entries else 0.0,
                "signed_q_mean_of_means": float(np.mean([
                    r["signed_q_mean"] for r in entries.values()
                    if "signed_q_mean" in r])) if any(
                    "signed_q_mean" in r for r in entries.values()) else None,
                "abs_q_mean_of_means": float(np.mean([
                    r["abs_q_mean"] for r in entries.values()
                    if "abs_q_mean" in r])) if any(
                    "abs_q_mean" in r for r in entries.values()) else None,
            }
            for name, entries in sorted(by_class.items())
        },
        "by_group": {
            name: {
                "samples": len(entries),
                "has_valid_s": sum(1 for r in entries.values() if r["has_valid_s"]),
                "interior_valid_ratio_mean": float(np.mean(
                    [r["interior_valid_ratio"] for r in entries.values()])),
                "band_valid_ratio_mean": float(np.mean(
                    [r["band_valid_ratio"] for r in entries.values()])),
            }
            for name, entries in sorted(by_group.items())
        },
        "brightness_vs_valid_ratio": correlation,
        "metadata_availability": {
            "exposure": "NOT_AVAILABLE",
            "gain": "NOT_AVAILABLE",
            "gamma": "NOT_AVAILABLE",
            "timestamp": "NOT_AVAILABLE",
            "sync_status": "unknown",
            "note": "no capture-side metadata files exist in this dataset",
        },
    }
    return aggregate, per_sample


def render_quality_plots(aggregate: dict, per_sample: list[dict],
                         output_dir: Path) -> list[str]:
    """Two small diagnostic PNGs (histogram + brightness scatter)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    interior = [r["interior_valid_ratio"] for r in per_sample]
    band = [r["band_valid_ratio"] for r in per_sample]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(interior, bins=30, range=(0, 1), color="#3b6ea5")
    axes[0].set_title("interior valid ratio")
    axes[1].hist(band, bins=30, range=(0, 1), color="#a55b3b")
    axes[1].set_title("contour-band valid ratio")
    figure.tight_layout()
    path = output_dir / "valid_ratio_histograms.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    written.append(path.name)

    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    axis.scatter([r["brightness_0_1"] for r in per_sample],
                 [r["valid_ratio_quality"] for r in per_sample],
                 s=4, alpha=0.35, color="#3b6ea5")
    pearson = aggregate["brightness_vs_valid_ratio"]["pearson"]
    axis.set_title(f"brightness vs valid ratio (pearson={pearson})")
    axis.set_xlabel("mean gray in mask [0,1]")
    axis.set_ylabel("quality valid_ratio")
    figure.tight_layout()
    path = output_dir / "brightness_vs_valid_ratio.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    written.append(path.name)
    return written


def build_probe_encoder(config: dict, project_root: Path):
    """The official MiT-B0 encoder (frozen), identity-verified."""
    import torch

    from models.cmx.dual_segformer import MiTEncoder, load_mit_b0_checkpoint

    mit = config["baselines"]["official_mit_b0"]
    encoder = MiTEncoder(img_size=int(config["features"]["imgsz"]))
    report = load_mit_b0_checkpoint(
        encoder, project_root / mit["path"],
        expected_sha256=mit["sha256"],
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, report


def extract_gray_features(
    rows: list[dict],
    data_root: Path,
    config: dict,
    device: str,
    log=lambda message: None,
    budget: BudgetWatch | None = None,
    progress: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Frozen MiT-B0 last-stage features, mask-pooled, batch by batch.

    The encoder runs under ``eval()`` + ``inference_mode``; parameters and
    buffers are snapshotted before the first batch and compared after the
    last one (plus every ``check_every`` batches). Requires_grad is
    disabled but that alone is never treated as proof — the snapshots are
    the evidence. ``budget`` (a shared :class:`BudgetWatch`) is checked at
    every batch boundary; on exhaustion the already-extracted batches are
    kept and the failure propagates so the entry can report PARTIAL.
    """
    import cv2
    import torch

    from models.material_fusion import mask_level_weights
    from models.polar_fusion import _fusion_tensors

    progress = progress if progress is not None else {}
    features: list[np.ndarray] = progress.setdefault("batches", [])
    progress["stage"] = "before encoder construction"
    if budget is not None:
        budget.check(progress["stage"])
    imgsz = int(config["features"]["imgsz"])
    encoder, load_report = build_probe_encoder(config, project_root=PROJECT_ROOT)
    from models.polar_fusion import torch_device_name

    torch_device = torch_device_name(str(device))
    encoder.to(torch_device)
    check_every = 200

    before = snapshot_module_state(encoder)
    started = time.perf_counter()
    batch_size = 32

    def check_budget(at):
        progress["stage"] = at
        if budget is not None:
            try:
                budget.check(at)
            except BudgetExceeded:
                progress["encoder_state_check"] = compare_module_states(
                    before, snapshot_module_state(encoder))
                raise

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            check_budget(f"extraction batch starting at sample {start}")
            batch_rows = rows[start:start + batch_size]
            gray_batch, mask_batch = [], []
            for row in batch_rows:
                sample = load_mask_sample(data_root / row["npz_path"])
                gray_t, _polar_t, _quality_t = _fusion_tensors(
                    sample, imgsz, cv2)  # [G, G, G] appearance, [0,1]
                mask_t = torch.from_numpy(np.ascontiguousarray(cv2.resize(
                    sample.instance_mask.astype(np.float32), (imgsz, imgsz),
                    interpolation=cv2.INTER_AREA))).float().clip_(0.0, 1.0)
                gray_batch.append(gray_t)
                mask_batch.append(mask_t)
            rgb = torch.stack(gray_batch).to(torch_device)
            masks = torch.stack(mask_batch).to(torch_device)
            stage_features = encoder(rgb)  # four levels; last stage only
            last = stage_features[-1]  # (B, C, H/32, W/32)
            weights = mask_level_weights(masks, last.shape[-2:]).to(
                dtype=last.dtype)
            total = weights.sum(dim=(2, 3))
            if bool((total <= 0).any()):
                raise ValueError(
                    "instance-mask pooling has zero weight for some batch "
                    "row; an empty mask is refused rather than GAP-pooled"
                )
            pooled = (last * weights).sum(dim=(2, 3)) / total
            features.append(pooled.cpu().numpy().astype(np.float32))
            done = start + len(batch_rows)
            check_budget(f"extraction batch ending at sample {done}")
            if done % check_every < batch_size or done == len(rows):
                report = compare_module_states(
                    before, snapshot_module_state(encoder))
                if not report["unchanged"]:
                    raise ValueError(
                        f"encoder parameters/buffers changed during feature "
                        f"extraction at sample {done}: "
                        f"{report['changed_tensors']}"
                    )
                log(f"  extracted {done}/{len(rows)} samples "
                    f"(encoder state verified unchanged)")
    after = snapshot_module_state(encoder)
    final_report = compare_module_states(before, after)
    if not final_report["unchanged"]:
        raise ValueError(
            "encoder parameters/buffers changed during feature extraction: "
            f"{final_report['changed_tensors']}"
        )
    return (
        np.concatenate(features, axis=0),
        {
            "load_report": {
                "loaded": load_report["loaded"],
                "missing": load_report["missing"],
                "unexpected": load_report["unexpected"],
                "random_init": load_report["random_init"],
            },
            "state_check": final_report,
            "forward_mode": "eval + inference_mode, requires_grad=False",
            "last_stage_shape_note": "features[-1] of MiTEncoder",
            "pooling": "instance_mask weighted mean (mask_level_weights)",
            "batch_size": batch_size,
            "elapsed_s": time.perf_counter() - started,
            "device": str(device),
        },
    )


def build_slot_features(
    rows: list[dict], data_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slot C / slot S / s_valid / class_id for every dev-pool row.

    Reads each sample's ORIGINAL crop npz once. The leakage rules live
    here: C is built from quality[0:3] + in-mask gray statistics only,
    S is computed from the signed_q/abs_q arrays on instance_mask &
    polar_valid — the quality vector's fourth entry is never read by C.
    """
    c_rows = np.zeros((len(rows), len(CONTROL_FEATURE_KEYS)), dtype=np.float32)
    s_rows = np.full((len(rows), len(POLAR_STAT_KEYS)), np.nan,
                     dtype=np.float32)
    s_valid = np.zeros(len(rows), dtype=bool)
    class_ids = np.zeros(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        sample = load_mask_sample(data_root / row["npz_path"])
        region = sample.instance_valid
        c_rows[index] = build_control_features(
            quality=sample.quality, gray=sample.gray,
            instance_mask=sample.instance_mask,
            s_region_nonempty=bool(region.any()))
        s_values, has_valid = build_polar_stats(
            signed_q=sample.signed_q, abs_q=sample.abs_q,
            valid=sample.valid, instance_mask=sample.instance_mask)
        s_rows[index] = s_values
        s_valid[index] = has_valid
        class_ids[index] = int(row["class_id"])
    if class_ids.size and class_ids.min() < 0:
        raise ValueError("unknown class id in the dev pool")
    return c_rows, s_rows, s_valid, class_ids


def write_json(path: Path, payload) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare frozen probe features and polarization "
                    "statistics (read-only; no fitting).",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True,
                        help="protocol.json (or its directory) from "
                        "prepare_existing_data_protocol")
    parser.add_argument("--output-dir", required=True,
                        help="new directory for the feature cache")
    parser.add_argument("--device", default="cpu",
                        help="encoder forward device (default: cpu)")
    args = parser.parse_args(argv)

    if Path(args.output_dir).exists():
        print(
            f"refused: output directory {args.output_dir} already exists; "
            "feature caches are never overwritten",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    started = time.perf_counter()

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        import yaml

        config = yaml.safe_load(
            Path(args.config).read_text(encoding="utf-8"))
        protocol = load_protocol(args.protocol)
        data_config = config["data"]
        fingerprint = data_config["fingerprint"]
        parent = data_config["parent_v4_fingerprint"]
        if protocol["data"]["fingerprint"] != fingerprint:
            raise ValueError(
                "protocol fingerprint "
                f"{protocol['data']['fingerprint']} does not match the "
                f"config {fingerprint}"
            )
        if protocol["config_sha256"] and protocol["config_sha256"] != _config_digest(args.config):
            raise ValueError(
                "the protocol was built from a different experiment config; "
                "re-run prepare_existing_data_protocol first"
            )

        project_root = PROJECT_ROOT
        data_root = project_root / data_config["root"]
        # The project's REAL full admission gate (not the fingerprint
        # identity helper): admit_mask_dataset re-verifies audit_passed,
        # the manifest/summary digests, EVERY referenced NPZ byte digest,
        # decodes every sample to re-check the mask invariants and recompute
        # the mask content digest, re-reads the annotation sources named by
        # the manifest, and finally compares the V5 fingerprint and the V4
        # parent fingerprint. A changed/edited NPZ with a stale audit is
        # refused here, before the encoder is even constructed.
        admission = admit_mask_dataset(
            data_root,
            expected_fingerprint=fingerprint,
            expected_parent_fingerprint=parent,
        )
        log(
            "V5 full admission passed: "
            f"{admission['verified_files']} files re-hashed, masks decoded "
            "and sources re-verified; fingerprint "
            f"{admission['audit_fingerprint'][:16]}... (V4 parent "
            f"{admission['v4_parent_fingerprint'][:16]}...)"
        )

        dev_splits = list(data_config["dev_pool_splits"])
        rows = read_manifest_rows(data_root, dev_splits)
        verify_sample_set(protocol, rows)
        roles = fold_roles(protocol)
        log(f"dev pool: {len(rows)} samples bound to the protocol")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True)

        log("polarization quality statistics...")
        aggregate, per_sample = polarization_quality_statistics(rows, data_root)
        plots = render_quality_plots(aggregate, per_sample, output_dir)
        write_json(output_dir / "polar_quality_summary.json", aggregate)
        with (output_dir / "polar_quality_per_sample.jsonl").open(
                "w", encoding="utf-8") as handle:
            for record in per_sample:
                handle.write(json.dumps(record, sort_keys=True,
                                        allow_nan=False) + "\n")

        log("frozen MiT-B0 feature extraction...")
        budget = BudgetWatch(
            float(config["features"].get("max_runtime_minutes", 30)))
        progress = {"batches": [], "stage": "before encoder construction"}
        try:
            g_features, extraction_report = extract_gray_features(
                rows, data_root, config, args.device, log=log, budget=budget,
                progress=progress)
        except BudgetExceeded as error:
            batches = progress["batches"]
            completed = sum(len(batch) for batch in batches)
            partial = {
                "status": "PARTIAL_BUDGET_EXCEEDED", "reason": str(error),
                "stage": progress["stage"], "completed_samples": completed,
                "total_samples": len(rows),
                "sample_names": [row["sample_name"] for row in rows[:completed]],
                "protocol_sha256": protocol["protocol_sha256"],
                "data_fingerprint": fingerprint,
                "official_mit_b0": config["baselines"]["official_mit_b0"],
                "imgsz": config["features"]["imgsz"],
                "preprocessing_version": config["features"]["preprocessing_version"],
                "encoder_state_check": progress.get("encoder_state_check", "NOT_VERIFIED"),
                "training_status": "NOT_RUN", "trainable_cache": False,
                "elapsed_s": time.perf_counter() - started,
                "partial_features": None, "feature_shape": None,
                "note": "Incomplete diagnostic G rows only; no cache binding and no resume support.",
            }
            if batches:
                partial_g = np.concatenate(batches, axis=0)
                partial["partial_features"] = "partial_gray_features.npz"
                partial["feature_shape"] = list(partial_g.shape)
                np.savez_compressed(output_dir / partial["partial_features"],
                                    g_features=partial_g)
            write_json(output_dir / "partial_extraction.json", partial)
            write_json(output_dir / "report.json", partial)
            log(str(error))
            return 2

        c_rows, s_rows, s_valid, class_ids = build_slot_features(
            rows, data_root)
        manifest_records = []
        for index, row in enumerate(rows):
            manifest_records.append({
                "sample_name": row["sample_name"],
                "npz_path": row["npz_path"],
                "split": row["split"],
                "group": row["group_name"],
                "source_frame": row["source_frame"],
                "object_index": int(row["object_index"]),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "physical_object_id": "unknown",
                "has_valid_s": bool(s_valid[index]),
                "fold_roles": roles[row["sample_name"]],
            })

        if len(manifest_records) != len(rows):
            raise ValueError("feature/manifest row count mismatch")

        np.savez_compressed(
            output_dir / "feature_cache.npz",
            g_features=g_features,
            c_features=c_rows,
            s_features=s_rows,
            s_valid=s_valid,
            class_id=class_ids,
        )
        definition = {
            "feature_definition_version": FEATURE_DEFINITION_VERSION,
            "data_root": str(data_root),
            "data_fingerprint": fingerprint,
            "protocol_sha256": protocol["protocol_sha256"],
            "class_names": list(data_config["class_names"]),
            "data_admission": {
                "gate": "core.material_mask_dataset.admit_mask_dataset",
                "checks": [
                    "audit_passed and the required V5 audit fields",
                    "manifest/summary digests re-verified",
                    "every referenced NPZ byte digest re-verified",
                    "every sample decoded; mask invariants and the mask "
                    "content digest recomputed",
                    "annotation sources re-read and re-hashed",
                    "V5 fingerprint and V4 parent fingerprint compared",
                ],
                "verified_files": admission["verified_files"],
                "audit_fingerprint": admission["audit_fingerprint"],
                "v4_parent_fingerprint": admission["v4_parent_fingerprint"],
            },
            "official_mit_b0": {
                "path": config["baselines"]["official_mit_b0"]["path"],
                "sha256": config["baselines"]["official_mit_b0"]["sha256"],
            },
            "preprocessing_version": config["features"]["preprocessing_version"],
            "imgsz": int(config["features"]["imgsz"]),
            "gray_input_mode": config["features"]["gray_input_mode"],
            "encoder": config["features"]["encoder"],
            "pooling": config["features"]["pooling"],
            "pooling_background_note": config["features"]["pooling_background_note"],
            "slot_G": {"dim": int(g_features.shape[1]),
                       "definition": "MiT-B0 last stage, mask-weighted mean"},
            "slot_C": {"dim": len(CONTROL_FEATURE_KEYS),
                       "keys": list(CONTROL_FEATURE_KEYS),
                       "leakage_rule": "quality[3] (mean_abs_q) never read"},
            "slot_S": {"dim": len(POLAR_STAT_KEYS),
                       "keys": list(POLAR_STAT_KEYS),
                       "region": "instance_mask & polar_valid of the original crop",
                       "quantile_method": QUANTILE_METHOD,
                       "missing_rule": "has_valid_s=false, values NaN in npz / null in json"},
            "cache_store": "raw (no standardization, no PCA, nothing fitted)",
            "interior_definition": INTERIOR_DEFINITION,
            "band_definition": BAND_DEFINITION,
            "extraction": extraction_report,
        }
        write_json(output_dir / "feature_definition.json", definition)
        write_json(output_dir / "feature_manifest.json", {
            "samples": manifest_records,
            "count": len(manifest_records),
        })
        binding = build_cache_binding(output_dir, protocol)
        report = {
            "status": "COMPLETED",
            "samples": len(rows),
            "g_dim": int(g_features.shape[1]),
            "has_valid_s": int(s_valid.sum()),
            "missing_s": int((~s_valid).sum()),
            "plots": plots,
            "cache_binding": {
                "binding_version": binding["binding_version"],
                "files": {name: record["sha256"]
                          for name, record in binding["files"].items()},
            },
            "elapsed_s": time.perf_counter() - started,
            "training_status": "NOT_RUN",
            "note": (
                "raw feature cache only; no model was fitted here. The "
                "binding record pins the byte digests of the three cache "
                "files for the training entry's admission"),
        }
        write_json(output_dir / "report.json", report)
        log(f"feature cache written: {output_dir}")
        return EXIT_OK
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


def _config_digest(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(
        Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
