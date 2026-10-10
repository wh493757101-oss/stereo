"""Build the frozen existing-data protocol (Part 1 of material_existing_data_v1).

Read-only over the datasets: verifies the pinned baseline identities, runs
the existing V5 admission gate, inventories the old train+val dev pool
(test excluded), assigns whole capture groups to 3 outer folds with
inner_train/inner_val parts, and writes protocol.json / folds.json /
baseline_identities.json / group_audit.json / member manifests / README.

Nothing here trains, copies or modifies dataset files. Exit codes:
0 = protocol built and folds OK; 3 = protocol built but the fold
assignment is BLOCKED (conflicts recorded; fitting admission refuses);
1 = refused (identity, admission, input or output error).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.existing_data_protocol import (  # noqa: E402
    CLASS_IDS,
    FOLD_STATUS_BLOCKED,
    FOLD_STATUS_OK,
    build_dev_inventory,
    build_folds,
    build_protocol,
    group_statistics,
    load_experiment_config,
    read_manifest_rows,
    read_split_rows,
    verify_baseline_identities,
)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_BLOCKED = 3


def write_json(path: Path, payload) -> Path:
    """JSON dump that refuses NaN/Inf instead of writing invalid JSON."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(
            f"output directory {path} already exists; protocol reports are "
            "never overwritten — pass a fresh --output-dir"
        )
    path.mkdir(parents=True)
    return path


def fill_fold_members(folds: dict, inventory: dict) -> None:
    """Attach the concrete frame/sample members to every fold level.

    Members reference the original dataset files (npz_path, frame stems);
    nothing is copied. Frames and samples stay listed even when they have
    no GT or no NPZ counterpart.
    """
    frames_by_group: dict[str, list[dict]] = {}
    for frame in inventory["frames"]:
        frames_by_group.setdefault(frame["group"], []).append(frame)
    samples_by_group: dict[str, list[dict]] = {}
    for sample in inventory["samples"]:
        samples_by_group.setdefault(sample["group"], []).append(sample)

    sample_fields = (
        "sample_name", "npz_path", "split", "group", "source_frame",
        "object_index", "class_id", "class_name", "physical_object_id",
    )
    for fold in folds["folds"]:
        for part in ("outer_eval", "inner_train", "inner_val"):
            groups = fold[f"{part}_groups"]
            part_frames = [
                {
                    "frame_id": frame["frame_id"],
                    "split": frame["split"],
                    "group": frame["group"],
                    "block_id": frame["block_id"],
                    "gt_instances": frame["gt_instances"],
                    "gt_class_counts": frame["gt_class_counts"],
                }
                for group in groups for frame in frames_by_group.get(group, [])
            ]
            part_samples = [
                {field: sample[field] for field in sample_fields}
                for group in groups
                for sample in samples_by_group.get(group, [])
            ]
            class_counts = {str(c): 0 for c in CLASS_IDS}
            for frame in part_frames:
                for class_text, count in frame["gt_class_counts"].items():
                    class_counts[class_text] = class_counts.get(class_text, 0) + count
            fold[f"{part}_members"] = {
                "groups": groups,
                "frames": part_frames,
                "samples": part_samples,
                "gt_instances_by_class": class_counts,
                "frames_without_gt": [
                    frame["frame_id"] for frame in part_frames
                    if frame["gt_instances"] == 0
                ],
            }


def render_readme(protocol: dict) -> str:
    folds = protocol["folds"]
    inventory = protocol["inventory"]["summary"]
    lines = [
        "# Existing-data protocol (material_existing_data_v1)",
        "",
        "Frozen grouping protocol over the OLD train+val pool. The old test",
        "split is excluded from every dev use in this round.",
        "",
        f"- dev pool: {inventory['frames']} frames / {inventory['samples']} V5 samples,"
        f" {inventory['gt_instances']} GT instances",
        f"- empty-GT frames kept: {len(inventory['empty_gt_frames'])};"
        f" GT without NPZ: {inventory['gt_without_npz_count']};"
        f" NPZ without GT source: {inventory['npz_without_gt_count']}",
        f"- fold assignment status: **{folds['status']}**"
        f" ({len(folds['folds'])} outer folds)",
        "",
        "## What this proves — and what it does not",
        "",
        "- It proves *training-level capture-group isolation* for anything",
        "  trained with these folds (inner_val selects the epoch,",
        "  outer_eval is touched once after the checkpoint is frozen).",
        "- The original val part already took part in project development,",
        "  so no fold here is a new blind test, and nothing evaluated with",
        "  weights trained on the old train/val is OOF evidence.",
        "- Physical-object identity is unknown (no source records it);",
        "  file names are never used as identity evidence.",
        "",
        "## Files",
        "",
        "- `protocol.json` — full protocol (data fingerprint, admission,",
        "  identities, inventory, folds, boundaries).",
        "- `folds.json` — per-fold group/frame/sample members.",
        "- `baseline_identities.json` — SHA256-verified baseline paths.",
        "- `group_audit.json` — group crossing / duplicate membership audit.",
        "",
        "The fitting entry (`scripts/train_material_probe.py`) refuses to",
        "fit unless this protocol exists and its fold status is OK.",
    ]
    if folds["status"] == FOLD_STATUS_BLOCKED:
        lines += [
            "",
            "## BLOCKED",
            "",
            "The group distribution could not form the required folds:",
            "",
        ]
        lines += [f"- {conflict}" for conflict in folds["conflicts"]]
        lines += [
            "",
            "Fitting is refused. Diagnostics that do not depend on the fold",
            "assignment may still run.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen existing-data protocol (read-only).",
    )
    parser.add_argument("--config", required=True,
                        help="experiment YAML (material_existing_data_v1)")
    parser.add_argument("--output-dir", required=True,
                        help="new directory for the protocol report")
    parser.add_argument("--project-root", default=None,
                        help="project root for resolving config paths "
                        "(default: the repository containing this script)")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root) if args.project_root else PROJECT_ROOT

    if Path(args.output_dir).exists():
        print(
            f"refused: output directory {args.output_dir} already exists; "
            "protocol reports are never overwritten — pass a fresh "
            "--output-dir",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    try:
        config = load_experiment_config(args.config)
        baselines = verify_baseline_identities(config, project_root)

        data_config = config["data"]
        data_root = project_root / data_config["root"]
        fingerprint = data_config["fingerprint"]
        parent = data_config["parent_v4_fingerprint"]

        from core.material_mask_dataset import (
            admit_dataset_for_protocol,
            admit_mask_dataset,
        )

        print(f"admitting {data_root} through the existing V5 gate...",
              file=sys.stderr)
        admission = admit_dataset_for_protocol(
            data_root,
            pooling_mode="instance_mask",
            expected_fingerprint=fingerprint,
            expected_parent_fingerprint=parent,
            expected_mask_schema=data_config.get("mask_schema", ""),
            expected_mask_resize_version=data_config.get("mask_resize_version", ""),
        )
        if admission["audit_fingerprint"] != fingerprint:
            raise ValueError(
                f"V5 fingerprint {admission['audit_fingerprint']} does not "
                f"match the configured {fingerprint}"
            )

        dev_splits = list(data_config["dev_pool_splits"])
        split_rows = read_split_rows(
            project_root / data_config["split_assignment"], dev_splits)
        manifest_rows = read_manifest_rows(data_root, dev_splits)
        inventory = build_dev_inventory(
            split_rows=split_rows,
            manifest_rows=manifest_rows,
            label_root=project_root / data_config["label_root"],
            data_root=data_root,
            project_root=project_root,
            class_names=tuple(data_config["class_names"]),
        )

        protocol_config = config["protocol"]
        group_stats = group_statistics(inventory)
        folds = build_folds(
            group_stats,
            n_folds=int(protocol_config["n_outer_folds"]),
            seed=int(protocol_config["seed"]),
            fraction=float(protocol_config.get("inner_val_fraction", 0.25)),
            inner_seed_base=int(protocol_config.get("inner_seed_base", 2026)),
            require_outer_eval_all_classes=bool(
                protocol_config.get("require_outer_eval_all_classes", True)),
            require_inner_train_all_classes=bool(
                protocol_config.get("require_inner_train_all_classes", True)),
            require_inner_val_all_classes=bool(
                protocol_config.get("require_inner_val_all_classes", False)),
        )
        fill_fold_members(folds, inventory)

        protocol = build_protocol(
            config=config,
            config_path=args.config,
            admission=admission,
            baselines=baselines,
            inventory=inventory,
            folds=folds,
            project_root=project_root,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED

    # The output directory is created only after every check passed, so a
    # refusal never leaves a half-built protocol directory behind.
    try:
        output_dir = prepare_output_dir(args.output_dir)
    except FileExistsError as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED

    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "folds.json", folds)
    write_json(output_dir / "baseline_identities.json", baselines)
    write_json(
        output_dir / "group_audit.json",
        {
            "fold_audit": protocol["fold_audit"],
            "group_statistics": protocol["group_statistics"],
            "conflicts": folds["conflicts"],
        },
    )
    (output_dir / "README.md").write_bytes(
        render_readme(protocol).encode("utf-8"))

    print(f"protocol written: {output_dir / 'protocol.json'}")
    print(f"fold status: {folds['status']}")
    if folds["status"] == FOLD_STATUS_BLOCKED:
        for conflict in folds["conflicts"]:
            print(f"  conflict: {conflict}", file=sys.stderr)
        return EXIT_BLOCKED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
