"""Linear probe training entry for the P0/P1/P2 material probe.

The USER runs this script manually; the agent never executes ``--fit``
against real data. Two explicit modes exist and nothing happens without
one of them:

- ``--dry-run`` — source admission, parameter checks and a task preview
  only. No model is built, no checkpoint is written; the report records
  ``training_status=NOT_RUN`` and ``classification_gain=NOT_MEASURED``.
- ``--fit`` — the real fit over the requested folds and variants.

Training contract (frozen in the experiment config, never auto-tuned):

1. the encoder is NOT loaded here — features come from the cache;
2. a linear 4-class softmax head only;
3. every variant uses the same [G, C, S] input slots; unused slots are
   exactly zero, so head dimension, initialization and budget are shared;
4. the standardizer is fitted on the fold's inner_train only (S columns
   from inner_train samples with a valid S; missing S -> zero after
   standardization; zero-variance/empty columns -> zero);
5. inner_val selects the checkpoint; outer_eval is evaluated ONCE after
   the checkpoint is frozen; no refit on the full data;
6. seed/optimizer/budget come from the protocol-bound config;
7. per fold/variant the model, scaler, config, history and per-sample
   outer-eval logits/predictions are saved; OOF, per-group and two key
   class-pair confusions are summarized, with P2-P1 / P2-P0 paired
   changes;
8. outer_eval never influences epoch/parameter/seed selection and no
   fold is cherry-picked;
9. an existing output directory, any cache/protocol/weight mismatch and
   unknown classes are refused BEFORE fitting;
10. without ``--fit`` nothing reports a trained model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_probe import (  # noqa: E402
    CACHE_BINDING_FILENAME,
    CONTROL_FEATURE_KEYS,
    FEATURE_DEFINITION_VERSION,
    NUM_CLASSES,
    POLAR_STAT_KEYS,
    Standardizer,
    accuracy,
    assemble_design_matrix,
    authoritative_sample_index,
    build_probe_head,
    compare_module_states,
    macro_f1,
    paired_changes,
    slot_dimensions,
    snapshot_module_state,
    train_linear_probe,
    verify_cache_arrays,
    verify_cache_binding,
    verify_cache_manifest,
)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_BLOCKED = 3

FOLD_STATUSES = ("OK", "BLOCKED")
KEY_CLASS_PAIRS = (("plastic_fish", "real_fish"),
                   ("metal_submarine", "plastic_submarine"))


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_protocol(path: str | Path) -> dict:
    protocol_path = Path(path)
    if protocol_path.is_dir():
        protocol_path = protocol_path / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"protocol not found: {protocol_path}")
    protocol = load_json(protocol_path)
    from core.existing_data_protocol import protocol_digest

    recorded = protocol.get("protocol_sha256", "")
    if not recorded or protocol_digest(protocol) != recorded:
        raise ValueError(
            f"protocol {protocol_path} failed its own digest check"
        )
    return protocol


def verify_feature_cache(
    feature_dir: str | Path, protocol: dict
) -> tuple[dict, dict, dict, dict]:
    """Full cache admission: binding, identities, arrays — before any fit.

    Chain (each step refuses before the next runs):
    1. ``verify_cache_binding`` — the saved binding record is the EXPECTED
       state; the actual bytes of the three cache files are re-hashed
       against it (never against a freshly computed "expected" value);
    2. ``authoritative_sample_index`` — the per-sample identity and
       per-fold role map derived from the protocol, validated fold by fold;
    3. ``verify_cache_manifest`` — row-by-row identity/role agreement with
       the protocol, including the frozen row order;
    4. ``verify_cache_arrays`` — keys, shapes, dtypes, class rows, finite
       values and the explicit missing-S rule.

    The cache's own fold_roles are never an independent split source: they
    are only accepted after the item-by-item comparison with the protocol.
    """
    feature_dir = Path(feature_dir)
    binding = verify_cache_binding(feature_dir, protocol)
    definition = load_json(feature_dir / "feature_definition.json")
    manifest = load_json(feature_dir / "feature_manifest.json")
    if definition.get("feature_definition_version") !=             FEATURE_DEFINITION_VERSION:
        raise ValueError(
            f"cache feature version "
            f"{definition.get('feature_definition_version')!r} != "
            f"{FEATURE_DEFINITION_VERSION!r}"
        )
    if definition.get("data_fingerprint") != protocol["data"]["fingerprint"]:
        raise ValueError(
            "cache data fingerprint does not match the protocol; refusing "
            "to fit on a different dataset state"
        )
    if definition.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError(
            "cache protocol digest does not match this protocol; the cache "
            "was built for a different grouping"
        )
    authority = authoritative_sample_index(protocol)
    samples = verify_cache_manifest(manifest, authority, protocol)
    cache = dict(np.load(feature_dir / "feature_cache.npz"))
    verify_cache_arrays(cache, samples, definition, protocol)
    return definition, manifest, cache, binding


def fold_sample_indices(
    manifest: dict, fold_id: str
) -> dict[str, list[int]]:
    """inner_train / inner_val / outer_eval row indices for one fold."""
    parts: dict[str, list[int]] = {"inner_train": [], "inner_val": [],
                                   "outer_eval": []}
    for index, sample in enumerate(manifest["samples"]):
        role = sample["fold_roles"].get(fold_id)
        if role is None:
            raise ValueError(
                f"sample {sample['sample_name']} has no {fold_id} role in "
                "the cache manifest"
            )
        parts[role].append(index)
    for role, indices in parts.items():
        if not indices:
            raise ValueError(f"fold {fold_id}: {role} part is empty")
    return parts


def resolve_budget(protocol: dict) -> dict:
    """The frozen probe budget from the protocol-bound config."""
    probe = protocol.get("config", {}).get("probe")
    if not isinstance(probe, dict):
        raise ValueError(
            "the protocol-bound config carries no probe section; the "
            "budget cannot be resolved"
        )
    variants = probe.get("variants")
    if not isinstance(variants, dict) or set(variants) != {"P0", "P1", "P2"}:
        raise ValueError(
            "the probe section must define exactly the variants P0/P1/P2"
        )
    return {
        "seed": int(probe["seed"]),
        "optimizer": str(probe["optimizer"]),
        "lr": float(probe["lr"]),
        "weight_decay": float(probe["weight_decay"]),
        "max_epochs": int(probe["max_epochs"]),
        "device": str(probe.get("device", "cpu")),
        "head": str(probe.get("head", "linear_softmax_4class")),
        "standardization": str(probe.get("standardization")),
        "selection": probe.get("selection", {}),
    }


def build_matrices(
    cache: dict, indices: list[int], variant: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw [G, C, S] slot values + the full-width per-column validity map.

    Only the S columns carry a validity flag (the sample's ``has_valid_s``);
    G and C are always valid.
    """
    g = cache["g_features"][indices]
    c = cache["c_features"][indices]
    s = cache["s_features"][indices]
    s_valid = cache["s_valid"][indices].astype(bool)
    if s.shape[1] != len(POLAR_STAT_KEYS):
        raise ValueError("cache S slot width does not match the definition")
    column_valid = np.ones((len(indices), g.shape[1] + c.shape[1] + s.shape[1]),
                           dtype=bool)
    column_valid[:, g.shape[1] + c.shape[1]:] = s_valid[:, None]
    return g, c, s, column_valid


def standardize_and_assemble(
    cache: dict, train_idx: list[int], eval_indices_lists: list[list[int]],
    variant: str,
) -> tuple[Standardizer, list[np.ndarray]]:
    """Fit the scaler on inner_train only, then transform every part."""
    g_train, c_train, s_train, valid_train = build_matrices(
        cache, train_idx, variant)
    raw_train = np.concatenate([g_train, c_train, s_train], axis=1)
    scaler = Standardizer.fit(raw_train, valid_train)
    outputs = []
    for indices in eval_indices_lists:
        g, c, s, valid = build_matrices(cache, indices, variant)
        raw = np.concatenate([g, c, s], axis=1)
        standardized = scaler.transform(raw, valid)
        g_dim = g.shape[1]
        g_std = standardized[:, :g_dim]
        c_std = standardized[:, g_dim:g_dim + len(CONTROL_FEATURE_KEYS)]
        s_std = standardized[:, g_dim + len(CONTROL_FEATURE_KEYS):]
        outputs.append(assemble_design_matrix(g_std, c_std, s_std, variant))
    return scaler, outputs


def confusion_pair(predictions, labels, class_names, pair) -> dict:
    """2x2 confusion of a class pair (rows = true pair member)."""
    first, second = pair
    matrix = {
        f"true_{first}_pred_{first}": 0,
        f"true_{first}_pred_{second}": 0,
        f"true_{second}_pred_{first}": 0,
        f"true_{second}_pred_{second}": 0,
    }
    for label, prediction in zip(labels, predictions):
        key = f"true_{class_names[label]}_pred_{class_names[prediction]}"
        if key in matrix:
            matrix[key] += 1
    return matrix


def write_json(path: Path, payload) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Linear P0/P1/P2 probe over the frozen feature cache "
                    "(--dry-run to preview, --fit to train).",
    )
    parser.add_argument("--protocol", required=True,
                        help="protocol.json from prepare_existing_data_protocol")
    parser.add_argument("--features", required=True,
                        help="feature cache directory from prepare_material_probe")
    parser.add_argument("--fold", default="all",
                        help="all | fold0 | fold1 | ...")
    parser.add_argument("--variants", nargs="+", default=["P0", "P1", "P2"],
                        choices=["P0", "P1", "P2"])
    parser.add_argument("--seed", type=int, default=None,
                        help="must equal the protocol seed; overrides nothing")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="admission + preview only; nothing is fitted")
    parser.add_argument("--fit", action="store_true",
                        help="explicitly start the fit (user-run only)")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.fit:
        parser.print_help()
        print(
            "\nrefused: choose an explicit mode — --dry-run (admission and "
            "preview only) or --fit (real training). Nothing runs by default.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if args.dry_run and args.fit:
        print("refused: --dry-run and --fit are mutually exclusive",
              file=sys.stderr)
        return EXIT_REFUSED
    if Path(args.output_dir).exists():
        print(
            f"refused: output directory {args.output_dir} already exists; "
            "probe runs are never overwritten",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    started = time.perf_counter()
    try:
        protocol = load_protocol(args.protocol)
        fold_status = protocol["folds"]["status"]
        if fold_status not in FOLD_STATUSES:
            raise ValueError(f"unknown protocol fold status {fold_status!r}")
        all_folds = [fold["fold_id"]
                     for fold in protocol["folds"]["folds"]]
        if args.fold == "all":
            selected_folds = all_folds
        elif args.fold in all_folds:
            selected_folds = [args.fold]
        else:
            raise ValueError(
                f"fold {args.fold!r} is not in the protocol folds {all_folds}"
            )
        budget = resolve_budget(protocol)
        if args.seed is not None and int(args.seed) != budget["seed"]:
            raise ValueError(
                f"--seed {args.seed} does not match the protocol seed "
                f"{budget['seed']}; the protocol budget is not re-seeded"
            )
        if args.fit and fold_status == "BLOCKED":
            # The protocol's top-level property is checked before the cache
            # binding: a blocked grouping is not fittable, whatever the
            # cache was built for.
            print(
                "refused: the protocol fold assignment is BLOCKED; fitting "
                "requires a valid group-isolated fold assignment (see the "
                "protocol conflicts)",
                file=sys.stderr,
            )
            return EXIT_BLOCKED
        definition, manifest, cache, binding = verify_feature_cache(
            args.features, protocol)
        for requested in args.variants:
            if requested not in {"P0", "P1", "P2"}:
                raise ValueError(f"unknown variant {requested!r}")
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED

    class_names = list(protocol["data"]["class_names"])

    # ---- preview table (identical for both modes) ----
    previews = []
    for fold_id in selected_folds:
        parts = fold_sample_indices(manifest, fold_id)
        previews.append({
            "fold": fold_id,
            "inner_train": len(parts["inner_train"]),
            "inner_val": len(parts["inner_val"]),
            "outer_eval": len(parts["outer_eval"]),
            "variants": list(args.variants),
        })

    # The admission record both modes share: it binds the verified protocol
    # and cache binding into the run so any output can be traced back.
    admission_record = {
        "protocol_sha256": protocol["protocol_sha256"],
        "cache_binding": {
            "binding_version": binding.get("binding_version"),
            "files": {
                name: {"sha256": record["sha256"], "bytes": record["bytes"]}
                for name, record in binding.get("files", {}).items()
            },
            "data_fingerprint": binding.get("data_fingerprint"),
            "official_mit_b0": binding.get("official_mit_b0"),
            "class_names": binding.get("class_names"),
            "binding_file": CACHE_BINDING_FILENAME,
        },
        "class_names": class_names,
        "feature_definition_version":
            definition["feature_definition_version"],
        "imgsz": definition.get("imgsz"),
        "preprocessing_version": definition.get("preprocessing_version"),
        "g_dim": int(cache["g_features"].shape[1]),
        "input_dim": int(cache["g_features"].shape[1]
                         + len(CONTROL_FEATURE_KEYS)
                         + len(POLAR_STAT_KEYS)),
        "folds": selected_folds,
        "variants": list(args.variants),
        "budget": budget,
        "sample_count": len(manifest["samples"]),
    }

    if args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True)
        report = {
            "mode": "dry-run",
            "training_status": "NOT_RUN",
            "classification_gain": "NOT_MEASURED",
            "fold_status": fold_status,
            "admission": admission_record,
            "preview": previews,
            "note": (
                "no model was built, no checkpoint was written; a BLOCKED "
                "protocol refuses --fit but a dry-run preview is allowed. "
                "A passing preview alone is NOT the fit command: run the "
                "fit command from manual_commands.ps1 afterwards."
            ),
        }
        write_json(output_dir / "report.json", report)
        write_json(output_dir / "admission.json", admission_record)
        print(f"dry-run preview written: {output_dir / 'report.json'}")
        for entry in previews:
            print(f"  {entry['fold']}: train {entry['inner_train']} / "
                  f"val {entry['inner_val']} / eval {entry['outer_eval']}")
        return EXIT_OK

    # ---- fit mode ---- (the BLOCKED refusal already happened above,
    # before the cache binding; fold_status is the protocol's top-level
    # property)
    try:
        import torch

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True)
        write_json(output_dir / "admission.json", admission_record)
        device = "cpu"  # cached features; deterministic CPU full-batch
        per_variant_records: dict[str, list[dict]] = {
            variant: [] for variant in args.variants}
        outer_eval_labels: list[int] = []
        outer_eval_groups: list[str] = []
        outer_eval_names: list[str] = []

        for fold_id in selected_folds:
            parts = fold_sample_indices(manifest, fold_id)
            labels = cache["class_id"]
            for role in ("inner_train", "inner_val"):
                if len(set(labels[parts[role]].tolist())) < NUM_CLASSES:
                    raise ValueError(
                        f"fold {fold_id}: {role} does not cover all four "
                        "classes; the protocol validity check should have "
                        "caught this"
                    )
            y_train = labels[parts["inner_train"]].astype(np.int64)
            y_val = labels[parts["inner_val"]].astype(np.int64)
            y_eval = labels[parts["outer_eval"]].astype(np.int64)
            eval_rows = [parts["outer_eval"]]
            scalers: dict[str, Standardizer] = {}
            design: dict[str, list[np.ndarray]] = {}
            for variant in args.variants:
                scaler, outputs = standardize_and_assemble(
                    cache, parts["inner_train"],
                    [parts["inner_train"], parts["inner_val"],
                     parts["outer_eval"]],
                    variant,
                )
                scalers[variant] = scaler
                design[variant] = outputs  # train / val / eval
            for variant in args.variants:
                fold_variant_dir = output_dir / "models" / f"{fold_id}_{variant}"
                fold_variant_dir.mkdir(parents=True)
                input_dim = design[variant][0].shape[1]
                head_before = snapshot_module_state(
                    build_probe_head(input_dim, seed=budget["seed"]))
                model, history, best_state, best = train_linear_probe(
                    design[variant][0], y_train, design[variant][1], y_val,
                    seed=budget["seed"], lr=budget["lr"],
                    weight_decay=budget["weight_decay"],
                    max_epochs=budget["max_epochs"],
                )
                head_after = snapshot_module_state(model)
                head_check = compare_module_states(head_before, head_after)
                if head_check["unchanged"]:
                    raise ValueError(
                        "the probe head did not change during training; "
                        "refusing to report a fit that never happened"
                    )
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    eval_logits = model(
                        torch.from_numpy(design[variant][2])).numpy()
                eval_pred = eval_logits.argmax(axis=1)
                write_json(fold_variant_dir / "scaler.json",
                           scalers[variant].to_dict())
                write_json(fold_variant_dir / "history.json",
                           {"selection": best, "epochs": history})
                torch.save(
                    {
                        "state_dict": best_state,
                        "input_dim": int(input_dim),
                        "num_classes": NUM_CLASSES,
                        "variant": variant,
                        "fold": fold_id,
                        "seed": budget["seed"],
                        "budget": budget,
                    },
                    fold_variant_dir / "model_state.pt",
                )
                with (fold_variant_dir / "outer_eval_predictions.jsonl").open(
                        "w", encoding="utf-8") as handle:
                    for row_index, name_index in enumerate(parts["outer_eval"]):
                        sample = manifest["samples"][name_index]
                        handle.write(json.dumps({
                            "sample_name": sample["sample_name"],
                            "group": sample["group"],
                            "class_id": int(y_eval[row_index]),
                            "prediction": int(eval_pred[row_index]),
                            "logits": [float(v) for v in eval_logits[row_index]],
                        }, sort_keys=True) + "\n")
                record = {
                    "fold": fold_id,
                    "variant": variant,
                    "selected_epoch": best["epoch"],
                    "inner_val_macro_f1": best["inner_val_macro_f1"],
                    "inner_val_accuracy": best["inner_val_accuracy"],
                    "outer_eval_macro_f1": macro_f1(y_eval, eval_pred),
                    "outer_eval_accuracy": accuracy(y_eval, eval_pred),
                    "support": len(y_eval),
                }
                write_json(fold_variant_dir / "fold_variant_report.json",
                           record)
                per_variant_records[variant].append(record)
                if variant == args.variants[0]:
                    # the pooled outer-eval labels/groups/names: collected
                    # once per fold (on the first variant of that fold)
                    outer_eval_labels.extend(int(v) for v in y_eval)
                    outer_eval_groups.extend(
                        manifest["samples"][i]["group"]
                        for i in parts["outer_eval"])
                    outer_eval_names.extend(
                        manifest["samples"][i]["sample_name"]
                        for i in parts["outer_eval"])
                print(f"  {fold_id}/{variant}: epoch {best['epoch']} "
                      f"(val F1 {best['inner_val_macro_f1']:.4f}) -> "
                      f"outer F1 {record['outer_eval_macro_f1']:.4f}")

        # ---- OOF summary over the completed folds (per variant) ----
        oof = {}
        for variant in args.variants:
            oof[variant] = {
                "folds": per_variant_records[variant],
                "macro_f1_mean": float(np.mean([
                    r["outer_eval_macro_f1"]
                    for r in per_variant_records[variant]])),
                "accuracy_mean": float(np.mean([
                    r["outer_eval_accuracy"]
                    for r in per_variant_records[variant]])),
            }

        # ---- paired deltas on the pooled outer-eval predictions ----
        pooled_preds: dict[str, list[int]] = {v: [] for v in args.variants}
        for variant in args.variants:
            for fold_id in selected_folds:
                path = (output_dir / "models" / f"{fold_id}_{variant}"
                        / "outer_eval_predictions.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines():
                    pooled_preds[variant].append(
                        int(json.loads(line)["prediction"]))
        paired = {}
        complete_pairing = (
            {"P0", "P1", "P2"} <= set(pooled_preds)
            and all(len(pooled_preds[v]) == len(outer_eval_labels)
                    for v in ("P0", "P1", "P2"))
            and bool(outer_eval_labels)
        )
        if complete_pairing:
            for base, other in (("P0", "P2"), ("P1", "P2")):
                changes = paired_changes(
                    [p == t for p, t in zip(pooled_preds[base],
                                            outer_eval_labels)],
                    [p == t for p, t in zip(pooled_preds[other],
                                            outer_eval_labels)],
                )
                paired[f"{other}_minus_{base}"] = {
                    **changes,
                    "pooled_macro_f1": {
                        base: macro_f1(outer_eval_labels, pooled_preds[base]),
                        other: macro_f1(outer_eval_labels, pooled_preds[other]),
                    },
                }

        per_group: dict[str, dict] = {}
        by_group_indices: dict[str, list[int]] = {}
        for index, group in enumerate(outer_eval_groups):
            by_group_indices.setdefault(group, []).append(index)
        for group, indices in sorted(by_group_indices.items()):
            per_group[group] = {
                variant: {
                    "macro_f1": macro_f1(
                        [outer_eval_labels[i] for i in indices],
                        [pooled_preds[variant][i] for i in indices]),
                    "accuracy": accuracy(
                        [outer_eval_labels[i] for i in indices],
                        [pooled_preds[variant][i] for i in indices]),
                }
                for variant in args.variants
            }
        class_id_to_name = {index: name
                            for index, name in enumerate(class_names)}
        key_pairs = {
            f"{pair[0]}_vs_{pair[1]}": {
                variant: confusion_pair(
                    pooled_preds[variant],
                    outer_eval_labels, class_id_to_name, pair)
                for variant in args.variants
            }
            for pair in KEY_CLASS_PAIRS
        }

        summary = {
            "mode": "fit",
            "training_status": "COMPLETED",
            "fold_status": fold_status,
            "admission": admission_record,
            "folds": selected_folds,
            "variants": list(args.variants),
            "budget": budget,
            "preview": previews,
            "oof": oof,
            "paired": paired,
            "per_group": per_group,
            "key_class_pairs": key_pairs,
            "selection_rule": (
                "inner_val macro_f1 > accuracy > earlier epoch; outer_eval "
                "evaluated once after the checkpoint was frozen; no refit "
                "on the full data"),
            "elapsed_s": time.perf_counter() - started,
            "conclusions": {
                "classification_gain": (
                    "measured on the pooled outer_eval of THIS protocol "
                    "only; it is not an independent-generalization claim "
                    "and not a polarization-benefit claim"),
                "independent_generalization": "NOT_CLAIMED",
            },
        }
        write_json(output_dir / "summary.json", summary)
        print(f"fit complete: {output_dir / 'summary.json'}")
        return EXIT_OK
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"failed: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
