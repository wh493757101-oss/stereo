"""material_polar_complementarity_v1 tests.

Unit tests cover the ledger identity refusals, the count semantics
(unique vs per-seed vs instance-model pairs), the changed-set definition,
the deterministic control selection and the empty-S representation. The
entry tests run the REAL diagnostic main() over a small synthetic stereo
world (dataset builds only — no model training anywhere): admission with
pinned hashes and fit-manifest binding expansion, the ledger, control
selection, the selected-frame rebuild checked against a production-built
cache, panels, the report, the shared deadline and the zero-training
call-path probes."""

from __future__ import annotations

import copy
import hashlib
import shutil
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

from core.material_polar_complementarity import (
    ARMS,
    build_conditional_analysis,
    q1_fish_pair_common_support,
    subset_layered_analysis,
    EXPECTED_COUNTS,
    FOLD_IDS,
    MAX_CONTROLS,
    MAX_PANELS,
    TRAIN_SEEDS,
    VARIANTS,
    ComplementarityRefused,
    assign_bins,
    build_ledger,
    conditional_tables,
    derive_counts,
    fold_bin_edges,
    prediction_set_digest,
    select_controls,
    verify_expected_counts,
)
from scripts import diagnose_material_polar_complementarity as diag
from scripts import prepare_material_residual_scale_features as prep
from core.material_residual_scale import CACHE_KIND_FULL
from .conftest import build_synthetic_v4, build_synthetic_v5
from .test_polar_region_quality_entry import (
    _make_cache_files_for_v5,
    _protocol_for_v5,
    _v5_rows,
)
from .test_material_residual_scale_features import (
    SYNTH_MATCHER,
    _build_stereo_source,
)

CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")

_ROOT = prep._PROJECT_ROOT


# ---------------------------------------------------------------------------
# Pure-row helpers for the count/selection unit tests


def _mini_row(name: str, cls: int, r0_pred: int, *, group: str = "g1",
              fold: str = "fold0", margin: float = 1.0,
              path_truth: dict | None = None, s_valid: bool = True,
              s_features: list | None = None) -> dict:
    """A ledger-shaped row. ``path_truth`` maps 'arm|seed|variant' to a
    correct flag; missing keys default to the R0 correctness."""
    r0_correct = r0_pred == cls
    paths = {}
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            for variant in VARIANTS:
                key = f"{arm}|{seed}|{variant}"
                correct = (path_truth or {}).get(key, r0_correct)
                paths[key] = {"logits": [0.1, 0.2, 0.3, 0.4],
                              "prediction": cls if correct
                              else (cls + 1) % 4,
                              "correct": correct}
    row = {
        "sample_name": name, "source_frame": f"{name}_frame",
        "split": "train", "group": group, "class_id": cls,
        "class_name": CLASS_NAMES[cls], "outer_fold": fold, "row": 0,
        "s_valid": s_valid,
        "blocks": {"interior_valid": s_valid, "band_valid": s_valid},
        "c_features": [0.5, 1.0, 1.0, 0.4] + [0.0] * 10,
        "s_features": (s_features if s_features is not None
                       else [0.1 * (i + 1) for i in range(16)]),
        "r0": {"logits": [0.1, 0.2, 0.3, 0.4],
               "prediction": r0_pred, "correct": r0_correct,
               "raw_logit_margin": margin},
        "paths": paths, "rcs_vs_r0": {}, "rcs_vs_rc": {},
    }
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            rcs = row["paths"][f"{arm}|{seed}|RCS"]
            for tag, other in (("rcs_vs_r0", row["r0"]),
                               ("rcs_vs_rc", row["paths"][
                                   f"{arm}|{seed}|RC"])):
                fixed = bool(rcs["correct"]) and not bool(other["correct"])
                broke = bool(other["correct"]) and not bool(rcs["correct"])
                row[tag][f"{arm}|{seed}"] = {
                    "fixed": fixed, "broke": broke,
                    "net": int(fixed) - int(broke)}
    return row


class TestCountSemantics:
    def test_unique_per_seed_and_pair_counts_are_separate(self):
        rows = [
            # corrected by (fixed,2026) and (fixed,2027): one unique
            # instance, two seeds, two pairs
            _mini_row("a", 2, 3, path_truth={
                "fixed|2026|RCS": True, "fixed|2027|RCS": True}),
            # broken by (scaled,2028): one instance, one seed, one pair
            _mini_row("b", 1, 1, path_truth={"scaled|2028|RCS": False}),
            _mini_row("c", 0, 0),
        ]
        stats = derive_counts(rows)
        assert stats["unique_instances"] == 3
        assert stats["corrected_union"] == 1
        assert stats["broken_union"] == 1
        assert stats["changed_union"] == 2
        assert stats["per_seed_corrected"]["2026"] == ["a"]
        assert stats["per_seed_corrected"]["2027"] == ["a"]
        assert stats["per_seed_corrected"]["2028"] == []
        assert stats["instance_model_pairs_evaluated"] == 3 * 12
        assert stats["corrected_pairs"] == 2
        assert stats["broken_pairs"] == 1

    def test_changed_set_is_correctness_only(self):
        # wrong -> another wrong class never enters; RC-only changes and
        # logit-only changes do not enter
        rows = [
            _mini_row("w1", 3, 2, path_truth={
                "fixed|2026|RCS": False, "fixed|2027|RCS": False,
                "fixed|2028|RCS": False, "scaled|2026|RCS": False,
                "scaled|2027|RCS": False, "scaled|2028|RCS": False}),
            _mini_row("w2", 1, 2, path_truth={"fixed|2026|RC": True}),
            _mini_row("w3", 2, 2),
        ]
        stats = derive_counts(rows)
        assert stats["changed_union"] == 0
        assert stats["corrected_union"] == 0
        assert stats["broken_union"] == 0

    def test_expected_count_checker_reports_every_mismatch(self):
        stats = derive_counts([_mini_row("a", 2, 3, path_truth={
            "fixed|2026|RCS": True})])
        problems = verify_expected_counts(stats)
        assert any("unique_instances" in p for p in problems)
        assert any("r0_errors" in p for p in problems)
        assert any("changed_union" in p for p in problems)
        assert any("confusion" in p for p in problems)


class TestControlSelection:
    def test_deterministic_median_nearest_with_name_tiebreak(self):
        rows = [
            _mini_row(f"e{i}", 1, 2, group="g1", margin=float(m),
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]})
            for i, m in enumerate((0.5, 1.0, 3.0, 2.0))]
        stats = derive_counts(rows)
        panels, notes = select_controls(rows, stats)
        error_panels = [p for p in panels if p["role"] == "error_control"]
        assert error_panels, "an error control is selected"
        # margins [0.5, 1, 2, 3]: median 1.5 -> nearest are 1.0 and 2.0;
        # the smaller sample_name wins deterministically
        by_dir = {p["direction"]: p["sample_name"] for p in error_panels}
        assert set(by_dir) == {"plastic_submarine->plastic_fish"}
        assert by_dir["plastic_submarine->plastic_fish"] == "e1"

    def test_dedup_shortage_and_cap(self):
        rows = [
            _mini_row("e1", 1, 2, group="g1", margin=1.0,
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
            _mini_row("e2", 0, 1, group="g2", margin=1.0,
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
            _mini_row("ok1", 1, 1, group="g1"),
        ]
        stats = derive_counts(rows)
        panels, notes = select_controls(rows, stats)
        names = [p["sample_name"] for p in panels]
        assert len(names) == len(set(names)), "no instance is reused"
        assert "e1" in names or any(
            p["role"] == "error_control" for p in panels)
        assert all(n not in stats["changed_names"] for n in names)
        # the 2->1 direction has no correct-control candidate in g2
        assert any("no eligible" in n for n in notes)

    def test_cap_is_enforced(self):
        assert MAX_CONTROLS <= 12 and MAX_PANELS == 41

    def test_panel_ids_are_filesystem_safe(self):
        rows = [
            _mini_row("e1", 1, 2, group="g1", margin=1.0,
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
            _mini_row("ok1", 1, 1, group="g1"),
        ]
        stats = derive_counts(rows)
        panels, _notes = select_controls(rows, stats)
        assert panels
        illegal = set('<>:"/\\|?*')
        for panel in panels:
            assert not (set(panel["panel_id"]) & illegal), \
                f"{panel['panel_id']} is not a safe filename base"


class TestEmptySRepresentation:
    def test_missing_blocks_are_none_and_flagged(self):
        row = _mini_row(
            "m1", 2, 2, s_valid=False,
            s_features=[None] * 8 + [0.0] * 8,
            )
        assert row["s_valid"] is False
        assert row["blocks"] == {"interior_valid": False,
                                 "band_valid": False}
        tables = conditional_tables([row], derive_counts([row]),
                                    {"m1": 4})
        subset = tables["groups"]["plastic_fish_vs_real_fish"][
            "subsets_overview"]["all_instances_of_the_pair"]
        assert subset["n_missing_s"] == 1
        # an all-missing interior block reports n_valid=0, never a
        # fabricated zero median
        assert subset["interior"]["signed_q_mean"]["n_valid"] == 0
        assert subset["interior"]["signed_q_mean"]["median"] is None

    def test_missing_rows_are_never_silent_in_ledger_counts(self):
        rows = [_mini_row("a", 2, 3, s_valid=False),
                _mini_row("b", 2, 2)]
        stats = derive_counts(rows)
        assert stats["r0_errors"] == 1
        assert stats["r0_errors_with_s"] == 0


# ---------------------------------------------------------------------------
# The synthetic world (dataset builds only; NO model training)


def _dump_cfg(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=True),
                    encoding="utf-8")
    return path


def _fabricate_predictions(cache, authority, fit_dir: Path) -> dict:
    """39 prediction files with crafted outcomes.

    Five REQUIRED scenarios, each a DIFFERENT instance (picked one at a
    time, the taken-set updated immediately):

    - c1: R0 wrong; every RC wrong; only fixed/2026 RCS correct;
    - c2: R0 wrong; every RC wrong; all six RCS correct;
    - b1: R0 correct; every RC correct; only scaled/2027 RCS wrong;
    - w1: R0 wrong; RC and RCS ALWAYS wrong, the wrong class rotating
      over the three non-true classes (never the true one);
    - rc1: R0 correct; every RCS correct; fixed/2026 RC wrong (the
      RC-only control marker).

    Every other instance is the plain all-correct case. A missing pick is
    a fixture failure, never a silent drop.
    """
    manifest = cache["manifest"]
    s_valid = cache["arrays"]["s_valid"]
    class_of = {sample["sample_name"]: int(sample["class_id"])
                for sample in manifest}
    fold_of = {}
    for sample in manifest:
        outer = [f for f, role in sample["fold_roles"].items()
                 if role == "outer_eval"]
        fold_of[sample["sample_name"]] = outer[0]
    names = sorted(class_of)
    group_of = {sample["sample_name"]: sample["group"]
                for sample in manifest}

    # one pick at a time; taken is updated immediately so no two
    # scenarios can ever share an instance
    taken: set[str] = set()

    def _pick(fold: str) -> str:
        name = next(
            (n for n in names if fold_of[n] == fold and n not in taken
             and bool(s_valid[cache["row_of"][n]])), None)
        if name is None:
            raise AssertionError(
                f"the synthetic world has no free with-S instance in "
                f"{fold}; the five scenarios cannot be built")
        taken.add(name)
        return name

    picks = {
        "c1": _pick("fold0"),
        "c2": _pick("fold1"),
        "b1": _pick("fold1"),
        "w1": _pick("fold2"),
        "rc1": _pick("fold0"),
    }
    scenario_of = {name: scenario for scenario, name in picks.items()}

    def logits_for(pred: int) -> list:
        values = [-1.0, -1.0, -1.0, -1.0]
        values[pred] = 3.0
        return values

    def r0_prediction_for(name: str) -> int:
        cls = class_of[name]
        if scenario_of.get(name) in ("c1", "c2", "w1"):
            return (cls + 1) % 4
        return cls

    def prediction_for(name: str, arm: str, seed: int, variant: str) -> int:
        scenario = scenario_of.get(name)
        cls = class_of[name]
        if scenario == "c1":
            correct = (arm, seed) == ("fixed", 2026) and variant == "RCS"
            return cls if correct else (cls + 1) % 4
        if scenario == "c2":
            correct = variant == "RCS"
            return cls if correct else (cls + 1) % 4
        if scenario == "b1":
            wrong = (arm, seed) == ("scaled", 2027) and variant == "RCS"
            return (cls + 1) % 4 if wrong else cls
        if scenario == "w1":
            # always wrong; the wrong class rotates over the THREE
            # non-true classes and never lands on the true one
            wrong_choices = [c for c in range(4) if c != cls]
            offset = (ARMS.index(arm) * len(TRAIN_SEEDS)
                      + TRAIN_SEEDS.index(seed)
                      + VARIANTS.index(variant))
            return wrong_choices[offset % 3]
        if scenario == "rc1":
            if (arm, seed) == ("fixed", 2026) and variant == "RC":
                return (cls + 1) % 4
            return cls
        return cls

    files = {}
    for fold in FOLD_IDS:
        fold_names = [n for n in names if fold_of[n] == fold]
        rows = []
        for name in fold_names:
            cls = class_of[name]
            pred = r0_prediction_for(name)
            rows.append({
                "sample_name": name, "class_id": cls,
                "group": group_of[name],
                "has_valid_s": bool(s_valid[
                    cache["row_of"][name]]),
                "logits": logits_for(pred), "prediction": pred,
                "prediction_source": "synthetic",
            })
        rel = (f"models/R0/fold{fold[-1]}_R0/"
               "outer_eval_predictions.jsonl")
        files[rel] = rows
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            for variant in VARIANTS:
                for fold in FOLD_IDS:
                    fold_names = [n for n in names if fold_of[n] == fold]
                    rows = []
                    for name in fold_names:
                        cls = class_of[name]
                        pred = prediction_for(name, arm, seed, variant)
                        rows.append({
                            "sample_name": name, "class_id": cls,
                            "group": group_of[name],
                            "has_valid_s": bool(s_valid[
                                cache["row_of"][name]]),
                            "logits": logits_for(pred),
                            "prediction": pred,
                            "prediction_source": "synthetic",
                        })
                    rel = (f"models/seed_{seed}/fold{fold[-1]}_{arm}_"
                           f"{variant}/outer_eval_predictions.jsonl")
                    files[rel] = rows
    for rel, rows in files.items():
        path = fit_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
            encoding="utf-8")
    # the five scenario instances must be distinct and carry S
    assert len(set(picks.values())) == 5
    for name in picks.values():
        assert bool(s_valid[cache["row_of"][name]]), \
            f"{name}: the scenario instance must carry S"
    return {"picks": picks, "scenario_of": scenario_of, "fold_of": fold_of}


@pytest.fixture(scope="module")
def diag_world(module_scratch):
    """The synthetic stereo world with a production-built fix cache and a
    fabricated (never trained) fit directory."""
    from core.material_probe import (
        authoritative_sample_index,
        build_cache_binding,
    )

    workspace = module_scratch.resolve()
    source = _build_stereo_source(module_scratch / "source")
    v4 = build_synthetic_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)
    from core.material_mask_dataset import mask_dataset_fingerprint
    from core.fusion_dataset import dataset_fingerprint

    fingerprint = mask_dataset_fingerprint(v5)
    rows = _v5_rows(v5)
    protocol = _protocol_for_v5(rows, fingerprint)
    protocol_dir = module_scratch / "protocol"
    protocol_dir.mkdir(parents=True)
    protocol_path = protocol_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True),
                             encoding="utf-8")
    authority = authoritative_sample_index(protocol)

    # the synthetic "frozen global cache" (arrays + manifest + binding,
    # built by the production helpers — no training anywhere)
    global_dir = module_scratch / "global_cache"
    global_dir.mkdir(parents=True)
    _make_cache_files_for_v5(v5, rows)(global_dir, protocol)
    (global_dir / "cache_binding.json").write_text(
        json.dumps(build_cache_binding(global_dir, protocol),
                   sort_keys=True), encoding="utf-8")

    cfg = {
        "experiment": "material_residual_scale_v1",
        "anchors": {
            "runtime_config": {
                "path": str(module_scratch / "placeholder.yaml"),
                "sha256": "0" * 64},
            "protocol_file": {"path": str(protocol_path),
                              "sha256": "0" * 64},
            "artifact_manifest": {"path": str(protocol_path),
                                  "sha256": "0" * 64},
            "upsample_impact_config": {"path": str(protocol_path),
                                       "sha256": "0" * 64},
            "batch1_source_manifest": {"path": str(protocol_path),
                                       "sha256": "0" * 64},
        },
        "protocol_source": {"path": str(protocol_path)},
        "features_source": {"path": str(global_dir)},
        "p0_baseline_source": {
            "fit_dir": str(module_scratch / "unused_fit"),
            "verification": str(module_scratch / "unused_ver.json")},
        "data": {
            "root": str(v5), "source_root": str(source),
            "fingerprint": fingerprint,
            "parent_v4_fingerprint": dataset_fingerprint(v4),
            "class_names": list(CLASS_NAMES),
            "dev_pool_splits": ["train", "val"],
            "excluded_splits": ["test"]},
        "fix_inputs": {**SYNTH_MATCHER,
                       "upsample_version": "valid_weighted_legacy_"
                                           "support_v1",
                       "band_margin": 20,
                       "full_image_threshold": 0.9, "crop_pad": 10},
        "local_features": {
            "region_rule": "local_partition_interior_band_v1",
            "feature_definition_version":
                "existing_data_probe_v1_local_interior_band_v1"},
        "support_expectations": None,  # filled below
        "scale_rule": {"source": "p0_logits_inner_train",
                       "statistic": "median", "dtype": "float64",
                       "min_scale": 1.0},
        "arms": {"fixed": {"magnitude_factor": 0.5},
                 "scaled": {"magnitude_factor": 0.5}},
        "seeds": [2026, 2027, 2028],
        "shuffle": {"content_seeds": [2026, 2027, 2028, 2029, 2030]},
        "training": {"device": "cpu", "full_batch": True, "epochs": 1,
                     "optimizer": "adamw", "lr": 0.01,
                     "weight_decay": 0.001, "aux_loss_weight": 0.01},
        "head": {"hidden": 8},
    }
    # register_p0_fit requires the configured P0 fit dir to exist; the
    # diagnostic never consumes it beyond registration, so a stub with an
    # empty summary satisfies the contract without any training
    p0_stub = module_scratch / "unused_fit"
    p0_stub.mkdir(parents=True, exist_ok=True)
    (p0_stub / "summary.json").write_text("{}", encoding="utf-8")

    dev_rows = [row for row in rows if row["split"] in ("train", "val")]
    dev_frames = sorted({row["source_frame"] for row in dev_rows})
    groups = sorted({row["group_name"] for row in dev_rows})
    cfg["support_expectations"] = {
        "frames": len(dev_frames), "instances": len(dev_rows),
        "groups": len(groups), "empty_frames": 0}

    # the production fix cache (dataset build only — no training)
    from core.material_residual_scale import DeadlineWatch

    ctx = {"registry": prep.SourceRegistry({}),
           "watch": DeadlineWatch(budget_minutes=30),
           "stage": lambda *a, **k: None,
           "log": lambda message: None}
    pre = prep.run_preflight(ctx, cfg)
    build = prep.run_feature_build(ctx, cfg, pre["census"],
                                   pre["global_info"], frame_limit=None)
    features = module_scratch / "cache_build" / "features"
    prep.write_scale_cache(
        features, cfg, pre["protocol"], build, kind=CACHE_KIND_FULL,
        complete=True, interrupted=False,
        data_fingerprint=cfg["data"]["fingerprint"],
        protocol_sha256=pre["protocol"]["protocol_sha256"],
        empty_frames=pre["census"]["empty_frames"])
    npz = np.load(features / "feature_cache.npz")
    manifest = json.loads((features / "feature_manifest.json").read_text(
        encoding="utf-8"))
    cache = {"arrays": {k: npz[k] for k in npz.files},
             "manifest": manifest["samples"],
             "row_of": {s["sample_name"]: i
                        for i, s in enumerate(manifest["samples"])}}

    # the fabricated fit (predictions only — nothing was trained)
    fit_dir = module_scratch / "fit"
    fabricated = _fabricate_predictions(cache, authority, fit_dir)
    fit_summary = {
        "mode": "fit", "experiment": "material_residual_scale_v1",
        "training_status": "COMPLETED",
        "arms": {arm: {variant: {"macro_f1_mean_of_seeds": value}
                       for variant, value in (
                           ("R0", diag.THIRD_BATCH_FACTS[
                               "macro_f1_mean_of_seeds"][
                               ("fixed", "R0")]),
                           ("RC", diag.THIRD_BATCH_FACTS[
                               "macro_f1_mean_of_seeds"][
                               (arm, "RC")]),
                           ("RCS", diag.THIRD_BATCH_FACTS[
                               "macro_f1_mean_of_seeds"][
                               (arm, "RCS")]))}
                 for arm in ARMS},
        "gates": {gate: {"pass": value} for gate, value
                  in diag.THIRD_BATCH_FACTS["gate_pass"].items()},
    }
    (fit_dir / "summary.json").write_text(
        json.dumps(fit_summary, indent=1), encoding="utf-8")
    (fit_dir / "admission.json").write_text("{}", encoding="utf-8")
    (fit_dir / "run_status.json").write_text(
        json.dumps({"overall_status": "COMPLETED", "exit_code": 0}),
        encoding="utf-8")

    # the fit's own source manifest: every consumed file with its real
    # digest, so the diagnostic's expansion bindings actually hit
    consumed = []
    consumed.append(str(protocol_path))
    for rel in (*[p for p in _fit_pred_rels(fit_dir)],):
        consumed.append(str(rel))
    for sample in cache["manifest"]:
        consumed.append(str(module_scratch / "v5" /
                            sample["npz_path"]))
    v5_manifest = v5 / "dataset_manifest.csv"
    consumed.append(str(v5_manifest))
    import csv as _csv

    with v5_manifest.open(encoding="utf-8-sig", newline="") as handle:
        v5_csv_rows = list(_csv.DictReader(handle))
    pair_rows = {}
    with (source / "pair_manifest.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        for row in _csv.DictReader(handle):
            pair_rows[str(row["output_stem"]).strip()] = row
    by_stem = {}
    for row in v5_csv_rows:
        by_stem.setdefault(row["source_frame"], row)
    for stem in dev_frames:
        row = by_stem[stem]
        consumed.append(str(source / row["left_path"]))
        consumed.append(str(source /
                            pair_rows[stem]["right_path"]))
        consumed.append(str(source / row["label_path"]))
    consumed.append(str(source / "pair_manifest.csv"))
    items = []
    for path in consumed:
        p = Path(path)
        if not p.is_file():
            continue
        rel = p.relative_to(_ROOT).as_posix()
        items.append({"path": rel, "role": "fabricated_fit_consumed",
                      "sha256_at_registration":
                          hashlib.sha256(p.read_bytes()).hexdigest()})
    (fit_dir / "source_manifest.json").write_text(
        json.dumps({"items": items}, indent=1), encoding="utf-8")

    verification = module_scratch / "verification"
    verification.mkdir()
    (verification / "verification_report.json").write_text(
        json.dumps({"experiment": "material_residual_scale_v1",
                    "problem_count": 0, "problems": []}),
        encoding="utf-8")

    return {
        "workspace": workspace, "source": source, "v5": v5,
        "protocol": protocol, "protocol_path": protocol_path,
        "authority": authority, "cfg": cfg, "features": features,
        "cache": cache, "fit_dir": fit_dir,
        "verification": verification, "fabricated": fabricated,
        "dev_frames": dev_frames,
    }


def _fit_pred_rels(fit_dir: Path) -> list[Path]:
    return sorted(
        p for p in fit_dir.rglob("outer_eval_predictions.jsonl"))


@pytest.fixture(scope="module")
def diag_patches(diag_world, module_scratch):
    """Point the entry's frozen module constants at the synthetic world.
    The expected counts come from the INDEPENDENT scenario definitions
    (never from calling the ledger/count functions under test)."""
    fit_rel = diag_world["fit_dir"].relative_to(_ROOT).as_posix()
    verif_rel = diag_world["verification"].relative_to(
        _ROOT).as_posix()
    cache_rel = diag_world["features"].relative_to(_ROOT).as_posix()
    config_path = module_scratch / "configs" / "diag.yaml"
    _dump_cfg(config_path, diag_world["cfg"])
    config_rel = config_path.relative_to(_ROOT).as_posix()
    pred_rels = [p.relative_to(_ROOT).as_posix()
                 for p in _fit_pred_rels(diag_world["fit_dir"])]
    digest = prediction_set_digest(_ROOT, pred_rels)
    pinned = {}
    for rel in pred_rels + [
            f"{fit_rel}/summary.json", f"{fit_rel}/admission.json",
            f"{fit_rel}/run_status.json",
            f"{fit_rel}/source_manifest.json",
            f"{verif_rel}/verification_report.json",
            f"{cache_rel}/cache_binding.json",
            f"{cache_rel}/feature_cache.npz",
            config_rel]:
        pinned[rel] = hashlib.sha256(
            (_ROOT / rel).read_bytes()).hexdigest()

    # ---- INDEPENDENT scenario expectations (no ledger/count call) ----
    picks = diag_world["fabricated"]["picks"]
    class_of = {sample["sample_name"]: int(sample["class_id"])
                for sample in diag_world["cache"]["manifest"]}
    unique_instances = len(diag_world["cache"]["manifest"])
    # the five scenarios: c1/c2/w1 are R0-wrong (pred = cls+1);
    # c1 corrected once, c2 six times; b1 broken once
    r0_error_names = sorted({picks["c1"], picks["c2"], picks["w1"]})
    corrected_names = sorted({picks["c1"], picks["c2"]})
    broken_names = [picks["b1"]]
    corrected_pairs = 1 + 6
    broken_pairs = 1
    # the corrected confusions come from the scenario definitions: the
    # R0 prediction is (cls + 1) % 4 for c1 and c2
    corrected_confusions = {}
    for name in corrected_names:
        key = (class_of[name], (class_of[name] + 1) % 4)
        corrected_confusions[key] = corrected_confusions.get(key, 0) + 1
    # the R0 error confusion directions: c1/c2/w1 each contribute one
    # (cls -> cls+1) error; every other instance is R0-correct
    error_pairs = [(class_of[n], (class_of[n] + 1) % 4)
                   for n in r0_error_names]
    import core.material_polar_complementarity as core_mod

    confusion_direction_counts = {}
    for group in core_mod.MAIN_CONFUSION_GROUPS:
        confusion_direction_counts[group["name"]] = {
            (a, b): sum(1 for pair in error_pairs if pair == (a, b))
            for a, b in group["directions"]}
    other_errors = len(error_pairs) - sum(
        count
        for per_group in confusion_direction_counts.values()
        for count in per_group.values())
    expected = {
        "unique_instances": unique_instances,
        "r0_errors": 3,
        "r0_errors_with_s": 3,
        "corrected_union": 2,
        "broken_union": 1,
        "changed_union": 3,
        "confusion_direction_counts": confusion_direction_counts,
        "corrected_confusions": corrected_confusions,
    }
    independent = {
        "r0_errors": 3,
        "r0_error_names": r0_error_names,
        "corrected_names": corrected_names,
        "broken_names": broken_names,
        "changed_union": 3,
        "corrected_pairs": corrected_pairs,
        "broken_pairs": broken_pairs,
        "other_errors": other_errors,
        "w1_never_in_changed": picks["w1"] not in set(corrected_names)
        | set(broken_names),
        "unique_instances": unique_instances,
    }

    patches = [
        mock.patch.object(core_mod, "PREDICTION_FILES", pred_rels),
        mock.patch.object(core_mod, "PREDICTION_SET_SHA256", digest),
        mock.patch.object(core_mod, "EXPECTED_COUNTS", expected),
        mock.patch.object(diag, "PREDICTION_FILES", pred_rels),
        mock.patch.object(diag, "PREDICTION_SET_SHA256", digest),
        mock.patch.object(diag, "PINNED_INPUTS", pinned),
        mock.patch.object(diag, "PINNED_CONFIG_REL", config_rel),
        mock.patch.object(diag, "FIT_DIR", fit_rel),
        mock.patch.object(diag, "FIT_DIR_VERIF_DIR", verif_rel),
        mock.patch.object(diag, "CACHE_DIR", cache_rel),
        mock.patch.object(diag, "DIAGNOSTIC_ROOT",
                          (module_scratch / "diagroot").relative_to(
                              _ROOT).as_posix()),
    ]
    for context in patches:
        context.start()
    try:
        yield {"expected": expected, "independent": independent,
               "picks": picks, "pred_rels": pred_rels,
               "config_rel": config_rel, "config_path": config_path,
               "fit_dir": diag_world["fit_dir"],
               "manifest": diag_world["cache"]["manifest"],
               "arrays": diag_world["cache"]["arrays"],
               "authority": diag_world["authority"]}
    finally:
        for context in patches:
            context.stop()


def _run_diag(module_scratch: Path, diag_world, **overrides) -> int:
    cfg_path = _dump_cfg(module_scratch / "configs" / "diag.yaml",
                         diag_world["cfg"])
    # the synthetic pin tracks the synthetic config file: each dump's
    # digest becomes the pinned expectation for this run (production
    # keeps one fixed review-pinned digest)
    diag.PINNED_INPUTS[diag.PINNED_CONFIG_REL] = hashlib.sha256(
        cfg_path.read_bytes()).hexdigest()
    argv = [
        "--config", str(cfg_path),
        "--fit-dir", str(diag_world["fit_dir"]),
        "--verification-dir", str(diag_world["verification"]),
        "--feature-dir", str(diag_world["features"]),
        "--output-dir", str(overrides.pop("output_dir")),
        "--budget-minutes", str(overrides.pop("budget_minutes", 30)),
    ]
    argv.extend(str(v) for pair in overrides.items() for v in pair)
    return diag.main(argv)


class TestDiagnosticEntry:
    def test_full_run_completes_with_panels_and_zero_training(
            self, diag_world, diag_patches, module_scratch):
        """Real-symbol zero-training probes: each bomb is first proven to
        fire on the REAL symbol it patches, then the full entry must run
        to COMPLETED with every bomb count at zero."""
        import torch
        import core.material_probe as probe_mod
        import core.material_conditional_residual as cond_mod

        calls = {"scaler_fit": 0, "head_build": 0, "cond_logits": 0,
                 "optimizer": 0}

        def scaler_bomb(*a, **k):
            calls["scaler_fit"] += 1
            raise AssertionError("Standardizer.fit must never run")

        def head_bomb(*a, **k):
            calls["head_build"] += 1
            raise AssertionError("build_conditional_head must never run")

        def logits_bomb(*a, **k):
            calls["cond_logits"] += 1
            raise AssertionError("conditional_logits must never run")

        class OptimBomb:
            def __init__(self, *a, **k):
                calls["optimizer"] += 1
                raise AssertionError("no optimizer may be constructed")

        # every patched symbol is a REAL existing one; prove each bomb
        # actually fires on that symbol before trusting the zero counts
        with mock.patch.object(probe_mod.Standardizer, "fit", scaler_bomb):
            with pytest.raises(AssertionError):
                probe_mod.Standardizer.fit(None, None)
        with mock.patch.object(cond_mod, "build_conditional_head",
                               head_bomb):
            with pytest.raises(AssertionError):
                cond_mod.build_conditional_head(seed=1, input_dim=4,
                                                hidden=2)
        with mock.patch.object(cond_mod, "conditional_logits",
                               logits_bomb):
            with pytest.raises(AssertionError):
                cond_mod.conditional_logits(None, None, None, None, 0.5)
        with mock.patch.object(torch.optim, "AdamW", OptimBomb):
            with pytest.raises(AssertionError):
                torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])

        # reset: the self-verification legitimately fired each bomb once;
        # the entry run itself must be the zero
        calls = {"scaler_fit": 0, "head_build": 0, "cond_logits": 0,
                 "optimizer": 0}
        out = module_scratch / "diagroot" / "full_run"
        with mock.patch.object(probe_mod.Standardizer, "fit",
                               scaler_bomb), \
                mock.patch.object(cond_mod, "build_conditional_head",
                                  head_bomb), \
                mock.patch.object(cond_mod, "conditional_logits",
                                  logits_bomb), \
                mock.patch.object(torch.optim, "AdamW", OptimBomb):
            code = _run_diag(module_scratch, diag_world,
                             output_dir=out)
        assert code == diag.EXIT_OK
        assert calls == {"scaler_fit": 0, "head_build": 0,
                         "cond_logits": 0, "optimizer": 0}, \
            "a forbidden call path fired"
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "COMPLETED"
        assert status["zero_training_calls"]["new_model_fits"] == 0
        assert status["protection"]["identical"] is True
        ledger = json.loads((out / "instance_ledger.json").read_text(
            encoding="utf-8"))
        assert len(ledger["rows"]) == \
            diag_patches["independent"]["unique_instances"]
        counts = json.loads((out / "counts.json").read_text(
            encoding="utf-8"))
        independent = diag_patches["independent"]
        assert counts["changed_union"] == independent["changed_union"]
        assert counts["corrected_pairs"] == independent["corrected_pairs"]
        assert counts["broken_pairs"] == independent["broken_pairs"]
        assert counts["r0_errors"] == independent["r0_errors"]
        assert sorted(counts["changed_names"]) == \
            sorted(independent["corrected_names"]
                   + independent["broken_names"])
        manifest = json.loads((out / "panels_manifest.json").read_text(
            encoding="utf-8"))
        panels = manifest["panels"]
        assert len(panels) <= MAX_PANELS
        changed = [p for p in panels if p["kind"] == "changed"]
        assert len(changed) == counts["changed_union"], \
            "every correctness-changed instance has a panel"
        assert all(p["rebuild_check"]["problems"] == []
                   for p in panels)
        assert all((out / "panels" / p["file"]).is_file()
                   for p in panels)
        assert (out / "index.html").is_file()
        assert (out / "report.md").is_file()
        assert (out / "conditional_tables.json").is_file()
        assert (out / "binning.json").is_file()
        selection = json.loads((out / "selection_manifest.json")
                               .read_text(encoding="utf-8"))
        assert len(selection["control_panels"]) <= MAX_CONTROLS

    def test_scenario_semantics_hold_in_the_saved_outcomes(
            self, diag_world, diag_patches):
        """The five scenarios must survive the REAL ledger build with the
        independent expectations (c1 corrected once, c2 six times, b1
        broken once, w1 never changed, rc1 RC-only)."""
        ledger, stats = build_ledger(
            diag_patches["fit_dir"], diag_patches["arrays"],
            diag_patches["manifest"], diag_patches["authority"])
        independent = diag_patches["independent"]
        picks = diag_patches["picks"]
        by_name = {row["sample_name"]: row for row in ledger}
        assert sorted(stats["changed_names"]) == \
            sorted(independent["corrected_names"]
                   + independent["broken_names"])
        assert stats["corrected_pairs"] == independent["corrected_pairs"]
        assert stats["broken_pairs"] == independent["broken_pairs"]
        assert stats["r0_errors"] == independent["r0_errors"]
        # w1 stays wrong on every path and never enters the changed set
        w1 = by_name[picks["w1"]]
        assert all(not w1["paths"][key]["correct"] for key in w1["paths"])
        assert picks["w1"] not in set(stats["changed_names"])
        # rc1: RC-only marker — the fixed/2026 RC is wrong while its RCS
        # is right, and it never changes correctness versus R0
        rc1 = by_name[picks["rc1"]]
        assert not rc1["paths"]["fixed|2026|RC"]["correct"]
        assert rc1["paths"]["fixed|2026|RCS"]["correct"]
        assert rc1["rcs_vs_rc"]["fixed|2026"]["fixed"] is True
        assert all(not tag["fixed"] and not tag["broke"]
                   for tag in rc1["rcs_vs_r0"].values())

    def test_input_drift_is_refused(self, diag_world, diag_patches,
                                    module_scratch):
        target = diag_world["verification"] / "verification_report.json"
        original = target.read_bytes()
        out = module_scratch / "diagroot" / "drift"
        try:
            target.write_text(
                json.dumps({"problem_count": 0, "problems": [],
                            "tampered": True}), encoding="utf-8")
            code = _run_diag(module_scratch, diag_world,
                             output_dir=out)
            assert code == diag.EXIT_REFUSED
            status = json.loads((out / "run_status.json").read_text(
                encoding="utf-8"))
            assert "pinned" in status["failure"] \
                or "sha256" in status["failure"]
        finally:
            target.write_bytes(original)

    def test_rebuild_mismatch_is_refused(self, diag_world,
                                         diag_patches, module_scratch):
        npz_path = diag_world["features"] / "feature_cache.npz"
        original = npz_path.read_bytes()
        out = module_scratch / "diagroot" / "rebuild_mismatch"
        try:
            with np.load(npz_path) as npz:
                arrays = {k: npz[k] for k in npz.files}
            # tamper the S16 row of a SELECTED (changed) instance so the
            # diagnostic's rebuild comparison is what catches it
            changed = diag_patches["independent"]["corrected_names"][0]
            row = diag_world["cache"]["row_of"][changed]
            if np.isnan(arrays["s_features"][row, 0]):
                arrays["s_features"][row, 0] = np.float32(0.25)
            else:
                arrays["s_features"][row, 0] = np.float32(
                    arrays["s_features"][row, 0] + 0.5)
            np.savez(npz_path, **arrays)
            # the tampered bytes are pinned as-is: the admission hash
            # gate must NOT be the catcher here, the rebuild-vs-cache
            # comparison is
            new_digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            cache_rel = diag.CACHE_DIR
            with mock.patch.object(
                    diag, "PINNED_INPUTS",
                    {**diag.PINNED_INPUTS,
                     f"{cache_rel}/feature_cache.npz": new_digest}):
                code = _run_diag(module_scratch, diag_world,
                                 output_dir=out)
            assert code == diag.EXIT_REFUSED
            status = json.loads((out / "run_status.json").read_text(
                encoding="utf-8"))
            assert "disagree" in status["failure"]
        finally:
            npz_path.write_bytes(original)

    def test_output_path_outside_the_root_is_refused(
            self, diag_world, diag_patches, module_scratch):
        code = _run_diag(module_scratch, diag_world,
                         output_dir=module_scratch / "elsewhere" / "x")
        assert code == diag.EXIT_REFUSED

    def test_existing_output_dir_is_refused(self, diag_world,
                                            diag_patches, module_scratch):
        out = module_scratch / "diagroot" / "existing"
        out.mkdir(parents=True)
        code = _run_diag(module_scratch, diag_world, output_dir=out)
        assert code == diag.EXIT_REFUSED

    def test_shared_deadline_crossing_publishes_partial(
            self, diag_world, diag_patches, module_scratch):
        out = module_scratch / "diagroot" / "deadline"
        code = _run_diag(module_scratch, diag_world,
                         output_dir=out, budget_minutes=0.000001)
        assert code == diag.EXIT_PARTIAL
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        assert status["unfinished"], "the unfinished list is kept"

    def test_substitute_config_is_refused_before_parsing(
            self, diag_world, diag_patches, module_scratch):
        """A substitute config path is refused even before the config
        parser is constructed; load_scale_config is never called."""
        parse_calls = {"n": 0}

        def parse_bomb(*a, **k):
            parse_calls["n"] += 1
            raise AssertionError("the config parser must never run")

        substitute = module_scratch / "configs" / "substitute.yaml"
        _dump_cfg(substitute, diag_world["cfg"])
        out = module_scratch / "diagroot" / "subcfg"
        code = diag.main([
            "--config", str(substitute),
            "--fit-dir", str(diag_world["fit_dir"]),
            "--verification-dir", str(diag_world["verification"]),
            "--feature-dir", str(diag_world["features"]),
            "--output-dir", str(out),
            "--budget-minutes", "30",
        ])
        assert code == diag.EXIT_REFUSED
        with mock.patch.object(diag, "load_scale_config", parse_bomb):
            code = diag.main([
                "--config", str(substitute),
                "--fit-dir", str(diag_world["fit_dir"]),
                "--verification-dir", str(diag_world["verification"]),
                "--feature-dir", str(diag_world["features"]),
                "--output-dir", str(out),
                "--budget-minutes", "30",
            ])
        assert code == diag.EXIT_REFUSED
        assert parse_calls["n"] == 0, \
            "the config parser ran despite the substitute path"

    def test_config_drift_is_refused_before_parsing(
            self, diag_world, diag_patches, module_scratch):
        """A VALID-content change of the pinned config is refused at the
        pinned-hash check — the parser never sees it."""
        parse_calls = {"n": 0}

        def parse_bomb(*a, **k):
            parse_calls["n"] += 1
            raise AssertionError("the config parser must never run")

        config_path = diag_patches["config_path"]
        original = config_path.read_bytes()
        out = module_scratch / "diagroot" / "cfgdrift"
        try:
            drifted = copy.deepcopy(diag_world["cfg"])
            drifted["head"] = {"hidden": 16}
            # NO _run_diag here: it would re-dump the original content
            # and heal the tamper before main runs
            config_path.write_text(
                yaml.safe_dump(drifted, sort_keys=True),
                encoding="utf-8")
            with mock.patch.object(diag, "load_scale_config", parse_bomb):
                code = diag.main([
                    "--config", str(config_path),
                    "--fit-dir", str(diag_world["fit_dir"]),
                    "--verification-dir", str(diag_world["verification"]),
                    "--feature-dir", str(diag_world["features"]),
                    "--output-dir", str(out),
                    "--budget-minutes", "30",
                ])
            assert code == diag.EXIT_REFUSED
            assert parse_calls["n"] == 0
            status = json.loads((out / "run_status.json").read_text(
                encoding="utf-8"))
            assert "sha256" in status["failure"] \
                or "pinned" in status["failure"]
        finally:
            config_path.write_bytes(original)

    def test_config_is_registered_and_bound_in_the_published_manifest(
            self, diag_world, diag_patches, module_scratch):
        """After a normal run the published source manifest holds the
        config's registration as a pinned anchor with the pinned digest
        as its binding — the registration happened before the parser
        consumed the file."""
        out = module_scratch / "diagroot" / "cfgorder"
        code = _run_diag(module_scratch, diag_world, output_dir=out)
        assert code == diag.EXIT_OK
        manifest = json.loads((out / "source_manifest.json").read_text(
            encoding="utf-8"))
        # rows are keyed by PATH: all eight pinned inputs share one role,
        # so the config row must be selected by its path, not its role
        config_rows = [row for row in manifest["items"]
                       if row["path"] == diag.PINNED_CONFIG_REL]
        assert len(config_rows) == 1, \
            "the config must be registered exactly once"
        config_row = config_rows[0]
        assert config_row["binding_source"] == \
            "config.anchors[diagnostic_input]"
        current_digest = hashlib.sha256(
            diag_patches["config_path"].read_bytes()).hexdigest()
        assert config_row["sha256_at_registration"] == current_digest, \
            "the registered config digest differs from the file on disk"
        assert current_digest == diag.PINNED_INPUTS[
            diag.PINNED_CONFIG_REL], (
                f"the pinned digest "
                f"{diag.PINNED_INPUTS[diag.PINNED_CONFIG_REL]} differs "
                f"from the on-disk config {current_digest}; the fixture "
                "hashing and the run saw different bytes")


class TestLedgerIdentityRefusals:
    """Direct calls to the REAL production build_ledger over tampered
    copies of the synthetic fit directory."""

    def _copy_fit(self, diag_world, module_scratch, name):
        target = module_scratch / name
        shutil.copytree(diag_world["fit_dir"], target)
        return target

    @staticmethod
    def _rows_of(path: Path) -> list:
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    @staticmethod
    def _write_rows(path: Path, rows: list) -> None:
        path.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
            encoding="utf-8")

    def _expect_refusal(self, diag_world, module_scratch, mutate,
                        name, fragment):
        fit_copy = self._copy_fit(diag_world, module_scratch, name)
        mutate(fit_copy)
        with pytest.raises(ComplementarityRefused) as excinfo:
            build_ledger(fit_copy, diag_world["cache"]["arrays"],
                         diag_world["cache"]["manifest"],
                         diag_world["authority"])
        assert fragment in str(excinfo.value)

    def test_missing_sample_is_refused(self, diag_world, diag_patches,
                                       module_scratch):
        def mutate(fit_copy):
            path = next(iter(fit_copy.rglob(
                "fold0_fixed_RC/outer_eval_predictions.jsonl")))
            rows = self._rows_of(path)
            self._write_rows(path, rows[:-1])

        self._expect_refusal(diag_world, module_scratch, mutate,
                             "ledger_missing", "differs from the fold")

    def test_duplicate_sample_is_refused(self, diag_world, diag_patches,
                                         module_scratch):
        def mutate(fit_copy):
            path = next(iter(fit_copy.rglob(
                "fold0_fixed_RC/outer_eval_predictions.jsonl")))
            rows = self._rows_of(path)
            self._write_rows(path, rows + [rows[0]])

        self._expect_refusal(diag_world, module_scratch, mutate,
                             "ledger_dup", "duplicate sample_name")

    def test_unknown_sample_is_refused(self, diag_world, diag_patches,
                                       module_scratch):
        def mutate(fit_copy):
            path = next(iter(fit_copy.rglob(
                "fold0_fixed_RC/outer_eval_predictions.jsonl")))
            rows = self._rows_of(path)
            foreign = dict(rows[0])
            foreign["sample_name"] = "not_in_the_manifest_obj000"
            self._write_rows(path, rows + [foreign])

        self._expect_refusal(diag_world, module_scratch, mutate,
                             "ledger_unknown", "differs from the fold")

    def test_class_conflict_is_refused(self, diag_world, diag_patches,
                                       module_scratch):
        def mutate(fit_copy):
            path = next(iter(fit_copy.rglob(
                "fold0_fixed_RC/outer_eval_predictions.jsonl")))
            rows = self._rows_of(path)
            rows[0]["class_id"] = (int(rows[0]["class_id"]) + 1) % 4
            self._write_rows(path, rows)

        self._expect_refusal(diag_world, module_scratch, mutate,
                             "ledger_class", "class_id conflict")

    def test_group_conflict_is_refused(self, diag_world, diag_patches,
                                       module_scratch):
        def mutate(fit_copy):
            path = next(iter(fit_copy.rglob(
                "fold0_fixed_RC/outer_eval_predictions.jsonl")))
            rows = self._rows_of(path)
            rows[0]["group"] = rows[0]["group"] + "_fabricated"
            self._write_rows(path, rows)

        self._expect_refusal(diag_world, module_scratch, mutate,
                             "ledger_group", "group conflict")

    def test_fold_role_conflict_is_refused(self, diag_world,
                                           diag_patches, module_scratch):
        """A sample moved into the WRONG fold's file is refused (the row
        set no longer matches that fold's outer_eval set)."""
        def mutate(fit_copy):
            source = next(iter(fit_copy.rglob(
                "fold0_fixed_RC/outer_eval_predictions.jsonl")))
            rows = self._rows_of(source)
            mover = dict(rows[0])
            self._write_rows(source, rows[1:])
            target = (fit_copy / "models" / "seed_2026" /
                      "fold1_fixed_RC" / "outer_eval_predictions.jsonl")
            self._write_rows(target, self._rows_of(target) + [mover])

        self._expect_refusal(diag_world, module_scratch, mutate,
                             "ledger_fold", "differs from the fold")


class _MiniWorld:
    """Six hand-built ledger rows covering the five scenarios plus one
    plain instance; the saved counts are written BY HAND from the
    scenario definitions, never from the functions under test."""

    def __init__(self):
        self.c1 = _mini_row("c1", 2, 3, group="gA", fold="fold0",
                            path_truth={"fixed|2026|RCS": True})
        self.c2 = _mini_row(
            "c2", 2, 3, group="gB", fold="fold1",
            path_truth={f"{a}|{s}|RCS": True for a in ARMS
                        for s in TRAIN_SEEDS})
        self.b1 = _mini_row("b1", 1, 1, group="gA", fold="fold0",
                            path_truth={"scaled|2027|RCS": False})
        self.rc1 = _mini_row("rc1", 1, 1, group="gB", fold="fold1",
                             path_truth={"fixed|2026|RC": False})
        self.n1 = _mini_row("n1", 0, 0, group="gA", fold="fold0")
        # w1: always wrong with the wrong class rotating over the three
        # non-true classes (true class 3 -> never 3)
        self.w1 = _mini_row("w1", 3, 0, group="gB", fold="fold1")
        for arm in ARMS:
            for seed in TRAIN_SEEDS:
                for variant in VARIANTS:
                    wrong_choices = [c for c in range(4) if c != 3]
                    offset = (ARMS.index(arm) * 6
                              + TRAIN_SEEDS.index(seed) * 2
                              + VARIANTS.index(variant))
                    pred = wrong_choices[offset % 3]
                    self.w1["paths"][f"{arm}|{seed}|{variant}"] = {
                        "logits": [0.1, 0.2, 0.3, 0.4],
                        "prediction": pred, "correct": False}
        for arm in ARMS:
            for seed in TRAIN_SEEDS:
                rcs = self.w1["paths"][f"{arm}|{seed}|RCS"]
                rc = self.w1["paths"][f"{arm}|{seed}|RC"]
                for tag, other in (("rcs_vs_r0", self.w1["r0"]),
                                   ("rcs_vs_rc", rc)):
                    fixed = bool(rcs["correct"]) and \
                        not bool(other["correct"])
                    broke = bool(other["correct"]) and \
                        not bool(rcs["correct"])
                    self.w1[tag][f"{arm}|{seed}"] = {
                        "fixed": fixed, "broke": broke,
                        "net": int(fixed) - int(broke)}

    @property
    def rows(self):
        return [self.c1, self.c2, self.b1, self.w1, self.rc1, self.n1]

    @property
    def independent_counts(self):
        return {
            "unique_instances": 6,
            "r0_errors": 3,
            "r0_errors_with_s": 3,
            "corrected_union": 2,
            "broken_union": 1,
            "changed_union": 3,
            "corrected_pairs": 7,
            "broken_pairs": 1,
            "instance_model_pairs_evaluated": 72,
            "rcs_vs_r0_comparison_pairs": 36,
        }


@pytest.fixture()
def postprocess_world(tmp_path_factory):
    """A micro saved-diagnostic directory built WITHOUT the full world:
    plain JSON files, placeholder panel bytes, hand-written counts."""
    root = tmp_path_factory.mktemp("postprocess_world")
    prior = root / "prior_diagnostic"
    prior.mkdir()
    world = _MiniWorld()
    rows = world.rows
    binning = {
        "edges_by_fold": {
            "fold0": {"valid_ratio": [0.2, 0.8],
                      "gray_mean": [0.05, 0.5],
                      "source": "fixture edges"},
            "fold1": {"valid_ratio": [0.3, 0.9],
                      "gray_mean": [0.1, 0.6],
                      "source": "fixture edges"}},
        "bins": {},
    }
    from core.material_residual import build_donor_bins

    for fold in ("fold0", "fold1"):
        fold_rows = [r for r in rows if r["outer_fold"] == fold]
        names = sorted(r["sample_name"] for r in fold_rows)
        by_name = {r["sample_name"]: r for r in fold_rows}
        vr = np.asarray([by_name[n]["c_features"][0] for n in names],
                        dtype=np.float64)
        gm = np.asarray([by_name[n]["c_features"][3] for n in names],
                        dtype=np.float64)
        bins = build_donor_bins(vr, gm, edges=binning["edges_by_fold"][
            fold])
        for local, name in enumerate(names):
            binning["bins"][name] = int(bins[local])
    (prior / "instance_ledger.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False),
        encoding="utf-8")
    (prior / "binning.json").write_text(
        json.dumps(binning), encoding="utf-8")
    # the saved counts are HAND-WRITTEN from the scenario definitions
    (prior / "counts.json").write_text(json.dumps(
        world.independent_counts), encoding="utf-8")
    panels = []
    panels_dir = prior / "panels"
    panels_dir.mkdir()
    for name, kind in (("c1", "changed"), ("c2", "changed"),
                       ("b1", "changed"), ("n1", "control")):
        payload = f"placeholder panel for {name}".encode("utf-8")
        (panels_dir / f"p_{name}.png").write_bytes(payload)
        panels.append({
            "panel_id": f"p_{name}", "sample_name": name, "kind": kind,
            "file": f"p_{name}.png", "bytes": len(payload),
            "rebuild_check": {"sample_name": name, "problems": []},
            "source_frame": f"frame_{name}"})
    (prior / "panels_manifest.json").write_text(json.dumps({
        "panels": panels, "panel_count": len(panels)}),
        encoding="utf-8")
    (prior / "selection_manifest.json").write_text(json.dumps({
        "control_panels": [{"sample_name": "n1",
                            "panel_id": "p_n1", "kind": "control",
                            "role": "correct_control",
                            "reason": "fixture"}],
        "control_notes": []}), encoding="utf-8")
    (prior / "run_status.json").write_text(json.dumps({
        "overall_status": "COMPLETED"}), encoding="utf-8")
    (prior / "source_manifest.json").write_text(json.dumps({
        "items": []}), encoding="utf-8")
    (prior / "report.md").write_text("old report", encoding="utf-8")
    return {"prior": prior, "rows": rows, "binning": binning,
            "independent": world.independent_counts}


def _postprocess_pins(postprocess_world) -> dict:
    return {
        name: hashlib.sha256(
            (postprocess_world["prior"] / name).read_bytes()).hexdigest()
        for name in ("instance_ledger.json", "binning.json", "counts.json",
                     "selection_manifest.json", "panels_manifest.json",
                     "source_manifest.json", "run_status.json")}


def _run_postprocess(module_scratch, postprocess_world, output_dir,
                     extra_patches=(), budget_minutes=30,
                     pinned=None):
    prior_rel = postprocess_world["prior"].relative_to(_ROOT).as_posix()
    pinned = pinned if pinned is not None else _postprocess_pins(
        postprocess_world)
    patches = [
        mock.patch.object(diag, "PRIOR_DIAGNOSTIC_REL", prior_rel),
        mock.patch.object(diag, "PINNED_PRIOR_ARTIFACTS", pinned),
        mock.patch.object(diag, "DIAGNOSTIC_ROOT",
                          module_scratch.relative_to(_ROOT).as_posix()),
        mock.patch.object(diag, "POSTPROCESS_EXPECTED", dict(
            postprocess_world["independent"])),
    ]
    patches.extend(extra_patches)
    for context in patches:
        context.start()
    try:
        return diag.main([
            "--mode", "postprocess",
            "--config", str(_ROOT / diag.PINNED_CONFIG_REL),
            "--diagnostic-dir", str(postprocess_world["prior"]),
            "--output-dir", str(output_dir),
            "--budget-minutes", str(budget_minutes),
        ])
    finally:
        for context in patches:
            context.stop()


class TestPostprocessMode:
    def test_full_postprocess_completes_with_zero_forbidden_calls(
            self, postprocess_world, module_scratch):
        """Real-symbol probes prove the offline branch never rebuilds,
        matches, computes pixels or renders; each bomb is first proven to
        fire on the symbol it patches."""
        calls = {"rebuild_frame": 0, "matcher": 0, "polar": 0,
                 "render": 0}

        def rebuild_bomb(*a, **k):
            calls["rebuild_frame"] += 1
            raise AssertionError("rebuild_frame must never run")

        def matcher_bomb(*a, **k):
            calls["matcher"] += 1
            raise AssertionError("StereoMatcher must never be built")

        def polar_bomb(*a, **k):
            calls["polar"] += 1
            raise AssertionError("compute_polar_features must never run")

        def render_bomb(*a, **k):
            calls["render"] += 1
            raise AssertionError("render_panel must never run")

        # self-verify each bomb against the real symbol
        with mock.patch.object(diag, "rebuild_frame", rebuild_bomb):
            with pytest.raises(AssertionError):
                diag.rebuild_frame(None, None, None, None, None, None,
                                   None)
        with mock.patch.object(diag, "StereoMatcher", matcher_bomb):
            with pytest.raises(AssertionError):
                diag.StereoMatcher(None)
        with mock.patch.object(diag, "compute_polar_features",
                               polar_bomb):
            with pytest.raises(AssertionError):
                diag.compute_polar_features(None, None, None)
        with mock.patch.object(diag, "render_panel", render_bomb):
            with pytest.raises(AssertionError):
                diag.render_panel(None, None, None, None, None)

        # reset: the self-verification legitimately fired each bomb once;
        # the postprocess run itself must be the zero
        calls = {"rebuild_frame": 0, "matcher": 0, "polar": 0,
                 "render": 0}
        out = module_scratch / "post_out"
        with mock.patch.object(diag, "rebuild_frame", rebuild_bomb), \
                mock.patch.object(diag, "StereoMatcher", matcher_bomb), \
                mock.patch.object(diag, "compute_polar_features",
                                  polar_bomb), \
                mock.patch.object(diag, "render_panel", render_bomb):
            code = _run_postprocess(module_scratch, postprocess_world,
                                    out)
        assert code == diag.EXIT_OK
        assert calls == {"rebuild_frame": 0, "matcher": 0, "polar": 0,
                         "render": 0}
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "COMPLETED"
        assert status["run_mode"] == "postprocess"
        assert status["protection"]["identical"] is True
        counts = json.loads((out / "counts.json").read_text(
            encoding="utf-8"))
        assert counts["unique_instances"] == 6
        assert counts["corrected_pairs"] == 7
        assert counts["broken_pairs"] == 1
        assert counts["denominators"][
            "rc_rcs_saved_prediction_pairs"] == 72
        assert counts["denominators"][
            "rcs_vs_r0_comparison_pairs"] == 36
        tables = json.loads((out / "conditional_tables.json").read_text(
            encoding="utf-8"))
        assert tables["tables"]["same_group_comparisons"]
        assert (out / "same_group_comparisons.json").is_file()
        assert (out / "postprocess_admission.json").is_file()
        assert (out / "route_suggestion.json").is_file()
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "本次重建 0 帧" in report
        assert "12×2446＝72" in report or "12×2446=72" in report
        assert "6×2446＝36" in report or "6×2446=36" in report
        assert "路线 C" in report
        index = (out / "index.html").read_text(encoding="utf-8")
        assert "p_c1.png" in index and "nothing was re-rendered" in index

    def test_layered_cells_use_within_condition_data(
            self, postprocess_world):
        """Give gA and gB clearly different S16 levels: each layered
        cell's median must come from its own members — the pooled value
        is never presented as the conditional one."""
        shifted = []
        for row in postprocess_world["rows"]:
            row = copy.deepcopy(row)
            if row["group"] == "gB":
                for i in range(16):
                    if row["s_features"][i] is not None:
                        row["s_features"][i] = \
                            row["s_features"][i] + 10.0
            shifted.append(row)
        bin_of = {r["sample_name"]: 4 for r in shifted}
        analysis = build_conditional_analysis(
            shifted, derive_counts(shifted), bin_of)
        group_cells = [c for c in analysis["groups"][
            "plastic_submarine_vs_plastic_fish"]["layered"]
            if c["true_class"] == "plastic_submarine"]
        assert group_cells
        for cell in group_cells:
            members = [r for r in shifted
                       if r["class_id"] == 2
                       and r["group"] == cell["group"]
                       and r["outer_fold"] == cell["outer_fold"]]
            values = [r["s_features"][8] for r in members
                      if r["s_features"][8] is not None]
            assert cell["band"]["signed_q_mean"]["median"] == \
                pytest.approx(float(np.median(values)))
        medians = {c["group"]: c["band"]["signed_q_mean"]["median"]
                   for c in group_cells}
        assert abs(medians["gA"] - medians["gB"]) > 5.0

    def test_class_separation_and_missing_parts(self):
        rows = [
            _mini_row("a1", 1, 2, group="g", fold="fold0",
                      s_features=[None] * 8 + [0.5] * 8),
            _mini_row("a2", 1, 1, group="g", fold="fold0",
                      s_features=[None] * 8 + [0.5] * 8),
            _mini_row("b1", 2, 2, group="g", fold="fold0",
                      s_features=[0.1] * 8 + [None] * 8),
        ]
        bin_of = {r["sample_name"]: 0 for r in rows}
        analysis = build_conditional_analysis(rows, derive_counts(rows),
                                              bin_of)
        for name, group in analysis["groups"].items():
            for cell in group["layered"]:
                if cell["true_class"] == "plastic_submarine":
                    assert cell["interior"]["signed_q_mean"]["n_valid"] \
                        == 0
                    assert cell["interior"]["signed_q_mean"][
                        "median"] is None
                    assert cell["band"]["signed_q_mean"]["n_valid"] == 2
        for name, group in analysis["groups"].items():
            for cell in group["layered"]:
                assert cell["true_class"] in (
                    "plastic_submarine", "plastic_fish",
                    "metal_submarine", "real_fish")

    def test_one_sided_comparisons_are_not_comparable(self):
        rows = [
            _mini_row("e1", 1, 2, group="g", fold="fold0",
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
            _mini_row("ok1", 1, 1, group="g", fold="fold0"),
        ]
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        analysis = build_conditional_analysis(rows, stats, bin_of)
        comparisons = analysis["same_group_comparisons"]
        paired = [c for c in comparisons
                  if c["pair"].startswith(("corrected_vs_",
                                           "broken_vs_"))]
        one_sided = [c for c in paired if not c.get("comparable")]
        assert one_sided, "the empty side must be recorded"
        for cell in one_sided:
            empty = cell["left"] if cell["left"]["support"] == 0 \
                else cell["right"]
            assert empty["support"] == 0
            assert empty["interior"]["signed_q_mean"]["median"] is None

    def test_bins_never_mix_folds(self):
        rows = [
            _mini_row("a1", 1, 1, group="g", fold="fold0"),
            _mini_row("a2", 1, 1, group="g", fold="fold1"),
        ]
        bin_of = {"a1": 4, "a2": 4}  # the SAME bin number in both folds
        analysis = build_conditional_analysis(rows, derive_counts(rows),
                                              bin_of)
        for name, group in analysis["groups"].items():
            cells = [c for c in group["layered"]]
            folds = {c["outer_fold"] for c in cells
                     if c["true_class"] == "plastic_submarine"}
            if name == "plastic_submarine_vs_plastic_fish":
                assert folds == {"fold0", "fold1"}, \
                    "two folds stay separate cells even with equal bins"
            for cell in cells:
                assert cell["outer_folds"] == [cell["outer_fold"]]

    def test_no_automatic_route_promotion(self):
        """Pooled IQR separation and a broken>corrected imbalance can
        NEVER select route A or B — the honest route is always C."""
        rows_imbalanced = [
            _mini_row("c1", 1, 2, group="g", fold="fold0",
                      path_truth={"fixed|2026|RCS": True}),
            _mini_row("b1", 2, 2, group="g", fold="fold0",
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
            _mini_row("b2", 2, 2, group="g", fold="fold0",
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
            _mini_row("b3", 2, 2, group="g", fold="fold0",
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
        ]
        stats_imbalanced = derive_counts(rows_imbalanced)
        assert stats_imbalanced["broken_union"] > \
            stats_imbalanced["corrected_union"]
        for candidate_stats in (derive_counts(
                [_mini_row("a", 1, 1)]), stats_imbalanced):
            route = diag.suggest_route({}, candidate_stats)
            assert route["route"] == "C"
            assert "测量侧主导" not in route["reason"]
            assert "dominant explanation is the measurement"                 not in route["reason"]

    def test_saved_count_disagreement_is_refused(
            self, postprocess_world, module_scratch):
        counts_path = postprocess_world["prior"] / "counts.json"
        original = counts_path.read_bytes()
        try:
            saved = json.loads(original)
            saved["corrected_pairs"] = 99
            counts_path.write_text(json.dumps(saved),
                                   encoding="utf-8")
            out = module_scratch / "post_disagree"
            code = _run_postprocess(module_scratch, postprocess_world,
                                    out)
            assert code == diag.EXIT_REFUSED
            status = json.loads((out / "run_status.json").read_text(
                encoding="utf-8"))
            assert "disagree" in status["failure"]
        finally:
            counts_path.write_bytes(original)

    def test_prior_artifact_drift_is_refused(
            self, postprocess_world, module_scratch):
        target = postprocess_world["prior"] / "binning.json"
        original = target.read_bytes()
        original_pins = _postprocess_pins(postprocess_world)
        try:
            doc = json.loads(original)
            doc["bins"]["c1"] = 8
            target.write_text(json.dumps(doc), encoding="utf-8")
            out = module_scratch / "post_drift"
            code = _run_postprocess(module_scratch, postprocess_world,
                                    out, pinned=original_pins)
            assert code == diag.EXIT_REFUSED
            status = json.loads((out / "run_status.json").read_text(
                encoding="utf-8"))
            assert "sha256" in status["failure"] \
                or "pinned" in status["failure"]
        finally:
            target.write_bytes(original)

    def test_substitute_diagnostic_dir_is_refused(
            self, postprocess_world, module_scratch):
        other = module_scratch / "other_diag"
        shutil.copytree(postprocess_world["prior"], other)
        out = module_scratch / "post_other"
        code = diag.main([
            "--mode", "postprocess",
            "--config", str(_ROOT / diag.PINNED_CONFIG_REL),
            "--diagnostic-dir", str(other),
            "--output-dir", str(out),
            "--budget-minutes", "30",
        ])
        assert code == diag.EXIT_REFUSED

    def test_final_publish_boundary_expiry_is_partial(
            self, postprocess_world, module_scratch):
        """NOT a tiny budget: a fake watch lets every stage complete and
        expires exactly at the final success-publish boundary, after the
        protection review — the run ends PARTIAL with the evidence kept."""
        from core.material_residual_scale import DeadlineExceeded

        class BoundaryWatch:
            def __init__(self, *, budget_minutes=None, deadline_utc=None):
                pass

            def check(self, label):
                if label == "before success status publish":
                    raise DeadlineExceeded(
                        "expired at the final publish boundary")

            def remaining_s(self):
                return 1.0

        out = module_scratch / "post_boundary"
        code = _run_postprocess(
            module_scratch, postprocess_world, out,
            extra_patches=[mock.patch.object(diag, "DeadlineWatch",
                                             BoundaryWatch)])
        assert code == diag.EXIT_PARTIAL
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        assert status["unfinished"]
        completed = {name for name, entry in status["stages"].items()
                     if entry["status"] == "COMPLETED"}
        assert {"postprocess_admission", "ledger_rederivation",
                "binning_recheck", "panel_coverage",
                "layered_statistics", "report",
                "protection"} <= completed
        assert (out / "report.md").is_file()
        assert (out / "route_suggestion.json").is_file()

    def test_full_mode_rejects_postprocess_only_arguments(
            self, diag_world, diag_patches, module_scratch):
        code = diag.main([
            "--config", str(diag_patches["config_path"]),
            "--fit-dir", str(diag_world["fit_dir"]),
            "--verification-dir", str(diag_world["verification"]),
            "--feature-dir", str(diag_world["features"]),
            "--diagnostic-dir", str(module_scratch / "somewhere"),
            "--output-dir", str(module_scratch / "diagroot" / "mixmode"),
            "--budget-minutes", "30",
        ])
        assert code == diag.EXIT_REFUSED


class TestRebuildRecipeWiring:
    def test_rebuild_uses_full_gt_bands_and_the_fix_version(
            self, diag_world, diag_patches):
        """A fake matcher records the recipe wiring: bands come from the
        frame's FULL GT instance set and the config's fix matcher carries
        the fix upsample version — without any real stereo pass."""
        observed = {}

        class FakeResult:
            disparity = np.zeros((96, 96), dtype=np.float32)
            valid = np.zeros((96, 96), dtype=np.uint8)

        class FakeStats:
            valid = False
            reason = "fake"

        class FakeMatcher:
            def __init__(self, config):
                observed["upsample_version"] = \
                    config["fix_inputs"]["upsample_version"]

            def compute_bands(self, left, right, bands,
                              full_image_threshold=None):
                observed["band_count"] = len(bands)
                observed["full_image_threshold"] = full_image_threshold
                return FakeResult()

            def instance_stats(self, result, mask, instance_id=0):
                return FakeStats()

        import csv as _csv2

        from core.gt_geometry import build_gt_geometry
        from scripts.prepare_cls_fusion_dataset import load_gray

        # the census's own pair-manifest reading keeps every column
        # (read_pair_manifest keeps only left_path)
        pair_manifest = {}
        with (diag_world["source"] / "pair_manifest.csv").open(
                encoding="utf-8-sig", newline="") as handle:
            for row in _csv2.DictReader(handle):
                pair_manifest[str(row["output_stem"]).strip()] = row
        frames = sorted({s["source_frame"]
                         for s in diag_world["cache"]["manifest"]})
        rows = [s for s in diag_world["cache"]["manifest"]
                if s["source_frame"] == frames[0]]
        import csv as _csv

        with (diag_world["v5"] / "dataset_manifest.csv").open(
                encoding="utf-8-sig", newline="") as handle:
            v5_rows = {r["sample_name"]: r
                       for r in _csv.DictReader(handle)}
        merged = []
        for row in rows:
            merged.append({**row, **{k: v for k, v in
                                     v5_rows[row["sample_name"]].items()
                                     if k in ("crop_x1", "crop_y1",
                                              "crop_x2", "crop_y2",
                                              "left_path", "label_path")}})
        registry = diag.SourceRegistry({})
        ctx = {"registry": registry, "watch": None,
               "stage": lambda *a, **k: None,
               "log": lambda message: None}
        real_build_bands = diag.build_horizontal_bands
        band_bboxes = []

        def recording_build_bands(bboxes, **kwargs):
            band_bboxes.extend(bboxes)
            return real_build_bands(bboxes, **kwargs)

        with mock.patch.object(diag, "StereoMatcher", FakeMatcher), \
                mock.patch.object(diag, "build_horizontal_bands",
                                  recording_build_bands), \
                mock.patch.object(diag, "compute_polar_features",
                                  lambda *a, **k: None):
            result = diag.rebuild_frame(
                ctx, diag_world["cfg"],
                FakeMatcher(diag_world["cfg"]),
                merged[0]["source_frame"], merged, pair_manifest,
                {merged[0]["sample_name"]})
        assert observed["upsample_version"] == \
            "valid_weighted_legacy_support_v1"
        first = merged[0]
        left = load_gray(_ROOT / first["left_path"])
        height, width = left.shape[:2]
        geometry = build_gt_geometry(
            _ROOT / first["label_path"], width, height, padding=10)
        # the band input covers EVERY GT instance of the frame
        assert len(band_bboxes) == len(geometry) == len(merged)
        assert observed["band_count"] >= 1
        assert list(result["diagnostics"]) == [merged[0]["sample_name"]]


class TestThreeGroupComparisons:
    """All three confusion groups must produce comparisons (nothing
    overwritten), the global list must equal the concatenation, and Q1
    must filter to the fish pair only."""

    @staticmethod
    def _three_group_rows():
        # three corrected instances, one per confusion group, each with
        # an always-wrong partner in the SAME group/fold/bin
        rows = [
            # group 1 (plastic_submarine 1 <-> plastic_fish 2): corrected
            _mini_row("c_g1", 1, 2, group="gA", fold="fold0",
                      path_truth={"fixed|2026|RCS": True}),
            _mini_row("w_g1", 1, 2, group="gA", fold="fold0",
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
            # group 2 (plastic_fish 2 <-> real_fish 3): corrected
            _mini_row("c_g2", 2, 3, group="gA", fold="fold0",
                      path_truth={"fixed|2026|RCS": True}),
            _mini_row("w_g2", 2, 3, group="gA", fold="fold0",
                      path_truth={k: False for k in
                                  [f"{a}|{s}|RCS" for a in ARMS
                                   for s in TRAIN_SEEDS]}),
            # group 3 (metal_submarine 0 <-> plastic_submarine 1): the
            # class-vs-class comparison has support even without a
            # corrected instance
            _mini_row("k_g3", 0, 0, group="gA", fold="fold0"),
            _mini_row("p_g3", 1, 1, group="gA", fold="fold0"),
        ]
        return rows

    def test_all_three_groups_produce_comparisons(self):
        rows = self._three_group_rows()
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        analysis = build_conditional_analysis(rows, stats, bin_of)
        per_group = analysis["comparisons_by_group"]
        assert set(per_group) == {
            "plastic_submarine_vs_plastic_fish",
            "plastic_fish_vs_real_fish",
            "metal_submarine_vs_plastic_submarine"}
        assert all(per_group[g] for g in per_group), \
            "every confusion group produced comparisons"
        glob = analysis["same_group_comparisons"]
        assert len(glob) == sum(len(v) for v in per_group.values())
        # mechanical conservation: global = sum(per-group)
        check = analysis["comparison_count_check"]
        assert check["global"] == sum(
            check["per_confusion_group"].values())
        # the first two groups have comparable corrected/always-wrong
        # cells (nothing lost from the earlier groups)
        for gname in ("plastic_submarine_vs_plastic_fish",
                      "plastic_fish_vs_real_fish"):
            comparable = [c for c in per_group[gname]
                          if c["comparison_type"] ==
                          "corrected_vs_always_wrong" and c["comparable"]]
            assert comparable, f"{gname} lost its comparable cells"

    def test_comparison_fields_and_unique_ids(self):
        rows = self._three_group_rows()
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        analysis = build_conditional_analysis(rows, stats, bin_of)
        glob = analysis["same_group_comparisons"]
        ids = [c["comparison_id"] for c in glob]
        assert len(ids) == len(set(ids)), "comparison_id must be unique"
        for c in glob:
            assert c["confusion_group"]
            assert c["comparison_type"] in (
                "corrected_vs_always_wrong", "broken_vs_stable_correct",
                "class_vs_class")
            assert "comparison_id" in c
            if c["comparison_type"] == "class_vs_class":
                assert "first_class" in c and "second_class" in c
                assert "mirror_note" in c
            else:
                for key in ("true_class", "group", "outer_fold", "bin"):
                    assert key in c

    def test_q1_filters_to_the_fish_pair_only(self):
        rows = self._three_group_rows()
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        analysis = build_conditional_analysis(rows, stats, bin_of)
        q1 = q1_fish_pair_common_support(analysis)
        assert q1["confusion_group"] == "plastic_fish_vs_real_fish"
        assert q1["comparison_type"] == "corrected_vs_always_wrong"
        # group 1's comparable corrected cell must NOT leak into Q1
        assert all(c["true_class"] in ("plastic_fish", "real_fish")
                   for c in q1["cell_table"])
        assert q1["cells_with_both_sides"] >= 1
        assert q1["cells_with_both_sides"] < len(
            analysis["same_group_comparisons"])


class TestCommonSupportVsCommonComponent:
    def test_both_sides_without_common_valid_component(self):
        # corrected side: interior missing; always-wrong side: band
        # missing -> both sides present, ZERO common valid components
        corrected = _mini_row(
            "c1", 2, 3, group="g", fold="fold0",
            path_truth={"fixed|2026|RCS": True},
            s_features=[None] * 8 + [0.4] * 8)
        always = _mini_row(
            "w1", 2, 3, group="g", fold="fold0",
            path_truth={k: False for k in
                        [f"{a}|{s}|RCS" for a in ARMS
                         for s in TRAIN_SEEDS]},
            s_features=[0.2] * 8 + [None] * 8)
        rows = [corrected, always]
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        analysis = build_conditional_analysis(rows, stats, bin_of)
        cells = [c for c in analysis["same_group_comparisons"]
                 if c.get("comparison_type") == "corrected_vs_always_wrong"
                 and c["comparable"]]
        assert len(cells) == 1
        assert cells[0]["n_common_valid_components"] == 0
        # the S16 gaps are empty; only the C0/C3 descriptive controls
        # (both sides valid) may appear
        assert not [k for k in cells[0]["median_gaps"]
                    if k.startswith(("interior.", "band."))]

    def test_full_overlap_counts_sixteen_components(self):
        corrected = _mini_row(
            "c1", 2, 3, group="g", fold="fold0",
            path_truth={"fixed|2026|RCS": True})
        always = _mini_row(
            "w1", 2, 3, group="g", fold="fold0",
            path_truth={k: False for k in
                        [f"{a}|{s}|RCS" for a in ARMS
                         for s in TRAIN_SEEDS]})
        rows = [corrected, always]
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        analysis = build_conditional_analysis(rows, stats, bin_of)
        cells = [c for c in analysis["same_group_comparisons"]
                 if c.get("comparison_type") == "corrected_vs_always_wrong"
                 and c["comparable"]]
        assert len(cells) == 1
        assert cells[0]["n_common_valid_components"] == 16


class TestFiveLayerSubsets:
    def _world_rows(self):
        world = _MiniWorld()
        return world.rows, world.independent_counts

    def test_all_layers_all_subsets_written_with_conservation(self):
        rows, _independent = self._world_rows()
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        layered = subset_layered_analysis(rows, stats, bin_of)
        assert layered["conservation_checks"]["verdict"] == "PASS", \
            layered["conservation_checks"]["problems"][:3]
        for gname, group in layered["groups"].items():
            assert set(group["subsets"]) == {
                "all_instances_of_the_pair", "r0_errors_in_pair",
                "r0_correct_in_pair", "corrected", "broken",
                "always_wrong", "stable_correct"}
            for subset_label, subset in group["subsets"].items():
                assert set(subset["layers"]) == {
                    "L1_true_class", "L2_class_x_fold",
                    "L3_class_x_group_keeps_fold",
                    "L4_class_x_fold_x_bin",
                    "L5_class_x_group_x_fold_x_bin"}
                for level, cells in subset["layers"].items():
                    for cell in cells:
                        assert cell["level"] == level
                        assert cell["subset"] == subset_label
                        assert cell["confusion_group"] == gname

    def test_subset_member_counts_are_hand_derived(self):
        # independent expectations from the scenario definitions:
        # classes are c1=2, c2=2, b1=1, w1=3, rc1=1, n1=0
        rows, _independent = self._world_rows()
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        layered = subset_layered_analysis(rows, stats, bin_of)
        expected_unique = {
            ("plastic_submarine_vs_plastic_fish",
             "all_instances_of_the_pair"): 4,
            ("plastic_fish_vs_real_fish",
             "all_instances_of_the_pair"): 3,
            ("metal_submarine_vs_plastic_submarine",
             "all_instances_of_the_pair"): 3,
            ("plastic_submarine_vs_plastic_fish", "corrected"): 0,
            ("plastic_fish_vs_real_fish", "corrected"): 2,
            ("metal_submarine_vs_plastic_submarine", "corrected"): 0,
            ("plastic_submarine_vs_plastic_fish", "broken"): 1,
            ("plastic_fish_vs_real_fish", "broken"): 0,
            ("metal_submarine_vs_plastic_submarine", "broken"): 1,
            ("plastic_fish_vs_real_fish", "always_wrong"): 0,
            ("plastic_submarine_vs_plastic_fish", "stable_correct"): 1,
        }
        for (gname, subset), expected in expected_unique.items():
            got = layered["groups"][gname]["subsets"][subset][
                "unique_members"]
            assert got == expected, f"{gname}/{subset}: {got} != {expected}"

    def test_imbalanced_group_median_comes_from_members(self):
        # 1 value 0.0 in bin 1, 9 values 0.8 in bin 2, same class/fold/
        # group: the L2 median must be 0.8 from ALL TEN members, never
        # the 0.4 average of the two cell medians
        rows = [_mini_row("single", 2, 2, group="g", fold="fold0",
                          s_features=[0.0] * 8 + [0.0] * 8)]
        for i in range(9):
            rows.append(_mini_row(f"many{i}", 2, 2, group="g",
                                  fold="fold0",
                                  s_features=[0.8] * 8 + [0.8] * 8))
        stats = derive_counts(rows)
        bin_of = {"single": 1}
        bin_of.update({f"many{i}": 2 for i in range(9)})
        layered = subset_layered_analysis(rows, stats, bin_of)
        subsets = layered["groups"]["plastic_submarine_vs_plastic_fish"][
            "subsets"]
        l2_cells = subsets["all_instances_of_the_pair"]["layers"][
            "L2_class_x_fold"]
        assert len(l2_cells) == 1
        assert l2_cells[0]["support"] == 10
        # the independent expectation, computed from the raw values
        raw = [0.0] + [0.8] * 9
        assert l2_cells[0]["band"]["signed_q_mean"]["median"] == \
            pytest.approx(float(np.median(raw))) == pytest.approx(0.8)
        l4_cells = subsets["all_instances_of_the_pair"]["layers"][
            "L4_class_x_fold_x_bin"]
        by_bin = {c["bin"]: c["band"]["signed_q_mean"]["median"]
                  for c in l4_cells}
        assert by_bin[1] == pytest.approx(0.0)
        assert by_bin[2] == pytest.approx(0.8)

    def test_corrected_subset_stats_differ_from_all_instances(self):
        # the corrected member carries a distinct S level; the corrected
        # subset's L1 median must reflect ITS member, not the pair pool
        rows = [
            _mini_row("c1", 2, 3, group="g", fold="fold0",
                      path_truth={"fixed|2026|RCS": True},
                      s_features=[0.9] * 8 + [0.9] * 8),
            _mini_row("plain1", 2, 2, group="g", fold="fold0",
                      s_features=[0.1] * 8 + [0.1] * 8),
            _mini_row("plain2", 2, 2, group="g", fold="fold0",
                      s_features=[0.2] * 8 + [0.2] * 8),
        ]
        stats = derive_counts(rows)
        bin_of = {r["sample_name"]: 0 for r in rows}
        layered = subset_layered_analysis(rows, stats, bin_of)
        subsets = layered["groups"]["plastic_fish_vs_real_fish"]["subsets"]
        corrected_l1 = subsets["corrected"]["layers"]["L1_true_class"]
        all_l1 = subsets["all_instances_of_the_pair"]["layers"][
            "L1_true_class"]
        assert len(corrected_l1) == 1
        assert corrected_l1[0]["support"] == 1
        assert corrected_l1[0]["band"]["signed_q_mean"]["median"] == \
            pytest.approx(0.9)
        all_by_class = {c["true_class"]: c for c in all_l1}
        assert all_by_class["plastic_fish"]["band"]["signed_q_mean"][
            "median"] == pytest.approx(
            float(np.median([0.9, 0.1, 0.2])))
        assert corrected_l1[0]["band"]["signed_q_mean"]["median"] != \
            all_by_class["plastic_fish"]["band"]["signed_q_mean"]["median"]
