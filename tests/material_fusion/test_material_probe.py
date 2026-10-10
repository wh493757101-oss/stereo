"""Synthetic tests for the probe feature contract (no real data, no fit).

The training-loop tests run on tiny synthetic design matrices in CPU
isolation; they never touch project data or real weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    NUM_CLASSES,
    POLAR_STAT_KEYS,
    Standardizer,
    accuracy,
    assemble_design_matrix,
    build_control_features,
    build_polar_stats,
    build_probe_head,
    compare_module_states,
    macro_f1,
    paired_changes,
    select_epoch,
    slot_masks,
    snapshot_module_state,
    total_input_dim,
    train_linear_probe,
)

from .conftest import (
    CLASS_NAMES,
    build_synthetic_source,
    build_synthetic_v4,
    build_synthetic_v5,
)


def _sample_arrays(n: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 255, size=(n, n), dtype=np.uint8)
    instance_mask = np.zeros((n, n), dtype=np.uint8)
    instance_mask[2:6, 2:6] = 1
    valid = instance_mask.copy()
    valid[2, 2] = 0  # one pixel of the mask is not measurable
    signed_q = np.zeros((n, n), dtype=np.float32)
    abs_q = np.zeros((n, n), dtype=np.float32)
    values = rng.uniform(-0.5, 0.5, size=signed_q.shape).astype(np.float32)
    signed_q[valid > 0] = values[valid > 0]
    abs_q = np.abs(signed_q)
    quality = np.asarray([0.875, 1.0, 1.0, 0.0625], dtype=np.float32)
    return gray, instance_mask, valid, signed_q, abs_q, quality


class TestBuildControlFeatures:
    def test_reads_quality_first_three_only(self):
        gray, mask, valid, signed, abs_q, quality = _sample_arrays()
        s_nonempty = True
        base = build_control_features(
            quality=quality, gray=gray, instance_mask=mask,
            s_region_nonempty=s_nonempty)
        perturbed = quality.copy()
        perturbed[3] = 0.999  # mean_abs_q is polarization content
        changed = build_control_features(
            quality=perturbed, gray=gray, instance_mask=mask,
            s_region_nonempty=s_nonempty)
        assert np.array_equal(base, changed)

    def test_quality_vector_shape_refused(self):
        with pytest.raises(ValueError, match="quality"):
            build_control_features(
                quality=np.zeros(3, dtype=np.float32),
                gray=np.zeros((4, 4), dtype=np.uint8),
                instance_mask=np.ones((4, 4), dtype=np.uint8),
                s_region_nonempty=True)

    def test_gray_statistics_inside_mask(self):
        gray, mask, *_ = _sample_arrays()
        features = build_control_features(
            quality=np.zeros(4, dtype=np.float32), gray=gray,
            instance_mask=mask, s_region_nonempty=False)
        pixels = gray[mask > 0].astype(np.float64) / 255.0
        assert features[3] == pytest.approx(pixels.mean(), abs=1e-6)
        assert features[4] == pytest.approx(pixels.std(ddof=0), abs=1e-6)
        assert features[5] == 0.0

    def test_empty_mask_rule_is_finite(self):
        features = build_control_features(
            quality=np.zeros(4, dtype=np.float32),
            gray=np.zeros((4, 4), dtype=np.uint8),
            instance_mask=np.zeros((4, 4), dtype=np.uint8),
            s_region_nonempty=False)
        assert features[3] == 0.0 and features[4] == 0.0


class TestBuildPolarStats:
    def test_region_is_instance_and_valid_only(self):
        gray, mask, valid, signed, abs_q, _ = _sample_arrays(seed=1)
        # put distinctive values OUTSIDE the measured region: ignored
        signed[0, 0] = 100.0
        abs_q[0, 0] = 100.0
        values, has_valid = build_polar_stats(
            signed_q=signed, abs_q=abs_q, valid=valid,
            instance_mask=mask)
        region = (valid > 0) & (mask > 0)
        expected = signed[region]
        assert has_valid
        assert values[0] == pytest.approx(float(expected.mean()), abs=1e-6)
        assert values[1] == pytest.approx(
            float(expected.std(ddof=0)), abs=1e-6)
        assert values[-1] == pytest.approx(
            float(np.abs(expected).mean()), abs=1e-6)

    def test_quantiles_ordered_and_labels_match(self):
        _, mask, valid, signed, abs_q, _ = _sample_arrays(seed=2)
        values, has_valid = build_polar_stats(
            signed_q=signed, abs_q=abs_q, valid=valid, instance_mask=mask)
        assert has_valid
        p10, p25, p50, p75, p90 = values[2:7]
        assert p10 <= p25 <= p50 <= p75 <= p90
        region = (valid > 0) & (mask > 0)
        expected_p50 = float(np.percentile(signed[region], 50.0,
                                           method="linear"))
        assert p50 == pytest.approx(expected_p50, abs=1e-6)

    def test_empty_region_is_explicit_missing(self):
        values, has_valid = build_polar_stats(
            signed_q=np.zeros((4, 4), dtype=np.float32),
            abs_q=np.zeros((4, 4), dtype=np.float32),
            valid=np.zeros((4, 4), dtype=np.uint8),
            instance_mask=np.ones((4, 4), dtype=np.uint8))
        assert not has_valid
        assert np.isnan(values).all()
        assert len(values) == len(POLAR_STAT_KEYS)

    def test_valid_outside_mask_is_not_measured(self):
        valid = np.ones((4, 4), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        signed = np.ones((4, 4), dtype=np.float32) * 0.5
        values, has_valid = build_polar_stats(
            signed_q=signed, abs_q=np.abs(signed), valid=valid,
            instance_mask=mask)
        assert not has_valid  # valid ⊄ instance_mask contributes nothing

    def test_non_finite_inside_region_refused(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        valid = np.ones((4, 4), dtype=np.uint8)
        signed = np.zeros((4, 4), dtype=np.float32)
        signed[1, 1] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            build_polar_stats(signed_q=signed,
                              abs_q=np.zeros((4, 4), dtype=np.float32),
                              valid=valid, instance_mask=mask)


class TestSlotAssembly:
    def test_unused_slots_are_exactly_zero(self):
        g = np.arange(1, 13, dtype=np.float32).reshape(2, 6)  # no zeros
        c = np.full((2, 6), 2.0, dtype=np.float32)
        s = np.full((2, 8), 3.0, dtype=np.float32)
        g_dim = 6
        for variant, expected_nonzero in {
            "P0": g_dim, "P1": g_dim + 6, "P2": g_dim + 6 + 8,
        }.items():
            matrix = assemble_design_matrix(g, c, s, variant)
            assert matrix.shape == (2, total_input_dim(g_dim))
            nonzero_rows = np.count_nonzero(matrix[0])
            assert nonzero_rows == expected_nonzero
        p0 = assemble_design_matrix(g, c, s, "P0")
        assert np.array_equal(p0[:, g_dim:], np.zeros((2, 14)))

    def test_mask_applied_by_default(self):
        g = np.ones((1, 4), dtype=np.float32)
        c = np.ones((1, 6), dtype=np.float32)
        s = np.ones((1, 8), dtype=np.float32)
        p1 = assemble_design_matrix(g, c, s, "P1")
        assert p1[:, 4:10].all()
        assert not p1[:, 10:].any()
        p2 = assemble_design_matrix(g, c, s, "P2")
        assert p2[:, 10:].all()

    def test_sample_count_mismatch_refused(self):
        with pytest.raises(ValueError, match="sample count"):
            assemble_design_matrix(
                np.zeros((2, 4), dtype=np.float32),
                np.zeros((3, 6), dtype=np.float32),
                np.zeros((2, 8), dtype=np.float32), "P2")

    def test_unknown_variant_refused(self):
        with pytest.raises(ValueError, match="unknown probe variant"):
            assemble_design_matrix(
                np.zeros((1, 4), dtype=np.float32),
                np.zeros((1, 6), dtype=np.float32),
                np.zeros((1, 8), dtype=np.float32), "P9")

    def test_surviving_nan_refused(self):
        s = np.full((1, 8), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN/Inf"):
            assemble_design_matrix(
                np.zeros((1, 4), dtype=np.float32),
                np.zeros((1, 6), dtype=np.float32), s, "P2")


class TestStandardizer:
    def test_fit_uses_only_valid_rows(self):
        matrix = np.array([[1.0, 10.0], [3.0, 30.0], [1000.0, 20.0]])
        column_valid = np.array([
            [True, True], [True, True], [False, True],  # row 2 excluded
        ])
        scaler = Standardizer.fit(matrix, column_valid)
        assert scaler.mean[0] == pytest.approx(2.0)
        assert scaler.mean[1] == pytest.approx(20.0)

    def test_zero_variance_column_outputs_zero(self):
        matrix = np.array([[5.0], [5.0], [5.0]])
        column_valid = np.ones((3, 1), dtype=bool)
        scaler = Standardizer.fit(matrix, column_valid)
        assert not scaler.usable[0]
        transformed = scaler.transform(matrix, column_valid)
        assert np.array_equal(transformed, np.zeros((3, 1)))

    def test_no_valid_rows_column_outputs_zero(self):
        matrix = np.array([[np.nan], [1.0]])
        column_valid = np.array([[False], [False]])
        scaler = Standardizer.fit(matrix, column_valid)
        assert not scaler.usable[0]
        transformed = scaler.transform(matrix, column_valid)
        assert np.array_equal(transformed, np.zeros((2, 1)))

    def test_missing_rows_transform_to_zero(self):
        matrix = np.array([[1.0], [3.0], [np.nan]])
        column_valid = np.array([[True], [True], [False]])
        scaler = Standardizer.fit(matrix, column_valid)
        transformed = scaler.transform(matrix, column_valid)
        assert transformed[2, 0] == 0.0
        assert transformed[0, 0] < 0.0 < transformed[1, 0]

    def test_fit_refuses_nan_in_valid_positions(self):
        matrix = np.array([[np.nan], [1.0]])
        column_valid = np.array([[True], [True]])
        with pytest.raises(ValueError, match="NaN/Inf"):
            Standardizer.fit(matrix, column_valid)

    def test_transform_does_not_mutate(self):
        matrix = np.array([[1.0], [3.0]])
        scaler = Standardizer.fit(matrix, np.ones((2, 1), dtype=bool))
        before = scaler.mean.copy()
        scaler.transform(np.array([[100.0], [900.0]]),
                         np.ones((2, 1), dtype=bool))
        assert scaler.mean[0] == before[0]

    def test_column_count_mismatch_refused(self):
        scaler = Standardizer.fit(np.zeros((2, 2)),
                                  np.ones((2, 2), dtype=bool))
        with pytest.raises(ValueError, match="column count"):
            scaler.transform(np.zeros((2, 3)), np.ones((2, 3), dtype=bool))

    def test_dict_round_trip(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaler = Standardizer.fit(matrix, np.ones((2, 2), dtype=bool))
        restored = Standardizer.from_dict(scaler.to_dict())
        assert np.array_equal(restored.mean, scaler.mean)
        assert np.array_equal(restored.usable, scaler.usable)


class TestMetrics:
    def test_macro_f1_fixed_class_set(self):
        # class 3 never predicted and never true -> F1 0, included in mean
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 1]
        f1s = macro_f1(y_true, y_pred, num_classes=4)
        # class1: tp2 fp1 fn0 -> F1 0.8; class2: tp1 fp0 fn1 -> F1 2/3
        per = [1.0, 0.8, 2 / 3, 0.0]
        assert f1s == pytest.approx(float(np.mean(per)))

    def test_accuracy(self):
        assert accuracy([0, 1, 2, 3], [0, 1, 2, 0]) == 0.75
        assert accuracy([], []) == 0.0

    def test_select_epoch_tie_prefers_earlier(self):
        history = [
            {"epoch": 1, "inner_val_macro_f1": 0.5, "inner_val_accuracy": 0.6},
            {"epoch": 2, "inner_val_macro_f1": 0.7, "inner_val_accuracy": 0.6},
            {"epoch": 3, "inner_val_macro_f1": 0.7, "inner_val_accuracy": 0.6},
            {"epoch": 4, "inner_val_macro_f1": 0.7, "inner_val_accuracy": 0.9},
        ]
        assert select_epoch(history)["epoch"] == 4
        history2 = history[:3]
        assert select_epoch(history2)["epoch"] == 2

    def test_paired_changes(self):
        result = paired_changes(
            [True, True, False, False, True],
            [True, False, True, False, True],
        )
        assert result["fixed"] == 1
        assert result["broke"] == 1
        assert result["both_correct"] == 2
        assert result["both_wrong"] == 1
        with pytest.raises(ValueError, match="equal length"):
            paired_changes([True], [])


class TestTrainLinearProbe:
    def test_separable_synthetic_data(self):
        rng = np.random.default_rng(0)
        centers = np.diag(np.full(NUM_CLASSES, 8.0))
        X = np.concatenate(
            [centers[c] + rng.normal(size=(40, NUM_CLASSES))
             for c in range(NUM_CLASSES)]
        ).astype(np.float32)
        y = np.repeat(np.arange(NUM_CLASSES), 40)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]  # every class on both sides of the split
        X_val, y_val = X[:40], y[:40]
        X_train, y_train = X[40:], y[40:]
        assert len(set(y_train.tolist())) == NUM_CLASSES
        model, history, best_state, best = train_linear_probe(
            X_train, y_train, X_val, y_val,
            seed=2026, lr=0.01, weight_decay=0.001, max_epochs=30,
        )
        assert len(history) == 30
        assert history[0]["epoch"] == 1
        assert best["epoch"] == select_epoch(history)["epoch"]
        # best_state reproduces the selected epoch's metrics
        with torch.no_grad():
            model.load_state_dict(best_state)
            pred = model(torch.from_numpy(X_val)).argmax(1).numpy()
        assert macro_f1(y_val, pred) == pytest.approx(
            best["inner_val_macro_f1"])
        assert best["inner_val_accuracy"] > 0.9

    def test_two_class_inner_train_refused(self):
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.int64)
        with pytest.raises(ValueError, match="fewer than two classes"):
            train_linear_probe(X, y, X, y, seed=2026, lr=0.01,
                               weight_decay=0.001, max_epochs=3)

    def test_invalid_budget_refused(self):
        X = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
        y = np.arange(10) % 4
        with pytest.raises(ValueError, match="max_epochs"):
            train_linear_probe(X, y, X, y, seed=2026, lr=0.01,
                               weight_decay=0.001, max_epochs=0)

    def test_head_initialization_is_seed_deterministic(self):
        a = build_probe_head(270, seed=2026)
        b = build_probe_head(270, seed=2026)
        for (name_a, tensor_a), (name_b, tensor_b) in zip(
                a.named_parameters(), b.named_parameters()):
            assert name_a == name_b
            assert torch.equal(tensor_a, tensor_b)


class TestModuleStateSnapshot:
    def test_eval_forward_leaves_state_unchanged(self):
        module = nn.Linear(6, 4)
        before = snapshot_module_state(module)
        module.eval()
        with torch.no_grad():
            module(torch.zeros(3, 6))
        after = snapshot_module_state(module)
        assert compare_module_states(before, after)["unchanged"]

    def test_optimizer_step_is_detected(self):
        module = nn.Linear(6, 4)
        before = snapshot_module_state(module)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        optimizer.zero_grad()
        module(torch.ones(3, 6)).sum().backward()
        optimizer.step()
        after = snapshot_module_state(module)
        report = compare_module_states(before, after)
        assert not report["unchanged"]
        assert report["changed_tensors"]

    def test_structure_change_is_detected(self):
        before = snapshot_module_state(nn.Linear(6, 4))
        after = snapshot_module_state(nn.Linear(6, 5))
        report = compare_module_states(before, after)
        assert not report["unchanged"]
        assert report["changed_tensors"]  # same names, different shapes


# ---------------------------------------------------------------------------
# prepare/train CLI over a full synthetic V5 chain (no real data/weights)


def _write_chain_config_with_probe(chain, path: Path, max_epochs: int = 8) -> Path:
    """An experiment config (with the probe budget) bound to a synthetic chain."""
    import yaml

    from core.fusion_dataset import dataset_fingerprint
    from core.material_mask_dataset import mask_dataset_fingerprint

    root = chain["project_root"]
    config = {
        "experiment": "material_existing_data_v1",
        "data": {
            "root": chain["v5"].relative_to(root).as_posix(),
            "fingerprint": mask_dataset_fingerprint(chain["v5"]),
            "parent_v4_fingerprint": dataset_fingerprint(chain["v4"]),
            "dev_pool_splits": ["train", "val"],
            "excluded_splits": ["test"],
            "split_assignment": chain["split_assignment"].relative_to(root).as_posix(),
            "label_root": (chain["source"] / "labels").relative_to(root).as_posix(),
            "class_names": ["metal_submarine", "plastic_submarine",
                            "plastic_fish", "real_fish"],
        },
        "baselines": {
            "model_a": {"path": "weights/model_a.pt",
                        "sha256": chain["weights"]["model_a.pt"][1]},
            "e1_candidate": {"path": "weights/e1.pt",
                             "sha256": chain["weights"]["e1.pt"][1]},
            "e4_candidate": {"path": "weights/e4.pt",
                             "sha256": chain["weights"]["e4.pt"][1]},
            "independent_gray_fallback": {"path": "weights/fb.pt",
                                          "sha256": chain["weights"]["fb.pt"][1]},
            "official_mit_b0": {"path": "weights/mit_b0.pth",
                                "sha256": chain["weights"]["mit_b0.pth"][1]},
        },
        "protocol": {"n_outer_folds": 3, "seed": 2026,
                     "inner_val_fraction": 0.25, "inner_seed_base": 2026},
        "detection": {"match_iou_threshold": 0.50},
        "features": {
            "imgsz": 224,
            "preprocessing_version": "v4_band_224_area_linear_v1",
            "gray_input_mode": "gray replicated to [G,G,G]",
            "encoder": "stub encoder (synthetic tests only)",
            "pooling": "instance_mask weighted mean on the last stage",
            "pooling_background_note": "test",
            "feature_definition_version": "existing_data_probe_v1",
        },
        "probe": {
            "variants": {"P0": {"slots": ["G"]}, "P1": {"slots": ["G", "C"]},
                         "P2": {"slots": ["G", "C", "S"]}},
            "head": "linear_softmax_4class",
            "seed": 2026,
            "optimizer": "adamw",
            "lr": 0.01,
            "weight_decay": 0.001,
            "max_epochs": max_epochs,
            "device": "cpu",
            "standardization": "per_column_inner_train_only",
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class _StubEncoder(nn.Module):
    """A tiny deterministic stand-in for the frozen MiT-B0 encoder."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x):
        features = self.pool(self.conv(x))
        return [features] * 4  # MiTEncoder returns four levels


def _stub_probe_encoder(config, project_root):
    return _StubEncoder().eval(), {
        "loaded": 1, "missing": [], "unexpected": [], "random_init": [],
    }


@pytest.fixture(scope="module")
def probe_chain(module_scratch):
    """synthetic source -> V4 -> V5 -> split_assignment -> weights/config."""
    import csv as csv_module

    from core.fusion_dataset import file_digest
    from .test_existing_data_protocol import _write_chain_config

    workspace = module_scratch.resolve()
    project_root = Path(__file__).resolve().parents[2]
    source = build_synthetic_source(
        module_scratch / "source", counts=(6, 3, 2), size=64,
        objects_per_frame=4,
    )
    v4 = build_synthetic_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)
    split_assignment = source / "split_assignment.csv"
    with split_assignment.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=[
            "output_stem", "split", "group_name", "block_id",
            "source_json", "left_path", "right_path"])
        writer.writeheader()
        for split, count in (("train", 6), ("val", 3), ("test", 2)):
            for index in range(count):
                stem = f"{split}_frame_{index:03d}"
                writer.writerow({
                    "output_stem": stem, "split": split,
                    "group_name": f"{split}/group_{index}", "block_id": "0",
                    "source_json": "",
                    "left_path": (source / "images" / split / f"{stem}.png").as_posix(),
                    "right_path": (source / "images_right" / split / f"{stem}.png").as_posix(),
                })
    weights = {}
    for name, payload in (
        ("model_a.pt", b"model-a"), ("e1.pt", b"e1"), ("e4.pt", b"e4"),
        ("fb.pt", b"fallback"), ("mit_b0.pth", b"mit"),
    ):
        path = module_scratch / "weights" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        weights[name] = (path, file_digest(path))
    config_path = _write_chain_config(
        {"v5": v5, "v4": v4, "source": source, "split_assignment": split_assignment,
         "weights": weights, "project_root": project_root},
        module_scratch / "config.yaml",
    )
    return {
        "workspace": workspace,
        "source": source,
        "v4": v4,
        "v5": v5,
        "weights": weights,
        "split_assignment": split_assignment,
        "project_root": project_root,
        "config_path": config_path,
    }


@pytest.fixture(scope="module")
def probe_protocol_dir(probe_chain, module_scratch):
    """A real protocol built over the synthetic chain via the protocol CLI."""
    from scripts.prepare_existing_data_protocol import main as protocol_main

    output = module_scratch / "protocol"
    code = protocol_main([
        "--config", str(probe_chain["config_path"]),
        "--output-dir", str(output),
        "--project-root", str(probe_chain["project_root"]),
    ])
    assert code == 0, (output / "protocol.json").exists()
    return output


@pytest.fixture(scope="module")
def probe_feature_dir(probe_chain, probe_protocol_dir, module_scratch):
    """The feature cache built by the prepare CLI with a stub encoder."""
    import pytest as _pytest

    from scripts.prepare_material_probe import main as prepare_main

    mp = _pytest.MonkeyPatch()
    mp.setattr("scripts.prepare_material_probe.build_probe_encoder",
               _stub_probe_encoder)
    try:
        output = module_scratch / "features"
        code = prepare_main([
            "--config", str(probe_chain["config_path"]),
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--output-dir", str(output),
            "--device", "cpu",
        ])
        assert code == 0
    finally:
        mp.undo()
    return output


class TestPrepareMaterialProbeCli:
    def test_cache_written_and_bound(self, probe_feature_dir):
        definition = json.loads(
            (probe_feature_dir / "feature_definition.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (probe_feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        report = json.loads(
            (probe_feature_dir / "report.json").read_text(encoding="utf-8"))
        assert report["training_status"] == "NOT_RUN"
        assert report["status"] == "COMPLETED"
        cache = np.load(probe_feature_dir / "feature_cache.npz")
        assert cache["g_features"].shape == (manifest["count"], 16)
        assert cache["c_features"].shape[1] == 6
        assert cache["s_features"].shape[1] == 8
        assert manifest["count"] == 36  # 9 groups x 4 objects
        assert definition["data_fingerprint"]
        assert definition["protocol_sha256"]
        assert (probe_feature_dir / "polar_quality_summary.json").is_file()
        assert (probe_feature_dir / "brightness_vs_valid_ratio.png").is_file()
        roles = manifest["samples"][0]["fold_roles"]
        assert set(roles.values()) <= {"outer_eval", "inner_train", "inner_val"}
        assert "outer_eval" in roles.values()

    def test_encoder_state_check_recorded(self, probe_feature_dir):
        definition = json.loads(
            (probe_feature_dir / "feature_definition.json").read_text(encoding="utf-8"))
        assert definition["extraction"]["state_check"]["unchanged"] is True
        assert "inference_mode" in definition["extraction"]["forward_mode"]

    def test_existing_output_refused(self, probe_feature_dir, probe_chain,
                                     probe_protocol_dir):
        from scripts.prepare_material_probe import main as prepare_main

        code = prepare_main([
            "--config", str(probe_chain["config_path"]),
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--output-dir", str(probe_feature_dir),
        ])
        assert code == 1

    def test_protocol_fingerprint_mismatch_refused(self, probe_chain,
                                                   probe_protocol_dir,
                                                   scratch_dir):
        import yaml as yaml_module

        from scripts.prepare_material_probe import main as prepare_main

        config = yaml_module.safe_load(
            Path(probe_chain["config_path"]).read_text(encoding="utf-8"))
        config["data"]["fingerprint"] = "f" * 64
        bad_config = scratch_dir / "bad_config.yaml"
        bad_config.write_text(yaml_module.safe_dump(config), encoding="utf-8")
        code = prepare_main([
            "--config", str(bad_config),
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--output-dir", str(scratch_dir / "bad_features"),
        ])
        assert code == 1
        assert not (scratch_dir / "bad_features").exists()


class TestBuildSlotFeaturesLeakage:
    """P1 must not read polarization content; P2's S must reflect changes."""

    def _rows_and_tree(self, scratch_dir):
        from core.material_mask_dataset import save_mask_sample

        rng = np.random.default_rng(3)
        data_root = scratch_dir / "v5"
        rows = []
        for index, (split, class_name, class_id) in enumerate([
                ("train", "metal_submarine", 0),
                ("val", "real_fish", 3)]):
            shape = (24, 20)
            instance_mask = np.zeros(shape, dtype=np.uint8)
            instance_mask[4:14, 4:12] = 1
            valid = instance_mask.copy()
            valid[4, 4] = 0
            gray = rng.integers(0, 255, size=shape, dtype=np.uint8)
            signed = np.zeros(shape, dtype=np.float32)
            values = rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)
            signed[valid > 0] = values[valid > 0]
            abs_q = np.abs(signed)
            quality = np.asarray([0.98, 1.0, 1.0, 0.05], dtype=np.float32)
            relative = Path(split) / class_name / f"sample_{index:03d}.npz"
            save_mask_sample(
                data_root / relative, gray=gray, signed_q=signed,
                abs_q=abs_q, valid=valid, quality=quality,
                class_id=class_id, instance_mask=instance_mask)
            rows.append({
                "sample_name": f"sample_{index:03d}",
                "npz_path": relative.as_posix(), "split": split,
                "group_name": "g", "source_frame": f"f{index:03d}",
                "object_index": 0, "class_id": class_id,
                "class_name": class_name,
            })
        return rows, data_root

    @staticmethod
    def _rewrite(data_root, row, **changes):
        from core.material_mask_dataset import load_mask_sample, save_mask_sample

        path = data_root / row["npz_path"]
        sample = load_mask_sample(path)
        save_mask_sample(
            path,
            gray=changes.get("gray", sample.gray),
            signed_q=changes.get("signed_q", sample.signed_q),
            abs_q=changes.get("abs_q", sample.abs_q),
            valid=changes.get("valid", sample.valid),
            quality=changes.get("quality", sample.quality),
            class_id=sample.class_id,
            instance_mask=changes.get("instance_mask",
                                      sample.instance_mask),
        )

    def test_quality_mean_abs_q_never_enters_c(self, scratch_dir):
        from core.material_mask_dataset import load_mask_sample

        from scripts.prepare_material_probe import build_slot_features

        rows, data_root = self._rows_and_tree(scratch_dir)
        c0, s0, v0, _ = build_slot_features(rows, data_root)
        for row in rows:
            sample = load_mask_sample(data_root / row["npz_path"])
            quality = sample.quality.copy()
            quality[3] = 0.987  # mean_abs_q is polarization content
            self._rewrite(data_root, row, quality=quality)
        c1, s1, v1, _ = build_slot_features(rows, data_root)
        assert np.array_equal(c0, c1)   # C untouched by mean_abs_q
        assert np.array_equal(s0, s1)   # S reads the arrays, not the field
        assert np.array_equal(v0, v1)

    def test_signed_q_change_moves_s_not_c(self, scratch_dir):
        from core.material_mask_dataset import load_mask_sample

        from scripts.prepare_material_probe import build_slot_features

        rows, data_root = self._rows_and_tree(scratch_dir)
        c0, s0, v0, _ = build_slot_features(rows, data_root)
        rng = np.random.default_rng(9)
        for row in rows:
            sample = load_mask_sample(data_root / row["npz_path"])
            signed = sample.signed_q.copy()
            flip = sample.valid > 0
            signed[flip] = rng.uniform(-0.9, 0.9, size=int(flip.sum()))
            self._rewrite(data_root, row, signed_q=signed,
                          abs_q=np.abs(signed))
        c1, s1, v1, _ = build_slot_features(rows, data_root)
        assert np.array_equal(c0, c1)   # P1 unchanged by polarization values
        assert not np.array_equal(s0, s1)  # P2's S reflects the change
        assert np.array_equal(v0, v1)


class TestTrainMaterialProbeCli:
    def test_no_mode_prints_help_and_refuses(self, probe_chain,
                                             probe_protocol_dir,
                                             probe_feature_dir, scratch_dir):
        from scripts.train_material_probe import main as train_main

        code = train_main([
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--features", str(probe_feature_dir),
            "--output-dir", str(scratch_dir / "no_mode"),
        ])
        assert code == 1
        assert not (scratch_dir / "no_mode").exists()

    def test_dry_run_does_not_fit(self, probe_chain, probe_protocol_dir,
                                  probe_feature_dir, scratch_dir):
        from scripts.train_material_probe import main as train_main

        output = scratch_dir / "dry_run"
        code = train_main([
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--features", str(probe_feature_dir),
            "--output-dir", str(output),
            "--dry-run",
        ])
        assert code == 0
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["training_status"] == "NOT_RUN"
        assert report["classification_gain"] == "NOT_MEASURED"
        assert report["fold_status"] == "OK"
        assert not (output / "models").exists()
        assert len(report["preview"]) == 3

    def test_fit_on_synthetic_cache(self, probe_chain, probe_protocol_dir,
                                    probe_feature_dir, scratch_dir):
        import torch

        from scripts.train_material_probe import main as train_main

        output = scratch_dir / "fit_run"
        code = train_main([
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--features", str(probe_feature_dir),
            "--output-dir", str(output),
            "--fit",
        ])
        assert code == 0
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        assert summary["training_status"] == "COMPLETED"
        assert summary["fold_status"] == "OK"
        assert set(summary["oof"]) == {"P0", "P1", "P2"}
        for fold_id in ("fold0", "fold1", "fold2"):
            for variant in ("P0", "P1", "P2"):
                directory = output / "models" / f"{fold_id}_{variant}"
                assert (directory / "model_state.pt").is_file()
                assert (directory / "scaler.json").is_file()
                assert (directory / "history.json").is_file()
                payload = torch.load(directory / "model_state.pt",
                                     weights_only=False)
                assert payload["variant"] == variant
                assert payload["state_dict"]["weight"].shape == (4, 30)
        for key in ("P2_minus_P1", "P2_minus_P0"):
            entry = summary["paired"][key]
            total = (entry["fixed"] + entry["broke"] + entry["both_correct"]
                     + entry["both_wrong"])
            assert total == summary["preview"][0]["outer_eval"] * 3
        assert set(summary["key_class_pairs"]) == {
            "plastic_fish_vs_real_fish",
            "metal_submarine_vs_plastic_submarine",
        }

    def test_selection_never_reads_outer_eval(self, probe_chain,
                                              probe_protocol_dir,
                                              probe_feature_dir,
                                              module_scratch, scratch_dir):
        """Corrupting the outer_eval labels must not move the selection.

        The alternate world is FULLY self-consistent (protocol members,
        manifest rows and cache class ids relabelled together, binding
        rebuilt), so the new admission accepts it: the comparison is purely
        about which labels the frozen selection rule saw.
        """
        import shutil as shutil_module

        from core.existing_data_protocol import protocol_digest
        from core.material_probe import build_cache_binding

        from scripts.train_material_probe import main as train_main

        protocol = json.loads(
            (probe_protocol_dir / "protocol.json").read_text(encoding="utf-8"))
        alternate = module_scratch / "features_outer_alt"
        shutil_module.copytree(probe_feature_dir, alternate)
        manifest = json.loads(
            (alternate / "feature_manifest.json").read_text(encoding="utf-8"))
        cache = dict(np.load(alternate / "feature_cache.npz"))
        # only fold0's outer_eval rows: the other folds' outer_eval members
        # sit inside fold0's inner parts and keep their labels
        outer_names = {
            sample["sample_name"]
            for sample in next(fold for fold in protocol["folds"]["folds"]
                               if fold["fold_id"] == "fold0")
            ["outer_eval_members"]["samples"]
        }
        flipped = 0
        for index, sample in enumerate(manifest["samples"]):
            if sample["sample_name"] in outer_names:
                new_id = (int(cache["class_id"][index]) + 1) % 4
                cache["class_id"][index] = new_id
                sample["class_id"] = int(new_id)
                sample["class_name"] = CLASS_NAMES[new_id]
                flipped += 1
        assert flipped > 0
        np.savez_compressed(alternate / "feature_cache.npz", **cache)
        (alternate / "feature_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        flipped_ids = {}
        fold0 = next(fold for fold in protocol["folds"]["folds"]
                     if fold["fold_id"] == "fold0")
        for sample in fold0["outer_eval_members"]["samples"]:
            flipped_ids[sample["sample_name"]] = (
                int(sample["class_id"]) + 1) % 4
        # the alternate world relabels these samples EVERYWHERE (identity
        # must stay consistent across folds), so the admission accepts it
        # and the comparison isolates which labels outer_eval saw
        for fold in protocol["folds"]["folds"]:
            for part in ("outer_eval", "inner_train", "inner_val"):
                for sample in fold[f"{part}_members"]["samples"]:
                    if sample["sample_name"] in flipped_ids:
                        sample["class_id"] = flipped_ids[sample["sample_name"]]
                        sample["class_name"] = CLASS_NAMES[
                            flipped_ids[sample["sample_name"]]]
        protocol["protocol_sha256"] = protocol_digest(protocol)
        alternate_protocol_dir = module_scratch / "protocol_sel_alt"
        alternate_protocol_dir.mkdir()
        (alternate_protocol_dir / "protocol.json").write_text(
            json.dumps(protocol, sort_keys=True), encoding="utf-8")
        definition = json.loads(
            (alternate / "feature_definition.json").read_text(encoding="utf-8"))
        definition["protocol_sha256"] = protocol["protocol_sha256"]
        (alternate / "feature_definition.json").write_text(
            json.dumps(definition, sort_keys=True), encoding="utf-8")
        build_cache_binding(alternate, protocol)

        def _run(protocol_dir, features, output):
            code = train_main([
                "--protocol", str(protocol_dir / "protocol.json"),
                "--features", str(features),
                "--output-dir", str(output),
                "--fold", "fold0",
                "--variants", "P2",
                "--fit",
            ])
            assert code == 0
            return json.loads(
                (output / "summary.json").read_text(encoding="utf-8"))

        honest = _run(probe_protocol_dir, probe_feature_dir,
                      scratch_dir / "sel_honest")
        corrupt = _run(alternate_protocol_dir, alternate,
                       scratch_dir / "sel_corrupt")
        assert (honest["oof"]["P2"]["folds"][0]["selected_epoch"]
                == corrupt["oof"]["P2"]["folds"][0]["selected_epoch"])

    def test_scaler_depends_on_inner_train_only(self, probe_chain,
                                                probe_protocol_dir,
                                                probe_feature_dir,
                                                module_scratch,
                                                scratch_dir):
        """Perturbing outer_eval features must not change the scaler."""
        import shutil as shutil_module

        from core.material_probe import build_cache_binding

        from scripts.train_material_probe import main as train_main

        alternate = module_scratch / "features_eval_perturbed"
        shutil_module.copytree(probe_feature_dir, alternate)
        manifest = json.loads(
            (alternate / "feature_manifest.json").read_text(encoding="utf-8"))
        cache = dict(np.load(alternate / "feature_cache.npz"))
        protocol = json.loads(
            (probe_protocol_dir / "protocol.json").read_text(encoding="utf-8"))
        # only fold0's outer_eval rows: exactly the rows outside fold0's
        # inner_train/inner_val. Feature VALUES are not part of the
        # protocol identity, so after re-hashing the binding the cache is
        # admissible and the comparison isolates the scaler's fit rows.
        outer_names = {
            sample["sample_name"]
            for sample in next(fold for fold in protocol["folds"]["folds"]
                               if fold["fold_id"] == "fold0")
            ["outer_eval_members"]["samples"]
        }
        for index, sample in enumerate(manifest["samples"]):
            if sample["sample_name"] in outer_names:
                cache["g_features"][index] += 7.0
        np.savez_compressed(alternate / "feature_cache.npz", **cache)
        build_cache_binding(alternate, protocol)

        for features, name in ((probe_feature_dir, "sc_honest"),
                               (alternate, "sc_perturbed")):
            code = train_main([
                "--protocol", str(probe_protocol_dir / "protocol.json"),
                "--features", str(features),
                "--output-dir", str(scratch_dir / name),
                "--fold", "fold0",
                "--variants", "P1",
                "--fit",
            ])
            assert code == 0
        honest = json.loads(
            (scratch_dir / "sc_honest" / "models" / "fold0_P1" / "scaler.json")
            .read_text(encoding="utf-8"))
        perturbed = json.loads(
            (scratch_dir / "sc_perturbed" / "models" / "fold0_P1" / "scaler.json")
            .read_text(encoding="utf-8"))
        assert honest["mean"] == perturbed["mean"]
        assert honest["std"] == perturbed["std"]

    def test_blocked_protocol_refuses_fit_but_allows_dry_run(
            self, probe_chain, probe_protocol_dir, probe_feature_dir,
            module_scratch, scratch_dir):
        import shutil as shutil_module

        from core.existing_data_protocol import protocol_digest

        from scripts.train_material_probe import main as train_main

        protocol = json.loads(
            (probe_protocol_dir / "protocol.json").read_text(encoding="utf-8"))
        protocol["folds"]["status"] = "BLOCKED"
        protocol["folds"]["conflicts"] = [
            {"fold": None, "reason": "synthetic conflict for the test"}]
        protocol["protocol_sha256"] = protocol_digest(protocol)
        blocked_dir = module_scratch / "protocol_blocked"
        blocked_dir.mkdir()
        (blocked_dir / "protocol.json").write_text(
            json.dumps(protocol, sort_keys=True), encoding="utf-8")

        # a cache built against THAT blocked protocol (as prepare would
        # produce for it): same arrays, definition protocol digest
        # re-pointed and the binding rebuilt against the blocked protocol
        blocked_cache = module_scratch / "features_blocked"
        shutil_module.copytree(probe_feature_dir, blocked_cache)
        definition = json.loads(
            (blocked_cache / "feature_definition.json")
            .read_text(encoding="utf-8"))
        definition["protocol_sha256"] = protocol["protocol_sha256"]
        (blocked_cache / "feature_definition.json").write_text(
            json.dumps(definition, sort_keys=True), encoding="utf-8")
        from core.material_probe import build_cache_binding

        build_cache_binding(blocked_cache, protocol)

        blocked_fit = scratch_dir / "blocked_fit"
        code = train_main([
            "--protocol", str(blocked_dir / "protocol.json"),
            "--features", str(blocked_cache),
            "--output-dir", str(blocked_fit),
            "--fit",
        ])
        assert code == 3
        assert not blocked_fit.exists()

        blocked_dry = scratch_dir / "blocked_dry"
        code = train_main([
            "--protocol", str(blocked_dir / "protocol.json"),
            "--features", str(blocked_cache),
            "--output-dir", str(blocked_dry),
            "--dry-run",
        ])
        assert code == 0
        report = json.loads(
            (blocked_dry / "report.json").read_text(encoding="utf-8"))
        assert report["training_status"] == "NOT_RUN"

    def test_cache_protocol_mismatch_refused(self, probe_chain,
                                             probe_protocol_dir,
                                             probe_feature_dir,
                                             module_scratch, scratch_dir):
        from core.existing_data_protocol import protocol_digest

        from scripts.train_material_probe import main as train_main

        protocol = json.loads(
            (probe_protocol_dir / "protocol.json").read_text(encoding="utf-8"))
        protocol["protocol_sha256"] = protocol_digest(
            {**protocol, "data": {**protocol["data"],
                                  "fingerprint": "e" * 64}})
        forged = module_scratch / "protocol_forged"
        forged.mkdir()
        (forged / "protocol.json").write_text(
            json.dumps(protocol, sort_keys=True), encoding="utf-8")
        code = train_main([
            "--protocol", str(forged / "protocol.json"),
            "--features", str(probe_feature_dir),
            "--output-dir", str(scratch_dir / "forged_run"),
            "--fit",
        ])
        assert code == 1
        assert not (scratch_dir / "forged_run").exists()

    def test_existing_output_dir_refused(self, probe_chain,
                                         probe_protocol_dir,
                                         probe_feature_dir, scratch_dir):
        from scripts.train_material_probe import main as train_main

        output = scratch_dir / "existing_run"
        assert train_main([
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--features", str(probe_feature_dir),
            "--output-dir", str(output),
            "--dry-run",
        ]) == 0
        assert train_main([
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--features", str(probe_feature_dir),
            "--output-dir", str(output),
            "--dry-run",
        ]) == 1

    def test_seed_mismatch_refused(self, probe_chain, probe_protocol_dir,
                                   probe_feature_dir, scratch_dir):
        from scripts.train_material_probe import main as train_main

        code = train_main([
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--features", str(probe_feature_dir),
            "--output-dir", str(scratch_dir / "seed_run"),
            "--seed", "1999",
            "--dry-run",
        ])
        assert code == 1


class TestPrepareRealAdmission:
    """The extraction entry must run the project's FULL admission gate.

    A tampered NPZ with a stale audit is refused before the encoder is
    even constructed.
    """

    def test_tampered_npz_refused_before_encoder(self, probe_chain,
                                                 probe_protocol_dir,
                                                 module_scratch,
                                                 scratch_dir, monkeypatch):
        import shutil as shutil_module

        from core.material_mask_dataset import load_mask_sample, save_mask_sample
        from scripts.prepare_material_probe import main as prepare_main

        # a full copy of the synthetic chain with one NPZ value changed,
        # the audit NOT updated
        v5_copy = module_scratch / "v5_tampered"
        shutil_module.copytree(probe_chain["v5"], v5_copy)
        manifest_path = v5_copy / "dataset_manifest.csv"
        import csv as csv_module

        with manifest_path.open(encoding="utf-8") as handle:
            rows = list(csv_module.DictReader(handle))
        target = v5_copy / rows[0]["npz_path"]
        sample = load_mask_sample(target)
        signed = sample.signed_q.copy()
        region = sample.instance_valid
        signed[region] = np.clip(signed[region] + 0.01, -1.0, 1.0)
        save_mask_sample(
            target, gray=sample.gray, signed_q=signed,
            abs_q=np.abs(signed), valid=sample.valid,
            quality=sample.quality, class_id=sample.class_id,
            instance_mask=sample.instance_mask)

        import yaml as yaml_module

        config = yaml_module.safe_load(
            Path(probe_chain["config_path"]).read_text(encoding="utf-8"))
        config["data"]["root"] = v5_copy.relative_to(
            probe_chain["project_root"]).as_posix()
        bad_config = scratch_dir / "config_tampered.yaml"
        bad_config.write_text(yaml_module.safe_dump(config), encoding="utf-8")

        def _bomb(config, project_root):
            raise AssertionError("encoder must not be constructed")

        monkeypatch.setattr(
            "scripts.prepare_material_probe.build_probe_encoder", _bomb)
        code = prepare_main([
            "--config", str(bad_config),
            "--protocol", str(probe_protocol_dir / "protocol.json"),
            "--output-dir", str(scratch_dir / "features_tampered"),
            "--device", "cpu",
        ])
        assert code == 1
        assert not (scratch_dir / "features_tampered").exists()


class TestExtractionBudget:
    """The shared budget is checked at every extraction batch boundary."""

    def test_budget_exhaustion_stops_before_encoder_forward(
            self, probe_chain, probe_protocol_dir, scratch_dir, monkeypatch):
        import pytest as _pytest

        from core.existing_data_protocol import BudgetExceeded, BudgetWatch

        import scripts.prepare_material_probe as prepare_module
        from scripts.prepare_material_probe import extract_gray_features

        mp = _pytest.MonkeyPatch()
        mp.setattr(prepare_module, "build_probe_encoder",
                   lambda config, project_root: (_StubEncoder().eval(), {
                       "loaded": 1, "missing": [], "unexpected": [],
                       "random_init": []}))
        try:
            clock = {"now": 0.0}
            budget = BudgetWatch(1.0, clock=lambda: clock["now"])
            clock["now"] = 3601.0  # exhausted before the first batch check
            config = {"features": {"imgsz": 224}}
            rows = [
                {"sample_name": f"s{i:03d}",
                 "npz_path": f"s{i:03d}.npz"}
                for i in range(4)
            ]
            # tiny synthetic npz files for the loader
            import numpy as _np

            from core.material_mask_dataset import save_mask_sample

            data_root = scratch_dir / "v5_budget"
            for index, name in enumerate(
                    ("metal_submarine", "plastic_submarine",
                     "plastic_fish", "real_fish")):
                shape = (24, 20)
                mask = _np.zeros(shape, dtype=_np.uint8)
                mask[4:14, 4:12] = 1
                save_mask_sample(
                    data_root / f"s{index:03d}.npz",
                    gray=_np.full(shape, 128, dtype=_np.uint8),
                    signed_q=_np.zeros(shape, dtype=_np.float32),
                    abs_q=_np.zeros(shape, dtype=_np.float32),
                    valid=mask.copy(), quality=_np.zeros(4, _np.float32),
                    class_id=index, instance_mask=mask)
            with _pytest.raises(BudgetExceeded):
                extract_gray_features(
                    rows, data_root, config, "cpu", budget=budget)
        finally:
            mp.undo()
