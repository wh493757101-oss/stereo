"""Budget failures through real CLI functions, with synthetic CPU providers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from core.existing_data_protocol import BudgetWatch


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def timing_entry(monkeypatch, scratch_dir):
    from scripts import diagnose_existing_detection as module

    clock = {"now": 0.0}
    budget = BudgetWatch(1, clock=lambda: clock["now"])
    state = SimpleNamespace(clock=clock, models=0, calls=0, expire_after=None,
                            before_timing=False, error=False)
    config = {"baselines": {"model_a": {"path": "synthetic.pt"}},
              "detection": {"split": "val", "max_runtime_minutes": 1,
                            "timing": {"enabled": True, "warmup": 2,
                                       "repeats": 3}}}
    state.config = config
    plan = [{"name": name, "conf": .25, "nms_iou": .7, "imgsz": 8}
            for name in ("D0", "D1")]
    monkeypatch.setattr(module, "load_experiment_config", lambda _: config)
    monkeypatch.setattr(module, "verify_baseline_identities", lambda *a: None)
    monkeypatch.setattr(module, "load_historical_d0", lambda *a: {})
    monkeypatch.setattr(module, "build_condition_plan", lambda _: plan)
    monkeypatch.setattr(module, "real_frames_provider", lambda *a: [
        {"frame_id": "f0", "image": np.zeros((8, 8, 3), np.uint8)}])
    monkeypatch.setattr(module, "probe_ultralytics_identity", lambda _: {})
    monkeypatch.setattr(module, "BudgetWatch", lambda _: budget)

    def diagnosis(**kwargs):
        result = {"status": "COMPLETED", "conditions": {"D0": {"matched": 3}}}
        module.write_json(kwargs["output_dir"] / "summary.json", result)
        if state.before_timing:
            clock["now"] = 61
        return result

    class Model:
        def __init__(self, *a):
            state.models += 1

        def predict(self, *a, **kw):
            state.calls += 1
            if state.error:
                raise RuntimeError("ordinary model failure")
            if state.calls == state.expire_after:
                clock["now"] = 61

    monkeypatch.setattr(module, "run_detection_diagnosis", diagnosis)
    monkeypatch.setattr(module, "_ConditionedModelA", Model)
    state.output = scratch_dir / "detection"
    state.run = lambda: module.main([
        "--config", "synthetic", "--output-dir", str(state.output),
        "--device", "cpu"])
    return state


@pytest.mark.parametrize("expire_after,completed", [(1, 0), (3, 1), (5, 3)])
def test_timing_timeout_publishes_partial(timing_entry, expire_after, completed):
    state = timing_entry
    state.expire_after = expire_after
    assert state.run() == 2
    summary = read_json(state.output / "summary.json")
    timing = read_json(state.output / "timing.json")
    assert summary["status"] == timing["status"] == "PARTIAL_BUDGET_EXCEEDED"
    assert summary["conditions"]["D0"]["matched"] == 3
    assert timing["reason"]
    assert state.calls == expire_after
    assert "D1" not in timing["conditions"]
    row = timing["conditions"]["D0"]
    assert len(row["samples_ms"]) == row["completed_repeats"] == completed
    assert row["completed_warmup"] == min(expire_after, 2)


def test_expired_before_timing_constructs_no_model(timing_entry):
    state = timing_entry
    state.before_timing = True
    assert state.run() == 2
    assert state.models == state.calls == 0
    assert read_json(state.output / "summary.json")["status"] == "PARTIAL_BUDGET_EXCEEDED"
    assert read_json(state.output / "timing.json")["conditions"] == {}


def test_normal_timing_still_completes(timing_entry):
    state = timing_entry
    assert state.run() == 0
    assert read_json(state.output / "summary.json")["status"] == "COMPLETED"
    assert set(read_json(state.output / "timing.json")["conditions"]) == {"D0", "D1"}
    assert state.calls == 10


def test_timing_keeps_previous_complete_condition(timing_entry):
    state = timing_entry
    state.expire_after = 8
    assert state.run() == 2
    timing = read_json(state.output / "timing.json")
    assert timing["conditions"]["D0"]["status"] == "COMPLETED"
    assert timing["conditions"]["D0"]["completed_repeats"] == 3
    assert timing["conditions"]["D1"]["completed_repeats"] == 1
    assert state.calls == 8


def test_disabled_timing_constructs_no_model(timing_entry):
    state = timing_entry
    state.config["detection"]["timing"]["enabled"] = False
    assert state.run() == 0
    assert state.models == state.calls == 0
    assert read_json(state.output / "timing.json")["enabled"] is False


def test_ordinary_timing_failure_is_not_budget(timing_entry):
    state = timing_entry
    state.error = True
    assert state.run() == 1
    assert read_json(state.output / "summary.json")["status"] == "FAILED"


@pytest.fixture
def feature_entry(monkeypatch, scratch_dir):
    from scripts import prepare_material_probe as module
    from models import polar_fusion

    clock = {"now": 0.0}
    budget = BudgetWatch(1, clock=lambda: clock["now"])
    state = SimpleNamespace(calls=0, models=0, expire=True, error=False,
                            clock=clock)
    rows = [{"sample_name": f"sample_{i:03}", "npz_path": f"{i}.npz",
             "split": "train", "group_name": "synthetic", "source_frame": str(i),
             "object_index": 0, "class_id": i % 4, "class_name": str(i % 4)}
            for i in range(35)]
    config = {
        "data": {"root": "unused", "fingerprint": "a" * 64,
                 "parent_v4_fingerprint": "b" * 64, "dev_pool_splits": ["train"],
                 "class_names": [str(i) for i in range(4)]},
        "features": {"imgsz": 8, "max_runtime_minutes": 1,
                     "preprocessing_version": "synthetic", "gray_input_mode": "gray",
                     "encoder": "synthetic", "pooling": "mask", "pooling_background_note": ""},
        "baselines": {"official_mit_b0": {"path": "synthetic.pt", "sha256": "c" * 64}}}
    protocol = {"data": config["data"], "config": config,
                "config_sha256": "", "protocol_sha256": "d" * 64}
    config_path = scratch_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(module, "load_protocol", lambda _: protocol)
    monkeypatch.setattr(module, "admit_mask_dataset", lambda *a, **k: {
        "verified_files": 35, "audit_fingerprint": "a" * 64,
        "v4_parent_fingerprint": "b" * 64})
    monkeypatch.setattr(module, "read_manifest_rows", lambda *a: rows)
    monkeypatch.setattr(module, "verify_sample_set", lambda *a: None)
    monkeypatch.setattr(module, "fold_roles", lambda _: {r["sample_name"]: {} for r in rows})
    monkeypatch.setattr(module, "polarization_quality_statistics", lambda *a: ({}, []))
    monkeypatch.setattr(module, "render_quality_plots", lambda *a: [])
    monkeypatch.setattr(module, "BudgetWatch", lambda _: budget)
    monkeypatch.setattr(module, "load_mask_sample", lambda _: SimpleNamespace(
        instance_mask=np.ones((8, 8), dtype=np.uint8)))
    monkeypatch.setattr(polar_fusion, "_fusion_tensors", lambda *a: (
        torch.ones(3, 8, 8), None, None))
    monkeypatch.setattr(module, "build_slot_features", lambda *a: (
        np.zeros((35, 6), np.float32), np.zeros((35, 8), np.float32),
        np.ones(35, bool), np.arange(35) % 4))

    class Encoder(torch.nn.Module):
        def forward(self, x):
            state.calls += 1
            if state.error:
                raise RuntimeError("ordinary encoder failure")
            if state.expire:
                clock["now"] = 61
            return [torch.full((len(x), 256, 1, 1), float(state.calls))]

    def encoder(*a, **k):
        state.models += 1
        return Encoder().eval(), {"loaded": [], "missing": [], "unexpected": [], "random_init": False}

    monkeypatch.setattr(module, "build_probe_encoder", encoder)
    state.output = scratch_dir / "features"
    state.protocol = protocol
    state.rows = rows
    state.run = lambda: module.main([
        "--config", str(config_path), "--protocol", "synthetic",
        "--output-dir", str(state.output), "--device", "cpu"])
    return state


def test_feature_timeout_preserves_completed_rows_and_refuses_training(feature_entry):
    from scripts.train_material_probe import verify_feature_cache

    state = feature_entry
    assert state.run() == 2
    report = read_json(state.output / "report.json")
    partial = read_json(state.output / "partial_extraction.json")
    assert report["status"] == partial["status"] == "PARTIAL_BUDGET_EXCEEDED"
    assert partial["completed_samples"] == 32
    assert partial["total_samples"] == 35
    assert partial["sample_names"] == [r["sample_name"] for r in state.rows[:32]]
    with np.load(state.output / "partial_gray_features.npz") as cache:
        np.testing.assert_array_equal(cache["g_features"], np.ones((32, 256), np.float32))
    assert state.calls == 1
    assert not (state.output / "cache_binding.json").exists()
    assert not (state.output / "feature_cache.npz").exists()
    with pytest.raises(FileNotFoundError, match="binding"):
        verify_feature_cache(state.output, state.protocol)


def test_feature_timeout_before_first_batch(feature_entry):
    state = feature_entry
    state.clock["now"] = 61
    assert state.run() == 2
    assert state.models == state.calls == 0
    assert read_json(state.output / "report.json")["completed_samples"] == 0
    assert not (state.output / "partial_gray_features.npz").exists()
    assert not (state.output / "cache_binding.json").exists()


def test_partial_features_refused_by_training_main(feature_entry, monkeypatch, capsys):
    from core.existing_data_protocol import protocol_digest
    from scripts import train_material_probe as train
    from .test_cache_admission import make_protocol

    state = feature_entry
    assert state.run() == 2
    protocol = make_protocol()
    protocol["config"]["probe"] = {
        "variants": {name: {} for name in ("P0", "P1", "P2")},
        "seed": 2026, "optimizer": "AdamW", "lr": .01,
        "weight_decay": .001, "max_epochs": 2}
    protocol["protocol_sha256"] = protocol_digest(protocol)
    path = state.output.parent / "train_protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    def forbidden(*a, **kw):
        pytest.fail("partial cache reached a fitting operation")

    monkeypatch.setattr(train, "build_probe_head", forbidden)
    monkeypatch.setattr(train, "train_linear_probe", forbidden)
    monkeypatch.setattr(train.Standardizer, "fit", forbidden)
    output = state.output.parent / "refused_fit"
    assert train.main(["--protocol", str(path), "--features", str(state.output),
                       "--output-dir", str(output), "--fit"]) == 1
    assert "binding record missing" in capsys.readouterr().err
    assert not output.exists()


def test_feature_complete_keeps_normal_cache(feature_entry):
    state = feature_entry
    state.expire = False
    assert state.run() == 0
    assert state.calls == 2
    assert read_json(state.output / "report.json")["status"] == "COMPLETED"
    assert (state.output / "cache_binding.json").is_file()
    assert not (state.output / "partial_gray_features.npz").exists()


def test_ordinary_feature_failure_is_not_budget(feature_entry):
    state = feature_entry
    state.error = True
    assert state.run() == 1
    assert not (state.output / "partial_extraction.json").exists()
