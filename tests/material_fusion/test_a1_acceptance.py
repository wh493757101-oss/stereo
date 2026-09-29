"""A1 acceptance regressions: eval admission, protocol sealing, diagnostics
binding, save-time evidence, fallback routing and annotation-source re-check.

Every test runs against a synthetic but *real* provenance chain (PNG frames
with YOLO-seg labels -> the real V4 generation helpers -> the real V5
generator), so breaking a link exercises the production code paths.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from core.fusion_dataset import dataset_fingerprint, load_fusion_sample
from core.material_fusion_dataset import MaterialFusionDataset
from core.material_mask_dataset import (
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    POOLING_GAP,
    POOLING_INSTANCE_MASK,
    admit_dataset_for_protocol,
    admit_mask_dataset,
    load_mask_sample,
    mask_dataset_fingerprint,
    resolve_source_path,
    save_mask_sample,
    verify_mask_sources,
)
from models.material_fusion import (
    MATERIAL_FUSION_MASK_VERSION,
    MATERIAL_FUSION_VERSION,
    build_material_model,
    material_fusion_metadata,
    read_material_checkpoint_metadata,
    save_material_checkpoint,
    validate_checkpoint_metadata,
)
from scripts.verify_material_mask import (
    _SaveObserver,
    check_finiteness,
    compare_saved_checkpoints,
    materialize_fixed_batch,
    verify_fallback_routing,
)

from .conftest import (
    ROOT,
    build_alternate_v5,
    build_synthetic_source,
    build_synthetic_v4,
    build_synthetic_v5,
    fabricate_completed_mask_run,
    run_synthetic_mask_training,
    write_synthetic_pretrained,
)


@pytest.fixture(scope="module")
def chain(module_scratch):
    workspace = module_scratch.resolve()
    source = build_synthetic_source(module_scratch / "source", counts=(4, 4, 4), size=64)
    v4 = build_synthetic_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)
    return {
        "workspace": workspace,
        "source": source,
        "v4": v4,
        "v5": v5,
        "parent": dataset_fingerprint(v4),
        "fingerprint": mask_dataset_fingerprint(v5),
    }


@pytest.fixture(scope="module")
def mask_run(chain, module_scratch):
    """A real v2 mask-pooling checkpoint trained on the synthetic V5."""
    result = run_synthetic_mask_training(
        module_scratch, "a1_mask_run", chain["v5"], chain["fingerprint"],
        chain["parent"], variant="gray_polar", epochs=2, warmup_epochs=1,
        limit_batches=1,
    )
    assert result["exit_code"] == 0
    return result


def _gap_metadata(chain, **overrides):
    kwargs = dict(
        variant="gray_polar", imgsz=64, data_fingerprint=chain["parent"],
        pretrained_sha256="b" * 64, seed=2026, epoch=1, stage="warmup",
        smoke=False, limit_batches=0, val_macro_f1=0.1, val_accuracy=0.2,
        physical_batch=2, accum_steps=2, effective_batch=4, drop_path_rate=0.0,
    )
    kwargs.update(overrides)
    return material_fusion_metadata(**kwargs)


def _write_raw_checkpoint(path: Path, metadata, model) -> Path:
    """Write a checkpoint payload directly, bypassing save-time validation.

    Used only to build deliberately contradictory checkpoints that the loader
    must refuse; the production save path would (correctly) reject them.
    """
    import dataclasses

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dataclasses.asdict(metadata),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _gap_checkpoint(scratch_dir: Path, chain, name="v1.pt") -> Path:
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_GAP)
    return save_material_checkpoint(
        scratch_dir / name, model, _gap_metadata(chain)
    )


# ---------------------------------------------------------------------------
# Fix 1: the eval entry admits the right dataset for the protocol


def _eval_main(argv):
    from scripts.eval_material_fusion import main as eval_main

    return eval_main(argv)


def test_eval_admits_a_legal_v2_checkpoint_end_to_end(chain, mask_run, scratch_dir):
    """The real CLI: admission, model load, mask loader and report generation."""
    output = scratch_dir / "eval_v2.json"
    code = _eval_main([
        "--checkpoint", str(mask_run["run_dir"] / "best.pt"),
        "--data", str(chain["v5"]), "--split", "val", "--device", "cpu",
        "--batch", "2", "--limit-batches", "1", "--no-fallback",
        "--output", str(output),
    ])
    assert code == 0, "a legal v2 checkpoint must evaluate through the real CLI"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["version"] == MATERIAL_FUSION_MASK_VERSION
    assert report["data_fingerprint"] == chain["fingerprint"]
    assert report["data_fingerprint"] != dataset_fingerprint(chain["v5"])
    assert report["config_consistency"]["asserted"]["pooling_mode"] == POOLING_INSTANCE_MASK
    assert report["evaluation"]["samples"] > 0
    assert report["evaluation"]["truncated"] is True
    assert report["class_names"] == [
        "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish",
    ]


def test_eval_v2_does_not_go_through_the_v4_gate(chain, mask_run, scratch_dir, monkeypatch):
    """v2 must use the V5 gate; the V4 gate must never be reached."""
    import scripts.eval_material_fusion as eval_module
    from core.material_fusion_dataset import admit_material_dataset

    called: list[str] = []

    def v4_gate(*args, **kwargs):
        called.append("v4")
        return admit_material_dataset(*args, **kwargs)

    monkeypatch.setattr(eval_module, "admit_material_dataset", v4_gate, raising=False)
    code = _eval_main([
        "--checkpoint", str(mask_run["run_dir"] / "best.pt"),
        "--data", str(chain["v5"]), "--split", "val", "--device", "cpu",
        "--batch", "2", "--limit-batches", "1", "--no-fallback",
        "--output", str(scratch_dir / "eval_v2_gate.json"),
    ])
    assert code == 0
    assert called == []


def test_eval_v2_refuses_a_wrong_v5_fingerprint(chain, mask_run, scratch_dir):
    """A checkpoint trained on another V5 state must not evaluate here."""
    alternate = build_alternate_v5(
        chain["v5"], scratch_dir / "v5_alternate", chain["v4"], chain["parent"]
    )
    alternate_fingerprint = mask_dataset_fingerprint(alternate)
    assert alternate_fingerprint != chain["fingerprint"]
    output = scratch_dir / "eval_wrong_fp.json"
    code = _eval_main([
        "--checkpoint", str(mask_run["run_dir"] / "best.pt"),
        "--data", str(alternate), "--split", "val", "--device", "cpu",
        "--batch", "2", "--limit-batches", "1", "--no-fallback",
        "--output", str(output),
    ])
    assert code == 2
    assert not output.exists(), "a refused evaluation must not leave a report"


def test_eval_v2_refuses_a_wrong_parent_fingerprint(chain, mask_run, scratch_dir, monkeypatch):
    """Same V5 data, but a checkpoint bound to another V4 parent: refused."""
    metadata = read_material_checkpoint_metadata(mask_run["run_dir"] / "best.pt")
    tampered = replace(metadata, parent_data_fingerprint="c" * 64)
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_INSTANCE_MASK)
    path = save_material_checkpoint(scratch_dir / "wrong_parent.pt", model, tampered)
    output = scratch_dir / "eval_wrong_parent.json"
    code = _eval_main([
        "--checkpoint", str(path), "--data", str(chain["v5"]),
        "--split", "val", "--device", "cpu", "--batch", "2",
        "--limit-batches", "1", "--no-fallback", "--output", str(output),
    ])
    assert code == 2
    assert not output.exists()


def test_eval_v2_refuses_a_wrong_mask_protocol(chain, mask_run, scratch_dir):
    """A checkpoint declaring another mask schema/resize is refused."""
    metadata = read_material_checkpoint_metadata(mask_run["run_dir"] / "best.pt")
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_INSTANCE_MASK)
    for field, value in (
        ("mask_schema", "instance_mask_polygon_crop_v0"),
        ("mask_resize_version", "nearest_v0"),
    ):
        tampered = replace(metadata, **{field: value})
        path = _write_raw_checkpoint(scratch_dir / f"bad_{field}.pt", tampered, model)
        output = scratch_dir / f"eval_bad_{field}.json"
        code = _eval_main([
            "--checkpoint", str(path), "--data", str(chain["v5"]),
            "--split", "val", "--device", "cpu", "--batch", "2",
            "--limit-batches", "1", "--no-fallback", "--output", str(output),
        ])
        assert code == 2, field
        assert not output.exists(), field


def test_eval_derives_the_v2_protocol_without_an_adjacent_run_config(chain, mask_run, scratch_dir):
    """A v2 checkpoint with no run_config.json inherits the v2 template."""
    isolated = scratch_dir / "isolated"
    isolated.mkdir()
    import shutil

    shutil.copy(mask_run["run_dir"] / "best.pt", isolated / "best.pt")
    output = scratch_dir / "eval_derived.json"
    code = _eval_main([
        "--checkpoint", str(isolated / "best.pt"), "--data", str(chain["v5"]),
        "--split", "val", "--device", "cpu", "--batch", "2",
        "--limit-batches", "1", "--no-fallback", "--output", str(output),
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config_source"]["source"] == "template"
    assert report["config_source"]["path"].endswith("material_fusion_v2_mask.yaml")
    assert "pooling_mode" in report["config_source"]["derived_fields"]
    assert "parent_data_fingerprint" in report["config_source"]["derived_fields"]
    assert report["version"] == MATERIAL_FUSION_MASK_VERSION


def test_eval_still_derives_v1_from_the_v1_template(chain, scratch_dir):
    """A v1 checkpoint with no run_config keeps the v1 template."""
    path = _gap_checkpoint(scratch_dir, chain, "v1_derive.pt")
    output = scratch_dir / "eval_v1_derived.json"
    code = _eval_main([
        "--checkpoint", str(path), "--data", str(chain["v4"]),
        "--split", "val", "--device", "cpu", "--batch", "2",
        "--limit-batches", "1", "--no-fallback", "--output", str(output),
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config_source"]["path"].endswith("material_fusion_v1.yaml")
    assert report["version"] == MATERIAL_FUSION_VERSION
    assert report["config_consistency"]["asserted"]["pooling_mode"] == POOLING_GAP


def test_eval_admits_a_legal_v1_checkpoint_end_to_end(chain, scratch_dir):
    path = _gap_checkpoint(scratch_dir, chain, "v1_e2e.pt")
    output = scratch_dir / "eval_v1_e2e.json"
    code = _eval_main([
        "--checkpoint", str(path), "--data", str(chain["v4"]),
        "--split", "val", "--device", "cpu", "--batch", "2",
        "--limit-batches", "1", "--no-fallback", "--output", str(output),
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data_fingerprint"] == chain["parent"]
    assert report["version"] == MATERIAL_FUSION_VERSION


# ---------------------------------------------------------------------------
# Fix 2: cross-protocol continuation and metadata mixing are sealed


def _payload(**meta_overrides) -> dict:
    meta = {
        "version": MATERIAL_FUSION_VERSION, "variant": "gray_polar",
        "class_names": ["metal_submarine", "plastic_submarine", "plastic_fish",
                        "real_fish"],
        "imgsz": 64, "preprocessing_version": "x", "data_fingerprint": "a" * 64,
        "upstream_commit": "0" * 40, "pretrained_sha256": "b" * 64, "seed": 2026,
        "epoch": 1, "stage": "warmup", "smoke": False, "limit_batches": 0,
        "val_macro_f1": 0.1, "val_accuracy": 0.2, "physical_batch": 2,
        "accum_steps": 2, "effective_batch": 4, "drop_path_rate": 0.0,
        "init_from": "", "fallback_weights": "", "fallback_identity": "",
    }
    meta.update(meta_overrides)
    return {"metadata": meta, "state_dict": {}}


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"version": MATERIAL_FUSION_MASK_VERSION, "pooling_mode": POOLING_GAP},
         "must record pooling_mode 'instance_mask'"),
        ({"version": MATERIAL_FUSION_MASK_VERSION}, "must record pooling_mode"),
        ({"version": MATERIAL_FUSION_VERSION, "pooling_mode": POOLING_INSTANCE_MASK},
         "must be a 'gap'"),
    ],
)
def test_protocol_pooling_contradictions_are_refused(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_checkpoint_metadata(_payload(**overrides))


def test_historical_v1_without_pooling_mode_is_still_gap():
    normalized = validate_checkpoint_metadata(_payload())
    assert normalized["pooling_mode"] == POOLING_GAP


def test_save_refuses_metadata_that_misdescribes_the_model(chain, scratch_dir):
    """Identical shapes do not let the metadata lie about how the model pools."""
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_GAP)
    with pytest.raises(ValueError, match="pooling_mode"):
        save_material_checkpoint(
            scratch_dir / "lie.pt", model,
            _gap_metadata(
                chain, version=MATERIAL_FUSION_MASK_VERSION,
                pooling_mode=POOLING_INSTANCE_MASK,
                mask_schema=MASK_SCHEMA_VERSION,
                mask_resize_version=MASK_RESIZE_VERSION,
                parent_data_fingerprint="c" * 64,
            ),
        )
    assert not (scratch_dir / "lie.pt").exists()


def test_v2_init_from_is_refused_before_any_side_effect(chain, mask_run, scratch_dir, monkeypatch):
    """--init-from is refused for v2, and no run directory is created."""
    import scripts.train_material_fusion as train_module

    run_root = scratch_dir / "runs" / "train"
    run_root.mkdir(parents=True, exist_ok=True)
    v1_path = _gap_checkpoint(scratch_dir, chain, "init_v1.pt")
    for name, target in (("from_v1", v1_path),
                         ("from_v2", mask_run["run_dir"] / "best.pt")):
        args = [
            "--run-id", f"probe_v2_{name}", "--config", str(mask_run["config_path"]),
            "--variant", "gray_polar", "--device", "cpu", "--epochs", "2",
            "--warmup-epochs", "1", "--limit-batches", "1",
            "--init-from", str(target),
        ]
        with monkeypatch.context() as patch:
            patch.setattr(train_module, "RUNS_ROOT", run_root)
            code = train_module.main(args)
        assert code == 2, name
        assert not (run_root / f"probe_v2_{name}").exists(), name


def test_v1_refuses_continuation_from_a_v2_checkpoint(chain, mask_run, scratch_dir, monkeypatch):
    import scripts.train_material_fusion as train_module
    from .conftest import write_experiment_config

    run_root = scratch_dir / "runs" / "train"
    run_root.mkdir(parents=True, exist_ok=True)
    data_root = build_synthetic_v4(
        chain["source"], scratch_dir / "v4_for_v1", chain["workspace"]
    )
    pretrained = write_synthetic_pretrained(scratch_dir / "mit_v1.pth")
    config = write_experiment_config(
        scratch_dir / "v1.yaml", data_root, pretrained, variant="gray_polar",
        effective_batch=4, epochs=2, warmup_epochs=1, imgsz=64,
    )
    args = [
        "--run-id", "probe_v1_from_v2", "--config", str(config),
        "--variant", "gray_polar", "--device", "cpu", "--epochs", "2",
        "--warmup-epochs", "1", "--limit-batches", "1",
        "--init-from", str(mask_run["run_dir"] / "best.pt"),
    ]
    with monkeypatch.context() as patch:
        patch.setattr(train_module, "RUNS_ROOT", run_root)
        code = train_module.main(args)
    assert code == 2
    assert not (run_root / "probe_v1_from_v2").exists()


def test_legal_v1_continuation_still_works(chain, scratch_dir, monkeypatch):
    import scripts.train_material_fusion as train_module
    from .conftest import write_experiment_config

    run_root = scratch_dir / "runs" / "train"
    run_root.mkdir(parents=True, exist_ok=True)
    data_root = build_synthetic_v4(
        chain["source"], scratch_dir / "v4_cont", chain["workspace"]
    )
    pretrained = write_synthetic_pretrained(scratch_dir / "mit_cont.pth")
    fingerprint = dataset_fingerprint(data_root)
    source_checkpoint = save_material_checkpoint(
        scratch_dir / "source_v1.pt",
        build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_GAP),
        _gap_metadata(chain, data_fingerprint=fingerprint),
    )
    config = write_experiment_config(
        scratch_dir / "v1_cont.yaml", data_root, pretrained, variant="gray_polar",
        effective_batch=4, epochs=2, warmup_epochs=1, imgsz=64,
    )
    args = [
        "--run-id", "probe_v1_continue", "--config", str(config),
        "--variant", "gray_polar", "--device", "cpu", "--epochs", "2",
        "--warmup-epochs", "1", "--limit-batches", "1",
        "--init-from", str(source_checkpoint),
    ]
    with monkeypatch.context() as patch:
        patch.setattr(train_module, "RUNS_ROOT", run_root)
        code = train_module.main(args)
    assert code == 0
    run_config = json.loads(
        (run_root / "probe_v1_continue" / "material_fusion" / "run_config.json")
        .read_text(encoding="utf-8")
    )
    assert run_config["version"] == MATERIAL_FUSION_VERSION
    assert run_config["config"]["experiment"] == MATERIAL_FUSION_VERSION


def test_v2_run_config_records_the_real_protocol(mask_run):
    """New outputs record the actual protocol at the run_config top level."""
    run_config = json.loads(
        (mask_run["run_dir"] / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config["version"] == MATERIAL_FUSION_MASK_VERSION
    assert run_config["config"]["experiment"] == MATERIAL_FUSION_MASK_VERSION
    assert run_config["pooling_mode"] == POOLING_INSTANCE_MASK
    assert run_config["mask_schema"] == MASK_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fix 3: the diagnostics bind the concrete V5 dataset


def test_diagnostics_refuse_a_different_v5_with_the_same_parent(chain, mask_run, scratch_dir):
    """Same V4 parent, different V5 state: refused."""
    alternate = build_alternate_v5(
        chain["v5"], scratch_dir / "v5_alt_diag", chain["v4"], chain["parent"]
    )
    with pytest.raises(ValueError, match="fingerprint"):
        admit_dataset_for_protocol(
            alternate,
            pooling_mode=POOLING_INSTANCE_MASK,
            expected_fingerprint=chain["fingerprint"],
            expected_parent_fingerprint=chain["parent"],
        )


def test_diagnostics_refuse_a_wrong_mask_schema(chain):
    with pytest.raises(ValueError, match="mask_schema"):
        admit_dataset_for_protocol(
            chain["v5"], pooling_mode=POOLING_INSTANCE_MASK,
            expected_fingerprint=chain["fingerprint"],
            expected_parent_fingerprint=chain["parent"],
            expected_mask_schema="instance_mask_polygon_crop_v0",
        )
    with pytest.raises(ValueError, match="mask_resize_version"):
        admit_dataset_for_protocol(
            chain["v5"], pooling_mode=POOLING_INSTANCE_MASK,
            expected_fingerprint=chain["fingerprint"],
            expected_parent_fingerprint=chain["parent"],
            expected_mask_resize_version="nearest_v0",
        )


def test_diagnostics_refusal_happens_before_any_model_forward(
    chain, mask_run, scratch_dir, monkeypatch
):
    """A refused diagnosis must not create a report or run the model."""
    import scripts.diagnose_material_mask as diag

    def fail_forward(*args, **kwargs):
        raise AssertionError("the model must not run before admission passes")

    monkeypatch.setattr(diag, "predict_samples", fail_forward)
    output = scratch_dir / "diag_refused"
    # a checkpoint that records a different V5 fingerprint than the data
    tampered = replace(
        read_material_checkpoint_metadata(mask_run["run_dir"] / "best.pt"),
        data_fingerprint="d" * 64,
    )
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_INSTANCE_MASK)
    path = save_material_checkpoint(scratch_dir / "wrong_v5.pt", model, tampered)
    args = diag.parse_args([
        "--mode", "controlled", "--data", str(chain["v5"]),
        "--checkpoint", str(path),
        "--split", "val", "--device", "cpu", "--seeds", "2026",
        "--output", str(output),
    ])
    with pytest.raises(ValueError, match="fingerprint"):
        diag.run_controlled(args)
    assert not output.exists(), "a refused diagnosis must not leave a report directory"


def test_diagnostics_report_identifies_the_checkpoint_and_protocol(
    chain, mask_run, scratch_dir
):
    import scripts.diagnose_material_mask as diag
    from models.material_fusion import sha256_of_file

    output = scratch_dir / "diag_ok"
    args = diag.parse_args([
        "--mode", "controlled", "--data", str(chain["v5"]),
        "--checkpoint", str(mask_run["run_dir"] / "best.pt"),
        "--split", "val", "--device", "cpu", "--seeds", "2026",
        "--output", str(output),
    ])
    report = diag.run_controlled(args)
    assert report["checkpoint_sha256"] == sha256_of_file(
        mask_run["run_dir"] / "best.pt"
    )
    assert len(report["donor_protocol_sha256"]) == 64
    assert report["dataset_binding"] == "checkpoint_v5_fingerprint_and_v4_parent"
    protocol = json.loads((output / "donor_protocol.json").read_text(encoding="utf-8"))
    assert protocol["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert protocol["data_fingerprint"] == chain["fingerprint"]


def test_v1_diagnostics_still_bind_the_v4_state(chain, scratch_dir):
    """The historical v1 path keeps working and stays bound to V4."""
    import scripts.diagnose_material_mask as diag

    checkpoint = _gap_checkpoint(scratch_dir, chain, "diag_v1.pt")
    output = scratch_dir / "diag_v1_out"
    args = diag.parse_args([
        "--mode", "controlled", "--data", str(chain["v5"]),
        "--v4", str(chain["v4"]), "--checkpoint", str(checkpoint),
        "--split", "val", "--device", "cpu", "--seeds", "2026",
        "--output", str(output),
    ])
    report = diag.run_controlled(args)
    assert report["checkpoint_pooling_mode"] == POOLING_GAP
    assert report["dataset_binding"] == "checkpoint_v4_fingerprint"


# ---------------------------------------------------------------------------
# Fix 4: save-time evidence, fallback routing, finiteness


def test_save_time_comparison_matches_the_live_model(chain, mask_run):
    """The rebuilt checkpoint reproduces the live logits captured at save time."""
    fixed = materialize_fixed_batch(chain["v5"], "gray_polar", 64, 3, True)
    observer = _SaveObserver(fixed, "cpu")
    model, _, _ = _load_mask_model(mask_run)
    observer(mask_run["run_dir"] / "best.pt", "best", 2, model)
    assert fixed["names"] and observer.records
    results = compare_saved_checkpoints(
        observer.records, fixed, "cpu", "gray_polar", 64
    )
    assert all(entry["ok"] for entry in results)
    assert all(entry["max_abs_error"] == 0.0 for entry in results)


def test_corrupted_reload_must_fail_the_save_time_comparison(chain, mask_run):
    """A reload that does not reproduce the saved weights must be detected."""
    fixed = materialize_fixed_batch(chain["v5"], "gray_polar", 64, 2, True)
    observer = _SaveObserver(fixed, "cpu")
    model, _, _ = _load_mask_model(mask_run)
    observer(mask_run["run_dir"] / "best.pt", "best", 2, model)

    def corrupt(rebuilt):
        # A real weight difference: scaling every parameter cannot be absorbed
        # by the head's LayerNorm the way a single small perturbation can.
        with torch.no_grad():
            for parameter in rebuilt.parameters():
                parameter.mul_(1.5)

    results = compare_saved_checkpoints(
        observer.records, fixed, "cpu", "gray_polar", 64, mutate=corrupt
    )
    assert not any(entry["ok"] for entry in results)
    assert results[0]["max_abs_error"] > 1e-2


def test_save_time_comparison_survives_a_later_best_overwrite(
    chain, scratch_dir, monkeypatch
):
    """best.pt is legitimately rewritten by a later epoch.

    The epoch-1 save must still compare against the bytes written at epoch 1,
    not against the final file (which now holds epoch 2).
    """
    import scripts.train_material_fusion as train_module
    from .conftest import write_mask_experiment_config, write_synthetic_pretrained

    run_root = scratch_dir / "runs" / "train"
    run_root.mkdir(parents=True, exist_ok=True)
    pretrained = write_synthetic_pretrained(scratch_dir / "mit_overwrite.pth")
    config_path = write_mask_experiment_config(
        scratch_dir / "overwrite.yaml", chain["v5"], pretrained,
        chain["fingerprint"], chain["parent"], variant="gray_polar",
        effective_batch=4, epochs=2, warmup_epochs=1, imgsz=64,
    )
    scripted = [
        {"accuracy": 0.2, "macro_f1": 0.2, "per_class_recall": {"0": 0.2}},
        {"accuracy": 0.9, "macro_f1": 0.9, "per_class_recall": {"0": 0.9}},
    ]
    calls = {"n": 0}

    def fake_evaluate(model, loader, device, limit_batches=0, with_instance_mask=False):
        result = scripted[min(calls["n"], len(scripted) - 1)]
        calls["n"] += 1
        return result

    monkeypatch.setattr(train_module, "evaluate_model", fake_evaluate)
    fixed = materialize_fixed_batch(chain["v5"], "gray_polar", 64, 2, True)
    observer = _SaveObserver(fixed, "cpu")
    previous = train_module.SAVE_OBSERVER
    train_module.SAVE_OBSERVER = observer
    try:
        with monkeypatch.context() as patch:
            patch.setattr(train_module, "RUNS_ROOT", run_root)
            code = train_module.main([
                "--run-id", "overwrite_probe", "--config", str(config_path),
                "--variant", "gray_polar", "--device", "cpu",
                "--epochs", "2", "--warmup-epochs", "1", "--limit-batches", "1",
            ])
    finally:
        train_module.SAVE_OBSERVER = previous
    assert code == 0
    roles = [record["role"] for record in observer.records]
    assert roles.count("best") == 2, "best.pt must have been rewritten at epoch 2"
    final_best = read_material_checkpoint_metadata(
        run_root / "overwrite_probe" / "material_fusion" / "best.pt"
    )
    assert final_best.epoch == 2
    results = compare_saved_checkpoints(
        observer.records, fixed, "cpu", "gray_polar", 64
    )
    assert all(entry["ok"] for entry in results), results
    assert [entry["epoch"] for entry in results] == [1, 2, 2]


def test_save_observer_is_read_only(chain, mask_run):
    """The observation restores module modes and consumes no RNG."""
    fixed = materialize_fixed_batch(chain["v5"], "gray_polar", 64, 2, True)
    observer = _SaveObserver(fixed, "cpu")
    model, _, _ = _load_mask_model(mask_run)
    model.train()
    before = {name: module.training for name, module in model.named_modules()}
    before_params = {name: p.detach().clone() for name, p in model.named_parameters()}
    before_buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    rng_before = torch.random.get_rng_state()

    observer(mask_run["run_dir"] / "best.pt", "best", 2, model)

    assert {name: m.training for name, m in model.named_modules()} == before
    for name, value in before_params.items():
        assert torch.equal(value, dict(model.named_parameters())[name]), name
    for name, value in before_buffers.items():
        assert torch.equal(value, dict(model.named_buffers())[name]), name
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert observer.records[0]["rng_state_unchanged"] is True
    assert observer.records[0]["module_modes_restored"] is True


def test_fallback_routing_functionally_routes_a_mixed_batch(chain, mask_run):
    """Invalid samples use the historical gray model, valid ones the v2 model."""
    from scripts.verify_material_mask import _ForcedInvalidDataset

    model, metadata, _ = _load_mask_model(mask_run)
    fallback = _stub_fallback()
    paths = _val_paths(chain["v5"], 4)
    forced = {0, 2}
    dataset = _ForcedInvalidDataset(
        MaterialFusionDataset(paths, variant="gray_polar", imgsz=64,
                              with_instance_mask=True),
        forced, dual=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    result = verify_fallback_routing(model, fallback, loader, "cpu", forced)
    assert result["quality_matches_forced"] is True
    assert result["invalid_rows"] == len(forced)
    assert result["fallback_rows"] == len(forced)
    assert result["used_fallback_matches"] is True
    assert result["predictions_match_expected"] is True
    assert result["invalid_predictions_from_fallback"] is True
    assert result["valid_predictions_from_model"] is True


def test_wrong_route_injection_must_fail_the_check(chain, mask_run):
    """A routing that also sends valid samples to the fallback must be caught."""
    from scripts.verify_material_mask import _ForcedInvalidDataset

    model, metadata, _ = _load_mask_model(mask_run)
    fallback = _stub_fallback()
    paths = _val_paths(chain["v5"], 4)
    forced = {0, 2}
    loader = torch.utils.data.DataLoader(
        _ForcedInvalidDataset(
            MaterialFusionDataset(paths, variant="gray_polar", imgsz=64,
                                  with_instance_mask=True),
            forced, dual=True,
        ),
        batch_size=4, shuffle=False,
    )

    def wrong_route(model_, fallback_, rgb, aux, invalid, mask=None):
        # sends everything through the fallback
        logits = fallback_(rgb)
        return logits.argmax(dim=1), logits, torch.ones_like(invalid)

    result = verify_fallback_routing(
        model, fallback, loader, "cpu", forced, route_fn=wrong_route
    )
    # The routing call pattern is the deterministic discriminator: a route that
    # sends valid samples through the fallback cannot pass.
    assert result["used_fallback_matches"] is False
    assert result["fallback_rows"] != result["invalid_rows"]
    assert not (
        result["used_fallback_matches"]
        and result["predictions_match_expected"]
        and result["fallback_rows"] == result["invalid_rows"]
    )


def test_finiteness_check_detects_a_non_finite_record(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"epoch": 1, "train_loss": 1.0, "val_macro_f1": 0.5,
                    "val_accuracy": 0.5, "gradients_finite": True}) + "\n",
        encoding="utf-8",
    )
    assert check_finiteness(good)["ok"] is True

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"epoch": 1, "train_loss": float("nan"), "val_macro_f1": 0.5,
                    "val_accuracy": 0.5, "gradients_finite": True}) + "\n",
        encoding="utf-8",
    )
    assert check_finiteness(bad)["ok"] is False

    gradients = tmp_path / "grad.jsonl"
    gradients.write_text(
        json.dumps({"epoch": 1, "train_loss": 1.0, "val_macro_f1": 0.5,
                    "val_accuracy": 0.5, "gradients_finite": False}) + "\n",
        encoding="utf-8",
    )
    assert check_finiteness(gradients)["ok"] is False


def _load_mask_model(mask_run):
    from models.material_fusion import load_material_checkpoint

    return load_material_checkpoint(
        mask_run["run_dir"] / "best.pt", variant="gray_polar", imgsz=64,
        pooling_mode=POOLING_INSTANCE_MASK,
    )


def _stub_fallback():
    """A deterministic stand-in for the historical gray Model B (4 logits)."""

    class _Stub(torch.nn.Module):
        is_dual = False

        def forward(self, rgb):
            mean = rgb.mean(dim=(2, 3))
            return torch.stack([mean[:, index % 3] for index in range(4)], dim=1)

    return _Stub()


def _val_paths(v5_root, count):
    from models.polar_fusion import read_manifest_split

    return list(read_manifest_split(v5_root, "val")[0])[:count]


# ---------------------------------------------------------------------------
# Fix 5: the annotation source digest is recomputed at admission


def _source_label_path(v5_root: Path) -> Path:
    with (v5_root / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return resolve_source_path(row["label_path"])


def test_admission_recomputes_the_label_source_digest(chain):
    """The untouched sources still pass, and the digests are recomputed."""
    with (chain["v5"] / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report = verify_mask_sources(
        rows,
        expected_label_source_sha256=json.loads(
            (chain["v5"] / "dataset_audit.json").read_text(encoding="utf-8")
        )["label_source_sha256"],
        expected_sample_mapping_sha256=json.loads(
            (chain["v5"] / "dataset_audit.json").read_text(encoding="utf-8")
        )["sample_mapping_sha256"],
    )
    assert report["label_files"] > 0
    assert report["manifest_rows"] == len(rows)


def _relocated_chain(chain, scratch_dir, name):
    """A self-consistent copy of the chain whose labels live in the scratch.

    The V5 manifest is re-pointed at the copied source (still project-relative,
    so admission can resolve it) and re-audited, so the copied dataset passes
    admission until a copied annotation is actually changed.
    """
    import shutil

    from scripts.prepare_material_mask_dataset import (
        audit_mask_dataset,
        records_from_manifest,
    )

    root = scratch_dir / name
    root.mkdir(parents=True)
    source = root / "source"
    shutil.copytree(chain["source"], source)
    v5 = root / "v5"
    shutil.copytree(chain["v5"], v5)

    manifest_path = v5 / "dataset_manifest.csv"
    with manifest_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = list(rows[0].keys())
    project_root = Path(__file__).resolve().parents[2]
    for row in rows:
        original = resolve_source_path(row["label_path"])
        relative = original.relative_to(chain["source"])
        relocated = (source / relative).resolve()
        row["label_path"] = relocated.relative_to(project_root).as_posix()
        left = Path(row["left_path"])
        row["left_path"] = left.as_posix()
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    records = records_from_manifest(v5)
    expected_splits = {
        split: sum(1 for r in records if r.split == split)
        for split in ("train", "val", "test")
    }
    audit_mask_dataset(
        output_root=v5, records=records, class_names=list(chain["v5"].glob("train").__next__().parent.name for _ in [0]) if False else [
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish",
        ],
        v4_root=chain["v4"], expected_splits=expected_splits, crop_pad=10,
        workspace=project_root, expected_parent_fingerprint=chain["parent"],
    )
    return v5, source


def test_relocated_chain_passes_before_any_edit(chain, scratch_dir):
    v5, _ = _relocated_chain(chain, scratch_dir, "reloc_ok")
    report = admit_mask_dataset(v5, expected_parent_fingerprint=chain["parent"])
    assert report["splits"]["val"] > 0


def test_admission_refuses_an_edited_source_annotation(chain, scratch_dir):
    """Editing a real (temporary) source annotation must be detected."""
    v5, source = _relocated_chain(chain, scratch_dir, "reloc_edit")
    label = _source_label_path(v5)
    assert str(label).startswith(str(source))
    original = label.read_text(encoding="utf-8")
    label.write_text(original.replace("0.05", "0.5", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="annotation sources changed"):
        admit_mask_dataset(v5, expected_parent_fingerprint=chain["parent"])
    label.write_text(original, encoding="utf-8")
    admit_mask_dataset(v5, expected_parent_fingerprint=chain["parent"])


def test_admission_refuses_a_deleted_source_annotation(chain, scratch_dir):
    v5, source = _relocated_chain(chain, scratch_dir, "reloc_delete")
    label = _source_label_path(v5)
    assert str(label).startswith(str(source))
    backup = label.with_suffix(".bak")
    label.rename(backup)
    try:
        with pytest.raises(ValueError, match="missing"):
            admit_mask_dataset(v5, expected_parent_fingerprint=chain["parent"])
    finally:
        backup.rename(label)
    admit_mask_dataset(v5, expected_parent_fingerprint=chain["parent"])


def test_admission_refuses_a_changed_sample_mapping(chain):
    with (chain["v5"] / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    audit = json.loads((chain["v5"] / "dataset_audit.json").read_text(encoding="utf-8"))
    tampered = [dict(row) for row in rows]
    tampered[0]["crop_x1"] = str(int(tampered[0]["crop_x1"]) + 1)
    with pytest.raises(ValueError, match="mapping"):
        verify_mask_sources(
            tampered,
            expected_label_source_sha256=audit["label_source_sha256"],
            expected_sample_mapping_sha256=audit["sample_mapping_sha256"],
        )


@pytest.mark.parametrize(
    "value, match",
    [
        ("", "empty source path"),
        ("C:/Windows/system32/drivers/etc/hosts", "absolute"),
        ("/etc/passwd", "absolute"),
        ("../../../etc/passwd", "outside the project root"),
        ("datasets/../../escape.txt", "outside the project root"),
    ],
)
def test_source_paths_are_constrained(value, match):
    with pytest.raises(ValueError, match=match):
        resolve_source_path(value)


def test_real_v5_admission_recomputes_its_sources():
    """The real V5 dataset still passes with the recomputed source digests."""
    root = Path("datasets/underwater_cls_fusion_v5_instance_mask")
    if not root.is_dir():
        pytest.skip("the real V5 dataset is not present")
    report = admit_mask_dataset(root)
    assert report["audit_fingerprint"] == (
        "bb9e3661fb1e71e94056893f6dfc50b9e3dee1c604f430ac0727d75362d32982"
    )
    assert report["source_label_files"] == 1121
    assert report["label_source_sha256"] == (
        "46579c1ca92834cfc27b91c570df797513f9a2b602d896c388612b33e59d63c8"
    )
    assert report["sample_mapping_sha256"] == (
        "fd2de70f3b8da08a0b92b1dce813bd1b2b8f29d2af9b32da3ca31107dd9aa7c6"
    )


class TestSmokeFormalAdmission:
    """The bounded-smoke verifier must judge a v2 smoke run as a v2 run.

    The production check used to build its task without a protocol, so it fell
    back to v1 and refused the smoke directory because its experiment is not
    ``material_fusion_v1`` -- a reason with no budget information in it. These
    tests cover the shared helper the production verifier now calls, plus the
    defect signature itself.
    """

    def test_smoke_admission_uses_the_v2_mask_protocol(
        self, scratch_dir, isolated_mask_runs, chain
    ):
        from scripts.verify_material_mask import smoke_formal_admission

        task = isolated_mask_runs.by_task["gray_s2026"]
        fabricate_completed_mask_run(
            task, chain["v5"], chain["fingerprint"], chain["parent"],
            smoke=True, limit_batches=3,
            config_overrides={"epochs": 2, "warmup_epochs": 1},
        )
        admitted, reason = smoke_formal_admission(
            task.run_id, task.run_dir, task.variant, task.seed
        )
        assert not admitted
        assert "fixed protocol" in reason, reason
        assert "epochs" in reason and "limit_batches" in reason, reason
        assert "not this task's protocol" not in reason, reason

    def test_the_v1_default_would_have_refused_for_the_wrong_reason(
        self, scratch_dir, isolated_mask_runs, chain
    ):
        """Characterises the defect: validating a v2 smoke run as v1."""
        from scripts.run_material_experiments import (
            V1_SPEC,
            ExperimentTask,
            validate_completed_task,
        )

        task = isolated_mask_runs.by_task["gray_s2026"]
        fabricate_completed_mask_run(
            task, chain["v5"], chain["fingerprint"], chain["parent"],
            smoke=True, limit_batches=3,
        )
        v1_task = ExperimentTask(
            task_id=task.task_id, variant=task.variant, seed=task.seed,
            run_id=task.run_id, run_dir=task.run_dir,  # no protocol -> v1 default
        )
        admitted, reason = validate_completed_task(v1_task, V1_SPEC)
        assert not admitted
        assert "not this task's protocol" in reason, reason
        assert "limit_batches" not in reason, reason

    def test_smoke_isolation_is_not_satisfied_by_an_unrelated_refusal(self):
        from scripts.verify_material_mask import smoke_refused_on_budget

        assert smoke_refused_on_budget(False, "run directory does not exist") is False
        assert smoke_refused_on_budget(
            False,
            "run config experiment 'material_fusion_v1' is not this task's "
            "protocol 'material_fusion_v2_mask'",
        ) is False
        assert smoke_refused_on_budget(True, "ok") is False
        assert smoke_refused_on_budget(
            False,
            "run config does not match the fixed protocol: epochs: config=2 "
            "expected=30; smoke: config=True expected=False; limit_batches: "
            "config=3 expected=0",
        ) is True

    def test_real_smoke_directory_is_refused_on_budget_by_the_production_helper(self):
        run_dir = (
            ROOT / "runs" / "train"
            / "mask_a1fix_smoke_gray_polar_a1fix_20260926T035110Z" / "material_fusion"
        )
        if not run_dir.is_dir():
            pytest.skip("the recorded v2 smoke run is not present in this checkout")
        from scripts.verify_material_mask import smoke_formal_admission

        run_config = json.loads((run_dir / "run_config.json").read_text("utf-8"))
        config = run_config["config"]
        admitted, reason = smoke_formal_admission(
            run_config["run_id"], run_dir, config["variant"], config["seed"]
        )
        assert not admitted
        assert "fixed protocol" in reason, reason
        assert "limit_batches" in reason, reason
        assert "not this task's protocol" not in reason, reason
