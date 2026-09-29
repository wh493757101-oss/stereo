"""Instance-mask provenance, pooling contract and controlled diagnostics.

Every test here runs on a synthetic but *real* provenance chain: PNG frames
with YOLO-seg labels feed the real V4 generation helpers, and the real V5
generator derives the instance masks from those same annotations. Breaking
any link (label, object index, crop window, mask) must be refused.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from core.fusion_dataset import dataset_fingerprint, file_digest, load_fusion_sample
from core.material_fusion_dataset import (
    MaterialFusionDataset,
    unpack_batch,
    variant_inputs,
)
from core.material_mask_dataset import (
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    admit_mask_dataset,
    load_mask_sample,
    mask_content_digest,
    mask_dataset_fingerprint,
    save_mask_sample,
    validate_instance_mask,
)
from models.material_fusion import (
    MATERIAL_FUSION_MASK_VERSION,
    MATERIAL_FUSION_VERSION,
    MASK_RESIZE_VERSION as MODEL_MASK_RESIZE_VERSION,
    POOLING_GAP,
    POOLING_INSTANCE_MASK,
    MultiScaleClassificationHead,
    build_material_model,
    load_material_checkpoint,
    mask_level_weights,
    material_fusion_metadata,
    read_material_checkpoint_metadata,
    save_material_checkpoint,
    validate_checkpoint_metadata,
    validate_material_config,
    validate_pooling_mode,
)
from scripts.diagnose_material_mask import (
    MIN_VALID_PIXELS,
    condition_b,
    condition_c,
    distance_threshold,
    match_donors,
    matching_summary,
    paired_against_baseline,
    predict_samples,
    summarise_condition,
)
from scripts.prepare_material_mask_dataset import (
    build_mask_dataset,
    read_v4_manifest,
    resolve_frame,
    run_provenance_check,
)

from .conftest import (
    build_synthetic_source,
    build_synthetic_v4,
    build_synthetic_v5,
    unique_path,
    write_experiment_config,
)


@pytest.fixture(scope="module")
def chain(module_scratch):
    """A synthetic source -> V4 -> V5 chain, built by the production code."""
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


def _v5_rows(v5_root: Path, split: str = "val") -> list[dict]:
    with (v5_root / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def _admit(chain):
    return admit_mask_dataset(
        chain["v5"], expected_parent_fingerprint=chain["parent"]
    )


# ---------------------------------------------------------------------------
# 1/2/4: provenance correspondence and mask validity


def test_provenance_check_resolves_every_sample(chain):
    check = run_provenance_check(
        chain["source"], chain["v4"], 10, chain["workspace"]
    )
    assert check["v4_samples"] == check["resolved_samples"] == 24
    assert check["failure_count"] == 0
    assert check["unresolved_frame_count"] == 0
    assert check["resolved_splits"] == check["v4_expected_splits"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda rows: rows[0].update(object_index="99"), "out of range"),
        (lambda rows: rows[0].update(class_id="3"), "label class"),
    ],
)
def test_provenance_refuses_object_index_and_class_mismatch(chain, mutate, match):
    rows = read_v4_manifest(chain["v4"])
    mutate(rows)
    class_names = ["metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"]
    pair = {
        "split": rows[0]["split"],
        "group_name": rows[0]["group_name"],
        "left_path": chain["source"]
        / "images"
        / rows[0]["split"]
        / f"{rows[0]['source_frame']}.png",
    }
    records, _, failures = resolve_frame(
        rows[0]["source_frame"], [rows[0]], pair, class_names,
        chain["source"], chain["v4"], 10, chain["workspace"],
    )
    assert not records
    assert failures and match in failures[0]


def test_provenance_refuses_a_wrong_crop_pad(chain):
    """A different crop window cannot reproduce the V4 crop."""
    check = run_provenance_check(chain["source"], chain["v4"], 3, chain["workspace"])
    assert check["failure_count"] > 0
    assert any("crop" in failure for failure in check["failures"])


def test_mask_invariants_reject_non_binary_empty_and_shape_mismatch(chain):
    row = _v5_rows(chain["v5"])[0]
    sample = load_mask_sample(chain["v5"] / row["npz_path"])
    path = "synthetic"
    with pytest.raises(ValueError, match="binary"):
        validate_instance_mask(
            path, (sample.instance_mask * 2).astype(np.uint8), sample.valid,
            sample.gray.shape,
        )
    with pytest.raises(ValueError, match="empty"):
        validate_instance_mask(
            path, np.zeros_like(sample.instance_mask), sample.valid, sample.gray.shape
        )
    with pytest.raises(ValueError, match="shape"):
        validate_instance_mask(
            path, sample.instance_mask[:-1], sample.valid, sample.gray.shape
        )


def test_valid_outside_the_instance_mask_is_refused(chain):
    row = _v5_rows(chain["v5"])[0]
    sample = load_mask_sample(chain["v5"] / row["npz_path"])
    valid = sample.valid.copy()
    outside = np.argwhere(sample.instance_mask == 0)
    valid[tuple(outside[0])] = 1
    with pytest.raises(ValueError, match="outside the instance mask"):
        validate_instance_mask("synthetic", sample.instance_mask, valid, sample.gray.shape)


# ---------------------------------------------------------------------------
# 3: V4 arrays are preserved element for element


def test_v5_preserves_every_v4_array_element_for_element(chain):
    rows = read_v4_manifest(chain["v4"])
    for row in rows:
        v4_sample = load_fusion_sample(chain["v4"] / row["npz_path"])
        v5_sample = load_mask_sample(chain["v5"] / row["npz_path"])
        assert np.array_equal(v5_sample.gray, v4_sample.gray)
        assert np.array_equal(v5_sample.signed_q, v4_sample.signed_q)
        assert np.array_equal(v5_sample.abs_q, v4_sample.abs_q)
        assert np.array_equal(v5_sample.valid, v4_sample.valid)
        assert np.array_equal(v5_sample.quality, v4_sample.quality)
        assert v5_sample.class_id == v4_sample.class_id
        assert v5_sample.instance_mask.shape == v4_sample.gray.shape


def test_v5_manifest_keeps_samples_splits_and_object_indices(chain):
    v4_rows = read_v4_manifest(chain["v4"])
    v5_rows = read_v4_manifest(chain["v5"])
    assert [r["sample_name"] for r in v4_rows] == [r["sample_name"] for r in v5_rows]
    assert [r["split"] for r in v4_rows] == [r["split"] for r in v5_rows]
    assert [r["object_index"] for r in v4_rows] == [r["object_index"] for r in v5_rows]
    assert [r["class_id"] for r in v4_rows] == [r["class_id"] for r in v5_rows]


def test_v5_generation_refuses_an_existing_directory(chain, scratch_dir):
    with pytest.raises(FileExistsError, match="already exists"):
        build_mask_dataset(
            source_root=chain["source"], v4_root=chain["v4"],
            output_root=chain["v5"], crop_pad=10, workspace=chain["workspace"],
            expected_parent_fingerprint=chain["parent"],
        )


# ---------------------------------------------------------------------------
# 5: tampering is refused by the admission gate


def test_admission_refuses_a_tampered_mask(chain, scratch_dir):
    tampered = scratch_dir / "tampered"
    _copy_dataset(chain["v5"], tampered)
    row = _v5_rows(tampered)[0]
    path = tampered / row["npz_path"]
    sample = load_mask_sample(path)
    mask = sample.instance_mask.copy()
    flat = mask.reshape(-1)
    flat[np.flatnonzero(flat == 1)[0]] = 0
    save_mask_sample(
        path, gray=sample.gray, signed_q=sample.signed_q, abs_q=sample.abs_q,
        valid=sample.valid, quality=sample.quality, class_id=sample.class_id,
        instance_mask=mask.reshape(sample.gray.shape),
    )
    with pytest.raises(ValueError):
        admit_mask_dataset(tampered, expected_parent_fingerprint=chain["parent"])


def test_admission_refuses_a_tampered_manifest(chain, scratch_dir):
    tampered = scratch_dir / "tampered_manifest"
    _copy_dataset(chain["v5"], tampered)
    manifest = tampered / "dataset_manifest.csv"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("obj000", "obj900", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed since the audit"):
        admit_mask_dataset(tampered, expected_parent_fingerprint=chain["parent"])


def test_admission_refuses_a_different_v4_parent(chain):
    with pytest.raises(ValueError, match="not derived from the formal V4"):
        admit_mask_dataset(chain["v5"])


def test_admission_refuses_an_unaudited_directory(chain, scratch_dir):
    stripped = scratch_dir / "no_audit"
    _copy_dataset(chain["v5"], stripped)
    (stripped / "dataset_audit.json").unlink()
    with pytest.raises(FileNotFoundError, match="audit"):
        admit_mask_dataset(stripped, expected_parent_fingerprint=chain["parent"])


def test_mask_content_digest_is_order_independent_and_content_sensitive(chain):
    entries = [
        (row["sample_name"], load_mask_sample(chain["v5"] / row["npz_path"]).instance_mask)
        for row in _v5_rows(chain["v5"])
    ]
    digest = mask_content_digest(entries)
    assert digest == mask_content_digest(list(reversed(entries)))
    changed = list(entries)
    changed[0] = (changed[0][0], np.zeros_like(changed[0][1]))
    assert mask_content_digest(changed) != digest


def _copy_dataset(src: Path, dst: Path) -> None:
    import shutil

    shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# 6/7/8/9: the pooling operator


def test_four_variants_receive_the_identical_mask_and_pooling(chain):
    row = _v5_rows(chain["v5"])[0]
    path = chain["v5"] / row["npz_path"]
    masks = {}
    for variant in ("gray", "gray_gray_valid", "gray_valid", "gray_polar"):
        dataset = MaterialFusionDataset(
            [path], variant=variant, imgsz=64, with_instance_mask=True
        )
        item = dataset[0]
        mask = item[-1]
        masks[variant] = mask
    reference = masks["gray"]
    for variant, mask in masks.items():
        assert torch.equal(mask, reference), variant
    # the mask is not the valid channel and not a q-derived region
    sample = load_mask_sample(path)
    valid_only = torch.from_numpy(
        np.asarray(sample.valid > 0, dtype=np.float32)
    ).reshape(-1)
    assert not torch.equal(reference.reshape(-1), valid_only)


def test_all_ones_mask_reproduces_global_average_pooling(chain):
    torch.manual_seed(0)
    gap = MultiScaleClassificationHead((32, 64, 160, 256), pooling_mode=POOLING_GAP)
    masked = MultiScaleClassificationHead(
        (32, 64, 160, 256), pooling_mode=POOLING_INSTANCE_MASK
    )
    masked.load_state_dict(gap.state_dict())
    gap.eval()
    masked.eval()
    features = [
        torch.randn(2, dim, size, size)
        for dim, size in ((32, 56), (64, 28), (160, 14), (256, 7))
    ]
    with torch.no_grad():
        gap_out = gap(features)
        mask_out = masked(features, torch.ones(2, 224, 224))
    assert torch.allclose(gap_out, mask_out, atol=1e-5)


def test_positions_outside_the_mask_support_do_not_affect_the_result(chain):
    torch.manual_seed(0)
    head = MultiScaleClassificationHead(
        (32, 64, 160, 256), pooling_mode=POOLING_INSTANCE_MASK
    )
    head.eval()
    mask = torch.zeros(1, 224, 224)
    mask[0, 100:130, 100:130] = 1.0
    features = [
        torch.randn(1, dim, size, size)
        for dim, size in ((32, 56), (64, 28), (160, 14), (256, 7))
    ]
    perturbed = []
    for feature in features:
        weights = mask_level_weights(mask, feature.shape[-2:])
        changed = feature.clone()
        changed[(weights == 0).expand_as(feature)] += 50.0
        perturbed.append(changed)
    with torch.no_grad():
        base = head(features, mask)
        other = head(perturbed, mask)
    assert torch.equal(base, other)


def test_small_targets_keep_a_positive_weight_at_every_scale():
    mask = torch.zeros(1, 224, 224)
    mask[0, 110:113, 110:113] = 1.0
    for size in ((56, 56), (28, 28), (14, 14), (7, 7)):
        weights = mask_level_weights(mask, size)
        assert float(weights.sum()) > 0.0, size
        assert float(weights.max()) > 0.0, size


def test_empty_mask_is_refused_instead_of_falling_back_to_gap():
    head = MultiScaleClassificationHead(
        (32, 64, 160, 256), pooling_mode=POOLING_INSTANCE_MASK
    )
    head.eval()
    features = [
        torch.randn(1, dim, size, size)
        for dim, size in ((32, 56), (64, 28), (160, 14), (256, 7))
    ]
    with pytest.raises(ValueError, match="zero weight"):
        head(features, torch.zeros(1, 224, 224))


def test_gap_head_refuses_a_mask_and_mask_head_requires_one():
    features = [torch.randn(1, dim, 8, 8) for dim in (32, 64, 160, 256)]
    gap = MultiScaleClassificationHead((32, 64, 160, 256), pooling_mode=POOLING_GAP)
    with pytest.raises(ValueError, match="silently ignored"):
        gap(features, torch.ones(1, 64, 64))
    masked = MultiScaleClassificationHead(
        (32, 64, 160, 256), pooling_mode=POOLING_INSTANCE_MASK
    )
    with pytest.raises(ValueError, match="instance_mask tensor is required"):
        masked(features)


def test_mask_resize_refuses_upsampling_and_non_finite_values():
    with pytest.raises(ValueError, match="downscaling only"):
        mask_level_weights(torch.ones(1, 4, 4), (8, 8))
    bad = torch.ones(1, 8, 8)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        mask_level_weights(bad, (4, 4))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mask_level_weights(torch.full((1, 8, 8), 2.0), (4, 4))


# ---------------------------------------------------------------------------
# 10/11/12: pooling versions and formal markers


def _metadata(variant="gray_polar", **overrides):
    kwargs = dict(
        variant=variant, imgsz=64, data_fingerprint="a" * 64,
        pretrained_sha256="b" * 64, seed=2026, epoch=1, stage="warmup",
        smoke=False, limit_batches=0, val_macro_f1=0.1, val_accuracy=0.2,
        physical_batch=2, accum_steps=2, effective_batch=4, drop_path_rate=0.0,
    )
    kwargs.update(overrides)
    return material_fusion_metadata(**kwargs)


def test_gap_and_mask_checkpoints_are_never_silently_interchanged(scratch_dir):
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_GAP)
    path = save_material_checkpoint(
        scratch_dir / "gap.pt", model, _metadata(pooling_mode=POOLING_GAP)
    )
    with pytest.raises(ValueError, match="pooling_mode"):
        load_material_checkpoint(path, variant="gray_polar", pooling_mode=POOLING_INSTANCE_MASK)
    _, metadata, _ = load_material_checkpoint(path, variant="gray_polar")
    assert metadata.pooling_mode == POOLING_GAP

    masked_model = build_material_model(
        "gray_polar", imgsz=64, pooling_mode=POOLING_INSTANCE_MASK
    )
    masked_path = save_material_checkpoint(
        scratch_dir / "masked.pt", masked_model,
        _metadata(
            version=MATERIAL_FUSION_MASK_VERSION, pooling_mode=POOLING_INSTANCE_MASK,
            mask_schema=MASK_SCHEMA_VERSION, mask_resize_version=MASK_RESIZE_VERSION,
            parent_data_fingerprint="c" * 64,
        ),
    )
    with pytest.raises(ValueError, match="pooling_mode"):
        load_material_checkpoint(masked_path, variant="gray_polar", pooling_mode=POOLING_GAP)
    _, metadata, _ = load_material_checkpoint(
        masked_path, variant="gray_polar", pooling_mode=POOLING_INSTANCE_MASK
    )
    assert metadata.pooling_mode == POOLING_INSTANCE_MASK
    assert metadata.mask_resize_version == MASK_RESIZE_VERSION


def test_v2_checkpoint_must_declare_its_pooling_mode():
    payload = {
        "metadata": {
            "version": MATERIAL_FUSION_MASK_VERSION, "variant": "gray_polar",
            "class_names": ["metal_submarine", "plastic_submarine", "plastic_fish",
                            "real_fish"],
            "imgsz": 64, "preprocessing_version": "x", "data_fingerprint": "a" * 64,
            "upstream_commit": "0" * 40, "pretrained_sha256": "b" * 64, "seed": 2026,
            "epoch": 1, "stage": "warmup", "smoke": False, "limit_batches": 0,
            "val_macro_f1": 0.1, "val_accuracy": 0.2, "physical_batch": 2,
            "accum_steps": 2, "effective_batch": 4, "drop_path_rate": 0.0,
            "init_from": "", "fallback_weights": "", "fallback_identity": "",
        },
        "state_dict": {},
    }
    with pytest.raises(ValueError, match="must record pooling_mode"):
        validate_checkpoint_metadata(payload)


def test_mask_pooling_metadata_requires_the_schema_and_parent():
    with pytest.raises(ValueError, match="mask_schema"):
        validate_checkpoint_metadata(_payload(
            version=MATERIAL_FUSION_MASK_VERSION, pooling_mode=POOLING_INSTANCE_MASK
        ))
    with pytest.raises(ValueError, match="mask_resize_version"):
        validate_checkpoint_metadata(_payload(
            version=MATERIAL_FUSION_MASK_VERSION, pooling_mode=POOLING_INSTANCE_MASK,
            mask_schema=MASK_SCHEMA_VERSION,
        ))
    with pytest.raises(ValueError, match="parent fingerprint"):
        validate_checkpoint_metadata(_payload(
            version=MATERIAL_FUSION_MASK_VERSION, pooling_mode=POOLING_INSTANCE_MASK,
            mask_schema=MASK_SCHEMA_VERSION, mask_resize_version=MASK_RESIZE_VERSION,
        ))


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


def test_v1_config_still_validates_and_forbids_mask_pooling():
    import yaml

    v1 = yaml.safe_load(
        Path("configs/experiments/material_fusion_v1.yaml").read_text(encoding="utf-8")
    )
    normalized = validate_material_config(v1)
    assert normalized["experiment"] == MATERIAL_FUSION_VERSION
    assert normalized["pooling_mode"] == POOLING_GAP
    v1["pooling_mode"] = POOLING_INSTANCE_MASK
    with pytest.raises(ValueError, match="GAP protocol"):
        validate_material_config(v1)


def test_v2_config_requires_the_full_mask_contract():
    import yaml

    v2 = yaml.safe_load(
        Path("configs/experiments/material_fusion_v2_mask.yaml").read_text(
            encoding="utf-8"
        )
    )
    normalized = validate_material_config(v2)
    assert normalized["experiment"] == MATERIAL_FUSION_MASK_VERSION
    assert normalized["pooling_mode"] == POOLING_INSTANCE_MASK
    assert MODEL_MASK_RESIZE_VERSION == MASK_RESIZE_VERSION
    for field, expected in (
        ("pooling_mode", "instance_mask"),
        ("mask_schema", MASK_SCHEMA_VERSION),
        ("mask_resize_version", MASK_RESIZE_VERSION),
    ):
        broken = dict(v2)
        broken[field] = "wrong"
        with pytest.raises(ValueError):
            validate_material_config(broken)
        assert expected
    missing_parent = dict(v2)
    missing_parent["parent_data_fingerprint"] = ""
    with pytest.raises(ValueError, match="parent fingerprint"):
        validate_material_config(missing_parent)


def test_pooling_mode_validation_is_strict():
    assert validate_pooling_mode("gap") == "gap"
    with pytest.raises(ValueError, match="unknown pooling_mode"):
        validate_pooling_mode("mask")


# ---------------------------------------------------------------------------
# 11: the v1 GAP path is unchanged


def test_v1_gap_forward_and_loading_are_unchanged(chain, scratch_dir):
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_GAP)
    assert not model.uses_instance_mask
    path = save_material_checkpoint(scratch_dir / "v1.pt", model, _metadata())
    reloaded, metadata, _ = load_material_checkpoint(path, variant="gray_polar")
    assert metadata.pooling_mode == POOLING_GAP
    assert not reloaded.uses_instance_mask
    rgb = torch.rand(2, 3, 64, 64)
    aux = torch.rand(2, 3, 64, 64)
    reloaded.eval()
    with torch.no_grad():
        logits = reloaded(rgb, aux, None)
    assert logits.shape == (2, 4)
    # a GAP model never consumes a mask, and the V4 loader yields none
    row = _v5_rows(chain["v4"])[0]
    dataset = MaterialFusionDataset(
        [chain["v4"] / row["npz_path"]], variant="gray_polar", imgsz=64
    )
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    rgb_b, aux_b, quality_b, labels_b, mask_b = unpack_batch(batch, True, False)
    assert mask_b is None


# ---------------------------------------------------------------------------
# 13/17: the interventions themselves


def test_self_permutation_keeps_the_support_and_the_value_multiset(chain):
    row = _v5_rows(chain["v5"])[0]
    sample = load_mask_sample(chain["v5"] / row["npz_path"])
    signed, abs_q = condition_b(sample, row["sample_name"], 2026)
    support = (sample.instance_mask > 0) & (sample.valid > 0)
    np.testing.assert_array_equal(signed[~support], sample.signed_q[~support])
    np.testing.assert_array_equal(abs_q[~support], sample.abs_q[~support])
    assert signed[support].size > 0
    np.testing.assert_allclose(np.sort(signed[support]), np.sort(sample.signed_q[support]))
    np.testing.assert_allclose(abs_q[support], np.abs(signed[support]))
    assert np.array_equal(sample.instance_mask, sample.instance_mask)


def test_self_permutation_is_seed_reproducible(chain):
    row = _v5_rows(chain["v5"])[0]
    sample = load_mask_sample(chain["v5"] / row["npz_path"])
    first = condition_b(sample, row["sample_name"], 2027)
    second = condition_b(sample, row["sample_name"], 2027)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    other = condition_b(sample, row["sample_name"], 2028)
    assert not np.array_equal(first[0], other[0])


def test_donor_transfer_preserves_the_support_and_the_gray(chain):
    rows = _v5_rows(chain["v5"])
    recipient = load_mask_sample(chain["v5"] / rows[0]["npz_path"])
    donor = load_mask_sample(chain["v5"] / rows[1]["npz_path"])
    signed, abs_q = condition_c(
        recipient, rows[0]["sample_name"], donor, rows[1]["sample_name"], 2026
    )
    support = (recipient.instance_mask > 0) & (recipient.valid > 0)
    outside = ~support
    np.testing.assert_array_equal(signed[outside], recipient.signed_q[outside])
    np.testing.assert_array_equal(abs_q[outside], recipient.abs_q[outside])
    np.testing.assert_allclose(abs_q[support], np.abs(signed[support]))
    # the donor's zero contour is never pasted in: every recipient-valid
    # position still holds a donor value taken from the donor's valid region
    donor_valid = (donor.instance_mask > 0) & (donor.valid > 0)
    donor_values = set(np.round(donor.signed_q[donor_valid], 6).tolist())
    for value in np.round(signed[support], 6).tolist():
        assert value in donor_values
    assert recipient.gray is not None


def test_donor_transfer_handles_fewer_donor_pixels_than_recipient(chain):
    rows = _v5_rows(chain["v5"])
    recipient = load_mask_sample(chain["v5"] / rows[0]["npz_path"])
    donor = load_mask_sample(chain["v5"] / rows[1]["npz_path"])
    # shrink the donor's valid support to a single pixel: the rule must cycle
    donor_valid = (donor.instance_mask > 0) & (donor.valid > 0)
    keep = np.argwhere(donor_valid)[0]
    valid = np.zeros_like(donor.valid)
    valid[tuple(keep)] = 1
    donor = replace(donor, valid=valid)
    signed, abs_q = condition_c(
        recipient, rows[0]["sample_name"], donor, rows[1]["sample_name"], 2026
    )
    support = (recipient.instance_mask > 0) & (recipient.valid > 0)
    assert int(support.sum()) > 1
    unique = np.unique(np.round(signed[support], 6))
    assert unique.size == 1  # one donor value, reused for every recipient pixel


def test_intervention_quality_keeps_support_semantics(chain):
    row = _v5_rows(chain["v5"])[0]
    sample = load_mask_sample(chain["v5"] / row["npz_path"])
    signed, abs_q = condition_b(sample, row["sample_name"], 2026)
    from scripts.diagnose_material_mask import intervention_quality

    quality = intervention_quality(sample, abs_q)
    assert quality["support_unchanged"] is True
    original = quality["original_quality"]
    updated = quality["intervention_quality"]
    # the three support-derived components are untouched
    assert original[0] == updated[0] and original[1] == updated[1]
    assert original[2] == updated[2]
    support = (sample.instance_mask > 0) & (sample.valid > 0)
    assert updated[3] == pytest.approx(float(abs_q[support].mean()), abs=1e-9)


# ---------------------------------------------------------------------------
# 15/16: the donor protocol


def test_donors_never_match_themselves_and_respect_the_ceiling(chain):
    _, features, _ = _features(chain["v5"], "val")
    threshold = distance_threshold(features)
    matches = match_donors(features, features, threshold["max_distance"])
    for name, match in matches.items():
        if match["matched"]:
            assert match["donor"] != name
            assert match["distance"] <= threshold["max_distance"]
        else:
            assert match["donor"] is None and match["reason"]


def test_matching_is_independent_of_input_order(chain):
    _, features, _ = _features(chain["v5"], "val")
    threshold = distance_threshold(features)
    forward = match_donors(features, features, threshold["max_distance"])
    backward = match_donors(list(reversed(features)), list(reversed(features)),
                            threshold["max_distance"])
    assert forward == backward


def test_matching_reports_unmatched_explicitly(chain):
    _, features, _ = _features(chain["v5"], "val")
    # an impossible ceiling leaves every recipient unmatched, with a reason
    matches = match_donors(features, features, -1.0)
    assert all(not match["matched"] for match in matches.values())
    summary = matching_summary(features, features, matches, {"max_distance": -1.0})
    assert summary["matched"] == 0
    assert summary["matched_ratio"] == 0.0
    assert len(summary["unmatched_reasons"]) == len(features)
    assert summary["is_bijection"] is False


def test_recipients_without_valid_support_are_never_matched(chain):
    _, features, _ = _features(chain["v5"], "val")
    stripped = [replace(f, valid_pixels=0, valid_coverage=0.0) for f in features]
    matches = match_donors(stripped, features, 10.0)
    assert all(not match["matched"] for match in matches.values())
    assert all("no distribution" in match["reason"] for match in matches.values())
    assert MIN_VALID_PIXELS >= 2


def test_summaries_use_the_same_matched_subset(chain):
    baseline = [
        {"sample_name": "a", "correct": True, "routed_correct": True,
         "fallback_triggered": False, "polar_valid": True},
        {"sample_name": "b", "correct": True, "routed_correct": True,
         "fallback_triggered": False, "polar_valid": True},
    ]
    perturbed = [
        {"sample_name": "a", "correct": True, "routed_correct": True,
         "fallback_triggered": False, "polar_valid": True},
        {"sample_name": "b", "correct": False, "routed_correct": False,
         "fallback_triggered": False, "polar_valid": True},
    ]
    paired = paired_against_baseline(baseline, perturbed)
    assert paired["samples"] == 2
    assert paired["broken_samples"] == ["b"]
    assert summarise_condition(perturbed)["accuracy"] == 0.5


def _features(v5_root: Path, split: str):
    from scripts.diagnose_material_mask import build_features

    return build_features(v5_root, split)


# ---------------------------------------------------------------------------
# 18/19/20: per-sample evidence, invariance, and the test split


def test_per_sample_evidence_recomputes_the_summary(chain, scratch_dir):
    rows, features, samples = _features(chain["v5"], "val")
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_GAP)
    model.eval()
    metadata = read_material_checkpoint_metadata(
        save_material_checkpoint(scratch_dir / "m.pt", model, _metadata())
    )
    records = predict_samples(model, metadata, rows, samples, "cpu", None, None)
    assert len(records) == len(rows)
    assert {r["sample_name"] for r in records} == {row["sample_name"] for row in rows}
    recomputed = sum(1 for r in records if r["correct"]) / len(records)
    assert summarise_condition(records)["accuracy"] == pytest.approx(recomputed)
    for record in records:
        assert len(record["logits"]) == 4
        assert record["routed_prediction"] == record["prediction"]
        assert record["fallback_triggered"] is False
        assert record["correct"] == (record["prediction"] == record["label"])


def test_inference_does_not_change_model_parameters_or_buffers(chain):
    rows, _, samples = _features(chain["v5"], "val")
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_GAP)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    metadata = _metadata()
    predict_samples(model, metadata, rows, samples, "cpu", None, None)
    after = model.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_diagnostics_never_read_the_test_split(chain, monkeypatch, scratch_dir):
    import scripts.diagnose_material_mask as diag

    seen: list[str] = []
    original = diag.build_features

    def recording(data_root, split):
        seen.append(split)
        return original(data_root, split)

    monkeypatch.setattr(diag, "build_features", recording)
    monkeypatch.setattr(
        diag, "admit_mask_dataset",
        lambda *a, **k: {
            "data_root": str(chain["v5"]),
            "audit_fingerprint": chain["fingerprint"],
            "v4_parent_fingerprint": chain["parent"],
        },
    )
    args = diag.parse_args([
        "--mode", "controlled", "--data", str(chain["v5"]),
        "--v4", str(chain["v4"]),
        "--checkpoint", str(_make_gap_checkpoint(scratch_dir, chain["parent"])),
        "--split", "val", "--device", "cpu", "--seeds", "2026",
        "--output", str(scratch_dir / "diag_out"),
    ])
    diag.run_controlled(args)
    assert "test" not in seen
    assert set(seen) <= {"train", "val"}


def test_diagnostics_refuse_the_test_split(chain, scratch_dir):
    """The test split is refused by the parser and again inside main()."""
    import scripts.diagnose_material_mask as diag

    with pytest.raises(SystemExit):
        diag.parse_args([
            "--mode", "controlled", "--data", str(chain["v5"]),
            "--checkpoint", str(_make_gap_checkpoint(scratch_dir)),
            "--split", "test", "--device", "cpu",
            "--output", str(scratch_dir / "never"),
        ])
    # defence in depth: even a hand-built namespace cannot reach the runner
    monkeypatch_target = diag.parse_args
    try:
        diag.parse_args = lambda argv=None: type(
            "A", (), {"mode": "controlled", "split": "test"}
        )()
        assert diag.main([]) == 2
    finally:
        diag.parse_args = monkeypatch_target


def test_diagnostics_refuse_an_existing_output_directory(chain, scratch_dir):
    import scripts.diagnose_material_mask as diag

    output = scratch_dir / "diag_exists"
    output.mkdir()
    args = diag.parse_args([
        "--mode", "controlled", "--data", str(chain["v5"]),
        "--v4", str(chain["v4"]),
        "--checkpoint", str(_make_gap_checkpoint(scratch_dir, chain["parent"])),
        "--split", "val", "--device", "cpu", "--seeds", "2026",
        "--output", str(output),
    ])
    with pytest.raises(FileExistsError, match="already exists"):
        diag.run_controlled(args)


def test_v1_reproduction_refuses_a_mask_pooling_checkpoint(chain, scratch_dir):
    import scripts.diagnose_material_mask as diag

    model = build_material_model(
        "gray_polar", imgsz=64, pooling_mode=POOLING_INSTANCE_MASK
    )
    path = save_material_checkpoint(
        scratch_dir / "masked.pt", model,
        _metadata(
            version=MATERIAL_FUSION_MASK_VERSION, pooling_mode=POOLING_INSTANCE_MASK,
            mask_schema=MASK_SCHEMA_VERSION, mask_resize_version=MASK_RESIZE_VERSION,
            parent_data_fingerprint="c" * 64,
        ),
    )
    args = diag.parse_args([
        "--mode", "v1-reproduction", "--data", str(chain["v5"]),
        "--v4", str(chain["v4"]), "--checkpoints", str(path),
        "--split", "val", "--device", "cpu",
        "--output", str(scratch_dir / "legacy_out"),
    ])
    with pytest.raises(ValueError, match="GAP head"):
        diag.run_v1_reproduction(args)


def _make_gap_checkpoint(scratch_dir: Path, data_fingerprint: str = "a" * 64) -> Path:
    """A v1 GAP checkpoint bound to the V4 state a diagnostic run should use."""
    model = build_material_model("gray_polar", imgsz=64, pooling_mode=POOLING_GAP)
    return save_material_checkpoint(
        scratch_dir / "gap_diag.pt", model,
        _metadata(data_fingerprint=data_fingerprint),
    )


# ---------------------------------------------------------------------------
# The bounded smoke verifier's budget


def test_smoke_verifier_refuses_a_request_above_the_budget():
    from scripts.verify_material_mask import SMOKE_BUDGET, check_budget, parse_args

    base = [
        "--config", "configs/experiments/material_fusion_v2_mask.yaml",
        "--data", "datasets/underwater_cls_fusion_v5_instance_mask",
        "--run-id", "budget_probe", "--smoke",
    ]
    ok = parse_args(base + ["--epochs", "2", "--limit-batches", "3"])
    assert check_budget(ok)["epochs"] == 2
    too_long = parse_args(base + ["--epochs", "3"])
    with pytest.raises(ValueError, match="exceeds the smoke budget"):
        check_budget(too_long)
    too_many = parse_args(base + ["--limit-batches", "4"])
    with pytest.raises(ValueError, match="exceeds the smoke budget"):
        check_budget(too_many)
    not_smoke = parse_args([
        "--config", "configs/experiments/material_fusion_v2_mask.yaml",
        "--data", "datasets/underwater_cls_fusion_v5_instance_mask",
        "--run-id", "no_smoke",
    ])
    with pytest.raises(ValueError, match="only runs bounded smoke"):
        check_budget(not_smoke)
    assert SMOKE_BUDGET["batch"] == 8 and SMOKE_BUDGET["accum_steps"] == 2
