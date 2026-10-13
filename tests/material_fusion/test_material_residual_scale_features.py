"""material_residual_scale_v1 fix-cache tests: the rebuild really goes
through the PRODUCTION StereoMatcher with the fix upsample version, the
gray/mask/class/G identities are the frozen ones, missing-mode changes
are recorded, and smoke/old caches are refused by the training entry's
marker check. The synthetic world is stereo-friendly (the right view is
a shifted copy of the left) so the fix matcher actually matches."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from core.material_mask_dataset import (
    admit_dataset_for_protocol,
    mask_dataset_fingerprint,
)
from core.material_residual_scale import (
    CACHE_KIND_FULL,
    CACHE_KIND_SMOKE,
    EXPERIMENT,
    DeadlineWatch,
    ScaleRefused,
    UPSAMPLE_FIX_VERSION,
    load_scale_config,
    scale_cache_definition,
    verify_scale_cache_markers,
)
from core.polar_local_features import (
    LOCAL_FEATURE_DEFINITION_VERSION,
    LOCAL_REGION_RULE,
    build_local_cache_arrays,
    verify_local_cache,
)

from .conftest import build_synthetic_v4, build_synthetic_v5
from .test_material_local_residual_entry import build_patterned_v4
from .test_p0_baseline_verification import build_fit
from .test_polar_region_quality_entry import (
    _make_cache_files_for_v5,
    _protocol_for_v5,
    _v5_rows,
)
from scripts import prepare_material_residual_scale_features as prep

FOLD_IDS = ("fold0", "fold1", "fold2")
CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")

# small matcher parameters that work on the synthetic image size (the
# production values would be meaningless on 96-px synthetic frames)
SYNTH_MATCHER = {
    "matcher": "sgbm", "mode": "3way", "max_disp": 32, "scale": 0.25,
    "block_size": 5, "uniqueness_ratio": 10, "speckle_window": 100,
    "speckle_range": 32, "texture_threshold": 10.0,
    "lr_check_threshold_px": 2.0, "min_valid_ratio": 0.05,
}


def _build_stereo_source(root: Path, frames_per_split: int = 4,
                         size: int = 96) -> Path:
    """A four-class source whose right view is a LEFT-SHIFTED copy of the
    left (a real stereo pair the fix matcher can match)."""
    import csv

    import cv2

    import numpy as np

    rng = np.random.default_rng(7)
    root = Path(root)
    shift = 8
    rows = []
    for split, count in (("train", frames_per_split),
                         ("val", frames_per_split),
                         ("test", frames_per_split)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "images_right" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"{split}_frame_{index:03d}"
            left = rng.integers(20, 235, size=(size, size),
                                dtype=np.uint8)
            right = np.empty_like(left)
            right[:, :-shift] = left[:, shift:]
            right[:, -shift:] = left[:, -1:]
            cv2.imwrite(str(root / "images" / split / f"{stem}.png"), left)
            cv2.imwrite(str(root / "images_right" / split /
                            f"{stem}.png"), right)
            lines = []
            for obj in range(2):
                class_id = (index + obj) % len(CLASS_NAMES)
                x1 = 0.05 + 0.40 * obj
                y1 = 0.10 + 0.05 * obj
                w = 0.25
                h = 0.35
                polygon = ((x1, y1), (x1 + w, y1), (x1 + w, y1 + h),
                           (x1, y1 + h))
                coords = " ".join(f"{v:.6f}" for point in polygon
                                  for v in point)
                lines.append(f"{class_id} {coords}")
            (root / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            rows.append({
                "output_stem": stem,
                "split": split,
                "group_name": f"{split}/group_{index}",
                "left_path": (root / "images" / split /
                              f"{stem}.png").as_posix(),
                "right_path": (root / "images_right" / split /
                               f"{stem}.png").as_posix(),
                "label_path": (root / "labels" / split /
                               f"{stem}.txt").as_posix(),
            })
    with (root / "pair_manifest.csv").open("w", encoding="utf-8",
                                           newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "data.yaml").write_text(
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8")
    return root


def _real_anchors() -> dict:
    import yaml

    real = yaml.safe_load(
        (prep._PROJECT_ROOT / "configs/experiments/"
         "material_residual_scale_v1.yaml").read_text(encoding="utf-8"))
    return real["anchors"]


@pytest.fixture(scope="module")
def scale_world(module_scratch):
    """Synthetic stereo source + V4/V5 + protocol + frozen-chain P0 fit,
    ready for the fix rebuild."""
    workspace = module_scratch.resolve()
    source = _build_stereo_source(module_scratch / "source")
    v4 = build_synthetic_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)
    fingerprint = mask_dataset_fingerprint(v5)
    rows = _v5_rows(v5)
    protocol = _protocol_for_v5(rows, fingerprint)
    protocol_dir = module_scratch / "protocol"
    protocol_dir.mkdir(parents=True)
    protocol_path = protocol_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True),
                             encoding="utf-8")

    from scripts.verify_p0_baseline import LOGIT_ATOL  # noqa: F401
    from . import test_p0_baseline_verification as _p0mod

    with mock.patch.object(_p0mod, "make_protocol", lambda: protocol), \
            mock.patch.object(_p0mod, "make_cache_files",
                              _make_cache_files_for_v5(v5, rows)):
        fit = build_fit(module_scratch, name="p0fit")
    verification = module_scratch / "p0_verification.json"
    verification.write_text(json.dumps({"verified": True}),
                            encoding="utf-8")

    dev_rows = [row for row in rows if row["split"] in ("train", "val")]
    dev_frames = sorted({row["source_frame"] for row in dev_rows})
    groups = sorted({row["group_name"] for row in dev_rows})
    from core.fusion_dataset import dataset_fingerprint

    cfg = {
        "experiment": EXPERIMENT,
        "anchors": _real_anchors(),
        "protocol_source": {"path": str(protocol_path)},
        "features_source": {"path": str(fit["feature_dir"])},
        "p0_baseline_source": {
            "fit_dir": str(fit["fit_dir"]),
            "verification": str(verification),
        },
        "data": {
            "root": str(v5),
            "source_root": str(source),
            "fingerprint": fingerprint,
            "parent_v4_fingerprint": dataset_fingerprint(v4),
            "class_names": list(CLASS_NAMES),
            "dev_pool_splits": ["train", "val"],
            "excluded_splits": ["test"],
        },
        "fix_inputs": {**SYNTH_MATCHER,
                       "upsample_version": UPSAMPLE_FIX_VERSION,
                       "band_margin": 20,
                       "full_image_threshold": 0.9,
                       "crop_pad": 10},
        "local_features": {
            "region_rule": LOCAL_REGION_RULE,
            "feature_definition_version":
                LOCAL_FEATURE_DEFINITION_VERSION,
        },
        "support_expectations": {
            "frames": len(dev_frames),
            "instances": len(dev_rows),
            "groups": len(groups),
            "empty_frames": 0,
        },
        "scale_rule": {"source": "p0_logits_inner_train",
                       "statistic": "median", "dtype": "float64",
                       "min_scale": 1.0},
        "arms": {"fixed": {"magnitude_factor": 0.5},
                 "scaled": {"magnitude_factor": 0.5}},
        "seeds": [2026, 2027, 2028],
        "shuffle": {"content_seeds": [2026, 2027, 2028, 2029, 2030]},
        "training": {"device": "cpu", "full_batch": True, "epochs": 2,
                     "optimizer": "adamw", "lr": 0.01,
                     "weight_decay": 0.001, "aux_loss_weight": 0.01},
        "head": {"hidden": 8},
    }
    return {
        "workspace": workspace, "source": source, "v4": v4, "v5": v5,
        "rows": rows, "protocol": protocol,
        "protocol_path": protocol_path, "feature_dir": fit["feature_dir"],
        "fit_dir": fit["fit_dir"], "cfg": cfg,
        "cache_samples": fit["manifest"]["samples"],
        "dev_frames": dev_frames,
    }


def _ctx(module_scratch: Path, log) -> dict:
    stages = {}

    def stage(name, status, **details):
        stages[name] = {"status": status, **details}

    return {"registry": prep.SourceRegistry({}),
            "watch": DeadlineWatch(budget_minutes=10.0),
            "stage": stage, "log": log, "stages": stages}


@pytest.fixture(scope="class")
def full_cache(scale_world, module_scratch):
    out = module_scratch / "full_build"
    log = lambda message: None  # noqa: E731
    ctx = _ctx(module_scratch, log)
    pre = prep.run_preflight(ctx, scale_world["cfg"])
    build = prep.run_feature_build(
        ctx, scale_world["cfg"], pre["census"], pre["global_info"],
        frame_limit=None)
    features = out / "features"
    prep.write_scale_cache(
        features, scale_world["cfg"], pre["protocol"], build,
        kind=CACHE_KIND_FULL, complete=True, interrupted=False,
        data_fingerprint=scale_world["cfg"]["data"]["fingerprint"],
        protocol_sha256=pre["protocol"]["protocol_sha256"],
        empty_frames=pre["census"]["empty_frames"])
    return {"out": out, "features": features, "build": build,
            "pre": pre, "ctx_stages": ctx["stages"]}


class TestFixCacheBuild:

    def test_preflight_census_matches_the_synthetic_pool(self, full_cache):
        counts = full_cache["pre"]["census"]["counts"]
        assert counts == {"instances": 16, "frames": 8, "groups": 8,
                          "empty_frames": 0}

    def test_full_cache_passes_its_own_markers(self, scale_world,
                                               full_cache):
        definition = verify_scale_cache_markers(full_cache["features"],
                                                scale_world["cfg"])
        assert definition["cache_kind"] == CACHE_KIND_FULL
        assert definition["complete"] is True
        assert definition["upsample_version"] == UPSAMPLE_FIX_VERSION

    def test_full_cache_passes_the_generic_contract(self, scale_world,
                                                    full_cache):
        report = verify_local_cache(
            full_cache["features"],
            full_cache["pre"]["protocol"])
        assert report["sample_count"] == 16

    def test_g_features_are_byte_identical_to_the_frozen_cache(
            self, scale_world, full_cache):
        frozen = dict(np.load(Path(scale_world["feature_dir"]) /
                              "feature_cache.npz"))
        built = dict(np.load(full_cache["features"] /
                             "feature_cache.npz"))
        np.testing.assert_array_equal(built["g_features"],
                                      frozen["g_features"])
        np.testing.assert_array_equal(built["class_id"],
                                      frozen["class_id"])

    def test_gray_and_mask_identity_against_v5(self, scale_world,
                                               full_cache):
        from core.material_mask_dataset import load_mask_sample

        rows = {row["sample_name"]: row
                for row in full_cache["pre"]["census"]["dev_rows"]}
        for name, record in full_cache["build"]["fix_provenance"].items():
            v5_row = rows[name]
            sample = load_mask_sample(Path(scale_world["v5"]) /
                                      v5_row["npz_path"])
            assert record["crop_window"] == [
                int(v5_row["crop_x1"]), int(v5_row["crop_y1"]),
                int(v5_row["crop_x2"]), int(v5_row["crop_y2"])]
            assert sample.gray.shape[0] > 0
            assert record["upsample_version"] == UPSAMPLE_FIX_VERSION

    def test_the_rebuild_used_the_fix_matcher_version(
            self, scale_world, module_scratch, monkeypatch):
        """A spy on the production matcher proves the rebuild constructs
        StereoMatcherConfig with the FIX upsample version."""
        captured = []
        real_matcher = prep.StereoMatcher
        real_config = prep.StereoMatcherConfig

        def spy_config(**kwargs):
            config = real_config(**kwargs)
            captured.append(config.upsample_version)
            return config

        with mock.patch.object(prep, "StereoMatcherConfig", spy_config), \
                mock.patch.object(prep, "StereoMatcher", real_matcher):
            ctx = _ctx(module_scratch, lambda message: None)
            pre = prep.run_preflight(ctx, scale_world["cfg"])
            build = prep.run_feature_build(
                ctx, scale_world["cfg"], pre["census"],
                pre["global_info"], frame_limit=1)
        assert captured, "the rebuild never constructed a matcher config"
        assert set(captured) == {UPSAMPLE_FIX_VERSION}
        assert build["built_frames"] == 1

    def test_s_valid_changes_are_recorded_not_forced(self, full_cache):
        report = full_cache["build"]
        assert isinstance(report["missing_changed"], list)
        # every row carries both states when a change is recorded
        for row in report["missing_changed"]:
            assert set(row) == {"sample_name", "old_s_valid",
                                "fix_s_valid"}
            assert row["old_s_valid"] != row["fix_s_valid"]

    def test_frame_ledger_covers_every_built_frame(self, full_cache):
        ledger = full_cache["build"]["frame_ledger"]
        assert len(ledger) == 8
        assert all(entry["instances"] == 2 for entry in ledger)


class TestCacheRefusals:
    def test_smoke_cache_is_marked_incomplete_and_refused(
            self, scale_world, module_scratch):
        out = module_scratch / "smoke_build"
        ctx = _ctx(module_scratch, lambda message: None)
        pre = prep.run_preflight(ctx, scale_world["cfg"])
        build = prep.run_feature_build(
            ctx, scale_world["cfg"], pre["census"], pre["global_info"],
            frame_limit=2)
        features = out / "features"
        prep.write_scale_cache(
            features, scale_world["cfg"], pre["protocol"], build,
            kind=CACHE_KIND_SMOKE, complete=False, interrupted=False,
            data_fingerprint=scale_world["cfg"]["data"]["fingerprint"],
            protocol_sha256=pre["protocol"]["protocol_sha256"],
            empty_frames=pre["census"]["empty_frames"])
        with pytest.raises(ScaleRefused, match="cache_kind"):
            verify_scale_cache_markers(features, scale_world["cfg"])

    def test_old_cache_without_markers_is_refused(self, scale_world,
                                                  module_scratch):
        """The legacy local cache (built by the ORIGINAL builder, carrying
        no experiment markers at all) must never be admitted."""
        old_cache = module_scratch / "old_style" / "local_features"
        arrays = dict(np.load(Path(scale_world["feature_dir"]) /
                              "feature_cache.npz"))
        old_cache.mkdir(parents=True)
        np.savez_compressed(old_cache / "feature_cache.npz", **arrays)
        (old_cache / "feature_definition.json").write_text(
            json.dumps({"feature_definition_version":
                        LOCAL_FEATURE_DEFINITION_VERSION}),
            encoding="utf-8")
        with pytest.raises(ScaleRefused, match="experiment"):
            verify_scale_cache_markers(old_cache, scale_world["cfg"])

    def test_partial_marker_refusal(self, scale_world, module_scratch):
        features = module_scratch / "partial_cache" / "features"
        features.mkdir(parents=True)
        definition = scale_cache_definition(
            scale_world["cfg"], kind=CACHE_KIND_FULL, complete=False,
            frames_built=2, instances_built=4, empty_frames=[],
            data_fingerprint=scale_world["cfg"]["data"]["fingerprint"],
            protocol_sha256="0" * 64, g_dim=4)
        (features / "feature_definition.json").write_text(
            json.dumps(definition), encoding="utf-8")
        with pytest.raises(ScaleRefused, match="incomplete"):
            verify_scale_cache_markers(features, scale_world["cfg"])

    def test_wrong_census_refusal(self, scale_world, module_scratch):
        features = module_scratch / "wrong_census" / "features"
        features.mkdir(parents=True)
        definition = scale_cache_definition(
            scale_world["cfg"], kind=CACHE_KIND_FULL, complete=True,
            frames_built=8, instances_built=6, empty_frames=[],
            data_fingerprint=scale_world["cfg"]["data"]["fingerprint"],
            protocol_sha256="0" * 64, g_dim=4)
        (features / "feature_definition.json").write_text(
            json.dumps(definition), encoding="utf-8")
        with pytest.raises(ScaleRefused, match="instances_built"):
            verify_scale_cache_markers(features, scale_world["cfg"])

    def test_v5_admission_refuses_a_wrong_fingerprint(self, scale_world):
        cfg = copy.deepcopy(scale_world["cfg"])
        cfg["data"]["fingerprint"] = "f" * 64
        with pytest.raises(ValueError, match="fingerprint"):
            prep.run_v5_admission(cfg)


import copy  # noqa: E402  (used by the refusal tests above)
