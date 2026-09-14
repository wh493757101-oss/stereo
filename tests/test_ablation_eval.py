"""Tests for scripts.eval_ablation: manifest pairing, pure metrics,
deterministic capture-group bootstrap, and the decision gate."""

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.eval_ablation as ab

MANIFEST_FIELDS = [
    "sample_name", "split", "group_name", "class_id", "class_name",
    "source_frame", "object_index", "stereo_valid", "stereo_reason",
    "valid_ratio", "disparity", "gray_path", "polar_path",
]


def make_row(name, split, group, class_id, class_name, rel_path):
    return {
        "sample_name": name, "split": split, "group_name": group,
        "class_id": str(class_id), "class_name": class_name,
        "source_frame": "frame_x", "object_index": "0",
        "stereo_valid": "true", "stereo_reason": "ok",
        "valid_ratio": "1.0", "disparity": "10.0",
        "gray_path": rel_path, "polar_path": rel_path,
    }


def write_manifest(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def workdir():
    path = Path(tempfile.mkdtemp(prefix="ablation_eval_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def base_rows():
    return [
        make_row("s1", "test", "G/0 NTU", 0, "metal_submarine",
                 "test/metal_submarine/s1.png"),
        make_row("s2", "test", "G/0 NTU", 1, "plastic_fish",
                 "test/plastic_fish/s2.png"),
        make_row("s3", "test", "G/10 NTU", 1, "plastic_fish",
                 "test/plastic_fish/s3.png"),
        make_row("s4", "train", "G/0 NTU", 0, "metal_submarine",
                 "train/metal_submarine/s4.png"),
    ]


def test_default_weights_target_grouped_formal_run():
    assert ab.DEFAULT_GRAY_WEIGHTS == (
        "runs/train/run_20260913_initial/model_b-gray/weights/best.pt"
    )
    assert ab.DEFAULT_POLAR_WEIGHTS == (
        "runs/train/run_20260913_initial/model_b-polar/weights/best.pt"
    )


class TestPairManifests:
    def test_matching_manifests_pair_successfully(self):
        pairs = ab.pair_manifests(
            {r["sample_name"]: r for r in base_rows()},
            {r["sample_name"]: r for r in base_rows()},
        )
        assert len(pairs) == 4
        by_name = {p.sample_name: p for p in pairs}
        assert by_name["s1"].group_name == "G/0 NTU"
        assert by_name["s2"].class_id == 1
        assert by_name["s2"].relative_path == "test/plastic_fish/s2.png"

    def test_class_mismatch_raises(self):
        gray = base_rows()
        polar = base_rows()
        polar[0]["class_name"] = "plastic_fish"
        with pytest.raises(ab.ManifestMismatchError, match="class_name"):
            ab.pair_manifests(
                {r["sample_name"]: r for r in gray},
                {r["sample_name"]: r for r in polar},
            )

    def test_relative_path_mismatch_raises(self):
        gray = base_rows()
        polar = base_rows()
        polar[1]["polar_path"] = "test/plastic_fish/other.png"
        with pytest.raises(ab.ManifestMismatchError, match="path"):
            ab.pair_manifests(
                {r["sample_name"]: r for r in gray},
                {r["sample_name"]: r for r in polar},
            )

    def test_group_mismatch_raises(self):
        gray = base_rows()
        polar = base_rows()
        polar[2]["group_name"] = "G/5 NTU"
        with pytest.raises(ab.ManifestMismatchError, match="group_name"):
            ab.pair_manifests(
                {r["sample_name"]: r for r in gray},
                {r["sample_name"]: r for r in polar},
            )

    def test_missing_sample_raises(self):
        gray = base_rows()
        polar = base_rows()[1:]
        with pytest.raises(ab.ManifestMismatchError, match="only in"):
            ab.pair_manifests(
                {r["sample_name"]: r for r in gray},
                {r["sample_name"]: r for r in polar},
            )

    def test_split_filter(self):
        pairs = ab.pair_manifests(
            {r["sample_name"]: r for r in base_rows()},
            {r["sample_name"]: r for r in base_rows()},
        )
        selected = ab.select_split(pairs, "test")
        assert [p.sample_name for p in selected] == ["s1", "s2", "s3"]
        with pytest.raises(ValueError, match="split"):
            ab.select_split(pairs, "nope")


class TestComputeMetrics:
    def test_hand_computed_metrics(self):
        y_true = [0, 0, 1, 1, 2]
        y_pred = [0, 1, 1, 1, 2]
        metrics = ab.compute_metrics(y_true, y_pred, ["a", "b", "c"])

        assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
        assert metrics["accuracy"] == pytest.approx(0.8)
        pc = metrics["per_class"]
        assert pc["a"]["precision"] == pytest.approx(1.0)
        assert pc["a"]["recall"] == pytest.approx(0.5)
        assert pc["a"]["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
        assert pc["b"]["precision"] == pytest.approx(2 / 3)
        assert pc["b"]["recall"] == pytest.approx(1.0)
        assert pc["b"]["f1"] == pytest.approx(0.8)
        assert pc["c"]["f1"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(
            (pc["a"]["f1"] + pc["b"]["f1"] + pc["c"]["f1"]) / 3
        )

    def test_zero_support_class_scores_zero(self):
        metrics = ab.compute_metrics([0, 0], [0, 0], ["a", "ghost"])
        assert metrics["per_class"]["ghost"]["f1"] == 0.0
        assert metrics["per_class"]["a"]["f1"] == 1.0

    def test_perfect_predictions(self):
        metrics = ab.compute_metrics([0, 1, 1], [0, 1, 1], ["a", "b"])
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0


class TestGroupBootstrap:
    def make_inputs(self):
        # Two groups with identical structure: model A misses one class-1
        # sample per group, model B is perfect, so every group-only
        # resample yields a strictly positive macro-F1 diff.
        y_true = [0, 0, 1, 1, 0, 0, 1, 1]
        y_pred_a = [0, 0, 0, 1, 0, 0, 0, 1]
        y_pred_b = [0, 0, 1, 1, 0, 0, 1, 1]
        groups = ["g0", "g0", "g0", "g0", "g1", "g1", "g1", "g1"]
        return y_true, y_pred_a, y_pred_b, groups

    def test_deterministic_same_seed(self):
        y_true, a, b, groups = self.make_inputs()
        r1 = ab.group_bootstrap_diff_ci(y_true, a, b, groups, n_bootstraps=50, seed=2026)
        r2 = ab.group_bootstrap_diff_ci(y_true, a, b, groups, n_bootstraps=50, seed=2026)
        assert r1 == r2

    def test_better_model_gets_positive_ci(self):
        y_true, a, b, groups = self.make_inputs()
        result = ab.group_bootstrap_diff_ci(y_true, a, b, groups,
                                            n_bootstraps=200, seed=2026)
        assert result["ci_low"] > 0.0
        assert result["ci_high"] <= 1.0 + 1e-9
        assert result["n_groups"] == 2

    def test_identical_predictions_zero_ci(self):
        y_true, a, _, groups = self.make_inputs()
        result = ab.group_bootstrap_diff_ci(y_true, a, a, groups,
                                            n_bootstraps=50, seed=2026)
        assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0

    def test_worse_model_gets_negative_ci(self):
        y_true, a, b, groups = self.make_inputs()
        result = ab.group_bootstrap_diff_ci(y_true, b, a, groups,
                                            n_bootstraps=200, seed=2026)
        assert result["ci_high"] < 0.0

    def test_groups_survive_together(self):
        # Both groups have identical structure, so every resample diff is
        # the same strictly positive value; the CI never dips to 0.
        y_true, a, b, groups = self.make_inputs()
        result = ab.group_bootstrap_diff_ci(y_true, a, b, groups,
                                            n_bootstraps=200, seed=2026)
        assert result["ci_low"] == pytest.approx(result["ci_high"])


class TestDecisionGate:
    def test_pass_when_gain_and_ci_both_good(self):
        d = ab.make_decision(0.70, 0.76, ci_low=0.01)
        assert d["gate"] == "pass"
        assert d["decision"] == "use_polar_aided"

    def test_fail_when_gain_below_threshold(self):
        d = ab.make_decision(0.70, 0.72, ci_low=0.05)
        assert d["gate"] == "fail"
        assert d["decision"] == "use_gray_only"

    def test_fail_when_ci_lower_bound_not_positive(self):
        d = ab.make_decision(0.70, 0.80, ci_low=-0.01)
        assert d["gate"] == "fail"
        assert d["decision"] == "use_gray_only"
        assert "no proven gain" in d["reason"]

    def test_custom_min_gain(self):
        d = ab.make_decision(0.70, 0.74, ci_low=0.01, min_gain=0.05)
        assert d["gate"] == "fail"


class TestEndToEndEvaluate:
    def test_evaluate_with_mock_classifiers(self, workdir, monkeypatch):
        gray_root = workdir / "gray"
        polar_root = workdir / "polar"
        rows = [
            make_row(f"s{i}", "test", f"G/{i % 3} NTU", i % 2,
                     "metal_submarine" if i % 2 == 0 else "plastic_fish",
                     f"test/c{i % 2}/s{i}.png")
            for i in range(6)
        ]
        for root in (gray_root, polar_root):
            root.mkdir(parents=True, exist_ok=True)
            write_manifest(root / "dataset_manifest.csv", rows)
            for i in range(6):
                (root / f"test/c{i % 2}").mkdir(parents=True, exist_ok=True)
                (root / f"test/c{i % 2}" / f"s{i}.png").write_bytes(b"\x89PNG\r\n")

        import cv2
        for i in range(6):
            # Truth is encoded in the pixel value so the mock classifier
            # can "read" it: gray always answers 0, polar reads the label.
            img = np.full((4, 4, 3), 128 + (i % 2), dtype=np.uint8)
            cv2.imwrite(str(gray_root / f"test/c{i % 2}/s{i}.png"), img)
            cv2.imwrite(str(polar_root / f"test/c{i % 2}/s{i}.png"), img)

        class FakeResult:
            def __init__(self, top1_id, top1_name):
                self.top1_id = top1_id
                self.top1_name = top1_name
                self.valid = True

        class FakeModel:
            def __init__(self, weights, device="cpu"):
                self.weights = weights

            def predict(self, image):
                if "polar" in self.weights:
                    label = int(image[0, 0, 0]) - 128
                else:
                    label = 0
                name = "metal_submarine" if label == 0 else "plastic_fish"
                return FakeResult(label, name)

        monkeypatch.setattr(
            "models.classification.ClassificationModel", FakeModel,
        )

        report = ab.evaluate(
            gray_root, polar_root,
            "gray.pt", "polar.pt",
            split="test", n_bootstraps=50, seed=2026,
        )

        assert report["n_samples"] == 6
        assert report["gray"]["n_samples"] == 6
        assert report["polar"]["n_samples"] == 6
        assert report["gray"]["accuracy"] == pytest.approx(0.5)
        assert report["polar"]["accuracy"] == pytest.approx(1.0)
        assert report["bootstrap"]["seed"] == 2026
        assert report["bootstrap"]["resamples"] == 50
        assert report["decision"]["gate"] == "pass"
        assert report["decision"]["decision"] == "use_polar_aided"

    def test_evaluate_fails_on_manifest_mismatch(self, workdir):
        gray_root = workdir / "gray"
        polar_root = workdir / "polar"
        gray_root.mkdir()
        polar_root.mkdir()
        rows = base_rows()
        write_manifest(gray_root / "dataset_manifest.csv", rows)
        mismatched = [dict(r) for r in rows]
        mismatched[0]["class_name"] = "plastic_fish"
        write_manifest(polar_root / "dataset_manifest.csv", mismatched)

        with pytest.raises(ab.ManifestMismatchError):
            ab.evaluate(gray_root, polar_root, "g.pt", "p.pt", split="test")

    def test_load_manifest_rejects_empty(self, workdir):
        path = workdir / "empty.csv"
        write_manifest(path, [])
        with pytest.raises(ValueError, match="empty"):
            ab.load_manifest(path)


class TestCanonicalIdMap:
    @staticmethod
    def pairs_with(*edits: tuple[int, str, str]):
        """base_rows() with (row_index, field, value) edits applied."""
        rows = base_rows()
        for index, field, value in edits:
            rows[index][field] = value
        return ab.pair_manifests(
            {r["sample_name"]: r for r in rows},
            {r["sample_name"]: r for r in rows},
        )

    def test_builds_one_to_one_map(self):
        pairs = ab.pair_manifests(
            {r["sample_name"]: r for r in base_rows()},
            {r["sample_name"]: r for r in base_rows()},
        )
        assert ab.build_canonical_id_map(pairs) == {
            "metal_submarine": 0, "plastic_fish": 1,
        }

    def test_conflicting_ids_for_one_name_raise(self):
        # Row 2 (plastic_fish, id 1) becomes metal_submarine while keeping
        # id 1, so metal_submarine maps to both 0 and 1.
        pairs = self.pairs_with((2, "class_name", "metal_submarine"))
        with pytest.raises(ab.ClassMappingError, match="conflicting"):
            ab.build_canonical_id_map(pairs)

    def test_conflicting_names_for_one_id_raise(self):
        # Row 3 (metal_submarine, id 0) is renamed plastic_fish, so id 0
        # maps to two different names.
        pairs = self.pairs_with((3, "class_name", "plastic_fish"))
        with pytest.raises(ab.ClassMappingError, match="conflicting"):
            ab.build_canonical_id_map(pairs)

    def test_non_contiguous_ids_raise(self):
        pairs = self.pairs_with((1, "class_id", "5"), (2, "class_id", "5"))
        with pytest.raises(ab.ClassMappingError, match="contiguous"):
            ab.build_canonical_id_map(pairs)


class StubResult:
    def __init__(self, top1_id, top1_name):
        self.top1_id = top1_id
        self.top1_name = top1_name
        self.valid = True


class TestPredictionNameMapping:
    canonical = {"metal_submarine": 0, "plastic_fish": 1}

    def test_names_map_to_canonical_ids(self):
        results = [StubResult(7, "plastic_fish"), StubResult(0, "metal_submarine")]
        assert ab.map_predictions_to_canonical(
            results, ["a.png", "b.png"], self.canonical, "gray") == [1, 0]

    def test_unknown_name_raises(self):
        results = [StubResult(0, "whale")]
        with pytest.raises(ab.ClassMappingError, match="unknown class name 'whale'"):
            ab.map_predictions_to_canonical(results, ["a.png"], self.canonical, "gray")

    def test_checkpoint_fallback_name_raises(self):
        results = [StubResult(1, "class_1")]
        with pytest.raises(ab.ClassMappingError, match="names"):
            ab.map_predictions_to_canonical(results, ["a.png"], self.canonical, "gray")

    def test_missing_name_raises(self):
        results = [StubResult(0, None)]
        with pytest.raises(ab.ClassMappingError, match="no class name"):
            ab.map_predictions_to_canonical(results, ["a.png"], self.canonical, "gray")


class TestSwappedCheckpointIds:
    """Ultralytics assigns checkpoint ids alphabetically, which can swap
    them relative to manifest ids; correct names must still score perfect."""

    def test_swapped_ids_but_correct_names_score_perfect(
            self, workdir, monkeypatch):
        gray_root = workdir / "gray"
        polar_root = workdir / "polar"
        # Manifest: metal_submarine=0, plastic_fish=1.
        rows = [
            make_row(f"s{i}", "test", f"G/{i % 2} NTU", i % 2,
                     "metal_submarine" if i % 2 == 0 else "plastic_fish",
                     f"test/c{i % 2}/s{i}.png")
            for i in range(6)
        ]
        for root in (gray_root, polar_root):
            root.mkdir(parents=True, exist_ok=True)
            write_manifest(root / "dataset_manifest.csv", rows)

        import cv2
        for root in (gray_root, polar_root):
            for i in range(6):
                crop_dir = root / f"test/c{i % 2}"
                crop_dir.mkdir(parents=True, exist_ok=True)
                img = np.full((4, 4, 3), 128 + (i % 2), dtype=np.uint8)
                cv2.imwrite(str(crop_dir / f"s{i}.png"), img)

        # Checkpoint names in alphabetical order: id 0 is plastic_fish,
        # id 1 is metal_submarine — the numeric ids are swapped vs the
        # manifest, but the names are correct.
        ckpt_names = {0: "plastic_fish", 1: "metal_submarine"}

        class FakeModel:
            def __init__(self, weights, device="cpu"):
                self.weights = weights

            def predict(self, image):
                truth = int(image[0, 0, 0]) - 128  # 0 or 1, manifest space
                truth_name = "metal_submarine" if truth == 0 else "plastic_fish"
                ckpt_id = next(k for k, v in ckpt_names.items() if v == truth_name)
                return StubResult(ckpt_id, truth_name)

        monkeypatch.setattr(
            "models.classification.ClassificationModel", FakeModel,
        )

        report = ab.evaluate(
            gray_root, polar_root,
            "gray.pt", "polar.pt",
            split="test", n_bootstraps=50, seed=2026,
        )

        for side in ("gray", "polar"):
            assert report[side]["accuracy"] == pytest.approx(1.0)
            assert report[side]["macro_f1"] == pytest.approx(1.0)
            assert report[side]["confusion_matrix"] == [[3, 0], [0, 3]]

    def test_unknown_checkpoint_name_fails_evaluate(self, workdir, monkeypatch):
        gray_root = workdir / "gray"
        polar_root = workdir / "polar"
        rows = [make_row("s1", "test", "G/0 NTU", 0,
                         "metal_submarine", "test/c0/s1.png")]
        for root in (gray_root, polar_root):
            root.mkdir(parents=True, exist_ok=True)
            write_manifest(root / "dataset_manifest.csv", rows)

        import cv2
        for root in (gray_root, polar_root):
            (root / "test/c0").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / "test/c0/s1.png"),
                        np.zeros((4, 4, 3), dtype=np.uint8))

        class FakeModel:
            def __init__(self, weights, device="cpu"):
                self.weights = weights

            def predict(self, image):
                return StubResult(0, "class_0")

        monkeypatch.setattr(
            "models.classification.ClassificationModel", FakeModel,
        )

        with pytest.raises(ab.ClassMappingError, match="class_0"):
            ab.evaluate(gray_root, polar_root, "gray.pt", "polar.pt", split="test")
