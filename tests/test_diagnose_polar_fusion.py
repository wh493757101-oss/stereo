"""Tests for scripts.diagnose_polar_fusion.

Pure-logic tests plus a tiny stub-model integration run over crafted npz
samples; no real checkpoints, no GPU, no training.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.diagnose_polar_fusion as dpf
from core.fusion_dataset import build_quality_vector, save_fusion_sample
from models.polar_fusion import GrayBackboneAdapter, PolarFusionModel

V3_CLASS_NAMES = [
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
]


@pytest.fixture
def tmp_path() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="diagnose_polar_fusion_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class TinyBackbone(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes)
        )
        torch.manual_seed(0)
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)

    def forward(self, gray):
        return self.net(gray)


def make_stub_model(num_classes=4):
    return PolarFusionModel(GrayBackboneAdapter(TinyBackbone(num_classes)), num_classes)


def write_npz(root: Path, name: str, class_id: int, polar_value: float) -> Path:
    shape = (12, 10)
    signed = np.full(shape, polar_value, np.float32)
    path = root / f"{name}.npz"
    save_fusion_sample(
        path,
        gray=np.full(shape, 60 + class_id * 20, np.uint8),
        signed_q=signed,
        abs_q=np.abs(signed),
        valid=np.ones(shape, np.uint8),
        quality=build_quality_vector(0.8, 0.9, 0.9, polar_value),
        class_id=class_id,
    )
    return path


class TestClassificationReport:
    def test_confusion_matrix_rows_are_true_columns_pred(self):
        report = dpf.classification_report([0, 1, 0, 1], [0, 0, 1, 1], 2)
        # rows = true class, columns = predicted class
        assert report["confusion_matrix"] == [[1, 1], [1, 1]]

    def test_per_class_metrics_and_accuracy(self):
        report = dpf.classification_report([0, 0, 1, 1], [0, 1, 1, 1], 2)
        assert report["accuracy"] == pytest.approx(0.75)
        assert report["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)
        class0 = report["per_class"]["0"]
        assert class0["precision"] == pytest.approx(1.0)
        assert class0["recall"] == pytest.approx(0.5)
        assert class0["f1"] == pytest.approx(2.0 / 3.0)
        assert class0["support"] == 2
        class1 = report["per_class"]["1"]
        assert class1["precision"] == pytest.approx(2.0 / 3.0)
        assert class1["recall"] == pytest.approx(1.0)

    def test_zero_denominators_are_zero_not_nan(self):
        report = dpf.classification_report([0, 0], [0, 0], 3)
        assert report["per_class"]["1"] == {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
        }
        assert report["accuracy"] == 1.0
        assert report["macro_f1"] == pytest.approx(1.0 / 3.0)
        # predicted but never true: precision denominator only
        report = dpf.classification_report([0], [1], 2)
        assert report["per_class"]["1"]["precision"] == 0.0
        assert report["per_class"]["1"]["support"] == 0


class TestChangeSummary:
    def test_counts_and_relations(self):
        labels = [0, 0, 1, 1, 2]
        gray = [0, 0, 1, 0, 2]
        fusion = [0, 1, 1, 2, 2]
        summary = dpf.change_summary(gray, fusion, labels)
        assert summary["correct_to_correct"] == 3
        assert summary["correct_to_wrong"] == 1
        assert summary["wrong_to_correct"] == 0
        assert summary["wrong_to_wrong"] == 1
        assert summary["wrong_to_wrong_changed_class"] == 1
        assert summary["changed_total"] == 2
        gray_wrong = sum(1 for l, p in zip(labels, gray) if l != p)
        fusion_wrong = sum(1 for l, p in zip(labels, fusion) if l != p)
        assert gray_wrong == summary["wrong_to_correct"] + summary["wrong_to_wrong"]
        assert fusion_wrong == summary["correct_to_wrong"] + summary["wrong_to_wrong"]

    def test_wrong_to_wrong_same_class_not_counted_as_changed(self):
        labels = [0, 1]
        gray = [2, 1]
        fusion = [2, 0]
        summary = dpf.change_summary(gray, fusion, labels)
        assert summary["wrong_to_wrong"] == 1
        assert summary["wrong_to_wrong_changed_class"] == 0
        assert summary["changed_total"] == 1


class TestDerangement:
    def test_full_bijection_without_fixed_points(self):
        perm = dpf.derangement(362, 2026)
        assert sorted(perm) == list(range(362))
        assert all(perm[i] != i for i in range(362))

    def test_deterministic_and_seed_dependent(self):
        first = dpf.derangement(50, 2027)
        assert first == dpf.derangement(50, 2027)
        assert first != dpf.derangement(50, 2028)

    def test_rejects_tiny_sizes(self):
        with pytest.raises(ValueError):
            dpf.derangement(1, 2026)


class TestBatchSlices:
    def test_covers_every_index_exactly_once(self):
        slices = dpf.batch_slices(362, 32)
        assert slices[0] == (0, 32)
        assert slices[-1] == (352, 362)
        covered = [i for start, stop in slices for i in range(start, stop)]
        assert covered == list(range(362))

    def test_exact_multiple_and_empty(self):
        assert dpf.batch_slices(64, 32) == [(0, 32), (32, 64)]
        assert dpf.batch_slices(0, 32) == []


class TestShuffledDonorConsistency:
    def test_polar_and_quality_come_from_the_same_donor(self, tmp_path):
        paths = [
            write_npz(tmp_path, "a", 0, 0.1),
            write_npz(tmp_path, "b", 1, 0.2),
            write_npz(tmp_path, "c", 2, 0.3),
        ]
        permutation = [2, 0, 1]
        polars, qualities = dpf.shuffled_polar_quality(paths, permutation, imgsz=8)
        for position, donor in enumerate(permutation):
            donor_polar, donor_quality = dpf._load_polar_quality(paths[donor], 8)
            assert torch.equal(polars[position], donor_polar)
            assert torch.equal(qualities[position], donor_quality)
        # spot check that donor polar values are actually distinct
        assert not torch.equal(polars[0], polars[1])


class TestFallbackReport:
    def test_exact_fallback_passes(self):
        gray_logits = torch.randn(3, 4)
        report = dpf._fallback_report(
            torch.zeros(3, 1), gray_logits.clone(), gray_logits
        )
        assert report == {"gate_failures": 0, "logits_failures": 0, "max_logits_diff": 0.0}

    def test_nonzero_gate_and_logit_drift_fail(self):
        gray_logits = torch.zeros(2, 4)
        gate = torch.tensor([[0.0], [1e-9]])
        final = gray_logits.clone()
        final[0, 0] = 1e-7
        report = dpf._fallback_report(gate, final, gray_logits)
        assert report["gate_failures"] == 1
        assert report["logits_failures"] == 1
        assert report["max_logits_diff"] == pytest.approx(1e-7)


class TestGrayBranchConsistency:
    def test_identical_branches_pass(self):
        module_a = TinyBackbone(4)
        module_b = TinyBackbone(4)
        module_b.load_state_dict(module_a.state_dict())
        report = dpf.verify_gray_branch(module_a, module_b)
        assert report["parameters_compared"] == sum(1 for _ in module_a.named_parameters())
        assert report["buffers_compared"] == sum(1 for _ in module_a.named_buffers())

    def test_parameter_drift_rejected(self):
        module_a = TinyBackbone(4)
        module_b = TinyBackbone(4)
        module_b.load_state_dict(module_a.state_dict())
        with torch.no_grad():
            module_b.net[2].weight += 1.0
        with pytest.raises(RuntimeError, match="drift|mismatch"):
            dpf.verify_gray_branch(module_a, module_b)


class TestCheckMetadata:
    def _inputs(self):
        import dataclasses

        from models.polar_fusion import fusion_metadata

        metadata = fusion_metadata(
            V3_CLASS_NAMES,
            imgsz=224,
            architecture="yolo26n-cls",
            gray_weights="runs/train/yolo26_formal_001/gray_fusion/best.pt",
            limit_batches=0,
            smoke=False,
        )
        fusion_config = {
            "run_id": "yolo26_formal_001",
            "phase": "freeze",
            "gray_weights": "runs/train/yolo26_formal_001/gray_fusion/best.pt",
            "imgsz": 224,
            "limit_batches": 0,
            "smoke": False,
            "dataset_fingerprint": "fp_v4",
        }
        gray_config = {
            "run_id": "yolo26_formal_001",
            "imgsz": 224,
            "limit_batches": 0,
            "smoke": False,
            "dataset_fingerprint": "fp_v4",
        }
        return metadata, fusion_config, gray_config

    def _check(self, metadata, fusion_config, gray_config, **overrides):
        kwargs = {
            "metadata": metadata,
            "fusion_config": fusion_config,
            "gray_config": gray_config,
            "fusion_fingerprint": "fp_v4",
            "expected_fingerprint": "fp_v4",
            "gray_weights_path": Path(
                "runs/train/yolo26_formal_001/gray_fusion/best.pt"
            ),
            "class_names": V3_CLASS_NAMES,
            "imgsz": 224,
        }
        kwargs.update(overrides)
        return dpf.check_metadata(**kwargs)

    def test_valid_metadata_passes(self):
        metadata, fusion_config, gray_config = self._inputs()
        report = self._check(metadata, fusion_config, gray_config)
        assert report["architecture"] == "yolo26n-cls"

    def test_fingerprint_mismatch_rejected(self):
        metadata, fusion_config, gray_config = self._inputs()
        with pytest.raises(RuntimeError, match="fingerprint"):
            self._check(metadata, fusion_config, gray_config,
                        expected_fingerprint="other")
        fusion_config["dataset_fingerprint"] = "other"
        with pytest.raises(RuntimeError, match="fingerprint"):
            self._check(metadata, fusion_config, gray_config)
        fusion_config["dataset_fingerprint"] = "fp_v4"
        with pytest.raises(RuntimeError, match="fingerprint"):
            self._check(metadata, fusion_config, gray_config,
                        fusion_fingerprint="other")

    def test_class_order_and_imgsz_rejected(self):
        metadata, fusion_config, gray_config = self._inputs()
        with pytest.raises(RuntimeError, match="class"):
            self._check(metadata, fusion_config, gray_config,
                        class_names=["a", "b", "c", "d"])
        with pytest.raises(RuntimeError, match="imgsz"):
            self._check(metadata, fusion_config, gray_config, imgsz=320)

    def test_gray_source_mismatch_rejected(self):
        metadata, fusion_config, gray_config = self._inputs()
        with pytest.raises(RuntimeError, match="gray"):
            self._check(
                metadata, fusion_config, gray_config,
                gray_weights_path=Path("runs/train/other/gray_fusion/best.pt"),
            )

    def test_smoke_or_truncated_fusion_rejected(self):
        metadata, fusion_config, gray_config = self._inputs()
        fusion_config["smoke"] = True
        with pytest.raises(RuntimeError, match="smoke"):
            self._check(metadata, fusion_config, gray_config)


class TestOutputDir:
    def test_existing_output_dir_refused(self, tmp_path):
        target = tmp_path / "diagnostics_x"
        target.mkdir()
        with pytest.raises(FileExistsError, match="exist"):
            dpf.prepare_output_dir(target)

    def test_new_output_dir_created(self, tmp_path):
        target = tmp_path / "diagnostics_x"
        assert dpf.prepare_output_dir(target) == target
        assert target.is_dir()


class TestEvaluateIntegration:
    def _fixture(self, tmp_path):
        paths = []
        labels = []
        for index, class_id in enumerate([0, 1, 2, 3, 0]):
            paths.append(write_npz(tmp_path, f"s{index}", class_id, 0.1 + 0.1 * index))
            labels.append(class_id)
        model = make_stub_model()
        return model, paths, labels

    def test_all_samples_evaluated_once_and_model_untouched(self, tmp_path):
        model, paths, labels = self._fixture(tmp_path)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = dpf.evaluate(
            model, paths, labels,
            imgsz=8, device="cpu", batch=2, shuffle_seeds=(2026,),
        )
        assert result["samples_evaluated"] == len(paths)
        assert result["gray"]["confusion_matrix"]
        assert model.training is False
        for key, value in model.state_dict().items():
            assert torch.equal(before[key], value)
        # fallbacks must be exact for the real forward
        assert result["fallbacks"]["valid_ratio_zero"]["gate_failures"] == 0
        assert result["fallbacks"]["valid_ratio_zero"]["logits_failures"] == 0
        assert result["fallbacks"]["polar_invalid"]["gate_failures"] == 0
        assert result["fallbacks"]["polar_invalid"]["logits_failures"] == 0

    def test_shuffle_donor_mapping_and_reproducibility(self, tmp_path):
        model, paths, labels = self._fixture(tmp_path)
        first = dpf.evaluate(
            model, paths, labels,
            imgsz=8, device="cpu", batch=2, shuffle_seeds=(2026,),
        )
        second = dpf.evaluate(
            model, paths, labels,
            imgsz=8, device="cpu", batch=4, shuffle_seeds=(2026,),
        )
        # batch-size independent mapping
        assert first["shuffles"]["2026"]["permutation"] == second["shuffles"]["2026"]["permutation"]
        assert first["shuffles"]["2026"]["predictions"] == second["shuffles"]["2026"]["predictions"]
        permutation = first["shuffles"]["2026"]["permutation"]
        assert sorted(permutation) == list(range(len(paths)))
        assert all(permutation[i] != i for i in range(len(paths)))
        assert first["shuffles"]["2026"]["fixed_points"] == 0
        for record in first["records"]:
            assert record["shuffle_donors"]["2026"] == permutation[record["index"]]
