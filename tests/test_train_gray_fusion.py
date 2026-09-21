"""Tests for scripts.train_gray_fusion (gray-only YOLO26 entry).

No real training and no network: the Ultralytics loader is faked. The real
YOLO26 GPU roundtrip is covered by scripts/verify_yolo26_fusion.py.
"""

import csv
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_gray_fusion as tgf
from core.fusion_dataset import build_quality_vector, save_fusion_sample
from models.polar_fusion import FusionClsDataset, prepare_gray_backbone, read_manifest_split

V3_CLASS_NAMES = [
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
]
GRAY_WEIGHTS_NAMES = {
    0: "metal_submarine",
    1: "plastic_fish",
    2: "plastic_submarine",
    3: "real_fish",
}


@pytest.fixture
def tmp_path() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="train_gray_fusion_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def write_sample(root: Path, split: str, class_name: str, name: str, class_id: int, seed: int):
    rng = np.random.default_rng(seed)
    shape = (20, 16)
    signed = rng.uniform(-1, 1, size=shape).astype(np.float32)
    save_fusion_sample(
        root / split / class_name / f"{name}.npz",
        gray=rng.integers(0, 256, size=shape, dtype=np.uint8),
        signed_q=signed,
        abs_q=np.abs(signed),
        valid=(rng.uniform(size=shape) > 0.5).astype(np.uint8),
        quality=build_quality_vector(0.5, 0.6, 0.7, 0.2),
        class_id=class_id,
    )
    return {
        "sample_name": name,
        "split": split,
        "group_name": f"group_{class_name}",
        "class_id": str(class_id),
        "class_name": class_name,
        "source_frame": name,
        "object_index": "0",
        "stereo_valid": "true",
        "stereo_reason": "ok",
        "stereo_valid_ratio": "0.900000",
        "disparity": "12.0000",
        "polar_valid_ratio": "0.500000",
        "quality_valid_ratio": "0.500000",
        "quality_in_bounds_ratio": "0.600000",
        "quality_brightness_valid_ratio": "0.700000",
        "quality_mean_abs_q": "0.200000",
        "npz_path": f"{split}/{class_name}/{name}.npz",
    }


@pytest.fixture
def mini_dataset(tmp_path):
    """4-class mini dataset (V3 order) with train + val + test splits."""
    root = tmp_path / "fusion_v3names"
    rows = []
    for class_id, name in enumerate(V3_CLASS_NAMES):
        rows.append(
            write_sample(root, "train", name, f"tr{class_id}", class_id, seed=class_id)
        )
    for class_id, name in enumerate(V3_CLASS_NAMES[:2]):
        rows.append(
            write_sample(root, "val", name, f"va{class_id}", class_id, seed=40 + class_id)
        )
    for class_id, name in enumerate(V3_CLASS_NAMES[:2]):
        rows.append(
            write_sample(root, "test", name, f"te{class_id}", class_id, seed=60 + class_id)
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    (root / "dataset_manifest.csv").write_text(buffer.getvalue(), encoding="utf-8")
    return root


class FakeClassifyHead(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)


class FakeUltralyticsModel(torch.nn.Module):
    """Train mode returns raw logits; eval mode returns (probs, logits).

    Carries ``yaml``/``names``/``task`` like a real Ultralytics model so the
    architecture and task admission is exercised.
    """

    def __init__(self, out_features, names, yaml_file, task="classify"):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Identity(), FakeClassifyHead(1280, out_features)
        )
        self.names = names
        self.yaml = {"yaml_file": yaml_file}
        self.task = task

    def forward(self, gray):
        features = torch.zeros(gray.shape[0], 1280, device=gray.device)
        logits = self.model[-1].linear(features)
        if self.training:
            return logits
        return (logits.softmax(-1), logits)


class FakeYOLO:
    last_weights: str | None = None

    def __init__(self, weights):
        FakeYOLO.last_weights = weights
        path = Path(str(weights))
        text = str(weights)
        payload = None
        if path.is_file():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                payload = None  # plain placeholder file (fake base)
        if isinstance(payload, dict) and "model" in payload:
            # Roundtrip: load what a previous save wrote.
            self.model = payload["model"]
        elif "legacy" in text:
            self.model = FakeUltralyticsModel(
                4, dict(GRAY_WEIGHTS_NAMES), "yolov8n-cls.yaml"
            )
        elif "detect" in text:
            self.model = FakeUltralyticsModel(
                1000, {i: f"imagenet_{i}" for i in range(1000)}, "yolo26n.yaml",
                task="detect",
            )
        else:
            self.model = FakeUltralyticsModel(
                1000, {i: f"imagenet_{i}" for i in range(1000)}, "yolo26n-cls.yaml"
            )


@pytest.fixture
def fake_ultralytics(monkeypatch):
    import types

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    FakeYOLO.last_weights = None
    return module


def allow_training(monkeypatch):
    """Bypass the dataset gate and force CPU for unit-level training runs."""
    monkeypatch.setattr(tgf, "validate_dataset", lambda root: {})
    monkeypatch.setattr(tgf, "resolve_device", lambda *a, **kw: "cpu")


def gray_argv(data, run_id, base, extra=()):
    return [
        "--run-id", run_id,
        "--data", str(data),
        "--base", str(base),
        "--imgsz", "16",
        "--batch", "2",
        "--epochs", "1",
        *extra,
    ]


def make_base_file(tmp_path: Path, name: str = "yolo26n-cls.pt") -> Path:
    base = tmp_path / name
    base.write_bytes(b"fake")
    return base


class TestGrayDatasetParity:
    """The gray-only view must produce exactly the Fusion gray tensor and
    label for the same sample (same resize/normalization, no augmentation)."""

    def test_gray_view_matches_fusion_inputs(self, mini_dataset):
        paths, _, _ = read_manifest_split(mini_dataset, "train")
        fusion_ds = FusionClsDataset(paths, imgsz=16)
        gray_ds = tgf.GrayFusionDataset(paths, imgsz=16)
        assert len(gray_ds) == len(fusion_ds)
        for index in range(len(paths)):
            gray_f, _, _, label_f = fusion_ds[index]
            gray_g, label_g = gray_ds[index]
            assert torch.equal(gray_g, gray_f)
            assert label_g == label_f

    def test_no_augmentation_is_deterministic(self, mini_dataset):
        paths, _, _ = read_manifest_split(mini_dataset, "train")
        gray_ds = tgf.GrayFusionDataset(paths, imgsz=16)
        gray_a, label_a = gray_ds[0]
        gray_b, label_b = gray_ds[0]
        assert torch.equal(gray_a, gray_b)
        assert label_a == label_b


class TestBaseAdmission:
    def test_missing_base_refused_before_outputs(self, tmp_path, mini_dataset, monkeypatch):
        allow_training(monkeypatch)
        run_id = "test_gray_missing_base"
        with pytest.raises(FileNotFoundError, match="no automatic download"):
            tgf.run_training(
                tgf.parse_args(gray_argv(mini_dataset, run_id, tmp_path / "missing.pt"))
            )
        assert not (tgf.PROJECT_ROOT / "runs" / "train" / run_id).exists()

    def test_legacy_base_refused(self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch):
        allow_training(monkeypatch)
        base = make_base_file(tmp_path, "legacy-cls.pt")
        with pytest.raises(ValueError, match="not YOLO26"):
            tgf.run_training(tgf.parse_args(gray_argv(mini_dataset, "test_gray_legacy", base)))

    def test_detection_base_refused(self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch):
        """yolo26n.pt is a detection model: the architecture name alone is
        not enough, the task must be classification."""
        allow_training(monkeypatch)
        base = make_base_file(tmp_path, "detect.pt")
        with pytest.raises(ValueError, match="task"):
            tgf.run_training(tgf.parse_args(gray_argv(mini_dataset, "test_gray_detect", base)))


class TestSelection:
    def test_selection_key_macro_f1_primary(self):
        better = tgf._selection_key({"macro_f1": 0.8, "accuracy": 0.7})
        worse = tgf._selection_key({"macro_f1": 0.75, "accuracy": 0.9})
        assert better > worse

    def test_best_follows_macro_f1_not_accuracy(
        self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch
    ):
        """Scripted val metrics: epoch 3 has the highest accuracy but a lower
        macro-F1 than epoch 2 and must not overwrite best.pt."""
        allow_training(monkeypatch)
        scripted = [
            ({"macro_f1": 0.5, "accuracy": 0.9, "per_class_recall": {}}, 1, 2),
            ({"macro_f1": 0.7, "accuracy": 0.6, "per_class_recall": {}}, 1, 2),
            ({"macro_f1": 0.6, "accuracy": 0.95, "per_class_recall": {}}, 1, 2),
        ]
        calls = {"i": 0}

        def fake_evaluate(model, loader, device, num_classes, limit_batches=0):
            metrics, batches, samples = scripted[calls["i"]]
            calls["i"] += 1
            return dict(metrics), batches, samples

        monkeypatch.setattr(tgf, "_evaluate_gray", fake_evaluate)
        base = make_base_file(tmp_path)
        saves = []
        run_id = "test_gray_selection"
        try:
            run_dir = tgf.run_training(
                tgf.parse_args(gray_argv(mini_dataset, run_id, base, ("--epochs", "3"))),
                save_hook=lambda model, path, info: saves.append((info["phase"], info["epoch"])),
            )
            assert saves == [("best", 1), ("best", 2), ("last", 3)]
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            assert [entry["val_macro_f1"] for entry in metrics] == [0.5, 0.7, 0.6]
            assert [entry["val_acc"] for entry in metrics] == [0.9, 0.6, 0.95]
            assert all(entry["val_batches"] == 1 for entry in metrics)
        finally:
            shutil.rmtree(tgf.PROJECT_ROOT / "runs" / "train" / run_id, ignore_errors=True)


class TestRunGate:
    def test_arg_validation_rejects_nonpositive(self):
        with pytest.raises(SystemExit):
            tgf.parse_args(["--run-id", "r", "--epochs", "0"])
        with pytest.raises(SystemExit):
            tgf.parse_args(["--run-id", "r", "--batch", "0"])
        with pytest.raises(SystemExit):
            tgf.parse_args(["--run-id", "r", "--imgsz", "0"])
        with pytest.raises(SystemExit):
            tgf.parse_args(["--run-id", "r", "--limit-batches", "-1"])

    def test_existing_run_dir_refused(
        self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch
    ):
        allow_training(monkeypatch)
        base = make_base_file(tmp_path)
        run_id = "test_gray_existing_dir"
        run_dir = tgf.PROJECT_ROOT / "runs" / "train" / run_id / "gray_fusion"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with pytest.raises(FileExistsError, match="new --run-id"):
                tgf.run_training(tgf.parse_args(gray_argv(mini_dataset, run_id, base)))
        finally:
            shutil.rmtree(tgf.PROJECT_ROOT / "runs" / "train" / run_id, ignore_errors=True)

    def test_training_requires_passed_audit(self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch):
        """Unpatched dataset gate: a dataset without an audit report is
        refused before any output is created."""
        monkeypatch.setattr(tgf, "resolve_device", lambda *a, **kw: "cpu")
        base = make_base_file(tmp_path)
        run_id = "test_gray_no_audit"
        with pytest.raises(FileNotFoundError, match="audit"):
            tgf.run_training(tgf.parse_args(gray_argv(mini_dataset, run_id, base)))
        assert not (tgf.PROJECT_ROOT / "runs" / "train" / run_id).exists()

    def test_corrupt_sample_refused(self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch):
        monkeypatch.setattr(tgf, "resolve_device", lambda *a, **kw: "cpu")
        victim = mini_dataset / "train" / V3_CLASS_NAMES[0] / "tr0.npz"
        victim.write_bytes(b"corrupt")
        base = make_base_file(tmp_path)
        with pytest.raises(ValueError):
            tgf.run_training(
                tgf.parse_args(gray_argv(mini_dataset, "test_gray_corrupt", base))
            )

    def test_only_train_and_val_splits_used(
        self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch
    ):
        """Training reads train; model selection reads val; test is never
        touched by the training entry."""
        allow_training(monkeypatch)
        requested = []
        original = tgf.read_manifest_split

        def recording_split(root, split):
            requested.append(split)
            return original(root, split)

        monkeypatch.setattr(tgf, "read_manifest_split", recording_split)
        base = make_base_file(tmp_path)
        run_id = "test_gray_splits"
        try:
            tgf.run_training(tgf.parse_args(gray_argv(mini_dataset, run_id, base)))
            assert requested == ["train", "val"]
        finally:
            shutil.rmtree(tgf.PROJECT_ROOT / "runs" / "train" / run_id, ignore_errors=True)


class TestCheckpointCompat:
    def _train_once(self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch, run_id):
        allow_training(monkeypatch)
        base = make_base_file(tmp_path)
        run_dir = tgf.run_training(tgf.parse_args(gray_argv(mini_dataset, run_id, base)))
        return base, run_dir

    def test_outputs_and_config(self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch):
        run_id = "test_gray_outputs"
        try:
            _, run_dir = self._train_once(
                tmp_path, mini_dataset, fake_ultralytics, monkeypatch, run_id
            )
            assert (run_dir / "best.pt").is_file()
            assert (run_dir / "last.pt").is_file()
            config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
            assert config["class_names"] == V3_CLASS_NAMES
            assert config["imgsz"] == 16
            assert config["limit_batches"] == 0
            assert config["smoke"] is False
            assert config["architecture"] == "yolo26n-cls"
            assert config["optimizer"] == "AdamW"
            assert config["train_samples"] == 4
            assert config["val_samples"] == 2
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            assert len(metrics) == 1
            assert set(metrics[0]) >= {
                "epoch", "train_loss", "train_batches", "val_batches",
                "val_macro_f1", "val_acc", "val_per_class_recall",
            }
        finally:
            shutil.rmtree(tgf.PROJECT_ROOT / "runs" / "train" / run_id, ignore_errors=True)

    def test_smoke_flag_recorded(self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch):
        allow_training(monkeypatch)
        base = make_base_file(tmp_path)
        run_id = "test_gray_smoke_flag"
        try:
            run_dir = tgf.run_training(
                tgf.parse_args(
                    gray_argv(mini_dataset, run_id, base, ("--limit-batches", "1"))
                )
            )
            config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
            assert config["limit_batches"] == 1
            assert config["smoke"] is True
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            assert metrics[0]["train_batches"] == 1
            assert metrics[0]["val_batches"] == 1
        finally:
            shutil.rmtree(tgf.PROJECT_ROOT / "runs" / "train" / run_id, ignore_errors=True)

    def test_saved_checkpoint_loads_via_prepare_gray_backbone(
        self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch
    ):
        """The checkpoint must be loadable by the existing fusion gray
        branch loader with a 4-class head in dataset class order."""
        run_id = "test_gray_reload"
        try:
            _, run_dir = self._train_once(
                tmp_path, mini_dataset, fake_ultralytics, monkeypatch, run_id
            )
            best = run_dir / "best.pt"
            backbone, info = prepare_gray_backbone(
                make_base_file(tmp_path), str(best), V3_CLASS_NAMES, "cpu"
            )
            assert info["head_replaced"] is False
            assert info["gray_class_names"] == V3_CLASS_NAMES
            assert backbone.module.model[-1].linear.out_features == 4
            backbone.eval()
            out = backbone(torch.zeros(2, 3, 16, 16))
            assert out.shape == (2, 4)
        finally:
            shutil.rmtree(tgf.PROJECT_ROOT / "runs" / "train" / run_id, ignore_errors=True)

    def test_checkpoint_carries_task_and_architecture(
        self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch
    ):
        run_id = "test_gray_meta"
        try:
            _, run_dir = self._train_once(
                tmp_path, mini_dataset, fake_ultralytics, monkeypatch, run_id
            )
            module = FakeYOLO(str(run_dir / "best.pt")).model
            assert module.task == "classify"
            assert module.yaml["yaml_file"] == "yolo26n-cls.yaml"
            assert [module.names[i] for i in sorted(module.names)] == V3_CLASS_NAMES
        finally:
            shutil.rmtree(tgf.PROJECT_ROOT / "runs" / "train" / run_id, ignore_errors=True)
