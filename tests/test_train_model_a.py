"""Tests for scripts.train_model_a (YOLO26 binary segmentation entry).

No real training: the Ultralytics loader/trainer is faked.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_model_a as tma


@pytest.fixture
def tmp_path() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="train_model_a_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class FakeSegModule(torch.nn.Module):
    def __init__(self, task, yaml_file, names):
        super().__init__()
        self.task = task
        self.yaml = {"yaml_file": yaml_file}
        self.names = names

    def forward(self, x):
        return x


class FakeTrainer:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)


class FakeYOLO:
    """Loader/trainer stand-in: ``train`` writes best/last into the run dir."""

    last_kwargs: dict | None = None
    last_weights: str | None = None

    def __init__(self, weights):
        FakeYOLO.last_weights = weights
        text = Path(str(weights)).name
        if "legacy" in text:
            self.model = FakeSegModule("segment", "yolov8n-seg.yaml", {0: "target"})
        elif "seg" in text:
            self.model = FakeSegModule("segment", "yolo26n-seg.yaml", {i: f"c{i}" for i in range(80)})
        else:
            # yolo26n.pt: a detection checkpoint, never a Model A base.
            self.model = FakeSegModule("detect", "yolo26n.yaml", {i: f"c{i}" for i in range(80)})

    def train(self, **kwargs):
        FakeYOLO.last_kwargs = kwargs
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"best")
        (save_dir / "weights" / "last.pt").write_bytes(b"last")
        self.trainer = FakeTrainer(save_dir)


@pytest.fixture
def fake_ultralytics(monkeypatch):
    FakeYOLO.last_kwargs = None
    FakeYOLO.last_weights = None
    monkeypatch.setattr(tma, "_load_yolo_model", FakeYOLO)
    return FakeYOLO


def make_base_file(tmp_path: Path, name: str = "yolo26n-seg.pt") -> Path:
    base = tmp_path / name
    base.write_bytes(b"fake")
    return base


def write_binary_dataset(
    tmp_path: Path,
    *,
    names_yaml: str = "names:\n  0: target\n",
    extra_yaml: str = "",
    train_label_text: str = "0 0.1 0.1 0.2 0.1 0.2 0.2\n",
    include_train_background: bool = True,
    train_positive: bool = True,
) -> Path:
    """Minimal binary seg dataset: 2 train images (one background), 1 val."""
    root = tmp_path / "seg_binary"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "images" / "train" / "a.png").write_bytes(b"x")
    (root / "labels" / "train" / "a.txt").write_text(
        train_label_text if train_positive else "", encoding="utf-8"
    )
    if include_train_background:
        (root / "images" / "train" / "b.png").write_bytes(b"x")  # background
    (root / "images" / "val" / "c.png").write_bytes(b"x")
    (root / "labels" / "val" / "c.txt").write_text(
        "0 0.3 0.3 0.4 0.3 0.4 0.4\n", encoding="utf-8"
    )
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"{names_yaml}{extra_yaml}",
        encoding="utf-8",
    )
    return root / "data.yaml"


def write_data_yaml(tmp_path: Path) -> Path:
    return write_binary_dataset(tmp_path)


def allow_device(monkeypatch):
    monkeypatch.setattr(tma, "resolve_device", lambda device, *a, **kw: device)


def make_argv(tmp_path, run_id="run_x", extra=()):
    return [
        "--run-id", run_id,
        "--base", str(make_base_file(tmp_path)),
        "--data", str(write_data_yaml(tmp_path)),
        "--device", "cpu",
        "--epochs", "2",
        "--batch", "2",
        "--imgsz", "64",
        *extra,
    ]


class TestAdmission:
    def test_missing_base_refused_before_outputs(self, tmp_path, fake_ultralytics, monkeypatch):
        allow_device(monkeypatch)
        monkeypatch.setattr(tma, "PROJECT_ROOT", tmp_path)
        run_id = "run_missing_base"
        with pytest.raises(FileNotFoundError, match="no automatic download"):
            tma.run_training(
                tma.parse_args(
                    ["--run-id", run_id, "--base", str(tmp_path / "missing.pt"),
                     "--data", str(write_data_yaml(tmp_path)), "--device", "cpu"]
                )
            )
        assert not (tmp_path / "runs" / "train" / run_id).exists()

    def test_detection_base_refused(self, tmp_path, fake_ultralytics, monkeypatch):
        allow_device(monkeypatch)
        monkeypatch.setattr(tma, "PROJECT_ROOT", tmp_path)
        base = make_base_file(tmp_path, "yolo26n.pt")
        with pytest.raises(ValueError, match="task"):
            tma.run_training(
                tma.parse_args(
                    ["--run-id", "run_detect", "--base", str(base),
                     "--data", str(write_data_yaml(tmp_path)), "--device", "cpu"]
                )
            )

    def test_legacy_base_refused(self, tmp_path, fake_ultralytics, monkeypatch):
        allow_device(monkeypatch)
        monkeypatch.setattr(tma, "PROJECT_ROOT", tmp_path)
        base = make_base_file(tmp_path, "legacy-seg.pt")
        with pytest.raises(ValueError, match="YOLO26"):
            tma.run_training(
                tma.parse_args(
                    ["--run-id", "run_legacy", "--base", str(base),
                     "--data", str(write_data_yaml(tmp_path)), "--device", "cpu"]
                )
            )

    def test_missing_data_yaml_refused(self, tmp_path, fake_ultralytics, monkeypatch):
        allow_device(monkeypatch)
        monkeypatch.setattr(tma, "PROJECT_ROOT", tmp_path)
        with pytest.raises(FileNotFoundError, match="data"):
            tma.run_training(
                tma.parse_args(
                    ["--run-id", "run_nodata", "--base", str(make_base_file(tmp_path)),
                     "--data", str(tmp_path / "missing.yaml"), "--device", "cpu"]
                )
            )

    def test_existing_output_dir_refused(self, tmp_path, fake_ultralytics, monkeypatch):
        allow_device(monkeypatch)
        monkeypatch.setattr(tma, "PROJECT_ROOT", tmp_path)
        run_id = "run_exists"
        (tmp_path / "runs" / "train" / run_id / "model_a").mkdir(parents=True)
        with pytest.raises(FileExistsError, match="new --run-id"):
            tma.run_training(tma.parse_args(make_argv(tmp_path, run_id)))

    def test_arg_validation(self):
        with pytest.raises(SystemExit):
            tma.parse_args(["--run-id", "r", "--epochs", "0"])
        with pytest.raises(SystemExit):
            tma.parse_args(["--run-id", "r", "--batch", "0"])
        with pytest.raises(SystemExit):
            tma.parse_args(["--run-id", "r", "--imgsz", "0"])
        with pytest.raises(SystemExit):
            tma.parse_args(["--run-id", "r", "--fraction", "0"])
        with pytest.raises(SystemExit):
            tma.parse_args(["--run-id", "r", "--fraction", "1.5"])


class TestBinaryDatasetValidation:
    """Single-class ``0: target`` enforcement, shared by the entry and the
    pipeline (issue 1)."""

    def test_four_class_yaml_rejected(self, tmp_path):
        data = write_binary_dataset(
            tmp_path,
            names_yaml="names:\n  0: metal\n  1: plastic\n  2: fish\n  3: real\n",
        )
        with pytest.raises(ValueError, match="target"):
            tma._validate_binary_dataset(data)

    def test_names_list_form_accepted(self, tmp_path):
        data = write_binary_dataset(tmp_path, names_yaml="names: [target]\n")
        info = tma._validate_binary_dataset(data)
        assert info["names"] == ["target"]
        assert info["nc"] == 1
        assert info["train_polygons"] >= 1
        assert info["val_polygons"] >= 1
        # Background images (no label file) are a legal convention.
        assert info["backgrounds"]["train"] >= 1

    def test_dict_form_with_nonzero_key_rejected(self, tmp_path):
        data = write_binary_dataset(tmp_path, names_yaml="names:\n  1: target\n")
        with pytest.raises(ValueError, match="target"):
            tma._validate_binary_dataset(data)

    def test_nc_mismatch_rejected(self, tmp_path):
        data = write_binary_dataset(tmp_path, extra_yaml="nc: 4\n")
        with pytest.raises(ValueError, match="nc"):
            tma._validate_binary_dataset(data)
        data_ok = write_binary_dataset(tmp_path / "ok", extra_yaml="nc: 1\n")
        assert tma._validate_binary_dataset(data_ok)["nc"] == 1

    def test_label_class_one_rejected(self, tmp_path):
        data = write_binary_dataset(
            tmp_path, train_label_text="1 0.1 0.1 0.2 0.1 0.2 0.2\n"
        )
        with pytest.raises(ValueError, match="class"):
            tma._validate_binary_dataset(data)

    def test_noninteger_class_rejected(self, tmp_path):
        data = write_binary_dataset(
            tmp_path, train_label_text="0.5 0.1 0.1 0.2 0.1 0.2 0.2\n"
        )
        with pytest.raises(ValueError, match="class"):
            tma._validate_binary_dataset(data)

    def test_detection_box_label_rejected(self, tmp_path):
        data = write_binary_dataset(
            tmp_path, train_label_text="0 0.5 0.5 0.2 0.3\n"
        )
        with pytest.raises(ValueError, match="polygon"):
            tma._validate_binary_dataset(data)

    def test_nonfinite_label_rejected(self, tmp_path):
        data = write_binary_dataset(
            tmp_path, train_label_text="0 nan 0.1 0.2 0.1 0.2 0.2\n"
        )
        with pytest.raises(ValueError, match="finite"):
            tma._validate_binary_dataset(data)

    def test_all_background_dataset_rejected(self, tmp_path):
        data = write_binary_dataset(tmp_path, train_positive=False)
        with pytest.raises(ValueError, match="annotation"):
            tma._validate_binary_dataset(data)

    def test_entry_and_pipeline_share_validation(
        self, tmp_path, fake_ultralytics, monkeypatch
    ):
        allow_device(monkeypatch)
        monkeypatch.setattr(tma, "PROJECT_ROOT", tmp_path)
        data = write_binary_dataset(
            tmp_path, names_yaml="names:\n  0: metal\n  1: plastic\n"
        )
        run_id = "run_four_class"
        with pytest.raises(ValueError, match="target"):
            tma.run_training(
                tma.parse_args(
                    ["--run-id", run_id, "--base", str(make_base_file(tmp_path)),
                     "--data", str(data), "--device", "cpu"]
                )
            )
        assert not (tmp_path / "runs" / "train" / run_id).exists()
        import scripts.train_pipeline as vp

        with pytest.raises(ValueError, match="target"):
            vp._check_seg_data(data)


class TestTrainingRun:
    def test_training_kwargs_and_config(self, tmp_path, fake_ultralytics, monkeypatch):
        allow_device(monkeypatch)
        monkeypatch.setattr(tma, "PROJECT_ROOT", tmp_path)
        run_id = "run_ok"
        run_dir = tma.run_training(tma.parse_args(make_argv(tmp_path, run_id)))
        kwargs = fake_ultralytics.last_kwargs
        assert kwargs["project"] == str(tmp_path / "runs" / "train" / run_id)
        assert kwargs["name"] == "model_a"
        assert kwargs["exist_ok"] is False
        assert kwargs["seed"] == 2026
        assert kwargs["deterministic"] is True
        assert kwargs["epochs"] == 2
        assert kwargs["batch"] == 2
        assert kwargs["imgsz"] == 64
        assert "fraction" not in kwargs
        assert (run_dir / "weights" / "best.pt").is_file()
        config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        assert config["entry"] == "train_model_a"
        assert config["architecture"] == "yolo26n-seg"
        assert config["task"] == "segment"
        assert config["smoke"] is False
        assert config["fraction"] is None
        assert config["data_names"] == ["target"]
        assert config["base_sha256"]

    def test_smoke_fraction_recorded(self, tmp_path, fake_ultralytics, monkeypatch):
        allow_device(monkeypatch)
        monkeypatch.setattr(tma, "PROJECT_ROOT", tmp_path)
        run_dir = tma.run_training(
            tma.parse_args(make_argv(tmp_path, "run_smoke", ("--fraction", "0.01")))
        )
        kwargs = fake_ultralytics.last_kwargs
        assert kwargs["fraction"] == 0.01
        config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        assert config["smoke"] is True
        assert config["fraction"] == 0.01
