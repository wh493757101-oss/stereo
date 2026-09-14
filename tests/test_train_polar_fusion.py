"""Tests for scripts.train_polar_fusion (CLI gate, dry-run, phases) and
scripts.eval_polar_fusion argument handling. No training is executed."""

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

import scripts.train_polar_fusion as tpf
from core.fusion_dataset import QUALITY_VECTOR_LENGTH, build_quality_vector, save_fusion_sample


@pytest.fixture
def tmp_path() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="train_polar_fusion_test_"))
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
    import csv
    import io

    root = tmp_path / "fusion_v3"
    rows = []
    for i in range(2):
        rows.append(write_sample(root, "train", "metal", f"m{i}", 0, seed=i))
        rows.append(write_sample(root, "train", "plastic", f"p{i}", 1, seed=10 + i))
    rows.append(write_sample(root, "val", "metal", "v0", 0, seed=20))
    rows.append(write_sample(root, "val", "plastic", "v1", 1, seed=21))
    rows.append(write_sample(root, "test", "metal", "t0", 0, seed=30))
    rows.append(write_sample(root, "test", "plastic", "t1", 1, seed=31))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    (root / "dataset_manifest.csv").write_text(buffer.getvalue(), encoding="utf-8")
    return root


class TestCLI:
    def test_run_id_required(self):
        with pytest.raises(SystemExit):
            tpf.parse_args([])

    def test_default_base_is_yolo26n_cls(self):
        args = tpf.parse_args(["--run-id", "run_x"])
        assert args.base == "yolo26n-cls.pt"
        assert args.phase == "freeze"
        assert args.data == "datasets/underwater_cls_fusion_v3"
        assert args.dry_run is False

    def test_invalid_run_id_rejected_before_any_work(self, monkeypatch):
        def boom(_):
            raise AssertionError("dry_run must not execute with a bad run id")

        monkeypatch.setattr(tpf, "dry_run", boom)
        assert tpf.main(["--run-id", "../escape", "--dry-run"]) == 2

    def test_phase_choices(self):
        args = tpf.parse_args(["--run-id", "r", "--phase", "joint"])
        assert args.phase == "joint"
        with pytest.raises(SystemExit):
            tpf.parse_args(["--run-id", "r", "--phase", "everything"])


class TestDryRun:
    def test_dry_run_validates_dataset_and_structure(
        self, mini_dataset, monkeypatch, capsys
    ):
        called = []
        monkeypatch.setattr(tpf, "run_training", lambda args: called.append(args))
        exit_code = tpf.main(
            ["--run-id", "run_dry", "--dry-run", "--data", str(mini_dataset)]
        )

        assert exit_code == 0
        assert called == []  # dry-run never trains
        out = capsys.readouterr().out
        assert '"train": 4' in out
        assert '"val": 2' in out
        assert '"test": 2' in out
        assert '"forced_gate_verified": true' in out

    def test_dry_run_reports_missing_base_without_download(
        self, mini_dataset, capsys
    ):
        exit_code = tpf.main(
            [
                "--run-id", "run_dry",
                "--dry-run",
                "--data", str(mini_dataset),
                "--base", "nonexistent-base.pt",
            ]
        )
        # Missing base is reported, not fatal, and never downloaded.
        assert exit_code == 0
        err = capsys.readouterr().err
        assert "nonexistent-base.pt" in err
        assert "no automatic download" in err.lower()

    def test_dry_run_creates_no_run_outputs(self, mini_dataset):
        tpf.main(["--run-id", "run_dry", "--dry-run", "--data", str(mini_dataset)])
        assert not (ROOT / "runs" / "train" / "run_dry").exists()

    def test_dry_run_fails_on_missing_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            tpf.validate_dataset(tmp_path / "empty")

    def test_dry_run_fails_when_val_empty(self, tmp_path):
        import csv
        import io

        root = tmp_path / "no_val"
        rows = [
            write_sample(root, "train", "metal", "m0", 0, seed=0),
            write_sample(root, "train", "plastic", "p0", 1, seed=1),
        ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        (root / "dataset_manifest.csv").write_text(buffer.getvalue(), encoding="utf-8")
        # Either the missing val split or the explicit non-empty gate fires.
        with pytest.raises(ValueError, match="val"):
            tpf.validate_dataset(root)


class TestStructureCheck:
    def test_structure_check_passes(self):
        report = tpf.structure_check(num_classes=4, imgsz=32)
        assert report["num_classes"] == 4
        assert report["forced_gate_verified"] is True


class TestTrainingGate:
    def test_training_refused_when_base_missing(self, mini_dataset, monkeypatch):
        monkeypatch.setattr(tpf, "validate_dataset", lambda root: None)
        with pytest.raises(FileNotFoundError, match="no automatic download"):
            tpf.run_training(
                tpf.parse_args(
                    [
                        "--run-id", "run_x",
                        "--data", str(mini_dataset),
                        "--base", "definitely-missing.pt",
                    ]
                )
            )

    def test_run_training_validates_dataset_first(self, mini_dataset, monkeypatch):
        def refuse(_):
            raise FileNotFoundError("base checkpoint missing; no automatic download")

        monkeypatch.setattr(tpf, "load_gray_backbone", refuse)
        with pytest.raises(FileNotFoundError):
            tpf.run_training(
                tpf.parse_args(["--run-id", "run_x", "--data", str(mini_dataset)])
            )


class TestEvalScript:
    def test_missing_checkpoint_exits_nonzero(self):
        import scripts.eval_polar_fusion as epf

        assert epf.main(["--checkpoint", "does/not/exist.pt"]) == 2

    def test_parse_args_defaults(self):
        import scripts.eval_polar_fusion as epf

        args = epf.parse_args(["--checkpoint", "x.pt"])
        assert args.split == "test"
        assert args.device == "cpu"
        assert args.data == "datasets/underwater_cls_fusion_v3"
