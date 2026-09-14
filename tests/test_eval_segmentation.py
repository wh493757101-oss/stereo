"""Tests for scripts.eval_segmentation: metric extraction, input validation,
report building/writing and lazy Ultralytics use.

No real checkpoints, datasets or Ultralytics models are loaded.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.eval_segmentation as ev


CLASS_NAMES = ["metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"]


def make_metrics(nc: int = 4, present_classes: int | None = None,
                 names: dict | None = None) -> SimpleNamespace:
    """Fake SegmentMetrics-like object (box + seg/mask Metric components)."""
    present = nc if present_classes is None else present_classes
    if names is None:
        names = {i: CLASS_NAMES[i] for i in range(nc)}
    box_ap = np.tile(np.linspace(0.40, 0.80, 10), (nc, 1))
    seg_ap = np.tile(np.linspace(0.30, 0.70, 10), (nc, 1))
    box = SimpleNamespace(all_ap=box_ap, ap_class_index=list(range(present)))
    seg = SimpleNamespace(all_ap=seg_ap, ap_class_index=list(range(present)))
    return SimpleNamespace(
        names=names,
        box=box,
        seg=seg,
        results_dict={
            "metrics/precision(B)": 0.9, "metrics/recall(B)": 0.8,
            "metrics/mAP50(B)": 0.7, "metrics/mAP50-95(B)": 0.6,
            "metrics/precision(M)": 0.75, "metrics/recall(M)": 0.65,
            "metrics/mAP50(M)": 0.55, "metrics/mAP50-95(M)": 0.45,
            "fitness": 0.6,
        },
        nt_per_class=np.arange(10, 10 + nc),
        nt_per_image=np.arange(1, nc + 1),
    )


class FakeYOLO:
    def __init__(self, metrics):
        self.metrics = metrics
        self.val_kwargs = None

    def val(self, **kwargs):
        self.val_kwargs = kwargs
        return self.metrics


def _write_dataset(tmp_path: Path, n_images: int = 2) -> tuple[Path, Path]:
    root = tmp_path / "ds"
    img_dir = root / "images" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        cv2.imwrite(str(img_dir / f"{i}.png"), np.zeros((8, 8), np.uint8))
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root.as_posix()}\ntest: images/test\nnames:\n  0: target\n",
        encoding="utf-8",
    )
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights")
    return ckpt, data_yaml


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_extract_overall_box_and_mask(self):
        overall = ev.extract_overall(make_metrics().results_dict)
        assert overall["box"] == {
            "precision": 0.9, "recall": 0.8, "map50": 0.7, "map50_95": 0.6,
        }
        assert overall["mask"] == {
            "precision": 0.75, "recall": 0.65, "map50": 0.55, "map50_95": 0.45,
        }

    def test_extract_overall_omits_missing_components(self):
        overall = ev.extract_overall({"metrics/precision(B)": 0.5})
        assert "box" in overall
        assert "mask" not in overall

    def test_extract_per_class_box_and_mask(self):
        per_class = ev.extract_per_class(make_metrics(), CLASS_NAMES)
        assert per_class[CLASS_NAMES[0]]["box"] == {"map50": 0.4, "map50_95": 0.6}
        assert per_class[CLASS_NAMES[0]]["mask"]["map50"] == pytest.approx(0.3)
        assert per_class[CLASS_NAMES[0]]["mask"]["map50_95"] == pytest.approx(0.5)

    def test_extract_per_class_skips_classes_without_ap(self):
        per_class = ev.extract_per_class(make_metrics(present_classes=3), CLASS_NAMES)
        assert CLASS_NAMES[3] not in per_class
        assert len(per_class) == 3

    def test_extract_per_class_supports_legacy_mask_attribute(self):
        metrics = make_metrics()
        legacy = SimpleNamespace(box=metrics.box, mask=metrics.seg)
        per_class = ev.extract_per_class(legacy, CLASS_NAMES)
        assert per_class[CLASS_NAMES[0]]["mask"]["map50"] == pytest.approx(0.3)
        assert per_class[CLASS_NAMES[0]]["mask"]["map50_95"] == pytest.approx(0.5)

    def test_extract_per_class_empty_without_components(self):
        assert ev.extract_per_class(SimpleNamespace(), CLASS_NAMES) == {}

    def test_extract_counts_instances_per_class(self):
        counts = ev.extract_counts(make_metrics(), CLASS_NAMES)
        assert counts["instances_per_class"][CLASS_NAMES[0]] == 10
        assert counts["instances_per_class"][CLASS_NAMES[3]] == 13

    def test_extract_counts_omitted_when_not_exposed(self):
        assert ev.extract_counts(SimpleNamespace(), CLASS_NAMES) == {}


# ---------------------------------------------------------------------------
# Input validation and data.yaml parsing
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_checkpoint_raises_with_path(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names:\n  0: target\n", encoding="utf-8")
        task = ev.EvalTask("model_a", tmp_path / "missing.pt", data_yaml)
        with pytest.raises(FileNotFoundError, match="missing.pt"):
            ev.validate_task(task)

    def test_missing_data_yaml_raises_with_path(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"x")
        task = ev.EvalTask("model_a", ckpt, tmp_path / "missing.yaml")
        with pytest.raises(FileNotFoundError, match="missing.yaml"):
            ev.validate_task(task)


class TestDataYaml:
    def test_load_class_names_from_mapping(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text("names:\n  0: a\n  1: b\n", encoding="utf-8")
        assert ev.load_class_names(path) == ["a", "b"]

    def test_load_class_names_from_list(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text("names: [a, b]\n", encoding="utf-8")
        assert ev.load_class_names(path) == ["a", "b"]

    def test_count_split_images_ignores_non_images(self, tmp_path):
        root = tmp_path / "ds"
        (root / "images" / "test").mkdir(parents=True)
        for name in ("a.png", "b.png", "notes.txt"):
            (root / "images" / "test" / name).write_bytes(b"x")
        path = root / "data.yaml"
        path.write_text(
            f"path: {root.as_posix()}\ntest: images/test\nnames:\n  0: target\n",
            encoding="utf-8",
        )
        assert ev.count_split_images(path, "test") == 2

    def test_count_split_images_missing_dir_returns_none(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text("path: .\ntest: images/test\nnames:\n  0: target\n",
                        encoding="utf-8")
        assert ev.count_split_images(path, "test") is None


# ---------------------------------------------------------------------------
# Task evaluation (fake YOLO factory; no Ultralytics import)
# ---------------------------------------------------------------------------

class TestEvaluateTask:
    def test_runs_val_and_builds_section(self, tmp_path):
        ckpt, data_yaml = _write_dataset(tmp_path)
        metrics = make_metrics(nc=1, names={0: "target"})
        fake = FakeYOLO(metrics)
        task = ev.EvalTask("model_a", ckpt, data_yaml)
        section = ev.evaluate_task(
            task, split="test", device="cpu", yolo_factory=lambda p: fake
        )
        assert fake.val_kwargs["data"] == str(data_yaml)
        assert fake.val_kwargs["split"] == "test"
        assert fake.val_kwargs["device"] == "cpu"
        assert fake.val_kwargs["imgsz"] == 640
        assert fake.val_kwargs["batch"] == 8
        assert fake.val_kwargs["workers"] == 0
        assert fake.val_kwargs["plots"] is False
        assert fake.val_kwargs["exist_ok"] is True
        assert fake.val_kwargs["name"] == "model_a-test"
        assert Path(fake.val_kwargs["project"]).is_absolute()
        assert Path(fake.val_kwargs["project"]).parts[-2:] == ("runs", "eval")
        assert section["checkpoint"] == ckpt.as_posix()
        assert section["data"] == data_yaml.as_posix()
        assert section["split"] == "test"
        assert section["device"] == "cpu"
        assert section["class_names"] == ["target"]
        assert section["n_split_images"] == 2
        assert section["overall"]["box"]["map50"] == 0.7
        assert section["overall"]["mask"]["map50_95"] == 0.45
        assert section["counts"]["instances_per_class"]["target"] == 10

    def test_default_factory_resolved_at_call_time(self, tmp_path, monkeypatch):
        ckpt, data_yaml = _write_dataset(tmp_path)
        fake = FakeYOLO(make_metrics(nc=1, names={0: "target"}))
        monkeypatch.setattr(ev, "_load_yolo", lambda p: fake)
        task = ev.EvalTask("model_a", ckpt, data_yaml)
        section = ev.evaluate_task(task, split="test", device="cpu")
        assert section["overall"]["box"]["map50"] == 0.7

    def test_missing_checkpoint_raises_before_model_load(self, tmp_path):
        _, data_yaml = _write_dataset(tmp_path)
        task = ev.EvalTask("model_a", tmp_path / "missing.pt", data_yaml)
        with pytest.raises(FileNotFoundError):
            ev.evaluate_task(task, split="test", device="cpu",
                             yolo_factory=lambda p: pytest.fail("must not load"))


# ---------------------------------------------------------------------------
# Report building and writing
# ---------------------------------------------------------------------------

class TestReport:
    def test_build_report_keys_models_by_name(self):
        tasks = [
            ev.EvalTask("model_baseline", Path("a.pt"), Path("a.yaml")),
            ev.EvalTask("model_a", Path("b.pt"), Path("b.yaml")),
        ]
        sections = [{"checkpoint": "a.pt"}, {"checkpoint": "b.pt"}]
        report = ev.build_report(tasks, sections, split="test", device="0")
        assert report["split"] == "test"
        assert report["device"] == "0"
        assert set(report["models"]) == {"model_baseline", "model_a"}
        assert report["models"]["model_a"]["checkpoint"] == "b.pt"

    def test_write_report_round_trips_stable_json(self, tmp_path):
        report = {
            "split": "test", "device": "cpu",
            "models": {"model_a": {"overall": {"box": {"map50": 0.5}}}},
        }
        out = ev.write_report(report, tmp_path / "nested" / "report.json")
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert json.loads(text) == report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_parse_args_defaults_target_formal_paths(self):
        assert ev.DEFAULT_BASELINE_CHECKPOINT.as_posix() == \
            "runs/train/run_20260913_initial/model_baseline/weights/best.pt"
        assert ev.DEFAULT_BASELINE_DATA.as_posix() == \
            "datasets/underwater_seg_v2/data.yaml"
        assert ev.DEFAULT_MODEL_A_CHECKPOINT.as_posix() == \
            "runs/train/run_20260913_initial/model_a/weights/best.pt"
        assert ev.DEFAULT_MODEL_A_DATA.as_posix() == \
            "datasets/underwater_seg_binary_v2/data.yaml"
        args = ev.parse_args(["--output", "out.json"])
        assert args.baseline_checkpoint == ev.DEFAULT_BASELINE_CHECKPOINT
        assert args.baseline_data == ev.DEFAULT_BASELINE_DATA
        assert args.model_a_checkpoint == ev.DEFAULT_MODEL_A_CHECKPOINT
        assert args.model_a_data == ev.DEFAULT_MODEL_A_DATA
        assert args.split == "test"
        assert args.device == "cpu"
        assert args.imgsz == 640
        assert args.batch == 8
        assert args.workers == 0

    def test_output_is_required(self):
        with pytest.raises(SystemExit):
            ev.parse_args([])

    def test_main_missing_checkpoint_returns_error(self, tmp_path, capsys):
        _, data_yaml = _write_dataset(tmp_path)
        rc = ev.main([
            "--baseline-checkpoint", str(tmp_path / "missing.pt"),
            "--baseline-data", str(data_yaml),
            "--model-a-checkpoint", str(tmp_path / "missing.pt"),
            "--model-a-data", str(data_yaml),
            "--output", str(tmp_path / "report.json"),
        ])
        assert rc == 2
        assert not (tmp_path / "report.json").exists()
        assert "missing.pt" in capsys.readouterr().err

    def test_main_writes_report_for_both_models(self, tmp_path, monkeypatch):
        ckpt, data_yaml = _write_dataset(tmp_path)
        monkeypatch.setattr(
            ev, "_load_yolo",
            lambda p: FakeYOLO(make_metrics(nc=1, names={0: "target"})),
        )
        out = tmp_path / "report.json"
        rc = ev.main([
            "--baseline-checkpoint", str(ckpt), "--baseline-data", str(data_yaml),
            "--model-a-checkpoint", str(ckpt), "--model-a-data", str(data_yaml),
            "--output", str(out),
        ])
        assert rc == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["split"] == "test"
        assert report["device"] == "cpu"
        assert set(report["models"]) == {"model_baseline", "model_a"}
        for section in report["models"].values():
            assert section["overall"]["mask"]["map50_95"] == 0.45


def test_import_does_not_load_ultralytics():
    code = (
        "import sys; "
        "import scripts.eval_segmentation; "
        "assert 'ultralytics' not in sys.modules"
    )
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root))
