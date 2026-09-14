"""Tests for scripts.eval_speed: warmup/benchmark argument forwarding, stats
and the end-to-end pipeline benchmark (fakes only, no real model loading)."""

import json
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.eval_speed as es


class RecordingModel:
    def __init__(self):
        self.calls: list[dict] = []

    def predict(self, source=None, **kwargs):
        self.calls.append({"source": source, **kwargs})
        return []


@pytest.fixture
def model():
    return RecordingModel()


@pytest.fixture
def img():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_warmup_forwards_device_and_imgsz(model, img):
    es.warmup(model, img, device="cuda:1", imgsz=224, n=3)
    assert len(model.calls) == 3
    for call in model.calls:
        assert call["device"] == "cuda:1"
        assert call["imgsz"] == 224
        assert call["verbose"] is False


def test_benchmark_forwards_device_and_imgsz(model, img):
    es.benchmark(model, img, device="0", imgsz=640, n=5)
    assert len(model.calls) == 5
    for call in model.calls:
        assert call["device"] == "0"
        assert call["imgsz"] == 640


def test_benchmark_reports_p50_p95_mean_fps(model, img):
    stats = es.benchmark(model, img, device="cpu", imgsz=64, n=20)
    for key in ("mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "fps"):
        assert key in stats, f"missing stat: {key}"
    assert stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]
    if stats["mean_ms"] > 0:
        assert stats["fps"] == pytest.approx(1000.0 / stats["mean_ms"], rel=0.02)
    else:
        assert stats["fps"] == 0.0


def test_benchmark_deterministic_bounds(model, img):
    stats = es.benchmark(model, img, device="cpu", imgsz=64, n=10)
    assert stats["mean_ms"] >= 0
    assert stats["fps"] >= 0


class FakePipelineEngine:
    device = "cpu"

    def __init__(self, n_instances: int = 3, n_valid_depths: int = 2):
        self.n_instances = n_instances
        self.n_valid_depths = n_valid_depths
        self.calls: list[dict] = []

    def process_frame(self, left, right, sync_skew_ms=None, already_rectified=False):
        self.calls.append({"already_rectified": already_rectified})
        instances = list(range(self.n_instances))
        depths = [
            {"instance_id": i, "valid": i < self.n_valid_depths}
            for i in range(self.n_instances)
        ]
        return instances, depths, None


class TestPipelineBenchmark:
    def _images(self):
        left = np.zeros((16, 16, 3), dtype=np.uint8)
        right = np.zeros((16, 16, 3), dtype=np.uint8)
        return left, right

    def test_reports_stats_and_last_frame_counts(self):
        engine = FakePipelineEngine()
        left, right = self._images()
        stats = es.benchmark_pipeline(
            engine, left, right, already_rectified=True, warmup_n=2, n=5
        )
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "fps"):
            assert key in stats, f"missing stat: {key}"
        assert stats["mode"] == "pipeline"
        assert stats["already_rectified"] is True
        assert stats["warmup"] == 2
        assert stats["repetitions"] == 5
        assert stats["last_frame_instances"] == 3
        assert stats["last_frame_valid_depths"] == 2
        assert len(engine.calls) == 7
        assert all(call["already_rectified"] for call in engine.calls)

    def test_syncs_around_every_timed_frame(self):
        engine = FakePipelineEngine()
        left, right = self._images()
        sync_calls: list[int] = []
        stats = es.benchmark_pipeline(
            engine, left, right, warmup_n=3, n=4, sync=lambda: sync_calls.append(1)
        )
        assert len(sync_calls) == 2 * 4
        assert stats["repetitions"] == 4

    def test_default_sync_resolution_is_safe_on_cpu(self):
        engine = FakePipelineEngine()
        left, right = self._images()
        stats = es.benchmark_pipeline(engine, left, right, warmup_n=0, n=1)
        assert stats["last_frame_instances"] == 3

    def test_last_frame_counts_follow_returned_depths(self):
        engine = FakePipelineEngine(n_instances=5, n_valid_depths=1)
        left, right = self._images()
        stats = es.benchmark_pipeline(engine, left, right, warmup_n=0, n=2)
        assert stats["last_frame_instances"] == 5
        assert stats["last_frame_valid_depths"] == 1


class TestRunPipelineBenchmark:
    def test_uses_build_pipeline_engine_and_reports_metadata(self, monkeypatch, tmp_path):
        engine = FakePipelineEngine()
        seen = {}

        def fake_build(config_path):
            seen["config"] = config_path
            return engine

        monkeypatch.setattr(es, "build_pipeline_engine", fake_build)
        left = tmp_path / "left.png"
        right = tmp_path / "right.png"
        cv2.imwrite(str(left), np.zeros((16, 16), np.uint8))
        cv2.imwrite(str(right), np.zeros((16, 16), np.uint8))

        report = es.run_pipeline_benchmark(
            Path("configs/default.yaml"), left, right,
            already_rectified=True, warmup=1, n=2,
        )
        assert seen["config"] == Path("configs/default.yaml")
        assert report["mode"] == "pipeline"
        assert report["config"] == "configs/default.yaml"
        assert report["left_image"] == str(left)
        assert report["right_image"] == str(right)
        assert report["device"] == "cpu"
        assert report["already_rectified"] is True
        assert report["last_frame_instances"] == 3

    def test_missing_image_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(es, "build_pipeline_engine", lambda cfg: FakePipelineEngine())
        with pytest.raises(FileNotFoundError):
            es.run_pipeline_benchmark(
                Path("cfg.yaml"), tmp_path / "nope_l.png", tmp_path / "nope_r.png",
                warmup=0, n=1,
            )


class TestBuildEngine:
    def test_build_pipeline_engine_uses_from_config(self, monkeypatch):
        from gui import inference_engine as gie

        sentinel = object()
        seen = {}

        def fake_from_config(cls, config_path=None, **kwargs):
            seen["config_path"] = config_path
            return sentinel

        monkeypatch.setattr(
            gie.DualStageInferenceEngine, "from_config", classmethod(fake_from_config)
        )
        assert es.build_pipeline_engine(Path("my.yaml")) is sentinel
        assert seen["config_path"] == Path("my.yaml")


class TestCli:
    def test_model_mode_defaults_preserved(self):
        args = es.parse_args(["--model", "best.pt"])
        assert args.mode == "model"
        assert args.device == "cuda"
        assert args.imgsz == 640
        assert args.n == 100
        assert args.warmup == 10

    def test_pipeline_mode_defaults(self):
        args = es.parse_args(["--mode", "pipeline", "--left", "l.png", "--right", "r.png"])
        assert args.mode == "pipeline"
        assert args.config == Path("configs/default.yaml")
        assert args.already_rectified is False

    def test_model_mode_requires_model(self):
        with pytest.raises(SystemExit):
            es.parse_args([])

    def test_pipeline_mode_requires_images(self):
        with pytest.raises(SystemExit):
            es.parse_args(["--mode", "pipeline"])

    @pytest.mark.parametrize(
        "argv",
        (["--model", "best.pt", "--n", "0"],
         ["--model", "best.pt", "--warmup", "-1"]),
    )
    def test_cli_rejects_invalid_iteration_counts(self, argv):
        with pytest.raises(SystemExit):
            es.parse_args(argv)

    def test_main_pipeline_mode_writes_json_report(self, monkeypatch, tmp_path):
        fake_report = {
            "mode": "pipeline", "mean_ms": 5.0, "fps": 200.0,
            "last_frame_instances": 1, "last_frame_valid_depths": 1,
        }
        monkeypatch.setattr(es, "run_pipeline_benchmark", lambda *a, **k: dict(fake_report))
        out = tmp_path / "speed.json"
        rc = es.main([
            "--mode", "pipeline", "--left", "l.png", "--right", "r.png",
            "--output", str(out),
        ])
        assert rc == 0
        assert json.loads(out.read_text(encoding="utf-8")) == fake_report

    def test_main_model_mode_writes_json_report(self, monkeypatch, tmp_path):
        fake_ultralytics = types.ModuleType("ultralytics")

        class FakeYOLO:
            def __init__(self, path):
                pass

            def predict(self, img, **kwargs):
                return []

        fake_ultralytics.YOLO = FakeYOLO
        monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
        out = tmp_path / "speed.json"
        rc = es.main([
            "--model", "best.pt", "--n", "3", "--warmup", "1",
            "--output", str(out),
        ])
        assert rc == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["mode"] == "model"
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "fps"):
            assert key in report, f"missing stat: {key}"


def test_script_bootstrap_adds_project_root_to_sys_path():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "eval_speed.py"
    code = (
        "import runpy, sys; from pathlib import Path; "
        f"root=Path({str(root)!r}).resolve(); "
        "sys.path=[p for p in sys.path if p and Path(p).resolve()!=root]; "
        f"runpy.run_path({str(script)!r}, run_name='eval_speed_test'); "
        "assert str(root) in sys.path"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(root))
