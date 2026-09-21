"""Tests for the archived four-stage trainer
(scripts.legacy_four_stage.train_models: stage defaults, argument
construction, CUDA fail-fast) and the archived train_seg shim.

The active three-stage mainline is covered by tests/test_train_pipeline.py
and tests/test_train_model_a.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.legacy_four_stage.train_models as tm
import scripts.legacy_four_stage.train_seg as ts


class TestStageDefaults:
    def test_baseline_and_a_use_seg_defaults(self):
        for stage in ("baseline", "a"):
            defaults = tm.STAGE_DEFAULTS[stage]
            assert defaults["base"] == "yolo26n-seg.pt"
            assert defaults["imgsz"] == 640
            assert defaults["batch"] == 8
            assert defaults["epochs"] == 100

    def test_baseline_data_is_v2_root(self):
        assert tm.STAGE_DEFAULTS["baseline"]["data"] == \
            "datasets/underwater_seg_v2/data.yaml"

    def test_model_a_data_is_binary_v2_root(self):
        assert tm.STAGE_DEFAULTS["a"]["data"] == \
            "datasets/underwater_seg_binary_v2/data.yaml"

    def test_b_stages_use_cls_defaults_and_v2_roots(self):
        for stage in ("b-gray", "b-polar"):
            defaults = tm.STAGE_DEFAULTS[stage]
            assert defaults["base"] == "yolo26n-cls.pt"
            assert defaults["imgsz"] == 224
            assert defaults["batch"] == 64
            assert defaults["epochs"] == 80
        assert tm.STAGE_DEFAULTS["b-gray"]["data"] == "datasets/underwater_cls_gray_v2"
        assert tm.STAGE_DEFAULTS["b-polar"]["data"] == "datasets/underwater_cls_polar_v2"

    def test_development_bases_are_yolo26_variants(self):
        assert tm.SEG_BASE == "yolo26n-seg.pt"
        assert tm.CLS_BASE == "yolo26n-cls.pt"

    def test_legacy_yolov8_bases_kept_for_reference(self):
        assert tm.LEGACY_SEG_BASE == "yolov8n-seg.pt"
        assert tm.LEGACY_CLS_BASE == "yolov8n-cls.pt"

    def test_detection_checkpoint_is_not_a_training_base(self):
        # The local yolo26n.pt is a detection model: it must never be the
        # base for Model A (seg) or Model B (cls).
        for defaults in tm.STAGE_DEFAULTS.values():
            assert defaults["base"] != "yolo26n.pt"


class TestBuildTrainKwargs:
    def test_seed_and_determinism(self):
        kwargs = tm.build_train_kwargs("baseline")
        assert kwargs["seed"] == 2026
        assert kwargs["deterministic"] is True
        assert kwargs["amp"] is True
        assert kwargs["patience"] == 20
        assert kwargs["save"] is True
        assert kwargs["project"] == tm.DEFAULT_PROJECT
        assert kwargs["name"] == "model_baseline"

    def test_b_stages_disable_hsv(self):
        for stage in ("b-gray", "b-polar"):
            kwargs = tm.build_train_kwargs(stage)
            assert kwargs["auto_augment"] is None
            assert kwargs["erasing"] == 0.0
            assert kwargs["hsv_h"] == 0.0
            assert kwargs["hsv_s"] == 0.0
            assert kwargs["hsv_v"] == 0.0

    def test_seg_stages_do_not_touch_hsv(self):
        for stage in ("baseline", "a"):
            kwargs = tm.build_train_kwargs(stage)
            assert "auto_augment" not in kwargs
            assert "erasing" not in kwargs
            assert "hsv_h" not in kwargs

    def test_b_gray_and_b_polar_differ_only_in_data_and_name(self):
        gray = tm.build_train_kwargs("b-gray")
        polar = tm.build_train_kwargs("b-polar")
        assert gray.pop("data") != polar.pop("data")
        assert gray.pop("name") != polar.pop("name")
        assert gray == polar

    def test_overrides_are_applied(self):
        kwargs = tm.build_train_kwargs("b-gray", epochs=2, imgsz=160, batch=4,
                                       name="smoke")
        assert kwargs["epochs"] == 2
        assert kwargs["imgsz"] == 160
        assert kwargs["batch"] == 4
        assert kwargs["name"] == "smoke"

    def test_default_workers_forwarded_to_every_stage(self):
        for stage in tm.STAGE_DEFAULTS:
            assert tm.build_train_kwargs(stage)["workers"] == tm.DEFAULT_WORKERS

    def test_workers_override_is_applied(self):
        assert tm.build_train_kwargs("b-gray", workers=2)["workers"] == 2

    def test_default_workers_zero_on_windows_only(self):
        assert tm.DEFAULT_WORKERS == (0 if sys.platform == "win32" else 8)

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError):
            tm.build_train_kwargs("c")


GROUPED_RUN_ID = "run_20260913_initial"
GROUPED_MODEL_B_GRAY = f"runs/train/{GROUPED_RUN_ID}/model_b-gray"


class TestValidateRunId:
    def test_accepts_safe_ids(self):
        for run_id in ("run_20260913_initial", "run_20260914_manual_labels",
                       "a.b-c_d", "Run.1-x_2", "r", "20260914"):
            assert tm.validate_run_id(run_id) == run_id

    def test_rejects_empty_dot_and_dotdot(self):
        for run_id in ("", ".", ".."):
            with pytest.raises(tm.InvalidRunIdError):
                tm.validate_run_id(run_id)

    def test_rejects_path_separators_and_traversal(self):
        for run_id in ("a/b", "a\\b", "/abs", "C:\\abs", "../escape",
                       "runs/train", "..\\escape"):
            with pytest.raises(tm.InvalidRunIdError):
                tm.validate_run_id(run_id)

    def test_rejects_other_unsafe_characters(self):
        for run_id in ("a b", "run;id", "run:id", "run(id)", "视角"):
            with pytest.raises(tm.InvalidRunIdError):
                tm.validate_run_id(run_id)


class TestRunIdCLI:
    def test_parse_args_requires_run_id(self):
        with pytest.raises(SystemExit):
            tm.parse_args(["--stage", "b-gray"])

    def test_parse_args_accepts_run_id(self):
        args = tm.parse_args(["--stage", "b-gray", "--run-id", "run_20260914_manual_labels"])
        assert args.run_id == "run_20260914_manual_labels"

    def test_main_forwards_run_id(self, monkeypatch):
        captured = {}

        def fake_train_stage(stage, **kwargs):
            captured.update(kwargs)
            return Path("runs/train/run_x/model_b-gray")

        monkeypatch.setattr(tm, "train_stage", fake_train_stage)
        exit_code = tm.main(["--stage", "b-gray", "--device", "cpu",
                             "--run-id", "run_x"])
        assert exit_code == 0
        assert captured["run_id"] == "run_x"

    def test_main_rejects_invalid_run_id(self, monkeypatch, capsys):
        def fake_train_stage(stage, **kwargs):
            raise AssertionError("train_stage must not run with a bad run id")

        monkeypatch.setattr(tm, "train_stage", fake_train_stage)
        exit_code = tm.main(["--stage", "b-gray", "--device", "cpu",
                             "--run-id", "../escape"])
        assert exit_code == 2
        assert "run id" in capsys.readouterr().err.lower()


class TestTrainStageRunId:
    def test_run_id_groups_output_under_project(
        self, fake_train_ultralytics, monkeypatch,
    ):
        monkeypatch.setattr(tm, "resolve_device", lambda device: device)
        tm.train_stage(
            "b-gray", device="cpu", run_id=GROUPED_RUN_ID,
            model_factory=fake_train_ultralytics,
        )
        assert FakeTrainYOLO.last_kwargs["project"] == str(
            tm.PROJECT_ROOT / "runs" / "train" / GROUPED_RUN_ID
        )
        assert FakeTrainYOLO.last_kwargs["name"] == "model_b-gray"
        assert FakeTrainYOLO.last_kwargs["exist_ok"] is False

    def test_run_id_without_project_override_uses_default_project(
        self, fake_train_ultralytics, monkeypatch,
    ):
        monkeypatch.setattr(tm, "resolve_device", lambda device: device)
        tm.train_stage(
            "baseline", device="cpu", run_id="run_x",
            model_factory=fake_train_ultralytics,
        )
        assert FakeTrainYOLO.last_kwargs["project"] == str(
            tm.PROJECT_ROOT / "runs" / "train" / "run_x"
        )

    def test_no_run_id_keeps_backward_compatible_project(
        self, fake_train_ultralytics, monkeypatch,
    ):
        monkeypatch.setattr(tm, "resolve_device", lambda device: device)
        tm.train_stage(
            "b-gray", device="cpu", model_factory=fake_train_ultralytics,
        )
        assert FakeTrainYOLO.last_kwargs["project"] == str(
            tm.PROJECT_ROOT / "runs" / "train"
        )

    def test_invalid_run_id_raises_before_training(
        self, fake_train_ultralytics, monkeypatch,
    ):
        monkeypatch.setattr(tm, "resolve_device", lambda device: device)
        with pytest.raises(tm.InvalidRunIdError):
            tm.train_stage(
                "b-gray", device="cpu", run_id="a/b",
                model_factory=fake_train_ultralytics,
            )
        assert FakeTrainYOLO.last_kwargs is None


class TestResolveProject:
    def test_relative_project_resolves_against_repo_root(self):
        assert tm.resolve_project("runs/train") == str(
            tm.PROJECT_ROOT / "runs" / "train"
        )

    def test_absolute_project_is_preserved(self):
        absolute = str(tm.PROJECT_ROOT / "custom_runs")
        assert tm.resolve_project(absolute) == absolute


class TestResolveDevice:
    def test_cpu_always_allowed(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        assert tm.resolve_device("cpu", cuda_available=False) == "cpu"

    def test_cuda_requested_but_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        with pytest.raises(tm.DeviceUnavailableError, match="cuda"):
            tm.resolve_device("0", cuda_available=False)
        with pytest.raises(tm.DeviceUnavailableError):
            tm.resolve_device("cuda", cuda_available=False)

    def test_cuda_available_passes(self):
        assert tm.resolve_device("0", cuda_available=True) == "0"


class FakeTrainer:
    save_dir = "runs/train/run_20260913_initial/model_b-gray"


class FakeTrainYOLO:
    last_kwargs: dict | None = None
    weight_loaded: str | None = None

    def __init__(self, weights: str):
        FakeTrainYOLO.weight_loaded = weights
        self.trainer = FakeTrainer()

    def train(self, **kwargs):
        FakeTrainYOLO.last_kwargs = kwargs


@pytest.fixture
def fake_train_ultralytics():
    FakeTrainYOLO.last_kwargs = None
    return FakeTrainYOLO


def test_train_stage_forwards_kwargs_and_override(
    fake_train_ultralytics, monkeypatch,
):
    monkeypatch.setattr(tm, "resolve_device", lambda device: device)
    save_dir = tm.train_stage(
        "b-gray", device="cpu", epochs=1, data="datasets/underwater_cls_gray_v2",
        model_factory=fake_train_ultralytics,
    )
    kwargs = FakeTrainYOLO.last_kwargs
    assert kwargs["data"] == "datasets/underwater_cls_gray_v2"
    assert kwargs["epochs"] == 1
    assert kwargs["seed"] == 2026
    assert kwargs["hsv_h"] == 0.0
    assert kwargs["device"] == "cpu"
    assert FakeTrainYOLO.weight_loaded == "yolo26n-cls.pt"
    assert Path(save_dir) == Path(FakeTrainer.save_dir)


def test_train_stage_forwards_cuda_device_after_validation(
    fake_train_ultralytics, monkeypatch,
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    tm.train_stage(
        "b-gray", device="0", model_factory=fake_train_ultralytics,
    )
    assert FakeTrainYOLO.last_kwargs["device"] == "0"


def test_train_stage_resolves_relative_project_against_repo_root(
    fake_train_ultralytics, monkeypatch,
):
    monkeypatch.setattr(tm, "resolve_device", lambda device: device)
    tm.train_stage(
        "b-gray", device="cpu", model_factory=fake_train_ultralytics,
    )
    assert FakeTrainYOLO.last_kwargs["project"] == str(
        tm.PROJECT_ROOT / "runs" / "train"
    )


def test_train_stage_preserves_absolute_project(
    fake_train_ultralytics, monkeypatch,
):
    monkeypatch.setattr(tm, "resolve_device", lambda device: device)
    absolute = str(tm.PROJECT_ROOT / "custom_runs")
    tm.train_stage(
        "b-gray", device="cpu", project=absolute,
        model_factory=fake_train_ultralytics,
    )
    assert FakeTrainYOLO.last_kwargs["project"] == absolute


def test_train_stage_forwards_workers(fake_train_ultralytics, monkeypatch):
    monkeypatch.setattr(tm, "resolve_device", lambda device: device)
    tm.train_stage(
        "b-gray", device="cpu", workers=3,
        model_factory=fake_train_ultralytics,
    )
    assert FakeTrainYOLO.last_kwargs["workers"] == 3


def test_train_stage_validates_device_before_forwarding(
    fake_train_ultralytics, monkeypatch,
):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(tm.DeviceUnavailableError):
        tm.train_stage(
            "b-gray", device="0", model_factory=fake_train_ultralytics,
        )
    assert FakeTrainYOLO.last_kwargs is None


def test_main_fails_fast_without_cuda(monkeypatch, capsys):
    def boom(device, cuda_available=None):
        raise tm.DeviceUnavailableError("CUDA requested but unavailable")

    monkeypatch.setattr(tm, "resolve_device", boom)
    exit_code = tm.main(["--stage", "baseline", "--device", "0",
                         "--run-id", GROUPED_RUN_ID])
    assert exit_code == 2
    assert "CUDA" in capsys.readouterr().err


def test_parse_args_workers_default_and_override():
    assert tm.parse_args(["--stage", "b-gray", "--run-id", "run_x"]).workers \
        == tm.DEFAULT_WORKERS
    assert tm.parse_args(["--stage", "b-gray", "--run-id", "run_x",
                          "--workers", "5"]).workers == 5


def test_main_forwards_workers(monkeypatch):
    captured = {}

    def fake_train_stage(stage, **kwargs):
        captured.update(kwargs)
        return Path("runs/train/run_x/model_b-gray")

    monkeypatch.setattr(tm, "train_stage", fake_train_stage)
    exit_code = tm.main(["--stage", "b-gray", "--device", "cpu",
                         "--run-id", "run_x", "--workers", "5"])
    assert exit_code == 0
    assert captured["workers"] == 5


class TestTrainSegShim:
    def test_shim_requires_run_id(self):
        with pytest.raises(SystemExit):
            ts.parse_args(["--stage", "b", "--data", "datasets/underwater_cls_polar_v2"])

    def test_shim_maps_stage_b_to_b_polar_and_forwards_run_id(self, monkeypatch):
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        monkeypatch.setattr(tm, "main", fake_main)
        exit_code = ts.main([
            "--stage", "b",
            "--data", "datasets/underwater_cls_polar_v2",
            "--run-id", "run_20260914_manual_labels",
        ])
        assert exit_code == 0
        argv = captured["argv"]
        assert argv[argv.index("--stage") + 1] == "b-polar"
        assert argv[argv.index("--run-id") + 1] == "run_20260914_manual_labels"
        assert "datasets/underwater_cls_polar_v2" in argv

    def test_shim_maps_stage_a(self, monkeypatch):
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        monkeypatch.setattr(tm, "main", fake_main)
        ts.main(["--stage", "a",
                 "--data", "datasets/underwater_seg_binary_v2/data.yaml",
                 "--run-id", "run_20260914_manual_labels"])
        argv = captured["argv"]
        assert argv[argv.index("--stage") + 1] == "a"
        assert argv[argv.index("--run-id") + 1] == "run_20260914_manual_labels"

    def test_first_conv_surgery_removed_from_production_path(self):
        source = Path(ts.__file__).read_text(encoding="utf-8")
        assert "modify_first_conv" not in source
        assert not hasattr(ts, "modify_first_conv")
