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


class TestDeviceMapping:
    def test_ultralytics_style_ids_map_to_torch_names(self):
        assert tpf.torch_device_name("0") == "cuda:0"
        assert tpf.torch_device_name("1") == "cuda:1"
        assert tpf.torch_device_name("cuda") == "cuda:0"
        assert tpf.torch_device_name("cuda:1") == "cuda:1"
        assert tpf.torch_device_name("cpu") == "cpu"

    def test_unsupported_device_raises(self):
        with pytest.raises(ValueError, match="device"):
            tpf.torch_device_name("tpu")

    def test_resolve_device_validates_cuda_before_mapping(self, monkeypatch):
        from scripts.train_models import DeviceUnavailableError

        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        with pytest.raises(DeviceUnavailableError):
            tpf.resolve_device("0", cuda_available=False)


class FakeClassifyHead(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)


class FakeUltralyticsModel(torch.nn.Module):
    """Mimics the pieces of prepare_gray_backbone that touch YOLO models.

    Like the real ClassificationModel: train mode returns a logits tensor,
    eval mode returns a (softmax_probs, logits) tuple.
    """

    def __init__(self, out_features, names):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Identity(), FakeClassifyHead(1280, out_features)
        )
        self.names = names

    def forward(self, gray):
        # Feature dimension matches the head's in_features=1280; the
        # backbone body itself is not exercised in these tests.
        features = torch.zeros(gray.shape[0], 1280, device=gray.device)
        logits = self.model[-1].linear(features)
        if self.training:
            return logits
        return (logits.softmax(-1), logits)


class FakeYOLO:
    last_weights: str | None = None
    last_model_head = None

    def __init__(self, weights):
        FakeYOLO.last_weights = weights
        if "gray" in str(weights):
            # Accepted 4-class gray weights: plastic_fish=1, plastic_submarine=2
            self.model = FakeUltralyticsModel(
                4,
                {0: "metal_submarine", 1: "plastic_fish", 2: "plastic_submarine", 3: "real_fish"},
            )
        else:
            # Stock ImageNet base: 1000 classes, generic names
            self.model = FakeUltralyticsModel(
                1000, {i: f"imagenet_{i}" for i in range(1000)}
            )
        FakeYOLO.last_model_head = self.model.model[-1]


@pytest.fixture
def fake_ultralytics(monkeypatch):
    import types

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    FakeYOLO.last_weights = None
    FakeYOLO.last_model_head = None
    return module


class TestPrepareGrayBackbone:
    V3_NAMES = [
        "metal_submarine",
        "plastic_submarine",
        "plastic_fish",
        "real_fish",
    ]

    def test_gray_weights_branch_derives_perm(
        self, tmp_path, fake_ultralytics
    ):
        weights = tmp_path / "model_b-gray.pt"
        weights.write_bytes(b"fake")
        backbone, info = tpf.prepare_gray_backbone(
            tmp_path / "base.pt", str(weights), self.V3_NAMES, "cpu"
        )

        assert info["head_replaced"] is False
        assert info["perm"] == [0, 2, 1, 3]  # plastic_fish/submarine swap fixed
        assert fake_ultralytics.YOLO.last_weights == str(weights)
        # Adapter must expose raw logits in eval mode
        backbone.eval()
        out = backbone(torch.zeros(2, 3, 16, 16))
        assert out.shape == (2, 4)

    def test_missing_gray_weights_refused_not_silent(self, tmp_path, fake_ultralytics):
        with pytest.raises(FileNotFoundError, match="no\\s+automatic|refuse|missing"):
            tpf.prepare_gray_backbone(
                tmp_path / "base.pt",
                str(tmp_path / "missing.pt"),
                self.V3_NAMES,
                "cpu",
            )

    def test_base_branch_replaces_head_to_dataset_classes(
        self, tmp_path, fake_ultralytics
    ):
        backbone, info = tpf.prepare_gray_backbone(
            tmp_path / "yolo26n-cls.pt", "", self.V3_NAMES, "cpu"
        )

        assert info["head_replaced"] is True
        assert info["perm"] is None
        head = fake_ultralytics.YOLO.last_model_head
        assert head.linear.out_features == 4  # was 1000
        assert info["gray_class_names"] == self.V3_NAMES

    def test_head_width_mismatch_raises(self, tmp_path, fake_ultralytics, monkeypatch):
        weights = tmp_path / "wrong_gray.pt"
        weights.write_bytes(b"fake")
        original_init = FakeUltralyticsModel.__init__

        def narrow_init(self, out_features, names):
            original_init(self, 3, names)  # wrong class count

        monkeypatch.setattr(FakeUltralyticsModel, "__init__", narrow_init)
        with pytest.raises(ValueError, match="classes"):
            tpf.prepare_gray_backbone(
                tmp_path / "base.pt", str(weights), self.V3_NAMES, "cpu"
            )


class TestRestoreBackbone:
    """Freeze -> joint resume must rebuild the gray branch the same way."""

    def test_gray_weights_checkpoint_restores_from_recorded_weights(
        self, tmp_path, fake_ultralytics
    ):
        weights = tmp_path / "model_b-gray.pt"
        weights.write_bytes(b"fake")
        backbone = tpf._restore_backbone(
            {
                "base_model": "yolo26n-cls.pt",
                "gray_weights": str(weights),
                "head_replaced": False,
            },
            ["metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"],
            "cpu",
        )
        assert fake_ultralytics.YOLO.last_weights == str(weights)
        assert backbone.perm is not None and backbone.perm.tolist() == [0, 2, 1, 3]

    def test_fresh_head_checkpoint_restores_via_base_not_gray_weights(
        self, tmp_path, fake_ultralytics
    ):
        # A fresh-head checkpoint records the 1000-class base; restoring
        # must rebuild with a replaced head (base branch), never feed the
        # base into the gray-weights branch (which would fail on width).
        base = tmp_path / "yolo26n-cls.pt"
        base.write_bytes(b"fake")
        backbone = tpf._restore_backbone(
            {
                "base_model": str(base),
                "gray_weights": "",
                "head_replaced": True,
            },
            ["metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"],
            "cpu",
        )
        assert fake_ultralytics.YOLO.last_weights == str(base)
        assert backbone.perm is None
        assert backbone.module.model[-1].linear.out_features == 4

    def test_fresh_head_restore_missing_base_refused(self, fake_ultralytics):
        with pytest.raises(FileNotFoundError, match="no automatic download"):
            tpf._restore_backbone(
                {
                    "base_model": "missing-base.pt",
                    "gray_weights": "",
                    "head_replaced": True,
                },
                ["metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"],
                "cpu",
            )


class TestPhaseRunDir:
    def test_conflicting_run_directory_refused(self, tmp_path):
        run_dir = tpf.PROJECT_ROOT / "runs" / "train" / "test_phase_conflict" / "polar_fusion" / "freeze"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with pytest.raises(FileExistsError, match="new --run-id"):
                tpf._phase_run_dir("test_phase_conflict", "freeze")
        finally:
            import shutil

            shutil.rmtree(
                tpf.PROJECT_ROOT / "runs" / "train" / "test_phase_conflict",
                ignore_errors=True,
            )

    def test_fresh_directory_created(self, tmp_path):
        run_dir = tpf._phase_run_dir("test_phase_fresh_dir", "freeze")
        try:
            assert run_dir.is_dir()
            assert run_dir.name == "freeze"
        finally:
            import shutil

            shutil.rmtree(
                tpf.PROJECT_ROOT / "runs" / "train" / "test_phase_fresh_dir",
                ignore_errors=True,
            )


class TestResolveInitCheckpoint:
    def make_args(self, **overrides):
        argv = ["--run-id", "run_x", "--phase", "joint"]
        for key, value in overrides.items():
            if value is not None:
                argv += [f"--{key.replace('_', '-')}", str(value)]
        args = tpf.parse_args(argv)
        for key, value in overrides.items():
            if value is None:
                setattr(args, key, None)
        return args

    def test_joint_without_freeze_checkpoint_raises(self):
        args = self.make_args(run_id="no_such_run_for_joint")
        with pytest.raises(FileNotFoundError, match="init-from|freeze"):
            tpf._resolve_init_checkpoint(args)

    def test_joint_uses_freeze_best_automatically(self, tmp_path):
        run_dir = tpf.PROJECT_ROOT / "runs" / "train" / "run_with_freeze" / "polar_fusion" / "freeze"
        run_dir.mkdir(parents=True, exist_ok=True)
        best = run_dir / "best.pt"
        best.write_bytes(b"fake")
        try:
            args = self.make_args(run_id="run_with_freeze")
            assert tpf._resolve_init_checkpoint(args) == best
        finally:
            import shutil

            shutil.rmtree(
                tpf.PROJECT_ROOT / "runs" / "train" / "run_with_freeze",
                ignore_errors=True,
            )

    def test_explicit_init_from(self, tmp_path):
        ckpt = tmp_path / "some.pt"
        ckpt.write_bytes(b"fake")
        args = self.make_args(init_from=str(ckpt))
        assert tpf._resolve_init_checkpoint(args) == ckpt

    def test_missing_explicit_init_from_raises(self):
        args = self.make_args(init_from="missing.pt")
        with pytest.raises(FileNotFoundError, match="init-from"):
            tpf._resolve_init_checkpoint(args)

    def test_freeze_phase_needs_no_init(self):
        args = tpf.parse_args(["--run-id", "run_x", "--phase", "freeze"])
        assert tpf._resolve_init_checkpoint(args) is None


class TinyBackbone(torch.nn.Module):
    """Local stand-in backbone (tests package is not importable by name)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, num_classes),
        )

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        return self.net(gray)


class TestPhaseModes:
    def test_freeze_keeps_gray_backbone_in_eval(self):
        from models.polar_fusion import GrayBackboneAdapter, PolarFusionModel

        model = PolarFusionModel(
            GrayBackboneAdapter(TinyBackbone(4)), num_classes=4
        )
        model.set_gray_frozen(True)
        tpf._apply_phase_modes(model, joint=False)

        # BN statistics of the frozen gray branch must not update.
        assert model.gray_backbone.training is False
        assert model.gray_backbone.module.training is False
        assert model.delta_net.training is True
        assert model.gate_net.training is True

    def test_joint_trains_everything(self):
        from models.polar_fusion import GrayBackboneAdapter, PolarFusionModel

        model = PolarFusionModel(
            GrayBackboneAdapter(TinyBackbone(4)), num_classes=4
        )
        tpf._apply_phase_modes(model, joint=True)
        assert model.gray_backbone.module.training is True


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
                "--gray-weights", "",
            ]
        )
        # Missing base is reported, not fatal, and never downloaded.
        assert exit_code == 0
        err = capsys.readouterr().err
        assert "nonexistent-base.pt" in err
        assert "no automatic download" in err.lower()

    def test_dry_run_skips_base_note_when_gray_weights_present(
        self, mini_dataset, capsys
    ):
        # Any existing file works: dry-run only checks presence.
        stand_in = str(mini_dataset / "dataset_manifest.csv")
        exit_code = tpf.main(
            [
                "--run-id", "run_dry",
                "--dry-run",
                "--data", str(mini_dataset),
                "--base", "nonexistent-base.pt",
                "--gray-weights", stand_in,
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "nonexistent-base.pt" not in captured.err
        assert '"gray_weights_present": true' in captured.out

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
        monkeypatch.setattr(tpf, "resolve_device", lambda *a, **kw: "cpu")
        with pytest.raises(FileNotFoundError, match="no automatic download"):
            tpf.run_training(
                tpf.parse_args(
                    [
                        "--run-id", "run_x",
                        "--data", str(mini_dataset),
                        "--base", "definitely-missing.pt",
                        "--gray-weights", "",
                    ]
                )
            )

    def test_training_with_gray_weights_does_not_require_base(
        self, mini_dataset, monkeypatch
    ):
        """The loaded gray checkpoint carries its own architecture, so a
        missing --base must not block the run (issue: base forced yolo26
        even for the YOLOv8 legacy weights)."""
        monkeypatch.setattr(tpf, "validate_dataset", lambda root: None)
        monkeypatch.setattr(tpf, "resolve_device", lambda *a, **kw: "cpu")

        def refuse(*a, **kw):
            raise FileNotFoundError("prepare_gray_backbone reached")

        monkeypatch.setattr(tpf, "prepare_gray_backbone", refuse)
        stand_in = str(mini_dataset / "dataset_manifest.csv")
        with pytest.raises(FileNotFoundError, match="prepare_gray_backbone"):
            tpf.run_training(
                tpf.parse_args(
                    [
                        "--run-id", "run_x",
                        "--data", str(mini_dataset),
                        "--base", "definitely-missing.pt",
                        "--gray-weights", stand_in,
                    ]
                )
            )

    def test_run_training_validates_dataset_first(self, mini_dataset, monkeypatch):
        def refuse(*a, **kw):
            raise FileNotFoundError("base checkpoint missing; no automatic download")

        monkeypatch.setattr(tpf, "prepare_gray_backbone", refuse)
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
