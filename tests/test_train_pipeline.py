"""Tests for scripts.train_pipeline (three-stage orchestrator).

No real training: subprocess execution, dataset gates and checkpoint probes
are faked. The real GPU chain is covered by the --smoke run.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_pipeline as vp

V3_CLASS_NAMES = [
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
]


@pytest.fixture
def tmp_path() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="train_pipeline_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _fake_runner(return_codes, calls):
    codes = iter(return_codes)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=next(codes))

    return run


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """Neutralize heavy preflight/artifact checks and redirect the project
    root to a temp dir."""
    monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        vp, "validate_dataset", lambda root: {"audit_fingerprint": "fp_v4"}
    )
    monkeypatch.setattr(vp, "resolve_device", lambda device, *a, **kw: device)
    monkeypatch.setattr(
        vp, "_check_seg_base",
        lambda path: {"task": "segment", "architecture": "yolo26n-seg"},
    )
    monkeypatch.setattr(
        vp, "_check_cls_base",
        lambda path: {"task": "classify", "architecture": "yolo26n-cls"},
    )
    monkeypatch.setattr(vp, "_check_seg_data", lambda path: {"names": ["target"]})
    monkeypatch.setattr(
        vp, "_check_gray_ready", lambda run_id, imgsz, fp: {"ready": True}
    )
    monkeypatch.setattr(
        vp, "_check_stage_artifacts",
        lambda stage, run_id, params, require_formal, fp: {"stage": stage},
    )
    return tmp_path


def _run(argv, runner_codes=(0, 0, 0), calls=None):
    calls = [] if calls is None else calls
    return vp.main(argv, runner=_fake_runner(runner_codes, calls)), calls


class TestStageSelection:
    def test_default_stages_are_all_three_in_fixed_order(self):
        args = vp.parse_args(["--run-id", "run_x"])
        assert tuple(args.stages) == ("model_a", "gray_fusion", "fusion_freeze")

    def test_unknown_stage_rejected_by_parser(self):
        with pytest.raises(SystemExit):
            vp.parse_args(["--run-id", "run_x", "--stages", "baseline"])

    def test_duplicate_stage_rejected(self, pipeline_env, capsys):
        exit_code, calls = _run(
            ["--run-id", "run_x", "--device", "cpu",
             "--stages", "gray_fusion", "gray_fusion"]
        )
        assert exit_code == 2
        assert calls == []

    def test_wrong_order_rejected(self, pipeline_env, capsys):
        exit_code, calls = _run(
            ["--run-id", "run_x", "--device", "cpu",
             "--stages", "fusion_freeze", "gray_fusion"]
        )
        assert exit_code == 2
        assert calls == []
        assert "order" in capsys.readouterr().err.lower()

    def test_invalid_run_id_rejected_before_any_work(self, pipeline_env, capsys):
        exit_code, calls = _run(
            ["--run-id", "../escape", "--device", "cpu"]
        )
        assert exit_code == 2
        assert calls == []
        assert "run id" in capsys.readouterr().err.lower()

    def test_imgsz_mismatch_between_gray_and_fusion_rejected(
        self, pipeline_env, capsys
    ):
        exit_code, calls = _run(
            ["--run-id", "run_x", "--device", "cpu",
             "--gray-imgsz", "224", "--fusion-imgsz", "320"]
        )
        assert exit_code == 2
        assert calls == []
        assert "imgsz" in capsys.readouterr().err.lower()


class TestRunFlow:
    def test_three_stages_run_in_order_as_separate_processes(
        self, pipeline_env
    ):
        exit_code, calls = _run(["--run-id", "run_x", "--device", "cpu"])
        assert exit_code == 0
        assert len(calls) == 3
        scripts = [Path(call[0][1]).name for call in calls]
        assert scripts == [
            "train_model_a.py", "train_gray_fusion.py", "train_polar_fusion.py",
        ]
        for command, kwargs in calls:
            assert command[0] == sys.executable
            assert command[command.index("--run-id") + 1] == "run_x"
            assert kwargs["cwd"] == pipeline_env
            assert kwargs["check"] is False

    def test_fusion_receives_this_round_gray_best(self, pipeline_env):
        _, calls = _run(["--run-id", "run_x", "--device", "cpu"])
        fusion = calls[2][0]
        assert fusion[fusion.index("--phase") + 1] == "freeze"
        expected = str(
            pipeline_env / "runs" / "train" / "run_x" / "gray_fusion" / "best.pt"
        )
        assert fusion[fusion.index("--gray-weights") + 1] == expected

    def test_stage_failure_stops_remaining_stages(self, pipeline_env, capsys):
        exit_code, calls = _run(
            ["--run-id", "run_x", "--device", "cpu"], runner_codes=(0, 5)
        )
        assert exit_code == 5
        assert len(calls) == 2
        assert "remaining" in capsys.readouterr().err.lower()

    def test_missing_artifacts_after_success_stops(
        self, pipeline_env, monkeypatch
    ):
        def artifact_check(stage, run_id, params, require_formal, fp):
            if stage == "gray_fusion":
                raise RuntimeError("gray_fusion best.pt missing")
            return {"stage": stage}

        monkeypatch.setattr(vp, "_check_stage_artifacts", artifact_check)
        exit_code, calls = _run(["--run-id", "run_x", "--device", "cpu"])
        assert exit_code != 0
        assert len(calls) == 2

    def test_output_conflict_blocks_before_first_subprocess(
        self, pipeline_env, capsys
    ):
        conflict = (
            pipeline_env / "runs" / "train" / "run_x" / "gray_fusion"
        )
        conflict.mkdir(parents=True)
        exit_code, calls = _run(["--run-id", "run_x", "--device", "cpu"])
        assert exit_code == 2
        assert calls == []
        assert str(conflict) in capsys.readouterr().err

    def test_selected_stages_continue_remaining_work(
        self, pipeline_env
    ):
        exit_code, calls = _run(
            ["--run-id", "run_x", "--device", "cpu",
             "--stages", "gray_fusion", "fusion_freeze"]
        )
        assert exit_code == 0
        assert [Path(call[0][1]).name for call in calls] == [
            "train_gray_fusion.py", "train_polar_fusion.py",
        ]


class TestSmokeMode:
    def test_smoke_limits_and_report_marking(self, pipeline_env):
        exit_code, calls = _run(
            ["--run-id", "run_smoke", "--device", "cpu", "--smoke"]
        )
        assert exit_code == 0
        model_a, gray, fusion = (call[0] for call in calls)
        assert model_a[model_a.index("--epochs") + 1] == "1"
        assert model_a[model_a.index("--fraction") + 1] == "0.01"
        for command in (gray, fusion):
            assert command[command.index("--epochs") + 1] == "2"
            assert command[command.index("--limit-batches") + 1] == "3"
        report = json.loads(
            (
                pipeline_env / "runs" / "train" / "run_smoke" / "pipeline_report.json"
            ).read_text(encoding="utf-8")
        )
        assert report["report_version"] == 2
        attempt = report["attempts"][0]
        assert attempt["smoke"] is True
        assert attempt["status"] == "PASSED"
        assert set(attempt["stages"]) == {"model_a", "gray_fusion", "fusion_freeze"}
        for stage in attempt["stages"].values():
            assert stage["exit_code"] == 0
            assert stage["command"]
            assert stage["started_at"] and stage["finished_at"]
        assert report["pipeline_complete"] is True
        assert report["remaining_stages"] == []

    def test_formal_run_records_smoke_false(self, pipeline_env):
        _run(["--run-id", "run_formal", "--device", "cpu"])
        report = json.loads(
            (
                pipeline_env / "runs" / "train" / "run_formal" / "pipeline_report.json"
            ).read_text(encoding="utf-8")
        )
        assert report["attempts"][0]["smoke"] is False

    def test_failure_report_keeps_record(self, pipeline_env):
        _run(["--run-id", "run_fail", "--device", "cpu"], runner_codes=(0, 9))
        report = json.loads(
            (
                pipeline_env / "runs" / "train" / "run_fail" / "pipeline_report.json"
            ).read_text(encoding="utf-8")
        )
        attempt = report["attempts"][0]
        assert attempt["status"] == "FAILED"
        assert attempt["stages"]["gray_fusion"]["exit_code"] == 9
        assert attempt["stages"]["gray_fusion"]["status"] == "FAILED"
        assert "fusion_freeze" not in attempt["stages"]
        assert report["pipeline_complete"] is False
        assert "fusion_freeze" in report["remaining_stages"]


class TestPreflightOnly:
    def test_preflight_only_runs_no_subprocess_and_writes_nothing(
        self, pipeline_env, capsys
    ):
        exit_code, calls = _run(
            ["--run-id", "preflight_only", "--device", "cpu", "--preflight-only"]
        )
        assert exit_code == 0
        assert calls == []
        assert not (pipeline_env / "runs").exists()
        out = capsys.readouterr().out
        assert "preflight" in out.lower()

    def test_preflight_only_reports_conflict(self, pipeline_env, capsys):
        conflict = (
            pipeline_env / "runs" / "train" / "preflight_only" / "model_a"
        )
        conflict.mkdir(parents=True)
        exit_code, calls = _run(
            ["--run-id", "preflight_only", "--device", "cpu", "--preflight-only"]
        )
        assert exit_code == 2
        assert calls == []
        assert str(conflict) in capsys.readouterr().err


class TestFusionOnlyDependency:
    def test_fusion_only_requires_gray_ready(self, pipeline_env, monkeypatch):
        def not_ready(run_id, imgsz, fingerprint):
            raise RuntimeError("gray_fusion/best.pt is missing for this run id")

        monkeypatch.setattr(vp, "_check_gray_ready", not_ready)
        exit_code, calls = _run(
            ["--run-id", "run_x", "--device", "cpu",
             "--stages", "fusion_freeze"]
        )
        assert exit_code == 2
        assert calls == []

    def test_fusion_only_proceeds_when_gray_ready(self, pipeline_env):
        exit_code, calls = _run(
            ["--run-id", "run_x", "--device", "cpu",
             "--stages", "fusion_freeze"]
        )
        assert exit_code == 0
        assert [Path(call[0][1]).name for call in calls] == [
            "train_polar_fusion.py",
        ]


class TestGrayArtifactChecks:
    """The real artifact checks (not the faked ones)."""

    def _write_gray_run(self, root: Path, run_id: str, *, smoke: bool, **overrides):
        run_dir = root / "runs" / "train" / run_id / "gray_fusion"
        run_dir.mkdir(parents=True)
        for name in ("best.pt", "last.pt"):
            (run_dir / name).write_bytes(b"fake")
        config = {
            "entry": "train_gray_fusion",
            "run_id": run_id,
            "architecture": "yolo26n-cls",
            "class_names": V3_CLASS_NAMES,
            "imgsz": 224,
            "smoke": smoke,
            "limit_batches": 3 if smoke else 0,
            "train_samples": 2078,
            "val_samples": 368,
            "selection": "val macro-F1 primary, accuracy tiebreak (full val split)",
            "best_epoch": 2,
            "dataset_fingerprint": "fp_v4",
        }
        config.update(overrides)
        (run_dir / "train_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        metrics = [
            {"epoch": 1, "val_macro_f1": 0.5, "val_acc": 0.9},
            {"epoch": 2, "val_macro_f1": 0.7, "val_acc": 0.6},
            {"epoch": 3, "val_macro_f1": 0.6, "val_acc": 0.95},
        ]
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        return run_dir

    @pytest.fixture
    def probe(self, monkeypatch):
        monkeypatch.setattr(
            vp, "_probe_checkpoint",
            lambda path: {
                "task": "classify",
                "architecture": "yolo26n-cls",
                "names": list(V3_CLASS_NAMES),
                "head_out_features": 4,
                "logits_shape": (2, 4),
                "logits_finite": True,
            },
        )

    def test_valid_gray_run_passes(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_gray_run(tmp_path, "run_x", smoke=False)
        result = vp._check_gray_artifacts("run_x", 224, True, "fp_v4")
        assert result["class_names"] == V3_CLASS_NAMES
        assert result["best_epoch"] == 2
        assert result["dataset_fingerprint"] == "fp_v4"

    def test_missing_files_rejected(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        run_dir = self._write_gray_run(tmp_path, "run_x", smoke=False)
        (run_dir / "best.pt").unlink()
        with pytest.raises(RuntimeError, match="best.pt"):
            vp._check_gray_artifacts("run_x", 224, True, "fp_v4")

    def test_smoke_gray_rejected_for_formal_use(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_gray_run(tmp_path, "run_smoke", smoke=True)
        with pytest.raises(RuntimeError, match="smoke"):
            vp._check_gray_artifacts("run_smoke", 224, True, "fp_v4")

    def test_smoke_gray_allowed_for_smoke_pipeline(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_gray_run(tmp_path, "run_smoke", smoke=True)
        result = vp._check_gray_artifacts("run_smoke", 224, False, "fp_v4")
        assert result["smoke"] is True

    def test_missing_smoke_fields_rejected_for_formal(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        run_dir = self._write_gray_run(tmp_path, "run_x", smoke=False)
        config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        del config["smoke"]
        del config["limit_batches"]
        (run_dir / "train_config.json").write_text(json.dumps(config), encoding="utf-8")
        with pytest.raises(RuntimeError, match="smoke"):
            vp._check_gray_artifacts("run_x", 224, True, "fp_v4")

    def test_wrong_type_limit_batches_rejected_for_formal(
        self, tmp_path, monkeypatch, probe
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        run_dir = self._write_gray_run(
            tmp_path, "run_x", smoke=False, limit_batches=True
        )
        with pytest.raises(RuntimeError, match="limit_batches"):
            vp._check_gray_artifacts("run_x", 224, True, "fp_v4")
        run_dir = self._write_gray_run(
            tmp_path, "run_y", smoke=False, limit_batches="0"
        )
        with pytest.raises(RuntimeError, match="limit_batches"):
            vp._check_gray_artifacts("run_y", 224, True, "fp_v4")

    def test_gray_fingerprint_mismatch_rejected(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_gray_run(tmp_path, "run_x", smoke=False)
        with pytest.raises(RuntimeError, match="fingerprint"):
            vp._check_gray_artifacts("run_x", 224, True, "other_fp")

    def test_gray_fingerprint_missing_rejected(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_gray_run(tmp_path, "run_x", smoke=False, dataset_fingerprint="")
        with pytest.raises(RuntimeError, match="fingerprint"):
            vp._check_gray_artifacts("run_x", 224, True, "fp_v4")

    def test_wrong_imgsz_rejected(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_gray_run(tmp_path, "run_x", smoke=False)
        with pytest.raises(RuntimeError, match="imgsz"):
            vp._check_gray_artifacts("run_x", 320, True, "fp_v4")

    def test_selection_must_follow_macro_f1_priority(
        self, tmp_path, monkeypatch, probe
    ):
        """best_epoch must be the argmax of (macro_f1, accuracy)."""
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        run_dir = self._write_gray_run(tmp_path, "run_x", smoke=False)
        config = json.loads(
            (run_dir / "train_config.json").read_text(encoding="utf-8")
        )
        config["best_epoch"] = 3  # epoch 3 has higher accuracy but lower F1
        (run_dir / "train_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="macro-F1"):
            vp._check_gray_artifacts("run_x", 224, True, "fp_v4")


_UNSET = object()


class TestFusionArtifactChecks:
    """Fusion smoke/formal separation, fingerprint binding and gray-source
    checks (issues 2 and 3)."""

    def _write_fusion_run(
        self,
        root: Path,
        run_id: str,
        *,
        smoke: bool,
        limit_batches,
        fingerprint: str,
        gray_weights: str | None = None,
        checkpoint_limit=_UNSET,
        checkpoint_smoke=_UNSET,
        checkpoint_fingerprint=_UNSET,
    ):
        run_dir = root / "runs" / "train" / run_id / "polar_fusion" / "freeze"
        run_dir.mkdir(parents=True)
        for name in ("best.pt", "last.pt"):
            (run_dir / name).write_bytes(b"fake")
        recorded = gray_weights or str(
            root / "runs" / "train" / run_id / "gray_fusion" / "best.pt"
        )
        config = {
            "run_id": run_id,
            "phase": "freeze",
            "gray_weights": recorded,
            "imgsz": 224,
            "limit_batches": limit_batches,
            "smoke": smoke,
            "dataset_fingerprint": fingerprint,
        }
        (run_dir / "train_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return run_dir, {
            "class_names": list(V3_CLASS_NAMES),
            "imgsz": 224,
            "gray_weights": recorded,
            "architecture": "yolo26n-cls",
            "state_finite": True,
            "logits_shape": (2, 4),
            "logits_finite": True,
            "limit_batches": limit_batches if checkpoint_limit is _UNSET else checkpoint_limit,
            "smoke": smoke if checkpoint_smoke is _UNSET else checkpoint_smoke,
            "dataset_fingerprint": (
                fingerprint
                if checkpoint_fingerprint is _UNSET
                else checkpoint_fingerprint
            ),
        }

    @pytest.fixture
    def probe_fusion(self, monkeypatch):
        holder = {}

        def fake(path, device="cpu"):
            return dict(holder["result"])

        monkeypatch.setattr(vp, "_probe_fusion_checkpoint", fake)
        return holder

    def test_formal_fusion_run_passes(self, tmp_path, monkeypatch, probe_fusion):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=0, fingerprint="fp_v4"
        )
        probe_fusion["result"] = probe
        result = vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")
        assert result["logits_shape"] == (2, 4)

    def test_smoke_fusion_passes_smoke_and_fails_formal(
        self, tmp_path, monkeypatch, probe_fusion
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_smoke", smoke=True, limit_batches=3, fingerprint="fp_v4"
        )
        probe_fusion["result"] = probe
        smoke_result = vp._check_fusion_artifacts("run_smoke", 224, False, "fp_v4")
        assert smoke_result["smoke"] is True
        with pytest.raises(RuntimeError, match="smoke|limit_batches"):
            vp._check_fusion_artifacts("run_smoke", 224, True, "fp_v4")

    def test_missing_smoke_fields_rejected_for_formal(
        self, tmp_path, monkeypatch, probe_fusion
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        run_dir, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=0, fingerprint="fp_v4",
            checkpoint_limit=None, checkpoint_smoke=None,
        )
        config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        del config["smoke"]
        del config["limit_batches"]
        (run_dir / "train_config.json").write_text(json.dumps(config), encoding="utf-8")
        probe["limit_batches"] = None
        probe["smoke"] = None
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="smoke|limit_batches"):
            vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")

    def test_bool_limit_batches_rejected_for_formal(
        self, tmp_path, monkeypatch, probe_fusion
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=True, fingerprint="fp_v4"
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="limit_batches"):
            vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")

    def test_config_and_checkpoint_smoke_mismatch_rejected(
        self, tmp_path, monkeypatch, probe_fusion
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=0, fingerprint="fp_v4",
            checkpoint_limit=3, checkpoint_smoke=True,
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="checkpoint"):
            vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")

    def test_fusion_fingerprint_mismatch_rejected(
        self, tmp_path, monkeypatch, probe_fusion
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=0, fingerprint="other_fp"
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="fingerprint"):
            vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")

    def test_checkpoint_fingerprint_mismatch_rejected(
        self, tmp_path, monkeypatch, probe_fusion
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=0, fingerprint="fp_v4",
            checkpoint_fingerprint="other_fp",
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="fingerprint"):
            vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")

    @pytest.mark.parametrize("bad_limit", [False, 0.0, "0", None, True])
    def test_checkpoint_limit_type_rejected_formal(
        self, tmp_path, monkeypatch, probe_fusion, bad_limit
    ):
        """The formal train_config is legal (0/false); the checkpoint
        metadata itself carries a type-invalid limit_batches. ``False == 0``
        and ``0.0 == 0`` must not pass the checkpoint admission."""
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=0, fingerprint="fp_v4",
            checkpoint_limit=bad_limit, checkpoint_smoke=False,
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="checkpoint metadata"):
            vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")

    @pytest.mark.parametrize("bad_limit", [3.0, True, "3", None])
    def test_checkpoint_limit_type_rejected_smoke(
        self, tmp_path, monkeypatch, probe_fusion, bad_limit
    ):
        """The smoke train_config is legal (3); the checkpoint carries
        3.0/True/"3"/None and must be rejected on its own type."""
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_smoke", smoke=True, limit_batches=3, fingerprint="fp_v4",
            checkpoint_limit=bad_limit, checkpoint_smoke=True,
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="checkpoint metadata"):
            vp._check_fusion_artifacts("run_smoke", 224, False, "fp_v4")

    @pytest.mark.parametrize("bad_smoke", [1, 0, "false", None])
    def test_checkpoint_smoke_non_bool_rejected(
        self, tmp_path, monkeypatch, probe_fusion, bad_smoke
    ):
        """A non-boolean checkpoint smoke marker must be rejected even when
        it is numerically equal to the config value."""
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=0, fingerprint="fp_v4",
            checkpoint_smoke=bad_smoke,
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="checkpoint metadata"):
            vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")

    @pytest.mark.parametrize(
        "config_smoke,config_limit,checkpoint_smoke,checkpoint_limit,require_formal",
        [
            (False, 0, False, 0, True),   # legal formal checkpoint
            (True, 3, True, 3, False),    # legal smoke checkpoint
        ],
    )
    def test_legal_checkpoint_types_pass(
        self, tmp_path, monkeypatch, probe_fusion,
        config_smoke, config_limit, checkpoint_smoke, checkpoint_limit,
        require_formal,
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        run_id = "run_x" if require_formal else "run_smoke"
        _, probe = self._write_fusion_run(
            tmp_path, run_id, smoke=config_smoke, limit_batches=config_limit,
            fingerprint="fp_v4",
            checkpoint_smoke=checkpoint_smoke, checkpoint_limit=checkpoint_limit,
        )
        probe_fusion["result"] = probe
        result = vp._check_fusion_artifacts(
            run_id, 224, require_formal, "fp_v4"
        )
        assert result["smoke"] is config_smoke
        assert result["limit_batches"] == config_limit

    def test_legal_types_numerically_inconsistent_rejected(
        self, tmp_path, monkeypatch, probe_fusion
    ):
        """Both fields are legal types but disagree in value (smoke config
        3 vs checkpoint 5): the consistency check must still reject."""
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_smoke", smoke=True, limit_batches=3, fingerprint="fp_v4",
            checkpoint_limit=5, checkpoint_smoke=True,
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="contradicts"):
            vp._check_fusion_artifacts("run_smoke", 224, False, "fp_v4")

    def test_fusion_config_must_point_at_this_round_gray(
        self, tmp_path, monkeypatch, probe_fusion
    ):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        _, probe = self._write_fusion_run(
            tmp_path, "run_x", smoke=False, limit_batches=0, fingerprint="fp_v4",
            gray_weights="runs/train/other_run/gray_fusion/best.pt",
        )
        probe_fusion["result"] = probe
        with pytest.raises(RuntimeError, match="gray"):
            vp._check_fusion_artifacts("run_x", 224, True, "fp_v4")


class TestModelAArtifactChecks:
    """Model A artifacts must be YOLO26 segment with the single class
    ``target`` on both best and last (issue 1)."""

    def _write_model_a_run(self, root: Path, run_id: str):
        run_dir = root / "runs" / "train" / run_id / "model_a"
        (run_dir / "weights").mkdir(parents=True)
        for name in ("best.pt", "last.pt"):
            (run_dir / "weights" / name).write_bytes(b"fake")
        config = {
            "entry": "train_model_a",
            "run_id": run_id,
            "architecture": "yolo26n-seg",
            "task": "segment",
            "imgsz": 640,
            "smoke": False,
            "data_names": ["target"],
            "train_images_used": 765,
            "data": str(root / "missing-data.yaml"),
        }
        (run_dir / "train_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return run_dir

    def _probe(self, name):
        return {
            "task": "segment",
            "architecture": "yolo26n-seg",
            "names": ["target"] if name == "ok" else ["target", "extra"],
        }

    def test_valid_model_a_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_model_a_run(tmp_path, "run_x")
        monkeypatch.setattr(vp, "_probe_checkpoint", lambda path: self._probe("ok"))
        result = vp._check_model_a_artifacts("run_x", {"imgsz": 640}, True)
        assert result["data_names"] == ["target"]

    def test_wrong_class_names_rejected_on_best_and_last(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_model_a_run(tmp_path, "run_x")
        monkeypatch.setattr(
            vp, "_probe_checkpoint", lambda path: self._probe("bad")
        )
        with pytest.raises(RuntimeError, match="target"):
            vp._check_model_a_artifacts("run_x", {"imgsz": 640}, True)

    def test_detection_task_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vp, "PROJECT_ROOT", tmp_path)
        self._write_model_a_run(tmp_path, "run_x")
        monkeypatch.setattr(
            vp, "_probe_checkpoint",
            lambda path: {"task": "detect", "architecture": "yolo26n", "names": ["target"]},
        )
        with pytest.raises(RuntimeError, match="segment"):
            vp._check_model_a_artifacts("run_x", {"imgsz": 640}, True)


class TestReportHistory:
    """Versioned cumulative report with attempts (issue 4)."""

    def _report(self, root: Path, run_id: str) -> dict:
        return json.loads(
            (root / "runs" / "train" / run_id / "pipeline_report.json").read_text(
                encoding="utf-8"
            )
        )

    def test_stage_continuation_preserves_history(self, pipeline_env):
        exit_code, _ = _run(
            ["--run-id", "run_cont", "--device", "cpu", "--stages", "model_a"]
        )
        assert exit_code == 0
        exit_code, _ = _run(
            ["--run-id", "run_cont", "--device", "cpu",
             "--stages", "gray_fusion", "fusion_freeze"]
        )
        assert exit_code == 0
        report = self._report(pipeline_env, "run_cont")
        assert len(report["attempts"]) == 2
        first, second = report["attempts"]
        assert set(first["stages"]) == {"model_a"}
        assert first["stages"]["model_a"]["status"] == "PASSED"
        assert set(second["stages"]) == {"gray_fusion", "fusion_freeze"}
        assert report["pipeline_complete"] is True
        assert report["remaining_stages"] == []
        assert report["stage_status"] == {
            "model_a": "PASSED", "gray_fusion": "PASSED", "fusion_freeze": "PASSED",
        }

    def test_partial_run_not_reported_as_complete(self, pipeline_env, capsys):
        exit_code, _ = _run(
            ["--run-id", "run_partial", "--device", "cpu",
             "--stages", "fusion_freeze"]
        )
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Full three-stage pipeline complete" not in out
        report = self._report(pipeline_env, "run_partial")
        assert report["pipeline_complete"] is False
        assert report["remaining_stages"] == ["model_a", "gray_fusion"]

    def test_running_status_persisted_before_launch(self, pipeline_env):
        seen = {}

        def inspecting_runner(command, **kwargs):
            report = self._report(pipeline_env, "run_running")
            entry = report["attempts"][0]["stages"]["model_a"]
            seen["status"] = entry["status"]
            seen["command"] = entry["command"]
            return SimpleNamespace(returncode=0)

        vp.run_pipeline(
            vp.parse_args(["--run-id", "run_running", "--device", "cpu",
                           "--stages", "model_a"]),
            runner=inspecting_runner,
        )
        assert seen["status"] == "RUNNING"
        assert seen["command"][1].endswith("train_model_a.py")

    def test_launch_failure_recorded(self, pipeline_env, capsys):
        def broken_runner(command, **kwargs):
            raise OSError("spawn failed")

        exit_code = vp.run_pipeline(
            vp.parse_args(["--run-id", "run_launch", "--device", "cpu"]),
            runner=broken_runner,
        )
        assert exit_code != 0
        report = self._report(pipeline_env, "run_launch")
        entry = report["attempts"][0]["stages"]["model_a"]
        assert entry["status"] == "LAUNCH_FAILED"
        assert "spawn failed" in entry["failure"]
        assert "gray_fusion" not in report["attempts"][0]["stages"]

    def test_interrupt_recorded_and_history_kept(self, pipeline_env):
        def interrupting_runner(command, **kwargs):
            raise KeyboardInterrupt

        exit_code = vp.run_pipeline(
            vp.parse_args(["--run-id", "run_int", "--device", "cpu"]),
            runner=interrupting_runner,
        )
        assert exit_code == 130
        report = self._report(pipeline_env, "run_int")
        attempt = report["attempts"][0]
        assert attempt["status"] == "INTERRUPTED"
        assert attempt["stages"]["model_a"]["status"] == "INTERRUPTED"

    def test_corrupted_report_refused_not_overwritten(self, pipeline_env, capsys):
        report_path = (
            pipeline_env / "runs" / "train" / "run_corrupt" / "pipeline_report.json"
        )
        report_path.parent.mkdir(parents=True)
        report_path.write_text("{not json", encoding="utf-8")
        exit_code, calls = _run(["--run-id", "run_corrupt", "--device", "cpu"])
        assert exit_code == 2
        assert calls == []
        assert report_path.read_text(encoding="utf-8") == "{not json"

    def test_report_run_id_mismatch_refused(self, pipeline_env):
        report_path = (
            pipeline_env / "runs" / "train" / "run_mismatch" / "pipeline_report.json"
        )
        report_path.parent.mkdir(parents=True)
        original = json.dumps(
            {"report_version": 2, "run_id": "someone_else", "attempts": []}
        )
        report_path.write_text(original, encoding="utf-8")
        exit_code, calls = _run(["--run-id", "run_mismatch", "--device", "cpu"])
        assert exit_code == 2
        assert calls == []
        assert report_path.read_text(encoding="utf-8") == original

    def test_v1_report_migrated_losslessly(self, pipeline_env):
        report_path = (
            pipeline_env / "runs" / "train" / "run_v1" / "pipeline_report.json"
        )
        report_path.parent.mkdir(parents=True)
        v1 = {
            "run_id": "run_v1",
            "device": "cpu",
            "seed": 2026,
            "smoke": True,
            "planned_stages": ["model_a"],
            "params": {"model_a": {"epochs": 1, "batch": 2, "imgsz": 640}},
            "preflight": {},
            "stages": {"model_a": {"status": "PASSED", "exit_code": 0,
                                    "command": ["python", "train_model_a.py"]}},
            "status": "PASSED",
        }
        report_path.write_text(json.dumps(v1), encoding="utf-8")
        exit_code, _ = _run(
            ["--run-id", "run_v1", "--device", "cpu",
             "--stages", "gray_fusion", "fusion_freeze"]
        )
        assert exit_code == 0
        report = self._report(pipeline_env, "run_v1")
        legacy = report["attempts"][0]
        assert legacy["migrated_from"] == "v1"
        assert legacy["stages"]["model_a"]["status"] == "PASSED"
        assert legacy["params"] == v1["params"]
        assert report["pipeline_complete"] is True


class TestPreflightAlias:
    def test_preflight_alias_is_read_only(self, pipeline_env, capsys):
        exit_code, calls = _run(
            ["--run-id", "preflight_alias", "--device", "cpu", "--preflight"]
        )
        assert exit_code == 0
        assert calls == []
        assert not (pipeline_env / "runs").exists()
        assert "preflight" in capsys.readouterr().out.lower()

    def test_preflight_alias_with_smoke_params(self, pipeline_env, capsys):
        exit_code, calls = _run(
            ["--run-id", "preflight_smoke", "--device", "cpu",
             "--smoke", "--preflight"]
        )
        assert exit_code == 0
        assert calls == []
        assert not (pipeline_env / "runs").exists()
