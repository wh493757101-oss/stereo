"""Retired-entry and archive boundary tests.

The old four-stage entries (scripts/train_all_models.py, train_models.py,
train_seg.py) must only print a retirement notice and exit non-zero; they
must never start training or silently forward to the new three-stage
pipeline. The archived implementations live under
scripts/legacy_four_stage/ and stay explicitly runnable.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_all_models as retired_launcher
import scripts.train_models as retired_stages
import scripts.train_seg as retired_shim

ACTIVE_SOURCES = (
    "training_common.py",
    "train_pipeline.py",
    "train_model_a.py",
    "train_gray_fusion.py",
    "train_polar_fusion.py",
    "verify_yolo26_fusion.py",
)
FORBIDDEN_IMPORTS = (
    "import scripts.train_models",
    "import scripts.train_all_models",
    "import scripts.train_seg",
    "from scripts.train_models",
    "from scripts.train_all_models",
    "from scripts.train_seg",
)


class TestRetiredEntries:
    @pytest.mark.parametrize(
        "module,argv",
        [
            (retired_stages, ["--stage", "b-gray", "--run-id", "run_x"]),
            (retired_launcher, ["--run-id", "run_x", "--stages", "a"]),
            (retired_shim, ["--stage", "a", "--data", "d.yaml", "--run-id", "run_x"]),
        ],
    )
    def test_retired_entry_prompts_and_refuses(self, module, argv, monkeypatch, capsys):
        spawned = []
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **kw: spawned.append((a, kw))
        )
        exit_code = module.main(argv)
        assert exit_code == 2
        err = capsys.readouterr().err
        assert "legacy_four_stage" in err
        assert "train_pipeline" in err
        assert spawned == []

    def test_retired_entries_do_not_import_training_logic(self):
        """The retirement stubs must stay thin: no training imports."""
        for name in ("train_models.py", "train_all_models.py", "train_seg.py"):
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            assert "import scripts.training_common" not in source
            assert "from scripts.training_common" not in source
            assert "ultralytics" not in source


class TestArchiveBoundary:
    def test_archived_entries_import_and_expose_expected_names(self):
        import scripts.legacy_four_stage.train_all_models as legacy_launcher
        import scripts.legacy_four_stage.train_models as legacy_stages
        import scripts.legacy_four_stage.train_seg as legacy_shim

        assert tuple(legacy_launcher.ALL_STAGES) == (
            "baseline", "a", "b-gray", "b-polar",
        )
        assert legacy_launcher.TRAIN_SCRIPT == (
            ROOT / "scripts" / "legacy_four_stage" / "train_models.py"
        )
        assert legacy_stages.PROJECT_ROOT == ROOT
        assert legacy_shim.PROJECT_ROOT == ROOT
        # Common helpers are reused, not duplicated.
        import scripts.training_common as common

        assert legacy_stages.validate_run_id is common.validate_run_id
        assert legacy_stages.resolve_device is common.resolve_device
        assert legacy_stages.InvalidRunIdError is common.InvalidRunIdError
        assert legacy_stages.DeviceUnavailableError is common.DeviceUnavailableError

    def test_archive_readme_documents_mapping(self):
        readme = (ROOT / "scripts" / "legacy_four_stage" / "README.md").read_text(
            encoding="utf-8"
        )
        for token in ("baseline", "b-gray", "b-polar", "train_pipeline", "mainline"):
            assert token in readme

    def test_active_code_does_not_depend_on_retired_entries(self):
        for name in ACTIVE_SOURCES:
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_IMPORTS:
                assert forbidden not in source, f"{name} imports retired entry"
