"""Regression guards: existing entries must keep working unchanged."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_polar_fusion_structure_check_still_passes():
    from scripts.train_polar_fusion import structure_check

    report = structure_check(4, 224)
    assert report["forced_gate_verified"] is True


def test_polar_fusion_imports_unchanged():
    from models.polar_fusion import FusionClsDataset, PolarFusionModel  # noqa: F401
    from core.fusion_dataset import load_fusion_sample  # noqa: F401


def test_material_fusion_modules_import_without_ultralytics():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import models.material_fusion\n"
            "import core.material_fusion_dataset\n"
            "import models.cmx.dual_segformer\n"
            "assert 'ultralytics' not in sys.modules, 'ultralytics eagerly imported'\n",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_retired_entries_still_refuse():
    result = subprocess.run(
        [sys.executable, "scripts/train_models.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0
