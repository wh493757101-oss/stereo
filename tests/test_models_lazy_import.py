"""Regression tests: importing model submodules must not load Ultralytics.

Ultralytics globally patches cv2.imread at import time (grayscale reads
return (H, W, 1)). ``models.export_tensorrt`` imports Ultralytics, so it
must only be imported when ``export_model`` is actually requested.
"""

import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_snippet(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_import_models_does_not_load_ultralytics():
    result = _run_snippet(
        "import sys\n"
        "import models\n"
        "import models.classification\n"
        "assert 'models.export_tensorrt' not in sys.modules, 'export_tensorrt eagerly imported'\n"
        "assert 'ultralytics' not in sys.modules, 'ultralytics eagerly imported'\n"
    )
    assert result.returncode == 0, result.stderr


def test_from_models_import_export_model_is_lazy_but_works():
    result = _run_snippet(
        "import sys\n"
        "from models import export_model\n"
        "assert callable(export_model)\n"
        "assert 'ultralytics' in sys.modules, 'export_model did not resolve lazily'\n"
    )
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_still_raises_attribute_error():
    result = _run_snippet(
        "import models\n"
        "try:\n"
        "    models.no_such_name\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected AttributeError')\n"
    )
    assert result.returncode == 0, result.stderr
