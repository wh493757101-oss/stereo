"""Upstream license retention for the ported CMX code (issue 7).

The vendored encoder/FRM/FFM code must ship the full upstream MIT license
text, byte-identical to the pinned commit's LICENSE, plus a provenance
note that separates upstream code from local modifications.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CMX_DIR = Path(__file__).resolve().parents[2] / "models" / "cmx"
UPSTREAM_LICENSE_SHA256 = (
    "a3fb69f7d2d7ab44ce80bba7f8c3a61f3c8a2775a2baac4bef815a60c4d8ba5e"
)
UPSTREAM_COMMIT = "e251d860aebc2f583a6c4919877e6bebe7f1aff3"


def test_license_file_matches_upstream_digest():
    license_path = CMX_DIR / "LICENSE"
    assert license_path.is_file(), "models/cmx/LICENSE must ship the upstream license"
    digest = hashlib.sha256(license_path.read_bytes()).hexdigest()
    assert digest == UPSTREAM_LICENSE_SHA256


def test_license_contains_full_mit_text():
    text = (CMX_DIR / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Copyright (c) 2022 Huayao Liu" in text
    assert "Permission is hereby granted, free of charge" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in text


def test_provenance_readme_records_source_commit_and_modifications():
    readme = (CMX_DIR / "README.md").read_text(encoding="utf-8")
    assert "huaaaliu/RGBX_Semantic_Segmentation" in readme
    assert UPSTREAM_COMMIT in readme
    assert "MIT" in readme
    # Local modifications must be separated from upstream code.
    assert "modification" in readme.lower()
    assert "timm" in readme
