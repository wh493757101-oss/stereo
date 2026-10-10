"""Build the local interior/band feature cache (read-only over inputs).

Steps, in order, all refusing on any mismatch:

1. validate the frozen local experiment config;
2. FULL V5 admission through the project's own gate
   (``admit_dataset_for_protocol`` -> ``admit_mask_dataset`` ->
   ``verify_mask_sources``) with the fingerprint taken from the frozen
   config — before any feature is computed;
3. verify the frozen GLOBAL cache (the P0 source) through the original
   chain — G, s_valid and class_id are copied byte-identically from it;
4. build the local cache (partition interior/band, 8 S statistics per
   part, per-region C controls) into a FRESH output directory;
5. self-verify and write a build report (no check is recorded as passed
   unless it actually ran).

Nothing is trained and no prior version is overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from core.material_mask_dataset import (  # noqa: E402
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    admit_dataset_for_protocol,
)
from core.polar_local_features import (  # noqa: E402
    LOCAL_FEATURE_DEFINITION_VERSION,
    LOCAL_REGION_RULE,
    LOCAL_TOTAL_DIM,
    build_local_cache_arrays,
)

EXIT_OK = 0
EXIT_REFUSED = 1


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the local interior/band feature cache (inputs "
                    "stay read-only).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory; must not exist")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    try:
        import yaml

        if output_dir.exists():
            raise ValueError(
                f"output directory {output_dir} already exists")
        config = yaml.safe_load(
            Path(args.config).read_text(encoding="utf-8"))
        if config.get("experiment") != \
                "material_polar_local_interior_band_v1":
            raise ValueError(
                f"config experiment {config.get('experiment')!r} is not "
                "material_polar_local_interior_band_v1")

        # ---- full V5 admission BEFORE any feature computation ----
        data_root = _PROJECT_ROOT / config["data"]["root"]
        expected_fingerprint = config["data"]["fingerprint"]
        expected_parent = config["data"].get("parent_v4_fingerprint") or ""
        v5_report = admit_dataset_for_protocol(
            data_root,
            pooling_mode="instance_mask",
            expected_fingerprint=expected_fingerprint,
            expected_parent_fingerprint=expected_parent,
            expected_mask_schema=MASK_SCHEMA_VERSION,
            expected_mask_resize_version=MASK_RESIZE_VERSION)
        if v5_report["audit_fingerprint"] != expected_fingerprint:
            raise ValueError(
                "V5 admission fingerprint does not match the expected "
                "frozen fingerprint")

        # ---- the frozen global cache (G source; the P0 chain's input) --
        import scripts.verify_p0_baseline as baseline

        protocol_path = _PROJECT_ROOT / \
            config["protocol_source"]["path"]
        protocol = baseline.load_protocol(protocol_path)
        if protocol["data"]["fingerprint"] != expected_fingerprint:
            raise ValueError(
                "protocol fingerprint does not match the config's data "
                "fingerprint")
        global_dir = _PROJECT_ROOT / config["features_source"]["path"]
        _binding, _definition, manifest, arrays, _authority = \
            baseline.verify_feature_cache(global_dir, protocol)
        definition_global = load_json(
            global_dir / "feature_definition.json")
        if definition_global.get("data_fingerprint") != \
                expected_fingerprint:
            raise ValueError(
                "global feature definition fingerprint does not match the "
                "config's data fingerprint")

        # ---- build the local cache into a fresh directory ----
        features_dir = output_dir / "local_features"
        report = build_local_cache_arrays(
            source_arrays=arrays,
            samples=manifest["samples"],
            data_root=data_root,
            protocol=protocol,
            output_dir=features_dir)

        # G byte-equality is part of the report evidence
        g_local = np.load(features_dir / "feature_cache.npz")["g_features"]
        g_equal = bool(np.array_equal(
            g_local, arrays["g_features"].astype(np.float32)))
        if not g_equal:
            raise ValueError("the local cache G is not byte-identical to "
                             "the global cache G")
        build_report = {
            "experiment": "material_polar_local_interior_band_v1",
            "region_rule": LOCAL_REGION_RULE,
            "feature_definition_version": LOCAL_FEATURE_DEFINITION_VERSION,
            "input_dim": LOCAL_TOTAL_DIM,
            "local_cache": report,
            "v5_admission": {
                "gate": "core.material_mask_dataset."
                        "admit_dataset_for_protocol (instance_mask)",
                "data_root": str(data_root.resolve()),
                "expected_fingerprint": expected_fingerprint,
                "actual_fingerprint": v5_report["audit_fingerprint"],
                "v4_parent_expected": expected_parent,
                "v4_parent_actual": v5_report["v4_parent_fingerprint"],
                "verified_files": v5_report["verified_files"],
                "mask_content_sha256": v5_report["mask_content_sha256"],
                "label_source_sha256": v5_report["label_source_sha256"],
                "sample_mapping_sha256":
                    v5_report["sample_mapping_sha256"],
                "source_label_files": v5_report["source_label_files"],
                "splits": v5_report["splits"],
            },
            "global_cache_source": {
                "path": str(global_dir),
                "binding_files": {name: record["sha256"]
                                  for name, record
                                  in _binding["files"].items()},
            },
            "protocol_sha256": protocol["protocol_sha256"],
            "g_byte_identical": g_equal,
            "checks_not_performed": {},
            "note": ("the local partition does not exclude any pixel; "
                     "s_valid is identical to the global definition by "
                     "construction and re-checked row by row"),
        }
        write_json(output_dir / "build_report.json", build_report)
        print(f"local cache built and verified: {features_dir} "
              f"({report['sample_count']} samples, input_dim "
              f"{LOCAL_TOTAL_DIM})", file=sys.stderr)
        return EXIT_OK
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
