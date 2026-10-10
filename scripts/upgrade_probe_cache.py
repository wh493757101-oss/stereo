"""Upgrade the previous round's feature cache to the binding contract.

The OLD evidence directory is strictly read-only. This entry:

1. verifies the old ``artifact_manifest.json`` covers the three original
   cache files with matching digests;
2. verifies the old protocol digest against the current data admission
   (full ``admit_mask_dataset`` gate) and the sample identity set;
3. copies the three files BYTE-FOR-BYTE into the new evidence directory
   (``corrected_features/``);
4. writes a new binding record and an upgrade report;
5. verifies everything with the repaired unified admission.

A byte-level before/after comparison proves the upgrade never re-wrapped
a changed cache as "the original trusted cache". If the old evidence
cannot be verified, the upgrade stops with a non-zero exit — no silent
re-extraction, no overwrite, no rebuild of historical artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_mask_dataset import admit_mask_dataset  # noqa: E402
from core.material_probe import (  # noqa: E402
    BOUND_CACHE_FILES,
    build_cache_binding,
    file_sha256,
    verify_cache_arrays,
    verify_cache_binding,
    verify_cache_manifest,
    authoritative_sample_index,
    FEATURE_DEFINITION_VERSION,
)

EXIT_OK = 0
EXIT_REFUSED = 1


def write_json(path: Path, payload) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_old_manifest(old_dir: Path) -> dict:
    """The old artifact manifest must cover the three cache files."""
    manifest_path = old_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"old artifact manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    files = manifest.get("files", {})
    covered = {}
    for name in BOUND_CACHE_FILES:
        relative = f"features/{name}"
        record = files.get(relative)
        if record is None:
            raise ValueError(
                f"old artifact manifest does not cover {relative}; the old "
                "evidence cannot be verified")
        actual = old_dir / relative
        if not actual.is_file():
            raise FileNotFoundError(f"old cache file missing: {actual}")
        digest = file_sha256(actual)
        if digest != record.get("sha256"):
            raise ValueError(
                f"{relative} sha256 {digest} != the old manifest record "
                f"{record.get('sha256')}; the old evidence changed since "
                "the previous round")
        covered[name] = {
            "old_manifest_relative": relative,
            "old_manifest_sha256": record.get("sha256"),
            "verified_sha256": digest,
            "bytes": actual.stat().st_size,
        }
    return covered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade a finished feature cache to the binding "
                    "contract (old evidence stays read-only).")
    parser.add_argument("--old-run-dir", required=True,
                        help="the previous round's evidence directory "
                        "(read-only)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True,
                        help="new directory; corrected_features/ is created "
                        "inside")
    args = parser.parse_args(argv)

    if Path(args.output_dir).exists():
        print(
            f"refused: output directory {args.output_dir} already exists",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    try:
        old_dir = Path(args.old_run_dir)
        if not old_dir.is_dir():
            raise FileNotFoundError(f"old run directory missing: {old_dir}")
        old_features = old_dir / "features"
        if not old_features.is_dir():
            raise FileNotFoundError(
                f"old features directory missing: {old_features}")

        # ---- step 1: the old manifest must cover and match the files ----
        verified = verify_old_manifest(old_dir)
        print("old manifest coverage verified for: "
              f"{sorted(verified)}", file=sys.stderr)

        # ---- step 2: protocol + current data admission + identity ----
        protocol_path = old_dir / "protocol" / "protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(f"old protocol missing: {protocol_path}")
        protocol = load_json(protocol_path)
        from core.existing_data_protocol import protocol_digest

        if protocol_digest(protocol) != protocol["protocol_sha256"]:
            raise ValueError("the old protocol fails its own digest check")
        # the upgrade binds the data identity from the config's data
        # section; the full experiment-config validation belongs to the
        # protocol/prepare entries, not to a byte-level cache upgrade
        import yaml

        config = yaml.safe_load(
            Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(
                config.get("data"), dict):
            raise ValueError(f"config {args.config} has no data section")
        data_config = config["data"]
        if protocol["data"]["fingerprint"] != data_config["fingerprint"]:
            raise ValueError(
                "the old protocol fingerprint does not match the config")
        import scripts.upgrade_probe_cache as _self

        admission = _self.admit_mask_dataset(
            PROJECT_ROOT / data_config["root"],
            expected_fingerprint=data_config["fingerprint"],
            expected_parent_fingerprint=data_config["parent_v4_fingerprint"],
        )
        print("current V5 admission passed "
              f"({admission['verified_files']} files re-verified)",
              file=sys.stderr)

        definition = load_json(old_features / "feature_definition.json")
        if definition.get("feature_definition_version") != \
                FEATURE_DEFINITION_VERSION:
            raise ValueError(
                "old feature definition version mismatch: "
                f"{definition.get('feature_definition_version')!r}")
        if definition.get("data_fingerprint") != \
                protocol["data"]["fingerprint"]:
            raise ValueError(
                "old cache data fingerprint does not match the old protocol")
        authority = authoritative_sample_index(protocol)
        manifest = load_json(old_features / "feature_manifest.json")
        samples = verify_cache_manifest(manifest, authority, protocol)
        cache = dict(np_load(old_features / "feature_cache.npz"))
        verify_cache_arrays(cache, samples, definition, protocol)
        print("old cache identity/arrays verified against the protocol",
              file=sys.stderr)

        # ---- step 3: byte-for-byte copy into the new directory ----
        output_dir = Path(args.output_dir)
        corrected = output_dir / "corrected_features"
        corrected.mkdir(parents=True)
        for name in BOUND_CACHE_FILES:
            source = old_features / name
            target = corrected / name
            shutil.copyfile(source, target)
            if file_sha256(target) != verified[name]["verified_sha256"]:
                raise ValueError(
                    f"{name}: the copied bytes differ from the verified "
                    "original; refusing to continue")
        print("byte-for-byte copy verified for: "
              f"{sorted(BOUND_CACHE_FILES)}", file=sys.stderr)

        # ---- step 4: the new binding record + upgrade report ----
        binding = build_cache_binding(corrected, protocol)
        byte_equality = {
            name: {
                "original_sha256": verified[name]["verified_sha256"],
                "copied_sha256": binding["files"][name]["sha256"],
                "identical": (verified[name]["verified_sha256"]
                              == binding["files"][name]["sha256"]),
            }
            for name in BOUND_CACHE_FILES
        }
        if not all(record["identical"] for record in byte_equality.values()):
            raise ValueError(
                "post-copy byte equality failed; the upgrade stops without "
                "publishing a binding")
        report = {
            "upgrade": "existing_data_probe_cache_binding_v2",
            "old_run_dir": str(old_dir),
            "old_manifest_coverage": verified,
            "data_admission": {
                "gate": "core.material_mask_dataset.admit_mask_dataset",
                "verified_files": admission["verified_files"],
                "audit_fingerprint": admission["audit_fingerprint"],
                "v4_parent_fingerprint": admission["v4_parent_fingerprint"],
            },
            "protocol_sha256": protocol["protocol_sha256"],
            "byte_equality": byte_equality,
            "binding_files": {name: record["sha256"] for name, record
                              in binding["files"].items()},
            "note": (
                "the three cache files were copied byte-for-byte and then "
                "re-verified by the repaired unified admission; no feature "
                "was re-extracted and no G/C/S value changed"),
        }
        write_json(output_dir / "upgrade_report.json", report)

        # ---- step 5: the repaired unified admission must accept it ----
        verify_cache_binding(corrected, protocol)
        definition_new = load_json(corrected / "feature_definition.json")
        manifest_new = load_json(corrected / "feature_manifest.json")
        samples_new = verify_cache_manifest(manifest_new, authority, protocol)
        cache_new = dict(np_load(corrected / "feature_cache.npz"))
        verify_cache_arrays(cache_new, samples_new, definition_new, protocol)
        write_json(output_dir / "admission.json", {
            "verified": True,
            "admission_chain": [
                "old artifact manifest coverage + digests",
                "old protocol digest",
                "current V5 full admission (admit_mask_dataset)",
                "authoritative sample index (per-fold roles)",
                "cache manifest row-by-row identity",
                "cache arrays shapes/dtypes/finite/missing rules",
                "byte equality original vs copied",
                "verify_cache_binding against the new binding",
            ],
            "sample_count": len(samples_new),
        })
        print(f"upgrade complete and admitted: {corrected}", file=sys.stderr)
        return EXIT_OK
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


def np_load(path: Path):
    import numpy as np

    return np.load(path)


if __name__ == "__main__":
    raise SystemExit(main())
