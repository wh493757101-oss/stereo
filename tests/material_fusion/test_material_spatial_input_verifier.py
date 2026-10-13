"""material_spatial_input_v1 verifier tests: the real synthetic
build→full chain, per-case fresh tamper refusals (definition, ledger,
plan, native, binding) with fresh verification evidence per case, the
independent-support saved-grid refusals and the late-deadline PARTIAL.
No model training."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from scripts import prepare_material_spatial_input as prep
from scripts import verify_material_spatial_input as verify

_ROOT = prep._PROJECT_ROOT

from .test_material_spatial_input_entry import (  # noqa: F401
    spatial_world,
    spatial_patches,
    _run_prepare,
    _dump_yaml,
)

_DEADLINE = "2030-01-01T00:00:00+00:00"


@pytest.fixture(scope="module")
def smoke_cache(spatial_world, spatial_patches, module_scratch):
    """A small smoke cache built once through the REAL prepare entry."""
    out = module_scratch / "spatial_root" / "verify_fixture"
    code = _run_prepare(spatial_world, out, "smoke", _DEADLINE,
                        max_frames=4)
    assert code == prep.EXIT_OK
    return {"cache_dir": out, "out": out}


def _binding_sha(cache_dir: Path) -> str:
    return hashlib.sha256(
        (cache_dir / "cache" / "binding.json").read_bytes()).hexdigest()


def _run_verify(spatial_world, cache_dir, module_scratch,
                extra_patches=(), mode="smoke",
                budget_deadline=_DEADLINE):
    """The verifier against the run root: --output-dir is the run
    root's fresh verification/ subdir and the expected binding digest
    comes from the builder status (as the wrapper passes it)."""
    binding_sha = _binding_sha(cache_dir)
    patches = [
        mock.patch.object(verify, "PINNED_CONFIG_REL",
                          spatial_world["cfg_path"].relative_to(
                              _ROOT).as_posix()),
        mock.patch.object(verify, "OUTPUT_ROOT",
                          module_scratch.relative_to(_ROOT).as_posix()),
    ]
    patches.extend(extra_patches)
    for context in patches:
        context.start()
    try:
        return verify.main([
            "--config", str(spatial_world["cfg_path"]),
            "--mode", mode,
            "--cache-dir", str(cache_dir),
            "--output-dir", str(cache_dir / "verification"),
            "--deadline-utc", budget_deadline,
            "--expected-binding-sha256", binding_sha,
        ])
    finally:
        for context in patches:
            context.stop()


def _fresh_cache(spatial_world, module_scratch, name, mode="smoke",
                 max_frames=3):
    """A fresh cache built through the REAL prepare entry: the run_id
    comes naturally from the production run, never hand-edited."""
    out = module_scratch / "spatial_root" / "caches" / name
    if mode == "smoke":
        code = _run_prepare(spatial_world, out, mode, _DEADLINE,
                            max_frames=max_frames)
    else:
        code = _run_prepare(spatial_world, out, mode, _DEADLINE)
    assert code == prep.EXIT_OK, name
    return out


def _window_bounds(cache_dir: Path) -> tuple:
    ledger = [json.loads(line) for line in
              (cache_dir / "cache" / "samples.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    row = ledger[0]
    x0, y0 = row["offset"]
    return y0, x0, row["h2"], row["w2"]


def _pick_zero_cell(planes: np.ndarray, cache_dir: Path) -> tuple:
    """A true-zero coverage cell INSIDE the letterbox window (never the
    padding, which the plain contract check would reject first)."""
    y0, x0, h2, w2 = _window_bounds(cache_dir)
    coverage = planes[6]
    zeros = np.argwhere(coverage.view(np.uint32) == 0)
    for r, c in zeros:
        if y0 <= r < y0 + h2 and x0 <= c < x0 + w2:
            return (int(r), int(c))
    raise AssertionError("no in-window zero cell in the fixture")


def _resync(cache_dir: Path) -> None:
    """Recompute every binding digest AND the builder status's binding
    digest, so ONLY the target contract / independent rebuild can catch
    a content tamper."""
    binding_path = cache_dir / "cache" / "binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    for record in binding["artifacts"]:
        path = cache_dir / record["path"]
        if path.is_file():
            record["bytes"] = path.stat().st_size
            record["sha256"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
            if record.get("npz_keys"):
                with np.load(path) as payload:
                    record["npz_keys"] = [
                        {"key": key, "dtype": str(payload[key].dtype),
                         "shape": list(payload[key].shape)}
                        for key in payload.files]
    binding_path.write_text(json.dumps(binding, indent=1),
                            encoding="utf-8")
    builder_status = json.loads(
        (cache_dir / "builder" / "run_status.json").read_text(
            encoding="utf-8"))
    builder_status["cache_binding_sha256"] = hashlib.sha256(
        binding_path.read_bytes()).hexdigest()
    (cache_dir / "builder" / "run_status.json").write_text(
        json.dumps(builder_status, indent=1), encoding="utf-8")


def _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                          expect_fragment, mode="smoke"):
    """Run the verifier on a fresh verification dir and assert the NEW
    status exists, belongs to this case and refuses for the expected
    reason — never a stale status from a previous case."""
    verification = cache_dir / "verification"
    assert not verification.exists(), \
        "the verification dir must not exist before the case runs"
    code = _run_verify(spatial_world, cache_dir,
                       cache_dir.parent.parent, mode=mode)
    assert code == verify.EXIT_REFUSED
    assert verification.exists(), "no fresh status was produced"
    status = json.loads((verification / "status.json").read_text(
        encoding="utf-8"))
    assert status["overall_status"] == "FAILED"
    assert status["run_id"] == cache_dir.name
    assert status["deadline_utc"] == _DEADLINE
    assert expect_fragment.lower() in status["failure"].lower(), \
        (expect_fragment, status["failure"])
    return status


class TestRealFullChain:
    def test_production_build_then_full_verify_succeeds(
            self, spatial_world, spatial_patches, module_scratch):
        """The REAL production chain: prepare.main(mode=build) over the
        synthetic dev pool, then verify.main(mode=full) — the complete
        cache passes under its own approved counts."""
        run_root = module_scratch / "spatial_root" / "full_chain" /             "synth_build"
        code = _run_prepare(spatial_world, run_root, "build", _DEADLINE)
        assert code == prep.EXIT_OK
        builder_status = json.loads(
            (run_root / "builder" / "run_status.json").read_text(
                encoding="utf-8"))
        assert builder_status["overall_status"] == "BUILT_UNVERIFIED"
        assert builder_status["cache_complete"] is True
        definition = json.loads((run_root / "cache" / "definition.json")
                                .read_text(encoding="utf-8"))
        pool = spatial_world["cfg"]["dev_pool"]
        assert definition["mode"] == "build"
        assert definition["kind"] == "material_spatial_input_full_v1"
        assert definition["complete"] is True
        assert definition["built"]["frames"] == pool["nonempty_frames"]
        assert definition["built"]["instances"] == pool["instances"]
        code = _run_verify(spatial_world, run_root,
                           run_root.parent.parent.parent, mode="full")
        assert code == verify.EXIT_OK
        status = json.loads((run_root / "verification" / "status.json")
                            .read_text(encoding="utf-8"))
        assert status["overall_status"] == "VERIFIED"
        outcome = status["outcome"]
        assert outcome["rebuilt_frames"] == pool["nonempty_frames"]
        assert outcome["rebuilt_instances"] == pool["instances"]
        # every non-empty frame's shard keys were checked against the
        # authoritative set and every plan against the frozen rule
        assert outcome["permutation"]["plans"] == \
            2 * pool["instances"]

    @pytest.mark.parametrize("field,value", [
        ("kind", "material_spatial_input_smoke_v1"),
        ("mode", "smoke"),
        ("complete", False),
    ])
    def test_full_cache_field_inconsistency_is_refused(
            self, spatial_world, spatial_patches, module_scratch,
            field, value):
        run_root = _fresh_cache(spatial_world, module_scratch,
                                f"full_bad_{field}", mode="build")
        definition_path = run_root / "cache" / "definition.json"
        definition = json.loads(definition_path.read_text(
            encoding="utf-8"))
        definition[field] = value
        definition_path.write_text(json.dumps(definition, indent=1),
                                   encoding="utf-8")
        _resync(run_root)
        _assert_fresh_refusal(spatial_world, module_scratch, run_root,
                              f"definition.{field}", mode="full")

    def test_full_cache_wrong_built_counts_are_refused(
            self, spatial_world, spatial_patches, module_scratch):
        run_root = _fresh_cache(spatial_world, module_scratch,
                                "full_bad_counts", mode="build")
        definition_path = run_root / "cache" / "definition.json"
        definition = json.loads(definition_path.read_text(
            encoding="utf-8"))
        definition["built"]["frames"] = \
            definition["built"]["frames"] - 1
        definition_path.write_text(json.dumps(definition, indent=1),
                                   encoding="utf-8")
        _resync(run_root)
        _assert_fresh_refusal(spatial_world, module_scratch, run_root,
                              "definition.built", mode="full")

    @pytest.mark.parametrize("mode,verify_mode", [("smoke", "smoke"),
                                                  ("build", "full")])
    def test_real_chain_includes_an_authoritative_empty_frame(
            self, spatial_world, spatial_patches, module_scratch,
            mode, verify_mode):
        """Add an empty source, not a fabricated instance or ledger row."""
        world = dict(spatial_world)
        cfg = copy.deepcopy(world["cfg"])
        owned = module_scratch / f"empty_world_{mode}"
        owned.mkdir()
        pair_path = Path(cfg["frozen_sources"]["pair_manifest"]["path"])
        with pair_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            pairs = list(reader)
        stem = "Synthetic_empty_000"
        row = dict(pairs[0])
        row["output_stem"] = stem
        for column in ("left_path", "right_path"):
            path = owned / f"{column}.png"
            path.write_bytes((_ROOT / row[column]).read_bytes())
            row[column] = path.relative_to(_ROOT).as_posix()
        new_pair = owned / "pairs.csv"
        with new_pair.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([*pairs, row])
        cfg["frozen_sources"]["pair_manifest"]["path"] = str(new_pair)
        cfg["dev_pool"]["frames"] += 1
        cfg["dev_pool"]["empty_frames"] = 1
        cfg["dev_pool"]["empty_frame_names"] = [stem]
        config_path = _dump_yaml(owned / "config.yaml", cfg)
        world.update(cfg=cfg, cfg_path=config_path)
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        with ExitStack() as stack:
            for module in (prep, verify):
                stack.enter_context(mock.patch.object(
                    module, "PINNED_CONFIG_REL",
                    config_path.relative_to(_ROOT).as_posix()))
                stack.enter_context(mock.patch.object(
                    module, "PINNED_CONFIG_SHA256", config_sha))
            root = _fresh_cache(world, module_scratch,
                                f"empty_chain_{mode}", mode=mode,
                                max_frames=4)
            assert _run_verify(world, root, root.parent.parent,
                               mode=verify_mode) == verify.EXIT_OK
            frames = [json.loads(line) for line in
                      (root / "cache" / "frames.jsonl").read_text(
                          encoding="utf-8").splitlines()]
            empty = next(item for item in frames if item["frame"] == stem)
            assert empty["instances"] == 0
            assert not any(stem in p.name for p in
                           (root / "cache" / "native").iterdir())

    @pytest.mark.parametrize("field", ["protocol_sha256", "v5_fingerprint",
                                       "recipe_source", "native_arrays"])
    def test_definition_frozen_source_values_are_verified(
            self, spatial_world, spatial_patches, module_scratch, field):
        root = _fresh_cache(spatial_world, module_scratch,
                            f"definition_source_{field}")
        path = root / "cache" / "definition.json"
        definition = json.loads(path.read_text(encoding="utf-8"))
        if field in ("protocol_sha256", "v5_fingerprint"):
            definition["source_identity"][field] = "0" * 64
        elif field == "native_arrays":
            definition[field]["region"] = "float32"
        else:
            definition[field] = "another_recipe.yaml"
        path.write_text(json.dumps(definition), encoding="utf-8")
        _resync(root)
        _assert_fresh_refusal(spatial_world, module_scratch, root, field)


class TestVerifierAccepts:
    def test_verifier_passes_on_the_untampered_cache(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = TestVerifierRefusals._copy_cache(
            self, smoke_cache, module_scratch, "verify_accept")
        out = cache_dir / "verification"
        code = _run_verify(spatial_world, cache_dir,
                           cache_dir.parent.parent)
        assert code == verify.EXIT_OK
        status = json.loads((out / "status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "VERIFIED"
        assert status["protection"]["identical"] is True
        assert status["cache_binding_sha256"] == _binding_sha(cache_dir)
        outcome = status["outcome"]
        ledger = [json.loads(line) for line in
                  (cache_dir / "cache" / "samples.jsonl")
                  .read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        assert outcome["rebuilt_instances"] == len(ledger)
        assert outcome["native_instances"] == len(ledger)
        assert outcome["permutation"]["plans"] == 2 * len(ledger)
        grid = outcome["grid"]
        assert grid["total_elements"] == 9 * 128 * 128 * len(ledger)
        report = json.loads((out / "report.json").read_text(
            encoding="utf-8"))
        assert report["grid_tolerance"] == {"atol": 1e-6, "rtol": 0.0}
        assert (out / "source_manifest.json").is_file()

    def test_inventory_and_ledgers_hold(self, spatial_world,
                                        spatial_patches, smoke_cache):
        cache_dir = smoke_cache["cache_dir"]
        binding = json.loads((cache_dir / "cache" / "binding.json")
                             .read_text(encoding="utf-8"))
        assert binding["cache_schema"] == \
            "material_spatial_input_cache_v1"
        assert binding["implementation_revision"] == "review_fix_v1"
        assert binding["artifact_count"] == len(binding["artifacts"])
        samples = [json.loads(line) for line in
                   (cache_dir / "cache" / "samples.jsonl").read_text(
                       encoding="utf-8").splitlines() if line.strip()]
        assert len({row["sample_name"] for row in samples}) == \
            len(samples) >= 1
        for row in samples:
            assert row["w2"] <= 128 and row["h2"] <= 128
            assert set(row["fold_roles"]) == {"fold0", "fold1", "fold2"}
        definition = json.loads((cache_dir / "cache" / "definition.json")
                                .read_text(encoding="utf-8"))
        assert definition["grid_schema"] == "area_parts_letterbox128_v1"
        assert definition["permutation_schema"] == \
            "coverage_matched_parts_pcg64_v1"
        assert definition["kind"] == "material_spatial_input_smoke_v1"
        assert definition["complete"] is False


class TestVerifierRefusals:
    def _copy_cache(self, smoke_cache, module_scratch, name):
        import shutil

        # the copy keeps the run-root NAME: the verifier refuses
        # cross-run evidence stitching on a renamed run
        target = module_scratch / "copies" / name / \
            smoke_cache["cache_dir"].name
        shutil.copytree(smoke_cache["cache_dir"], target)
        return target

    # ------------------------------------------------------------------
    # per-case fresh caches: every case runs its own production prepare
    # and gets its own verification evidence

    def _tampered_cache(self, spatial_world, module_scratch, name,
                        mutate, mode="smoke"):
        """A fresh production cache, one semantic tamper applied, every
        related digest re-synced so only the target contract can catch
        it."""
        cache_dir = _fresh_cache(spatial_world, module_scratch, name,
                                 mode=mode)
        mutate(cache_dir)
        _resync(cache_dir)
        return cache_dir

    @staticmethod
    def _edit_json(path: Path, mutate) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload, indent=1),
                        encoding="utf-8")

    @pytest.mark.parametrize("field,value,fragment", [
        ("cache_schema", "material_spatial_input_cache_v0", "definition"),
        ("implementation_revision", "some_old_revision", "definition"),
        ("kind", "material_spatial_input_full_v1", "definition"),
        ("complete", True, "definition"),
        ("content_seed", 2027, "definition"),
        ("grid_size", 64, "definition"),
        ("content_seed", 2026.0, "definition"),
    ])
    def test_definition_field_tamper_is_refused(
            self, spatial_world, spatial_patches, module_scratch,
            field, value, fragment):
        cache_dir = self._tampered_cache(
            spatial_world, module_scratch, f"def_{field}_{id(value)}",
            lambda cache: self._edit_json(
                cache / "cache" / "definition.json",
                lambda definition: definition.__setitem__(field, value)))
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              fragment)

    def test_definition_unknown_field_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        cache_dir = self._tampered_cache(
            spatial_world, module_scratch, "def_extra_field",
            lambda cache: self._edit_json(
                cache / "cache" / "definition.json",
                lambda definition: definition.__setitem__(
                    "extra_note", "hello")))
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "unknown fields")

    def test_definition_missing_field_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        cache_dir = self._tampered_cache(
            spatial_world, module_scratch, "def_missing_field",
            lambda cache: self._edit_json(
                cache / "cache" / "definition.json",
                lambda definition: definition.pop("planes")))
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "missing fields")

    def _first_ledger_row(self, cache_dir: Path) -> dict:
        rows = [json.loads(line) for line in
                (cache_dir / "cache" / "samples.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()]
        return rows[0]

    def _rewrite_ledger(self, cache_dir: Path, rows: list) -> None:
        (cache_dir / "cache" / "samples.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows), encoding="utf-8")

    @pytest.mark.parametrize("label,mutate,fragment", [
        ("cache_row_float", lambda row: row.update(cache_row=0.9),
         "cache_row"),
        ("cache_row_wrong", lambda row: row.update(cache_row=999),
         "cache_row"),
        ("class_id_float", lambda row: row.update(class_id=1.0),
         "class_id"),
        ("w2_float", lambda row: row.update(w2=4.9), "w2"),
        ("offset_float", lambda row: row.update(offset=[0.5, 0]),
         "offset"),
        ("stereo_valid_string", lambda row: row.update(
            stereo_valid="false"), "stereo_valid"),
        ("has_valid_s_int", lambda row: row.update(has_valid_s=1),
         "has_valid_s"),
        ("group_wrong", lambda row: row.update(group="Some/Other/Group"),
         "group"),
        ("split_wrong", lambda row: row.update(split="test"), "split"),
        ("crop_wrong", lambda row: row.update(crop_window=[0, 0, 4, 4]),
         "crop_window"),
        ("fold_roles_wrong", lambda row: row.update(
            fold_roles={"fold0": "inner_train", "fold1": "inner_train",
                        "fold2": "inner_train"}), "fold_roles"),
        ("blocks_wrong", lambda row: row.update(
            blocks={"interior_valid": False, "band_valid": False}),
         "blocks"),
    ])
    def test_ledger_field_tamper_is_refused(
            self, spatial_world, spatial_patches, module_scratch,
            label, mutate, fragment):
        def apply(cache_dir: Path) -> None:
            rows_path = cache_dir / "cache" / "samples.jsonl"
            rows = [json.loads(line) for line in
                    rows_path.read_text(
                        encoding="utf-8").splitlines() if line.strip()]
            mutate(rows[0])
            self._rewrite_ledger(cache_dir, rows)
        cache_dir = self._tampered_cache(spatial_world, module_scratch,
                                         f"ledger_{label}", apply)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              fragment)

    def test_ledger_unknown_field_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        def apply(cache_dir: Path) -> None:
            rows_path = cache_dir / "cache" / "samples.jsonl"
            rows = [json.loads(line) for line in
                    rows_path.read_text(
                        encoding="utf-8").splitlines() if line.strip()]
            rows[0]["note"] = "extra"
            self._rewrite_ledger(cache_dir, rows)
        cache_dir = self._tampered_cache(spatial_world, module_scratch,
                                         "ledger_extra_field", apply)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "unknown fields")

    def test_frame_ledger_group_tamper_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        def apply(cache_dir: Path) -> None:
            path = cache_dir / "cache" / "frames.jsonl"
            rows = [json.loads(line) for line in
                    path.read_text(
                        encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                if row["instances"] > 0:
                    row["group"] = "Wrong/Group/0 NTU"
                    break
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n"
                        for row in rows), encoding="utf-8")
        cache_dir = self._tampered_cache(spatial_world, module_scratch,
                                         "frame_group", apply)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "group")

    def test_frame_ledger_extra_frame_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        def apply(cache_dir: Path) -> None:
            path = cache_dir / "cache" / "frames.jsonl"
            rows = [json.loads(line) for line in
                    path.read_text(
                        encoding="utf-8").splitlines() if line.strip()]
            rows.append({"frame": "Ghost_Frame_000",
                         "split": "train", "group": "g",
                         "instances": 0, "note": ""})
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n"
                        for row in rows), encoding="utf-8")
        cache_dir = self._tampered_cache(spatial_world, module_scratch,
                                         "frame_extra", apply)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "extra frames")

    # ------------------------------------------------------------------
    # saved-grid support tampering against the independent support set

    def _fake_support_cell(self, cache_dir: Path) -> tuple[str, object]:
        """A 3x4-crop-like single-support instance: write coverage=5e-7
        into a true-T==0 cell and keep the permutation plan's counts
        untouched (the tamper must be caught by the support check, not
        by the plan)."""
        grid_path = next(iter((cache_dir / "cache" / "grid").glob(
            "frame_*.npz")))
        with np.load(grid_path) as payload:
            arrays = {key: payload[key] for key in payload.files}
        plane_key = next(key for key in arrays
                         if key.endswith("|planes"))
        planes = arrays[plane_key]
        instance = plane_key.split("|")[0]
        target = _pick_zero_cell(planes, grid_path.parent.parent.parent)
        planes[6][target] = np.float32(5e-7)
        arrays[plane_key] = planes
        np.savez_compressed(grid_path, **arrays)
        return instance, target

    def test_fake_support_with_synced_binding_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        cache_dir = self._tampered_cache(
            spatial_world, module_scratch, "fake_support",
            self._fake_support_cell)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "support")

    def test_deleted_support_with_synced_binding_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        def apply(cache_dir: Path) -> None:
            grid_path = next(iter((cache_dir / "cache" / "grid").glob(
                "frame_*.npz")))
            with np.load(grid_path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            plane_key = next(key for key in arrays
                             if key.endswith("|planes"))
            planes = arrays[plane_key]
            coverage = planes[6]
            supported = np.argwhere(coverage.view(np.uint32) != 0)
            target = tuple(int(v) for v in supported[0])
            coverage[target] = np.float32(0.0)
            arrays[plane_key] = planes
            np.savez_compressed(grid_path, **arrays)
        cache_dir = self._tampered_cache(spatial_world, module_scratch,
                                         "deleted_support", apply)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "support")

    def test_negative_zero_response_with_synced_binding_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        def apply(cache_dir: Path) -> None:
            grid_path = next(iter((cache_dir / "cache" / "grid").glob(
                "frame_*.npz")))
            with np.load(grid_path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            plane_key = next(key for key in arrays
                             if key.endswith("|planes"))
            planes = arrays[plane_key]
            response = planes[4]        # q_interior
            target = _pick_zero_cell(planes,
                                     grid_path.parent.parent.parent)
            response[target] = np.float32(-0.0)
            arrays[plane_key] = planes
            np.savez_compressed(grid_path, **arrays)
        cache_dir = self._tampered_cache(spatial_world, module_scratch,
                                         "negative_zero", apply)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "response")

    def test_small_nonzero_response_without_support_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        def apply(cache_dir: Path) -> None:
            grid_path = next(iter((cache_dir / "cache" / "grid").glob(
                "frame_*.npz")))
            with np.load(grid_path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            plane_key = next(key for key in arrays
                             if key.endswith("|planes"))
            planes = arrays[plane_key]
            response = planes[0]        # L_interior
            target = _pick_zero_cell(planes,
                                     grid_path.parent.parent.parent)
            response[target] = np.float32(1e-30)
            arrays[plane_key] = planes
            np.savez_compressed(grid_path, **arrays)
        cache_dir = self._tampered_cache(spatial_world, module_scratch,
                                         "tiny_response", apply)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "response")

    # ------------------------------------------------------------------
    # binding-level refusals

    def test_missing_metadata_file_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "missing_metadata")
        (cache_dir / "cache" / "frames.jsonl").unlink()
        binding_path = cache_dir / "cache" / "binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        binding["artifacts"] = [record for record in binding["artifacts"]
                                if record["path"] !=
                                "cache/frames.jsonl"]
        binding["artifact_count"] = len(binding["artifacts"])
        binding_path.write_text(json.dumps(binding, indent=1),
                                encoding="utf-8")
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "missing")

    def test_extra_cache_file_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "extra_file")
        extra = cache_dir / "cache" / "native" / "frame_extra.npz"
        np.savez_compressed(extra, dummy=np.zeros(3, dtype=np.float32))
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "extra")

    def test_extra_artifact_synced_into_binding_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "extra_synced")
        extra = cache_dir / "cache" / "grid" / "frame_extra.npz"
        np.savez_compressed(extra, dummy=np.zeros(3, dtype=np.float32))
        binding_path = cache_dir / "cache" / "binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        binding["artifacts"].append({
            "path": "cache/grid/frame_extra.npz", "kind": "grid",
            "bytes": extra.stat().st_size,
            "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
            "npz_keys": [{"key": "dummy", "dtype": "float32",
                          "shape": [3]}]})
        binding["artifact_count"] = len(binding["artifacts"])
        binding_path.write_text(json.dumps(binding, indent=1),
                                encoding="utf-8")
        _resync(cache_dir)
        # the binding now documents the extra artifact too, but the
        # DERIVED expected set does not: still refused
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "unexpected artifacts")

    def test_duplicate_binding_entry_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "duplicate_entry")
        binding_path = cache_dir / "cache" / "binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        binding["artifacts"].append(copy.deepcopy(
            binding["artifacts"][0]))
        binding_path.write_text(json.dumps(binding, indent=1),
                                encoding="utf-8")
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "duplicate")

    def test_npz_key_tamper_with_synced_binding_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "key_rename")
        shard = next(iter((cache_dir / "cache" / "native").glob(
            "frame_*.npz")))
        with np.load(shard) as payload:
            arrays = {key: payload[key] for key in payload.files}
        renamed = {}
        for key, value in arrays.items():
            if key.endswith("|L"):
                renamed[key[:-1] + "X"] = value
            else:
                renamed[key] = value
        np.savez_compressed(shard, **renamed)
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "authoritative")

    def test_region_float32_with_synced_binding_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "region_f32")
        shard = next(iter((cache_dir / "cache" / "native").glob(
            "frame_*.npz")))
        with np.load(shard) as payload:
            arrays = {key: payload[key] for key in payload.files}
        region_key = next(key for key in arrays if key.endswith("|region"))
        arrays[region_key] = arrays[region_key].astype(np.float32)
        np.savez_compressed(shard, **arrays)
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "native contract")

    def test_deleted_instance_keys_with_synced_binding_are_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "deleted_instance")
        ledger = [json.loads(line) for line in
                  (cache_dir / "cache" / "samples.jsonl").read_text(
                      encoding="utf-8").splitlines() if line.strip()]
        victim = ledger[0]["sample_name"]
        frame = ledger[0]["frame"]
        for subdir in ("native", "grid"):
            shard = cache_dir / "cache" / subdir / f"frame_{frame}.npz"
            with np.load(shard) as payload:
                arrays = {key: payload[key] for key in payload.files
                          if not key.startswith(victim + "|")}
            np.savez_compressed(shard, **arrays)
        perm_path = cache_dir / "cache" / "permutation" / \
            f"frame_{frame}.json"
        records = json.loads(perm_path.read_text(encoding="utf-8"))
        records.pop(victim)
        perm_path.write_text(json.dumps(records, ensure_ascii=False),
                             encoding="utf-8")
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "authoritative")

    def test_tampered_native_with_synced_binding_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "tamper_native")
        shard = next(iter((cache_dir / "cache" / "native").glob(
            "frame_*.npz")))
        with np.load(shard) as payload:
            arrays = {key: payload[key] for key in payload.files}
        pixel_key = next(key for key in arrays if key.endswith("|L"))
        arrays[pixel_key] = arrays[pixel_key] ^ np.uint8(0xFF)
        np.savez_compressed(shard, **arrays)
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "native")

    def test_wrong_dtype_is_refused(self, spatial_world,
                                    spatial_patches, smoke_cache,
                                    module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "wrong_dtype")
        shard = next(iter((cache_dir / "cache" / "grid").glob(
            "frame_*.npz")))
        with np.load(shard) as payload:
            arrays = {key: payload[key] for key in payload.files}
        plane_key = next(key for key in arrays
                         if key.endswith("|planes"))
        arrays[plane_key] = arrays[plane_key].astype(np.float64)
        np.savez_compressed(shard, **arrays)
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "shape/dtype")

    def test_missing_shard_is_refused(self, spatial_world,
                                      spatial_patches, smoke_cache,
                                      module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "missing_shard")
        shard = next(iter((cache_dir / "cache" / "grid").glob(
            "frame_*.npz")))
        shard.unlink()
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "missing")

    # ------------------------------------------------------------------
    # permutation plan refusals (per-case fresh caches)

    def _perm_path(self, cache_dir: Path) -> Path:
        return next(iter((cache_dir / "cache" / "permutation").glob(
            "frame_*.json")))

    @pytest.mark.parametrize("label,mutate,fragment", [
        ("extra_instance", lambda records: records.update(
            Ghost_obj999=records[next(iter(records))]), "instance set"),
        ("extra_top_field", lambda records: next(iter(records.values()))
         .update(note="x"), "unknown fields"),
        ("content_seed_float", lambda records: next(
            iter(records.values())).update(content_seed=2026.0),
         "true int"),
        ("seed_float", lambda records: next(
            iter(records.values()))["regions"]["interior"]["groups"]
         and None or None, "plan"),
        ("permuted_int", lambda records: _flip_one_multi_cell(
            records, {"permuted": 1}), "true bool"),
        ("extra_region_field", lambda records: next(
            iter(records.values()))["regions"]["interior"]
         .update(extra=1), "unknown fields"),
    ])
    def test_plan_tamper_is_refused(self, spatial_world, spatial_patches,
                                    module_scratch, label, mutate,
                                    fragment):
        def apply(cache_dir: Path) -> None:
            path = self._perm_path(cache_dir)
            records = json.loads(path.read_text(encoding="utf-8"))
            mutate(records)
            path.write_text(json.dumps(records, ensure_ascii=False),
                            encoding="utf-8")
        if label == "seed_float":
            # a targeted float seed inside one multi-cell group
            def apply(cache_dir: Path) -> None:
                path = self._perm_path(cache_dir)
                records = json.loads(path.read_text(encoding="utf-8"))
                hit = False
                for plan in records.values():
                    for region in plan["regions"].values():
                        for group in region["groups"].values():
                            if group.get("permuted"):
                                group["seed64"] = float(group["seed64"])
                                hit = True
                if not hit:
                    raise AssertionError(
                        "no multi-cell group in the fixture")
                path.write_text(json.dumps(records, ensure_ascii=False),
                                encoding="utf-8")
        cache_dir = self._tampered_cache(spatial_world, module_scratch,
                                         f"plan_{label}", apply)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              fragment)

    def test_permutation_summary_tamper_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "tamper_perm")
        perm_path = self._perm_path(cache_dir)
        records = json.loads(perm_path.read_text(encoding="utf-8"))
        name = next(iter(records))
        for region in records[name]["regions"].values():
            region["moved_indices"] = region["moved_indices"] + 7
        perm_path.write_text(json.dumps(records, ensure_ascii=False),
                             encoding="utf-8")
        _resync(cache_dir)
        _assert_fresh_refusal(spatial_world, module_scratch, cache_dir,
                              "permutation")

    def test_expected_binding_digest_mismatch_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "binding_mismatch")
        patches = [
            mock.patch.object(verify, "PINNED_CONFIG_REL",
                              spatial_world["cfg_path"].relative_to(
                                  _ROOT).as_posix()),
            mock.patch.object(verify, "OUTPUT_ROOT",
                              cache_dir.parent.parent.relative_to(
                                  _ROOT).as_posix()),
        ]
        for context in patches:
            context.start()
        try:
            code = verify.main([
                "--config", str(spatial_world["cfg_path"]),
                "--mode", "smoke",
                "--cache-dir", str(cache_dir),
                "--output-dir", str(cache_dir / "verification"),
                "--deadline-utc", _DEADLINE,
                "--expected-binding-sha256", "0" * 64])
        finally:
            for context in patches:
                context.stop()
        assert code == verify.EXIT_REFUSED

    def test_wrong_output_dir_is_refused(
            self, spatial_world, spatial_patches, smoke_cache,
            module_scratch):
        patches = [
            mock.patch.object(verify, "PINNED_CONFIG_REL",
                              spatial_world["cfg_path"].relative_to(
                                  _ROOT).as_posix()),
            mock.patch.object(verify, "OUTPUT_ROOT",
                              module_scratch.relative_to(
                                  _ROOT).as_posix()),
        ]
        for context in patches:
            context.start()
        try:
            code = verify.main([
                "--config", str(spatial_world["cfg_path"]),
                "--mode", "smoke",
                "--cache-dir", str(smoke_cache["cache_dir"]),
                "--output-dir", str(module_scratch / "elsewhere"),
                "--deadline-utc", _DEADLINE,
                "--expected-binding-sha256",
                _binding_sha(smoke_cache["cache_dir"])])
        finally:
            for context in patches:
                context.stop()
        assert code == verify.EXIT_REFUSED

    def test_late_deadline_is_partial(self, spatial_world,
                                      spatial_patches, smoke_cache,
                                      module_scratch):
        from core.material_residual_scale import DeadlineExceeded

        class BoundaryWatch:
            def __init__(self, *, deadline_utc=None, budget_minutes=None):
                pass

            def check(self, label):
                if label == "before success status publish":
                    raise DeadlineExceeded("expired at the final publish")

            def remaining_s(self):
                return 1.0

        cache_dir = self._copy_cache(smoke_cache, module_scratch,
                                     "verify_deadline_copy")
        out = cache_dir / "verification"
        code = _run_verify(
            spatial_world, cache_dir, cache_dir.parent.parent,
            extra_patches=[mock.patch.object(verify, "DeadlineWatch",
                                             BoundaryWatch)])
        assert code == verify.EXIT_PARTIAL
        status = json.loads((out / "status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        assert status["unfinished"]
        assert (out / "report.json").is_file()


def _flip_one_multi_cell(records: dict, update: dict) -> None:
    for plan in records.values():
        for region in plan["regions"].values():
            for group in region["groups"].values():
                if group.get("permuted"):
                    group.update(update)
                    return
    raise AssertionError("no multi-cell group in the fixture")
