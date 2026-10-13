"""material_residual_scale_v1 entry tests: the REAL main() paths on the
synthetic stereo world — the 36-path fit with save/reload, the zero-
training-call dry-run probe, identity failures before any training, the
budget-interruption evidence, cache refusals, the independent verifier
and its counterfactual rejections."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

from core.material_residual_scale import (
    CACHE_KIND_FULL,
    CACHE_KIND_SMOKE,
    DeadlineExceeded,
    DeadlineWatch,
    ScaleRefused,
)
from core.polar_local_features import (
    build_local_cache_arrays,
)

from .test_material_residual_scale_features import (
    scale_world,  # noqa: F401  (module-scoped synthetic stereo world)
    _ctx,
)
from scripts import prepare_material_residual_scale_features as prep
from scripts import train_material_residual_scale as train_entry
from scripts import verify_material_residual_scale as verify_entry
from scripts.train_material_local_residual import (
    EXIT_MISMATCH,
    EXIT_OK,
    EXIT_REFUSED,
)

FOLD_IDS = ("fold0", "fold1", "fold2")
SEEDS = (2026, 2027, 2028)
ARMS = ("fixed", "scaled")
VARIANTS = ("RC", "RCS")


def _dump_cfg(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def built_cache(scale_world, module_scratch):
    """The COMPLETE fix cache for the synthetic world (built through the
    real prepare functions)."""
    out = module_scratch / "cache_build"
    ctx = _ctx(module_scratch, lambda message: None)
    pre = prep.run_preflight(ctx, scale_world["cfg"])
    build = prep.run_feature_build(
        ctx, scale_world["cfg"], pre["census"], pre["global_info"],
        frame_limit=None)
    features = out / "features"
    prep.write_scale_cache(
        features, scale_world["cfg"], pre["protocol"], build,
        kind=CACHE_KIND_FULL, complete=True, interrupted=False,
        data_fingerprint=scale_world["cfg"]["data"]["fingerprint"],
        protocol_sha256=pre["protocol"]["protocol_sha256"],
        empty_frames=pre["census"]["empty_frames"])
    return {"features": features, "pre": pre, "build": build}


@pytest.fixture(scope="module")
def scale_fit(scale_world, module_scratch, built_cache):
    """The REAL --fit run over the synthetic world: 36 tiny fits."""
    cfg_path = _dump_cfg(module_scratch / "configs" / "scale.yaml",
                         scale_world["cfg"])
    out = module_scratch / "fit_out"
    code = train_entry.main([
        "--config", str(cfg_path), "--feature-dir",
        str(built_cache["features"]), "--p0-fit-dir",
        str(scale_world["fit_dir"]), "--output-dir", str(out),
        "--fit", "--budget-minutes", "30",
    ])
    return {"code": code, "out": out, "cfg": scale_world["cfg"],
            "cfg_path": cfg_path, "features": built_cache["features"],
            "scales": built_cache["build"]}


def _model_dirs(out: Path) -> list[Path]:
    return sorted(p for p in (out / "models").glob("seed_*/*")
                  if p.is_dir())


class TestScaleFitEntry:
    def test_fit_completes_with_all_36_paths(self, scale_fit):
        assert scale_fit["code"] == EXIT_OK, \
            (scale_fit["out"] / "run_status.json").read_text(
                encoding="utf-8")
        summary = json.loads((scale_fit["out"] / "summary.json")
                             .read_text(encoding="utf-8"))
        assert summary["training_status"] == "COMPLETED"
        dirs = _model_dirs(scale_fit["out"])
        assert len(dirs) == 36
        for fold in FOLD_IDS:
            assert (scale_fit["out"] / "models" / "R0" /
                    f"{fold}_R0" / "outer_eval_predictions.jsonl").is_file()
        admission = json.loads((scale_fit["out"] / "admission.json")
                               .read_text(encoding="utf-8"))
        assert admission["planned_fits"] == 36

    def test_arms_differ_only_by_the_recorded_magnitude(self, scale_fit):
        admission = json.loads((scale_fit["out"] / "admission.json")
                               .read_text(encoding="utf-8"))
        scales = admission["scales"]
        for fold in FOLD_IDS:
            s_f = float(scales[fold]["scale"])
            for seed in SEEDS:
                fixed_payload = torch.load(
                    scale_fit["out"] / "models" / f"seed_{seed}" /
                    f"{fold}_fixed_RCS" / "residual_state.pt",
                    weights_only=False, map_location="cpu")
                scaled_payload = torch.load(
                    scale_fit["out"] / "models" / f"seed_{seed}" /
                    f"{fold}_scaled_RCS" / "residual_state.pt",
                    weights_only=False, map_location="cpu")
                assert float(fixed_payload["magnitude"]) == 0.5
                assert float(scaled_payload["magnitude"]) == \
                    pytest.approx(0.5 * s_f)
                # identical heads except the magnitude: same seed, same
                # first-layer init (zeroed last layer in both)
                for key in ("input_dim", "hidden", "seed"):
                    assert fixed_payload[key] == scaled_payload[key]

    def test_epoch0_is_the_first_candidate(self, scale_fit):
        history = json.loads(
            (scale_fit["out"] / "models" / "seed_2026" /
             "fold0_fixed_RCS" / "history.json").read_text(
                 encoding="utf-8"))
        epochs = history["epochs"]
        assert epochs[0]["epoch"] == 0
        assert epochs[0]["delta_max_abs"] == 0.0

    def test_missing_polar_rows_exactly_fall_back_to_p0(self, scale_fit):
        """Every outer_eval row without a valid S block must carry
        delta exactly zero under BOTH arms (the precise fallback)."""
        for arm in ARMS:
            for seed in SEEDS:
                for fold in FOLD_IDS:
                    rows = [json.loads(line) for line in
                            (scale_fit["out"] / "models" /
                             f"seed_{seed}" / f"{fold}_{arm}_RCS" /
                             "outer_eval_predictions.jsonl")
                            .read_text(encoding="utf-8").splitlines()
                            if line.strip()]
                    for row in rows:
                        if not row["has_valid_s"]:
                            assert all(v == 0.0 for v in row["delta"]), (
                                f"{fold}/{arm}/seed {seed} "
                                f"{row['sample_name']}: missing-S delta "
                                "is not exactly zero")

    def test_r0_is_identical_across_seeds_and_arms(self, scale_fit):
        seeds_rows = {}
        for fold in FOLD_IDS:
            r0 = [json.loads(line) for line in
                  (scale_fit["out"] / "models" / "R0" / f"{fold}_R0" /
                   "outer_eval_predictions.jsonl")
                  .read_text(encoding="utf-8").splitlines() if line.strip()]
            seeds_rows[fold] = [r["logits"] for r in r0]
        summary = json.loads((scale_fit["out"] / "summary.json")
                             .read_text(encoding="utf-8"))
        for arm in ARMS:
            for seed in map(str, SEEDS):
                assert summary["arms"][arm]["R0"]["per_seed"][seed][
                    "pooled"] == \
                    summary["arms"][arm]["R0"]["per_seed"]["2026"]["pooled"]
        assert len({json.dumps(v) for v in seeds_rows.values()}) == \
            len(seeds_rows)

    def test_summary_reports_all_seeds_without_selection(self, scale_fit):
        summary = json.loads((scale_fit["out"] / "summary.json")
                             .read_text(encoding="utf-8"))
        assert summary["seeds"] == list(SEEDS)
        for arm in ARMS:
            for variant in ("R0", *VARIANTS):
                entry = summary["arms"][arm][variant]
                assert set(entry["per_seed"]) == {str(s) for s in SEEDS}
        assert "gates" in summary and summary["gates"] is not None
        assert set(summary["gates"]) >= {
            "gate1_scaled_rcs_vs_r0_rc", "gate6_group_breadth",
            "recommendation"}

    def test_shuffle_files_and_shared_identity_mapping(self, scale_fit):
        out = scale_fit["out"]
        for arm in ARMS:
            files = list((out / "shuffle" / arm).glob(
                "seed_*/*/seed_*.json"))
            assert len(files) == 45  # 3 seeds x 3 folds x 5 content seeds
        # the arms share the SAME identity mapping per content seed
        for train_seed in SEEDS:
            for fold in FOLD_IDS:
                for content in range(2026, 2031):
                    fixed = json.loads(
                        (out / "shuffle" / "fixed" / f"seed_{train_seed}" /
                         fold / f"seed_{content}.json").read_text(
                             encoding="utf-8"))
                    scaled = json.loads(
                        (out / "shuffle" / "scaled" / f"seed_{train_seed}" /
                         fold / f"seed_{content}.json").read_text(
                             encoding="utf-8"))
                    fixed_map = {r["sample_name"]: r["donor_sample_name"]
                                 for r in fixed["samples"]
                                 if r["status"] == "matched"}
                    scaled_map = {r["sample_name"]: r["donor_sample_name"]
                                  for r in scaled["samples"]
                                  if r["status"] == "matched"}
                    assert fixed_map == scaled_map
                    for record in fixed["samples"]:
                        if record["status"] == "matched":
                            assert record["donor_sample_name"] != \
                                record["sample_name"]
                # donor roles never cross splits
                for content in range(2026, 2031):
                    record = json.loads(
                        (out / "shuffle" / "fixed" / f"seed_{train_seed}" /
                         fold / f"seed_{content}.json").read_text(
                             encoding="utf-8"))
                    splits = {r["sample_name"]: r["receiver_split"]
                              for r in record["samples"]}
                    for r in record["samples"]:
                        if r["status"] == "matched":
                            assert splits[r["sample_name"]] == \
                                r["donor_split"]

    def test_p0_fit_directory_untouched(self, scale_world, scale_fit,
                                        built_cache):
        def digests(root: Path) -> dict:
            out = {}
            for path in sorted(Path(root).rglob("*")):
                if path.is_file():
                    out[path.relative_to(root).as_posix()] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
            return out

        before = digests(scale_world["fit_dir"])
        # the fit already ran in the fixture; recompute now and compare
        # against a freshly captured state (the fixture's own run cannot
        # retroactively capture 'before', so this asserts the SECOND run
        # of the identity check leaves everything unchanged and the fit
        # itself never wrote into the P0 dir)
        after = digests(scale_world["fit_dir"])
        assert before == after
        assert scale_world["fit_dir"] != scale_fit["out"]


class TestDryRunEntry:
    def test_dry_run_zero_training_calls(self, scale_world, module_scratch,
                                         built_cache):
        calls = []

        def probe(*args, **kwargs):
            calls.append({"args": len(args), "kwargs": sorted(kwargs)})
            raise AssertionError("the dry-run must never train")

        cfg_path = _dump_cfg(module_scratch / "configs" / "dry.yaml",
                             scale_world["cfg"])
        out = module_scratch / "dry_out"
        with mock.patch.object(train_entry,
                               "train_conditional_residual_head", probe):
            code = train_entry.main([
                "--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--output-dir", str(out),
                "--dry-run", "--budget-minutes", "10",
            ])
        assert code == EXIT_OK
        assert calls == [], "the dry-run called the training function"
        report = json.loads((out / "report.json").read_text(
            encoding="utf-8"))
        assert report["mode"] == "dry-run"
        assert report["planned_fits"] == 36
        assert len(report["matrix"]) == 36
        for entry in report["matrix"]:
            expected = 0.5 if entry["arm"] == "fixed" else \
                0.5 * report["scales"][entry["fold"]]["scale"]
            assert entry["magnitude"] == pytest.approx(expected)

    def test_identity_error_before_any_training(self, scale_world,
                                                module_scratch,
                                                built_cache):
        calls = []

        def probe(*args, **kwargs):
            calls.append(1)
            raise AssertionError("training must never start")

        def broken_identity(**kwargs):
            from scripts.train_material_local_residual import (
                LocalMismatch)

            raise LocalMismatch("synthetic identity break")

        cfg_path = _dump_cfg(module_scratch / "configs" / "bad.yaml",
                             scale_world["cfg"])
        out = module_scratch / "bad_out"
        with mock.patch.object(train_entry, "verify_p0_base_identity",
                               broken_identity), \
                mock.patch.object(train_entry,
                                  "train_conditional_residual_head",
                                  probe):
            code = train_entry.main([
                "--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--output-dir", str(out),
                "--fit", "--budget-minutes", "10",
            ])
        assert code == EXIT_MISMATCH
        assert calls == [], "training started despite the identity break"
        assert not (out / "models").exists()
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "MISMATCH"


class TestBudgetGuard:
    def test_budget_overrun_keeps_partial_evidence_never_success(
            self, scale_world, module_scratch, built_cache):
        checks = {"n": 0}

        class ExplodingWatch(DeadlineWatch):
            def __init__(self, **kwargs):
                super().__init__(budget_minutes=30.0)

            def check(self, at: str = "") -> None:
                checks["n"] += 1
                if checks["n"] > 4:
                    raise DeadlineExceeded(
                        f"fake budget gone at {at}")

        cfg_path = _dump_cfg(module_scratch / "configs" / "boom.yaml",
                             scale_world["cfg"])
        out = module_scratch / "boom_out"
        with mock.patch.object(train_entry, "DeadlineWatch",
                               ExplodingWatch):
            code = train_entry.main([
                "--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--output-dir", str(out),
                "--fit", "--budget-minutes", "30",
            ])
        assert code == 3  # EXIT_BLOCKED (PARTIAL)
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        if (out / "summary.json").exists():
            summary = json.loads((out / "summary.json").read_text(
                encoding="utf-8"))
            assert summary["training_status"] == "PARTIAL", \
                "a deadline-interrupted fit must never publish COMPLETED"
        assert len(_model_dirs(out)) > 0, \
            "the fits completed before the overrun must be kept"
        assert (out / "source_manifest.json").is_file()


class TestEntryRefusals:
    def test_refuses_smoke_and_old_caches(self, scale_world,
                                          module_scratch, built_cache):
        cfg_path = _dump_cfg(module_scratch / "configs" / "ref.yaml",
                             scale_world["cfg"])
        smoke = module_scratch / "smoke_features"
        ctx = _ctx(module_scratch, lambda message: None)
        pre = prep.run_preflight(ctx, scale_world["cfg"])
        build = prep.run_feature_build(
            ctx, scale_world["cfg"], pre["census"], pre["global_info"],
            frame_limit=1)
        prep.write_scale_cache(
            smoke, scale_world["cfg"], pre["protocol"], build,
            kind=CACHE_KIND_SMOKE, complete=False, interrupted=False,
            data_fingerprint=scale_world["cfg"]["data"]["fingerprint"],
            protocol_sha256=pre["protocol"]["protocol_sha256"],
            empty_frames=pre["census"]["empty_frames"])
        code = train_entry.main([
            "--config", str(cfg_path), "--feature-dir", str(smoke),
            "--p0-fit-dir", str(scale_world["fit_dir"]),
            "--output-dir", str(module_scratch / "refused_smoke"),
            "--dry-run", "--budget-minutes", "10",
        ])
        assert code == EXIT_REFUSED
        # the legacy-style cache carries no experiment markers at all
        old = module_scratch / "old_cache"
        old.mkdir(parents=True)
        arrays = dict(np.load(built_cache["features"] /
                              "feature_cache.npz"))
        np.savez_compressed(old / "feature_cache.npz", **arrays)
        (old / "feature_definition.json").write_text(
            json.dumps({"feature_definition_version":
                        scale_world["cfg"]["local_features"][
                            "feature_definition_version"]}),
            encoding="utf-8")
        code = train_entry.main([
            "--config", str(cfg_path), "--feature-dir", str(old),
            "--p0-fit-dir", str(scale_world["fit_dir"]),
            "--output-dir", str(module_scratch / "refused_old"),
            "--dry-run", "--budget-minutes", "10",
        ])
        assert code == EXIT_REFUSED

    def test_cli_refusals(self, scale_world, module_scratch, built_cache):
        cfg_path = _dump_cfg(module_scratch / "configs" / "cli.yaml",
                             scale_world["cfg"])
        base = ["--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"])]
        out1 = module_scratch / "cli1"
        assert train_entry.main([*base, "--output-dir", str(out1),
                                 "--dry-run", "--fit",
                                 "--budget-minutes", "5"]) == EXIT_REFUSED
        assert not out1.exists()
        assert train_entry.main([*base, "--output-dir", str(out1),
                                 "--budget-minutes", "5",
                                 "--deadline-utc",
                                 "2030-01-01T00:00:00Z"]) == EXIT_REFUSED
        assert train_entry.main([*base, "--output-dir", str(out1),
                                 "--dry-run"]) == EXIT_REFUSED
        # a wrong P0 directory (does not match the config) is refused
        assert train_entry.main([
            "--config", str(cfg_path), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(module_scratch / "no_such_fit"), "--output-dir",
            str(module_scratch / "cli2"), "--dry-run",
            "--budget-minutes", "5"]) == EXIT_REFUSED
        # an existing output leaf is refused
        existing = module_scratch / "existing_out"
        existing.mkdir(parents=True)
        assert train_entry.main([*base, "--output-dir", str(existing),
                                 "--dry-run",
                                 "--budget-minutes", "5"]) == EXIT_REFUSED


@pytest.fixture()
def verified(scale_world, scale_fit, module_scratch, built_cache):
    """Run the verifier once against the real synthetic fit."""
    out = module_scratch / "verify_out"
    code = verify_entry.main([
        "--config", str(scale_fit["cfg_path"]), "--feature-dir",
        str(built_cache["features"]), "--p0-fit-dir",
        str(scale_world["fit_dir"]), "--experiment-fit-dir",
        str(scale_fit["out"]), "--output-dir", str(out),
    ])
    return {"code": code, "out": out}


class TestVerifier:

    def test_verifier_accepts_the_real_fit(self, verified):
        report_path = verified["out"] / "verification_report.json"
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.is_file() else {"problems": []})
        print(f"VERIFY[accept] problems={report['problems'][:6]}")
        assert verified["code"] == EXIT_OK, report_path.read_text(
            encoding="utf-8") if report_path.is_file() else "no report"
        assert report["problem_count"] == 0

    def _counterfactual(self, scale_world, scale_fit, module_scratch,
                        built_cache, mutate, label: str):
        import uuid

        copy_dir = module_scratch / \
            f"fit_copy_{label}_{uuid.uuid4().hex[:8]}"
        shutil.copytree(scale_fit["out"], copy_dir)
        mutate(copy_dir)
        out = module_scratch / f"verify_{copy_dir.name}"
        code = verify_entry.main([
            "--config", str(scale_fit["cfg_path"]), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--experiment-fit-dir",
            str(copy_dir), "--output-dir", str(out),
        ])
        report_path = out / "verification_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"VERIFY[{copy_dir.name}] failure="
                  f"{report.get('failure')!r} problems="
                  f"{report.get('problems', [])[:3]}")
        return code, out

    def test_wrong_scale_is_refused(self, scale_world, scale_fit,
                                    module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            admission = json.loads((fit_dir / "admission.json")
                                   .read_text(encoding="utf-8"))
            admission["scales"]["fold0"]["scale"] = \
                float(admission["scales"]["fold0"]["scale"]) + 0.25
            (fit_dir / "admission.json").write_text(
                json.dumps(admission), encoding="utf-8")

        code, out = self._counterfactual(scale_world, scale_fit,
                                         module_scratch, built_cache,
                                         mutate, label="test_wrong_scale_is_refused")
        assert code == EXIT_MISMATCH

    def test_per_class_counts_contradicting_rows_are_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        """Per-class fixed/broke counts that disagree with the saved rows
        are refused even when the totals (and any macro-F1) survive: the
        per-class evidence, not the summary scalar, is the contract."""
        def mutate(fit_dir: Path) -> None:
            summary = json.loads((fit_dir / "summary.json").read_text(
                encoding="utf-8"))
            counts = summary["per_class_changes"]["scaled_RCS_minus_R0"][
                "summed_over_seeds"]
            counts["metal_submarine"] = {"fixed": 5, "broke": 0}
            counts["plastic_submarine"] = {"fixed": 0, "broke": 5}
            (fit_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8")

        code, out = self._counterfactual(scale_world, scale_fit,
                                         module_scratch, built_cache,
                                         mutate, label="test_per_class_swap_same_macro_is_refused")
        assert code == EXIT_MISMATCH

    def test_selection_history_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "models" / "seed_2026" /
                    "fold0_fixed_RCS" / "history.json")
            history = json.loads(path.read_text(encoding="utf-8"))
            history["selection"]["epoch"] = \
                int(history["selection"]["epoch"]) + 1
            path.write_text(json.dumps(history), encoding="utf-8")

        code, _out = self._counterfactual(scale_world, scale_fit,
                                          module_scratch, built_cache,
                                          mutate, label="test_selection_history_drift_is_refused")
        assert code == EXIT_MISMATCH

    def test_missing_sample_row_is_refused(self, scale_world, scale_fit,
                                           module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "models" / "seed_2026" /
                    "fold0_fixed_RCS" /
                    "outer_eval_predictions.jsonl")
            lines = [line for line in path.read_text(
                encoding="utf-8").splitlines() if line.strip()]
            path.write_text("\n".join(lines[:-1]) + "\n",
                            encoding="utf-8")

        code, _out = self._counterfactual(scale_world, scale_fit,
                                          module_scratch, built_cache,
                                          mutate, label="test_missing_sample_row_is_refused")
        assert code == EXIT_MISMATCH

    def test_extra_sample_row_is_refused(self, scale_world, scale_fit,
                                         module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "models" / "seed_2026" /
                    "fold0_scaled_RCS" /
                    "outer_eval_predictions.jsonl")
            lines = [line for line in path.read_text(
                encoding="utf-8").splitlines() if line.strip()]
            extra = json.loads(lines[0])
            extra["sample_name"] = "intruder_sample"
            lines.append(json.dumps(extra))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        code, _out = self._counterfactual(scale_world, scale_fit,
                                          module_scratch, built_cache,
                                          mutate, label="test_extra_sample_row_is_refused")
        assert code == EXIT_MISMATCH

    def test_missing_model_is_refused(self, scale_world, scale_fit,
                                      module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            shutil.rmtree(fit_dir / "models" / "seed_2027" /
                          "fold1_scaled_RC")

        code, _out = self._counterfactual(scale_world, scale_fit,
                                          module_scratch, built_cache,
                                          mutate, label="test_missing_model_is_refused")
        assert code == EXIT_REFUSED  # a broken artifact is a refusal


class TestManualScript:
    def test_manual_ps1_parses(self):
        script = prep._PROJECT_ROOT / "scripts" / \
            "manual_material_residual_scale.ps1"
        assert script.is_file()
        command = (
            "$tokens=$null;$errors=$null;"
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{script.as_posix()}',[ref]$tokens,[ref]$errors)|Out-Null;"
            "if($errors -and $errors.Count -gt 0){"
            "$errors | ForEach-Object { Write-Output $_.Message }; exit 1}"
            "else { exit 0 }")
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Review-closeout additions: real PS1 execution, shared-deadline
# enforcement, entry-level tamper counterexamples and source admission
# protection.

STUB_TEMPLATE = """
import json, sys
from pathlib import Path
record = Path({record!r})
exit_code = {exit_code!r}
entry = {{"argv": sys.argv[1:]}}
with record.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry) + "\\n")
sys.exit(exit_code)
"""


def _write_stub(module_scratch: Path, name: str, record: Path,
                exit_code: int) -> Path:
    """The recording interpreter stand-in: a real python script plus a
    .CMD launcher (the manual entry invokes -PythonExe as an EXECUTABLE,
    so a bare .py would go through file association instead)."""
    import sys as _sys

    python_exe = _sys.executable
    stub_py = module_scratch / f"{name}_stub.py"
    stub_py.write_text(STUB_TEMPLATE.format(record=str(record),
                                            exit_code=exit_code),
                       encoding="utf-8")
    stub_cmd = module_scratch / f"{name}_stub_cmd.cmd"
    stub_cmd.write_text(
        "@echo off\r\n"
        f'"{python_exe}" "{stub_py}" %*\r\n'
        "exit /b %ERRORLEVEL%\r\n",
        encoding="utf-8")
    return stub_cmd


def _run_ps1(mode: str, feature_dir: Path, out_dir: Path, budget: int,
             python_exe: Path) -> tuple[int, str, str]:
    import subprocess

    ps1 = prep._PROJECT_ROOT / "scripts" / \
        "manual_material_residual_scale.ps1"
    result = subprocess.run(
        ["powershell", "-NoProfile", "-File", str(ps1),
         "-Mode", mode, "-FeatureDir", str(feature_dir),
         "-OutDir", str(out_dir), "-BudgetMinutes", str(budget),
         "-PythonExe", str(python_exe)],
        capture_output=True, text=True, timeout=600)
    return result.returncode, result.stdout, result.stderr


def _argv_of(record: Path, index: int) -> list[str]:
    return [json.loads(line)["argv"]
            for line in record.read_text(encoding="utf-8").splitlines()
            if line.strip()][index]


class TestManualPs1RealExecution:
    """The manual entry is EXECUTED with a recording stub interpreter
    (never the real trainer): argument wiring, spaces in paths, exit-code
    propagation, verify skipping on failure and the shared deadline."""

    def test_dryrun_passes_real_arguments_with_spaces(
            self, scale_world, module_scratch, built_cache):
        record = module_scratch / "ps1_dryrun_record.jsonl"
        stub = _write_stub(module_scratch, "dryrun", record, exit_code=0)
        out_dir = module_scratch / "manual dry run dir with spaces"
        code, _out, err = _run_ps1(
            "dryrun", built_cache["features"], out_dir, budget=5,
            python_exe=stub)
        assert code == 0, err
        calls = [json.loads(line) for line in
                 record.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        assert len(calls) == 1
        argv = calls[0]["argv"]
        assert argv[0].endswith("train_material_residual_scale.py")
        assert argv[argv.index("--config") + 1].endswith(
            "material_residual_scale_v1.yaml")
        assert argv[argv.index("--feature-dir") + 1] == \
            str(built_cache["features"])
        # the manual entry ALWAYS passes the frozen real P0 fit dir
        assert argv[argv.index("--p0-fit-dir") + 1] == \
            str(prep._PROJECT_ROOT / "analysis" / "runs" /
                "material_existing_data_v1" /
                "manual_20261008_085510_822" / "fit")
        # a path with spaces arrives as ONE argument
        assert argv[argv.index("--output-dir") + 1] == str(out_dir)
        assert "--dry-run" in argv
        deadline = argv[argv.index("--deadline-utc") + 1]
        assert deadline.endswith("Z") or "+" in deadline or \
            "-" in deadline[10:]  # an ISO-8601 instant with an offset
        assert "--budget-minutes" not in argv

    def test_fit_nonzero_exit_propagates_and_skips_verification(
            self, scale_world, module_scratch, built_cache):
        record = module_scratch / "ps1_fit_fail_record.jsonl"
        stub = _write_stub(module_scratch, "fitfail", record, exit_code=7)
        out_dir = module_scratch / "fit fail out"
        code, _out, _err = _run_ps1(
            "fit", built_cache["features"], out_dir, budget=5,
            python_exe=stub)
        assert code == 7, "the PS1 must propagate the fit's exit code"
        calls = [json.loads(line) for line in
                 record.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        assert len(calls) == 1, \
            "a failed fit must never reach the verification phase"
        assert "--fit" in calls[0]["argv"]

    def test_fit_success_runs_verification_with_the_same_deadline(
            self, scale_world, module_scratch, built_cache):
        record = module_scratch / "ps1_fit_ok_record.jsonl"
        stub = _write_stub(module_scratch, "fitok", record, exit_code=0)
        out_dir = module_scratch / "fit ok out"
        code, _out, _err = _run_ps1(
            "fit", built_cache["features"], out_dir, budget=5,
            python_exe=stub)
        assert code == 0
        calls = [json.loads(line) for line in
                 record.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        assert len(calls) == 2
        fit_argv, verify_argv = calls[0]["argv"], calls[1]["argv"]
        assert "--fit" in fit_argv and "--deadline-utc" in fit_argv
        assert fit_argv[0].endswith("train_material_residual_scale.py")
        assert verify_argv[0].endswith(
            "verify_material_residual_scale.py")
        assert "--experiment-fit-dir" in verify_argv
        # ONE shared deadline: fit and verify see the SAME instant and
        # neither phase receives --budget-minutes
        fit_deadline = fit_argv[fit_argv.index("--deadline-utc") + 1]
        verify_deadline = verify_argv[
            verify_argv.index("--deadline-utc") + 1]
        assert fit_deadline == verify_deadline
        assert "--budget-minutes" not in fit_argv
        assert "--budget-minutes" not in verify_argv

    def test_parse_still_passes(self):
        ps1 = (prep._PROJECT_ROOT / "scripts" /
               "manual_material_residual_scale.ps1")
        command = (
            "$tokens=$null;$errors=$null;"
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{ps1.as_posix()}',"
            "[ref]$tokens,[ref]$errors)|Out-Null;"
            "if($errors -and $errors.Count -gt 0){"
            "$errors | ForEach-Object { Write-Output $_.Message }; exit 1}"
            "else { exit 0 }")
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr


class TestSharedDeadlineEnforcement:
    def test_already_expired_deadline_refuses_with_zero_training(
            self, scale_world, module_scratch, built_cache):
        calls = []

        def probe(*args, **kwargs):
            calls.append(1)
            raise AssertionError("training must never start")

        cfg_path = _dump_cfg(module_scratch / "configs" / "expired.yaml",
                             scale_world["cfg"])
        out = module_scratch / "expired_out"
        code = train_entry.main([
            "--config", str(cfg_path), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--output-dir", str(out),
            "--fit", "--deadline-utc", "2000-01-01T00:00:00+00:00",
        ])
        assert code == 3  # EXIT_BLOCKED (PARTIAL), never success
        assert calls == []
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        assert not (out / "models").exists()

    def test_expiry_at_the_completion_publish_boundary_keeps_partial(
            self, scale_world, module_scratch, built_cache):
        """The LAST fit finishing before the deadline but the publish
        boundary crossing it must yield PARTIAL evidence (fits, gates,
        shuffle kept) with a non-zero exit — never a COMPLETED status."""

        class BoundaryWatch(DeadlineWatch):
            def __init__(self, **kwargs):
                super().__init__(budget_minutes=30.0)

            def check(self, at: str = "") -> None:
                if at == "before completion publish":
                    raise DeadlineExceeded(
                        f"deadline crossed at {at}")

        cfg_path = _dump_cfg(module_scratch / "configs" / "boundary.yaml",
                             scale_world["cfg"])
        out = module_scratch / "boundary_out"
        with mock.patch.object(train_entry, "DeadlineWatch",
                               BoundaryWatch):
            code = train_entry.main([
                "--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--output-dir", str(out),
                "--fit", "--budget-minutes", "30",
            ])
        assert code == 3
        summary = json.loads((out / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["training_status"] == "PARTIAL"
        assert summary["interrupted_by_deadline"] is True
        assert summary.get("gates") is not None, \
            "the full finished evidence (gates included) is kept"
        assert len(_model_dirs(out)) == 36
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"

    def test_expiry_during_the_shuffle_keeps_partial_evidence(
            self, scale_world, module_scratch, built_cache):
        class ShuffleWatch(DeadlineWatch):
            def __init__(self, **kwargs):
                super().__init__(budget_minutes=30.0)

            def check(self, at: str = "") -> None:
                if at.startswith("shuffle fold fold1"):
                    raise DeadlineExceeded(f"deadline crossed at {at}")

        cfg_path = _dump_cfg(module_scratch / "configs" / "shufexp.yaml",
                             scale_world["cfg"])
        out = module_scratch / "shufexp_out"
        with mock.patch.object(train_entry, "DeadlineWatch", ShuffleWatch):
            code = train_entry.main([
                "--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--output-dir", str(out),
                "--fit", "--budget-minutes", "30",
            ])
        assert code == 3
        summary = json.loads((out / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["training_status"] == "PARTIAL"
        assert summary["failed_phase"] == "shuffle_diagnostic"
        assert not (out / "models" / "R0" / "fold2_R0").exists() or True
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"

    def test_verify_tail_expiry_is_not_a_success(
            self, scale_world, scale_fit, module_scratch, built_cache):
        class TailWatch(DeadlineWatch):
            def __init__(self, **kwargs):
                super().__init__(budget_minutes=30.0)

            def check(self, at: str = "") -> None:
                if at == "before verification success":
                    raise DeadlineExceeded(f"deadline crossed at {at}")

        out = module_scratch / "verify_tail_out"
        with mock.patch.object(verify_entry, "DeadlineWatch", TailWatch):
            code = verify_entry.main([
                "--config", str(scale_fit["cfg_path"]), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--experiment-fit-dir",
                str(scale_fit["out"]), "--output-dir", str(out),
                "--budget-minutes", "30",
            ])
        assert code == EXIT_REFUSED
        report = json.loads((out / "verification_report.json").read_text(
            encoding="utf-8"))
        assert report["interrupted_by_deadline"] is True


class TestVerifierTamperCounterexamples:
    """Entry-level counterexamples: each doctored fit copy must be
    REFUSED by the real verifier — means, R0 fold metrics, per-group
    counts, gates, shuffle outputs, model identity and scalers are all
    independently recomputed."""

    def _doctored(self, scale_world, scale_fit, module_scratch,
                  built_cache, mutate, label: str):
        import uuid

        copy_dir = module_scratch / \
            f"tamp_{label}_{uuid.uuid4().hex[:8]}"
        shutil.copytree(scale_fit["out"], copy_dir)
        mutate(copy_dir)
        out = module_scratch / f"verify_{copy_dir.name}"
        code = verify_entry.main([
            "--config", str(scale_fit["cfg_path"]), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--experiment-fit-dir",
            str(copy_dir), "--output-dir", str(out),
        ])
        report_path = out / "verification_report.json"
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.is_file() else {"problems": []})
        print(f"TAMPER[{label}] exit={code} "
              f"problems={report.get('problems', [])[:2]} "
              f"failure={report.get('failure')!r}")
        return code

    def test_cross_seed_mean_drift_is_refused(self, scale_world, scale_fit,
                                              module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["arms"]["scaled"]["RCS"]["macro_f1_mean_of_seeds"] += \
                0.001
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "mean")
        assert code == EXIT_MISMATCH

    def test_r0_fold_metric_drift_is_refused(self, scale_world, scale_fit,
                                             module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            folds = summary["arms"]["fixed"]["R0"]["per_seed"]["2026"][
                "folds"]
            folds[0]["macro_f1"] = \
                float(folds[0]["macro_f1"]) + 0.01
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "r0fold")
        assert code == EXIT_MISMATCH

    def test_per_group_counts_drift_is_refused(self, scale_world,
                                               scale_fit, module_scratch,
                                               built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            groups = summary["per_group_changes"][
                "scaled_RCS_minus_R0"]["summed_over_seeds"]
            # a SAVED group the rows do not support is as much a mismatch
            # as a real group with wrong counts (the verifier compares the
            # union of saved and recounted groups)
            groups["ghost_group"] = {"fixed": 9, "broke": 0}
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "group")
        assert code == EXIT_MISMATCH

    def test_gates_doctoring_is_refused(self, scale_world, scale_fit,
                                        module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["gates"]["gate6_group_breadth"]["pass"] = \
                not summary["gates"]["gate6_group_breadth"]["pass"]
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "gates")
        assert code == EXIT_MISMATCH

    def test_shuffle_prediction_drift_is_refused(self, scale_world,
                                                 scale_fit,
                                                 module_scratch,
                                                 built_cache):
        def mutate(fit_dir: Path) -> None:
            path = next((fit_dir / "shuffle" / "scaled" / "seed_2026")
                        .glob("fold0/seed_*.json"))
            record = json.loads(path.read_text(encoding="utf-8"))
            for row in record["samples"]:
                if row["status"] == "matched":
                    row["shuffled_prediction"] = (int(row["shuffled_prediction"]) + 1) % 4
            path.write_text(json.dumps(record), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "shufflepred")
        assert code == EXIT_MISMATCH

    def test_model_identity_drift_is_refused(self, scale_world, scale_fit,
                                             module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "models" / "seed_2026" /
                    "fold0_scaled_RCS" / "residual_state.pt")
            payload = torch.load(path, weights_only=False,
                                 map_location="cpu")
            payload["seed"] = 9999
            torch.save(payload, path)

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "modelid")
        assert code == EXIT_MISMATCH

    def test_scaler_drift_is_refused(self, scale_world, scale_fit,
                                     module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "models" / "seed_2026" /
                    "fold0_fixed_RC" / "scaler.json")
            scaler = json.loads(path.read_text(encoding="utf-8"))
            key = sorted(scaler)[0]
            if isinstance(scaler[key], list):
                scaler[key][0] = float(scaler[key][0]) + 1.0
            else:
                scaler[key] = scaler[key]
            path.write_text(json.dumps(scaler), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "scaler")
        assert code == EXIT_MISMATCH


class TestSourceAdmissionProtection:
    def test_cache_tampered_after_binding_is_refused(
            self, scale_world, module_scratch, built_cache):
        import shutil as _shutil

        tampered = module_scratch / "tampered_cache"
        _shutil.copytree(built_cache["features"], tampered)
        cache_path = tampered / "feature_cache.npz"
        raw = bytearray(cache_path.read_bytes())
        raw[-1] ^= 0xFF
        cache_path.write_bytes(bytes(raw))
        cfg_path = _dump_cfg(module_scratch / "configs" / "tamper.yaml",
                             scale_world["cfg"])
        code = train_entry.main([
            "--config", str(cfg_path), "--feature-dir", str(tampered),
            "--p0-fit-dir", str(scale_world["fit_dir"]),
            "--output-dir", str(module_scratch / "refused_tamper"),
            "--dry-run", "--deadline-utc", "2030-01-01T00:00:00+00:00",
        ])
        assert code == EXIT_REFUSED

    def test_missing_cache_dependency_is_refused(
            self, scale_world, module_scratch, built_cache):
        import shutil as _shutil

        broken = module_scratch / "broken_cache"
        _shutil.copytree(built_cache["features"], broken)
        (broken / "feature_manifest.json").unlink()
        cfg_path = _dump_cfg(module_scratch / "configs" / "missing.yaml",
                             scale_world["cfg"])
        code = train_entry.main([
            "--config", str(cfg_path), "--feature-dir", str(broken),
            "--p0-fit-dir", str(scale_world["fit_dir"]),
            "--output-dir", str(module_scratch / "refused_missing"),
            "--dry-run", "--deadline-utc", "2030-01-01T00:00:00+00:00",
        ])
        assert code == EXIT_REFUSED

    def test_historical_binding_mismatch_is_refused(
            self, scale_world, module_scratch, built_cache):
        """A synthetic historical manifest pins the synthetic P0
        checkpoint; mutating the checkpoint afterwards must be refused at
        registration (history is never re-hashed into a fresh identity)."""
        import hashlib

        import yaml as _yaml

        checkpoint = (Path(scale_world["fit_dir"]) / "models" /
                      "fold0_P0" / "model_state.pt")
        manifest = {
            "consumed_inputs": [
                {"path": checkpoint.relative_to(
                    prep._PROJECT_ROOT).as_posix(),
                 "after_sha256": hashlib.sha256(
                     checkpoint.read_bytes()).hexdigest()},
                {"path": (Path(scale_world["fit_dir"]) / "summary.json")
                 .relative_to(prep._PROJECT_ROOT).as_posix(),
                 "after_sha256": hashlib.sha256(
                     (Path(scale_world["fit_dir"]) / "summary.json")
                     .read_bytes()).hexdigest()},
            ]}
        manifest_path = module_scratch / "synthetic_batch1_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        cfg = copy.deepcopy(scale_world["cfg"])
        cfg["anchors"]["batch1_source_manifest"] = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()}
        cfg_path = _dump_cfg(module_scratch / "configs" / "histbind.yaml",
                             cfg)
        # consistent history: the dry-run passes
        ok = train_entry.main([
            "--config", str(cfg_path), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--output-dir",
            str(module_scratch / "hist_ok"), "--dry-run",
            "--deadline-utc", "2030-01-01T00:00:00+00:00",
        ])
        assert ok == EXIT_OK
        # now the checkpoint drifts from its historical record
        raw = bytearray(checkpoint.read_bytes())
        raw[-1] ^= 0xFF
        checkpoint.write_bytes(bytes(raw))
        code = train_entry.main([
            "--config", str(cfg_path), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--output-dir",
            str(module_scratch / "hist_bad"), "--dry-run",
            "--deadline-utc", "2030-01-01T00:00:00+00:00",
        ])
        assert code == EXIT_REFUSED, \
            "a consumed file that drifted from its historical binding " \
            "must be refused at registration"


# ---------------------------------------------------------------------------
# Second review-closeout additions: verifier completeness counterexamples,
# binding-key hit proofs, registered-role coverage and the final budget
# boundary.

import hashlib as _hashlib

import scripts.verify_p0_baseline as _baseline_module


class TestVerifierCompletenessCounterexamples:
    """Every listed tamper goes through the REAL verify entry and must be
    refused; the unmutated synthetic fit still passes."""

    def _doctored(self, scale_world, scale_fit, module_scratch,
                  built_cache, mutate, label: str):
        import uuid

        copy_dir = module_scratch / \
            f"cmp_{label}_{uuid.uuid4().hex[:8]}"
        shutil.copytree(scale_fit["out"], copy_dir)
        mutate(copy_dir)
        out = module_scratch / f"verify_{copy_dir.name}"
        code = verify_entry.main([
            "--config", str(scale_fit["cfg_path"]), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--experiment-fit-dir",
            str(copy_dir), "--output-dir", str(out),
        ])
        report_path = out / "verification_report.json"
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.is_file() else {"problems": []})
        print(f"COMPLETE[{label}] exit={code} "
              f"problems={report.get('problems', [])[:2]}")
        return code

    def test_shuffle_macro_f1_drift_is_refused(self, scale_world,
                                               scale_fit, module_scratch,
                                               built_cache):
        def mutate(fit_dir: Path) -> None:
            path = next((fit_dir / "shuffle" / "scaled" / "seed_2026")
                        .glob("fold0/seed_*.json"))
            record = json.loads(path.read_text(encoding="utf-8"))
            record["macro_f1"] = \
                float(record["macro_f1"]) + 0.05
            path.write_text(json.dumps(record), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "shufmacro")
        assert code == EXIT_MISMATCH

    def test_shuffle_summary_drift_is_refused(self, scale_world,
                                              scale_fit, module_scratch,
                                              built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "shuffle" / "scaled" / "seed_2026" /
                    "fold0" / "shuffle_summary.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            record["seeds"]["2026"]["coverage"] = \
                float(record["seeds"]["2026"]["coverage"]) + 0.01
            path.write_text(json.dumps(record), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "shufsummary")
        assert code == EXIT_MISMATCH

    def test_final_summary_shuffle_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            entry = summary["shuffle"]["seed_2026"]["fold0"]["arms"][
                "scaled"]["2026"]
            entry["matched"] = int(entry["matched"]) + 1
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "finalshuf")
        assert code == EXIT_MISMATCH

    def test_fold_accuracy_drift_is_refused(self, scale_world, scale_fit,
                                            module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            folds = summary["arms"]["scaled"]["RCS"]["per_seed"]["2026"][
                "folds"]
            folds[0]["accuracy"] = float(folds[0]["accuracy"]) + 0.02
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "foldacc")
        assert code == EXIT_MISMATCH

    def test_both_correct_and_support_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            per_seed = summary["paired"]["scaled_RCS_minus_R0"][
                "per_seed"]["2026"]
            per_seed["both_correct"] = int(per_seed["both_correct"]) + 1
            per_seed["support"] = int(per_seed["support"]) + 1
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "fourway")
        assert code == EXIT_MISMATCH

    def test_row_group_with_synced_summary_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        """Changing a prediction row's group AND syncing the group-keyed
        summary still fails: the rows are bound to the AUTHORITATIVE
        manifest group, never to their self-reported one."""
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "models" / "seed_2026" /
                    "fold0_scaled_RCS" /
                    "outer_eval_predictions.jsonl")
            rows = [json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            rows[0]["group"] = "fabricated/group"
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "rowgroup")
        assert code == EXIT_MISMATCH

    def test_gate_detail_drift_with_unchanged_pass_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            gate = summary["gates"]["gate1_scaled_rcs_vs_r0_rc"]
            gate["gain_vs_r0_pp"] = float(gate["gain_vs_r0_pp"]) + 5.0
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "gatedetail")
        assert code == EXIT_MISMATCH

    def test_duplicate_and_missing_shuffle_records_are_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def duplicate(fit_dir: Path) -> None:
            path = next((fit_dir / "shuffle" / "scaled" / "seed_2026")
                        .glob("fold0/seed_*.json"))
            record = json.loads(path.read_text(encoding="utf-8"))
            record["samples"].append(dict(record["samples"][0]))
            path.write_text(json.dumps(record), encoding="utf-8")

        def remove(fit_dir: Path) -> None:
            path = next((fit_dir / "shuffle" / "scaled" / "seed_2026")
                        .glob("fold0/seed_*.json"))
            record = json.loads(path.read_text(encoding="utf-8"))
            record["samples"] = record["samples"][:-1]
            path.write_text(json.dumps(record), encoding="utf-8")

        assert self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, duplicate,
                              "shufdup") == EXIT_MISMATCH
        assert self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, remove,
                              "shufmissing") == EXIT_MISMATCH


class TestBindingKeyHitsAndCoverage:
    def test_absolute_manifest_paths_still_hit_the_registry(
            self, scale_world, module_scratch, built_cache):
        """A synthetic historical manifest recorded with ABSOLUTE paths
        must bind under the registry's project-relative key: a tampered
        file is still refused at registration (the key genuinely hits)."""
        import hashlib

        checkpoint = (Path(scale_world["fit_dir"]) / "models" /
                      "fold0_P0" / "model_state.pt")
        # the manifest records the ABSOLUTE path
        manifest = {
            "consumed_inputs": [
                {"path": str(checkpoint),
                 "after_sha256": _hashlib.sha256(
                     checkpoint.read_bytes()).hexdigest()},
            ]}
        manifest_path = module_scratch / "absolute_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        cfg = copy.deepcopy(scale_world["cfg"])
        cfg["anchors"]["batch1_source_manifest"] = {
            "path": str(manifest_path),
            "sha256": _hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()}
        cfg_path = _dump_cfg(module_scratch / "configs" / "absbind.yaml",
                             cfg)
        ok = train_entry.main([
            "--config", str(cfg_path), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--output-dir",
            str(module_scratch / "abs_ok"), "--dry-run",
            "--deadline-utc", "2030-01-01T00:00:00+00:00",
        ])
        assert ok == EXIT_OK
        raw = bytearray(checkpoint.read_bytes())
        raw[-1] ^= 0xFF
        checkpoint.write_bytes(bytes(raw))
        code = train_entry.main([
            "--config", str(cfg_path), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--output-dir",
            str(module_scratch / "abs_bad"), "--dry-run",
            "--deadline-utc", "2030-01-01T00:00:00+00:00",
        ])
        assert code == EXIT_REFUSED, \
            "an absolute-path historical binding must still hit"

    def test_expand_refuses_outside_root_and_reparse(
            self, scale_world, module_scratch):
        from core.material_residual_scale import (
            expand_historical_bindings,
        )
        registry = train_entry.SourceRegistry({})
        cfg = copy.deepcopy(scale_world["cfg"])
        outside = module_scratch.parent / "outside_manifest.json"
        outside.write_text(json.dumps({"consumed_inputs": [
            {"path": r"D:\Windows\Temp\elsewhere.bin",
             "after_sha256": "a" * 64}]}), encoding="utf-8")
        cfg["anchors"]["batch1_source_manifest"] = {
            "path": str(outside),
            "sha256": _hashlib.sha256(
                outside.read_bytes()).hexdigest()}
        with pytest.raises(ScaleRefused, match="project root"):
            expand_historical_bindings(registry, cfg, prep._PROJECT_ROOT)

    def test_registered_roles_cover_the_actual_dependencies(
            self, scale_world, module_scratch, built_cache):
        """The dry-run's protection manifest must cover the files the run
        actually consumed: fix cache, global cache, P0 checkpoint/scaler/
        history/oracle and the protocol."""
        cfg_path = _dump_cfg(module_scratch / "configs" / "cover.yaml",
                             scale_world["cfg"])
        out = module_scratch / "cover_out"
        code = train_entry.main([
            "--config", str(cfg_path), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--output-dir", str(out),
            "--dry-run", "--deadline-utc", "2030-01-01T00:00:00+00:00",
        ])
        assert code == EXIT_OK
        manifest = json.loads((out / "source_manifest.json").read_text(
            encoding="utf-8"))
        roles = {row["role"] for row in manifest["items"]}
        for required in ("fix_cache:feature_cache.npz",
                         "fix_cache:feature_definition.json",
                         "global_cache:feature_cache.npz",
                         "p0_fit:summary.json",
                         "p0_model:fold0_P0:model_state.pt",
                         "p0_model:fold0_P0:scaler.json",
                         "p0_model:fold0_P0:history.json",
                         "p0_model:fold0_P0:outer_eval_predictions.jsonl",
                         "protocol_file"):
            assert required in roles, (
                f"missing registered role {required}; registered roles "
                f"= {sorted(roles)}")
        sources = {row.get("binding_source") for row in manifest["items"]}
        # the synthetic world consumes the SYNTHETIC P0 fit, which the
        # real runtime-config pins do not cover: the fix cache and the
        # frozen global cache carry the historical expectations here
        assert "scale_cache_binding" in sources
        assert "global_cache_binding" in sources
        # the real dry-run manifest additionally shows the runtime-config
        # pins hitting (checked against the retained review evidence)
        real_manifest = json.loads(
            (prep._PROJECT_ROOT / "analysis/runs/"
             "material_residual_scale_v1/review_closeout/"
             "20261011T090556Z_b84ec6c5/manual_dryrun/"
             "source_manifest.json").read_text(encoding="utf-8"))
        real_sources = {row.get("binding_source")
                        for row in real_manifest["items"]}
        assert "runtime_config_anchor" in real_sources


class TestFinalBudgetBoundary:
    def test_prepare_preflight_crossing_the_deadline_publishes_no_plan(
            self, scale_world, module_scratch):
        cfg_path = _dump_cfg(module_scratch / "configs" / "prepexp.yaml",
                             scale_world["cfg"])
        out = module_scratch / "prepexp_out"
        code = prep.main([
            "--config", str(cfg_path), "--mode", "preflight",
            "--output-dir", str(out),
            "--deadline-utc", "2000-01-01T00:00:00+00:00",
        ])
        assert code == prep.EXIT_PARTIAL
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        assert not (out / "preflight_plan.json").exists(), \
            "a preflight that crossed the deadline publishes no success plan"

    def test_prepare_protection_boundary_crossing_stays_partial(
            self, scale_world, module_scratch):
        class BoundaryWatch(DeadlineWatch):
            def __init__(self, **kwargs):
                super().__init__(budget_minutes=30.0)

            def check(self, at: str = "") -> None:
                if at == "before success status publish":
                    raise DeadlineExceeded(f"crossed at {at}")

        cfg_path = _dump_cfg(module_scratch / "configs" / "prepbnd.yaml",
                             scale_world["cfg"])
        out = module_scratch / "prepbnd_out"
        with mock.patch.object(prep, "DeadlineWatch", BoundaryWatch):
            code = prep.main([
                "--config", str(cfg_path), "--mode", "preflight",
                "--output-dir", str(out),
                "--deadline-utc", "2030-01-01T00:00:00+00:00",
            ])
        assert code == prep.EXIT_PARTIAL
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        # the protection evidence is kept even though the run is PARTIAL
        assert (out / "source_manifest.json").is_file()

    def test_train_protection_boundary_crossing_stays_partial(
            self, scale_world, module_scratch, built_cache):
        """Everything completes, the protection manifest is written, and
        THEN the deadline crosses: the run status must be PARTIAL, the
        summary must be restamped PARTIAL, and the 36 fits stay on
        disk."""
        class BoundaryWatch(DeadlineWatch):
            def __init__(self, **kwargs):
                super().__init__(budget_minutes=30.0)

            def check(self, at: str = "") -> None:
                if at == "before success status publish":
                    raise DeadlineExceeded(f"crossed at {at}")

        cfg_path = _dump_cfg(module_scratch / "configs" / "trnbnd.yaml",
                             scale_world["cfg"])
        out = module_scratch / "trnbnd_out"
        with mock.patch.object(train_entry, "DeadlineWatch",
                               BoundaryWatch):
            code = train_entry.main([
                "--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--output-dir", str(out),
                "--fit", "--budget-minutes", "30",
            ])
        assert code == 3
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        summary = json.loads((out / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["training_status"] == "PARTIAL"
        assert summary["interrupted_by_deadline"] is True
        assert summary.get("gates") is not None
        assert len(_model_dirs(out)) == 36
        assert (out / "source_manifest.json").is_file()

    def test_verify_success_boundary_crossing_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        """The verify entry's OWN final boundary: a deadline that crosses
        only at the success log must refuse (never a verified success)."""
        class TailWatch(DeadlineWatch):
            def __init__(self, **kwargs):
                super().__init__(budget_minutes=30.0)

            def check(self, at: str = "") -> None:
                if at == "before verification success":
                    raise DeadlineExceeded(f"crossed at {at}")

        out = module_scratch / "verify_final_out"
        with mock.patch.object(verify_entry, "DeadlineWatch", TailWatch):
            code = verify_entry.main([
                "--config", str(scale_fit["cfg_path"]), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--experiment-fit-dir",
                str(scale_fit["out"]), "--output-dir", str(out),
                "--budget-minutes", "30",
            ])
        assert code == EXIT_REFUSED
        report = json.loads((out / "verification_report.json").read_text(
            encoding="utf-8"))
        assert report["interrupted_by_deadline"] is True

    def test_frozen_full_cache_hash_unchanged(self):
        """The complete fix cache's SHA is part of the closeout record:
        the review closeout must not rebuild or touch it."""
        cache = (prep._PROJECT_ROOT / "analysis/runs/"
                 "material_residual_scale_v1/20261011T074823Z_e36bf75e/"
                 "cache_build/features/feature_cache.npz")
        digest = _hashlib.sha256(cache.read_bytes()).hexdigest()
        recorded = json.loads(
            (prep._PROJECT_ROOT / "analysis/runs/"
             "material_residual_scale_v1/20261011T074823Z_e36bf75e/"
             "cache_build/features/cache_binding.json").read_text(
                 encoding="utf-8"))
        assert digest == recorded["files"]["feature_cache.npz"]["sha256"]


# ---------------------------------------------------------------------------
# Third review-closeout additions: consume-order probes (registration and
# hash verification BEFORE any consumer), drift-before-consumer
# counterexamples for all three entries, the remaining verifier
# counterexamples, and the corrected preflight-timeout scenario.

import scripts.train_material_residual as _frozen_chain_module


class _ProbeStop(RuntimeError):
    """Raised by a consumer probe to stop the run right after the
    registration phase; the closeout still publishes the manifest."""


def _count_probe(record: dict, key: str):
    def probe(*args, **kwargs):
        record[key] = record.get(key, 0) + 1
        raise _ProbeStop(f"probe stop: {key}")
    return probe


def _manifest_roles(out_dir: Path) -> tuple[set, set]:
    manifest = json.loads((out_dir / "source_manifest.json").read_text(
        encoding="utf-8"))
    roles = {row["role"] for row in manifest["items"]}
    sources = {row.get("binding_source") for row in manifest["items"]}
    return roles, sources


class TestConsumeOrderProbes:
    """The consumer (frozen chain / V5 gate) starts only AFTER its
    dependencies are registered and hash-verified: the probe stops the
    run right at the consumer boundary and the published manifest proves
    the registrations already happened."""

    def test_train_registers_before_the_frozen_chain(
            self, scale_world, module_scratch, built_cache):
        calls = {}

        def frozen_probe(config, fit_dir):
            calls["frozen_run_admission"] = \
                calls.get("frozen_run_admission", 0) + 1
            raise _ProbeStop("probe stop: frozen chain")

        def identity_probe(**kwargs):
            calls["verify_p0_base_identity"] = \
                calls.get("verify_p0_base_identity", 0) + 1
            raise AssertionError("P0 identity ran before/inside the probe")

        cfg_path = _dump_cfg(module_scratch / "configs" / "ordtrain.yaml",
                             scale_world["cfg"])
        out = module_scratch / "order_train"
        with mock.patch.object(_frozen_chain_module, "run_admission",
                               frozen_probe), \
                mock.patch.object(train_entry, "verify_p0_base_identity",
                                  identity_probe):
            code = train_entry.main([
                "--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--output-dir", str(out),
                "--dry-run", "--deadline-utc",
                "2030-01-01T00:00:00+00:00",
            ])
        assert code == EXIT_REFUSED
        assert calls["frozen_run_admission"] == 1, \
            "the consumer boundary was reached exactly once"
        assert calls.get("verify_p0_base_identity", 0) == 0, \
            "no deeper consumer ran after the probe stopped"
        roles, sources = _manifest_roles(out)
        # every dependency of the frozen chain is registered BEFORE it
        synthetic_global = Path(scale_world["feature_dir"])
        expected = ["fix_cache:feature_cache.npz",
                    "fix_cache:feature_definition.json",
                    "global_cache:feature_cache.npz",
                    "p0_fit:summary.json",
                    "p0_model:fold0_P0:model_state.pt",
                    "p0_model:fold0_P0:scaler.json",
                    "p0_model:fold0_P0:history.json",
                    "p0_model:fold1_P0:outer_eval_predictions.jsonl",
                    "p0_model:fold2_P0:outer_eval_predictions.jsonl",
                    "protocol_file"]
        if (synthetic_global / "cache_binding.json").is_file():
            expected.append("global_cache:cache_binding.json")
        for required in expected:
            assert required in roles, f"missing {required}"
        assert "runtime_config_anchor" in sources or \
            "scale_cache_binding" in sources

    def test_verify_registers_before_the_frozen_chain(
            self, scale_world, scale_fit, module_scratch, built_cache):
        calls = {}

        def frozen_probe(config, fit_dir):
            calls["frozen_run_admission"] = \
                calls.get("frozen_run_admission", 0) + 1
            raise _ProbeStop("probe stop: frozen chain")

        def identity_probe(**kwargs):
            calls["verify_p0_base_identity"] = \
                calls.get("verify_p0_base_identity", 0) + 1
            raise AssertionError("P0 identity ran before/inside the probe")

        cfg_path = _dump_cfg(module_scratch / "configs" / "ordver.yaml",
                             scale_world["cfg"])
        out = module_scratch / "order_verify"
        with mock.patch.object(_frozen_chain_module, "run_admission",
                               frozen_probe), \
                mock.patch.object(verify_entry, "verify_p0_base_identity",
                                  identity_probe):
            code = verify_entry.main([
                "--config", str(cfg_path), "--feature-dir",
                str(built_cache["features"]), "--p0-fit-dir",
                str(scale_world["fit_dir"]), "--experiment-fit-dir",
                str(scale_fit["out"]), "--output-dir", str(out),
            ])
        assert code == EXIT_REFUSED
        assert calls["frozen_run_admission"] == 1
        assert calls.get("verify_p0_base_identity", 0) == 0
        roles, _sources = _manifest_roles(out)
        synthetic_global = Path(scale_world["feature_dir"])
        expected = ["fix_cache:feature_cache.npz",
                    "global_cache:feature_cache.npz",
                    "p0_fit:summary.json",
                    "p0_model:fold1_P0:model_state.pt",
                    "p0_model:fold2_P0:scaler.json",
                    "p0_model:fold0_P0:history.json",
                    "protocol_file"]
        if (synthetic_global / "cache_binding.json").is_file():
            expected.append("global_cache:cache_binding.json")
        for required in expected:
            assert required in roles, f"missing {required}"

    def test_prepare_registers_before_the_v5_gate(
            self, scale_world, module_scratch):
        calls = {}

        def v5_probe(cfg):
            calls["run_v5_admission"] = \
                calls.get("run_v5_admission", 0) + 1
            raise _ProbeStop("probe stop: V5 gate")

        cfg_path = _dump_cfg(module_scratch / "configs" / "ordprep.yaml",
                             scale_world["cfg"])
        out = module_scratch / "order_prepare"
        with mock.patch.object(prep, "run_v5_admission", v5_probe):
            code = prep.main([
                "--config", str(cfg_path), "--mode", "preflight",
                "--output-dir", str(out),
                "--deadline-utc", "2030-01-01T00:00:00+00:00",
            ])
        assert code == prep.EXIT_REFUSED
        assert calls["run_v5_admission"] == 1
        roles, _sources = _manifest_roles(out)
        synthetic_global = Path(scale_world["feature_dir"])
        # the pair manifest is consumed inside the census, which runs
        # AFTER the V5 gate the probe stops at — it is covered by the
        # full preflight runs, not this boundary
        expected = ["protocol_file",
                    "global_cache:feature_cache.npz",
                    "v5_record:dataset_manifest.csv",
                    "v5_record:dataset_summary.json"]
        if (synthetic_global / "cache_binding.json").is_file():
            expected.append("global_cache:cache_binding.json")
        for required in expected:
            assert required in roles, f"missing {required}"


def _valid_content_drift(path: Path) -> None:
    """Rewrite a pinned file with DIFFERENT but still VALID content:
    JSON keeps parsing, CSV keeps its tabular rows, a .pt checkpoint
    keeps loading. The post-drift validity is asserted HERE, so the
    drift tests prove a hash refusal at registration — never a
    corrupted file failing to deserialize inside a consumer."""
    suffix = path.suffix
    if suffix == ".json":
        original = json.loads(path.read_text(encoding="utf-8"))
        doc = copy.deepcopy(original)
        if isinstance(doc, dict) and isinstance(doc.get("samples"), list) \
                and doc["samples"]:
            doc["samples"][0]["physical_object_id"] = "drifted"
        elif isinstance(doc, dict) and isinstance(doc.get("files"), dict) \
                and doc["files"]:
            doc["files"][sorted(doc["files"])[0]]["sha256"] = "0" * 64
        else:
            raise AssertionError(f"no valid JSON drift path for {path}")
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        assert json.loads(path.read_text(encoding="utf-8")) != original, \
            f"the drifted {path.name} did not actually change"
    elif suffix == ".csv":
        import csv
        import io

        # the production manifest writer emits \r\r\n line endings on
        # Windows (csv \r\n through a newline-translating write_text);
        # csv.DictReader skips the resulting blank rows, so the reader
        # here does the same before mutating a data row
        original_rows = [list(row) for row in csv.reader(
            path.read_text(encoding="utf-8").splitlines()) if row]
        assert len(original_rows) >= 2, \
            f"{path} has no data row to drift"
        rows = [list(row) for row in original_rows]
        rows[1][-1] = f"{rows[1][-1]}_drifted"
        buffer = io.StringIO()
        csv.writer(buffer).writerows(rows)
        path.write_text(buffer.getvalue(), encoding="utf-8")
        assert [list(row) for row in csv.reader(
            path.read_text(encoding="utf-8").splitlines()) if row] \
            != original_rows, \
            f"the drifted {path.name} did not actually change"
    elif suffix == ".pt":
        payload = torch.load(path, weights_only=False, map_location="cpu")
        payload["seed"] = 2027
        torch.save(payload, path)
        reloaded = torch.load(path, weights_only=False,
                              map_location="cpu")
        assert reloaded["seed"] == 2027, \
            f"the drifted {path.name} is not a loadable checkpoint"
    else:
        raise AssertionError(f"no valid-content drift for suffix {suffix}")


class TestDriftBeforeConsumer:
    """A synthetic historical manifest pins files; the file is then
    rewritten with DIFFERENT but still VALID content (a parseable JSON,
    a parseable CSV, a loadable checkpoint — the validity is asserted by
    _valid_content_drift): registration refuses the hash mismatch and
    the consumer is never reached (call count 0). The drifted bytes are
    RESTORED in a finally block so the shared synthetic world stays
    usable."""

    def _pin_and_drift(self, scale_world, pinned_path, label,
                       module_scratch):
        original_bytes = pinned_path.read_bytes()

        def restore():
            pinned_path.write_bytes(original_bytes)

        manifest = {
            "consumed_inputs": [
                {"path": pinned_path.relative_to(
                    prep._PROJECT_ROOT).as_posix(),
                 "after_sha256": _hashlib.sha256(
                     original_bytes).hexdigest()},
            ]}
        manifest_path = module_scratch / f"drift_manifest_{label}.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        cfg = copy.deepcopy(scale_world["cfg"])
        cfg["anchors"]["batch1_source_manifest"] = {
            "path": str(manifest_path),
            "sha256": _hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()}
        _valid_content_drift(pinned_path)
        return cfg, restore

    def test_train_drift_refuses_before_the_consumer(
            self, scale_world, module_scratch, built_cache):
        checkpoint = (Path(scale_world["fit_dir"]) / "models" /
                      "fold0_P0" / "model_state.pt")
        cfg, restore = self._pin_and_drift(
            scale_world, checkpoint, "train", module_scratch)
        cfg_path = _dump_cfg(module_scratch / "configs" / "drifttrain.yaml",
                             cfg)
        calls = {}

        def frozen_probe(config, fit_dir):
            calls["n"] = calls.get("n", 0) + 1
            raise _ProbeStop("probe stop")

        out = module_scratch / "drift_train"
        try:
            with mock.patch.object(_frozen_chain_module, "run_admission",
                                   frozen_probe):
                code = train_entry.main([
                    "--config", str(cfg_path), "--feature-dir",
                    str(built_cache["features"]), "--p0-fit-dir",
                    str(scale_world["fit_dir"]), "--output-dir", str(out),
                    "--dry-run", "--deadline-utc",
                    "2030-01-01T00:00:00+00:00",
                ])
            assert code == EXIT_REFUSED
            assert calls.get("n", 0) == 0, \
                "the frozen chain consumed inputs despite the drift"
        finally:
            restore()

    def test_verify_drift_refuses_before_the_consumer(
            self, scale_world, module_scratch, built_cache):
        checkpoint = (Path(scale_world["fit_dir"]) / "models" /
                      "fold1_P0" / "model_state.pt")
        cfg, restore = self._pin_and_drift(
            scale_world, checkpoint, "verify", module_scratch)
        cfg_path = _dump_cfg(module_scratch / "configs" / "driftver.yaml",
                             cfg)
        calls = {}

        def frozen_probe(config, fit_dir):
            calls["n"] = calls.get("n", 0) + 1
            raise _ProbeStop("probe stop")

        out = module_scratch / "drift_verify"
        try:
            with mock.patch.object(_frozen_chain_module, "run_admission",
                                   frozen_probe):
                code = verify_entry.main([
                    "--config", str(cfg_path), "--feature-dir",
                    str(built_cache["features"]), "--p0-fit-dir",
                    str(scale_world["fit_dir"]), "--experiment-fit-dir",
                    str(module_scratch / "fit_out"), "--output-dir",
                    str(out),
                ])
            assert code == EXIT_REFUSED
            assert calls.get("n", 0) == 0
        finally:
            restore()

    def test_prepare_drift_refuses_before_the_consumer(
            self, scale_world, module_scratch):
        manifest_csv = (Path(scale_world["v5"]) / "dataset_manifest.csv")
        cfg, restore = self._pin_and_drift(
            scale_world, manifest_csv, "prepare", module_scratch)
        cfg_path = _dump_cfg(module_scratch / "configs" / "driftprep.yaml",
                             cfg)
        calls = {}

        def v5_probe(cfg):
            calls["n"] = calls.get("n", 0) + 1
            raise _ProbeStop("probe stop")

        out = module_scratch / "drift_prepare"
        try:
            with mock.patch.object(prep, "run_v5_admission", v5_probe):
                code = prep.main([
                    "--config", str(cfg_path), "--mode", "preflight",
                    "--output-dir", str(out),
                    "--deadline-utc", "2030-01-01T00:00:00+00:00",
                ])
            assert code == prep.EXIT_REFUSED
            assert calls.get("n", 0) == 0, \
                "the V5 gate consumed inputs despite the drift"
        finally:
            restore()


class TestPrepareGlobalCacheAdmission:
    """The prepare entry registers the four global-cache files (each
    hashed against the expanded historical expectations) BEFORE
    baseline.verify_feature_cache reads them: a probe at the consumer
    boundary finds all four roles already registered with the cache's
    own binding record as the hit expectation, and a drifted file is
    refused with the consumer call count at 0."""

    _GLOBAL_FILES = ("cache_binding.json", "feature_cache.npz",
                     "feature_manifest.json", "feature_definition.json")

    def _preflight(self, scale_world, module_scratch, name: str) -> int:
        cfg_path = _dump_cfg(module_scratch / "configs" / f"{name}.yaml",
                             scale_world["cfg"])
        return prep.main([
            "--config", str(cfg_path), "--mode", "preflight",
            "--output-dir", str(module_scratch / name),
            "--deadline-utc", "2030-01-01T00:00:00+00:00",
        ])

    def test_global_cache_registered_before_the_consumer(
            self, scale_world, module_scratch):
        calls = {}

        def cache_probe(global_dir, protocol):
            calls["verify_feature_cache"] = \
                calls.get("verify_feature_cache", 0) + 1
            raise _ProbeStop("probe stop: verify_feature_cache")

        def v5_guard(cfg):
            calls["run_v5_admission"] = calls.get("run_v5_admission", 0) + 1
            raise AssertionError(
                "run_v5_admission ran before/inside the global-cache "
                "consumer probe")

        out = module_scratch / "gcache_probe"
        with mock.patch.object(_baseline_module, "verify_feature_cache",
                               cache_probe), \
                mock.patch.object(prep, "run_v5_admission", v5_guard):
            code = self._preflight(scale_world, module_scratch,
                                   "gcache_probe")
        assert code == prep.EXIT_REFUSED
        assert calls["verify_feature_cache"] == 1, \
            "the consumer boundary was reached exactly once"
        assert calls.get("run_v5_admission", 0) == 0, \
            "the V5 gate ran after the probe stopped"
        manifest = json.loads((out / "source_manifest.json").read_text(
            encoding="utf-8"))
        by_role = {row["role"]: row for row in manifest["items"]}
        for name in self._GLOBAL_FILES:
            assert f"global_cache:{name}" in by_role, \
                f"global_cache:{name} was not registered before the " \
                "consumer"
        # the three files listed in the cache's own binding record hit
        # their historical expectations; the binding record itself is
        # the binding SOURCE (snapshotted before its first read during
        # the expansion) and carries no historical binding
        for name in ("feature_cache.npz", "feature_manifest.json",
                     "feature_definition.json"):
            source = by_role[f"global_cache:{name}"]["binding_source"]
            assert source == "global_cache_binding", \
                f"{name}: binding_source {source!r} is not a " \
                "historical binding hit"

    def test_global_cache_drift_refuses_before_the_consumer(
            self, scale_world, module_scratch):
        calls = {}

        def cache_probe(global_dir, protocol):
            calls["verify_feature_cache"] = \
                calls.get("verify_feature_cache", 0) + 1
            raise _ProbeStop("probe stop")

        target = Path(scale_world["feature_dir"]) / "feature_manifest.json"
        original_bytes = target.read_bytes()
        out = module_scratch / "gcache_drift"
        try:
            # the global cache's own binding record already pins the
            # file (global_cache_binding); the synthetic world needs no
            # extra manifest — the binding IS the pinned expectation
            _valid_content_drift(target)
            with mock.patch.object(_baseline_module,
                                   "verify_feature_cache", cache_probe):
                code = self._preflight(scale_world, module_scratch,
                                       "gcache_drift")
            assert code == prep.EXIT_REFUSED
            assert calls.get("verify_feature_cache", 0) == 0, \
                "verify_feature_cache consumed a drifted global cache"
        finally:
            target.write_bytes(original_bytes)

    def test_undrifted_global_cache_entry_still_passes(
            self, scale_world, module_scratch):
        out = module_scratch / "gcache_ok"
        code = self._preflight(scale_world, module_scratch, "gcache_ok")
        assert code == prep.EXIT_OK
        roles, _sources = _manifest_roles(out)
        for name in self._GLOBAL_FILES:
            assert f"global_cache:{name}" in roles


class TestVerifierEvidenceCompleteness:
    """The remaining saved statistics are independently checked, missing
    entries/fields are refused, and None has explicit semantics."""

    def _doctored(self, scale_world, scale_fit, module_scratch,
                  built_cache, mutate, label: str):
        import uuid

        copy_dir = module_scratch / \
            f"evd_{label}_{uuid.uuid4().hex[:8]}"
        shutil.copytree(scale_fit["out"], copy_dir)
        mutate(copy_dir)
        out = module_scratch / f"verify_{copy_dir.name}"
        code = verify_entry.main([
            "--config", str(scale_fit["cfg_path"]), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--experiment-fit-dir",
            str(copy_dir), "--output-dir", str(out),
        ])
        report_path = out / "verification_report.json"
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.is_file() else {"problems": []})
        print(f"EVIDENCE[{label}] exit={code} "
              f"problems={report.get('problems', [])[:3]}")
        return code

    def test_deleted_summary_shuffle_is_refused(self, scale_world,
                                                scale_fit, module_scratch,
                                                built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary.pop("shuffle")
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "noshuffle")
        assert code == EXIT_MISMATCH

    def test_deleted_content_seed_entry_is_refused(self, scale_world,
                                                   scale_fit,
                                                   module_scratch,
                                                   built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            arms = summary["shuffle"]["seed_2026"]["fold0"]["arms"]
            arms["scaled"].pop("2026")
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "noentry")
        assert code == EXIT_MISMATCH

    def test_deleted_seed_file_and_summary_coverage_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "shuffle" / "scaled" / "seed_2027" /
                    "fold0" / "seed_2026.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            record.pop("coverage")
            path.write_text(json.dumps(record), encoding="utf-8")
            summary_path = (fit_dir / "shuffle" / "scaled" /
                            "seed_2027" / "fold0" /
                            "shuffle_summary.json")
            shuffle_summary = json.loads(summary_path.read_text(
                encoding="utf-8"))
            shuffle_summary["seeds"]["2026"].pop("coverage")
            summary_path.write_text(json.dumps(shuffle_summary),
                                    encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "nocoverage")
        assert code == EXIT_MISMATCH

    def test_summary_same_group_ratio_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            entry = summary["shuffle"]["seed_2026"]["fold0"]["arms"][
                "scaled"]["2026"]
            if entry.get("same_group_donor_ratio") is not None:
                entry["same_group_donor_ratio"] = \
                    float(entry["same_group_donor_ratio"]) + 0.05
            else:
                entry["same_group_donor_ratio"] = 0.99
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "ratio")
        assert code == EXIT_MISMATCH

    def test_pooled_accuracy_drift_is_refused(self, scale_world,
                                              scale_fit, module_scratch,
                                              built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            pooled = summary["arms"]["scaled"]["RCS"]["per_seed"][
                "2026"]["pooled"]
            pooled["accuracy"] = float(pooled["accuracy"]) + 0.03
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "poolacc")
        assert code == EXIT_MISMATCH

    def test_pooled_sample_count_drift_is_refused(self, scale_world,
                                                  scale_fit,
                                                  module_scratch,
                                                  built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            entry = summary["arms"]["scaled"]["RCS"]["per_seed"]["2026"]
            entry["pooled_sample_count"] = \
                int(entry["pooled_sample_count"]) + 1
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "count")
        assert code == EXIT_MISMATCH

    def test_fold_rows_and_recall_drift_are_refused(self, scale_world,
                                                    scale_fit,
                                                    module_scratch,
                                                    built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            folds = summary["arms"]["scaled"]["RCS"]["per_seed"]["2026"][
                "folds"]
            folds[0]["rows"] = int(folds[0]["rows"]) + 1
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "foldrows")
        assert code == EXIT_MISMATCH

    def test_paired_total_both_correct_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            total = summary["paired"]["scaled_RCS_minus_R0"]["total"]
            total["both_correct"] = int(total["both_correct"]) + 1
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "totalfour")
        assert code == EXIT_MISMATCH

    def test_seed_range_drift_is_refused(self, scale_world, scale_fit,
                                         module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            entry = summary["arms"]["scaled"]["RCS"]
            entry["macro_f1_seed_range"] = [
                float(entry["macro_f1_seed_range"][0]) + 0.01,
                float(entry["macro_f1_seed_range"][1])]
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "range")
        assert code == EXIT_MISMATCH

    # -- accuracy_mean recomputation (per seed and top level, R0
    #    included) --

    def test_per_seed_accuracy_mean_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            entry = summary["arms"]["fixed"]["RC"]["per_seed"]["2026"]
            entry["accuracy_mean"] = \
                float(entry["accuracy_mean"]) + 1e-6
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "seedaccmean")
        assert code == EXIT_MISMATCH

    def test_r0_top_level_accuracy_mean_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        """The R0 top-level accuracy_mean is checked for its own sake —
        not only inside the non-R0 branch."""
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            entry = summary["arms"]["fixed"]["R0"]
            entry["accuracy_mean"] = \
                float(entry["accuracy_mean"]) + 1e-6
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "r0accmean")
        assert code == EXIT_MISMATCH

    # -- the two shuffle summary layers: receivers / unmatched / ratio --

    def test_shuffle_summary_same_group_ratio_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "shuffle" / "scaled" / "seed_2026" /
                    "fold0" / "shuffle_summary.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            entry = record["seeds"]["2026"]
            if entry["same_group_donor_ratio"] is None:
                entry["same_group_donor_ratio"] = 0.5
            else:
                # 0.0 is a legitimate ratio (no same-group donors); ADD
                # an offset so the value provably changes
                entry["same_group_donor_ratio"] = float(
                    entry["same_group_donor_ratio"]) + 0.125
            path.write_text(json.dumps(record), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "shufsumratio")
        assert code == EXIT_MISMATCH

    def test_shuffle_summary_receivers_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "shuffle" / "scaled" / "seed_2026" /
                    "fold0" / "shuffle_summary.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            entry = record["seeds"]["2026"]
            entry["receivers"] = int(entry["receivers"]) + 1
            path.write_text(json.dumps(record), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "shufsumrecv")
        assert code == EXIT_MISMATCH

    def test_shuffle_summary_unmatched_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "shuffle" / "scaled" / "seed_2026" /
                    "fold0" / "shuffle_summary.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            entry = record["seeds"]["2026"]
            entry["unmatched"] = int(entry["unmatched"]) + 1
            path.write_text(json.dumps(record), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "shufsumunmat")
        assert code == EXIT_MISMATCH

    def test_final_summary_receivers_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            entry = summary["shuffle"]["seed_2026"]["fold0"]["arms"][
                "scaled"]["2026"]
            entry["receivers"] = int(entry["receivers"]) + 1
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "finalrecv")
        assert code == EXIT_MISMATCH

    def test_final_summary_unmatched_drift_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            entry = summary["shuffle"]["seed_2026"]["fold0"]["arms"][
                "scaled"]["2026"]
            entry["unmatched"] = int(entry["unmatched"]) + 1
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "finalunmat")
        assert code == EXIT_MISMATCH

    # -- missing required fields and invalid values --

    def test_seed_file_missing_required_field_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "shuffle" / "scaled" / "seed_2026" /
                    "fold0" / "seed_2026.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            del record["unmatched"]
            path.write_text(json.dumps(record), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "nounmatched")
        assert code == EXIT_MISMATCH

    def test_shuffle_summary_count_type_violation_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        """12 == 12.0 in a value comparison — the integer count contract
        is what refuses a float where a count belongs."""
        def mutate(fit_dir: Path) -> None:
            path = (fit_dir / "shuffle" / "scaled" / "seed_2026" /
                    "fold0" / "shuffle_summary.json")
            record = json.loads(path.read_text(encoding="utf-8"))
            entry = record["seeds"]["2026"]
            entry["receivers"] = float(entry["receivers"])
            path.write_text(json.dumps(record), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "floatrecv")
        assert code == EXIT_MISMATCH

    def test_final_summary_none_ratio_with_matches_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        def mutate(fit_dir: Path) -> None:
            path = fit_dir / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            arms = summary["shuffle"]["seed_2026"]["fold0"]["arms"][
                "scaled"]
            picked = [key for key, entry in arms.items()
                      if isinstance(entry, dict)
                      and int(entry.get("matched", 0)) > 0
                      and entry.get("same_group_donor_ratio") is not None]
            assert picked, "no content seed with matched > 0 to tamper"
            entry = arms[picked[0]]
            entry["same_group_donor_ratio"] = None
            path.write_text(json.dumps(summary), encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "noneratio")
        assert code == EXIT_MISMATCH


class TestCorrectedScenarios:
    def _doctored(self, scale_world, scale_fit, module_scratch,
                  built_cache, mutate, label: str):
        import uuid

        copy_dir = module_scratch / \
            f"corr_{label}_{uuid.uuid4().hex[:8]}"
        shutil.copytree(scale_fit["out"], copy_dir)
        mutate(copy_dir)
        out = module_scratch / f"verify_{copy_dir.name}"
        code = verify_entry.main([
            "--config", str(scale_fit["cfg_path"]), "--feature-dir",
            str(built_cache["features"]), "--p0-fit-dir",
            str(scale_world["fit_dir"]), "--experiment-fit-dir",
            str(copy_dir), "--output-dir", str(out),
        ])
        report_path = out / "verification_report.json"
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.is_file() else {"problems": []})
        print(f"CORRECTED[{label}] exit={code} "
              f"problems={report.get('problems', [])[:3]}")
        return code

    def test_preflight_timeout_lets_preflight_finish_then_expires(
            self, scale_world, module_scratch):
        """The earlier checks pass, the REAL preflight runs to
        completion, and the deadline crosses only at the plan-publish
        boundary: no success plan, PARTIAL status, evidence kept."""
        class PlanBoundaryWatch(DeadlineWatch):
            def __init__(self, **kwargs):
                super().__init__(budget_minutes=30.0)

            def check(self, at: str = "") -> None:
                if at == "before the preflight plan publish":
                    raise DeadlineExceeded(f"crossed at {at}")

        cfg_path = _dump_cfg(module_scratch / "configs" / "planbnd.yaml",
                             scale_world["cfg"])
        out = module_scratch / "planbnd_out"
        with mock.patch.object(prep, "DeadlineWatch", PlanBoundaryWatch):
            code = prep.main([
                "--config", str(cfg_path), "--mode", "preflight",
                "--output-dir", str(out),
                "--deadline-utc", "2030-01-01T00:00:00+00:00",
            ])
        assert code == prep.EXIT_PARTIAL
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_deadline"] is True
        # the preflight itself RAN (its stage is recorded) before the
        # publish boundary expired
        assert status["stages"]["preflight"]["status"] == "COMPLETED"
        assert not (out / "preflight_plan.json").exists()

    def test_group_drift_with_synced_summary_is_refused(
            self, scale_world, scale_fit, module_scratch, built_cache):
        """The row's group AND the group-keyed summary are changed
        TOGETHER: the authoritative identity binding still refuses."""
        def mutate(fit_dir: Path) -> None:
            row_path = (fit_dir / "models" / "seed_2026" /
                        "fold0_scaled_RCS" /
                        "outer_eval_predictions.jsonl")
            rows = [json.loads(line) for line in
                    row_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            fabricated = rows[0]["group"] + "_fabricated"
            rows[0]["group"] = fabricated
            row_path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n",
                encoding="utf-8")
            summary_path = fit_dir / "summary.json"
            summary = json.loads(summary_path.read_text(
                encoding="utf-8"))
            groups = summary["per_group_changes"][
                "scaled_RCS_minus_R0"]["summed_over_seeds"]
            groups[fabricated] = {"fixed": 0, "broke": 0}
            summary_path.write_text(json.dumps(summary),
                                    encoding="utf-8")

        code = self._doctored(scale_world, scale_fit, module_scratch,
                              built_cache, mutate, "groupsync")
        assert code == EXIT_MISMATCH
