"""Stage-evidence identity, integrity, and checkpoint binding (guard round).

The evidence must be bound to the run and checkpoint it belongs to:
identity fields, per-file digests, and exact state association with the
verified checkpoint. Structural problems in any compared state (missing
or extra tensors, shape/dtype mismatch, non-finite values, incomplete
group coverage) fail; a zeroed or modified in-memory model, snapshots
from another run/variant, and swapped snapshot roles must all be refused.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from models.material_fusion import (
    CLASS_NAMES,
    build_material_model,
    load_material_checkpoint,
    read_material_checkpoint_metadata,
)
from scripts.verify_material_fusion import (
    compare_states,
    verify_gradient_finiteness,
    verify_stage_updates,
)

from .conftest import run_production_training, unique_path


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _save(path, payload):
    torch.save(payload, path)


@pytest.fixture(scope="module")
def evidence(module_scratch):
    """One real bounded production run with v2 evidence (shared, read-only)."""
    scratch = module_scratch
    snapshot_dir = unique_path(scratch, "snaps")
    result = run_production_training(
        scratch, "guard_evidence_run", variant="gray_polar",
        epochs=2, warmup_epochs=1, limit_batches=1, snapshot_dir=snapshot_dir,
    )
    assert result["exit_code"] == 0
    yield result


class TestEvidenceIdentityAndBinding:
    def test_same_run_evidence_passes(self, evidence):
        for name in ("last.pt", "best.pt"):
            checkpoint = evidence["run_dir"] / name
            report = verify_stage_updates(
                evidence["snapshot_dir"], checkpoint, model=None
            )
            assert report["available"] is True
            assert report["passed"] is True, report["failed_items"]
            assert report["format_version"] == 2

    def test_zeroed_in_memory_model_rejected(self, evidence):
        checkpoint = evidence["run_dir"] / "last.pt"
        model, _, _ = load_material_checkpoint(
            checkpoint, variant="gray_polar", imgsz=64
        )
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        report = verify_stage_updates(evidence["snapshot_dir"], checkpoint, model=model)
        assert report["passed"] is False
        assert any("model" in item or "checkpoint" in item for item in report["failed_items"])

    def test_single_modified_param_in_memory_model_rejected(self, evidence):
        checkpoint = evidence["run_dir"] / "last.pt"
        model, _, _ = load_material_checkpoint(
            checkpoint, variant="gray_polar", imgsz=64
        )
        with torch.no_grad():
            next(model.parameters()).add_(0.01)
        report = verify_stage_updates(evidence["snapshot_dir"], checkpoint, model=model)
        assert report["passed"] is False

    def test_cross_run_snapshots_rejected(self, evidence, scratch_dir):
        other = run_production_training(
            scratch_dir, "guard_evidence_other_run", variant="gray_polar",
            epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=scratch_dir / "other_snaps",
        )
        assert other["exit_code"] == 0
        checkpoint = evidence["run_dir"] / "last.pt"
        report = verify_stage_updates(
            other["snapshot_dir"], checkpoint, model=None
        )
        assert report["passed"] is False
        assert report["failed_items"]

    def test_cross_variant_snapshots_rejected(self, evidence, scratch_dir):
        e3 = run_production_training(
            scratch_dir, "guard_evidence_e3", variant="gray_valid",
            epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=scratch_dir / "e3_snaps",
        )
        assert e3["exit_code"] == 0
        checkpoint = evidence["run_dir"] / "last.pt"  # gray_polar
        report = verify_stage_updates(e3["snapshot_dir"], checkpoint, model=None)
        assert report["passed"] is False
        assert any("variant" in item for item in report["failed_items"])

    def test_swapped_snapshot_files_rejected(self, evidence, scratch_dir):
        snapshots = scratch_dir / "swapped"
        snapshots.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt",
                     "stage_snapshots.json"):
            source = evidence["snapshot_dir"] / name
            (snapshots / name).write_bytes(source.read_bytes())
        # Swap init and post_warmup contents.
        init = _load(snapshots / "init.pt")
        warmup = _load(snapshots / "post_warmup.pt")
        _save(snapshots / "init.pt", warmup)
        _save(snapshots / "post_warmup.pt", init)
        checkpoint = evidence["run_dir"] / "last.pt"
        report = verify_stage_updates(snapshots, checkpoint, model=None)
        assert report["passed"] is False
        assert report["failed_items"]

    def test_missing_metadata_is_legacy_not_pass(self, evidence, scratch_dir):
        snapshots = scratch_dir / "no_meta"
        snapshots.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt"):
            (snapshots / name).write_bytes((evidence["snapshot_dir"] / name).read_bytes())
        checkpoint = evidence["run_dir"] / "last.pt"
        report = verify_stage_updates(snapshots, checkpoint, model=None)
        assert report["passed"] is False
        assert report["available"] is False
        assert report["format"] == "legacy"
        assert any("legacy" in item or "metadata" in item for item in report["failed_items"])

    def test_missing_required_stage_snapshot_rejected(self, evidence, scratch_dir):
        snapshots = scratch_dir / "no_warmup"
        snapshots.mkdir()
        for name in ("init.pt", "post_finetune.pt", "stage_snapshots.json"):
            (snapshots / name).write_bytes((evidence["snapshot_dir"] / name).read_bytes())
        checkpoint = evidence["run_dir"] / "last.pt"
        report = verify_stage_updates(snapshots, checkpoint, model=None)
        assert report["passed"] is False
        assert any("post_warmup" in item for item in report["failed_items"])

    def test_corrupted_snapshot_digest_rejected(self, evidence, scratch_dir):
        snapshots = scratch_dir / "corrupt"
        snapshots.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt",
                     "stage_snapshots.json"):
            (snapshots / name).write_bytes((evidence["snapshot_dir"] / name).read_bytes())
        payload = _load(snapshots / "init.pt")
        key = next(iter(payload["state_dict"]))
        payload["state_dict"][key] = payload["state_dict"][key] + 1.0
        _save(snapshots / "init.pt", payload)
        checkpoint = evidence["run_dir"] / "last.pt"
        report = verify_stage_updates(snapshots, checkpoint, model=None)
        assert report["passed"] is False
        assert any("digest" in item or "sha256" in item for item in report["failed_items"])

    def test_missing_tensor_in_snapshot_rejected(self, evidence, scratch_dir):
        snapshots = scratch_dir / "missing_tensor"
        snapshots.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt",
                     "stage_snapshots.json"):
            (snapshots / name).write_bytes((evidence["snapshot_dir"] / name).read_bytes())
        payload = _load(snapshots / "post_finetune.pt")
        del payload["state_dict"]["encoder.block1.0.attn.q.weight"]
        _save(snapshots / "post_finetune.pt", payload)
        # Recompute the recorded digest so only the tensor set is wrong.
        meta = json.loads((snapshots / "stage_snapshots.json").read_text(encoding="utf-8"))
        meta["snapshots"]["post_finetune"]["sha256"] = _sha256(snapshots / "post_finetune.pt")
        (snapshots / "stage_snapshots.json").write_text(json.dumps(meta), encoding="utf-8")
        checkpoint = evidence["run_dir"] / "last.pt"
        report = verify_stage_updates(snapshots, checkpoint, model=None)
        assert report["passed"] is False
        assert any("block1.0.attn.q.weight" in item or "missing" in item
                   for item in report["failed_items"])

    def test_dtype_mismatch_in_snapshot_rejected(self, evidence, scratch_dir):
        snapshots = scratch_dir / "dtype"
        snapshots.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt",
                     "stage_snapshots.json"):
            (snapshots / name).write_bytes((evidence["snapshot_dir"] / name).read_bytes())
        payload = _load(snapshots / "post_finetune.pt")
        key = "head.fc.weight"
        payload["state_dict"][key] = payload["state_dict"][key].double()
        _save(snapshots / "post_finetune.pt", payload)
        meta = json.loads((snapshots / "stage_snapshots.json").read_text(encoding="utf-8"))
        meta["snapshots"]["post_finetune"]["sha256"] = _sha256(snapshots / "post_finetune.pt")
        (snapshots / "stage_snapshots.json").write_text(json.dumps(meta), encoding="utf-8")
        checkpoint = evidence["run_dir"] / "last.pt"
        report = verify_stage_updates(snapshots, checkpoint, model=None)
        assert report["passed"] is False
        assert any("dtype" in item for item in report["failed_items"])

    def test_non_finite_snapshot_rejected(self, evidence, scratch_dir):
        snapshots = scratch_dir / "nan"
        snapshots.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt",
                     "stage_snapshots.json"):
            (snapshots / name).write_bytes((evidence["snapshot_dir"] / name).read_bytes())
        payload = _load(snapshots / "post_warmup.pt")
        key = "head.fc.bias"
        payload["state_dict"][key][0] = float("nan")
        _save(snapshots / "post_warmup.pt", payload)
        meta = json.loads((snapshots / "stage_snapshots.json").read_text(encoding="utf-8"))
        meta["snapshots"]["post_warmup"]["sha256"] = _sha256(snapshots / "post_warmup.pt")
        (snapshots / "stage_snapshots.json").write_text(json.dumps(meta), encoding="utf-8")
        checkpoint = evidence["run_dir"] / "last.pt"
        report = verify_stage_updates(snapshots, checkpoint, model=None)
        assert report["passed"] is False
        assert any("finite" in item or "nan" in item.lower() for item in report["failed_items"])

    def test_checkpoint_not_recorded_in_evidence_rejected(self, evidence, scratch_dir):
        """A checkpoint the evidence does not record (different run) fails."""
        other = run_production_training(
            scratch_dir, "guard_evidence_unrecorded", variant="gray_polar",
            epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=scratch_dir / "unrecorded_snaps",
        )
        report = verify_stage_updates(
            evidence["snapshot_dir"], other["run_dir"] / "last.pt", model=None
        )
        assert report["passed"] is False
        assert any("not recorded" in item or "digest" in item or "variant" in item
                   for item in report["failed_items"])


@pytest.fixture(scope="module")
def early_best(module_scratch):
    """Scripted metrics: epoch 1 wins, epoch 2 is worse -> best != last."""
    import scripts.train_material_fusion as train_module

    scratch = module_scratch
    scripted = [
        {"accuracy": 0.9, "macro_f1": 0.9, "per_class_recall": {"0": 0.9}},
        {"accuracy": 0.1, "macro_f1": 0.1, "per_class_recall": {"0": 0.1}},
    ]
    calls = {"n": 0}
    original = train_module.evaluate_model

    def fake_evaluate(model, loader, device, limit_batches=0):
        result = scripted[min(calls["n"], len(scripted) - 1)]
        calls["n"] += 1
        return result

    train_module.evaluate_model = fake_evaluate
    try:
        result = run_production_training(
            scratch, "guard_early_best", variant="gray_polar",
            epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=unique_path(scratch, "snaps"),
        )
    finally:
        train_module.evaluate_model = original
    assert result["exit_code"] == 0
    yield result


class TestBestLastAssociation:
    def test_early_best_and_last_both_verified(self, early_best):
        last = verify_stage_updates(
            early_best["snapshot_dir"], early_best["run_dir"] / "last.pt", model=None
        )
        best = verify_stage_updates(
            early_best["snapshot_dir"], early_best["run_dir"] / "best.pt", model=None
        )
        assert last["passed"] is True, last["failed_items"]
        assert best["passed"] is True, best["failed_items"]
        assert last["checkpoint_role"] == "last"
        assert best["checkpoint_role"] == "best"

    def test_best_state_identical_to_final_rejected(self, early_best, scratch_dir):
        """An early best must differ from the final snapshot; substituting the
        final state for the early best is refused."""
        snapshots = scratch_dir / "best_as_final"
        snapshots.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt",
                     "stage_snapshots.json"):
            (snapshots / name).write_bytes((early_best["snapshot_dir"] / name).read_bytes())
        payload = _load(snapshots / "post_finetune.pt")
        _save(snapshots / "best.pt", payload)  # final state claimed as best
        meta = json.loads((snapshots / "stage_snapshots.json").read_text(encoding="utf-8"))
        meta["checkpoints"]["best"]["sha256"] = _sha256(snapshots / "best.pt")
        (snapshots / "stage_snapshots.json").write_text(json.dumps(meta), encoding="utf-8")
        report = verify_stage_updates(snapshots, snapshots / "best.pt", model=None)
        assert report["passed"] is False
        assert report["failed_items"]


class TestHistoryVerification:
    def _history(self, evidence):
        return evidence["run_dir"] / "metrics_history.jsonl"

    def _write(self, path, entries):
        path.write_text(
            "\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8"
        )

    def _valid_entries(self):
        return [
            {"epoch": 1, "stage": "warmup", "gradients_finite": True},
            {"epoch": 2, "stage": "finetune", "gradients_finite": True},
        ]

    def test_valid_history_passes(self, evidence):
        report = verify_gradient_finiteness(
            self._history(evidence), epochs=2, warmup_epochs=1
        )
        assert report["passed"] is True

    def test_missing_rows_rejected(self, evidence, scratch_dir):
        path = scratch_dir / "history.jsonl"
        self._write(path, self._valid_entries()[:1])
        report = verify_gradient_finiteness(path, epochs=2, warmup_epochs=1)
        assert report["passed"] is False

    def test_duplicate_epoch_rejected(self, evidence, scratch_dir):
        path = scratch_dir / "history.jsonl"
        entries = self._valid_entries()
        entries[1] = {**entries[1], "epoch": 1}
        self._write(path, entries)
        report = verify_gradient_finiteness(path, epochs=2, warmup_epochs=1)
        assert report["passed"] is False

    def test_wrong_stage_rejected(self, evidence, scratch_dir):
        path = scratch_dir / "history.jsonl"
        entries = self._valid_entries()
        entries[0] = {**entries[0], "stage": "finetune"}
        self._write(path, entries)
        report = verify_gradient_finiteness(path, epochs=2, warmup_epochs=1)
        assert report["passed"] is False

    def test_non_bool_gradients_finite_rejected(self, evidence, scratch_dir):
        path = scratch_dir / "history.jsonl"
        entries = self._valid_entries()
        entries[0] = {**entries[0], "gradients_finite": 1}
        self._write(path, entries)
        report = verify_gradient_finiteness(path, epochs=2, warmup_epochs=1)
        assert report["passed"] is False

    def test_false_gradients_finite_rejected(self, evidence, scratch_dir):
        path = scratch_dir / "history.jsonl"
        entries = self._valid_entries()
        entries[0] = {**entries[0], "gradients_finite": False}
        self._write(path, entries)
        report = verify_gradient_finiteness(path, epochs=2, warmup_epochs=1)
        assert report["passed"] is False

    def test_missing_file_rejected(self, scratch_dir):
        report = verify_gradient_finiteness(
            scratch_dir / "nope.jsonl", epochs=2, warmup_epochs=1
        )
        assert report["passed"] is False


class TestCompareStatesStrictness:
    def _model_and_state(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        return model, {k: v.clone() for k, v in model.state_dict().items()}

    def test_identical_states_pass(self):
        model, state = self._model_and_state()
        report = compare_states(model, state, state)
        assert report["params"]["fusion"]["total"] > 0
        assert report["params"]["fusion"]["changed"] == 0

    def test_missing_key_rejected(self):
        model, state = self._model_and_state()
        after = dict(state)
        del after["head.fc.weight"]
        with pytest.raises(ValueError, match="head.fc.weight"):
            compare_states(model, state, after)

    def test_extra_key_rejected(self):
        model, state = self._model_and_state()
        after = dict(state)
        after["head.extra.weight"] = torch.zeros(1)
        with pytest.raises(ValueError, match="head.extra.weight"):
            compare_states(model, state, after)

    def test_dtype_mismatch_rejected(self):
        model, state = self._model_and_state()
        after = {k: v.clone() for k, v in state.items()}
        after["head.fc.weight"] = after["head.fc.weight"].double()
        with pytest.raises(ValueError, match="dtype"):
            compare_states(model, state, after)

    def test_shape_mismatch_rejected(self):
        model, state = self._model_and_state()
        after = {k: v.clone() for k, v in state.items()}
        after["head.fc.bias"] = after["head.fc.bias"][:2]
        with pytest.raises(ValueError, match="shape"):
            compare_states(model, state, after)

    def test_non_finite_rejected(self):
        model, state = self._model_and_state()
        after = {k: v.clone() for k, v in state.items()}
        after["head.fc.bias"][0] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            compare_states(model, state, after)

    def test_float32_downcast_not_used_for_exactness(self):
        """A float64 difference below float32 resolution must still count as
        a change: exact comparison uses torch.equal, never a float32 cast."""
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        before = {k: v.double() for k, v in model.state_dict().items()}
        after = {k: v.clone() for k, v in before.items()}
        key = "head.fc.bias"
        after[key][0] = after[key][0] + 1e-12
        assert float(after[key][0] - before[key][0]) != 0.0
        report = compare_states(model, before, after)
        assert report["params"]["head"]["changed"] >= 1

    def test_dual_variant_requires_nonempty_groups(self):
        model, state = self._model_and_state()
        emptied = {k: v for k, v in state.items() if "FRMs" not in k and "FFMs" not in k}
        with pytest.raises(ValueError, match="fusion|missing"):
            compare_states(model, state, emptied)

    def test_v1_metadata_classified_as_legacy(self, evidence, scratch_dir):
        """Pre-v2 metadata (no format/identity fields) is legacy, never a pass."""
        snapshots = scratch_dir / "v1_meta"
        snapshots.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt"):
            (snapshots / name).write_bytes((evidence["snapshot_dir"] / name).read_bytes())
        (snapshots / "stage_snapshots.json").write_text(
            json.dumps({"epochs": 2, "warmup_epochs": 1, "variant": "gray_polar",
                        "run_id": "x"}),
            encoding="utf-8",
        )
        report = verify_stage_updates(snapshots, evidence["run_dir"] / "last.pt")
        assert report["passed"] is False
        assert report["format"] == "legacy"
        assert report["legacy_findings"]["missing_evidence"]


def _model_fingerprint(model):
    return {
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "training": model.training,
        "requires_grad": [p.requires_grad for p in model.parameters()],
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }


def _assert_model_unchanged(before, after):
    assert after["device"] == before["device"]
    assert after["dtype"] == before["dtype"]
    assert after["training"] == before["training"]
    assert after["requires_grad"] == before["requires_grad"]
    assert set(after["state"]) == set(before["state"])
    for key, value in before["state"].items():
        assert torch.equal(after["state"][key], value), key


class TestModelStateInvariance:
    def test_verify_does_not_change_the_passed_model(self, evidence):
        checkpoint = evidence["run_dir"] / "last.pt"
        model, _, _ = load_material_checkpoint(checkpoint, variant="gray_polar", imgsz=64)
        model.train()
        before = _model_fingerprint(model)
        verify_stage_updates(evidence["snapshot_dir"], checkpoint, model=model)
        _assert_model_unchanged(before, _model_fingerprint(model))

    def test_dtype_mismatch_rejected_even_if_values_equal(self, evidence):
        checkpoint = evidence["run_dir"] / "last.pt"
        model, _, _ = load_material_checkpoint(checkpoint, variant="gray_polar", imgsz=64)
        model = model.double()  # float32 -> float64 is value-exact, dtype differs
        report = verify_stage_updates(evidence["snapshot_dir"], checkpoint, model=model)
        assert report["passed"] is False
        assert any("dtype" in item for item in report["failed_items"])


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"
)


@requires_cuda
class TestCudaModelBinding:
    def test_cuda_model_with_identical_weights_passes(self, evidence):
        checkpoint = evidence["run_dir"] / "last.pt"
        model, _, _ = load_material_checkpoint(checkpoint, variant="gray_polar", imgsz=64)
        model = model.to("cuda")
        report = verify_stage_updates(evidence["snapshot_dir"], checkpoint, model=model)
        assert report["passed"] is True, report["failed_items"]

    def test_cuda_model_modified_param_rejected(self, evidence):
        checkpoint = evidence["run_dir"] / "last.pt"
        model, _, _ = load_material_checkpoint(checkpoint, variant="gray_polar", imgsz=64)
        model = model.to("cuda")
        with torch.no_grad():
            next(model.parameters()).add_(0.01)
        report = verify_stage_updates(evidence["snapshot_dir"], checkpoint, model=model)
        assert report["passed"] is False
        assert any("model" in item or "checkpoint" in item for item in report["failed_items"])

    def test_cuda_model_modified_buffer_rejected(self, evidence):
        checkpoint = evidence["run_dir"] / "last.pt"
        model, _, _ = load_material_checkpoint(checkpoint, variant="gray_polar", imgsz=64)
        model = model.to("cuda")
        with torch.no_grad():
            next(model.buffers()).add_(1.0)
        report = verify_stage_updates(evidence["snapshot_dir"], checkpoint, model=model)
        assert report["passed"] is False

    def test_cuda_model_unchanged_by_verification(self, evidence):
        checkpoint = evidence["run_dir"] / "last.pt"
        model, _, _ = load_material_checkpoint(checkpoint, variant="gray_polar", imgsz=64)
        model = model.to("cuda").eval()
        before = _model_fingerprint(model)
        verify_stage_updates(evidence["snapshot_dir"], checkpoint, model=model)
        _assert_model_unchanged(before, _model_fingerprint(model))


@pytest.fixture(scope="module")
def zero_warmup_runs(module_scratch):
    """Two real bounded runs without warmup: E4 (dual) and E1 (single)."""
    scratch = module_scratch
    runs = {}
    for tag, variant in (("e4", "gray_polar"), ("e1", "gray")):
        run_id = f"guard_zero_warmup_{tag}"
        result = run_production_training(
            scratch, run_id, variant=variant, epochs=2, warmup_epochs=0,
            limit_batches=1, snapshot_dir=scratch / f"snaps_{tag}",
        )
        assert result["exit_code"] == 0
        runs[tag] = result
    yield runs


class TestZeroWarmupEvidence:
    @pytest.mark.parametrize("tag", ["e4", "e1"])
    def test_zero_warmup_full_evidence_passes(self, zero_warmup_runs, tag):
        run = zero_warmup_runs[tag]
        assert not (run["snapshot_dir"] / "post_warmup.pt").exists()
        report = verify_stage_updates(run["snapshot_dir"], run["run_dir"] / "last.pt")
        assert report["passed"] is True, report["failed_items"]
        assert report["warmup_applicable"] is False
        checks = {c["check"]: c for c in report["checks"]}
        assert checks["warmup_not_applicable"]["passed"] is True
        assert "warmup_backbone_frozen" not in checks
        assert checks["finetune_backbone_unfroze_and_trained"]["passed"] is True

    def test_zero_warmup_finetune_backbone_unchanged_rejected(
        self, zero_warmup_runs, scratch_dir
    ):
        run = zero_warmup_runs["e4"]
        snapshots = scratch_dir / "frozen_finetune"
        snapshots.mkdir()
        for name in ("init.pt", "post_finetune.pt", "stage_snapshots.json"):
            (snapshots / name).write_bytes((run["snapshot_dir"] / name).read_bytes())
        # The finetune snapshot equals init: nothing was trained.
        init = _load(snapshots / "init.pt")
        _save(snapshots / "post_finetune.pt", init)
        meta = json.loads((snapshots / "stage_snapshots.json").read_text(encoding="utf-8"))
        meta["snapshots"]["post_finetune"]["sha256"] = _sha256(snapshots / "post_finetune.pt")
        (snapshots / "stage_snapshots.json").write_text(json.dumps(meta), encoding="utf-8")
        report = verify_stage_updates(snapshots, run["run_dir"] / "last.pt")
        assert report["passed"] is False
        assert any("finetune" in item for item in report["failed_items"])

    def test_missing_warmup_evidence_with_warmup_protocol_fails_cleanly(
        self, evidence, scratch_dir
    ):
        """warmup>0 but post_warmup missing: explicit failure, no KeyError."""
        snapshots = scratch_dir / "missing_warmup"
        snapshots.mkdir()
        for name in ("init.pt", "post_finetune.pt", "stage_snapshots.json"):
            (snapshots / name).write_bytes((evidence["snapshot_dir"] / name).read_bytes())
        report = verify_stage_updates(snapshots, evidence["run_dir"] / "last.pt")
        assert report["passed"] is False
        assert report["failed_items"]

    def test_zero_warmup_history_all_finetune(self, zero_warmup_runs):
        run = zero_warmup_runs["e4"]
        report = verify_gradient_finiteness(
            run["run_dir"] / "metrics_history.jsonl", epochs=2, warmup_epochs=0
        )
        assert report["passed"] is True

    def test_illegal_warmup_row_rejected(self, zero_warmup_runs, scratch_dir):
        run = zero_warmup_runs["e4"]
        path = scratch_dir / "history.jsonl"
        entries = [
            {"epoch": 1, "stage": "warmup", "gradients_finite": True},
            {"epoch": 2, "stage": "finetune", "gradients_finite": True},
        ]
        path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )
        report = verify_gradient_finiteness(path, epochs=2, warmup_epochs=0)
        assert report["passed"] is False
