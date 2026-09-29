"""Verification harness: real stage evidence, reload, exact routing, exit codes."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from core.material_fusion_dataset import MaterialFusionDataset
from models.material_fusion import build_material_model
from models.polar_fusion import read_manifest_split
from scripts.eval_material_fusion import load_fallback_gray
from scripts.verify_material_fusion import (
    compare_states,
    main,
    route_batch,
    verify_fallback_routing,
    verify_pretrained_sha,
    verify_reload,
    verify_stage_updates,
)

from .conftest import (
    build_v4_dataset,
    run_production_training,
    write_synthetic_pretrained,
)
from .test_eval import FixedLogitsModel, GrayFallbackStub


@pytest.fixture(scope="module")
def verified_run(module_scratch):
    """A real bounded production run with v2 evidence and a fallback stub."""
    scratch = module_scratch
    result = run_production_training(
        scratch, "guard_verify_run", variant="gray_polar",
        epochs=2, warmup_epochs=1, limit_batches=1,
        snapshot_dir=scratch / "stage_snapshots",
    )
    assert result["exit_code"] == 0
    fallback = scratch / "gray_fallback.pt"
    torch.save({"model": GrayFallbackStub(imgsz=64)}, fallback)
    result["fallback"] = fallback
    yield result


class TestRouteBatch:
    def test_route_batch_scatters_exact_outputs(self):
        cmx = FixedLogitsModel([0.0, 5.0, 0.0, 0.0])
        fallback = FixedLogitsModel([3.0, 0.0, 0.0, 0.0])
        rgb = torch.rand(4, 3, 64, 64)
        aux = torch.rand(4, 3, 64, 64)
        invalid = torch.tensor([False, True, False, True])
        routed, routed_logits = route_batch(cmx, fallback, rgb, aux, invalid)
        assert routed.tolist() == [1, 0, 1, 0]
        assert torch.equal(routed_logits[invalid], fallback(rgb[invalid]))
        assert torch.equal(routed_logits[~invalid], cmx(rgb[~invalid], aux[~invalid]))

    def test_route_batch_all_invalid_never_calls_cmx(self):
        class CountingCmx(FixedLogitsModel):
            def __init__(self):
                super().__init__([1.0, 0.0, 0.0, 0.0])
                self.calls = 0

            def forward(self, rgb, aux=None):
                self.calls += 1
                return super().forward(rgb, aux)

        cmx = CountingCmx()
        fallback = FixedLogitsModel([0.0, 1.0, 0.0, 0.0])
        rgb = torch.rand(3, 3, 64, 64)
        routed, _ = route_batch(cmx, fallback, rgb, torch.rand(3, 3, 64, 64),
                                torch.tensor([True, True, True]))
        assert cmx.calls == 0
        assert routed.tolist() == [1, 1, 1]


class TestReload:
    def test_reload_reproduces_raw_logits_within_tolerance(self, scratch_dir):
        from models.material_fusion import material_fusion_metadata, save_material_checkpoint

        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        metadata = material_fusion_metadata(
            variant="gray_polar", imgsz=64, data_fingerprint="fp",
            pretrained_sha256="0" * 64, seed=2026, epoch=2, stage="finetune",
            smoke=True, limit_batches=4, val_macro_f1=0.3, val_accuracy=0.4,
            physical_batch=2, accum_steps=2, effective_batch=4, drop_path_rate=0.0,
        )
        checkpoint = save_material_checkpoint(scratch_dir / "best.pt", model, metadata)
        report = verify_reload(checkpoint, variant="gray_polar", imgsz=64, device="cpu")
        assert report["max_abs_diff"] == 0.0
        assert report["within_tolerance"] is True
        assert report["atol"] == 1e-5
        assert report["rtol"] == 0.0

    def test_reload_detects_changed_weights(self, scratch_dir):
        from models.material_fusion import material_fusion_metadata, save_material_checkpoint

        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        metadata = material_fusion_metadata(
            variant="gray_polar", imgsz=64, data_fingerprint="fp",
            pretrained_sha256="0" * 64, seed=2026, epoch=2, stage="finetune",
            smoke=True, limit_batches=4, val_macro_f1=0.3, val_accuracy=0.4,
            physical_batch=2, accum_steps=2, effective_batch=4, drop_path_rate=0.0,
        )
        checkpoint = save_material_checkpoint(scratch_dir / "best.pt", model, metadata)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["state_dict"]["head.fc.bias"] = payload["state_dict"]["head.fc.bias"] + 1.0
        torch.save(payload, checkpoint)
        report = verify_reload(
            checkpoint, variant="gray_polar", imgsz=64, device="cpu",
            reference_state=model.state_dict(),
        )
        assert report["within_tolerance"] is False


class TestCompareStates:
    def test_identical_states_show_no_updates(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        report = compare_states(model, state, state)
        for group in ("backbone", "fusion", "head"):
            assert report["params"][group]["changed"] == 0
        assert report["params"]["fusion"]["total"] > 0

    def test_param_and_buffer_updates_are_separate(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.add_(0.01)
        report = compare_states(model, before, model.state_dict())
        assert report["params"]["backbone"]["changed"] == report["params"]["backbone"]["total"]
        assert report["buffers"]["fusion"]["changed"] == 0
        assert report["buffers"]["fusion"]["total"] > 0

    def test_buffer_only_change_detected(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            model(torch.rand(4, 3, 64, 64), torch.rand(4, 3, 64, 64))
        report = compare_states(model, before, model.state_dict())
        assert report["buffers"]["fusion"]["changed"] > 0
        assert report["params"]["backbone"]["changed"] == 0


class TestStageUpdates:
    def test_evidence_from_production_run_passes(self, verified_run):
        report = verify_stage_updates(
            verified_run["snapshot_dir"], verified_run["run_dir"] / "last.pt"
        )
        assert report["passed"] is True, report["failed_items"]
        assert report["format_version"] == 2
        assert report["checkpoint_role"] == "last"

    def test_missing_evidence_is_not_provable(self, scratch_dir, verified_run):
        report = verify_stage_updates(
            scratch_dir / "no_such_dir", verified_run["run_dir"] / "last.pt"
        )
        assert report["available"] is False
        assert report["passed"] is False
        assert report["failed_items"]


class TestPretrainedSha:
    def test_matching_sha_passes(self, scratch_dir):
        path = write_synthetic_pretrained(scratch_dir / "syn.pth")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report = verify_pretrained_sha(path, digest)
        assert report["matches"] is True

    def test_wrong_sha_fails(self, scratch_dir):
        path = write_synthetic_pretrained(scratch_dir / "syn.pth")
        report = verify_pretrained_sha(path, "0" * 64)
        assert report["matches"] is False


class TestFallbackRoutingVerification:
    def test_exact_routing_with_explicit_invalid_flags(self, scratch_dir):
        data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 6, 4))
        paths, _, _ = read_manifest_split(data_root, "val")
        dataset = MaterialFusionDataset(paths, variant="gray_polar", imgsz=64)
        loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        fallback_path = scratch_dir / "gray_best.pt"
        torch.save({"model": GrayFallbackStub(imgsz=64)}, fallback_path)
        fallback, identity = load_fallback_gray(fallback_path, "cpu", imgsz=64)
        report = verify_fallback_routing(
            model, fallback, loader, "cpu", forced_invalid_indices=[0, 3]
        )
        assert report["invalid_samples"] == 2
        assert report["invalid_rows_exact"] is True
        assert report["valid_rows_exact"] is True
        assert identity


class TestVerifyEntry:
    def _args(self, run, output, extra=None):
        return [
            "--checkpoint", str(run["run_dir"] / "last.pt"),
            "--stage-snapshots", str(run["snapshot_dir"]),
            "--fallback-weights", str(run["fallback"]),
            "--pretrained", str(run["pretrained"]),
            "--device", "cpu",
            "--batch", "3",
            "--limit-batches", "1",
            "--output", str(output),
        ] + (extra or [])

    def test_verify_passes_with_full_evidence(self, verified_run, scratch_dir):
        output = scratch_dir / "verify.json"
        code = main(self._args(verified_run, output))
        assert code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["reload"]["within_tolerance"] is True
        assert report["fallback_routing"]["invalid_rows_exact"] is True
        assert report["stage_updates"]["passed"] is True
        assert report["gradient_finiteness"]["passed"] is True
        assert report["config_source"]["source"] == "run_config"
        assert report["passed"] is True
        assert report["failed_items"] == []

    def test_verify_fails_without_stage_evidence(self, verified_run, scratch_dir):
        output = scratch_dir / "verify_no_evidence.json"
        code = main(self._args(verified_run, output, [
            "--stage-snapshots", str(scratch_dir / "missing_snapshots"),
        ]))
        assert code == 1
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["passed"] is False
        assert report["stage_updates"]["available"] is False
        assert report["failed_items"]

    def test_verify_fails_with_doctored_warmup(self, verified_run, scratch_dir):
        doctored = scratch_dir / "doctored_snaps"
        doctored.mkdir()
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt",
                     "stage_snapshots.json"):
            (doctored / name).write_bytes((verified_run["snapshot_dir"] / name).read_bytes())
        warmup_path = doctored / "post_warmup.pt"
        warmup = torch.load(warmup_path, map_location="cpu", weights_only=True)
        key = "encoder.block1.0.attn.q.weight"
        warmup["state_dict"][key] = warmup["state_dict"][key] + 1.0
        torch.save(warmup, warmup_path)
        output = scratch_dir / "verify_doctored.json"
        code = main(self._args(verified_run, output, ["--stage-snapshots", str(doctored)]))
        assert code == 1
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["passed"] is False
        assert report["failed_items"]

    def test_wrong_pretrained_sha_refused(self, verified_run, scratch_dir):
        wrong = scratch_dir / "wrong_pretrained.pth"
        wrong.write_bytes(b"not the recorded checkpoint")
        output = scratch_dir / "verify_bad_sha.json"
        code = main(self._args(verified_run, output, ["--pretrained", str(wrong)]))
        assert code == 2
        assert not output.exists()

    def test_data_mismatch_refused(self, verified_run, scratch_dir):
        other_root = build_v4_dataset(scratch_dir / "other_v4", counts=(4, 4, 4), seed=123)
        output = scratch_dir / "verify_wrong_data.json"
        code = main(self._args(verified_run, output, ["--data", str(other_root)]))
        assert code == 2
        assert not output.exists()
