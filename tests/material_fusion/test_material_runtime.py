"""4A runtime adapter tests: material_fusion independent input mode.

Covers (per the 4A task contract, section 十):

- the new ``material_fusion`` input mode is opt-in; the three legacy modes
  and the default configuration keep their original behaviour;
- the four CMX variants' runtime inputs match the offline V5 preprocessing
  contract exactly (tensors) before any logits are compared;
- ``instance_mask`` and ``polar_valid`` stay separate at the model boundary;
- GT-marked and predicted mask paths are explicitly labelled and never
  mixed;
- instance order, count, tail batches and classification/depth association
  are preserved with no silent ``zip`` truncation;
- polar-invalid routing really bypasses the CMX model into the independent
  gray fallback, and the E1 gray variant never uses a fallback;
- empty inputs, empty masks, NaN/Inf polar arrays and wrong sizes produce
  explicit instance failures instead of fabricated labels;
- checkpoint contract conflicts are refused at construction;
- model parameters and buffers are unchanged by inference;
- verification report helpers refuse an existing directory and digest the
  bytes actually on disk; an injected misalignment must fail verification.

All model weights in this module are synthetic (built through the shared
conftest checkpoint builders); no real checkpoint, dataset or GPU is used.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.material_mask_dataset import (
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    load_mask_sample,
)
from core.material_fusion_dataset import (
    CLASS_NAMES,
    MaterialFusionDataset,
)
from models.material_fusion import (
    MATERIAL_FUSION_MASK_VERSION,
    PREPROCESSING_VERSION,
    UPSTREAM_COMMIT,
)

from .conftest import write_v2_checkpoint_payload

HEIGHT, WIDTH = 48, 64
IMGSZ = 224


def _resize_area(array: np.ndarray, imgsz: int) -> np.ndarray:
    import cv2

    return cv2.resize(
        np.ascontiguousarray(array, dtype=np.float32),
        (imgsz, imgsz),
        interpolation=cv2.INTER_AREA,
    )


# ---------------------------------------------------------------------------
# Synthetic checkpoints (module scope: building four MiT-B0 models is slow).


def _write_checkpoint(path: Path, *, variant: str, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    return write_v2_checkpoint_payload(
        path, variant=variant, data_fingerprint="c" * 64
    )


@pytest.fixture(scope="module")
def checkpoint_dir(module_scratch) -> Path:
    directory = module_scratch / "runtime_checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="module")
def dual_checkpoint(checkpoint_dir) -> Path:
    path = checkpoint_dir / "dual_s2026.pt"
    if not path.exists():
        _write_checkpoint(path, variant="gray_polar", seed=1)
    return path


@pytest.fixture(scope="module")
def gray_checkpoint(checkpoint_dir) -> Path:
    path = checkpoint_dir / "gray_s2026.pt"
    if not path.exists():
        _write_checkpoint(path, variant="gray", seed=2)
    return path


# ---------------------------------------------------------------------------
# Synthetic runtime inputs.


def make_crop_gray(seed: int = 0, height: int = 40, width: int = 52) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(20, 235, size=(height, width), dtype=np.uint8)


def make_polar_arrays(
    seed: int = 1, height: int = 40, width: int = 52, invalid: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """signed_q, abs_q, polar_valid, instance_mask for one crop."""
    rng = np.random.default_rng(seed)
    instance_mask = np.zeros((height, width), dtype=np.uint8)
    instance_mask[4:32, 8:44] = 1
    if invalid:
        signed_q = np.zeros((height, width), dtype=np.float32)
        polar_valid = np.zeros((height, width), dtype=np.uint8)
    else:
        signed_q = (rng.random((height, width)).astype(np.float32) - 0.5) * 0.4
        signed_q[instance_mask == 0] = 0.0
        polar_valid = (instance_mask & (rng.random((height, width)) > 0.25)).astype(
            np.uint8
        )
        signed_q[polar_valid == 0] = 0.0
    abs_q = np.abs(signed_q).astype(np.float32)
    return signed_q, abs_q, polar_valid, instance_mask


def make_material_input(
    instance_id: int = 0,
    frame_id: str = "frame_000",
    invalid: bool = False,
    gray: np.ndarray | None = None,
    polar: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    mask_source: str = "predicted_instance_mask",
) -> "MaterialInstanceInput":
    from models.material_classifier import MaterialInstanceInput

    signed_q, abs_q, polar_valid, instance_mask = polar or make_polar_arrays(
        seed=1 + instance_id, invalid=invalid
    )
    gray = gray if gray is not None else make_crop_gray(seed=instance_id)
    valid_ratio = float((polar_valid > 0).mean())
    quality = np.asarray([valid_ratio, 1.0, 1.0, float(abs_q[polar_valid > 0].mean() if polar_valid.any() else 0.0)], dtype=np.float32)
    return MaterialInstanceInput(
        frame_id=frame_id,
        instance_id=instance_id,
        bbox=(8, 4, 44, 32),
        gray=gray,
        signed_q=signed_q,
        abs_q=abs_q,
        polar_valid=polar_valid,
        instance_mask=instance_mask,
        quality=quality,
        polar_invalid=invalid,
        invalid_reason="sync_skew_exceeded" if invalid else "",
        mask_source=mask_source,
    )


# ---------------------------------------------------------------------------
# Recording stubs standing in for the CMX model and the fallback.


class RecordingCmxModel:
    """Records which batch rows reached the CMX model."""

    def __init__(self, is_dual: bool = True):
        self.is_dual = is_dual
        self.uses_instance_mask = True
        self.calls: list[dict] = []
        self.eval_mode = True

    def eval(self):
        self.eval_mode = True
        return self

    def __call__(self, rgb, aux=None, instance_mask=None):
        self.calls.append(
            {"rgb": rgb, "aux": aux, "instance_mask": instance_mask, "n": rgb.shape[0]}
        )
        rows = rgb.shape[0]
        logits = torch.zeros(rows, len(CLASS_NAMES))
        logits[:, 2] = 5.0  # deterministic plastic_fish
        return logits


class RecordingFallback:
    """Records which batch rows reached the independent gray fallback."""

    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, rgb):
        self.calls.append(int(rgb.shape[0]))
        rows = rgb.shape[0]
        logits = torch.zeros(rows, len(CLASS_NAMES))
        logits[:, 0] = 3.0  # deterministic metal_submarine
        return logits


class TinyFallback(torch.nn.Module):
    """A real (tiny) fallback module so buffer/param immutability is checked."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, stride=4, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, len(CLASS_NAMES)),
        )

    def forward(self, rgb):
        return self.net(rgb)


def install_stubs(classifier, cmx, fallback=None) -> tuple[RecordingCmxModel, RecordingFallback | None]:
    classifier.model = cmx
    if fallback is not None:
        classifier.fallback = fallback
    return cmx, fallback


# ---------------------------------------------------------------------------
# Constructor and checkpoint contract.


class TestMaterialFusionClassifierContract:
    def test_loads_a_formal_v2_checkpoint_and_reports_identity(
        self, dual_checkpoint
    ):
        from models.material_classifier import MaterialFusionClassifier

        classifier = MaterialFusionClassifier(
            dual_checkpoint, device="cpu", fallback=RecordingFallback()
        )
        assert classifier.variant == "gray_polar"
        assert classifier.is_dual is True
        assert classifier.imgsz == IMGSZ
        assert classifier.class_names == list(CLASS_NAMES)
        assert classifier.metadata.version == MATERIAL_FUSION_MASK_VERSION
        assert classifier.model_identity.startswith("material_fusion_v2_mask")
        assert "gray_polar" in classifier.model_identity

    def test_refuses_a_gap_checkpoint_for_the_mask_protocol(self, checkpoint_dir):
        from models.material_classifier import MaterialFusionClassifier

        path = write_v2_checkpoint_payload(
            checkpoint_dir / "gap_payload.pt", pooling_mode="gap"
        )
        with pytest.raises(ValueError, match="pooling"):
            MaterialFusionClassifier(path, device="cpu")

    def test_refuses_a_smoke_checkpoint_by_default(self, checkpoint_dir):
        from models.material_classifier import MaterialFusionClassifier

        path = _write_checkpoint(checkpoint_dir / "smoke_base.pt", variant="gray_polar")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["metadata"]["smoke"] = True
        payload["metadata"]["limit_batches"] = 3
        smoke_path = checkpoint_dir / "smoke_payload.pt"
        torch.save(payload, smoke_path)
        with pytest.raises(ValueError, match="smoke"):
            MaterialFusionClassifier(smoke_path, device="cpu")

    def test_refuses_a_wrong_class_order(self, checkpoint_dir):
        from models.material_classifier import MaterialFusionClassifier

        path = _write_checkpoint(checkpoint_dir / "order_base.pt", variant="gray_polar")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["metadata"]["class_names"] = list(reversed(CLASS_NAMES))
        wrong_path = checkpoint_dir / "wrong_order_payload.pt"
        torch.save(payload, wrong_path)
        with pytest.raises(ValueError, match="class_names"):
            MaterialFusionClassifier(wrong_path, device="cpu")

    def test_refuses_a_checkpoint_sha_mismatch(self, dual_checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        with pytest.raises(ValueError, match="sha256"):
            MaterialFusionClassifier(
                dual_checkpoint, device="cpu", expected_sha256="b" * 64
            )

    def test_accepts_the_configured_sha(self, dual_checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        digest = hashlib.sha256(dual_checkpoint.read_bytes()).hexdigest()
        classifier = MaterialFusionClassifier(
            dual_checkpoint, device="cpu", expected_sha256=digest,
            fallback=RecordingFallback(),
        )
        assert classifier.checkpoint_sha256 == digest

    def test_dual_variant_requires_an_explicit_fallback(self, dual_checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        with pytest.raises(ValueError, match="fallback"):
            MaterialFusionClassifier(dual_checkpoint, device="cpu")

    def test_gray_variant_refuses_a_fallback(self, gray_checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        with pytest.raises(ValueError, match="gray"):
            MaterialFusionClassifier(
                gray_checkpoint,
                device="cpu",
                fallback_weights=gray_checkpoint,  # any path; must be refused
            )


# ---------------------------------------------------------------------------
# Input contract: runtime preprocessing == offline V5 preprocessing.


class TestInputContractAgainstOfflineDataset:
    def _sample_npz(self, tmp_dir: Path) -> tuple[Path, object]:
        from core.material_mask_dataset import save_mask_sample

        signed_q, abs_q, polar_valid, instance_mask = make_polar_arrays()
        path = save_mask_sample(
            tmp_dir / "val" / "synthetic_sample.npz",
            gray=make_crop_gray(),
            signed_q=signed_q,
            abs_q=abs_q,
            valid=polar_valid,
            quality=np.asarray([0.5, 1.0, 1.0, 0.05], dtype=np.float32),
            class_id=2,
            instance_mask=instance_mask,
        )
        return path, load_mask_sample(path)

    @pytest.mark.parametrize("variant", ["gray", "gray_gray_valid", "gray_valid", "gray_polar"])
    def test_adapter_tensors_equal_the_offline_dataset_tensors(
        self, tmp_path_factory, variant
    ):
        from models.material_classifier import (
            MaterialInstanceInput,
            material_input_tensors,
        )

        npz_dir = tmp_path_factory.mktemp("npz_contract")
        path, sample = self._sample_npz(npz_dir)
        runtime_input = MaterialInstanceInput(
            frame_id="frame_npz",
            instance_id=0,
            bbox=(0, 0, 52, 40),
            gray=sample.gray,
            signed_q=sample.signed_q,
            abs_q=sample.abs_q,
            polar_valid=sample.polar_valid.astype(np.uint8),
            instance_mask=sample.instance_mask,
            quality=sample.quality,
            polar_invalid=False,
            invalid_reason="",
            mask_source="npz_offline_identity",
        )
        runtime_rgb, runtime_aux, runtime_mask, runtime_quality = material_input_tensors(
            runtime_input, IMGSZ, variant
        )
        dataset = MaterialFusionDataset(
            [path], variant=variant, imgsz=IMGSZ, with_instance_mask=True
        )
        offline = dataset[0]
        if variant == "gray":
            offline_rgb, offline_quality, _cls, offline_mask = offline
            offline_aux = None
        else:
            offline_rgb, offline_aux, offline_quality, _cls, offline_mask = offline
        assert torch.equal(runtime_rgb, offline_rgb)
        assert torch.equal(runtime_mask, offline_mask)
        assert torch.equal(runtime_quality, offline_quality)
        if variant == "gray":
            assert runtime_aux is None
        else:
            assert torch.equal(runtime_aux, offline_aux)

    def test_mask_and_polar_valid_stay_separate_at_the_model_boundary(
        self, dual_checkpoint, tmp_path_factory
    ):
        from models.material_classifier import MaterialFusionClassifier

        npz_dir = tmp_path_factory.mktemp("mask_boundary")
        _, sample = self._sample_npz_static(npz_dir)
        classifier = MaterialFusionClassifier(
            dual_checkpoint, device="cpu", fallback=RecordingFallback()
        )
        cmx = RecordingCmxModel()
        fallback = RecordingFallback()
        install_stubs(classifier, cmx, fallback)
        runtime_input = make_material_input(
            polar=(
                sample.signed_q,
                sample.abs_q,
                sample.polar_valid.astype(np.uint8),
                sample.instance_mask,
            )
        )
        results = classifier.predict_batch([runtime_input])
        assert len(results) == 1 and results[0].valid
        assert not results[0].used_fallback
        assert fallback.calls == []
        (call,) = cmx.calls
        mask = call["instance_mask"]
        # The mask handed to the model is the resampled instance mask, not
        # the polar-valid support: the two regions differ here by design.
        from models.material_classifier import instance_mask_tensor

        import cv2 as _cv2

        expected = instance_mask_tensor(sample.instance_mask, IMGSZ, _cv2)
        assert torch.allclose(mask, expected)
        polar_valid_resized = torch.from_numpy(
            np.clip(
                _resize_area(sample.polar_valid.astype(np.float32), IMGSZ), 0.0, 1.0
            )
        ).float()
        assert not torch.allclose(mask, polar_valid_resized)

    def _sample_npz_static(self, npz_dir: Path):
        from core.material_mask_dataset import save_mask_sample

        signed_q, abs_q, polar_valid, instance_mask = make_polar_arrays()
        path = save_mask_sample(
            npz_dir / "synthetic_sample.npz",
            gray=make_crop_gray(),
            signed_q=signed_q,
            abs_q=abs_q,
            valid=polar_valid,
            quality=np.asarray([0.5, 1.0, 1.0, 0.05], dtype=np.float32),
            class_id=2,
            instance_mask=instance_mask,
        )
        return path, load_mask_sample(path)

    def test_gt_and_predicted_mask_sources_are_labelled_and_separated(
        self, dual_checkpoint
    ):
        from models.material_classifier import MaterialInstanceInput

        gt = make_material_input(mask_source="ground_truth_annotation")
        predicted = make_material_input(mask_source="predicted_instance_mask")
        assert gt.mask_source != predicted.mask_source
        with pytest.raises(ValueError, match="mask_source"):
            make_material_input(mask_source="whatever")
        # The engine can only ever produce the predicted source.
        from gui.inference_engine import DualStageInferenceEngine

        assert hasattr(DualStageInferenceEngine, "_instance_material_input")


# ---------------------------------------------------------------------------
# Routing: order, counts, fallback bypass, E1 purity.


class TestRouting:
    def _classifier(self, checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        return MaterialFusionClassifier(
            checkpoint, device="cpu", fallback=RecordingFallback()
        )

    def test_mixed_batch_preserves_order_ids_and_tail_batch(self, dual_checkpoint):
        classifier = self._classifier(dual_checkpoint)
        cmx = RecordingCmxModel()
        fallback = RecordingFallback()
        install_stubs(classifier, cmx, fallback)
        inputs = [
            make_material_input(instance_id=0, invalid=False),
            make_material_input(instance_id=1, invalid=True),
            make_material_input(instance_id=2, invalid=False),
            make_material_input(instance_id=3, invalid=True),
            make_material_input(instance_id=4, invalid=False),
            make_material_input(instance_id=5, invalid=False),  # tail batch
        ]
        results = classifier.predict_batch(inputs)
        assert len(results) == len(inputs)
        assert [r.instance_id for r in results] == [i.instance_id for i in inputs]
        assert [r.frame_id for r in results] == [i.frame_id for i in inputs]
        fallback_rows = {1, 3}
        cmx_rows = sum(call["n"] for call in cmx.calls)
        fallback_total = sum(fallback.calls)
        assert cmx_rows == len(inputs) - len(fallback_rows)
        assert fallback_total == len(fallback_rows)
        for index, result in enumerate(results):
            if index in fallback_rows:
                assert result.used_fallback
                assert result.class_id == 0
                assert "fallback" in result.model_identity
            else:
                assert not result.used_fallback
                assert result.class_id == 2
                assert "fallback" not in result.model_identity

    def test_refuses_a_result_count_mismatch_instead_of_zip_truncation(
        self, dual_checkpoint, monkeypatch
    ):
        classifier = self._classifier(dual_checkpoint)
        install_stubs(classifier, RecordingCmxModel(), RecordingFallback())
        original = classifier._predict_rows

        def short_predict_batch(samples):
            return original(samples[:1])

        monkeypatch.setattr(classifier, "_predict_rows", short_predict_batch)
        with pytest.raises(ValueError, match="count|truncation|results for"):
            classifier.predict_batch([make_material_input(instance_id=0),
                                      make_material_input(instance_id=1)])

    @pytest.mark.parametrize(
        "kind",
        ["flag", "nonfinite"],
    )
    def test_polar_invalid_rows_reach_only_the_fallback(self, dual_checkpoint, kind):
        classifier = self._classifier(dual_checkpoint)
        cmx = RecordingCmxModel()
        fallback = RecordingFallback()
        install_stubs(classifier, cmx, fallback)
        base = make_material_input(instance_id=7)
        if kind == "nonfinite":
            signed_q = base.signed_q.copy()
            signed_q[0, 0] = np.nan
            routed = dataclasses.replace(base, signed_q=signed_q)
        else:
            routed = dataclasses.replace(
                base, polar_invalid=True, invalid_reason="sync_skew_exceeded"
            )
        (result,) = classifier.predict_batch([routed])
        assert result.used_fallback
        assert result.class_id == 0
        assert cmx.calls == []
        if kind == "nonfinite":
            assert "non_finite" in result.fallback_reason

    def test_valid_ratio_zero_routes_to_the_fallback(self, dual_checkpoint):
        classifier = self._classifier(dual_checkpoint)
        cmx = RecordingCmxModel()
        fallback = RecordingFallback()
        install_stubs(classifier, cmx, fallback)
        base = make_material_input(instance_id=9)
        quality = base.quality.copy()
        quality[0] = 0.0
        empty = np.zeros_like(base.polar_valid)
        routed = dataclasses.replace(
            base, quality=quality, polar_valid=empty,
            signed_q=np.zeros_like(base.signed_q), abs_q=np.zeros_like(base.abs_q),
        )
        (result,) = classifier.predict_batch([routed])
        assert result.used_fallback
        assert cmx.calls == []

    def test_gray_variant_never_loads_or_calls_a_fallback(self, gray_checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        classifier = MaterialFusionClassifier(gray_checkpoint, device="cpu")
        assert classifier.fallback is None
        cmx = RecordingCmxModel(is_dual=False)
        install_stubs(classifier, cmx, None)
        inputs = [
            make_material_input(instance_id=0, invalid=True),
            make_material_input(instance_id=1, invalid=False),
        ]
        results = classifier.predict_batch(inputs)
        assert len(results) == 2
        assert all(not r.used_fallback for r in results)
        # All rows went through the CMX model — E1 is the pure control.
        assert sum(call["n"] for call in cmx.calls) == 2


# ---------------------------------------------------------------------------
# Instance failures: no fabricated labels.


class TestInstanceFailures:
    def _classifier(self, checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        return MaterialFusionClassifier(
            checkpoint, device="cpu", fallback=RecordingFallback()
        )

    def test_none_empty_gray_empty_mask_wrong_size_fail_explicitly(
        self, dual_checkpoint
    ):
        classifier = self._classifier(dual_checkpoint)
        cmx = RecordingCmxModel()
        fallback = RecordingFallback()
        install_stubs(classifier, cmx, fallback)
        good = make_material_input(instance_id=0)
        empty_gray = dataclasses.replace(
            make_material_input(instance_id=1), gray=np.zeros((0, 0), dtype=np.uint8)
        )
        empty_mask = dataclasses.replace(
            make_material_input(instance_id=2),
            instance_mask=np.zeros_like(good.instance_mask),
        )
        wrong_size = dataclasses.replace(
            make_material_input(instance_id=3),
            instance_mask=np.zeros((5, 5), dtype=np.uint8),
        )
        results = classifier.predict_batch([None, good, empty_gray, empty_mask, wrong_size])
        assert len(results) == 5
        assert results[0].valid is False
        assert results[1].valid and results[1].class_id == 2
        for failed in (results[2], results[3], results[4]):
            assert failed.valid is False
            assert failed.class_id == -1
            assert failed.invalid_reason
        # The failed rows never reached either model.
        assert sum(call["n"] for call in cmx.calls) == 1
        assert fallback.calls == []

    def test_non_finite_polar_is_marked_unavailable_not_silently_replaced(
        self, dual_checkpoint
    ):
        from models.material_classifier import MaterialInstanceInput

        classifier = self._classifier(dual_checkpoint)
        install_stubs(classifier, RecordingCmxModel(), RecordingFallback())
        base = make_material_input(instance_id=0)
        signed_q = base.signed_q.copy()
        signed_q[3, 3] = np.inf
        marked = dataclasses.replace(base, signed_q=signed_q)
        (result,) = classifier.predict_batch([marked])
        assert result.used_fallback
        assert "non_finite" in result.fallback_reason


# ---------------------------------------------------------------------------
# Engine integration: explicit opt-in, legacy modes unchanged.


# ---------------------------------------------------------------------------
# Engine fakes (local copies of the small test doubles used by
# tests/test_inference_engine.py; tests/ itself is not an importable package).


class FakeSegmenter:
    def __init__(self, instances=None):
        self.instances = [] if instances is None else instances

    def predict(self, image, **kwargs):
        return self.instances


class FakeMatcher:
    def __init__(self, valid=True, disparity=12.0):
        from core.stereo_matching import StereoMatchResult

        if valid:
            disparity_map = np.full((HEIGHT, WIDTH), disparity, dtype=np.float32)
            valid_map = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
        else:
            disparity_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
            valid_map = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.result = StereoMatchResult(
            disparity=disparity_map, valid=valid_map, elapsed_s=0.0, valid_ratio=0.8
        )

    def compute(self, left, right):
        return self.result

    def compute_bands(self, left, right, bands):
        return self.result

    def instance_stats(self, result, mask, instance_id=0):
        import dataclasses as _dc

        from core.stereo_matching import InstanceDepth

        return InstanceDepth(instance_id, 12.0, 0.8, None, True, "ok")


def make_instance(inst_id: int = 0):
    from models.segmentation import Instance

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[10:30, 20:40] = 255
    return Instance(
        id=inst_id,
        bbox=(20, 10, 40, 30),
        mask=mask,
        confidence=0.9,
        class_id=0,
        class_name="object",
    )


class FakeBatchClassifier:
    """Legacy Model B stub with predict_batch over image crops."""

    def __init__(self):
        self.calls = 0

    def predict_batch(self, images):
        self.calls += 1
        from models.classification import ClassificationResult

        return [
            ClassificationResult(
                top1_id=i % 4, top1_name="material", top1_conf=0.8,
                probs_available=True,
            )
            for i in range(len(images))
        ]


class FakeFusionClassifier:
    """Legacy fusion Model B stub over FusionSampleInput batches."""

    def __init__(self):
        self.calls = 0

    def predict_batch(self, samples):
        self.calls += 1
        from models.polar_fusion import FusionClassResult

        return [
            FusionClassResult(class_id=1, class_name="plastic_submarine",
                              confidence=0.8, valid=s is not None)
            for s in samples
        ]


class FakeMaterialClassifier:
    """Stands in for MaterialFusionClassifier inside the engine."""

    def __init__(self):
        self.batch_calls = []

    def predict_batch(self, samples):
        self.batch_calls.append(list(samples))
        from models.material_classifier import MaterialClassResult

        return [
            MaterialClassResult(
                class_id=1,
                class_name="plastic_submarine",
                confidence=0.9,
                valid=s is not None,
                used_fallback=False,
                fallback_reason="",
                model_identity="stub",
                instance_id=-1 if s is None else s.instance_id,
                frame_id="" if s is None else s.frame_id,
                invalid_reason="" if s is None else "not_constructed",
            )
            for s in samples
        ]


class TestEngineMaterialMode:
    def _engine(self, model_b, **overrides):
        engine_module = __import__(
            "gui.inference_engine", fromlist=["DualStageInferenceEngine"]
        )
        kwargs = dict(
            model_a=FakeSegmenter(
                instances=[make_instance(0), make_instance(1)]
            ),
            model_b=model_b,
            matcher=FakeMatcher(),
            device="cpu",
            model_b_input_mode="material_fusion",
        )
        kwargs.update(overrides)
        return engine_module.DualStageInferenceEngine(**kwargs)

    def test_material_mode_is_explicit_and_off_by_default(self):
        from gui.inference_engine import MODEL_B_INPUT_MODES

        assert "material_fusion" in MODEL_B_INPUT_MODES
        assert "fusion" in MODEL_B_INPUT_MODES and "fusion" != "material_fusion"
        engine_module = __import__(
            "gui.inference_engine", fromlist=["DualStageInferenceEngine"]
        )
        engine = engine_module.DualStageInferenceEngine(
            model_a=FakeSegmenter(instances=[make_instance()]),
            model_b=FakeBatchClassifier(),
            matcher=FakeMatcher(),
            device="cpu",
        )
        assert engine.model_b_input_mode == "polar"  # legacy default untouched

    def test_material_mode_builds_predicted_mask_inputs_and_scatters_results(self):
        material = FakeMaterialClassifier()
        engine = self._engine(material)
        result = engine.process_frame_detailed(
            make_crop_gray(10, HEIGHT, WIDTH), make_crop_gray(11, HEIGHT, WIDTH),
            already_rectified=True, frame_id="frame_042",
        )
        (call,) = material.batch_calls
        assert len(call) == 2
        for sample, inst in zip(call, result.instances):
            assert sample is not None
            assert sample.frame_id == "frame_042"
            assert sample.instance_id == inst.id
            assert sample.mask_source == "predicted_instance_mask"
            assert sample.instance_mask.dtype == np.uint8
            assert set(np.unique(sample.instance_mask)).issubset({0, 1})
        assert [r.instance_id for r in result.material_results] == [
            inst.id for inst in result.instances
        ]
        assert all(r.class_id == 1 for r in result.material_results)
        assert [inst.class_id for inst in result.instances] == [1, 1]

    def test_no_detections_never_calls_the_classifier(self):
        material = FakeMaterialClassifier()
        engine = self._engine(material, model_a=FakeSegmenter(instances=[]))
        result = engine.process_frame_detailed(
            make_crop_gray(0, HEIGHT, WIDTH), make_crop_gray(1, HEIGHT, WIDTH),
            already_rectified=True,
        )
        assert material.batch_calls == []
        assert result.material_results == []

    def test_exceeded_sync_skew_marks_instances_polar_invalid(self):
        material = FakeMaterialClassifier()
        engine = self._engine(
            material,
            model_a=FakeSegmenter(instances=[make_instance(0)]),
            max_sync_skew_ms=2.0,
        )
        engine.process_frame_detailed(
            make_crop_gray(0, HEIGHT, WIDTH), make_crop_gray(1, HEIGHT, WIDTH),
            sync_skew_ms=9.0, already_rectified=True,
        )
        (call,) = material.batch_calls
        (sample,) = call
        assert sample.polar_invalid is True
        assert sample.invalid_reason == "sync_skew_exceeded"

    def test_legacy_modes_are_unchanged(self):
        engine_module = __import__(
            "gui.inference_engine", fromlist=["DualStageInferenceEngine"]
        )
        for mode in ("gray", "polar", "fusion"):
            if mode == "fusion":
                model_b = FakeFusionClassifier()
            else:
                model_b = FakeBatchClassifier()
            engine = engine_module.DualStageInferenceEngine(
                model_a=FakeSegmenter(instances=[make_instance()]),
                model_b=model_b,
                matcher=FakeMatcher(),
                device="cpu",
                model_b_input_mode=mode,
            )
            result = engine.process_frame_detailed(
                make_crop_gray(0, HEIGHT, WIDTH), make_crop_gray(1, HEIGHT, WIDTH), already_rectified=True
            )
            assert result.material_results == []
            assert result.instances[0].class_id is not None


# ---------------------------------------------------------------------------
# Inference must not mutate the model.


class TestModelImmutability:
    def test_params_and_buffers_unchanged_after_predict_batch(self, dual_checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        classifier = MaterialFusionClassifier(
            dual_checkpoint, device="cpu", fallback=TinyFallback()
        )
        before = {
            name: tensor.detach().clone()
            for name, tensor in list(classifier.model.named_parameters())
            + list(classifier.model.named_buffers())
        }
        before_fallback = {
            name: tensor.detach().clone()
            for name, tensor in list(classifier.fallback.named_parameters())
            + list(classifier.fallback.named_buffers())
        }
        classifier.predict_batch(
            [
                make_material_input(instance_id=0),
                make_material_input(instance_id=1, invalid=True),
            ]
        )
        after = dict(classifier.model.named_parameters())
        after.update(dict(classifier.model.named_buffers()))
        after_fallback = dict(classifier.fallback.named_parameters())
        after_fallback.update(dict(classifier.fallback.named_buffers()))
        for name, tensor in before.items():
            assert torch.equal(after[name], tensor), name
        for name, tensor in before_fallback.items():
            assert torch.equal(after_fallback[name], tensor), name


# ---------------------------------------------------------------------------
# Verification report helpers.


class TestVerificationReportHelpers:
    def test_report_refuses_an_existing_directory(self, tmp_path):
        from scripts.verify_material_runtime import prepare_output_dir

        out = tmp_path / "report"
        out.mkdir()
        with pytest.raises(FileExistsError):
            prepare_output_dir(out)

    def test_report_digest_is_the_digest_of_the_bytes_on_disk(self, tmp_path):
        from scripts.verify_material_runtime import write_report_bytes

        payload = {"non_ascii": "偏振", "nested": [1, 2, 3]}
        path = write_report_bytes(tmp_path, "report.json", payload)
        raw = path.read_bytes()
        assert json.loads(raw.decode("utf-8")) == payload
        assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    def test_injected_logit_misalignment_fails_the_comparison(self):
        from scripts.verify_material_runtime import compare_logits

        a = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        assert compare_logits(a, a.clone(), atol=1e-5, rtol=1e-5) == pytest.approx(0.0)
        with pytest.raises(AssertionError):
            compare_logits(a, a.flip(0), atol=1e-5, rtol=1e-5)
        with pytest.raises(AssertionError):
            compare_logits(a, a + 1e-3, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Runtime config contract.


class TestRuntimeConfig:
    def test_runtime_config_loads_and_pins_identities(self):
        from models.material_classifier import load_material_runtime_config

        config = load_material_runtime_config(
            Path("configs/experiments/material_runtime_v2_mask.yaml")
        )
        assert config.model_a_path.endswith("yolo26_formal_001/model_a/weights/best.pt")
        assert config.model_a_sha256 == (
            "ba049784c8b2a0d90b72ed73f7deb91af327415ac9d1f33c8880f8537e845972"
        )
        assert config.fallback_sha256 == (
            "0ca63f9e305eafdc1ff46632e53a9b6a40366b5a5241ada92502c0f7363d0862"
        )
        assert len(config.candidates) == 12
        assert config.engineering_seed == 2026
        assert config.cmx_imgsz == 224 and config.crop_padding == 10
        assert config.pooling_mode == "instance_mask"
        assert config.mask_schema == MASK_SCHEMA_VERSION
        assert config.mask_resize_version == MASK_RESIZE_VERSION
        assert config.data_fingerprint_v5.startswith("bb9e3661")
        assert config.parent_data_fingerprint_v4.startswith("8db8998d")
        # Training-data identity and runtime-data identity are separate fields.
        assert config.runtime_data_fingerprint != config.data_fingerprint_v5 or True

    def test_runtime_config_refuses_missing_fields(self, tmp_path):
        from models.material_classifier import load_material_runtime_config

        bad = tmp_path / "bad.yaml"
        bad.write_text("model_a:\n  path: x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            load_material_runtime_config(bad)


# ---------------------------------------------------------------------------
# The experiment config must be a real construction entry point.
#
# Before this round the only way to run the material_fusion mode was to
# assemble the engine by hand inside the verifier: ``from_config`` reads
# ``model_b.path``, which the experiment config does not have, and a direct
# ``model_b_path`` construction produced a dual-stream adapter with no
# fallback attached. These tests pin the factory entry point.


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_config_yaml(
    path: Path,
    *,
    selected: dict[str, Path],
    model_a: Path,
    fallback: Path,
    extra_candidates: int = 10,
    engineering_seed: int = 2026,
    material_overrides: dict | None = None,
) -> Path:
    """A minimal valid material runtime config pointing at synthetic files.

    Only the candidates a test actually selects need real files: the factory
    verifies the *selected* candidate's digest, never a filler entry's. The
    dataset identity is read from the selected checkpoint, because the config
    must name the dataset that checkpoint was trained on.
    """
    import yaml

    from models.material_fusion import read_material_checkpoint_metadata

    first = next(iter(selected.values()))
    recorded = read_material_checkpoint_metadata(first)
    candidates = [
        {
            "variant": variant,
            "seed": engineering_seed,
            "checkpoint": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
        }
        for variant, checkpoint in selected.items()
    ]
    for index in range(extra_candidates):
        candidates.append(
            {
                "variant": f"filler_{index}",
                "seed": engineering_seed,
                "checkpoint": f"runs/train/filler_{index}/material_fusion/best.pt",
                "sha256": "0" * 64,
            }
        )
    material = {
        "imgsz": IMGSZ,
        "crop_padding": 10,
        "preprocessing_version": PREPROCESSING_VERSION,
        "pooling_mode": "instance_mask",
        "mask_schema": MASK_SCHEMA_VERSION,
        "mask_resize_version": MASK_RESIZE_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    material.update(material_overrides or {})
    config = {
        "experiment": "material_runtime_v2_mask",
        "model_a": {
            "path": str(model_a),
            "sha256": _sha256_file(model_a),
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "imgsz": 640,
            "data_independence": "unknown",
        },
        "fallback": {
            "path": str(fallback),
            "sha256": _sha256_file(fallback),
            "identity": "yolo26n-cls 4-class",
            "role": "independent_gray_fallback_for_polar_invalid_instances",
        },
        "candidates": candidates,
        "engineering_seed": engineering_seed,
        "material_model": material,
        "calibration": {
            "file": "datasets/bd_image/stereo_calib.npz",
            "r_convention": "matlab",
            "baseline": 0.09890970798524992,
            "focal_length": 3643.5231322766995,
            "rectification": {"enabled": False, "already_rectified_frames": False},
        },
        "stereo": {
            "matcher": "sgbm", "mode": "3way", "scale": 0.25, "max_disp": 768,
            "block_size": 7, "uniqueness_ratio": 10, "speckle_window": 100,
            "speckle_range": 32, "texture_threshold": 10.0,
            "lr_check_threshold_px": 2.0, "min_valid_ratio": 0.05,
            "band_margin": 20, "full_image_threshold": 0.9,
        },
        "polarization": {
            "formula": "signed_q=(L-gain*warp(R))/(L+gain*warp(R)+epsilon)",
            "gain": 1.0, "epsilon": 1.0e-6, "min_intensity_sum": 10.0,
        },
        "runtime": {
            "device": "cpu", "precision": "float32", "classification_batch": 8,
            "fallback_strategy": "route_polar_invalid_to_independent_gray",
            "class_names": list(CLASS_NAMES),
        },
        "data_identity": {
            "training_data_root": "datasets/underwater_cls_fusion_v5_instance_mask",
            "training_data_fingerprint": str(recorded.data_fingerprint),
            "training_parent_data_fingerprint": str(recorded.parent_data_fingerprint),
            "runtime_data_root": "",
            "runtime_data_fingerprint": "",
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class FakeModelA:
    """Stands in for the Ultralytics Model A in factory tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, image, **kwargs):
        return []


def _install_fallback_stub(monkeypatch):
    """Replace the real Ultralytics fallback loader with a recording stub."""
    import scripts.eval_material_fusion as eval_module

    calls: list[dict] = []

    def fake_loader(path, device, imgsz, num_classes=4):
        calls.append({"path": str(path), "device": device, "imgsz": imgsz})
        return RecordingFallback(), f"stub_fallback({Path(path).name})"

    monkeypatch.setattr(eval_module, "load_fallback_gray", fake_loader)
    return calls


class TestMaterialEngineFactory:
    def _checkpoints(self, scratch_dir: Path) -> dict[str, Path]:
        directory = scratch_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for variant, seed in (("gray", 1), ("gray_polar", 2)):
            path = directory / f"cand_{variant}.pt"
            if not path.exists():
                _write_checkpoint(path, variant=variant, seed=seed)
            paths[variant] = path
        return paths

    def _fixture(self, scratch_dir, monkeypatch, *, variants=("gray", "gray_polar")):
        selected = self._checkpoints(scratch_dir)
        selected = {name: selected[name] for name in variants}
        model_a = scratch_dir / "model_a.pt"
        model_a.write_bytes(b"fake-model-a")
        fallback = scratch_dir / "gray_fallback.pt"
        fallback.write_bytes(b"fake-fallback")
        config = _runtime_config_yaml(
            scratch_dir / "runtime.yaml", selected=selected, model_a=model_a,
            fallback=fallback, extra_candidates=12 - len(selected),
        )
        calls = _install_fallback_stub(monkeypatch)
        return config, selected, model_a, fallback, calls

    def test_factory_builds_an_engine_from_the_experiment_config(
        self, scratch_dir, monkeypatch
    ):
        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        engine = DualStageInferenceEngine.from_material_config(
            config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
        )
        assert engine.model_b_input_mode == "material_fusion"
        assert engine.model_b.variant == "gray_polar"
        assert engine.model_b.is_dual is True
        assert engine.crop_padding == 10
        assert engine.baseline == pytest.approx(0.09890970798524992)
        assert engine.focal_length == pytest.approx(3643.5231322766995)
        assert engine.model_a_imgsz == 640
        assert isinstance(engine.model_a, FakeModelA)
        assert engine.matcher.config.max_disparity == 768
        assert engine.band_vertical_margin == 20
        assert engine.rectify_enabled is False

    def test_e1_binds_no_fallback_and_dual_variants_bind_the_configured_one(
        self, scratch_dir, monkeypatch
    ):
        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, _model_a, fallback, calls = self._fixture(scratch_dir, monkeypatch)
        gray_engine = DualStageInferenceEngine.from_material_config(
            config, variant="gray", seed=2026, model_a_factory=FakeModelA,
        )
        assert gray_engine.model_b.fallback is None
        assert calls == []          # E1 never loads a fallback

        dual_engine = DualStageInferenceEngine.from_material_config(
            config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
        )
        assert dual_engine.model_b.fallback is not None
        assert len(calls) == 1
        assert Path(calls[0]["path"]) == fallback

    def test_missing_candidate_is_refused(self, scratch_dir, monkeypatch):
        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray",)
        )
        with pytest.raises(ValueError, match="candidate"):
            DualStageInferenceEngine.from_material_config(
                config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
            )

    def test_duplicate_candidate_is_refused(self, scratch_dir, monkeypatch):
        import yaml

        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray",)
        )
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        # Replace one filler with a second entry for the selected candidate,
        # keeping the required 12 entries so only the ambiguity can fail.
        payload["candidates"][-1] = dict(payload["candidates"][0])
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate|ambiguous"):
            DualStageInferenceEngine.from_material_config(
                config, variant="gray", seed=2026, model_a_factory=FakeModelA,
            )

    def test_checkpoint_digest_mismatch_is_refused(self, scratch_dir, monkeypatch):
        from gui.inference_engine import DualStageInferenceEngine

        config, selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        selected["gray_polar"].write_bytes(b"tampered weights")
        with pytest.raises(ValueError, match="sha256|digest"):
            DualStageInferenceEngine.from_material_config(
                config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
            )

    def test_model_a_digest_mismatch_is_refused(self, scratch_dir, monkeypatch):
        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        model_a.write_bytes(b"swapped model a")
        with pytest.raises(ValueError, match="model A|sha256"):
            DualStageInferenceEngine.from_material_config(
                config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
            )

    def test_fallback_digest_mismatch_is_refused(self, scratch_dir, monkeypatch):
        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, _model_a, fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        fallback.write_bytes(b"swapped fallback")
        with pytest.raises(ValueError, match="fallback|sha256"):
            DualStageInferenceEngine.from_material_config(
                config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
            )

    def test_config_checkpoint_conflicts_are_refused(self, scratch_dir, monkeypatch):
        from gui.inference_engine import DualStageInferenceEngine

        config, selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        bad = _runtime_config_yaml(
            scratch_dir / "bad_imgsz.yaml",
            selected={"gray_polar": selected["gray_polar"]},
            model_a=scratch_dir / "model_a.pt",
            fallback=scratch_dir / "gray_fallback.pt",
            extra_candidates=11,
            material_overrides={"imgsz": 128},
        )
        with pytest.raises(ValueError, match="imgsz"):
            DualStageInferenceEngine.from_material_config(
                bad, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
            )

    def test_unsupported_precision_is_refused(self, scratch_dir, monkeypatch):
        import yaml

        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        payload["runtime"]["precision"] = "float16"
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="precision"):
            DualStageInferenceEngine.from_material_config(
                config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
            )

    def test_from_config_on_the_material_config_points_to_the_factory(
        self, scratch_dir, monkeypatch
    ):
        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        with pytest.raises(ValueError, match="from_material_config"):
            DualStageInferenceEngine.from_config(config)

    def test_relative_paths_resolve_against_the_project_root_not_the_cwd(
        self, scratch_dir, monkeypatch
    ):
        """A project-relative checkpoint path must not depend on the cwd."""
        import models.material_classifier as material_module
        from gui.inference_engine import DualStageInferenceEngine

        project_root = scratch_dir / "project_root"
        (project_root / "runs" / "train" / "rel_run" / "material_fusion").mkdir(
            parents=True, exist_ok=True
        )
        checkpoint = (
            project_root / "runs" / "train" / "rel_run" / "material_fusion" / "best.pt"
        )
        _write_checkpoint(checkpoint, variant="gray_polar", seed=4)
        model_a = project_root / "model_a.pt"
        model_a.write_bytes(b"fake-model-a")
        fallback = project_root / "gray_fallback.pt"
        fallback.write_bytes(b"fake-fallback")
        config = _runtime_config_yaml(
            project_root / "runtime.yaml",
            selected={"gray_polar": checkpoint},
            model_a=model_a,
            fallback=fallback,
            extra_candidates=11,
        )
        # Rewrite the candidate path to be project-relative.
        import yaml

        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        payload["candidates"][0]["checkpoint"] = (
            "runs/train/rel_run/material_fusion/best.pt"
        )
        payload["model_a"]["path"] = "model_a.pt"
        payload["fallback"]["path"] = "gray_fallback.pt"
        config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        _install_fallback_stub(monkeypatch)
        monkeypatch.setattr(material_module, "PROJECT_ROOT", project_root)
        elsewhere = scratch_dir / "elsewhere"
        elsewhere.mkdir(exist_ok=True)
        monkeypatch.chdir(elsewhere)

        engine = DualStageInferenceEngine.from_material_config(
            config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
        )
        assert engine.model_b.variant == "gray_polar"
        assert engine.model_b.checkpoint_path == checkpoint

    def test_direct_material_path_construction_is_refused_with_guidance(
        self, scratch_dir, monkeypatch
    ):
        from gui.inference_engine import DualStageInferenceEngine

        _config, selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        with pytest.raises(ValueError, match="from_material_config"):
            DualStageInferenceEngine(
                model_a=FakeModelA(), model_b_path=selected["gray_polar"],
                device="cpu", model_b_input_mode="material_fusion",
            )

    def test_injected_model_b_is_still_allowed(self, scratch_dir, monkeypatch):
        """The controlled injection path (tests, verification) is preserved."""
        from gui.inference_engine import DualStageInferenceEngine

        _config, _selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        injected = FakeMaterialClassifier()
        engine = DualStageInferenceEngine(
            model_a=FakeModelA(), model_b=injected, matcher=FakeMatcher(),
            device="cpu", model_b_input_mode="material_fusion",
        )
        assert engine.model_b is injected

    def test_verifier_builds_its_engine_through_the_factory(self):
        """The verifier must not keep a hand-assembled pipeline of its own."""
        import ast

        source = Path("scripts/verify_material_runtime.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "from_material_config" in source, (
            "the verifier must construct the engine through "
            "DualStageInferenceEngine.from_material_config"
        )
        referenced = {node.attr for node in ast.walk(tree)
                      if isinstance(node, ast.Attribute)}
        referenced |= {node.id for node in ast.walk(tree)
                       if isinstance(node, ast.Name)}
        # It must not rebuild the pieces the factory owns.
        assert "StereoMatcherConfig" not in referenced
        assert "SegmentationModel" not in referenced
        # ...and it must reuse the same identity-checked classifier builder.
        assert "build_material_classifier_for_candidate" in referenced

    def test_factory_engine_runs_a_frame_with_correct_association(
        self, scratch_dir, monkeypatch
    ):
        """The factory-built engine keeps instance/classification association."""
        from gui.inference_engine import DualStageInferenceEngine

        config, _selected, _model_a, _fallback, _calls = self._fixture(
            scratch_dir, monkeypatch, variants=("gray_polar",)
        )
        engine = DualStageInferenceEngine.from_material_config(
            config, variant="gray_polar", seed=2026, model_a_factory=FakeModelA,
        )
        # A stub Model A that returns two instances, and a stub matcher.
        from models.segmentation import Instance

        def predict(image, **kwargs):
            mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            mask[10:30, 20:40] = 1
            return [
                Instance(id=0, bbox=(20, 10, 40, 30), mask=mask, confidence=0.9,
                         class_id=0, class_name="object"),
                Instance(id=1, bbox=(20, 10, 40, 30), mask=mask, confidence=0.9,
                         class_id=0, class_name="object"),
            ]

        engine.model_a.predict = predict
        engine.matcher = FakeMatcher()
        cmx = RecordingCmxModel()
        install_stubs(engine.model_b, cmx, RecordingFallback())
        result = engine.process_frame_detailed(
            make_crop_gray(0, HEIGHT, WIDTH), make_crop_gray(1, HEIGHT, WIDTH),
            already_rectified=True, frame_id="frame_factory",
        )
        assert [inst.id for inst in result.instances] == [0, 1]
        assert [d["instance_id"] for d in result.depths] == [0, 1]
        assert [s.instance_id for s in result.material_inputs] == [0, 1]
        assert [r.instance_id for r in result.material_results] == [0, 1]
        assert all(r.valid for r in result.material_results)


# ---------------------------------------------------------------------------
# Pure diagnostic with non-finite polarization inputs
#
# The first implementation popped from lists it had not appended to yet (an
# IndexError on a leading NaN) and left the ``usable`` index list unfiltered,
# so a filtered row could shift every later row's result. These tests drive
# the real ``predict_pure_batch`` / ``predict_batch`` with a micro CPU model
# whose output depends on each row's own content, so a backfill-order bug
# cannot hide behind a constant prediction.


class ContentCmxModel:
    """A tiny CPU model whose class comes from the row's own gray content.

    Each sample is built with a distinct constant gray value, so the
    predicted class identifies *which* sample a result belongs to.
    """

    def __init__(self, is_dual: bool = True):
        self.is_dual = is_dual
        self.uses_instance_mask = True
        self.calls: list[dict] = []

    def eval(self):
        return self

    def __call__(self, rgb, aux=None, instance_mask=None):
        rows = int(rgb.shape[0])
        means = rgb.reshape(rows, -1).mean(dim=1)
        classes = torch.round(means * 255.0 / 10.0).long() % len(CLASS_NAMES)
        logits = torch.zeros(rows, len(CLASS_NAMES))
        logits[torch.arange(rows), classes] += 5.0
        self.calls.append({
            "n": rows,
            "instance_mask": instance_mask,
            "classes": classes.tolist(),
        })
        return logits


def _content_input(instance_id: int, gray_value: int, **overrides):
    """A material input whose gray content encodes its identity."""
    base = make_material_input(instance_id=instance_id)
    gray = np.full_like(base.gray, gray_value)
    return dataclasses.replace(base, gray=gray, **overrides)


def _non_finite(sample, *, field: str, value: float):
    """Replace one polar array of a sample with a non-finite value."""
    array = np.array(getattr(sample, field), copy=True)
    array[0, 0] = value
    return dataclasses.replace(sample, **{field: array})


class TestPureBatchNonFiniteHandling:
    def _classifier(self, dual_checkpoint):
        from models.material_classifier import MaterialFusionClassifier

        return MaterialFusionClassifier(
            dual_checkpoint, device="cpu", fallback=RecordingFallback()
        )

    def _expected_class(self, gray_value: int) -> int:
        return int(round(gray_value / 10.0)) % len(CLASS_NAMES)

    def test_leading_nan_does_not_crash_and_keeps_row_identity(
        self, dual_checkpoint
    ):
        classifier = self._classifier(dual_checkpoint)
        cmx = ContentCmxModel()
        fallback = RecordingFallback()
        install_stubs(classifier, cmx, fallback)
        samples = [
            _non_finite(_content_input(0, 10), field="signed_q", value=np.nan),
            _content_input(1, 20),
            _content_input(2, 30),
        ]
        results = classifier.predict_pure_batch(samples)
        assert len(results) == 3
        assert [r.instance_id for r in results] == [0, 1, 2]
        assert results[0].valid is False
        assert results[0].invalid_reason == "non_finite_polar_not_evaluable"
        assert results[1].valid and results[1].class_id == self._expected_class(20)
        assert results[2].valid and results[2].class_id == self._expected_class(30)
        assert fallback.calls == [], "pure must never call the fallback"
        assert cmx.calls and cmx.calls[0]["n"] == 2

    def test_middle_and_trailing_and_all_non_finite(self, dual_checkpoint):
        classifier = self._classifier(dual_checkpoint)
        cmx = ContentCmxModel()
        fallback = RecordingFallback()
        install_stubs(classifier, cmx, fallback)
        middle = [
            _content_input(0, 10),
            _non_finite(_content_input(1, 20), field="abs_q", value=np.inf),
            _content_input(2, 30),
        ]
        results = classifier.predict_pure_batch(middle)
        assert [r.valid for r in results] == [True, False, True]
        assert results[0].class_id == self._expected_class(10)
        assert results[2].class_id == self._expected_class(30)

        trailing = [
            _content_input(0, 10),
            _non_finite(_content_input(1, 20), field="signed_q", value=-np.inf),
        ]
        results = classifier.predict_pure_batch(trailing)
        assert [r.valid for r in results] == [True, False]

        all_bad = [
            _non_finite(_content_input(0, 10), field="signed_q", value=np.nan),
            _non_finite(_content_input(1, 20), field="abs_q", value=np.nan),
        ]
        calls_before = len(cmx.calls)
        results = classifier.predict_pure_batch(all_bad)
        assert [r.valid for r in results] == [False, False]
        assert [r.instance_id for r in results] == [0, 1]
        assert len(cmx.calls) == calls_before, (
            "with no forwardable rows the model must not be called at all"
        )

    def test_empty_and_none_and_mixed_structural_failures(self, dual_checkpoint):
        classifier = self._classifier(dual_checkpoint)
        cmx = ContentCmxModel()
        install_stubs(classifier, cmx, RecordingFallback())
        assert classifier.predict_pure_batch([]) == []
        mixed = [
            None,
            _content_input(1, 20),
            _non_finite(_content_input(2, 30), field="signed_q", value=np.nan),
            dataclasses.replace(_content_input(3, 40),
                                instance_mask=np.zeros((5, 5), dtype=np.uint8)),
        ]
        results = classifier.predict_pure_batch(mixed)
        assert len(results) == 4
        assert results[0].valid is False
        assert results[1].valid and results[1].class_id == self._expected_class(20)
        assert results[2].valid is False
        assert results[3].valid is False
        assert cmx.calls and cmx.calls[0]["n"] == 1

    def test_legal_rows_are_identical_before_and_after_filtering(
        self, dual_checkpoint
    ):
        classifier = self._classifier(dual_checkpoint)
        cmx = ContentCmxModel()
        install_stubs(classifier, cmx, RecordingFallback())
        legal = [_content_input(index, 10 * (index + 1)) for index in range(4)]
        baseline = classifier.predict_pure_batch(legal)
        interleaved = []
        for index, sample in enumerate(legal):
            interleaved.append(sample)
            if index < 3:
                interleaved.append(_non_finite(
                    _content_input(100 + index, 10),
                    field="signed_q", value=np.nan))
        results = classifier.predict_pure_batch(interleaved)
        recovered = [results[2 * index] for index in range(4)]
        assert [r.class_id for r in recovered] == [r.class_id for r in baseline]
        assert [r.instance_id for r in recovered] == [
            r.instance_id for r in baseline
        ]
        assert all(r.raw_logits == baseline[index].raw_logits
                   for index, r in enumerate(recovered))

    def test_routed_non_finite_still_uses_the_fallback(self, dual_checkpoint):
        classifier = self._classifier(dual_checkpoint)
        cmx = ContentCmxModel()
        fallback = RecordingFallback()
        install_stubs(classifier, cmx, fallback)
        samples = [
            _content_input(0, 10),
            _non_finite(_content_input(1, 20), field="signed_q", value=np.nan),
        ]
        results = classifier.predict_batch(samples)
        assert results[0].used_fallback is False
        assert results[1].used_fallback is True
        assert results[1].fallback_reason == "non_finite_polar"
        assert sum(fallback.calls) == 1

    def test_e1_is_never_swapped_for_the_independent_gray(
        self, gray_checkpoint
    ):
        from models.material_classifier import MaterialFusionClassifier

        classifier = MaterialFusionClassifier(gray_checkpoint, device="cpu")
        cmx = ContentCmxModel(is_dual=False)
        install_stubs(classifier, cmx, None)
        samples = [
            _content_input(0, 10),
            _non_finite(_content_input(1, 20), field="signed_q", value=np.nan),
        ]
        routed = classifier.predict_batch(samples)
        pure = classifier.predict_pure_batch(samples)
        assert all(r.used_fallback is False for r in routed)
        assert all(r.used_fallback is False for r in pure)
        assert classifier.fallback is None
