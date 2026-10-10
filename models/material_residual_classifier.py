"""Runtime adapter for the locked residual candidate
(material_conditional_residual_v1, fit manual_20261009T185911).

This module wires the residual candidate — frozen MiT-B0 gray feature G,
the frozen per-fold P0 linear head and the frozen correction branch
(``Linear(35,8) -> SiLU -> Linear(8,4,bias=False)`` with the last layer
zero-initialized) — into the inference engine's ``material_fusion`` input
mode. It is NOT a CMX checkpoint and never enters the historical CMX
loader; every file it loads is pinned by sha256 in
``configs/experiments/material_residual_runtime_v1.yaml`` and re-verified
here before anything runs.

No new model mathematics exists here:

- the gray feature reuses the offline extraction path
  (:func:`models.polar_fusion._fusion_tensors`, the resized instance mask
  and :func:`models.material_fusion.mask_level_weights` pooling over the
  last MiT-B0 stage);
- the local C14/S16 statistics reuse
  :func:`core.polar_local_features.build_local_polar_stats` /
  :func:`build_local_control_features` verbatim;
- the P0 design reuses :func:`core.material_probe.assemble_design_matrix`
  with the frozen per-fold scaler (C/S slots exactly zero);
- the correction reuses :func:`core.material_conditional_residual` verbatim.

Fallback semantics are the candidate's own frozen rules: a missing or
non-finite polarization measurement gates ``valid_s`` to zero, so the
correction falls back to the candidate's OWN frozen P0 exactly (the raw
P0 logits, bit for bit). An unusable gray crop is an explicit per-instance
failure and reaches neither the encoder nor any head. Ground truth is
never read: the class decision uses only the instance arrays.

Fold routing: each frame belongs to exactly one protocol fold's
``outer_eval`` (through its capture group) and is evaluated only by that
fold's models — never a best fold, no ensemble, no full-data refit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.polar_local_features import (
    LOCAL_C_WIDTH,
    LOCAL_S_WIDTH,
    build_local_polar_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT = "material_residual_runtime_v1"
CANDIDATE_EXPERIMENT = "material_conditional_residual_v1"
VARIANTS = ("R0", "RC", "RCS")
CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")
P0_INPUT_DIM = 270           # G(256) + global C(6) + global S(8)
GLOBAL_C_WIDTH = 6
GLOBAL_S_WIDTH = 8

FALLBACK_REASON_NO_VALID_S = "no_valid_s_region"
FALLBACK_REASON_NON_FINITE = "non_finite_polar"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

_CONFIG_REQUIRED = (
    "experiment", "candidate", "protocol", "encoder", "features",
    "p0_models", "residual_models", "r0_records", "model_a",
    "calibration", "stereo", "polarization", "runtime", "data_identity",
    "tolerances", "benchmark",
)


@dataclasses.dataclass(frozen=True)
class FileRef:
    """A pinned runtime file: path plus the sha256 the config records."""

    path: str
    sha256: str


@dataclasses.dataclass(frozen=True)
class ResidualCandidate:
    """One frozen candidate of the fixed evaluation matrix.

    ``family`` selects the frozen forward: ``p0`` (the shared gray
    baseline), ``linear_local`` (C14+S16, input 30, no context) or
    ``conditional`` (C14+S16+P0-context, input 35). Every candidate
    loads its OWN per-fold residual state, scaler and saved outer-eval
    oracle; the raw G/C14/S16 features may be shared, the standardized
    inputs may not.
    """

    candidate_id: str
    family: str
    variant: str
    seed: int | None
    input_dim: int
    fold_files: dict[str, dict[str, FileRef]]


@dataclasses.dataclass(frozen=True)
class ResidualRuntimeConfig:
    """Validated view of the residual runtime experiment config."""

    experiment: str
    fit_dir: str
    summary_sha256: str
    admission_sha256: str
    input_dim: int
    context_dim: int
    hidden: int
    engineering_seed: int
    variants: tuple[str, ...]
    fallback_rule: str
    protocol_path: str
    protocol_sha256: str
    encoder_path: str
    encoder_sha256: str
    encoder_imgsz: int
    crop_padding: int
    region_rule: str
    global_cache: tuple[FileRef, ...]
    local_cache: tuple[FileRef, ...]
    p0_models: dict[str, dict[str, FileRef]]
    residual_models: dict[str, dict[str, dict[str, dict[str, FileRef]]]]
    r0_records: dict[str, FileRef]
    candidates: tuple[ResidualCandidate, ...]
    detection_reference: str
    data_identity: dict[str, Any]
    evaluation_budget: dict[str, Any]
    model_a: dict[str, Any]
    stereo: dict[str, Any]
    calibration: dict[str, Any]
    polarization: dict[str, Any]
    runtime: dict[str, Any]
    tolerances: dict[str, Any]
    benchmark: dict[str, Any]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_of_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, field: str) -> str:
    value = str(value)
    if _HEX64.match(value) is None:
        raise ValueError(
            f"{field} must be 64 lowercase hex chars, got {value!r}")
    return value


def _file_ref(entry: Any, field: str) -> FileRef:
    if not isinstance(entry, dict) or "path" not in entry or \
            "sha256" not in entry:
        raise ValueError(f"{field} must be a path/sha256 mapping")
    return FileRef(path=str(entry["path"]),
                   sha256=_require_sha(entry["sha256"], f"{field}.sha256"))


def load_residual_runtime_config(path: str | Path) -> ResidualRuntimeConfig:
    """Load and strictly validate the residual runtime config."""
    import yaml

    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"runtime config {path} must be a mapping")
    missing = [field for field in _CONFIG_REQUIRED if field not in raw]
    if missing:
        raise ValueError(
            f"runtime config is missing sections/fields: {missing}")
    if str(raw["experiment"]) not in (EXPERIMENT,
                                      EXPERIMENT + "_complete"):
        raise ValueError(
            f"experiment {raw['experiment']!r} is not {EXPERIMENT!r} "
            f"(or {EXPERIMENT + '_complete'!r})")
    candidate = raw["candidate"]
    if str(candidate["experiment"]) != CANDIDATE_EXPERIMENT:
        raise ValueError(
            f"candidate experiment {candidate['experiment']!r} is not "
            f"{CANDIDATE_EXPERIMENT!r}")
    variants = tuple(str(v) for v in candidate["variants"])
    if variants != VARIANTS:
        raise ValueError(
            f"candidate variants {variants} != {VARIANTS}")
    encoder = raw["encoder"]
    features = raw["features"]
    global_root = str(features["global_cache"]["path"]).rstrip("/\\")
    local_root = str(features["local_cache"]["path"]).rstrip("/\\")
    p0_models = {
        fold: {name: _file_ref(entry, f"p0_models.{fold}.{name}")
               for name, entry in per_fold.items()}
        for fold, per_fold in raw["p0_models"].items()
    }
    residual_models = {}
    for seed_key, per_seed in raw["residual_models"].items():
        residual_models[str(seed_key)] = {
            fold: {variant: {name: _file_ref(
                       entry,
                       f"residual_models.{seed_key}.{fold}.{variant}.{name}")
                     for name, entry in per_variant.items()}
                   for variant, per_variant in per_fold.items()}
            for fold, per_fold in per_seed.items()
        }
    model_a = dict(raw["model_a"])
    _require_sha(model_a.get("sha256", ""), "model_a.sha256")
    candidates: list[ResidualCandidate] = []
    for entry in raw.get("candidates", []):
        fold_files: dict[str, dict[str, FileRef]] = {}
        for fold, files in entry.get("files", {}).items():
            fold_files[str(fold)] = {
                name: _file_ref(ref, f"candidates.{entry['candidate_id']}."
                                      f"{fold}.{name}")
                for name, ref in files.items()}
        candidates.append(ResidualCandidate(
            candidate_id=str(entry["candidate_id"]),
            family=str(entry["family"]),
            variant=str(entry["variant"]),
            seed=(int(entry["seed"]) if entry.get("seed") is not None
                  else None),
            input_dim=int(entry["input_dim"]),
            fold_files=fold_files))
    return ResidualRuntimeConfig(
        experiment=str(raw["experiment"]),
        fit_dir=str(candidate["fit_dir"]),
        summary_sha256=_require_sha(candidate["summary_sha256"],
                                    "candidate.summary_sha256"),
        admission_sha256=_require_sha(candidate["admission_sha256"],
                                      "candidate.admission_sha256"),
        input_dim=int(candidate["input_dim"]),
        context_dim=int(candidate["context_dim"]),
        hidden=int(candidate["hidden"]),
        engineering_seed=int(candidate["engineering_seed"]),
        variants=variants,
        fallback_rule=str(candidate["fallback"]["rule"]),
        protocol_path=str(raw["protocol"]["path"]),
        protocol_sha256=_require_sha(raw["protocol"]["sha256"],
                                     "protocol.sha256"),
        encoder_path=str(encoder["path"]),
        encoder_sha256=_require_sha(encoder["sha256"], "encoder.sha256"),
        encoder_imgsz=int(encoder["imgsz"]),
        crop_padding=int(features["crop_padding"]),
        region_rule=str(features["region_rule"]),
        global_cache=tuple(
            FileRef(path=f"{global_root}/{name}", sha256=_require_sha(
                digest, f"features.global_cache.files.{name}"))
            for name, digest in features["global_cache"]["files"].items()),
        local_cache=tuple(
            FileRef(path=f"{local_root}/{name}", sha256=_require_sha(
                digest, f"features.local_cache.files.{name}"))
            for name, digest in features["local_cache"]["files"].items()),
        p0_models=p0_models,
        residual_models=residual_models,
        r0_records={fold: _file_ref(entry, f"r0_records.{fold}")
                    for fold, entry in raw["r0_records"].items()},
        candidates=tuple(candidates),
        detection_reference=str(raw.get("detection_reference", "")),
        data_identity=dict(raw["data_identity"]),
        evaluation_budget=dict(raw.get("evaluation_budget", {})),
        model_a=dict(raw["model_a"]),
        stereo=dict(raw["stereo"]),
        calibration=dict(raw["calibration"]),
        polarization=dict(raw["polarization"]),
        runtime=dict(raw["runtime"]),
        tolerances=dict(raw["tolerances"]),
        benchmark=dict(raw["benchmark"]),
    )


_encoder_cache: dict[tuple, Any] = {}


def _encoder_cache_key(config: ResidualRuntimeConfig, device: str) -> tuple:
    """The encoder identity: resolved weights path, sha256, normalized
    device and input size. Different devices or weight identities can
    never share an entry (CPU and GPU G features differ numerically)."""
    from models.polar_fusion import torch_device_name

    path = str(resolve_path(config.encoder_path)).lower()
    normalized = str(torch_device_name(str(device))).lower()
    return (path, str(config.encoder_sha256), normalized,
            int(config.encoder_imgsz))


def _module_device(obj: Any) -> str:
    """The declared device of an encoder-like object."""
    declared = getattr(obj, "_declared_device", None)
    if declared is not None:
        return str(declared).lower()
    try:
        for parameter in obj.parameters():
            return str(parameter.device).lower()
        for buffer in obj.buffers():
            return str(buffer.device).lower()
    except (AttributeError, TypeError):
        pass
    return "cpu"


def _build_encoder(config: ResidualRuntimeConfig, device: str, *,
                   builder: Any = None):
    """The frozen official MiT-B0 encoder, identity-verified and cached
    by (weights, sha256, device, imgsz).

    A cache hit must still match the requested device — a shared encoder
    that was moved elsewhere is refused, never migrated in place (an
    in-place ``.to()`` would silently change every other classifier).
    """
    from models.polar_fusion import torch_device_name

    key = _encoder_cache_key(config, device)
    normalized = str(torch_device_name(str(device))).lower()
    if key in _encoder_cache:
        cached = _encoder_cache[key]
        if _module_device(cached) != normalized:
            raise RuntimeError(
                f"encoder device mismatch: the cached encoder for this "
                f"identity lives on {_module_device(cached)!r}, not the "
                f"requested device {normalized!r}; refusing to migrate a "
                "shared encoder in place")
        return cached
    if builder is not None:
        encoder = builder(config, normalized)
    else:
        encoder = _construct_verified_encoder(config, normalized)
    _encoder_cache[key] = encoder
    return encoder


def _construct_verified_encoder(config: ResidualRuntimeConfig,
                                normalized_device: str):
    probe_config = {
        "baselines": {"official_mit_b0": {
            "path": config.encoder_path,
            "sha256": config.encoder_sha256,
        }},
        "features": {"imgsz": config.encoder_imgsz},
    }
    from scripts.prepare_material_probe import build_probe_encoder

    encoder, _report = build_probe_encoder(probe_config, PROJECT_ROOT)
    encoder.to(normalized_device)
    return encoder


# ---------------------------------------------------------------------------
# Per-fold frozen heads


class FoldHeads:
    """The frozen P0 + correction heads of one fold (CPU float32 only)."""

    def __init__(self, *, fold_id: str, p0_head, p0_scaler, local_scaler,
                 correction_heads: dict[str, Any], p0_sha256: str):
        self.fold_id = fold_id
        self.p0_head = p0_head
        self.p0_scaler = p0_scaler
        self.local_scaler = local_scaler
        self.correction_heads = correction_heads
        self.p0_sha256 = p0_sha256

    def torch_modules(self) -> dict[str, Any]:
        modules = {"p0_head": self.p0_head}
        for name, head in self.correction_heads.items():
            modules[f"correction_{name}"] = head
        return modules


def _load_p0_head(payload_path: Path, expected_fold: str,
                  expected_input_dim: int = P0_INPUT_DIM) -> Any:
    import torch

    from core.material_residual import p0_head_from_payload

    payload = torch.load(payload_path, weights_only=False,
                         map_location="cpu")
    if payload.get("variant") != "P0":
        raise ValueError(
            f"{payload_path}: variant {payload.get('variant')!r} is not P0")
    if payload.get("fold") != expected_fold:
        raise ValueError(
            f"{payload_path}: payload fold {payload.get('fold')!r} != "
            f"expected {expected_fold!r}")
    if int(payload.get("input_dim", -1)) != int(expected_input_dim):
        raise ValueError(
            f"{payload_path}: P0 input_dim {payload.get('input_dim')!r} != "
            f"{expected_input_dim}")
    return p0_head_from_payload(payload)


def _load_local_scaler(scaler_path: Path) -> Any:
    from core.material_probe import Standardizer

    payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    return Standardizer.from_dict(payload)


def _load_p0_scaler(scaler_path: Path) -> Any:
    return _load_local_scaler(scaler_path)


def _load_linear_head(payload_path: Path, *, expected_fold: str,
                      expected_variant: str, expected_seed: int,
                      expected_input_dim: int) -> Any:
    """The frozen LINEAR local residual head (bias-free Linear(input,4)).

    Reuses :class:`core.material_residual.ResidualHead` verbatim — the
    linear-local family never enters the conditional head builder.
    """
    import torch

    payload = torch.load(payload_path, weights_only=False,
                         map_location="cpu")
    if payload.get("variant") != expected_variant:
        raise ValueError(
            f"{payload_path}: variant {payload.get('variant')!r} != "
            f"expected {expected_variant!r}")
    if payload.get("fold") != expected_fold:
        raise ValueError(
            f"{payload_path}: payload fold {payload.get('fold')!r} != "
            f"expected {expected_fold!r}")
    if int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            f"{payload_path}: payload seed {payload.get('seed')!r} != "
            f"expected {expected_seed}")
    if int(payload.get("input_dim", -1)) != expected_input_dim:
        raise ValueError(
            f"{payload_path}: input_dim {payload.get('input_dim')!r} != "
            f"{expected_input_dim}")
    if payload.get("region_rule") != "local_partition_interior_band_v1":
        raise ValueError(
            f"{payload_path}: region rule {payload.get('region_rule')!r} "
            "is not the frozen local rule")
    from core.material_residual import build_residual_head

    head = build_residual_head(seed=int(payload["seed"]),
                               input_dim=int(payload["input_dim"]))
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def _load_correction_head(payload_path: Path, *, expected_fold: str,
                          expected_variant: str, expected_seed: int,
                          expected_input_dim: int,
                          expected_hidden: int) -> Any:
    import torch

    payload = torch.load(payload_path, weights_only=False,
                         map_location="cpu")
    if payload.get("variant") != expected_variant:
        raise ValueError(
            f"{payload_path}: variant {payload.get('variant')!r} != "
            f"expected {expected_variant!r}")
    if payload.get("fold") != expected_fold:
        raise ValueError(
            f"{payload_path}: payload fold {payload.get('fold')!r} != "
            f"expected {expected_fold!r}")
    if int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError(
            f"{payload_path}: payload seed {payload.get('seed')!r} != "
            f"expected {expected_seed}")
    if int(payload.get("input_dim", -1)) != expected_input_dim:
        raise ValueError(
            f"{payload_path}: input_dim {payload.get('input_dim')!r} != "
            f"{expected_input_dim}")
    if int(payload.get("hidden", -1)) != expected_hidden:
        raise ValueError(
            f"{payload_path}: hidden {payload.get('hidden')!r} != "
            f"{expected_hidden}")
    from core.material_conditional_residual import build_conditional_head

    head = build_conditional_head(
        seed=int(payload["seed"]), input_dim=int(payload["input_dim"]),
        hidden=int(payload["hidden"]))
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


# ---------------------------------------------------------------------------
# The adapter


def _p0_logits(heads: FoldHeads, g: np.ndarray) -> np.ndarray:
    """The frozen P0 logits for one (n, G_dim) batch of G features.

    The (G_dim + C + S)-wide design is built exactly like the offline
    path: the frozen scaler transforms [G, 0, 0] and
    ``assemble_design_matrix(..., "P0")`` zeroes the C/S slots, so the
    head input is the standardized G plus exact zeros. G_dim comes from
    the head itself (the real world pins 256; the synthetic tests use
    their own width) — never a hardcoded constant.
    """
    import torch

    from core.material_probe import assemble_design_matrix

    g_dim = heads.p0_head.in_features - GLOBAL_C_WIDTH - GLOBAL_S_WIDTH
    if g_dim <= 0:
        raise ValueError(
            f"the frozen P0 head implies a non-positive G width "
            f"{g_dim}")
    raw = np.concatenate(
        [np.asarray(g, dtype=np.float32),
         np.zeros((len(g), GLOBAL_C_WIDTH), dtype=np.float32),
         np.zeros((len(g), GLOBAL_S_WIDTH), dtype=np.float32)],
        axis=1).astype(np.float64)
    column_valid = np.ones((len(g), g_dim + GLOBAL_C_WIDTH
                            + GLOBAL_S_WIDTH), dtype=bool)
    standardized = heads.p0_scaler.transform(raw, column_valid)
    design = assemble_design_matrix(
        standardized[:, :g_dim],
        standardized[:, g_dim:g_dim + GLOBAL_C_WIDTH],
        standardized[:, g_dim + GLOBAL_C_WIDTH:], "P0")
    with torch.inference_mode():
        logits = heads.p0_head(torch.from_numpy(
            np.ascontiguousarray(design, dtype=np.float32)))
    return logits.cpu().numpy().astype(np.float32)


def _structure_problem(sample) -> str:
    """The instance-structure problem, or "" when the row is well-formed."""
    from models.material_classifier import _structure_problem

    return _structure_problem(sample)


class MaterialResidualClassifier:
    """Batch inference over the locked residual candidate.

    One instance is bound to one ``variant`` (R0 = pure P0, RC = C +
    context correction, RCS = C + S + context correction) at the
    engineering seed; three classifier instances share the loaded heads
    through the module-level caches. Every pinned file digest is verified
    at construction, and every payload's own identity fields must agree
    with the config before anything runs.
    """

    def __init__(self, config, *, variant: str, device: str = "cpu",
                 frame_folds: dict[str, str], encoder=None,
                 seed: int | None = None, candidate: str | None = None,
                 encoder_builder: Any = None):
        self.config = (load_residual_runtime_config(config)
                       if not isinstance(config, ResidualRuntimeConfig)
                       else config)
        if variant not in VARIANTS:
            raise ValueError(
                f"variant {variant!r} is not one of {VARIANTS}")
        if seed is not None and int(seed) != self.config.engineering_seed:
            raise ValueError(
                f"seed {seed} is not the pinned engineering seed "
                f"{self.config.engineering_seed}; the runtime never picks "
                "a seed by results")
        self.engineering_seed = self.config.engineering_seed
        if candidate is not None:
            matches = [spec for spec in self.config.candidates
                       if spec.candidate_id == candidate]
            if not matches:
                raise ValueError(
                    f"candidate {candidate!r} is not in the frozen registry "
                    f"{sorted(s.candidate_id for s in self.config.candidates)}")
            if len(matches) > 1:
                raise ValueError(
                    f"candidate id {candidate!r} is ambiguous in the registry")
            spec = matches[0]
            if spec.variant != variant:
                raise ValueError(
                    f"candidate {candidate!r} is variant {spec.variant!r}, "
                    f"not the requested {variant!r}")
            self.candidate_spec = spec
        else:
            self.candidate_spec = self._legacy_candidate_spec(variant)
        self.variant = variant
        self.imgsz = self.config.encoder_imgsz
        self.frame_folds = dict(frame_folds)
        known_folds = set(self.config.p0_models)
        for frame, fold in self.frame_folds.items():
            if fold not in known_folds:
                raise ValueError(
                    f"frame {frame!r} maps to fold {fold!r}, which the "
                    f"config does not pin (known: {sorted(known_folds)})")

        self.file_identities = self._verify_pinned_files()

        self.encoder = (encoder if encoder is not None
                        else _build_encoder(self.config, device,
                                            builder=encoder_builder))
        self.encoder_device = _module_device(self.encoder)
        # the P0 input width follows the runtime encoder's G width
        # (the real world pins 256 -> 270; synthetic worlds derive their
        # own width from the injected encoder)
        encoder_g_dim = getattr(self.encoder, "g_dim", None)
        self._p0_input_dim = (int(encoder_g_dim) + GLOBAL_C_WIDTH
                              + GLOBAL_S_WIDTH
                              if encoder_g_dim is not None
                              else P0_INPUT_DIM)

        self._fold_heads = {
            fold: self._load_fold_heads(fold) for fold in known_folds}

    def _legacy_candidate_spec(self, variant: str) -> ResidualCandidate:
        """The single-candidate view of the ORIGINAL config (no registry):
        the conditional family at the engineering seed."""
        seed_key = f"seed_{self.engineering_seed}"
        family = "p0" if variant == "R0" else "conditional"
        fold_files = {}
        if family != "p0":
            for fold, per_fold in self.config.residual_models.get(
                    seed_key, {}).items():
                entries = per_fold.get(variant, {})
                if entries:
                    fold_files[fold] = {
                        "residual_state": entries["residual_state"],
                        "scaler": entries["scaler"],
                        "saved_predictions":
                            entries["outer_eval_predictions"],
                        "fold_variant_report": entries.get(
                            "fold_variant_report",
                            entries["outer_eval_predictions"]),
                    }
        return ResidualCandidate(
            candidate_id=f"conditional_{variant}_s{self.engineering_seed}",
            family=family, variant=variant, seed=self.engineering_seed,
            input_dim=self.config.input_dim, fold_files=fold_files)

    # ------------------------------------------------------------------
    # Construction helpers

    def _verify_pinned_files(self) -> dict[str, str]:
        identities: dict[str, str] = {}
        checks = [
            ("fit_summary", resolve_path(self.config.fit_dir) / "summary.json",
             self.config.summary_sha256),
            ("fit_admission",
             resolve_path(self.config.fit_dir) / "admission.json",
             self.config.admission_sha256),
            ("protocol", resolve_path(self.config.protocol_path),
             self.config.protocol_sha256),
        ]
        for ref in self.config.global_cache:
            checks.append((f"global_cache:{Path(ref.path).name}",
                           resolve_path(ref.path), ref.sha256))
        for ref in self.config.local_cache:
            checks.append((f"local_cache:{Path(ref.path).name}",
                           resolve_path(ref.path), ref.sha256))
        for fold, entries in self.config.p0_models.items():
            for name, ref in entries.items():
                checks.append((f"p0:{fold}:{name}",
                               resolve_path(ref.path), ref.sha256))
        for seed_key, per_seed in self.config.residual_models.items():
            for fold, per_fold in per_seed.items():
                for variant, entries in per_fold.items():
                    for name, ref in entries.items():
                        checks.append((f"residual:{seed_key}:{fold}:"
                                       f"{variant}:{name}",
                                       resolve_path(ref.path), ref.sha256))
        for fold, ref in self.config.r0_records.items():
            checks.append((f"r0:{fold}", resolve_path(ref.path), ref.sha256))
        for fold, entries in self.candidate_spec.fold_files.items():
            for name, ref in entries.items():
                checks.append((f"candidate:{self.candidate_spec.candidate_id}:"
                               f"{fold}:{name}",
                               resolve_path(ref.path), ref.sha256))
        for name, path, expected in checks:
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"pinned runtime file missing: {path}")
            digest = sha256_of_file(path)
            if digest != expected:
                raise ValueError(
                    f"{name}: sha256 {digest} != the pinned {expected}")
            identities[name] = digest
        return identities

    def _load_fold_heads(self, fold: str) -> FoldHeads:
        entries = self.config.p0_models[fold]
        p0_head = _load_p0_head(resolve_path(entries["checkpoint"].path),
                                expected_fold=fold,
                                expected_input_dim=self._p0_input_dim)
        p0_scaler = _load_local_scaler(resolve_path(entries["scaler"].path))
        from core.material_probe import Standardizer as _Standardizer

        spec = self.candidate_spec
        if spec.family == "p0":
            # the P0 family never standardizes local features; a local
            # scaler is only loaded when the config happens to pin one
            fallback = (self.config.residual_models.get(
                f"seed_{self.engineering_seed}", {}).get(fold, {})
                .get("RCS", {}).get("scaler"))
            local_scaler_path = (resolve_path(fallback.path)
                                 if fallback is not None else None)
            correction_heads: dict[str, Any] = {}
        else:
            local_scaler_path = resolve_path(
                spec.fold_files[fold]["scaler"].path)
            correction_heads = {}
            if spec.variant in ("RC", "RCS"):
                state_path = resolve_path(
                    spec.fold_files[fold]["residual_state"].path)
                if spec.family == "conditional":
                    correction_heads[spec.variant] = _load_correction_head(
                        state_path, expected_fold=fold,
                        expected_variant=spec.variant,
                        expected_seed=spec.seed,
                        expected_input_dim=spec.input_dim,
                        expected_hidden=self.config.hidden)
                elif spec.family == "linear_local":
                    correction_heads[spec.variant] = _load_linear_head(
                        state_path, expected_fold=fold,
                        expected_variant=spec.variant,
                        expected_seed=spec.seed,
                        expected_input_dim=spec.input_dim)
                else:
                    raise ValueError(
                        f"unknown candidate family {spec.family!r}")
        if local_scaler_path is None:
            local_scaler = None
        else:
            local_payload = json.loads(local_scaler_path.read_text(
                encoding="utf-8"))
            local_scaler = _Standardizer.from_dict(local_payload)
            if len(local_scaler.mean) != LOCAL_C_WIDTH + LOCAL_S_WIDTH:
                raise ValueError(
                    f"{local_scaler_path}: local scaler width "
                    f"{len(local_scaler.mean)} != "
                    f"{LOCAL_C_WIDTH + LOCAL_S_WIDTH}")
        return FoldHeads(fold_id=fold, p0_head=p0_head,
                         p0_scaler=p0_scaler, local_scaler=local_scaler,
                         correction_heads=correction_heads,
                         p0_sha256=self.file_identities[f"p0:{fold}:checkpoint"])

    def _state_key(self, fold: str) -> str:
        """The file-identity key of this candidate's residual state."""
        return (f"candidate:{self.candidate_spec.candidate_id}:{fold}:"
                f"residual_state")

    # ------------------------------------------------------------------
    # Public interface

    def torch_modules(self) -> dict[str, Any]:
        modules = {}
        for fold, heads in self._fold_heads.items():
            for name, module in heads.torch_modules().items():
                modules[f"{fold}:{name}"] = module
        if hasattr(self.encoder, "parameters"):
            modules["encoder"] = self.encoder
        return modules

    def predict_batch(
        self, samples: Sequence[Any | None],
        gray: Sequence[np.ndarray | None] | None = None,
    ) -> list[Any]:
        """Classify a batch of instance inputs, preserving order and count.

        Structurally invalid rows yield explicit failures and reach neither
        the encoder nor any head. ``gray`` may carry one precomputed
        (256,) feature per input (same order; entries for invalid rows are
        ignored) so a shared evaluation does not re-run the encoder per
        candidate. The output length is checked against the input length —
        a shorter result is an error, never a silent ``zip`` truncation.
        """
        from models.material_classifier import MaterialClassResult

        outcomes = self._predict_rows(list(samples), gray=gray)
        if len(outcomes) != len(samples):
            raise ValueError(
                f"classification returned {len(outcomes)} results for "
                f"{len(samples)} inputs; refusing a silent zip truncation")
        return outcomes

    # ------------------------------------------------------------------
    # Stage functions (also used by the Layer A offline comparison)

    def gray_features(self, samples: Sequence[Any]) -> np.ndarray:
        """Frozen MiT-B0 last-stage mask-pooled gray features, (n, 256)."""
        import cv2
        import torch

        from models.material_fusion import mask_level_weights
        from models.polar_fusion import _fusion_tensors

        if not samples:
            return np.zeros((0, 256), dtype=np.float32)
        gray_batch, mask_batch = [], []
        for sample in samples:
            gray_t, _polar_t, _quality_t = _fusion_tensors(
                sample, self.imgsz, cv2)
            mask_t = torch.from_numpy(np.ascontiguousarray(cv2.resize(
                sample.instance_mask.astype(np.float32),
                (self.imgsz, self.imgsz),
                interpolation=cv2.INTER_AREA))).float().clip_(0.0, 1.0)
            gray_batch.append(gray_t)
            mask_batch.append(mask_t)
        rgb = torch.stack(gray_batch).to(self.encoder_device)
        masks = torch.stack(mask_batch).to(self.encoder_device)
        with torch.inference_mode():
            stage_features = self.encoder(rgb)
            last = stage_features[-1]
            weights = mask_level_weights(masks, last.shape[-2:]).to(
                dtype=last.dtype)
            total = weights.sum(dim=(2, 3))
            if bool((total <= 0).any()):
                raise ValueError(
                    "instance-mask pooling has zero weight for some batch "
                    "row; an empty mask is refused rather than GAP-pooled")
            pooled = (last * weights).sum(dim=(2, 3)) / total
        return pooled.cpu().numpy().astype(np.float32)

    def local_features(self, sample) -> dict:
        """The C14/S16 stage of one instance, with the fallback decision.

        Returns ``{"c_features", "s_features", "s_valid", "reason"}``.
        A missing or non-finite polarization measurement is an explicit
        S miss (all-NaN S block + a fallback reason); the values are never
        fabricated, zeroed into validity or silently restored.
        """
        valid = np.asarray(sample.valid, dtype=bool)
        mask = np.asarray(sample.instance_mask) > 0
        region = valid & mask
        c_features = build_local_control_features_record(sample)
        finite = bool(np.isfinite(sample.signed_q).all()
                      and np.isfinite(sample.abs_q).all())
        if not region.any():
            s_features = np.full(LOCAL_S_WIDTH, np.nan, dtype=np.float32)
            return {"c_features": c_features, "s_features": s_features,
                    "s_valid": False, "reason": FALLBACK_REASON_NO_VALID_S}
        if not finite or not (np.isfinite(sample.signed_q[region]).all()
                              and np.isfinite(sample.abs_q[region]).all()):
            s_features = np.full(LOCAL_S_WIDTH, np.nan, dtype=np.float32)
            return {"c_features": c_features, "s_features": s_features,
                    "s_valid": False, "reason": FALLBACK_REASON_NON_FINITE}
        s_values, _blocks = build_local_polar_stats(sample)
        return {"c_features": c_features, "s_features": s_values,
                "s_valid": True, "reason": ""}

    def p0_logits_for(self, sample, fold: str, gray: np.ndarray | None = None) \
            -> np.ndarray:
        """The frozen P0 logits of ``fold`` for one instance (CPU float32)."""
        heads = self._fold_heads[fold]
        g = (gray if gray is not None
             else self.gray_features([sample]))
        return _p0_logits(heads, np.asarray(g, dtype=np.float32)[None, :])[0]

    def candidate_final(self, *, fold: str, c14: np.ndarray,
                        s16: np.ndarray, s_valid: bool,
                        p0_logits: np.ndarray) -> dict:
        """The candidate-specific stage: standardize with THIS candidate's
        frozen scaler, assemble per family, apply its own head.

        ``c14``/``s16`` are the RAW per-instance features (shared across
        candidates); the standardized inputs are never shared.
        """
        import torch

        from core.material_conditional_residual import (
            assemble_conditional_inputs,
            conditional_logits,
            p0_context,
        )
        from core.material_residual import compute_delta
        from core.polar_local_features import (
            assemble_local_residual_inputs,
        )

        heads = self._fold_heads[fold]
        spec = self.candidate_spec
        if spec.family == "p0":
            return {"final_logits": np.asarray(p0_logits, dtype=np.float32),
                    "used_fallback": False, "reason": ""}
        matrix30 = np.concatenate(
            [np.asarray(c14, dtype=np.float32)[None, :],
             np.asarray(s16, dtype=np.float32)[None, :]],
            axis=1).astype(np.float64)
        column_valid = np.ones((1, matrix30.shape[1]), dtype=bool)
        column_valid[:, LOCAL_C_WIDTH:] = ~np.isnan(matrix30[0, LOCAL_C_WIDTH:])
        standardized30 = heads.local_scaler.transform(matrix30, column_valid)
        valid_vector = np.asarray([bool(s_valid)], dtype=bool)
        if spec.family == "linear_local":
            x = assemble_local_residual_inputs(
                standardized30[:, :LOCAL_C_WIDTH],
                standardized30[:, LOCAL_C_WIDTH:], valid_vector, spec.variant)
            head = heads.correction_heads[spec.variant]
            with torch.inference_mode():
                delta = compute_delta(
                    head, torch.from_numpy(x),
                    torch.from_numpy(valid_vector.astype(np.float32)))
            final = np.asarray(p0_logits, dtype=np.float32) + \
                delta.cpu().numpy().astype(np.float32)[0]
        elif spec.family == "conditional":
            context = p0_context(np.asarray(p0_logits,
                                            dtype=np.float32)[None, :])
            x = assemble_conditional_inputs(
                standardized30[:, :LOCAL_C_WIDTH],
                standardized30[:, LOCAL_C_WIDTH:], context, valid_vector,
                spec.variant)
            head = heads.correction_heads[spec.variant]
            with torch.inference_mode():
                final = conditional_logits(
                    head, torch.from_numpy(x),
                    torch.from_numpy(valid_vector.astype(np.float32)),
                    torch.from_numpy(np.asarray(p0_logits,
                                                dtype=np.float32)[None, :])
                ).cpu().numpy().astype(np.float32)[0]
        else:
            raise ValueError(f"unknown candidate family {spec.family!r}")
        return {"final_logits": final, "used_fallback": not bool(s_valid),
                "reason": ""}

    def predict_logits(self, sample, gray: np.ndarray | None = None,
                       fold: str | None = None) -> dict:
        """The full per-instance prediction record (all stages).

        Used by :meth:`_predict_rows` and by the Layer A offline
        comparison; ``gray`` may carry a precomputed (256,) feature from
        :meth:`gray_features` so batched callers do not re-run the
        encoder per instance. ``fold`` is the Layer A override: offline
        rows carry their fold from the authoritative protocol, while
        live records are routed through the frame->fold map. The class
        decision reads no ground truth.
        """
        if fold is None:
            fold = self.frame_folds.get(sample.frame_id)
        if fold is None or fold not in self._fold_heads:
            return {"status": "unknown_frame_fold"}
        heads = self._fold_heads[fold]
        local = self.local_features(sample)
        g = np.asarray(gray if gray is not None
                       else self.gray_features([sample]), dtype=np.float32)
        if g.ndim == 1:
            g = g[None, :]
        p0_logits = _p0_logits(heads, g)[0]
        final_stage = self.candidate_final(
            fold=fold, c14=local["c_features"], s16=local["s_features"],
            s_valid=local["s_valid"], p0_logits=p0_logits)
        used_fallback = final_stage["used_fallback"]
        return {
            "status": "ok", "fold": fold,
            "candidate_id": self.candidate_spec.candidate_id,
            "p0_logits": p0_logits,
            "final_logits": final_stage["final_logits"],
            "s_valid": local["s_valid"],
            "used_fallback": used_fallback,
            "reason": local["reason"] if used_fallback else "",
            "c_features": local["c_features"],
            "s_features": local["s_features"],
        }

    # ------------------------------------------------------------------
    # Internals

    def _predict_rows(self, samples: list,
                      gray: Sequence[np.ndarray | None] | None = None) -> list:
        from models.material_classifier import MaterialClassResult

        rows = [MaterialClassResult.failure(sample, "none")
                for sample in samples]
        for index, sample in enumerate(samples):
            if sample is None:
                continue
            reason = _structure_problem(sample)
            if reason:
                rows[index] = MaterialClassResult.failure(sample, reason)
                continue
            sample_gray = gray[index] if gray is not None else None
            record = self.predict_logits(sample, gray=sample_gray)
            if record["status"] != "ok":
                rows[index] = MaterialClassResult.failure(
                    sample, record["status"])
                continue
            final = record["final_logits"]
            class_id = int(final.argmax())
            shift = float(final.max())
            exp = np.exp(final.astype(np.float64) - shift)
            probs = exp / exp.sum()
            heads = self._fold_heads[record["fold"]]
            if self.candidate_spec.family == "p0":
                identity = (f"material_residual_runtime_v1 "
                            f"candidate={self.candidate_spec.candidate_id} "
                            f"fold={record['fold']} "
                            f"seed={self.engineering_seed} "
                            f"p0_sha256={heads.p0_sha256[:12]}")
            elif record["used_fallback"]:
                identity = (f"material_residual_runtime_v1 "
                            f"candidate={self.candidate_spec.candidate_id} "
                            f"fold={record['fold']} "
                            f"seed={self.engineering_seed} "
                            f"fallback=own_P0 "
                            f"p0_sha256={heads.p0_sha256[:12]}")
            else:
                state_digest = self.file_identities[self._state_key(
                    record["fold"])]
                identity = (f"material_residual_runtime_v1 "
                            f"candidate={self.candidate_spec.candidate_id} "
                            f"fold={record['fold']} "
                            f"seed={self.engineering_seed} "
                            f"state_sha256={state_digest[:12]}")
            rows[index] = MaterialClassResult(
                class_id=class_id, class_name=CLASS_NAMES[class_id],
                confidence=float(probs[class_id]), valid=True,
                used_fallback=record["used_fallback"],
                fallback_reason=record["reason"],
                model_identity=identity,
                instance_id=sample.instance_id, frame_id=sample.frame_id,
                invalid_reason=sample.invalid_reason,
                raw_logits=tuple(float(v) for v in final.tolist()),
                polar_invalid=bool(sample.polar_invalid))
        return rows


# ---------------------------------------------------------------------------
# Local feature computation on runtime records
#
# The offline functions read the attributes they need from any object that
# carries the V5 npz contract (gray/signed_q/abs_q/valid/instance_mask/
# quality). MaterialInstanceInput already does — ``valid`` is a property
# there — so the SAME code runs on both paths.


def build_local_control_features_record(sample) -> np.ndarray:
    from core.polar_local_features import build_local_control_features

    return build_local_control_features(sample)


def build_frame_fold_map(split_rows: Sequence[dict],
                         protocol: dict) -> dict[str, str]:
    """frame output_stem -> the fold whose outer_eval holds its group.

    Every frame's capture group must belong to exactly one fold's
    ``outer_eval_groups``; an ambiguous or uncovered group is refused
    instead of guessed.
    """
    group_to_fold: dict[str, str] = {}
    for fold in protocol["folds"]["folds"]:
        for group in fold["outer_eval_groups"]:
            if group in group_to_fold:
                raise ValueError(
                    f"capture group {group!r} is outer_eval in both "
                    f"{group_to_fold[group]} and {fold['fold_id']}")
            group_to_fold[group] = fold["fold_id"]
    mapping = {}
    for row in split_rows:
        group = row["group_name"]
        if group not in group_to_fold:
            raise ValueError(
                f"frame {row['output_stem']!r}: group {group!r} is not any "
                "fold's outer_eval group")
        mapping[row["output_stem"]] = group_to_fold[group]
    return mapping


__all__ = [
    "CANDIDATE_EXPERIMENT",
    "EXPERIMENT",
    "FALLBACK_REASON_NON_FINITE",
    "FALLBACK_REASON_NO_VALID_S",
    "FileRef",
    "FoldHeads",
    "MaterialResidualClassifier",
    "ResidualRuntimeConfig",
    "VARIANTS",
    "build_frame_fold_map",
    "load_residual_runtime_config",
    "sha256_of_file",
]
