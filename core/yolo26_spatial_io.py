"""Hash-bound inputs and fold-only standardization for YOLO26 transfer."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import cv2
import numpy as np

from core.material_probe import Standardizer
from core.material_spatial_classifier import require_partition
from core.material_spatial_classifier_io import (
    FROZEN_CONFIG_SHA256 as SOURCE_CONFIG_SHA256,
    InputRegistry, digest, load_inputs, read_json,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = '57bb9817f9f6b12d1cf31031d7d75ff20870ba39efd4ed8d8e442db8d4d5a656'
CLASS_NAMES = ['metal_submarine', 'plastic_submarine', 'plastic_fish', 'real_fish']


def validate_contract(cfg: dict) -> None:
    """Reject changes to preregistered data, model and fitting parameters."""
    expected = {
        'experiment': 'yolo26_spatial_v1',
        'source_config': 'configs/experiments/material_spatial_classifier_v1.yaml',
        'source_config_sha256': SOURCE_CONFIG_SHA256,
        'official_base': 'yolo26n-cls.pt',
        'official_base_sha256': '0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81',
        'class_names': CLASS_NAMES,
        'fold_ids': ['fold0', 'fold1', 'fold2'],
        'seeds': [2026, 2027, 2028],
        'variants': ['B_stats', 'B_L', 'B_R', 'B_Q'],
        'planned_gray_fits': 9, 'planned_spatial_fits': 36,
        'gray_training': {'epochs': 30, 'batch': 32, 'lr': .001,
                          'weight_decay': .01, 'augmentation': False,
                          'device': 'cuda:0', 'precision': 'float32'},
        'spatial_training': {'epochs': 60, 'patience': 10, 'batch': 32,
                             'lr': .001, 'weight_decay': .001,
                             'augmentation': False, 'device': 'cuda:0',
                             'precision': 'float32'},
        'model': {'gray_feature_dim': 1280, 'context_dim': 1310,
                  'spatial_dim': 64, 'hidden': 64, 'classes': 4},
        'budgets': {'total_minutes': 480, 'gray_fit_minutes': 20,
                    'spatial_fit_minutes': 10},
        'selection': ['inner_val_macro_f1', 'inner_val_accuracy', 'earlier_epoch'],
    }
    import json

    for key, value in expected.items():
        if json.dumps(cfg.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f'frozen YOLO26 contract mismatch: {key}')
    if set(cfg) != set(expected) | {'science'}:
        raise ValueError('unknown YOLO26 configuration fields')


def decode_native_gray(path: Path, rows: list[dict]) -> np.ndarray:
    """Read only L responses, while checking the complete shard identity set."""
    fields = ('L', 'R', 'q', 'M', 'V', 'region', 'quality')
    names = [r['sample_name'] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError('duplicate native identity')
    expected = {name + '|' + field for name in names for field in fields}
    images = []
    with np.load(path, allow_pickle=False) as shard:
        if set(shard.files) != expected:
            raise ValueError('native shard identity keys differ')
        for row in rows:
            x1, y1, x2, y2 = row['crop_window']
            image = shard[row['sample_name'] + '|L']
            if image.dtype != np.uint8 or image.shape != (y2-y1, x2-x1):
                raise ValueError('native L dtype or crop geometry differs')
            if not image.size:
                raise ValueError('empty native L')
            images.append(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA))
    return np.stack(images)


def load_experiment_inputs(config_path: Path, registry: InputRegistry,
                           *, load_grid: bool = True) -> dict:
    """Reuse admitted spatial sources, adding hash-protected native grayscale."""
    config_path = Path(config_path).resolve()
    rel = config_path.relative_to(ROOT).as_posix()
    registry.check(rel, CONFIG_SHA256)
    cfg = read_json(config_path)
    validate_contract(cfg)
    source = registry.check(cfg['source_config'], cfg['source_config_sha256'])
    source_cfg = read_json(source)
    registry.check(cfg['official_base'], cfg['official_base_sha256'])
    data = load_inputs(source_cfg, registry, load_grid=load_grid)
    base = data['base']
    for name in ('run_status.json',):
        path = base/name
        registry.check(path.relative_to(ROOT).as_posix(), digest(path))
    gray = np.empty((len(data['rows']), 224, 224), np.uint8)
    frames: dict[str, list[int]] = {}
    for i, row in enumerate(data['rows']):
        frames.setdefault(row['frame'], []).append(i)
    for frame, ids in frames.items():
        shard_rel = f'cache/native/frame_{frame}.npz'
        record = data['binding'][shard_rel]
        path = registry.check((base/shard_rel).relative_to(ROOT).as_posix(),
                              record['sha256'])
        gray[ids] = decode_native_gray(path, [data['rows'][i] for i in ids])
    data.update(gray=gray, cfg=cfg, config_sha256=CONFIG_SHA256)
    # Admission checks bind the old feature file; its G256 is deliberately
    # removed so no consumer can accidentally use it as YOLO26 appearance.
    data['features'].pop('g_features')
    return data


def fold_context(g: np.ndarray, features: dict, rows: list[dict], fold: str) -> dict:
    """Only this fold's inner_train contributes to normalization."""
    parts = require_partition(rows, fold)
    if g.shape != (len(rows), 1280) or not np.isfinite(g).all():
        raise ValueError('wrong YOLO26 feature contract')
    c, s = features['c_features'], features['s_features']
    if c.shape != (len(rows), 14) or s.shape != (len(rows), 16):
        raise ValueError('wrong common statistics contract')
    if not np.isfinite(c).all() or np.isinf(s).any():
        raise ValueError('nonfinite common statistics')
    raw = np.concatenate((g, c, s), axis=1)
    flags = np.isfinite(raw)
    ids = parts['inner_train']
    scaler = Standardizer.fit(raw[ids], flags[ids])
    return {'parts': parts, 'context': scaler.transform(raw, flags),
            'scaler': scaler.to_dict()}


def fresh_output(parent: Path, name: str) -> Path:
    """Return a fresh strict descendant, refusing links and path traversal."""
    parent = Path(parent).resolve()
    raw = Path(name)
    if '..' in raw.parts:
        raise ValueError('output traversal refused')
    path = Path(os.path.abspath(parent/raw))
    if path == parent or not path.resolve().is_relative_to(parent) or path.exists():
        raise ValueError('fresh experiment descendant required')
    node = path
    while node != node.parent:
        if node.exists():
            info = node.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('output link/reparse point refused')
        node = node.parent
    return path


def prediction_rows(data: dict, ids: list[int], logits: np.ndarray,
                    y0: np.ndarray, *, baseline: bool = False) -> list[dict]:
    """Preserve sample identity when scattering fold predictions."""
    if logits.shape != (len(ids), 4) or y0.shape != (len(data['rows']), 4):
        raise ValueError('prediction identity shape differs')
    if not np.isfinite(logits).all() or not np.isfinite(y0).all():
        raise ValueError('nonfinite prediction')
    if len(set(ids)) != len(ids) or any(not 0 <= i < len(data['rows']) for i in ids):
        raise ValueError('prediction identities invalid')
    result = []
    for j, i in enumerate(ids):
        row = data['rows'][i]
        record = {key: row[key] for key in ('sample_name', 'class_id', 'group')}
        record.update(logits=logits[j].tolist(), prediction=int(logits[j].argmax()),
                      y0_prediction=int(y0[i].argmax()),
                      used_fallback=False if baseline else not bool(data['features']['s_valid'][i]))
        result.append(record)
    return result
