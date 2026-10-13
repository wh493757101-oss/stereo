"""Fourth-batch source admission and fold-specific data assembly."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np
import torch

from core.material_probe import Standardizer
from core.material_spatial_classifier import require_partition
from core.material_spatial_input import GRID_PLANES, apply_q_permutation

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG_SHA256 = '8f77f35e9506db17aa694f2ce7bf80e8aa9729287c41c3eff19b55ce1b643851'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class InputRegistry:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root.resolve()
        self.records = {}

    def check(self, rel: str, expected: str) -> Path:
        raw = Path(rel)
        if '..' in raw.parts:
            raise ValueError('path traversal refused')
        path = Path(os.path.abspath(self.root / raw))
        node = path
        while True:
            info = node.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError(f'link/reparse point refused: {node}')
            if node.parent == node:
                break
            node = node.parent
        if not path.resolve().is_relative_to(self.root) or not path.is_file():
            raise ValueError('input outside project or not a regular file')
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'input identity mismatch: {rel}')
        key = path.relative_to(self.root).as_posix()
        if key in self.records and self.records[key]['sha256'] != expected:
            raise ValueError('conflicting source identity')
        self.records[key] = {'path': key, 'sha256': expected, 'bytes': path.stat().st_size}
        return path

    def finish(self) -> dict:
        for key, record in list(self.records.items()):
            self.check(key, record['sha256'])
        return {'status': 'IDENTICAL', 'unique_files': len(self.records),
                'files': list(self.records.values())}


def load_inputs(cfg: dict, registry: InputRegistry, *, load_grid: bool = True) -> dict:
    if cfg['experiment'] != 'material_spatial_classifier_v1' or cfg['planned_fits'] != 36:
        raise ValueError('wrong experiment contract')
    for anchor in cfg['anchors']:
        registry.check(anchor['path'], anchor['sha256'])
    base = ROOT / cfg['spatial_root']
    binding = read_json(base / 'cache/binding.json')
    if binding['artifact_count'] != len(binding['artifacts']):
        raise ValueError('invalid artifact count')
    records = {a['path']: a for a in binding['artifacts']}
    if len(records) != len(binding['artifacts']):
        raise ValueError('duplicate artifact path')
    for name in ('definition.json', 'samples.jsonl', 'frames.jsonl'):
        key = 'cache/' + name
        registry.check((base / key).relative_to(ROOT).as_posix(), records[key]['sha256'])
    status = read_json(base / 'run_status.json')
    if status['input_admitted'] is not True or status['overall_status'] != 'COMPLETED':
        raise ValueError('unadmitted cache')
    definition = read_json(base/'cache/definition.json')
    if definition['kind'] != 'material_spatial_input_full_v1' or definition['complete'] is not True or tuple(definition['planes']) != GRID_PLANES:
        raise ValueError('wrong full cache definition')
    rows = [json.loads(line) for line in (base/'cache/samples.jsonl').read_text().splitlines()]
    frames = [json.loads(line) for line in (base/'cache/frames.jsonl').read_text().splitlines()]
    if len(rows) != 2446 or len(frames) != 945:
        raise ValueError('wrong spatial census')
    if len({r['sample_name'] for r in rows}) != len(rows):
        raise ValueError('duplicate sample identity')
    features = dict(np.load(ROOT/cfg['features'], allow_pickle=False))
    manifest = read_json(ROOT/cfg['feature_manifest'])['samples']
    if len(manifest) != len(rows):
        raise ValueError('feature manifest length mismatch')
    for i, row in enumerate(rows):
        reference = manifest[i]
        if type(row['cache_row']) is not int or row['cache_row'] != i:
            raise ValueError('cache row identity mismatch')
        for k in ('sample_name', 'class_id', 'group', 'fold_roles', 'has_valid_s'):
            if row[k] != reference[k] or type(row[k]) is not type(reference[k]):
                raise ValueError(f'sample identity mismatch: {k}')
        if row['split'] not in ('train', 'val') or type(row['class_id']) is not int:
            raise ValueError('invalid split/class')
    for key, shape in [('g_features',(2446,256)), ('c_features',(2446,14)),
                       ('s_features',(2446,16)), ('s_valid',(2446,)), ('class_id',(2446,))]:
        if features[key].shape != shape:
            raise ValueError(f'wrong feature shape: {key}')
    g, c, s = (features[k] for k in ('g_features','c_features','s_features'))
    if not np.isfinite(g).all() or not np.isfinite(c).all() or np.isinf(s).any():
        raise ValueError('nonfinite features')
    for block in (s[:, :8], s[:, 8:]):
        if ((np.isnan(block).sum(1) != 0) & (np.isnan(block).sum(1) != 8)).any():
            raise ValueError('partial NaN block')
    available = features['s_valid']
    if available.dtype != bool or not np.array_equal(available, np.isfinite(s).any(1)):
        raise ValueError('availability mismatch')
    if int(available.sum()) != 2341:
        raise ValueError('wrong available count')
    if not np.array_equal(features['class_id'], [r['class_id'] for r in rows]):
        raise ValueError('label identity mismatch')
    folds = read_json(ROOT/cfg['folds'])['folds']
    for f in folds:
        parts = require_partition(rows, f['fold_id'])
        for role, ids in parts.items():
            groups = {rows[i]['group'] for i in ids}
            if groups != set(f[role+'_groups']):
                raise ValueError('fold differs from frozen protocol')
    grid = None
    if load_grid:
        grid = np.empty((2446,9,128,128), np.float32)
        by_frame = {}
        for i, r in enumerate(rows):
            by_frame.setdefault(r['frame'], []).append(i)
        for frame, ids in by_frame.items():
            rel = f'cache/grid/frame_{frame}.npz'
            path = registry.check((base/rel).relative_to(ROOT).as_posix(), records[rel]['sha256'])
            with np.load(path, allow_pickle=False) as z:
                expected = {rows[i]['sample_name']+'|planes' for i in ids}
                if set(z.files) != expected:
                    raise ValueError('grid instance key mismatch')
                for i in ids:
                    arr = z[rows[i]['sample_name']+'|planes']
                    if arr.dtype != np.float32 or arr.shape != (9,128,128):
                        raise ValueError('invalid grid contract')
                    grid[i] = arr
    return {'rows':rows, 'features':features, 'grid':grid, 'binding':records, 'base':base}


def fold_inputs(data: dict, fold: str, cfg: dict) -> dict:
    rows, f = data['rows'], data['features']
    parts = require_partition(rows, fold)
    raw = np.concatenate((f['g_features'], f['c_features'], f['s_features']), axis=1)
    flags = np.isfinite(raw)
    ids = parts['inner_train']
    scaler = Standardizer.fit(raw[ids], flags[ids])
    context = scaler.transform(raw, flags)
    p0_dir = ROOT/cfg['p0_root']/'models'/f'{fold}_P0'
    p0_scaler = Standardizer.from_dict(read_json(p0_dir/'scaler.json'))
    full_raw = np.zeros((len(rows),270), np.float64)
    full_raw[:, :256] = f['g_features']
    design = p0_scaler.transform(full_raw, np.ones(full_raw.shape, bool))
    design[:,256:] = 0
    payload = torch.load(p0_dir/'model_state.pt', map_location='cpu', weights_only=False)
    if payload['variant'] != 'P0' or payload['fold'] != fold or payload['input_dim'] != 270:
        raise ValueError('wrong P0 checkpoint identity')
    head = torch.nn.Linear(270,4)
    head.load_state_dict(payload['state_dict'])
    with torch.inference_mode():
        p0 = head(torch.from_numpy(design)).numpy()
    oracle = [json.loads(line) for line in (p0_dir/'outer_eval_predictions.jsonl').read_text().splitlines()]
    by_name = {r['sample_name']:r for r in oracle}
    ev = parts['outer_eval']
    if set(by_name) != {rows[i]['sample_name'] for i in ev} or len(by_name) != len(oracle):
        raise ValueError('P0 oracle coverage mismatch')
    expected = np.array([by_name[rows[i]['sample_name']]['logits'] for i in ev], np.float32)
    if not np.allclose(p0[ev], expected, atol=1e-4, rtol=0) or not np.array_equal(p0[ev].argmax(1), expected.argmax(1)):
        raise ValueError('P0 identity mismatch')
    return {'parts':parts, 'context':context, 'p0':p0, 'scaler':scaler.to_dict(),
            'p0_max_error':float(np.abs(p0[ev]-expected).max())}


def permuted_eval(data: dict, ids: list[int], registry: InputRegistry) -> np.ndarray:
    grids = data['grid'][ids].copy()
    loaded = {}
    for offset, i in enumerate(ids):
        row = data['rows'][i]
        frame = row['frame']
        if frame not in loaded:
            rel = f'cache/permutation/frame_{frame}.json'
            record = data['binding'][rel]
            path = registry.check((data['base']/rel).relative_to(ROOT).as_posix(), record['sha256'])
            loaded[frame] = read_json(path)
        plane_map = dict(zip(GRID_PLANES, grids[offset]))
        plan = loaded[frame][row['sample_name']]
        mapped = apply_q_permutation(plane_map, plan)
        grids[offset] = np.stack([mapped[k] for k in GRID_PLANES])
    return grids
