"""Independent 45-fit YOLO26 replay, provenance and outcome verification.

This script deliberately does not import production metrics or summarizers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

SEEDS = (2026, 2027, 2028)
FOLDS = ('fold0', 'fold1', 'fold2')
VARIANTS = ('B_stats', 'B_L', 'B_R', 'B_Q')


def independent_metrics(truth: list[int], pred: list[int]) -> dict:
    if len(truth) != len(pred) or not truth:
        raise ValueError('unaligned or empty predictions')
    cm = [[0] * 4 for _ in range(4)]
    for actual, predicted in zip(truth, pred):
        if (type(actual) is not int or type(predicted) is not int
                or not 0 <= actual < 4 or not 0 <= predicted < 4):
            raise ValueError('invalid class type/range')
        cm[actual][predicted] += 1
    f1, recall = [], []
    for k in range(4):
        actual = sum(cm[k])
        predicted = sum(row[k] for row in cm)
        f1.append(2 * cm[k][k] / (actual + predicted) if actual + predicted else 0.)
        recall.append(cm[k][k] / actual if actual else 0.)
    return {'accuracy': sum(cm[k][k] for k in range(4)) / len(truth),
            'macro_f1': sum(f1) / 4, 'recall': recall,
            'confusion_matrix': cm, 'support': len(truth)}


def _row_metrics(rows: list[dict]) -> dict:
    return independent_metrics([r['class_id'] for r in rows],
                               [r['prediction'] for r in rows])


def _identity_map(rows: list[dict]) -> dict:
    by_name = {row['sample_name']: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError('duplicate OOF identity')
    return by_name


def summarize_records(records: dict, reference: dict) -> dict:
    """Independent seed-specific OOF counting, paired errors and frozen gates."""
    variants = {}
    for variant in ('Y0',) + VARIANTS:
        per_seed = {}
        for seed in SEEDS:
            fold_metrics, pooled = [], []
            for fold in FOLDS:
                rows = reference[(seed, fold)] if variant == 'Y0' else records[(variant, seed, fold)]
                fold_metrics.append(_row_metrics(rows))
                pooled.extend(rows)
            _identity_map(pooled)
            per_seed[str(seed)] = {'folds': fold_metrics, 'pooled': _row_metrics(pooled),
                                   'fold_mean': sum(m['macro_f1'] for m in fold_metrics) / 3}
        variants[variant] = {'per_seed': per_seed,
                             'mean': sum(p['fold_mean'] for p in per_seed.values()) / 3}
    paired = {}
    for base in ('Y0', 'B_L', 'B_R', 'B_stats'):
        per_seed = {}
        for seed in SEEDS:
            fixed, broke, nets = 0, 0, {}
            for fold in FOLDS:
                left = reference[(seed, fold)] if base == 'Y0' else records[(base, seed, fold)]
                lookup = _identity_map(left)
                right = _identity_map(records[('B_Q', seed, fold)])
                if lookup.keys() != right.keys():
                    raise ValueError('paired identity coverage mismatch')
                for name, row in right.items():
                    old = lookup[name]
                    if any(old[k] != row[k] for k in ('class_id', 'group')):
                        raise ValueError('paired identity truth/group mismatch')
                    a = old['prediction'] == old['class_id']
                    b = row['prediction'] == row['class_id']
                    fixed += int(b and not a)
                    broke += int(a and not b)
                    nets[row['group']] = nets.get(row['group'], 0) + int(b) - int(a)
            per_seed[str(seed)] = {'fixed': fixed, 'broke': broke, 'group_net': nets}
        paired[base] = per_seed
    q = variants['B_Q']
    gates = {'gate1': q['mean'] - variants['Y0']['mean'] >= .01
                     and q['mean'] - variants['B_L']['mean'] >= .005,
             'gate2': q['mean'] > variants['B_R']['mean']
                     and sum(p['fixed'] - p['broke'] for p in paired['B_R'].values()) > 0,
             'gate3': sum(all(paired[b][str(s)]['fixed'] > paired[b][str(s)]['broke']
                             for b in ('Y0', 'B_L')) for s in SEEDS) >= 2}
    diffs = [[sum(q['per_seed'][str(s)]['folds'][f]['macro_f1']
                  - variants[b]['per_seed'][str(s)]['folds'][f]['macro_f1']
                  for s in SEEDS) / 3 for f in range(3)] for b in ('Y0', 'B_L')]
    gates['gate4'] = (sum(diffs[0][f] >= 0 and diffs[1][f] >= 0 for f in range(3)) >= 2
                      and min(min(x) for x in diffs) >= -.01)
    gates['gate5'] = all(sum(q['per_seed'][str(s)]['pooled']['recall'][k]
                            - variants[b]['per_seed'][str(s)]['pooled']['recall'][k]
                            for s in SEEDS) / 3 >= -.01
                        for b in ('Y0', 'B_L') for k in range(4))
    groups = set().union(*(p['group_net'] for p in paired['Y0'].values()))
    gates['gate6'] = sum(sum(paired['Y0'][str(s)]['group_net'].get(g, 0)
                            for s in SEEDS) > 0 for g in groups) >= 2
    delta = q['mean'] - variants['B_stats']['mean']
    net = sum(p['fixed'] - p['broke'] for p in paired['B_stats'].values())
    overall_gain_passed = all(gates.values())
    spatial_passed = delta > 0 and net > 0
    return {'variants': variants, 'paired': paired, 'gates': gates,
            'overall_gain_passed': overall_gain_passed,
            'advance': overall_gain_passed and spatial_passed,
            'spatial_contribution': {'mean_f1_delta': delta, 'net_correct': net,
                                     'passed': spatial_passed},
            'note': 'Exploratory existing-data comparison; no blind test or material causality claim.'}


def compare(actual: Any, expected: Any, path: str = '') -> None:
    if type(actual) is not type(expected):
        raise ValueError(f'type mismatch: {path}')
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f'key mismatch: {path}')
        for key in expected:
            compare(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f'length mismatch: {path}')
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{path}[{i}]')
    elif isinstance(expected, float):
        if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > 1e-6:
            raise ValueError(f'numeric mismatch: {path}')
    elif actual != expected:
        raise ValueError(f'value mismatch: {path}')


def check_prediction_rows(saved: list[dict], logits: np.ndarray, data: dict,
                          ids: list[int], y0: np.ndarray, *, baseline: bool = False) -> None:
    if len(saved) != len(ids) or len(_identity_map(saved)) != len(ids):
        raise ValueError('prediction coverage mismatch')
    if logits.shape != (len(ids), 4) or not np.isfinite(logits).all():
        raise ValueError('invalid replay logits')
    for offset, (row, i) in enumerate(zip(saved, ids)):
        for key in ('sample_name', 'group', 'class_id'):
            compare(row[key], data['rows'][i][key], key)
        compare(row['prediction'], int(logits[offset].argmax()), 'prediction')
        compare(row['y0_prediction'], int(y0[i].argmax()), 'Y0 prediction')
        stored = np.array(row['logits'], np.float32)
        if stored.shape != (4,) or not np.isfinite(stored).all() or not np.allclose(
                stored, logits[offset], atol=1e-6, rtol=0):
            raise ValueError('weight replay logits mismatch')
        fallback = False if baseline else not bool(data['features']['s_valid'][i])
        compare(row['used_fallback'], fallback, 'fallback')
        if fallback and not np.array_equal(stored, y0[i]):
            raise ValueError('fallback is not exact fold/seed Y0')


def verify_scaler(payload: dict, raw: np.ndarray, train_ids: list[int]) -> np.ndarray:
    """Recompute each statistic without invoking the production Standardizer."""
    training = raw[train_ids].astype(np.float64)
    width = raw.shape[1]
    if any(len(payload[key]) != width for key in ('mean', 'std', 'usable')):
        raise ValueError('scaler dimension mismatch')
    out = np.zeros(raw.shape, np.float32)
    for k in range(width):
        finite = training[:, k][np.isfinite(training[:, k])]
        mean = float(finite.mean()) if len(finite) else 0.
        std = float(finite.std(ddof=0)) if len(finite) else 0.
        usable = std > 0
        if (not np.isfinite(payload['mean'][k]) or not np.isfinite(payload['std'][k])
                or abs(payload['mean'][k] - mean) > 1e-6
                or abs(payload['std'][k] - std) > 1e-6
                or type(payload['usable'][k]) is not bool or payload['usable'][k] != usable):
            raise ValueError('independent inner_train scaler mismatch')
        if usable:
            valid = np.isfinite(raw[:, k])
            out[valid, k] = (raw[valid, k].astype(np.float64) - mean) / std
    return out


def check_history(history: list[dict], result: dict, *, train_used: int,
                  max_epochs: int, exact_epochs: bool = False) -> dict:
    if ([r['epoch'] for r in history] != list(range(1, len(history) + 1))
            or not 1 <= len(history) <= max_epochs
            or (exact_epochs and len(history) != max_epochs)):
        raise ValueError('invalid epoch history')
    for row in history:
        if any(not np.isfinite(row[k]) or not 0 <= row[k] <= 1
               for k in ('macro_f1', 'accuracy')):
            raise ValueError('invalid history metrics')
        if type(row['train_used']) is not int or row['train_used'] != train_used:
            raise ValueError('incorrect instances entered training loss')
    selected = sorted(history, key=lambda r: (-r['macro_f1'], -r['accuracy'], r['epoch']))[0]
    if (result['selected_epoch'] != selected['epoch']
            or result['completed_epochs'] != len(history) or result.get('smoke') is not False):
        raise ValueError('selection/smoke mismatch')
    if not exact_epochs:
        best_key, stale, stop_epoch = None, 0, None
        for row in history:
            key = (row['macro_f1'], row['accuracy'], -row['epoch'])
            if best_key is None or key > best_key:
                best_key, stale = key, 0
            else:
                stale += 1
            if stale == 10:
                stop_epoch = row['epoch']
                break
        if ((stop_epoch is not None and stop_epoch != len(history))
                or (stop_epoch is None and len(history) < max_epochs)):
            raise ValueError('formal spatial patience10 stopping mismatch')
    return selected


def require_formal_status(status: dict, config_hash: str) -> None:
    if (status['mode'] != 'fit' or status['overall_status'] != 'COMPLETED'
            or status['completed_fits'] != 45 or status['gray_fits'] != 9
            or status['spatial_fits'] != 36 or status.get('training_status') != 'COMPLETED'):
        raise ValueError('fit not formally complete: require 9 gray + 36 spatial fits')
    if status['config_sha256'] != config_hash:
        raise ValueError('config changed after training')


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def compare_backbone_to_official(selected: Any, official: Any) -> dict:
    """Measure learning from actual selected tensors, excluding replaced linear."""
    import torch
    old_parameters = dict(official.named_parameters())
    selected_parameters = [(name, value) for name, value in selected.named_parameters()
                           if '.linear.' not in name and not name.startswith('linear.')]
    changed = []
    for name, value in selected_parameters:
        if name not in old_parameters or value.shape != old_parameters[name].shape:
            raise ValueError('official/selected backbone identity differs')
        if not torch.equal(value.detach().cpu(), old_parameters[name].detach().cpu()):
            changed.append(name)
    early = selected_parameters[0][0] if selected_parameters else ''
    if early not in changed:
        raise ValueError('actual selected early backbone did not learn')
    old_buffers = dict(official.named_buffers())
    bn_changes = []
    for name, value in selected.named_buffers():
        if not name.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            continue
        if name not in old_buffers or value.shape != old_buffers[name].shape:
            raise ValueError('official/selected BN buffer identity differs')
        if not torch.equal(value.detach().cpu(), old_buffers[name].detach().cpu()):
            bn_changes.append(name)
    if not bn_changes:
        raise ValueError('actual selected BN buffers did not learn')
    return {'actual_backbone_changed_count': len(changed),
            'actual_bn_buffers_changed_count': len(bn_changes),
            'actual_early_backbone_parameter': early, 'actual_early_backbone_changed': True}


def _verify_roles(result: dict, parts: dict, rows: list[dict], fold: str, seed: int) -> None:
    compare(result['fold'], fold, 'fold')
    compare(result['seed'], seed, 'seed')
    for role, prefix in [('inner_train', 'train'), ('inner_val', 'val'), ('outer_eval', 'outer')]:
        compare(result[prefix + '_ids'], parts[role], role)
        groups = sorted({rows[i]['group'] for i in parts[role]})
        compare(result[prefix + '_groups'], groups, role + ' groups')


def _verify_manifest(fit: Path, registry: Any, expected_hash: str, check: Any) -> None:
    from core.material_spatial_classifier_io import read_json
    manifest_path = fit / 'artifact_manifest.json'
    registry.check(manifest_path.relative_to(ROOT).as_posix(), expected_hash)
    manifest = read_json(manifest_path)
    listed = set()
    for record in manifest['artifacts']:
        check()
        rel = Path(record['path'])
        target = (fit / rel).resolve()
        if (rel.is_absolute() or '..' in rel.parts or not target.is_relative_to(fit.resolve())
                or record['path'] in listed or type(record['bytes']) is not int):
            raise ValueError('invalid artifact identity/path')
        listed.add(record['path'])
        path = registry.check(target.relative_to(ROOT).as_posix(), record['sha256'])
        if path.stat().st_size != record['bytes']:
            raise ValueError('artifact bytes mismatch')
    required = {'config_snapshot.json', 'summary.json', 'run_status.json',
                'source_protection.json', 'protocol_snapshot.md'}
    for fold in FOLDS:
        for seed in SEEDS:
            prefix = f'gray/{fold}_s{seed}/'
            required.update(prefix + name for name in ('best.pt', 'history.jsonl', 'result.json',
                            'features.npz', 'extraction.json', 'context_scaler.json',
                            'predictions.jsonl', 'val_predictions.jsonl'))
            for variant in VARIANTS:
                prefix = f'models/{fold}_{variant}_s{seed}/'
                required.update(prefix + name for name in ('model.pt', 'result.json', 'history.jsonl',
                                                           'predictions.jsonl', 'val_predictions.jsonl'))
                if variant == 'B_Q':
                    required.add(prefix + 'permuted_predictions.jsonl')
    if not required.issubset(listed):
        raise ValueError('artifact manifest missing consumed evidence')


def verify(config: Path, fit: Path, manifest_hash: str, output: Path,
           deadline_utc: str) -> dict:
    import torch
    from core.material_residual_scale import DeadlineWatch
    from core.material_spatial_classifier import assemble_planes, predict_with_fallback, require_partition
    from core.material_spatial_classifier_io import InputRegistry, digest, read_json, permuted_eval
    from core.yolo26_spatial import Yolo26SpatialClassifier, extract_gray, state_digest
    from core.yolo26_spatial_io import load_experiment_inputs
    from models.polar_fusion import architecture_name, prepare_gray_backbone
    from ultralytics import YOLO

    registry = InputRegistry()
    watch = DeadlineWatch(deadline_utc=deadline_utc)
    check = lambda: watch.check('independent verification')
    config_hash = digest(config)
    cfg = read_json(registry.check(config.relative_to(ROOT).as_posix(), config_hash))
    _verify_manifest(fit, registry, manifest_hash, check)
    status = read_json(fit / 'run_status.json')
    require_formal_status(status, config_hash)
    compare(read_json(fit / 'config_snapshot.json'), cfg, 'configuration snapshot')
    protection = read_json(fit / 'source_protection.json')
    if protection['status'] != 'IDENTICAL':
        raise ValueError('training source protection failed')
    for source in protection['files']:
        registry.check(source['path'], source['sha256'])
    protocol_path = ROOT / 'docs/experiments/yolo26_spatial_v1.md'
    protocol_sources = [item for item in protection['files']
                        if item['path'] == protocol_path.relative_to(ROOT).as_posix()]
    if len(protocol_sources) != 1:
        raise ValueError('frozen protocol source is not protected')
    if (fit / 'protocol_snapshot.md').read_text(encoding='utf-8-sig').replace('\r\n', '\n') != \
            protocol_path.read_text(encoding='utf-8-sig').replace('\r\n', '\n'):
        raise ValueError('frozen protocol snapshot differs')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = load_experiment_inputs(config, registry, load_grid=True)
    official = YOLO(str(ROOT / cfg['official_base'])).model.float().cpu().eval()
    records, reference, extraction_checks = {}, {}, []
    for fold in FOLDS:
        parts = require_partition(data['rows'], fold)
        train_ids, val_ids, eval_ids = (parts[k] for k in ('inner_train', 'inner_val', 'outer_eval'))
        permuted = permuted_eval(data, eval_ids, registry)
        for seed in SEEDS:
            check()
            job = fit / 'gray' / f'{fold}_s{seed}'
            result = read_json(job / 'result.json')
            _verify_roles(result, parts, data['rows'], fold, seed)
            history = _read_rows(job / 'history.jsonl')
            selected = check_history(history, result, train_used=len(train_ids),
                                     max_epochs=30, exact_epochs=True)
            evidence = result['training_evidence']
            for key in ('all_parameters_trainable', 'optimizer_covers_all',
                        'early_backbone_gradient_nonzero', 'early_backbone_changed', 'head_changed'):
                compare(evidence[key], True, 'training evidence ' + key)
            if evidence['backbone_changed_count'] <= 0 or evidence['bn_buffers_changed_count'] <= 0:
                raise ValueError('gray backbone did not learn')
            compare(result['official_base'], cfg['official_base'], 'official base path')
            compare(result['official_base_sha256'], cfg['official_base_sha256'], 'official base hash')
            adapter, info = prepare_gray_backbone(ROOT / cfg['official_base'],
                str(job / 'best.pt'), ['metal_submarine', 'plastic_submarine',
                                      'plastic_fish', 'real_fish'], 'cuda:0')
            if (getattr(adapter.module, 'task', None) != 'classify'
                    or architecture_name(adapter.module) != 'yolo26n-cls' or info['head_replaced']):
                raise ValueError('gray checkpoint is not trained four-class YOLO26')
            compare(info['gray_class_names'], cfg['class_names'], 'gray class order')
            if adapter.module.model[-1].linear.in_features != 1280:
                raise ValueError('wrong gray feature dimension')
            # Training constructed a dataset-order model with perm=None. The loader
            # provides a redundant identity permutation; remove it before hashing.
            compare(info['perm'], [0, 1, 2, 3], 'gray class permutation')
            adapter.perm = None
            adapter.eval()
            learning = compare_backbone_to_official(adapter.module, official)
            before = state_digest(adapter)
            compare(before, result['selected_state_sha256'], 'selected gray state')
            g1280, y0, extraction = extract_gray(adapter, data['gray'], device='cuda:0', batch=32, check=check)
            if state_digest(adapter) != before:
                raise ValueError('gray replay changed parameters or BN buffers')
            frozen = read_json(job / 'extraction.json')
            if (frozen['state_before'] != frozen['state_after']
                    or frozen['state_before'] != before
                    or frozen['logit_reconstruction_max_error'] > 1e-4):
                raise ValueError('training extraction altered gray state or linear identity')
            for key in ('state_identical', 'parameters_frozen', 'eval_mode'):
                compare(frozen[key], True, 'extraction ' + key)
                compare(extraction[key], True, 'replay extraction ' + key)
            with np.load(job / 'features.npz', allow_pickle=False) as cached:
                if set(cached.files) != {'g1280', 'logits'}:
                    raise ValueError('invalid gray feature cache keys')
                for key, actual in [('g1280', g1280), ('logits', y0)]:
                    if cached[key].shape != actual.shape or not np.allclose(
                            cached[key], actual, atol=1e-5, rtol=0):
                        raise ValueError('gray extraction replay mismatch')
            if g1280.shape != (2446, 1280) or y0.shape != (2446, 4):
                raise ValueError('invalid gray extraction census')
            saved = _read_rows(job / 'predictions.jsonl')
            check_prediction_rows(saved, y0[eval_ids], data, eval_ids, y0, baseline=True)
            check_prediction_rows(_read_rows(job / 'val_predictions.jsonl'), y0[val_ids],
                                  data, val_ids, y0, baseline=True)
            reference[(seed, fold)] = saved
            val_metrics = independent_metrics(data['features']['class_id'][val_ids].tolist(),
                                               y0[val_ids].argmax(1).tolist())
            for key in ('macro_f1', 'accuracy', 'confusion_matrix', 'recall', 'support'):
                compare(selected[key], val_metrics[key], 'selected gray val ' + key)
            compare(result['metrics'], _row_metrics(saved), 'gray outer metrics')
            raw = np.concatenate((g1280, data['features']['c_features'],
                                  data['features']['s_features']), axis=1)
            context = verify_scaler(read_json(job / 'context_scaler.json'), raw, train_ids)
            extraction_checks.append({'fold': fold, 'seed': seed, 'state_unchanged': True,
                                      'linear_max_error': float(extraction['logit_reconstruction_max_error']),
                                      **learning})
            del adapter
            for variant in VARIANTS:
                check()
                space = fit / 'models' / f'{fold}_{variant}_s{seed}'
                result = read_json(space / 'result.json')
                _verify_roles(result, parts, data['rows'], fold, seed)
                compare(result['variant'], variant, 'variant')
                history = _read_rows(space / 'history.jsonl')
                available = data['features']['s_valid']
                selected = check_history(history, result, train_used=int(available[train_ids].sum()),
                                         max_epochs=60)
                checkpoint = torch.load(space / 'model.pt', map_location='cpu', weights_only=False)
                for key, expected in [('fold', fold), ('variant', variant), ('seed', seed),
                        ('selected_epoch', selected['epoch']), ('config_sha256', config_hash)]:
                    compare(checkpoint[key], expected, key)
                model = Yolo26SpatialClassifier().to('cuda:0')
                model.load_state_dict(checkpoint['state_dict'])
                def replay(ids: list[int], grids: np.ndarray) -> np.ndarray:
                    check()
                    return predict_with_fallback(model, assemble_planes(grids, variant), context[ids],
                                                 available[ids], y0[ids], 'cuda:0', 32)
                logits = replay(eval_ids, data['grid'][eval_ids])
                saved = _read_rows(space / 'predictions.jsonl')
                check_prediction_rows(saved, logits, data, eval_ids, y0)
                compare(result['metrics'], _row_metrics(saved), 'space outer metrics')
                val_logits = replay(val_ids, data['grid'][val_ids])
                check_prediction_rows(_read_rows(space / 'val_predictions.jsonl'), val_logits,
                                      data, val_ids, y0)
                metrics = independent_metrics(data['features']['class_id'][val_ids].tolist(),
                                               val_logits.argmax(1).tolist())
                for key in ('macro_f1', 'accuracy', 'confusion_matrix', 'recall', 'support'):
                    compare(selected[key], metrics[key], 'selected space val ' + key)
                if variant == 'B_Q':
                    perm_saved = _read_rows(space / 'permuted_predictions.jsonl')
                    check_prediction_rows(perm_saved, replay(eval_ids, permuted), data, eval_ids, y0)
                records[(variant, seed, fold)] = saved
                del model
    rebuilt = summarize_records(records, reference)
    production = read_json(fit / 'summary.json')
    for key in ('variants', 'paired', 'gates', 'overall_gain_passed', 'advance', 'spatial_contribution'):
        compare(production[key], rebuilt[key], 'summary.' + key)
    from core.material_spatial_classifier_io import write_json
    write_json(output / 'independent_summary.json', rebuilt)
    write_json(output / 'source_protection.json', registry.finish())
    check()
    return {'status': 'PASSED', 'problem_count': 0, 'gray_models': 9, 'spatial_models': 36,
            'unique_samples': 2446, 'folds': 3, 'gray_extraction_checks': extraction_checks,
            'gates': rebuilt['gates'], 'advance': rebuilt['advance'],
            'overall_gain_passed': rebuilt['overall_gain_passed'],
            'spatial_contribution': rebuilt['spatial_contribution']}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--fit-dir', required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args(argv)
    output = (ROOT / args.output_dir).resolve()
    allowed = (ROOT / 'analysis/runs/yolo26_spatial_v1').resolve()
    if output == allowed or not output.is_relative_to(allowed) or output.exists():
        raise ValueError('fresh experiment-local verification output required')
    output.mkdir(parents=True)
    report = {'status': 'STARTED', 'problem_count': 0}
    try:
        report = verify((ROOT / args.config).resolve(), (ROOT / args.fit_dir).resolve(),
                        args.expected_manifest_sha256, output, args.deadline_utc)
        return 0
    except Exception as exc:
        report.update(status='FAILED', problem_count=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output / 'verification_report.json').write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
