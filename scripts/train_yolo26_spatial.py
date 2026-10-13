"""Authorized fold-isolated YOLO26 grayscale and spatial validation."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from core.material_residual_scale import DeadlineWatch
from core.material_spatial_classifier import (
    classification_metrics, predict_with_fallback, require_partition,
)
from core.material_spatial_classifier_io import (
    InputRegistry, digest, permuted_eval, write_json,
)
from core.yolo26_spatial import build_gray, extract_gray, fit_gray, fit_spatial
from core.material_spatial_classifier import assemble_planes
from core.yolo26_spatial_io import (
    CONFIG_SHA256, fold_context, fresh_output, load_experiment_inputs,
    prediction_rows,
)

SEEDS = (2026, 2027, 2028)
FOLDS = ('fold0', 'fold1', 'fold2')
VARIANTS = ('B_stats', 'B_L', 'B_R', 'B_Q')


def configured_output(value: str) -> Path:
    parent = ROOT/'analysis/runs/yolo26_spatial_v1'
    raw = Path(value)
    if '..' in raw.parts:
        raise ValueError('output traversal refused')
    candidate = raw if raw.is_absolute() else ROOT/raw
    try:
        relative = candidate.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError('output outside this experiment') from exc
    return fresh_output(parent, relative.as_posix())


def save_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows),
                    encoding='utf-8')


def summarize(records: dict, reference: dict) -> dict:
    """Training-side summarizer, independently replayed by the verifier."""
    result = {'variants': {}, 'paired': {}, 'gates': {}}
    for variant in ('Y0',) + VARIANTS:
        per_seed = {}
        for seed in SEEDS:
            fold_metrics, pooled = [], []
            for fold in FOLDS:
                rows = reference[(seed, fold)] if variant == 'Y0' else records[(variant, seed, fold)]
                if len({r['sample_name'] for r in rows}) != len(rows):
                    raise ValueError('duplicate evaluation identity')
                baseline = reference[(seed, fold)]
                if {(r['sample_name'], r['class_id'], r['group']) for r in rows} != {
                    (r['sample_name'], r['class_id'], r['group']) for r in baseline
                }:
                    raise ValueError('evaluation identities differ')
                fold_metrics.append(classification_metrics(
                    [r['class_id'] for r in rows], [r['prediction'] for r in rows]))
                pooled.extend(rows)
            if len({r['sample_name'] for r in pooled}) != len(pooled):
                raise ValueError('OOF sample repeated across folds')
            per_seed[str(seed)] = {
                'folds': fold_metrics,
                'pooled': classification_metrics([r['class_id'] for r in pooled],
                                                  [r['prediction'] for r in pooled]),
                'fold_mean': float(np.mean([m['macro_f1'] for m in fold_metrics])),
            }
        result['variants'][variant] = {
            'per_seed': per_seed,
            'mean': float(np.mean([v['fold_mean'] for v in per_seed.values()])),
        }
    for baseline in ('Y0', 'B_L', 'B_R', 'B_stats'):
        per_seed = {}
        for seed in SEEDS:
            fixed, broke, group_net = 0, 0, {}
            for fold in FOLDS:
                before = reference[(seed, fold)] if baseline == 'Y0' else records[(baseline, seed, fold)]
                by_name = {r['sample_name']: r for r in before}
                for row in records[('B_Q', seed, fold)]:
                    other = by_name[row['sample_name']]
                    change = int(row['prediction'] == row['class_id']) - int(other['prediction'] == other['class_id'])
                    fixed += int(change == 1)
                    broke += int(change == -1)
                    group_net[row['group']] = group_net.get(row['group'], 0) + change
            per_seed[str(seed)] = {'fixed': fixed, 'broke': broke,
                                   'group_net': dict(sorted(group_net.items()))}
        result['paired'][baseline] = per_seed
    q = result['variants']['B_Q']
    delta = {b: q['mean']-result['variants'][b]['mean']
             for b in ('Y0', 'B_L', 'B_R', 'B_stats')}
    paired = result['paired']
    result['gates']['gate1'] = delta['Y0'] >= .01 and delta['B_L'] >= .005
    result['gates']['gate2'] = delta['B_R'] > 0 and sum(
        p['fixed']-p['broke'] for p in paired['B_R'].values()) > 0
    result['gates']['gate3'] = sum(all(
        paired[b][str(s)]['fixed'] > paired[b][str(s)]['broke']
        for b in ('Y0', 'B_L')) for s in SEEDS) >= 2
    fold_deltas, recall_deltas = [], []
    for b in ('Y0', 'B_L'):
        for f in range(3):
            fold_deltas.append(float(np.mean([
                q['per_seed'][str(s)]['folds'][f]['macro_f1'] -
                result['variants'][b]['per_seed'][str(s)]['folds'][f]['macro_f1']
                for s in SEEDS])))
        recall_deltas.extend(np.mean([
            np.array(q['per_seed'][str(s)]['pooled']['recall']) -
            result['variants'][b]['per_seed'][str(s)]['pooled']['recall']
            for s in SEEDS], axis=0).tolist())
    result['gates']['gate4'] = sum(
        fold_deltas[f] >= 0 and fold_deltas[f+3] >= 0 for f in range(3)
    ) >= 2 and min(fold_deltas) >= -.01
    result['gates']['gate5'] = min(recall_deltas) >= -.01
    groups = set().union(*(p['group_net'] for p in paired['Y0'].values()))
    result['gates']['gate6'] = sum(sum(
        paired['Y0'][str(s)]['group_net'].get(g, 0) for s in SEEDS
    ) > 0 for g in groups) >= 2
    net = sum(p['fixed']-p['broke'] for p in paired['B_stats'].values())
    result['spatial_contribution'] = {
        'mean_f1_delta': delta['B_stats'], 'net_correct': net,
        'passed': delta['B_stats'] > 0 and net > 0,
    }
    result['overall_gain_passed'] = all(result['gates'].values())
    result['advance'] = result['overall_gain_passed'] and result['spatial_contribution']['passed']
    result['note'] = 'Existing-data capture-group validation; shared S16; no blind-test, detection or material-causality claim.'
    return result


def role_metadata(data: dict, parts: dict) -> dict:
    result = {}
    for role, prefix in (('inner_train', 'train'), ('inner_val', 'val'), ('outer_eval', 'outer')):
        ids = parts[role]
        result[prefix+'_ids'] = ids
        result[prefix+'_groups'] = sorted({data['rows'][i]['group'] for i in ids})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/yolo26_spatial_v1.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'fit'), required=True)
    parser.add_argument('--budget-minutes', type=float, default=480.)
    parser.add_argument('--deadline-utc')
    args = parser.parse_args(argv)
    output = configured_output(args.output_dir)
    output.mkdir(parents=True)
    watch = DeadlineWatch(deadline_utc=args.deadline_utc) if args.deadline_utc else DeadlineWatch(budget_minutes=args.budget_minutes)
    status = {'mode': args.mode, 'training_status': 'NOT_RUN', 'overall_status': 'STARTED',
              'completed_fits': 0, 'gray_fits': 0, 'spatial_fits': 0,
              'config_sha256': CONFIG_SHA256,
              'deadline_utc': datetime.fromtimestamp(watch._deadline, timezone.utc).isoformat()}
    registry, started = InputRegistry(), time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('frozen CUDA device unavailable')
        for rel in ('core/yolo26_spatial.py', 'core/yolo26_spatial_io.py',
                    'scripts/train_yolo26_spatial.py', 'scripts/verify_yolo26_spatial.py',
                    'core/material_spatial_classifier.py', 'core/material_spatial_classifier_io.py',
                    'core/material_probe.py', 'core/material_spatial_input.py',
                    'core/material_residual_scale.py', 'models/polar_fusion.py',
                    'docs/experiments/yolo26_spatial_v1.md'):
            registry.check(rel, digest(ROOT/rel))
        data = load_experiment_inputs(ROOT/args.config, registry, load_grid=args.mode != 'preflight')
        cfg = data['cfg']
        write_json(output/'config_snapshot.json', cfg)
        (output/'protocol_snapshot.md').write_text(
            (ROOT/'docs/experiments/yolo26_spatial_v1.md').read_text(encoding='utf-8'),
            encoding='utf-8')
        watch.check('input admission')
        write_json(output/'admission.json', {'samples': len(data['rows']),
                   'available': int(data['features']['s_valid'].sum()), 'groups': 21,
                   'gray_shape': list(data['gray'].shape), 'gray_dtype': str(data['gray'].dtype),
                   'gray_source': 'hash-bound admitted native L', 'MiT_G256_used': False})
        if args.mode == 'preflight':
            probe = build_gray(ROOT/cfg['official_base'], seed=2026, device='cuda:0')
            if probe.module.model[-1].linear.in_features != 1280:
                raise ValueError('unexpected YOLO26 pre-linear width')
            del probe
            for fold in FOLDS:
                parts = require_partition(data['rows'], fold)
                write_json(output/f'{fold}_preflight.json', role_metadata(data, parts))
            status['overall_status'] = 'COMPLETED'
            print('PREFLIGHT PASSED: 2446 identities, native L, official YOLO26, isolated folds.', flush=True)
            return 0
        records, reference = {}, {}
        folds = ('fold0',) if args.mode == 'smoke' else FOLDS
        seeds = (2026,) if args.mode == 'smoke' else SEEDS
        for fold in folds:
            all_parts = require_partition(data['rows'], fold)
            parts = {role: ids[:32] if args.mode == 'smoke' else ids
                     for role, ids in all_parts.items()}
            ids_eval = parts['outer_eval']
            permutation = permuted_eval(data, ids_eval, registry)
            for seed in seeds:
                gray_job = output/'gray'/f'{fold}_s{seed}'
                gray_job.mkdir(parents=True)
                fit_started = time.monotonic()

                def gray_check() -> None:
                    watch.check('gray fit, extraction or evaluation')
                    if time.monotonic()-fit_started > 1200:
                        raise TimeoutError('gray fit exceeded20 minutes')

                def gray_log(row: dict) -> None:
                    with (gray_job/'history.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(row, allow_nan=False)+'\n')
                    print(f'GRAY {fold}/s{seed} epoch={row["epoch"]} val_f1={row["macro_f1"]:.5f}', flush=True)

                def gray_subset(role: str) -> dict:
                    ids = parts[role]
                    return {'gray': data['gray'][ids], 'labels': data['features']['class_id'][ids]}

                print(f'FIT GRAY {fold}/s{seed}', flush=True)
                model, history, best = fit_gray(gray_subset('inner_train'), gray_subset('inner_val'),
                    base_path=ROOT/cfg['official_base'], seed=seed, device='cuda:0',
                    epochs=2 if args.mode == 'smoke' else 30, check=gray_check, epoch_callback=gray_log)
                torch.save({'model': copy.deepcopy(model.module).cpu().eval()}, gray_job/'best.pt')
                registry.check((gray_job/'best.pt').relative_to(ROOT).as_posix(), digest(gray_job/'best.pt'))
                g, y0, extraction = extract_gray(model, data['gray'], device='cuda:0', check=gray_check)
                np.savez_compressed(gray_job/'features.npz', g1280=g, logits=y0)
                write_json(gray_job/'extraction.json', extraction)
                wired = fold_context(g, data['features'], data['rows'], fold)
                write_json(gray_job/'context_scaler.json', wired['scaler'])
                predictions = prediction_rows(data, ids_eval, y0[ids_eval], y0, baseline=True)
                save_rows(gray_job/'predictions.jsonl', predictions)
                save_rows(gray_job/'val_predictions.jsonl', prediction_rows(
                    data, parts['inner_val'], y0[parts['inner_val']], y0, baseline=True))
                result = dict(best)
                result.update(fold=fold, seed=seed, selected_epoch=best['epoch'],
                              completed_epochs=len(history), elapsed_s=time.monotonic()-fit_started,
                              smoke=args.mode == 'smoke', official_base=cfg['official_base'],
                              official_base_sha256=cfg['official_base_sha256'],
                              metrics=classification_metrics([r['class_id'] for r in predictions],
                                                             [r['prediction'] for r in predictions]),
                              **role_metadata(data, parts))
                write_json(gray_job/'result.json', result)
                reference[(seed, fold)] = predictions
                status['gray_fits'] += 1
                status['completed_fits'] += 1
                print(f'GRAY DONE {fold}/s{seed} epoch={best["epoch"]} outer_f1={result["metrics"]["macro_f1"]:.6f}', flush=True)
                del model
                torch.cuda.empty_cache()
                for variant in VARIANTS:
                    images = assemble_planes(data['grid'], variant)

                    def spatial_subset(role: str) -> dict:
                        ids = parts[role]
                        return {'images': images[ids], 'context': wired['context'][ids],
                                'available': data['features']['s_valid'][ids],
                                'p0': y0[ids], 'labels': data['features']['class_id'][ids]}

                    job = output/'models'/f'{fold}_{variant}_s{seed}'
                    job.mkdir(parents=True)
                    fit_started = time.monotonic()

                    def spatial_check() -> None:
                        watch.check('spatial fit or evaluation')
                        if time.monotonic()-fit_started > 600:
                            raise TimeoutError('spatial fit exceeded10 minutes')

                    def spatial_log(row: dict) -> None:
                        with (job/'history.jsonl').open('a', encoding='utf-8') as handle:
                            handle.write(json.dumps(row, allow_nan=False)+'\n')

                    print(f'FIT SPATIAL {fold}/{variant}/s{seed}', flush=True)
                    spatial, history, best = fit_spatial(
                        spatial_subset('inner_train'), spatial_subset('inner_val'),
                        seed=seed, device='cuda:0', epochs=2 if args.mode == 'smoke' else 60,
                        check=spatial_check, epoch_callback=spatial_log)
                    logits = predict_with_fallback(spatial, images[ids_eval], wired['context'][ids_eval],
                        data['features']['s_valid'][ids_eval], y0[ids_eval], 'cuda:0', 32)
                    predictions = prediction_rows(data, ids_eval, logits, y0)
                    save_rows(job/'predictions.jsonl', predictions)
                    val_ids = parts['inner_val']
                    val_logits = predict_with_fallback(spatial, images[val_ids], wired['context'][val_ids],
                        data['features']['s_valid'][val_ids], y0[val_ids], 'cuda:0', 32)
                    save_rows(job/'val_predictions.jsonl', prediction_rows(data, val_ids, val_logits, y0))
                    if variant == 'B_Q':
                        plogits = predict_with_fallback(spatial, assemble_planes(permutation, variant),
                            wired['context'][ids_eval], data['features']['s_valid'][ids_eval],
                            y0[ids_eval], 'cuda:0', 32)
                        save_rows(job/'permuted_predictions.jsonl', prediction_rows(data, ids_eval, plogits, y0))
                    torch.save({'state_dict': spatial.cpu().state_dict(), 'fold': fold,
                                'variant': variant, 'seed': seed, 'selected_epoch': best['epoch'],
                                'config_sha256': CONFIG_SHA256}, job/'model.pt')
                    result = dict(best)
                    result.update(fold=fold, variant=variant, seed=seed, selected_epoch=best['epoch'],
                                  completed_epochs=len(history), elapsed_s=time.monotonic()-fit_started,
                                  smoke=args.mode == 'smoke', **role_metadata(data, parts),
                                  metrics=classification_metrics([r['class_id'] for r in predictions],
                                                                 [r['prediction'] for r in predictions]))
                    write_json(job/'result.json', result)
                    records[(variant, seed, fold)] = predictions
                    status['spatial_fits'] += 1
                    status['completed_fits'] += 1
                    spatial_check()
                    print(f'SPATIAL DONE epoch={best["epoch"]} outer_f1={result["metrics"]["macro_f1"]:.6f}', flush=True)
                    del spatial, images
                write_json(output/'progress.json', status)
        if args.mode == 'fit':
            write_json(output/'summary.json', summarize(records, reference))
        status.update(overall_status='COMPLETED', training_status='COMPLETED' if args.mode == 'fit' else 'SMOKE_ONLY')
        watch.check('publication')
        return 0
    except Exception as exc:
        status.update(overall_status='FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            write_json(output/'source_protection.json', registry.finish())
            watch.check('source protection')
        except Exception as exc:
            status.update(overall_status='FAILED', protection_error=str(exc))
            write_json(output/'run_status.json', status)
            raise
        status['elapsed_s'] = time.monotonic()-started
        write_json(output/'run_status.json', status)
        inventory = [{'path': p.relative_to(output).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size}
                     for p in sorted(output.rglob('*')) if p.is_file()]
        write_json(output/'artifact_manifest.json', {'config_sha256': CONFIG_SHA256, 'artifacts': inventory})


if __name__ == '__main__':
    sys.exit(main())
