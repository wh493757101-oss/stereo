"""Fourth-batch preflight, bounded smoke and explicitly authorized fit."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from core.material_residual_scale import DeadlineWatch
from core.material_spatial_classifier import (
    SEEDS, VARIANTS, assemble_planes, classification_metrics, fit_head,
    predict_with_fallback,
)
from core.material_spatial_classifier_io import (
    FROZEN_CONFIG_SHA256, InputRegistry, digest, fold_inputs, load_inputs, permuted_eval,
    read_json, write_json,
)


def prediction_rows(data: dict, ids: list[int], logits: np.ndarray,
                    p0: np.ndarray) -> list[dict]:
    result = []
    for j, i in enumerate(ids):
        row = data['rows'][i]
        result.append({k: row[k] for k in ('sample_name', 'class_id', 'group')})
        result[-1].update(logits=logits[j].tolist(), prediction=int(logits[j].argmax()),
                          p0_prediction=int(p0[i].argmax()),
                          used_fallback=not bool(data['features']['s_valid'][i]))
    return result


def save_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


def summarize(records: dict, reference: dict) -> dict:
    summary = {'variants': {}, 'paired': {}, 'gates': {}}
    for v in ('P0',) + VARIANTS:
        per_seed = {}
        for seed in SEEDS:
            fold_metrics, all_rows = [], []
            for fold in ('fold0','fold1','fold2'):
                rows = reference[fold] if v == 'P0' else records[(v, seed, fold)]
                fold_metrics.append(classification_metrics(
                    [r['class_id'] for r in rows], [r['prediction'] for r in rows]))
                all_rows += rows
            pooled = classification_metrics([r['class_id'] for r in all_rows],
                                            [r['prediction'] for r in all_rows])
            per_seed[str(seed)] = {'folds':fold_metrics, 'pooled':pooled,
                                   'fold_mean':float(np.mean([m['macro_f1'] for m in fold_metrics]))}
        summary['variants'][v] = {'per_seed':per_seed,
                                  'mean':float(np.mean([x['fold_mean'] for x in per_seed.values()]))}
    for base in ('P0','B_L','B_R'):
        per_seed = {}
        for seed in SEEDS:
            fixed, broke, group_net = 0, 0, {}
            for fold in ('fold0','fold1','fold2'):
                left = reference[fold] if base == 'P0' else records[(base,seed,fold)]
                right = records[('B_Q',seed,fold)]
                lm = {r['sample_name']:r for r in left}
                for r in right:
                    b = lm[r['sample_name']]
                    a = int(r['prediction']==r['class_id']) - int(b['prediction']==b['class_id'])
                    fixed += int(a==1)
                    broke += int(a==-1)
                    group_net[r['group']] = group_net.get(r['group'],0)+a
            per_seed[str(seed)] = {'fixed':fixed,'broke':broke,'group_net':group_net}
        summary['paired'][base] = per_seed
    q = summary['variants']['B_Q']
    delta = {b: q['mean']-summary['variants'][b]['mean'] for b in ('P0','B_L','B_R')}
    summary['gates']['gate1'] = delta['P0'] >= .01 and delta['B_L'] >= .005
    summary['gates']['gate2'] = delta['B_R'] > 0 and sum(x['fixed']-x['broke'] for x in summary['paired']['B_R'].values()) > 0
    summary['gates']['gate3'] = sum(all(summary['paired'][b][str(s)]['fixed'] > summary['paired'][b][str(s)]['broke'] for b in ('P0','B_L')) for s in SEEDS) >= 2
    fold_deltas = []
    recall_deltas = []
    for b in ('P0','B_L'):
        for f in range(3):
            fold_deltas.append(float(np.mean([q['per_seed'][str(s)]['folds'][f]['macro_f1']-summary['variants'][b]['per_seed'][str(s)]['folds'][f]['macro_f1'] for s in SEEDS])))
        recall_deltas += np.mean([np.array(q['per_seed'][str(s)]['pooled']['recall'])-summary['variants'][b]['per_seed'][str(s)]['pooled']['recall'] for s in SEEDS],axis=0).tolist()
    summary['gates']['gate4'] = sum(fold_deltas[f]>=0 and fold_deltas[f+3]>=0 for f in range(3)) >= 2 and min(fold_deltas) >= -.01
    summary['gates']['gate5'] = min(recall_deltas) >= -.01
    groups = set().union(*(x['group_net'] for x in summary['paired']['P0'].values()))
    summary['gates']['gate6'] = sum(sum(summary['paired']['P0'][str(s)]['group_net'].get(g,0) for s in SEEDS)>0 for g in groups) >= 2
    summary['advance'] = all(summary['gates'].values())
    summary['note'] = 'Exploratory existing-data comparison; no independent blind test or material causality claim.'
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=('preflight','smoke','fit'), required=True)
    parser.add_argument('--budget-minutes', type=float, default=480.)
    parser.add_argument('--deadline-utc')
    args = parser.parse_args(argv)
    output = ROOT/args.output_dir
    if output.exists() or '..' in Path(args.output_dir).parts or not output.resolve().is_relative_to(ROOT/'analysis/runs/material_spatial_classifier_v1'):
        raise ValueError('fresh project experiment output required')
    output.mkdir(parents=True)
    watch = DeadlineWatch(deadline_utc=args.deadline_utc) if args.deadline_utc else DeadlineWatch(budget_minutes=args.budget_minutes)
    status = {'mode':args.mode, 'training_status':'NOT_RUN', 'overall_status':'STARTED'}
    registry = InputRegistry()
    started = time.monotonic()
    try:
        config_path = ROOT/args.config
        config_hash = digest(config_path)
        if config_hash != FROZEN_CONFIG_SHA256:
            raise ValueError('fourth-batch frozen config identity mismatch')
        registry.check(args.config, config_hash)
        for rel in ('core/material_spatial_classifier.py', 'core/material_spatial_classifier_io.py',
                    'scripts/train_material_spatial_classifier.py', 'scripts/verify_material_spatial_classifier.py'):
            registry.check(rel, digest(ROOT/rel))
        cfg = read_json(config_path)
        write_json(output/'config_snapshot.json', cfg)
        status['config_sha256'] = config_hash
        from datetime import datetime, timezone
        status['deadline_utc'] = datetime.fromtimestamp(watch._deadline, timezone.utc).isoformat()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.mode != 'preflight' and not torch.cuda.is_available():
            raise RuntimeError('frozen CUDA device unavailable')
        data = load_inputs(cfg, registry, load_grid=args.mode!='preflight')
        watch.check('input admission')
        write_json(output/'admission.json', {'samples':2446,'available':2341,'groups':21,'config_sha256':config_hash})
        if args.mode == 'preflight':
            for fold in ('fold0','fold1','fold2'):
                wired = fold_inputs(data,fold,cfg)
                write_json(output/f'{fold}_preflight.json', {'p0_max_error':wired['p0_max_error'], 'counts':{k:len(v) for k,v in wired['parts'].items()}})
            status['overall_status']='COMPLETED'
            return 0
        records, reference = {}, {}
        folds = ('fold0',) if args.mode=='smoke' else ('fold0','fold1','fold2')
        seeds = (2026,) if args.mode=='smoke' else SEEDS
        for fold in folds:
            watch.check('fold start')
            wired = fold_inputs(data,fold,cfg)
            ids_eval = wired['parts']['outer_eval']
            if args.mode == 'smoke':
                ids_eval = ids_eval[:16]
            reference[fold] = prediction_rows(data,ids_eval,wired['p0'][ids_eval],wired['p0'])
            ref_dir=output/'P0'/fold
            ref_dir.mkdir(parents=True)
            save_rows(ref_dir/'predictions.jsonl', reference[fold])
            write_json(ref_dir/'scaler.json', wired['scaler'])
            perm_grid = permuted_eval(data,ids_eval,registry)
            for variant in VARIANTS:
                images = assemble_planes(data['grid'], variant)
                def subset(role: str) -> dict:
                    ids = wired['parts'][role]
                    if args.mode=='smoke':
                        ids=ids[:32]
                    return {'images':images[ids], 'context':wired['context'][ids],
                            'available':data['features']['s_valid'][ids],
                            'p0':wired['p0'][ids], 'labels':data['features']['class_id'][ids]}
                train, val = subset('inner_train'), subset('inner_val')
                for seed in seeds:
                    job=output/'models'/f'{fold}_{variant}_s{seed}'
                    job.mkdir(parents=True)
                    fit_started=time.monotonic()
                    def check() -> None:
                        watch.check('fit or evaluation')
                        if time.monotonic()-fit_started > 600:
                            raise TimeoutError('single fit exceeded 10 minutes')
                    def epoch_log(record: dict) -> None:
                        with (job/'history.jsonl').open('a',encoding='utf-8') as handle:
                            handle.write(json.dumps(record,allow_nan=False)+'\n')
                    print(f'FIT {fold}/{variant}/{seed}', flush=True)
                    model,history,best=fit_head(train,val,seed=seed,device='cuda:0',
                        epochs=2 if args.mode=='smoke' else 60, check=check,epoch_callback=epoch_log)
                    check()
                    logits=predict_with_fallback(model,images[ids_eval],wired['context'][ids_eval],
                        data['features']['s_valid'][ids_eval],wired['p0'][ids_eval],'cuda:0',32)
                    check()
                    pred_rows=prediction_rows(data,ids_eval,logits,wired['p0'])
                    save_rows(job/'predictions.jsonl',pred_rows)
                    result={'fold':fold,'variant':variant,'seed':seed,'selected_epoch':best['epoch'],
                            'completed_epochs':len(history),'elapsed_s':time.monotonic()-fit_started,
                            'metrics':classification_metrics([r['class_id'] for r in pred_rows], [r['prediction'] for r in pred_rows])}
                    if variant=='B_Q':
                        pimages=assemble_planes(perm_grid,variant)
                        plogits=predict_with_fallback(model,pimages,wired['context'][ids_eval],
                            data['features']['s_valid'][ids_eval],wired['p0'][ids_eval],'cuda:0',32)
                        save_rows(job/'permuted_predictions.jsonl',prediction_rows(data,ids_eval,plogits,wired['p0']))
                    torch.save({'state_dict':model.cpu().state_dict(), 'fold':fold,'variant':variant,
                                'seed':seed,'selected_epoch':best['epoch'],'config_sha256':config_hash}, job/'model.pt')
                    write_json(job/'result.json',result)
                    records[(variant,seed,fold)] = pred_rows
                    print(f'DONE epoch={best["epoch"]} outer_f1={result["metrics"]["macro_f1"]:.6f} seconds={result["elapsed_s"]:.1f}',flush=True)
                    del model
                del images,train,val
        if args.mode=='fit':
            write_json(output/'summary.json',summarize(records,reference))
        status.update(overall_status='COMPLETED', training_status='COMPLETED' if args.mode=='fit' else 'SMOKE_ONLY',
                      completed_fits=len(records),elapsed_s=time.monotonic()-started)
        watch.check('publication')
        return 0
    except Exception as exc:
        status.update(overall_status='FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            write_json(output/'source_protection.json',registry.finish())
            watch.check('final source protection')
        except Exception as exc:
            status.update(overall_status='FAILED', protection_error=str(exc))
            write_json(output/'run_status.json',status)
            raise
        status['elapsed_s']=time.monotonic()-started
        write_json(output/'run_status.json',status)
        inventory=[]
        for p in sorted(output.rglob('*')):
            if p.is_file():
                inventory.append({'path':p.relative_to(output).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size})
        write_json(output/'artifact_manifest.json',{'config_sha256':status.get('config_sha256'),'artifacts':inventory})


if __name__=='__main__':
    sys.exit(main())
