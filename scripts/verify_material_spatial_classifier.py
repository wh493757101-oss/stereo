"""Independent fourth-batch counting, selection, fallback and weight replay."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from core.material_residual_scale import DeadlineWatch
from core.material_spatial_classifier import (
    SpatialClassifier, SEEDS, VARIANTS, assemble_planes, predict_with_fallback,
)
from core.material_spatial_classifier_io import (
    FROZEN_CONFIG_SHA256, InputRegistry, digest, read_json, write_json, load_inputs, fold_inputs, permuted_eval,
)


def independent_metrics(truth: list[int], pred: list[int]) -> dict:
    if len(truth) != len(pred) or not truth:
        raise ValueError('unaligned or empty predictions')
    cm = [[0]*4 for _ in range(4)]
    for a,b in zip(truth,pred):
        if type(a) is not int or type(b) is not int or not 0<=a<4 or not 0<=b<4:
            raise ValueError('invalid class type/range')
        cm[a][b]+=1
    f1, recall = [], []
    for k in range(4):
        tp=cm[k][k]
        actual=sum(cm[k])
        predicted=sum(row[k] for row in cm)
        f1.append(2*tp/(actual+predicted) if actual+predicted else 0.)
        recall.append(tp/actual if actual else 0.)
    return {'accuracy':sum(cm[k][k] for k in range(4))/len(truth),
            'macro_f1':sum(f1)/4,'recall':recall,'confusion_matrix':cm,'support':len(truth)}


def independently_summarize(records: dict, reference: dict) -> dict:
    variants={}
    for v in ('P0',)+VARIANTS:
        ps={}
        for seed in SEEDS:
            fm=[]
            all_rows=[]
            for fold in ('fold0','fold1','fold2'):
                rows=reference[fold] if v=='P0' else records[(v,seed,fold)]
                fm.append(independent_metrics([r['class_id'] for r in rows],[r['prediction'] for r in rows]))
                all_rows.extend(rows)
            ps[str(seed)]={'folds':fm,'pooled':independent_metrics([r['class_id'] for r in all_rows],[r['prediction'] for r in all_rows]),
                           'fold_mean':sum(m['macro_f1'] for m in fm)/3}
        variants[v]={'per_seed':ps,'mean':sum(p['fold_mean'] for p in ps.values())/3}
    paired={}
    for base in ('P0','B_L','B_R'):
        ps={}
        for seed in SEEDS:
            fixed=0
            broke=0
            nets={}
            for fold in ('fold0','fold1','fold2'):
                a=reference[fold] if base=='P0' else records[(base,seed,fold)]
                lookup={r['sample_name']:r for r in a}
                for r in records[('B_Q',seed,fold)]:
                    b=lookup[r['sample_name']]
                    right=r['prediction']==r['class_id']
                    left=b['prediction']==b['class_id']
                    fixed+=int(right and not left)
                    broke+=int(left and not right)
                    nets[r['group']]=nets.get(r['group'],0)+int(right)-int(left)
            ps[str(seed)]={'fixed':fixed,'broke':broke,'group_net':nets}
        paired[base]=ps
    q=variants['B_Q']
    gates={}
    gates['gate1']=q['mean']-variants['P0']['mean']>=.01 and q['mean']-variants['B_L']['mean']>=.005
    gates['gate2']=q['mean']>variants['B_R']['mean'] and sum(p['fixed']-p['broke'] for p in paired['B_R'].values())>0
    gates['gate3']=sum(all(paired[b][str(s)]['fixed']>paired[b][str(s)]['broke'] for b in ('P0','B_L')) for s in SEEDS)>=2
    diffs=[[sum(q['per_seed'][str(s)]['folds'][f]['macro_f1']-variants[b]['per_seed'][str(s)]['folds'][f]['macro_f1'] for s in SEEDS)/3 for f in range(3)] for b in ('P0','B_L')]
    gates['gate4']=sum(diffs[0][f]>=0 and diffs[1][f]>=0 for f in range(3))>=2 and min(min(x) for x in diffs)>=-.01
    gates['gate5']=all(sum(q['per_seed'][str(s)]['pooled']['recall'][k]-variants[b]['per_seed'][str(s)]['pooled']['recall'][k] for s in SEEDS)/3>=-.01 for b in ('P0','B_L') for k in range(4))
    groups=set().union(*(p['group_net'] for p in paired['P0'].values()))
    gates['gate6']=sum(sum(paired['P0'][str(s)]['group_net'].get(g,0) for s in SEEDS)>0 for g in groups)>=2
    return {'variants':variants,'paired':paired,'gates':gates,'advance':all(gates.values()),
            'note':'Exploratory existing-data comparison; no independent blind test or material causality claim.'}


def compare(actual, expected, path: str = '') -> None:
    if type(actual) is not type(expected):
        raise ValueError(f'type mismatch: {path}')
    if isinstance(expected, dict):
        if actual.keys()!=expected.keys():
            raise ValueError(f'key mismatch: {path}')
        for k in expected:
            compare(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):
            raise ValueError(f'length mismatch: {path}')
        for i,(a,b) in enumerate(zip(actual,expected)):
            compare(a,b,f'{path}[{i}]')
    elif isinstance(expected,float):
        if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual-expected)>1e-6:
            raise ValueError(f'numeric mismatch: {path}')
    elif actual!=expected:
        raise ValueError(f'value mismatch: {path}')


def check_rows(saved: list[dict], logits: np.ndarray, data: dict, ids: list[int], p0: np.ndarray) -> None:
    if len(saved)!=len(ids) or len({r['sample_name'] for r in saved})!=len(ids):
        raise ValueError('prediction coverage mismatch')
    for offset,(row,i) in enumerate(zip(saved,ids)):
        for k in ('sample_name','group','class_id'):
            compare(row[k],data['rows'][i][k],k)
        if type(row['prediction']) is not int or row['prediction']!=int(logits[offset].argmax()):
            raise ValueError('class prediction mismatch')
        if type(row['p0_prediction']) is not int or row['p0_prediction']!=int(p0[i].argmax()):
            raise ValueError('P0 reference prediction metadata mismatch')
        if not np.isfinite(row['logits']).all() or not np.allclose(row['logits'],logits[offset],atol=1e-6,rtol=0):
            raise ValueError('weight replay logits mismatch')
        fallback=not bool(data['features']['s_valid'][i])
        if type(row['used_fallback']) is not bool or row['used_fallback']!=fallback:
            raise ValueError('fallback mismatch')
        if fallback and not np.array_equal(np.array(row['logits'],np.float32),p0[i]):
            raise ValueError('fallback is not exact original P0')


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--fit-dir',required=True)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--deadline-utc',required=True)
    args=p.parse_args(argv)
    output=ROOT/args.output_dir
    if output.exists() or '..' in Path(args.output_dir).parts or not output.resolve().is_relative_to(ROOT/'analysis/runs/material_spatial_classifier_v1'):
        raise ValueError('fresh verification output required')
    output.mkdir(parents=True)
    report={'status':'STARTED','problem_count':0}
    reg=InputRegistry()
    watch=DeadlineWatch(deadline_utc=args.deadline_utc)
    try:
        cfg=read_json(reg.check(args.config,FROZEN_CONFIG_SHA256))
        fit=ROOT/args.fit_dir
        manifest=read_json(reg.check((fit/'artifact_manifest.json').relative_to(ROOT).as_posix(),args.expected_manifest_sha256))
        for record in manifest['artifacts']:
            watch.check('artifact identity')
            if type(record['bytes']) is not int:
                raise ValueError('invalid artifact bytes')
            path=reg.check((fit/record['path']).relative_to(ROOT).as_posix(),record['sha256'])
            if path.stat().st_size!=record['bytes']:
                raise ValueError('artifact bytes mismatch')
        status=read_json(fit/'run_status.json')
        if status['mode']!='fit' or status['overall_status']!='COMPLETED' or status['completed_fits']!=36:
            raise ValueError('fit not formally complete')
        if status['config_sha256']!=digest(ROOT/args.config):
            raise ValueError('config changed after training')
        compare(read_json(fit/'config_snapshot.json'),cfg,'config')
        frozen_inputs=read_json(fit/'source_protection.json')
        if frozen_inputs['status']!='IDENTICAL':
            raise ValueError('training source protection failed')
        for source in frozen_inputs['files']:
            reg.check(source['path'],source['sha256'])
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        data=load_inputs(cfg,reg)
        records,reference={},{}
        for fold in ('fold0','fold1','fold2'):
            wired=fold_inputs(data,fold,cfg)
            ev=wired['parts']['outer_eval']
            va=wired['parts']['inner_val']
            scaler=read_json(fit/'P0'/fold/'scaler.json')
            raw=np.concatenate([data['features'][k] for k in ('g_features','c_features','s_features')],axis=1)
            training=raw[wired['parts']['inner_train']].astype(np.float64)
            for k in range(286):
                finite=training[:,k][np.isfinite(training[:,k])]
                mean=float(finite.mean()) if len(finite) else 0.
                std=float(finite.std(ddof=0)) if len(finite) else 0.
                if abs(scaler['mean'][k]-mean)>1e-6 or abs(scaler['std'][k]-std)>1e-6 or scaler['usable'][k]!=(std>0):
                    raise ValueError('independent scaler mismatch')
            reference[fold]=[json.loads(x) for x in (fit/'P0'/fold/'predictions.jsonl').read_text().splitlines()]
            check_rows(reference[fold],wired['p0'][ev],data,ev,wired['p0'])
            perm=permuted_eval(data,ev,reg)
            for variant in VARIANTS:
                images=assemble_planes(data['grid'],variant)
                for seed in SEEDS:
                    watch.check('model replay')
                    job=fit/'models'/f'{fold}_{variant}_s{seed}'
                    result=read_json(job/'result.json')
                    history=[json.loads(x) for x in (job/'history.jsonl').read_text().splitlines()]
                    if [r['epoch'] for r in history]!=list(range(1,len(history)+1)) or not 1<=len(history)<=60:
                        raise ValueError('invalid epoch history')
                    selected=sorted(history,key=lambda r:(-r['macro_f1'],-r['accuracy'],r['epoch']))[0]
                    if result['selected_epoch']!=selected['epoch'] or result['completed_epochs']!=len(history):
                        raise ValueError('selection mismatch')
                    checkpoint=torch.load(job/'model.pt',map_location='cpu',weights_only=False)
                    for k,expected in [('fold',fold),('variant',variant),('seed',seed),('selected_epoch',selected['epoch']),('config_sha256',status['config_sha256'])]:
                        compare(checkpoint[k],expected,k)
                    model=SpatialClassifier().to('cuda:0')
                    model.load_state_dict(checkpoint['state_dict'])
                    def replay(ids, chosen):
                        return predict_with_fallback(model,chosen,wired['context'][ids],data['features']['s_valid'][ids],wired['p0'][ids],'cuda:0',32)
                    logits=replay(ev,images[ev])
                    saved=[json.loads(x) for x in (job/'predictions.jsonl').read_text().splitlines()]
                    check_rows(saved,logits,data,ev,wired['p0'])
                    compare(result['metrics'],independent_metrics([r['class_id'] for r in saved],[r['prediction'] for r in saved]),'result metrics')
                    vlogits=replay(va,images[va])
                    vm=independent_metrics(data['features']['class_id'][va].tolist(),vlogits.argmax(1).tolist())
                    for k in ('macro_f1','accuracy','confusion_matrix','recall','support'):
                        compare(selected[k],vm[k],f'selected val {k}')
                    expected_train=int(data['features']['s_valid'][wired['parts']['inner_train']].sum())
                    if any(type(r['train_used']) is not int or r['train_used']!=expected_train for r in history):
                        raise ValueError('missing rows entered loss')
                    if variant=='B_Q':
                        saved_perm=[json.loads(x) for x in (job/'permuted_predictions.jsonl').read_text().splitlines()]
                        check_rows(saved_perm,replay(ev,assemble_planes(perm,variant)),data,ev,wired['p0'])
                    records[(variant,seed,fold)]=saved
                    del model
                del images
        rebuilt=independently_summarize(records,reference)
        compare(read_json(fit/'summary.json'),rebuilt,'summary')
        write_json(output/'independent_summary.json',rebuilt)
        protection=reg.finish()
        write_json(output/'source_protection.json',protection)
        watch.check('verification publication')
        report.update(status='PASSED',models=36,unique_samples=2446,folds=3,gates=rebuilt['gates'],advance=rebuilt['advance'])
        return 0
    except Exception as exc:
        report.update(status='FAILED',problem_count=1,error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        write_json(output/'verification_report.json',report)


if __name__=='__main__':
    sys.exit(main())
