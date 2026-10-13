"""Input identity, fold isolation and prediction identity for YOLO26 transfer."""
import json
from pathlib import Path

import numpy as np
import pytest

from core.yolo26_spatial_io import (
    decode_native_gray, fold_context, fresh_output, prediction_rows,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_contract():
    cfg = json.loads((ROOT/'configs/experiments/yolo26_spatial_v1.yaml').read_text())
    validate_contract(cfg)
    cfg['gray_training']['epochs'] = 1
    with pytest.raises(ValueError):
        validate_contract(cfg)


@pytest.mark.parametrize('key,value', [
    ('official_base', 'runs/train/yolo26_formal_001/gray_fusion/best.pt'),
    ('seeds', [2026]), ('planned_gray_fits', 3),
    ('source_config_sha256', '0'*64),
])
def test_unregistered_contract_refused(key, value):
    cfg = json.loads((ROOT/'configs/experiments/yolo26_spatial_v1.yaml').read_text())
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_contract(cfg)


def native_shard(tmp_path, *, dtype=np.uint8, extra=False, missing=False):
    rows = [{'sample_name': 'a', 'crop_window': [0, 0, 2, 3]},
            {'sample_name': 'b', 'crop_window': [1, 2, 5, 4]}]
    arrays = {}
    for i, row in enumerate(rows):
        _, _, x2, y2 = row['crop_window']
        x1, y1, _, _ = row['crop_window']
        for field in ('L', 'R', 'q', 'M', 'V', 'region', 'quality'):
            shape = (4,) if field == 'quality' else (y2-y1, x2-x1)
            arrays[row['sample_name']+'|'+field] = np.full(shape, 20+i*50, dtype)
    if extra:
        arrays['foreign|L'] = np.zeros((2,2), np.uint8)
    if missing:
        arrays.pop('a|L')
    path = tmp_path/'native.npz'
    np.savez(path, **arrays)
    return path, rows


def test_native_gray_keeps_identity_and_values(tmp_path):
    path, rows = native_shard(tmp_path)
    gray = decode_native_gray(path, rows)
    assert gray.shape == (2,224,224) and gray.dtype == np.uint8
    np.testing.assert_array_equal(gray[:,0,0], [20,70])
    np.testing.assert_array_equal(decode_native_gray(path, rows[::-1]), gray[::-1])


@pytest.mark.parametrize('options', [{'dtype':np.float32}, {'extra':True}, {'missing':True}])
def test_native_contract_refused(tmp_path, options):
    path, rows = native_shard(tmp_path, **options)
    with pytest.raises(ValueError):
        decode_native_gray(path, rows)


def test_native_shape_refused(tmp_path):
    path, rows = native_shard(tmp_path)
    rows[0]['crop_window'][2] += 1
    with pytest.raises(ValueError):
        decode_native_gray(path, rows)


def test_standardization_only_inner_train():
    rows = [{'group':str(i), 'fold_roles':{'fold0':role}} for i,role in enumerate(
        ('inner_train','inner_train','inner_val','outer_eval'))]
    g = np.arange(4*1280, dtype=np.float32).reshape(4,1280)
    f = {'c_features':np.ones((4,14),np.float32),
         's_features':np.ones((4,16),np.float32)}
    first = fold_context(g, f, rows, 'fold0')
    g[-1] = 1e8
    second = fold_context(g, f, rows, 'fold0')
    assert first['scaler'] == second['scaler']
    np.testing.assert_array_equal(first['context'][:3], second['context'][:3])
    assert first['context'].shape == (4,1310)


def test_missing_blocks_are_standardized_zero():
    rows = [{'group':str(i), 'fold_roles':{'fold0':role}} for i,role in enumerate(
        ('inner_train','inner_train','inner_val','outer_eval'))]
    g = np.ones((4,1280),np.float32)
    f = {'c_features':np.ones((4,14),np.float32),
         's_features':np.ones((4,16),np.float32)}
    f['s_features'][-1] = np.nan
    result = fold_context(g, f, rows, 'fold0')
    np.testing.assert_array_equal(result['context'][-1,-16:], np.zeros(16))


def test_group_overlap_refused():
    rows = [{'group':'same', 'fold_roles':{'fold0':role}} for role in
            ('inner_train','inner_val','outer_eval')]
    with pytest.raises(ValueError):
        fold_context(np.ones((3,1280),np.float32),
                     {'c_features':np.ones((3,14)), 's_features':np.ones((3,16))},
                     rows, 'fold0')


def test_fresh_output_rejects_existing_and_escape(tmp_path):
    parent = tmp_path/'evidence'
    parent.mkdir()
    path = fresh_output(parent, 'new')
    assert path == parent/'new'
    path.mkdir()
    with pytest.raises(ValueError):
        fresh_output(parent, 'new')
    with pytest.raises(ValueError):
        fresh_output(parent, '../other')


def test_predictions_scatter_identity_and_fallback():
    data = {'rows':[{'sample_name':str(i),'class_id':i,'group':'g'} for i in range(3)],
            'features':{'s_valid':np.array([True,False,True])}}
    baseline = np.eye(4, dtype=np.float32)[:3]
    logits = baseline[[2,1]]
    out = prediction_rows(data, [2,1], logits, baseline)
    assert [r['sample_name'] for r in out] == ['2','1']
    assert [r['used_fallback'] for r in out] == [False,True]
    assert [r['y0_prediction'] for r in out] == [2,1]
    y0 = prediction_rows(data, [2,1], logits, baseline, baseline=True)
    assert not any(r['used_fallback'] for r in y0)


def test_prediction_shape_refused():
    with pytest.raises(ValueError):
        prediction_rows({'rows':[], 'features':{'s_valid':np.zeros(0,bool)}},
                        [0], np.zeros((0,4)), np.zeros((0,4)))
