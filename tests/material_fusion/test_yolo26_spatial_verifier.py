"""Independent YOLO26 replay must reject contaminated or altered evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.verify_yolo26_spatial import (
    check_history, check_prediction_rows, compare, independent_metrics,
    require_formal_status, summarize_records, verify_scaler, _verify_roles,
    _verify_manifest, ROOT,
)


def test_metrics_count_absent_classes_in_macro_average():
    metrics = independent_metrics([0, 0], [0, 1])
    assert metrics['macro_f1'] == pytest.approx(1 / 6)
    assert metrics['support'] == 2
    assert metrics['recall'] == [.5, 0., 0., 0.]


@pytest.mark.parametrize('truth,pred', [([], []), ([0], []), ([True], [0]),
                         ([4], [0]), ([0], [np.int64(0)])])
def test_metrics_refuse_bad_evidence(truth, pred):
    with pytest.raises(ValueError):
        independent_metrics(truth, pred)


def records_fixture():
    records, reference = {}, {}
    for seed in (2026, 2027, 2028):
        for fold in ('fold0', 'fold1', 'fold2'):
            rows = [dict(sample_name=f'{fold}_{k}', class_id=k,
                         group=f'{fold}_g{k}', prediction=k) for k in range(4)]
            reference[(seed, fold)] = copy.deepcopy(rows)
            for variant in ('B_stats', 'B_L', 'B_R', 'B_Q'):
                records[(variant, seed, fold)] = copy.deepcopy(rows)
    return records, reference


def test_seed_specific_yolo_reference_not_repeated_old_p0():
    records, reference = records_fixture()
    for fold in ('fold0', 'fold1', 'fold2'):
        reference[(2026, fold)][0]['prediction'] = 1
    summary = summarize_records(records, reference)
    assert summary['variants']['Y0']['per_seed']['2026']['fold_mean'] < 1
    assert summary['variants']['Y0']['per_seed']['2027']['fold_mean'] == 1
    assert summary['paired']['Y0']['2026']['fixed'] == 3
    assert summary['paired']['Y0']['2027']['fixed'] == 0
    assert summary['spatial_contribution']['passed'] is False


def test_all_gates_and_spatial_contribution_pass_with_distributed_corrections():
    records, reference = records_fixture()
    for seed in (2026, 2027, 2028):
        for fold in ('fold0', 'fold1', 'fold2'):
            reference[(seed, fold)][0]['prediction'] = 1
            for variant in ('B_stats', 'B_L', 'B_R'):
                records[(variant, seed, fold)][0]['prediction'] = 1
    summary = summarize_records(records, reference)
    assert all(summary['gates'].values())
    assert summary['advance'] is True
    assert summary['spatial_contribution']['passed'] is True
    assert summary['spatial_contribution']['net_correct'] == 9
    assert summary['variants']['B_Q']['per_seed']['2026']['pooled']['support'] == 12


def test_six_gain_gates_pass_but_no_q_spatial_contribution_blocks_advancement():
    records, reference = records_fixture()
    for seed in (2026, 2027, 2028):
        for fold in ('fold0', 'fold1', 'fold2'):
            reference[(seed, fold)][0]['prediction'] = 1
            for variant in ('B_L', 'B_R'):
                records[(variant, seed, fold)][0]['prediction'] = 1
    summary = summarize_records(records, reference)
    assert all(summary['gates'].values())
    assert summary['spatial_contribution']['passed'] is False
    assert summary['advance'] is False
    assert summary['overall_gain_passed'] is True


def test_duplicated_oof_identity_is_rejected():
    records, reference = records_fixture()
    reference[(2026, 'fold1')][0]['sample_name'] = 'fold0_0'
    with pytest.raises(ValueError, match='identity'):
        summarize_records(records, reference)


def test_pairing_checks_truth_and_group_not_only_sample_name():
    records, reference = records_fixture()
    records[('B_Q', 2026, 'fold0')][0]['class_id'] = 1
    with pytest.raises(ValueError, match='identity'):
        summarize_records(records, reference)


def prediction_fixture():
    logits = np.array([[2., 1., 0., 0.], [0., 0., 2., 0.]], np.float32)
    data = {'rows': [dict(sample_name='a', group='g1', class_id=0),
                     dict(sample_name='b', group='g2', class_id=2)],
            'features': {'s_valid': np.array([True, False])}}
    rows = [dict(**data['rows'][i], logits=logits[i].tolist(), prediction=int(logits[i].argmax()),
                 y0_prediction=int(logits[i].argmax()), used_fallback=(i == 1)) for i in range(2)]
    return rows, logits, data


def test_prediction_replay_and_exact_fallback():
    rows, logits, data = prediction_fixture()
    check_prediction_rows(rows, logits, data, [0, 1], logits)
    rows[1]['logits'][0] = 1e-7
    with pytest.raises(ValueError, match='exact'):
        check_prediction_rows(rows, logits, data, [0, 1], logits)


@pytest.mark.parametrize('mutation', ['identity', 'probability', 'fallback', 'y0'])
def test_prediction_rejects_tampering(mutation):
    rows, logits, data = prediction_fixture()
    if mutation == 'identity':
        rows.reverse()
    elif mutation == 'probability':
        rows[0]['logits'] = [float('nan')] * 4
    elif mutation == 'fallback':
        rows[1]['used_fallback'] = False
    else:
        rows[0]['y0_prediction'] = 3
    with pytest.raises(ValueError):
        check_prediction_rows(rows, logits, data, [0, 1], logits)


def test_baseline_missing_row_is_never_reported_as_fallback():
    rows, logits, data = prediction_fixture()
    for row in rows:
        row['used_fallback'] = False
    check_prediction_rows(rows, logits, data, [0, 1], logits, baseline=True)


def test_scaler_rebuilt_from_inner_train_with_missing_and_zero_variance():
    raw = np.array([[1., np.nan, 4.], [3., 2., 4.], [100., 8., 8.]])
    scaler = dict(mean=[2., 2., 4.], std=[1., 0., 0.], usable=[True, False, False])
    context = verify_scaler(scaler, raw, [0, 1])
    assert context.tolist() == [[-1., 0., 0.], [1., 0., 0.], [98., 0., 0.]]
    scaler['mean'][0] = float(raw[:, 0].mean())
    with pytest.raises(ValueError, match='scaler'):
        verify_scaler(scaler, raw, [0, 1])


def test_history_uses_accuracy_then_earliest_epoch():
    history = [dict(epoch=1, macro_f1=.5, accuracy=.6, train_used=4),
               dict(epoch=2, macro_f1=.5, accuracy=.7, train_used=4),
               dict(epoch=3, macro_f1=.5, accuracy=.7, train_used=4)]
    result = dict(selected_epoch=2, completed_epochs=3, smoke=False)
    assert check_history(history, result, train_used=4, max_epochs=3)['epoch'] == 2
    result['selected_epoch'] = 3
    with pytest.raises(ValueError, match='selection'):
        check_history(history, result, train_used=4, max_epochs=3)


def test_history_requires_contiguous_epochs_and_no_missing_training():
    result = dict(selected_epoch=1, completed_epochs=1, smoke=False)
    with pytest.raises(ValueError, match='epoch'):
        check_history([dict(epoch=2, macro_f1=.5, accuracy=.5, train_used=3)], result,
                      train_used=3, max_epochs=60)
    with pytest.raises(ValueError, match='loss'):
        check_history([dict(epoch=1, macro_f1=.5, accuracy=.5, train_used=4)], result,
                      train_used=3, max_epochs=60)


def test_formal_status_rejects_smoke_and_incomplete_counts():
    status = dict(mode='fit', overall_status='COMPLETED', completed_fits=45,
                  gray_fits=9, spatial_fits=36, config_sha256='hash', training_status='COMPLETED')
    require_formal_status(status, 'hash')
    for field, value in [('mode', 'smoke'), ('completed_fits', 36), ('gray_fits', 8),
                         ('config_sha256', 'changed'), ('training_status', 'SMOKE_ONLY')]:
        bad = dict(status, **{field: value})
        with pytest.raises(ValueError):
            require_formal_status(bad, 'hash')


def test_compare_refuses_nan_and_bool_as_integer():
    with pytest.raises(ValueError):
        compare(float('nan'), 0.)
    with pytest.raises(ValueError):
        compare(True, 1)


def test_role_provenance_has_exact_train_val_outer_ids_and_groups():
    rows = [dict(group='g0'), dict(group='g1'), dict(group='g2')]
    parts = dict(inner_train=[0], inner_val=[1], outer_eval=[2])
    result = dict(fold='fold0', seed=2026, train_ids=[0], val_ids=[1], outer_ids=[2],
                  train_groups=['g0'], val_groups=['g1'], outer_groups=['g2'])
    _verify_roles(result, parts, rows, 'fold0', 2026)
    result['train_ids'] = [0, 2]
    with pytest.raises(ValueError):
        _verify_roles(result, parts, rows, 'fold0', 2026)


@pytest.mark.parametrize('field,value', [('macro_f1', float('nan')), ('accuracy', 1.1)])
def test_nonfinite_or_impossible_history_metrics_refused(field, value):
    row = dict(epoch=1, macro_f1=.5, accuracy=.5, train_used=3)
    row[field] = value
    with pytest.raises(ValueError, match='metrics'):
        check_history([row], dict(selected_epoch=1, completed_epochs=1, smoke=False),
                      train_used=3, max_epochs=60)


@pytest.mark.parametrize('kind', ['traversal', 'duplicate', 'bytes', 'missing'])
def test_artifact_manifest_refuses_unsafe_or_incomplete_evidence(tmp_path, kind):
    fit = tmp_path / 'fit'
    fit.mkdir()
    (fit / 'result.json').write_text('{}', encoding='utf-8')
    record = dict(path='result.json', sha256='fake', bytes=2)
    records = [record]
    if kind == 'traversal':
        record['path'] = '../result.json'
    elif kind == 'duplicate':
        records.append(dict(record))
    elif kind == 'bytes':
        record['bytes'] = 3
    (fit / 'artifact_manifest.json').write_text(json.dumps({'artifacts': records}), encoding='utf-8')
    class Registry:
        def check(self, rel, digest):
            return ROOT / rel
    with pytest.raises(ValueError):
        _verify_manifest(fit, Registry(), 'fake', lambda: None)


@pytest.mark.parametrize('actual,expected', [({'a': 1}, {'b': 1}), ([1], [1, 2]),
                                           ([1], [2]), (1., 2.), ('a', 'b')])
def test_recursive_comparison_detects_changed_metadata(actual, expected):
    with pytest.raises(ValueError):
        compare(actual, expected)


def test_compare_accepts_complete_nested_evidence():
    compare({'a': [1, .2, False], 'b': {}}, {'a': [1, .2, False], 'b': {}})


@pytest.mark.parametrize('epochs', [1, 12])
def test_formal_spatial_history_refuses_unearned_or_late_early_stopping(epochs):
    history = [dict(epoch=i, macro_f1=.5, accuracy=.5, train_used=3)
               for i in range(1, epochs + 1)]
    result = dict(selected_epoch=1, completed_epochs=epochs, smoke=False)
    with pytest.raises(ValueError, match='patience'):
        check_history(history, result, train_used=3, max_epochs=60)


def test_formal_spatial_history_stops_at_first_ten_stale_epochs():
    history = [dict(epoch=i, macro_f1=.5, accuracy=.5, train_used=3)
               for i in range(1, 12)]
    result = dict(selected_epoch=1, completed_epochs=11, smoke=False)
    assert check_history(history, result, train_used=3, max_epochs=60)['epoch'] == 1


def test_spatial_staleness_resets_when_accuracy_improves():
    history = [dict(epoch=i, macro_f1=.5, accuracy=.5 if i < 11 else .6, train_used=3)
               for i in range(1, 22)]
    result = dict(selected_epoch=11, completed_epochs=21, smoke=False)
    assert check_history(history, result, train_used=3, max_epochs=60)['epoch'] == 11


def test_selected_backbone_changes_are_measured_against_actual_official_state():
    import torch
    from torch import nn
    from scripts.verify_yolo26_spatial import compare_backbone_to_official
    class Head(nn.Module):
        def __init__(self, classes):
            super().__init__()
            self.linear = nn.Linear(2, classes)
    def model(classes):
        return nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2), Head(classes))
    official, selected = model(1000), model(4)
    for name, parameter in selected.named_parameters():
        if '.linear.' not in name:
            parameter.data.copy_(dict(official.named_parameters())[name])
    for name, buffer in selected.named_buffers():
        buffer.copy_(dict(official.named_buffers())[name])
    with pytest.raises(ValueError, match='backbone'):
        compare_backbone_to_official(selected, official)
    with torch.no_grad():
        selected[0].weight.add_(.01)
    with pytest.raises(ValueError, match='BN'):
        compare_backbone_to_official(selected, official)
    selected[1].running_mean.add_(.1)
    evidence = compare_backbone_to_official(selected, official)
    assert evidence['actual_backbone_changed_count'] == 1
    assert evidence['actual_bn_buffers_changed_count'] == 1
    assert evidence['actual_early_backbone_changed'] is True
