"""Decision examples for the frozen YOLO26 spatial comparison."""
from pathlib import Path

import pytest

from scripts.train_yolo26_spatial import configured_output, summarize


def records():
    reference, candidates = {}, {}
    for seed in (2026,2027,2028):
        for f in range(3):
            fold = f'fold{f}'
            truth = [{'sample_name':f'{fold}_{c}', 'class_id':c, 'group':fold,
                      'prediction':c} for c in range(4)]
            gray = [dict(r) for r in truth]
            gray[0]['prediction'] = 1
            reference[(seed,fold)] = gray
            for variant in ('B_stats','B_L','B_R','B_Q'):
                out = [dict(r) for r in truth]
                if variant != 'B_Q':
                    out[0]['prediction'] = 1
                if variant in ('B_stats','B_R'):
                    out[2]['prediction'] = 3
                candidates[(variant,seed,fold)] = out
    return candidates, reference


def test_known_improvement_passes_and_counts_pairs():
    candidates, reference = records()
    result = summarize(candidates, reference)
    assert result['variants']['B_Q']['mean'] == 1
    assert result['advance'] is True
    assert result['spatial_contribution']['passed'] is True
    assert result['spatial_contribution']['net_correct'] == 18
    assert result['paired']['Y0']['2026']['fixed'] == 3
    assert result['paired']['Y0']['2026']['broke'] == 0
    assert result['variants']['Y0']['per_seed']['2026']['pooled']['support'] == 12


def test_no_spatial_contribution_when_q_equals_statistics():
    candidates, reference = records()
    for seed in (2026,2027,2028):
        for fold in ('fold0','fold1','fold2'):
            candidates[('B_Q',seed,fold)] = candidates[('B_stats',seed,fold)]
    result = summarize(candidates, reference)
    assert result['spatial_contribution'] == {
        'mean_f1_delta':0., 'net_correct':0, 'passed':False}
    assert result['advance'] is False


def test_all_six_gates_cannot_advance_without_spatial_contribution():
    candidates, reference = records()
    for seed in (2026,2027,2028):
        for fold in ('fold0','fold1','fold2'):
            candidates[('B_stats',seed,fold)] = candidates[('B_Q',seed,fold)]
    result = summarize(candidates, reference)
    assert all(result['gates'].values())
    assert result['spatial_contribution']['passed'] is False
    assert result['advance'] is False


def test_missing_job_is_not_silently_ignored():
    candidates, reference = records()
    del candidates[('B_Q',2028,'fold2')]
    with pytest.raises((KeyError,ValueError)):
        summarize(candidates, reference)


def test_duplicate_identity_is_not_counted_twice():
    candidates, reference = records()
    candidates[('B_Q',2026,'fold0')].append(dict(candidates[('B_Q',2026,'fold0')][0]))
    with pytest.raises(ValueError):
        summarize(candidates, reference)


def test_output_has_to_be_inside_own_experiment(tmp_path):
    with pytest.raises(ValueError):
        configured_output(str(tmp_path/'elsewhere'))
    with pytest.raises(ValueError):
        configured_output('analysis/runs/yolo26_spatial_v1/../outside')
