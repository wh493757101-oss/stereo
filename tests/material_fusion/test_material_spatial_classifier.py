"""Independent examples for the fourth-batch information barriers."""
import inspect

import numpy as np
import pytest
import torch

from core.material_spatial_classifier import (
    SpatialClassifier, assemble_planes, predict_with_fallback, fit_head,
    classification_metrics, require_partition, select_best,
)


@pytest.fixture
def grid():
    x = np.zeros((2, 9, 128, 128), np.float32)
    for i in range(6):
        x[:, i, 20:100, 20:100] = (i + 1) / 10
    x[:, 6:9, 20:100, 20:100] = 1
    return x


@pytest.mark.parametrize('variant,source', [('B_L', 0), ('B_R', 2), ('B_Q', 4)])
def test_channel_mapping(grid, variant, source):
    out = assemble_planes(grid, variant)
    assert out.shape == (2, 7, 128, 128)
    np.testing.assert_array_equal(out[:, :2], grid[:, :2])
    np.testing.assert_array_equal(out[:, 2:4], grid[:, source:source+2])
    np.testing.assert_array_equal(out[:, 4:], grid[:, 6:])


def test_statistics_control_cannot_read_response(grid):
    a = assemble_planes(grid, 'B_stats')
    grid[:, :6] *= -1
    np.testing.assert_array_equal(a, assemble_planes(grid, 'B_stats'))
    assert not a[:, :4].any()


def test_left_control_cannot_read_right_or_q(grid):
    a = assemble_planes(grid, 'B_L')
    grid[:, 2:6] *= -1
    np.testing.assert_array_equal(a, assemble_planes(grid, 'B_L'))


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_nonfinite_refused(grid, invalid):
    grid[0, 0, 0, 0] = invalid
    with pytest.raises(ValueError):
        assemble_planes(grid, 'B_Q')


def test_unknown_variant_refused(grid):
    with pytest.raises(ValueError):
        assemble_planes(grid, 'B_foo')


def test_zero_coverage_refused(grid):
    grid[:, 8] = 0
    with pytest.raises(ValueError):
        assemble_planes(grid, 'B_Q')


def test_shared_initialization_and_shape(grid):
    torch.manual_seed(2026)
    a = SpatialClassifier()
    torch.manual_seed(2026)
    b = SpatialClassifier()
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])
    out = a(torch.from_numpy(assemble_planes(grid, 'B_Q')), torch.zeros(2, 286))
    assert out.shape == (2, 4)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())


def test_missing_rows_never_enter_model(grid):
    class Probe(torch.nn.Module):
        def forward(self, images, context):
            assert len(images) == 1
            return torch.full((1, 4), 99.)
    p0 = np.array([[1, 2, 3, 4], [9, 8, 7, 6]], np.float32)
    out = predict_with_fallback(Probe(), assemble_planes(grid, 'B_Q'),
                                np.zeros((2, 286), np.float32),
                                np.array([True, False]), p0, 'cpu', 1)
    np.testing.assert_array_equal(out[1], p0[1])
    np.testing.assert_array_equal(out[0], [99]*4)


def test_all_missing_skips_forward(grid):
    class Probe(torch.nn.Module):
        def forward(self, *args):
            raise AssertionError('must not run')
    p0 = np.ones((2, 4), np.float32)
    np.testing.assert_array_equal(predict_with_fallback(
        Probe(), assemble_planes(grid, 'B_Q'), np.zeros((2, 286), np.float32),
        np.zeros(2, bool), p0, 'cpu', 2), p0)


def test_metrics_hand_computed():
    m = classification_metrics([0, 1, 2, 3], [0, 0, 2, 3])
    assert m['accuracy'] == .75
    assert m['macro_f1'] == pytest.approx((2/3 + 0 + 1 + 1)/4)
    assert m['recall'] == [1, 0, 1, 1]


@pytest.mark.parametrize('truth,pred', [([0], []), ([True], [0]), ([4], [0]), ([0], [3.0])])
def test_bad_metric_identity_rejected(truth, pred):
    with pytest.raises(ValueError):
        classification_metrics(truth, pred)


def test_partition_rejects_group_leakage():
    with pytest.raises(ValueError):
        require_partition([{'group':'g','fold_roles':{'fold0':'inner_train'}},
                           {'group':'g','fold_roles':{'fold0':'outer_eval'}}], 'fold0')


def test_partition_rejects_unknown_role():
    with pytest.raises(ValueError):
        require_partition([{'group':'g','fold_roles':{'fold0':'test'}}], 'fold0')


def test_selection_ties_choose_earlier():
    h = [{'epoch':2,'macro_f1':.8,'accuracy':.9},
         {'epoch':1,'macro_f1':.8,'accuracy':.9}]
    assert select_best(h)['epoch'] == 1


def test_training_api_cannot_receive_outer_eval():
    assert 'eval' not in inspect.signature(fit_head).parameters
    assert 'outer_eval' not in inspect.signature(fit_head).parameters


def test_registered_file_rejects_drift(tmp_path):
    from core.material_spatial_classifier_io import InputRegistry
    p = tmp_path / 'input.txt'
    p.write_text('before')
    reg = InputRegistry(tmp_path)
    import hashlib
    expected = hashlib.sha256(b'before').hexdigest()
    reg.check('input.txt', expected)
    p.write_text('after')
    with pytest.raises(ValueError):
        reg.check('input.txt', expected)
    with pytest.raises(ValueError):
        reg.finish()


def test_registry_refuses_escape(tmp_path):
    from core.material_spatial_classifier_io import InputRegistry
    with pytest.raises(ValueError):
        InputRegistry(tmp_path).check('../outside', '0'*64)


def test_independent_metric_does_not_use_production(monkeypatch):
    from scripts.verify_material_spatial_classifier import independent_metrics
    import core.material_spatial_classifier as production
    monkeypatch.setattr(production, 'classification_metrics', lambda *a: (_ for _ in ()).throw(AssertionError()))
    result = independent_metrics([0, 1, 2, 3], [0, 0, 2, 3])
    assert result['macro_f1'] == pytest.approx(2/3)


def test_synthetic_training_integration(grid):
    from core.material_spatial_classifier import fit_head
    data = dict(images=assemble_planes(grid, 'B_Q'), context=np.zeros((2, 286), np.float32),
                available=np.array([True, False]), p0=np.zeros((2, 4), np.float32),
                labels=np.array([0, 1], np.int64))
    model, history, best = fit_head(data, data, seed=2026, device='cpu', epochs=2)
    assert len(history) == 2
    assert all(r['train_used'] == 1 for r in history)
    assert best == select_best(history)
    actual = predict_with_fallback(model, data['images'], data['context'],
                                   data['available'], data['p0'], 'cpu', 2)
    np.testing.assert_array_equal(actual[1], data['p0'][1])


@pytest.mark.parametrize('value', [None, [], True, 0])
def test_verifier_refuses_wrong_summary_container(value):
    from scripts.verify_material_spatial_classifier import compare
    with pytest.raises(ValueError):
        compare(value, {'gate': True})


@pytest.mark.parametrize('value', [True, 1.0, float('nan'), float('inf')])
def test_verifier_refuses_bad_counts(value):
    from scripts.verify_material_spatial_classifier import compare
    with pytest.raises(ValueError):
        compare(value, 1)


def test_verifier_checks_reference_class_metadata():
    from scripts.verify_material_spatial_classifier import check_rows
    data = {'rows':[dict(sample_name='a',group='g',class_id=0)],
            'features':{'s_valid':np.array([True])}}
    logits = np.array([[2, 1, 0, 0]], np.float32)
    saved = [dict(sample_name='a',group='g',class_id=0,logits=[2,1,0,0],
                  prediction=0,p0_prediction=3,used_fallback=False)]
    with pytest.raises(ValueError):
        check_rows(saved,logits,data,[0],logits)


def test_production_and_independent_summary_agree_synthetic():
    from scripts.train_material_spatial_classifier import summarize
    from scripts.verify_material_spatial_classifier import independently_summarize, compare
    records, reference = {}, {}
    for fold in ('fold0','fold1','fold2'):
        rows = [dict(sample_name=f'{fold}_{i}',group=f'{fold}_g',class_id=i,
                     prediction=i) for i in range(4)]
        reference[fold] = rows
        for seed in (2026,2027,2028):
            for variant in ('B_stats','B_L','B_R','B_Q'):
                records[(variant,seed,fold)] = rows
    result = summarize(records,reference)
    compare(result,independently_summarize(records,reference))
    assert result['gates'] == dict(gate1=False,gate2=False,gate3=False,
                                  gate4=True,gate5=True,gate6=False)


def test_all_gates_pass_on_independently_constructed_world():
    from scripts.train_material_spatial_classifier import summarize
    from scripts.verify_material_spatial_classifier import independently_summarize, compare
    records, reference = {}, {}
    def rows(fold, mistakes):
        return [dict(sample_name=f'{fold}_{i}', group=f'{fold}_group', class_id=i%4,
                     prediction=(i+1)%4 if i<mistakes else i%4) for i in range(40)]
    for fold in ('fold0','fold1','fold2'):
        reference[fold] = rows(fold,8)
        for seed in (2026,2027,2028):
            for variant,n in [('B_stats',8),('B_L',6),('B_R',4),('B_Q',0)]:
                records[(variant,seed,fold)] = rows(fold,n)
    result = summarize(records,reference)
    compare(result,independently_summarize(records,reference))
    assert all(result['gates'].values())
    assert result['paired']['P0']['2026']['fixed'] == 24
    assert result['paired']['P0']['2026']['broke'] == 0


@pytest.mark.parametrize('case', ['availability','p0','result'])
def test_prediction_invalid_inputs_or_outputs_rejected(grid, case):
    class Probe(torch.nn.Module):
        def forward(self, images, context):
            return torch.full((len(images),4),float('nan') if case=='result' else 1.)
    available=np.ones(2,bool) if case!='availability' else np.ones(2,np.float32)
    p0=np.zeros((2,4),np.float32)
    if case=='p0':
        p0[0,0]=np.inf
    with pytest.raises(ValueError):
        predict_with_fallback(Probe(),assemble_planes(grid,'B_Q'),np.zeros((2,286),np.float32),available,p0,'cpu',2)


def test_deadline_interrupts_training(grid):
    data=dict(images=assemble_planes(grid,'B_Q'),context=np.zeros((2,286),np.float32),
              labels=np.array([0,1],np.int64),available=np.ones(2,bool),p0=np.zeros((2,4),np.float32))
    def stop():
        raise TimeoutError('budget')
    with pytest.raises(TimeoutError):
        fit_head(data,data,seed=2026,device='cpu',check=stop)


def test_no_available_training_refused(grid):
    data=dict(images=assemble_planes(grid,'B_Q'),context=np.zeros((2,286),np.float32),
              labels=np.array([0,1],np.int64),available=np.zeros(2,bool),p0=np.zeros((2,4),np.float32))
    with pytest.raises(ValueError):
        fit_head(data,data,seed=2026,device='cpu')


def test_missing_class_still_counts_in_macro_f1():
    assert classification_metrics([0,0],[0,0])['macro_f1'] == .25


def test_valid_partition_and_empty_role_rejection():
    rows=[dict(group=r,fold_roles={'fold0':r}) for r in ('inner_train','inner_val','outer_eval')]
    assert require_partition(rows,'fold0') == dict(inner_train=[0],inner_val=[1],outer_eval=[2])
    with pytest.raises(ValueError):
        require_partition(rows[:2],'fold0')


def test_instance_pool_rejects_zero_mass():
    with pytest.raises(ValueError):
        SpatialClassifier()(torch.zeros(1,7,128,128),torch.zeros(1,286))
