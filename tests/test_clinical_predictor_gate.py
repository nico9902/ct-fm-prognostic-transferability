"""Verify clinical-gradient isolation, masking, same capacity, and cluster metrics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef
from lung_jepa.models.clinical_future import ClinicalFuturePredictor
from run_nlst_clinical_predictor_gate import cluster_weights,weighted_scores


def model(variant='C4',mode='residual'):
    return ClinicalFuturePredictor([0.]*12,[.1]*12,[0.]*12,[1.]*12,[0.]*12,[1.]*12,
        oracle_heads=2,hidden=32,variant=variant,mode=mode)


def test_clinical_gradients_only_reach_predictor():
    m=model();m.train()
    assert not m.oracle.training
    current=torch.randn(8,12);time=torch.zeros(8,1);future=torch.randn(8,12)
    labels=torch.randint(0,2,(8,3)).float();pred=m(current,time)
    loss=m.losses(current,pred,future,labels)['loss'];loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.network.parameters())
    assert all(not p.requires_grad and p.grad is None for p in m.oracle.parameters())
    optimizer=m.configure_optimizers()
    assert set(id(p) for group in optimizer.param_groups for p in group['params'])==set(id(p) for p in m.network.parameters())


def test_forward_has_no_future_or_outcome_input():
    m=model().eval();x=torch.randn(5,12);t=torch.zeros(5,1)
    a=m(x,t);b=m(x,t)
    torch.testing.assert_close(a,b)
    assert a.shape==x.shape


def test_missing_labels_mask_clinical_losses():
    m=model();x=torch.randn(4,12);pred=m(x,torch.zeros(4,1));future=torch.randn(4,12)
    losses=m.losses(x,pred,future,torch.full((4,3),float('nan')))
    assert losses['distillation'].item()==0 and losses['label'].item()==0
    assert torch.isfinite(losses['loss'])


def test_identical_predictor_capacity_and_old_objective():
    residual=model();direct=model(mode='direct')
    assert sum(p.numel() for p in residual.network.parameters())==sum(p.numel() for p in direct.network.parameters())
    x=torch.randn(4,12);future=torch.randn(4,12);pred=residual(x,torch.zeros(4,1))
    expected=((pred-x-residual.delta_mean)/residual.delta_scale-(future-x-residual.delta_mean)/residual.delta_scale).square().mean()
    actual=residual.losses(x,pred,future,torch.ones(4,3))['state']
    torch.testing.assert_close(actual,expected)


def test_cluster_resamples_keep_patient_pairs_together():
    w=cluster_weights(np.array(['a','a','b','c','c']),30,42)
    np.testing.assert_array_equal(w[:,0],w[:,1]);np.testing.assert_array_equal(w[:,3],w[:,4])
    assert np.all(w[:,[0,2,3]].sum(1)==3)


def test_fast_metrics_match_explicit_sklearn_resamples_with_ties():
    y=np.array([0,1,0,1,1,0]);p=np.array([.2,.2,.5,.7,.7,.8])
    w=np.array([[1,1,1,1,1,1],[2,3,1,0,2,1],[0,0,1,2,3,1]])
    fast=weighted_scores(y,p,w)
    for i,row in enumerate(w):
        indices=np.repeat(np.arange(len(y)),row)
        expected=[roc_auc_score(y[indices],p[indices]),average_precision_score(y[indices],p[indices]),
                  matthews_corrcoef(y[indices],p[indices]>=.5)]
        np.testing.assert_allclose(fast[i],expected,atol=1e-12)

if __name__=='__main__':
    suite=unittest.TestSuite(unittest.FunctionTestCase(test) for name,test in list(globals().items()) if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
