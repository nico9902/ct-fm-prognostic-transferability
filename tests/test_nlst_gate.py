"""Scientific invariants: patient leakage, query correspondence, missing targets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import unittest
import torch
from lung_jepa.models.disease_state import DiseaseStateProjector, GateModule
from run_nlst_gate import validate_splits, representations, metrics


def test_split_rejects_conflicting_patient_rows():
    with unittest.TestCase().assertRaisesRegex(ValueError,'one assignment'):
        validate_splits(pd.DataFrame({'patient_id':['p','p'],'split':['train','test']}))


def test_current_input_and_invertible_delta():
    current=np.ones((2,3),dtype=np.float32)
    future=np.full((2,3),7,dtype=np.float32)
    reps=representations(current,future)
    changed=representations(current,future+10)
    np.testing.assert_array_equal(reps['G0'],changed['G0'])
    np.testing.assert_array_equal(reps['G1'][:,3:]-reps['G1'][:,:3],reps['G2'][:,3:])
    np.testing.assert_array_equal(reps['G_identity'][:,:3],reps['G_identity'][:,3:])


def test_shared_queries_identical_scans_have_zero_delta():
    model=DiseaseStateProjector(input_dim=12,d_state=16,queries=4,tokens=8).eval()
    x=torch.randn(3,8,12)
    a,b=model(x),model(x)
    assert a.shape==(3,4,16)
    torch.testing.assert_close(a-b,torch.zeros_like(a))
    assert sum(p.numel() for p in model.parameters())<10000


def test_missing_supervision_has_no_gradient():
    module=GateModule(8,3,[1.,1.,1.])
    logits=torch.randn(2,3,requires_grad=True)
    y=torch.tensor([[1.,float('nan'),0.],[0.,float('nan'),1.]])
    module.loss(logits,y).backward()
    torch.testing.assert_close(logits.grad[:,1],torch.zeros(2))
    assert torch.isfinite(logits.grad).all()


def test_single_class_metrics_do_not_claim_auroc():
    result=metrics(np.zeros(3),np.array([.1,.3,.8]))
    assert np.isnan(result['auroc'])
    assert np.isnan(result['balanced_accuracy'])
    unittest.TestCase().assertAlmostEqual(result['specificity'],2/3)

if __name__ == '__main__':
    suite=unittest.TestSuite(unittest.FunctionTestCase(test) for name,test in list(globals().items()) if name.startswith('test_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
