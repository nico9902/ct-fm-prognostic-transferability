"""Regression tests for information preservation and the export alignment bug."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from lung_jepa.models.preserving_adapter import CharacteristicSIGReg, PreservingAdapter, FrozenSpaceTransition
from run_colipri_temporal_adapter_ablation import collect


class PreservationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)

    def test_sigreg_matches_reference_quadrature_and_gradient(self):
        z=torch.randn(64,12,requires_grad=True)
        a=F.normalize(torch.randn(12,32),dim=0)
        loss=CharacteristicSIGReg(32)(z,a)
        # Independent trapezoid integration of both positive/negative frequencies.
        t=torch.linspace(0,3,17)
        phase=(z@a)[...,None]*t
        err=(phase.cos().mean(0)-torch.exp(-t*t/2)).square()+phase.sin().mean(0).square()
        expected=2*len(z)*torch.trapezoid(err*torch.exp(-t*t/2),t,dim=-1).mean()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.norm(),0)

    def test_sigreg_detects_collapse_and_mean_shift(self):
        normal=torch.randn(2048,16)
        a=F.normalize(torch.randn(16,64),dim=0)
        reg=CharacteristicSIGReg(64)
        self.assertLess(reg(normal,a),reg(torch.zeros_like(normal),a)/10)
        self.assertLess(reg(normal,a),reg(normal+3,a)/10)

    def test_residual_starts_at_identity_and_decoder_has_no_input_bypass(self):
        model=PreservingAdapter(input_dim=12,latent_dim=12,hidden_dim=8,decoder_hidden=8)
        x=torch.randn(10,12)
        z,decoded=model(x)
        torch.testing.assert_close(z,x,rtol=0,atol=0)
        loss=F.mse_loss(decoded,x)
        loss.backward()
        self.assertGreater(model.encoder.net[-1].weight.grad.norm(),0)
        constant=torch.zeros_like(z)
        self.assertEqual(model.decoder(constant).unique(dim=0).shape[0],1)

    def test_frozen_adapter_cannot_receive_transition_gradients(self):
        adapter=PreservingAdapter(input_dim=12,latent_dim=6,hidden_dim=8,decoder_hidden=8)
        adapter.eval().requires_grad_(False)
        before={k:v.clone() for k,v in adapter.state_dict().items()}
        with torch.no_grad():
            source=adapter.encode(torch.randn(10,12)); target=adapter.encode(torch.randn(10,12))
        predictor=FrozenSpaceTransition(6,8)
        loss=F.mse_loss(predictor(source,torch.ones(10,1)),target)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in adapter.parameters()))
        for k,v in adapter.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_export_ignores_training_shuffle(self):
        class Echo(torch.nn.Module):
            def forward(self,batch):
                return batch['source'],batch['source'],batch['source']
        data=[{'source':torch.tensor([float(i)])} for i in range(31)]
        loader=DataLoader(data,batch_size=7,shuffle=True)
        pred,_,_=collect(Echo(),loader,'cpu')
        np.testing.assert_array_equal(pred[:,0],np.arange(31))


if __name__=='__main__': unittest.main()
