import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_lung1_e2_mlp import EfronLoss  # noqa: E402
from run_lung1_e2_sqa import SurvivalQueryAdapter  # noqa: E402


def test_efron_tied_events_matches_hand_calculation():
    risk = torch.tensor([0.2, -0.1, 0.5, 0.4], requires_grad=True)
    time = np.array([4.0, 3.0, 3.0, 1.0])
    event = np.array([1, 1, 1, 0])
    weights = torch.exp(risk)
    expected = (-risk[0] + torch.log(weights[0]) - risk[1] - risk[2]
                + torch.log(weights[:3].sum())
                + torch.log(weights[:3].sum() - (weights[1] + weights[2]) / 2)) / 3
    actual = EfronLoss(time, event)(risk)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert torch.isfinite(risk.grad).all()


def test_sqa_returns_one_finite_risk_per_patient():
    model = SurvivalQueryAdapter(input_dim=24, dropout=0.1)
    result = model(torch.randn(3, 16, 24))
    assert result.shape == (3,)
    assert torch.isfinite(result).all()


def test_two_pass_risk_gradient_matches_direct_cox_gradient():
    features = torch.tensor([[1.0, 0.1], [0.2, 1.0], [1.2, -0.4], [-0.2, 0.3]])
    time = np.array([4.0, 3.0, 3.0, 1.0])
    event = np.array([1, 1, 1, 0])
    direct = torch.nn.Linear(2, 1, bias=False)
    direct.weight.data[:] = torch.tensor([[0.2, -0.1]])
    EfronLoss(time, event)(direct(features).flatten()).backward()
    expected = direct.weight.grad.clone()

    recompute = torch.nn.Linear(2, 1, bias=False)
    recompute.weight.data.copy_(direct.weight.data)
    risk = recompute(features).flatten().detach().requires_grad_(True)
    derivative = torch.autograd.grad(EfronLoss(time, event)(risk), risk)[0].detach()
    for index in range(len(features)):
        (recompute(features[index:index + 1]).flatten()[0] * derivative[index]).backward()
    assert torch.allclose(recompute.weight.grad, expected, atol=1e-6)
