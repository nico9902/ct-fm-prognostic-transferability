"""CPU checks; GPU experiments are launched in background with W&B."""
import torch
from lung_jepa.models.prognostic_adapter import PrognosticAdapter, masked_prognostic_loss


def test_missing_labels_have_zero_gradient():
    logits = torch.zeros(3, 3, requires_grad=True)
    labels = torch.tensor([[1., 1., float('nan')], [0., 1., float('nan')], [0., 0., 1.]])
    loss = masked_prognostic_loss(logits, labels, torch.ones(3))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 2] == 0 and logits.grad[1, 2] == 0
    assert logits.grad[2, 2] != 0


def test_all_missing_batch_safe():
    logits = torch.zeros(2, 3, requires_grad=True)
    loss = masked_prognostic_loss(logits, torch.full((2, 3), float('nan')), torch.ones(3))
    loss.backward()
    assert loss == 0 and torch.all(logits.grad == 0)


def test_residual_identity_and_supervised_gradient():
    model = PrognosticAdapter(input_dim=12, hidden_dim=6, decoder_hidden=8)
    x = torch.randn(4, 12)
    z, _ = model(x)
    torch.testing.assert_close(z, x, rtol=0, atol=0)
    y = torch.randint(0, 2, (4, 3)).float()
    model.losses(x, y)['prognosis'].backward()
    assert model.encoder.net[-1].weight.grad.abs().sum() > 0
    assert model.decoder[0].in_features == 12


def test_variants_and_checkpoint_roundtrip(tmp_path):
    for kind in ['linear', 'mlp', 'adapter']:
        m = PrognosticAdapter(kind=kind, input_dim=12, hidden_dim=6, decoder_hidden=8)
        x = torch.randn(4, 12)
        losses = m.losses(x, torch.randint(0, 2, (4, 3)).float())
        assert all(torch.isfinite(v) for v in losses.values())
        clone = PrognosticAdapter(**dict(m.hparams))
        clone.load_state_dict(m.state_dict())
        torch.testing.assert_close(m(x)[1], clone(x)[1])
