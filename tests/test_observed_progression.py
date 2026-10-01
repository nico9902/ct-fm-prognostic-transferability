import torch

from lung_jepa.models.observed_progression import ObservedProgressionPrognosticator


def test_all_model_shapes_and_gradients():
    current, delta, time = torch.randn(5, 12), torch.randn(5, 12), torch.randn(5, 1)
    labels = torch.randint(0, 2, (5, 3)).float()
    for kind in ObservedProgressionPrognosticator.KINDS:
        model = ObservedProgressionPrognosticator(kind, input_dim=12, hidden_dim=8, time_dim=4)
        logits, fused = model(current, delta, time)
        assert logits.shape == (5, 3)
        loss = model.training_step((current, delta, time, labels), 0)
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None for p in model.parameters())
        if kind == 'history_gated':
            assert fused.shape == (5, 8)


def test_current_models_ignore_delta():
    current, time = torch.randn(5, 12), torch.randn(5, 1)
    for kind in ['current_linear', 'current_time_linear', 'current_time_mlp', 'oracle_t2_linear']:
        model = ObservedProgressionPrognosticator(kind, input_dim=12, hidden_dim=8, time_dim=4).eval()
        a = model(current, torch.randn(5, 12), time)[0]
        b = model(current, torch.randn(5, 12), time)[0]
        torch.testing.assert_close(a, b)


def test_history_models_use_delta():
    current, time = torch.randn(5, 12), torch.randn(5, 1)
    for kind in ['history_linear', 'history_mlp', 'history_gated']:
        model = ObservedProgressionPrognosticator(kind, input_dim=12, hidden_dim=8, time_dim=4).eval()
        a = model(current, torch.zeros(5, 12), time)[0]
        b = model(current, torch.ones(5, 12), time)[0]
        assert not torch.allclose(a, b)
