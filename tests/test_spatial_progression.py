import torch

from lung_jepa.models.spatial_progression import SpatialProgressionPrognosticator


def test_shapes_probabilities_and_gradients():
    state, delta, time = torch.randn(4, 8, 12), torch.randn(4, 8, 12), torch.randn(4, 1)
    labels = torch.randint(0,2,(4,3)).float()
    for kind in SpatialProgressionPrognosticator.KINDS:
        model=SpatialProgressionPrognosticator(kind,token_dim=12,hidden_dim=6,time_dim=4,tokens=8)
        logits,attention,gate=model(state,delta,time)
        assert logits.shape==(4,3)
        if attention is not None:
            assert attention.shape==(4,8)
            torch.testing.assert_close(attention.sum(1),torch.ones(4))
        if gate is not None:
            assert gate.shape==(4,8) and torch.all((gate>=0)&(gate<=1))
        loss=model.training_step((state,delta,time,labels),0);loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())


def test_capacity_matched_control_ignores_delta():
    model=SpatialProgressionPrognosticator('token_gated_control',token_dim=12,hidden_dim=6,time_dim=4,tokens=8).eval()
    state,time=torch.randn(4,8,12),torch.randn(4,1)
    a=model(state,torch.randn(4,8,12),time)[0]
    b=model(state,torch.randn(4,8,12),time)[0]
    torch.testing.assert_close(a,b)


def test_history_uses_delta_and_parameter_count_matches_control():
    kwargs=dict(token_dim=12,hidden_dim=6,time_dim=4,tokens=8)
    history=SpatialProgressionPrognosticator('token_gated_history',**kwargs).eval()
    control=SpatialProgressionPrognosticator('token_gated_control',**kwargs).eval()
    assert sum(p.numel() for p in history.parameters())==sum(p.numel() for p in control.parameters())
    state,time=torch.randn(4,8,12),torch.randn(4,1)
    assert not torch.allclose(history(state,torch.zeros_like(state),time)[0],
                              history(state,torch.ones_like(state),time)[0])
