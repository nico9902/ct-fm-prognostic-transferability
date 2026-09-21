"""Losses for compact latent future-state prediction."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def future_loss(prediction: torch.Tensor, target: torch.Tensor, alpha: float = 1.0):
    mse = F.mse_loss(prediction, target)
    cosine_loss = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
    total = mse if alpha >= 1.0 else alpha * mse + (1.0 - alpha) * cosine_loss
    return total, {"mse": mse, "cosine_loss": cosine_loss}


def sigreg_loss(z: torch.Tensor, num_slices: int = 32) -> torch.Tensor:
    """Legacy sliced moment penalty (NOT official characteristic-function SIGReg).

    Retained for checkpoint compatibility. New experiments use
    preserving_adapter.CharacteristicSIGReg instead.
    """
    directions = torch.randn(z.shape[1], min(num_slices, z.shape[1]), device=z.device, dtype=z.dtype)
    directions = F.normalize(directions, dim=0)
    projected = z @ directions
    centered = projected - projected.mean(dim=0, keepdim=True)
    variance = centered.pow(2).mean(dim=0)
    skew = centered.pow(3).mean(dim=0).abs()
    kurtosis = centered.pow(4).mean(dim=0) / (variance.pow(2) + 1e-6)
    return (variance - 1.0).pow(2).mean() + skew.mean() + (kurtosis - 3.0).pow(2).mean()
