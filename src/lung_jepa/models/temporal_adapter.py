"""Small, reusable temporal latent adapter and future-state predictors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


class TemporalLatentAdapter(nn.Module):
    """Maps frozen COLIPRI features to a compact learnable temporal state."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeEmbedding(nn.Module):
    def __init__(self, output_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        return self.net(delta_t)


class ResidualDeltaPredictor(nn.Module):
    def __init__(self, latent_dim: int, use_time: bool = False, use_velocity: bool = False, time_dim: int = 32, hidden_dim: int = 256, blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.use_time, self.use_velocity = use_time, use_velocity
        self.time_embedding = TimeEmbedding(time_dim) if use_time else None
        input_dim = latent_dim + (latent_dim if use_velocity else 0) + (time_dim if use_time else 0)
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        for block in range(blocks):
            layers.extend([nn.Linear(input_dim if block == 0 else hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, delta_t: torch.Tensor | None = None, velocity: torch.Tensor | None = None) -> torch.Tensor:
        parts = [z_t]
        if self.use_velocity:
            parts.append(torch.zeros_like(z_t) if velocity is None else velocity)
        if self.use_time:
            if delta_t is None:
                raise ValueError("delta_t is required when use_time=True")
            parts.append(self.time_embedding(delta_t))
        return self.net(torch.cat(parts, dim=-1))


class DirectLatentPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    space: Literal["frozen", "temporal"]
    residual: bool
    use_time: bool = False
    use_velocity: bool = False


MODEL_SPECS = {
    "identity": ModelSpec("identity", "frozen", False),
    "linear_direct": ModelSpec("linear_direct", "frozen", False),
    "raw_residual_mlp": ModelSpec("raw_residual_mlp", "frozen", True),
    "adapter_direct": ModelSpec("adapter_direct", "temporal", False),
    "adapter_residual": ModelSpec("adapter_residual", "temporal", True),
    "adapter_residual_time": ModelSpec("adapter_residual_time", "temporal", True, True),
    "adapter_residual_velocity_time": ModelSpec("adapter_residual_velocity_time", "temporal", True, True, True),
}
