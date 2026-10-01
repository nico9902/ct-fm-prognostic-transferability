"""Lightweight token pooling for observed spatial COLIPRI progression."""
from __future__ import annotations

import lightning as L
import torch
from torch import nn

from .prognostic_adapter import masked_prognostic_loss
from .temporal_adapter import TimeEmbedding


class SpatialProgressionPrognosticator(L.LightningModule):
    """Attention pooling and capacity-matched gated history without self-attention."""

    KINDS = {'mean_linear', 'token_attention_t1', 'token_gated_control',
             'token_gated_history', 'oracle_mean_linear', 'oracle_token_attention'}

    def __init__(self, kind: str, token_dim: int = 864, hidden_dim: int = 128,
                 time_dim: int = 32, tokens: int = 64, dropout: float = .1,
                 pos_weights: list[float] | None = None, lr: float = .001):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(kind)
        self.save_hyperparameters()
        self.register_buffer('pos_weights', torch.tensor(pos_weights or [1., 1., 1.]))
        self.validation_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.is_mean = kind in {'mean_linear', 'oracle_mean_linear'}
        self.is_gated = kind in {'token_gated_control', 'token_gated_history'}
        if self.is_mean:
            self.head = nn.Linear(token_dim + 1, 3)
        else:
            self.position = nn.Parameter(torch.zeros(1, tokens, hidden_dim))
            nn.init.trunc_normal_(self.position, std=.02)
            self.state_encoder = nn.Sequential(
                nn.Linear(token_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.time_embedding = TimeEmbedding(time_dim)
            if self.is_gated:
                self.delta_encoder = nn.Sequential(
                    nn.Linear(token_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                    nn.GELU(), nn.Dropout(dropout))
                self.gate = nn.Linear(hidden_dim * 2 + time_dim, hidden_dim)
            self.attention = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim + time_dim, 3)

    def forward(self, state: torch.Tensor, delta: torch.Tensor,
                time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None,
                                              torch.Tensor | None]:
        if self.is_mean:
            return self.head(torch.cat([state.mean(1), time], dim=1)), None, None
        state_features = self.state_encoder(state) + self.position
        time_features = self.time_embedding(time)
        gate_values = None
        if self.is_gated:
            # Same parameters in control/history; only patient-specific delta is ablated.
            delta_input = delta if self.hparams.kind == 'token_gated_history' else torch.zeros_like(delta)
            delta_features = self.delta_encoder(delta_input)
            expanded_time = time_features.unsqueeze(1).expand(-1, state.shape[1], -1)
            gate_values = torch.sigmoid(self.gate(torch.cat(
                [state_features, delta_features, expanded_time], dim=-1)))
            token_features = state_features + gate_values * delta_features
        else:
            token_features = state_features
        attention = torch.softmax(self.attention(token_features).squeeze(-1), dim=1)
        pooled = self.output_norm((token_features * attention.unsqueeze(-1)).sum(1))
        logits = self.head(torch.cat([pooled, time_features], dim=1))
        return logits, attention, None if gate_values is None else gate_values.mean(-1)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        state, delta, time, labels = batch
        loss = masked_prognostic_loss(self(state, delta, time)[0], labels, self.pos_weights)
        self.log('train/prognosis', loss, on_step=False, on_epoch=True, batch_size=len(state))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_outputs.clear()

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        state, delta, time, labels = batch
        self.validation_outputs.append((self(state, delta, time)[0].detach(), labels.detach()))

    def on_validation_epoch_end(self) -> None:
        logits, labels = (torch.cat([row[i] for row in self.validation_outputs]) for i in [0, 1])
        self.log('val/prognosis', masked_prognostic_loss(logits, labels, self.pos_weights))
        self.validation_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=.0001)
