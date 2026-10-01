"""Small prognosis models using one or two observed frozen COLIPRI states."""
from __future__ import annotations

import lightning as L
import torch
from torch import nn

from .prognostic_adapter import masked_prognostic_loss
from .temporal_adapter import TimeEmbedding


class ObservedProgressionPrognosticator(L.LightningModule):
    """Controlled current-only/history models with a common training objective."""

    KINDS = {'current_linear', 'current_time_linear', 'current_time_mlp',
             'history_linear', 'history_mlp', 'history_gated', 'oracle_t2_linear'}

    def __init__(self, kind: str, input_dim: int = 864, hidden_dim: int = 256,
                 time_dim: int = 32, dropout: float = .1,
                 pos_weights: list[float] | None = None, lr: float = .001):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f'Unknown kind: {kind}')
        self.save_hyperparameters()
        self.register_buffer('pos_weights', torch.tensor(pos_weights or [1., 1., 1.]))
        self.validation_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.time_embedding = TimeEmbedding(time_dim) if kind in {
            'current_time_mlp', 'history_mlp', 'history_gated'} else None

        if kind in {'current_linear', 'oracle_t2_linear'}:
            self.head = nn.Linear(input_dim, 3)
        elif kind == 'current_time_linear':
            self.head = nn.Linear(input_dim + 1, 3)
        elif kind == 'history_linear':
            self.head = nn.Linear(input_dim * 2 + 1, 3)
        elif kind == 'current_time_mlp':
            self.current_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.head = nn.Linear(hidden_dim + time_dim, 3)
        elif kind == 'history_mlp':
            self.encoder = nn.Sequential(
                nn.LayerNorm(input_dim * 2 + time_dim),
                nn.Linear(input_dim * 2 + time_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.head = nn.Linear(hidden_dim, 3)
        else:
            self.current_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.delta_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.gate = nn.Linear(hidden_dim * 2 + time_dim, hidden_dim)
            self.fused_norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, 3)

    def forward(self, current: torch.Tensor, delta: torch.Tensor,
                time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        kind = self.hparams.kind
        fused = None
        if kind in {'current_linear', 'oracle_t2_linear'}:
            logits = self.head(current)
        elif kind == 'current_time_linear':
            logits = self.head(torch.cat([current, time], dim=1))
        elif kind == 'history_linear':
            logits = self.head(torch.cat([current, delta, time], dim=1))
        elif kind == 'current_time_mlp':
            fused = torch.cat([self.current_encoder(current), self.time_embedding(time)], dim=1)
            logits = self.head(fused)
        elif kind == 'history_mlp':
            fused = self.encoder(torch.cat([current, delta, self.time_embedding(time)], dim=1))
            logits = self.head(fused)
        else:
            state = self.current_encoder(current)
            change = self.delta_encoder(delta)
            time_embedding = self.time_embedding(time)
            gate = torch.sigmoid(self.gate(torch.cat([state, change, time_embedding], dim=1)))
            fused = self.fused_norm(state + gate * change)
            logits = self.head(fused)
        return logits, fused

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        current, delta, time, labels = batch
        loss = masked_prognostic_loss(self(current, delta, time)[0], labels, self.pos_weights)
        self.log('train/prognosis', loss, on_step=False, on_epoch=True, batch_size=len(current))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_outputs.clear()

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        current, delta, time, labels = batch
        self.validation_outputs.append((self(current, delta, time)[0].detach(), labels.detach()))

    def on_validation_epoch_end(self) -> None:
        logits, labels = (torch.cat([row[i] for row in self.validation_outputs]) for i in [0, 1])
        loss = masked_prognostic_loss(logits, labels, self.pos_weights)
        self.log('val/prognosis', loss)
        self.validation_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=.0001)
