"""Small disease-query readout of cached, frozen COLIPRI spatial features."""
from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn
from .prognostic_adapter import masked_prognostic_loss


class DiseaseStateProjector(nn.Module):
    def __init__(self, input_dim=864, d_state=128, queries=4, tokens=64, heads=4):
        super().__init__()
        self.input = nn.Linear(input_dim, d_state)
        self.position = nn.Parameter(torch.zeros(1, tokens, d_state))
        self.queries = nn.Parameter(torch.randn(1, queries, d_state) * .02)
        self.attention = nn.MultiheadAttention(d_state, heads, dropout=0., batch_first=True)
        self.norm1 = nn.LayerNorm(d_state)
        self.mlp = nn.Sequential(nn.Linear(d_state, 2*d_state), nn.GELU(), nn.Linear(2*d_state, d_state))
        self.norm2 = nn.LayerNorm(d_state)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.position.shape[1]:
            raise ValueError('Expected [batch, configured spatial tokens, channels]')
        x = self.input(tokens) + self.position
        q = self.queries.expand(len(x), -1, -1)
        attended, _ = self.attention(q, x, x, need_weights=False)
        state = self.norm1(q + attended)
        return self.norm2(state + self.mlp(state))


class GateModule(pl.LightningModule):
    """Either a multi-task projector or a deliberately small fixed-state probe."""
    def __init__(self, input_dim, tasks, pos_weights, kind='linear', d_state=128,
                 queries=4, tokens=64, hidden=128, lr=.001, auxiliary_weight=.25):
        super().__init__()
        self.save_hyperparameters()
        self.projector = (DiseaseStateProjector(input_dim, d_state, queries, tokens)
                          if kind == 'projector' else None)
        dim = queries*d_state if self.projector is not None else input_dim
        self.head = (nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(.1), nn.Linear(hidden, tasks))
                     if kind == 'mlp' else nn.Linear(dim, tasks))
        self.register_buffer('pos_weights', torch.tensor(pos_weights, dtype=torch.float32))
        self.validation_outputs = []

    def forward(self, x):
        if self.projector is not None:
            x = self.projector(x).flatten(1)
        return self.head(x)

    def loss(self, logits, y):
        if self.projector is None:
            return masked_prognostic_loss(logits, y, self.pos_weights)
        prognosis = masked_prognostic_loss(logits[:, :3], y[:, :3], self.pos_weights[:3])
        auxiliary = masked_prognostic_loss(logits[:, 3:], y[:, 3:], self.pos_weights[3:])
        return prognosis + self.hparams.auxiliary_weight*auxiliary

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log('train/loss', loss, on_step=False, on_epoch=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.validation_outputs.append((self(x).detach(), y.detach()))

    def on_validation_epoch_end(self):
        logits, y = [torch.cat([b[i] for b in self.validation_outputs]) for i in (0, 1)]
        self.log('val/loss', self.loss(logits, y))
        self.validation_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=.0001)
