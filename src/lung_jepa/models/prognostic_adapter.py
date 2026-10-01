"""Supervised preservation of frozen COLIPRI features; no temporal predictor."""
from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from .temporal_adapter import TemporalLatentAdapter


def masked_prognostic_loss(logits: torch.Tensor, labels: torch.Tensor,
                           pos_weight: torch.Tensor) -> torch.Tensor:
    """Equal horizon weighting, excluding unknown labels (never negatives)."""
    mask = torch.isfinite(labels)
    values = F.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(labels), pos_weight=pos_weight, reduction='none')
    counts = mask.sum(0)
    per_task = (values * mask).sum(0) / counts.clamp_min(1)
    return per_task[counts > 0].mean() if (counts > 0).any() else logits.sum() * 0


class PrognosticAdapter(L.LightningModule):
    """Linear, capacity-matched nonlinear, or identity-initialized residual model.

    Inputs are training-standardized *current* frozen COLIPRI states. All heads
    predict the same three post-t+1 endpoints. No round/time input is provided.
    The decoder sees z only; supervision reaches the encoder, never COLIPRI.
    """

    def __init__(self, kind: str = 'adapter', input_dim: int = 864,
                 hidden_dim: int = 256, decoder_hidden: int = 512,
                 gamma: float = .1, reconstruction_weight: float = 1.,
                 anchor_weight: float = .1, prognosis_weight: float = 1.,
                 pos_weights: list[float] | None = None, lr: float = .001):
        super().__init__()
        if kind not in {'linear', 'mlp', 'adapter'}:
            raise ValueError(kind)
        self.save_hyperparameters()
        self.encoder = (nn.Identity() if kind == 'linear' else
                        TemporalLatentAdapter(input_dim, hidden_dim, input_dim))
        if kind == 'adapter':
            nn.init.zeros_(self.encoder.net[-1].weight)
            nn.init.zeros_(self.encoder.net[-1].bias)
        self.head = nn.Linear(input_dim, 3)
        self.decoder = (nn.Sequential(nn.Linear(input_dim, decoder_hidden), nn.GELU(),
                                      nn.Linear(decoder_hidden, input_dim))
                        if kind == 'adapter' and reconstruction_weight > 0 else None)
        self.register_buffer('pos_weights', torch.tensor(pos_weights or [1., 1., 1.]))
        self.validation_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return x + self.hparams.gamma * z if self.hparams.kind == 'adapter' else z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.head(z)

    def losses(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        z, logits = self(x)
        prognosis = masked_prognostic_loss(logits, y, self.pos_weights)
        rec = F.mse_loss(self.decoder(z), x) if self.decoder is not None else x.new_zeros(())
        anchor = F.mse_loss(z, x) if self.hparams.kind == 'adapter' else x.new_zeros(())
        total = (self.hparams.prognosis_weight * prognosis +
                 self.hparams.reconstruction_weight * rec + self.hparams.anchor_weight * anchor)
        return {'loss': total, 'prognosis': prognosis, 'reconstruction': rec, 'anchor': anchor}

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        losses = self.losses(x, y)
        for key, value in losses.items():
            self.log('train/' + key, value, on_step=False, on_epoch=True, batch_size=len(x))
        return losses['loss']

    def on_validation_epoch_start(self) -> None:
        self.validation_outputs.clear()

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        self.validation_outputs.append((self(x)[1].detach(), y.detach()))
        for key, value in self.losses(x, y).items():
            if key != 'prognosis':
                self.log('val/' + key, value, on_epoch=True, batch_size=len(x))

    def on_validation_epoch_end(self) -> None:
        logits, labels = (torch.cat([v[i] for v in self.validation_outputs]) for i in [0, 1])
        # Exact full-validation masked macro loss, independent of batch sizes.
        self.log('val/prognosis', masked_prognostic_loss(logits, labels, self.pos_weights))
        self.validation_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=.0001)
