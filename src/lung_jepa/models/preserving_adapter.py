"""Two-stage frozen-COLIPRI adapters; no medical labels enter these modules."""
from __future__ import annotations

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from .temporal_adapter import ResidualDeltaPredictor, TemporalLatentAdapter


class CharacteristicSIGReg(nn.Module):
    """Single-device Epps–Pulley SIGReg, matching official LeWM quadrature.

    Source: https://github.com/lucas-maes/le-wm/blob/main/module.py
    Uses unit random directions, frequencies [0,3], Gaussian integration
    weights, and the sample-count multiplier. No batch centering/whitening
    is performed inside the loss. Input is [samples, features].
    """

    def __init__(self, projections: int = 256, knots: int = 17):
        super().__init__()
        if projections < 1 or knots < 2:
            raise ValueError('SIGReg needs projections >= 1 and knots >= 2')
        self.projections = projections
        frequencies = torch.linspace(0, 3, knots)
        normal_cf = torch.exp(-frequencies.square()/2)
        quadrature = torch.ones(knots) * (6/(knots-1))
        quadrature[0] /= 2
        quadrature[-1] /= 2
        self.register_buffer('frequencies', frequencies)
        self.register_buffer('normal_cf', normal_cf)
        self.register_buffer('weights', quadrature * normal_cf)

    def forward(self, z: torch.Tensor, directions: torch.Tensor | None = None) -> torch.Tensor:
        if z.ndim != 2 or len(z) < 2:
            raise ValueError('SIGReg requires a [N,D] batch with N >= 2')
        if directions is None:
            directions = F.normalize(torch.randn(z.shape[1], self.projections, device=z.device), dim=0)
        phase = (z.float() @ directions).unsqueeze(-1) * self.frequencies
        distance = (phase.cos().mean(0)-self.normal_cf).square() + phase.sin().mean(0).square()
        return (distance @ self.weights).mean() * len(z)


class PreservingAdapter(LightningModule):
    """Reconstruct standardized frozen features through z alone, then freeze."""

    def __init__(self, input_dim: int = 864, latent_dim: int = 864, hidden_dim: int = 256,
                 decoder_hidden: int = 512, gamma: float = .1, anchor_weight: float = .1,
                 sigreg_weight: float = 0., projections: int = 256, lr: float = .001):
        super().__init__()
        self.save_hyperparameters()
        self.residual = latent_dim == input_dim
        self.encoder = TemporalLatentAdapter(input_dim, hidden_dim, latent_dim)
        if self.residual:
            nn.init.zeros_(self.encoder.net[-1].weight)
            nn.init.zeros_(self.encoder.net[-1].bias)
        # All information enters the decoder via z: never a bypass from x.
        self.decoder = nn.Sequential(nn.Linear(latent_dim, decoder_hidden), nn.GELU(),
                                     nn.Linear(decoder_hidden, input_dim))
        self.sigreg = CharacteristicSIGReg(projections)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.encoder(x)
        return x + self.hparams.gamma * transformed if self.residual else transformed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decoder(z)

    def step(self, batch: tuple[torch.Tensor], stage: str) -> torch.Tensor:
        x = batch[0]
        z, reconstructed = self(x)
        reconstruction = F.mse_loss(reconstructed, x)
        anchor = F.mse_loss(z, x) if self.residual else x.new_zeros(())
        preservation = reconstruction + self.hparams.anchor_weight * anchor
        # Fixed projection seed makes validation diagnostics repeatable.
        if self.hparams.sigreg_weight > 0:
            if stage == 'train':
                regularization = self.sigreg(z)
            else:
                g = torch.Generator(device=x.device).manual_seed(314159)
                directions = F.normalize(torch.randn(z.shape[1], self.sigreg.projections,
                                                     device=x.device, generator=g), dim=0)
                regularization = self.sigreg(z, directions)
        else:
            regularization = x.new_zeros(())
        total = preservation + self.hparams.sigreg_weight * regularization
        for key, value in {'loss':total, 'preservation':preservation, 'reconstruction':reconstruction,
                           'anchor':anchor, 'sigreg':regularization,
                           'feature_std':z.std(0, unbiased=False).mean()}.items():
            self.log(f'{stage}/{key}', value, on_step=False, on_epoch=True, batch_size=len(x))
        return total

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'train')

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'val')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=.0001)


class FrozenSpaceTransition(LightningModule):
    """Predict normalized deltas against cached states from a frozen adapter."""

    def __init__(self, dim: int, hidden_dim: int = 256, lr: float = .001):
        super().__init__()
        self.save_hyperparameters()
        self.predictor = ResidualDeltaPredictor(dim, use_time=True, hidden_dim=hidden_dim,
                                                blocks=2, dropout=.1)
        nn.init.zeros_(self.predictor.net[-1].weight)
        nn.init.zeros_(self.predictor.net[-1].bias)
        self.register_buffer('delta_mean', torch.zeros(dim))
        self.register_buffer('delta_scale', torch.ones(dim))

    def forward(self, z: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return z + self.predictor(z,time) * self.delta_scale + self.delta_mean

    def step(self, batch: tuple, stage: str) -> torch.Tensor:
        current, time, target = batch
        expected = (target-current-self.delta_mean)/self.delta_scale
        normalized = self.predictor(current,time)
        prediction = current + normalized*self.delta_scale+self.delta_mean
        loss = F.mse_loss(normalized,expected)
        for key,value in {'loss':loss, 'mse':F.mse_loss(prediction,target),
                          'identity_mse':F.mse_loss(current,target),
                          'cosine':F.cosine_similarity(prediction,target).mean()}.items():
            self.log(f'{stage}/{key}', value, on_step=False, on_epoch=True, batch_size=len(current))
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step(batch,'train')

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step(batch,'val')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(),lr=self.hparams.lr,weight_decay=.0001)
