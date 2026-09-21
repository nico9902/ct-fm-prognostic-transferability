"""Lightning module for frozen-COLIPRI temporal adapter ablations."""
from __future__ import annotations

import torch
from lightning import LightningModule
from torch import nn

from .temporal_adapter import DirectLatentPredictor, MODEL_SPECS, ResidualDeltaPredictor, TemporalLatentAdapter
from .temporal_losses import future_loss, sigreg_loss


class TemporalFutureStateModule(LightningModule):
    def __init__(self, input_dim: int, model_name: str, latent_dim: int = 128, adapter_hidden_dim: int = 256, predictor_hidden_dim: int = 256, predictor_blocks: int = 2, dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4, loss_alpha: float = 1.0, lambda_sigreg: float = 0.0):
        super().__init__()
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model_name={model_name}; choose from {sorted(MODEL_SPECS)}")
        self.save_hyperparameters()
        self.spec = MODEL_SPECS[model_name]
        self.adapter = TemporalLatentAdapter(input_dim, adapter_hidden_dim, latent_dim) if self.spec.space == "temporal" else None
        if model_name == "identity":
            self.predictor = nn.Identity()
        elif model_name == "linear_direct":
            self.predictor = nn.Linear(input_dim, input_dim)
        elif model_name == "raw_residual_mlp":
            self.predictor = ResidualDeltaPredictor(input_dim, hidden_dim=predictor_hidden_dim, blocks=predictor_blocks, dropout=dropout)
        elif self.spec.residual:
            self.predictor = ResidualDeltaPredictor(latent_dim, use_time=self.spec.use_time, use_velocity=self.spec.use_velocity, hidden_dim=predictor_hidden_dim, blocks=predictor_blocks, dropout=dropout)
        else:
            self.predictor = DirectLatentPredictor(latent_dim, predictor_hidden_dim, dropout)

    def _states(self, batch):
        source, target = batch["source"], batch["target"]
        if self.adapter is None:
            return source, target, None, batch["delta_t"]
        z_t, z_t1 = self.adapter(source), self.adapter(target)
        velocity = z_t - self.adapter(batch["previous"])
        return z_t, z_t1, velocity, batch["delta_t"]

    def forward(self, batch):
        z_t, z_t1, velocity, delta_t = self._states(batch)
        if self.spec.name == "identity":
            prediction = z_t
        elif self.spec.name == "linear_direct":
            prediction = self.predictor(z_t)
        elif self.spec.residual:
            prediction = z_t + self.predictor(z_t, delta_t if self.spec.use_time else None, velocity if self.spec.use_velocity else None)
        else:
            prediction = self.predictor(z_t)
        return prediction, z_t1, z_t

    def _step(self, batch, stage: str):
        prediction, target, z_t = self(batch)
        loss, parts = future_loss(prediction, target, float(self.hparams.loss_alpha))
        reg = sigreg_loss(z_t) if float(self.hparams.lambda_sigreg) > 0 else prediction.new_zeros(())
        total = loss + float(self.hparams.lambda_sigreg) * reg
        self.log(f"{stage}/loss", total, on_epoch=True, on_step=False, prog_bar=stage != "test")
        self.log(f"{stage}/future_mse", parts["mse"], on_epoch=True, on_step=False)
        self.log(f"{stage}/future_cosine", 1.0 - parts["cosine_loss"], on_epoch=True, on_step=False)
        self.log(f"{stage}/r2_batch", 1.0 - parts["mse"] / (target.var(unbiased=False) + 1e-8), on_epoch=True, on_step=False)
        if float(self.hparams.lambda_sigreg) > 0:
            self.log(f"{stage}/sigreg", reg, on_epoch=True, on_step=False)
        return total

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay))
