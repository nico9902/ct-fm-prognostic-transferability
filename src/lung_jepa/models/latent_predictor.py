from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class LatentPredictorNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        clinical_dim: int,
        delta_dim: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = embedding_dim + clinical_dim + delta_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        source: torch.Tensor,
        clinical: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([source, clinical, delta], dim=-1))


class LatentTransformerCLSPredictorNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        clinical_dim: int,
        delta_dim: int = 1,
        transformer_dim: int = 384,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_dim: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        if transformer_dim % transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads.")

        self.clinical_dim = clinical_dim
        self.delta_dim = delta_dim
        self.source_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, transformer_dim),
        )
        self.clinical_proj = (
            nn.Sequential(
                nn.LayerNorm(clinical_dim),
                nn.Linear(clinical_dim, transformer_dim),
            )
            if clinical_dim > 0
            else None
        )
        self.delta_proj = (
            nn.Sequential(
                nn.LayerNorm(delta_dim),
                nn.Linear(delta_dim, transformer_dim),
            )
            if delta_dim > 0
            else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 4, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, embedding_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        source: torch.Tensor,
        clinical: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        if source.ndim == 3:
            source = source.mean(dim=1)
        batch_size = source.shape[0]
        tokens = [
            self.cls_token.expand(batch_size, -1, -1),
            self.source_proj(source).unsqueeze(1),
        ]
        if self.clinical_proj is not None:
            tokens.append(self.clinical_proj(clinical).unsqueeze(1))
        if self.delta_proj is not None:
            tokens.append(self.delta_proj(delta).unsqueeze(1))

        sequence = torch.cat(tokens, dim=1)
        sequence = sequence + self.pos_embedding[:, : sequence.shape[1], :]
        encoded = self.encoder(sequence)
        return self.head(encoded[:, 0])


class TokenTransformerCLSPredictorNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        clinical_dim: int,
        delta_dim: int = 1,
        transformer_dim: int = 384,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_dim: int = 1536,
        dropout: float = 0.1,
        max_tokens: int = 1024,
    ):
        super().__init__()
        if transformer_dim % transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads.")

        self.max_tokens = max_tokens
        self.source_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, transformer_dim),
        )
        self.clinical_proj = (
            nn.Sequential(nn.LayerNorm(clinical_dim), nn.Linear(clinical_dim, transformer_dim))
            if clinical_dim > 0
            else None
        )
        self.delta_proj = (
            nn.Sequential(nn.LayerNorm(delta_dim), nn.Linear(delta_dim, transformer_dim))
            if delta_dim > 0
            else None
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens + 3, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, embedding_dim),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        source: torch.Tensor,
        clinical: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        if source.ndim == 2:
            source = source.unsqueeze(1)
        if source.shape[1] > self.max_tokens:
            raise ValueError(f"source has {source.shape[1]} tokens, max_tokens={self.max_tokens}.")

        batch_size = source.shape[0]
        tokens = [
            self.cls_token.expand(batch_size, -1, -1),
            self.source_proj(source),
        ]
        if self.clinical_proj is not None:
            tokens.append(self.clinical_proj(clinical).unsqueeze(1))
        if self.delta_proj is not None:
            tokens.append(self.delta_proj(delta).unsqueeze(1))

        sequence = torch.cat(tokens, dim=1)
        sequence = sequence + self.pos_embedding[:, : sequence.shape[1], :]
        encoded = self.encoder(sequence)
        return self.head(encoded[:, 0])


class TokenTransformerPredictorNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        clinical_dim: int,
        delta_dim: int = 1,
        transformer_dim: int = 384,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_dim: int = 1536,
        dropout: float = 0.1,
        max_tokens: int = 1024,
    ):
        super().__init__()
        if transformer_dim % transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads.")

        self.max_tokens = max_tokens
        self.source_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, transformer_dim),
        )
        self.clinical_proj = (
            nn.Sequential(nn.LayerNorm(clinical_dim), nn.Linear(clinical_dim, transformer_dim))
            if clinical_dim > 0
            else None
        )
        self.delta_proj = (
            nn.Sequential(nn.LayerNorm(delta_dim), nn.Linear(delta_dim, transformer_dim))
            if delta_dim > 0
            else None
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_tokens + 2, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, embedding_dim),
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(
        self,
        source: torch.Tensor,
        clinical: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        if source.ndim == 2:
            source = source.unsqueeze(1)
        if source.shape[1] > self.max_tokens:
            raise ValueError(f"source has {source.shape[1]} tokens, max_tokens={self.max_tokens}.")

        tokens = [self.source_proj(source)]
        if self.clinical_proj is not None:
            tokens.append(self.clinical_proj(clinical).unsqueeze(1))
        if self.delta_proj is not None:
            tokens.append(self.delta_proj(delta).unsqueeze(1))

        sequence = torch.cat(tokens, dim=1)
        sequence = sequence + self.pos_embedding[:, : sequence.shape[1], :]
        encoded = self.encoder(sequence)
        return self.head(encoded[:, : source.shape[1], :])


class LatentPredictorModule(LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        clinical_dim: int,
        delta_dim: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cosine_weight: float = 0.1,
        token_loss_weight: float = 1.0,
        prediction_mode: Literal["direct", "residual"] = "direct",
        model_arch: Literal["mlp", "transformer_cls", "transformer_token_cls", "transformer_token"] = "mlp",
        transformer_dim: int = 384,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_dim: int = 1536,
        max_tokens: int = 1024,
    ):
        super().__init__()
        if prediction_mode not in {"direct", "residual"}:
            raise ValueError("prediction_mode must be 'direct' or 'residual'.")
        if model_arch not in {"mlp", "transformer_cls", "transformer_token_cls", "transformer_token"}:
            raise ValueError(
                "model_arch must be 'mlp', 'transformer_cls', 'transformer_token_cls' "
                "or 'transformer_token'."
            )
        self.save_hyperparameters()
        if model_arch == "transformer_token":
            self.predictor = TokenTransformerPredictorNet(
                embedding_dim=embedding_dim,
                clinical_dim=clinical_dim,
                delta_dim=delta_dim,
                transformer_dim=transformer_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_mlp_dim=transformer_mlp_dim,
                dropout=dropout,
                max_tokens=max_tokens,
            )
        elif model_arch == "transformer_token_cls":
            self.predictor = TokenTransformerCLSPredictorNet(
                embedding_dim=embedding_dim,
                clinical_dim=clinical_dim,
                delta_dim=delta_dim,
                transformer_dim=transformer_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_mlp_dim=transformer_mlp_dim,
                dropout=dropout,
                max_tokens=max_tokens,
            )
        elif model_arch == "transformer_cls":
            self.predictor = LatentTransformerCLSPredictorNet(
                embedding_dim=embedding_dim,
                clinical_dim=clinical_dim,
                delta_dim=delta_dim,
                transformer_dim=transformer_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_mlp_dim=transformer_mlp_dim,
                dropout=dropout,
            )
        else:
            self.predictor = LatentPredictorNet(
                embedding_dim=embedding_dim,
                clinical_dim=clinical_dim,
                delta_dim=delta_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

    def forward(
        self,
        source: torch.Tensor,
        clinical: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        update = self.predictor(source, clinical, delta)
        if self.hparams.prediction_mode == "residual":
            if update.ndim == 3:
                if source.ndim == 2:
                    source = source.unsqueeze(1)
                return source + update
            return self._pool_if_tokens(source) + update
        return update

    @staticmethod
    def _pool_if_tokens(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 3:
            return latent.mean(dim=1)
        return latent

    def _latent_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l1 = F.l1_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        loss = l1 + self.hparams.cosine_weight * cosine
        return loss, l1, cosine

    @staticmethod
    def _relative_improvement(
        baseline: torch.Tensor,
        model: torch.Tensor,
    ) -> torch.Tensor:
        eps = torch.finfo(model.dtype).eps
        return (baseline - model) / baseline.clamp_min(eps)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pred = self(batch["source"], batch["clinical"], batch["delta"])
        pred_pooled = self._pool_if_tokens(pred)
        source = self._pool_if_tokens(batch["source"])
        target = self._pool_if_tokens(batch["target"])

        loss, l1, cosine = self._latent_metrics(pred_pooled, target)
        optimization_loss = loss
        persistence_loss, persistence_l1, persistence_cosine = self._latent_metrics(
            source,
            target,
        )
        loss_improvement = self._relative_improvement(persistence_loss, loss)
        l1_improvement = self._relative_improvement(persistence_l1, l1)
        cosine_improvement = self._relative_improvement(persistence_cosine, cosine)

        batch_size = target.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}/l1", l1, batch_size=batch_size)
        self.log(f"{stage}/cosine", cosine, batch_size=batch_size)
        self.log(f"{stage}/persistence_loss", persistence_loss, batch_size=batch_size)
        self.log(f"{stage}/persistence_l1", persistence_l1, batch_size=batch_size)
        self.log(f"{stage}/persistence_cosine", persistence_cosine, batch_size=batch_size)
        self.log(f"{stage}/relative_loss_improvement", loss_improvement, batch_size=batch_size)
        self.log(f"{stage}/relative_l1_improvement", l1_improvement, batch_size=batch_size)
        self.log(
            f"{stage}/relative_cosine_improvement",
            cosine_improvement,
            batch_size=batch_size,
        )
        if pred.ndim == 3 and batch["target"].ndim == 3 and pred.shape == batch["target"].shape:
            token_loss, token_l1, token_cosine = self._latent_metrics(
                pred.flatten(0, 1),
                batch["target"].flatten(0, 1),
            )
            optimization_loss = loss + self.hparams.token_loss_weight * token_loss
            self.log(f"{stage}/token_loss", token_loss, batch_size=batch_size)
            self.log(f"{stage}/token_l1", token_l1, batch_size=batch_size)
            self.log(f"{stage}/token_cosine", token_cosine, batch_size=batch_size)
            self.log(f"{stage}/optimization_loss", optimization_loss, batch_size=batch_size)
        return optimization_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
