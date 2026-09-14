from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def select_numeric_clinical_columns(df: pd.DataFrame) -> list[str]:
    clinical_cols = [c for c in df.columns if c.startswith("clinical_")]
    numeric = []
    for col in clinical_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            df[col] = values
            numeric.append(col)
    return numeric


def split_by_patient(
    df: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to < 1.")

    patients = sorted(df["patient_id"].astype(str).unique())
    random.Random(seed).shuffle(patients)
    n_patients = len(patients)
    n_test = max(1, int(round(n_patients * test_fraction))) if n_patients > 2 and test_fraction > 0 else 0
    n_val = max(1, int(round(n_patients * val_fraction))) if n_patients - n_test > 1 and val_fraction > 0 else 0
    if n_val + n_test >= n_patients:
        n_test = max(0, min(n_test, n_patients - 2))
        n_val = max(0, min(n_val, n_patients - n_test - 1))

    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test : n_test + n_val])
    test = df[df["patient_id"].astype(str).isin(test_patients)].copy()
    val = df[df["patient_id"].astype(str).isin(val_patients)].copy()
    train = df[
        ~df["patient_id"].astype(str).isin(test_patients | val_patients)
    ].copy()
    if train.empty:
        train = df.copy()
        val = df.iloc[0:0].copy()
        test = df.iloc[0:0].copy()
    return train, val, test


def normalize_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    stats = {}
    for col in cols:
        mean = float(train[col].mean(skipna=True))
        std = float(train[col].std(skipna=True))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        fill = mean if np.isfinite(mean) else 0.0
        train[col] = (train[col].fillna(fill) - fill) / std
        val[col] = (val[col].fillna(fill) - fill) / std
        test[col] = (test[col].fillna(fill) - fill) / std
        stats[col] = {"mean": fill, "std": std}
    return train, val, test, stats


class LatentTransitionDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        clinical_columns: list[str],
        use_delta: bool = True,
    ):
        self.table = table.reset_index(drop=True)
        self.clinical_columns = clinical_columns
        self.use_delta = use_delta

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        source = np.load(row["source_embedding_path"]).astype(np.float32)
        target = np.load(row["target_embedding_path"]).astype(np.float32)
        clinical = (
            row[self.clinical_columns].to_numpy(dtype=np.float32)
            if self.clinical_columns
            else np.zeros(0, dtype=np.float32)
        )
        delta = (
            np.asarray([float(row["delta_days"]) / 365.25], dtype=np.float32)
            if self.use_delta
            else np.zeros(0, dtype=np.float32)
        )

        return {
            "source": torch.from_numpy(source),
            "target": torch.from_numpy(target),
            "clinical": torch.from_numpy(clinical),
            "delta": torch.from_numpy(delta),
            "patient_id": str(row["patient_id"]),
            "source_series_uid": str(row["source_series_uid"]),
            "target_series_uid": str(row["target_series_uid"]),
            "source_embedding_path": str(row["source_embedding_path"]),
            "target_embedding_path": str(row["target_embedding_path"]),
            "source_timepoint": str(row.get("source_timepoint", "")),
            "target_timepoint": str(row.get("target_timepoint", "")),
        }


class LatentTransitionDataModule(LightningDataModule):
    def __init__(
        self,
        transitions_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
        val_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 42,
        pin_memory: bool = False,
        use_clinical: bool = True,
        use_delta: bool = True,
    ):
        super().__init__()
        self.transitions_path = Path(transitions_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed
        self.pin_memory = pin_memory
        self.use_clinical = use_clinical
        self.use_delta = use_delta
        self.clinical_columns: list[str] = []
        self.feature_stats: dict[str, dict[str, float]] = {}
        self.embedding_dim: int | None = None
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

    @property
    def clinical_dim(self) -> int:
        return len(self.clinical_columns)

    @property
    def delta_dim(self) -> int:
        return 1 if self.use_delta else 0

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.transitions_path)
        df = df.dropna(
            subset=["source_embedding_path", "target_embedding_path", "delta_days"],
        ).copy()
        if df.empty:
            raise RuntimeError("No complete transitions found. Extract embeddings first.")

        self.clinical_columns = select_numeric_clinical_columns(df) if self.use_clinical else []
        train_df, val_df, test_df = split_by_patient(
            df,
            self.val_fraction,
            self.test_fraction,
            self.seed,
        )
        train_df, val_df, test_df, self.feature_stats = normalize_features(
            train_df,
            val_df,
            test_df,
            self.clinical_columns,
        )
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        sample_embedding = np.load(train_df.iloc[0]["source_embedding_path"])
        self.embedding_dim = int(sample_embedding.shape[-1])

    def train_dataloader(self) -> DataLoader:
        if self.train_df is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            LatentTransitionDataset(self.train_df, self.clinical_columns, self.use_delta),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_df is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        if len(self.val_df) == 0:
            return None
        return DataLoader(
            LatentTransitionDataset(self.val_df, self.clinical_columns, self.use_delta),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_df is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        if len(self.test_df) == 0:
            return None
        return DataLoader(
            LatentTransitionDataset(self.test_df, self.clinical_columns, self.use_delta),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
