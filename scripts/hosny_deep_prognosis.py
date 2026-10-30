#!/usr/bin/env python3
"""PyTorch port of the Hosny et al. 2018 (PLOS Medicine) radiotherapy prognosis CNN.

Weights: modelhub-ai/deep-prognosis (Keras 1.2.2, TensorFlow dim ordering, MIT licence),
trained on HarvardRT, tuned on Radboud, tested on Maastro (= LUNG1). Input is a
50 x 50 x 50 patch (central crop of a 150^3, 1 mm array centred on the tumour),
HU mapped from [-1024, 3071] to [0, 1]. Output is softmax(deceased, alive) for
2-year survival; `features` returns the 256-d dense_2 output (Keras layer 23),
the vector the modelhub wrapper exports.

Run directly to check the port against the authors' published LUNG1 outputs.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1] / "external/hosny_deep_prognosis/repo"
WEIGHTS = REPO / "contrib_src/model/weights.h5"


def normalize(hu: np.ndarray) -> np.ndarray:
    return (hu.astype(np.float32) + 1024.0) / 4095.0


class HosnyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv3d(1, 64, 5), nn.Conv3d(64, 128, 3),
                                    nn.Conv3d(128, 256, 3), nn.Conv3d(256, 512, 3)])
        self.conv_bns = nn.ModuleList([nn.BatchNorm3d(c, eps=1e-3) for c in (64, 128, 256, 512)])
        self.dense = nn.ModuleList([nn.Linear(13824, 512), nn.Linear(512, 256), nn.Linear(256, 2)])
        self.dense_bns = nn.ModuleList([nn.BatchNorm1d(c, eps=1e-3) for c in (512, 256, 2)])
        self.act = nn.LeakyReLU(0.1)
        self.drop_conv, self.drop_dense = nn.Dropout(0.25), nn.Dropout(0.5)
        self.pool = nn.MaxPool3d(3)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """x: (batch, 50, 50, 50) in the Keras array order, already normalized."""
        x = x.unsqueeze(1)
        for i, (conv, bn) in enumerate(zip(self.convs, self.conv_bns)):
            x = self.act(bn(conv(x)))
            if i in (1, 3):
                x = self.pool(x)
            x = self.drop_conv(x)
        x = x.permute(0, 2, 3, 4, 1).flatten(1)  # Keras flattens channels-last
        x = self.drop_dense(self.act(self.dense_bns[0](self.dense[0](x))))
        features = self.dense[1](x)
        x = self.drop_dense(self.act(self.dense_bns[1](features)))
        logits = self.dense_bns[2](self.dense[2](x))
        return (logits, features) if return_features else logits


def load(weights: Path = WEIGHTS) -> HosnyCNN:
    model = HosnyCNN()
    with h5py.File(weights, "r") as f:
        def get(layer, name):
            return torch.from_numpy(np.asarray(f[layer][f"{layer}_{name}:0"]))
        for i, conv in enumerate(model.convs, start=1):
            conv.weight.data = get(f"convolution3d_{i}", "W").permute(4, 3, 0, 1, 2).contiguous()
            conv.bias.data = get(f"convolution3d_{i}", "b")
        for i, dense in enumerate(model.dense, start=1):
            dense.weight.data = get(f"dense_{i}", "W").T.contiguous()
            dense.bias.data = get(f"dense_{i}", "b")
        for i, bn in enumerate([*model.conv_bns, *model.dense_bns], start=1):
            name = f"batchnormalization_{i}"
            bn.weight.data, bn.bias.data = get(name, "gamma"), get(name, "beta")
            bn.running_mean.data = get(name, "running_mean")
            bn.running_var.data = get(name, "running_std")  # Keras 1 stores the variance under this name
    return model.eval()


def main() -> None:
    import pandas as pd
    model = load()
    published = pd.read_csv(REPO / "data/Maastro.csv", usecols=["id", "logit_0", "logit_1"])
    published["id"] = published.id.str.strip()
    published = published.set_index("id")
    for path in sorted((REPO / "contrib_src/sample_data").glob("*.npy")):
        patch = normalize(np.load(path)[50:100, 50:100, 50:100])
        with torch.no_grad():
            prob = torch.softmax(model(torch.from_numpy(patch)[None]), dim=1)[0].numpy()
        ref = published.loc[path.stem, ["logit_0", "logit_1"]].to_numpy(float) if path.stem in published.index else None
        print(f"{path.stem}: port {prob.round(6)}  published {None if ref is None else ref.round(6)}")


if __name__ == "__main__":
    main()
