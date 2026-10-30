#!/usr/bin/env python3
"""Summary figure for E3 with DINOv3: label-efficiency curves at PCA8 and PCA16."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/lung1_e3_frozen_linear/e3_dinov3_summary.png"
FM_NAMES = {"colipri": "COLIPRI", "tangerine": "TANGERINE",
            "radjepa": "Rad-JEPA-3D", "ctclip": "CT-CLIP"}
BASELINES = {"stage_volume": ("Stage + tumor volume", "#eb6834"),
             "radiomics": ("GTV radiomics", "#1baf7a")}
DINO = "#2a78d6"
CONTEXT = "#a3a29b"
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a64", "#e6e5e0", "#fcfcfb"
DEATHS = (29, 74, 148, 295)


def main() -> None:
    base = pd.read_csv(ROOT / "outputs/lung1_e3_baselines/summary.csv")
    base_ci = json.loads((ROOT / "outputs/lung1_e3_baselines/bootstrap_intervals.json").read_text())["intervals"]
    panels = (("PCA 8 (primary)", "outputs/lung1_e3_frozen_linear"),
              ("PCA 16 (post hoc sensitivity)", "outputs/lung1_e3_frozen_linear_pca16"))
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for axis, (title, folder) in zip(axes, panels):
        axis.set_facecolor(SURFACE)
        summary = pd.read_csv(ROOT / folder / "summary.csv")
        ci = json.loads((ROOT / folder / "paired_uno_bootstrap.json").read_text())["model_budget"]
        ends = []
        for model, name in FM_NAMES.items():
            rows = summary[summary.model == model].sort_values("budget_pct")
            axis.plot(rows.events_train_full, rows.uno_c_36m, color=CONTEXT, linewidth=1.5,
                      marker="o", markersize=4, zorder=2)
            ends.append([rows.uno_c_36m.iloc[-1], name, rows.events_train_full.iloc[-1]])
        # Spread end labels so close curves stay readable.
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.0055)
        for y, name, x in ends:
            axis.annotate(name, (x, y), xytext=(7, 0), textcoords="offset points", va="center",
                          fontsize=8, color=MUTED)
        for model, (name, color) in BASELINES.items():
            rows = base[base.model == model].sort_values("budget_pct")
            axis.plot(rows.events_train_full, rows.uno_c_36m, color=color, linewidth=2,
                      linestyle=(0, (5, 3)), marker="s", markersize=5, zorder=3, label=name)
        rows = summary[summary.model == "dinov3"].sort_values("budget_pct")
        low = [ci["dinov3"][str(b)]["uno_c_36m_ci95"][0] for b in rows.budget_pct]
        high = [ci["dinov3"][str(b)]["uno_c_36m_ci95"][1] for b in rows.budget_pct]
        axis.fill_between(rows.events_train_full, low, high, color=DINO, alpha=0.14, linewidth=0, zorder=1)
        axis.plot(rows.events_train_full, rows.uno_c_36m, color=DINO, linewidth=2.5, marker="o",
                  markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                  label="DINOv3 (95% band)")
        for x, y in zip(rows.events_train_full, rows.uno_c_36m):
            axis.annotate(f"{y:.3f}", (x, y), xytext=(0, -14), textcoords="offset points",
                          ha="center", fontsize=8, color=INK, zorder=5,
                          bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.9))
        axis.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
        axis.set_xscale("log", base=2)
        axis.set_xticks(DEATHS, labels=[f"{d}\n({p}%)" for d, p in zip(DEATHS, (10, 25, 50, 100))])
        axis.minorticks_off()
        axis.set_xlim(24, 470)
        axis.set_title(title, loc="left", fontsize=10, color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xlabel("Observed deaths in the training subset (label budget)")
    axes[0].set_ylabel("Held-out Uno C at 36 months (mean of 5 folds)")
    axes[0].set_ylim(0.49, 0.63)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=CONTEXT, linewidth=1.5, marker="o", markersize=4))
    labels.append("Other frozen FMs (labelled)")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("E3 label efficiency on LUNG1: DINOv3 vs 3-D CT foundation models and baselines",
                 x=0.01, y=1.07, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.06, "Frozen embeddings, linear Cox, 5 locked outer folds, 10 nested draws per budget below 100%. "
             "Baselines use all covariates (no PCA), identical in both panels. Band: conditional patient bootstrap.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(OUTPUT)


if __name__ == "__main__":
    main()
