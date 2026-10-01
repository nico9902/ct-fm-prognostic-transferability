"""Generate figures for the Italian project-status presentation.

The figures deliberately use only values already reported in the repository's
completed result documents. They are presentation aids, not new analyses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation_assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17324D"
TEAL = "#167D8D"
CYAN = "#6FC2D0"
ORANGE = "#E07A3F"
GOLD = "#E9B44C"
RED = "#C84A4A"
GREEN = "#4B9B72"
LIGHT = "#EEF3F6"
MID = "#B9C8D3"
GRAY = "#5D6B78"
WHITE = "#FFFFFF"


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "axes.edgecolor": MID,
            "axes.linewidth": 0.8,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "text.color": NAVY,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def box(ax, xy, wh, text, color, *, fontsize=13, subtext=None, text_color=WHITE):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=color,
        edgecolor="none",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.58 if subtext else 0.5), text, ha="center", va="center", color=text_color, fontsize=fontsize, fontweight="bold")
    if subtext:
        ax.text(x + w / 2, y + h * 0.27, subtext, ha="center", va="center", color=text_color, fontsize=fontsize - 2)
    return patch


def arrow(ax, start, end, color=MID, lw=2.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=lw, color=color))


def cohort_flow() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (0.02, "Coorte selezionata", "1.000 pazienti\n3.000 scansioni", NAVY),
        (0.27, "Coorte analizzata", "999 pazienti\n2.984 scansioni", TEAL),
        (0.52, "Coppie longitudinali", "998 pazienti\n1.984 coppie", ORANGE),
        (0.77, "Triplette complete", "986 pazienti\nT0–T1–T2", GREEN),
    ]
    for i, (x, title, subtitle, color) in enumerate(stages):
        box(ax, (x, 0.29), (0.20, 0.42), title, color, fontsize=13, subtext=subtitle)
        if i < len(stages) - 1:
            arrow(ax, (x + 0.20, 0.50), (stages[i + 1][0] - 0.015, 0.50))
    ax.text(0.5, 0.10, "Split fisso a livello paziente: 599 train / 200 validation / 200 test", ha="center", color=GRAY, fontsize=13)
    save(fig, "cohort_flow.png")


def label_timeline() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 2.7)
    ax.axis("off")
    ax.plot([0.6, 9.3], [0, 0], color=NAVY, lw=3)
    for x, label in [(1.4, "t\nscan corrente"), (4.1, "t+1\nscan futura osservata"), (8.4, "orizzonte\n1/2/3 anni")]:
        ax.scatter([x], [0], s=180, color=TEAL if x < 5 else ORANGE, zorder=3)
        ax.text(x, -0.45, label, ha="center", va="top", fontsize=12, fontweight="bold")
    ax.add_patch(FancyBboxPatch((4.25, 0.55), 4.0, 0.75, boxstyle="round,pad=0.02", facecolor=LIGHT, edgecolor=TEAL, linewidth=2))
    ax.text(6.25, 0.93, "Label primaria: diagnosi dopo t+1", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.annotate("finestra di outcome", xy=(8.25, 0.25), xytext=(6.2, 1.55), arrowprops=dict(arrowstyle="->", color=TEAL, lw=2), ha="center", color=TEAL)
    ax.text(1.4, 1.9, "Input disponibile", ha="center", fontsize=13, color=NAVY)
    ax.text(4.1, 1.9, "Target latente / oracle", ha="center", fontsize=13, color=NAVY)
    ax.text(5.0, -1.1, "1 = diagnosi entro H   •   0 = evento dopo H o follow-up cancer-free ≥ H   •   NA = follow-up insufficiente", ha="center", fontsize=12, color=GRAY)
    save(fig, "label_timeline.png")


def fm_comparison() -> None:
    targets = ["Sospetto\nisolato", "Nodulo\n≥4 mm", "Enfisema", "Fibrosi /\nscarring"]
    rad = [0.507, 0.504, 0.555, 0.554]
    tang = [0.542, 0.528, 0.700, 0.603]
    coli = [0.795, 0.788, 0.752, 0.679]
    x = np.arange(len(targets))
    width = 0.23
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    bars = [
        ax.bar(x - width, rad, width, label="Rad-JEPA", color=MID),
        ax.bar(x, tang, width, label="TANGERINE", color=CYAN),
        ax.bar(x + width, coli, width, label="COLIPRI mean", color=TEAL),
    ]
    ax.axhline(0.5, color=GRAY, lw=1.3, ls="--")
    ax.set_ylim(0.45, 0.86)
    ax.set_ylabel("AUROC test")
    ax.set_xticks(x, targets)
    ax.set_title("Decodificabilità diagnostica con encoder congelati")
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.grid(axis="y", color=LIGHT, lw=1)
    for group in bars:
        for b in group:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9, color=NAVY)
    fig.tight_layout()
    save(fig, "foundation_model_comparison.png")


def colipri_representation() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        (0.03, 0.68, "CT preprocessata", "192³, 2 mm isotropica", NAVY),
        (0.28, 0.68, "COLIPRI congelato", "dense grid 24³×864", TEAL),
        (0.53, 0.68, "Mean pooling", "z ∈ ℝ⁸⁶⁴", GREEN),
        (0.78, 0.68, "Probe lineare", "diagnosi / prognosi", ORANGE),
    ]
    for i, (x, y, title, subtitle, color) in enumerate(stages):
        box(ax, (x, y - 0.16), (0.19, 0.32), title, color, fontsize=12, subtext=subtitle)
        if i < 3:
            arrow(ax, (x + 0.19, y), (stages[i + 1][0] - 0.01, y))
    box(ax, (0.30, 0.10), (0.22, 0.27), "Pooling spaziale", TEAL, fontsize=12, subtext="4³ token × 864")
    box(ax, (0.57, 0.10), (0.22, 0.27), "World model spaziale", ORANGE, fontsize=12, subtext="64 token nativi")
    arrow(ax, (0.375, 0.52), (0.41, 0.37), color=TEAL)
    arrow(ax, (0.52, 0.235), (0.57, 0.235), color=MID)
    ax.text(0.5, 0.01, "Sanity check: mean(64 token) ≃ embedding COLIPRI mean-pooled (MSE 1,32×10⁻¹⁶)", ha="center", fontsize=11, color=GRAY)
    save(fig, "colipri_representation.png")


def oracle_gap() -> None:
    horizons = ["1 anno", "2 anni", "3 anni"]
    current = [0.7154, 0.6540, 0.6403]
    predicted = [0.7206, 0.6663, 0.6531]
    future = [0.8701, 0.7419, 0.6719]
    x = np.arange(3)
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    groups = [
        ax.bar(x - width, current, width, label="Stato corrente zₜ", color=MID),
        ax.bar(x, predicted, width, label="Miglior futuro predetto", color=ORANGE),
        ax.bar(x + width, future, width, label="Futuro reale zₜ₊₁", color=TEAL),
    ]
    ax.set_ylim(0.55, 0.92)
    ax.set_ylabel("AUROC test")
    ax.set_xticks(x, horizons)
    ax.set_title("Il futuro reale contiene un forte segnale: il modello ne recupera poco")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", color=LIGHT)
    for group in groups:
        for b in group:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.007, f"{b.get_height():.3f}", ha="center", fontsize=10)
    save(fig, "oracle_gap.png")


def world_model_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.03, 0.55), (0.16, 0.24), "zₜ", NAVY, fontsize=18, subtext="COLIPRI 864-d")
    box(ax, (0.03, 0.15), (0.16, 0.20), "Δt", GRAY, fontsize=17, subtext="giorni")
    box(ax, (0.27, 0.43), (0.25, 0.32), "Predittore residuale", ORANGE, fontsize=15, subtext="MLP o Transformer")
    box(ax, (0.61, 0.55), (0.16, 0.24), "Δ̂", GOLD, fontsize=18, subtext="variazione stimata", text_color=NAVY)
    box(ax, (0.82, 0.55), (0.15, 0.24), "ẑₜ₊₁", TEAL, fontsize=18, subtext="zₜ + Δ̂")
    arrow(ax, (0.19, 0.67), (0.27, 0.62))
    arrow(ax, (0.19, 0.25), (0.27, 0.52))
    arrow(ax, (0.52, 0.60), (0.61, 0.67))
    arrow(ax, (0.77, 0.67), (0.82, 0.67))
    arrow(ax, (0.11, 0.79), (0.88, 0.81), color=TEAL, lw=2)
    ax.text(0.48, 0.85, "skip connection: persistenza come prior", ha="center", color=TEAL, fontsize=12)
    ax.text(0.50, 0.26, "Target principale corretto: Δ standardizzato con statistiche train-only", ha="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(0.50, 0.08, "Nessuna label clinica entra nel training del world model", ha="center", fontsize=12, color=GRAY)
    save(fig, "world_model_architecture.png")


def forecasting_tradeoff() -> None:
    labels = ["Identità", "MLP raw", "Δ standardizzato", "Transformer\nspaziale nativo"]
    mse_rel = [100.0, 115.4, 91.6, 93.4]
    prognosis_1y = [0.715, 0.717, 0.722, 0.716]
    colors = [MID, RED, GREEN, ORANGE]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.1), gridspec_kw={"wspace": 0.34})
    y = np.arange(len(labels))
    ax1.barh(y, mse_rel, color=colors)
    ax1.axvline(100, color=NAVY, ls="--", lw=1.5)
    ax1.set_yticks(y, labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("MSE relativo all’identità (%)")
    ax1.set_title("Ricostruzione")
    ax1.grid(axis="x", color=LIGHT)
    for yi, v in zip(y, mse_rel):
        ax1.text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=10)
    ax2.barh(y, prognosis_1y, color=colors)
    ax2.axvline(0.870, color=TEAL, ls="--", lw=2, label="oracle futuro 0,870")
    ax2.set_yticks(y, [""] * len(y))
    ax2.invert_yaxis()
    ax2.set_xlim(0.68, 0.89)
    ax2.set_xlabel("AUROC prognosi 1 anno")
    ax2.set_title("Decodificabilità prognostica")
    ax2.grid(axis="x", color=LIGHT)
    ax2.legend(frameon=False, loc="lower right")
    for yi, v in zip(y, prognosis_1y):
        ax2.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=10)
    fig.suptitle("Migliorare MSE non equivale a recuperare il segnale clinico", fontsize=18, fontweight="bold", color=NAVY)
    save(fig, "forecasting_tradeoff.png")


def strict_diagnostics() -> None:
    targets = ["Sospetto", "Nodulo ≥4 mm", "Enfisema", "Fibrosi"]
    current = [0.698, 0.700, 0.752, 0.679]
    predicted = [0.706, 0.712, 0.753, 0.678]
    future = [0.790, 0.786, 0.743, 0.698]
    x = np.arange(4)
    width = 0.23
    fig, ax = plt.subplots(figsize=(11.3, 5.0))
    ax.bar(x - width, current, width, label="corrente", color=MID)
    ax.bar(x, predicted, width, label="spaziale predetto", color=ORANGE)
    ax.bar(x + width, future, width, label="futuro reale", color=TEAL)
    ax.set_ylim(0.62, 0.82)
    ax.set_xticks(x, targets)
    ax.set_ylabel("AUROC contro la stessa label t+1")
    ax.set_title("Recupero diagnostico futuro: piccoli guadagni sui target focali")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", color=LIGHT)
    save(fig, "strict_future_diagnostics.png")


def observed_progression() -> None:
    horizons = ["1 anno", "2 anni", "3 anni"]
    current = [0.7152, 0.6320, 0.6500]
    history = [0.6776, 0.6231, 0.6396]
    oracle = [0.8822, 0.7704, 0.6878]
    x = np.arange(3)
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    ax.bar(x - width, current, width, label="T1 corrente", color=MID)
    ax.bar(x, history, width, label="T1 + Δ osservato", color=ORANGE)
    ax.bar(x + width, oracle, width, label="T2 reale oracle", color=TEAL)
    ax.set_ylim(0.55, 0.92)
    ax.set_xticks(x, horizons)
    ax.set_ylabel("AUROC test")
    ax.set_title("Neppure la progressione realmente osservata T0→T1 migliora la prognosi post-T2")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", color=LIGHT)
    save(fig, "observed_progression.png")


def alignment_audit() -> None:
    labels = ["T0→T1", "T1→T2"]
    true = [0.823, 0.822]
    random = [0.433, 0.410]
    fixed = [0.989, 0.993]
    x = np.arange(2)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.bar(x - width, random, width, label="paziente casuale", color=MID)
    ax.bar(x, true, width, label="stessa posizione", color=TEAL)
    ax.bar(x + width, fixed, width, label="Hungarian: posizione fissa", color=GREEN)
    ax.set_ylim(0.3, 1.03)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cosine / quota di posizioni")
    ax.set_title("Audit di allineamento: la griglia 4³ è già stabile")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.grid(axis="y", color=LIGHT)
    save(fig, "alignment_audit.png")


def evidence_map() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.03, 0.57), (0.25, 0.27), "Cosa funziona", GREEN, fontsize=16, subtext="COLIPRI + futuro reale")
    box(ax, (0.375, 0.57), (0.25, 0.27), "Cosa abbiamo escluso", ORANGE, fontsize=15, subtext="solo scaling / PCA / alignment")
    box(ax, (0.72, 0.57), (0.25, 0.27), "Ipotesi residua", NAVY, fontsize=16, subtext="segnale focale e stocastico")
    arrow(ax, (0.28, 0.705), (0.375, 0.705))
    arrow(ax, (0.625, 0.705), (0.72, 0.705))
    ax.text(0.155, 0.38, "• diagnostica forte\n• oracle prognostico forte\n• identità difficile da battere", ha="center", va="top", fontsize=12, color=GRAY)
    ax.text(0.50, 0.38, "• MSE migliore ≠ AUROC migliore\n• Transformer ≈ MLP pooled\n• SIGReg non risolve", ha="center", va="top", fontsize=12, color=GRAY)
    ax.text(0.845, 0.38, "• cambiamenti piccoli/lesionali\n• vicinanza alla diagnosi\n• dati e confondimento temporale", ha="center", va="top", fontsize=12, color=GRAY)
    ax.text(0.5, 0.08, "Direzione: predire/rankare cambiamenti clinicamente rilevanti, non ricostruire tutto l’embedding", ha="center", fontsize=15, fontweight="bold", color=TEAL)
    save(fig, "evidence_map.png")


def main() -> None:
    setup()
    cohort_flow()
    label_timeline()
    fm_comparison()
    colipri_representation()
    oracle_gap()
    world_model_architecture()
    forecasting_tradeoff()
    strict_diagnostics()
    observed_progression()
    alignment_audit()
    evidence_map()
    print(f"Wrote presentation figures to {OUT}")


if __name__ == "__main__":
    main()
