"""Figures for docs/lung1_fm_sintesi_finale.md (summary of all FM experiments).

Reads saved outputs only; no model is refitted. Values that exist only in the
reports (E2 adaptation table, E4 token masking) are copied from those reports.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "fm_sintesi"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill, light mode) + neutral gray for weak models.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY, INK, INK2, GRID = "#9a9893", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#9ec5f4", "#4a90e2", "#1c4f91"]  # one hue, light -> dark (low/mid/high risk)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

NAMES = {"colipri": "COLIPRI", "tangerine": "TANGERINE", "dinov3": "DINOv3",
         "radjepa": "Rad-JEPA-3D", "ctclip": "CT-CLIP", "pai": "Pai 2024",
         "hosny": "Hosny 2018 (frozen)", "hosny_zero_shot": "Hosny 2018 zero-shot"}


def fig_forest():
    ladder = pd.read_csv(ROOT / "outputs/lung1_ceiling_ladder/primary_tumour_centred/summary.csv")
    ladder = ladder.set_index("model")
    ci = json.load(open(ROOT / "outputs/lung1_e3_baselines/bootstrap_intervals.json"))["intervals"]
    base = pd.read_csv(ROOT / "outputs/lung1_e3_baselines/summary.csv")
    base = base[base.budget_pct == 100].set_index("model")
    rows = []  # (label, c, lo, hi, color, group)
    def add(key, label, color, group):
        r = ladder.loc[key]
        rows.append((label, r.uno_c_36m, r.ci95_lower, r.ci95_upper, color, group))
    add("C0", "C0  clinica (età, sesso, T, N, stadio, istologia)", ORANGE, "Clinica")
    add("C1", "C1  clinica + volume GTV", ORANGE, "Clinica")
    for k, lab in [("stage_volume", "Stadio + volume GTV"), ("radiomics", "Radiomica GTV (107 feature)")]:
        rows.append((lab, base.loc[k, "uno_c_36m"], *ci[k]["100"], AQUA if k == "radiomics" else ORANGE,
                     "Clinica" if k == "stage_volume" else "Centrati sul tumore"))
    add("pai", "Pai 2024 (FM sul patch tumorale)", AQUA, "Centrati sul tumore")
    add("hosny", "Hosny 2018 (CNN prognostica, frozen)", AQUA, "Centrati sul tumore")
    add("hosny_zero_shot", "Hosny 2018 zero-shot", AQUA, "Centrati sul tumore")
    for k in ["colipri", "tangerine", "dinov3"]:
        add(k, NAMES[k], BLUE, "FM su CT intera")
    for k in ["radjepa", "ctclip"]:
        add(k, NAMES[k], GRAY, "FM su CT intera")
    add("C3-colipri", "C1 + COLIPRI", VIOLET, "Combinazioni")
    add("C3-tangerine", "C1 + TANGERINE", VIOLET, "Combinazioni")
    add("C4-colipri", "C1 + radiomica + COLIPRI", VIOLET, "Combinazioni")

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, c, lo, hi, col, _) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(c, yi, "o", ms=8, color=col, mec="white", mew=1.5)
        ax.text(hi + 0.004, yi, f"{c:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.axvline(ladder.loc["colipri", "uno_c_36m"], color=BLUE, lw=1, ls=":", alpha=0.8)
    ax.set_xlim(0.47, 0.69)
    ax.set_xlabel("Uno C a 36 mesi (media 5 fold, IC 95% bootstrap)")
    ax.grid(axis="y", visible=False)
    prev = None
    for yi, r in zip(y, rows):
        if prev is not None and r[5] != prev:
            ax.axhline(yi + 0.5, color=GRID, lw=1)
        prev = r[5]
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker="o", lw=2, label=l) for c, l in
         [(ORANGE, "Clinica"), (AQUA, "Centrati sul tumore"), (BLUE, "FM generaliste utili"),
          (GRAY, "FM deboli (~caso)"), (VIOLET, "Combinazioni")]]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.35, -0.08), ncol=3, fontsize=8.5, frameon=False)
    ax.set_title("LUNG1 interno (416 pazienti): tutti i metodi sugli stessi 5 fold", loc="left", fontsize=11)
    fig.savefig(OUT / "fig1_forest_lung1.png")
    plt.close(fig)


def fig_adaptation():
    # Pooled Uno C 36 m, selected-checkpoint protocol (lung1_foundation_models_training_preprocessing_results.md)
    rungs = ["Linear Cox", "MLP Cox", "SQA (token)", "LoRA", "Partial tuning"]
    models = ["COLIPRI", "TANGERINE", "DINOv3", "Rad-JEPA-3D", "CT-CLIP"]
    v = np.array([
        [0.619, 0.616, 0.605, 0.535, 0.490],
        [0.595, 0.611, 0.608, 0.520, 0.509],
        [0.546, 0.561, 0.533, 0.489, 0.506],
        [0.528, 0.530, 0.499, 0.492, 0.499],
        [0.513, 0.485, 0.501, 0.517, 0.514],
    ])
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(rungs))
    cols = [BLUE, VIOLET, AQUA, ORANGE, GRAY]
    for m, row, c, mk in zip(models, v, cols, ["o", "s", "D", "^", "v"]):
        ax.plot(x, row, marker=mk, color=c, lw=2, ms=8, mec="white", mew=1.2, label=m)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.axhline(0.557, color=ORANGE, lw=1, ls=":")
    ax.text(4.25, 0.559, "stadio + volume", fontsize=8, color=INK2, ha="right", va="bottom")
    ax.axvspan(2.5, 4.5, color="#f3f2ef", zorder=0)
    ax.text(3.5, 0.625, "encoder modificato", ha="center", fontsize=8.5, color=INK2)
    ax.text(1.0, 0.625, "encoder congelato", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, rungs)
    ax.set_xlim(-0.3, 4.5)
    ax.set_ylim(0.47, 0.635)
    ax.set_ylabel("Uno C 36 m (pooled)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Più capacità addestrabile non aiuta: il fine-tuning porta tutti al caso", loc="left", fontsize=11)
    fig.savefig(OUT / "fig2_adattamento.png")
    plt.close(fig)


def fig_learning_curves():
    fm = pd.read_csv(ROOT / "outputs/lung1_e3_frozen_linear_pca16/summary.csv")
    bl = pd.read_csv(ROOT / "outputs/lung1_e3_baselines/summary.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    spec = [("colipri", fm, BLUE, "-", "o"), ("tangerine", fm, VIOLET, "-", "s"),
            ("dinov3", fm, BLUE, "--", "D"), ("pai", fm, AQUA, "-", "^"),
            ("hosny", fm, AQUA, "--", "v"), ("stage_volume", bl, ORANGE, "-", "P"),
            ("radiomics", bl, ORANGE, "--", "X"), ("radjepa", fm, GRAY, "-", "<"),
            ("ctclip", fm, GRAY, "--", ">")]
    labels = {**NAMES, "stage_volume": "Stadio + volume", "radiomics": "Radiomica GTV"}
    for k, df, c, ls, mk in spec:
        d = df[df.model == k].sort_values("budget_pct")
        ax.plot(d.events_train_full, d.uno_c_36m, ls=ls, marker=mk, color=c, lw=2, ms=7,
                mec="white", mew=1, label=labels[k])
    ax.set_xscale("log", base=2)
    ev = fm[fm.model == "colipri"].sort_values("budget_pct")
    ax.set_xticks(ev.events_train_full, [f"{p}%\n({e:.0f} decessi)" for p, e in zip(ev.budget_pct, ev.events_train_full)])
    ax.minorticks_off()
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.set_ylabel("Uno C 36 m (media 5 fold)")
    ax.set_xlabel("Quota del training set usata (decessi osservati)")
    ax.legend(fontsize=8, frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("E3 – efficienza delle etichette (PCA16, Cox lineare)", loc="left", fontsize=11)
    fig.savefig(OUT / "fig3_curve_apprendimento.png")
    plt.close(fig)


def fig_probing():
    p = pd.read_csv(ROOT / "outputs/lung1_embedding_probing/summary.csv")
    p = p[p.capacity == "pca16"]
    targets = [("age", "Età (ρ)"), ("male", "Sesso (AUC)"), ("log_gtv_volume", "Volume GTV (ρ)"),
               ("T3_T4", "T3–T4 (AUC)"), ("N2_N3", "N2–N3 (AUC)"), ("stage_III", "Stadio III (AUC)"),
               ("squamous", "Squamoso (AUC)")]
    models = ["colipri", "tangerine", "dinov3", "radjepa", "ctclip"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, cond, title in zip(axes, ["original", "gtv_masked"], ["CT originale", "Token del tumore nascosti"]):
        m = np.full((len(models), len(targets)), np.nan)
        for i, mo in enumerate(models):
            for j, (t, _) in enumerate(targets):
                r = p[(p.model == mo) & (p.condition == cond) & (p.target == t)]
                if len(r):
                    val = r.value.iloc[0]
                    # put correlations and AUCs on one 0-1 "above chance" scale for colour only
                    m[i, j] = val if r.metric.iloc[0] == "spearman" else 2 * (val - 0.5)
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if m[i, j] > 0.55 else INK)
        ax.imshow(np.clip(m, 0, 1), cmap="Blues", vmin=0, vmax=1.05, aspect="auto")
        ax.set_xticks(range(len(targets)), [t[1] for t in targets], rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(models)), [NAMES[mo] for mo in models])
        ax.grid(False)
        ax.set_title(title, fontsize=10, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Cosa codificano gli embedding (probe lineare, PCA16; colore = distanza dal caso)",
                 x=0.06, ha="left", fontsize=11)
    fig.savefig(OUT / "fig4_probing.png")
    plt.close(fig)


def fig_masking():
    # lung1_e4_token_masking_report.md, global readout, 416 patients, original - hidden
    models = ["COLIPRI", "TANGERINE", "DINOv3", "Rad-JEPA-3D", "CT-CLIP"]
    d = [(0.011, -0.007, 0.028), (0.006, 0.001, 0.011), (-0.004, -0.016, 0.007),
         (-0.009, -0.017, -0.001), (0.001, -0.005, 0.008)]
    l2 = [0.508, 0.002, 0.017, 0.027, 0.023]       # GTV hidden, global, 403 pts
    l2c = [0.018, 0.002, 0.016, 0.027, 0.024]      # count-matched control
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    y = np.arange(len(models))[::-1]
    for yi, (c, lo, hi) in zip(y, d):
        a1.plot([lo, hi], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
        a1.plot(c, yi, "o", color=BLUE, ms=8, mec="white", mew=1.5)
    a1.axvline(0, color=INK2, lw=1)
    a1.axvspan(-0.03, 0.03, color="#eef4fc", zorder=0)
    a1.text(0.029, y[-1] - 0.45, "margine ±0.03", ha="right", fontsize=8, color=INK2)
    a1.set_xlim(-0.05, 0.05)
    a1.set_yticks(y, models)
    a1.set_xlabel("Δ Uno C (originale − tumore nascosto)")
    a1.set_title("Effetto sulla prognosi: ~0", loc="left", fontsize=10)
    a1.grid(axis="y", visible=False)
    h = 0.36
    a2.barh(y + h / 2, l2, height=h, color=BLUE, label="tumore nascosto")
    a2.barh(y - h / 2, l2c, height=h, color=GRAY, label="controllo (stesso n. di token altrove)")
    for yi, v in zip(y, l2):
        a2.text(v + 0.01, yi + h / 2, f"{v:.0%}" if v >= 0.01 else f"{v:.1%}", va="center", fontsize=8, color=INK2)
    a2.set_xlabel("Variazione relativa dell'embedding (L2, mediana)")
    a2.set_title("Effetto sulla rappresentazione", loc="left", fontsize=10)
    a2.set_xlim(0, 0.62)
    a2.legend(fontsize=8, frameon=False, loc="lower right")
    a2.grid(axis="y", visible=False)
    fig.subplots_adjust(top=0.82)
    fig.suptitle("E4 – nascondere ~0.3% dei token (quelli del tumore) dentro l'encoder",
                 x=0.06, ha="left", fontsize=11)
    fig.savefig(OUT / "fig5_mascheramento_tumore.png")
    plt.close(fig)


GROUP = {"C0": ORANGE, "C1": ORANGE, "baseline_stage_volume": ORANGE, "C2": AQUA,
         "baseline_radiomics": AQUA, "pai": AQUA, "hosny": AQUA, "hosny_zero_shot": AQUA,
         "C3-pai": AQUA, "C3-hosny_score": AQUA, "colipri": BLUE, "tangerine": BLUE, "dinov3": BLUE,
         "radjepa": GRAY, "ctclip": GRAY}


def fig_external():
    ext = json.load(open(ROOT / "outputs/phase4_external/external_validation.json"))["populations"]["primary"]["models"]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    lab = {"colipri": "COLIPRI", "tangerine": "TANGERINE", "dinov3": "DINOv3", "radjepa": "Rad-JEPA",
           "ctclip": "CT-CLIP", "pai": "Pai", "hosny": "Hosny frozen", "hosny_zero_shot": "Hosny 0-shot",
           "C1": "C1", "C0": "C0", "baseline_stage_volume": "stadio+vol"}
    for k, r in ext.items():
        col = GROUP.get(k, VIOLET)
        xi, yi = r["internal_lung1_uno_c_36m"], r["uno_c_36m"]
        ax.plot(xi, yi, "o", ms=8, color=col, mec="white", mew=1.3)
        if k in lab:
            ax.annotate(lab[k], (xi, yi), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK2)
    lims = [0.46, 0.70]
    ax.plot(lims, lims, color=INK2, lw=1, ls="--")
    ax.text(0.62, 0.612, "y = x", fontsize=8, color=INK2)
    ax.axhline(0.5, color=GRID, lw=1.2)
    ax.set_xlim(0.51, 0.63)
    ax.set_ylim(lims)
    ax.set_xlabel("Uno C interna (LUNG1, CV)")
    ax.set_ylabel("Uno C esterna (Radiogenomics, n = 143)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker="o", lw=0, label=l) for c, l in
         [(ORANGE, "Clinica"), (AQUA, "Centrati sul tumore / radiomica"), (BLUE, "FM generaliste utili"),
          (GRAY, "FM deboli (~caso)"), (VIOLET, "Combinazioni C3/C4")]]
    ax.legend(handles=h, fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Validazione esterna: l'ordinamento si trasferisce (ρ = 0.67)", loc="left", fontsize=11)
    fig.savefig(OUT / "fig6_interna_vs_esterna.png")
    plt.close(fig)


def km_panel(ax, df, title):
    """df: time_days, event, group in {0,1,2} (0 = lowest predicted risk)."""
    kmf = KaplanMeierFitter()
    names = ["rischio basso", "rischio intermedio", "rischio alto"]
    for g in range(3):
        s = df[df.group == g]
        kmf.fit(s.time / 365.25, s.event, label=f"{names[g]} (n={len(s)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=SEQ[g], lw=2)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Anni")
    ax.set_ylabel("Sopravvivenza globale")
    ax.set_title(title, loc="left", fontsize=10)
    from lifelines.statistics import multivariate_logrank_test
    p = multivariate_logrank_test(df.time, df.group, df.event).p_value
    ax.text(0.03, 0.04, f"log-rank p = {p:.1g}" if p >= 1e-4 else "log-rank p < 0.0001",
            transform=ax.transAxes, fontsize=8, color=INK2)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right" if df.event.mean() > 0.6 else "lower left",
              bbox_to_anchor=None if df.event.mean() > 0.6 else (0.0, 0.08))


def fig_km():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    # LUNG1: out-of-fold COLIPRI (E3 PCA16, 100%), tertiles inside each test fold.
    oof = pd.read_csv(ROOT / "outputs/lung1_e3_frozen_linear_pca16/out_of_fold_predictions.csv")
    oof = oof[(oof.model == "colipri") & (oof.budget_pct == 100)].copy()
    oof["group"] = oof.groupby("fold").risk.transform(lambda r: pd.qcut(r.rank(method="first"), 3, labels=False))
    km_panel(axes[0], oof.rename(columns={"os_time_days": "time", "os_event": "event"}),
             "LUNG1 interno – COLIPRI (out-of-fold)")
    # Radiogenomics secondary population (199), frozen models, tertiles of external risk.
    pred = pd.read_csv(ROOT / "outputs/phase4_external/predictions.csv")
    coh = pd.read_csv(ROOT / "data/derived/radiogenomics/cohort_with_gtv_volume.csv")[["patient_id", "os_time_days", "os_event"]]
    for ax, mo in zip(axes[1:], ["tangerine", "colipri"]):
        d = pred[(pred.population == "secondary") & (pred.model == mo)].merge(coh, on="patient_id")
        d["group"] = pd.qcut(d.risk.rank(method="first"), 3, labels=False)
        km_panel(ax, d.rename(columns={"os_time_days": "time", "os_event": "event"}),
                 f"Radiogenomics esterno – {NAMES[mo]} (n={len(d)})")
    fig.suptitle("Curve di Kaplan-Meier per terzili di rischio predetto", x=0.05, ha="left", fontsize=11)
    fig.savefig(OUT / "fig7_kaplan_meier.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_forest()
    fig_adaptation()
    fig_learning_curves()
    fig_probing()
    fig_masking()
    fig_external()
    fig_km()
    print("figures in", OUT)
