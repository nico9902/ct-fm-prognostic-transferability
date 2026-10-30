#!/usr/bin/env python3
"""Build the Results figures (one per research question) and the trainable and external tables.

Reads aggregate outputs only. Missing results are shown as pending, so the script can be rerun
as experiments complete. Run from the manuscript directory: ../.venv/bin/python scripts/build_results_figures.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"

# Reference categorical palette (validated light-mode slots 1-5), fixed per encoder in every figure.
ENCODERS = ["colipri", "tangerine", "radjepa", "ctclip", "dinov3"]
NAMES = {"colipri": "COLIPRI", "tangerine": "TANGERINE", "radjepa": "Rad-JEPA-3D",
         "ctclip": "CT-CLIP", "dinov3": "DINOv3", "pai": "Pai lesion encoder",
         "hosny": "Hosny network"}
COLOR = {"colipri": "#2a78d6", "tangerine": "#eb6834", "radjepa": "#1baf7a",
         "ctclip": "#eda100", "dinov3": "#e87ba4"}
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f1f0ec"
MARGIN = 0.03  # equivalence margin in Uno C, fixed in advance

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def recessive(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=8.5, color=INK)


def interval(ax, y, est, lo, hi, colour, filled=True, size=4.5):
    ax.plot([lo, hi], [y, y], color=colour, linewidth=1.3, solid_capstyle="round")
    ax.plot(est, y, "o", markersize=size, markeredgewidth=1.0 if not filled else 0.8,
            color=colour if filled else "white", markeredgecolor=colour if not filled else "white")


def save(fig, name):
    fig.savefig(OUT / f"figures/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def ladder_summary():
    return {r["model"]: r for r in rows(O / "lung1_ceiling_ladder/primary_tumour_centred/summary.csv")}


# ---------------------------------------------------------------- Figure 1 (RQ1): frozen benchmark and clinical increment
def rq1_figure() -> None:
    ladder = ladder_summary()
    base = {(r["model"], r["budget_pct"]): float(r["uno_c_36m"]) for r in rows(O / "lung1_e3_baselines/summary.csv")}
    base_ci = json.load(open(O / "lung1_e3_baselines/bootstrap_intervals.json"))["intervals"]
    tests = {(r["family"], r["contrast"]): r for r in rows(O / "lung1_ceiling_ladder/primary_tumour_centred/tests.csv")}
    pca32 = json.load(open(O / "lung1_nested/pca_vs_baselines.json"))["standalone"]["pca32"]  # sensitivity head

    def lad(key):
        r = ladder[key]
        return float(r["uno_c_36m"]), float(r["ci95_lower"]), float(r["ci95_upper"])

    items = [  # (group, label, (estimate, lower, upper), encoder or None)
        ("Clinical", "Clinical", lad("C0"), None),
        ("Clinical", "Clinical+Volume", lad("C1"), None),
        ("Clinical", "Stage+Volume", (base[("stage_volume", "100")], *base_ci["stage_volume"]["100"]), None),
        ("Tumour-centred", "GTV radiomics", (base[("radiomics", "100")], *base_ci["radiomics"]["100"]), None),
        ("Tumour-centred", NAMES["pai"], lad("pai"), None),
        ("Tumour-centred", NAMES["hosny"], lad("hosny"), None),
        ("Tumour-centred", "Hosny zero-shot", lad("hosny_zero_shot"), None),
    ] + [("Frozen encoders", NAMES[e], lad(e), e) for e in ENCODERS]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    y, ticks, labels, previous = 0.0, [], [], None
    for group, label, (est, lo, hi), enc in items:
        if group != previous:
            y += 0.6 if previous is not None else 0
            ax.text(0.455, y, group, fontsize=7.5, fontweight="bold", color=INK, va="center")
            y += 1.0
            previous = group
        if enc:  # PCA16 (filled) above, PCA32 sensitivity head (open) below
            interval(ax, y - 0.18, est, lo, hi, COLOR[enc])
            interval(ax, y + 0.18, *pca32[enc], COLOR[enc], filled=False, size=4)
        else:
            interval(ax, y, est, lo, hi, INK2)
        ticks.append(y)
        labels.append(label)
        y += 1.0
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y + 0.9, -0.6)  # room for the head legend
    ax.set_xlim(0.455, 0.68)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="-", color=INK2, markersize=4, label="PCA16"),
                       Line2D([], [], marker="o", linestyle="-", color=INK2, markerfacecolor="white",
                              markersize=4, label="PCA32")],
              loc="lower center", ncol=2, fontsize=6.8, handlelength=1.4, columnspacing=0.8, borderaxespad=0.3)
    ax.set_xlabel("Uno C at 36 months (95% interval)")
    ax.tick_params(axis="y", length=0)
    recessive(ax)
    title(ax, "a  Discrimination")

    contrasts = [  # (label, family, contrast, encoder)
        ("Volume over Clinical", "ladder_primary", "C1 - C0", None),
        ("Radiomics over Clinical+Volume", "ladder_primary", "C2 - C1", None),
        ("COLIPRI over Clinical+Volume", "ladder_primary", "C3-colipri - C1", "colipri"),
        ("Radiomics over Clinical+Volume+COLIPRI", "ladder_primary", "C4-colipri - C3-colipri", None),
        ("COLIPRI over Clinical+Volume+Radiomics", "ladder_primary", "C4-colipri - C2", "colipri"),
        (None, None, None, None),
        ("TANGERINE over Clinical+Volume", "image_over_C1", "C3-tangerine - C1", "tangerine"),
        ("Rad-JEPA-3D over Clinical+Volume", "image_over_C1", "C3-radjepa - C1", "radjepa"),
        ("CT-CLIP over Clinical+Volume", "image_over_C1", "C3-ctclip - C1", "ctclip"),
        ("DINOv3 over Clinical+Volume", "image_over_C1", "C3-dinov3 - C1", "dinov3"),
        ("Pai encoder over Clinical+Volume", "tumour_centred_over_C1", "C3-pai - C1", None),
        ("Clinical+Volume over COLIPRI", "C1_over_image", "C3-colipri - colipri", "colipri"),
    ]
    bx.axvspan(-MARGIN, MARGIN, color=BAND, zorder=0)
    bx.axvline(0, color=INK2, linewidth=0.8)
    y, ticks, labels = 0.0, [], []
    for label, family, contrast, enc in contrasts:
        if label is None:
            bx.text(-0.062, y + 0.15, "Secondary", fontsize=7.5, fontweight="bold", color=INK, va="center")
            y += 1.0
            continue
        r = tests[(family, contrast)]
        d, lo, hi = (float(r[k]) for k in ("difference", "ci95_lower", "ci95_upper"))
        lo90, hi90 = float(r["ci90_lower"]), float(r["ci90_upper"])
        colour = COLOR[enc] if enc else INK2
        bx.plot([lo90, hi90], [y, y], color=colour, linewidth=3.0, solid_capstyle="butt", alpha=0.35)
        interval(bx, y, d, lo, hi, colour, filled=float(r["p_holm"]) < 0.05)
        verdict = "equivalent" if r["equivalent_within_margin"] == "True" else (
            f"$p_\\mathrm{{Holm}}$ {float(r['p_holm']):.3f}" if float(r["p_holm"]) < 0.05 else "inconclusive")
        bx.text(0.084, y, verdict, fontsize=6.8, color=INK2, va="center")
        ticks.append(y)
        labels.append(label)
        y += 1.0
    bx.text(-0.062, -1.0, "Primary", fontsize=7.5, fontweight="bold", color=INK, va="center")
    bx.set_yticks(ticks, labels)
    bx.set_ylim(y - 0.4, -1.6)
    bx.set_xlim(-0.062, 0.082)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Paired difference in Uno C")
    recessive(bx)
    title(bx, "b  Clinical increment")
    fig.tight_layout(w_pad=1.2)
    save(fig, "results_rq1")


# ---------------------------------------------------------------- Figure 2 (RQ2): label budget
def rq2_figure() -> None:
    base = {(r["model"], r["budget_pct"]): float(r["uno_c_36m"]) for r in rows(O / "lung1_e3_baselines/summary.csv")}
    fr = rows(O / "lung1_e3_frozen_linear_pca16/summary.csv")
    tests = {(r["budget_pct"], r["fm"], r["baseline"]): r
             for r in rows(O / "lung1_e3_baselines/statistical_tests_fm_vs_baselines.csv") if r["pca_components"] == "16"}
    pct, xs = [10, 25, 50, 100], [0, 1, 2, 3]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    ends = []
    for name, key, style in (("Stage+Volume", "stage_volume", (0, (3, 2))), ("GTV radiomics", "radiomics", (0, (1, 1.5)))):
        vals = [base[(key, str(p))] for p in pct]
        ax.plot(xs, vals, color=MUTED, linewidth=1.4, linestyle=style)
        ends.append([vals[-1], name])
    for e in ENCODERS:
        vals = [float(next(r["uno_c_36m"] for r in fr if r["model"] == e and r["budget_pct"] == str(p))) for p in pct]
        ax.plot(xs, vals, color=COLOR[e], linewidth=2, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8)
        ends.append([vals[-1], e])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.0085)
    for value, e in ends:
        ax.text(3.08, value, NAMES.get(e, e), color=INK if e in NAMES else INK2, fontsize=7, va="center")
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    budget_ticks = ["10%\n(33)", "25%\n(83)", "50%\n(167)", "100%\n(333)"]
    ax.set_xticks(xs, budget_ticks)
    ax.set_xlim(-0.2, 3.95)
    ax.set_xlabel("Training labels (patients per fold)")
    ax.set_ylabel("Mean-fold Uno C at 36 months")
    recessive(ax, "y")
    title(ax, "a  Discrimination by label budget")

    bx.axhline(0, color=INK2, linewidth=0.8)
    for i, e in enumerate(ENCODERS):
        for x in xs:
            r = tests[(str(pct[x]), e, "stage_volume")]
            d, lo, hi = (float(r[k]) for k in ("difference", "ci95_lower", "ci95_upper"))
            xx = x + (i - 2) * 0.13
            bx.plot([xx, xx], [lo, hi], color=COLOR[e], linewidth=1.2, solid_capstyle="round")
            bx.plot(xx, d, "o", color=COLOR[e], markersize=3.8, markeredgecolor="white", markeredgewidth=0.6)
    bx.set_xticks(xs, budget_ticks)
    bx.set_xlabel("Training labels (patients per fold)")
    bx.set_ylabel("Difference in Uno C (95% interval)")
    recessive(bx, "y")
    bx.legend(handles=[Line2D([], [], marker="o", linestyle="", color=COLOR[e], markersize=4, label=NAMES[e]) for e in ENCODERS],
              loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6.8, handletextpad=0.2, columnspacing=0.8)
    title(bx, "b  Paired difference from Stage+Volume")
    fig.tight_layout(w_pad=1.8)
    save(fig, "results_rq2")


# ---------------------------------------------------------------- Figure 3 (RQ3): tumour dependence
def rq3_figure() -> None:
    regional = {(r["model"], r["region"]): float(r["uno_c_36m"])
                for path in (O / "lung1_e4_regional_cox/summary.csv", O / "lung1_e4_regional_cox_dinov3/summary.csv")
                for r in rows(path)}
    masking = O / "lung1_e4_token_masking/cox_control403"
    diff = {(r["model"], r["contrast"]): r for r in rows(masking / "paired_mask_differences.csv")}
    change = {(r["model"], r["feature"]): float(r["median_relative_l2_vs_original"])
              for r in rows(masking / "embedding_change_summary.csv")}
    probe = {(r["model"], r["condition"], r["target"]): r for r in rows(O / "lung1_embedding_probing/summary.csv")
             if r["capacity"] == "pca16"}

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.9), gridspec_kw={"height_ratios": [1, 0.8]})
    (ax, bx), (cx, dx) = axes

    # (a) regional pooling
    regions = [("all", "All"), ("non_gtv", "Non-\ntumour"), ("peri20", "Shell"),
               ("tumor_peri20", "Tumour\n+shell"), ("tumor", "Tumour")]
    for e in ENCODERS:
        vals = [regional[(e, k)] for k, _ in regions]
        ax.plot(range(len(regions)), vals, color=COLOR[e], linewidth=1.8, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=NAMES[e])
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xticks(range(len(regions)), [label for _, label in regions])
    ax.set_ylabel("Mean-fold Uno C at 36 months")
    recessive(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.8, ncol=3, handlelength=1.2, columnspacing=0.8)
    title(ax, "a  Regional pooling of final tokens")

    # (b) native-token masking, change in discrimination
    y, ticks, labels = 0.0, [], []
    for e in ENCODERS:
        for key, short in (("keep_both", "retained"), ("global", "global")):
            for offset, contrast, filled in ((-0.17, f"{key}__original_minus_{key}__gtv_masked", True),
                                             (0.17, f"{key}__control_masked_minus_{key}__gtv_masked", False)):
                r = diff[(e, contrast)]
                interval(bx, y + offset, *(float(r[k]) for k in ("uno_c_36m_difference", "ci95_lower", "ci95_upper")),
                         COLOR[e], filled=filled, size=4)
            ticks.append(y)
            labels.append(f"{NAMES[e]}, {short}")
            y += 1.0
        y += 0.4
    bx.axvspan(-MARGIN, MARGIN, color=BAND, zorder=0)
    bx.axvline(0, color=INK2, linewidth=0.8)
    bx.set_yticks(ticks, labels, fontsize=7)
    bx.set_ylim(y - 0.4, -0.7)
    bx.set_xlim(-0.035, 0.035)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Difference in Uno C (95% interval)")
    recessive(bx)
    bx.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK2, markersize=4, label="original $-$ tumour hidden"),
                       Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK2,
                              markersize=4, label="control hidden $-$ tumour hidden")],
              loc="upper center", bbox_to_anchor=(0.4, -0.14), ncol=1, fontsize=6.8, handletextpad=0.3)
    title(bx, "b  Tumour hidden inside the encoder")

    # (c) change of the global vector
    for i, e in enumerate(ENCODERS):
        g, c = change[(e, "global__gtv_masked")], change[(e, "global__control_masked")]
        cx.plot([c, g], [i, i], color=GRID, linewidth=1.2, zorder=1)
        cx.plot(c, i, "o", color="white", markeredgecolor=COLOR[e], markersize=7, markeredgewidth=1.0, zorder=2)
        cx.plot(g, i, "o", color=COLOR[e], markersize=4, markeredgewidth=0, zorder=3)
    cx.set_xscale("log")
    cx.set_yticks(range(len(ENCODERS)), [NAMES[e] for e in ENCODERS])
    cx.set_ylim(len(ENCODERS) - 0.5, -0.6)
    cx.tick_params(axis="y", length=0)
    cx.set_xlabel("Median relative change of global vector")
    recessive(cx)
    cx.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK2, markersize=4, label="tumour hidden"),
                       Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK2,
                              markersize=7, label="control hidden")],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6.8, handletextpad=0.3)
    title(cx, "c  Change in representation")

    # (d) decodability of tumour volume, original vs tumour hidden
    for i, e in enumerate(ENCODERS):
        o = float(probe[(e, "original", "log_gtv_volume")]["value"])
        m = float(probe[(e, "gtv_masked", "log_gtv_volume")]["value"])
        dx.annotate("", xy=(m, i), xytext=(o, i), arrowprops=dict(arrowstyle="-|>", color=COLOR[e], lw=1.2, mutation_scale=7))
        dx.plot(o, i, "o", color=COLOR[e], markersize=4.5, markeredgecolor="white", markeredgewidth=0.8)
    dx.axvline(0, color=INK2, linewidth=0.8)
    dx.set_yticks(range(len(ENCODERS)), [NAMES[e] for e in ENCODERS])
    dx.set_ylim(len(ENCODERS) - 0.5, -0.6)
    dx.set_xlim(-0.05, 0.75)
    dx.tick_params(axis="y", length=0)
    dx.set_xlabel("Spearman $\\rho$, decoded vs true log GTV volume")
    recessive(dx)
    dx.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK2, markersize=4, label="original"),
                       Line2D([], [], marker=r"$\rightarrow$", linestyle="", color=INK2, markersize=8, label="tumour hidden")],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6.8, handletextpad=0.3)
    title(dx, "d  Tumour volume encoded in the global vector")
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save(fig, "results_rq3")


# ---------------------------------------------------------------- Figure 4 (RQ4): external transport
EXTERNAL_LABELS = {"C0": "Clinical", "C1": "Clinical+Volume", "baseline_stage_volume": "Stage+Volume",
                   "baseline_radiomics": "GTV radiomics", "C3-colipri": "Clinical+Volume+COLIPRI",
                   "C4-colipri": "Clinical+Volume+Radiomics+COLIPRI", **{e: NAMES[e] for e in ENCODERS},
                   "pai": "Pai lesion encoder", "hosny_zero_shot": "Hosny zero-shot"}


def rq4_figure(external: dict) -> None:
    p = external["populations"]["primary"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1, 1.1]})
    lo_lim, hi_lim = 0.46, 0.70
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    for key, m in p["models"].items():
        enc = key if key in ENCODERS else None
        ax.plot(m["internal_lung1_uno_c_36m"], m["uno_c_36m"], "o", color=COLOR[enc] if enc else MUTED,
                markersize=5.5 if enc else 4, markeredgecolor="white", markeredgewidth=0.8, zorder=3 if enc else 2)
        if enc:
            left = enc == "dinov3"
            ax.text(m["internal_lung1_uno_c_36m"] + (-0.006 if left else 0.006), m["uno_c_36m"], NAMES[enc], fontsize=6.8,
                    color=INK, va="center", ha="right" if left else "left")
    rt = p["ranking_transport"]
    ax.text(0.47, 0.69, f"Spearman $\\rho$ = {rt['spearman_internal_vs_external']:.2f} ({rt['n_models']} models)",
            fontsize=7, color=INK2, va="top")
    ax.set(xlim=(lo_lim, hi_lim), ylim=(lo_lim, hi_lim))
    ax.set_xlabel("Internal Uno C (LUNG1)")
    ax.set_ylabel("External Uno C (NSCLC-Radiogenomics)")
    ax.set_aspect("equal")
    recessive(ax, "both")
    title(ax, "a  Transport of discrimination")

    keys = ["C0", "C1", "baseline_stage_volume", "baseline_radiomics", "C3-colipri", "C4-colipri", *ENCODERS, "pai"]
    observed = p["models"]["colipri"]["observed_risk_36m"]
    bx.axvline(observed, color=INK, linewidth=1.0)
    bx.text(observed + 0.01, -0.9, "observed (Kaplan–Meier)", fontsize=6.8, color=INK, va="center")
    for i, key in enumerate(keys):
        m = p["models"][key]
        enc = key if key in ENCODERS else None
        bx.plot([observed, m["expected_risk_36m"]], [i, i], color=GRID, linewidth=1.2, zorder=1)
        bx.plot(m["expected_risk_36m"], i, "o", color=COLOR[enc] if enc else INK2, markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=2)
        bx.text(1.08, i, f"{m['calibration_slope']:.2f}", fontsize=6.8, color=INK2, va="center", ha="right")
    bx.text(1.08, -0.9, "Slope", fontsize=6.8, color=INK2, va="center", ha="right")
    bx.set_yticks(range(len(keys)), [EXTERNAL_LABELS[k] for k in keys], fontsize=7)
    bx.set_ylim(len(keys) - 0.5, -1.5)
    bx.set_xlim(0.15, 1.09)
    bx.set_xticks([0.2, 0.4, 0.6, 0.8])
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Predicted 36-month mortality")
    recessive(bx)
    title(bx, "b  Calibration of absolute risk")
    fig.tight_layout(w_pad=1.2)
    save(fig, "results_rq4")


def external_table(external: dict) -> None:
    p = external["populations"]["primary"]
    lines = []
    groups = [("Clinical and radiomic", ["C0", "C1", "baseline_stage_volume", "baseline_radiomics"]),
              ("Clinical and image", ["C3-colipri", "C4-colipri"]),
              ("Frozen image representations", [*ENCODERS, "pai", "hosny_zero_shot"])]
    for group, keys in groups:
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{group}}}}}\\\\")
        for key in keys:
            m, ch = p["models"][key], p["chance"][key]
            label = EXTERNAL_LABELS[key]
            label = f"\\model{{{label}}}" if key in ("C0", "C1", "baseline_stage_volume", "C3-colipri", "C4-colipri") else label
            exp = m["expected_risk_36m"]
            risk = "--" if exp is None or (isinstance(exp, float) and math.isnan(exp)) else f"{exp:.2f}"
            lines.append(f"{label} & {m['internal_lung1_uno_c_36m']:.3f} & {m['uno_c_36m']:.3f} ({m['ci95'][0]:.3f} to {m['ci95'][1]:.3f})"
                         f" & {ch['p_holm']:.3f} & {m['calibration_slope']:.2f} & {risk} \\\\")
    lines.append("\\multicolumn{6}{@{}l}{\\textit{Exploratory LoRA adaptation}$^{\\ddagger}$}\\\\")
    lora_cal = json.load(open(O / "lung1_e5_lora_external_calibration.json"))["models"]  # chance test and calibration
    for e in ENCODERS:
        # COLIPRI/TANGERINE: sweep fits on V100 (FP16); the others: fixed-configuration fits on A100 (BF16).
        swept = e in ("colipri", "tangerine")
        internal_root = "lung1_phase3" if swept else "lung1_adaptation_fixed"
        internal = json.load(open(O / f"{internal_root}/lr1e-3_ep20/lora/{e}/summary.json"))["mean_fold"]["uno_c_36m"]
        path = O / ("lung1_e5_lora_v100_fp16" if swept else "lung1_e5_lora_a100_bf16") / e / "summary.json"
        label = f"{NAMES[e]}, LoRA" + ("$^{\\S}$" if e == "ctclip" else "")
        if path.exists():
            d = json.load(open(path))["populations"]["primary"]
            c = lora_cal[e]
            lines.append(f"{label} & {internal:.3f} & {d['uno_c_36m']:.3f} ({d['ci95'][0]:.3f} to {d['ci95'][1]:.3f})"
                         f" & {'$<$0.001' if c['p_vs_chance'] < 0.001 else format(c['p_vs_chance'], '.3f')}$^{{\\dagger}}$ & {c['calibration_slope']:.2f} & {c['expected_risk_36m']:.2f} \\\\")
        else:
            lines.append(f"{label} & {internal:.3f} & pending & -- & -- & -- \\\\")
    observed = p["models"]["colipri"]["observed_risk_36m"]
    table = r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{External evaluation in NSCLC-Radiogenomics (143 patients, 33 deaths by 36 months). Models were fitted on all 416 LUNG1 patients and frozen before external outcomes were read. Internal and external values are Uno C at 36 months; internal is the mean-fold LUNG1 estimate. External intervals come from 2,000 patient resamples and condition on the frozen models. $p_\mathrm{Holm}$ tests external Uno C against 0.5, Holm-adjusted across all models. Calibration slope: coefficient of the linear predictor in an external Cox regression (1 is ideal). Predicted: mean predicted 36-month mortality with the LUNG1 baseline hazard; observed Kaplan--Meier mortality was """ + f"{observed:.2f}" + r""". The Hosny zero-shot score has no baseline hazard. $^{\ddagger}$Exploratory: fixed LoRA configuration, trained on all LUNG1 patients after the external cohort had been examined; it was chosen on the COLIPRI and TANGERINE test folds. Internal values are the LoRA estimates of Table~\ref{tab:trainable_results}. $^{\dagger}$Unadjusted $p$ against 0.5; the exploratory LoRA analysis was not corrected for multiplicity. $^{\S}$Training did not converge; predicted risks were almost constant.}
\label{tab:external}
\begin{tabular}{@{}lrlrrr@{}}
\toprule
Model & Internal & External (95\% interval) & $p_\mathrm{Holm}$ & Slope & Predicted \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tables/external.tex").write_text(table)


# ---------------------------------------------------------------- Table: trainable models
def mean_fold(path: Path, model: str | None = None, repetition: str | None = "0") -> float | None:
    if not path.exists():
        return None
    values = [float(r["uno_c_36m"]) for r in rows(path)
              if (model is None or r.get("model") == model) and (repetition is None or r.get("repetition", "0") == repetition)]
    return sum(values) / len(values) if len(values) == 5 else None


def summary_mean(path: Path) -> float | None:
    if not path.exists():
        return None
    d = json.load(open(path))
    return d["mean_fold"]["uno_c_36m"] if d.get("completed_folds") == 5 else None


def trainable_table() -> None:
    tests_path = O / "lung1_nested/trainable_tests.json"
    tests = json.load(open(tests_path)) if tests_path.exists() else {}

    def cell(value):
        return "pending" if value is None else f"{value:.3f}"

    lines = []
    for e in ENCODERS + ["pai", "hosny"]:
        fm = e in ENCODERS
        entries = [
            ("Frozen linear Cox, PCA16", mean_fold(O / "lung1_nested/linear_pca16/fold_metrics.csv", e), "reference"),
            ("Frozen linear Cox, PCA32", mean_fold(O / "lung1_nested/linear_pca32/fold_metrics.csv", e), "sensitivity"),
            ("Multilayer perceptron", summary_mean(O / f"lung1_nested/heads/final/mlp/{e}/summary.json"), "primary"),
        ]
        if fm:
            entries.append(("Survival query adapter", summary_mean(O / f"lung1_nested/heads/final/sqa/{e}/summary.json"), "primary"))
            if e in ("colipri", "tangerine"):
                lora = summary_mean(O / f"lung1_phase3/lr1e-3_ep20/lora/{e}/summary.json")
                part = summary_mean(O / f"lung1_phase3/lr1e-4_ep20/partial/{e}/summary.json")
                mark = "$^{\\dagger}$"
            else:
                lora = summary_mean(O / f"lung1_adaptation_fixed/lr1e-3_ep20/lora/{e}/summary.json")
                part = summary_mean(O / f"lung1_adaptation_fixed/lr1e-4_ep20/partial/{e}/summary.json")
                # CT-CLIP training loss spiked and then plateaued near its initial value in every fold.
                mark = "$^{\\ddagger}$" if e == "ctclip" else ""
            entries += [(f"LoRA{mark}", lora, "primary"), (f"Partial fine-tuning{mark}", part, "primary")]
        for i, (method, value, role) in enumerate(entries):
            t = tests.get(e, {}).get(method.replace("$^{\\dagger}$", "").replace("$^{\\ddagger}$", ""), {})
            delta = "--" if role == "reference" else (
                f"{t['difference']:+.3f} ({t['lower']:+.3f} to {t['upper']:+.3f})" if t else "pending")
            p = "--" if role != "primary" else (f"{t['p_holm']:.3f}" if t else "pending")
            name = NAMES[e] if i == 0 else ""
            lines.append(f"{name} & {method} & {cell(value)} & {delta} & {p} \\\\")
        lines.append("\\addlinespace")
    body = "\n".join(lines[:-1])
    table = r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Frozen and trainable survival models with all training labels. Mean-fold Uno C at 36 months over the five outer folds; every model was trained on the whole development set of each fold and evaluated on the same test patients. Differences are against the PCA16 frozen head of the same representation, with paired bootstrap intervals (1,000 resamples within folds) and Holm-adjusted $p$ values across 22 comparisons. $^{\dagger}$The adaptation configuration was chosen on these test folds in a preliminary analysis (Section~\ref{sec:res-adaptation}), so these estimates are optimistic. $^{\ddagger}$Training was unstable: the loss spiked during the first epochs and the predicted risks varied little between patients.}
\label{tab:trainable_results}
\begin{tabular}{@{}llrlr@{}}
\toprule
Representation & Model & Uno C & Difference vs PCA16 (95\% interval) & $p_\mathrm{Holm}$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tables/trainable_results.tex").write_text(table)


if __name__ == "__main__":
    external = json.load(open(O / "phase4_external/external_validation.json"))
    rq1_figure()
    rq2_figure()
    rq3_figure()
    rq4_figure(external)
    external_table(external)
    trainable_table()
    print("Wrote figures/results_rq1-4.pdf, tables/trainable_results.tex, tables/external.tex")
