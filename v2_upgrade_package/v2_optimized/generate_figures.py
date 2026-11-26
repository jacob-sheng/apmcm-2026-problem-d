#!/usr/bin/env python3
"""
Publication-Grade (300 DPI, English-Only) Figure Generator for v2_optimized.
Reads verified JSON/CSV outputs from v2_optimized/results/ and renders 6
high-impact figures into v2_optimized/figures/.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(CUR_DIR, "results")
FIG_DIR = os.path.join(CUR_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Clean academic visual style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "legend.fontsize": 8.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

NAVY = "#1B4965"
TEAL = "#1B9E77"
ORANGE = "#D95F02"
CRIMSON = "#C0392B"
PURPLE = "#7570B3"
SLATE = "#6C757D"
GOLD = "#E69F00"
BLUE = "#2B7BBA"


def fig01_v1_vs_v2_surrogate():
    with open(os.path.join(RES_DIR, "v1_baseline_metrics.json"), "r", encoding="utf-8") as f:
        v1 = json.load(f)
    with open(os.path.join(RES_DIR, "v2_q1_diagnostics.json"), "r", encoding="utf-8") as f:
        v2 = json.load(f)
    val_df = pd.read_csv(os.path.join(RES_DIR, "v2_val_predictions.csv"))

    by_name = {r["model"]: r for r in (v1["q1_metrics"] + v2["metrics_table"])}
    ordered_keys = [
        ("v1_OneHot_FM_k8", "v1 One-Hot\nFM (k=8)", SLATE),
        ("v1_Direct_LGBM_V", "v1 Direct\nLGBM (V)", "#8DA0CB"),
        ("v1_Direct_LGBM_log1pV", "v1 Direct\nLGBM (log1p)", ORANGE),
        ("v2_Direct_4FamPhys", "v2 Direct\n4-Fam Phys", BLUE),
        ("v2_PhysHurdle_Bayes", "v2 PhysHurdle\nBayes (Ours)", TEAL),
    ]
    rows = [by_name[k] for k, _, _ in ordered_keys]
    labels = [lbl for _, lbl, _ in ordered_keys]
    colors = [col for _, _, col in ordered_keys]

    rmse = [r["RMSE"] for r in rows]
    rec50 = [100.0 * r["Rec@50(V<0.5)"] for r in rows]
    ef50 = [r["EF@50(V<0.5)"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel A: RMSE (lower is better)
    ax = axes[0]
    bars = ax.bar(range(len(labels)), rmse, color=colors, width=0.62, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Validation RMSE on V (Lower is Better)")
    ax.set_title("(a) Global Surrogate Error on 03_val")
    ax.set_ylim(0, max(rmse) * 1.18)
    for b, val in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Panel B: Rec@50 and EF@50
    ax = axes[1]
    x = np.arange(len(labels))
    bars = ax.bar(x, rec50, color=colors, width=0.62, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Top-50 Recall for V < 0.5 (%)")
    ax.set_title("(b) Rare-Target Recall & Enrichment (EF@50)")
    ax.set_ylim(0, 114)
    for b, r_val, ef_val in zip(bars, rec50, ef50):
        ax.text(b.get_x() + b.get_width() / 2, r_val + 2.0, f"{r_val:.1f}%\n({ef_val:.1f}x)",
                ha="center", va="bottom", fontweight="bold", fontsize=8.3)

    # Panel C: Exact Butler-Ginley Parity Plot
    ax = axes[2]
    ec_t = val_df["ec_true"].values
    ec_p = val_df["ec_exact"].values
    in_win = (ec_t > 4.18) & (ec_t < 6.05)
    ax.axvspan(4.18, 6.05, color=TEAL, alpha=0.10, label="Feasible Window (4.18–6.05 eV)")
    ax.scatter(ec_t[~in_win], ec_p[~in_win], s=7, alpha=0.35, color=SLATE, rasterized=True, label="Outside Window")
    ax.scatter(ec_t[in_win], ec_p[in_win], s=10, alpha=0.55, color=NAVY, rasterized=True, label="Inside Window")
    lims = [min(ec_t.min(), ec_p.min()) - 0.15, max(ec_t.max(), ec_p.max()) + 0.15]
    ax.plot(lims, lims, "--", color=CRIMSON, linewidth=1.4, label="Identity y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT Band Center E_c = (VBM+CBM)/2 (eV)")
    ax.set_ylabel("Butler–Ginley Predicted E_c (eV)")
    ax.set_title("(c) Exact Log-Additive Band-Center Law")
    ax.text(0.05, 0.84, f"Unseen (A,B) Val (n=3,969)\nR² = 1.0000000000\nRMSE = {v2['band_center_val_rmse_eV']:.2e} eV",
            transform=ax.transAxes, fontsize=8.8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F9F9", edgecolor=NAVY, alpha=0.95))
    ax.legend(loc="lower right", frameon=True, fontsize=8.0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig01_v1_vs_v2_surrogate_metrics.png"))
    plt.close()


def fig02_residual_and_calibration():
    with open(os.path.join(RES_DIR, "v2_q1_diagnostics.json"), "r", encoding="utf-8") as f:
        diag = json.load(f)
    val_df = pd.read_csv(os.path.join(RES_DIR, "v2_val_predictions.csv"))
    resid = val_df["V_hat"].values - val_df["V"].values
    is_semi = val_df["gap"].values > 0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel A: Residual Distribution (Metal vs Semiconductor)
    ax = axes[0]
    bins = np.linspace(-15, 15, 55)
    ax.hist(resid[~is_semi], bins=bins, density=True, alpha=0.55, color=NAVY, label=f"Metals (n={(~is_semi).sum()}, RMSE={diag['residual_diagnostics']['metal_rmse']:.2f})")
    ax.hist(resid[is_semi], bins=bins, density=True, alpha=0.65, color=ORANGE, label=f"Semiconductors (n={is_semi.sum()}, RMSE={diag['residual_diagnostics']['semi_rmse']:.2f})")
    ax.axvline(0.0, color=CRIMSON, linestyle="--", linewidth=1.3)
    ax.set_xlabel("Prediction Residual (V_hat - V_true)")
    ax.set_ylabel("Probability Density")
    ax.set_title("(a) Residual Distribution by Electronic Regime")
    ax.text(0.04, 0.72, f"Overall Bias: {diag['residual_diagnostics']['overall_bias']:+.3f}\nNear-Zero (V<1) MAE: {diag['residual_diagnostics']['near_zero_Vlt1_mae']:.3f}",
            transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=SLATE, alpha=0.9))
    ax.legend(loc="upper right", fontsize=8.0)

    # Panel B: Absolute Error across Band Center E_c bins
    ax = axes[1]
    ec = val_df["ec_true"].values
    abs_err = np.abs(resid)
    edges = np.linspace(3.5, 7.0, 11)
    centers, means, cis = [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (ec >= e0) & (ec < e1)
        if m.sum() >= 10:
            centers.append(0.5 * (e0 + e1))
            means.append(abs_err[m].mean())
            cis.append(1.96 * abs_err[m].std() / np.sqrt(m.sum()))
    ax.axvspan(4.18, 6.05, color=TEAL, alpha=0.12, label="Feasible Band-Center Window")
    ax.errorbar(centers, means, yerr=cis, fmt="o-", color=NAVY, linewidth=2.0, markersize=6, capsize=4, label="Mean |V_hat - V_true| ± 95% CI")
    ax.set_xlabel("Band Center E_c (eV)")
    ax.set_ylabel("Mean Absolute Error on V")
    ax.set_title("(b) Error Profile Across Band-Center Spectrum")
    ax.legend(loc="upper left", fontsize=8.2)

    # Panel C: Reliability / Calibration Curve for P(gap > 0)
    ax = axes[2]
    cal = diag["calibration_curve"]
    mp = [c["mean_pred"] for c in cal]
    of = [c["obs_freq"] for c in cal]
    cnt = [c["count"] for c in cal]
    ax.plot([0, 1], [0, 1], "--", color=SLATE, linewidth=1.3, label="Perfect Calibration")
    ax.plot(mp, of, "o-", color=TEAL, linewidth=2.2, markersize=7, label="v2 Calibrated Ensemble")
    for x0, y0, n0 in zip(mp, of, cnt):
        if n0 >= 10:
            dx = 12 if x0 < 0.05 else (-10 if x0 > 0.92 else 0)
            dy = 8 if x0 < 0.92 else 6
            ax.annotate(f"n={n0}", (x0, y0), textcoords="offset points", xytext=(dx, dy), ha="center", fontsize=7.5, color="#333333")
    cm = diag["component_val_metrics"]
    ax.text(0.05, 0.72, f"ROC-AUC = {cm['p_semi_ROC_AUC']:.4f}\nPR-AUC  = {cm['p_semi_PR_AUC']:.4f}\nBrier   = {cm['p_semi_Brier']:.5f}\nECE     = {cm['p_semi_ECE']:.5f}",
            transform=ax.transAxes, fontsize=8.6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F9F9", edgecolor=TEAL, alpha=0.95))
    ax.set_xlabel("Predicted Semiconductor Probability P(gap > 0)")
    ax.set_ylabel("Observed Empirical Frequency")
    ax.set_title("(c) Hurdle Classifier Reliability Curve (03_val)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.08)
    ax.legend(loc="lower right", fontsize=8.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig02_residual_and_calibration.png"))
    plt.close()


def fig03_closed_loop_fmqa():
    with open(os.path.join(RES_DIR, "v2_experiments.json"), "r", encoding="utf-8") as f:
        exp = json.load(f)
    fm = exp["closed_loop_fmqa"]
    budgets = np.array(fm["budgets"])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    specs = [
        ("Random", "Random Search Baseline", SLATE, "s--"),
        ("v1_FMQA", "v1 Iterative One-Hot FMQA", ORANGE, "^-."),
        ("v2_PhysQUBO", "v2 Physics-Hurdle Active-Subspace QUBO", TEAL, "o-"),
    ]

    for ax, key, tot, title, ylab in [
        (axes[0], "rare05", fm["total_rare05"], f"(a) Closed-Loop Rare-Target Discovery (V < 0.5, {fm['total_rare05']} Targets)", "Cumulative Rare Targets Found (V < 0.5)"),
        (axes[1], "rare10", fm["total_rare10"], f"(b) Closed-Loop Near-Zero Discovery (V < 1.0, {fm['total_rare10']} Targets)", "Cumulative Near-Zero Targets Found (V < 1.0)"),
    ]:
        for m, lbl, col, sty in specs:
            mean = np.array(fm["methods"][m][f"{key}_mean"])
            std = np.array(fm["methods"][m][f"{key}_std"])
            ax.plot(budgets, mean, sty, color=col, linewidth=2.3, markersize=5.5, label=f"{lbl} ({mean[-1]:.1f}/{tot})")
            ax.fill_between(budgets, np.maximum(0, mean - std), mean + std, color=col, alpha=0.14)
        ax.axhline(tot, color=CRIMSON, linestyle=":", linewidth=1.2, label=f"Pool Total ({tot})")
        ax.set_xlabel("Cumulative DFT Evaluation Budget (Compounds Queried out of 3,969)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xlim(budgets[0] - 3, budgets[-1] + 5)
        ax.set_ylim(0, tot * 1.12)
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig03_closed_loop_fmqa_benchmark.png"))
    plt.close()


def fig04_q2_hardware_and_cost():
    with open(os.path.join(RES_DIR, "v2_experiments.json"), "r", encoding="utf-8") as f:
        exp = json.load(f)
    q2b = pd.read_csv(os.path.join(RES_DIR, "v2_q2_blind_11.csv")).head(10).iloc[::-1]

    bw_df = pd.DataFrame(exp["bitwidth_sweep"])
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))

    # Panel A: Bit-Width Sweep (v1 Naive vs v2 Exact-Ratio Active-Subspace)
    ax = axes[0]
    bws = bw_df["bit_width"].values
    ax.plot(bws, bw_df["v1_feas_pct"].values, "s--", color=CRIMSON, linewidth=2.2, markersize=6.5, label="v1 Naive 104-Var QUBO Feasibility (%)")
    ax.plot(bws, bw_df["v2_feas_pct"].values, "o-", color=TEAL, linewidth=2.5, markersize=7.0, label="v2 Active-Subspace Exact-Ratio Feasibility (%)")
    ax.axvline(8, color=NAVY, linestyle=":", linewidth=1.6, label="SPQC Hardware Limit (8-bit Int8)")
    ax.set_xlabel("Ising Coefficient Precision (Signed Bit-Width)")
    ax.set_ylabel("One-Hot Feasibility Rate (%)")
    ax.set_title("(a) Proof of Theorem 1: 8-Bit Quantization Drift vs v2 Fix")
    ax.set_ylim(-5, 112)
    ax.set_xticks(bws)
    # Annotate Theorem 1 drift at 8-bit
    v1_8 = bw_df.loc[bw_df["bit_width"] == 8].iloc[0]
    v2_8 = bw_df.loc[bw_df["bit_width"] == 8].iloc[0]
    ax.annotate(f"v1 @ 8-bit: {v1_8['v1_feas_pct']:.0f}% feasible\n(mean k_A = {v1_8['v1_mean_kA']:.2f} vs 1.0!)",
                xy=(8, v1_8["v1_feas_pct"]), xytext=(4.2, 22),
                arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.4),
                fontsize=8.5, fontweight="bold", color=CRIMSON,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5", edgecolor=CRIMSON))
    ax.annotate(f"v2 @ 8-bit: {v2_8['v2_feas_pct']:.0f}% feasible\n(Top-10 capture: {int(v2_8['v2_top10_hits'])}/10)",
                xy=(8, v2_8["v2_feas_pct"]), xytext=(8.6, 68),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4),
                fontsize=8.5, fontweight="bold", color=TEAL,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F2FBF8", edgecolor=TEAL))
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.44), fontsize=8.0, frameon=True)

    # Panel B: Q2 Blind Top-10 5-Term Cost Breakdown
    ax = axes[1]
    y_pos = np.arange(len(q2b))
    terms = [
        ("v_form", "Formation Energy (e_form > 0.21)", NAVY),
        ("v_gap_low", "Lower Band Gap (gap < 1.4 eV)", ORANGE),
        ("v_gap_high", "Upper Band Gap (gap > 3.1 eV)", GOLD),
        ("v_vbm", "VBM Alignment (VBM < 5.73 eV)", TEAL),
        ("v_cbm", "CBM Alignment (CBM > 4.50 eV)", PURPLE),
    ]
    left = np.zeros(len(q2b))
    for col, lbl, c in terms:
        vals = q2b[col].values
        ax.barh(y_pos, vals, left=left, height=0.62, color=c, label=lbl, edgecolor="white", linewidth=0.4)
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{10 - i}  {f}" for i, f in enumerate(q2b["formula"].values)], fontweight="bold", fontsize=9.2)
    for i, (tot, ph) in enumerate(zip(left, q2b["P_hit"].values)):
        ax.text(tot + 0.04, i, f"V={tot:.2f} | P_hit={100*ph:.1f}%", va="center", fontsize=8.3, fontweight="bold", color="#222222")
    ax.set_xlabel("Expected 5-Term Violation Cost V_hat (Lower is Better)")
    ax.set_title("(b) Q2 Blind Pool (11, X=NO2): Top-10 Cost Decomposition")
    ax.set_xlim(0, left.max() * 1.46)
    ax.legend(loc="upper right", fontsize=7.6, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig04_q2_v1_vs_v2_hardware_drift.png"))
    plt.close()


def fig05_q3_compression_and_cost():
    with open(os.path.join(RES_DIR, "v1_baseline_metrics.json"), "r", encoding="utf-8") as f:
        v1 = json.load(f)
    with open(os.path.join(RES_DIR, "v2_qubo_results.json"), "r", encoding="utf-8") as f:
        v2 = json.load(f)
    q3b = pd.read_csv(os.path.join(RES_DIR, "v2_q3_blind_12.csv")).head(15).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4))

    # Panel A: Spin Count (log scale) & SA vs Tabu Top-15 Capture
    ax = axes[0]
    m_3a_08 = v2["Q3"]["Q3_full_08_Model3A_SubmodularHOBO"]
    m_3b_08 = v2["Q3"]["Q3_full_08_Model3B_CompactANOVA"]
    m_3a_12 = v2["Q3"]["Q3_blind_12_Model3A_SubmodularHOBO"]
    m_3b_12 = v2["Q3"]["Q3_blind_12_Model3B_CompactANOVA"]
    v1_fm8 = v1["q3_results"]["Q3_full_08_v1_FM107_Naive8bit_SA"]
    v1_ros = v1["q3_results"]["Q3_v1_NaiveRosenberg_Structure"]["total_spins_AB_pairing"]

    names = [
        "v1 Naive\nRosenberg\n(Full 08)",
        "v1 Pairwise\nFM-107\n(8-bit)",
        "v2 Model 3A\nSubmod HOBO\n(Full 08)",
        "v2 Model 3A\nSubmod HOBO\n(Blind 12)",
        "v2 Model 3B\nCompact ANOVA\n(Full 08)",
        "v2 Model 3B\nCompact ANOVA\n(Blind 12)",
    ]
    spins = [v1_ros, v1_fm8["spins"], m_3a_08["spins"], m_3a_12["spins"], m_3b_08["spins"], m_3b_12["spins"]]
    cols = [CRIMSON, ORANGE, NAVY, BLUE, TEAL, "#2CA02C"]
    notes = [
        "Exceeds 1000\nCIM Limit!",
        f"Feas: {v1_fm8['feasibility_pct']:.0f}%\nHits: {v1_fm8['top15_hits']}/15",
        f"SA: {m_3a_08['SA']['top15_hits']}/15\nTabu: {m_3a_08['Tabu']['top15_hits']}/15",
        f"SA: {m_3a_12['SA']['top15_hits']}/15\nTabu: {m_3a_12['Tabu']['top15_hits']}/15",
        f"SA: {m_3b_08['SA']['top15_hits']}/15\nTabu: {m_3b_08['Tabu']['top15_hits']}/15",
        f"SA: {m_3b_12['SA']['top15_hits']}/15\nTabu: {m_3b_12['Tabu']['top15_hits']}/15",
    ]
    bars = ax.bar(range(len(names)), spins, color=cols, width=0.60, edgecolor="#222222", linewidth=0.7)
    ax.set_yscale("log")
    ax.axhline(1000, color=CRIMSON, linestyle="--", linewidth=1.4, label="1,000-Qubit CIM Capacity")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.2)
    ax.set_ylabel("Total Hardware Spins (Log Scale)")
    ax.set_title("(a) Q3 Qubit Compression & Multi-Solver Capture (8-Bit)")
    ax.set_ylim(10, 8500)
    for b, sp, nt in zip(bars, spins, notes):
        ax.text(b.get_x() + b.get_width() / 2, sp * 1.15, f"{sp} spins\n{nt}",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.4)

    # Panel B: Q3 Blind Pool (12) Top-15 5-Term Cost Breakdown
    ax = axes[1]
    y_pos = np.arange(len(q3b))
    terms = [
        ("v_form", "e_form", NAVY),
        ("v_gap_low", "gap < 1.4", ORANGE),
        ("v_gap_high", "gap > 3.1", GOLD),
        ("v_vbm", "VBM < 5.73", TEAL),
        ("v_cbm", "CBM > 4.50", PURPLE),
    ]
    left = np.zeros(len(q3b))
    for col, lbl, c in terms:
        vals = q3b[col].values
        ax.barh(y_pos, vals, left=left, height=0.66, color=c, label=lbl, edgecolor="white", linewidth=0.4)
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{15 - i} {f} [{x}]" for i, (f, x) in enumerate(zip(q3b["formula"].values, q3b["X"].values))],
                       fontweight="bold", fontsize=8.4)
    for i, (tot, ph) in enumerate(zip(left, q3b["P_hit"].values)):
        ax.text(tot + 0.03, i, f"V={tot:.2f} (P={100*ph:.1f}%)", va="center", fontsize=7.9, fontweight="bold")
    ax.set_xlabel("Expected 5-Term Violation Cost V_hat")
    ax.set_title("(b) Q3 Blind Pool (12, O3/NO2/FO2): Top-15 Cost Breakdown")
    ax.set_xlim(0, left.max() * 1.48)
    ax.legend(loc="upper right", ncol=1, fontsize=7.8, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig05_q3_qubit_compression_and_solvers.png"))
    plt.close()


def fig06_sensitivity_and_mc():
    with open(os.path.join(RES_DIR, "v2_experiments.json"), "r", encoding="utf-8") as f:
        exp = json.load(f)

    pen_df = pd.DataFrame(exp["penalty_sweep"])
    cap_df = pd.DataFrame(exp["cap_sweep"])
    mc_q2 = exp["monte_carlo_robustness"]["Q2_blind_11"]
    mc_q3 = exp["monte_carlo_robustness"]["Q3_blind_12"]

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))

    # Panel A: Penalty Multiplier Sweep
    ax = axes[0, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(pen_df["alpha_P"], pen_df["feasibility_pct"], "o-", color=NAVY, linewidth=2.2, label="Feasibility Rate (%)")
    l2 = ax2.plot(pen_df["alpha_P"], pen_df["top10_hits"], "s--", color=TEAL, linewidth=2.2, label="Top-10 Hits (/10)")
    ax.axvline(1.50, color=CRIMSON, linestyle=":", linewidth=1.6, label="Theorem 2 Bound (alpha_P = 1.50)")
    ax.set_xlabel("Penalty Multiplier alpha_P (P / C_cap)")
    ax.set_ylabel("One-Hot Feasibility (%)", color=NAVY)
    ax2.set_ylabel("Top-10 Capture Count", color=TEAL)
    ax.set_ylim(-5, 110)
    ax2.set_ylim(0, 11.5)
    ax.set_title("(a) Penalty Sensitivity Sweep (Q2 Blind 11, 8-Bit)")
    lines = l1 + l2 + [ax.lines[-1]]
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8.2)
    ax2.spines["top"].set_visible(False)

    # Panel B: Upper-Tail Cap Pareto Frontier
    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(cap_df["C_cap"], cap_df["q2_spins"], "o-", color=NAVY, linewidth=2.2, label="Q2 Full Active Spins")
    l2 = ax.plot(cap_df["C_cap"], cap_df["q3_spins"], "^-", color=ORANGE, linewidth=2.2, label="Q3 Blind Active Spins")
    l3 = ax2.plot(cap_df["C_cap"], 100 * cap_df["q2_top10_hits"] / 10.0, "s--", color=TEAL, linewidth=1.9, label="Q2 Top-10 Capture (%)")
    l4 = ax2.plot(cap_df["C_cap"], 100 * cap_df["q3_top15_hits"] / 15.0, "d:", color=PURPLE, linewidth=1.9, label="Q3 Top-15 Capture (%)")
    ax.set_xlabel("Upper-Tail Cap Threshold C_cap")
    ax.set_ylabel("Active Hardware Spins", color=NAVY)
    ax2.set_ylabel("Top-K Capture Rate (%)", color=TEAL)
    ax.set_ylim(5, 82)
    ax2.set_ylim(0, 132)
    ax.set_title("(b) Active-Subspace Cap C_cap Pareto Trade-Off")
    lines = l1 + l2 + l3 + l4
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", ncol=2, fontsize=7.6, frameon=True)
    ax2.spines["top"].set_visible(False)

    # Panel C: Monte Carlo Q2 Blind Inclusion Probability
    ax = axes[1, 0]
    items_q2 = list(mc_q2["per_candidate_inclusion_pct"].items())[::-1]
    f_q2 = [k for k, _ in items_q2]
    v_q2 = [v for _, v in items_q2]
    bars = ax.barh(range(len(f_q2)), v_q2, color=NAVY, height=0.64, alpha=0.88)
    ax.set_yticks(range(len(f_q2)))
    ax.set_yticklabels(f_q2, fontweight="bold", fontsize=8.8)
    ax.set_xlim(0, 115)
    for b, val in zip(bars, v_q2):
        ax.text(val + 1.5, b.get_y() + b.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8.3, fontweight="bold")
    ax.set_xlabel("Top-10 Inclusion Probability Across 500 MC Draws (%)")
    ax.set_title(f"(c) Q2 Blind MC Robustness (Retention: {mc_q2['mean_retention_pct']:.1f}% ± {mc_q2['std_retention_pct']:.1f}%, ρ={mc_q2['mean_spearman']:.4f})")

    # Panel D: Monte Carlo Q3 Blind Inclusion Probability
    ax = axes[1, 1]
    items_q3 = list(mc_q3["per_candidate_inclusion_pct"].items())[::-1]
    f_q3 = [k for k, _ in items_q3]
    v_q3 = [v for _, v in items_q3]
    bars = ax.barh(range(len(f_q3)), v_q3, color=TEAL, height=0.66, alpha=0.88)
    ax.set_yticks(range(len(f_q3)))
    ax.set_yticklabels(f_q3, fontweight="bold", fontsize=8.2)
    ax.set_xlim(0, 115)
    for b, val in zip(bars, v_q3):
        ax.text(val + 1.5, b.get_y() + b.get_height() / 2, f"{val:.1f}%", va="center", fontsize=7.9, fontweight="bold")
    ax.set_xlabel("Top-15 Inclusion Probability Across 500 MC Draws (%)")
    ax.set_title(f"(d) Q3 Blind MC Robustness (Retention: {mc_q3['mean_retention_pct']:.1f}% ± {mc_q3['std_retention_pct']:.1f}%, ρ={mc_q3['mean_spearman']:.4f})")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig06_sensitivity_and_mc_robustness.png"))
    plt.close()


def fig07_spqc550_real_hardware():
    spqc_path = os.path.join(RES_DIR, "v2_spqc550_results.json")
    if not os.path.exists(spqc_path):
        print("  [Skip] v2_spqc550_results.json not found.")
        return
    with open(spqc_path, "r", encoding="utf-8") as f:
        spqc = json.load(f)["tasks"]

    # Verified SPQC-550 real hardware comparison:
    # Teammate Full-Space K-Copy Ref-Split (table4_spqc550_hardware_vs_classical.csv) vs Our v2 Compact Active-Subspace
    hw_rows = [
        ("TM\nQ2 Uncap\n(105 sp)", 105, 0.0, "Teammate", "INFEAS"),
        ("TM\nQ2 Full\n(224 sp)", 224, 11.1, "Teammate", "LaTiO2N\n#1"),
        ("TM\nQ2 Blind\n(224 sp)", 224, 0.0, "Teammate", "INFEAS"),
        ("TM\nQ3 QFM\n(257 sp)", 257, 0.0, "Teammate", "INFEAS"),
        ("TM\nQ3 CMP\n(468 sp)", 468, 10.0, "Teammate", "YTiO2N\n#2"),
        ("TM\nQ3 ROS\n(413 sp)", 413, 10.0, "Teammate", "AgTaO3\n#21"),
        ("v2\nQ2 Blind\n(12 sp)", 12, spqc["apmcm26_v2_q2_blind_12"]["spqc550"]["feasible_rate_pct"], "v2", "SrTaO2N #1\n+CaTa/YTi"),
        ("v2\nQ3 ANOVA\n(24 sp)", 24, spqc["apmcm26_v2_q3_anova_24"]["spqc550"]["feasible_rate_pct"], "v2", "LaTiO2F\n(2nd-ord)"),
        ("v2\nQ3 Submod\n(33 sp)", 33, spqc["apmcm26_v2_q3_submod_33"]["spqc550"]["feasible_rate_pct"], "v2", "CaTaO2N #2\n(4/10 feas)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4))

    # Panel A: Real SPQC-550 Hardware Feasibility Rate across all 9 real runs
    ax = axes[0]
    x = np.arange(len(hw_rows))
    feas = [r[2] for r in hw_rows]
    cols = [SLATE if r[3] == "Teammate" and r[2] == 0 else (ORANGE if r[3] == "Teammate" else (BLUE if "ANOVA" in r[0] else TEAL)) for r in hw_rows]
    bars = ax.bar(x, feas, color=cols, width=0.64, edgecolor="#222222", linewidth=0.7)
    ax.axvline(5.5, color=NAVY, linestyle="--", linewidth=1.4)
    ax.text(2.5, 55.5, "Full-Space K-Split\n(105–468 spins, 0–11%)", ha="center", va="top", fontsize=8.4, fontweight="bold", color="#8B4513")
    ax.text(7.0, 55.5, "v2 Compact Subspace\n(12–33 spins, 20–40%)", ha="center", va="top", fontsize=8.4, fontweight="bold", color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in hw_rows], fontsize=7.6)
    ax.set_ylabel("Real SPQC-550 Feasibility Rate (%)")
    ax.set_title("(a) Real SPQC-550 Feasibility: 9 Cloud Quantum Runs")
    ax.set_ylim(0, 58)
    for b, val, r in zip(bars, feas, hw_rows):
        ax.text(b.get_x() + b.get_width() / 2, val + 1.0, f"{val:.0f}%\n{r[4]}",
                ha="center", va="bottom", fontsize=6.7, fontweight="bold")

    # Panel B: Raw Ising Energy Excess Spectrum across all 10 SPQC-550 samples (our 3 tasks)
    ax = axes[1]
    t_keys = [
        ("apmcm26_v2_q2_blind_12", "Q2 Blind\n(12 spins)"),
        ("apmcm26_v2_q3_anova_24", "Q3 ANOVA\n(24 spins)"),
        ("apmcm26_v2_q3_submod_33", "Q3 Submod\n(33 spins)"),
    ]
    for xi, (tk, tlbl) in enumerate(t_keys):
        samples = spqc[tk]["spqc550"]["samples"]
        e_min = spqc[tk]["spqc550"]["best_ising_energy"]
        rng = np.random.default_rng(42 + xi)
        for s in samples:
            de = s["ising_energy"] - e_min
            jx = xi + rng.uniform(-0.16, 0.16)
            if s["feasible"]:
                ax.scatter(jx, de + 1.0, s=68, color=TEAL, edgecolor="#111111", zorder=5,
                           label="SPQC-550 Feasible Sample" if (xi == 0 and s["sample_idx"] == 0) else "")
            else:
                ax.scatter(jx, de + 1.0, s=46, marker="x", color=CRIMSON, linewidths=1.5, zorder=4,
                           label="SPQC-550 Infeasible Sample" if (xi == 0 and s["sample_idx"] == 3) else "")
    ax.set_yscale("log")
    ax.axhline(1.0, color=NAVY, linestyle=":", linewidth=1.3, label="Exact Ground State (ΔE = 0)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([t[1] for t in t_keys], fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Ising Energy Excess (ΔE + 1, Log Scale)")
    ax.set_title("(b) Raw SPQC-550 Energy Spectrum (10 Shots/Task)")
    ax.annotate("SrTaO2N #1 & CaTaO2N #2\n(ΔE=0) + YTiO2N #6 (ΔE=8)", xy=(0, 1.0), xytext=(0.15, 6.0),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2), fontsize=7.8, fontweight="bold", color=TEAL,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F2FBF8", edgecolor=TEAL))
    ax.annotate("4x CaTaO2N #2\n(Exact ΔE=0!)", xy=(2, 1.0), xytext=(1.22, 42.0),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2), fontsize=7.8, fontweight="bold", color=TEAL,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F2FBF8", edgecolor=TEAL))
    ax.legend(loc="upper left", fontsize=7.8, frameon=True)

    # Panel C: Spin Compression vs Feasibility Trade-Off (Pareto Scatter)
    ax = axes[2]
    for name, sp, fr, grp, hit in hw_rows:
        short = name.replace("\n", " ").replace("Teammate ", "TM ").replace("v2 Ours ", "v2 ")
        if grp == "Teammate":
            ax.scatter(sp, fr, s=95, marker="s", color=ORANGE if fr > 0 else SLATE, edgecolor="#222222", zorder=4)
        else:
            ax.scatter(sp, fr, s=135, marker="o", color=TEAL if "Submod" in short or "Blind" in short else BLUE, edgecolor="#111111", zorder=5)
    # Clean non-overlapping callouts with leader arrows for clustered points
    callouts = [
        ("TM\nQ2 Uncap\n(105 sp)", "TM Q2 Uncap (0%)", (40, 4.5), True),
        ("TM\nQ2 Full\n(224 sp)", "TM Q2 Full (LaTiO2N #1)", (65, 16.5), True),
        ("TM\nQ2 Blind\n(224 sp)", "TM Q2 Blind & Q3 QFM (0%)", (135, -3.6), True),
        ("TM\nQ3 CMP\n(468 sp)", "TM Q3 CMP (YTiO2N #2)", (120, 23.0), True),
        ("TM\nQ3 ROS\n(413 sp)", "TM Q3 ROS (AgTaO3 #21)", (110, 5.2), True),
        ("v2\nQ2 Blind\n(12 sp)", "v2 Q2 Blind (12sp: #1 SrTaO2N)", (15.5, 30.2), False),
        ("v2\nQ3 ANOVA\n(24 sp)", "v2 Q3 ANOVA (24sp: 2nd-ord)", (30.5, 19.8), False),
        ("v2\nQ3 Submod\n(33 sp)", "v2 Q3 Submod (33sp: #2 CaTaO2N)", (42.0, 39.8), False),
    ]
    row_by_key = {r[0]: r for r in hw_rows}
    for key, txt, xy_txt, use_arrow in callouts:
        r = row_by_key[key]
        is_v2 = (r[3] == "v2")
        if use_arrow:
            ax.annotate(txt, xy=(r[1], r[2]), xytext=xy_txt,
                        arrowprops=dict(arrowstyle="->", color="#777777", lw=0.9),
                        fontsize=7.3, color="#333333")
        else:
            ax.text(xy_txt[0], xy_txt[1], txt, fontsize=7.6, fontweight="bold", color=NAVY, va="center")
    ax.set_xscale("log")
    ax.set_xlim(8, 720)
    ax.set_ylim(-6, 50)
    ax.set_xlabel("Hardware Spin Count on SPQC-550 (Log Scale)")
    ax.set_ylabel("Real SPQC-550 Feasibility Rate (%)")
    ax.set_title("(c) Spin Compression vs Real Quantum Feasibility")
    ax.axvline(550, color=CRIMSON, linestyle="--", linewidth=1.2, label="SPQC-550 Limit (550 spins)")
    ax.legend(handles=[
        Patch(facecolor=TEAL, edgecolor="#111", label="v2 Compact Active-Subspace (12–33 sp)"),
        Patch(facecolor=ORANGE, edgecolor="#222", label="Teammate Full-Space K-Split (224–468 sp)"),
        Patch(facecolor=SLATE, edgecolor="#222", label="Zero-Feasibility Hardware Runs"),
    ], loc="upper right", fontsize=7.6, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig07_spqc550_real_hardware_comparison.png"))
    plt.close()


def main():
    print("=== Generating 7 Publication-Grade Figures (300 DPI, English-Only) ===")
    fig01_v1_vs_v2_surrogate()
    print("  [1/7] fig01_v1_vs_v2_surrogate_metrics.png")
    fig02_residual_and_calibration()
    print("  [2/7] fig02_residual_and_calibration.png")
    fig03_closed_loop_fmqa()
    print("  [3/7] fig03_closed_loop_fmqa_benchmark.png")
    fig04_q2_hardware_and_cost()
    print("  [4/7] fig04_q2_v1_vs_v2_hardware_drift.png")
    fig05_q3_compression_and_cost()
    print("  [5/7] fig05_q3_qubit_compression_and_solvers.png")
    fig06_sensitivity_and_mc()
    print("  [6/7] fig06_sensitivity_and_mc_robustness.png")
    fig07_spqc550_real_hardware()
    print("  [7/7] fig07_spqc550_real_hardware_comparison.png")
    print("All figures saved cleanly to v2_optimized/figures/.")


if __name__ == "__main__":
    main()
