#!/usr/bin/env python3
"""
Deep Scientific Analysis for Problem D Paper (F06/F07/F10 Remediated):
  1. In-Data Butler-Ginley Elemental Weight Extraction vs Mulliken Electronegativity
  2. (A,B)-Group-Disjoint Mondrian Split-Conformal Uncertainty Calibration on 03_val
     (200 group-disjoint 50/50 splits of 567 distinct (A,B) pairs, zero group overlap)
  3. 03_val Semiconductor Residual Correlation (e_form - mu_ef vs gap - mu_gap)
  4. Scoped Model Expectation Sum(P_hit) Analysis on 04_test_holdout (NOT true recall)
  5. Publication-Grade Figure 8: Goldilocks Phase Diagram, Scoped Sum(P_hit) Curve,
     and 4-Panel Predicted Band-Edge Audit of Top Blind Candidates.
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)
from common_physics import DATA_DIR, ELEMENT_DATA, fit_band_center_model

RES_DIR = os.path.join(CUR_DIR, "results")
FIG_DIR = os.path.join(CUR_DIR, "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.8,
    "axes.titlesize": 11.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.0,
    "axes.labelweight": "bold",
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.0,
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


def run_deep_analysis():
    tr = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    val_df = pd.read_csv(os.path.join(RES_DIR, "v2_val_predictions.csv"))
    hold_df = pd.read_csv(os.path.join(RES_DIR, "v2_holdout_04.csv")).sort_values("V_hat").reset_index(drop=True)
    hold02_path = os.path.join(RES_DIR, "v2_holdout_04_train02only.csv")
    hold02_df = pd.read_csv(hold02_path).sort_values("V_hat").reset_index(drop=True) if os.path.isfile(hold02_path) else None

    # 1. In-Data Butler-Ginley Elemental Weight Recovery vs Textbook Mulliken Electronegativity
    bc_model, bc_cols = fit_band_center_model(tr)
    coefs = dict(zip(bc_cols, bc_model.coef_))
    elems = sorted(ELEMENT_DATA.keys())
    wA = np.array([coefs.get(f"A_{e}", 0.0) for e in elems])
    wB = np.array([coefs.get(f"B_{e}", 0.0) for e in elems])
    ab_diff_std = float(np.std(wA - wB))
    ab_corr = float(pearsonr(wA, wB).statistic)
    chi_mull = np.array([0.5 * (ELEMENT_DATA[e][5] + ELEMENT_DATA[e][6]) for e in elems])
    bg_mull_r = float(pearsonr(np.log(chi_mull), wA).statistic)
    bg_mull_rho = float(spearmanr(chi_mull, wA).statistic)

    # 2. (A,B)-Group-Disjoint Mondrian Split-Conformal Calibration on 03_val (Frozen 02-only model)
    #    Split the 567 distinct (A,B) groups 50/50 (283 cal / 284 test) across 200 permutations.
    rng = np.random.default_rng(2026)
    groups = (val_df["A"].astype(str) + "_" + val_df["B"].astype(str)).values
    uniq_groups = np.unique(groups)
    n_groups = len(uniq_groups)
    n_cal_groups = n_groups // 2
    n_tst_groups = n_groups - n_cal_groups

    y_true = val_df["V"].values
    y_pred = val_df["V_hat"].values
    p_semi = val_df["p_semi"].values
    abs_err = np.abs(y_true - y_pred)
    stratum = (p_semi >= 0.20).astype(int)  # 0: confident metallic; 1: semiconducting/borderline

    cov90_all, cov95_all = [], []
    cov90_s0_l, cov90_s1_l, cov95_s0_l, cov95_s1_l = [], [], [], []
    q90_s0_l, q90_s1_l, q95_s0_l, q95_s1_l = [], [], [], []
    max_group_overlap = 0

    for _ in range(200):
        perm_g = rng.permutation(uniq_groups)
        cal_g = set(perm_g[:n_cal_groups])
        tst_g = set(perm_g[n_cal_groups:])
        overlap_g = len(cal_g & tst_g)
        assert overlap_g == 0, f"Group overlap detected: {overlap_g}"

        cal_mask = np.isin(groups, list(cal_g))
        tst_mask = np.isin(groups, list(tst_g))
        row_group_overlap = len(set(groups[cal_mask]) & set(groups[tst_mask]))
        assert row_group_overlap == 0, f"Row group overlap detected: {row_group_overlap}"
        max_group_overlap = max(max_group_overlap, row_group_overlap)

        q90, q95 = {}, {}
        for s in (0, 1):
            sc = abs_err[cal_mask & (stratum == s)]
            n_s = len(sc)
            q90[s] = float(np.quantile(sc, min(1.0, np.ceil(0.90 * (n_s + 1)) / n_s))) if n_s > 0 else 0.0
            q95[s] = float(np.quantile(sc, min(1.0, np.ceil(0.95 * (n_s + 1)) / n_s))) if n_s > 0 else 0.0
        q90_s0_l.append(q90[0])
        q90_s1_l.append(q90[1])
        q95_s0_l.append(q95[0])
        q95_s1_l.append(q95[1])

        tst_str = stratum[tst_mask]
        tst_err = abs_err[tst_mask]
        rad90 = np.where(tst_str == 0, q90[0], q90[1])
        rad95 = np.where(tst_str == 0, q95[0], q95[1])
        cov90_all.append(float(np.mean(tst_err <= rad90)))
        cov95_all.append(float(np.mean(tst_err <= rad95)))

        m0 = tst_str == 0
        m1 = tst_str == 1
        if m0.any():
            cov90_s0_l.append(float(np.mean(tst_err[m0] <= q90[0])))
            cov95_s0_l.append(float(np.mean(tst_err[m0] <= q95[0])))
        if m1.any():
            cov90_s1_l.append(float(np.mean(tst_err[m1] <= q90[1])))
            cov95_s1_l.append(float(np.mean(tst_err[m1] <= q95[1])))

    # 3. 03_val Semiconductor (gap > 0) Residual Correlation Check (F06)
    semi_val = val_df[val_df["gap"] > 0].copy()
    res_ef = semi_val["e_form"].values - semi_val["mu_ef"].values
    res_gap = semi_val["gap"].values - semi_val["mu_gap"].values
    pr_res = pearsonr(res_ef, res_gap)
    sr_res = spearmanr(res_ef, res_gap)

    # 4. Scoped Model Expectation Sum(P_hit) on 04_test_holdout (and 03_val diagnostic sum)
    p_by_v = hold_df["P_hit"].values
    p_by_phit = np.sort(hold_df["P_hit"].values)[::-1]
    cum_yield_v = np.cumsum(p_by_v)
    cum_yield_phit = np.cumsum(p_by_phit)
    cum_sd_v = np.sqrt(np.cumsum(p_by_v * (1.0 - p_by_v)))
    total_exp_hits = float(p_by_v.sum())
    total_exp_sd = float(np.sqrt(np.sum(p_by_v * (1.0 - p_by_v))))

    val_sorted = val_df.sort_values("V_hat").reset_index(drop=True)
    sum_phit_03_all = float(val_sorted["P_hit"].sum())
    sum_phit_03_top75 = float(val_sorted.head(75)["P_hit"].sum())

    anion_exp = hold_df.groupby("X")["P_hit"].sum().to_dict()
    anion_top30 = hold_df.head(30)["X"].value_counts().to_dict()

    top20_hold = hold_df.head(20)[
        ["formula", "X", "V_hat", "P_hit", "mu_ef", "p_semi", "mu_gap", "ec_exact",
         "v_form", "v_gap_low", "v_gap_high", "v_vbm", "v_cbm"]
    ].copy()
    top20_hold["vbm_hat"] = top20_hold["ec_exact"] + 0.5 * top20_hold["mu_gap"]
    top20_hold["cbm_hat"] = top20_hold["ec_exact"] - 0.5 * top20_hold["mu_gap"]
    top20_hold.to_csv(os.path.join(RES_DIR, "v2_holdout_top20_detailed.csv"), index=False)

    summary = {
        "butler_ginley_physics": {
            "scope_note": "In-data empirically verified log-additive Butler-Ginley band-center relation on 02_train -> 03_val.",
            "wA_vs_wB_pearson_r": round(ab_corr, 8),
            "wA_minus_wB_std": round(ab_diff_std, 8),
            "log_mulliken_vs_wA_pearson_r": round(bg_mull_r, 4),
            "mulliken_vs_wA_spearman_rho": round(bg_mull_rho, 4),
        },
        "mondrian_conformal_03val": {
            "split_protocol": f"200 (A,B)-group-disjoint 50/50 splits ({n_cal_groups} cal groups / {n_tst_groups} test groups)",
            "predictor_scope": (
                "Scoped strictly to frozen 02-only model predictions on unseen (A,B) pairs in 03_castelli_val.csv. "
                "Does NOT transfer as a finite-sample guarantee to 02+03 refitted 04_holdout or adaptive Top-K subsets."
            ),
            "n_splits": 200,
            "n_unique_ab_groups": int(n_groups),
            "cal_groups_per_split": int(n_cal_groups),
            "test_groups_per_split": int(n_tst_groups),
            "group_overlap": int(max_group_overlap),
            "empirical_coverage_90_pct": round(100.0 * float(np.mean(cov90_all)), 2),
            "empirical_coverage_90_std_pct": round(100.0 * float(np.std(cov90_all)), 2),
            "empirical_coverage_95_pct": round(100.0 * float(np.mean(cov95_all)), 2),
            "empirical_coverage_95_std_pct": round(100.0 * float(np.std(cov95_all)), 2),
            "metal_coverage_90_pct": round(100.0 * float(np.mean(cov90_s0_l)), 2),
            "metal_coverage_95_pct": round(100.0 * float(np.mean(cov95_s0_l)), 2),
            "semi_coverage_90_pct": round(100.0 * float(np.mean(cov90_s1_l)), 2),
            "semi_coverage_95_pct": round(100.0 * float(np.mean(cov95_s1_l)), 2),
            "metal_stratum_q90_radius": round(float(np.mean(q90_s0_l)), 3),
            "metal_stratum_q90_radius_median": round(float(np.median(q90_s0_l)), 3),
            "metal_stratum_q90_radius_std": round(float(np.std(q90_s0_l)), 3),
            "metal_stratum_q95_radius": round(float(np.mean(q95_s0_l)), 3),
            "metal_stratum_q95_radius_median": round(float(np.median(q95_s0_l)), 3),
            "metal_stratum_q95_radius_std": round(float(np.std(q95_s0_l)), 3),
            "semi_stratum_q90_radius": round(float(np.mean(q90_s1_l)), 3),
            "semi_stratum_q90_radius_median": round(float(np.median(q90_s1_l)), 3),
            "semi_stratum_q90_radius_std": round(float(np.std(q90_s1_l)), 3),
            "semi_stratum_q95_radius": round(float(np.mean(q95_s1_l)), 3),
            "semi_stratum_q95_radius_median": round(float(np.median(q95_s1_l)), 3),
            "semi_stratum_q95_radius_std": round(float(np.std(q95_s1_l)), 3),
        },
        "semiconductor_residual_correlation_03val": {
            "n_semiconductors": int(len(semi_val)),
            "pearson_r": round(float(pr_res.statistic), 6),
            "pearson_p": float(pr_res.pvalue),
            "spearman_rho": round(float(sr_res.statistic), 6),
            "spearman_p": float(sr_res.pvalue),
            "res_ef_mean": round(float(np.mean(res_ef)), 4),
            "res_ef_std": round(float(np.std(res_ef)), 4),
            "res_gap_mean": round(float(np.mean(res_gap)), 4),
            "res_gap_std": round(float(np.std(res_gap)), 4),
        },
        "holdout_04_expected_yield": {
            "definition_scope": (
                "Model expectation sums of approximate hit score P_hit = P_form * P_semi * P_gap under assumed "
                "predictive Gaussian marginals and conditional independence. Percentages are fractions of the "
                "model expectation sum sum_P_hit_all, NOT unseen true-positive recall against the 19 hidden V=0 targets."
            ),
            "total_compounds": int(len(hold_df)),
            "known_true_targets": 19,
            "sum_P_hit_all": round(total_exp_hits, 4),
            "sum_P_hit_all_3703": round(total_exp_hits, 2),
            "sum_P_hit_sd": round(total_exp_sd, 2),
            "sum_P_hit_top10": round(float(cum_yield_v[9]), 4),
            "sum_P_hit_top20": round(float(cum_yield_v[19]), 4),
            "sum_P_hit_top30": round(float(cum_yield_v[29]), 4),
            "sum_P_hit_top50": round(float(cum_yield_v[49]), 4),
            "sum_P_hit_top100": round(float(cum_yield_v[99]), 4),
            "expected_hits_top10_by_V": round(float(cum_yield_v[9]), 2),
            "expected_hits_top20_by_V": round(float(cum_yield_v[19]), 2),
            "expected_hits_top30_by_V": round(float(cum_yield_v[29]), 2),
            "expected_hits_top50_by_V": round(float(cum_yield_v[49]), 2),
            "expected_hits_top100_by_V": round(float(cum_yield_v[99]), 2),
            "expected_hits_top20_by_Phit": round(float(cum_yield_phit[19]), 2),
            "expected_hits_top30_by_Phit": round(float(cum_yield_phit[29]), 2),
            "expected_hits_top50_by_Phit": round(float(cum_yield_phit[49]), 2),
            "frac_model_expectation_sum_at_K30_pct": round(100.0 * float(cum_yield_v[29]) / max(total_exp_hits, 1e-6), 1),
            "frac_model_expectation_sum_at_K50_pct": round(100.0 * float(cum_yield_v[49]) / max(total_exp_hits, 1e-6), 1),
            "pct_yield_captured_at_K30": round(100.0 * float(cum_yield_v[29]) / max(total_exp_hits, 1e-6), 1),
            "pct_yield_captured_at_K50": round(100.0 * float(cum_yield_v[49]) / max(total_exp_hits, 1e-6), 1),
            "sum_P_hit_03val_all": round(sum_phit_03_all, 4),
            "sum_P_hit_03val_top75": round(sum_phit_03_top75, 4),
            "sum_P_hit_04_train02only_all": round(float(hold02_df["P_hit"].sum()), 4) if hold02_df is not None else None,
            "anion_expected_hits": {k: round(float(v), 2) for k, v in anion_exp.items()},
            "anion_top30_counts": {k: int(v) for k, v in anion_top30.items()},
        },
    }

    with open(os.path.join(RES_DIR, "v2_deep_physics_conformal.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Deep Scientific Analysis Complete ===")
    print(json.dumps(summary, indent=2))
    print("\nTop-15 Holdout (04) Candidates:")
    print(top20_hold.head(15)[["formula", "X", "V_hat", "P_hit", "mu_ef", "mu_gap", "vbm_hat", "cbm_hat"]].to_string(index=False))

    # Build Figure 8: Scoped Electronic Phase Diagram + Model Expectation Sum(P_hit) + 4-Band Audit
    fig = plt.figure(figsize=(16.5, 9.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.05, 1.0], hspace=0.34, wspace=0.25)

    # Panel A: Goldilocks Photocatalytic Phase Diagram
    ax_a = fig.add_subplot(gs[0, 0])
    ec_h = hold_df["ec_exact"].values
    ef_h = hold_df["mu_ef"].values
    ph_h = hold_df["P_hit"].values
    ax_a.axvspan(4.18, 6.05, color=TEAL, alpha=0.07)
    ax_a.fill_between([4.18, 6.05], -1.2, 0.21, color=TEAL, alpha=0.14, label="Feasible Window (4.18<Ec<6.05, e_form≤0.21)")
    ax_a.axvline(4.18, color=TEAL, linestyle="--", lw=1.1)
    ax_a.axvline(6.05, color=TEAL, linestyle="--", lw=1.1)
    ax_a.axhline(0.21, color=NAVY, linestyle="--", lw=1.1)

    low_m = ph_h < 0.05
    hi_m = ph_h >= 0.05
    ax_a.scatter(ec_h[low_m], ef_h[low_m], s=8, color=SLATE, alpha=0.20, rasterized=True, label="Low-Score Bulk (P_hit < 5%)")
    sc = ax_a.scatter(ec_h[hi_m], ef_h[hi_m], c=100 * ph_h[hi_m], cmap="viridis", s=42, edgecolor="#111111", lw=0.45, zorder=5)
    cb = plt.colorbar(sc, ax=ax_a, pad=0.02)
    cb.set_label("Approx. Hit Score P_hit (%)", fontsize=8.8, fontweight="bold")

    star_labels = {
        "SrTaO2N": (-56, -24),
        "BaTaO2N": (14, -18),
        "LaTiO2N": (52, -6),
        "KNbO3": (10, -24),
        "RbNbO3": (-64, 8),
        "YTiO2N": (14, 18),
        "SrNbO2N": (-58, 28),
    }
    for _, row in hold_df.head(20).iterrows():
        f = row["formula"]
        if f in star_labels:
            dx, dy = star_labels[f]
            ax_a.annotate(f, (row["ec_exact"], row["mu_ef"]), textcoords="offset points", xytext=(dx, dy),
                          fontsize=7.5, fontweight="bold", color="#111111",
                          arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7))
    ax_a.set_xlim(4.0, 7.2)
    ax_a.set_ylim(-0.65, 2.2)
    ax_a.set_xlabel("In-Data Butler–Ginley Band Center E_c (eV)")
    ax_a.set_ylabel("Predicted e_form (eV/atom)")
    ax_a.set_title("(a) Band-Center Phase Map (04_holdout, n=3,703, 02+03 refit)", pad=10)
    ax_a.legend(loc="upper right", fontsize=7.6, frameon=True)

    # Panel B: Cumulative Model Expectation Sum(P_hit) (Explicitly Scoped, Not True Recall)
    ax_b = fig.add_subplot(gs[0, 1])
    ks = np.arange(1, 151)
    ax_b.plot(ks, cum_yield_v[:150], "-", color=NAVY, lw=2.4,
              label=f"Ranked by V_hat (Sum@30={cum_yield_v[29]:.2f}, @50={cum_yield_v[49]:.2f})")
    ax_b.fill_between(ks, np.maximum(0, cum_yield_v[:150] - cum_sd_v[:150]), cum_yield_v[:150] + cum_sd_v[:150],
                      color=NAVY, alpha=0.14, label="±1 Poisson-Binomial SD (indep. approx.)")
    ax_b.plot(ks, cum_yield_phit[:150], "--", color=TEAL, lw=2.1,
              label=f"Ranked by P_hit (Sum@30={cum_yield_phit[29]:.2f}, @50={cum_yield_phit[49]:.2f})")
    ax_b.plot(ks, 19.0 * ks / len(hold_df), ":", color=SLATE, lw=1.8, label="Uniform Random Expectation (0.15 @ K=30)")
    ax_b.axhline(19.0, color=CRIMSON, linestyle="-.", lw=1.4, label="Hidden V=0 Count in 04 (19 total; labels unseen)")
    ax_b.axhline(total_exp_hits, color=ORANGE, linestyle=":", lw=1.4,
                 label=f"Total Model Sum(P_hit) = {total_exp_hits:.2f} ± {total_exp_sd:.2f}")
    ax_b.axvline(30, color=GOLD, linestyle="--", lw=1.2)
    pct30 = 100.0 * cum_yield_v[29] / max(total_exp_hits, 1e-6)
    ax_b.text(32, 3.0,
              f"K=30 (0.81% pool):\nSum(P_hit)={cum_yield_v[29]:.2f}/{total_exp_hits:.2f}\n({pct30:.1f}% of model sum;\nnot unseen true recall)",
              fontsize=8.0, fontweight="bold", color="#8B5A00",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDF2", edgecolor=GOLD))
    ax_b.set_xlabel("Top-K Budget in 04_holdout (out of 3,703)")
    ax_b.set_ylabel("Cumulative Model Expectation Sum(P_hit)")
    ax_b.set_title("(b) Model Expectation Sum(P_hit) vs Budget (Scoped, Not True Recall)")
    ax_b.set_xlim(0, 150)
    ax_b.set_ylim(0, max(22.0, total_exp_hits + 3.0))
    ax_b.legend(loc="lower right", fontsize=7.2, frameon=True)

    # Panel C: 4 aligned predicted band-edge audit subpanels
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, :], wspace=0.14)
    audit_formulas = ["SrTaO2N", "CaTaO2N", "LaTiO2N", "BaTaO2N", "SrNbO2N",
                      "YTiO2N", "CaNbO2N", "KNbO3", "RbNbO3", "SrGeO3"]
    sub_audit = hold_df[hold_df["formula"].isin(audit_formulas)].copy()
    sub_audit["order"] = sub_audit["formula"].map({f: i for i, f in enumerate(audit_formulas)})
    sub_audit = sub_audit.sort_values("order").iloc[::-1].reset_index(drop=True)
    sub_audit["vbm_hat"] = sub_audit["ec_exact"] + 0.5 * sub_audit["mu_gap"]
    sub_audit["cbm_hat"] = sub_audit["ec_exact"] - 0.5 * sub_audit["mu_gap"]
    y_idx = np.arange(len(sub_audit))

    band_specs = [
        ("mu_ef", "Predicted e_form (eV/at)", (-0.45, 0.55), (-0.45, 0.21), lambda v: v <= 0.21, "Target ≤ 0.21"),
        ("mu_gap", "Predicted Band Gap E_g (eV)", (1.1, 3.4), (1.4, 3.1), lambda v: 1.4 <= v <= 3.1, "Target [1.4, 3.1]"),
        ("vbm_hat", "Predicted VBM (eV)", (5.4, 7.2), (5.73, 7.2), lambda v: v >= 5.73, "Target ≥ 5.73"),
        ("cbm_hat", "Predicted CBM (eV)", (3.5, 4.8), (3.5, 4.50), lambda v: v <= 4.50, "Target ≤ 4.50"),
    ]
    for bi, (col, xlabel, xlim, (tlo, thi), ok_fn, t_lbl) in enumerate(band_specs):
        ax = fig.add_subplot(gs_bot[0, bi])
        ax.axvspan(tlo, thi, color=TEAL, alpha=0.14)
        if col == "mu_ef":
            ax.axvline(0.21, color=TEAL, linestyle="--", lw=1.2)
        elif col == "mu_gap":
            ax.axvline(1.4, color=TEAL, linestyle="--", lw=1.2)
            ax.axvline(3.1, color=TEAL, linestyle="--", lw=1.2)
        elif col == "vbm_hat":
            ax.axvline(5.73, color=TEAL, linestyle="--", lw=1.2)
        else:
            ax.axvline(4.50, color=TEAL, linestyle="--", lw=1.2)

        vals = sub_audit[col].values
        for yi, val in enumerate(vals):
            ok = ok_fn(val)
            c = TEAL if ok else ORANGE
            ax.scatter(val, yi, s=70, color=c, edgecolor="#111111", lw=0.7, zorder=5)
            ax.text(val, yi + 0.26, f"{val:.2f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#222222")
        ax.set_xlim(xlim)
        ax.set_ylim(-0.6, len(sub_audit) - 0.1)
        ax.set_yticks(y_idx)
        if bi == 0:
            ax.set_yticklabels([f"{r['formula']} [{r['X']}]" for _, r in sub_audit.iterrows()], fontweight="bold", fontsize=8.8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(xlabel)
        ax.set_title(f"(c{bi+1}) {t_lbl}", fontsize=9.8)

    plt.savefig(os.path.join(FIG_DIR, "fig08_electronic_phase_and_conformal_audit.png"))
    plt.close()
    print("Saved fig08_electronic_phase_and_conformal_audit.png successfully.")


if __name__ == "__main__":
    run_deep_analysis()
