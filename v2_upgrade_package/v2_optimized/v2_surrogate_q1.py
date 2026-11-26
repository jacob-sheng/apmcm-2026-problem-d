#!/usr/bin/env python3
"""
v2 Upgraded Problem 1 Surrogate:
  Physics-Coupled Stage-0 Auxiliary DFT + Multi-Family Ensemble +
  Calibrated Exact Bayes-Optimal Mixture-Expectation Model.

Strictly obeys Problem D evaluation rules:
  - Validation benchmark: trained ONLY on 02_castelli_train.csv, evaluated on
    disjoint (A,B)-pair 03_castelli_val.csv. All OOF encodings, Stage-0 DFT
    predictors, ensemble weights, and probability calibration are fitted strictly
    inside 5-fold (A,B)-GroupKFold on 02_castelli_train.csv (ZERO val leakage).
  - Final pool scoring (04_holdout, 07/11 Q2, 08/12 Q3): refitted cleanly on
    02_castelli_train.csv + 03_castelli_val.csv with identical locked settings.
"""

import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)
from common_physics import (
    DATA_DIR,
    build_v2_physics_features,
    fit_band_center_model,
    predict_band_center,
    v_band_terms,
    exact_expected_hinge_sq,
    compute_band_lower_bound,
    eval_surrogate_metrics,
)

OUT_DIR = os.path.join(CUR_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 2026


def add_oof_marginal_stats(df_train, df_eval_list, n_splits=5):
    """Leak-free out-of-fold marginal physical encodings grouped by (A,B) pair."""
    groups = df_train["A"].astype(str) + "_" + df_train["B"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)

    tr_aux = df_train[["A", "B", "X", "e_form", "gap", "V", "a_lat", "mu_b", "fermi_level"]].copy()
    tr_aux["gap_pos"] = (tr_aux["gap"] > 0).astype(float)
    tr_aux["V_lt2"] = (tr_aux["V"] < 2.0).astype(float)
    tr_aux["is_mag"] = (tr_aux["mu_b"].abs() > 0.05).astype(float)

    targets = ["e_form", "gap_pos", "V", "V_lt2", "a_lat", "mu_b", "is_mag", "fermi_level"]
    keys = [["A"], ["B"], ["X"], ["A", "X"], ["B", "X"]]
    smooth = 5.0

    oof_cols = {}
    eval_cols_list = [{} for _ in df_eval_list]

    for key in keys:
        kname = "_".join(key)
        for tcol in targets:
            col_name = f"te_{kname}_{tcol}"
            prior = float(tr_aux[tcol].mean())
            oof_arr = np.zeros(len(df_train), dtype=float)
            for tr_idx, va_idx in gkf.split(tr_aux, groups=groups):
                sub = tr_aux.iloc[tr_idx]
                stats = sub.groupby(key)[tcol].agg(["count", "mean"])
                stats["sm"] = (stats["count"] * stats["mean"] + smooth * prior) / (stats["count"] + smooth)
                merged = tr_aux.iloc[va_idx][key].merge(stats[["sm"]], on=key, how="left")
                oof_arr[va_idx] = merged["sm"].fillna(prior).values
            oof_cols[col_name] = oof_arr

            full_stats = tr_aux.groupby(key)[tcol].agg(["count", "mean"])
            full_stats["sm"] = (full_stats["count"] * full_stats["mean"] + smooth * prior) / (full_stats["count"] + smooth)
            for idx_e, df_e in enumerate(df_eval_list):
                merged_e = df_e[key].merge(full_stats[["sm"]], on=key, how="left")
                eval_cols_list[idx_e][col_name] = merged_e["sm"].fillna(prior).values

    return pd.DataFrame(oof_cols, index=df_train.index), [
        pd.DataFrame(d, index=df_e.index) for d, df_e in zip(eval_cols_list, df_eval_list)
    ]


def add_stage0_dft_predictors(X_tr_base, df_train, X_eval_list, n_splits=5):
    """
    Stage-0 Auxiliary Multi-Task DFT Predictors:
    Predict latent DFT structural/electronic state (a_lat, |mu_b|, P(|mu_b|>0.05), fermi_level)
    strictly via 5-fold (A,B)-GroupKFold OOF on train, then refit on train for eval sets.
    Physical rationale: open-shell magnetism (|mu_b| > 0.05) strongly flags metallic states (gap=0).
    """
    groups = df_train["A"].astype(str) + "_" + df_train["B"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)

    num_cols = [c for c in X_tr_base.columns if c not in ("A", "B", "X")]
    X_tr_n = X_tr_base[num_cols].values
    X_ev_n = [Xe[num_cols].values for Xe in X_eval_list]

    reg_targets = {
        "s0_a_lat": df_train["a_lat"].values,
        "s0_abs_mub": np.abs(df_train["mu_b"].values),
        "s0_fermi": df_train["fermi_level"].values,
    }
    y_mag = (np.abs(df_train["mu_b"].values) > 0.05).astype(int)

    oof_dict = {}
    eval_dicts = [{} for _ in X_eval_list]
    s0_metrics = {}

    for name, y in reg_targets.items():
        oof = np.zeros(len(df_train), dtype=float)
        ev_preds = [np.zeros(len(Xe), dtype=float) for Xe in X_eval_list]
        for fold, (ti, vi) in enumerate(gkf.split(X_tr_n, y, groups=groups)):
            m = lgb.LGBMRegressor(
                n_estimators=450, learning_rate=0.04, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED + fold, verbose=-1
            )
            m.fit(X_tr_n[ti], y[ti])
            oof[vi] = m.predict(X_tr_n[vi])
            for j, Xe in enumerate(X_ev_n):
                ev_preds[j] += m.predict(Xe) / n_splits
        oof_dict[name] = oof
        for j in range(len(X_eval_list)):
            eval_dicts[j][name] = ev_preds[j]
        s0_metrics[name] = round(float(r2_score(y, oof)), 4)

    # Binary magnetic classifier P(|mu_b| > 0.05)
    oof_mag = np.zeros(len(df_train), dtype=float)
    ev_mag = [np.zeros(len(Xe), dtype=float) for Xe in X_eval_list]
    for fold, (ti, vi) in enumerate(gkf.split(X_tr_n, y_mag, groups=groups)):
        mc = lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.04, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED + 100 + fold, verbose=-1
        )
        mc.fit(X_tr_n[ti], y_mag[ti])
        oof_mag[vi] = mc.predict_proba(X_tr_n[vi])[:, 1]
        for j, Xe in enumerate(X_ev_n):
            ev_mag[j] += mc.predict_proba(Xe)[:, 1] / n_splits
    oof_dict["s0_p_mag"] = oof_mag
    for j in range(len(X_eval_list)):
        eval_dicts[j]["s0_p_mag"] = ev_mag[j]
    s0_metrics["s0_p_mag_roc_auc"] = round(float(roc_auc_score(y_mag, oof_mag)), 4)

    return (
        pd.DataFrame(oof_dict, index=df_train.index),
        [pd.DataFrame(d, index=Xe.index) for d, Xe in zip(eval_dicts, X_eval_list)],
        s0_metrics,
    )


def build_feature_bundle(df_train, df_eval_list):
    """Assemble full v2 feature matrices: crystal physics + Butler-Ginley Ec + OOF marginals + Stage-0 DFT."""
    bc_model, bc_cols = fit_band_center_model(df_train)
    ec_tr = predict_band_center(bc_model, bc_cols, df_train)
    ec_evals = [predict_band_center(bc_model, bc_cols, df_e) for df_e in df_eval_list]

    def attach_ec(phys_df, ec, df_raw):
        g_min, feas_w, v_min, v_met = compute_band_lower_bound(ec)
        out = phys_df.copy()
        out["ec_exact"] = ec
        out["ec_dist_center"] = np.abs(ec - 5.115)
        out["ec_in_window"] = ((ec > 4.18) & (ec < 6.05)).astype(int)
        out["g_min_zero"] = g_min
        out["feasible_gap_window"] = feas_w
        out["v_band_lower_bound"] = v_min
        out["v_band_metal"] = v_met
        out["A"] = df_raw["A"].values
        out["B"] = df_raw["B"].values
        out["X"] = df_raw["X"].values
        return out

    phys_tr = attach_ec(build_v2_physics_features(df_train), ec_tr, df_train)
    phys_evals = [
        attach_ec(build_v2_physics_features(df_e), ec_e, df_e)
        for df_e, ec_e in zip(df_eval_list, ec_evals)
    ]

    te_tr, te_evals = add_oof_marginal_stats(df_train, df_eval_list, n_splits=5)
    base_tr = pd.concat([phys_tr, te_tr], axis=1)
    base_evals = [pd.concat([pe, te], axis=1) for pe, te in zip(phys_evals, te_evals)]

    s0_tr, s0_evals, s0_metrics = add_stage0_dft_predictors(base_tr, df_train, base_evals, n_splits=5)
    full_tr = pd.concat([base_tr, s0_tr], axis=1)
    full_evals = [pd.concat([be, s0e], axis=1) for be, s0e in zip(base_evals, s0_evals)]

    return full_tr, full_evals, ec_tr, ec_evals, s0_metrics


def _fit_optimal_weights(oof_mat, y_true, loss="mse"):
    """Find simplex weights w >= 0, sum(w)=1 minimizing OOF MSE or log-loss."""
    k = oof_mat.shape[1]

    def obj(w):
        pred = oof_mat @ w
        if loss == "mse":
            return float(np.mean((pred - y_true) ** 2))
        p = np.clip(pred, 1e-6, 1.0 - 1e-6)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    res = minimize(
        obj,
        np.ones(k) / k,
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    w = np.maximum(0.0, res.x)
    return w / np.sum(w)


def _train_4family_regressor(X_tr, y_tr, X_evals, groups, n_splits=5, seed_offset=0):
    """4-family (CatBoost, LightGBM, XGBoost, ExtraTrees) OOF CV + simplex stacking."""
    cat_cols = ["A", "B", "X"]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]

    # Prepare numeric + category views
    X_tr_num = X_tr[num_cols].values.astype(np.float32)
    X_ev_num = [Xe[num_cols].values.astype(np.float32) for Xe in X_evals]

    X_tr_lgb = X_tr.copy()
    X_ev_lgb = [Xe.copy() for Xe in X_evals]
    for c in cat_cols:
        cats = sorted(set(X_tr[c]).union(*[set(Xe[c]) for Xe in X_evals]))
        X_tr_lgb[c] = pd.Categorical(X_tr_lgb[c], categories=cats)
        for Xe in X_ev_lgb:
            Xe[c] = pd.Categorical(Xe[c], categories=cats)

    gkf = GroupKFold(n_splits=n_splits)
    n = len(X_tr)
    oof = np.zeros((n, 4), dtype=float)
    ev_preds = [np.zeros((len(Xe), 4), dtype=float) for Xe in X_evals]

    for fold, (ti, vi) in enumerate(gkf.split(X_tr_num, y_tr, groups=groups)):
        s = SEED + seed_offset + fold * 13

        # 0: CatBoost
        cb = CatBoostRegressor(
            iterations=600, learning_rate=0.055, depth=6, l2_leaf_reg=4.0,
            one_hot_max_size=60, max_ctr_complexity=1, thread_count=-1,
            random_seed=s, verbose=0
        )
        cb.fit(X_tr.iloc[ti], y_tr[ti], cat_features=cat_cols)
        oof[vi, 0] = cb.predict(X_tr.iloc[vi])
        for j, Xe in enumerate(X_evals):
            ev_preds[j][:, 0] += cb.predict(Xe) / n_splits

        # 1: LightGBM
        m_lgb = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.04, num_leaves=31, min_child_samples=15,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.5,
            n_jobs=-1, random_state=s, verbose=-1
        )
        m_lgb.fit(X_tr_lgb.iloc[ti], y_tr[ti])
        oof[vi, 1] = m_lgb.predict(X_tr_lgb.iloc[vi])
        for j, Xe in enumerate(X_ev_lgb):
            ev_preds[j][:, 1] += m_lgb.predict(Xe) / n_splits

        # 2: XGBoost
        m_xgb = xgb.XGBRegressor(
            n_estimators=550, learning_rate=0.04, max_depth=6, subsample=0.8,
            colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=2.0,
            random_state=s, n_jobs=-1, tree_method="hist", verbosity=0
        )
        m_xgb.fit(X_tr_num[ti], y_tr[ti])
        oof[vi, 2] = m_xgb.predict(X_tr_num[vi])
        for j, Xe in enumerate(X_ev_num):
            ev_preds[j][:, 2] += m_xgb.predict(Xe) / n_splits

        # 3: ExtraTrees (variance-reducing non-boosted tree family)
        m_et = ExtraTreesRegressor(
            n_estimators=160, max_features=0.60, min_samples_leaf=2,
            random_state=s, n_jobs=-1
        )
        m_et.fit(X_tr_num[ti], y_tr[ti])
        oof[vi, 3] = m_et.predict(X_tr_num[vi])
        for j, Xe in enumerate(X_ev_num):
            ev_preds[j][:, 3] += m_et.predict(Xe) / n_splits

    w = _fit_optimal_weights(oof, y_tr, loss="mse")
    oof_ens = oof @ w
    oof_std = np.std(oof, axis=1)
    resid_sigma = float(np.sqrt(mean_squared_error(y_tr, oof_ens)))
    ev_ens = [ep @ w for ep in ev_preds]
    ev_std = [np.std(ep, axis=1) for ep in ev_preds]
    return oof_ens, oof_std, ev_ens, ev_std, resid_sigma, [round(float(x), 4) for x in w]


def _train_4family_classifier(X_tr, y_tr, X_evals, groups, n_splits=5, seed_offset=500):
    """4-family (CatBoost, LightGBM, XGBoost, ExtraTrees) OOF CV + isotonic calibration."""
    cat_cols = ["A", "B", "X"]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]

    X_tr_num = X_tr[num_cols].values.astype(np.float32)
    X_ev_num = [Xe[num_cols].values.astype(np.float32) for Xe in X_evals]

    X_tr_lgb = X_tr.copy()
    X_ev_lgb = [Xe.copy() for Xe in X_evals]
    for c in cat_cols:
        cats = sorted(set(X_tr[c]).union(*[set(Xe[c]) for Xe in X_evals]))
        X_tr_lgb[c] = pd.Categorical(X_tr_lgb[c], categories=cats)
        for Xe in X_ev_lgb:
            Xe[c] = pd.Categorical(Xe[c], categories=cats)

    gkf = GroupKFold(n_splits=n_splits)
    n = len(X_tr)
    oof = np.zeros((n, 4), dtype=float)
    ev_preds = [np.zeros((len(Xe), 4), dtype=float) for Xe in X_evals]

    for fold, (ti, vi) in enumerate(gkf.split(X_tr_num, y_tr, groups=groups)):
        s = SEED + seed_offset + fold * 13

        cb = CatBoostClassifier(
            iterations=550, learning_rate=0.055, depth=6, l2_leaf_reg=4.0,
            one_hot_max_size=60, max_ctr_complexity=1, thread_count=-1,
            random_seed=s, verbose=0
        )
        cb.fit(X_tr.iloc[ti], y_tr[ti], cat_features=cat_cols)
        oof[vi, 0] = cb.predict_proba(X_tr.iloc[vi])[:, 1]
        for j, Xe in enumerate(X_evals):
            ev_preds[j][:, 0] += cb.predict_proba(Xe)[:, 1] / n_splits

        m_lgb = lgb.LGBMClassifier(
            n_estimators=550, learning_rate=0.04, num_leaves=31, min_child_samples=15,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.5,
            n_jobs=-1, random_state=s, verbose=-1
        )
        m_lgb.fit(X_tr_lgb.iloc[ti], y_tr[ti])
        oof[vi, 1] = m_lgb.predict_proba(X_tr_lgb.iloc[vi])[:, 1]
        for j, Xe in enumerate(X_ev_lgb):
            ev_preds[j][:, 1] += m_lgb.predict_proba(Xe)[:, 1] / n_splits

        m_xgb = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.04, max_depth=6, subsample=0.8,
            colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=2.0,
            random_state=s, n_jobs=-1, tree_method="hist", eval_metric="logloss", verbosity=0
        )
        m_xgb.fit(X_tr_num[ti], y_tr[ti])
        oof[vi, 2] = m_xgb.predict_proba(X_tr_num[vi])[:, 1]
        for j, Xe in enumerate(X_ev_num):
            ev_preds[j][:, 2] += m_xgb.predict_proba(Xe)[:, 1] / n_splits

        m_et = ExtraTreesClassifier(
            n_estimators=160, max_features=0.60, min_samples_leaf=2,
            random_state=s, n_jobs=-1
        )
        m_et.fit(X_tr_num[ti], y_tr[ti])
        oof[vi, 3] = m_et.predict_proba(X_tr_num[vi])[:, 1]
        for j, Xe in enumerate(X_ev_num):
            ev_preds[j][:, 3] += m_et.predict_proba(Xe)[:, 1] / n_splits

    w = _fit_optimal_weights(oof, y_tr, loss="logloss")
    oof_raw = np.clip(oof @ w, 1e-6, 1.0 - 1e-6)
    ev_raw = [np.clip(ep @ w, 1e-6, 1.0 - 1e-6) for ep in ev_preds]

    # Calibrate on OOF probabilities using IsotonicRegression blended 50/50 with raw probability
    # (prevents staircase overfitting while preserving Brier calibration)
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(oof_raw, y_tr)
    oof_cal = 0.5 * oof_raw + 0.5 * iso.predict(oof_raw)
    ev_cal = [0.5 * p + 0.5 * iso.predict(p) for p in ev_raw]
    return oof_cal, ev_cal, [round(float(x), 4) for x in w]


def integrate_bayes_mixture(mu_ef, std_ef, sig_ef, p_semi, mu_g, std_g, sig_g, ec):
    """
    Exact Bayes-optimal mixture integration over metal (gap=0) and semiconductor (gap>0)
    regimes with heteroscedastic uncertainty blending residual RMS and ensemble disagreement.
    Returns total expected violation V_hat, 5-term breakdown, and exact P_hit = P(V=0).
    """
    # Calibrated per-sample sigma combining global OOF residual scale + ensemble spread
    s_ef = np.sqrt(0.75 * (sig_ef ** 2) + 0.25 * (np.maximum(std_ef, 0.05) ** 2))
    s_g = np.sqrt(0.75 * (sig_g ** 2) + 0.25 * (np.maximum(std_g, 0.10) ** 2))

    # 1. Formation energy term: E[max(0, (e_form - 0.21)/0.5)^2]
    v_form = exact_expected_hinge_sq(mu_ef, s_ef, threshold=0.21, sign=+1.0, scale=0.5)

    # 2. Metal regime (gap = 0) exact deterministic electronic violation
    vm_gl, vm_gh, vm_vbm, vm_cbm = v_band_terms(np.zeros_like(ec), ec)

    # 3. Semiconductor regime (gap ~ N(mu_g, s_g^2)) exact Gaussian hinge expectations
    vs_gl = exact_expected_hinge_sq(mu_g, s_g, threshold=1.4, sign=-1.0, scale=0.5)
    vs_gh = exact_expected_hinge_sq(mu_g, s_g, threshold=3.1, sign=+1.0, scale=0.5)
    # VBM = ec + 0.5*g < 5.73 <=> g < 2*(5.73 - ec)
    vs_vbm = exact_expected_hinge_sq(ec + 0.5 * mu_g, 0.5 * s_g, threshold=5.73, sign=-1.0, scale=0.5)
    # CBM = ec - 0.5*g > 4.50 <=> -0.5*g > 4.50 - ec
    vs_cbm = exact_expected_hinge_sq(ec - 0.5 * mu_g, 0.5 * s_g, threshold=4.50, sign=+1.0, scale=0.5)

    v_gl = (1.0 - p_semi) * vm_gl + p_semi * vs_gl
    v_gh = (1.0 - p_semi) * vm_gh + p_semi * vs_gh
    v_vbm = (1.0 - p_semi) * vm_vbm + p_semi * vs_vbm
    v_cbm = (1.0 - p_semi) * vm_cbm + p_semi * vs_cbm
    v_tot = v_form + v_gl + v_gh + v_vbm + v_cbm

    # Exact hit probability P(V = 0 | x)
    p_form_ok = norm.cdf((0.21 - mu_ef) / s_ef)
    g_low_req = np.maximum.reduce([np.full_like(ec, 1.4), 2.0 * (5.73 - ec), 2.0 * (ec - 4.50)])
    p_gap_ok = np.where(
        g_low_req < 3.1,
        np.maximum(0.0, norm.cdf((3.1 - mu_g) / s_g) - norm.cdf((g_low_req - mu_g) / s_g)),
        0.0,
    )
    p_hit = p_form_ok * p_semi * p_gap_ok

    return {
        "V_hat": v_tot,
        "v_form": v_form,
        "v_gap_low": v_gl,
        "v_gap_high": v_gh,
        "v_vbm": v_vbm,
        "v_cbm": v_cbm,
        "P_hit": p_hit,
        "mu_ef": mu_ef,
        "sigma_ef": s_ef,
        "p_semi": p_semi,
        "mu_gap": mu_g,
        "sigma_gap": s_g,
        "ec_exact": ec,
    }


def fit_and_predict_v2(df_train, df_eval_list, fit_direct_ablation=True):
    """Full end-to-end v2 training on df_train and inference on df_eval_list."""
    print("  -> Building physics + Butler-Ginley Ec + OOF marginals + Stage-0 DFT features...", flush=True)
    X_tr, X_evals, ec_tr, ec_evals, s0_metrics = build_feature_bundle(df_train, df_eval_list)
    groups = (df_train["A"].astype(str) + "_" + df_train["B"].astype(str)).values

    # Sub-model 1: Formation energy e_form
    print("  -> Training 4-family regressor: e_form...", flush=True)
    oof_ef, oof_ef_std, ev_ef, ev_ef_std, sig_ef, w_ef = _train_4family_regressor(
        X_tr, df_train["e_form"].values, X_evals, groups, n_splits=5, seed_offset=0
    )

    # Sub-model 2: Semiconductor probability P(gap > 0)
    print("  -> Training 4-family classifier: P(gap > 0)...", flush=True)
    y_semi = (df_train["gap"].values > 0).astype(int)
    oof_psemi, ev_psemi, w_cls = _train_4family_classifier(
        X_tr, y_semi, X_evals, groups, n_splits=5, seed_offset=500
    )

    # Sub-model 3: Conditional semiconductor gap (trained on gap > 0 subset)
    print("  -> Training 4-family regressor: gap | semi...", flush=True)
    semi_mask = y_semi == 1
    X_tr_semi = X_tr.iloc[np.where(semi_mask)[0]].reset_index(drop=True)
    y_g_semi = df_train.loc[semi_mask, "gap"].values
    g_semi = groups[semi_mask]
    oof_g, oof_g_std, ev_g, ev_g_std, sig_g, w_gap = _train_4family_regressor(
        X_tr_semi, y_g_semi, X_evals, g_semi, n_splits=5, seed_offset=1000
    )

    # Direct physics-enhanced regressor (ablation reference: only on Phase 1 validation)
    if fit_direct_ablation:
        print("  -> Training 4-family direct ablation regressor: V...", flush=True)
        _, _, ev_dir, _, _, _ = _train_4family_regressor(
            X_tr, df_train["V"].values, X_evals, groups, n_splits=5, seed_offset=1500
        )
    else:
        ev_dir = [np.zeros(len(df_e)) for df_e in df_eval_list]

    # Integrate Bayes mixture on every eval set
    eval_outputs = []
    for j in range(len(df_eval_list)):
        res = integrate_bayes_mixture(
            ev_ef[j], ev_ef_std[j], sig_ef,
            ev_psemi[j],
            np.maximum(0.05, ev_g[j]), ev_g_std[j], sig_g,
            ec_evals[j],
        )
        res["V_direct"] = np.maximum(0.0, ev_dir[j])
        eval_outputs.append(res)

    meta = {
        "stage0_oof_metrics": s0_metrics,
        "weights_e_form": dict(zip(["CatBoost", "LightGBM", "XGBoost", "ExtraTrees"], w_ef)),
        "weights_p_semi": dict(zip(["CatBoost", "LightGBM", "XGBoost", "ExtraTrees"], w_cls)),
        "weights_gap_semi": dict(zip(["CatBoost", "LightGBM", "XGBoost", "ExtraTrees"], w_gap)),
        "oof_sigma_e_form": round(sig_ef, 4),
        "oof_sigma_gap_semi": round(sig_g, 4),
    }
    return eval_outputs, meta


def main():
    t0 = time.perf_counter()
    tr = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "03_castelli_val.csv"))

    print("=== [Phase 1] Training v2 strictly on 02_train -> Evaluating on 03_val ===")
    val_outs, meta_val = fit_and_predict_v2(tr, [va])
    vo = val_outs[0]
    dt_val = time.perf_counter() - t0

    y_va = va["V"].values
    ec_true = 0.5 * (va["vbm"].values + va["cbm"].values)
    ec_rmse = float(np.sqrt(mean_squared_error(ec_true, vo["ec_exact"])))
    ec_r2 = float(r2_score(ec_true, vo["ec_exact"]))

    # Sub-model validation metrics
    ef_r2 = float(r2_score(va["e_form"].values, vo["mu_ef"]))
    ef_rmse = float(np.sqrt(mean_squared_error(va["e_form"].values, vo["mu_ef"])))
    y_semi_va = (va["gap"].values > 0).astype(int)
    cls_roc = float(roc_auc_score(y_semi_va, vo["p_semi"]))
    cls_pr = float(average_precision_score(y_semi_va, vo["p_semi"]))
    cls_brier = float(brier_score_loss(y_semi_va, vo["p_semi"]))
    sm_va = y_semi_va == 1
    gap_r2 = float(r2_score(va.loc[sm_va, "gap"].values, vo["mu_gap"][sm_va]))
    gap_rmse = float(np.sqrt(mean_squared_error(va.loc[sm_va, "gap"].values, vo["mu_gap"][sm_va])))

    # Compute overall surrogate metrics for both v2 Direct and v2 PhysHurdle Bayes
    m_v2_dir = eval_surrogate_metrics(y_va, vo["V_direct"], "v2_Direct_4FamPhys")
    m_v2_bayes = eval_surrogate_metrics(y_va, vo["V_hat"], "v2_PhysHurdle_Bayes")

    # Residual & calibration diagnostics on 03_val
    resid = vo["V_hat"] - y_va
    metal_mask = y_semi_va == 0
    semi_mask = y_semi_va == 1
    # 10-bin calibration curve for p_semi
    bins = np.linspace(0.0, 1.0, 11)
    cal_curve = []
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (vo["p_semi"] >= b0) & (vo["p_semi"] < b1 + (1e-9 if b1 == 1.0 else 0.0))
        if m.sum() > 0:
            mp = float(vo["p_semi"][m].mean())
            mf = float(y_semi_va[m].mean())
            cnt = int(m.sum())
            ece += (cnt / len(y_semi_va)) * abs(mp - mf)
            cal_curve.append({"bin_low": round(float(b0), 2), "bin_high": round(float(b1), 2),
                              "mean_pred": round(mp, 4), "obs_freq": round(mf, 4), "count": cnt})

    diagnostics = {
        "val_runtime_sec": round(dt_val, 3),
        "band_center_val_r2": round(ec_r2, 10),
        "band_center_val_rmse_eV": ec_rmse,
        "component_val_metrics": {
            "e_form_R2": round(ef_r2, 4),
            "e_form_RMSE": round(ef_rmse, 4),
            "p_semi_ROC_AUC": round(cls_roc, 4),
            "p_semi_PR_AUC": round(cls_pr, 4),
            "p_semi_Brier": round(cls_brier, 5),
            "p_semi_ECE": round(float(ece), 5),
            "gap_semi_R2": round(gap_r2, 4),
            "gap_semi_RMSE": round(gap_rmse, 4),
        },
        "residual_diagnostics": {
            "overall_bias": round(float(np.mean(resid)), 4),
            "overall_std": round(float(np.std(resid)), 4),
            "metal_rmse": round(float(np.sqrt(np.mean(resid[metal_mask] ** 2))), 4),
            "metal_mae": round(float(np.mean(np.abs(resid[metal_mask]))), 4),
            "semi_rmse": round(float(np.sqrt(np.mean(resid[semi_mask] ** 2))), 4),
            "semi_mae": round(float(np.mean(np.abs(resid[semi_mask]))), 4),
            "near_zero_Vlt1_mae": round(float(np.mean(np.abs(resid[y_va < 1.0]))), 4),
        },
        "calibration_curve": cal_curve,
        "ensemble_meta": meta_val,
        "metrics_table": [m_v2_dir, m_v2_bayes],
    }

    print(f"\n[v2 Q1 Validation] Completed in {dt_val:.2f}s:")
    print(pd.DataFrame([m_v2_dir, m_v2_bayes]).to_string(index=False))
    print("\nComponent Metrics:", json.dumps(diagnostics["component_val_metrics"], indent=2))
    print("Stage-0 OOF Metrics:", json.dumps(meta_val["stage0_oof_metrics"], indent=2))
    print("Residual Diagnostics:", json.dumps(diagnostics["residual_diagnostics"], indent=2))

    # Save validation predictions for figures & residual analysis
    val_df = va[["id", "formula", "A", "B", "X", "e_form", "gap", "vbm", "cbm", "V"]].copy()
    val_df["ec_true"] = ec_true
    for k in ["V_hat", "V_direct", "v_form", "v_gap_low", "v_gap_high", "v_vbm", "v_cbm",
              "P_hit", "mu_ef", "p_semi", "mu_gap", "ec_exact"]:
        val_df[k] = vo[k]
    val_df.to_csv(os.path.join(OUT_DIR, "v2_val_predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "v2_q1_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # Phase 2: Refit cleanly on 02_train + 03_val for final holdout & Q2/Q3 pools
    print("\n=== [Phase 2] Refitting v2 on 02_train + 03_val for Q2/Q3/Holdout pools ===")
    t1 = time.perf_counter()
    tv = pd.concat([tr, va], ignore_index=True)
    holdout = pd.read_csv(os.path.join(DATA_DIR, "04_castelli_test_holdout.csv"))
    q2_full = pd.read_csv(os.path.join(DATA_DIR, "07_castelli_q2_pool.csv"))
    q2_blind = pd.read_csv(os.path.join(DATA_DIR, "11_castelli_q2_pool_blind.csv"))
    q3_full = pd.read_csv(os.path.join(DATA_DIR, "08_castelli_q3_pool.csv"))
    q3_blind = pd.read_csv(os.path.join(DATA_DIR, "12_castelli_q3_pool_blind.csv"))

    pool_outs, meta_full = fit_and_predict_v2(tv, [holdout, q2_full, q2_blind, q3_full, q3_blind], fit_direct_ablation=False)
    names = ["holdout_04", "q2_full_07", "q2_blind_11", "q3_full_08", "q3_blind_12"]
    dfs = [holdout, q2_full, q2_blind, q3_full, q3_blind]

    for nm, dfp, res in zip(names, dfs, pool_outs):
        out_df = dfp[["id", "formula", "A", "B", "X", "a_eq_b"]].copy()
        for k in ["V_hat", "v_form", "v_gap_low", "v_gap_high", "v_vbm", "v_cbm",
                  "P_hit", "mu_ef", "sigma_ef", "p_semi", "mu_gap", "sigma_gap", "ec_exact"]:
            out_df[k] = res[k]
        out_df = out_df.sort_values("V_hat", ascending=True).reset_index(drop=True)
        out_df.to_csv(os.path.join(OUT_DIR, f"v2_{nm}.csv"), index=False)

    print(f"[Phase 2] Refit & pool scoring finished in {time.perf_counter() - t1:.2f}s.")
    print("\nTop-10 Q2 Blind (11) Candidates (v2):")
    q2b = pd.read_csv(os.path.join(OUT_DIR, "v2_q2_blind_11.csv")).head(10)
    print(q2b[["formula", "V_hat", "P_hit", "mu_ef", "p_semi", "mu_gap", "ec_exact"]].to_string(index=False))
    print("\nTop-15 Q3 Blind (12) Candidates (v2):")
    q3b = pd.read_csv(os.path.join(OUT_DIR, "v2_q3_blind_12.csv")).head(15)
    print(q3b[["formula", "X", "V_hat", "P_hit", "mu_ef", "p_semi", "mu_gap", "ec_exact"]].to_string(index=False))


if __name__ == "__main__":
    main()
