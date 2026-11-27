#!/usr/bin/env python3
"""
v2 Upgraded Problem 1 Surrogate (Strict Zero-Leakage Nested Outer-(A,B)-GroupKFold):
  Physics-Coupled Stage-0 Auxiliary DFT + Multi-Family Ensemble +
  Calibrated Bayes-Optimal Mixture-Expectation Model.

Strictly obeys Problem D evaluation rules and F05/F12/F25/F26 remediation:
  - Zero-leakage outer (A,B)-GroupKFold: inside EVERY outer fold (ti, vi),
    Butler-Ginley Ridge weights, marginal target-encoding priors & inner-OOF stats,
    Stage-0 DFT predictors, 4-family main models, and conditional semiconductor
    gap models (using the exact same outer fold partition) are fitted strictly
    on df_tr_fold ONLY. Ensemble weights, residual scales, and IsotonicRegression
    calibrators are fitted strictly on leakage-free outer OOF predictions.
  - Automated +100 label-perturbation isolation test verifies max_feature_delta < 1e-12
    and max_pred_delta < 1e-12.
  - Fair baseline alignment loads M4_PIMT_Physics_Only and M5_PIMT_Hybrid_Ensemble
    from the original repo table1_surrogate_ablation.csv alongside v1 baselines and
    near-target MAE(V<1) / MAE(V<0.5) metrics on the 02 -> 03 protocol.
  - Dual holdout exports: v2_holdout_04_train02only.csv (frozen 02-only) and
    v2_holdout_04.csv (02+03 refit), plus A=B sensitivity diagnostics.
"""

import hashlib
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(CUR_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
sys.path.insert(0, CUR_DIR)

from common_physics import (
    DATA_DIR,
    build_basic_v1_features,
    build_v2_physics_features,
    fit_band_center_model,
    predict_band_center,
    v_band_terms,
    exact_expected_hinge_sq,
    compute_band_lower_bound,
    eval_surrogate_metrics,
)
from v1_baseline import SimpleOneHotFM

OUT_DIR = os.path.join(CUR_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 2026
N_JOBS = 2  # CPU throttling constraint (R6)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _eval_extended_metrics(y_true, y_pred, name=""):
    """Extend common_physics.eval_surrogate_metrics with near-target MAE and Top-K(V<1) recall."""
    m = eval_surrogate_metrics(y_true, y_pred, name=name)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(y_pred)
    n = len(y_true)
    mask10 = y_true < 1.0
    mask05 = y_true < 0.5
    n10 = max(1, int(mask10.sum()))
    rec20_10 = float(mask10[order[:20]].sum() / n10)
    rec50_10 = float(mask10[order[:50]].sum() / n10)
    ef50_10 = float(rec50_10 / (50.0 / n))
    mae_lt1 = float(np.mean(np.abs(y_pred[mask10] - y_true[mask10]))) if mask10.any() else None
    mae_lt05 = float(np.mean(np.abs(y_pred[mask05] - y_true[mask05]))) if mask05.any() else None
    m["Rec@20(V<1.0)"] = round(rec20_10, 4)
    m["Rec@50(V<1.0)"] = round(rec50_10, 4)
    m["EF@50(V<1.0)"] = round(ef50_10, 2)
    m["MAE_V_lt_1"] = round(mae_lt1, 4) if mae_lt1 is not None else None
    m["MAE_V_lt_0_5"] = round(mae_lt05, 4) if mae_lt05 is not None else None
    return m


def add_oof_marginal_stats(df_tr_fold, df_eval_list, n_splits=5):
    """
    Leak-free marginal physical encodings fitted strictly on df_tr_fold ONLY.
    Inner OOF uses inner-fold priors; eval sets use df_tr_fold prior & full df_tr_fold stats.
    Never reads any label column from df_eval_list.
    """
    groups = df_tr_fold["A"].astype(str) + "_" + df_tr_fold["B"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)

    tr_aux = df_tr_fold[["A", "B", "X", "e_form", "gap", "V", "a_lat", "mu_b", "fermi_level"]].copy()
    tr_aux["gap_pos"] = (tr_aux["gap"] > 0).astype(float)
    tr_aux["V_lt2"] = (tr_aux["V"] < 2.0).astype(float)
    tr_aux["is_mag"] = (tr_aux["mu_b"].abs() > 0.05).astype(float)

    targets = ["e_form", "gap_pos", "V", "V_lt2", "a_lat", "mu_b", "is_mag", "fermi_level"]
    keys = [["A"], ["B"], ["X"], ["A", "X"], ["B", "X"]]
    smooth = 5.0

    oof_cols = {}
    eval_cols_list = [{} for _ in df_eval_list]
    splits = list(gkf.split(tr_aux, groups=groups))

    for key in keys:
        kname = "_".join(key)
        for tcol in targets:
            col_name = f"te_{kname}_{tcol}"
            oof_arr = np.zeros(len(df_tr_fold), dtype=float)
            for tr_idx, va_idx in splits:
                sub = tr_aux.iloc[tr_idx]
                inner_prior = float(sub[tcol].mean())
                stats = sub.groupby(key)[tcol].agg(["count", "mean"])
                stats["sm"] = (stats["count"] * stats["mean"] + smooth * inner_prior) / (stats["count"] + smooth)
                merged = tr_aux.iloc[va_idx][key].merge(stats[["sm"]], on=key, how="left")
                oof_arr[va_idx] = merged["sm"].fillna(inner_prior).values
            oof_cols[col_name] = oof_arr

            fold_prior = float(tr_aux[tcol].mean())
            full_stats = tr_aux.groupby(key)[tcol].agg(["count", "mean"])
            full_stats["sm"] = (full_stats["count"] * full_stats["mean"] + smooth * fold_prior) / (full_stats["count"] + smooth)
            for idx_e, df_e in enumerate(df_eval_list):
                merged_e = df_e[key].merge(full_stats[["sm"]], on=key, how="left")
                eval_cols_list[idx_e][col_name] = merged_e["sm"].fillna(fold_prior).values

    return pd.DataFrame(oof_cols, index=df_tr_fold.index), [
        pd.DataFrame(d, index=df_e.index) for d, df_e in zip(eval_cols_list, df_eval_list)
    ]


def add_stage0_dft_predictors(X_tr_base, df_tr_fold, X_eval_list, n_splits=5, seed_base=SEED):
    """
    Stage-0 Auxiliary Multi-Task DFT Predictors fitted strictly on df_tr_fold ONLY.
    Uses inner (A,B)-GroupKFold OOF on df_tr_fold, averages inner models on X_eval_list.
    """
    groups = df_tr_fold["A"].astype(str) + "_" + df_tr_fold["B"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)

    num_cols = [c for c in X_tr_base.columns if c not in ("A", "B", "X")]
    X_tr_n = X_tr_base[num_cols].values
    X_ev_n = [Xe[num_cols].values for Xe in X_eval_list]

    reg_targets = {
        "s0_a_lat": df_tr_fold["a_lat"].values,
        "s0_abs_mub": np.abs(df_tr_fold["mu_b"].values),
        "s0_fermi": df_tr_fold["fermi_level"].values,
    }
    y_mag = (np.abs(df_tr_fold["mu_b"].values) > 0.05).astype(int)

    oof_dict = {}
    eval_dicts = [{} for _ in X_eval_list]
    s0_metrics = {}
    splits = list(gkf.split(X_tr_n, groups=groups))

    for t_idx, (name, y) in enumerate(reg_targets.items()):
        oof = np.zeros(len(df_tr_fold), dtype=float)
        ev_preds = [np.zeros(len(Xe), dtype=float) for Xe in X_eval_list]
        for fold, (ti, vi) in enumerate(splits):
            m = lgb.LGBMRegressor(
                n_estimators=450, learning_rate=0.04, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS,
                random_state=seed_base + t_idx * 50 + fold, verbose=-1
            )
            m.fit(X_tr_n[ti], y[ti])
            oof[vi] = m.predict(X_tr_n[vi])
            for j, Xe in enumerate(X_ev_n):
                ev_preds[j] += m.predict(Xe) / n_splits
        oof_dict[name] = oof
        for j in range(len(X_eval_list)):
            eval_dicts[j][name] = ev_preds[j]
        s0_metrics[name] = round(float(r2_score(y, oof)), 4)

    oof_mag = np.zeros(len(df_tr_fold), dtype=float)
    ev_mag = [np.zeros(len(Xe), dtype=float) for Xe in X_eval_list]
    for fold, (ti, vi) in enumerate(splits):
        mc = lgb.LGBMClassifier(
            n_estimators=450, learning_rate=0.04, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS,
            random_state=seed_base + 300 + fold, verbose=-1
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
        pd.DataFrame(oof_dict, index=df_tr_fold.index),
        [pd.DataFrame(d, index=Xe.index) for d, Xe in zip(eval_dicts, X_eval_list)],
        s0_metrics,
    )


def build_fold_feature_bundle(df_tr_fold, df_eval_list, inner_splits=5, seed_base=SEED):
    """
    Assemble full v2 feature matrices strictly from df_tr_fold:
      crystal physics + Butler-Ginley Ec + inner-OOF marginals + Stage-0 DFT.
    Strips df_eval_list to input descriptor columns ONLY (structurally zero target leakage).
    """
    df_eval_clean = [
        df_e[["A", "B", "X", "a_eq_b"]].reset_index(drop=True)
        for df_e in df_eval_list
    ]

    bc_model, bc_cols = fit_band_center_model(df_tr_fold)
    ec_tr = predict_band_center(bc_model, bc_cols, df_tr_fold)
    ec_evals = [predict_band_center(bc_model, bc_cols, df_e) for df_e in df_eval_clean]

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

    phys_tr = attach_ec(build_v2_physics_features(df_tr_fold), ec_tr, df_tr_fold)
    phys_evals = [
        attach_ec(build_v2_physics_features(df_e), ec_e, df_e)
        for df_e, ec_e in zip(df_eval_clean, ec_evals)
    ]

    te_tr, te_evals = add_oof_marginal_stats(df_tr_fold, df_eval_clean, n_splits=inner_splits)
    base_tr = pd.concat([phys_tr, te_tr], axis=1)
    base_evals = [pd.concat([pe, te], axis=1) for pe, te in zip(phys_evals, te_evals)]

    s0_tr, s0_evals, s0_metrics = add_stage0_dft_predictors(
        base_tr, df_tr_fold, base_evals, n_splits=inner_splits, seed_base=seed_base
    )
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


def _prepare_family_views(X_tr, X_evals):
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

    return cat_cols, num_cols, X_tr_num, X_ev_num, X_tr_lgb, X_ev_lgb


def _fit_predict_4family_regressor(X_tr, y_tr, X_evals, seed_val):
    """Fit 4 regressor families strictly on (X_tr, y_tr) and predict (len(Xe), 4) on X_evals."""
    cat_cols, _, X_tr_num, X_ev_num, X_tr_lgb, X_ev_lgb = _prepare_family_views(X_tr, X_evals)
    preds = [np.zeros((len(Xe), 4), dtype=float) for Xe in X_evals]

    cb = CatBoostRegressor(
        iterations=600, learning_rate=0.055, depth=6, l2_leaf_reg=4.0,
        one_hot_max_size=60, max_ctr_complexity=1, thread_count=N_JOBS,
        random_seed=seed_val, verbose=0
    )
    cb.fit(X_tr, y_tr, cat_features=cat_cols)
    for j, Xe in enumerate(X_evals):
        preds[j][:, 0] = cb.predict(Xe)

    m_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=31, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.5,
        n_jobs=N_JOBS, random_state=seed_val, verbose=-1
    )
    m_lgb.fit(X_tr_lgb, y_tr)
    for j, Xe in enumerate(X_ev_lgb):
        preds[j][:, 1] = m_lgb.predict(Xe)

    m_xgb = xgb.XGBRegressor(
        n_estimators=550, learning_rate=0.04, max_depth=6, subsample=0.8,
        colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=2.0,
        random_state=seed_val, n_jobs=N_JOBS, tree_method="hist", verbosity=0
    )
    m_xgb.fit(X_tr_num, y_tr)
    for j, Xe in enumerate(X_ev_num):
        preds[j][:, 2] = m_xgb.predict(Xe)

    m_et = ExtraTreesRegressor(
        n_estimators=160, max_features=0.60, min_samples_leaf=2,
        random_state=seed_val, n_jobs=N_JOBS
    )
    m_et.fit(X_tr_num, y_tr)
    for j, Xe in enumerate(X_ev_num):
        preds[j][:, 3] = m_et.predict(Xe)

    return preds


def _fit_predict_4family_classifier(X_tr, y_tr, X_evals, seed_val):
    """Fit 4 classifier families strictly on (X_tr, y_tr) and predict probs (len(Xe), 4) on X_evals."""
    cat_cols, _, X_tr_num, X_ev_num, X_tr_lgb, X_ev_lgb = _prepare_family_views(X_tr, X_evals)
    preds = [np.zeros((len(Xe), 4), dtype=float) for Xe in X_evals]

    cb = CatBoostClassifier(
        iterations=550, learning_rate=0.055, depth=6, l2_leaf_reg=4.0,
        one_hot_max_size=60, max_ctr_complexity=1, thread_count=N_JOBS,
        random_seed=seed_val, verbose=0
    )
    cb.fit(X_tr, y_tr, cat_features=cat_cols)
    for j, Xe in enumerate(X_evals):
        preds[j][:, 0] = cb.predict_proba(Xe)[:, 1]

    m_lgb = lgb.LGBMClassifier(
        n_estimators=550, learning_rate=0.04, num_leaves=31, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.5,
        n_jobs=N_JOBS, random_state=seed_val, verbose=-1
    )
    m_lgb.fit(X_tr_lgb, y_tr)
    for j, Xe in enumerate(X_ev_lgb):
        preds[j][:, 1] = m_lgb.predict_proba(Xe)[:, 1]

    m_xgb = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=6, subsample=0.8,
        colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=2.0,
        random_state=seed_val, n_jobs=N_JOBS, tree_method="hist", eval_metric="logloss", verbosity=0
    )
    m_xgb.fit(X_tr_num, y_tr)
    for j, Xe in enumerate(X_ev_num):
        preds[j][:, 2] = m_xgb.predict_proba(Xe)[:, 1]

    m_et = ExtraTreesClassifier(
        n_estimators=160, max_features=0.60, min_samples_leaf=2,
        random_state=seed_val, n_jobs=N_JOBS
    )
    m_et.fit(X_tr_num, y_tr)
    for j, Xe in enumerate(X_ev_num):
        preds[j][:, 3] = m_et.predict_proba(Xe)[:, 1]

    return preds


def fit_fold_isolated_pipeline(df_tr_fold, df_eval_list, fit_direct_ablation=True, inner_splits=5, seed_base=SEED):
    """
    Strictly fold-isolated pipeline:
      1. Fits Butler-Ginley band-center Ridge on df_tr_fold ONLY.
      2. Fits marginal target-encoding priors & inner-CV stats on df_tr_fold ONLY.
      3. Fits Stage-0 DFT predictors via inner-CV on df_tr_fold ONLY.
      4. Fits 4-family e_form, p_semi, conditional gap|semi (using df_tr_fold[gap>0]),
         and optional direct-V models strictly on df_tr_fold ONLY.
    Returns features and 4-family predictions on all DataFrames in df_eval_list.
    """
    X_tr, X_evals, ec_tr, ec_evals, s0_metrics = build_fold_feature_bundle(
        df_tr_fold, df_eval_list, inner_splits=inner_splits, seed_base=seed_base
    )

    preds_ef = _fit_predict_4family_regressor(
        X_tr, df_tr_fold["e_form"].values, X_evals, seed_val=seed_base + 13
    )

    y_semi_tr = (df_tr_fold["gap"].values > 0).astype(int)
    preds_psemi = _fit_predict_4family_classifier(
        X_tr, y_semi_tr, X_evals, seed_val=seed_base + 513
    )

    semi_mask = y_semi_tr == 1
    X_tr_semi = X_tr.iloc[np.where(semi_mask)[0]].reset_index(drop=True)
    y_gap_semi = df_tr_fold.loc[semi_mask, "gap"].values
    preds_gap = _fit_predict_4family_regressor(
        X_tr_semi, y_gap_semi, X_evals, seed_val=seed_base + 1013
    )

    if fit_direct_ablation:
        preds_dir = _fit_predict_4family_regressor(
            X_tr, df_tr_fold["V"].values, X_evals, seed_val=seed_base + 1513
        )
    else:
        preds_dir = [np.zeros((len(Xe), 4), dtype=float) for Xe in X_evals]

    return {
        "X_tr": X_tr,
        "X_evals": X_evals,
        "ec_tr": ec_tr,
        "ec_evals": ec_evals,
        "s0_metrics": s0_metrics,
        "preds_ef": preds_ef,
        "preds_psemi": preds_psemi,
        "preds_gap": preds_gap,
        "preds_dir": preds_dir,
    }


def integrate_bayes_mixture(mu_ef, std_ef, sig_ef, p_semi, mu_g, std_g, sig_g, ec):
    """
    Bayes-optimal mixture integration over metal (gap=0) and semiconductor (gap>0)
    regimes under assumed Gaussian marginal error distributions with empirical
    0.75/0.25 residual/disagreement scale blending.
    """
    s_ef = np.sqrt(0.75 * (sig_ef ** 2) + 0.25 * (np.maximum(std_ef, 0.05) ** 2))
    s_g = np.sqrt(0.75 * (sig_g ** 2) + 0.25 * (np.maximum(std_g, 0.10) ** 2))

    v_form = exact_expected_hinge_sq(mu_ef, s_ef, threshold=0.21, sign=+1.0, scale=0.5)
    vm_gl, vm_gh, vm_vbm, vm_cbm = v_band_terms(np.zeros_like(ec), ec)

    vs_gl = exact_expected_hinge_sq(mu_g, s_g, threshold=1.4, sign=-1.0, scale=0.5)
    vs_gh = exact_expected_hinge_sq(mu_g, s_g, threshold=3.1, sign=+1.0, scale=0.5)
    vs_vbm = exact_expected_hinge_sq(ec + 0.5 * mu_g, 0.5 * s_g, threshold=5.73, sign=-1.0, scale=0.5)
    vs_cbm = exact_expected_hinge_sq(ec - 0.5 * mu_g, 0.5 * s_g, threshold=4.50, sign=+1.0, scale=0.5)

    v_gl = (1.0 - p_semi) * vm_gl + p_semi * vs_gl
    v_gh = (1.0 - p_semi) * vm_gh + p_semi * vs_gh
    v_vbm = (1.0 - p_semi) * vm_vbm + p_semi * vs_vbm
    v_cbm = (1.0 - p_semi) * vm_cbm + p_semi * vs_cbm
    v_tot = v_form + v_gl + v_gh + v_vbm + v_cbm

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


def test_outer_fold_isolation(df_train, n_splits=5):
    """
    Automated +100 Label-Perturbation Isolation Self-Test (F05):
    Perturb fold-0 validation targets (V, e_form, gap) by +100.0, re-run fold-0
    feature engineering & 4-family model fitting, and assert max_feature_delta < 1e-12
    AND max_pred_delta < 1e-12 on unperturbed validation inputs.
    """
    t0 = time.perf_counter()
    groups = (df_train["A"].astype(str) + "_" + df_train["B"].astype(str)).values
    gkf = GroupKFold(n_splits=n_splits)
    ti, vi = next(gkf.split(df_train, groups=groups))

    df_tr_0 = df_train.iloc[ti].reset_index(drop=True)
    df_va_0 = df_train.iloc[vi].reset_index(drop=True)

    res_clean = fit_fold_isolated_pipeline(
        df_tr_0, [df_va_0], fit_direct_ablation=True, inner_splits=5, seed_base=SEED
    )

    # Perturb fold-0 validation targets (V, e_form, gap) by +100.0
    df_pert = df_train.copy()
    for col in ("V", "e_form", "gap"):
        df_pert.iloc[vi, df_pert.columns.get_loc(col)] = df_pert.iloc[vi][col].values + 100.0

    df_tr_p = df_pert.iloc[ti].reset_index(drop=True)
    df_va_p = df_pert.iloc[vi].reset_index(drop=True)

    # Evaluate on both perturbed df_va_p (proving label columns are ignored) and unperturbed df_va_0
    res_pert = fit_fold_isolated_pipeline(
        df_tr_p, [df_va_p, df_va_0], fit_direct_ablation=True, inner_splits=5, seed_base=SEED
    )

    num_cols = [c for c in res_clean["X_tr"].columns if c not in ("A", "B", "X")]
    tr_feat_delta = float(np.max(np.abs(
        res_clean["X_tr"][num_cols].values - res_pert["X_tr"][num_cols].values
    )))
    va_feat_delta_unpert = float(np.max(np.abs(
        res_clean["X_evals"][0][num_cols].values - res_pert["X_evals"][1][num_cols].values
    )))
    va_feat_delta_pert = float(np.max(np.abs(
        res_clean["X_evals"][0][num_cols].values - res_pert["X_evals"][0][num_cols].values
    )))
    max_feat_delta = max(tr_feat_delta, va_feat_delta_unpert, va_feat_delta_pert)

    pred_deltas = []
    for key in ("preds_ef", "preds_psemi", "preds_gap", "preds_dir"):
        pred_deltas.append(float(np.max(np.abs(res_clean[key][0] - res_pert[key][1]))))
        pred_deltas.append(float(np.max(np.abs(res_clean[key][0] - res_pert[key][0]))))
    pred_deltas.append(float(np.max(np.abs(res_clean["ec_evals"][0] - res_pert["ec_evals"][1]))))
    max_pred_delta = max(pred_deltas)

    assert max_feat_delta < 1e-12, f"Outer-fold feature leakage detected: max_feat_delta={max_feat_delta}"
    assert max_pred_delta < 1e-12, f"Outer-fold prediction leakage detected: max_pred_delta={max_pred_delta}"

    dt = time.perf_counter() - t0
    return {
        "passed": True,
        "fold_tested": 0,
        "perturbation": "+100.0 on fold-0 validation V, e_form, gap",
        "n_tr_fold": int(len(ti)),
        "n_va_fold": int(len(vi)),
        "n_tr_ab_groups": int(len(set(groups[ti]))),
        "n_va_ab_groups": int(len(set(groups[vi]))),
        "group_overlap": int(len(set(groups[ti]) & set(groups[vi]))),
        "max_feature_delta": max_feat_delta,
        "max_pred_delta": max_pred_delta,
        "threshold": 1e-12,
        "runtime_sec": round(dt, 2),
    }


def fit_and_predict_v2(df_train, df_eval_list, fit_direct_ablation=True, n_splits=5):
    """
    Strictly fold-isolated outer (A,B)-GroupKFold training on df_train
    and out-of-sample ensemble inference on df_eval_list.
    """
    groups = (df_train["A"].astype(str) + "_" + df_train["B"].astype(str)).values
    gkf = GroupKFold(n_splits=n_splits)
    n = len(df_train)

    oof_ef = np.zeros((n, 4), dtype=float)
    oof_psemi = np.zeros((n, 4), dtype=float)
    oof_gap = np.zeros((n, 4), dtype=float)
    oof_dir = np.zeros((n, 4), dtype=float)
    oof_ec = np.zeros(n, dtype=float)

    ev_ef_4 = [np.zeros((len(df_e), 4), dtype=float) for df_e in df_eval_list]
    ev_psemi_4 = [np.zeros((len(df_e), 4), dtype=float) for df_e in df_eval_list]
    ev_gap_4 = [np.zeros((len(df_e), 4), dtype=float) for df_e in df_eval_list]
    ev_dir_4 = [np.zeros((len(df_e), 4), dtype=float) for df_e in df_eval_list]
    ev_ec = [np.zeros(len(df_e), dtype=float) for df_e in df_eval_list]
    fold_s0_metrics = []

    for fold, (ti, vi) in enumerate(gkf.split(df_train, groups=groups)):
        assert len(set(groups[ti]) & set(groups[vi])) == 0, "Outer fold group overlap!"
        print(f"  -> [Outer Fold {fold+1}/{n_splits}] tr={len(ti)} rows, va={len(vi)} rows...", flush=True)
        df_tr_fold = df_train.iloc[ti].reset_index(drop=True)
        df_va_fold = df_train.iloc[vi].reset_index(drop=True)

        fold_res = fit_fold_isolated_pipeline(
            df_tr_fold,
            [df_va_fold] + df_eval_list,
            fit_direct_ablation=fit_direct_ablation,
            inner_splits=5,
            seed_base=SEED + fold * 200,
        )
        fold_s0_metrics.append(fold_res["s0_metrics"])

        oof_ef[vi] = fold_res["preds_ef"][0]
        oof_psemi[vi] = fold_res["preds_psemi"][0]
        oof_gap[vi] = fold_res["preds_gap"][0]
        oof_dir[vi] = fold_res["preds_dir"][0]
        oof_ec[vi] = fold_res["ec_evals"][0]

        for j in range(len(df_eval_list)):
            ev_ef_4[j] += fold_res["preds_ef"][1 + j] / n_splits
            ev_psemi_4[j] += fold_res["preds_psemi"][1 + j] / n_splits
            ev_gap_4[j] += fold_res["preds_gap"][1 + j] / n_splits
            ev_dir_4[j] += fold_res["preds_dir"][1 + j] / n_splits
            ev_ec[j] += fold_res["ec_evals"][1 + j] / n_splits

    # Fit simplex ensemble weights, residual scales, and isotonic calibrator strictly on outer OOF
    y_ef = df_train["e_form"].values
    w_ef = _fit_optimal_weights(oof_ef, y_ef, loss="mse")
    oof_ef_ens = oof_ef @ w_ef
    oof_ef_std = np.std(oof_ef, axis=1)
    sig_ef = float(np.sqrt(mean_squared_error(y_ef, oof_ef_ens)))

    y_semi = (df_train["gap"].values > 0).astype(int)
    w_cls = _fit_optimal_weights(oof_psemi, y_semi, loss="logloss")
    oof_psemi_raw = np.clip(oof_psemi @ w_cls, 1e-6, 1.0 - 1e-6)
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(oof_psemi_raw, y_semi)
    oof_psemi_cal = 0.5 * oof_psemi_raw + 0.5 * iso.predict(oof_psemi_raw)

    semi_mask = y_semi == 1
    y_gap_semi = df_train.loc[semi_mask, "gap"].values
    w_gap = _fit_optimal_weights(oof_gap[semi_mask], y_gap_semi, loss="mse")
    oof_gap_ens = np.maximum(0.05, oof_gap @ w_gap)
    oof_gap_std = np.std(oof_gap, axis=1)
    sig_g = float(np.sqrt(mean_squared_error(y_gap_semi, oof_gap_ens[semi_mask])))

    oof_mix = integrate_bayes_mixture(
        oof_ef_ens, oof_ef_std, sig_ef, oof_psemi_cal, oof_gap_ens, oof_gap_std, sig_g, oof_ec
    )
    outer_oof_metrics = _eval_extended_metrics(df_train["V"].values, oof_mix["V_hat"], "v2_PhysHurdle_Bayes_OuterOOF")

    if fit_direct_ablation:
        w_dir = _fit_optimal_weights(oof_dir, df_train["V"].values, loss="mse")
        oof_dir_ens = np.maximum(0.0, oof_dir @ w_dir)
        outer_oof_dir_metrics = _eval_extended_metrics(df_train["V"].values, oof_dir_ens, "v2_Direct_4FamPhys_OuterOOF")
    else:
        w_dir = np.ones(4) / 4.0
        outer_oof_dir_metrics = None

    # Average Stage-0 inner-OOF R2/AUC across the 5 outer folds
    s0_keys = list(fold_s0_metrics[0].keys())
    s0_mean = {k: round(float(np.mean([fm[k] for fm in fold_s0_metrics])), 4) for k in s0_keys}

    eval_outputs = []
    for j in range(len(df_eval_list)):
        mu_ef = ev_ef_4[j] @ w_ef
        std_ef = np.std(ev_ef_4[j], axis=1)
        p_raw = np.clip(ev_psemi_4[j] @ w_cls, 1e-6, 1.0 - 1e-6)
        p_cal = 0.5 * p_raw + 0.5 * iso.predict(p_raw)
        mu_g = np.maximum(0.05, ev_gap_4[j] @ w_gap)
        std_g = np.std(ev_gap_4[j], axis=1)
        res = integrate_bayes_mixture(
            mu_ef, std_ef, sig_ef, p_cal, mu_g, std_g, sig_g, ev_ec[j]
        )
        res["V_direct"] = np.maximum(0.0, ev_dir_4[j] @ w_dir) if fit_direct_ablation else np.zeros(len(df_eval_list[j]))
        eval_outputs.append(res)

    meta = {
        "outer_cv_protocol": f"{n_splits}-fold (A,B)-GroupKFold strictly isolated per fold",
        "stage0_oof_metrics": s0_mean,
        "stage0_per_fold_metrics": fold_s0_metrics,
        "weights_e_form": dict(zip(["CatBoost", "LightGBM", "XGBoost", "ExtraTrees"], [round(float(x), 4) for x in w_ef])),
        "weights_p_semi": dict(zip(["CatBoost", "LightGBM", "XGBoost", "ExtraTrees"], [round(float(x), 4) for x in w_cls])),
        "weights_gap_semi": dict(zip(["CatBoost", "LightGBM", "XGBoost", "ExtraTrees"], [round(float(x), 4) for x in w_gap])),
        "oof_sigma_e_form": round(sig_ef, 4),
        "oof_sigma_gap_semi": round(sig_g, 4),
        "outer_oof_metrics_bayes": outer_oof_metrics,
        "outer_oof_metrics_direct": outer_oof_dir_metrics,
    }
    return eval_outputs, meta


def evaluate_baselines_and_align(tr, va, m_v2_dir, m_v2_bayes):
    """
    Fair Baseline Alignment (F12):
      - Recompute v1 simple baselines (v1_Direct_LGBM_V, v1_Direct_LGBM_log1pV, v1_OneHot_FM_k8)
        on the exact 02 -> 03 protocol (including MAE_V_lt_1 and MAE_V_lt_0_5).
      - Load M4_PIMT_Physics_Only and M5_PIMT_Hybrid_Ensemble from the original repository's
        table1_surrogate_ablation.csv (resolved dynamically relative to REPO_ROOT).
    """
    X_tr = build_basic_v1_features(tr)
    X_va = build_basic_v1_features(va)
    for col in ["A", "B", "X"]:
        cats = sorted(set(tr[col]).union(set(va[col])))
        X_tr[col] = pd.Categorical(tr[col], categories=cats)
        X_va[col] = pd.Categorical(va[col], categories=cats)

    y_tr = tr["V"].values
    y_va = va["V"].values

    m_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=SEED, verbose=-1
    )
    m_lgb.fit(X_tr, y_tr)
    pred_lgb = np.maximum(0.0, m_lgb.predict(X_va))

    m_log = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS, random_state=SEED, verbose=-1
    )
    m_log.fit(X_tr, np.log1p(y_tr))
    pred_log = np.maximum(0.0, np.expm1(m_log.predict(X_va)))

    fm = SimpleOneHotFM(k=8, lr=0.03, reg=1e-3, epochs=40, batch_size=256, seed=SEED)
    fm.fit(tr, y_tr)
    pred_fm = np.maximum(0.0, fm.predict(va))

    v1_rows = [
        _eval_extended_metrics(y_va, pred_lgb, "v1_Direct_LGBM_V"),
        _eval_extended_metrics(y_va, pred_log, "v1_Direct_LGBM_log1pV"),
        _eval_extended_metrics(y_va, pred_fm, "v1_OneHot_FM_k8"),
    ]
    for r in v1_rows:
        r["source"] = "Recomputed on 02->03 via v1_baseline.py protocol (n_jobs=2)"

    # Load original repository table1_surrogate_ablation.csv dynamically
    orig_outputs_dir = os.path.join(
        REPO_ROOT, "01_2026赛题与数据集", "Problem_D_量子计算钙钛矿稀有目标搜索", "project", "outputs"
    )
    table1_path = os.path.join(orig_outputs_dir, "table1_surrogate_ablation.csv")
    orig_val_pred_path = os.path.join(orig_outputs_dir, "03_val_predictions.csv")

    if not os.path.isfile(table1_path):
        raise FileNotFoundError(f"Missing original baseline ablation CSV: {table1_path}")

    t1 = pd.read_csv(table1_path)
    repo_rows = []
    has_per_row_val = os.path.isfile(orig_val_pred_path)
    for model_name in ("M4_PIMT_Physics_Only", "M5_PIMT_Hybrid_Ensemble"):
        sub = t1[t1["model"] == model_name]
        if len(sub) == 0:
            raise ValueError(f"Model {model_name} not found in {table1_path}")
        row = sub.iloc[0]
        rec = {
            "model": model_name,
            "RMSE": round(float(row["RMSE"]), 4),
            "MAE": round(float(row["MAE"]), 4),
            "R2": round(float(row["R2"]), 4),
            "Spearman": round(float(row["Spearman_rho"]), 4),
            "Rec@20(V<1.0)": round(float(row["Recall_Vlt1_Top20"]), 4),
            "Rec@50(V<1.0)": round(float(row["Recall_Vlt1_Top50"]), 4),
            "Rec@100(V<1.0)": round(float(row["Recall_Vlt1_Top100"]), 4),
            "EF@50(V<1.0)": round(float(row["Enrichment_Vlt1_Top50"]), 2),
            "MAE_V_lt_1": None,
            "MAE_V_lt_0_5": None,
            "source": (
                "01_2026赛题与数据集/Problem_D_量子计算钙钛矿稀有目标搜索/project/outputs/table1_surrogate_ablation.csv "
                "(02->03 protocol; per-row 03_val_predictions.csv not archived in project/outputs/)"
            ),
        }
        repo_rows.append(rec)

    v2_rows = []
    for r in (m_v2_dir, m_v2_bayes):
        rc = dict(r)
        rc["source"] = "Fold-isolated v2 surrogate on 02->03 protocol"
        v2_rows.append(rc)

    all_rows = v1_rows + repo_rows + v2_rows
    by_name = {r["model"]: r for r in all_rows}

    # Relative RMSE & MAE improvements of v2_PhysHurdle_Bayes against key references
    rel_cmp = {}
    v2b = by_name["v2_PhysHurdle_Bayes"]
    for ref_name in ("M4_PIMT_Physics_Only", "M5_PIMT_Hybrid_Ensemble", "v1_Direct_LGBM_log1pV", "v1_OneHot_FM_k8", "v2_Direct_4FamPhys"):
        ref = by_name[ref_name]
        rel_cmp[ref_name] = {
            "ref_RMSE": ref["RMSE"],
            "ref_MAE": ref["MAE"],
            "rmse_pct_change": round(100.0 * (v2b["RMSE"] - ref["RMSE"]) / ref["RMSE"], 2),
            "mae_pct_change": round(100.0 * (v2b["MAE"] - ref["MAE"]) / ref["MAE"], 2),
        }

    return {
        "table1_path_rel": os.path.relpath(table1_path, REPO_ROOT).replace("\\", "/"),
        "orig_03_val_predictions_present": bool(has_per_row_val),
        "models": all_rows,
        "v2_phys_hurdle_relative_delta": rel_cmp,
    }


def _compute_holdout_lineage_diff(df_02only, df_02plus03):
    """Compare frozen 02-only vs 02+03 refit predictions on 04_test_holdout (F25)."""
    a = df_02only.sort_values("V_hat", ascending=True).reset_index(drop=True)
    b = df_02plus03.sort_values("V_hat", ascending=True).reset_index(drop=True)
    merged = a[["id", "formula", "V_hat", "P_hit"]].merge(
        b[["id", "V_hat", "P_hit"]], on="id", suffixes=("_02only", "_02plus03")
    )
    rho_v = float(spearmanr(merged["V_hat_02only"], merged["V_hat_02plus03"]).statistic)
    r_v = float(pearsonr(merged["V_hat_02only"], merged["V_hat_02plus03"]).statistic)
    rho_p = float(spearmanr(merged["P_hit_02only"], merged["P_hit_02plus03"]).statistic)
    mae_diff = float(np.mean(np.abs(merged["V_hat_02only"] - merged["V_hat_02plus03"])))
    max_diff = float(np.max(np.abs(merged["V_hat_02only"] - merged["V_hat_02plus03"])))

    overlaps = {}
    for k in (10, 20, 50):
        s_a = set(a.head(k)["formula"])
        s_b = set(b.head(k)["formula"])
        inter = sorted(s_a & s_b)
        overlaps[f"top{k}"] = {
            "overlap_count": int(len(inter)),
            "jaccard": round(float(len(inter) / max(1, len(s_a | s_b))), 4),
            "only_in_02only": sorted(s_a - s_b),
            "only_in_02plus03": sorted(s_b - s_a),
        }

    return {
        "n_rows": int(len(merged)),
        "spearman_V_hat": round(rho_v, 6),
        "pearson_V_hat": round(r_v, 6),
        "spearman_P_hit": round(rho_p, 6),
        "mae_V_hat_diff": round(mae_diff, 4),
        "max_V_hat_diff": round(max_diff, 4),
        "sum_P_hit_02only": round(float(a["P_hit"].sum()), 4),
        "sum_P_hit_02plus03": round(float(b["P_hit"].sum()), 4),
        "top_k_overlap": overlaps,
    }


def _compute_a_eq_b_sensitivity(df_pool, name):
    """Quantify retain vs exclude A=B (a_eq_b == 1) sensitivity on a scored pool (F26)."""
    df_ret = df_pool.sort_values("V_hat", ascending=True).reset_index(drop=True)
    df_exc = df_ret[df_ret["a_eq_b"] == 0].reset_index(drop=True)
    df_eq = df_ret[df_ret["a_eq_b"] == 1].reset_index(drop=True)

    ret_top20 = set(df_ret.head(20)["formula"])
    exc_top20 = set(df_exc.head(20)["formula"])
    ret_top50 = set(df_ret.head(50)["formula"])
    exc_top50 = set(df_exc.head(50)["formula"])

    eq_in_top20 = df_ret.head(20)[df_ret.head(20)["a_eq_b"] == 1][["formula", "V_hat"]]
    eq_in_top50 = df_ret.head(50)[df_ret.head(50)["a_eq_b"] == 1][["formula", "V_hat"]]

    best_eq = None
    if len(df_eq) > 0:
        r0 = df_eq.iloc[0]
        rank_ret = int(df_ret.index[df_ret["id"] == r0["id"]][0]) + 1
        best_eq = {
            "formula": str(r0["formula"]),
            "V_hat": round(float(r0["V_hat"]), 6),
            "rank_when_retained": rank_ret,
        }

    return {
        "pool": name,
        "total_rows": int(len(df_ret)),
        "n_a_eq_b": int(len(df_eq)),
        "n_a_neq_b": int(len(df_exc)),
        "min_V_hat_retain_a_eq_b": round(float(df_ret["V_hat"].min()), 6),
        "min_V_hat_exclude_a_eq_b": round(float(df_exc["V_hat"].min()), 6),
        "best_a_eq_b_candidate": best_eq,
        "n_a_eq_b_in_top20": int(len(eq_in_top20)),
        "n_a_eq_b_in_top50": int(len(eq_in_top50)),
        "top20_overlap_retain_vs_exclude": int(len(ret_top20 & exc_top20)),
        "top50_overlap_retain_vs_exclude": int(len(ret_top50 & exc_top50)),
        "a_eq_b_formulas_in_top50": [
            {"formula": str(r["formula"]), "V_hat": round(float(r["V_hat"]), 4)}
            for _, r in eq_in_top50.iterrows()
        ],
    }


def main():
    t_start = time.perf_counter()
    tr = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "03_castelli_val.csv"))
    holdout = pd.read_csv(os.path.join(DATA_DIR, "04_castelli_test_holdout.csv"))
    q2_full = pd.read_csv(os.path.join(DATA_DIR, "07_castelli_q2_pool.csv"))
    q2_blind = pd.read_csv(os.path.join(DATA_DIR, "11_castelli_q2_pool_blind.csv"))
    q3_full = pd.read_csv(os.path.join(DATA_DIR, "08_castelli_q3_pool.csv"))
    q3_blind = pd.read_csv(os.path.join(DATA_DIR, "12_castelli_q3_pool_blind.csv"))

    print("=== [Self-Test] Running +100 Outer-Fold Label-Perturbation Isolation Test ===")
    isolation_report = test_outer_fold_isolation(tr, n_splits=5)
    print(
        f"  [PASS] Fold-0 isolation verified in {isolation_report['runtime_sec']}s: "
        f"max_feat_delta={isolation_report['max_feature_delta']:.3e}, "
        f"max_pred_delta={isolation_report['max_pred_delta']:.3e}",
        flush=True,
    )

    print("\n=== [Phase 1] Strict Fold-Isolated Training on 02_train -> 03_val & Frozen 04_holdout ===")
    t0 = time.perf_counter()
    phase1_outs, meta_val = fit_and_predict_v2(tr, [va, holdout], fit_direct_ablation=True, n_splits=5)
    vo = phase1_outs[0]
    hold02_res = phase1_outs[1]
    dt_val = time.perf_counter() - t0

    y_va = va["V"].values
    ec_true = 0.5 * (va["vbm"].values + va["cbm"].values)
    ec_rmse = float(np.sqrt(mean_squared_error(ec_true, vo["ec_exact"])))
    ec_r2 = float(r2_score(ec_true, vo["ec_exact"]))

    ef_r2 = float(r2_score(va["e_form"].values, vo["mu_ef"]))
    ef_rmse = float(np.sqrt(mean_squared_error(va["e_form"].values, vo["mu_ef"])))
    y_semi_va = (va["gap"].values > 0).astype(int)
    cls_roc = float(roc_auc_score(y_semi_va, vo["p_semi"]))
    cls_pr = float(average_precision_score(y_semi_va, vo["p_semi"]))
    cls_brier = float(brier_score_loss(y_semi_va, vo["p_semi"]))
    sm_va = y_semi_va == 1
    gap_r2 = float(r2_score(va.loc[sm_va, "gap"].values, vo["mu_gap"][sm_va]))
    gap_rmse = float(np.sqrt(mean_squared_error(va.loc[sm_va, "gap"].values, vo["mu_gap"][sm_va])))

    m_v2_dir = _eval_extended_metrics(y_va, vo["V_direct"], "v2_Direct_4FamPhys")
    m_v2_bayes = _eval_extended_metrics(y_va, vo["V_hat"], "v2_PhysHurdle_Bayes")

    print("\n=== [Baselines] Evaluating v1 Simple Baselines & Loading Repo M4/M5 (02 -> 03) ===")
    baseline_cmp = evaluate_baselines_and_align(tr, va, m_v2_dir, m_v2_bayes)

    resid = vo["V_hat"] - y_va
    metal_mask = y_semi_va == 0
    semi_mask = y_semi_va == 1
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

    # Export 03_val predictions (frozen 02-only)
    val_df = va[["id", "formula", "A", "B", "X", "a_eq_b", "e_form", "gap", "vbm", "cbm", "V"]].copy()
    val_df["ec_true"] = ec_true
    for k in ["V_hat", "V_direct", "v_form", "v_gap_low", "v_gap_high", "v_vbm", "v_cbm",
              "P_hit", "mu_ef", "sigma_ef", "p_semi", "mu_gap", "sigma_gap", "ec_exact"]:
        val_df[k] = vo[k]
    val_pred_path = os.path.join(OUT_DIR, "v2_val_predictions.csv")
    val_df.to_csv(val_pred_path, index=False)

    # Export frozen 02-only holdout_04 predictions (F25)
    hold02_df = holdout[["id", "formula", "A", "B", "X", "a_eq_b"]].copy()
    for k in ["V_hat", "V_direct", "v_form", "v_gap_low", "v_gap_high", "v_vbm", "v_cbm",
              "P_hit", "mu_ef", "sigma_ef", "p_semi", "mu_gap", "sigma_gap", "ec_exact"]:
        hold02_df[k] = hold02_res[k]
    hold02_df = hold02_df.sort_values("V_hat", ascending=True).reset_index(drop=True)
    hold02_path = os.path.join(OUT_DIR, "v2_holdout_04_train02only.csv")
    hold02_df.to_csv(hold02_path, index=False)

    print(f"\n[v2 Q1 Validation] Completed in {dt_val:.2f}s:")
    print(pd.DataFrame(baseline_cmp["models"])[["model", "RMSE", "MAE", "R2", "Spearman", "MAE_V_lt_1", "MAE_V_lt_0_5", "Rec@50(V<1.0)", "Rec@100(V<1.0)"]].to_string(index=False))

    # Phase 2: Refit cleanly on 02_train + 03_val for final holdout & Q2/Q3 pools
    print("\n=== [Phase 2] Fold-Isolated Refit on 02_train + 03_val for 04/Q2/Q3 Pools ===")
    t1 = time.perf_counter()
    tv = pd.concat([tr, va], ignore_index=True)
    pool_outs, meta_full = fit_and_predict_v2(
        tv, [holdout, q2_full, q2_blind, q3_full, q3_blind], fit_direct_ablation=False, n_splits=5
    )
    names = ["holdout_04", "q2_full_07", "q2_blind_11", "q3_full_08", "q3_blind_12"]
    dfs = [holdout, q2_full, q2_blind, q3_full, q3_blind]

    exported_dfs = {}
    exported_sha256 = {
        "v2_val_predictions.csv": _sha256_file(val_pred_path),
        "v2_holdout_04_train02only.csv": _sha256_file(hold02_path),
    }
    for nm, dfp, res in zip(names, dfs, pool_outs):
        out_df = dfp[["id", "formula", "A", "B", "X", "a_eq_b"]].copy()
        for k in ["V_hat", "v_form", "v_gap_low", "v_gap_high", "v_vbm", "v_cbm",
                  "P_hit", "mu_ef", "sigma_ef", "p_semi", "mu_gap", "sigma_gap", "ec_exact"]:
            out_df[k] = res[k]
        out_df = out_df.sort_values("V_hat", ascending=True).reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, f"v2_{nm}.csv")
        out_df.to_csv(out_path, index=False)
        exported_dfs[nm] = out_df
        exported_sha256[f"v2_{nm}.csv"] = _sha256_file(out_path)

    dt_phase2 = time.perf_counter() - t1

    # Dual holdout lineage comparison (F25) & A=B sensitivity (F26)
    holdout_diff = _compute_holdout_lineage_diff(hold02_df, exported_dfs["holdout_04"])
    a_eq_b_sens = {
        "holdout_04_02plus03": _compute_a_eq_b_sensitivity(exported_dfs["holdout_04"], "04_holdout (02+03 refit)"),
        "holdout_04_02only": _compute_a_eq_b_sensitivity(hold02_df, "04_holdout (02-only frozen)"),
        "q2_full_07": _compute_a_eq_b_sensitivity(exported_dfs["q2_full_07"], "07_q2_full (02+03 refit)"),
        "q3_full_08": _compute_a_eq_b_sensitivity(exported_dfs["q3_full_08"], "08_q3_full (02+03 refit)"),
        "q2_blind_11_n_a_eq_b": int((exported_dfs["q2_blind_11"]["a_eq_b"] == 1).sum()),
        "q3_blind_12_n_a_eq_b": int((exported_dfs["q3_blind_12"]["a_eq_b"] == 1).sum()),
    }

    diagnostics = {
        "val_runtime_sec": round(dt_val, 3),
        "phase2_runtime_sec": round(dt_phase2, 3),
        "total_runtime_sec": round(time.perf_counter() - t_start, 3),
        "outer_fold_isolation_test": isolation_report,
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
            "near_zero_Vlt05_mae": round(float(np.mean(np.abs(resid[y_va < 0.5]))), 4),
        },
        "calibration_curve": cal_curve,
        "ensemble_meta": meta_val,
        "phase2_ensemble_meta": meta_full,
        "metrics_table": [m_v2_dir, m_v2_bayes],
        "baseline_comparison": baseline_cmp,
        "holdout_04_02only_vs_02plus03": holdout_diff,
        "a_eq_b_sensitivity": a_eq_b_sens,
        "exported_csv_sha256": exported_sha256,
    }

    with open(os.path.join(OUT_DIR, "v2_q1_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\n[Phase 2] Refit & pool scoring finished in {dt_phase2:.2f}s.")
    print("Holdout 02-only vs 02+03 lineage:", json.dumps(holdout_diff, indent=2))
    print("A=B Sensitivity:", json.dumps(a_eq_b_sens, indent=2))
    print("\nTop-10 Q2 Blind (11) Candidates (v2):")
    print(exported_dfs["q2_blind_11"].head(10)[["formula", "V_hat", "P_hit", "mu_ef", "p_semi", "mu_gap", "ec_exact"]].to_string(index=False))
    print("\nTop-15 Q3 Blind (12) Candidates (v2):")
    print(exported_dfs["q3_blind_12"].head(15)[["formula", "X", "V_hat", "P_hit", "mu_ef", "p_semi", "mu_gap", "ec_exact"]].to_string(index=False))


if __name__ == "__main__":
    main()
