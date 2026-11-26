'''Problem 1: Gentle-CPU (4-Thread) Physics-Informed Multi-Task Surrogate for Perovskite V.'''

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from element_features import (
    build_intrinsic_features,
    build_oof_marginal_features,
    build_test_marginal_features,
)

DATA_DIR = os.path.expanduser("~/apmcm2026_d/data")
OUT_DIR = os.path.expanduser("~/apmcm2026_d/outputs")
os.makedirs(OUT_DIR, exist_ok=True)
N_THREADS = 4


def make_features(df: pd.DataFrame, ref_df: pd.DataFrame = None, is_train_oof: bool = False) -> pd.DataFrame:
    intr = build_intrinsic_features(df)
    if is_train_oof:
        marg = build_oof_marginal_features(df, n_splits=5)
    else:
        assert ref_df is not None
        marg = build_test_marginal_features(ref_df, df)
    return pd.concat([intr.reset_index(drop=True), marg.reset_index(drop=True)], axis=1)


def fit_lgbm_reg(X_tr, y_tr, seed=42, n_est=550, lr=0.04, num_leaves=31):
    m = LGBMRegressor(
        n_estimators=n_est,
        learning_rate=lr,
        num_leaves=num_leaves,
        subsample_for_bin=20000,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_samples=15,
        random_state=seed,
        n_jobs=N_THREADS,
        verbose=-1,
    )
    m.fit(X_tr, y_tr)
    return m


def fit_cat_reg(X_tr, y_tr, seed=42, iters=550, depth=6, lr=0.05):
    m = CatBoostRegressor(
        iterations=iters,
        depth=depth,
        learning_rate=lr,
        l2_leaf_reg=5.0,
        random_seed=seed,
        thread_count=N_THREADS,
        verbose=0,
    )
    m.fit(X_tr, y_tr)
    return m


def fit_xgb_reg(X_tr, y_tr, seed=42, n_est=500, max_depth=6, lr=0.04):
    m = XGBRegressor(
        n_estimators=n_est,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=N_THREADS,
    )
    m.fit(X_tr, y_tr)
    return m


def compute_V_from_props(e_form, p_semi, gap_semi, vbm, cbm):
    v_form = np.maximum(0.0, (e_form - 0.21) / 0.5) ** 2
    v_gap_semi = (
        np.maximum(0.0, (1.4 - gap_semi) / 0.5) ** 2
        + np.maximum(0.0, (gap_semi - 3.1) / 0.5) ** 2
    )
    v_gap = (1.0 - p_semi) * ((1.4 / 0.5) ** 2) + p_semi * v_gap_semi
    v_vbm = np.maximum(0.0, (5.73 - vbm) / 0.5) ** 2
    v_cbm = np.maximum(0.0, (cbm - 4.50) / 0.5) ** 2
    return v_form + v_gap + v_vbm + v_cbm


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> dict:
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rho = float(spearmanr(y_true, y_pred).statistic)

    order = np.argsort(y_pred)
    rare1 = y_true < 1.0
    rare2 = y_true < 2.0
    n1 = max(1, int(rare1.sum()))
    n2 = max(1, int(rare2.sum()))

    return {
        "model": tag,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Spearman_rho": rho,
        "Recall_Vlt1_Top20": float(rare1[order[:20]].sum() / n1),
        "Recall_Vlt1_Top50": float(rare1[order[:50]].sum() / n1),
        "Recall_Vlt1_Top100": float(rare1[order[:100]].sum() / n1),
        "Recall_Vlt2_Top50": float(rare2[order[:50]].sum() / n2),
        "Recall_Vlt2_Top100": float(rare2[order[:100]].sum() / n2),
        "Enrichment_Vlt1_Top50": float((rare1[order[:50]].mean()) / (rare1.mean() + 1e-12)),
        "True_min_in_Top10": float(y_true[order[:10]].min()),
        "True_median_in_Top20": float(np.median(y_true[order[:20]])),
    }


class PIMTPipeline:
    '''Physics-Informed Multi-Task + Direct/Decomposed Ensemble Surrogate.'''

    def __init__(self, blend_weights=None):
        self.blend_weights = blend_weights

    def fit(self, df_tr: pd.DataFrame, X_tr: pd.DataFrame, tune_weights: bool = False):
        t_start = time.time()
        groups = (df_tr["A"] + "_" + df_tr["B"]).values
        gkf = GroupKFold(n_splits=5)

        stage1_cols = ["a_lat", "volume", "mu_b", "fermi_level"]
        y_V = df_tr["V"].values
        y_logV = np.log1p(y_V)
        root_cols = ["v_form", "v_gap_low", "v_gap_high", "v_vbm", "v_cbm"]
        y_roots = np.sqrt(np.maximum(0.0, df_tr[root_cols].values))

        # 1. Stage-1 OOF latent structural/electronic predictions (fast LGBM)
        oof_s1 = np.zeros((len(df_tr), len(stage1_cols)), dtype=float)
        for tr_i, va_i in gkf.split(X_tr, groups=groups):
            for j, col in enumerate(stage1_cols):
                m = fit_lgbm_reg(X_tr.iloc[tr_i], df_tr[col].iloc[tr_i], seed=100 + j, n_est=300, lr=0.05)
                oof_s1[va_i, j] = m.predict(X_tr.iloc[va_i])

        self.s1_models = {}
        for j, col in enumerate(stage1_cols):
            self.s1_models[col] = fit_lgbm_reg(X_tr, df_tr[col].values, seed=100 + j, n_est=350, lr=0.05)

        X_tr_aug = X_tr.copy()
        for j, col in enumerate(stage1_cols):
            X_tr_aug[f"pred_{col}"] = oof_s1[:, j]
        print(f"  [+{time.time()-t_start:.1f}s] Stage-1 latent OOF ready.", flush=True)

        # 2. Tune convex blend weights on a single GroupShuffleSplit (80/20) inside 02_train
        if tune_weights and self.blend_weights is None:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
            tr_i, va_i = next(gss.split(X_tr_aug, groups=groups))
            X_it, X_iv = X_tr_aug.iloc[tr_i], X_tr_aug.iloc[va_i]
            y_iv = y_V[va_i]

            p_raw = np.maximum(0.0, fit_lgbm_reg(X_it, y_V[tr_i], seed=201, n_est=450).predict(X_iv))
            p_log = np.maximum(0.0, np.expm1(fit_lgbm_reg(X_it, y_logV[tr_i], seed=301, n_est=450).predict(X_iv)))
            p_comp = np.zeros(len(va_i))
            for k in range(5):
                uk = np.maximum(0.0, fit_lgbm_reg(X_it, y_roots[tr_i, k], seed=400 + k, n_est=350).predict(X_iv))
                p_comp += uk ** 2

            is_semi_it = (df_tr["gap"].iloc[tr_i].values > 0).astype(int)
            clf_it = LGBMClassifier(
                n_estimators=400, learning_rate=0.04, num_leaves=31,
                subsample=0.85, colsample_bytree=0.8, random_state=501, n_jobs=N_THREADS, verbose=-1
            )
            clf_it.fit(X_it, is_semi_it)
            p_semi_iv = clf_it.predict_proba(X_iv)[:, 1]
            semi_mask_it = is_semi_it == 1
            gap_iv = np.clip(fit_lgbm_reg(X_it.iloc[semi_mask_it], df_tr["gap"].iloc[tr_i].values[semi_mask_it], seed=510, n_est=300).predict(X_iv), 0.0, 10.0)
            ef_iv = fit_lgbm_reg(X_it, df_tr["e_form"].iloc[tr_i].values, seed=520, n_est=400).predict(X_iv)
            vbm_iv = fit_lgbm_reg(X_it, df_tr["vbm"].iloc[tr_i].values, seed=530, n_est=400).predict(X_iv)
            cbm_iv = fit_lgbm_reg(X_it, df_tr["cbm"].iloc[tr_i].values, seed=540, n_est=400).predict(X_iv)
            p_phys = compute_V_from_props(ef_iv, p_semi_iv, gap_iv, vbm_iv, cbm_iv)

            mat = np.column_stack([p_phys, p_comp, p_log, p_raw])
            sample_w = 1.0 + 3.0 * np.exp(-y_iv / 3.0)
            def loss_fn(w):
                w = w / np.sum(w)
                return float(np.mean(sample_w * (mat @ w - y_iv) ** 2))
            res = minimize(loss_fn, np.array([0.25, 0.25, 0.25, 0.25]), bounds=[(0.0, 1.0)] * 4)
            w_opt = res.x / np.sum(res.x)
            self.blend_weights = w_opt.tolist()
            print(f"  [+{time.time()-t_start:.1f}s] Tuned blend weights [phys, comp, logV, rawV]: {[round(float(x), 4) for x in w_opt]}", flush=True)

        # 3. Full Stage-2 training (4 threads)
        self.raw_models = [
            fit_lgbm_reg(X_tr_aug, y_V, seed=601, n_est=550),
            fit_cat_reg(X_tr_aug, y_V, seed=602, iters=550),
            fit_xgb_reg(X_tr_aug, y_V, seed=603, n_est=500),
        ]
        self.log_models = [
            fit_lgbm_reg(X_tr_aug, y_logV, seed=701, n_est=550),
            fit_cat_reg(X_tr_aug, y_logV, seed=702, iters=550),
            fit_xgb_reg(X_tr_aug, y_logV, seed=703, n_est=500),
        ]
        print(f"  [+{time.time()-t_start:.1f}s] Direct RawV & LogV ensembles fitted.", flush=True)

        self.root_models = []
        for k in range(5):
            self.root_models.append((
                fit_lgbm_reg(X_tr_aug, y_roots[:, k], seed=800 + k, n_est=500),
                fit_cat_reg(X_tr_aug, y_roots[:, k], seed=810 + k, iters=450),
            ))
        print(f"  [+{time.time()-t_start:.1f}s] 5-term Root-Penalty models fitted.", flush=True)

        is_semi = (df_tr["gap"].values > 0).astype(int)
        self.semi_clfs = [
            LGBMClassifier(
                n_estimators=500, learning_rate=0.04, num_leaves=31,
                subsample=0.85, colsample_bytree=0.8, random_state=901, n_jobs=N_THREADS, verbose=-1
            ),
            CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05, random_seed=902, thread_count=N_THREADS, verbose=0
            ),
        ]
        for clf in self.semi_clfs:
            clf.fit(X_tr_aug, is_semi)

        semi_mask = is_semi == 1
        self.gap_models = [
            fit_lgbm_reg(X_tr_aug.iloc[semi_mask], df_tr["gap"].values[semi_mask], seed=911, n_est=400),
            fit_cat_reg(X_tr_aug.iloc[semi_mask], df_tr["gap"].values[semi_mask], seed=912, iters=450),
        ]
        self.ef_models = [
            fit_lgbm_reg(X_tr_aug, df_tr["e_form"].values, seed=921, n_est=500),
            fit_cat_reg(X_tr_aug, df_tr["e_form"].values, seed=922, iters=450),
        ]
        self.vbm_models = [
            fit_lgbm_reg(X_tr_aug, df_tr["vbm"].values, seed=931, n_est=500),
            fit_cat_reg(X_tr_aug, df_tr["vbm"].values, seed=932, iters=450),
        ]
        self.cbm_models = [
            fit_lgbm_reg(X_tr_aug, df_tr["cbm"].values, seed=941, n_est=500),
            fit_cat_reg(X_tr_aug, df_tr["cbm"].values, seed=942, iters=450),
        ]
        print(f"  [+{time.time()-t_start:.1f}s] Physics-property models fitted.", flush=True)

    def predict_components(self, X_te: pd.DataFrame) -> dict:
        X_aug = X_te.copy()
        for col, m in self.s1_models.items():
            X_aug[f"pred_{col}"] = m.predict(X_te)

        pred_rawV = np.maximum(0.0, np.mean([m.predict(X_aug) for m in self.raw_models], axis=0))
        pred_logV = np.maximum(0.0, np.expm1(np.mean([m.predict(X_aug) for m in self.log_models], axis=0)))

        pred_comp = np.zeros(len(X_aug))
        for m_lgb, m_cat in self.root_models:
            uk = np.maximum(0.0, 0.5 * m_lgb.predict(X_aug) + 0.5 * m_cat.predict(X_aug))
            pred_comp += uk ** 2

        p_semi = np.mean([c.predict_proba(X_aug)[:, 1] for c in self.semi_clfs], axis=0)
        gap_semi = np.clip(np.mean([m.predict(X_aug) for m in self.gap_models], axis=0), 0.0, 10.0)
        e_form = np.mean([m.predict(X_aug) for m in self.ef_models], axis=0)
        vbm = np.mean([m.predict(X_aug) for m in self.vbm_models], axis=0)
        cbm = np.mean([m.predict(X_aug) for m in self.cbm_models], axis=0)
        pred_phys = compute_V_from_props(e_form, p_semi, gap_semi, vbm, cbm)

        w = self.blend_weights if self.blend_weights is not None else [0.25, 0.25, 0.25, 0.25]
        pred_hybrid = w[0] * pred_phys + w[1] * pred_comp + w[2] * pred_logV + w[3] * pred_rawV

        return {
            "V_hybrid": pred_hybrid,
            "V_phys": pred_phys,
            "V_comp": pred_comp,
            "V_logV": pred_logV,
            "V_rawV": pred_rawV,
            "p_semi": p_semi,
            "gap_semi": gap_semi,
            "gap_expected": p_semi * gap_semi,
            "e_form": e_form,
            "vbm": vbm,
            "cbm": cbm,
            "X_aug": X_aug,
        }


def main():
    t0 = time.time()
    tr = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "03_castelli_val.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "04_castelli_test_holdout.csv"))
    q2_pool = pd.read_csv(os.path.join(DATA_DIR, "07_castelli_q2_pool.csv"))
    q2_blind = pd.read_csv(os.path.join(DATA_DIR, "11_castelli_q2_pool_blind.csv"))
    q3_pool = pd.read_csv(os.path.join(DATA_DIR, "08_castelli_q3_pool.csv"))
    q3_blind = pd.read_csv(os.path.join(DATA_DIR, "12_castelli_q3_pool_blind.csv"))

    print("Building leak-free features for 02_train and 03_val...", flush=True)
    X_tr = make_features(tr, is_train_oof=True)
    X_va = make_features(va, ref_df=tr, is_train_oof=False)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_ohe_tr = ohe.fit_transform(tr[["A", "B", "X"]])
    X_ohe_va = ohe.transform(va[["A", "B", "X"]])
    scaler = StandardScaler()
    intr_cols = [c for c in X_tr.columns if not c.startswith("marg_")]
    ridge = Ridge(alpha=15.0)
    ridge.fit(np.hstack([X_ohe_tr, scaler.fit_transform(X_tr[intr_cols])]), np.log1p(tr["V"].values))
    pred_ridge = np.maximum(0.0, np.expm1(ridge.predict(np.hstack([X_ohe_va, scaler.transform(X_va[intr_cols])]))))

    print("Training PIMT Surrogate on 02_train (4 threads)...", flush=True)
    pipe_val = PIMTPipeline()
    pipe_val.fit(tr, X_tr, tune_weights=True)
    va_out = pipe_val.predict_components(X_va)

    y_va = va["V"].values
    ablation = [
        eval_metrics(y_va, pred_ridge, "M1_Ridge_LogV_Baseline"),
        eval_metrics(y_va, va_out["V_rawV"], "M2a_Direct_GBDT_RawV"),
        eval_metrics(y_va, va_out["V_logV"], "M2b_Direct_GBDT_LogV"),
        eval_metrics(y_va, va_out["V_comp"], "M3_RootPenalty_Decomp"),
        eval_metrics(y_va, va_out["V_phys"], "M4_PIMT_Physics_Only"),
        eval_metrics(y_va, va_out["V_hybrid"], "M5_PIMT_Hybrid_Ensemble"),
    ]
    abl_df = pd.DataFrame(ablation)
    abl_df.to_csv(os.path.join(OUT_DIR, "problem1_ablation_table.csv"), index=False)
    print("\n=== PROBLEM 1 VALIDATION ABLATION (03_castelli_val.csv, N=3969) ===", flush=True)
    print(abl_df[["model", "RMSE", "MAE", "R2", "Spearman_rho", "Recall_Vlt1_Top50", "Recall_Vlt1_Top100", "Enrichment_Vlt1_Top50"]].to_string(index=False), flush=True)

    semi_true = (va["gap"].values > 0).astype(int)
    semi_auc = float(roc_auc_score(semi_true, va_out["p_semi"]))
    semi_ap = float(average_precision_score(semi_true, va_out["p_semi"]))
    print(f"Semiconductor classifier on 03_val: ROC-AUC={semi_auc:.4f}, PR-AUC={semi_ap:.4f} (pos={semi_true.sum()}/{len(semi_true)})", flush=True)

    va_save = va[["id", "formula", "A", "B", "X", "V", "e_form", "gap", "vbm", "cbm"]].copy()
    va_save["V_pred_hybrid"] = va_out["V_hybrid"]
    va_save["V_pred_ridge"] = pred_ridge
    va_save["p_semi_pred"] = va_out["p_semi"]
    va_save.to_csv(os.path.join(OUT_DIR, "03_val_predictions.csv"), index=False)

    lgb_imp = pd.DataFrame({
        "feature": va_out["X_aug"].columns,
        "importance": pipe_val.raw_models[0].feature_importances_,
    }).sort_values("importance", ascending=False)
    lgb_imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

    print("\nRefitting winning PIMT_Hybrid pipeline on full 02 + 03 (15,225 rows)...", flush=True)
    full_tr = pd.concat([tr, va], ignore_index=True)
    X_full = make_features(full_tr, is_train_oof=True)
    pipe_full = PIMTPipeline(blend_weights=pipe_val.blend_weights)
    pipe_full.fit(full_tr, X_full, tune_weights=False)

    X_te = make_features(te, ref_df=full_tr, is_train_oof=False)
    te_out = pipe_full.predict_components(X_te)
    te_pred = te.copy()
    te_pred["V_pred"] = te_out["V_hybrid"]
    te_pred["V_phys_pred"] = te_out["V_phys"]
    te_pred["V_comp_pred"] = te_out["V_comp"]
    te_pred["V_logV_pred"] = te_out["V_logV"]
    te_pred["p_semi_pred"] = te_out["p_semi"]
    te_pred["e_form_pred"] = te_out["e_form"]
    te_pred["gap_semi_pred"] = te_out["gap_semi"]
    te_pred["gap_expected_pred"] = te_out["gap_expected"]
    te_pred["vbm_pred"] = te_out["vbm"]
    te_pred["cbm_pred"] = te_out["cbm"]
    te_pred.to_csv(os.path.join(OUT_DIR, "04_castelli_test_holdout_predictions.csv"), index=False)

    te_sorted = te_pred.sort_values("V_pred")
    print("\n=== TOP 20 PREDICTED RARE PEROVSKITES IN 04_castelli_test_holdout.csv ===", flush=True)
    print(te_sorted[["id", "formula", "A", "B", "X", "V_pred", "p_semi_pred", "e_form_pred", "gap_semi_pred"]].head(20).to_string(index=False), flush=True)

    known_map = full_tr.set_index("id")["V"].to_dict()
    for tag, pool_df, blind_df, out_name in [
        ("Q2", q2_pool, q2_blind, "q2_full_pool_predictions.csv"),
        ("Q3", q3_pool, q3_blind, "q3_full_pool_predictions.csv"),
    ]:
        X_p = make_features(pool_df, ref_df=full_tr, is_train_oof=False)
        p_out = pipe_full.predict_components(X_p)
        res_df = pool_df.copy()
        res_df["V_pred"] = p_out["V_hybrid"]
        res_df["V_phys_pred"] = p_out["V_phys"]
        res_df["p_semi_pred"] = p_out["p_semi"]
        res_df["in_blind_pool"] = res_df["id"].isin(set(blind_df["id"])).astype(int)
        res_df["in_labeled"] = res_df["id"].isin(known_map).astype(int)
        res_df["V_true_if_known"] = res_df["id"].map(known_map)
        res_df.to_csv(os.path.join(OUT_DIR, out_name), index=False)
        top_p = res_df.sort_values("V_pred").head(10)
        print(f"\n=== TOP 10 SURROGATE CANDIDATES IN {tag} FULL POOL ({len(res_df)} rows) ===", flush=True)
        print(top_p[["id", "formula", "A", "B", "X", "V_pred", "p_semi_pred", "in_blind_pool"]].to_string(index=False), flush=True)

    metrics_payload = {
        "blend_weights": {
            "V_phys": pipe_val.blend_weights[0],
            "V_comp": pipe_val.blend_weights[1],
            "V_logV": pipe_val.blend_weights[2],
            "V_rawV": pipe_val.blend_weights[3],
        },
        "val_semiconductor_roc_auc": semi_auc,
        "val_semiconductor_pr_auc": semi_ap,
        "holdout_count_Vpred_lt_0_5": int((te_pred["V_pred"] < 0.5).sum()),
        "holdout_count_Vpred_lt_1_0": int((te_pred["V_pred"] < 1.0).sum()),
        "holdout_count_Vpred_lt_2_0": int((te_pred["V_pred"] < 2.0).sum()),
        "holdout_min_Vpred": float(te_pred["V_pred"].min()),
        "runtime_sec": round(time.time() - t0, 2),
        "ablation": ablation,
    }
    with open(os.path.join(OUT_DIR, "problem1_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nPROBLEM 1 COMPLETE in {metrics_payload['runtime_sec']}s!", flush=True)


if __name__ == "__main__":
    main()
