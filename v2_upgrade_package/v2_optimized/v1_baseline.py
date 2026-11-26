#!/usr/bin/env python3
"""
v1 Baseline Benchmark (Faithful Implementation of Teammate's Roadmap in
ref_teammate/01_2026赛题与数据集/.../D题_中文精读与建模路线图.md).

Because ref_teammate/ (commit 80f31c8) contains the modeling design document
without pushed Python source files, this module faithfully implements every
v1 modeling choice proposed in the teammate's roadmap so v1 vs v2 can be
compared apples-to-apples on identical official data splits and Kaiwu solvers:
  - v1 Q1: Direct LightGBM on V, Direct LightGBM on log1p(V), and standalone
           2nd-order one-hot Factorization Machine (FM, rank k=8).
  - v1 Q2: Naive 104-variable one-hot QUBO (52 A + 52 B) with fixed penalty
           P=20 and naive 8-bit adjust_ising_matrix_precision(..., bit_width=8).
  - v1 Q3: Standard 107-variable pairwise FM-QUBO + standard naive Rosenberg
           3-body HOBO->QUBO reduction (52*52*3 = 8,112 cubic terms -> 2,704 aux
           variables on full space, or active subset) + naive 8-bit quantization.
"""

import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import kaiwu as kw

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)
from common_physics import (
    DATA_DIR,
    build_basic_v1_features,
    eval_surrogate_metrics,
    safe_ising_bit_width,
    sort_ising_solutions,
)

OUT_DIR = os.path.join(CUR_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 2026


class SimpleOneHotFM:
    """
    Standalone 2nd-order Factorization Machine on binary one-hot (A, B, X):
        y_hat(z) = w0 + w^T z + 0.5 * sum_f [ (v_f^T z)^2 - sum_i (v_{i,f}^2 z_i^2) ]
    Trained via Adam / mini-batch SGD with L2 regularization.
    """

    def __init__(self, k=8, lr=0.03, reg=1e-3, epochs=35, batch_size=256, seed=SEED):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, df, y):
        self.cols_A = sorted(df["A"].unique())
        self.cols_B = sorted(df["B"].unique())
        self.cols_X = sorted(df["X"].unique())
        self.map_A = {e: i for i, e in enumerate(self.cols_A)}
        self.map_B = {e: len(self.cols_A) + i for i, e in enumerate(self.cols_B)}
        self.map_X = {e: len(self.cols_A) + len(self.cols_B) + i for i, e in enumerate(self.cols_X)}
        self.d = len(self.cols_A) + len(self.cols_B) + len(self.cols_X)

        ia = df["A"].map(self.map_A).values
        ib = df["B"].map(self.map_B).values
        ix = df["X"].map(self.map_X).values
        idx = np.stack([ia, ib, ix], axis=1)
        y = np.asarray(y, dtype=float)

        rng = np.random.default_rng(self.seed)
        self.w0 = float(np.mean(y))
        self.w = np.zeros(self.d, dtype=float)
        self.V = rng.normal(0.0, 0.1, size=(self.d, self.k))

        # Adam state
        m_w0, v_w0 = 0.0, 0.0
        m_w, v_w = np.zeros_like(self.w), np.zeros_like(self.w)
        m_V, v_V = np.zeros_like(self.V), np.zeros_like(self.V)
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        n = len(y)

        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                step += 1
                sl = perm[start:start + self.batch_size]
                bi = idx[sl]  # (B, 3)
                by = y[sl]
                bs = len(sl)

                # Forward: exactly 3 active one-hot indices per row
                v_act = self.V[bi]  # (B, 3, k)
                sum_v = v_act.sum(axis=1)  # (B, k)
                sum_v2 = (v_act ** 2).sum(axis=1)  # (B, k)
                inter = 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1)
                lin = self.w[bi].sum(axis=1)
                pred = self.w0 + lin + inter
                err = (pred - by) / bs  # dL/dpred for 0.5*MSE

                # Gradients
                g_w0 = float(err.sum())
                g_w = self.reg * self.w
                g_V = self.reg * self.V
                for j in range(3):
                    col_j = bi[:, j]
                    np.add.at(g_w, col_j, err)
                    # d(inter)/d(v_{col_j}) = sum_v - v_{col_j}
                    grad_vj = err[:, None] * (sum_v - v_act[:, j, :])
                    np.add.at(g_V, col_j, grad_vj)

                # Adam updates
                m_w0 = b1 * m_w0 + (1 - b1) * g_w0
                v_w0 = b2 * v_w0 + (1 - b2) * (g_w0 ** 2)
                self.w0 -= self.lr * (m_w0 / (1 - b1 ** step)) / (np.sqrt(v_w0 / (1 - b2 ** step)) + eps)

                m_w = b1 * m_w + (1 - b1) * g_w
                v_w = b2 * v_w + (1 - b2) * (g_w ** 2)
                self.w -= self.lr * (m_w / (1 - b1 ** step)) / (np.sqrt(v_w / (1 - b2 ** step)) + eps)

                m_V = b1 * m_V + (1 - b1) * g_V
                v_V = b2 * v_V + (1 - b2) * (g_V ** 2)
                self.V -= self.lr * (m_V / (1 - b1 ** step)) / (np.sqrt(v_V / (1 - b2 ** step)) + eps)
        return self

    def predict(self, df):
        ia = df["A"].map(self.map_A).values
        ib = df["B"].map(self.map_B).values
        ix = df["X"].map(self.map_X).values
        bi = np.stack([ia, ib, ix], axis=1)
        v_act = self.V[bi]
        sum_v = v_act.sum(axis=1)
        sum_v2 = (v_act ** 2).sum(axis=1)
        return self.w0 + self.w[bi].sum(axis=1) + 0.5 * np.sum(sum_v ** 2 - sum_v2, axis=1)

    def get_qubo_matrix_q3(self, x_subset=("O3", "NO2", "FO2"), penalty=20.0):
        """Extract 107-variable pairwise QUBO (52 A + 52 B + 3 X) directly from FM weights."""
        idx_A = [self.map_A[e] for e in self.cols_A]
        idx_B = [self.map_B[e] for e in self.cols_B]
        idx_X = [self.map_X[e] for e in x_subset]
        keep = idx_A + idx_B + idx_X
        nA, nB, nX = len(idx_A), len(idx_B), len(idx_X)
        N = nA + nB + nX
        Q = np.zeros((N, N), dtype=float)
        # Linear terms
        for i, gi in enumerate(keep):
            Q[i, i] += self.w[gi]
        # Pairwise cross terms from FM inner product <v_i, v_j>
        V_sub = self.V[keep]
        G = V_sub @ V_sub.T
        for i in range(N):
            for j in range(i + 1, N):
                Q[i, j] += G[i, j]
        # One-hot penalties on A, B, X
        for start, size in [(0, nA), (nA, nB), (nA + nB, nX)]:
            for i in range(start, start + size):
                Q[i, i] -= penalty
                for j in range(i + 1, start + size):
                    Q[i, j] += 2.0 * penalty
        # A != B penalty
        for i in range(nA):
            Q[i, nA + i] += penalty
        Q_sym = 0.5 * (Q + Q.T)
        return Q_sym, self.cols_A, self.cols_B, list(x_subset)


def run_v1_q1():
    t0 = time.perf_counter()
    tr = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "03_castelli_val.csv"))

    X_tr = build_basic_v1_features(tr)
    X_va = build_basic_v1_features(va)
    # Add categorical A, B, X as category dtype for LightGBM
    for col in ["A", "B", "X"]:
        cats = sorted(set(tr[col]).union(set(va[col])))
        X_tr[col] = pd.Categorical(tr[col], categories=cats)
        X_va[col] = pd.Categorical(va[col], categories=cats)

    y_tr = tr["V"].values
    y_va = va["V"].values

    # 1. Direct LightGBM on V
    m_lgb = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1
    )
    m_lgb.fit(X_tr, y_tr)
    pred_lgb = np.maximum(0.0, m_lgb.predict(X_va))

    # 2. Direct LightGBM on log1p(V)
    m_log = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1
    )
    m_log.fit(X_tr, np.log1p(y_tr))
    pred_log = np.maximum(0.0, np.expm1(m_log.predict(X_va)))

    # 3. Standalone 2nd-order One-Hot Factorization Machine (FM)
    fm = SimpleOneHotFM(k=8, lr=0.03, reg=1e-3, epochs=40, batch_size=256, seed=SEED)
    fm.fit(tr, y_tr)
    pred_fm = np.maximum(0.0, fm.predict(va))

    metrics = [
        eval_surrogate_metrics(y_va, pred_lgb, "v1_Direct_LGBM_V"),
        eval_surrogate_metrics(y_va, pred_log, "v1_Direct_LGBM_log1pV"),
        eval_surrogate_metrics(y_va, pred_fm, "v1_OneHot_FM_k8"),
    ]
    elapsed = time.perf_counter() - t0

    # Refit best v1 GBDT (log1p) and FM on train+val for Q2/Q3 pool scoring
    tv = pd.concat([tr, va], ignore_index=True)
    X_tv = build_basic_v1_features(tv)
    for col in ["A", "B", "X"]:
        cats = sorted(tv[col].unique())
        X_tv[col] = pd.Categorical(tv[col], categories=cats)
    m_log_full = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1
    )
    m_log_full.fit(X_tv, np.log1p(tv["V"].values))

    fm_full = SimpleOneHotFM(k=8, lr=0.03, reg=1e-3, epochs=40, batch_size=256, seed=SEED)
    fm_full.fit(tv, tv["V"].values)

    return metrics, elapsed, m_log_full, fm_full, sorted(tv["A"].unique()), sorted(tv["B"].unique()), sorted(tv["X"].unique())


def _build_naive_q2_qubo(score_mat, elems_A, elems_B, penalty=20.0):
    """Standard 104-variable one-hot QUBO (52 A + 52 B) without zero-shift or pruning."""
    nA, nB = len(elems_A), len(elems_B)
    N = nA + nB
    Q = np.zeros((N, N), dtype=float)
    # Bilinear objective sum_{i,j} M_{ij} a_i b_j
    for i in range(nA):
        for j in range(nB):
            Q[i, nA + j] += score_mat[i, j]
    # One-hot constraints: P*(sum a_i - 1)^2 + P*(sum b_j - 1)^2
    for start, size in [(0, nA), (nA, nB)]:
        for i in range(start, start + size):
            Q[i, i] -= penalty
            for j in range(i + 1, start + size):
                Q[i, j] += 2.0 * penalty
    # A != B penalty
    for i in range(nA):
        if elems_A[i] in elems_B:
            j = elems_B.index(elems_A[i])
            Q[i, nA + j] += penalty
    return 0.5 * (Q + Q.T)


def _solve_qubo_sa(Q_sym, n_runs=40, bit_width=None, seed_base=100):
    """Convert QUBO->Ising, optionally apply naive bit-width scaling, and solve with Kaiwu SA."""
    ising_mat, bias = kw.qubo_matrix_to_ising_matrix(Q_sym)
    bw_raw = safe_ising_bit_width(ising_mat)
    if bit_width is not None:
        ising_use = kw.adjust_ising_matrix_precision(ising_mat, bit_width=bit_width)
    else:
        ising_use = ising_mat
    bw_used = safe_ising_bit_width(ising_use)
    dr = float(kw.get_dynamic_range_metric(ising_use))

    t0 = time.perf_counter()
    sols = []
    for r in range(n_runs):
        opt = kw.SimulatedAnnealingOptimizer(
            initial_temperature=120.0,
            alpha=0.985,
            cutoff_temperature=0.01,
            iterations_per_t=20,
            size_limit=5,
            rand_seed=seed_base + r,
        )
        out = opt.solve(ising_use)
        if out is not None and len(out) > 0:
            out_s = sort_ising_solutions(out, ising_use)
            s = out_s[0]
            s_aux = s[-1]
            x_bin = ((s[:-1] * s_aux) + 1) // 2
            sols.append(x_bin.astype(int))
    dt = time.perf_counter() - t0
    return sols, bw_raw, bw_used, dr, dt


def run_v1_q2_q3(m_log_full, fm_full, cats_A, cats_B, cats_X):
    q2_full = pd.read_csv(os.path.join(DATA_DIR, "07_castelli_q2_pool.csv"))
    q2_blind = pd.read_csv(os.path.join(DATA_DIR, "11_castelli_q2_pool_blind.csv"))
    q3_full = pd.read_csv(os.path.join(DATA_DIR, "08_castelli_q3_pool.csv"))
    q3_blind = pd.read_csv(os.path.join(DATA_DIR, "12_castelli_q3_pool_blind.csv"))

    # Score Q2 pools with v1_Direct_LGBM_log1pV
    def score_df(df):
        X = build_basic_v1_features(df)
        X["A"] = pd.Categorical(df["A"], categories=cats_A)
        X["B"] = pd.Categorical(df["B"], categories=cats_B)
        X["X"] = pd.Categorical(df["X"], categories=cats_X)
        return np.maximum(0.0, np.expm1(m_log_full.predict(X)))

    q2_full["v1_pred"] = score_df(q2_full)
    q2_blind["v1_pred"] = score_df(q2_blind)

    # Build 52x52 score matrix for Q2 full & Q2 blind (non-pool entries penalized +50)
    elems = sorted(cats_A)
    e2i = {e: i for i, e in enumerate(elems)}

    q2_results = {}
    for label, df_pool in [("Q2_full_07", q2_full), ("Q2_blind_11", q2_blind)]:
        M = np.full((52, 52), 50.0, dtype=float)
        pool_set = set()
        for a, b, val in zip(df_pool["A"], df_pool["B"], df_pool["v1_pred"]):
            M[e2i[a], e2i[b]] = val
            pool_set.add((a, b))
        true_min = float(df_pool["v1_pred"].min())
        top10_set = set(zip(df_pool.nsmallest(10, "v1_pred")["A"], df_pool.nsmallest(10, "v1_pred")["B"]))

        Q_naive = _build_naive_q2_qubo(M, elems, elems, penalty=20.0)
        for mode, bw in [("v1_Float_SA", None), ("v1_Naive8bit_SA", 8)]:
            sols, bw_raw, bw_used, dr, dt = _solve_qubo_sa(Q_naive, n_runs=40, bit_width=bw, seed_base=200)
            kA_list, kB_list = [], []
            feas = 0
            best_val = None
            hit_top10 = set()
            for x in sols:
                a_idx = np.where(x[:52] == 1)[0]
                b_idx = np.where(x[52:104] == 1)[0]
                kA_list.append(len(a_idx))
                kB_list.append(len(b_idx))
                if len(a_idx) == 1 and len(b_idx) == 1 and a_idx[0] != b_idx[0]:
                    pair = (elems[a_idx[0]], elems[b_idx[0]])
                    if pair in pool_set:
                        feas += 1
                        val = M[a_idx[0], b_idx[0]]
                        if best_val is None or val < best_val:
                            best_val = float(val)
                        if pair in top10_set:
                            hit_top10.add(pair)
            q2_results[f"{label}_{mode}"] = {
                "spins": int(Q_naive.shape[0] + 1),
                "raw_bit_width": bw_raw,
                "used_bit_width": bw_used,
                "dynamic_range": round(dr, 2),
                "feasibility_pct": round(100.0 * feas / max(1, len(sols)), 2),
                "mean_kA": round(float(np.mean(kA_list)), 3),
                "mean_kB": round(float(np.mean(kB_list)), 3),
                "best_score": round(best_val, 4) if best_val is not None else None,
                "exact_min_score": round(true_min, 4),
                "gap_to_opt": round(best_val - true_min, 4) if best_val is not None else None,
                "top10_hits": len(hit_top10),
                "runtime_sec": round(dt, 3),
            }

    # v1 Q3: 107-var pairwise FM-QUBO (no 3-body) + naive Rosenberg auxiliary count analysis
    Q_fm107, cA, cB, cX = fm_full.get_qubo_matrix_q3(x_subset=("O3", "NO2", "FO2"), penalty=20.0)
    q3_full["v1_fm_pred"] = np.maximum(0.0, fm_full.predict(q3_full))
    q3_blind["v1_fm_pred"] = np.maximum(0.0, fm_full.predict(q3_blind))
    q3_blind["v1_lgb_pred"] = score_df(q3_blind)
    q3_full_lookup = {(a, b, x): v for a, b, x, v in zip(q3_full["A"], q3_full["B"], q3_full["X"], q3_full["v1_fm_pred"])}
    q3_full_top15 = set(zip(q3_full.nsmallest(15, "v1_fm_pred")["A"], q3_full.nsmallest(15, "v1_fm_pred")["B"], q3_full.nsmallest(15, "v1_fm_pred")["X"]))
    q3_full_min = float(q3_full["v1_fm_pred"].min())

    q3_results = {}
    for mode, bw in [("v1_FM107_Float_SA", None), ("v1_FM107_Naive8bit_SA", 8)]:
        sols, bw_raw, bw_used, dr, dt = _solve_qubo_sa(Q_fm107, n_runs=40, bit_width=bw, seed_base=300)
        kA_l, kB_l, kX_l = [], [], []
        feas = 0
        best_val = None
        hit_top15 = set()
        for x in sols:
            ai = np.where(x[:52] == 1)[0]
            bi = np.where(x[52:104] == 1)[0]
            xi = np.where(x[104:107] == 1)[0]
            kA_l.append(len(ai))
            kB_l.append(len(bi))
            kX_l.append(len(xi))
            if len(ai) == 1 and len(bi) == 1 and len(xi) == 1 and ai[0] != bi[0]:
                key = (cA[ai[0]], cB[bi[0]], cX[xi[0]])
                if key in q3_full_lookup:
                    feas += 1
                    val = q3_full_lookup[key]
                    if best_val is None or val < best_val:
                        best_val = float(val)
                    if key in q3_full_top15:
                        hit_top15.add(key)
        q3_results[f"Q3_full_08_{mode}"] = {
            "spins": int(Q_fm107.shape[0] + 1),
            "aux_vars": 0,
            "raw_bit_width": bw_raw,
            "used_bit_width": bw_used,
            "dynamic_range": round(dr, 2),
            "feasibility_pct": round(100.0 * feas / max(1, len(sols)), 2),
            "mean_kA": round(float(np.mean(kA_l)), 3),
            "mean_kB": round(float(np.mean(kB_l)), 3),
            "mean_kX": round(float(np.mean(kX_l)), 3),
            "best_score": round(best_val, 4) if best_val is not None else None,
            "exact_min_score": round(q3_full_min, 4),
            "gap_to_opt": round(best_val - q3_full_min, 4) if best_val is not None else None,
            "top15_hits": len(hit_top15),
            "runtime_sec": round(dt, 3),
        }

    # Exact structural count of naive full Rosenberg reduction for 3-body Q3 (52*52 = 2704 aux)
    q3_results["Q3_v1_NaiveRosenberg_Structure"] = {
        "base_vars": 107,
        "aux_vars_AB_pairing": 52 * 51,
        "total_spins_AB_pairing": 107 + 52 * 51 + 1,
        "note": "Pairing (a_i, b_j) requires 2,652 auxiliary qubits (2,760 spins) before pruning, exceeding 1000-qubit SPQC capacity.",
    }
    return q2_results, q3_results


def main():
    print("=== Running v1 Baseline Benchmark ===")
    q1_metrics, q1_dt, m_log_full, fm_full, cA, cB, cX = run_v1_q1()
    print(f"\n[v1 Q1] Completed in {q1_dt:.2f}s:")
    print(pd.DataFrame(q1_metrics).to_string(index=False))

    q2_res, q3_res = run_v1_q2_q3(m_log_full, fm_full, cA, cB, cX)
    print("\n[v1 Q2] Results:")
    print(json.dumps(q2_res, indent=2))
    print("\n[v1 Q3] Results:")
    print(json.dumps(q3_res, indent=2))

    out = {
        "q1_metrics": q1_metrics,
        "q1_runtime_sec": round(q1_dt, 3),
        "q2_results": q2_res,
        "q3_results": q3_res,
    }
    with open(os.path.join(OUT_DIR, "v1_baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved v1_baseline_metrics.json successfully.")


if __name__ == "__main__":
    main()
