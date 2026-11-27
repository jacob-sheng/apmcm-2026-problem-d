#!/usr/bin/env python3
"""
v3 Remediated Closed-Loop Acquisition & Fair Multi-Axis Sensitivity Suite (M4A):
  1. Closed-Loop Benchmark on 03_castelli_val.csv (F04, F14, F15):
     - 20 paired random seeds (N=3,969; M=18 for V<0.5; M=27 for V<1.0)
     - Exact analytical hypergeometric expectation baseline E[H(K)] = K * M / N
     - Converged L-BFGS-B 2nd-order One-Hot FM (explicit scalar-V oracle budget)
     - Composition-physics scalar-V LGBM ablation (isolating features under scalar-V)
     - Side-by-side acquisition routes under multi-property oracle:
         (a) v2_Classical_SeqRank (np.argsort)
         (b) v2_Real_8Bit_QUBO (genuine per-round 8-bit Model 3A-Comp masked QUBO via Kaiwu SA)
         and (c) v1_FM_LBFGS_SeqRank vs (d) v1_FM_LBFGS_QUBO (per-round 112-spin FM QUBO via Kaiwu SA)
  2. Equal-Budget Solver Top-K Comparison (F13):
     - Same budget (n_runs=50, 1 best shot/run -> 50 shots + unique Top-K) on frozen V_hat pools
     - Compares v1 Naive, v2 Q2 Exact-8bit, v2 Q3 3A-Comp, 3A-HOBO, 3B-ANOVA (SA & Tabu)
  3. Independent 4-12-Bit Quantization Sweep (F16):
     - Independently searches maximal usable M_max(B) bounded by M_B = 2^(B-1)-1 for every B
     - Full global ground-state certification + continuous regret + SA (n_runs=50, 1 best/run)
  4. Refined Fixed-M_max=4 Penalty Sweep Around Phase Transition (F16) + C_cap Pareto Sweep:
     - Fixes M_max=4 on Q2 Blind (12 spins) across alpha_P in [0.50 .. 3.00]
     - Reports BOTH exact global enumeration feasibility AND SA empirical feasibility
  5. Multi-Scale Surrogate Ranking Perturbation Stability (F17):
     - n_trials=500 across scales [0.5, 1.0, 1.5, 2.0] for independent Gaussian AND
       (A,B)-group-correlated + p_semi logit perturbation on Q2 & Q3 blind pools
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
from scipy.stats import spearmanr
import lightgbm as lgb
import kaiwu as kw
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)
from common_physics import (
    DATA_DIR,
    build_v2_physics_features,
    compute_band_lower_bound,
    v_band_terms,
    exact_expected_hinge_sq,
    safe_ising_bit_width,
    sort_ising_solutions,
)
from v1_baseline import _build_naive_q2_qubo
from v2_qubo_q2_q3 import (
    build_exact8_q2_qubo,
    build_q3_comp_qubo,
    build_q3_submodular_hobo_qubo,
    build_q3_compact_anova_qubo,
    certify_global_ground_states,
    _solve_ising_multi_solver,
    evaluate_q2_solutions,
    evaluate_q3_solutions,
)

OUT_DIR = os.path.join(CUR_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def _ising_sha256(ising_mat):
    arr = np.asarray(ising_mat)
    if np.issubdtype(arr.dtype, np.integer):
        buf = arr.astype(np.int64).tobytes()
    else:
        buf = np.asarray(arr, dtype=np.float64).tobytes()
    return hashlib.sha256(buf).hexdigest()


# ---------------------------------------------------------------------------
# Converged 2nd-Order One-Hot Factorization Machine via L-BFGS-B (F15)
# ---------------------------------------------------------------------------

def fit_onehot_fm_lbfgs(
    df_obs,
    y_obs,
    cols_A,
    cols_B,
    cols_X,
    k=8,
    reg=0.05,
    maxiter=600,
    seed=2026,
):
    """
    Train a 2nd-order One-Hot Factorization Machine on (A, B, X) to convergence
    using scipy.optimize.minimize(..., method="L-BFGS-B") with exact analytical gradients.
    """
    map_A = {e: i for i, e in enumerate(cols_A)}
    map_B = {e: len(cols_A) + i for i, e in enumerate(cols_B)}
    map_X = {e: len(cols_A) + len(cols_B) + i for i, e in enumerate(cols_X)}
    d = len(cols_A) + len(cols_B) + len(cols_X)

    ia = df_obs["A"].map(map_A).values
    ib = df_obs["B"].map(map_B).values
    ix = df_obs["X"].map(map_X).values
    bi = np.stack([ia, ib, ix], axis=1)
    y = np.asarray(y_obs, dtype=np.float64)
    n = len(y)

    rng = np.random.default_rng(seed)
    w0_init = float(np.mean(y))
    w_init = np.zeros(d, dtype=np.float64)
    V_init = rng.normal(0.0, 0.05, size=(d, k))
    theta0 = np.concatenate([[w0_init], w_init, V_init.ravel()])

    def obj_and_grad(theta):
        w0 = theta[0]
        w = theta[1:1 + d]
        V = theta[1 + d:].reshape(d, k)

        v_act = V[bi]  # (n, 3, k)
        sv = v_act.sum(axis=1)  # (n, k)
        sv2 = (v_act ** 2).sum(axis=1)  # (n, k)
        pred = w0 + w[bi].sum(axis=1) + 0.5 * np.sum(sv ** 2 - sv2, axis=1)
        diff = pred - y

        loss = 0.5 * float(np.mean(diff ** 2)) + 0.5 * reg * (
            float(np.sum(w ** 2)) + float(np.sum(V ** 2))
        )

        err = diff / n
        gw0 = float(err.sum())
        gw = reg * w
        gV = reg * V
        for j in range(3):
            col_j = bi[:, j]
            np.add.at(gw, col_j, err)
            np.add.at(gV, col_j, err[:, None] * (sv - v_act[:, j, :]))

        grad = np.concatenate([[gw0], gw, gV.ravel()])
        return loss, grad

    res = minimize(
        obj_and_grad,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6},
    )
    theta = res.x
    w0 = float(theta[0])
    w = theta[1:1 + d].copy()
    V = theta[1 + d:].reshape(d, k).copy()

    # Final training MSE diagnostic
    v_act = V[bi]
    sv = v_act.sum(axis=1)
    sv2 = (v_act ** 2).sum(axis=1)
    pred_tr = w0 + w[bi].sum(axis=1) + 0.5 * np.sum(sv ** 2 - sv2, axis=1)
    train_mse = float(np.mean((pred_tr - y) ** 2))

    return {
        "w0": w0,
        "w": w,
        "V": V,
        "cols_A": cols_A,
        "cols_B": cols_B,
        "cols_X": cols_X,
        "map_A": map_A,
        "map_B": map_B,
        "map_X": map_X,
        "d": d,
        "k": k,
        "converged": bool(res.success),
        "nit": int(res.nit),
        "train_mse": train_mse,
        "grad_norm": float(np.linalg.norm(res.jac, ord=np.inf)),
    }


def predict_onehot_fm(fm_model, df):
    ia = df["A"].map(fm_model["map_A"]).values
    ib = df["B"].map(fm_model["map_B"]).values
    ix = df["X"].map(fm_model["map_X"]).values
    bi = np.stack([ia, ib, ix], axis=1)
    v_act = fm_model["V"][bi]
    sv = v_act.sum(axis=1)
    sv2 = (v_act ** 2).sum(axis=1)
    return fm_model["w0"] + fm_model["w"][bi].sum(axis=1) + 0.5 * np.sum(sv ** 2 - sv2, axis=1)


def build_fm_q3_qubo(fm_model, valid_ab_set, fully_queried_ab_set=None, penalty=None):
    """Build 111-variable (52 A + 52 B + 7 X) pairwise FM QUBO from converged FM weights."""
    cols_A = fm_model["cols_A"]
    cols_B = fm_model["cols_B"]
    cols_X = fm_model["cols_X"]
    nA, nB, nX = len(cols_A), len(cols_B), len(cols_X)
    N = nA + nB + nX
    w = fm_model["w"]
    V = fm_model["V"]
    G = V @ V.T

    if penalty is None:
        max_lin = float(np.max(np.abs(w))) if len(w) else 1.0
        max_pair = float(np.max(np.abs(G))) if G.size else 1.0
        penalty = max(25.0, 2.0 * (max_lin + 3.0 * max_pair) + 5.0)

    Q = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        Q[i, i] += w[i]
    for i in range(N):
        for j in range(i + 1, N):
            Q[i, j] += G[i, j]

    for start, sz in [(0, nA), (nA, nB), (nA + nB, nX)]:
        for i in range(start, start + sz):
            Q[i, i] -= penalty
            for j in range(i + 1, start + sz):
                Q[i, j] += 2.0 * penalty

    fq_set = fully_queried_ab_set or set()
    for i, ea in enumerate(cols_A):
        for j, eb in enumerate(cols_B):
            if ea == eb or (ea, eb) not in valid_ab_set or (ea, eb) in fq_set:
                Q[i, nA + j] += penalty

    return 0.5 * (Q + Q.T), float(penalty)


# ---------------------------------------------------------------------------
# Per-Round 8-Bit Masked Model 3A-Comp QUBO Builder for Closed-Loop (F04)
# ---------------------------------------------------------------------------

def build_masked_8bit_comp_qubo_round(df_unq_sorted, val_trip_set, obs_triples, batch_size=10):
    """
    Build a genuine 8-bit integer Ising matrix over top unqueried candidates
    using Model 3A-Comp (zero-auxiliary composite (A,X)-B bipartite QUBO),
    where any (A,B,X) already in obs_triples or absent from val_trip_set is
    forbidden via P_forbid >= P_block (multiple of 8).
    Shrinks active pool size K_sub within [22, 10] if needed so M_max >= 1
    always fits strictly in [-127, 127].
    """
    for K_sub in [20, 18, 16, 14, 12, 10]:
        sub = df_unq_sorted.head(min(K_sub, len(df_unq_sorted)))
        if len(sub) < batch_size:
            continue
        ax_pairs = sorted(set(zip(sub["A"], sub["X"])))
        elems_B = sorted(sub["B"].unique())
        nAX, nB = len(ax_pairs), len(elems_B)
        N = nAX + nB

        c_cap = max(float(sub["v_score"].max()) * 1.15, float(sub["v_score"].min()) + 0.5)
        score_map = {(a, b, x): float(s) for a, b, x, s in zip(sub["A"], sub["B"], sub["X"], sub["v_score"])}

        M_bar = np.zeros((nAX, nB), dtype=float)
        valid_mask = np.zeros((nAX, nB), dtype=bool)
        unq_in_sub = 0
        for p_idx, (ea, ex) in enumerate(ax_pairs):
            for j, eb in enumerate(elems_B):
                trip = (ea, eb, ex)
                if ea != eb and (trip in val_trip_set) and (trip not in obs_triples):
                    valid_mask[p_idx, j] = True
                    unq_in_sub += 1
                    v = score_map.get(trip, c_cap)
                    if v < c_cap:
                        M_bar[p_idx, j] = v - c_cap

        if unq_in_sub < batch_size:
            continue

        for M_max in range(12, 0, -1):
            m_int = np.round((M_bar / c_cap) * M_max).astype(int)
            amax = int(np.max(np.abs(m_int)))
            if amax == 0:
                continue
            j_pen = max(3 * amax + 1, int(np.ceil(2.0 * 1.60 * amax)))
            P_block = 4.0 * j_pen
            P_forbid = 8.0 * int(np.ceil(j_pen / 2.0))

            Q = np.zeros((N, N), dtype=float)
            for p_idx in range(nAX):
                for j in range(nB):
                    if not valid_mask[p_idx, j]:
                        Q[p_idx, nAX + j] += P_forbid
                    elif m_int[p_idx, j] != 0:
                        Q[p_idx, nAX + j] += 8.0 * m_int[p_idx, j]

            for start, sz in [(0, nAX), (nAX, nB)]:
                for i in range(start, start + sz):
                    Q[i, i] -= P_block
                    for j in range(i + 1, start + sz):
                        Q[i, j] += 2.0 * P_block

            Q_sym = 0.5 * (Q + Q.T)
            ising_raw, _ = kw.qubo_matrix_to_ising_matrix(Q_sym)
            ising_rnd = np.round(ising_raw)
            max_err = float(np.max(np.abs(ising_raw - ising_rnd)))
            max_abs = int(np.max(np.abs(ising_rnd)))
            if max_err < 1e-6 and max_abs <= 127:
                return ising_rnd.astype(np.int64), {
                    "ax_pairs": ax_pairs,
                    "elems_B": elems_B,
                    "nAX": nAX,
                    "nB": nB,
                    "spins": N + 1,
                    "M_max": int(M_max),
                    "max_abs": max_abs,
                    "bit_width": safe_ising_bit_width(ising_rnd.astype(np.int64)),
                    "exact_8bit": True,
                    "K_sub": len(sub),
                    "unq_in_sub": unq_in_sub,
                    "matrix_sha256": _ising_sha256(ising_rnd.astype(np.int64)),
                }

    # Rare fallback if even K_sub=10 exceeds 127: rescale precision to 8-bit
    sub = df_unq_sorted.head(min(10, len(df_unq_sorted)))
    ax_pairs = sorted(set(zip(sub["A"], sub["X"])))
    elems_B = sorted(sub["B"].unique())
    nAX, nB = len(ax_pairs), len(elems_B)
    N = nAX + nB
    c_cap = max(float(sub["v_score"].max()) * 1.15, float(sub["v_score"].min()) + 0.5)
    score_map = {(a, b, x): float(s) for a, b, x, s in zip(sub["A"], sub["B"], sub["X"], sub["v_score"])}
    Q = np.zeros((N, N), dtype=float)
    for p_idx, (ea, ex) in enumerate(ax_pairs):
        for j, eb in enumerate(elems_B):
            trip = (ea, eb, ex)
            if ea != eb and (trip in val_trip_set) and (trip not in obs_triples):
                v = score_map.get(trip, c_cap)
                if v < c_cap:
                    Q[p_idx, nAX + j] += 8.0 * round(((v - c_cap) / c_cap) * 2)
            else:
                Q[p_idx, nAX + j] += 32.0
    for start, sz in [(0, nAX), (nAX, nB)]:
        for i in range(start, start + sz):
            Q[i, i] -= 24.0
            for j in range(i + 1, start + sz):
                Q[i, j] += 48.0
    ising_raw, _ = kw.qubo_matrix_to_ising_matrix(0.5 * (Q + Q.T))
    ising_8 = np.round(kw.adjust_ising_matrix_precision(ising_raw, bit_width=8)).astype(np.int64)
    return ising_8, {
        "ax_pairs": ax_pairs,
        "elems_B": elems_B,
        "nAX": nAX,
        "nB": nB,
        "spins": N + 1,
        "M_max": 0,
        "max_abs": int(np.max(np.abs(ising_8))),
        "bit_width": safe_ising_bit_width(ising_8),
        "exact_8bit": False,
        "K_sub": len(sub),
        "unq_in_sub": len(sub),
        "matrix_sha256": _ising_sha256(ising_8),
    }


def _solve_round_comp_qubo_sa(ising, meta, val_idx_map, obs_set, batch_size=10, n_sa_runs=15, seed_base=0):
    """Solve per-round 8-bit masked Model 3A-Comp QUBO via Kaiwu SA and extract batch_size unique unqueried hits."""
    nAX, nB = meta["nAX"], meta["nB"]
    ax_pairs = meta["ax_pairs"]
    elems_B = meta["elems_B"]

    total_shots = 0
    feasible_shots = 0
    top1_total = 0
    top1_feasible = 0
    cand_stats = {}  # idx -> [min_energy, -hit_count]

    for r in range(n_sa_runs):
        opt = kw.SimulatedAnnealingOptimizer(
            initial_temperature=80.0,
            alpha=0.98,
            cutoff_temperature=0.01,
            iterations_per_t=15,
            size_limit=20,
            rand_seed=seed_base + r,
        )
        out = opt.solve(ising)
        if out is None or len(out) == 0:
            continue
        out_s = sort_ising_solutions(out, ising)
        for s_idx, row in enumerate(out_s):
            total_shots += 1
            if s_idx == 0:
                top1_total += 1
            e_val = float(-row @ ising @ row)
            s_aux = row[-1]
            x = (((row[:-1] * s_aux) + 1) // 2).astype(int)
            axi = np.where(x[:nAX] == 1)[0]
            bi = np.where(x[nAX:nAX + nB] == 1)[0]
            if len(axi) == 1 and len(bi) == 1:
                ea, ex = ax_pairs[int(axi[0])]
                eb = elems_B[int(bi[0])]
                if ea != eb:
                    idx = val_idx_map.get((ea, eb, ex))
                    if idx is not None and idx not in obs_set:
                        feasible_shots += 1
                        if s_idx == 0:
                            top1_feasible += 1
                        if idx not in cand_stats:
                            cand_stats[idx] = [e_val, -1]
                        else:
                            cand_stats[idx][0] = min(cand_stats[idx][0], e_val)
                            cand_stats[idx][1] -= 1

    ordered = sorted(cand_stats.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    picks = [idx for idx, _ in ordered[:batch_size]]
    strict_feas_pct = round(100.0 * feasible_shots / max(1, total_shots), 2)
    top1_feas_pct = round(100.0 * top1_feasible / max(1, top1_total), 2)
    return picks, strict_feas_pct, top1_feas_pct, len(cand_stats)


def _solve_round_fm_qubo_sa(ising, cols_A, cols_B, cols_X, val_idx_map, obs_set, batch_size=10, n_sa_runs=15, seed_base=0):
    """Solve per-round 112-spin FM QUBO via Kaiwu SA and extract batch_size unique unqueried hits."""
    nA, nB, nX = len(cols_A), len(cols_B), len(cols_X)
    total_shots = 0
    feasible_shots = 0
    top1_total = 0
    top1_feasible = 0
    cand_stats = {}

    for r in range(n_sa_runs):
        opt = kw.SimulatedAnnealingOptimizer(
            initial_temperature=100.0,
            alpha=0.98,
            cutoff_temperature=0.01,
            iterations_per_t=15,
            size_limit=20,
            rand_seed=seed_base + r,
        )
        out = opt.solve(ising)
        if out is None or len(out) == 0:
            continue
        out_s = sort_ising_solutions(out, ising)
        for s_idx, row in enumerate(out_s):
            total_shots += 1
            if s_idx == 0:
                top1_total += 1
            e_val = float(-row @ ising @ row)
            s_aux = row[-1]
            x = (((row[:-1] * s_aux) + 1) // 2).astype(int)
            ai = np.where(x[:nA] == 1)[0]
            bi = np.where(x[nA:nA + nB] == 1)[0]
            xi = np.where(x[nA + nB:nA + nB + nX] == 1)[0]
            if len(ai) == 1 and len(bi) == 1 and len(xi) == 1:
                ea, eb, ex = cols_A[int(ai[0])], cols_B[int(bi[0])], cols_X[int(xi[0])]
                if ea != eb:
                    idx = val_idx_map.get((ea, eb, ex))
                    if idx is not None and idx not in obs_set:
                        feasible_shots += 1
                        if s_idx == 0:
                            top1_feasible += 1
                        if idx not in cand_stats:
                            cand_stats[idx] = [e_val, -1]
                        else:
                            cand_stats[idx][0] = min(cand_stats[idx][0], e_val)
                            cand_stats[idx][1] -= 1

    ordered = sorted(cand_stats.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
    picks = [idx for idx, _ in ordered[:batch_size]]
    strict_feas_pct = round(100.0 * feasible_shots / max(1, total_shots), 2)
    top1_feas_pct = round(100.0 * top1_feasible / max(1, top1_total), 2)
    return picks, strict_feas_pct, top1_feas_pct, len(cand_stats)


def _score_v2_physics_hurdle_round(va, df_obs, obs_idx, X_phys_all, X_aug_all_oh, phys_va, seed_r):
    """Train Butler-Ginley Ec Ridge + physics-hurdle LightGBM strictly on observed indices."""
    n_total = len(va)
    X_oh_obs = X_aug_all_oh[obs_idx]
    y_log_ec = np.log(0.5 * (df_obs["vbm"].values + df_obs["cbm"].values))
    bc = Ridge(alpha=1e-3, fit_intercept=True).fit(X_oh_obs, y_log_ec)
    ec_hat = np.exp(bc.predict(X_aug_all_oh))
    _, feas_w, v_min_ec, _ = compute_band_lower_bound(ec_hat)

    X_aug_all = np.column_stack([X_phys_all, ec_hat, feas_w, v_min_ec])
    X_aug_obs = X_aug_all[obs_idx]

    m_ef = lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.06, num_leaves=15, min_child_samples=5,
        n_jobs=2, random_state=seed_r, verbose=-1
    ).fit(X_aug_obs, df_obs["e_form"].values)
    mu_ef = m_ef.predict(X_aug_all)

    y_semi_obs = (df_obs["gap"].values > 0).astype(int)
    if y_semi_obs.sum() >= 3 and (len(y_semi_obs) - y_semi_obs.sum()) >= 3:
        m_cls = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.06, num_leaves=15, min_child_samples=5,
            n_jobs=2, random_state=seed_r, verbose=-1
        ).fit(X_aug_obs, y_semi_obs)
        p_semi = m_cls.predict_proba(X_aug_all)[:, 1]
        sm = y_semi_obs == 1
        m_gap = lgb.LGBMRegressor(
            n_estimators=80, learning_rate=0.06, num_leaves=7, min_child_samples=3,
            n_jobs=2, random_state=seed_r, verbose=-1
        ).fit(X_aug_obs[sm], df_obs["gap"].values[sm])
        mu_g = np.maximum(0.1, m_gap.predict(X_aug_all))
    else:
        p_semi = np.where(phys_va["is_cs_balanced"].values == 1, 0.35, 0.02)
        mu_g = np.full(n_total, 2.0)

    v_f = exact_expected_hinge_sq(mu_ef, np.full(n_total, 0.25), 0.21, +1.0, 0.5)
    vm_gl, vm_gh, vm_v, vm_c = v_band_terms(np.zeros(n_total), ec_hat)
    vs_gl = exact_expected_hinge_sq(mu_g, np.full(n_total, 0.8), 1.4, -1.0, 0.5)
    vs_gh = exact_expected_hinge_sq(mu_g, np.full(n_total, 0.8), 3.1, +1.0, 0.5)
    vs_v = exact_expected_hinge_sq(ec_hat + 0.5 * mu_g, np.full(n_total, 0.4), 5.73, -1.0, 0.5)
    vs_c = exact_expected_hinge_sq(ec_hat - 0.5 * mu_g, np.full(n_total, 0.4), 4.50, +1.0, 0.5)
    return v_f + (1 - p_semi) * (vm_gl + vm_gh + vm_v + vm_c) + p_semi * (vs_gl + vs_gh + vs_v + vs_c)


def run_closed_loop_fmqa_benchmark(n_seeds=20, n_init=50, batch_size=10, n_rounds=15):
    """
    20-seed closed-loop simulated DFT screening benchmark on 03_castelli_val.csv
    (N=3,969 compounds, M=18 rare targets V<0.5, M=27 near-zero targets V<1.0).
    Distinguishes oracle information budgets (scalar V vs multi-property DFT labels)
    and classical SeqRank vs genuine per-round QUBO acquisition.
    """
    t0 = time.perf_counter()
    va = pd.read_csv(os.path.join(DATA_DIR, "03_castelli_val.csv")).reset_index(drop=True)
    n_total = len(va)
    y_true = va["V"].values
    is_rare05 = (y_true < 0.5).astype(int)
    is_rare10 = (y_true < 1.0).astype(int)
    n_rare05 = int(is_rare05.sum())
    n_rare10 = int(is_rare10.sum())

    cols_A = sorted(va["A"].unique())
    cols_B = sorted(va["B"].unique())
    cols_X = sorted(va["X"].unique())
    val_idx_map = {(a, b, x): int(i) for i, (a, b, x) in enumerate(zip(va["A"], va["B"], va["X"]))}
    val_trip_set = set(val_idx_map.keys())
    val_ab_set = set(zip(va["A"], va["B"]))
    ab_to_indices = {}
    for idx, (a, b) in enumerate(zip(va["A"], va["B"])):
        ab_to_indices.setdefault((a, b), []).append(idx)

    # Precompute composition-only physics features on 03_val (zero label leakage)
    phys_va = build_v2_physics_features(va)
    X_phys_all = phys_va.values.astype(np.float32)
    X_oh_all_df = pd.get_dummies(va[["A", "B", "X"]], dtype=float)
    X_oh_all = X_oh_all_df.values

    budgets = [n_init + r * batch_size for r in range(n_rounds + 1)]
    method_names = [
        "Random",
        "v1_FM_LBFGS_SeqRank",
        "v1_FM_LBFGS_QUBO",
        "v2_PhysLGBM_ScalarV_SeqRank",
        "v2_Classical_SeqRank",
        "v2_Real_8Bit_QUBO",
    ]
    curves = {
        m: {"rare05": np.zeros((n_seeds, n_rounds + 1)), "rare10": np.zeros((n_seeds, n_rounds + 1))}
        for m in method_names
    }

    fm_seq_conv_logs = []
    fm_qubo_round_logs = []
    v2_qubo_round_logs = []

    for s_idx in range(n_seeds):
        seed = 101 + s_idx
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(n_total, size=n_init, replace=False)

        # 1. Empirical Random (paired initial set)
        obs_rand = set(init_idx.tolist())
        curves["Random"]["rare05"][s_idx, 0] = is_rare05[list(obs_rand)].sum()
        curves["Random"]["rare10"][s_idx, 0] = is_rare10[list(obs_rand)].sum()
        rem_rand = [i for i in range(n_total) if i not in obs_rand]
        rng.shuffle(rem_rand)
        ptr = 0
        for r in range(1, n_rounds + 1):
            pick = rem_rand[ptr:ptr + batch_size]
            ptr += batch_size
            obs_rand.update(pick)
            curves["Random"]["rare05"][s_idx, r] = is_rare05[list(obs_rand)].sum()
            curves["Random"]["rare10"][s_idx, r] = is_rare10[list(obs_rand)].sum()

        # 2. v1_FM_LBFGS_SeqRank (scalar V -> L-BFGS-B FM -> np.argsort)
        obs_fms = list(init_idx.tolist())
        obs_fms_set = set(obs_fms)
        curves["v1_FM_LBFGS_SeqRank"]["rare05"][s_idx, 0] = is_rare05[obs_fms].sum()
        curves["v1_FM_LBFGS_SeqRank"]["rare10"][s_idx, 0] = is_rare10[obs_fms].sum()
        for r in range(1, n_rounds + 1):
            df_obs = va.iloc[obs_fms]
            fm = fit_onehot_fm_lbfgs(
                df_obs, df_obs["V"].values, cols_A, cols_B, cols_X,
                k=8, reg=0.05, maxiter=600, seed=seed + r
            )
            fm_seq_conv_logs.append((fm["converged"], fm["nit"], fm["train_mse"], fm["grad_norm"]))
            preds_all = predict_onehot_fm(fm, va)
            preds_masked = preds_all.copy()
            preds_masked[list(obs_fms_set)] = 1e9
            pick = np.argsort(preds_masked)[:batch_size].tolist()
            obs_fms.extend(pick)
            obs_fms_set.update(pick)
            curves["v1_FM_LBFGS_SeqRank"]["rare05"][s_idx, r] = is_rare05[obs_fms].sum()
            curves["v1_FM_LBFGS_SeqRank"]["rare10"][s_idx, r] = is_rare10[obs_fms].sum()

        # 3. v1_FM_LBFGS_QUBO (scalar V -> L-BFGS-B FM -> per-round 112-spin FM QUBO via Kaiwu SA)
        obs_fmq = list(init_idx.tolist())
        obs_fmq_set = set(obs_fmq)
        curves["v1_FM_LBFGS_QUBO"]["rare05"][s_idx, 0] = is_rare05[obs_fmq].sum()
        curves["v1_FM_LBFGS_QUBO"]["rare10"][s_idx, 0] = is_rare10[obs_fmq].sum()
        for r in range(1, n_rounds + 1):
            df_obs = va.iloc[obs_fmq]
            fm = fit_onehot_fm_lbfgs(
                df_obs, df_obs["V"].values, cols_A, cols_B, cols_X,
                k=8, reg=0.05, maxiter=600, seed=seed + 1000 + r
            )
            fq_ab = {ab for ab, idxs in ab_to_indices.items() if all(i in obs_fmq_set for i in idxs)}
            Q_fm, pen_fm = build_fm_q3_qubo(fm, val_ab_set, fully_queried_ab_set=fq_ab)
            ising_fm, _ = kw.qubo_matrix_to_ising_matrix(Q_fm)
            picks, feas_pct, top1_feas_pct, n_uniq = _solve_round_fm_qubo_sa(
                ising_fm, cols_A, cols_B, cols_X, val_idx_map, obs_fmq_set,
                batch_size=batch_size, n_sa_runs=15, seed_base=seed * 200 + r * 20
            )
            fallback_fills = 0
            if len(picks) < batch_size:
                preds_masked = predict_onehot_fm(fm, va)
                preds_masked[list(obs_fmq_set)] = 1e9
                for idx_p in np.argsort(preds_masked):
                    idx_i = int(idx_p)
                    if idx_i not in picks and idx_i not in obs_fmq_set:
                        picks.append(idx_i)
                        fallback_fills += 1
                        if len(picks) == batch_size:
                            break
            fm_qubo_round_logs.append({
                "seed": seed,
                "round": r,
                "spins": int(ising_fm.shape[0]),
                "bit_width": safe_ising_bit_width(ising_fm),
                "penalty": round(pen_fm, 2),
                "strict_feasibility_pct": top1_feas_pct,
                "all_shots_strict_feasibility_pct": feas_pct,
                "unique_qubo_hits": n_uniq,
                "fallback_fills": fallback_fills,
                "matrix_sha256": _ising_sha256(ising_fm),
            })
            obs_fmq.extend(picks)
            obs_fmq_set.update(picks)
            curves["v1_FM_LBFGS_QUBO"]["rare05"][s_idx, r] = is_rare05[obs_fmq].sum()
            curves["v1_FM_LBFGS_QUBO"]["rare10"][s_idx, r] = is_rare10[obs_fmq].sum()

        # 4. v2_PhysLGBM_ScalarV_SeqRank (scalar V -> v2 composition physics + log1p(V) LGBM -> np.argsort)
        obs_sv = list(init_idx.tolist())
        obs_sv_set = set(obs_sv)
        curves["v2_PhysLGBM_ScalarV_SeqRank"]["rare05"][s_idx, 0] = is_rare05[obs_sv].sum()
        curves["v2_PhysLGBM_ScalarV_SeqRank"]["rare10"][s_idx, 0] = is_rare10[obs_sv].sum()
        for r in range(1, n_rounds + 1):
            y_obs_log = np.log1p(va["V"].values[obs_sv])
            m_sv = lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.06, num_leaves=15, min_child_samples=5,
                n_jobs=2, random_state=seed + r, verbose=-1
            ).fit(X_phys_all[obs_sv], y_obs_log)
            pred_sv = np.expm1(m_sv.predict(X_phys_all))
            pred_sv[list(obs_sv_set)] = 1e9
            pick = np.argsort(pred_sv)[:batch_size].tolist()
            obs_sv.extend(pick)
            obs_sv_set.update(pick)
            curves["v2_PhysLGBM_ScalarV_SeqRank"]["rare05"][s_idx, r] = is_rare05[obs_sv].sum()
            curves["v2_PhysLGBM_ScalarV_SeqRank"]["rare10"][s_idx, r] = is_rare10[obs_sv].sum()

        # 5. v2_Classical_SeqRank (multi-property DFT -> Ridge Ec + physics-hurdle LGBM -> np.argsort)
        obs_v2c = list(init_idx.tolist())
        obs_v2c_set = set(obs_v2c)
        curves["v2_Classical_SeqRank"]["rare05"][s_idx, 0] = is_rare05[obs_v2c].sum()
        curves["v2_Classical_SeqRank"]["rare10"][s_idx, 0] = is_rare10[obs_v2c].sum()
        for r in range(1, n_rounds + 1):
            v_score = _score_v2_physics_hurdle_round(
                va, va.iloc[obs_v2c], obs_v2c, X_phys_all, X_oh_all, phys_va, seed + r
            )
            v_score[list(obs_v2c_set)] = 1e9
            pick = np.argsort(v_score)[:batch_size].tolist()
            obs_v2c.extend(pick)
            obs_v2c_set.update(pick)
            curves["v2_Classical_SeqRank"]["rare05"][s_idx, r] = is_rare05[obs_v2c].sum()
            curves["v2_Classical_SeqRank"]["rare10"][s_idx, r] = is_rare10[obs_v2c].sum()

        # 6. v2_Real_8Bit_QUBO (multi-property DFT -> same physics-hurdle surrogate -> per-round 8-bit Masked Model 3A-Comp QUBO via Kaiwu SA)
        obs_v2q = list(init_idx.tolist())
        obs_v2q_set = set(obs_v2q)
        obs_triples_v2q = {(va.iloc[i]["A"], va.iloc[i]["B"], va.iloc[i]["X"]) for i in obs_v2q}
        curves["v2_Real_8Bit_QUBO"]["rare05"][s_idx, 0] = is_rare05[obs_v2q].sum()
        curves["v2_Real_8Bit_QUBO"]["rare10"][s_idx, 0] = is_rare10[obs_v2q].sum()
        for r in range(1, n_rounds + 1):
            v_score = _score_v2_physics_hurdle_round(
                va, va.iloc[obs_v2q], obs_v2q, X_phys_all, X_oh_all, phys_va, seed + 500 + r
            )
            unq_mask = np.ones(n_total, dtype=bool)
            unq_mask[list(obs_v2q_set)] = False
            df_unq = va.loc[unq_mask, ["A", "B", "X"]].copy()
            df_unq["v_score"] = v_score[unq_mask]
            df_unq_sorted = df_unq.sort_values("v_score", ascending=True).reset_index(drop=True)

            ising_8, qmeta = build_masked_8bit_comp_qubo_round(
                df_unq_sorted, val_trip_set, obs_triples_v2q, batch_size=batch_size
            )
            picks, feas_pct, top1_feas_pct, n_uniq = _solve_round_comp_qubo_sa(
                ising_8, qmeta, val_idx_map, obs_v2q_set,
                batch_size=batch_size, n_sa_runs=15, seed_base=seed * 300 + r * 20
            )
            fallback_fills = 0
            if len(picks) < batch_size:
                v_masked = v_score.copy()
                v_masked[list(obs_v2q_set)] = 1e9
                for idx_p in np.argsort(v_masked):
                    idx_i = int(idx_p)
                    if idx_i not in picks and idx_i not in obs_v2q_set:
                        picks.append(idx_i)
                        fallback_fills += 1
                        if len(picks) == batch_size:
                            break
            v2_qubo_round_logs.append({
                "seed": seed,
                "round": r,
                "spins": qmeta["spins"],
                "nAX": qmeta["nAX"],
                "nB": qmeta["nB"],
                "M_max": qmeta["M_max"],
                "max_abs": qmeta["max_abs"],
                "bit_width": qmeta["bit_width"],
                "exact_8bit": qmeta["exact_8bit"],
                "strict_feasibility_pct": top1_feas_pct,
                "all_shots_strict_feasibility_pct": feas_pct,
                "unique_qubo_hits": n_uniq,
                "fallback_fills": fallback_fills,
                "matrix_sha256": qmeta["matrix_sha256"],
            })
            obs_v2q.extend(picks)
            obs_v2q_set.update(picks)
            for i_new in picks:
                obs_triples_v2q.add((va.iloc[i_new]["A"], va.iloc[i_new]["B"], va.iloc[i_new]["X"]))
            curves["v2_Real_8Bit_QUBO"]["rare05"][s_idx, r] = is_rare05[obs_v2q].sum()
            curves["v2_Real_8Bit_QUBO"]["rare10"][s_idx, r] = is_rare10[obs_v2q].sum()

    # Exact analytical hypergeometric baseline E[H(K)] = K * M / N
    b_arr = np.asarray(budgets, dtype=float)
    p05 = n_rare05 / n_total
    p10 = n_rare10 / n_total
    hg_exp05 = b_arr * p05
    hg_std05 = np.sqrt(b_arr * p05 * (1.0 - p05) * (n_total - b_arr) / (n_total - 1.0))
    hg_exp10 = b_arr * p10
    hg_std10 = np.sqrt(b_arr * p10 * (1.0 - p10) * (n_total - b_arr) / (n_total - 1.0))

    emp_rand_final05 = float(curves["Random"]["rare05"][:, -1].mean())
    emp_rand_final10 = float(curves["Random"]["rare10"][:, -1].mean())
    hg_final05 = float(hg_exp05[-1])
    hg_final10 = float(hg_exp10[-1])

    oracle_meta = {
        "Random": {
            "oracle_labels_used": "none",
            "surrogate_model": "None",
            "acquisition_mechanism": "uniform_random_without_replacement",
            "uses_qubo_solver": False,
        },
        "v1_FM_LBFGS_SeqRank": {
            "oracle_labels_used": "scalar_V",
            "surrogate_model": "2nd_order_OneHot_FM_LBFGSB_k8",
            "acquisition_mechanism": "classical_np_argsort_unqueried",
            "uses_qubo_solver": False,
        },
        "v1_FM_LBFGS_QUBO": {
            "oracle_labels_used": "scalar_V",
            "surrogate_model": "2nd_order_OneHot_FM_LBFGSB_k8",
            "acquisition_mechanism": "per_round_112spin_FM_QUBO_KaiwuSA",
            "uses_qubo_solver": True,
        },
        "v2_PhysLGBM_ScalarV_SeqRank": {
            "oracle_labels_used": "scalar_V",
            "surrogate_model": "Composition_Physics_LGBM_log1pV",
            "acquisition_mechanism": "classical_np_argsort_unqueried",
            "uses_qubo_solver": False,
        },
        "v2_Classical_SeqRank": {
            "oracle_labels_used": "multi_property(e_form, gap, vbm, cbm)",
            "surrogate_model": "ButlerGinley_Ec_Ridge + PhysicsHurdle_LGBM_ExpectedHinge",
            "acquisition_mechanism": "classical_np_argsort_unqueried",
            "uses_qubo_solver": False,
        },
        "v2_Real_8Bit_QUBO": {
            "oracle_labels_used": "multi_property(e_form, gap, vbm, cbm)",
            "surrogate_model": "ButlerGinley_Ec_Ridge + PhysicsHurdle_LGBM_ExpectedHinge",
            "acquisition_mechanism": "per_round_8bit_Model3A_Comp_MaskedQUBO_KaiwuSA",
            "uses_qubo_solver": True,
        },
    }

    summary = {
        "label": "Closed-Loop Simulated DFT Discovery on 03_val (20 Paired Seeds)",
        "pool_size": n_total,
        "total_rare05": n_rare05,
        "total_rare10": n_rare10,
        "budgets": budgets,
        "n_seeds": n_seeds,
        "n_init": n_init,
        "batch_size": batch_size,
        "n_rounds": n_rounds,
        "runtime_sec": round(time.perf_counter() - t0, 2),
        "analytical_hypergeometric": {
            "formula": "E[H(K)] = K * M / N",
            "rare05_mean": [round(float(x), 6) for x in hg_exp05],
            "rare05_std": [round(float(x), 6) for x in hg_std05],
            "rare10_mean": [round(float(x), 6) for x in hg_exp10],
            "rare10_std": [round(float(x), 6) for x in hg_std10],
            "final_expected_rare05": round(hg_final05, 7),
            "final_recall05_pct": round(100.0 * hg_final05 / n_rare05, 4),
            "final_expected_rare10": round(hg_final10, 7),
            "final_recall10_pct": round(100.0 * hg_final10 / n_rare10, 4),
        },
        "methods": {},
        "fm_lbfgs_convergence": {
            "total_fits": len(fm_seq_conv_logs),
            "converged_pct": round(100.0 * float(np.mean([c[0] for c in fm_seq_conv_logs])), 2),
            "mean_nit": round(float(np.mean([c[1] for c in fm_seq_conv_logs])), 2),
            "mean_train_mse": round(float(np.mean([c[2] for c in fm_seq_conv_logs])), 4),
            "mean_grad_inf_norm": round(float(np.mean([c[3] for c in fm_seq_conv_logs])), 6),
        },
        "per_round_qubo_audit": {
            "v1_FM_LBFGS_QUBO": {
                "total_rounds": len(fm_qubo_round_logs),
                "mean_spins": round(float(np.mean([d["spins"] for d in fm_qubo_round_logs])), 2),
                "mean_strict_feasibility_pct": round(float(np.mean([d["strict_feasibility_pct"] for d in fm_qubo_round_logs])), 2),
                "mean_all_shots_feasibility_pct": round(float(np.mean([d["all_shots_strict_feasibility_pct"] for d in fm_qubo_round_logs])), 2),
                "mean_unique_qubo_hits": round(float(np.mean([d["unique_qubo_hits"] for d in fm_qubo_round_logs])), 2),
                "total_fallback_fills": int(sum(d["fallback_fills"] for d in fm_qubo_round_logs)),
                "sample_seed101_logs": [d for d in fm_qubo_round_logs if d["seed"] == 101],
            },
            "v2_Real_8Bit_QUBO": {
                "total_rounds": len(v2_qubo_round_logs),
                "mean_spins": round(float(np.mean([d["spins"] for d in v2_qubo_round_logs])), 2),
                "min_spins": int(min(d["spins"] for d in v2_qubo_round_logs)),
                "max_spins": int(max(d["spins"] for d in v2_qubo_round_logs)),
                "mean_M_max": round(float(np.mean([d["M_max"] for d in v2_qubo_round_logs])), 2),
                "exact_8bit_pct": round(100.0 * float(np.mean([d["exact_8bit"] for d in v2_qubo_round_logs])), 2),
                "mean_strict_feasibility_pct": round(float(np.mean([d["strict_feasibility_pct"] for d in v2_qubo_round_logs])), 2),
                "mean_all_shots_feasibility_pct": round(float(np.mean([d["all_shots_strict_feasibility_pct"] for d in v2_qubo_round_logs])), 2),
                "mean_unique_qubo_hits": round(float(np.mean([d["unique_qubo_hits"] for d in v2_qubo_round_logs])), 2),
                "total_fallback_fills": int(sum(d["fallback_fills"] for d in v2_qubo_round_logs)),
                "sample_seed101_logs": [d for d in v2_qubo_round_logs if d["seed"] == 101],
            },
        },
    }

    for m in method_names:
        r05 = curves[m]["rare05"]
        r10 = curves[m]["rare10"]
        m05_fin = float(r05[:, -1].mean())
        m10_fin = float(r10[:, -1].mean())
        summary["methods"][m] = {
            **oracle_meta[m],
            "rare05_mean": [round(float(x), 3) for x in r05.mean(axis=0)],
            "rare05_std": [round(float(x), 3) for x in r05.std(axis=0)],
            "rare05_se": [round(float(x), 3) for x in (r05.std(axis=0) / np.sqrt(n_seeds))],
            "rare10_mean": [round(float(x), 3) for x in r10.mean(axis=0)],
            "rare10_std": [round(float(x), 3) for x in r10.std(axis=0)],
            "rare10_se": [round(float(x), 3) for x in (r10.std(axis=0) / np.sqrt(n_seeds))],
            "final_mean_rare05": round(m05_fin, 3),
            "final_mean_rare10": round(m10_fin, 3),
            "final_recall05_pct": round(100.0 * m05_fin / n_rare05, 2),
            "final_recall10_pct": round(100.0 * m10_fin / n_rare10, 2),
            "enrichment_vs_hypergeom_rare05": round(m05_fin / hg_final05, 2),
            "enrichment_vs_empirical_random_rare05": round(m05_fin / max(emp_rand_final05, 1e-9), 2),
            "enrichment_vs_hypergeom_rare10": round(m10_fin / hg_final10, 2),
            "enrichment_vs_empirical_random_rare10": round(m10_fin / max(emp_rand_final10, 1e-9), 2),
        }

    return summary


# ---------------------------------------------------------------------------
# Task 2: Fair Equal-Budget Solver Top-K Comparison (F13)
# ---------------------------------------------------------------------------

def _eval_v1_naive_q2(sols, ground_sols, energies, all52, df_pool, lookup, declared_min_v):
    df_s = df_pool.sort_values("V_hat").reset_index(drop=True)
    formula_map = {(r["A"], r["B"]): str(r["formula"]) for _, r in df_s.iterrows()}
    rank_map = {(r["A"], r["B"]): int(idx + 1) for idx, (_, r) in enumerate(df_s.iterrows())}
    top10 = set(zip(df_s.head(10)["A"], df_s.head(10)["B"]))

    nA = len(all52)
    feas_g = 0
    kA_l, kB_l = [], []
    for x in ground_sols:
        ai = np.where(x[:nA] == 1)[0]
        bi = np.where(x[nA:2 * nA] == 1)[0]
        kA_l.append(len(ai))
        kB_l.append(len(bi))
        if len(ai) == 1 and len(bi) == 1:
            ea, eb = all52[int(ai[0])], all52[int(bi[0])]
            if ea != eb and (ea, eb) in lookup:
                feas_g += 1

    uniq = {}
    hits = set()
    for x in sols:
        ai = np.where(x[:nA] == 1)[0]
        bi = np.where(x[nA:2 * nA] == 1)[0]
        if len(ai) == 1 and len(bi) == 1:
            ea, eb = all52[int(ai[0])], all52[int(bi[0])]
            if ea != eb and (ea, eb) in lookup:
                v = float(lookup[(ea, eb)])
                uniq[(ea, eb)] = {
                    "formula": formula_map[(ea, eb)],
                    "V_hat": round(v, 6),
                    "regret": round(v - declared_min_v, 6),
                    "pool_rank": rank_map[(ea, eb)],
                }
                if (ea, eb) in top10:
                    hits.add((ea, eb))

    ucands = sorted(uniq.values(), key=lambda d: (d["V_hat"], d["formula"]))
    top5 = ucands[:5]
    best_v = ucands[0]["V_hat"] if ucands else None
    return {
        "n_returned_shots": len(sols),
        "feasible_shot_count": int(feas_g),
        "primary_feasibility_pct": round(100.0 * feas_g / max(1, len(ground_sols)), 2),
        "y_aux_consistent_pct": 100.0,
        "strict_feasibility_pct": round(100.0 * feas_g / max(1, len(ground_sols)), 2),
        "unique_feasible_count": len(ucands),
        "best_candidate": ucands[0]["formula"] if ucands else None,
        "best_V_hat": best_v,
        "continuous_regret_top1": round(best_v - declared_min_v, 6) if best_v is not None else None,
        "top5_V_hat": [d["V_hat"] for d in top5],
        "top5_mean_V_hat": round(float(np.mean([d["V_hat"] for d in top5])), 6) if top5 else None,
        "top5_mean_regret": round(float(np.mean([d["regret"] for d in top5])), 6) if top5 else None,
        "topk_hits": len(hits),
        "topk_target": 10,
        "mean_kA": round(float(np.mean(kA_l)), 3) if kA_l else 0.0,
        "mean_kB": round(float(np.mean(kB_l)), 3) if kB_l else 0.0,
    }


def _eval_v1_fm107_q3(sols, ground_sols, all52, cols_X, df_pool, lookup, declared_min_v):
    df_s = df_pool.sort_values("V_hat").reset_index(drop=True)
    formula_map = {(r["A"], r["B"], r["X"]): str(r["formula"]) for _, r in df_s.iterrows()}
    rank_map = {(r["A"], r["B"], r["X"]): int(idx + 1) for idx, (_, r) in enumerate(df_s.iterrows())}
    top15 = set(zip(df_s.head(15)["A"], df_s.head(15)["B"], df_s.head(15)["X"]))

    nA, nB, nX = len(all52), len(all52), len(cols_X)
    feas_g = 0
    kA_l, kB_l, kX_l = [], [], []
    for x in ground_sols:
        ai = np.where(x[:nA] == 1)[0]
        bi = np.where(x[nA:nA + nB] == 1)[0]
        xi = np.where(x[nA + nB:nA + nB + nX] == 1)[0]
        kA_l.append(len(ai))
        kB_l.append(len(bi))
        kX_l.append(len(xi))
        if len(ai) == 1 and len(bi) == 1 and len(xi) == 1:
            ea, eb, ex = all52[int(ai[0])], all52[int(bi[0])], cols_X[int(xi[0])]
            if ea != eb and (ea, eb, ex) in lookup:
                feas_g += 1

    uniq = {}
    hits = set()
    for x in sols:
        ai = np.where(x[:nA] == 1)[0]
        bi = np.where(x[nA:nA + nB] == 1)[0]
        xi = np.where(x[nA + nB:nA + nB + nX] == 1)[0]
        if len(ai) == 1 and len(bi) == 1 and len(xi) == 1:
            ea, eb, ex = all52[int(ai[0])], all52[int(bi[0])], cols_X[int(xi[0])]
            if ea != eb and (ea, eb, ex) in lookup:
                v = float(lookup[(ea, eb, ex)])
                uniq[(ea, eb, ex)] = {
                    "formula": formula_map[(ea, eb, ex)],
                    "V_hat": round(v, 6),
                    "regret": round(v - declared_min_v, 6),
                    "pool_rank": rank_map[(ea, eb, ex)],
                }
                if (ea, eb, ex) in top15:
                    hits.add((ea, eb, ex))

    ucands = sorted(uniq.values(), key=lambda d: (d["V_hat"], d["formula"]))
    top5 = ucands[:5]
    best_v = ucands[0]["V_hat"] if ucands else None
    return {
        "n_returned_shots": len(sols),
        "feasible_shot_count": int(feas_g),
        "primary_feasibility_pct": round(100.0 * feas_g / max(1, len(ground_sols)), 2),
        "y_aux_consistent_pct": 100.0,
        "strict_feasibility_pct": round(100.0 * feas_g / max(1, len(ground_sols)), 2),
        "unique_feasible_count": len(ucands),
        "best_candidate": ucands[0]["formula"] if ucands else None,
        "best_V_hat": best_v,
        "continuous_regret_top1": round(best_v - declared_min_v, 6) if best_v is not None else None,
        "top5_V_hat": [d["V_hat"] for d in top5],
        "top5_mean_V_hat": round(float(np.mean([d["V_hat"] for d in top5])), 6) if top5 else None,
        "top5_mean_regret": round(float(np.mean([d["regret"] for d in top5])), 6) if top5 else None,
        "topk_hits": len(hits),
        "topk_target": 15,
        "mean_kA": round(float(np.mean(kA_l)), 3) if kA_l else 0.0,
        "mean_kB": round(float(np.mean(kB_l)), 3) if kB_l else 0.0,
        "mean_kX": round(float(np.mean(kX_l)), 3) if kX_l else 0.0,
    }


def run_equal_budget_solver_topk(n_runs=50):
    """
    Fair Equal-Budget Top-K Solver Comparison (F13):
      - Fixed budget: n_runs=50, 1 best spin configuration per run -> 50 returned shots.
      - Frozen surrogate V_hat and identical P_declared evaluation pools.
      - Both SA and Tabu across:
          Q2 (Full 07 & Blind 11): v1_Naive104_Float, v1_Naive104_Naive8bit, v2_Q2_Exact8bit
          Q3 (Full 08 & Blind 12): v1_FM107_Float, v1_FM107_Naive8bit, v2_Model3A_Comp,
                                   v2_Model3A_HOBO, v2_Model3B_CompactANOVA
    """
    t0 = time.perf_counter()
    tr = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    all52 = sorted(tr["A"].unique())
    e2i = {e: i for i, e in enumerate(all52)}
    rows = []

    # Q2 Instances
    for label, fname, c_cap in [
        ("Q2_full_07", "v2_q2_full_07.csv", 2.0),
        ("Q2_blind_11", "v2_q2_blind_11.csv", 2.5),
    ]:
        dfp = pd.read_csv(os.path.join(OUT_DIR, fname))
        declared_min_v = float(dfp["V_hat"].min())
        M = np.full((52, 52), 50.0, dtype=float)
        lkp_q2 = {}
        for a, b, v in zip(dfp["A"], dfp["B"], dfp["V_hat"]):
            M[e2i[a], e2i[b]] = float(v)
            lkp_q2[(a, b)] = float(v)
        Q_v1 = _build_naive_q2_qubo(M, all52, all52, penalty=20.0)
        ising_v1_float, _ = kw.qubo_matrix_to_ising_matrix(Q_v1)
        ising_v1_8bit = np.round(kw.adjust_ising_matrix_precision(ising_v1_float, bit_width=8)).astype(np.int64)

        for enc_name, ising_mat in [
            ("v1_Naive104_Float", ising_v1_float),
            ("v1_Naive104_Naive8bit", ising_v1_8bit),
        ]:
            pack = _solve_ising_multi_solver(ising_mat, n_sa=n_runs, n_tabu=n_runs, seed_base=4000)
            for s_name, sols, rt in [
                ("SA", pack["sa_sols"], pack["sa_runtime_sec"]),
                ("Tabu", pack["tabu_sols"], pack["tabu_runtime_sec"]),
            ]:
                ev = _eval_v1_naive_q2(sols, sols, None, all52, dfp, lkp_q2, declared_min_v)
                rows.append({
                    "instance": label,
                    "problem": "Q2",
                    "encoding": enc_name,
                    "solver": s_name,
                    "n_runs": n_runs,
                    "spins": pack["spins"],
                    "aux_vars": 0,
                    "bit_width": pack["bit_width"],
                    "dynamic_range": pack["dynamic_range"],
                    **ev,
                    "runtime_sec": rt,
                })

        ising_v2, meta_v2, lkp_v2 = build_exact8_q2_qubo(dfp, c_cap=c_cap, penalty_mult=2.25)
        cert_v2 = certify_global_ground_states(ising_v2, meta_v2, dfp, lkp_v2, task_kind="q2", verify_zero_illegal=True)
        pack_v2 = _solve_ising_multi_solver(ising_v2, n_sa=n_runs, n_tabu=n_runs, seed_base=1000)
        for s_name, sols, en, rt in [
            ("SA", pack_v2["sa_sols"], pack_v2["sa_energies"], pack_v2["sa_runtime_sec"]),
            ("Tabu", pack_v2["tabu_sols"], pack_v2["tabu_energies"], pack_v2["tabu_runtime_sec"]),
        ]:
            ev = evaluate_q2_solutions(sols, meta_v2, dfp, lkp_v2, ground_sols=sols, energies=en, cert=cert_v2)
            rows.append({
                "instance": label,
                "problem": "Q2",
                "encoding": "v2_Q2_Exact8bit",
                "solver": s_name,
                "n_runs": n_runs,
                "spins": pack_v2["spins"],
                "aux_vars": 0,
                "bit_width": pack_v2["bit_width"],
                "dynamic_range": pack_v2["dynamic_range"],
                "n_returned_shots": ev["n_returned_shots"],
                "feasible_shot_count": ev["feasible_shot_count"],
                "primary_feasibility_pct": ev["primary_feasibility_pct"],
                "y_aux_consistent_pct": ev["y_aux_consistent_pct"],
                "strict_feasibility_pct": ev["feasibility_pct"],
                "unique_feasible_count": ev["unique_feasible_count"],
                "best_candidate": ev["best_pair"],
                "best_V_hat": ev["best_V_hat"],
                "continuous_regret_top1": ev["continuous_regret"],
                "top5_V_hat": ev["top5_V_hat"],
                "top5_mean_V_hat": ev["top5_mean_V_hat"],
                "top5_mean_regret": ev["top5_mean_regret"],
                "topk_hits": ev["top10_hits"],
                "topk_target": 10,
                "runtime_sec": rt,
            })

    # Q3 Instances
    cols_X_q3 = ["FO2", "NO2", "O3"]
    for label, fname, c_cap, tk in [
        ("Q3_full_08", "v2_q3_full_08.csv", 1.25, 35),
        ("Q3_blind_12", "v2_q3_blind_12.csv", 1.50, 25),
    ]:
        dfp = pd.read_csv(os.path.join(OUT_DIR, fname))
        declared_min_v = float(dfp["V_hat"].min())
        lkp_q3 = {(a, b, x): float(v) for a, b, x, v in zip(dfp["A"], dfp["B"], dfp["X"], dfp["V_hat"])}
        valid_ab = set(zip(dfp["A"], dfp["B"]))

        # v1 Pairwise FM-107 (converged L-BFGS-B FM fit on frozen V_hat)
        fm_q3 = fit_onehot_fm_lbfgs(
            dfp, dfp["V_hat"].values, all52, all52, cols_X_q3,
            k=8, reg=1e-3, maxiter=300, seed=2026
        )
        Q_fm107, _ = build_fm_q3_qubo(fm_q3, valid_ab, penalty=20.0)
        ising_fm_float, _ = kw.qubo_matrix_to_ising_matrix(Q_fm107)
        ising_fm_8bit = np.round(kw.adjust_ising_matrix_precision(ising_fm_float, bit_width=8)).astype(np.int64)

        for enc_name, ising_mat in [
            ("v1_PairwiseFM107_Float", ising_fm_float),
            ("v1_PairwiseFM107_Naive8bit", ising_fm_8bit),
        ]:
            pack = _solve_ising_multi_solver(ising_mat, n_sa=n_runs, n_tabu=n_runs, seed_base=4500)
            for s_name, sols, rt in [
                ("SA", pack["sa_sols"], pack["sa_runtime_sec"]),
                ("Tabu", pack["tabu_sols"], pack["tabu_runtime_sec"]),
            ]:
                ev = _eval_v1_fm107_q3(sols, sols, all52, cols_X_q3, dfp, lkp_q3, declared_min_v)
                rows.append({
                    "instance": label,
                    "problem": "Q3",
                    "encoding": enc_name,
                    "solver": s_name,
                    "n_runs": n_runs,
                    "spins": pack["spins"],
                    "aux_vars": 0,
                    "bit_width": pack["bit_width"],
                    "dynamic_range": pack["dynamic_range"],
                    **ev,
                    "runtime_sec": rt,
                })

        # v2 Q3 Models: 3A-Comp, 3A-HOBO, 3B-ANOVA
        ising_comp, meta_comp, lkp_comp = build_q3_comp_qubo(dfp, c_cap=c_cap, penalty_mult=1.60)
        cert_comp = certify_global_ground_states(ising_comp, meta_comp, dfp, lkp_comp, task_kind="q3", verify_zero_illegal=True)

        ising_hobo, meta_hobo, lkp_hobo = build_q3_submodular_hobo_qubo(dfp, c_cap=c_cap)
        cert_hobo = certify_global_ground_states(ising_hobo, meta_hobo, dfp, lkp_hobo, task_kind="q3", verify_zero_illegal=True)

        ising_anov, meta_anov, lkp_anov = build_q3_compact_anova_qubo(dfp, top_k_subspace=tk, c_cap=c_cap)
        cert_anov = certify_global_ground_states(ising_anov, meta_anov, dfp, lkp_anov, task_kind="q3", verify_zero_illegal=True)

        for enc_name, ising_m, meta_m, lkp_m, cert_m, sb in [
            ("v2_Q3_Model3A_Comp", ising_comp, meta_comp, lkp_comp, cert_comp, 1500),
            ("v2_Q3_Model3A_HOBO", ising_hobo, meta_hobo, lkp_hobo, cert_hobo, 2000),
            ("v2_Q3_Model3B_ANOVA", ising_anov, meta_anov, lkp_anov, cert_anov, 3000),
        ]:
            pack = _solve_ising_multi_solver(ising_m, n_sa=n_runs, n_tabu=n_runs, seed_base=sb)
            for s_name, sols, en, rt in [
                ("SA", pack["sa_sols"], pack["sa_energies"], pack["sa_runtime_sec"]),
                ("Tabu", pack["tabu_sols"], pack["tabu_energies"], pack["tabu_runtime_sec"]),
            ]:
                ev = evaluate_q3_solutions(sols, meta_m, dfp, lkp_m, ground_sols=sols, energies=en, cert=cert_m)
                rows.append({
                    "instance": label,
                    "problem": "Q3",
                    "encoding": enc_name,
                    "solver": s_name,
                    "n_runs": n_runs,
                    "spins": pack["spins"],
                    "aux_vars": int(meta_m.get("nY", 0)),
                    "bit_width": pack["bit_width"],
                    "dynamic_range": pack["dynamic_range"],
                    "n_returned_shots": ev["n_returned_shots"],
                    "feasible_shot_count": ev["feasible_shot_count"],
                    "primary_feasibility_pct": ev["primary_feasibility_pct"],
                    "y_aux_consistent_pct": ev["y_aux_consistent_pct"],
                    "strict_feasibility_pct": ev["strict_feasibility_pct"],
                    "unique_feasible_count": ev["unique_feasible_count"],
                    "best_candidate": ev["best_candidate"],
                    "best_V_hat": ev["best_V_hat"],
                    "continuous_regret_top1": ev["continuous_regret"],
                    "top5_V_hat": ev["top5_V_hat"],
                    "top5_mean_V_hat": ev["top5_mean_V_hat"],
                    "top5_mean_regret": ev["top5_mean_regret"],
                    "topk_hits": ev["top15_hits"],
                    "topk_target": 15,
                    "runtime_sec": rt,
                })

    return {
        "protocol": f"Equal-budget n_runs={n_runs}, 1 best shot/run (50 returned shots) on identical frozen V_hat",
        "runtime_sec": round(time.perf_counter() - t0, 2),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Task 3: Independent 4-12-Bit Quantization Sweep (F16)
# ---------------------------------------------------------------------------

def _build_q2_at_bitwidth(df_pool, bw, c_cap=2.5, penalty_mult=2.25):
    """
    Independently search the maximal usable integer scale M_max(B) bounded by
    M_B = 2^(B-1) - 1 for bit-width B (NEVER reusing the 8-bit matrix for B >= 8).
    """
    M_B = (1 << (bw - 1)) - 1
    active = df_pool[df_pool["V_hat"] < c_cap].copy()
    elems_A = sorted(active["A"].unique())
    elems_B = sorted(active["B"].unique())
    nA, nB = len(elems_A), len(elems_B)
    a2i = {e: i for i, e in enumerate(elems_A)}
    b2i = {e: i for i, e in enumerate(elems_B)}

    M_bar = np.zeros((nA, nB), dtype=float)
    lookup = {}
    for a, b, v in zip(df_pool["A"], df_pool["B"], df_pool["V_hat"]):
        lookup[(a, b)] = float(v)
        if a in a2i and b in b2i and v < c_cap:
            M_bar[a2i[a], b2i[b]] = v - c_cap

    best_ising = None
    best_meta = None
    ising_M1 = None

    def _build_for_M(M_max):
        m_int = np.round((M_bar / c_cap) * M_max).astype(int)
        j_pen = int(np.ceil(2.0 * penalty_mult * M_max))
        P_block = 4.0 * j_pen
        P_forbid = 8.0 * int(np.ceil(j_pen / 2.0))
        N = nA + nB
        Q = np.zeros((N, N), dtype=float)
        for i in range(nA):
            for j in range(nB):
                if m_int[i, j] != 0:
                    Q[i, nA + j] += 8.0 * m_int[i, j]
        for start, sz in [(0, nA), (nA, nB)]:
            for i in range(start, start + sz):
                Q[i, i] -= P_block
                for j in range(i + 1, start + sz):
                    Q[i, j] += 2.0 * P_block
        for i, ea in enumerate(elems_A):
            for j, eb in enumerate(elems_B):
                if ea == eb or (ea, eb) not in lookup:
                    Q[i, nA + j] += P_forbid
        Q_sym = 0.5 * (Q + Q.T)
        ising_raw, _ = kw.qubo_matrix_to_ising_matrix(Q_sym)
        ising_rnd = np.round(ising_raw)
        max_err = float(np.max(np.abs(ising_raw - ising_rnd)))
        max_abs = int(np.max(np.abs(ising_rnd)))
        return ising_rnd.astype(np.int64), max_err, max_abs, m_int, j_pen, P_block

    # Direct descending search for exact maximal M_max(B) <= M_B (upper-bounded safely by min(M_B, 250))
    ising_1, _, _, _, jp1, pb1 = _build_for_M(1)
    ising_M1 = ising_1
    for cand_m in range(min(M_B, 250), 0, -1):
        is_c, err_c, abs_c, m_c, jp_c, pb_c = _build_for_M(cand_m)
        if err_c < 1e-6 and abs_c <= M_B:
            best_ising = is_c
            best_meta = {
                "formulation": "q2_bipartite",
                "nA": nA, "nB": nB, "nY": 0, "N": nA + nB,
                "c_cap": float(c_cap),
                "M_max": int(cand_m),
                "M_B": int(M_B),
                "quant_mode": "exact_integer_M_max",
                "j_pen": int(jp_c),
                "P_block": float(pb_c),
                "max_abs_ising": int(abs_c),
                "elems_A": elems_A,
                "elems_B": elems_B,
                "m_int": m_c,
            }
            break

    if best_ising is None:
        # At narrow bitwidths (e.g. 4..6 bits), even M_max=1 exceeds M_B, so rescale precision
        ising_scaled = np.round(kw.adjust_ising_matrix_precision(ising_M1.astype(float), bit_width=bw)).astype(np.int64)
        best_ising = ising_scaled
        best_meta = {
            "formulation": "q2_bipartite",
            "nA": nA, "nB": nB, "nY": 0, "N": nA + nB,
            "c_cap": float(c_cap),
            "M_max": None,
            "M_B": int(M_B),
            "quant_mode": "kaiwu_rescaled_fallback",
            "j_pen": int(jp1),
            "P_block": float(pb1),
            "max_abs_ising": int(np.max(np.abs(best_ising))),
            "elems_A": elems_A,
            "elems_B": elems_B,
            "m_int": None,
        }

    return best_ising, best_meta, lookup


def run_bitwidth_sensitivity(n_runs=50):
    """
    Independent 4-12-bit quantization sweep (F16) on Q2 Blind (11) vs v1 Naive 104-var QUBO.
    Each bitwidth B in [4, 5, 6, 7, 8, 10, 12] independently finds M_max(B) <= 2^(B-1)-1.
    """
    dfp = pd.read_csv(os.path.join(OUT_DIR, "v2_q2_blind_11.csv"))
    declared_min_v = float(dfp["V_hat"].min())
    tr = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    all52 = sorted(tr["A"].unique())
    e2i = {e: i for i, e in enumerate(all52)}
    M = np.full((52, 52), 50.0, dtype=float)
    lkp_q2 = {}
    for a, b, v in zip(dfp["A"], dfp["B"], dfp["V_hat"]):
        M[e2i[a], e2i[b]] = float(v)
        lkp_q2[(a, b)] = float(v)
    Q_v1 = _build_naive_q2_qubo(M, all52, all52, penalty=20.0)
    ising_v1_raw, _ = kw.qubo_matrix_to_ising_matrix(Q_v1)

    bit_widths = [4, 5, 6, 7, 8, 10, 12]
    rows = []
    for bw in bit_widths:
        # 1. v1 Naive at bit_width=bw (n_runs=50, 1 best/run)
        ising_v1_bw = np.round(kw.adjust_ising_matrix_precision(ising_v1_raw, bit_width=bw)).astype(np.int64)
        sols_v1 = []
        for r in range(n_runs):
            opt = kw.SimulatedAnnealingOptimizer(
                initial_temperature=120.0, alpha=0.985, cutoff_temperature=0.01,
                iterations_per_t=25, size_limit=15, rand_seed=400 + bw * 100 + r
            )
            out = opt.solve(ising_v1_bw)
            if out is not None and len(out) > 0:
                row = sort_ising_solutions(out, ising_v1_bw)[0]
                s_aux = row[-1]
                sols_v1.append((((row[:-1] * s_aux) + 1) // 2).astype(int))
        ev1 = _eval_v1_naive_q2(sols_v1, sols_v1, None, all52, dfp, lkp_q2, declared_min_v)

        # 2. v2 Active-Subspace independently built for bit_width=bw
        ising_v2, meta_v2, lkp_v2 = _build_q2_at_bitwidth(dfp, bw, c_cap=2.5, penalty_mult=2.25)
        cert_v2 = certify_global_ground_states(
            ising_v2, meta_v2, dfp, lkp_v2, task_kind="q2", verify_zero_illegal=False
        )
        # Count distinct integer energy levels across legal active pool
        if meta_v2["m_int"] is not None:
            eA, eB = meta_v2["elems_A"], meta_v2["elems_B"]
            distinct_levels = len({
                int(meta_v2["m_int"][i, j])
                for i, ea in enumerate(eA)
                for j, eb in enumerate(eB)
                if ea != eb and (ea, eb) in lkp_v2
            })
        else:
            distinct_levels = None

        sols_v2, energies_v2 = [], []
        for r in range(n_runs):
            opt = kw.SimulatedAnnealingOptimizer(
                initial_temperature=80.0, alpha=0.985, cutoff_temperature=0.01,
                iterations_per_t=25, size_limit=15, rand_seed=600 + bw * 100 + r
            )
            out = opt.solve(ising_v2)
            if out is not None and len(out) > 0:
                row = sort_ising_solutions(out, ising_v2)[0]
                energies_v2.append(float(-row @ ising_v2 @ row))
                s_aux = row[-1]
                sols_v2.append((((row[:-1] * s_aux) + 1) // 2).astype(int))
        ev2 = evaluate_q2_solutions(
            sols_v2, meta_v2, dfp, lkp_v2, ground_sols=sols_v2, energies=energies_v2, cert=cert_v2
        )
        sa_gs_hits = sum(1 for e in energies_v2 if round(e) == cert_v2["cert_min_ising_energy"])

        rows.append({
            "bit_width": int(bw),
            "M_B": int(meta_v2["M_B"]),
            "v1_actual_bw": safe_ising_bit_width(ising_v1_bw),
            "v1_feas_pct": ev1["strict_feasibility_pct"],
            "v1_mean_kA": ev1["mean_kA"],
            "v1_top10_hits": ev1["topk_hits"],
            "v1_best_V_hat": ev1["best_V_hat"],
            "v2_M_max": meta_v2["M_max"],
            "v2_quant_mode": meta_v2["quant_mode"],
            "v2_actual_bw": safe_ising_bit_width(ising_v2),
            "v2_max_abs": int(meta_v2["max_abs_ising"]),
            "v2_distinct_levels": distinct_levels,
            "v2_cert_min_ising_energy": cert_v2["cert_min_ising_energy"],
            "v2_legal_ground_count": cert_v2["legal_ground_count"],
            "v2_illegal_ground_count": cert_v2["illegal_ground_count"],
            "v2_ground_state_degeneracy": cert_v2["ground_state_degeneracy"],
            "v2_cert_regret_top1": cert_v2["continuous_regret_top1"],
            "v2_cert_regret_worst_gs": cert_v2["continuous_regret_worst_gs"],
            "v2_cert_regret_mean_gs": cert_v2["continuous_regret_mean_gs"],
            "v2_feas_pct": ev2["feasibility_pct"],
            "v2_mean_kA": ev2["mean_kA"],
            "v2_sa_unique_feasible": ev2["unique_feasible_count"],
            "v2_sa_best_V_hat": ev2["best_V_hat"],
            "v2_sa_regret": ev2["continuous_regret"],
            "v2_gap_to_opt": ev2["gap_to_opt"],
            "v2_sa_gs_hit_pct": round(100.0 * sa_gs_hits / max(1, len(energies_v2)), 2),
            "v2_top10_hits": ev2["top10_hits"],
        })
    return rows


# ---------------------------------------------------------------------------
# Task 4: Refined Fixed-M_max Penalty Sweep & C_cap Pareto Sweep (F16)
# ---------------------------------------------------------------------------

def run_penalty_and_cap_sensitivity(n_runs=50):
    """
    1. Fixed M_max=4 Penalty Sweep on Q2 Blind (12 spins) across
       alpha_P in [0.50, 1.00, 1.25, 1.40, 1.45, 1.50, 1.55, 1.75, 2.25, 3.00],
       computing BOTH exact global enumeration (2^11 = 2048 states) AND SA empirical rate.
    2. Upper-Tail Cap C_cap Pareto Sweep on Q2 Full (07) and Q3 Blind (12)
       (equal-budget n_runs=50, 1 best/run, covering both Model 3A-Comp & 3A-HOBO).
    """
    q2b = pd.read_csv(os.path.join(OUT_DIR, "v2_q2_blind_11.csv"))
    q3b = pd.read_csv(os.path.join(OUT_DIR, "v2_q3_blind_12.csv"))

    # 1. Fixed M_max=4 Penalty Sweep on Q2 Blind (12 spins)
    c_cap = 2.5
    M_max_fixed = 4
    active = q2b[q2b["V_hat"] < c_cap].copy()
    elems_A = sorted(active["A"].unique())
    elems_B = sorted(active["B"].unique())
    nA, nB = len(elems_A), len(elems_B)
    a2i = {e: i for i, e in enumerate(elems_A)}
    b2i = {e: i for i, e in enumerate(elems_B)}
    lkp_q2b = {(a, b): float(v) for a, b, v in zip(q2b["A"], q2b["B"], q2b["V_hat"])}

    M_bar = np.zeros((nA, nB), dtype=float)
    for a, b, v in zip(active["A"], active["B"], active["V_hat"]):
        M_bar[a2i[a], b2i[b]] = float(v) - c_cap
    m_int = np.round((M_bar / c_cap) * M_max_fixed).astype(int)
    max_q_obj = 8.0 * M_max_fixed  # 32.0

    # Precompute all 2^11 = 2048 primary states for exact global enumeration
    n_prim = nA + nB
    masks = np.arange(1 << n_prim, dtype=np.uint32)
    Z_all = ((masks[:, None] >> np.arange(n_prim, dtype=np.uint32)) & 1).astype(np.float64)
    kA_all = Z_all[:, :nA].sum(axis=1)
    kB_all = Z_all[:, nA:].sum(axis=1)
    legal_mask = np.zeros(len(Z_all), dtype=bool)
    for r in range(len(Z_all)):
        if kA_all[r] == 1 and kB_all[r] == 1:
            ea = elems_A[int(np.argmax(Z_all[r, :nA]))]
            eb = elems_B[int(np.argmax(Z_all[r, nA:]))]
            if ea != eb and (ea, eb) in lkp_q2b:
                legal_mask[r] = True

    Q_obj = np.zeros((n_prim, n_prim), dtype=np.float64)
    for i in range(nA):
        for j in range(nB):
            if m_int[i, j] != 0:
                Q_obj[i, nA + j] += 8.0 * m_int[i, j]
    Q_obj = 0.5 * (Q_obj + Q_obj.T)

    Q_pen_unit = np.zeros((n_prim, n_prim), dtype=np.float64)
    for start, sz in [(0, nA), (nA, nB)]:
        for i in range(start, start + sz):
            Q_pen_unit[i, i] -= 1.0
            for j in range(i + 1, start + sz):
                Q_pen_unit[i, j] += 2.0
    for i, ea in enumerate(elems_A):
        for j, eb in enumerate(elems_B):
            if ea == eb or (ea, eb) not in lkp_q2b:
                Q_pen_unit[i, nA + j] += 1.0
    Q_pen_unit = 0.5 * (Q_pen_unit + Q_pen_unit.T)

    e_obj_all = np.einsum("ij,jk,ik->i", Z_all, Q_obj, Z_all)
    e_pen_all = np.einsum("ij,jk,ik->i", Z_all, Q_pen_unit, Z_all)

    meta_fixed = {
        "formulation": "q2_bipartite",
        "nA": nA, "nB": nB, "nY": 0, "N": n_prim,
        "c_cap": c_cap, "M_max": M_max_fixed,
        "elems_A": elems_A, "elems_B": elems_B,
    }

    alpha_grid = [0.50, 1.00, 1.25, 1.40, 1.45, 1.50, 1.55, 1.75, 2.25, 3.00]
    pen_rows = []
    for alpha_p in alpha_grid:
        P_block = float(alpha_p * max_q_obj)  # 32.0 * alpha_P (fixed M_max=4)
        e_tot = e_obj_all + P_block * e_pen_all
        min_e = float(np.min(e_tot))
        gs_mask = np.abs(e_tot - min_e) <= 1e-9
        leg_gs = int(np.sum(gs_mask & legal_mask))
        ill_gs = int(np.sum(gs_mask & (~legal_mask)))
        tot_gs = leg_gs + ill_gs
        leg_min = float(np.min(e_tot[legal_mask]))
        ill_min = float(np.min(e_tot[~legal_mask]))

        Q_tot = Q_obj + P_block * Q_pen_unit
        ising, _ = kw.qubo_matrix_to_ising_matrix(Q_tot)
        max_abs = float(np.max(np.abs(ising)))
        bw = safe_ising_bit_width(ising)

        sols, energies = [], []
        for r in range(n_runs):
            opt = kw.SimulatedAnnealingOptimizer(
                initial_temperature=80.0, alpha=0.985, cutoff_temperature=0.01,
                iterations_per_t=25, size_limit=15, rand_seed=900 + int(round(alpha_p * 1000)) + r
            )
            out = opt.solve(ising)
            if out is not None and len(out) > 0:
                row = sort_ising_solutions(out, ising)[0]
                energies.append(float(-row @ ising @ row))
                s_aux = row[-1]
                sols.append((((row[:-1] * s_aux) + 1) // 2).astype(int))
        ev = evaluate_q2_solutions(sols, meta_fixed, q2b, lkp_q2b, ground_sols=sols, energies=energies)

        pen_rows.append({
            "alpha_P": float(alpha_p),
            "M_max": M_max_fixed,
            "P_block": round(P_block, 4),
            "spins": n_prim + 1,
            "bit_width": bw,
            "max_abs_ising": round(max_abs, 2),
            "exact_total_states": int(1 << n_prim),
            "exact_legal_ground_count": leg_gs,
            "exact_illegal_ground_count": ill_gs,
            "exact_gs_feasibility_pct": round(100.0 * leg_gs / max(1, tot_gs), 2),
            "exact_zero_illegal_gs": bool(ill_gs == 0),
            "exact_legal_min_qubo_energy": round(leg_min, 4),
            "exact_illegal_min_qubo_energy": round(ill_min, 4),
            "exact_energy_margin": round(ill_min - leg_min, 4),
            "sa_n_runs": n_runs,
            "feasibility_pct": ev["feasibility_pct"],
            "sa_unique_feasible_count": ev["unique_feasible_count"],
            "sa_best_V_hat": ev["best_V_hat"],
            "sa_regret": ev["continuous_regret"],
            "gap_to_opt": ev["gap_to_opt"],
            "top10_hits": ev["top10_hits"],
        })

    # 2. Cap C_cap Pareto Sweep on Q2 Full (07) and Q3 Blind (12)
    q2f = pd.read_csv(os.path.join(OUT_DIR, "v2_q2_full_07.csv"))
    cap_rows = []
    for c_cap_v in [0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]:
        ising2, meta2, lkp2 = build_exact8_q2_qubo(q2f, c_cap=c_cap_v, penalty_mult=2.25)
        sols2, en2 = [], []
        for r in range(n_runs):
            opt = kw.SimulatedAnnealingOptimizer(
                initial_temperature=80.0, alpha=0.985, cutoff_temperature=0.01,
                iterations_per_t=25, size_limit=15, rand_seed=1200 + int(c_cap_v * 100) + r
            )
            out = opt.solve(ising2)
            if out is not None and len(out) > 0:
                row = sort_ising_solutions(out, ising2)[0]
                en2.append(float(-row @ ising2 @ row))
                sols2.append((((row[:-1] * row[-1]) + 1) // 2).astype(int))
        ev2 = evaluate_q2_solutions(sols2, meta2, q2f, lkp2, ground_sols=sols2, energies=en2)

        ising3h, meta3h, lkp3h = build_q3_submodular_hobo_qubo(q3b, c_cap=c_cap_v)
        sols3h, en3h = [], []
        for r in range(n_runs):
            opt = kw.SimulatedAnnealingOptimizer(
                initial_temperature=80.0, alpha=0.985, cutoff_temperature=0.01,
                iterations_per_t=25, size_limit=15, rand_seed=1400 + int(c_cap_v * 100) + r
            )
            out = opt.solve(ising3h)
            if out is not None and len(out) > 0:
                row = sort_ising_solutions(out, ising3h)[0]
                en3h.append(float(-row @ ising3h @ row))
                sols3h.append((((row[:-1] * row[-1]) + 1) // 2).astype(int))
        ev3h = evaluate_q3_solutions(sols3h, meta3h, q3b, lkp3h, ground_sols=sols3h, energies=en3h)

        ising3c, meta3c, lkp3c = build_q3_comp_qubo(q3b, c_cap=c_cap_v, penalty_mult=1.60)
        sols3c, en3c = [], []
        for r in range(n_runs):
            opt = kw.SimulatedAnnealingOptimizer(
                initial_temperature=80.0, alpha=0.985, cutoff_temperature=0.01,
                iterations_per_t=25, size_limit=15, rand_seed=1600 + int(c_cap_v * 100) + r
            )
            out = opt.solve(ising3c)
            if out is not None and len(out) > 0:
                row = sort_ising_solutions(out, ising3c)[0]
                en3c.append(float(-row @ ising3c @ row))
                sols3c.append((((row[:-1] * row[-1]) + 1) // 2).astype(int))
        ev3c = evaluate_q3_solutions(sols3c, meta3c, q3b, lkp3c, ground_sols=sols3c, energies=en3c)

        cap_rows.append({
            "C_cap": float(c_cap_v),
            "q2_spins": int(ising2.shape[0]),
            "q2_M_max": meta2.get("M_max"),
            "q2_bw": safe_ising_bit_width(ising2),
            "q2_feas_pct": ev2["feasibility_pct"],
            "q2_top10_hits": ev2["top10_hits"],
            "q2_regret": ev2["continuous_regret"],
            "q3_spins": int(ising3h.shape[0]),
            "q3_aux": int(meta3h["nY"]),
            "q3_M_max": meta3h.get("M_max"),
            "q3_bw": safe_ising_bit_width(ising3h),
            "q3_feas_pct": ev3h["strict_feasibility_pct"],
            "q3_y_aux_pct": ev3h["y_aux_consistent_pct"],
            "q3_top15_hits": ev3h["top15_hits"],
            "q3_regret": ev3h["continuous_regret"],
            "q3_comp_spins": int(ising3c.shape[0]),
            "q3_comp_M_max": meta3c.get("M_max"),
            "q3_comp_bw": safe_ising_bit_width(ising3c),
            "q3_comp_feas_pct": ev3c["strict_feasibility_pct"],
            "q3_comp_top15_hits": ev3c["top15_hits"],
            "q3_comp_regret": ev3c["continuous_regret"],
        })

    return pen_rows, cap_rows


# ---------------------------------------------------------------------------
# Task 5: Multi-Scale Surrogate Ranking Perturbation Stability (F17)
# ---------------------------------------------------------------------------

def _eval_mc_perturbation(df, scale, mode, n_trials, rng):
    """
    Run n_trials Monte Carlo ranking perturbations at given scale and noise mode
    ('independent_gaussian' vs 'group_correlated_logit') around frozen surrogate.
    """
    base_v = df["V_hat"].values
    base_top10 = set(df.head(10)["formula"].values)
    base_top15 = set(df.head(15)["formula"].values)
    base_top1 = df.iloc[0]["formula"]

    mu_ef = df["mu_ef"].values
    s_ef = df["sigma_ef"].values
    p_semi = np.clip(df["p_semi"].values, 1e-4, 1.0 - 1e-4)
    logit_p = np.log(p_semi / (1.0 - p_semi))
    mu_g = df["mu_gap"].values
    s_g = df["sigma_gap"].values
    ec = df["ec_exact"].values
    formulas = df["formula"].values
    n = len(df)

    # Group indices for (A,B)-pair and A-site correlations
    ab_keys = list(zip(df["A"], df["B"]))
    uniq_ab = {k: i for i, k in enumerate(sorted(set(ab_keys)))}
    ab_idx = np.array([uniq_ab[k] for k in ab_keys], dtype=int)
    n_ab = len(uniq_ab)

    uniq_a = {k: i for i, k in enumerate(sorted(df["A"].unique()))}
    a_idx = np.array([uniq_a[a] for a in df["A"]], dtype=int)
    n_a = len(uniq_a)

    ret10, ret15 = [], []
    top1_exact = 0
    top1_in_top10 = 0
    top1_in_top15 = 0
    rhos = []
    appear10 = {f: 0 for f in df.head(10)["formula"].values}
    appear15 = {f: 0 for f in df.head(15)["formula"].values}

    vm_gl, vm_gh, vm_v, vm_c = v_band_terms(np.zeros_like(ec), ec)
    vm_sum = vm_gl + vm_gh + vm_v + vm_c

    for _ in range(n_trials):
        if mode == "independent_gaussian":
            z_ef = rng.normal(0.0, 1.0, size=n)
            z_g = rng.normal(0.0, 1.0, size=n)
            p_sim = p_semi
        else:
            # (A,B)-group + A-site correlated physical shocks + p_semi logit perturbation
            z_ef = (
                np.sqrt(0.50) * rng.normal(0.0, 1.0, size=n_ab)[ab_idx]
                + np.sqrt(0.25) * rng.normal(0.0, 1.0, size=n_a)[a_idx]
                + np.sqrt(0.25) * rng.normal(0.0, 1.0, size=n)
            )
            z_g = (
                np.sqrt(0.50) * rng.normal(0.0, 1.0, size=n_ab)[ab_idx]
                + np.sqrt(0.25) * rng.normal(0.0, 1.0, size=n_a)[a_idx]
                + np.sqrt(0.25) * rng.normal(0.0, 1.0, size=n)
            )
            z_log = (
                np.sqrt(0.50) * rng.normal(0.0, 1.0, size=n_ab)[ab_idx]
                + np.sqrt(0.50) * rng.normal(0.0, 1.0, size=n)
            )
            logit_sim = logit_p + (0.5 * scale) * z_log
            p_sim = 1.0 / (1.0 + np.exp(-np.clip(logit_sim, -12.0, 12.0)))

        ef_p = mu_ef + (scale * s_ef) * z_ef
        g_p = np.maximum(0.05, mu_g + (scale * s_g) * z_g)

        v_f = exact_expected_hinge_sq(ef_p, s_ef, 0.21, +1.0, 0.5)
        vs_gl = exact_expected_hinge_sq(g_p, s_g, 1.4, -1.0, 0.5)
        vs_gh = exact_expected_hinge_sq(g_p, s_g, 3.1, +1.0, 0.5)
        vs_v = exact_expected_hinge_sq(ec + 0.5 * g_p, 0.5 * s_g, 5.73, -1.0, 0.5)
        vs_c = exact_expected_hinge_sq(ec - 0.5 * g_p, 0.5 * s_g, 4.50, +1.0, 0.5)
        v_sim = v_f + (1.0 - p_sim) * vm_sum + p_sim * (vs_gl + vs_gh + vs_v + vs_c)

        order = np.argsort(v_sim)
        sim_top10 = set(formulas[order[:10]])
        sim_top15 = set(formulas[order[:15]])
        ret10.append(len(base_top10.intersection(sim_top10)) / 10.0)
        ret15.append(len(base_top15.intersection(sim_top15)) / 15.0)
        if formulas[order[0]] == base_top1:
            top1_exact += 1
        if base_top1 in sim_top10:
            top1_in_top10 += 1
        if base_top1 in sim_top15:
            top1_in_top15 += 1
        for f in appear10:
            if f in sim_top10:
                appear10[f] += 1
        for f in appear15:
            if f in sim_top15:
                appear15[f] += 1
        rhos.append(float(spearmanr(base_v, v_sim).statistic))

    return {
        "scale_sigma": float(scale),
        "noise_mode": mode,
        "n_trials": int(n_trials),
        "top1_exact_pct": round(100.0 * top1_exact / n_trials, 2),
        "top1_in_top10_pct": round(100.0 * top1_in_top10 / n_trials, 2),
        "top1_in_top15_pct": round(100.0 * top1_in_top15 / n_trials, 2),
        "top10_retention_mean_pct": round(100.0 * float(np.mean(ret10)), 2),
        "top10_retention_std_pct": round(100.0 * float(np.std(ret10)), 2),
        "top15_retention_mean_pct": round(100.0 * float(np.mean(ret15)), 2),
        "top15_retention_std_pct": round(100.0 * float(np.std(ret15)), 2),
        "mean_spearman": round(float(np.mean(rhos)), 4),
        "per_candidate_top10_inclusion_pct": {k: round(100.0 * v / n_trials, 1) for k, v in appear10.items()},
        "per_candidate_top15_inclusion_pct": {k: round(100.0 * v / n_trials, 1) for k, v in appear15.items()},
    }


def run_monte_carlo_robustness(n_trials=500):
    """
    Surrogate Ranking Perturbation Stability (F17) across scales [0.5, 1.0, 1.5, 2.0]
    under BOTH independent Gaussian noise AND (A,B)-group-correlated + p_semi logit noise.
    """
    rng = np.random.default_rng(2026)
    scales = [0.5, 1.0, 1.5, 2.0]
    mc_out = {
        "label": "Surrogate Ranking Perturbation Stability (Parametric Perturbation around Frozen Surrogate; NOT True-Target Validation)",
        "n_trials": int(n_trials),
        "scales": scales,
        "multi_scale_independent": {},
        "multi_scale_group_correlated_logit": {},
    }

    for label, fname, default_k in [
        ("Q2_blind_11", "v2_q2_blind_11.csv", 10),
        ("Q3_blind_12", "v2_q3_blind_12.csv", 15),
    ]:
        df = pd.read_csv(os.path.join(OUT_DIR, fname))
        ind_rows = [_eval_mc_perturbation(df, sc, "independent_gaussian", n_trials, rng) for sc in scales]
        grp_rows = [_eval_mc_perturbation(df, sc, "group_correlated_logit", n_trials, rng) for sc in scales]

        mc_out["multi_scale_independent"][label] = ind_rows
        mc_out["multi_scale_group_correlated_logit"][label] = grp_rows

        # Baseline 0.5-sigma summary keys for backward compatibility
        r05 = ind_rows[0]
        if default_k == 10:
            mc_out[label] = {
                "label": "Surrogate Ranking Perturbation Stability",
                "n_trials": n_trials,
                "top_k": 10,
                "scale_sigma": 0.5,
                "mean_retention_pct": r05["top10_retention_mean_pct"],
                "std_retention_pct": r05["top10_retention_std_pct"],
                "top1_exact_pct": r05["top1_exact_pct"],
                "top1_survival_pct": r05["top1_in_top10_pct"],
                "mean_spearman": r05["mean_spearman"],
                "per_candidate_inclusion_pct": r05["per_candidate_top10_inclusion_pct"],
            }
        else:
            mc_out[label] = {
                "label": "Surrogate Ranking Perturbation Stability",
                "n_trials": n_trials,
                "top_k": 15,
                "scale_sigma": 0.5,
                "mean_retention_pct": r05["top15_retention_mean_pct"],
                "std_retention_pct": r05["top15_retention_std_pct"],
                "top10_retention_mean_pct": r05["top10_retention_mean_pct"],
                "top1_exact_pct": r05["top1_exact_pct"],
                "top1_survival_pct": r05["top1_in_top15_pct"],
                "mean_spearman": r05["mean_spearman"],
                "per_candidate_inclusion_pct": r05["per_candidate_top15_inclusion_pct"],
            }

    return mc_out


def main():
    t_start = time.perf_counter()
    print("=== Running v3 Remediated Closed-Loop & Fair Multi-Axis Sensitivity Experiments (M4A) ===")

    # 1. Closed-Loop 20-Seed Benchmark (F04, F14, F15)
    fmqa_res = run_closed_loop_fmqa_benchmark(n_seeds=20, n_init=50, batch_size=10, n_rounds=15)
    print(f"\n[Experiment 1: Closed-Loop on 03_val (20 seeds, budget=200/3,969) in {fmqa_res['runtime_sec']}s]")
    hg = fmqa_res["analytical_hypergeometric"]
    print(f"  Hypergeometric_Analytical: V<0.5={hg['final_expected_rare05']:.4f}/18 ({hg['final_recall05_pct']:.2f}%), "
          f"V<1.0={hg['final_expected_rare10']:.4f}/27 ({hg['final_recall10_pct']:.2f}%)")
    for m, d in fmqa_res["methods"].items():
        print(
            f"  {m:28s} [{d['oracle_labels_used']:38s}]: "
            f"V<0.5={d['final_mean_rare05']:5.2f}/18 ({d['final_recall05_pct']:5.2f}%, {d['enrichment_vs_hypergeom_rare05']:5.2f}x HG), "
            f"V<1.0={d['final_mean_rare10']:5.2f}/27 ({d['final_recall10_pct']:5.2f}%)"
        )

    # 2. Equal-Budget Solver Top-K Comparison (F13)
    topk_res = run_equal_budget_solver_topk(n_runs=50)
    print(f"\n[Experiment 2: Equal-Budget Solver Top-K (n_runs=50, 1 best/run) in {topk_res['runtime_sec']}s]")
    df_topk = pd.DataFrame(topk_res["rows"])
    print(df_topk[["instance", "encoding", "solver", "spins", "strict_feasibility_pct", "unique_feasible_count", "best_V_hat", "continuous_regret_top1", "topk_hits"]].to_string(index=False))

    # 3. Independent 4-12-Bit Quantization Sweep (F16)
    bw_res = run_bitwidth_sensitivity(n_runs=50)
    print("\n[Experiment 3: Independent 4..12-Bit Quantization Sweep (Q2 Blind 11)]")
    print(pd.DataFrame(bw_res)[[
        "bit_width", "M_B", "v1_feas_pct", "v2_M_max", "v2_max_abs",
        "v2_distinct_levels", "v2_legal_ground_count", "v2_illegal_ground_count",
        "v2_cert_regret_worst_gs", "v2_feas_pct", "v2_top10_hits"
    ]].to_string(index=False))

    # 4. Refined Fixed-M_max=4 Penalty Sweep & C_cap Pareto Sweep (F16)
    pen_res, cap_res = run_penalty_and_cap_sensitivity(n_runs=50)
    print("\n[Experiment 4A: Fixed M_max=4 Penalty Sweep on Q2 Blind (12 spins)]")
    print(pd.DataFrame(pen_res)[[
        "alpha_P", "P_block", "exact_legal_ground_count", "exact_illegal_ground_count",
        "exact_gs_feasibility_pct", "exact_energy_margin", "feasibility_pct", "sa_regret", "top10_hits"
    ]].to_string(index=False))
    print("\n[Experiment 4B: Cap C_cap Pareto Sweep]")
    print(pd.DataFrame(cap_res)[[
        "C_cap", "q2_spins", "q2_feas_pct", "q2_top10_hits",
        "q3_spins", "q3_aux", "q3_feas_pct", "q3_y_aux_pct", "q3_top15_hits",
        "q3_comp_spins", "q3_comp_feas_pct", "q3_comp_top15_hits"
    ]].to_string(index=False))

    # 5. Multi-Scale Surrogate Ranking Perturbation Stability (F17)
    mc_res = run_monte_carlo_robustness(n_trials=500)
    print("\n[Experiment 5: Surrogate Ranking Perturbation Stability (500 trials, 0.5..2.0 sigma)]")
    for mode_key in ["multi_scale_independent", "multi_scale_group_correlated_logit"]:
        print(f"  -- {mode_key} --")
        for lbl in ["Q2_blind_11", "Q3_blind_12"]:
            for r in mc_res[mode_key][lbl]:
                print(
                    f"    {lbl} scale={r['scale_sigma']:.1f}: "
                    f"top1_exact={r['top1_exact_pct']:5.1f}%, top10_ret={r['top10_retention_mean_pct']:5.1f}%, "
                    f"top15_ret={r['top15_retention_mean_pct']:5.1f}%, spearman={r['mean_spearman']:.4f}"
                )

    out = {
        "total_runtime_sec": round(time.perf_counter() - t_start, 2),
        "closed_loop_fmqa": fmqa_res,
        "equal_budget_solver_topk": topk_res,
        "bitwidth_sweep": bw_res,
        "penalty_sweep": pen_res,
        "cap_sweep": cap_res,
        "monte_carlo_robustness": mc_res,
    }
    out_path = os.path.join(OUT_DIR, "v2_experiments.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path} successfully in {out['total_runtime_sec']}s.")


if __name__ == "__main__":
    main()
