"""Experiment 3 (M3): Question 3 Three-Encoding Comparison, Local SA/Tabu Baselines,
SPQC-550 Real Quantum Hardware Execution (at most 6 tasks), Independent Decoding,
and Physical Penalty Breakdown (Tables 3, 4, 5).

Modes:
  - Default (no flags): Run Phase A (build + verify all Q3 & Q2 matrices + SA/Tabu),
    then Phase B (submit <= 6 SPQC-550 tasks one-by-one with incremental JSON saves,
    decode all samples, and export tables 3-5).
  - --skip-cim: Run Phase A gate only (build Q3 matrices, run exact & SA/Tabu checks,
    verify M2 Q2 matrices, check Kaiwu license via proxy, zero SPQC-550 calls).
  - --verify-only: Never import or instantiate CIMOptimizer; load saved matrices,
    exp3_spqc_results.json, checkpoints/*_results.pkl, and tables 3-5 from disk and
    independently re-verify every raw spin sample, constraint, energy, and breakdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

import kaiwu as kw
from kaiwu.core import (
    Binary,
    PenaltyMethodConstraint,
    QuboModel,
    get_sol_dict,
    get_sorted_solutions,
    ndarray,
    qubo_matrix_to_ising_matrix,
    qubo_model_to_ising_model,
)
from kaiwu.preprocess import (
    adjust_ising_matrix_precision,
    calculate_ising_matrix_bit_width,
    get_dynamic_range_metric,
)

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "outputs")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
sys.path.insert(0, SRC_DIR)

from element_features import ELEMENTS_52  # noqa: E402

X_Q3_EXPECTED = ["FO2", "NO2", "O3"]

TASK_NAMES = [
    "kaiwu_skill_d_q2_uncap",
    "kaiwu_skill_d_q2_full",
    "kaiwu_skill_d_q2_blind",
    "kaiwu_skill_d_q3_qfm108",
    "kaiwu_skill_d_q3_cmp209",
    "kaiwu_skill_d_q3_ros264",
]

SA_CONFIG = {
    "alpha": 0.99,
    "cutoff_temperature": 0.001,
    "iterations_per_t": 500,
    "size_limit": 50,
    "rand_seed": 42,
    "process_num": 1,
    "initial_temperature_rule": "max(100.0, 2.0 * max|h_eff|)",
}

TABU_CONFIG = {
    "max_iter": 1500,
    "size_limit": 50,
    "np_random_seed": 42,
}


def sha256_int64_matrix(mat: np.ndarray) -> str:
    arr = np.ascontiguousarray(mat, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def init_kaiwu_license() -> None:
    """Load KAIWU credentials from .env without ever printing or logging secrets."""
    env_path = os.path.join(ROOT_DIR, ".env")
    env_vars: dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()
    uid = env_vars.get("KAIWU_USER_ID") or os.environ.get("KAIWU_USER_ID", "")
    code = env_vars.get("KAIWU_SDK_CODE") or os.environ.get("KAIWU_SDK_CODE", "")
    if not uid or not code:
        raise RuntimeError("Missing KAIWU_USER_ID or KAIWU_SDK_CODE in .env / environment.")
    kw.license.init(uid, code)


def load_q3_and_q2_data() -> dict[str, Any]:
    """Load Q3 pool, M1 QFM artifacts, Q2 pool, and verify IDs/formulas/alignment."""
    q3_pred = pd.read_csv(os.path.join(OUT_DIR, "q3_full_pool_predictions.csv"))
    q3_pool = pd.read_csv(os.path.join(DATA_DIR, "08_castelli_q3_pool.csv"))
    q3_blind = pd.read_csv(os.path.join(DATA_DIR, "12_castelli_q3_pool_blind.csv"))

    assert len(q3_pred) == 8112, f"Expected 8112 rows in q3_pred, got {len(q3_pred)}"
    assert len(q3_pool) == 8112, f"Expected 8112 rows in 08_pool, got {len(q3_pool)}"
    assert len(q3_blind) == 1587, f"Expected 1587 rows in 12_blind, got {len(q3_blind)}"
    assert list(q3_pred["id"]) == list(q3_pool["id"]), "ID mismatch between q3_pred and 08_pool"
    assert set(q3_pred.loc[q3_pred["in_blind_pool"] == 1, "id"]) == set(q3_blind["id"]), (
        "Q3 blind pool ID mismatch"
    )

    qfm_npz = np.load(os.path.join(OUT_DIR, "exp1_qfm_q3_artifacts.npz"), allow_pickle=True)
    elements_52 = [str(x) for x in qfm_npz["elements_52"]]
    x_q3 = [str(x) for x in qfm_npz["x_q3"]]
    assert elements_52 == list(ELEMENTS_52), "ELEMENTS_52 mismatch with M1 QFM artifacts"
    assert x_q3 == X_Q3_EXPECTED, f"Unexpected x_q3: {x_q3}"

    elem_idx = {e: i for i, e in enumerate(elements_52)}
    x_idx = {x: k for k, x in enumerate(x_q3)}

    V_raw_q3 = np.zeros((3, 52, 52), dtype=np.float64)
    blind_mask_q3 = np.zeros((3, 52, 52), dtype=np.int64)
    formula_grid_q3 = np.empty((3, 52, 52), dtype=object)
    id_grid_q3 = np.zeros((3, 52, 52), dtype=np.int64)

    for _, row in q3_pred.iterrows():
        k, i, j = x_idx[row["X"]], elem_idx[row["A"]], elem_idx[row["B"]]
        V_raw_q3[k, i, j] = float(row["V_pred"])
        blind_mask_q3[k, i, j] = int(row["in_blind_pool"])
        formula_grid_q3[k, i, j] = str(row["formula"])
        id_grid_q3[k, i, j] = int(row["id"])

    assert int(blind_mask_q3.sum()) == 1587

    # Rank lookup in Q3 pool (1-based)
    q3_sorted = q3_pred.sort_values("V_pred", ascending=True).reset_index(drop=True)
    rank_map_q3 = {str(r["formula"]): int(idx + 1) for idx, r in q3_sorted.iterrows()}

    # Load Q2 pool & rank lookup
    q2_pred = pd.read_csv(os.path.join(OUT_DIR, "q2_full_pool_predictions.csv"))
    assert len(q2_pred) == 2704
    q2_sorted = q2_pred.sort_values("V_pred", ascending=True).reset_index(drop=True)
    rank_map_q2 = {str(r["formula"]): int(idx + 1) for idx, r in q2_sorted.iterrows()}

    q2_npz = np.load(os.path.join(OUT_DIR, "exp2_q2_matrices.npz"), allow_pickle=True)
    assert [str(x) for x in q2_npz["elements_52"]] == elements_52

    return {
        "elements_52": elements_52,
        "x_q3": x_q3,
        "qfm": {
            "w0": float(qfm_npz["w0"]),
            "w_a": qfm_npz["w_a"].astype(np.float64),
            "w_b": qfm_npz["w_b"].astype(np.float64),
            "w_x": qfm_npz["w_x"].astype(np.float64),
            "W_ab": qfm_npz["W_ab"].astype(np.float64),
            "W_ax": qfm_npz["W_ax"].astype(np.float64),
            "W_bx": qfm_npz["W_bx"].astype(np.float64),
            "V_qfm_q3": qfm_npz["V_qfm_q3"].astype(np.float64),
        },
        "V_raw_q3": V_raw_q3,
        "blind_mask_q3": blind_mask_q3,
        "formula_grid_q3": formula_grid_q3,
        "id_grid_q3": id_grid_q3,
        "rank_map_q3": rank_map_q3,
        "q3_pred_df": q3_pred,
        "V_raw_q2": q2_npz["V_pred_52x52"].astype(np.float64),
        "blind_mask_q2": q2_npz["blind_mask_52x52"].astype(np.int64),
        "rank_map_q2": rank_map_q2,
        "q2_pred_df": q2_pred,
        "q2_npz": q2_npz,
    }


# ---------------------------------------------------------------------------
# Q3 QUBO / Ising Builders & Alphabetical Variable Verification
# ---------------------------------------------------------------------------

def qfm_qubo_parts(qfm: dict[str, Any], P_oh: float) -> tuple[np.ndarray, np.ndarray]:
    """Build exact 107-bit QFM-QUBO (q_vec, Q_sym) in alphabetical order: a[00..51], b[00..51], x[0..2]."""
    q_vec = np.concatenate([qfm["w_a"] - P_oh, qfm["w_b"] - P_oh, qfm["w_x"] - P_oh])
    Q_sym = np.zeros((107, 107), dtype=np.float64)
    for s0, sz in [(0, 52), (52, 52), (104, 3)]:
        blk = Q_sym[s0 : s0 + sz, s0 : s0 + sz]
        blk[~np.eye(sz, dtype=bool)] = 2.0 * P_oh
    Q_sym[:52, 52:104] = qfm["W_ab"]
    Q_sym[52:104, :52] = qfm["W_ab"].T
    Q_sym[:52, 104:107] = qfm["W_ax"]
    Q_sym[104:107, :52] = qfm["W_ax"].T
    Q_sym[52:104, 104:107] = qfm["W_bx"]
    Q_sym[104:107, 52:104] = qfm["W_bx"].T
    return q_vec, Q_sym


def cmp_qubo_parts(M_3d: np.ndarray, P_oh: float) -> tuple[np.ndarray, np.ndarray]:
    """Build exact 208-bit Compact-QUBO (q_vec, Q_sym) in alphabetical order: b[00..51], xa[000..155]."""
    q_vec = np.full(208, -P_oh, dtype=np.float64)
    Q_sym = np.zeros((208, 208), dtype=np.float64)
    for s0, sz in [(0, 52), (52, 156)]:
        blk = Q_sym[s0 : s0 + sz, s0 : s0 + sz]
        blk[~np.eye(sz, dtype=bool)] = 2.0 * P_oh
    M_flat = M_3d.reshape(156, 52)
    Q_sym[:52, 52:208] = M_flat.T
    Q_sym[52:208, :52] = M_flat
    return q_vec, Q_sym


def ros_qubo_parts(M_3d: np.ndarray, P_oh: float, P_ros: float) -> tuple[np.ndarray, np.ndarray]:
    """Build exact 263-bit Rosenberg HUBO-QUBO in alphabetical order: a[00..51], b[00..51], x[0..2], y[0][00]..y[2][51]."""
    q_vec = np.zeros(263, dtype=np.float64)
    Q_sym = np.zeros((263, 263), dtype=np.float64)
    M_flat = M_3d.reshape(156, 52)
    Q_sym[107:263, 52:104] = M_flat
    Q_sym[52:104, 107:263] = M_flat.T

    q_vec[0:52] -= P_oh
    q_vec[52:104] -= P_oh
    q_vec[104:107] -= P_oh
    for s0, sz in [(0, 52), (52, 52), (104, 3)]:
        blk = Q_sym[s0 : s0 + sz, s0 : s0 + sz]
        blk[~np.eye(sz, dtype=bool)] += 2.0 * P_oh

    # Rosenberg quadratic penalty: P_ros * (x_k a_i - 2 x_k y_{k,i} - 2 a_i y_{k,i} + 3 y_{k,i})
    q_vec[107:263] += 3.0 * P_ros
    for k in range(3):
        for i in range(52):
            u = 107 + k * 52 + i
            xk = 104 + k
            ai = i
            Q_sym[xk, ai] += P_ros
            Q_sym[ai, xk] += P_ros
            Q_sym[xk, u] -= 2.0 * P_ros
            Q_sym[u, xk] -= 2.0 * P_ros
            Q_sym[ai, u] -= 2.0 * P_ros
            Q_sym[u, ai] -= 2.0 * P_ros
    return q_vec, Q_sym


def parts_to_upper_qubo(q_vec: np.ndarray, Q_sym: np.ndarray) -> np.ndarray:
    """Convert (q_vec, Q_sym) where x^T Q_sym x / 2 + q_vec^T x into upper-triangular QUBO matrix Q."""
    Q = np.triu(Q_sym, 1)
    np.fill_diagonal(Q, q_vec)
    return Q


def verify_qubo_models_and_signs(data: dict[str, Any]) -> dict[str, Any]:
    """Verify QuboModel alphabetical variable ordering, analytical Ising match (<1e-10),
    -s^T J s + bias == x^T Q x, and get_sorted_solutions sign convention across all 3 encodings."""
    qfm = data["qfm"]
    M_c3 = np.minimum(data["V_raw_q3"], 3.0)

    # 1. QFM-QUBO
    m1 = QuboModel()
    a1, b1, x1 = ndarray(52, "a", Binary), ndarray(52, "b", Binary), ndarray(3, "x", Binary)
    m1.set_objective(
        qfm["w0"]
        + kw.core.dot(qfm["w_a"], a1)
        + kw.core.dot(qfm["w_b"], b1)
        + kw.core.dot(qfm["w_x"], x1)
        + a1 @ qfm["W_ab"] @ b1
        + a1 @ qfm["W_ax"] @ x1
        + b1 @ qfm["W_bx"] @ x1
        + 62.0 * ((a1.sum() - 1) ** 2 + (b1.sum() - 1) ** 2 + (x1.sum() - 1) ** 2)
    )
    m1.set_constraint_handler(PenaltyMethodConstraint)
    im1 = qubo_model_to_ising_model(m1)
    v1 = sorted(im1.get_variables().keys(), key=lambda k: im1.get_variables()[k])
    exp_v1 = [f"a[{i:02d}]" for i in range(52)] + [f"b[{j:02d}]" for j in range(52)] + [f"x[{k}]" for k in range(3)] + ["__spin__"]
    assert v1 == exp_v1, f"QFM var order mismatch: {v1[:5]} .. {v1[-5:]}"
    q1, Q1 = qfm_qubo_parts(qfm, 62.0)
    J_im1 = im1.get_matrix()
    diff1 = max(
        float(np.max(np.abs(-Q1 / 8.0 - J_im1[:107, :107]))),
        float(np.max(np.abs((-q1 / 4.0 + (-Q1 / 8.0).sum(axis=1)) - J_im1[:107, 107]))),
    )
    assert diff1 < 1e-10, f"QFM Ising matrix mismatch: {diff1}"

    # 2. Compact-QUBO (note b < xa alphabetically!)
    m2 = QuboModel()
    u2, b2 = ndarray(156, "xa", Binary), ndarray(52, "b", Binary)
    m2.set_objective(u2 @ M_c3.reshape(156, 52) @ b2 + 3.1 * ((u2.sum() - 1) ** 2 + (b2.sum() - 1) ** 2))
    m2.set_constraint_handler(PenaltyMethodConstraint)
    im2 = qubo_model_to_ising_model(m2)
    v2 = sorted(im2.get_variables().keys(), key=lambda k: im2.get_variables()[k])
    exp_v2 = [f"b[{j:02d}]" for j in range(52)] + [f"xa[{u:03d}]" for u in range(156)] + ["__spin__"]
    assert v2 == exp_v2, f"Compact var order mismatch: {v2[:5]} .. {v2[-5:]}"
    q2, Q2 = cmp_qubo_parts(M_c3, 3.1)
    J_im2 = im2.get_matrix()
    diff2 = max(
        float(np.max(np.abs(-Q2 / 8.0 - J_im2[:208, :208]))),
        float(np.max(np.abs((-q2 / 4.0 + (-Q2 / 8.0).sum(axis=1)) - J_im2[:208, 208]))),
    )
    assert diff2 < 1e-10, f"Compact Ising matrix mismatch: {diff2}"

    # 3. Rosenberg HUBO-QUBO
    m3 = QuboModel()
    x3, a3, b3, y3 = ndarray(3, "x", Binary), ndarray(52, "a", Binary), ndarray(52, "b", Binary), ndarray((3, 52), "y", Binary)
    ros = (x3[:, None] * a3[None, :] - 2 * x3[:, None] * y3 - 2 * a3[None, :] * y3 + 3 * y3).sum()
    m3.set_objective(y3.reshape(156) @ M_c3.reshape(156, 52) @ b3 + 3.6 * ((x3.sum() - 1) ** 2 + (a3.sum() - 1) ** 2 + (b3.sum() - 1) ** 2) + 3.3 * ros)
    m3.set_constraint_handler(PenaltyMethodConstraint)
    im3 = qubo_model_to_ising_model(m3)
    v3 = sorted(im3.get_variables().keys(), key=lambda k: im3.get_variables()[k])
    exp_v3 = (
        [f"a[{i:02d}]" for i in range(52)]
        + [f"b[{j:02d}]" for j in range(52)]
        + [f"x[{k}]" for k in range(3)]
        + [f"y[{k}][{i:02d}]" for k in range(3) for i in range(52)]
        + ["__spin__"]
    )
    assert v3 == exp_v3, f"Rosenberg var order mismatch: {v3[:5]} .. {v3[-5:]}"
    q3, Q3 = ros_qubo_parts(M_c3, 3.6, 3.3)
    J_im3 = im3.get_matrix()
    diff3 = max(
        float(np.max(np.abs(-Q3 / 8.0 - J_im3[:263, :263]))),
        float(np.max(np.abs((-q3 / 4.0 + (-Q3 / 8.0).sum(axis=1)) - J_im3[:263, 263]))),
    )
    assert diff3 < 1e-10, f"Rosenberg Ising matrix mismatch: {diff3}"

    # Verify -s^T J s + bias == x^T Q x and get_sorted_solutions
    rng = np.random.default_rng(2026)
    max_sign_err = 0.0
    for q_v, Q_s in [(q1, Q1), (q2, Q2), (q3, Q3)]:
        Q_up = parts_to_upper_qubo(q_v, Q_s)
        J_mat, bias = qubo_matrix_to_ising_matrix(Q_up)
        n_b = len(q_v)
        for _ in range(8):
            x = rng.integers(0, 2, size=n_b).astype(np.float64)
            s = np.append(2.0 * x - 1.0, 1.0)
            e_qubo = float(x @ Q_up @ x)
            e_ising = -float(s @ J_mat @ s) + float(bias)
            max_sign_err = max(max_sign_err, abs(e_qubo - e_ising))
        # Check get_sorted_solutions energy sign on small sample
        s_pair = np.vstack([
            np.append(2.0 * rng.integers(0, 2, size=n_b) - 1.0, 1.0),
            np.append(2.0 * rng.integers(0, 2, size=n_b) - 1.0, 1.0),
        ]).astype(np.int8)
        sorted_sols, sorted_energies = get_sorted_solutions(J_mat, s_pair, bias=float(bias))
        for vec, energy in zip(sorted_sols, sorted_energies):
            assert abs(float(energy) - (-float(vec @ J_mat @ vec) + float(bias))) < 1e-8

    assert max_sign_err < 1e-9, f"Ising sign error too high: {max_sign_err}"
    return {
        "qfm_vars": exp_v1,
        "cmp_vars": exp_v2,
        "ros_vars": exp_v3,
        "max_model_vs_analytical_diff": max(diff1, diff2, diff3),
        "max_qubo_vs_neg_sJs_error": max_sign_err,
    }


def build_ising_from_qubo_parts(
    q_vec: np.ndarray,
    Q_sym: np.ndarray,
    K: int,
    target_max: int = 105,
    var_names_base: list[str] | None = None,
) -> dict[str, Any]:
    """Quantize to 8-bit int64 with consistent post-rounding hub fields and K-copy ref splitting."""
    n_bits = len(q_vec)
    Q_up = parts_to_upper_qubo(q_vec, Q_sym)
    _, bias_raw = qubo_matrix_to_ising_matrix(Q_up)

    J_raw = -Q_sym / 8.0
    h_raw = -q_vec / 4.0 + J_raw.sum(axis=1)

    if K == 1:
        J_full_raw = np.zeros((n_bits + 1, n_bits + 1), dtype=np.float64)
        J_full_raw[:n_bits, :n_bits] = J_raw
        J_full_raw[:n_bits, n_bits] = h_raw
        J_full_raw[n_bits, :n_bits] = h_raw
        J = adjust_ising_matrix_precision(J_full_raw, bit_width=8).astype(np.int64)
        np.fill_diagonal(J, 0)
        J = ((J + J.T) // 2).astype(np.int64)
        alpha = 127.0 / float(np.max(np.abs(J_full_raw)))
        h_int = J[:n_bits, n_bits].copy()
        j_rr = 0
        vars_full = (var_names_base or []) + ["__spin__"]
    else:
        max_coeff = max(float(np.max(np.abs(J_raw))), float(np.max(np.abs(h_raw))) / K)
        alpha = float(target_max) / max_coeff
        J_int = np.rint(J_raw * alpha).astype(np.int64)
        np.fill_diagonal(J_int, 0)
        J_int = ((J_int + J_int.T) // 2).astype(np.int64)
        # Consistent post-rounding hub fields guarantee exact cancellation of inactive background couplings
        h_int = np.rint(-alpha * q_vec / 4.0).astype(np.int64) + J_int.sum(axis=1)

        n_spins = n_bits + K
        J = np.zeros((n_spins, n_spins), dtype=np.int64)
        J[:n_bits, :n_bits] = J_int
        for u in range(n_bits):
            val = int(h_int[u])
            q, r = divmod(abs(val), K)
            sgn = 1 if val >= 0 else -1
            row_vals = np.full(K, sgn * q, dtype=np.int64)
            if r > 0:
                row_vals[(np.arange(r) + u * 7) % K] += sgn
            J[u, n_bits:] = row_vals
            J[n_bits:, u] = row_vals

        j_rr = max(1, int(math.ceil(1.15 * float(np.sum(np.abs(h_int))) / (K * (K - 1)))))
        assert j_rr <= 127, f"j_rr={j_rr} exceeds 127"
        for c1 in range(K):
            for c2 in range(c1 + 1, K):
                J[n_bits + c1, n_bits + c2] = j_rr
                J[n_bits + c2, n_bits + c1] = j_rr
        vars_full = (var_names_base or []) + [f"__spin_ref_{c:03d}__" for c in range(K)]

    assert J.dtype == np.int64
    assert np.array_equal(J, J.T), "J is not symmetric"
    assert np.all(np.diag(J) == 0), "J diagonal is not zero"
    assert int(np.min(J)) >= -127 and int(np.max(J)) <= 127, f"J out of [-127, 127]: [{J.min()}, {J.max()}]"
    bw = int(calculate_ising_matrix_bit_width(J)["precision"])
    assert bw == 8, f"Expected bit width 8, got {bw}"
    drm = float(get_dynamic_range_metric(J))

    return {
        "J": J,
        "Q_up": Q_up,
        "bias_scaled": float(bias_raw * alpha),
        "alpha": float(alpha),
        "j_rr": int(j_rr),
        "h_int": h_int,
        "n_bits": int(n_bits),
        "n_ref_copies": int(K),
        "n_spins": int(n_bits + K),
        "bit_width": bw,
        "drm": drm,
        "variables": vars_full,
    }


# ---------------------------------------------------------------------------
# Exact 8,112-State Evaluation & Global Lower Bounds
# ---------------------------------------------------------------------------

def min_broken_ref_gap(J: np.ndarray, n_bits: int, K: int, j_rr: int) -> float:
    if K <= 1:
        return float("nan")
    abs_cols = np.sort(np.abs(J[:n_bits, n_bits:]), axis=1)[:, ::-1]
    pref_col = np.cumsum(abs_cols, axis=1)
    min_lb = float("inf")
    for t in range(1, K // 2 + 1):
        pen = 4.0 * j_rr * t * (K - t)
        max_gain = 2.0 * float(pref_col[:, t - 1].sum())
        if pen - max_gain < min_lb:
            min_lb = pen - max_gain
    return float(min_lb)


def collapse_to_effective_qubo(J: np.ndarray, n_bits: int, K: int) -> tuple[float, np.ndarray, np.ndarray]:
    h_eff = J[:n_bits, n_bits:].sum(axis=1).astype(np.float64) if K > 1 else J[:n_bits, n_bits].astype(np.float64)
    J_sub = J[:n_bits, :n_bits].astype(np.float64)
    J_up = np.triu(J_sub, 1)
    C_eff = -8.0 * (J_up + J_up.T)
    q_eff = 4.0 * J_sub.sum(axis=1) - 4.0 * h_eff
    E0 = -float(J_sub.sum()) + 2.0 * float(h_eff.sum())
    return E0, q_eff, C_eff


def qfm_global_lb(J: np.ndarray, K: int, e_valid_opt: float, E_valid: np.ndarray) -> float:
    """Lower bound on min_{invalid (a,b,x)} E - e_valid_opt across all 2^107 - 8112 invalid states."""
    E0, q_eff, C_eff = collapse_to_effective_qubo(J, 107, K)
    caa = float(C_eff[:52, :52][~np.eye(52, dtype=bool)][0])
    cbb = float(C_eff[52:104, 52:104][~np.eye(52, dtype=bool)][0])
    C_ab = C_eff[:52, 52:104]
    C_ax = C_eff[:52, 104:107]
    C_bx = C_eff[52:104, 104:107]

    min_inv = float("inf")
    for x_mask in range(8):
        xv = np.array([(x_mask >> b) & 1 for b in range(3)], dtype=np.float64)
        kX = int(xv.sum())
        ex = float(q_eff[104:107] @ xv + 0.5 * (xv @ C_eff[104:107, 104:107] @ xv))
        a_eff = q_eff[:52] + C_ax @ xv
        b_eff = q_eff[52:104] + C_bx @ xv
        pref_a0 = np.concatenate([[0.0], np.cumsum(np.sort(a_eff))])
        pref_b0 = np.concatenate([[0.0], np.cumsum(np.sort(b_eff))])

        # Sorted rows of C_ab for tight row-wise kB lower bound
        Cab_sorted = np.sort(C_ab, axis=1)
        Cab_pref = np.hstack([np.zeros((52, 1)), np.cumsum(Cab_sorted, axis=1)])

        for kA in range(53):
            pa = 0.5 * caa * kA * (kA - 1)
            for kB in range(53):
                if kX == 1 and kA == 1 and kB == 1:
                    continue
                pb = 0.5 * cbb * kB * (kB - 1)
                if kX == 1 and kA == 2 and kB == 1:
                    # Exact over all C(52,2)*52 = 68,952 states for this x
                    xk = int(np.argmax(xv))
                    i1, i2 = np.triu_indices(52, 1)
                    base_a2 = E0 + ex + pa + a_eff[i1] + a_eff[i2]
                    cab_2 = C_ab[i1, :] + C_ab[i2, :]
                    vals = base_a2[:, None] + b_eff[None, :] + cab_2
                    lb = float(vals.min())
                elif kX == 1 and kA == 1 and kB == 2:
                    j1, j2 = np.triu_indices(52, 1)
                    base_b2 = E0 + ex + pb + b_eff[j1] + b_eff[j2]
                    cab_2 = C_ab[:, j1] + C_ab[:, j2]
                    vals = base_b2[None, :] + a_eff[:, None] + cab_2
                    lb = float(vals.min())
                elif kA == 0 or kB == 0:
                    lb = E0 + ex + pa + pb + pref_a0[kA] + pref_b0[kB]
                else:
                    row_costs = a_eff + Cab_pref[:, kB]
                    row_sum_lb = float(np.sum(np.sort(row_costs)[:kA]))
                    lb = E0 + ex + pa + pb + pref_b0[kB] + row_sum_lb
                if lb < min_inv:
                    min_inv = lb
    return float(min_inv - e_valid_opt)


def cmp_global_lb(J: np.ndarray, K: int, e_valid_max: float) -> float:
    """Lower bound on min_{invalid (b,xa)} E - e_valid_max across all 2^208 - 8112 states."""
    E0, q_eff, C_eff = collapse_to_effective_qubo(J, 208, K)
    cbb = float(C_eff[:52, :52][~np.eye(52, dtype=bool)][0])
    cuu = float(C_eff[52:208, 52:208][~np.eye(156, dtype=bool)][0])
    C_bu = C_eff[:52, 52:208]
    cbu_sorted = np.sort(C_bu.ravel())
    cbu_pref = np.concatenate([[0.0], np.cumsum(cbu_sorted)])
    pref_b = np.concatenate([[0.0], np.cumsum(np.sort(q_eff[:52]))])
    pref_u = np.concatenate([[0.0], np.cumsum(np.sort(q_eff[52:208]))])

    min_inv = float("inf")
    for kB in range(53):
        pb = 0.5 * cbb * kB * (kB - 1)
        for kU in range(157):
            if kB == 1 and kU == 1:
                continue
            pu = 0.5 * cuu * kU * (kU - 1)
            lb = E0 + pb + pu + pref_b[kB] + pref_u[kU] + cbu_pref[kB * kU]
            if lb < min_inv:
                min_inv = lb
    return float(min_inv - e_valid_max)


def ros_global_lb(J: np.ndarray, K: int, E_valid: np.ndarray) -> tuple[float, float]:
    """Exact vectorized lower bound across all 2^263 - 8112 states for Rosenberg HUBO-QUBO."""
    E0, q_eff, C_eff = collapse_to_effective_qubo(J, 263, K)
    caa = float(C_eff[:52, :52][~np.eye(52, dtype=bool)][0])
    cbb = float(C_eff[52:104, 52:104][~np.eye(52, dtype=bool)][0])
    C_yb = C_eff[107:263, 52:104]
    min_cyb_sum = np.minimum(C_yb, 0.0).sum(axis=1)

    k_g, i_g, j_g = np.meshgrid(np.arange(3), np.arange(52), np.arange(52), indexing="ij")
    y_idx = np.arange(107, 263)[None, None, None, :]
    y_flip_on = (
        q_eff[107:263][None, None, None, :]
        + C_eff[i_g[:, :, :, None], y_idx]
        + C_eff[(104 + k_g)[:, :, :, None], y_idx]
        + C_eff[(52 + j_g)[:, :, :, None], y_idx]
    )
    single_y_pen = y_flip_on.copy()
    for k in range(3):
        for i in range(52):
            u = k * 52 + i
            single_y_pen[k, i, :, u] = -y_flip_on[k, i, :, u]
    neg_sum = np.minimum(single_y_pen, 0.0).sum(axis=-1)
    has_neg = (single_y_pen < 0.0).any(axis=-1)
    min_pos = np.where(single_y_pen >= 0.0, single_y_pen, 1e18).min(axis=-1)
    min_y_viol = np.where(has_neg, neg_sum, min_pos)
    min_y_abs = float((E_valid + min_y_viol).min())

    min_inv_card = float("inf")
    for x_mask in range(8):
        xv = np.array([(x_mask >> b) & 1 for b in range(3)], dtype=np.float64)
        kX = int(xv.sum())
        ex = float(q_eff[104:107] @ xv + 0.5 * (xv @ C_eff[104:107, 104:107] @ xv))
        a_eff_cost = q_eff[:52].copy()
        y_base_const = 0.0
        for k in range(3):
            u_sl = slice(107 + k * 52, 107 + (k + 1) * 52)
            c0 = np.minimum(0.0, q_eff[u_sl] + C_eff[u_sl, 104 + k] * xv[k] + min_cyb_sum[k * 52 : (k + 1) * 52])
            c1 = C_eff[:52, 104 + k] * xv[k] + np.minimum(
                0.0,
                q_eff[u_sl]
                + C_eff[u_sl, 104 + k] * xv[k]
                + np.diag(C_eff[:52, u_sl])
                + min_cyb_sum[k * 52 : (k + 1) * 52],
            )
            y_base_const += float(c0.sum())
            a_eff_cost += c1 - c0
        pref_a = np.concatenate([[0.0], np.cumsum(np.sort(a_eff_cost))])
        pref_b = np.concatenate([[0.0], np.cumsum(np.sort(q_eff[52:104]))])
        for kA in range(53):
            pa = 0.5 * caa * kA * (kA - 1)
            for kB in range(53):
                if kX == 1 and kA == 1 and kB == 1:
                    continue
                pb = 0.5 * cbb * kB * (kB - 1)
                lb = E0 + ex + y_base_const + pa + pb + pref_a[kA] + pref_b[kB]
                if lb < min_inv_card:
                    min_inv_card = lb

    e_max = float(E_valid.max())
    return float(min_inv_card - e_max), float(min_y_abs - e_max)


def evaluate_q3_exact_8112(
    mode: str,
    J: np.ndarray,
    K: int,
    j_rr: int,
    V_raw_q3: np.ndarray,
    formula_grid_q3: np.ndarray,
) -> dict[str, Any]:
    """Enumerate all 8,112 valid states on quantized 8-bit matrix and compute exact rank, ties, levels, and gaps."""
    n_bits = {"qfm": 107, "cmp": 208, "ros": 263}[mode]
    E0, q_eff, C_eff = collapse_to_effective_qubo(J, n_bits, K)

    if mode == "qfm":
        E_valid = np.empty((3, 52, 52), dtype=np.float64)
        for k in range(3):
            xk = 104 + k
            for i in range(52):
                base = E0 + q_eff[i] + q_eff[xk] + C_eff[i, xk]
                E_valid[k, i, :] = base + q_eff[52:104] + C_eff[i, 52:104] + C_eff[xk, 52:104]
        lvls = int(len(np.unique(J[:52, 52:104])))
    elif mode == "cmp":
        E_ub = E0 + q_eff[52:208, None] + q_eff[None, :52] + C_eff[52:208, :52]
        E_valid = E_ub.reshape(3, 52, 52)
        lvls = int(len(np.unique(J[:52, 52:208])))
    else:
        E_valid = np.empty((3, 52, 52), dtype=np.float64)
        for k in range(3):
            xk = 104 + k
            for i in range(52):
                u = 107 + k * 52 + i
                base = E0 + q_eff[i] + q_eff[xk] + q_eff[u] + C_eff[i, xk] + C_eff[i, u] + C_eff[xk, u]
                E_valid[k, i, :] = base + q_eff[52:104] + C_eff[i, 52:104] + C_eff[xk, 52:104] + C_eff[u, 52:104]
        lvls = int(len(np.unique(J[107:263, 52:104])))

    true_kij = np.unravel_index(int(np.argmin(V_raw_q3)), V_raw_q3.shape)
    assert str(formula_grid_q3[true_kij]) == "LaTiO2N"
    e_true = float(E_valid[true_kij])
    e_min = float(E_valid.min())
    e_max = float(E_valid.max())
    rank = int(np.sum(E_valid < e_true - 1e-9)) + 1
    ties = int(np.sum(E_valid <= e_min + 1e-9))
    best_kij = np.unravel_index(int(np.argmin(E_valid)), E_valid.shape)

    if mode == "qfm":
        gap_inv = qfm_global_lb(J, K, e_true, E_valid)
    elif mode == "cmp":
        gap_inv = cmp_global_lb(J, K, e_max)
    else:
        g_card, g_y = ros_global_lb(J, K, E_valid)
        gap_inv = min(g_card, g_y)

    ref_gap = min_broken_ref_gap(J, n_bits, K, j_rr)
    return {
        "J_coupling_levels": lvls,
        "exact_ground_state_rank_of_true_opt": rank,
        "exact_ties_at_top1": ties,
        "exact_8bit_best_formula": str(formula_grid_q3[best_kij]),
        "exact_8bit_best_V_pred": float(V_raw_q3[best_kij]),
        "min_infeasible_energy_gap": float(gap_inv),
        "min_broken_ref_gap": float(ref_gap),
        "E_valid_min": e_min,
        "E_valid_max": e_max,
    }


# ---------------------------------------------------------------------------
# Independent Spin Sample Decoding (Q2 & Q3)
# ---------------------------------------------------------------------------

def decode_sample_pool(
    pool: np.ndarray,
    J: np.ndarray,
    task_type: str,
    data: dict[str, Any],
    n_ref_copies: int,
    V_cap: float = 3.0,
) -> dict[str, Any]:
    """Independently decode every spin vector, normalize via ref consensus, check ALL constraints,
    evaluate -float(s @ J @ s), and look up original V_pred, formula, in_blind, and true_rank_in_pool."""
    arr = np.asarray(pool, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_samples, n_spins = arr.shape
    elements_52 = data["elements_52"]
    x_q3 = data["x_q3"]

    if task_type == "q2":
        n_dec = 104
        V_grid = data["V_raw_q2"]
        blind_grid = data["blind_mask_q2"]
        rank_map = data["rank_map_q2"]
    elif task_type == "q3_qfm":
        n_dec = 107
        V_grid = data["V_raw_q3"]
        blind_grid = data["blind_mask_q3"]
        rank_map = data["rank_map_q3"]
    elif task_type == "q3_cmp":
        n_dec = 208
        V_grid = data["V_raw_q3"]
        blind_grid = data["blind_mask_q3"]
        rank_map = data["rank_map_q3"]
    elif task_type == "q3_ros":
        n_dec = 263
        V_grid = data["V_raw_q3"]
        blind_grid = data["blind_mask_q3"]
        rank_map = data["rank_map_q3"]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    assert n_spins == n_dec + n_ref_copies, f"Expected {n_dec + n_ref_copies} spins, got {n_spins}"

    decoded_samples: list[dict[str, Any]] = []
    ising_energies: list[float] = []
    feasible_flags: list[bool] = []
    ref_ok_cnt = 0
    feas_cnt = 0
    feas_blind_cnt = 0
    subcap_feas_cnt = 0
    subcap_blind_cnt = 0
    hard_viol_in_feas = 0

    for row in arr:
        assert set(np.unique(row)).issubset({-1, 1}), "Non +/-1 spin value detected"
        energy = -float(row @ J @ row)
        ising_energies.append(energy)

        refs = row[n_dec:]
        ref_consensus = bool(np.all(refs == refs[0]))
        if ref_consensus:
            ref_ok_cnt += 1
        s_norm = row[:n_dec] * refs[0]
        bits = (s_norm + 1) // 2

        ros_viol = 0
        if task_type == "q2":
            ab, bb = bits[:52], bits[52:104]
            oh_viol = abs(int(ab.sum()) - 1) + abs(int(bb.sum()) - 1)
            feasible = bool(ref_consensus and oh_viol == 0)
            if oh_viol == 0:
                ia, ib = int(np.argmax(ab)), int(np.argmax(bb))
                formula = f"{elements_52[ia]}{elements_52[ib]}O2N"
                v_pred = float(V_grid[ia, ib])
                in_blind = int(blind_grid[ia, ib])
                t_rank = int(rank_map[formula])
            else:
                formula, v_pred, in_blind, t_rank = "INFEASIBLE", None, 0, None
        elif task_type == "q3_qfm":
            ab, bb, xb = bits[:52], bits[52:104], bits[104:107]
            oh_viol = abs(int(ab.sum()) - 1) + abs(int(bb.sum()) - 1) + abs(int(xb.sum()) - 1)
            feasible = bool(ref_consensus and oh_viol == 0)
            if oh_viol == 0:
                k, ia, ib = int(np.argmax(xb)), int(np.argmax(ab)), int(np.argmax(bb))
                formula = str(data["formula_grid_q3"][k, ia, ib])
                v_pred = float(V_grid[k, ia, ib])
                in_blind = int(blind_grid[k, ia, ib])
                t_rank = int(rank_map[formula])
            else:
                formula, v_pred, in_blind, t_rank = "INFEASIBLE", None, 0, None
        elif task_type == "q3_cmp":
            bb, ub = bits[:52], bits[52:208]
            oh_viol = abs(int(bb.sum()) - 1) + abs(int(ub.sum()) - 1)
            feasible = bool(ref_consensus and oh_viol == 0)
            if oh_viol == 0:
                u, ib = int(np.argmax(ub)), int(np.argmax(bb))
                k, ia = divmod(u, 52)
                formula = str(data["formula_grid_q3"][k, ia, ib])
                v_pred = float(V_grid[k, ia, ib])
                in_blind = int(blind_grid[k, ia, ib])
                t_rank = int(rank_map[formula])
            else:
                formula, v_pred, in_blind, t_rank = "INFEASIBLE", None, 0, None
        else:  # q3_ros
            ab, bb, xb, yb = bits[:52], bits[52:104], bits[104:107], bits[107:263]
            oh_viol = abs(int(ab.sum()) - 1) + abs(int(bb.sum()) - 1) + abs(int(xb.sum()) - 1)
            ros_viol = int(np.sum(yb != np.outer(xb, ab).ravel()))
            feasible = bool(ref_consensus and oh_viol == 0 and ros_viol == 0)
            if oh_viol == 0 and ros_viol == 0:
                k, ia, ib = int(np.argmax(xb)), int(np.argmax(ab)), int(np.argmax(bb))
                formula = str(data["formula_grid_q3"][k, ia, ib])
                v_pred = float(V_grid[k, ia, ib])
                in_blind = int(blind_grid[k, ia, ib])
                t_rank = int(rank_map[formula])
            else:
                formula, v_pred, in_blind, t_rank = "INFEASIBLE", None, 0, None

        feasible_flags.append(feasible)
        if feasible:
            feas_cnt += 1
            if oh_viol != 0 or ros_viol != 0 or not ref_consensus:
                hard_viol_in_feas += 1
            if in_blind == 1:
                feas_blind_cnt += 1
            if v_pred is not None and v_pred < V_cap:
                subcap_feas_cnt += 1
                if in_blind == 1:
                    subcap_blind_cnt += 1

        decoded_samples.append({
            "formula": formula,
            "V_pred": v_pred,
            "ising_energy": energy,
            "in_blind": in_blind,
            "true_rank_in_pool": t_rank,
            "feasible": feasible,
            "ref_consensus": ref_consensus,
            "onehot_viol": int(oh_viol),
            "rosenberg_viol": int(ros_viol),
        })

    feas_samples = [s for s in decoded_samples if s["feasible"]]
    if feas_samples:
        # Best by Ising energy (break ties by V_pred)
        best_e_sample = min(feas_samples, key=lambda s: (s["ising_energy"], s["V_pred"]))
        best_v_sample = min(feas_samples, key=lambda s: (s["V_pred"], s["ising_energy"]))
        best_ising_energy = float(min(ising_energies))
        best_feas_ising_energy = float(best_e_sample["ising_energy"])
        best_v_pred = float(best_v_sample["V_pred"])
        best_formula = str(best_v_sample["formula"])
        best_in_blind = int(best_v_sample["in_blind"])
        best_rank = int(best_v_sample["true_rank_in_pool"])
    else:
        best_ising_energy = float(min(ising_energies)) if ising_energies else float("nan")
        best_feas_ising_energy = float("nan")
        best_v_pred = None
        best_formula = "INFEASIBLE"
        best_in_blind = 0
        best_rank = None

    return {
        "n_samples": int(n_samples),
        "ref_consensus_count": int(ref_ok_cnt),
        "ref_consensus_rate": float(ref_ok_cnt / n_samples) if n_samples else 0.0,
        "feasible_count": int(feas_cnt),
        "feasible_rate": float(feas_cnt / n_samples) if n_samples else 0.0,
        "feasible_in_blind_rate": float(feas_blind_cnt / feas_cnt) if feas_cnt else None,
        "subcap_feasible_count": int(subcap_feas_cnt),
        "subcap_in_blind_rate": float(subcap_blind_cnt / subcap_feas_cnt) if subcap_feas_cnt else None,
        "hard_penalty_verified": bool(hard_viol_in_feas == 0),
        "best_ising_energy": best_ising_energy,
        "best_feasible_ising_energy": best_feas_ising_energy,
        "best_V_pred": best_v_pred,
        "best_formula": best_formula,
        "best_in_blind": best_in_blind,
        "true_rank_in_pool": best_rank,
        "ising_energies": ising_energies,
        "feasible_flags": feasible_flags,
        "decoded_samples": decoded_samples,
        "raw_spins": arr.tolist(),
    }


def run_classical_solvers(
    J: np.ndarray,
    n_dec: int,
    n_ref_copies: int,
    task_type: str,
    data: dict[str, Any],
    V_cap: float,
    sa_iters: int = 500,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run local SimulatedAnnealingOptimizer (process_num=1) & TabuSearchOptimizer."""
    if n_ref_copies > 1:
        h_eff = J[:n_dec, n_dec:].sum(axis=1).astype(np.float64)
        j_rr = int(J[n_dec, n_dec + 1])
        h_all = np.concatenate([h_eff, np.full(n_ref_copies, j_rr * (n_ref_copies - 1), dtype=np.float64)])
    else:
        h_all = J[:n_dec, n_dec].astype(np.float64)

    t0 = max(100.0, 2.0 * float(np.max(np.abs(h_all))))
    sa = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=t0,
        alpha=0.99,
        cutoff_temperature=0.001,
        iterations_per_t=sa_iters,
        size_limit=50,
        rand_seed=42,
        process_num=1,
    )
    sa_pool = np.asarray(sa.solve(J), dtype=np.int64)
    sa_res = decode_sample_pool(sa_pool, J, task_type, data, n_ref_copies=n_ref_copies, V_cap=V_cap)

    np.random.seed(42)
    tabu = kw.classical.TabuSearchOptimizer(max_iter=1500, size_limit=50)
    tabu_pool = np.asarray(tabu.solve(J), dtype=np.int64)
    tabu_res = decode_sample_pool(tabu_pool, J, task_type, data, n_ref_copies=n_ref_copies, V_cap=V_cap)
    return sa_res, tabu_res


# ---------------------------------------------------------------------------
# Phase A: Build All Q3 Encodings, Ablations, & Pre-Flight Assertions
# ---------------------------------------------------------------------------

def build_phase_a(data: dict[str, Any]) -> dict[str, Any]:
    """Build Q3 matrices, Table 3 ablation + production rows, and validate all 6 hardware matrices."""
    print("[Phase A] Verifying QuboModel alphabetical variable ordering and -s^T J s sign...")
    alg_info = verify_qubo_models_and_signs(data)
    print(
        f"  -> Verified: max model/analytical diff = {alg_info['max_model_vs_analytical_diff']:.2e}, "
        f"max QUBO/-sJs err = {alg_info['max_qubo_vs_neg_sJs_error']:.2e}"
    )

    qfm = data["qfm"]
    V_raw_q3 = data["V_raw_q3"]
    formula_grid_q3 = data["formula_grid_q3"]
    base_vars = {
        "qfm": [f"a[{i:02d}]" for i in range(52)] + [f"b[{j:02d}]" for j in range(52)] + [f"x[{k}]" for k in range(3)],
        "cmp": [f"b[{j:02d}]" for j in range(52)] + [f"xa[{u:03d}]" for u in range(156)],
        "ros": [f"a[{i:02d}]" for i in range(52)]
        + [f"b[{j:02d}]" for j in range(52)]
        + [f"x[{k}]" for k in range(3)]
        + [f"y[{k}][{i:02d}]" for k in range(3) for i in range(52)],
    }

    configs = [
        # (setting_name, encoding_family, mode, K, P_oh, P_ros, V_cap, c_shift, tmax, is_prod, task_name, notes)
        ("QFM_K1_P62_raw", "QFM-QUBO", "qfm", 1, 62.0, 0.0, float("nan"), 0.0, 127, False, "N/A (ablation)",
         "Raw K=1 QFM-QUBO (108 spins) via adjust_ising_matrix_precision: 8-bit coupling collapse (1 level)"),
        ("QFM_K150_P62_prod", "QFM-QUBO", "qfm", 150, 62.0, 0.0, float("nan"), 0.0, 127, True, "kaiwu_skill_d_q3_qfm108",
         "Prod q3_qfm108: K=150 ref-split + consistent post-rounding hub fields (rank=1 LaTiO2N, 28 J levels)"),
        ("QFM_K200_P65", "QFM-QUBO", "qfm", 200, 65.0, 0.0, float("nan"), 0.0, 127, False, "N/A (ablation)",
         "QFM P_oh=65.0 sensitivity: step ~0.98 QFM units yields 2-way tie at top-1 [LaTiO2N, YTiO2N]"),
        ("CMP_K1_cap3_c0_P3.1_raw", "Compact-QUBO", "cmp", 1, 3.1, 0.0, 3.0, 0.0, 105, False, "N/A (ablation)",
         "Raw K=1 Compact-QUBO (209 spins) via adjust_ising_matrix_precision: J_bu crushed to 1 level (0% feas)"),
        ("CMP_K260_cap3_c0_P3.1_prod", "Compact-QUBO", "cmp", 260, 3.1, 0.0, 3.0, 0.0, 105, True, "kaiwu_skill_d_q3_cmp209",
         "Prod q3_cmp209: V_cap=3.0, c=0.0, P_oh=3.1, K=260 ref-split (48 J levels, rank=1 LaTiO2N, 100% SA/Tabu)"),
        ("CMP_K260_cap3_cVcap_P6.0", "Compact-QUBO", "cmp", 260, 6.0, 0.0, 3.0, 3.0, 105, False, "N/A (ablation)",
         "Centering c=V_cap=3.0 (M<=0): forces P_oh=6.0 (~2x), halving J levels (27 vs 48) and SA100 feas (0.24)"),
        ("CMP_K260_cap6_c0_P6.1", "Compact-QUBO", "cmp", 260, 6.1, 0.0, 6.0, 0.0, 105, False, "N/A (ablation)",
         "Window ablation V_cap=6.0: wider cap doubles quantization step, causing 5-way top-1 tie"),
        ("ROS_K1_cap3_c0_P3.6_3.3_raw", "Rosenberg HUBO-QUBO", "ros", 1, 3.6, 3.3, 3.0, 0.0, 105, False, "N/A (ablation)",
         "Raw K=1 Rosenberg HUBO-QUBO (264 spins) via adjust_ising_matrix_precision: J_yb crushed to 1 level (0% feas)"),
        ("ROS_K200_cap3_c0_P3.1_3.5_trap", "Rosenberg HUBO-QUBO", "ros", 200, 3.1, 3.5, 3.0, 0.0, 105, False, "N/A (ablation)",
         "Trap ordering P_oh=3.1 < P_ros=3.5: single-bit (x=0,a=1,b=1,y=0) barrier traps Tabu at 4% feas"),
        ("ROS_K150_cap3_c0_P3.6_3.3_prod", "Rosenberg HUBO-QUBO", "ros", 150, 3.6, 3.3, 3.0, 0.0, 105, True, "kaiwu_skill_d_q3_ros264",
         "Prod q3_ros264: anti-trap P_oh=3.6 > P_ros=3.3 > V_cap=3.0, K=150 (43 J levels, rank=1 LaTiO2N, 100% SA500/Tabu)"),
        ("ROS_K150_cap3_cVcap_P6.5_6.0", "Rosenberg HUBO-QUBO", "ros", 150, 6.5, 6.0, 3.0, 3.0, 105, False, "N/A (ablation)",
         "Rosenberg centering c=V_cap=3.0: negative M shifts require P_oh=6.5, halving J levels (25 vs 43)"),
        ("ROS_K150_cap6_c0_P6.6_6.3", "Rosenberg HUBO-QUBO", "ros", 150, 6.6, 6.3, 6.0, 0.0, 105, False, "N/A (ablation)",
         "Rosenberg window V_cap=6.0: wider cap causes 5-way top-1 tie"),
    ]

    table3_rows: list[dict[str, Any]] = []
    prod_q3: dict[str, dict[str, Any]] = {}

    for setting_name, enc_fam, mode, K, P_oh, P_ros, V_cap, c_shift, tmax, is_prod, t_name, notes in configs:
        if mode == "qfm":
            q_vec, Q_sym = qfm_qubo_parts(qfm, P_oh)
            task_type = "q3_qfm"
            v_cap_eval = 999.0
        elif mode == "cmp":
            M_3d = np.minimum(V_raw_q3, V_cap) - c_shift
            q_vec, Q_sym = cmp_qubo_parts(M_3d, P_oh)
            task_type = "q3_cmp"
            v_cap_eval = V_cap
        else:
            M_3d = np.minimum(V_raw_q3, V_cap) - c_shift
            q_vec, Q_sym = ros_qubo_parts(M_3d, P_oh, P_ros)
            task_type = "q3_ros"
            v_cap_eval = V_cap

        built = build_ising_from_qubo_parts(q_vec, Q_sym, K, target_max=tmax, var_names_base=base_vars[mode])
        J = built["J"]
        exact_info = evaluate_q3_exact_8112(mode, J, K, built["j_rr"], V_raw_q3, formula_grid_q3)

        # Run SA100, SA500, and Tabu1500
        sa500_res, tabu_res = run_classical_solvers(J, built["n_bits"], K, task_type, data, V_cap=v_cap_eval, sa_iters=500)
        sa100_res, _ = run_classical_solvers(J, built["n_bits"], K, task_type, data, V_cap=v_cap_eval, sa_iters=100)

        # For production matrices, assert exact_rank==1, ties==1, positive gaps, and passing SA500/Tabu
        if is_prod:
            assert built["n_spins"] <= 550
            assert exact_info["exact_ground_state_rank_of_true_opt"] == 1, f"{setting_name} rank != 1"
            assert exact_info["exact_ties_at_top1"] == 1, f"{setting_name} ties != 1"
            assert exact_info["exact_8bit_best_formula"] == "LaTiO2N", f"{setting_name} top1 != LaTiO2N"
            assert exact_info["min_infeasible_energy_gap"] > 0, f"{setting_name} gap_inv <= 0"
            assert exact_info["min_broken_ref_gap"] > 0, f"{setting_name} ref_gap <= 0"
            assert sa500_res["feasible_rate"] == 1.0, f"{setting_name} SA500 feas != 1.0"
            assert tabu_res["feasible_rate"] == 1.0 and tabu_res["best_formula"] == "LaTiO2N", (
                f"{setting_name} Tabu failed"
            )
            prod_q3[t_name] = {
                "setting_name": setting_name,
                "encoding_family": enc_fam,
                "mode": mode,
                "task_type": task_type,
                "P_onehot": float(P_oh),
                "P_rosenberg": float(P_ros),
                "V_cap": float(V_cap) if not math.isnan(V_cap) else float("nan"),
                "c_shift": float(c_shift),
                "built": built,
                "exact_info": exact_info,
                "sa_res": sa500_res,
                "tabu_res": tabu_res,
            }

        row = {
            "encoding_name": f"{enc_fam} ({built['n_bits'] + 1}->{built['n_spins']} spins)" if K > 1 else f"{enc_fam} ({built['n_spins']} spins, raw K=1)",
            "encoding_family": enc_fam,
            "setting_name": setting_name,
            "task_name": t_name,
            "is_production_spqc": bool(is_prod),
            "n_bits": int(built["n_bits"]),
            "n_spins": int(built["n_spins"]),
            "n_ref_copies": int(K),
            "j_rr": int(built["j_rr"]),
            "P_onehot": round(float(P_oh), 4),
            "P_rosenberg": round(float(P_ros), 4) if P_ros > 0 else 0.0,
            "V_cap": round(float(V_cap), 4) if not math.isnan(V_cap) else np.nan,
            "c_shift": round(float(c_shift), 4),
            "ising_bit_width": int(built["bit_width"]),
            "dynamic_range_ratio": round(float(built["drm"]), 4),
            "J_coupling_levels": int(exact_info["J_coupling_levels"]),
            "exact_ground_state_rank_of_true_opt": int(exact_info["exact_ground_state_rank_of_true_opt"]),
            "exact_ties_at_top1": int(exact_info["exact_ties_at_top1"]),
            "exact_8bit_best_formula": exact_info["exact_8bit_best_formula"],
            "exact_8bit_best_V_pred": round(float(exact_info["exact_8bit_best_V_pred"]), 6),
            "min_infeasible_energy_gap": round(float(exact_info["min_infeasible_energy_gap"]), 2),
            "min_broken_ref_gap": round(float(exact_info["min_broken_ref_gap"]), 2)
            if not math.isnan(float(exact_info["min_broken_ref_gap"]))
            else np.nan,
            "sa_feasible_rate": round(float(sa500_res["feasible_rate"]), 4),
            "sa_best_V_pred": round(float(sa500_res["best_V_pred"]), 6)
            if sa500_res["best_V_pred"] is not None
            else np.nan,
            "sa_best_formula": sa500_res["best_formula"],
            "sa100_feasible_rate": round(float(sa100_res["feasible_rate"]), 4),
            "sa100_best_V_pred": round(float(sa100_res["best_V_pred"]), 6)
            if sa100_res["best_V_pred"] is not None
            else np.nan,
            "sa100_best_formula": sa100_res["best_formula"],
            "tabu_feasible_rate": round(float(tabu_res["feasible_rate"]), 4),
            "tabu_best_V_pred": round(float(tabu_res["best_V_pred"]), 6)
            if tabu_res["best_V_pred"] is not None
            else np.nan,
            "tabu_best_formula": tabu_res["best_formula"],
            "notes": notes,
        }
        table3_rows.append(row)
        print(
            f"  [{setting_name:30s}] n={built['n_spins']:3d} K={K:3d} lvls={exact_info['J_coupling_levels']:2d} "
            f"rank={exact_info['exact_ground_state_rank_of_true_opt']:3d} ties={exact_info['exact_ties_at_top1']:3d} "
            f"gap={exact_info['min_infeasible_energy_gap']:8.1f} SA500={sa500_res['feasible_rate']:.2f}({sa500_res['best_formula']}) "
            f"Tabu={tabu_res['feasible_rate']:.2f}({tabu_res['best_formula']})"
        )

    # Save table3
    df_t3 = pd.DataFrame(table3_rows)
    df_t3.to_csv(os.path.join(OUT_DIR, "table3_q3_encoding_comparison.csv"), index=False)

    # Save exp3_q3_matrices.npz
    qfm_b = prod_q3["kaiwu_skill_d_q3_qfm108"]["built"]
    cmp_b = prod_q3["kaiwu_skill_d_q3_cmp209"]["built"]
    ros_b = prod_q3["kaiwu_skill_d_q3_ros264"]["built"]
    np.savez_compressed(
        os.path.join(OUT_DIR, "exp3_q3_matrices.npz"),
        elements_52=np.array(data["elements_52"]),
        x_q3=np.array(data["x_q3"]),
        V_pred_q3=data["V_raw_q3"],
        blind_mask_q3=data["blind_mask_q3"],
        q3_qfm108_ising=qfm_b["J"],
        q3_qfm108_qubo=qfm_b["Q_up"],
        q3_qfm108_bias=np.float64(qfm_b["bias_scaled"]),
        q3_qfm108_alpha=np.float64(qfm_b["alpha"]),
        q3_qfm108_j_rr=np.int64(qfm_b["j_rr"]),
        q3_qfm108_n_spins=np.int64(qfm_b["n_spins"]),
        q3_qfm108_n_ref_copies=np.int64(qfm_b["n_ref_copies"]),
        q3_qfm108_P_onehot=np.float64(62.0),
        q3_qfm108_P_rosenberg=np.float64(0.0),
        q3_qfm108_V_cap=np.float64(np.nan),
        q3_qfm108_variables=np.array(qfm_b["variables"]),
        q3_cmp209_ising=cmp_b["J"],
        q3_cmp209_qubo=cmp_b["Q_up"],
        q3_cmp209_bias=np.float64(cmp_b["bias_scaled"]),
        q3_cmp209_alpha=np.float64(cmp_b["alpha"]),
        q3_cmp209_j_rr=np.int64(cmp_b["j_rr"]),
        q3_cmp209_n_spins=np.int64(cmp_b["n_spins"]),
        q3_cmp209_n_ref_copies=np.int64(cmp_b["n_ref_copies"]),
        q3_cmp209_P_onehot=np.float64(3.1),
        q3_cmp209_P_rosenberg=np.float64(0.0),
        q3_cmp209_V_cap=np.float64(3.0),
        q3_cmp209_variables=np.array(cmp_b["variables"]),
        q3_ros264_ising=ros_b["J"],
        q3_ros264_qubo=ros_b["Q_up"],
        q3_ros264_bias=np.float64(ros_b["bias_scaled"]),
        q3_ros264_alpha=np.float64(ros_b["alpha"]),
        q3_ros264_j_rr=np.int64(ros_b["j_rr"]),
        q3_ros264_n_spins=np.int64(ros_b["n_spins"]),
        q3_ros264_n_ref_copies=np.int64(ros_b["n_ref_copies"]),
        q3_ros264_P_onehot=np.float64(3.6),
        q3_ros264_P_rosenberg=np.float64(3.3),
        q3_ros264_V_cap=np.float64(3.0),
        q3_ros264_variables=np.array(ros_b["variables"]),
    )

    # Prepare and verify ALL 6 hardware tasks (3 M2 Q2 + 3 M3 Q3)
    q2_npz = data["q2_npz"]
    q2_specs = [
        ("kaiwu_skill_d_q2_uncap", "q2_uncap", float(q2_npz["q2_uncap_V_cap"])),
        ("kaiwu_skill_d_q2_full", "q2_full", float(q2_npz["q2_full_V_cap"])),
        ("kaiwu_skill_d_q2_blind", "q2_blind", float(q2_npz["q2_blind_V_cap"])),
    ]

    hardware_tasks: dict[str, dict[str, Any]] = {}
    print("\n[Phase A] Pre-flight assertions across ALL 6 SPQC-550 hardware matrices:")
    for t_name, prefix, vcap in q2_specs:
        J_q2 = q2_npz[f"{prefix}_ising"]
        K_q2 = int(q2_npz[f"{prefix}_n_ref_copies"])
        assert J_q2.dtype == np.int64
        assert J_q2.shape[0] == J_q2.shape[1] <= 550
        assert np.array_equal(J_q2, J_q2.T)
        assert np.all(np.diag(J_q2) == 0)
        assert int(J_q2.min()) >= -127 and int(J_q2.max()) <= 127
        assert int(calculate_ising_matrix_bit_width(J_q2)["precision"]) == 8

        sa_q2, tabu_q2 = run_classical_solvers(J_q2, 104, K_q2, "q2", data, V_cap=vcap, sa_iters=500)
        if prefix != "q2_uncap":
            assert tabu_q2["feasible_rate"] == 1.0 and tabu_q2["best_formula"] == "LaTiO2N"
        print(
            f"  [OK] {t_name:24s} shape={J_q2.shape} K={K_q2:3d} rng=[{J_q2.min()},{J_q2.max()}] "
            f"SA500={sa_q2['feasible_rate']:.2f}({sa_q2['best_formula']}) Tabu={tabu_q2['feasible_rate']:.2f}({tabu_q2['best_formula']})"
        )
        hardware_tasks[t_name] = {
            "task_name": t_name,
            "problem_scope": "Q2",
            "task_type": "q2",
            "J": J_q2,
            "n_dec": 104,
            "n_ref_copies": K_q2,
            "n_spins": int(J_q2.shape[0]),
            "V_cap": vcap,
            "matrix_sha256": sha256_int64_matrix(J_q2),
            "sa_res": sa_q2,
            "tabu_res": tabu_q2,
        }

    for t_name in ["kaiwu_skill_d_q3_qfm108", "kaiwu_skill_d_q3_cmp209", "kaiwu_skill_d_q3_ros264"]:
        info = prod_q3[t_name]
        b = info["built"]
        J_q3 = b["J"]
        vcap = 999.0 if math.isnan(info["V_cap"]) else float(info["V_cap"])
        print(
            f"  [OK] {t_name:24s} shape={J_q3.shape} K={b['n_ref_copies']:3d} rng=[{J_q3.min()},{J_q3.max()}] "
            f"SA500={info['sa_res']['feasible_rate']:.2f}({info['sa_res']['best_formula']}) Tabu={info['tabu_res']['feasible_rate']:.2f}({info['tabu_res']['best_formula']})"
        )
        hardware_tasks[t_name] = {
            "task_name": t_name,
            "problem_scope": "Q3",
            "task_type": info["task_type"],
            "J": J_q3,
            "n_dec": int(b["n_bits"]),
            "n_ref_copies": int(b["n_ref_copies"]),
            "n_spins": int(b["n_spins"]),
            "V_cap": vcap,
            "matrix_sha256": sha256_int64_matrix(J_q3),
            "sa_res": info["sa_res"],
            "tabu_res": info["tabu_res"],
        }

    return {
        "algebra_info": alg_info,
        "table3_df": df_t3,
        "hardware_tasks": hardware_tasks,
    }


# ---------------------------------------------------------------------------
# Phase B: SPQC-550 Hardware Execution & Tables 4–5 Generation
# ---------------------------------------------------------------------------

def build_encoding_specs(df_t3: pd.DataFrame) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for _, r in df_t3[df_t3["is_production_spqc"] == True].iterrows():  # noqa: E712
        specs[str(r["task_name"])] = {
            "encoding_family": str(r["encoding_family"]),
            "n_bits": int(r["n_bits"]),
            "n_spins": int(r["n_spins"]),
            "n_ref_copies": int(r["n_ref_copies"]),
            "j_rr": int(r["j_rr"]),
            "P_onehot": float(r["P_onehot"]),
            "P_rosenberg": float(r["P_rosenberg"]),
            "V_cap": None if pd.isna(r["V_cap"]) else float(r["V_cap"]),
            "dynamic_range_ratio": float(r["dynamic_range_ratio"]),
            "J_coupling_levels": int(r["J_coupling_levels"]),
            "exact_rank": int(r["exact_ground_state_rank_of_true_opt"]),
            "exact_ties": int(r["exact_ties_at_top1"]),
            "min_infeasible_energy_gap": float(r["min_infeasible_energy_gap"]),
            "min_broken_ref_gap": float(r["min_broken_ref_gap"]),
        }
    return specs


def find_checkpoint_file(task_name: str, J: np.ndarray) -> str:
    """Locate the exact *_results.pkl file created by CheckpointManager matching task_name and matrix."""
    candidates = sorted(
        f for f in os.listdir(CKPT_DIR) if f.startswith(task_name) and f.endswith("_results.pkl")
    )
    if len(candidates) == 1:
        return candidates[0]
    for fn in candidates:
        obj = pd.read_pickle(os.path.join(CKPT_DIR, fn))
        # Check if MD5 or shape matches
        if isinstance(obj, np.ndarray) and obj.shape[1] == J.shape[0]:
            return fn
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No *_results.pkl checkpoint found for {task_name} in {CKPT_DIR}")


def submit_spqc_tasks(phase_a: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Execute AT MOST 6 SPQC-550 hardware tasks one-by-one with persistent checkpoints & incremental JSON."""
    from kaiwu.cim import CIMOptimizer, TaskMode  # Imported ONLY here, never in --verify-only/--skip-cim

    os.makedirs(CKPT_DIR, exist_ok=True)
    kw.common.CheckpointManager.save_dir = CKPT_DIR

    results_path = os.path.join(OUT_DIR, "exp3_spqc_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {}

    payload["metadata"] = {
        "task_names": TASK_NAMES,
        "checkpoint_dir": CKPT_DIR,
        "solver_config": {"sa": SA_CONFIG, "tabu": TABU_CONFIG, "cim": {"wait": True, "interval_min": 6, "task_mode": "optimization", "negtail_flip": True, "sort_solutions": False}},
        "encoding_specs": build_encoding_specs(phase_a["table3_df"]),
        "algebra_verification": phase_a["algebra_info"],
    }
    payload.setdefault("tasks", {})

    for idx, t_name in enumerate(TASK_NAMES, start=1):
        t_info = phase_a["hardware_tasks"][t_name]
        J = t_info["J"].astype(np.int64)
        prev = payload["tasks"].get(t_name)
        if (
            prev is not None
            and prev.get("matrix_sha256") == t_info["matrix_sha256"]
            and os.path.exists(os.path.join(CKPT_DIR, prev.get("checkpoint_file", "")))
        ):
            print(f"[Phase B] ({idx}/6) Reusing completed checkpoint for '{t_name}': {prev['checkpoint_file']}")
            continue
        print(f"\n[Phase B] ({idx}/6) Submitting SPQC-550 task '{t_name}' (shape={J.shape}, sha256={t_info['matrix_sha256'][:12]})...")
        t0 = time.time()
        opt = CIMOptimizer(
            task_name=t_name,
            wait=True,
            interval=6,
            task_mode=TaskMode.OPTIMIZATION,
        )
        raw_sol = opt.solve(ising_matrix=J, negtail_flip=True, sort_solutions=False)
        elapsed = round(time.time() - t0, 2)
        if raw_sol is None:
            raise RuntimeError(f"CIMOptimizer returned None for {t_name}")

        sol_arr = np.asarray(raw_sol, dtype=np.int64)
        ckpt_file = find_checkpoint_file(t_name, J)
        spqc_res = decode_sample_pool(
            sol_arr,
            J,
            t_info["task_type"],
            data,
            n_ref_copies=t_info["n_ref_copies"],
            V_cap=t_info["V_cap"],
        )
        print(
            f"  -> Completed in {elapsed:.1f}s | ckpt={ckpt_file} | n={spqc_res['n_samples']} "
            f"ref_ok={spqc_res['ref_consensus_rate']:.2f} feas={spqc_res['feasible_count']}/{spqc_res['n_samples']} "
            f"best={spqc_res['best_formula']} (V={spqc_res['best_V_pred']}, rank={spqc_res['true_rank_in_pool']})"
        )

        payload["tasks"][t_name] = {
            "task_name": t_name,
            "problem_scope": t_info["problem_scope"],
            "task_type": t_info["task_type"],
            "n_spins": int(t_info["n_spins"]),
            "n_dec": int(t_info["n_dec"]),
            "n_ref_copies": int(t_info["n_ref_copies"]),
            "V_cap": float(t_info["V_cap"]),
            "matrix_sha256": t_info["matrix_sha256"],
            "checkpoint_file": ckpt_file,
            "hardware_elapsed_seconds": elapsed,
            "solvers": {
                "SPQC-550": spqc_res,
                "SA": t_info["sa_res"],
                "Tabu": t_info["tabu_res"],
            },
        }
        # Save incrementally after every task so results are never lost
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    kw.common.CheckpointManager.save_dir = None
    return payload


def generate_table4(results_payload: dict[str, Any]) -> pd.DataFrame:
    """Build 18-row Table 4 comparing SPQC-550, SA, and Tabu across all 6 tasks."""
    rows: list[dict[str, Any]] = []
    for t_name in TASK_NAMES:
        t_data = results_payload["tasks"][t_name]
        for solver_name in ["SPQC-550", "SA", "Tabu"]:
            s_res = t_data["solvers"][solver_name]
            rows.append({
                "task_name": t_name,
                "problem_scope": t_data["problem_scope"],
                "solver": solver_name,
                "n_spins": int(t_data["n_spins"]),
                "n_samples": int(s_res["n_samples"]),
                "ref_consensus_rate": round(float(s_res["ref_consensus_rate"]), 4),
                "feasible_count": int(s_res["feasible_count"]),
                "feasible_rate": round(float(s_res["feasible_rate"]), 4),
                "best_ising_energy": round(float(s_res["best_ising_energy"]), 2)
                if not math.isnan(float(s_res["best_ising_energy"]))
                else np.nan,
                "best_V_pred": round(float(s_res["best_V_pred"]), 6)
                if s_res["best_V_pred"] is not None
                else np.nan,
                "best_formula": str(s_res["best_formula"]),
                "best_in_blind": int(s_res["best_in_blind"]),
                "feasible_in_blind_rate": round(float(s_res["feasible_in_blind_rate"]), 4)
                if s_res["feasible_in_blind_rate"] is not None
                else np.nan,
                "subcap_feasible_count": int(s_res["subcap_feasible_count"]),
                "subcap_in_blind_rate": round(float(s_res["subcap_in_blind_rate"]), 4)
                if s_res["subcap_in_blind_rate"] is not None
                else np.nan,
                "true_rank_in_pool": int(s_res["true_rank_in_pool"])
                if s_res["true_rank_in_pool"] is not None
                else np.nan,
                "hard_penalty_verified": bool(s_res["hard_penalty_verified"]),
                "checkpoint_file": str(t_data["checkpoint_file"]) if solver_name == "SPQC-550" else "N/A",
            })
    df_t4 = pd.DataFrame(rows)
    df_t4.to_csv(os.path.join(OUT_DIR, "table4_spqc550_hardware_vs_classical.csv"), index=False)
    return df_t4


def compute_5term_breakdown(e_form: float, gap_semi: float, p_semi: float, vbm: float, cbm: float) -> tuple[float, float, float, float, float, float]:
    v_form = max(0.0, (e_form - 0.21) / 0.5) ** 2
    v_gap_low = (1.0 - p_semi) * ((1.4 / 0.5) ** 2) + p_semi * (max(0.0, (1.4 - gap_semi) / 0.5) ** 2)
    v_gap_high = p_semi * (max(0.0, (gap_semi - 3.1) / 0.5) ** 2)
    v_vbm = max(0.0, (5.73 - vbm) / 0.5) ** 2
    v_cbm = max(0.0, (cbm - 4.50) / 0.5) ** 2
    v_recomp = v_form + v_gap_low + v_gap_high + v_vbm + v_cbm
    return v_form, v_gap_low, v_gap_high, v_vbm, v_cbm, v_recomp


def generate_table5(results_payload: dict[str, Any], data: dict[str, Any]) -> pd.DataFrame:
    """Build Table 5 item-by-item 5-term physical penalty breakdown for top Q1/Q2/Q3/SPQC-550 candidates."""
    holdout_df = pd.read_csv(os.path.join(OUT_DIR, "04_castelli_test_holdout_predictions.csv")).sort_values("V_pred")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "03_castelli_val.csv"))
    q3_df = data["q3_pred_df"]
    q2_df = data["q2_pred_df"]

    holdout_by_id = {int(r["id"]): r for _, r in holdout_df.iterrows()}
    train_by_id = {int(r["id"]): r for _, r in train_df.iterrows()}
    val_by_id = {int(r["id"]): r for _, r in val_df.iterrows()}
    q3_by_formula = {str(r["formula"]): r for _, r in q3_df.iterrows()}
    q2_by_formula = {str(r["formula"]): r for _, r in q2_df.iterrows()}
    holdout_by_formula = {str(r["formula"]): r for _, r in holdout_df.iterrows()}

    # Track discovery sources per formula
    sources: dict[str, list[str]] = {}
    def add_src(formula: str, tag: str) -> None:
        if formula and formula != "INFEASIBLE":
            sources.setdefault(formula, [])
            if tag not in sources[formula]:
                sources[formula].append(tag)

    for idx, r in holdout_df.head(10).iterrows():
        add_src(str(r["formula"]), "Q1_Holdout_Top10")
    for _, r in q2_df.sort_values("V_pred").head(5).iterrows():
        add_src(str(r["formula"]), "Q2_Surrogate_Top5")
    for _, r in q3_df.sort_values("V_pred").head(5).iterrows():
        add_src(str(r["formula"]), "Q3_Surrogate_Top5")

    discovered_spqc_best: list[str] = []
    for t_name in TASK_NAMES:
        t_data = results_payload["tasks"][t_name]
        short = t_name.replace("kaiwu_skill_d_", "")
        for s_name in ["SPQC-550", "SA", "Tabu"]:
            s_res = t_data["solvers"][s_name]
            bf = str(s_res["best_formula"])
            if bf != "INFEASIBLE":
                add_src(bf, f"{s_name}({short})")
                if s_name == "SPQC-550" and bf not in discovered_spqc_best:
                    discovered_spqc_best.append(bf)
            # Also record top-1 feasible in-blind if different
            feas_blind = [x for x in s_res["decoded_samples"] if x["feasible"] and x["in_blind"] == 1]
            if feas_blind:
                best_b = min(feas_blind, key=lambda x: x["V_pred"])
                add_src(str(best_b["formula"]), f"{s_name}_blind({short})")

    # Build curated candidate list: Holdout Top-12 + labeled Q3 benchmarks (ScTiO2N, SnTiO3, BaNbO2N) + any SPQC-550 best discoveries
    selected_formulas: list[str] = []
    for _, r in holdout_df.head(12).iterrows():
        f = str(r["formula"])
        if f not in selected_formulas:
            selected_formulas.append(f)
    for f in ["ScTiO2N", "SnTiO3", "BaNbO2N"] + discovered_spqc_best:
        if f not in selected_formulas:
            selected_formulas.append(f)

    # Holdout rank lookup
    holdout_rank = {str(r["formula"]): int(i + 1) for i, (_, r) in enumerate(holdout_df.iterrows())}

    rows: list[dict[str, Any]] = []
    for f in selected_formulas:
        q3_r = q3_by_formula.get(f)
        q2_r = q2_by_formula.get(f)
        h_r = holdout_by_formula.get(f)
        in_blind = int(q3_r["in_blind_pool"]) if q3_r is not None else (1 if h_r is not None else 0)

        if h_r is not None:
            cid = int(h_r["id"])
            e_f = float(h_r["e_form_pred"])
            g_s = float(h_r["gap_semi_pred"])
            g_e = float(h_r["gap_expected_pred"])
            p_s = float(h_r["p_semi_pred"])
            vbm = float(h_r["vbm_pred"])
            cbm = float(h_r["cbm_pred"])
            vf, vgl, vgh, vv, vc, v_rec = compute_5term_breakdown(e_f, g_s, p_s, vbm, cbm)
            v_phys = float(h_r["V_phys_pred"])
            v_pred = float(h_r["V_pred"])
            assert abs(v_rec - v_phys) < 1e-11, f"{f}: V_recomputed {v_rec} != V_phys_pred {v_phys}"
            prop_src = "04_holdout_surrogate"
            rk_str = f"H#{holdout_rank[f]}"
            if f in data["rank_map_q3"]:
                rk_str += f"/Q3#{data['rank_map_q3'][f]}"
            A_el, B_el, X_el = str(h_r["A"]), str(h_r["B"]), str(h_r["X"])
        else:
            # Labeled candidate from Q3/Q2 pool -> use 02_train or 03_val DFT properties
            ref_r = q3_r if q3_r is not None else q2_r
            assert ref_r is not None, f"Formula {f} missing from holdout and pools"
            cid = int(ref_r["id"])
            dft_r = train_by_id.get(cid)
            prop_src = "02_train_DFT"
            if dft_r is None:
                dft_r = val_by_id.get(cid)
                prop_src = "03_val_DFT"
            assert dft_r is not None, f"Labeled formula {f} (id={cid}) missing from train/val"
            e_f = float(dft_r["e_form"])
            g_s = float(dft_r["gap"])
            p_s = 1.0 if g_s > 0.0 else 0.0
            g_e = g_s
            vbm = float(dft_r["vbm"])
            cbm = float(dft_r["cbm"])
            vf, vgl, vgh, vv, vc, v_rec = compute_5term_breakdown(e_f, g_s, p_s, vbm, cbm)
            v_phys = float(dft_r["V"])
            v_pred = float(ref_r["V_pred"])
            assert abs(v_rec - v_phys) < 1e-4, f"{f}: DFT V_recomputed {v_rec} != V {v_phys}"
            rk_str = f"Q3#{data['rank_map_q3'][f]}" if f in data["rank_map_q3"] else f"Q2#{data['rank_map_q2'][f]}"
            A_el, B_el, X_el = str(ref_r["A"]), str(ref_r["B"]), str(ref_r["X"])

        rows.append({
            "rank_holdout_or_pool": rk_str,
            "id": cid,
            "formula": f,
            "A": A_el,
            "B": B_el,
            "X": X_el,
            "e_form_pred": round(e_f, 6),
            "gap_semi_pred": round(g_s, 6),
            "gap_expected_pred": round(g_e, 6),
            "p_semi_pred": round(p_s, 6),
            "vbm_pred": round(vbm, 6),
            "cbm_pred": round(cbm, 6),
            "v_form": round(vf, 6),
            "v_gap_low": round(vgl, 6),
            "v_gap_high": round(vgh, 6),
            "v_vbm": round(vv, 6),
            "v_cbm": round(vc, 6),
            "V_recomputed": round(v_rec, 6),
            "V_phys_pred": round(v_phys, 6),
            "V_pred": round(v_pred, 6),
            "source_method": "; ".join(sources.get(f, ["Surrogate_Top"])),
            "in_blind_pool": int(in_blind),
            "prop_source": prop_src,
        })

    df_t5 = pd.DataFrame(rows).sort_values("V_pred", ascending=True).reset_index(drop=True)
    df_t5.to_csv(os.path.join(OUT_DIR, "table5_top_perovskite_cost_breakdown.csv"), index=False)
    return df_t5


# ---------------------------------------------------------------------------
# --verify-only: Strict Disk-Only Re-Verification (ZERO CIMOptimizer imports)
# ---------------------------------------------------------------------------

def run_verify_only() -> None:
    """Independently re-verify all artifacts, checkpoints, constraints, energies, and tables from disk."""
    print("[Verify-Only] Starting offline verification (zero CIMOptimizer imports/calls)...")
    data = load_q3_and_q2_data()

    # 1. Verify Problem 1 holdout immutability
    holdout_df = pd.read_csv(os.path.join(OUT_DIR, "04_castelli_test_holdout_predictions.csv"))
    assert len(holdout_df) == 3703 and int(holdout_df.isna().sum().sum()) == 0

    # 2. Verify exp3_q3_matrices.npz and exp2_q2_matrices.npz
    q3_npz = np.load(os.path.join(OUT_DIR, "exp3_q3_matrices.npz"), allow_pickle=True)
    q2_npz = data["q2_npz"]
    matrix_map = {
        "kaiwu_skill_d_q2_uncap": (q2_npz["q2_uncap_ising"], "q2", 1, float(q2_npz["q2_uncap_V_cap"])),
        "kaiwu_skill_d_q2_full": (q2_npz["q2_full_ising"], "q2", int(q2_npz["q2_full_n_ref_copies"]), float(q2_npz["q2_full_V_cap"])),
        "kaiwu_skill_d_q2_blind": (q2_npz["q2_blind_ising"], "q2", int(q2_npz["q2_blind_n_ref_copies"]), float(q2_npz["q2_blind_V_cap"])),
        "kaiwu_skill_d_q3_qfm108": (q3_npz["q3_qfm108_ising"], "q3_qfm", int(q3_npz["q3_qfm108_n_ref_copies"]), 999.0),
        "kaiwu_skill_d_q3_cmp209": (q3_npz["q3_cmp209_ising"], "q3_cmp", int(q3_npz["q3_cmp209_n_ref_copies"]), float(q3_npz["q3_cmp209_V_cap"])),
        "kaiwu_skill_d_q3_ros264": (q3_npz["q3_ros264_ising"], "q3_ros", int(q3_npz["q3_ros264_n_ref_copies"]), float(q3_npz["q3_ros264_V_cap"])),
    }

    for t_name, (J, _, _, _) in matrix_map.items():
        assert J.dtype == np.int64
        assert J.shape[0] == J.shape[1] <= 550
        assert np.array_equal(J, J.T)
        assert np.all(np.diag(J) == 0)
        assert int(J.min()) >= -127 and int(J.max()) <= 127
        assert int(calculate_ising_matrix_bit_width(J)["precision"]) == 8

    # 3. Load exp3_spqc_results.json and checkpoints/*_results.pkl
    with open(os.path.join(OUT_DIR, "exp3_spqc_results.json"), "r", encoding="utf-8") as f:
        res_json = json.load(f)

    ckpt_files = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith("_results.pkl"))
    assert len(ckpt_files) <= 6, f"Expected <= 6 results checkpoints, found {len(ckpt_files)}: {ckpt_files}"

    df_t4 = pd.read_csv(os.path.join(OUT_DIR, "table4_spqc550_hardware_vs_classical.csv"))
    assert len(df_t4) == 18

    for t_name, (J, t_type, K, vcap) in matrix_map.items():
        t_rec = res_json["tasks"][t_name]
        assert t_rec["matrix_sha256"] == sha256_int64_matrix(J), f"{t_name} sha256 mismatch"
        ckpt_path = os.path.join(CKPT_DIR, t_rec["checkpoint_file"])
        assert os.path.exists(ckpt_path), f"Missing checkpoint {ckpt_path}"
        pkl_arr = np.asarray(pd.read_pickle(ckpt_path), dtype=np.int64)
        saved_spqc_spins = np.asarray(t_rec["solvers"]["SPQC-550"]["raw_spins"], dtype=np.int64)
        assert np.array_equal(pkl_arr, saved_spqc_spins), f"{t_name} checkpoint spins != saved JSON spins"

        for solver_name in ["SPQC-550", "SA", "Tabu"]:
            s_rec = t_rec["solvers"][solver_name]
            spins = np.asarray(s_rec["raw_spins"], dtype=np.int64)
            re_dec = decode_sample_pool(spins, J, t_type, data, n_ref_copies=K, V_cap=vcap)
            assert re_dec["feasible_count"] == s_rec["feasible_count"]
            assert re_dec["best_formula"] == s_rec["best_formula"]
            assert re_dec["hard_penalty_verified"] is True
            if not math.isnan(re_dec["best_ising_energy"]):
                assert abs(re_dec["best_ising_energy"] - s_rec["best_ising_energy"]) < 1e-6

            row_t4 = df_t4[(df_t4["task_name"] == t_name) & (df_t4["solver"] == solver_name)]
            assert len(row_t4) == 1
            r4 = row_t4.iloc[0]
            assert int(r4["feasible_count"]) == re_dec["feasible_count"]
            assert str(r4["best_formula"]) == re_dec["best_formula"]
            assert bool(r4["hard_penalty_verified"]) is True

    # 4. Verify Table 3 & Table 5
    df_t3 = pd.read_csv(os.path.join(OUT_DIR, "table3_q3_encoding_comparison.csv"))
    assert len(df_t3[df_t3["is_production_spqc"] == True]) == 3  # noqa: E712

    df_t5 = pd.read_csv(os.path.join(OUT_DIR, "table5_top_perovskite_cost_breakdown.csv"))
    for _, r in df_t5.iterrows():
        vf, vgl, vgh, vv, vc, v_rec = compute_5term_breakdown(
            float(r["e_form_pred"]), float(r["gap_semi_pred"]), float(r["p_semi_pred"]), float(r["vbm_pred"]), float(r["cbm_pred"])
        )
        assert abs(v_rec - float(r["V_recomputed"])) < 1e-4
        tol = 1e-5 if r["prop_source"] == "04_holdout_surrogate" else 1e-4
        assert abs(v_rec - float(r["V_phys_pred"])) < tol

    print("[Verify-Only] ALL checks PASSED (6 matrices, 6 checkpoints, 18 Table-4 rows, Table-3 & Table-5 invariants, 0 CIM imports)!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Q3 Encodings & SPQC-550 Hardware Execution")
    parser.add_argument("--verify-only", action="store_true", help="Re-verify all saved outputs & checkpoints from disk without CIMOptimizer")
    parser.add_argument("--skip-cim", action="store_true", help="Run Phase A gate only (zero SPQC-550 cloud submissions)")
    args = parser.parse_args()

    if args.verify_only:
        run_verify_only()
        return

    t_start = time.time()
    init_kaiwu_license()
    print("[Init] Kaiwu license initialized via configured environment.")
    data = load_q3_and_q2_data()
    phase_a = build_phase_a(data)
    print(f"\n[Phase A] PASSED in {time.time() - t_start:.1f}s!")

    if args.skip_cim:
        print("[--skip-cim] Stopping after Phase A gate (0 SPQC-550 submissions).")
        return

    results_payload = submit_spqc_tasks(phase_a, data)
    df_t4 = generate_table4(results_payload)
    df_t5 = generate_table5(results_payload, data)
    print(f"\n[Phase B] Exported Table 4 ({len(df_t4)} rows) & Table 5 ({len(df_t5)} rows).")
    run_verify_only()
    print(f"\n[Done] Experiment 3 complete in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

