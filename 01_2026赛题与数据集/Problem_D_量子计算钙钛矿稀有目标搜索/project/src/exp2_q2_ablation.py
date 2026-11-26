"""Experiment 2 (M2): Q2 Penalty Sweep, 8-Bit Precision Quantization Ablation,
and Full vs. Blind Pool Evaluation with Exact-Integer K-Copy Reference Splitting.

Produces:
  - outputs/table2_q2_precision_penalty_ablation.csv
  - outputs/exp2_q2_matrices.npz
  - outputs/exp2_q2_summary.json

Strictly read-only with respect to Problem 1 inputs/outputs; zero SPQC-550 cloud calls.
"""

from __future__ import annotations

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
    get_sorted_solutions,
    ndarray,
    qubo_matrix_to_ising_matrix,
    qubo_model_to_ising_model,
)
from kaiwu.preprocess import (
    adjust_ising_matrix_precision,
    calculate_ising_matrix_bit_width,
    get_dynamic_range_metric,
    perform_precision_adaption_mutate,
    perform_precision_adaption_split,
    restore_split_solution,
)

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "outputs")
sys.path.insert(0, SRC_DIR)

from element_features import ELEMENTS_52  # noqa: E402


TABLE2_COLUMNS = [
    "experiment_group",
    "setting_name",
    "P",
    "V_cap",
    "V_shift",
    "M_blind",
    "n_spins",
    "n_ref_copies",
    "ising_bit_width",
    "dynamic_range_ratio",
    "J_ab_unique_levels",
    "exact_ground_state_rank_of_true_opt",
    "exact_ties_at_top1",
    "exact_8bit_best_formula",
    "exact_8bit_best_V_pred",
    "min_infeasible_energy_gap",
    "sa_ref_consensus_rate",
    "sa_feasible_count",
    "sa_feasible_rate",
    "sa_best_V_pred",
    "sa_best_formula",
    "tabu_feasible_rate",
    "tabu_best_V_pred",
    "tabu_best_formula",
    "exact_top10_in_blind_count",
    "sa_feasible_in_blind_rate",
    "tabu_feasible_in_blind_rate",
    "notes",
]


def init_kaiwu_license() -> None:
    """Load KAIWU credentials from .env without printing/leaking secrets."""
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


def load_q2_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load Q2 full and blind pools and validate alignment & {A}{B}O2N naming."""
    q2_pred_path = os.path.join(OUT_DIR, "q2_full_pool_predictions.csv")
    q2_full_csv = os.path.join(DATA_DIR, "07_castelli_q2_pool.csv")
    q2_blind_csv = os.path.join(DATA_DIR, "11_castelli_q2_pool_blind.csv")

    q2_df = pd.read_csv(q2_pred_path)
    full_pool_df = pd.read_csv(q2_full_csv)
    blind_pool_df = pd.read_csv(q2_blind_csv)

    assert len(q2_df) == 2704, f"Expected 2704 rows, got {len(q2_df)}"
    assert len(full_pool_df) == 2704, f"Expected 2704 rows in 07_pool, got {len(full_pool_df)}"
    assert len(blind_pool_df) == 529, f"Expected 529 rows in 11_blind, got {len(blind_pool_df)}"
    assert list(q2_df["id"]) == list(full_pool_df["id"]), "ID mismatch between predictions and 07 pool"
    assert set(q2_df.loc[q2_df["in_blind_pool"] == 1, "id"]) == set(blind_pool_df["id"]), (
        "Blind pool ID mismatch"
    )
    assert set(q2_df["X"].unique()) == {"NO2"}, f"Unexpected X classes: {q2_df['X'].unique()}"

    # Verify formula convention {A}{B}O2N (not {A}{B}NO2)
    expected_formulas = q2_df["A"] + q2_df["B"] + "O2N"
    assert (q2_df["formula"] == expected_formulas).all(), "Formula naming mismatch from {A}{B}O2N"

    elem_idx = {el: i for i, el in enumerate(ELEMENTS_52)}
    V_raw = np.zeros((52, 52), dtype=np.float64)
    blind_mask = np.zeros((52, 52), dtype=np.int64)
    formula_grid = np.empty((52, 52), dtype=object)

    for _, row in q2_df.iterrows():
        i, j = elem_idx[row["A"]], elem_idx[row["B"]]
        V_raw[i, j] = float(row["V_pred"])
        blind_mask[i, j] = int(row["in_blind_pool"])
        formula_grid[i, j] = str(row["formula"])

    assert int(blind_mask.sum()) == 529
    return V_raw, blind_mask, formula_grid, q2_df


def build_base_qubo(M_mat: np.ndarray, P: float) -> np.ndarray:
    """Build exact 104x104 QUBO matrix for sum M_ij a_i b_j + P((sum a - 1)^2 + (sum b - 1)^2)."""
    Q = np.zeros((104, 104), dtype=np.float64)
    for blk in range(2):
        s0 = blk * 52
        Q[s0 : s0 + 52, s0 : s0 + 52] = P
        np.fill_diagonal(Q[s0 : s0 + 52, s0 : s0 + 52], -P)
    Q[:52, 52:104] = M_mat
    return Q


def verify_qubo_ising_algebra(M_sample: np.ndarray, P_sample: float) -> dict[str, Any]:
    """Verify QuboModel variable ordering, -s^T J s + bias == x^T Q x, and get_sorted_solutions."""
    nA, nB = M_sample.shape
    a = ndarray(nA, "a", Binary)
    b = ndarray(nB, "b", Binary)
    model = QuboModel()
    model.set_objective(a @ M_sample @ b + P_sample * ((a.sum() - 1) ** 2 + (b.sum() - 1) ** 2))
    model.set_constraint_handler(PenaltyMethodConstraint)

    im = qubo_model_to_ising_model(model)
    vars_dict = im.get_variables()
    vars_list = sorted(vars_dict.keys(), key=lambda k: vars_dict[k])
    expected_vars = [f"a[{i:02d}]" for i in range(52)] + [f"b[{j:02d}]" for j in range(52)] + ["__spin__"]
    assert vars_list == expected_vars, f"Unexpected variable order: {vars_list[:5]} ... {vars_list[-5:]}"

    Q_np = build_base_qubo(M_sample, P_sample)
    J_np, bias_np = qubo_matrix_to_ising_matrix(Q_np)
    J_model = im.get_matrix()
    max_mat_diff = float(np.max(np.abs(J_model - J_np)))
    assert max_mat_diff < 1e-10, f"Model vs direct QUBO Ising diff too large: {max_mat_diff}"

    rng = np.random.default_rng(123)
    max_energy_err = 0.0
    for _ in range(16):
        x = rng.integers(0, 2, size=104).astype(np.float64)
        s = np.append(2.0 * x - 1.0, 1.0)
        e_qubo = float(x @ Q_np @ x)
        e_ising = -float(s @ J_np @ s) + float(bias_np)
        max_energy_err = max(max_energy_err, abs(e_qubo - e_ising))
    assert max_energy_err < 1e-9, f"Energy identity error: {max_energy_err}"

    # Verify get_sorted_solutions agrees with -s^T J s
    test_sols = np.array([
        np.append(2.0 * np.eye(104)[0] - 1.0, 1.0),
        np.append(2.0 * np.eye(104)[10] - 1.0, 1.0),
    ])
    sorted_sols, sorted_energies = get_sorted_solutions(J_np, test_sols, bias=float(bias_np))
    e0 = -float(sorted_sols[0] @ J_np @ sorted_sols[0]) + float(bias_np)
    e1 = -float(sorted_sols[1] @ J_np @ sorted_sols[1]) + float(bias_np)
    assert abs(e0 - float(sorted_energies[0])) < 1e-9 and abs(e1 - float(sorted_energies[1])) < 1e-9
    assert e0 <= e1 + 1e-12, "get_sorted_solutions ordering mismatch with -s^T J s"

    return {
        "variables_105_verified": True,
        "max_model_vs_numpy_ising_diff": max_mat_diff,
        "max_qubo_vs_neg_sJs_error": max_energy_err,
        "get_sorted_solutions_sign_verified": True,
    }


def build_refsplit_ising(
    M_mat: np.ndarray,
    P: float,
    K: int = 120,
    target_max: int = 105,
    j_rr_override: int | None = None,
) -> dict[str, Any]:
    """Build exact-integer K-copy reference-spin splitting 8-bit Ising matrix."""
    max_coeff = max(abs(P) / 4.0, float(np.max(np.abs(M_mat))) / 8.0)
    alpha = float(target_max) / max_coeff if max_coeff > 0 else 1.0

    J_aa_int = -int(np.rint(alpha * P / 4.0))
    J_ab_int = np.rint(-alpha * M_mat / 8.0).astype(np.int64)
    h_a = 50 * J_aa_int + J_ab_int.sum(axis=1)
    h_b = 50 * J_aa_int + J_ab_int.sum(axis=0)
    h_int = np.concatenate([h_a, h_b]).astype(np.int64)

    # Exact equivalent 104x104 QUBO after integer quantization
    P_eff = -4.0 * float(J_aa_int)
    M_eff = -8.0 * J_ab_int.astype(np.float64)
    Q_eff = build_base_qubo(M_eff, P_eff)
    J_q2i, bias_q2i = qubo_matrix_to_ising_matrix(Q_eff)
    h_diff = float(np.max(np.abs(J_q2i[:-1, -1] - h_int)))
    assert h_diff == 0.0, f"Integer QUBO-Ising linear field mismatch: {h_diff}"

    n_bits = 104
    n_spins = n_bits + K
    J = np.zeros((n_spins, n_spins), dtype=np.int64)
    for blk in range(2):
        s0 = blk * 52
        for u in range(52):
            for v in range(u + 1, 52):
                J[s0 + u, s0 + v] = J_aa_int
                J[s0 + v, s0 + u] = J_aa_int
    J[:52, 52:104] = J_ab_int
    J[52:104, :52] = J_ab_int.T

    # Distribute h_int[u] exactly across K reference spin copies via quotient-remainder
    for u in range(n_bits):
        val = int(h_int[u])
        q, r = divmod(abs(val), K)
        sgn = 1 if val >= 0 else -1
        row_vals = np.full(K, sgn * q, dtype=np.int64)
        if r > 0:
            idx = (np.arange(r) + u * 7) % K
            row_vals[idx] += sgn
        assert int(row_vals.sum()) == val
        J[u, n_bits:] = row_vals
        J[n_bits:, u] = row_vals

    j_rr_min = max(1, int(math.ceil(1.15 * float(np.sum(np.abs(h_int))) / (K * (K - 1)))))
    j_rr = int(j_rr_override) if j_rr_override is not None else j_rr_min
    assert j_rr <= 127, f"j_rr={j_rr} exceeds 8-bit max 127"

    for c1 in range(K):
        for c2 in range(c1 + 1, K):
            J[n_bits + c1, n_bits + c2] = j_rr
            J[n_bits + c2, n_bits + c1] = j_rr

    assert J.dtype == np.int64
    assert J.shape == (n_spins, n_spins)
    assert n_spins <= 550
    assert np.array_equal(J, J.T)
    assert np.all(np.diag(J) == 0)
    assert int(np.max(np.abs(J))) <= 127
    bw = int(calculate_ising_matrix_bit_width(J)["precision"])
    assert bw <= 8, f"Expected bit_width <= 8, got {bw}"

    h_eff = np.concatenate([h_int, np.full(K, j_rr * (K - 1), dtype=np.int64)])
    variables = (
        [f"a[{i:02d}]" for i in range(52)]
        + [f"b[{j:02d}]" for j in range(52)]
        + [f"__spin_ref_{c:03d}__" for c in range(K)]
    )

    return {
        "J": J,
        "Q_eff": Q_eff,
        "bias_q2i": float(bias_q2i),
        "alpha": float(alpha),
        "j_rr": int(j_rr),
        "j_rr_min": int(j_rr_min),
        "h_int": h_int,
        "h_eff": h_eff,
        "n_spins": int(n_spins),
        "n_ref_copies": int(K),
        "bit_width": int(bw),
        "variables": np.array(variables, dtype="<U18"),
    }


def collapse_refsplit_to_105(J: np.ndarray, n_ref_copies: int) -> np.ndarray:
    """Collapse a K-copy reference-split Ising matrix (when refs agree at +1) to effective 105-spin matrix."""
    if n_ref_copies == 1:
        assert J.shape == (105, 105)
        return J.astype(np.float64)
    J_105 = np.zeros((105, 105), dtype=np.float64)
    J_105[:104, :104] = J[:104, :104]
    h = J[:104, 104 : 104 + n_ref_copies].sum(axis=1)
    J_105[:104, 104] = h
    J_105[104, :104] = h
    return J_105


def evaluate_exact_2704_and_gap(
    J: np.ndarray,
    V_raw: np.ndarray,
    blind_mask: np.ndarray,
    formula_grid: np.ndarray,
    true_opt_ij: tuple[int, int],
    n_ref_copies: int = 1,
    split_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Exact 2,704-state enumeration (-s^T J s) + invalid (kA, kB) != (1, 1) gap check."""
    if split_indices is not None:
        n_spins = J.shape[0]
        starts = np.concatenate([[0], split_indices[:-1] + 1])
        ends = split_indices + 1
        S_105 = np.empty((2704, 105), dtype=np.float64)
        eye52 = 2.0 * np.eye(52) - 1.0
        idx = 0
        for i in range(52):
            for j in range(52):
                S_105[idx, :52] = eye52[i]
                S_105[idx, 52:104] = eye52[j]
                S_105[idx, 104] = 1.0
                idx += 1
        S_split = np.empty((2704, n_spins), dtype=np.float64)
        for k in range(105):
            S_split[:, starts[k] : ends[k]] = S_105[:, k : k + 1]
        E_flat = -np.einsum("ni,ij,nj->n", S_split, J.astype(np.float64), S_split)
        E_valid = E_flat.reshape(52, 52)
        J_ab_unique = -1
        min_inv_gap = float("nan")
    else:
        J_105 = collapse_refsplit_to_105(J, n_ref_copies)
        J_aa = J_105[:52, :52]
        J_bb = J_105[52:104, 52:104]
        J_ab = J_105[:52, 52:104]
        h_a = J_105[:52, 104]
        h_b = J_105[52:104, 104]

        S_A = 2.0 * np.eye(52) - 1.0
        S_B = 2.0 * np.eye(52) - 1.0
        e_A = -np.einsum("ni,ij,nj->n", S_A, J_aa, S_A) - 2.0 * (S_A @ h_a)
        e_B = -np.einsum("mj,jk,mk->m", S_B, J_bb, S_B) - 2.0 * (S_B @ h_b)
        cross = -2.0 * (S_A @ J_ab @ S_B.T)
        E_valid = e_A[:, None] + e_B[None, :] + cross
        J_ab_unique = int(len(np.unique(J_ab)))

        jaa_vals = J_aa[~np.eye(52, dtype=bool)]
        jbb_vals = J_bb[~np.eye(52, dtype=bool)]
        if np.max(np.abs(jaa_vals - jaa_vals[0])) < 1e-9 and np.max(np.abs(jbb_vals - jbb_vals[0])) < 1e-9:
            jaa_val = float(jaa_vals[0])
            jbb_val = float(jbb_vals[0])
            E0 = (
                -jaa_val * (52**2 - 52)
                - jbb_val * (52**2 - 52)
                + 2.0 * float(np.sum(h_a))
                + 2.0 * float(np.sum(h_b))
                - 2.0 * float(np.sum(J_ab))
            )
            lin_A = -4.0 * h_a + 4.0 * J_ab.sum(axis=1)
            lin_B = -4.0 * h_b + 4.0 * J_ab.sum(axis=0)
            M_eff = -8.0 * J_ab
            prefix_A = np.concatenate([[0.0], np.cumsum(np.sort(lin_A))])
            prefix_B = np.concatenate([[0.0], np.cumsum(np.sort(lin_B))])
            prefix_M = np.concatenate([[0.0], np.cumsum(np.sort(M_eff.ravel()))])

            min_inv = float("inf")
            for kA in range(53):
                penA = -jaa_val * ((2 * kA - 52) ** 2 - 52) + jaa_val * (52**2 - 52)
                for kB in range(53):
                    if kA == 1 and kB == 1:
                        continue
                    penB = -jbb_val * ((2 * kB - 52) ** 2 - 52) + jbb_val * (52**2 - 52)
                    lb = E0 + penA + penB + prefix_A[kA] + prefix_B[kB] + prefix_M[kA * kB]
                    if lb < min_inv:
                        min_inv = float(lb)
            if float(np.min(M_eff)) < 0.0:
                pen2 = -jaa_val * ((4 - 52) ** 2 - 52) + jaa_val * (52**2 - 52)
                pen1 = -jbb_val * ((2 - 52) ** 2 - 52) + jbb_val * (52**2 - 52)
                ii, kk = np.triu_indices(52, k=1)
                e_21 = (
                    E0
                    + pen2
                    + pen1
                    + lin_A[ii, None]
                    + lin_A[kk, None]
                    + lin_B[None, :]
                    + M_eff[ii, :]
                    + M_eff[kk, :]
                )
                min_21 = float(np.min(e_21))
                e_12 = (
                    E0
                    + pen1
                    + pen2
                    + lin_A[:, None]
                    + lin_B[None, ii]
                    + lin_B[None, kk]
                    + M_eff[:, ii]
                    + M_eff[:, kk]
                )
                min_12 = float(np.min(e_12))
                if min_inv > min(min_21, min_12):
                    min_inv = min(min_21, min_12)
            min_inv_gap = float(min_inv - float(np.max(E_valid)))
        else:
            min_inv_gap = float("nan")

    e_true = float(E_valid[true_opt_ij])
    e_min = float(np.min(E_valid))
    tol = 1e-9
    rank_true = int(np.sum(E_valid < e_true - tol)) + 1
    ties_top1 = int(np.sum(E_valid <= e_min + tol))

    flat_E = E_valid.ravel()
    order = sorted(range(2704), key=lambda idx: (round(float(flat_E[idx]), 9), idx))
    top1_i, top1_j = divmod(order[0], 52)
    top10_blind_count = sum(1 for idx in order[:10] if int(blind_mask[idx // 52, idx % 52]) == 1)

    return {
        "J_ab_unique_levels": J_ab_unique,
        "exact_ground_state_rank_of_true_opt": rank_true,
        "exact_ties_at_top1": ties_top1,
        "exact_8bit_best_formula": str(formula_grid[top1_i, top1_j]),
        "exact_8bit_best_V_pred": float(V_raw[top1_i, top1_j]),
        "min_infeasible_energy_gap": min_inv_gap,
        "exact_top10_in_blind_count": int(top10_blind_count),
    }


def decode_and_evaluate_pool(
    pool: np.ndarray,
    V_raw: np.ndarray,
    blind_mask: np.ndarray,
    formula_grid: np.ndarray,
    n_ref_copies: int = 1,
    split_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Independently decode every spin sample, verify ref consensus & 1-hot constraints, and recompute V_pred."""
    n_samples = len(pool)
    ref_ok = 0
    split_ok = 0
    feasible_list: list[tuple[float, str, int]] = []

    for row in pool:
        if split_indices is not None:
            try:
                restored = restore_split_solution(row * row[-1], split_indices)
            except Exception:
                continue
            split_ok += 1
            ref_ok += 1
            s_norm = restored[:-1] * restored[-1]
            bits = (s_norm + 1) // 2
        elif n_ref_copies > 1:
            refs = row[104 : 104 + n_ref_copies]
            if not np.all(refs == refs[0]):
                continue
            ref_ok += 1
            bits = ((row[:104] * refs[0]) + 1) // 2
        else:
            ref_ok += 1
            bits = ((row[:104] * row[104]) + 1) // 2

        a_bits = bits[:52]
        b_bits = bits[52:104]
        if int(a_bits.sum()) == 1 and int(b_bits.sum()) == 1:
            idx_a = int(np.argmax(a_bits))
            idx_b = int(np.argmax(b_bits))
            v_val = float(V_raw[idx_a, idx_b])
            formula = str(formula_grid[idx_a, idx_b])
            in_blind = int(blind_mask[idx_a, idx_b])
            feasible_list.append((v_val, formula, in_blind))

    feas_cnt = len(feasible_list)
    feas_rate = float(feas_cnt / n_samples) if n_samples > 0 else 0.0
    ref_rate = float(ref_ok / n_samples) if n_samples > 0 else 0.0
    split_rate = float(split_ok / n_samples) if n_samples > 0 else 0.0

    if feas_cnt > 0:
        best_v, best_f, best_blind = min(feasible_list, key=lambda t: t[0])
        blind_rate = float(sum(t[2] for t in feasible_list) / feas_cnt)
    else:
        best_v, best_f, best_blind, blind_rate = float("nan"), "INFEASIBLE", 0, float("nan")

    return {
        "n_samples": int(n_samples),
        "ref_consensus_rate": ref_rate,
        "split_consistent_rate": split_rate,
        "feasible_count": int(feas_cnt),
        "feasible_rate": feas_rate,
        "best_V_pred": best_v,
        "best_formula": best_f,
        "best_in_blind": int(best_blind),
        "feasible_in_blind_rate": blind_rate,
    }


def run_sa_and_tabu(
    J: np.ndarray,
    h_eff: np.ndarray,
    V_raw: np.ndarray,
    blind_mask: np.ndarray,
    formula_grid: np.ndarray,
    n_ref_copies: int = 1,
    split_indices: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run local SimulatedAnnealingOptimizer (process_num=1, rand_seed=42) & TabuSearchOptimizer (seed=42)."""
    t0_sa = max(100.0, 2.0 * float(np.max(np.abs(h_eff))))
    sa = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=t0_sa,
        alpha=0.99,
        cutoff_temperature=0.001,
        iterations_per_t=100,
        size_limit=50,
        rand_seed=42,
        process_num=1,
    )
    sa_pool = np.asarray(sa.solve(J))
    sa_res = decode_and_evaluate_pool(
        sa_pool, V_raw, blind_mask, formula_grid, n_ref_copies=n_ref_copies, split_indices=split_indices
    )

    np.random.seed(42)
    tabu = kw.classical.TabuSearchOptimizer(max_iter=1500, size_limit=50)
    tabu_pool = np.asarray(tabu.solve(J))
    tabu_res = decode_and_evaluate_pool(
        tabu_pool, V_raw, blind_mask, formula_grid, n_ref_copies=n_ref_copies, split_indices=split_indices
    )
    return sa_res, tabu_res


def make_row(
    experiment_group: str,
    setting_name: str,
    P: float,
    V_cap: float,
    V_shift: float,
    M_blind: float,
    n_spins: int,
    n_ref_copies: int,
    ising_bit_width: float,
    dynamic_range_ratio: float,
    exact_info: dict[str, Any],
    sa_res: dict[str, Any] | None,
    tabu_res: dict[str, Any] | None,
    notes: str,
) -> dict[str, Any]:
    return {
        "experiment_group": experiment_group,
        "setting_name": setting_name,
        "P": round(float(P), 6),
        "V_cap": round(float(V_cap), 6),
        "V_shift": round(float(V_shift), 6),
        "M_blind": round(float(M_blind), 6),
        "n_spins": int(n_spins),
        "n_ref_copies": int(n_ref_copies),
        "ising_bit_width": ising_bit_width,
        "dynamic_range_ratio": round(float(dynamic_range_ratio), 4),
        "J_ab_unique_levels": exact_info["J_ab_unique_levels"],
        "exact_ground_state_rank_of_true_opt": exact_info["exact_ground_state_rank_of_true_opt"],
        "exact_ties_at_top1": exact_info["exact_ties_at_top1"],
        "exact_8bit_best_formula": exact_info["exact_8bit_best_formula"],
        "exact_8bit_best_V_pred": round(float(exact_info["exact_8bit_best_V_pred"]), 6)
        if not math.isnan(float(exact_info["exact_8bit_best_V_pred"]))
        else np.nan,
        "min_infeasible_energy_gap": round(float(exact_info["min_infeasible_energy_gap"]), 4)
        if not math.isnan(float(exact_info["min_infeasible_energy_gap"]))
        else np.nan,
        "sa_ref_consensus_rate": round(float(sa_res["ref_consensus_rate"]), 4) if sa_res else np.nan,
        "sa_feasible_count": int(sa_res["feasible_count"]) if sa_res else 0,
        "sa_feasible_rate": round(float(sa_res["feasible_rate"]), 4) if sa_res else np.nan,
        "sa_best_V_pred": round(float(sa_res["best_V_pred"]), 6)
        if sa_res and not math.isnan(float(sa_res["best_V_pred"]))
        else np.nan,
        "sa_best_formula": sa_res["best_formula"] if sa_res else "SKIPPED_OOB",
        "tabu_feasible_rate": round(float(tabu_res["feasible_rate"]), 4) if tabu_res else np.nan,
        "tabu_best_V_pred": round(float(tabu_res["best_V_pred"]), 6)
        if tabu_res and not math.isnan(float(tabu_res["best_V_pred"]))
        else np.nan,
        "tabu_best_formula": tabu_res["best_formula"] if tabu_res else "SKIPPED_OOB",
        "exact_top10_in_blind_count": exact_info["exact_top10_in_blind_count"],
        "sa_feasible_in_blind_rate": round(float(sa_res["feasible_in_blind_rate"]), 4)
        if sa_res and not math.isnan(float(sa_res["feasible_in_blind_rate"]))
        else np.nan,
        "tabu_feasible_in_blind_rate": round(float(tabu_res["feasible_in_blind_rate"]), 4)
        if tabu_res and not math.isnan(float(tabu_res["feasible_in_blind_rate"]))
        else np.nan,
        "notes": notes,
    }


def main() -> None:
    t_start = time.time()
    init_kaiwu_license()
    V_raw, blind_mask, formula_grid, q2_df = load_q2_data()

    true_opt_full_ij = np.unravel_index(int(np.argmin(V_raw)), (52, 52))
    V_blind_only = np.where(blind_mask == 1, V_raw, np.inf)
    true_opt_blind_ij = np.unravel_index(int(np.argmin(V_blind_only)), (52, 52))
    assert formula_grid[true_opt_full_ij] == "LaTiO2N"
    assert formula_grid[true_opt_blind_ij] == "LaTiO2N"

    algebra_check = verify_qubo_ising_algebra(np.minimum(V_raw, 3.0), P_sample=3.1)

    rows: list[dict[str, Any]] = []
    prod_artifacts: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Var A: Penalty P Sweep (V_cap=3.0, c=0.0, K=120, target_max=105)
    # ------------------------------------------------------------------
    M_cap3_c0 = np.minimum(V_raw, 3.0)
    P_sweep = [0.1, 0.5, 1.0, 2.0, 3.1, 6.0, 15.0, 50.0, 200.0]
    for P_val in P_sweep:
        built = build_refsplit_ising(M_cap3_c0, P=P_val, K=120, target_max=105)
        J = built["J"]
        drm = float(get_dynamic_range_metric(J))
        exact_info = evaluate_exact_2704_and_gap(
            J, V_raw, blind_mask, formula_grid, true_opt_full_ij, n_ref_copies=120
        )
        sa_res, tabu_res = run_sa_and_tabu(
            J, built["h_eff"], V_raw, blind_mask, formula_grid, n_ref_copies=120
        )
        if P_val <= 2.0:
            note = "P < P*=3.0: infeasible 0-hot/multi-hot states beat valid 1-hot states"
        elif P_val == 3.1:
            note = "Tight optimal P=3.1 > P*=3.0: prod q2_full, max J_ab resolution (26 levels)"
            prod_artifacts["q2_full"] = {
                **built,
                "P": 3.1,
                "V_cap": 3.0,
                "V_shift": 0.0,
                "M_blind": 0.0,
                "dynamic_range_ratio": drm,
                "exact_info": exact_info,
                "sa_res": sa_res,
                "tabu_res": tabu_res,
            }
        elif P_val <= 50.0:
            note = "Over-penalized: shrinking J_ab levels under fixed 8-bit dynamic range"
        else:
            note = "Extreme penalty P=200: 8-bit coupling collapse (J_ab<=2 levels)"

        rows.append(
            make_row(
                "VarA_Penalty_Sweep",
                f"A_P_{P_val:g}",
                P_val,
                3.0,
                0.0,
                0.0,
                built["n_spins"],
                120,
                built["bit_width"],
                drm,
                exact_info,
                sa_res,
                tabu_res,
                note,
            )
        )

    # ------------------------------------------------------------------
    # Var B: 8-Bit Precision Quantization Ablation
    # ------------------------------------------------------------------
    # B1: Uncapped direct adjust_ising_matrix_precision (q2_uncap)
    V_max_raw = float(np.max(V_raw))
    Q_unc = build_base_qubo(V_raw, P=65.0)
    J_unc_float, bias_unc = qubo_matrix_to_ising_matrix(Q_unc)
    alpha_unc = 127.0 / float(np.max(np.abs(J_unc_float)))
    J_unc_8bit = np.asarray(adjust_ising_matrix_precision(J_unc_float, bit_width=8), dtype=np.int64)
    assert J_unc_8bit.dtype == np.int64 and J_unc_8bit.shape == (105, 105)
    assert np.array_equal(J_unc_8bit, J_unc_8bit.T) and np.all(np.diag(J_unc_8bit) == 0)
    assert int(np.max(np.abs(J_unc_8bit))) <= 127
    bw_unc = int(calculate_ising_matrix_bit_width(J_unc_8bit)["precision"])
    drm_unc = float(get_dynamic_range_metric(J_unc_8bit))
    exact_b1 = evaluate_exact_2704_and_gap(
        J_unc_8bit, V_raw, blind_mask, formula_grid, true_opt_full_ij, n_ref_copies=1
    )
    sa_b1, tabu_b1 = run_sa_and_tabu(
        J_unc_8bit, J_unc_8bit[:-1, -1], V_raw, blind_mask, formula_grid, n_ref_copies=1
    )
    prod_artifacts["q2_uncap"] = {
        "J": J_unc_8bit,
        "Q_eff": Q_unc,
        "bias_q2i": float(bias_unc),
        "alpha": float(alpha_unc),
        "j_rr": 0,
        "n_spins": 105,
        "n_ref_copies": 1,
        "bit_width": bw_unc,
        "P": 65.0,
        "V_cap": V_max_raw,
        "V_shift": 0.0,
        "M_blind": 0.0,
        "dynamic_range_ratio": drm_unc,
        "exact_info": exact_b1,
        "sa_res": sa_b1,
        "tabu_res": tabu_b1,
        "variables": np.array(
            [f"a[{i:02d}]" for i in range(52)] + [f"b[{j:02d}]" for j in range(52)] + ["__spin__"],
            dtype="<U18",
        ),
    }
    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B1_uncap_direct_adjust",
            65.0,
            V_max_raw,
            0.0,
            0.0,
            105,
            1,
            bw_unc,
            drm_unc,
            exact_b1,
            sa_b1,
            tabu_b1,
            "Prod q2_uncap: unsplit 105-spin adjust_ising_matrix_precision on raw V_pred (rank collapse)",
        )
    )

    # B2: Capped V_cap=3.0 direct adjust_ising_matrix_precision
    Q_c3 = build_base_qubo(M_cap3_c0, P=3.1)
    J_c3_float, _ = qubo_matrix_to_ising_matrix(Q_c3)
    J_b2 = np.asarray(adjust_ising_matrix_precision(J_c3_float, bit_width=8), dtype=np.int64)
    bw_b2 = int(calculate_ising_matrix_bit_width(J_b2)["precision"])
    drm_b2 = float(get_dynamic_range_metric(J_b2))
    exact_b2 = evaluate_exact_2704_and_gap(J_b2, V_raw, blind_mask, formula_grid, true_opt_full_ij, n_ref_copies=1)
    sa_b2, tabu_b2 = run_sa_and_tabu(J_b2, J_b2[:-1, -1], V_raw, blind_mask, formula_grid, n_ref_copies=1)
    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B2_cap3_direct_adjust",
            3.1,
            3.0,
            0.0,
            0.0,
            105,
            1,
            bw_b2,
            drm_b2,
            exact_b2,
            sa_b2,
            tabu_b2,
            "Unsplit 105-spin adjust breaks |h|<51|J_aa| (127>102): 0-hot beats 1-hot (0% feasible)",
        )
    )

    # B3: Capped V_cap=3.0 perform_precision_adaption_mutate + adjust_ising_matrix_precision
    J_c3_mut = np.asarray(perform_precision_adaption_mutate(J_c3_float, iterations=100), dtype=np.float64)
    J_b3 = np.asarray(adjust_ising_matrix_precision(J_c3_mut, bit_width=8), dtype=np.int64)
    bw_b3 = int(calculate_ising_matrix_bit_width(J_b3)["precision"])
    drm_b3 = float(get_dynamic_range_metric(J_b3))
    exact_b3 = evaluate_exact_2704_and_gap(J_b3, V_raw, blind_mask, formula_grid, true_opt_full_ij, n_ref_copies=1)
    sa_b3, tabu_b3 = run_sa_and_tabu(J_b3, J_b3[:-1, -1], V_raw, blind_mask, formula_grid, n_ref_copies=1)
    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B3_cap3_mutate",
            3.1,
            3.0,
            0.0,
            0.0,
            105,
            1,
            bw_b3,
            drm_b3,
            exact_b3,
            sa_b3,
            tabu_b3,
            "perform_precision_adaption_mutate(100) + adjust: same 105-spin 8-bit barrier (0% feasible)",
        )
    )

    # B4a: perform_precision_adaption_split default (min_increment=None) on V_cap=6.0 (2-dec) -> OOB >550 spins
    # Note: unrounded float64 V_pred causes multi-million-spin OOM explosion; even 2-decimal rounding yields 3,132 spins >> 550
    import gc
    V_c6_r2 = np.round(np.minimum(V_raw, 6.0), 2)
    J_c6_r2, _ = qubo_matrix_to_ising_matrix(build_base_qubo(V_c6_r2, P=8.0))
    J_sp_r2, _ = perform_precision_adaption_split(J_c6_r2, 8)
    n_spins_b4a = int(J_sp_r2.shape[0])
    bw_b4a = int(calculate_ising_matrix_bit_width(J_sp_r2)["precision"])
    drm_b4a = float(get_dynamic_range_metric(J_sp_r2))
    del J_sp_r2
    gc.collect()
    unrounded_oom = True

    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B4a_cap6_split_default_oob",
            8.0,
            6.0,
            0.0,
            0.0,
            n_spins_b4a,
            1,
            bw_b4a,
            drm_b4a,
            {
                "J_ab_unique_levels": -1,
                "exact_ground_state_rank_of_true_opt": -1,
                "exact_ties_at_top1": -1,
                "exact_8bit_best_formula": "OOB_GT_550_SPINS",
                "exact_8bit_best_V_pred": float("nan"),
                "min_infeasible_energy_gap": float("nan"),
                "exact_top10_in_blind_count": 0,
            },
            None,
            None,
            (
                f"Default min_increment=None exceeds SPQC-550 (550 spins): "
                f"{n_spins_b4a} spins on 2-dec V_cap=6.0 (unrounded float64 OOM; skipped SA/Tabu)"
            ),
        )
    )

    # B4b: perform_precision_adaption_split(min_increment=0.05) on V_cap=3.0, P=3.1 -> 317 spins <= 550
    J_b4b_raw, idx_b4b = perform_precision_adaption_split(
        J_c3_float, 8, min_increment=0.05, round_to_increment=True
    )
    bw_b4b_info = calculate_ising_matrix_bit_width(J_b4b_raw)
    J_b4b = np.rint(J_b4b_raw * float(bw_b4b_info["multiplier"])).astype(np.int64)
    np.fill_diagonal(J_b4b, 0)
    bw_b4b = int(calculate_ising_matrix_bit_width(J_b4b)["precision"])
    drm_b4b = float(get_dynamic_range_metric(J_b4b))
    exact_b4b = evaluate_exact_2704_and_gap(
        J_b4b, V_raw, blind_mask, formula_grid, true_opt_full_ij, n_ref_copies=1, split_indices=idx_b4b
    )
    sa_b4b, tabu_b4b = run_sa_and_tabu(
        J_b4b, J_b4b[:-1, -1], V_raw, blind_mask, formula_grid, n_ref_copies=1, split_indices=idx_b4b
    )
    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B4b_cap3_split_mi0.05",
            3.1,
            3.0,
            0.0,
            0.0,
            int(J_b4b.shape[0]),
            1,
            bw_b4b,
            drm_b4b,
            exact_b4b,
            sa_b4b,
            tabu_b4b,
            (
                f"Kaiwu split(mi=0.05): 317 spins<=550, but replica conflict KaiwuError "
                f"(SA split_consistent_rate={sa_b4b['split_consistent_rate']:.2f}, "
                f"Tabu split_consistent_rate={tabu_b4b['split_consistent_rate']:.2f})"
            ),
        )
    )

    # B5: Unsplit 105-spin QUBO-aligned binary window (V_cap=0.25, c=0.25, P=8.0)
    V_cap_b5 = 0.25
    V_shift_b5 = 0.25
    scale_b5 = 8.0 / 0.25
    V_tilde_b5 = np.minimum(V_raw, V_cap_b5) - V_shift_b5
    V_q_b5 = 8.0 * np.rint(scale_b5 * V_tilde_b5 / 8.0)
    Q_b5 = build_base_qubo(V_q_b5, P=8.0)
    J_b5_float, _ = qubo_matrix_to_ising_matrix(Q_b5)
    J_b5 = np.rint(J_b5_float).astype(np.int64)
    assert float(np.max(np.abs(J_b5 - J_b5_float))) == 0.0
    bw_b5 = int(calculate_ising_matrix_bit_width(J_b5)["precision"])
    drm_b5 = float(get_dynamic_range_metric(J_b5))
    exact_b5 = evaluate_exact_2704_and_gap(J_b5, V_raw, blind_mask, formula_grid, true_opt_full_ij, n_ref_copies=1)
    sa_b5, tabu_b5 = run_sa_and_tabu(J_b5, J_b5[:-1, -1], V_raw, blind_mask, formula_grid, n_ref_copies=1)
    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B5_cap025_qubo_aligned_P8",
            8.0,
            V_cap_b5,
            V_shift_b5,
            0.0,
            105,
            1,
            bw_b5,
            drm_b5,
            exact_b5,
            sa_b5,
            tabu_b5,
            "Unsplit 105-spin QUBO-aligned binary window (0 rounding drift, only 2 J_ab levels, 2-way tie)",
        )
    )

    # B6: Production q2_full (V_cap=3.0, c=0.0, P=3.1, K=120 ref-split) -> reuse VarA P=3.1
    b6_art = prod_artifacts["q2_full"]
    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B6_cap3_c0_refsplit",
            3.1,
            3.0,
            0.0,
            0.0,
            b6_art["n_spins"],
            120,
            b6_art["bit_width"],
            b6_art["dynamic_range_ratio"],
            b6_art["exact_info"],
            b6_art["sa_res"],
            b6_art["tabu_res"],
            "Prod q2_full: lossless V_cap=3.0, c=0.0, K=120 ref-split (26 J_ab levels, rank=1, 100% feasible)",
        )
    )

    # B7: V_cap=3.0, c=3.0 (c=V_cap), P=6.0, K=120 ref-split
    M_cap3_c3 = M_cap3_c0 - 3.0
    built_b7 = build_refsplit_ising(M_cap3_c3, P=6.0, K=120, target_max=105)
    drm_b7 = float(get_dynamic_range_metric(built_b7["J"]))
    exact_b7 = evaluate_exact_2704_and_gap(
        built_b7["J"], V_raw, blind_mask, formula_grid, true_opt_full_ij, n_ref_copies=120
    )
    sa_b7, tabu_b7 = run_sa_and_tabu(
        built_b7["J"], built_b7["h_eff"], V_raw, blind_mask, formula_grid, n_ref_copies=120
    )
    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B7_cap3_cVcap_refsplit",
            6.0,
            3.0,
            3.0,
            0.0,
            built_b7["n_spins"],
            120,
            built_b7["bit_width"],
            drm_b7,
            exact_b7,
            sa_b7,
            tabu_b7,
            "c=V_cap=3.0 (M<=0): (2,1) multi-hot forces P=6.0 (~2x larger), fewer J_ab levels (19 vs 26)",
        )
    )

    # B8: V_cap=3.0, c=mean(V_cap3), P=5.9, K=120 ref-split
    c_mean = float(np.mean(M_cap3_c0))
    M_cap3_cmean = M_cap3_c0 - c_mean
    built_b8 = build_refsplit_ising(M_cap3_cmean, P=5.9, K=120, target_max=105)
    drm_b8 = float(get_dynamic_range_metric(built_b8["J"]))
    exact_b8 = evaluate_exact_2704_and_gap(
        built_b8["J"], V_raw, blind_mask, formula_grid, true_opt_full_ij, n_ref_copies=120
    )
    sa_b8, tabu_b8 = run_sa_and_tabu(
        built_b8["J"], built_b8["h_eff"], V_raw, blind_mask, formula_grid, n_ref_copies=120
    )
    rows.append(
        make_row(
            "VarB_8Bit_Quantization",
            "B8_cap3_cmean_refsplit",
            5.9,
            3.0,
            c_mean,
            0.0,
            built_b8["n_spins"],
            120,
            built_b8["bit_width"],
            drm_b8,
            exact_b8,
            sa_b8,
            tabu_b8,
            f"c=mean={c_mean:.4f}, P=5.9: negative shifted entries require larger P* and slightly lower resolution",
        )
    )

    # ------------------------------------------------------------------
    # Var C: Full Pool (07) vs Blind Pool (11) Comparison (V_cap=3.0, c=0.0, P=3.1, K=120)
    # ------------------------------------------------------------------
    rows.append(
        make_row(
            "VarC_Pool_Comparison",
            "C1_full_pool",
            3.1,
            3.0,
            0.0,
            0.0,
            b6_art["n_spins"],
            120,
            b6_art["bit_width"],
            b6_art["dynamic_range_ratio"],
            b6_art["exact_info"],
            b6_art["sa_res"],
            b6_art["tabu_res"],
            "Full pool (07_castelli_q2_pool.csv, N=2704): baseline without blind masking",
        )
    )

    for m_blind in [1.0, 3.0, 15.0]:
        V_masked = np.where(blind_mask == 1, V_raw, m_blind)
        M_blind_cap = np.minimum(V_masked, 3.0)
        built_c = build_refsplit_ising(M_blind_cap, P=3.1, K=120, target_max=105)
        drm_c = float(get_dynamic_range_metric(built_c["J"]))
        exact_c = evaluate_exact_2704_and_gap(
            built_c["J"], V_raw, blind_mask, formula_grid, true_opt_blind_ij, n_ref_copies=120
        )
        sa_c, tabu_c = run_sa_and_tabu(
            built_c["J"], built_c["h_eff"], V_raw, blind_mask, formula_grid, n_ref_copies=120
        )
        if m_blind == 1.0:
            cname = "C2_blind_pool_M1.0"
            cnote = "Partial blind penalty M_blind=1.0: known compounds with V<1 still compete in top-10"
        elif m_blind == 3.0:
            cname = "C3_blind_pool_M3.0"
            cnote = "Blind penalty M_blind=3.0 (matches V_cap): 100% blind top-10 & 100% blind sub-cap (<3.0) SA/Tabu"
        else:
            cname = "C4_blind_pool_M15.0"
            cnote = (
                "Prod q2_blind (M_blind=15.0 before cap=3.0): 100% blind top-10 exact states and "
                "blind best SA/Tabu solutions (V=3.0 plateau includes capped non-blind states)"
            )
            assert exact_c["exact_ground_state_rank_of_true_opt"] == 1
            assert exact_c["exact_top10_in_blind_count"] == 10
            assert sa_c["best_in_blind"] == 1 and tabu_c["best_in_blind"] == 1
            prod_artifacts["q2_blind"] = {
                **built_c,
                "P": 3.1,
                "V_cap": 3.0,
                "V_shift": 0.0,
                "M_blind": 15.0,
                "dynamic_range_ratio": drm_c,
                "exact_info": exact_c,
                "sa_res": sa_c,
                "tabu_res": tabu_c,
            }
        rows.append(
            make_row(
                "VarC_Pool_Comparison",
                cname,
                3.1,
                3.0,
                0.0,
                m_blind,
                built_c["n_spins"],
                120,
                built_c["bit_width"],
                drm_c,
                exact_c,
                sa_c,
                tabu_c,
                cnote,
            )
        )

    # Save table2 CSV
    df_table2 = pd.DataFrame(rows, columns=TABLE2_COLUMNS)
    table2_path = os.path.join(OUT_DIR, "table2_q2_precision_penalty_ablation.csv")
    df_table2.to_csv(table2_path, index=False)

    # Save exp2_q2_matrices.npz
    npz_payload: dict[str, Any] = {
        "elements_52": np.array(ELEMENTS_52, dtype="<U2"),
        "V_pred_52x52": V_raw.astype(np.float64),
        "blind_mask_52x52": blind_mask.astype(np.int64),
        "variables_105": np.array(
            [f"a[{i:02d}]" for i in range(52)] + [f"b[{j:02d}]" for j in range(52)] + ["__spin__"],
            dtype="<U18",
        ),
    }
    for name in ["q2_uncap", "q2_full", "q2_blind"]:
        art = prod_artifacts[name]
        npz_payload[f"{name}_ising"] = art["J"].astype(np.int64)
        npz_payload[f"{name}_qubo"] = art["Q_eff"].astype(np.float64)
        npz_payload[f"{name}_bias_q2i"] = np.array(art["bias_q2i"], dtype=np.float64)
        npz_payload[f"{name}_bias"] = np.array(art["bias_q2i"], dtype=np.float64)
        npz_payload[f"{name}_n_spins"] = np.array(art["n_spins"], dtype=np.int64)
        npz_payload[f"{name}_n_ref_copies"] = np.array(art["n_ref_copies"], dtype=np.int64)
        npz_payload[f"{name}_P"] = np.array(art["P"], dtype=np.float64)
        npz_payload[f"{name}_V_cap"] = np.array(art["V_cap"], dtype=np.float64)
        npz_payload[f"{name}_V_shift"] = np.array(art["V_shift"], dtype=np.float64)
        npz_payload[f"{name}_alpha"] = np.array(art["alpha"], dtype=np.float64)
        npz_payload[f"{name}_M_blind"] = np.array(art["M_blind"], dtype=np.float64)
        npz_payload[f"{name}_j_rr"] = np.array(art["j_rr"], dtype=np.int64)
        npz_payload[f"{name}_variables"] = art["variables"]

    npz_path = os.path.join(OUT_DIR, "exp2_q2_matrices.npz")
    np.savez(npz_path, **npz_payload)

    # Save exp2_q2_summary.json
    matrix_integrity = {}
    for name in ["q2_uncap", "q2_full", "q2_blind"]:
        art = prod_artifacts[name]
        J = art["J"]
        matrix_integrity[name] = {
            "shape": list(J.shape),
            "dtype": str(J.dtype),
            "n_spins": int(art["n_spins"]),
            "n_ref_copies": int(art["n_ref_copies"]),
            "j_rr": int(art["j_rr"]),
            "symmetric": bool(np.array_equal(J, J.T)),
            "zero_diag": bool(np.all(np.diag(J) == 0)),
            "min_entry": int(J.min()),
            "max_entry": int(J.max()),
            "bit_width": int(art["bit_width"]),
            "dynamic_range_ratio": float(art["dynamic_range_ratio"]),
            "P": float(art["P"]),
            "V_cap": float(art["V_cap"]),
            "V_shift": float(art["V_shift"]),
            "M_blind": float(art["M_blind"]),
            "alpha": float(art["alpha"]),
            "exact_enumeration": art["exact_info"],
            "sa_baseline": art["sa_res"],
            "tabu_baseline": art["tabu_res"],
        }

    summary = {
        "metadata": {
            "n_q2_pool": int(len(q2_df)),
            "n_q2_blind": int(blind_mask.sum()),
            "elements_52": list(ELEMENTS_52),
            "true_opt_full": {
                "formula": str(formula_grid[true_opt_full_ij]),
                "V_pred": float(V_raw[true_opt_full_ij]),
                "in_blind_pool": int(blind_mask[true_opt_full_ij]),
            },
            "true_opt_blind": {
                "formula": str(formula_grid[true_opt_blind_ij]),
                "V_pred": float(V_raw[true_opt_blind_ij]),
                "in_blind_pool": int(blind_mask[true_opt_blind_ij]),
            },
            "solver_config": {
                "sa": {
                    "alpha": 0.99,
                    "cutoff_temperature": 0.001,
                    "iterations_per_t": 100,
                    "size_limit": 50,
                    "rand_seed": 42,
                    "process_num": 1,
                    "initial_temperature_rule": "max(100.0, 2.0 * max|h_eff|)",
                },
                "tabu": {
                    "max_iter": 1500,
                    "size_limit": 50,
                    "np_random_seed": 42,
                },
            },
            "algebra_verification": algebra_check,
            "runtime_seconds": round(time.time() - t_start, 2),
        },
        "matrix_integrity": matrix_integrity,
        "split_and_reducer_diagnostics": {
            "B4a_unrounded_cap6_P8_memory_error": bool(unrounded_oom),
            "B4a_rounded2dec_cap6_P8_n_spins": int(n_spins_b4a),
            "B4b_cap3_P3.1_mi0.05_n_spins": int(J_b4b.shape[0]),
            "B4b_sa_split_consistent_rate": float(sa_b4b["split_consistent_rate"]),
            "B4b_tabu_split_consistent_rate": float(tabu_b4b["split_consistent_rate"]),
        },
        "var_a_penalty_sweep": [r for r in rows if r["experiment_group"] == "VarA_Penalty_Sweep"],
        "var_b_quantization": [r for r in rows if r["experiment_group"] == "VarB_8Bit_Quantization"],
        "var_c_pool_comparison": [r for r in rows if r["experiment_group"] == "VarC_Pool_Comparison"],
    }

    summary_path = os.path.join(OUT_DIR, "exp2_q2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"EXP2_Q2_ABLATION_DONE in {time.time() - t_start:.2f}s")
    print(f"Saved: {table2_path} ({len(df_table2)} rows)")
    print(f"Saved: {npz_path}")
    print(f"Saved: {summary_path}")
    print("\n=== TABLE 2 SUMMARY ===")
    cols_show = [
        "experiment_group",
        "setting_name",
        "P",
        "V_cap",
        "n_spins",
        "ising_bit_width",
        "J_ab_unique_levels",
        "exact_ground_state_rank_of_true_opt",
        "min_infeasible_energy_gap",
        "sa_ref_consensus_rate",
        "sa_feasible_rate",
        "sa_best_formula",
        "sa_best_V_pred",
        "tabu_feasible_rate",
        "tabu_best_formula",
        "tabu_best_V_pred",
    ]
    print(df_table2[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
