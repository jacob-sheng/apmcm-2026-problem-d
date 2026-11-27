#!/usr/bin/env python3
"""
v3 Remediated Problem 2 & Problem 3 Quantum/Ising QUBO & HOBO Engine:
  - Q2 Exact-8bit Active-Subspace Bipartite QUBO ([-127, 127] exact integer Ising)
  - Model 3A-Comp: Zero-Auxiliary Composite (A,X)-B Bipartite QUBO (Theorem 2 P_block > 1.5 max|Q_obj|)
  - Model 3A-HOBO: Repaired Asymmetric Submodular HOBO->QUBO with Cumulative Row-Sum P_aux(i,k)
                   & Exact Cardinality-Bound P_block (0 illegal global ground states certified)
  - Model 3B-ANOVA: Repaired Uniform 8x Diagonal/Off-Diagonal Scaling & Effect-Amplitude Normalization
                    with 3-Layer Error Decomposition (Surrogate -> Float ANOVA -> 8-bit Ising -> Solver)
  - Global Ground-State Certification (`certify_global_ground_states`):
      Analytic Y-elimination + exact algebraic Ising-to-QUBO recovery cross-check +
      Full 2^{n_primary} enumeration (n_primary <= 22) or Exact Cardinality-Partitioned enumeration (n_primary > 22)
  - 4-Row Compression Attribution Table (F18), C_cap Top-10/Top-25 Retention Audit (F19),
    and Fair Equal-Budget Solver Benchmark (n_runs=50, 1 best run -> 50 returned solutions)
"""

import json
import math
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import kaiwu as kw

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)
from common_physics import safe_ising_bit_width, sort_ising_solutions

OUT_DIR = os.path.join(CUR_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Exact Algebraic QUBO <-> Ising Recovery & Global Ground-State Certifier
# ---------------------------------------------------------------------------

def _recover_qubo_from_ising_sec92(ising_mat):
    """
    Recover exact symmetric integer QUBO Q and offset C from int64 Ising matrix J:
      x_i = (1 + s_i * s_ref) / 2,   E(s) = -s^T J s,
      Q_ij = -4 J_ij (i != j),   Q_ii = -4 J_{i,ref} - sum_{j != i} Q_ij,
      x^T Q x = -s^T J s + C,    C = (sum_{ij} Q_ij + tr(Q)) / 4.
    """
    J = np.asarray(ising_mat, dtype=np.int64)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] < 2:
        raise ValueError(f"Invalid Ising matrix shape: {J.shape}")
    if not np.array_equal(J, J.T):
        raise ValueError("Ising matrix must be symmetric")
    if not np.all(np.diag(J) == 0):
        raise ValueError("Ising matrix diagonal must be zero")

    n = J.shape[0] - 1
    Q = -4 * J[:n, :n].copy()
    np.fill_diagonal(Q, 0)
    row_off_sum = Q.sum(axis=1)
    np.fill_diagonal(Q, -4 * J[:n, n] - row_off_sum)

    # Round-trip algebraic QUBO <-> Ising verification
    H = np.zeros((n + 1, n + 1), dtype=np.int64)
    H[:n, :n] = -Q // 4
    np.fill_diagonal(H[:n, :n], 0)
    assert np.all((-Q + 4 * np.diag(np.diag(Q // 4))) % 4 == 0) or np.array_equal(-4 * H[:n, :n], Q - np.diag(np.diag(Q))), (
        "Non-integer off-diagonal recovery in algebraic Ising check"
    )
    row_sums_q = -Q.sum(axis=1)
    assert np.all(row_sums_q % 4 == 0), "Non-integer reference-spin coupling in algebraic Ising check"
    h_ref = row_sums_q // 4
    H[:n, n] = h_ref
    H[n, :n] = h_ref

    c_num = int(Q.sum()) + int(np.trace(Q))
    assert c_num % 4 == 0, "Non-integer Ising constant C in algebraic Ising check"
    C = c_num // 4

    assert np.array_equal(H, J), "Algebraic Ising round-trip matrix H != J"

    rng = np.random.default_rng(2026)
    test_s = rng.choice(np.array([-1, 1], dtype=np.int64), size=(32, n + 1))
    test_s[0, -1] = 1
    test_s[1, -1] = -1
    for s in test_s:
        x = ((s[:-1] * s[-1]) + 1) // 2
        lhs = int(x @ Q @ x)
        rhs = int(-s @ J @ s + C)
        assert lhs == rhs, f"Algebraic QUBO-Ising energy identity failed: {lhs} != {rhs}"

    return Q, int(C)


def _decode_primary_state_info(z_row, meta, lookup, task_kind):
    """Check primary one-hot + pool eligibility + A!=B for a single primary binary vector z."""
    form = meta.get("formulation", "")
    if form == "comp_ax_b":
        nAX, nB = meta["nAX"], meta["nB"]
        ax_idx = np.where(z_row[:nAX] == 1)[0]
        b_idx = np.where(z_row[nAX:nAX + nB] == 1)[0]
        kAX, kB = len(ax_idx), len(b_idx)
        one_hot_ok = (kAX == 1 and kB == 1)
        if not one_hot_ok:
            return False, None, None
        ea, ex = meta["ax_pairs"][int(ax_idx[0])]
        eb = meta["B_elems"][int(b_idx[0])]
        if ea == eb or (ea, eb, ex) not in lookup:
            return False, None, None
        return True, (ea, eb, ex), float(lookup[(ea, eb, ex)])

    if task_kind == "q2" or form == "q2_bipartite":
        nA, nB = meta["nA"], meta["nB"]
        ai = np.where(z_row[:nA] == 1)[0]
        bi = np.where(z_row[nA:nA + nB] == 1)[0]
        if len(ai) != 1 or len(bi) != 1:
            return False, None, None
        ea, eb = meta["elems_A"][int(ai[0])], meta["elems_B"][int(bi[0])]
        if ea == eb or (ea, eb) not in lookup:
            return False, None, None
        return True, (ea, eb), float(lookup[(ea, eb)])

    nA, nB, nX = meta["nA"], meta["nB"], meta["nX"]
    off_B, off_X = meta["off_B"], meta["off_X"]
    ai = np.where(z_row[:nA] == 1)[0]
    bi = np.where(z_row[off_B:off_B + nB] == 1)[0]
    xi = np.where(z_row[off_X:off_X + nX] == 1)[0]
    if len(ai) != 1 or len(bi) != 1 or len(xi) != 1:
        return False, None, None
    ea = meta["elems_A"][int(ai[0])]
    eb = meta["elems_B"][int(bi[0])]
    ex = meta["elems_X"][int(xi[0])]
    if ea == eb or (ea, eb, ex) not in lookup:
        return False, None, None
    return True, (ea, eb, ex), float(lookup[(ea, eb, ex)])


def _check_hobo_aux_uniqueness(z_row, Q_zy, diag_Qyy, meta):
    """
    For a HOBO primary state z_row, check whether every auxiliary variable y_r
    achieves its minimum uniquely at y_r = a_i * c_k (no illegal aux tie at optimal energy).
    Returns number of auxiliary ties or inconsistent optimal values.
    """
    nY = int(meta.get("nY", 0))
    if nY <= 0:
        return 0
    active_ax = meta["active_ax"]
    a2i = {e: i for i, e in enumerate(meta["elems_A"])}
    x2i = {e: i for i, e in enumerate(meta["elems_X"])}
    off_X = meta["off_X"]

    coeffs = diag_Qyy + 2 * (z_row.astype(np.int64) @ Q_zy)
    bad = 0
    for r, (ea, ex) in enumerate(active_ax):
        expected_y = int(z_row[a2i[ea]] * z_row[off_X + x2i[ex]])
        c_r = int(coeffs[r])
        if expected_y == 0:
            # Must strictly prefer y_r = 0 (c_r > 0) so y_r = 1 cannot tie
            if c_r <= 0:
                bad += 1
        else:
            # Must strictly prefer y_r = 1 (c_r < 0) so y_r = 0 cannot tie
            if c_r >= 0:
                bad += 1
    return bad


def _count_min_subset_tie_multiplicity(vals_sorted, min_val_arr):
    """
    Given 1D sorted values v_0 <= v_1 <= ... <= v_{n-1} and target k,
    compute exact combinatorial multiplicity of subsets of size k achieving the minimum sum.
    """
    n = len(vals_sorted)
    counts = np.zeros(n + 1, dtype=object)
    counts[0] = 1
    # Group equal values
    i = 0
    k_pos = 0
    while i < n:
        j = i + 1
        while j < n and vals_sorted[j] == vals_sorted[i]:
            j += 1
        m = j - i
        # For k in [k_pos, k_pos + m], we can pick (k - k_pos) out of m equal items
        for k in range(k_pos + 1, k_pos + m + 1):
            counts[k] = math.comb(m, k - k_pos)
        k_pos += m
        i = j
    return counts


def _certify_cardinality_partitioned(Q, C, meta, lookup, task_kind):
    """
    Exact zero-gap global certification for n_primary > 22 by exploiting exact
    block-diagonal/separable structure over the leading one-hot block (A or AX).
    Raises AssertionError if structural separability conditions do not hold.
    """
    nY = int(meta.get("nY", 0))
    N = Q.shape[0]
    n_primary = N - nY
    Q_zz = Q[:n_primary, :n_primary]
    Q_zy = Q[:n_primary, n_primary:]
    diag_Qyy = np.diag(Q[n_primary:, n_primary:]).astype(np.int64)

    form = meta.get("formulation", "")
    if form == "comp_ax_b":
        n_lead = int(meta["nAX"])
        m_rest = int(meta["nB"])
        assert n_lead + m_rest == n_primary
    elif form == "q2_bipartite" or task_kind == "q2":
        n_lead = int(meta["nA"])
        m_rest = int(meta["nB"])
        assert n_lead + m_rest == n_primary
    else:
        n_lead = int(meta["nA"])
        m_rest = n_primary - n_lead
        nB = int(meta["nB"])
        nX = int(meta["nX"])
        assert n_lead + nB + nX == n_primary

    # Verify leading block has uniform off-diagonal P_lead (no i-i' objective coupling)
    Q_ll = Q_zz[:n_lead, :n_lead]
    if n_lead >= 2:
        P_lead = int(Q_ll[0, 1])
        expected_off = np.full((n_lead, n_lead), P_lead, dtype=np.int64)
        np.fill_diagonal(expected_off, 0)
        actual_off = Q_ll.copy()
        np.fill_diagonal(actual_off, 0)
        assert np.array_equal(actual_off, expected_off), (
            "Leading block has non-uniform off-diagonal entries; cannot apply exact separable partition."
        )
    else:
        P_lead = 0

    # For HOBO (nY > 0), verify each auxiliary variable couples to at most ONE leading variable a_i
    aux_owner_lead = None
    if nY > 0:
        Q_ly = Q_zy[:n_lead, :]
        aux_owner_lead = []
        for r in range(nY):
            nz = np.where(Q_ly[:, r] != 0)[0]
            assert len(nz) <= 1, f"Aux var {r} couples to >1 leading vars ({nz})"
            aux_owner_lead.append(int(nz[0]) if len(nz) == 1 else -1)

    d_lead = np.diag(Q_ll).astype(np.int64)
    Q_lr = Q_zz[:n_lead, n_lead:].astype(np.int64)
    Q_rr = Q_zz[n_lead:, n_lead:].astype(np.int64)
    Q_ry = Q_zy[n_lead:, :].astype(np.int64)

    # Enumerate ALL 2^m_rest states of the remaining primary variables (B, [X])
    n_rest_states = 1 << m_rest
    masks = np.arange(n_rest_states, dtype=np.uint32)
    Z_rest = ((masks[:, None] >> np.arange(m_rest, dtype=np.uint32)) & 1).astype(np.int64)

    base_zz = np.einsum("ij,jk,ik->i", Z_rest, Q_rr, Z_rest, dtype=np.int64)
    lin_lead = d_lead[None, :] + 2 * (Z_rest @ Q_lr.T)  # (n_rest_states, n_lead)

    if nY > 0:
        # Base aux contribution when a = 0:
        coeff_a0 = diag_Qyy[None, :] + 2 * (Z_rest @ Q_ry)  # (n_rest_states, nY)
        min_a0 = np.minimum(0, coeff_a0)
        base_E = base_zz + min_a0.sum(axis=1)
        # Delta when setting a_i = 1 for each r owned by i:
        delta_lead = lin_lead.copy()
        for r, owner in enumerate(aux_owner_lead):
            if owner >= 0:
                coeff_a1_r = coeff_a0[:, r] + 2 * int(Q_zy[owner, r])
                delta_r = np.minimum(0, coeff_a1_r) - min_a0[:, r]
                delta_lead[:, owner] += delta_r
    else:
        base_E = base_zz
        delta_lead = lin_lead

    order_lead = np.argsort(delta_lead, axis=1, kind="mergesort")
    delta_sorted = np.take_along_axis(delta_lead, order_lead, axis=1)
    zeros = np.zeros((n_rest_states, 1), dtype=np.int64)
    pref = np.hstack([zeros, np.cumsum(delta_sorted, axis=1, dtype=np.int64)])

    # Pass 1: find exact global minimum QUBO energy across ALL (k_lead, z_rest)
    global_min_q = None
    for k_lead in range(n_lead + 1):
        e_k = base_E + pref[:, k_lead] + P_lead * k_lead * (k_lead - 1)
        mk = int(e_k.min())
        if global_min_q is None or mk < global_min_q:
            global_min_q = mk

    # Pass 2: count ALL legal and illegal ground states achieving global_min_q
    legal_ground_count = 0
    illegal_ground_count = 0
    illegal_aux_tie_count = 0
    gs_candidates = {}  # key -> {formula, V_hat, count}

    if form in ("comp_ax_b", "q2_bipartite") or task_kind == "q2":
        kB_arr = Z_rest.sum(axis=1)
        rest_one_hot = (kB_arr == 1)
    else:
        nB = int(meta["nB"])
        kB_arr = Z_rest[:, :nB].sum(axis=1)
        kX_arr = Z_rest[:, nB:].sum(axis=1)
        rest_one_hot = (kB_arr == 1) & (kX_arr == 1)

    for k_lead in range(n_lead + 1):
        e_k = base_E + pref[:, k_lead] + P_lead * k_lead * (k_lead - 1)
        hit_rows = np.where(e_k == global_min_q)[0]
        if len(hit_rows) == 0:
            continue
        for r in hit_rows:
            ds = delta_sorted[r]
            mult_map = _count_min_subset_tie_multiplicity(ds, None)
            mult = int(mult_map[k_lead])
            if k_lead == 1 and bool(rest_one_hot[r]):
                # Inspect every individual i achieving the minimum delta_lead[r, i] == ds[0]
                min_d = ds[0]
                tied_i = np.where(delta_lead[r] == min_d)[0]
                assert len(tied_i) == mult
                z_r = Z_rest[r]
                for i in tied_i:
                    z_full = np.zeros(n_primary, dtype=np.int64)
                    z_full[int(i)] = 1
                    z_full[n_lead:] = z_r
                    ok, key, v_hat = _decode_primary_state_info(z_full, meta, lookup, task_kind)
                    if ok:
                        legal_ground_count += 1
                        if nY > 0:
                            illegal_aux_tie_count += _check_hobo_aux_uniqueness(z_full, Q_zy, diag_Qyy, meta)
                        if key not in gs_candidates:
                            gs_candidates[key] = {"key": key, "V_hat": v_hat, "multiplicity": 0}
                        gs_candidates[key]["multiplicity"] += 1
                    else:
                        illegal_ground_count += 1
            else:
                illegal_ground_count += mult

    return {
        "cert_min_qubo_energy": int(global_min_q),
        "cert_min_ising_energy": int(global_min_q - C),
        "legal_ground_count": int(legal_ground_count),
        "illegal_ground_count": int(illegal_ground_count),
        "illegal_aux_tie_count": int(illegal_aux_tie_count),
        "gs_candidates": gs_candidates,
        "enumeration_method": "exact_cardinality_partitioned",
    }


def _certify_full_vectorized(Q, C, meta, lookup, task_kind):
    """Exact vectorized enumeration over ALL 2^{n_primary} primary states (n_primary <= 22)."""
    nY = int(meta.get("nY", 0))
    N = Q.shape[0]
    n_primary = N - nY
    Q_zz = Q[:n_primary, :n_primary].astype(np.int64)
    Q_zy = Q[:n_primary, n_primary:].astype(np.int64)
    diag_Qyy = np.diag(Q[n_primary:, n_primary:]).astype(np.int64)

    total_states = 1 << n_primary
    chunk_size = min(total_states, 1 << 16)
    bit_pos = np.arange(n_primary, dtype=np.uint32)

    global_min_q = None
    min_chunks = []

    for start in range(0, total_states, chunk_size):
        end = min(total_states, start + chunk_size)
        masks = np.arange(start, end, dtype=np.uint32)
        Z = ((masks[:, None] >> bit_pos) & 1).astype(np.int64)
        e_zz = np.einsum("ij,jk,ik->i", Z, Q_zz, Z, dtype=np.int64)
        if nY > 0:
            coeffs = diag_Qyy[None, :] + 2 * (Z @ Q_zy)
            e_tot = e_zz + np.minimum(0, coeffs).sum(axis=1)
        else:
            e_tot = e_zz

        chunk_min = int(e_tot.min())
        if global_min_q is None or chunk_min < global_min_q:
            global_min_q = chunk_min
            min_chunks = [Z[e_tot == global_min_q]]
        elif chunk_min == global_min_q:
            min_chunks.append(Z[e_tot == global_min_q])

    Z_min = np.vstack(min_chunks)
    legal_ground_count = 0
    illegal_ground_count = 0
    illegal_aux_tie_count = 0
    gs_candidates = {}

    for z_row in Z_min:
        ok, key, v_hat = _decode_primary_state_info(z_row, meta, lookup, task_kind)
        if ok:
            legal_ground_count += 1
            if nY > 0:
                illegal_aux_tie_count += _check_hobo_aux_uniqueness(z_row, Q_zy, diag_Qyy, meta)
            if key not in gs_candidates:
                gs_candidates[key] = {"key": key, "V_hat": v_hat, "multiplicity": 0}
            gs_candidates[key]["multiplicity"] += 1
        else:
            illegal_ground_count += 1

    return {
        "cert_min_qubo_energy": int(global_min_q),
        "cert_min_ising_energy": int(global_min_q - C),
        "legal_ground_count": int(legal_ground_count),
        "illegal_ground_count": int(illegal_ground_count),
        "illegal_aux_tie_count": int(illegal_aux_tie_count),
        "gs_candidates": gs_candidates,
        "enumeration_method": "full_2^n_vectorized",
    }


def certify_global_ground_states(ising_mat, meta, df_pool, lookup, task_kind="q2", verify_zero_illegal=True):
    """
    Independently certify the exact global ground state(s) of an integer Ising matrix:
      1. Recover exact QUBO Q and offset C via exact algebraic Ising formulas & verify H == J.
      2. Analytically eliminate any auxiliary variables y_r via exact min_y.
      3. Enumerate ALL 2^{n_primary} states (if n_primary <= 22) or run exact
         zero-gap cardinality-partitioned enumeration (if n_primary > 22).
      4. Assert illegal_ground_count == 0 when verify_zero_illegal=True.
      5. Compute ground-state degeneracy and continuous surrogate regret against P_declared.
    """
    Q, C = _recover_qubo_from_ising_sec92(ising_mat)
    nY = int(meta.get("nY", 0))
    N = Q.shape[0]
    n_primary = N - nY
    ising_spins = N + 1

    if n_primary <= 22:
        raw_cert = _certify_full_vectorized(Q, C, meta, lookup, task_kind)
    else:
        raw_cert = _certify_cardinality_partitioned(Q, C, meta, lookup, task_kind)

    if verify_zero_illegal:
        assert raw_cert["illegal_ground_count"] == 0, (
            f"Global ground-state certification FAILED: illegal_ground_count={raw_cert['illegal_ground_count']} "
            f"(cert_min_ising={raw_cert['cert_min_ising_energy']})"
        )

    df_sorted = df_pool.sort_values("V_hat").reset_index(drop=True)
    declared_min_v = float(df_sorted.iloc[0]["V_hat"])
    declared_best_formula = str(df_sorted.iloc[0]["formula"])

    formula_map = {}
    rank_map = {}
    for idx_r, r in df_sorted.iterrows():
        key = (r["A"], r["B"]) if task_kind == "q2" else (r["A"], r["B"], r["X"])
        formula_map[key] = str(r["formula"])
        rank_map[key] = int(idx_r + 1)

    gs_list = []
    for key, info in raw_cert["gs_candidates"].items():
        vh = float(info["V_hat"])
        gs_list.append({
            "formula": formula_map.get(key, "".join(key)),
            "tuple": list(key),
            "V_hat": round(vh, 6),
            "regret": round(vh - declared_min_v, 6),
            "pool_rank": rank_map.get(key),
            "multiplicity": int(info["multiplicity"]),
        })
    gs_list.sort(key=lambda d: (d["V_hat"], d["formula"]))

    regrets = [d["regret"] for d in gs_list] if gs_list else [None]
    best_gs_reg = round(min(regrets), 6) if gs_list else None
    worst_gs_reg = round(max(regrets), 6) if gs_list else None
    mean_gs_reg = round(float(np.mean(regrets)), 6) if gs_list else None
    return {
        "n_primary_vars": int(n_primary),
        "n_aux_vars": int(nY),
        "ising_spins": int(ising_spins),
        "n_primary_states_total": int(1 << n_primary),
        "enumeration_method": raw_cert["enumeration_method"],
        "M_max": meta.get("M_max"),
        "cert_min_ising_energy": int(raw_cert["cert_min_ising_energy"]),
        "cert_min_qubo_energy": int(raw_cert["cert_min_qubo_energy"]),
        "ising_offset_C": int(C),
        "legal_ground_count": int(raw_cert["legal_ground_count"]),
        "illegal_ground_count": int(raw_cert["illegal_ground_count"]),
        "illegal_aux_tie_count": int(raw_cert["illegal_aux_tie_count"]),
        "ground_state_degeneracy": int(raw_cert["legal_ground_count"]),
        "contains_continuous_surrogate_minimizer": bool(gs_list and best_gs_reg == 0.0),
        "best_regret_among_integer_ground_states": best_gs_reg,
        "worst_regret_among_integer_ground_states": worst_gs_reg,
        "mean_regret_among_integer_ground_states": mean_gs_reg,
        "ground_state_candidates": gs_list,
        "declared_pool_min_V_hat": round(declared_min_v, 6),
        "declared_pool_best_formula": declared_best_formula,
        "continuous_regret_top1": best_gs_reg,
        "continuous_regret_worst_gs": worst_gs_reg,
        "continuous_regret_mean_gs": mean_gs_reg,
    }


# ---------------------------------------------------------------------------
# Equal-Budget Classical Ising Solvers (SA & Tabu, n_runs=50, 1 best/run)
# ---------------------------------------------------------------------------

def _solve_ising_multi_solver(ising_int, n_sa=50, n_tabu=50, seed_base=1000):
    """
    Run equal-budget Kaiwu SimulatedAnnealingOptimizer and TabuSearchOptimizer
    (n_runs=50 each, taking the single best spin configuration out_s[0] per run -> 50 shots).
    """
    bw = safe_ising_bit_width(ising_int)
    dr = float(kw.get_dynamic_range_metric(ising_int))

    t0 = time.perf_counter()
    sa_ground = []
    sa_energies = []
    for r in range(n_sa):
        opt = kw.SimulatedAnnealingOptimizer(
            initial_temperature=80.0,
            alpha=0.985,
            cutoff_temperature=0.01,
            iterations_per_t=25,
            size_limit=15,
            rand_seed=seed_base + r,
        )
        out = opt.solve(ising_int)
        if out is not None and len(out) > 0:
            out_s = sort_ising_solutions(out, ising_int)
            row = out_s[0]
            e_val = float(-row @ ising_int @ row)
            s_aux = row[-1]
            x = (((row[:-1] * s_aux) + 1) // 2).astype(int)
            sa_ground.append(x)
            sa_energies.append(e_val)
    dt_sa = time.perf_counter() - t0

    t1 = time.perf_counter()
    tabu_ground = []
    tabu_energies = []
    for r in range(n_tabu):
        np.random.seed(seed_base + 5000 + r)
        opt_t = kw.TabuSearchOptimizer(max_iter=250, size_limit=10)
        out_t = opt_t.solve(ising_int)
        if out_t is not None and len(out_t) > 0:
            out_ts = sort_ising_solutions(out_t, ising_int)
            row = out_ts[0]
            e_val = float(-row @ ising_int @ row)
            s_aux = row[-1]
            x = (((row[:-1] * s_aux) + 1) // 2).astype(int)
            tabu_ground.append(x)
            tabu_energies.append(e_val)
    dt_tabu = time.perf_counter() - t1

    return {
        "spins": int(ising_int.shape[0]),
        "bit_width": bw,
        "dynamic_range": round(dr, 2),
        "max_abs_coeff": int(np.max(np.abs(ising_int))),
        "n_sa_runs": int(n_sa),
        "n_tabu_runs": int(n_tabu),
        "sa_ground": sa_ground,
        "sa_sols": sa_ground,
        "sa_energies": sa_energies,
        "sa_runtime_sec": round(dt_sa, 3),
        "tabu_ground": tabu_ground,
        "tabu_sols": tabu_ground,
        "tabu_energies": tabu_energies,
        "tabu_runtime_sec": round(dt_tabu, 3),
    }


# ---------------------------------------------------------------------------
# Q2 & Q3 Formulations (Q2 Exact-8bit, Q3 3A-Comp, Q3 3A-HOBO, Q3 3B-ANOVA)
# ---------------------------------------------------------------------------

def build_exact8_q2_qubo(df_pool, c_cap=2.0, penalty_mult=2.25):
    """
    Lossless Active-Subspace + Upper-Tail Capped Zero-Shifted + Exact-Ratio 8-bit QUBO for Q2.
    Guarantees every Ising matrix entry is an exact integer in [-127, 127].
    """
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
    ising_raw = None
    for M_max in range(12, 2, -1):
        m_int = np.round((M_bar / c_cap) * M_max).astype(int)
        j_pen = int(np.ceil(2.0 * penalty_mult * M_max))
        P_block = 4.0 * j_pen
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
                    Q[i, nA + j] += P_block

        Q_sym = 0.5 * (Q + Q.T)
        ising_raw, _ = kw.qubo_matrix_to_ising_matrix(Q_sym)
        ising_rnd = np.round(ising_raw)
        max_err = float(np.max(np.abs(ising_raw - ising_rnd)))
        max_abs = int(np.max(np.abs(ising_rnd)))
        if max_err < 1e-6 and max_abs <= 127:
            best_ising = ising_rnd.astype(int)
            best_meta = {
                "formulation": "q2_bipartite",
                "nA": nA, "nB": nB, "nY": 0, "N": N,
                "c_cap": float(c_cap), "M_max": M_max,
                "actual_max_m": int(np.max(np.abs(m_int))),
                "j_pen": j_pen, "P_block": P_block, "max_abs_ising": max_abs,
                "elems_A": elems_A, "elems_B": elems_B,
            }
            break

    if best_ising is None:
        ising_scaled = kw.adjust_ising_matrix_precision(ising_raw, bit_width=8)
        best_ising = np.round(ising_scaled).astype(int)
        best_meta = {
            "formulation": "q2_bipartite",
            "nA": nA, "nB": nB, "nY": 0, "N": nA + nB,
            "c_cap": float(c_cap), "M_max": None,
            "elems_A": elems_A, "elems_B": elems_B,
        }

    return best_ising, best_meta, lookup


def build_q3_comp_qubo(df_pool, c_cap=1.50, penalty_mult=1.60):
    """
    Model 3A-Comp: Zero-Auxiliary Composite-Pair Bipartite QUBO for Q3.
    Active (A, X) pairs A = {(i,k) : min_j V_hat(A_i, B_j, X_k) < c_cap} form one
    one-hot block z_{ik} (sum z_{ik} = 1) coupled directly to B-site one-hot block b_j (sum b_j = 1).
    Applies exact Theorem 2 bipartite block penalty P_block > (3/2) * max|Q_obj|
    and forbidden-pair penalty P_forbid >= P_block (multiple of 8 for integer Ising).
    """
    active = df_pool[df_pool["V_hat"] < c_cap].copy()
    if len(active) == 0:
        raise ValueError(f"No Q3 candidates satisfy V_hat < {c_cap}")

    ax_pairs = sorted(set(zip(active["A"], active["X"])))
    elems_B = sorted(active["B"].unique())
    elems_A = sorted(active["A"].unique())
    elems_X = sorted(active["X"].unique())
    nAX, nB = len(ax_pairs), len(elems_B)
    N = nAX + nB

    lookup = {
        (a, b, x): float(v)
        for a, b, x, v in zip(df_pool["A"], df_pool["B"], df_pool["X"], df_pool["V_hat"])
    }

    M_bar = np.zeros((nAX, nB), dtype=float)
    valid_mask = np.zeros((nAX, nB), dtype=bool)
    for p_idx, (ea, ex) in enumerate(ax_pairs):
        for j, eb in enumerate(elems_B):
            if ea != eb and (ea, eb, ex) in lookup:
                valid_mask[p_idx, j] = True
                v = lookup[(ea, eb, ex)]
                if v < c_cap:
                    M_bar[p_idx, j] = v - c_cap

    best_ising = None
    best_meta = None
    ising_raw = None
    for M_max in range(12, 0, -1):
        m_int = np.round((M_bar / c_cap) * M_max).astype(int)
        amax = int(np.max(np.abs(m_int)))
        if amax == 0:
            continue
        # Theorem 2 bound: P_block = 4*j_pen > (3/2)*(8*amax) = 12*amax  <=>  j_pen >= 3*amax + 1
        j_pen = max(3 * amax + 1, int(np.ceil(2.0 * penalty_mult * amax)))
        P_block = 4.0 * j_pen
        P_forbid = 8.0 * int(np.ceil(j_pen / 2.0))  # >= P_block and multiple of 8

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
            best_ising = ising_rnd.astype(int)
            best_meta = {
                "formulation": "comp_ax_b",
                "ax_pairs": ax_pairs,
                "B_elems": elems_B,
                "elems_A": elems_A,
                "elems_B": elems_B,
                "elems_X": elems_X,
                "pool_lookup": lookup,
                "nAX": nAX,
                "nB": nB,
                "nY": 0,
                "N": N,
                "off_B": nAX,
                "c_cap": float(c_cap),
                "M_max": int(M_max),
                "actual_max_m": amax,
                "j_pen": int(j_pen),
                "P_block": float(P_block),
                "P_forbid": float(P_forbid),
                "theorem2_min_P": float(12 * amax),
                "max_abs_ising": max_abs,
            }
            break

    if best_ising is None:
        ising_scaled = kw.adjust_ising_matrix_precision(ising_raw, bit_width=8)
        best_ising = np.round(ising_scaled).astype(int)
        best_meta = {
            "formulation": "comp_ax_b",
            "ax_pairs": ax_pairs,
            "B_elems": elems_B,
            "elems_A": elems_A,
            "elems_B": elems_B,
            "elems_X": elems_X,
            "pool_lookup": lookup,
            "nAX": nAX, "nB": nB, "nY": 0, "N": N, "off_B": nAX,
            "c_cap": float(c_cap), "M_max": None,
            "max_abs_ising": int(np.max(np.abs(best_ising))),
        }

    return best_ising, best_meta, lookup


def _compute_hobo_block_penalty_bound(T8, nA, nB, nX):
    """
    Compute exact lower bound (or tight upper bound if nB+nX is large) on
      max_{(kA, kB, kX) != (1, 1, 1)} (-min H_obj(kA, kB, kX)) / ((kA-1)^2 + (kB-1)^2 + (kX-1)^2).
    """
    if (1 << (nB + nX)) <= 4096:
        max_ratio = 0.0
        for x_mask in range(1, 1 << nX):
            xs = [k for k in range(nX) if (x_mask >> k) & 1]
            kX = len(xs)
            M_AB = T8[:, :, xs].sum(axis=2)
            for b_mask in range(1, 1 << nB):
                bs = [j for j in range(nB) if (b_mask >> j) & 1]
                kB = len(bs)
                row_sums = np.sort(M_AB[:, bs].sum(axis=1))
                for kA in range(1, nA + 1):
                    if kA == 1 and kB == 1 and kX == 1:
                        continue
                    h_min = int(row_sums[:kA].sum())
                    denom = (kA - 1) ** 2 + (kB - 1) ** 2 + (kX - 1) ** 2
                    ratio = (-h_min) / denom
                    if ratio > max_ratio:
                        max_ratio = ratio
        return float(max_ratio)

    # Exact safe upper bound for larger B/X sets (e.g., wide C_cap sweeps)
    neg_sorted = np.sort(T8.ravel())
    neg_sorted = neg_sorted[neg_sorted < 0]
    if len(neg_sorted) == 0:
        return 0.0
    pref = np.concatenate([[0], np.cumsum(neg_sorted)])
    max_ratio = 0.0
    for kA in range(1, nA + 1):
        for kB in range(1, nB + 1):
            for kX in range(1, nX + 1):
                if kA == 1 and kB == 1 and kX == 1:
                    continue
                cnt = min(len(neg_sorted), kA * kB * kX)
                h_lb = int(pref[cnt])
                denom = (kA - 1) ** 2 + (kB - 1) ** 2 + (kX - 1) ** 2
                ratio = (-h_lb) / denom
                if ratio > max_ratio:
                    max_ratio = ratio
    return float(max_ratio)


def build_q3_submodular_hobo_qubo(df_pool, c_cap=1.5):
    """
    Model 3A-HOBO: Repaired Asymmetric One-Sided Submodular HOBO->QUBO.
    Fixes F01 via:
      1. Cumulative row-sum auxiliary penalty per active (i,k):
           P_aux(i,k) = sum_j |Q_obj(y_{ik}, b_j)| + 8 > sum_j |Q_obj(y_{ik}, b_j)|,
         guaranteeing y_{ik}=1 when a_i c_k = 0 strictly increases energy even if ALL b_j = 1.
      2. Exact cardinality-partitioned one-hot penalty:
           P_block > max_{(kA,kB,kX) != (1,1,1)} (-min H_obj(kA,kB,kX)) / ((kA-1)^2+(kB-1)^2+(kX-1)^2).
    """
    active = df_pool[df_pool["V_hat"] < c_cap].copy()
    if len(active) == 0:
        raise ValueError(f"No Q3 candidates satisfy V_hat < {c_cap}")

    elems_A = sorted(active["A"].unique())
    elems_B = sorted(active["B"].unique())
    elems_X = sorted(active["X"].unique())
    nA, nB, nX = len(elems_A), len(elems_B), len(elems_X)
    a2i = {e: i for i, e in enumerate(elems_A)}
    b2i = {e: i for i, e in enumerate(elems_B)}
    x2i = {e: i for i, e in enumerate(elems_X)}

    active_ax = sorted(set(zip(active["A"], active["X"])))
    ax2y = {pair: idx for idx, pair in enumerate(active_ax)}
    nY = len(active_ax)

    off_B = nA
    off_X = nA + nB
    off_Y = nA + nB + nX
    N = nA + nB + nX + nY

    lookup = {
        (a, b, x): float(v)
        for a, b, x, v in zip(df_pool["A"], df_pool["B"], df_pool["X"], df_pool["V_hat"])
    }

    best_ising = None
    best_meta = None
    ising_raw = None
    for M_max in range(8, 0, -1):
        T8 = np.zeros((nA, nB, nX), dtype=int)
        for a, b, x, v in zip(active["A"], active["B"], active["X"], active["V_hat"]):
            if a in a2i and b in b2i and x in x2i:
                m_int = int(np.round(((v - c_cap) / c_cap) * M_max))
                if m_int < 0:
                    T8[a2i[a], b2i[b], x2i[x]] = 8 * m_int

        amax = int(-T8.min() // 8)
        if amax == 0:
            continue

        strong_P_bound = _compute_hobo_block_penalty_bound(T8, nA, nB, nX)
        j_pen = int(np.floor(strong_P_bound / 4.0)) + 1
        P_block = 4.0 * j_pen
        P_forbid = 8.0 * int(np.ceil(j_pen / 2.0))

        Q = np.zeros((N, N), dtype=float)
        R_row = {}
        P_aux_map = {}
        for (ea, ex), yidx in ax2y.items():
            ai = a2i[ea]
            xi = x2i[ex]
            yi = off_Y + yidx
            r_sum = 0
            for j, eb in enumerate(elems_B):
                val = int(T8[ai, j, xi])
                if val < 0:
                    bi = off_B + j
                    Q[min(bi, yi), max(bi, yi)] += float(val)
                    r_sum += abs(val)
            p_aux_ik = float(r_sum + 8)
            R_row[f"{ea}_{ex}"] = int(r_sum)
            P_aux_map[f"{ea}_{ex}"] = int(p_aux_ik)
            Q[yi, yi] += 2.0 * p_aux_ik
            Q[ai, yi] -= p_aux_ik
            Q[off_X + xi, yi] -= p_aux_ik

        for start, sz in [(0, nA), (off_B, nB), (off_X, nX)]:
            for i in range(start, start + sz):
                Q[i, i] -= P_block
                for j in range(i + 1, start + sz):
                    Q[i, j] += 2.0 * P_block

        for i, ea in enumerate(elems_A):
            for j, eb in enumerate(elems_B):
                if ea == eb or all((ea, eb, ex) not in lookup for ex in elems_X):
                    Q[i, off_B + j] += P_forbid

        Q_sym = 0.5 * (Q + Q.T)
        ising_raw, _ = kw.qubo_matrix_to_ising_matrix(Q_sym)
        ising_rnd = np.round(ising_raw)
        max_err = float(np.max(np.abs(ising_raw - ising_rnd)))
        max_abs = int(np.max(np.abs(ising_rnd)))
        if max_err < 1e-6 and max_abs <= 127:
            best_ising = ising_rnd.astype(int)
            best_meta = {
                "formulation": "hobo_ax_y",
                "nA": nA, "nB": nB, "nX": nX, "nY": nY, "N": N,
                "off_B": off_B, "off_X": off_X, "off_Y": off_Y,
                "elems_A": elems_A, "elems_B": elems_B, "elems_X": elems_X,
                "active_ax": active_ax,
                "c_cap": float(c_cap),
                "M_max": int(M_max),
                "actual_max_m": amax,
                "j_pen": int(j_pen),
                "P_block": float(P_block),
                "P_forbid": float(P_forbid),
                "strong_P_bound": round(strong_P_bound, 4),
                "max_R_row": int(max(R_row.values())),
                "max_P_aux": int(max(P_aux_map.values())),
                "R_row_per_y": R_row,
                "P_aux_per_y": P_aux_map,
                "max_abs_ising": max_abs,
            }
            break

    if best_ising is None:
        best_ising = np.round(kw.adjust_ising_matrix_precision(ising_raw, bit_width=8)).astype(int)
        best_meta = {
            "formulation": "hobo_ax_y",
            "nA": nA, "nB": nB, "nX": nX, "nY": nY, "N": N,
            "off_B": off_B, "off_X": off_X, "off_Y": off_Y,
            "elems_A": elems_A, "elems_B": elems_B, "elems_X": elems_X,
            "active_ax": active_ax,
            "c_cap": float(c_cap), "M_max": None,
            "max_abs_ising": int(np.max(np.abs(best_ising))),
        }

    return best_ising, best_meta, lookup


def _anova_exact_illegal_min_check(Q_sym, elems_A, elems_B, elems_X, lookup):
    """Exact separable (S_B, S_X) global check for Model 3B ANOVA returning (legal_min, illegal_min)."""
    nA, nB, nX = len(elems_A), len(elems_B), len(elems_X)
    P_A = float(Q_sym[0, 1]) if nA >= 2 else 0.0
    Q_rr = Q_sym[nA:, nA:]
    d_A = np.diag(Q_sym[:nA, :nA])
    Q_Ar = Q_sym[:nA, nA:]

    m_rest = nB + nX
    masks = np.arange(1 << m_rest, dtype=np.uint32)
    Z_rest = ((masks[:, None] >> np.arange(m_rest, dtype=np.uint32)) & 1).astype(float)
    base_E = np.einsum("ij,jk,ik->i", Z_rest, Q_rr, Z_rest)
    lin_A = d_A[None, :] + 2.0 * (Z_rest @ Q_Ar.T)
    kB_arr = Z_rest[:, :nB].sum(axis=1).astype(int)
    kX_arr = Z_rest[:, nB:].sum(axis=1).astype(int)

    order_A = np.argsort(lin_A, axis=1)
    lin_sorted = np.take_along_axis(lin_A, order_A, axis=1)
    pref = np.hstack([np.zeros((len(masks), 1)), np.cumsum(lin_sorted, axis=1)])

    legal_min = float("inf")
    illegal_min = float("inf")
    for kA in range(nA + 1):
        e_k = base_E + pref[:, kA] + P_A * kA * (kA - 1)
        if kA == 1:
            is_oh = (kB_arr == 1) & (kX_arr == 1)
            for r in np.where(is_oh)[0]:
                eb = elems_B[int(np.argmax(Z_rest[r, :nB]))]
                ex = elems_X[int(np.argmax(Z_rest[r, nB:]))]
                for i, ea in enumerate(elems_A):
                    ei = float(base_E[r] + lin_A[r, i])
                    if ea != eb and (ea, eb, ex) in lookup:
                        if ei < legal_min:
                            legal_min = ei
                    else:
                        if ei < illegal_min:
                            illegal_min = ei
            non_oh = e_k[~is_oh]
            if len(non_oh) > 0:
                illegal_min = min(illegal_min, float(non_oh.min()))
        else:
            illegal_min = min(illegal_min, float(e_k.min()))
    return legal_min, illegal_min


def build_q3_compact_anova_qubo(df_pool, top_k_subspace=35, c_cap=1.5):
    """
    Model 3B-ANOVA: Zero-Auxiliary-Variable Compact ANOVA-QUBO on the Top-K Subspace.
    Fixes F02 via:
      1. Uniform 8.0 * round(...) scaling on BOTH main-effect diagonals (tA, tB, tX)
         AND pairwise off-diagonals (tAB, tAX, tBX) after Q_sym = 0.5*(Q + Q.T).
      2. Effect-amplitude scaling max_eff = max(|tA|, |tB|, |tX|, |tAB|, |tAX|, |tBX|).
      3. Maximal M_max and largest j_pen within [-127, 127] verified with
         illegal_min > legal_min (zero illegal global ground states).
    """
    sub = df_pool.nsmallest(top_k_subspace, "V_hat")
    elems_A = sorted(sub["A"].unique())
    elems_B = sorted(sub["B"].unique())
    elems_X = sorted(sub["X"].unique())
    nA, nB, nX = len(elems_A), len(elems_B), len(elems_X)

    lookup = {
        (a, b, x): float(v)
        for a, b, x, v in zip(df_pool["A"], df_pool["B"], df_pool["X"], df_pool["V_hat"])
    }
    formula_map = {
        (a, b, x): str(f)
        for a, b, x, f in zip(df_pool["A"], df_pool["B"], df_pool["X"], df_pool["formula"])
    }

    T = np.zeros((nA, nB, nX), dtype=float)
    for i, ea in enumerate(elems_A):
        for j, eb in enumerate(elems_B):
            for k, ex in enumerate(elems_X):
                v = lookup.get((ea, eb, ex), c_cap)
                T[i, j, k] = min(v, c_cap) - c_cap

    mu = float(T.mean())
    tA = T.mean(axis=(1, 2)) - mu
    tB = T.mean(axis=(0, 2)) - mu
    tX = T.mean(axis=(0, 1)) - mu
    tAB = T.mean(axis=2) - mu - tA[:, None] - tB[None, :]
    tAX = T.mean(axis=1) - mu - tA[:, None] - tX[None, :]
    tBX = T.mean(axis=0) - mu - tB[:, None] - tX[None, :]

    T_anova = (
        mu + tA[:, None, None] + tB[None, :, None] + tX[None, None, :]
        + tAB[:, :, None] + tAX[:, None, :] + tBX[None, :, :]
    )
    ss_tot = float(np.sum((T - mu) ** 2))
    ss_res = float(np.sum((T - T_anova) ** 2))
    anova_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    max_eff = max(
        float(np.max(np.abs(arr))) for arr in (tA, tB, tX, tAB, tAX, tBX)
    )
    max_eff = max(max_eff, 1e-6)

    off_B = nA
    off_X = nA + nB
    N = nA + nB + nX

    # Layer (a): compare V_hat (and T_bar) vs unquantized float ANOVA on legal subspace candidates
    legal_triples = []
    for i, ea in enumerate(elems_A):
        for j, eb in enumerate(elems_B):
            for k, ex in enumerate(elems_X):
                if ea != eb and (ea, eb, ex) in lookup:
                    legal_triples.append((i, j, k, ea, eb, ex, lookup[(ea, eb, ex)], float(T[i, j, k]), float(T_anova[i, j, k])))

    v_arr = np.array([r[6] for r in legal_triples], dtype=float)
    t_bar_arr = np.array([r[7] for r in legal_triples], dtype=float)
    fa_arr = np.array([r[8] for r in legal_triples], dtype=float)
    fa_opt_idx = int(np.argmin(fa_arr))
    fa_opt_rec = legal_triples[fa_opt_idx]
    declared_min_v = float(df_pool["V_hat"].min())

    layer_a = {
        "n_legal_subspace_candidates": len(legal_triples),
        "anova_r2_full_tensor": round(anova_r2, 4),
        "rmse_Tbar_vs_float_anova": round(float(np.sqrt(np.mean((t_bar_arr - fa_arr) ** 2))), 6),
        "spearman_Vhat_vs_float_anova": round(float(spearmanr(v_arr, fa_arr).statistic), 4),
        "float_anova_opt_formula": formula_map[(fa_opt_rec[3], fa_opt_rec[4], fa_opt_rec[5])],
        "float_anova_opt_val": round(float(fa_opt_rec[8]), 6),
        "float_anova_opt_V_hat": round(float(fa_opt_rec[6]), 6),
        "float_anova_opt_regret": round(float(fa_opt_rec[6]) - declared_min_v, 6),
    }

    best_ising = None
    best_meta = None
    ising_raw = None
    for M_max in range(10, 0, -1):
        scale = M_max / max_eff
        qA = 8 * np.round(tA * scale).astype(int)
        qB = 8 * np.round(tB * scale).astype(int)
        qX = 8 * np.round(tX * scale).astype(int)
        qAB = 8 * np.round(tAB * scale).astype(int)
        qAX = 8 * np.round(tAX * scale).astype(int)
        qBX = 8 * np.round(tBX * scale).astype(int)

        best_for_M = None
        for j_pen in range(1, 40):
            P_block = 4.0 * j_pen
            P_forbid = 8.0 * int(np.ceil(j_pen / 2.0))
            Q = np.zeros((N, N), dtype=float)

            for i in range(nA):
                Q[i, i] += float(qA[i])
            for j in range(nB):
                Q[off_B + j, off_B + j] += float(qB[j])
            for k in range(nX):
                Q[off_X + k, off_X + k] += float(qX[k])

            for i in range(nA):
                for j in range(nB):
                    Q[i, off_B + j] += float(qAB[i, j])
            for i in range(nA):
                for k in range(nX):
                    Q[i, off_X + k] += float(qAX[i, k])
            for j in range(nB):
                for k in range(nX):
                    Q[off_B + j, off_X + k] += float(qBX[j, k])

            for start, sz in [(0, nA), (off_B, nB), (off_X, nX)]:
                for i in range(start, start + sz):
                    Q[i, i] -= P_block
                    for j in range(i + 1, start + sz):
                        Q[i, j] += 2.0 * P_block
            for i, ea in enumerate(elems_A):
                for j, eb in enumerate(elems_B):
                    if ea == eb or all((ea, eb, ex) not in lookup for ex in elems_X):
                        Q[i, off_B + j] += P_forbid

            Q_sym = 0.5 * (Q + Q.T)
            ising_raw, _ = kw.qubo_matrix_to_ising_matrix(Q_sym)
            ising_rnd = np.round(ising_raw)
            max_err = float(np.max(np.abs(ising_raw - ising_rnd)))
            max_abs = int(np.max(np.abs(ising_rnd)))
            if max_err >= 1e-6 or max_abs > 127:
                break

            leg_min, illeg_min = _anova_exact_illegal_min_check(Q_sym, elems_A, elems_B, elems_X, lookup)
            if illeg_min > leg_min + 1e-6:
                best_for_M = (ising_rnd.astype(int), j_pen, P_block, P_forbid, max_abs, leg_min, illeg_min)

        if best_for_M is not None:
            best_ising, j_pen_b, P_blk_b, P_fbd_b, max_abs_b, leg_min_b, illeg_min_b = best_for_M
            int_vals = np.array([
                int(qA[i] + qB[j] + qX[k] + qAB[i, j] + qAX[i, k] + qBX[j, k])
                for i, j, k, *_ in legal_triples
            ], dtype=float)
            uniq_lvls, lvl_counts = np.unique(int_vals.astype(int), return_counts=True)
            distinct_levels = int(len(uniq_lvls))
            level_occupancies = [int(c) for c in lvl_counts.tolist()]
            sp_int_float = float(spearmanr(fa_arr, int_vals).statistic)
            sp_int_vhat = float(spearmanr(v_arr, int_vals).statistic)
            best_meta = {
                "formulation": "anova_8x",
                "nA": nA, "nB": nB, "nX": nX, "nY": 0, "N": N,
                "off_B": off_B, "off_X": off_X,
                "elems_A": elems_A, "elems_B": elems_B, "elems_X": elems_X,
                "top_k_subspace": int(top_k_subspace),
                "c_cap": float(c_cap),
                "anova_r2": round(anova_r2, 4),
                "max_eff": round(max_eff, 6),
                "M_max": int(M_max),
                "j_pen": int(j_pen_b),
                "P_block": float(P_blk_b),
                "P_forbid": float(P_fbd_b),
                "max_abs_ising": int(max_abs_b),
                "distinct_legal_energy_levels": distinct_levels,
                "level_occupancies": level_occupancies,
                "quantization_note": (
                    "Uniform 8x integer scaling eliminates the main-vs-interaction factor mismatch, "
                    "while round(effect * scale) still discretizes relative float ANOVA weights."
                ),
                "spearman_float_anova_vs_int": round(sp_int_float, 4),
                "spearman_Vhat_vs_int": round(sp_int_vhat, 4),
                "layer_a_surrogate_vs_float_anova": layer_a,
            }
            break

    if best_ising is None:
        best_ising = np.round(kw.adjust_ising_matrix_precision(ising_raw, bit_width=8)).astype(int)
        best_meta = {
            "formulation": "anova_8x",
            "nA": nA, "nB": nB, "nX": nX, "nY": 0, "N": N,
            "off_B": off_B, "off_X": off_X,
            "elems_A": elems_A, "elems_B": elems_B, "elems_X": elems_X,
            "anova_r2": round(anova_r2, 4),
            "M_max": None,
            "layer_a_surrogate_vs_float_anova": layer_a,
        }

    return best_ising, best_meta, lookup


# ---------------------------------------------------------------------------
# Solution Decoding & Evaluation (Strict Feasibility + Regret + Top-K)
# ---------------------------------------------------------------------------

def evaluate_q2_solutions(sols, meta, df_pool, lookup, ground_sols=None, energies=None, cert=None):
    nA, nB = meta["nA"], meta["nB"]
    elems_A, elems_B = meta["elems_A"], meta["elems_B"]

    df_sorted = df_pool.sort_values("V_hat").reset_index(drop=True)
    exact_min = float(df_sorted.iloc[0]["V_hat"])
    exact_best_formula = str(df_sorted.iloc[0]["formula"])
    formula_map = {(r["A"], r["B"]): str(r["formula"]) for _, r in df_sorted.iterrows()}
    rank_map = {(r["A"], r["B"]): int(idx + 1) for idx, (_, r) in enumerate(df_sorted.iterrows())}
    top10 = set(zip(df_sorted.head(10)["A"], df_sorted.head(10)["B"]))

    g_list = ground_sols if ground_sols is not None else sols
    kA_l, kB_l = [], []
    feas = 0
    for x in g_list:
        ai = np.where(x[:nA] == 1)[0]
        bi = np.where(x[nA:nA + nB] == 1)[0]
        kA_l.append(len(ai))
        kB_l.append(len(bi))
        if len(ai) == 1 and len(bi) == 1:
            ea, eb = elems_A[int(ai[0])], elems_B[int(bi[0])]
            if ea != eb and (ea, eb) in lookup:
                feas += 1

    unique_map = {}
    hits = set()
    for x in sols:
        ai = np.where(x[:nA] == 1)[0]
        bi = np.where(x[nA:nA + nB] == 1)[0]
        if len(ai) == 1 and len(bi) == 1:
            ea, eb = elems_A[int(ai[0])], elems_B[int(bi[0])]
            if ea != eb and (ea, eb) in lookup:
                v = float(lookup[(ea, eb)])
                f = formula_map[(ea, eb)]
                unique_map[(ea, eb)] = {"formula": f, "V_hat": round(v, 6), "regret": round(v - exact_min, 6), "pool_rank": rank_map[(ea, eb)]}
                if (ea, eb) in top10:
                    hits.add((ea, eb))

    unique_cands = sorted(unique_map.values(), key=lambda d: (d["V_hat"], d["formula"]))
    best_v = unique_cands[0]["V_hat"] if unique_cands else None
    best_formula = unique_cands[0]["formula"] if unique_cands else None
    top5 = unique_cands[:5]

    out = {
        "n_returned_shots": len(sols),
        "feasible_shot_count": int(feas),
        "primary_feasibility_pct": round(100.0 * feas / max(1, len(g_list)), 2),
        "y_aux_consistent_pct": 100.0,
        "feasibility_pct": round(100.0 * feas / max(1, len(g_list)), 2),
        "mean_kA": round(float(np.mean(kA_l)), 3) if kA_l else 0.0,
        "mean_kB": round(float(np.mean(kB_l)), 3) if kB_l else 0.0,
        "unique_feasible_count": len(unique_cands),
        "unique_feasible_candidates": [d["formula"] for d in unique_cands],
        "best_V_hat": round(best_v, 6) if best_v is not None else None,
        "exact_min_V_hat": round(exact_min, 6),
        "continuous_regret": round(best_v - exact_min, 6) if best_v is not None else None,
        "gap_to_opt": round(best_v - exact_min, 6) if best_v is not None else None,
        "top5_V_hat": [d["V_hat"] for d in top5],
        "top5_mean_V_hat": round(float(np.mean([d["V_hat"] for d in top5])), 6) if top5 else None,
        "top5_mean_regret": round(float(np.mean([d["regret"] for d in top5])), 6) if top5 else None,
        "best_pair": best_formula,
        "exact_best_formula": exact_best_formula,
        "top10_hits": len(hits),
    }
    if energies:
        best_e = float(min(energies))
        out["best_ising_energy"] = round(best_e, 2)
        if cert is not None:
            out["energy_gap_to_cert"] = round(best_e - cert["cert_min_ising_energy"], 2)
    return out


def evaluate_q3_solutions(sols, meta, df_pool, lookup, ground_sols=None, energies=None, cert=None):
    form = meta.get("formulation", "")
    df_sorted = df_pool.sort_values("V_hat").reset_index(drop=True)
    exact_min = float(df_sorted.iloc[0]["V_hat"])
    exact_best = str(df_sorted.iloc[0]["formula"])
    formula_map = {(r["A"], r["B"], r["X"]): str(r["formula"]) for _, r in df_sorted.iterrows()}
    rank_map = {(r["A"], r["B"], r["X"]): int(idx + 1) for idx, (_, r) in enumerate(df_sorted.iterrows())}
    top15 = set(zip(df_sorted.head(15)["A"], df_sorted.head(15)["B"], df_sorted.head(15)["X"]))

    g_list = ground_sols if ground_sols is not None else sols
    nY = int(meta.get("nY", 0))
    active_ax = meta.get("active_ax", []) if nY > 0 else []
    a2i = {e: i for i, e in enumerate(meta.get("elems_A", []))}
    x2i = {e: i for i, e in enumerate(meta.get("elems_X", []))}

    kA_l, kB_l, kX_l = [], [], []
    prim_feas = 0
    aux_ok_cnt = 0
    strict_feas = 0

    def _decode_one(x):
        if form == "comp_ax_b":
            nAX, nB = meta["nAX"], meta["nB"]
            axi = np.where(x[:nAX] == 1)[0]
            bi = np.where(x[nAX:nAX + nB] == 1)[0]
            kA = len(axi)
            kB = len(bi)
            kX = len(axi)
            oh_ok = (kA == 1 and kB == 1)
            trip = None
            if oh_ok:
                ea, ex = meta["ax_pairs"][int(axi[0])]
                eb = meta["B_elems"][int(bi[0])]
                if ea != eb and (ea, eb, ex) in lookup:
                    trip = (ea, eb, ex)
            return kA, kB, kX, oh_ok, trip, True

        nA, nB, nX = meta["nA"], meta["nB"], meta["nX"]
        off_B, off_X = meta["off_B"], meta["off_X"]
        ai = np.where(x[:nA] == 1)[0]
        bi = np.where(x[off_B:off_B + nB] == 1)[0]
        xi = np.where(x[off_X:off_X + nX] == 1)[0]
        kA, kB, kX = len(ai), len(bi), len(xi)
        oh_ok = (kA == 1 and kB == 1 and kX == 1)
        trip = None
        if oh_ok:
            ea = meta["elems_A"][int(ai[0])]
            eb = meta["elems_B"][int(bi[0])]
            ex = meta["elems_X"][int(xi[0])]
            if ea != eb and (ea, eb, ex) in lookup:
                trip = (ea, eb, ex)
        aux_ok = True
        if nY > 0:
            off_Y = meta["off_Y"]
            for y_idx, (ea, ex) in enumerate(active_ax):
                expected_y = int(x[a2i[ea]] * x[off_X + x2i[ex]])
                if int(x[off_Y + y_idx]) != expected_y:
                    aux_ok = False
                    break
        return kA, kB, kX, oh_ok, trip, aux_ok

    for x in g_list:
        kA, kB, kX, _, trip, aux_ok = _decode_one(x)
        kA_l.append(kA)
        kB_l.append(kB)
        kX_l.append(kX)
        p_ok = (trip is not None)
        if p_ok:
            prim_feas += 1
        if aux_ok:
            aux_ok_cnt += 1
        if p_ok and aux_ok:
            strict_feas += 1

    unique_map = {}
    hits = set()
    for x in sols:
        _, _, _, _, trip, aux_ok = _decode_one(x)
        if trip is not None and aux_ok:
            v = float(lookup[trip])
            f = formula_map[trip]
            unique_map[trip] = {
                "formula": f,
                "V_hat": round(v, 6),
                "regret": round(v - exact_min, 6),
                "pool_rank": rank_map[trip],
            }
            if trip in top15:
                hits.add(trip)

    unique_cands = sorted(unique_map.values(), key=lambda d: (d["V_hat"], d["formula"]))
    best_v = unique_cands[0]["V_hat"] if unique_cands else None
    best_cand = unique_cands[0]["formula"] if unique_cands else None
    top5 = unique_cands[:5]

    n_g = max(1, len(g_list))
    out = {
        "n_returned_shots": len(sols),
        "feasible_shot_count": int(strict_feas),
        "primary_feasible_shot_count": int(prim_feas),
        "y_aux_consistent_count": int(aux_ok_cnt),
        "primary_feasibility_pct": round(100.0 * prim_feas / n_g, 2),
        "y_aux_consistent_pct": round(100.0 * aux_ok_cnt / n_g, 2),
        "strict_feasibility_pct": round(100.0 * strict_feas / n_g, 2),
        "feasibility_pct": round(100.0 * strict_feas / n_g, 2),
        "mean_kA": round(float(np.mean(kA_l)), 3) if kA_l else 0.0,
        "mean_kB": round(float(np.mean(kB_l)), 3) if kB_l else 0.0,
        "mean_kX": round(float(np.mean(kX_l)), 3) if kX_l else 0.0,
        "unique_feasible_count": len(unique_cands),
        "unique_feasible_candidates": [d["formula"] for d in unique_cands],
        "best_V_hat": round(best_v, 6) if best_v is not None else None,
        "exact_min_V_hat": round(exact_min, 6),
        "continuous_regret": round(best_v - exact_min, 6) if best_v is not None else None,
        "gap_to_opt": round(best_v - exact_min, 6) if best_v is not None else None,
        "top5_V_hat": [d["V_hat"] for d in top5],
        "top5_mean_V_hat": round(float(np.mean([d["V_hat"] for d in top5])), 6) if top5 else None,
        "top5_mean_regret": round(float(np.mean([d["regret"] for d in top5])), 6) if top5 else None,
        "best_candidate": best_cand,
        "exact_best_formula": exact_best,
        "top15_hits": len(hits),
    }
    if energies:
        best_e = float(min(energies))
        out["best_ising_energy"] = round(best_e, 2)
        if cert is not None:
            out["energy_gap_to_cert"] = round(best_e - cert["cert_min_ising_energy"], 2)
    return out


# ---------------------------------------------------------------------------
# Compression Attribution (F18) & C_cap Top-10/Top-25 Retention Audit (F19)
# ---------------------------------------------------------------------------

def compute_compression_attribution_table(q3_entries):
    """
    Build the 4-row compression attribution table requested in F18:
      1) Full-space A-B Rosenberg (52x51 = 2,652 aux)
      2) Full-space A-X Rosenberg (52x3 = 156 aux, 264 spins)
      3) Active-subspace A-X HOBO (Model 3A-HOBO for Blind & Full)
      4) Active-subspace Composite (A,X) Bipartite (Model 3A-Comp for Blind & Full)
    """
    rows = [
        {
            "row_id": 1,
            "encoding": "Full-space A-B Rosenberg",
            "pool_scope": "Full 52x52x3",
            "n_primary_vars": 107,
            "n_aux_vars": 52 * 51,
            "ising_spins": 52 + 52 + 3 + 52 * 51 + 1,
            "fits_550_spins": False,
            "aux_compression_vs_AB": 1.0,
            "aux_compression_vs_full_AX": round((52 * 3) / (52 * 51), 4),
            "note": "Baseline A-B pairing; exceeds 550-spin SPQC-550 capacity.",
        },
        {
            "row_id": 2,
            "encoding": "Full-space A-X Rosenberg",
            "pool_scope": "Full 52x52x3",
            "n_primary_vars": 107,
            "n_aux_vars": 52 * 3,
            "ising_spins": 52 + 52 + 3 + 52 * 3 + 1,
            "fits_550_spins": True,
            "aux_compression_vs_AB": round((52 * 51) / (52 * 3), 2),
            "aux_compression_vs_full_AX": 1.0,
            "note": "Axis-reordering alone (52*3=156 aux, 264 spins) already fits 550-spin limit.",
        },
    ]

    row_idx = 3
    for scope_key, enc_label in [
        ("Q3_full_08_Model3A_HOBO", "Active-subspace A-X HOBO (Model 3A-HOBO, Full 08)"),
        ("Q3_blind_12_Model3A_HOBO", "Active-subspace A-X HOBO (Model 3A-HOBO, Blind 12)"),
        ("Q3_full_08_Model3A_Comp", "Active-subspace Composite (A,X)-B (Model 3A-Comp, Full 08)"),
        ("Q3_blind_12_Model3A_Comp", "Active-subspace Composite (A,X)-B (Model 3A-Comp, Blind 12)"),
    ]:
        e = q3_entries[scope_key]
        cert = e["certification"]
        n_aux = cert["n_aux_vars"]
        n_prim = cert["n_primary_vars"]
        spins = cert["ising_spins"]
        rows.append({
            "row_id": row_idx,
            "encoding": enc_label,
            "pool_scope": "Full_08" if "full" in scope_key else "Blind_12",
            "n_primary_vars": n_prim,
            "n_aux_vars": n_aux,
            "ising_spins": spins,
            "fits_550_spins": bool(spins <= 550),
            "ising_couplings": e.get("ising_couplings"),
            "dynamic_range": e.get("dynamic_range"),
            "M_max": cert.get("M_max"),
            "illegal_ground_count": cert["illegal_ground_count"],
            "continuous_regret_top1": cert["continuous_regret_top1"],
            "aux_compression_vs_AB": round(2652 / n_aux, 2) if n_aux > 0 else "inf (0 aux)",
            "aux_compression_vs_full_AX": round(156 / n_aux, 2) if n_aux > 0 else "inf (0 aux)",
            "note": (
                f"Same active-space A-X standard Rosenberg also requires {n_aux} aux vars."
                if n_aux > 0
                else "Zero auxiliary variables via composite (A,X) site encoding."
            ),
        })
        row_idx += 1
    return rows


def compute_ccap_retention_table(pools_cfg):
    """
    Compute exact Top-10 and Top-25 retention counts and dropped Top-25 candidate lists
    directly from the real matrix builders' active-subspace metadata across four domains:
      1. V_hat < c_cap (truncated pool threshold, both all-rows and legal A != B rows)
      2. Q2 / Model 3A-HOBO element-product subspace (elems_A x elems_B [x elems_X] from builder)
      3. Model 3A-Comp composite subspace ((A,X) x B from build_q3_comp_qubo metadata)
      4. Model 3B-ANOVA Top-K element subspace (from build_q3_compact_anova_qubo, Q3 only)
    """
    anova_tk_defaults = {"Q3_full_08": 35, "Q3_blind_12": 25}
    records = []
    for item in pools_cfg:
        if len(item) == 5:
            label, df_pool, default_cap, cap_grid, anova_tk = item
        else:
            label, df_pool, default_cap, cap_grid = item
            anova_tk = anova_tk_defaults.get(label)
        df_s = df_pool.sort_values("V_hat").reset_index(drop=True)
        top10 = df_s.head(10)
        top25 = df_s.head(25)
        is_q3 = ("X" in df_s.columns) and (df_s["X"].nunique() > 1)

        anova_meta = None
        if is_q3 and anova_tk is not None:
            _, anova_meta, _ = build_q3_compact_anova_qubo(df_s, top_k_subspace=int(anova_tk), c_cap=float(default_cap))
            anovA = set(anova_meta["elems_A"])
            anovB = set(anova_meta["elems_B"])
            anovX = set(anova_meta["elems_X"])
            t10_anov = sum(
                1 for a, b, x in zip(top10["A"], top10["B"], top10["X"])
                if a != b and a in anovA and b in anovB and x in anovX
            )
            t25_anov = sum(
                1 for a, b, x in zip(top25["A"], top25["B"], top25["X"])
                if a != b and a in anovA and b in anovB and x in anovX
            )
            dropped_anov = [
                {
                    "rank": int(idx + 1),
                    "formula": str(row["formula"]),
                    "V_hat": round(float(row["V_hat"]), 6),
                }
                for idx, row in top25.iterrows()
                if not (row["A"] != row["B"] and row["A"] in anovA and row["B"] in anovB and row["X"] in anovX)
            ]
        else:
            t10_anov = None
            t25_anov = None
            dropped_anov = None

        for c_cap in cap_grid:
            act_all = df_s[df_s["V_hat"] < c_cap]
            act_legal = act_all[act_all["A"] != act_all["B"]]

            if is_q3:
                _, meta_hobo, _ = build_q3_submodular_hobo_qubo(df_s, c_cap=float(c_cap))
                _, meta_comp, _ = build_q3_comp_qubo(df_s, c_cap=float(c_cap), penalty_mult=1.60)
                actA = set(meta_hobo["elems_A"])
                actB = set(meta_hobo["elems_B"])
                actX = set(meta_hobo["elems_X"])
                actAX = {tuple(p) for p in meta_comp["ax_pairs"]}
                compB = set(meta_comp["elems_B"])

                t10_elem = sum(
                    1 for a, b, x in zip(top10["A"], top10["B"], top10["X"])
                    if a != b and a in actA and b in actB and x in actX
                )
                t25_elem = sum(
                    1 for a, b, x in zip(top25["A"], top25["B"], top25["X"])
                    if a != b and a in actA and b in actB and x in actX
                )
                t10_comp = sum(
                    1 for a, b, x in zip(top10["A"], top10["B"], top10["X"])
                    if a != b and (a, x) in actAX and b in compB
                )
                t25_comp = sum(
                    1 for a, b, x in zip(top25["A"], top25["B"], top25["X"])
                    if a != b and (a, x) in actAX and b in compB
                )
                dropped_elem = [
                    {
                        "rank": int(idx + 1),
                        "formula": str(row["formula"]),
                        "V_hat": round(float(row["V_hat"]), 6),
                    }
                    for idx, row in top25.iterrows()
                    if not (row["A"] != row["B"] and row["A"] in actA and row["B"] in actB and row["X"] in actX)
                ]
                dropped_comp = [
                    {
                        "rank": int(idx + 1),
                        "formula": str(row["formula"]),
                        "V_hat": round(float(row["V_hat"]), 6),
                    }
                    for idx, row in top25.iterrows()
                    if not (row["A"] != row["B"] and (row["A"], row["X"]) in actAX and row["B"] in compB)
                ]
            else:
                _, meta_q2, _ = build_exact8_q2_qubo(df_s, c_cap=float(c_cap), penalty_mult=2.25)
                actA = set(meta_q2["elems_A"])
                actB = set(meta_q2["elems_B"])
                t10_elem = sum(1 for a, b in zip(top10["A"], top10["B"]) if a != b and a in actA and b in actB)
                t25_elem = sum(1 for a, b in zip(top25["A"], top25["B"]) if a != b and a in actA and b in actB)
                t10_comp = t10_elem
                t25_comp = t25_elem
                dropped_elem = [
                    {
                        "rank": int(idx + 1),
                        "formula": str(row["formula"]),
                        "V_hat": round(float(row["V_hat"]), 6),
                    }
                    for idx, row in top25.iterrows()
                    if not (row["A"] != row["B"] and row["A"] in actA and row["B"] in actB)
                ]
                dropped_comp = dropped_elem

            t10_lt = int((top10["V_hat"] < c_cap).sum())
            t25_lt = int((top25["V_hat"] < c_cap).sum())
            t25_legal_lt = int(((top25["V_hat"] < c_cap) & (top25["A"] != top25["B"])).sum())
            dropped_lt_cap = [
                {
                    "rank": int(idx + 1),
                    "formula": str(row["formula"]),
                    "A": str(row["A"]),
                    "B": str(row["B"]),
                    **({"X": str(row["X"])} if is_q3 else {}),
                    "V_hat": round(float(row["V_hat"]), 6),
                }
                for idx, row in top25.iterrows()
                if float(row["V_hat"]) >= c_cap
            ]

            records.append({
                "pool": label,
                "is_default_cap": bool(abs(c_cap - default_cap) < 1e-9),
                "c_cap": float(c_cap),
                "declared_pool_size": int(len(df_s)),
                "declared_min_V_hat": round(float(df_s.iloc[0]["V_hat"]), 6),
                "active_rows_lt_cap": int(len(act_all)),
                "legal_active_rows_lt_cap": int(len(act_legal)),
                "top10_retained_lt_cap": t10_lt,
                "top10_in_element_product_subspace": int(t10_elem),
                "top10_in_composite_ax_subspace": int(t10_comp),
                "top10_in_anova_topk_subspace": int(t10_anov) if t10_anov is not None else None,
                "top10_in_active_subspace": int(t10_elem),
                "top10_max_V_hat": round(float(top10["V_hat"].max()), 6),
                "top25_retained_lt_cap": t25_lt,
                "top25_legal_lt_cap": t25_legal_lt,
                "top25_in_element_product_subspace": int(t25_elem),
                "top25_in_composite_ax_subspace": int(t25_comp),
                "anova_top_k_subspace": int(anova_tk) if anova_tk is not None else None,
                "top25_in_anova_topk_subspace": int(t25_anov) if t25_anov is not None else None,
                "top25_in_active_subspace": int(t25_elem),
                "top25_max_V_hat": round(float(top25["V_hat"].max()), 6),
                "dropped_top25_count_lt_cap": len(dropped_lt_cap),
                "dropped_top25_lt_cap": dropped_lt_cap,
                "dropped_top25_count_element_product": len(dropped_elem),
                "dropped_top25_element_product": dropped_elem,
                "dropped_top25_count_composite_ax": len(dropped_comp),
                "dropped_top25_composite_ax": dropped_comp,
                "dropped_top25_count_anova_topk": len(dropped_anov) if dropped_anov is not None else None,
                "dropped_top25_anova_topk": dropped_anov,
                "dropped_top25_count_subspace": len(dropped_elem),
                "dropped_top25_subspace": dropped_elem,
            })
    return records


def main():
    print("=== Running v3 Certified 8-Bit Q2 & Q3 QUBO/HOBO Engine (F01/F02/F03/F13/F18/F19) ===")
    results = {
        "Q2": {},
        "Q3": {},
        "anova_error_decomposition": {},
        "compression_attribution": [],
        "c_cap_retention": [],
        "certification_summary": [],
    }

    # 1. Q2 Full (07) and Blind (11)
    q2_dfs = {}
    for label, fname, c_cap in [
        ("Q2_full_07", "v2_q2_full_07.csv", 2.0),
        ("Q2_blind_11", "v2_q2_blind_11.csv", 2.5),
    ]:
        dfp = pd.read_csv(os.path.join(OUT_DIR, fname))
        q2_dfs[label] = (dfp, c_cap)
        ising, meta, lookup = build_exact8_q2_qubo(dfp, c_cap=c_cap, penalty_mult=2.25)
        cert = certify_global_ground_states(ising, meta, dfp, lookup, task_kind="q2", verify_zero_illegal=True)

        sol_pack = _solve_ising_multi_solver(ising, n_sa=50, n_tabu=50, seed_base=1000)
        ev_sa = evaluate_q2_solutions(
            sol_pack["sa_sols"], meta, dfp, lookup,
            ground_sols=sol_pack["sa_ground"], energies=sol_pack["sa_energies"], cert=cert,
        )
        ev_tabu = evaluate_q2_solutions(
            sol_pack["tabu_sols"], meta, dfp, lookup,
            ground_sols=sol_pack["tabu_ground"], energies=sol_pack["tabu_energies"], cert=cert,
        )
        n_couplings = int(np.count_nonzero(np.triu(ising, k=1)))
        results["Q2"][label] = {
            "spins": sol_pack["spins"],
            "nA_active": meta["nA"],
            "nB_active": meta["nB"],
            "M_max": meta.get("M_max"),
            "P_block": meta.get("P_block"),
            "bit_width": sol_pack["bit_width"],
            "dynamic_range": sol_pack["dynamic_range"],
            "max_abs_coeff": sol_pack["max_abs_coeff"],
            "ising_couplings": n_couplings,
            "certification": cert,
            "SA": {**ev_sa, "n_runs": 50, "runtime_sec": sol_pack["sa_runtime_sec"]},
            "Tabu": {**ev_tabu, "n_runs": 50, "runtime_sec": sol_pack["tabu_runtime_sec"]},
        }
        results["certification_summary"].append({"matrix_id": label, **cert})
        print(
            f"\n[{label}] spins={sol_pack['spins']} M_max={meta.get('M_max')} bw={sol_pack['bit_width']} "
            f"cert_E={cert['cert_min_ising_energy']} legal_gs={cert['legal_ground_count']} "
            f"illegal_gs={cert['illegal_ground_count']} regret={cert['continuous_regret_top1']}"
        )
        print("  SA  :", {k: ev_sa[k] for k in ["n_returned_shots", "feasibility_pct", "unique_feasible_count", "best_V_hat", "continuous_regret", "top10_hits"]}, f"({sol_pack['sa_runtime_sec']}s)")
        print("  Tabu:", {k: ev_tabu[k] for k in ["n_returned_shots", "feasibility_pct", "unique_feasible_count", "best_V_hat", "continuous_regret", "top10_hits"]}, f"({sol_pack['tabu_runtime_sec']}s)")

    # 2. Q3 Full (08) and Blind (12): Model 3A-Comp, Model 3A-HOBO, Model 3B-ANOVA
    q3_dfs = {}
    for label, fname, c_cap, tk in [
        ("Q3_full_08", "v2_q3_full_08.csv", 1.25, 35),
        ("Q3_blind_12", "v2_q3_blind_12.csv", 1.50, 25),
    ]:
        dfp = pd.read_csv(os.path.join(OUT_DIR, fname))
        q3_dfs[label] = (dfp, c_cap)

        # Model 3A-Comp: Zero-Aux Composite (A,X)-B Bipartite QUBO
        ising_comp, meta_comp, lkp_comp = build_q3_comp_qubo(dfp, c_cap=c_cap, penalty_mult=1.60)
        cert_comp = certify_global_ground_states(ising_comp, meta_comp, dfp, lkp_comp, task_kind="q3", verify_zero_illegal=True)
        pack_comp = _solve_ising_multi_solver(ising_comp, n_sa=50, n_tabu=50, seed_base=1500)
        sa_comp = evaluate_q3_solutions(pack_comp["sa_sols"], meta_comp, dfp, lkp_comp, ground_sols=pack_comp["sa_ground"], energies=pack_comp["sa_energies"], cert=cert_comp)
        tb_comp = evaluate_q3_solutions(pack_comp["tabu_sols"], meta_comp, dfp, lkp_comp, ground_sols=pack_comp["tabu_ground"], energies=pack_comp["tabu_energies"], cert=cert_comp)

        # Model 3A-HOBO: Repaired Cumulative-P_aux Submodular HOBO->QUBO
        ising_hobo, meta_hobo, lkp_hobo = build_q3_submodular_hobo_qubo(dfp, c_cap=c_cap)
        cert_hobo = certify_global_ground_states(ising_hobo, meta_hobo, dfp, lkp_hobo, task_kind="q3", verify_zero_illegal=True)
        pack_hobo = _solve_ising_multi_solver(ising_hobo, n_sa=50, n_tabu=50, seed_base=2000)
        sa_hobo = evaluate_q3_solutions(pack_hobo["sa_sols"], meta_hobo, dfp, lkp_hobo, ground_sols=pack_hobo["sa_ground"], energies=pack_hobo["sa_energies"], cert=cert_hobo)
        tb_hobo = evaluate_q3_solutions(pack_hobo["tabu_sols"], meta_hobo, dfp, lkp_hobo, ground_sols=pack_hobo["tabu_ground"], energies=pack_hobo["tabu_energies"], cert=cert_hobo)

        # Model 3B-ANOVA: Repaired Uniform 8x Compact ANOVA-QUBO
        ising_anov, meta_anov, lkp_anov = build_q3_compact_anova_qubo(dfp, top_k_subspace=tk, c_cap=c_cap)
        cert_anov = certify_global_ground_states(ising_anov, meta_anov, dfp, lkp_anov, task_kind="q3", verify_zero_illegal=True)
        pack_anov = _solve_ising_multi_solver(ising_anov, n_sa=50, n_tabu=50, seed_base=3000)
        sa_anov = evaluate_q3_solutions(pack_anov["sa_sols"], meta_anov, dfp, lkp_anov, ground_sols=pack_anov["sa_ground"], energies=pack_anov["sa_energies"], cert=cert_anov)
        tb_anov = evaluate_q3_solutions(pack_anov["tabu_sols"], meta_anov, dfp, lkp_anov, ground_sols=pack_anov["tabu_ground"], energies=pack_anov["tabu_energies"], cert=cert_anov)

        comp_key = f"{label}_Model3A_Comp"
        hobo_key = f"{label}_Model3A_HOBO"
        anov_key = f"{label}_Model3B_CompactANOVA"

        results["Q3"][comp_key] = {
            "spins": pack_comp["spins"],
            "nAX_active": meta_comp["nAX"],
            "nB_active": meta_comp["nB"],
            "aux_vars": 0,
            "M_max": meta_comp.get("M_max"),
            "P_block": meta_comp.get("P_block"),
            "bit_width": pack_comp["bit_width"],
            "dynamic_range": pack_comp["dynamic_range"],
            "max_abs_coeff": pack_comp["max_abs_coeff"],
            "ising_couplings": int(np.count_nonzero(np.triu(ising_comp, k=1))),
            "certification": cert_comp,
            "SA": {**sa_comp, "n_runs": 50, "runtime_sec": pack_comp["sa_runtime_sec"]},
            "Tabu": {**tb_comp, "n_runs": 50, "runtime_sec": pack_comp["tabu_runtime_sec"]},
        }
        results["Q3"][hobo_key] = {
            "spins": pack_hobo["spins"],
            "aux_vars": meta_hobo["nY"],
            "aux_compression_vs_v1": round((52 * 51) / max(1, meta_hobo["nY"]), 2),
            "aux_compression_vs_full_AX": round((52 * 3) / max(1, meta_hobo["nY"]), 2),
            "M_max": meta_hobo.get("M_max"),
            "P_block": meta_hobo.get("P_block"),
            "max_P_aux": meta_hobo.get("max_P_aux"),
            "bit_width": pack_hobo["bit_width"],
            "dynamic_range": pack_hobo["dynamic_range"],
            "max_abs_coeff": pack_hobo["max_abs_coeff"],
            "ising_couplings": int(np.count_nonzero(np.triu(ising_hobo, k=1))),
            "certification": cert_hobo,
            "SA": {**sa_hobo, "n_runs": 50, "runtime_sec": pack_hobo["sa_runtime_sec"]},
            "Tabu": {**tb_hobo, "n_runs": 50, "runtime_sec": pack_hobo["tabu_runtime_sec"]},
        }
        results["Q3"][anov_key] = {
            "spins": pack_anov["spins"],
            "aux_vars": 0,
            "anova_r2": meta_anov["anova_r2"],
            "M_max": meta_anov.get("M_max"),
            "P_block": meta_anov.get("P_block"),
            "distinct_legal_energy_levels": meta_anov.get("distinct_legal_energy_levels"),
            "level_occupancies": meta_anov.get("level_occupancies"),
            "bit_width": pack_anov["bit_width"],
            "dynamic_range": pack_anov["dynamic_range"],
            "max_abs_coeff": pack_anov["max_abs_coeff"],
            "ising_couplings": int(np.count_nonzero(np.triu(ising_anov, k=1))),
            "certification": cert_anov,
            "SA": {**sa_anov, "n_runs": 50, "runtime_sec": pack_anov["sa_runtime_sec"]},
            "Tabu": {**tb_anov, "n_runs": 50, "runtime_sec": pack_anov["tabu_runtime_sec"]},
        }

        # 3-Layer Error Decomposition for Model 3B-ANOVA (F02)
        results["anova_error_decomposition"][label] = {
            "layer_1_surrogate_vs_float_anova": meta_anov.get("layer_a_surrogate_vs_float_anova"),
            "layer_2_float_anova_vs_8bit_ising": {
                "M_max": meta_anov.get("M_max"),
                "distinct_legal_energy_levels": meta_anov.get("distinct_legal_energy_levels"),
                "level_occupancies": meta_anov.get("level_occupancies"),
                "quantization_note": meta_anov.get("quantization_note"),
                "spearman_float_anova_vs_int": meta_anov.get("spearman_float_anova_vs_int"),
                "spearman_Vhat_vs_int": meta_anov.get("spearman_Vhat_vs_int"),
                "cert_min_ising_energy": cert_anov["cert_min_ising_energy"],
                "cert_min_qubo_energy": cert_anov["cert_min_qubo_energy"],
                "legal_ground_count": cert_anov["legal_ground_count"],
                "illegal_ground_count": cert_anov["illegal_ground_count"],
                "ground_state_degeneracy": cert_anov["ground_state_degeneracy"],
                "ground_state_V_hat_min": cert_anov["ground_state_candidates"][0]["V_hat"] if cert_anov["ground_state_candidates"] else None,
                "ground_state_V_hat_max": cert_anov["ground_state_candidates"][-1]["V_hat"] if cert_anov["ground_state_candidates"] else None,
                "continuous_regret_top1": cert_anov["continuous_regret_top1"],
                "continuous_regret_worst_gs": cert_anov["continuous_regret_worst_gs"],
            },
            "layer_3_8bit_ising_vs_solvers": {
                "SA_best_ising_energy": sa_anov.get("best_ising_energy"),
                "SA_energy_gap_to_cert": sa_anov.get("energy_gap_to_cert"),
                "SA_best_V_hat": sa_anov.get("best_V_hat"),
                "SA_regret": sa_anov.get("continuous_regret"),
                "SA_feasibility_pct": sa_anov.get("feasibility_pct"),
                "Tabu_best_ising_energy": tb_anov.get("best_ising_energy"),
                "Tabu_energy_gap_to_cert": tb_anov.get("energy_gap_to_cert"),
                "Tabu_best_V_hat": tb_anov.get("best_V_hat"),
                "Tabu_regret": tb_anov.get("continuous_regret"),
                "Tabu_feasibility_pct": tb_anov.get("feasibility_pct"),
            },
        }

        for mid, cert in [(comp_key, cert_comp), (hobo_key, cert_hobo), (anov_key, cert_anov)]:
            results["certification_summary"].append({"matrix_id": mid, **cert})

        print(f"\n[{comp_key}] spins={pack_comp['spins']} aux=0 M_max={meta_comp.get('M_max')} cert_E={cert_comp['cert_min_ising_energy']} legal_gs={cert_comp['legal_ground_count']} illegal_gs={cert_comp['illegal_ground_count']} regret={cert_comp['continuous_regret_top1']}")
        print("  SA  :", {k: sa_comp[k] for k in ["n_returned_shots", "feasibility_pct", "unique_feasible_count", "best_V_hat", "continuous_regret", "top15_hits"]}, f"({pack_comp['sa_runtime_sec']}s)")
        print("  Tabu:", {k: tb_comp[k] for k in ["n_returned_shots", "feasibility_pct", "unique_feasible_count", "best_V_hat", "continuous_regret", "top15_hits"]}, f"({pack_comp['tabu_runtime_sec']}s)")
        print(f"[{hobo_key}] spins={pack_hobo['spins']} aux={meta_hobo['nY']} M_max={meta_hobo.get('M_max')} cert_E={cert_hobo['cert_min_ising_energy']} legal_gs={cert_hobo['legal_ground_count']} illegal_gs={cert_hobo['illegal_ground_count']} regret={cert_hobo['continuous_regret_top1']}")
        print("  SA  :", {k: sa_hobo[k] for k in ["n_returned_shots", "feasibility_pct", "y_aux_consistent_pct", "unique_feasible_count", "best_V_hat", "continuous_regret", "top15_hits"]}, f"({pack_hobo['sa_runtime_sec']}s)")
        print("  Tabu:", {k: tb_hobo[k] for k in ["n_returned_shots", "feasibility_pct", "y_aux_consistent_pct", "unique_feasible_count", "best_V_hat", "continuous_regret", "top15_hits"]}, f"({pack_hobo['tabu_runtime_sec']}s)")
        print(f"[{anov_key}] spins={pack_anov['spins']} aux=0 M_max={meta_anov.get('M_max')} levels={meta_anov.get('distinct_legal_energy_levels')} cert_E={cert_anov['cert_min_ising_energy']} legal_gs={cert_anov['legal_ground_count']} illegal_gs={cert_anov['illegal_ground_count']} regret={cert_anov['continuous_regret_top1']}")
        print("  SA  :", {k: sa_anov[k] for k in ["n_returned_shots", "feasibility_pct", "unique_feasible_count", "best_V_hat", "continuous_regret", "top15_hits"]}, f"({pack_anov['sa_runtime_sec']}s)")
        print("  Tabu:", {k: tb_anov[k] for k in ["n_returned_shots", "feasibility_pct", "unique_feasible_count", "best_V_hat", "continuous_regret", "top15_hits"]}, f"({pack_anov['tabu_runtime_sec']}s)")

    # 3. 4-Row Compression Attribution Table (F18)
    results["compression_attribution"] = compute_compression_attribution_table(results["Q3"])

    # 4. C_cap Top-10 / Top-25 Retention Table (F19)
    results["c_cap_retention"] = compute_ccap_retention_table([
        ("Q2_full_07", q2_dfs["Q2_full_07"][0], 2.0, [1.5, 2.0, 2.5, 3.0, 4.0]),
        ("Q2_blind_11", q2_dfs["Q2_blind_11"][0], 2.5, [1.5, 2.0, 2.5, 3.0, 4.0]),
        ("Q3_full_08", q3_dfs["Q3_full_08"][0], 1.25, [1.0, 1.25, 1.5, 2.0, 2.5]),
        ("Q3_blind_12", q3_dfs["Q3_blind_12"][0], 1.50, [1.0, 1.25, 1.5, 2.0, 2.5]),
    ])

    out_path = os.path.join(OUT_DIR, "v2_qubo_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    print(f"\nSaved {out_path} successfully.")


if __name__ == "__main__":
    main()
