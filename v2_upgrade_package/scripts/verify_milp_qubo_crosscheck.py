#!/usr/bin/env python3
"""
Independent HiGHS MILP (McCormick Linearization) Cross-Check for All 8 Q2/Q3 Integer Ising Matrices.

For each of the 8 active-subspace matrices:
  1. Verifies SHA-256 integrity of the 4 declared pool CSVs against v2_q1_diagnostics.json.
  2. Builds the exact 8-bit integer Ising matrix H in [-127, 127] and computes its int64 SHA-256.
  3. Recovers the exact integer QUBO (Q, C) via x^T Q x = -s^T H s + C.
  4. Linearizes min_{x in {0,1}^n} x^T Q x via Fortet/McCormick constraints:
       w_{ij} - x_i <= 0,  w_{ij} - x_j <= 0,  x_i + x_j - w_{ij} <= 1   (x_i in {0,1}, w_{ij} in [0,1])
     and solves to provable global optimality (mip_rel_gap = 0.0) using SciPy/HiGHS.
  5. Cross-checks exact global equality against structured primary-variable enumeration.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPTS_DIR)
V2_DIR = os.path.join(PKG_DIR, "v2_optimized")
RES_DIR = os.path.join(V2_DIR, "results")
DEFAULT_OUT = os.path.join(RES_DIR, "v2_milp_crosscheck.json")

if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

import v2_qubo_q2_q3 as q23


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _matrix_int64_sha256(mat: np.ndarray) -> str:
    arr = np.ascontiguousarray(mat, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def solve_qubo_milp_mccormick(Q: np.ndarray) -> dict:
    """Exact Fortet/McCormick MILP linearization of min_{x in {0,1}^n} x^T Q x using HiGHS."""
    n = int(Q.shape[0])
    Q_sym = 0.5 * (Q + Q.T)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(Q_sym[i, j]) > 0]
    m = len(pairs)

    c = np.zeros(n + m, dtype=float)
    c[:n] = np.diag(Q_sym)
    for k, (i, j) in enumerate(pairs):
        c[n + k] = 2.0 * float(Q_sym[i, j])

    rows, cols, vals, ub = [], [], [], []
    for k, (i, j) in enumerate(pairs):
        r0 = 3 * k
        rows.extend([r0, r0, r0 + 1, r0 + 1, r0 + 2, r0 + 2, r0 + 2])
        cols.extend([n + k, i, n + k, j, n + k, i, j])
        vals.extend([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
        ub.extend([0.0, 0.0, 1.0])

    integrality = np.zeros(n + m, dtype=int)
    integrality[:n] = 1
    bounds = Bounds(np.zeros(n + m, dtype=float), np.ones(n + m, dtype=float))
    if m > 0:
        mat = csc_matrix((vals, (rows, cols)), shape=(3 * m, n + m))
        constraints = LinearConstraint(mat, np.full(3 * m, -np.inf, dtype=float), np.array(ub, dtype=float))
    else:
        constraints = ()

    t0 = time.perf_counter()
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints, options={"mip_rel_gap": 0.0})
    dt = time.perf_counter() - t0

    x_sol = np.round(res.x[:n]).astype(np.int64)
    exact_q_val = int(round(float(x_sol @ Q_sym @ x_sol)))
    mip_gap = float(getattr(res, "mip_gap", 0.0) if getattr(res, "mip_gap", None) is not None else 0.0)
    if not math.isfinite(mip_gap) or not math.isfinite(float(res.fun)):
        raise ValueError(f"Non-finite MILP return: status={res.status}, fun={res.fun}, mip_gap={mip_gap}")
    return {
        "status": int(res.status),
        "success": bool(res.success),
        "mip_gap": mip_gap,
        "milp_obj": int(round(float(res.fun))),
        "x_binary_qubo_val": exact_q_val,
        "n_binary_vars": n,
        "n_mccormick_aux_vars": m,
        "n_linear_constraints": 3 * m,
        "solve_sec": round(dt, 4),
    }


def run_milp_crosscheck(verify_saved: bool = True, write_json: bool = False, out_path: str = DEFAULT_OUT) -> dict:
    diag = json.load(open(os.path.join(RES_DIR, "v2_q1_diagnostics.json"), "r", encoding="utf-8"))
    qubo = json.load(open(os.path.join(RES_DIR, "v2_qubo_results.json"), "r", encoding="utf-8"))
    exp_sha = diag["exported_csv_sha256"]

    pool_files = {
        "q2_full": ("v2_q2_full_07.csv", "q2"),
        "q2_blind": ("v2_q2_blind_11.csv", "q2"),
        "q3_full": ("v2_q3_full_08.csv", "q3"),
        "q3_blind": ("v2_q3_blind_12.csv", "q3"),
    }
    pool_dfs = {}
    pool_shas = {}
    for key, (csv_fn, _) in pool_files.items():
        p = os.path.join(RES_DIR, csv_fn)
        sha = _sha256_file(p)
        assert sha == exp_sha[csv_fn], f"Pool CSV SHA mismatch for {csv_fn}"
        pool_dfs[key] = pd.read_csv(p)
        pool_shas[key] = sha

    specs = [
        ("Q2_full_07", "q2", "q2_full", lambda df: q23.build_exact8_q2_qubo(df, c_cap=2.0, penalty_mult=2.25)),
        ("Q2_blind_11", "q2", "q2_blind", lambda df: q23.build_exact8_q2_qubo(df, c_cap=2.5, penalty_mult=2.25)),
        ("Q3_full_08_Model3A_Comp", "q3", "q3_full", lambda df: q23.build_q3_comp_qubo(df, c_cap=1.25, penalty_mult=1.60)),
        ("Q3_full_08_Model3A_HOBO", "q3", "q3_full", lambda df: q23.build_q3_submodular_hobo_qubo(df, c_cap=1.25)),
        ("Q3_full_08_Model3B_CompactANOVA", "q3", "q3_full", lambda df: q23.build_q3_compact_anova_qubo(df, top_k_subspace=35, c_cap=1.25)),
        ("Q3_blind_12_Model3A_Comp", "q3", "q3_blind", lambda df: q23.build_q3_comp_qubo(df, c_cap=1.50, penalty_mult=1.60)),
        ("Q3_blind_12_Model3A_HOBO", "q3", "q3_blind", lambda df: q23.build_q3_submodular_hobo_qubo(df, c_cap=1.50)),
        ("Q3_blind_12_Model3B_CompactANOVA", "q3", "q3_blind", lambda df: q23.build_q3_compact_anova_qubo(df, top_k_subspace=25, c_cap=1.50)),
    ]

    certs_by_id = {c["matrix_id"]: c for c in qubo["certification_summary"]}
    rows = []
    for mid, kind, pkey, builder in specs:
        dfp = pool_dfs[pkey]
        H, meta, lookup = builder(dfp)
        h_sha = _matrix_int64_sha256(H)
        Q, C = q23._recover_qubo_from_ising_sec92(H)
        cert = q23.certify_global_ground_states(H, meta, dfp, lookup, task_kind=kind, verify_zero_illegal=True)
        milp_res = solve_qubo_milp_mccormick(Q)
        milp_ising_min = int(milp_res["milp_obj"] - C)

        assert milp_res["status"] == 0 and milp_res["success"], f"{mid}: MILP did not succeed ({milp_res})"
        assert milp_res["mip_gap"] <= 1e-12, f"{mid}: non-zero mip_gap={milp_res['mip_gap']}"
        assert milp_res["milp_obj"] == milp_res["x_binary_qubo_val"], f"{mid}: McCormick relaxation gap != 0"
        assert milp_res["milp_obj"] == cert["cert_min_qubo_energy"], (
            f"{mid}: MILP QUBO min {milp_res['milp_obj']} != cert {cert['cert_min_qubo_energy']}"
        )
        assert milp_ising_min == cert["cert_min_ising_energy"], (
            f"{mid}: MILP Ising min {milp_ising_min} != cert {cert['cert_min_ising_energy']}"
        )
        assert cert["cert_min_ising_energy"] == certs_by_id[mid]["cert_min_ising_energy"], (
            f"{mid}: saved cert mismatch"
        )

        rows.append({
            "matrix_id": mid,
            "pool_csv": pool_files[pkey][0],
            "pool_csv_sha256": pool_shas[pkey],
            "spins": int(H.shape[0]),
            "matrix_int64_sha256": h_sha,
            "ising_offset_C": int(C),
            "cert_min_qubo_energy": int(cert["cert_min_qubo_energy"]),
            "cert_min_ising_energy": int(cert["cert_min_ising_energy"]),
            "milp_status": milp_res["status"],
            "milp_mip_gap": milp_res["mip_gap"],
            "milp_min_qubo_energy": milp_res["milp_obj"],
            "milp_min_ising_energy": milp_ising_min,
            "n_binary_vars": milp_res["n_binary_vars"],
            "n_mccormick_aux_vars": milp_res["n_mccormick_aux_vars"],
            "n_linear_constraints": milp_res["n_linear_constraints"],
            "solve_sec": milp_res["solve_sec"],
            "exact_match_with_structured_cert": True,
        })

    payload = {
        "protocol": "Exact Fortet/McCormick MILP linearization via SciPy/HiGHS (mip_rel_gap=0.0)",
        "provenance_note": (
            "Independent MILP cross-check prompted by external manuscript review [15]; "
            "implemented, executed, and verified directly by the team in this standalone script."
        ),
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "solver": "scipy.optimize.milp (HiGHS)",
        },
        "n_matrices": len(rows),
        "all_optimal_zero_gap": all(r["milp_status"] == 0 and r["milp_mip_gap"] == 0.0 for r in rows),
        "all_exact_match_structured_cert": all(r["exact_match_with_structured_cert"] for r in rows),
        "matrices": rows,
    }

    if write_json:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
    elif verify_saved:
        assert os.path.isfile(out_path), f"Missing {out_path}; run with --write-json first"
        saved = json.load(open(out_path, "r", encoding="utf-8"))
        saved_by_id = {r["matrix_id"]: r for r in saved["matrices"]}
        assert len(saved_by_id) == 8
        for r in rows:
            sr = saved_by_id[r["matrix_id"]]
            for fk in (
                "pool_csv_sha256",
                "spins",
                "matrix_int64_sha256",
                "ising_offset_C",
                "cert_min_qubo_energy",
                "cert_min_ising_energy",
                "milp_status",
                "milp_mip_gap",
                "milp_min_qubo_energy",
                "milp_min_ising_energy",
                "n_binary_vars",
                "n_mccormick_aux_vars",
                "n_linear_constraints",
            ):
                assert sr[fk] == r[fk], f"{r['matrix_id']}.{fk}: saved={sr[fk]} != live={r[fk]}"
    return payload


def main():
    parser = argparse.ArgumentParser(description="Independent Fortet/McCormick MILP Cross-Check (T06)")
    parser.add_argument("--write-json", action="store_true", help="Write results/v2_milp_crosscheck.json")
    args = parser.parse_args()

    payload = run_milp_crosscheck(verify_saved=not args.write_json, write_json=args.write_json)
    for r in payload["matrices"]:
        print(
            f"[MILP-PASS] {r['matrix_id']:34s} spins={r['spins']:2d} Q*={r['milp_min_qubo_energy']:4d} "
            f"C={r['ising_offset_C']:4d} E*={r['milp_min_ising_energy']:5d} gap={r['milp_mip_gap']:.1f} "
            f"sha12={r['matrix_int64_sha256'][:12]} ({r['solve_sec']:.3f}s)"
        )
    print(f"[PASS] All {payload['n_matrices']}/8 matrices MILP-verified (mip_gap=0, exact match with structured cert).")


if __name__ == "__main__":
    main()
