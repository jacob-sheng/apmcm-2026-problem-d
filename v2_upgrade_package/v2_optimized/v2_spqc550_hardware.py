#!/usr/bin/env python3
"""
SPQC-550 Real Optical Quantum Computer (CIM) Execution & Independent Verification
for v2 Compact Lossless Active-Subspace Exact-8-Bit Ising Formulations.

Executes at most 3 real SPQC-550 tasks (strictly <= 550 spins, int64, symmetric,
zero diagonal, [-127, 127], verified <= 8-bit):
  1. apmcm26_v2_q2_blind_12  : Q2 Blind Pool (11, X=NO2)             - 12 spins, 8-bit
  2. apmcm26_v2_q3_anova_24  : Q3 Blind Pool (12, Model 3B ANOVA)    - 24 spins, 0 aux, 8-bit
  3. apmcm26_v2_q3_submod_33 : Q3 Blind Pool (12, Model 3A Submod)   - 33 spins, 13 aux, 7-bit

Credentials are read strictly from environment variables KAIWU_USER_ID and
KAIWU_SDK_CODE (never hardcoded or logged). Supports --dry-run, --run-spqc,
and --verify-only modes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import kaiwu as kw

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)
from common_physics import safe_ising_bit_width, sort_ising_solutions
from v2_qubo_q2_q3 import (
    build_exact8_q2_qubo,
    build_q3_compact_anova_qubo,
    build_q3_submodular_hobo_qubo,
    evaluate_q2_solutions,
    evaluate_q3_solutions,
)

RES_DIR = os.path.join(CUR_DIR, "results")
CKPT_DIR = os.path.join(RES_DIR, "spqc_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def matrix_sha256(mat: np.ndarray) -> str:
    arr = np.ascontiguousarray(mat, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def verify_hardware_matrix(mat: np.ndarray, name: str) -> dict:
    """Strict SPQC-550 hardware compatibility assertions."""
    J = np.asarray(mat, dtype=np.int64)
    assert J.ndim == 2 and J.shape[0] == J.shape[1], f"{name}: must be square"
    assert J.shape[0] <= 550, f"{name}: exceeds 550 spins ({J.shape[0]})"
    assert np.array_equal(J, J.T), f"{name}: Ising matrix must be symmetric"
    assert np.all(np.diag(J) == 0), f"{name}: Ising diagonal must be zero"
    assert int(J.min()) >= -127 and int(J.max()) <= 127, f"{name}: out of [-127, 127]"
    bw = safe_ising_bit_width(J)
    assert bw <= 8, f"{name}: bit width {bw} > 8"
    dr = float(kw.get_dynamic_range_metric(J))
    return {
        "spins": int(J.shape[0]),
        "min_val": int(J.min()),
        "max_val": int(J.max()),
        "bit_width": int(bw),
        "dynamic_range": round(dr, 3),
        "sha256": matrix_sha256(J),
    }


def decode_raw_spins(raw_sols: np.ndarray, J: np.ndarray, task_kind: str, meta: dict, df_pool: pd.DataFrame, lookup: dict) -> dict:
    """
    Independently decode every raw Ising spin vector s in {-1, +1}^N returned by
    SPQC-550 (or classical solver) via x = ((s[:-1] * s[-1]) + 1) // 2, check all
    hard one-hot / A!=B / blind-pool constraints, and compute exact Ising energy
    E(s) = -s^T J s alongside physical candidate ranking.
    """
    arr = np.asarray(raw_sols, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_samples = int(arr.shape[0])
    energies = -np.einsum("ij,jk,ik->i", arr, J.astype(float), arr)
    order = np.argsort(energies)
    arr = arr[order]
    energies = energies[order]

    # Exact rank lookup in pool
    rank_map = {}
    formula_map = {}
    for idx_r, r in df_pool.reset_index(drop=True).iterrows():
        if task_kind == "q2":
            key = (r["A"], r["B"])
        else:
            key = (r["A"], r["B"], r["X"])
        rank_map[key] = int(idx_r + 1)
        formula_map[key] = str(r["formula"])

    top_k = 10 if task_kind == "q2" else 15
    top_set = set(list(rank_map.keys())[:top_k])

    decoded_rows = []
    feas_cnt = 0
    hits_top = set()
    best_feas = None

    for i in range(n_samples):
        s = arr[i]
        s_aux = s[-1]
        x = (((s[:-1] * s_aux) + 1) // 2).astype(int)
        e_val = float(energies[i])

        if task_kind == "q2":
            nA, nB = meta["nA"], meta["nB"]
            ai = np.where(x[:nA] == 1)[0]
            bi = np.where(x[nA:nA + nB] == 1)[0]
            kA, kB, kX = len(ai), len(bi), 1
            is_feas = False
            cand_formula = "INFEASIBLE"
            v_hat = None
            pool_rank = None
            if kA == 1 and kB == 1:
                ea, eb = meta["elems_A"][ai[0]], meta["elems_B"][bi[0]]
                if ea != eb and (ea, eb) in lookup:
                    is_feas = True
                    v_hat = float(lookup[(ea, eb)])
                    cand_formula = formula_map[(ea, eb)]
                    pool_rank = rank_map[(ea, eb)]
                    if (ea, eb) in top_set:
                        hits_top.add(cand_formula)
        else:
            nA, nB, nX = meta["nA"], meta["nB"], meta["nX"]
            off_B, off_X = meta["off_B"], meta["off_X"]
            ai = np.where(x[:nA] == 1)[0]
            bi = np.where(x[off_B:off_B + nB] == 1)[0]
            xi = np.where(x[off_X:off_X + nX] == 1)[0]
            kA, kB, kX = len(ai), len(bi), len(xi)
            is_feas = False
            cand_formula = "INFEASIBLE"
            v_hat = None
            pool_rank = None
            if kA == 1 and kB == 1 and kX == 1:
                ea, eb, ex = meta["elems_A"][ai[0]], meta["elems_B"][bi[0]], meta["elems_X"][xi[0]]
                if ea != eb and (ea, eb, ex) in lookup:
                    is_feas = True
                    v_hat = float(lookup[(ea, eb, ex)])
                    cand_formula = formula_map[(ea, eb, ex)]
                    pool_rank = rank_map[(ea, eb, ex)]
                    if (ea, eb, ex) in top_set:
                        hits_top.add(cand_formula)

        if is_feas:
            feas_cnt += 1
            if best_feas is None or v_hat < best_feas["V_hat"]:
                best_feas = {
                    "formula": cand_formula,
                    "V_hat": round(v_hat, 4),
                    "pool_rank": pool_rank,
                    "ising_energy": round(e_val, 2),
                }
        decoded_rows.append({
            "sample_idx": i,
            "ising_energy": round(e_val, 2),
            "kA": int(kA), "kB": int(kB), "kX": int(kX),
            "feasible": bool(is_feas),
            "formula": cand_formula,
            "V_hat": round(v_hat, 4) if v_hat is not None else None,
            "pool_rank": pool_rank,
        })

    exact_min_v = float(df_pool["V_hat"].min())
    return {
        "n_samples": n_samples,
        "feasible_count": feas_cnt,
        "feasible_rate_pct": round(100.0 * feas_cnt / max(1, n_samples), 2),
        "best_ising_energy": round(float(energies[0]), 2),
        "best_feasible": best_feas,
        "exact_min_V_hat": round(exact_min_v, 4),
        "gap_to_opt": round(best_feas["V_hat"] - exact_min_v, 4) if best_feas else None,
        "unique_topK_hits": sorted(list(hits_top)),
        "unique_topK_count": len(hits_top),
        "top_k_target": top_k,
        "samples": decoded_rows,
    }


def build_all_tasks():
    q2b = pd.read_csv(os.path.join(RES_DIR, "v2_q2_blind_11.csv"))
    q3b = pd.read_csv(os.path.join(RES_DIR, "v2_q3_blind_12.csv"))

    # Task 1: Q2 Blind (11), 12 spins, exact 8-bit
    J1, m1, l1 = build_exact8_q2_qubo(q2b, c_cap=2.5, penalty_mult=2.25)
    # Task 2: Q3 Blind (12) Model 3B Zero-Aux Compact ANOVA, 24 spins, exact 8-bit
    J2, m2, l2 = build_q3_compact_anova_qubo(q3b, top_k_subspace=25, c_cap=1.50)
    # Task 3: Q3 Blind (12) Model 3A Exact Submodular HOBO->QUBO, 33 spins, 13 aux, 7-bit
    J3, m3, l3 = build_q3_submodular_hobo_qubo(q3b, c_cap=1.50)

    tasks = [
        {
            "task_name": "apmcm26_v2_q2_blind_12",
            "label": "v2 Q2 Blind (11) Active-Subspace",
            "scope": "Q2_blind_11",
            "kind": "q2",
            "aux_vars": 0,
            "J": np.asarray(J1, dtype=np.int64),
            "meta": m1,
            "lookup": l1,
            "df": q2b,
        },
        {
            "task_name": "apmcm26_v2_q3_anova_24",
            "label": "v2 Q3 Blind (12) Compact ANOVA (Model 3B)",
            "scope": "Q3_blind_12",
            "kind": "q3",
            "aux_vars": 0,
            "J": np.asarray(J2, dtype=np.int64),
            "meta": m2,
            "lookup": l2,
            "df": q3b,
        },
        {
            "task_name": "apmcm26_v2_q3_submod_33",
            "label": "v2 Q3 Blind (12) Submodular HOBO (Model 3A)",
            "scope": "Q3_blind_12",
            "kind": "q3",
            "aux_vars": int(m3["nY"]),
            "J": np.asarray(J3, dtype=np.int64),
            "meta": m3,
            "lookup": l3,
            "df": q3b,
        },
    ]
    return tasks


def run_classical_gate(t):
    """Run local SA (50 reads) and Tabu (30 reads) on exact same int64 Ising matrix."""
    J = t["J"]
    sa_raw = []
    t0 = time.perf_counter()
    for r in range(50):
        opt = kw.SimulatedAnnealingOptimizer(
            initial_temperature=80.0, alpha=0.985, cutoff_temperature=0.01,
            iterations_per_t=25, size_limit=10, rand_seed=7000 + r
        )
        out = opt.solve(J)
        if out is not None and len(out) > 0:
            sa_raw.append(sort_ising_solutions(out, J)[0])
    dt_sa = time.perf_counter() - t0

    tb_raw = []
    t1 = time.perf_counter()
    for r in range(30):
        np.random.seed(8000 + r)
        opt_t = kw.TabuSearchOptimizer(max_iter=250, size_limit=10)
        out_t = opt_t.solve(J)
        if out_t is not None and len(out_t) > 0:
            tb_raw.append(sort_ising_solutions(out_t, J)[0])
    dt_tb = time.perf_counter() - t1

    sa_dec = decode_raw_spins(np.array(sa_raw), J, t["kind"], t["meta"], t["df"], t["lookup"])
    tb_dec = decode_raw_spins(np.array(tb_raw), J, t["kind"], t["meta"], t["df"], t["lookup"])
    sa_dec["runtime_sec"] = round(dt_sa, 3)
    tb_dec["runtime_sec"] = round(dt_tb, 3)
    # Trim verbose per-sample list for classical baselines
    sa_dec.pop("samples", None)
    tb_dec.pop("samples", None)
    return sa_dec, tb_dec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Pre-flight hardware gate only (zero cloud calls)")
    parser.add_argument("--run-spqc", action="store_true", help="Submit the 3 tasks to real SPQC-550 quantum hardware")
    parser.add_argument("--verify-only", action="store_true", help="Re-verify saved SPQC-550 outputs from disk")
    args = parser.parse_args()

    tasks = build_all_tasks()
    out_json = os.path.join(RES_DIR, "v2_spqc550_results.json")
    payload = {}
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    payload.setdefault("tasks", {})

    print("=== [Phase A] Pre-Flight SPQC-550 Hardware Gate Check ===", flush=True)
    for t in tasks:
        hw_info = verify_hardware_matrix(t["J"], t["task_name"])
        sa_dec, tb_dec = run_classical_gate(t)
        print(
            f"  [PASS] {t['task_name']:26s} spins={hw_info['spins']:2d} aux={t['aux_vars']:2d} "
            f"bw={hw_info['bit_width']} rng=[{hw_info['min_val']},{hw_info['max_val']}] "
            f"SA={sa_dec['feasible_rate_pct']:5.1f}%({sa_dec['best_feasible']['formula']}#{sa_dec['best_feasible']['pool_rank']}) "
            f"Tabu={tb_dec['feasible_rate_pct']:5.1f}%({tb_dec['best_feasible']['formula']}#{tb_dec['best_feasible']['pool_rank']})",
            flush=True,
        )
        entry = payload["tasks"].setdefault(t["task_name"], {})
        entry.update({
            "task_name": t["task_name"],
            "label": t["label"],
            "scope": t["scope"],
            "aux_vars": t["aux_vars"],
            "hardware_spec": hw_info,
            "classical_SA": sa_dec,
            "classical_Tabu": tb_dec,
        })

    if args.dry_run and not args.run_spqc:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("\n[Dry-Run Complete] All 3 matrices passed strict SPQC-550 hardware assertions (0 cloud calls).", flush=True)
        return

    if args.verify_only:
        print("\n=== [Verify-Only] Independently re-verifying saved SPQC-550 samples ===", flush=True)
        for t in tasks:
            tn = t["task_name"]
            raw_path = os.path.join(CKPT_DIR, f"{tn}_raw_spins.npy")
            if not os.path.exists(raw_path):
                print(f"  [MISSING] {raw_path}")
                continue
            raw_spins = np.load(raw_path)
            dec = decode_raw_spins(raw_spins, t["J"], t["kind"], t["meta"], t["df"], t["lookup"])
            print(f"  [VERIFIED] {tn}: feas={dec['feasible_count']}/{dec['n_samples']} ({dec['feasible_rate_pct']}%), "
                  f"best={dec['best_feasible']}, topK_hits={dec['unique_topK_count']}/{dec['top_k_target']} {dec['unique_topK_hits']}")
        return

    if args.run_spqc:
        uid = os.environ.get("KAIWU_USER_ID", "").strip()
        code = os.environ.get("KAIWU_SDK_CODE", "").strip()
        if not uid or not code:
            raise RuntimeError("Set KAIWU_USER_ID and KAIWU_SDK_CODE environment variables before --run-spqc.")
        kw.license.init(uid, code)

        from kaiwu.cim import CIMOptimizer, TaskMode
        kw.common.CheckpointManager.save_dir = CKPT_DIR

        print("\n=== [Phase B] Executing 3 Tasks on Real Bose Quantum SPQC-550 ===", flush=True)
        for idx, t in enumerate(tasks, start=1):
            tn = t["task_name"]
            J = t["J"]
            raw_path = os.path.join(CKPT_DIR, f"{tn}_raw_spins.npy")
            prev = payload["tasks"].get(tn, {})
            if os.path.exists(raw_path) and prev.get("spqc550") and prev.get("hardware_spec", {}).get("sha256") == matrix_sha256(J):
                print(f"  ({idx}/3) Reusing cached SPQC-550 result for {tn}", flush=True)
                continue

            print(f"  ({idx}/3) Submitting/Fetching '{tn}' (spins={J.shape[0]}, bw={safe_ising_bit_width(J)}) on SPQC-550...", flush=True)
            t0 = time.perf_counter()
            raw_sols = None
            for attempt in range(1, 6):
                try:
                    opt = CIMOptimizer(
                        task_name=tn,
                        wait=True,
                        interval=10,
                        task_mode=TaskMode.OPTIMIZATION,
                        sample_number=10,
                    )
                    raw_sols = opt.solve(J, negtail_flip=True, sort_solutions=False)
                    break
                except Exception as exc:
                    if attempt == 5:
                        raise
                    print(f"    [Retry {attempt}/5] Transient network error for '{tn}': {exc.__class__.__name__}; waiting 10s...", flush=True)
                    time.sleep(10)
            dt = time.perf_counter() - t0
            if raw_sols is None or len(raw_sols) == 0:
                raise RuntimeError(f"SPQC-550 returned empty solution for {tn}")

            raw_arr = np.asarray(raw_sols, dtype=np.int8)
            np.save(raw_path, raw_arr)
            dec = decode_raw_spins(raw_arr, J, t["kind"], t["meta"], t["df"], t["lookup"])
            dec["hardware_runtime_sec"] = round(dt, 2)
            payload["tasks"][tn]["spqc550"] = dec

            # Save incrementally after every single real hardware call!
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            bf = dec["best_feasible"]
            bf_str = f"{bf['formula']} (#{bf['pool_rank']}, V={bf['V_hat']})" if bf else "INFEASIBLE"
            print(
                f"  -> [DONE {dt:.1f}s] {tn}: SPQC-550 feasible={dec['feasible_count']}/{dec['n_samples']} "
                f"({dec['feasible_rate_pct']}%), best={bf_str}, "
                f"unique Top-{dec['top_k_target']} hits={dec['unique_topK_count']} {dec['unique_topK_hits']}",
                flush=True,
            )

        print(f"\nAll 3 SPQC-550 hardware runs finished and saved to {out_json}.", flush=True)


if __name__ == "__main__":
    main()
