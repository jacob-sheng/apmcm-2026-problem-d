#!/usr/bin/env python3
"""
SPQC-550 Real Optical Quantum Computer (CIM) Execution & Independent Verification
for Compact Lossless Active-Subspace Exact-8-Bit Ising Formulations.

Supported task sets:
  - v3 (M4B repaired Q2/Q3 tasks; archived v2 tasks preserved under historical_v2_archived_tasks):
      1. v3_q2_blind_12          : Q2 Blind Pool (11, Exact-8bit)        - 12 spins, 0 aux, 8-bit
      2. v3_q3_comp_18           : Q3 Blind Pool (12, Model 3A-Comp)     - 18 spins, 0 aux, 8-bit
      3. v3_q3_hobo_29           : Q3 Blind Pool (12, Model 3A-HOBO)     - 29 spins, 11 aux, 8-bit
      4. v3_q3_anova_25          : Q3 Blind Pool (12, Model 3B-ANOVA)    - 25 spins, 0 aux, 8-bit
  - v2 (historical checkpoint integrity audit ONLY; does not rebuild pre-fix matrices):
      1. apmcm26_v2_q2_blind_12  : Historical Q2 Blind archive           - 12 spins
      2. apmcm26_v2_q3_anova_24  : Historical Q3 ANOVA archive           - 24 spins
      3. apmcm26_v2_q3_submod_33 : Historical Q3 Submod archive          - 33 spins

SDK inspection note (kaiwu==1.4.4):
  `kaiwu.cim._client` does not exist in `kaiwu==1.4.4`; the compiled module is
  `kaiwu.cim._cim_optimizer` (`CIMOptimizer`, `TaskMode`; endpoints
  `CREATE_TASK_URL`, `GET_TASK_RESULT_URL`, `SIGN_OSS_URL`) alongside
  `kaiwu.license` (`ensure_license`, `init`, `license_settings`, `track_data`,
  `verify_license`). Neither `kaiwu.cim` nor `kaiwu.license` exposes a public
  account credit/balance query API, so remaining SPQC-550 credits must be
  checked via the QBoson cloud console and tracked carefully to preserve >= 5
  unused credits.
  The SDK signature `CIMOptimizer(..., interval=...)` defaults to `interval=6`
  seconds (minimum SDK-enforced effective polling interval is `5` seconds).

Credentials are read strictly from environment variables KAIWU_USER_ID and
KAIWU_SDK_CODE (never hardcoded or logged). Supports --dry-run (writes ONLY to
results/v2_spqc550_dryrun.json), --verify-only (strict bit-exact SHA-256 +
algebraic QUBO-to-Ising recovery + full record-level comparison), and --run-spqc.
"""

import argparse
import copy
import csv
import getpass
import hashlib
import json
import os
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
import kaiwu as kw

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
sys.path.insert(0, CUR_DIR)
from common_physics import safe_ising_bit_width, sort_ising_solutions
import v2_qubo_q2_q3 as q23

RES_DIR = os.path.join(CUR_DIR, "results")
CKPT_DIR = os.path.join(RES_DIR, "spqc_checkpoints")
DEFAULT_RESULTS_JSON = os.path.join(RES_DIR, "v2_spqc550_results.json")
DEFAULT_DRYRUN_JSON = os.path.join(RES_DIR, "v2_spqc550_dryrun.json")
DEFAULT_TRACK_A_SUMMARY = os.path.join(RES_DIR, "track_a_spqc_archive_summary.json")
_BUNDLED_TRACK_A_CSV = os.path.join(CUR_DIR, "reference_baselines", "table4_spqc550_hardware_vs_classical.csv")
_REPO_TRACK_A_CSV = os.path.join(
    REPO_ROOT,
    "01_2026赛题与数据集",
    "Problem_D_量子计算钙钛矿稀有目标搜索",
    "project",
    "outputs",
    "table4_spqc550_hardware_vs_classical.csv",
)
DEFAULT_TRACK_A_CSV = _BUNDLED_TRACK_A_CSV if os.path.isfile(_BUNDLED_TRACK_A_CSV) else _REPO_TRACK_A_CSV
os.makedirs(CKPT_DIR, exist_ok=True)


SENSITIVE_PATH_RE = re.compile(
    r"([A-Za-z]:[\\/]|/"
    + "Users/"
    + r"|/"
    + "home/"
    + r"|/"
    + "mnt/[a-zA-Z]/|\\\\)",
    re.I,
)


def _safe_relative_path(
    path: str,
    base_dir: str = CUR_DIR,
    fallback_base: str = REPO_ROOT,
) -> str:
    """Return a POSIX relative path within base_dir/fallback_base, or basename if external."""
    if not path:
        return ""
    raw_str = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", raw_str) and os.name != "nt":
        return raw_str.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    abs_p = os.path.abspath(raw_str)
    for root in (base_dir, fallback_base):
        if not root:
            continue
        abs_root = os.path.abspath(str(root))
        try:
            if os.path.commonpath([abs_p, abs_root]) == abs_root:
                rel = os.path.relpath(abs_p, abs_root)
                return "." if rel == "." else rel.replace("\\", "/")
        except ValueError:
            continue
    return os.path.basename(abs_p.replace("\\", "/").rstrip("/"))


def _contains_sensitive_path(text: str) -> bool:
    """Detect host/user-home paths or runtime username/home directory strings."""
    if not isinstance(text, str) or not text:
        return False
    if SENSITIVE_PATH_RE.search(text):
        return True
    norm_text = text.replace("\\", "/").lower()
    home_dir = os.path.expanduser("~")
    if home_dir and len(home_dir) > 3:
        home_norm = home_dir.replace("\\", "/").lower()
        if home_dir.lower() in text.lower() or home_norm in norm_text:
            return True
    try:
        user = getpass.getuser().strip()
    except Exception:
        user = ""
    if user and len(user) >= 3:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", text, re.I):
            return True
    return False


def _assert_no_sensitive_paths(obj, path: str = "$") -> None:
    """Recursively fail-closed guard preventing local paths/PII from being serialized to JSON."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and _contains_sensitive_path(k):
                raise ValueError(f"Refusing to serialize sensitive/absolute path key at {path}.{k}")
            _assert_no_sensitive_paths(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            _assert_no_sensitive_paths(v, f"{path}[{idx}]")
    elif isinstance(obj, str):
        if _contains_sensitive_path(obj):
            raise ValueError(f"Refusing to serialize sensitive/absolute path at {path}: {obj!r}")


def matrix_sha256(mat: np.ndarray) -> str:
    arr = np.ascontiguousarray(mat, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def spins_int8_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.int8).tobytes()).hexdigest()


def verify_section92_ising_recovery(mat: np.ndarray, extra_spins: np.ndarray = None) -> dict:
    """
    Independently verify exact algebraic QUBO <-> Ising equivalence:
      x_i = (1 + s_i * s_ref) / 2,  E(s) = -s^T H s,
      H_ij = -Q_ij / 4 (i != j),  H_{i,ref} = -1/4 * sum_j Q_ij,  H_ii = 0,
      C = 1/4 * (sum_{ij} Q_ij + tr(Q)),   x^T Q x = -s^T H s + C.
    Reconstructs integer symmetric Q from int64 Ising J and asserts exact recovery
    H == J (bit-for-bit SHA-256 match) and exact energy identity on test spins.
    """
    J = np.asarray(mat, dtype=np.int64)
    n = J.shape[0] - 1
    assert n >= 1, "Ising matrix must include >= 1 primary variable plus reference spin"

    Q = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = -4 * J[i, j]
    for i in range(n):
        Q[i, i] = -4 * J[i, n] - int(np.sum(Q[i, :]))

    H = np.zeros((n + 1, n + 1), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i != j:
                num = -int(Q[i, j])
                assert num % 4 == 0, f"Non-integer Section 9.2 H[{i},{j}] recovery"
                H[i, j] = num // 4
        row_sum = -int(np.sum(Q[i, :]))
        assert row_sum % 4 == 0, f"Non-integer Section 9.2 H[{i},ref] recovery"
        h_ref = row_sum // 4
        H[i, n] = h_ref
        H[n, i] = h_ref

    c_num = int(np.sum(Q)) + int(np.trace(Q))
    assert c_num % 4 == 0, "Non-integer Section 9.2 Ising offset C"
    C = c_num // 4

    exact_identity = bool(np.array_equal(H, J))
    rec_sha = matrix_sha256(H)
    orig_sha = matrix_sha256(J)
    assert exact_identity and rec_sha == orig_sha, (
        f"Section 9.2 Ising recovery mismatch: rec={rec_sha} vs orig={orig_sha}"
    )

    rng = np.random.default_rng(2026)
    test_spins = rng.choice(np.array([-1, 1], dtype=np.int64), size=(64, n + 1))
    test_spins[0, -1] = 1
    test_spins[1, -1] = -1
    if extra_spins is not None:
        arr_extra = np.asarray(extra_spins, dtype=np.int64)
        if arr_extra.ndim == 1:
            arr_extra = arr_extra[None, :]
        test_spins = np.vstack([test_spins, arr_extra])

    max_diff = 0
    for s in test_spins:
        x = ((s[:-1] * s[-1]) + 1) // 2
        lhs = int(x @ Q @ x)
        rhs = int(-s @ H @ s + C)
        diff = abs(lhs - rhs)
        if diff > max_diff:
            max_diff = diff
    assert max_diff == 0, f"Section 9.2 QUBO-Ising energy identity failed (max_diff={max_diff})"

    return {
        "ising_offset_C": int(C),
        "recovered_sha256": rec_sha,
        "exact_matrix_identity": exact_identity,
        "qubo_ising_identity_max_diff": int(max_diff),
    }


verify_algebraic_ising_recovery = verify_section92_ising_recovery


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
    rec92 = verify_section92_ising_recovery(J)
    return {
        "spins": int(J.shape[0]),
        "min_val": int(J.min()),
        "max_val": int(J.max()),
        "bit_width": int(bw),
        "dynamic_range": round(dr, 3),
        "sha256": matrix_sha256(J),
        "ising_offset_C": rec92["ising_offset_C"],
    }


def _resolve_active_ax_pairs(meta: dict, df_pool: pd.DataFrame):
    """Resolve auxiliary variable (A, X) pairs in exact builder order for y_ik = a_i * c_k checks."""
    nY = int(meta.get("nY", 0))
    if nY <= 0:
        return []
    if "active_ax" in meta and meta["active_ax"] is not None:
        pairs = [tuple(p) for p in meta["active_ax"]]
        if len(pairs) != nY:
            raise ValueError(f"meta['active_ax'] length {len(pairs)} != nY={nY}")
        return pairs
    if "ax2y" in meta and meta["ax2y"] is not None:
        ordered = [k for k, _ in sorted(meta["ax2y"].items(), key=lambda kv: kv[1])]
        if len(ordered) != nY:
            raise ValueError(f"meta['ax2y'] length {len(ordered)} != nY={nY}")
        return ordered
    if "c_cap" in meta:
        active = df_pool[df_pool["V_hat"] < float(meta["c_cap"])]
        pairs = sorted(set(zip(active["A"], active["X"])))
        if len(pairs) != nY:
            raise ValueError(f"Reconstructed active_ax length {len(pairs)} != nY={nY}")
        return pairs
    raise ValueError(f"Cannot resolve auxiliary (A, X) mapping for nY={nY} without active_ax or c_cap.")


def decode_raw_spins(
    raw_sols: np.ndarray,
    J: np.ndarray,
    task_kind: str,
    meta: dict,
    df_pool: pd.DataFrame,
    lookup: dict,
) -> dict:
    """
    Independently decode every raw Ising spin vector s in {-1, +1}^N returned by
    SPQC-550 (or classical solver) via x = ((s[:-1] * s[-1]) + 1) // 2, check:
      1. One-hot block constraints (kA == 1, kB == 1[, kX == 1])
      2. Pool eligibility & A != B constraint
      3. Explicit auxiliary consistency y_ik == a_i * c_k (when nY > 0)
    Separates unique feasible candidates from total feasible shot hits.
    """
    arr = np.asarray(raw_sols, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    n_samples = int(arr.shape[0])
    energies = -np.einsum("ij,jk,ik->i", arr, J.astype(float), arr)
    order = np.argsort(energies)
    arr = arr[order]
    energies = energies[order]

    df_sorted = df_pool.sort_values("V_hat").reset_index(drop=True)
    rank_map = {}
    formula_map = {}
    for idx_r, r in df_sorted.iterrows():
        key = (r["A"], r["B"]) if task_kind == "q2" else (r["A"], r["B"], r["X"])
        rank_map[key] = int(idx_r + 1)
        formula_map[key] = str(r["formula"])

    top_k = 10 if task_kind == "q2" else 15
    top_set = set(list(rank_map.keys())[:top_k])

    nY = int(meta.get("nY", 0))
    active_ax = _resolve_active_ax_pairs(meta, df_pool) if nY > 0 else []
    a2i = {e: i for i, e in enumerate(meta["elems_A"])}
    x2i = {e: i for i, e in enumerate(meta.get("elems_X", []))}
    formulation = meta.get("formulation", "")
    exact_min_v = float(df_pool["V_hat"].min())

    decoded_rows = []
    feas_shot_cnt = 0
    prim_feas_shot_cnt = 0
    y_aux_ok_cnt = 0
    unique_feas_map = {}
    hits_top = set()
    best_feas = None
    best_feas_raw_v = None

    for i in range(n_samples):
        s = arr[i]
        s_aux = s[-1]
        x = (((s[:-1] * s_aux) + 1) // 2).astype(int)
        e_val = float(energies[i])

        kAX = None
        if formulation == "comp_ax_b":
            nAX, nB = meta["nAX"], meta["nB"]
            ax_idx = np.where(x[:nAX] == 1)[0]
            b_idx = np.where(x[nAX:nAX + nB] == 1)[0]
            kAX, kB = len(ax_idx), len(b_idx)
            kA, kX = kAX, kAX
            one_hot_ok = (kAX == 1 and kB == 1)
            pool_ok = False
            cand_formula = "INFEASIBLE"
            v_hat = None
            pool_rank = None
            key = None
            if one_hot_ok:
                ea, ex = meta["ax_pairs"][int(ax_idx[0])]
                eb = meta["B_elems"][int(b_idx[0])]
                if ea != eb and (ea, eb, ex) in lookup:
                    pool_ok = True
                    key = (ea, eb, ex)
                    v_hat = float(lookup[key])
                    cand_formula = formula_map[key]
                    pool_rank = rank_map[key]
        elif task_kind == "q2" or formulation == "q2_bipartite":
            nA, nB = meta["nA"], meta["nB"]
            ai = np.where(x[:nA] == 1)[0]
            bi = np.where(x[nA:nA + nB] == 1)[0]
            kA, kB, kX = len(ai), len(bi), 1
            one_hot_ok = (kA == 1 and kB == 1)
            pool_ok = False
            cand_formula = "INFEASIBLE"
            v_hat = None
            pool_rank = None
            key = None
            if one_hot_ok:
                ea, eb = meta["elems_A"][ai[0]], meta["elems_B"][bi[0]]
                if ea != eb and (ea, eb) in lookup:
                    pool_ok = True
                    key = (ea, eb)
                    v_hat = float(lookup[key])
                    cand_formula = formula_map[key]
                    pool_rank = rank_map[key]
        else:
            nA, nB, nX = meta["nA"], meta["nB"], meta["nX"]
            off_B, off_X = meta["off_B"], meta["off_X"]
            ai = np.where(x[:nA] == 1)[0]
            bi = np.where(x[off_B:off_B + nB] == 1)[0]
            xi = np.where(x[off_X:off_X + nX] == 1)[0]
            kA, kB, kX = len(ai), len(bi), len(xi)
            one_hot_ok = (kA == 1 and kB == 1 and kX == 1)
            pool_ok = False
            cand_formula = "INFEASIBLE"
            v_hat = None
            pool_rank = None
            key = None
            if one_hot_ok:
                ea, eb, ex = meta["elems_A"][ai[0]], meta["elems_B"][bi[0]], meta["elems_X"][xi[0]]
                if ea != eb and (ea, eb, ex) in lookup:
                    pool_ok = True
                    key = (ea, eb, ex)
                    v_hat = float(lookup[key])
                    cand_formula = formula_map[key]
                    pool_rank = rank_map[key]

        y_mismatches = 0
        if nY > 0:
            off_Y = meta["off_Y"]
            for y_idx, (ea, ex) in enumerate(active_ax):
                expected_y = int(x[a2i[ea]] * x[off_X + x2i[ex]])
                if int(x[off_Y + y_idx]) != expected_y:
                    y_mismatches += 1
        y_aux_consistent = (y_mismatches == 0)
        if y_aux_consistent:
            y_aux_ok_cnt += 1

        primary_feasible = bool(one_hot_ok and pool_ok)
        if primary_feasible:
            prim_feas_shot_cnt += 1

        is_feas = bool(primary_feasible and y_aux_consistent)
        sample_regret = round(v_hat - exact_min_v, 6) if is_feas and v_hat is not None else None
        if is_feas:
            feas_shot_cnt += 1
            unique_feas_map[cand_formula] = (pool_rank, v_hat)
            if key in top_set:
                hits_top.add(cand_formula)
            if best_feas is None or v_hat < best_feas_raw_v:
                best_feas_raw_v = v_hat
                best_feas = {
                    "formula": cand_formula,
                    "V_hat": round(v_hat, 6),
                    "regret": round(v_hat - exact_min_v, 6),
                    "pool_rank": pool_rank,
                    "ising_energy": round(e_val, 2),
                }
        else:
            if primary_feasible and not y_aux_consistent:
                cand_formula = f"AUX_INCONSISTENT({cand_formula})"

        row_dict = {
            "sample_idx": i,
            "ising_energy": round(e_val, 2),
            "kA": int(kA), "kB": int(kB), "kX": int(kX),
            **({"kAX": int(kAX)} if kAX is not None else {}),
            "one_hot_ok": bool(one_hot_ok),
            "pool_ok": bool(pool_ok),
            "y_aux_consistent": bool(y_aux_consistent),
            "y_aux_mismatch_count": int(y_mismatches),
            "primary_feasible": primary_feasible,
            "feasible": bool(is_feas),
            "formula": cand_formula,
            "V_hat": round(v_hat, 6) if v_hat is not None else None,
            "regret": sample_regret,
            "pool_rank": pool_rank,
        }
        decoded_rows.append(row_dict)

    unique_feas_list = [
        f for f, _ in sorted(unique_feas_map.items(), key=lambda kv: (kv[1][0], kv[0]))
    ]
    best_regret = round(best_feas_raw_v - exact_min_v, 6) if best_feas is not None else None
    return {
        "n_samples": n_samples,
        "feasible_shot_count": feas_shot_cnt,
        "feasible_count": feas_shot_cnt,
        "feasible_rate_pct": round(100.0 * feas_shot_cnt / max(1, n_samples), 2),
        "primary_feasible_shot_count": prim_feas_shot_cnt,
        "primary_feasible_rate_pct": round(100.0 * prim_feas_shot_cnt / max(1, n_samples), 2),
        "y_aux_consistent_count": y_aux_ok_cnt,
        "unique_feasible_candidates": unique_feas_list,
        "unique_feasible_count": len(unique_feas_list),
        "duplicate_feasible_hits": max(0, feas_shot_cnt - len(unique_feas_list)),
        "best_ising_energy": round(float(energies[0]), 2),
        "best_feasible": best_feas,
        "exact_min_V_hat": round(exact_min_v, 6),
        "gap_to_opt": best_regret,
        "continuous_regret": best_regret,
        "unique_topK_hits": sorted(list(hits_top)),
        "unique_topK_count": len(hits_top),
        "top_k_target": top_k,
        "samples": decoded_rows,
    }


evaluate_hardware_samples = decode_raw_spins


def build_tasks(task_set: str = "v3", task_filter=None):
    """
    Build active repaired 'v3' SPQC-550 task specifications.
    Note: Legacy 'v2' archives use historical pre-fix formulations and are verified via
    checkpoint-integrity audit ONLY rather than rebuilt from current v3 builders (R11).
    """
    if task_set == "v2":
        return []
    q2b = pd.read_csv(os.path.join(RES_DIR, "v2_q2_blind_11.csv"))
    q3b = pd.read_csv(os.path.join(RES_DIR, "v2_q3_blind_12.csv"))

    tasks = []
    if task_set in ("v3", "all"):
        if not hasattr(q23, "build_q3_comp_qubo"):
            raise AttributeError(
                "v2_qubo_q2_q3.build_q3_comp_qubo is not available yet (implemented in Milestone M3)."
            )
        Jq2, mq2, lq2 = q23.build_exact8_q2_qubo(q2b, c_cap=2.5, penalty_mult=2.25)
        Jc, mc, lc = q23.build_q3_comp_qubo(q3b, c_cap=1.50, penalty_mult=1.60)
        Jh, mh, lh = q23.build_q3_submodular_hobo_qubo(q3b, c_cap=1.50)
        Ja, ma, la = q23.build_q3_compact_anova_qubo(q3b, top_k_subspace=25, c_cap=1.50)
        tasks.extend([
            {
                "task_name": "v3_q2_blind_12",
                "label": "v3 Q2 Blind (11) Active-Subspace (Exact-8bit)",
                "scope": "Q2_blind_11",
                "kind": "q2",
                "aux_vars": 0,
                "J": np.asarray(Jq2, dtype=np.int64),
                "meta": mq2,
                "lookup": lq2,
                "df": q2b,
            },
            {
                "task_name": "v3_q3_comp_18",
                "label": "v3 Q3 Blind (12) Zero-Aux Composite (Model 3A-Comp)",
                "scope": "Q3_blind_12",
                "kind": "q3",
                "aux_vars": int(mc.get("nY", 0)),
                "J": np.asarray(Jc, dtype=np.int64),
                "meta": mc,
                "lookup": lc,
                "df": q3b,
            },
            {
                "task_name": "v3_q3_hobo_29",
                "label": "v3 Q3 Blind (12) Repaired Submodular HOBO (Model 3A-HOBO)",
                "scope": "Q3_blind_12",
                "kind": "q3",
                "aux_vars": int(mh["nY"]),
                "J": np.asarray(Jh, dtype=np.int64),
                "meta": mh,
                "lookup": lh,
                "df": q3b,
            },
            {
                "task_name": "v3_q3_anova_25",
                "label": "v3 Q3 Blind (12) Repaired Compact ANOVA (Model 3B)",
                "scope": "Q3_blind_12",
                "kind": "q3",
                "aux_vars": int(ma.get("nY", 0)),
                "J": np.asarray(Ja, dtype=np.int64),
                "meta": ma,
                "lookup": la,
                "df": q3b,
            },
        ])

    if task_filter:
        allowed = set(task_filter)
        tasks = [t for t in tasks if t["task_name"] in allowed]
    return tasks


def build_all_tasks():
    """Backward-compatible wrapper returning the default v3 task list."""
    return build_tasks(task_set="v3")


def export_track_a_archive_summary(
    csv_path: str = DEFAULT_TRACK_A_CSV,
    out_path: str = DEFAULT_TRACK_A_SUMMARY,
) -> dict:
    """
    Parse Track A's historical SPQC-550 vs Classical CSV directly into a structured
    JSON archive summary with zero hardcoded numerical rows (F22).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Track A CSV not found: {_safe_relative_path(csv_path, base_dir=REPO_ROOT, fallback_base=CUR_DIR)}"
        )

    with open(csv_path, "rb") as f:
        raw_bytes = f.read()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    lf_sha256 = hashlib.sha256(raw_bytes.replace(b"\r\n", b"\n")).hexdigest()

    def _parse_int(val):
        return None if val == "" or val is None else int(float(val))

    def _parse_float(val):
        return None if val == "" or val is None else float(val)

    def _parse_bool(val):
        if val == "" or val is None:
            return None
        low = str(val).strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
        raise ValueError(f"Unexpected boolean value in Track A CSV: {val!r}")

    parsed_rows = []
    tasks_grouped = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "task_name": str(row["task_name"]),
                "problem_scope": str(row["problem_scope"]),
                "solver": str(row["solver"]),
                "n_spins": _parse_int(row["n_spins"]),
                "n_samples": _parse_int(row["n_samples"]),
                "ref_consensus_rate": _parse_float(row["ref_consensus_rate"]),
                "feasible_count": _parse_int(row["feasible_count"]),
                "feasible_rate": _parse_float(row["feasible_rate"]),
                "best_ising_energy": _parse_float(row["best_ising_energy"]),
                "best_V_pred": _parse_float(row["best_V_pred"]),
                "best_formula": str(row["best_formula"]),
                "best_in_blind": _parse_int(row["best_in_blind"]),
                "feasible_in_blind_rate": _parse_float(row["feasible_in_blind_rate"]),
                "subcap_feasible_count": _parse_int(row["subcap_feasible_count"]),
                "subcap_in_blind_rate": _parse_float(row["subcap_in_blind_rate"]),
                "true_rank_in_pool": _parse_int(row["true_rank_in_pool"]),
                "hard_penalty_verified": _parse_bool(row["hard_penalty_verified"]),
                "checkpoint_file": str(row["checkpoint_file"]),
            }
            parsed_rows.append(rec)
            tn = rec["task_name"]
            if tn not in tasks_grouped:
                tasks_grouped[tn] = {
                    "task_name": tn,
                    "problem_scope": rec["problem_scope"],
                    "n_spins": rec["n_spins"],
                    "solvers": {},
                }
            tasks_grouped[tn]["solvers"][rec["solver"]] = rec

    spqc_rows = [r for r in parsed_rows if r["solver"] == "SPQC-550"]
    total_spqc_samples = sum(r["n_samples"] or 0 for r in spqc_rows)
    total_spqc_feas = sum(r["feasible_count"] or 0 for r in spqc_rows)
    summary = {
        "source_csv_relative_path": _safe_relative_path(csv_path, base_dir=REPO_ROOT, fallback_base=CUR_DIR),
        "source_csv_sha256": raw_sha256,
        "source_csv_lf_normalized_sha256": lf_sha256,
        "row_count": len(parsed_rows),
        "task_count": len(tasks_grouped),
        "solvers": sorted({r["solver"] for r in parsed_rows}),
        "spqc550_summary": {
            "n_tasks": len(spqc_rows),
            "total_samples": total_spqc_samples,
            "total_feasible_shots": total_spqc_feas,
            "overall_feasible_rate": round(total_spqc_feas / max(1, total_spqc_samples), 6),
            "tasks_with_at_least_one_feasible": sum(1 for r in spqc_rows if (r["feasible_count"] or 0) > 0),
            "min_spins": min(r["n_spins"] for r in spqc_rows) if spqc_rows else None,
            "max_spins": max(r["n_spins"] for r in spqc_rows) if spqc_rows else None,
        },
        "tasks": tasks_grouped,
        "rows": parsed_rows,
    }
    _assert_no_sensitive_paths(summary)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _checkpoint_paths(task_name: str, sha256_hex: str, ckpt_dir: str = CKPT_DIR):
    sha8 = sha256_hex[:8]
    hash_path = os.path.join(ckpt_dir, f"{task_name}_{sha8}_raw_spins.npy")
    legacy_path = os.path.join(ckpt_dir, f"{task_name}_raw_spins.npy")
    return hash_path, legacy_path


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
    sa_dec.pop("samples", None)
    tb_dec.pop("samples", None)
    return sa_dec, tb_dec


def _sanitize_meta_str(key_hint: str, val: str) -> str:
    lk = key_hint.lower()
    if any(p in lk for p in ("dir", "path", "file", "root", "home")) or _contains_sensitive_path(val) or os.path.isabs(val):
        rel = _safe_relative_path(val, base_dir=CUR_DIR, fallback_base=REPO_ROOT)
        if _contains_sensitive_path(rel):
            return "[REDACTED_PATH]"
        return rel
    return val


def _extract_safe_optimizer_meta(opt) -> dict:
    """Extract non-sensitive metadata from CIMOptimizer without leaking credentials or local paths."""
    blocked = ("user", "sdk", "code", "token", "auth", "key", "secret", "license", "password")
    meta = {}
    for k, v in getattr(opt, "__dict__", {}).items():
        lk = str(k).lower()
        if any(b in lk for b in blocked):
            continue
        clean_k = str(k).lstrip("_")
        if clean_k == "save_dir":
            meta[clean_k] = (
                _safe_relative_path(str(v), base_dir=CUR_DIR, fallback_base=REPO_ROOT)
                if v
                else "results/spqc_checkpoints"
            )
            continue
        if isinstance(v, os.PathLike):
            v = os.fspath(v)
        if isinstance(v, str):
            meta[clean_k] = _sanitize_meta_str(clean_k, v)
        elif isinstance(v, (int, float, bool)) or v is None:
            meta[clean_k] = v
        elif isinstance(v, dict):
            safe_sub = {}
            for sk, sv in v.items():
                if any(b in str(sk).lower() for b in blocked):
                    continue
                if isinstance(sv, os.PathLike):
                    sv = os.fspath(sv)
                if isinstance(sv, str):
                    safe_sub[str(sk)] = _sanitize_meta_str(str(sk), sv)
                elif isinstance(sv, (int, float, bool, type(None))):
                    safe_sub[str(sk)] = sv
            if safe_sub:
                meta[clean_k] = safe_sub
    return meta


def _update_credit_accounting(payload: dict):
    """Maintain explicit SPQC-550 credit accounting ensuring >= 5 credits remain unused."""
    hist = payload.get("historical_v2_archived_tasks", {})
    active = payload.get("tasks", {})
    v2_used = len([k for k in hist if k.startswith("apmcm26_v2_")]) or 3
    v3_used = len([k for k, v in active.items() if k.startswith("v3_") and v.get("spqc550")])
    credits_initial = 31
    track_a_used = 6
    total_used = track_a_used + v2_used + v3_used
    est_remaining = credits_initial - total_used
    payload["credit_accounting"] = {
        "credits_initial": credits_initial,
        "track_a_used": track_a_used,
        "v2_archived_used": v2_used,
        "v3_used": v3_used,
        "total_used": total_used,
        "estimated_remaining": est_remaining,
        "min_required_remaining": 5,
        "credit_safety_margin_satisfied": bool(est_remaining >= 5),
        "sample_number_per_task": 10,
        "note": "kaiwu==1.4.4 exposes no balance query API; tracked conservatively from initial 31 credits.",
    }


def verify_historical_v2_checkpoints_integrity(prior_archived: dict, ckpt_dir: str) -> list:
    """
    Historical checkpoint integrity audit ONLY (R11):
    Verifies archived v2 matrix SHA length, raw .npy checkpoint existence, spin alphabet {-1,+1},
    file/array SHA-256 digests (when present), array shape (n_samples, spins), and sample list count
    without pretending to rebuild pre-fix v2 Hamiltonians from current v3 builders.
    """
    failures = []
    for h_tn, h_entry in prior_archived.items():
        h_spec = h_entry.get("hardware_spec", {})
        h_sha = h_spec.get("sha256", "")
        h_spins = int(h_spec.get("spins", 0))
        h_spqc = h_entry.get("spqc550", {})
        h_spqc_sha = h_spqc.get("matrix_sha256")
        if not h_sha or len(h_sha) != 64:
            failures.append(f"historical:{h_tn}: invalid hardware_spec.sha256")
            continue
        if h_spqc_sha is not None and h_spqc_sha != h_sha:
            failures.append(f"historical:{h_tn}: spqc550.matrix_sha256 != hardware_spec.sha256")
            continue
        hp, lp = _checkpoint_paths(h_tn, h_sha, ckpt_dir)
        rp = hp if os.path.exists(hp) else (lp if os.path.exists(lp) else None)
        if rp is None:
            failures.append(f"historical:{h_tn}: missing archived raw spin checkpoint")
            continue
        arr_h = np.load(rp)
        exp_n = int(h_spqc.get("n_samples", arr_h.shape[0]))
        if arr_h.ndim != 2 or arr_h.shape != (exp_n, h_spins):
            failures.append(f"historical:{h_tn}: checkpoint shape {arr_h.shape} != ({exp_n}, {h_spins})")
            continue
        if not np.all(np.isin(arr_h, [-1, 1])):
            failures.append(f"historical:{h_tn}: checkpoint contains non-{{-1,+1}} spin values")
            continue
        exp_fsha = h_spqc.get("raw_spins_file_sha256")
        if exp_fsha and file_sha256(rp) != exp_fsha:
            failures.append(f"historical:{h_tn}: raw_spins_file_sha256 mismatch")
            continue
        exp_asha = h_spqc.get("raw_spins_int8_sha256")
        if exp_asha and spins_int8_sha256(arr_h) != exp_asha:
            failures.append(f"historical:{h_tn}: raw_spins_int8_sha256 mismatch")
            continue
        samples_h = h_spqc.get("samples", [])
        if samples_h and len(samples_h) != exp_n:
            failures.append(f"historical:{h_tn}: len(samples)={len(samples_h)} != n_samples={exp_n}")
            continue
        print(
            f"  [HISTORICAL-CKPT-INTEGRITY-ONLY] {h_tn:26s} sha={h_sha[:12]}.. spins={h_spins} "
            f"ckpt={os.path.basename(rp)} shape={arr_h.shape}",
            flush=True,
        )
    return failures


def _compare_decoded_with_archived(tn: str, dec: dict, archived: dict) -> list:
    """
    Strictly compare freshly recomputed decode_raw_spins() output against archived spqc550 JSON (R10).
    Returns a list of mismatch descriptions (empty list means exact agreement).
    """
    errs = []
    exact_keys = (
        "n_samples",
        "feasible_shot_count",
        "feasible_count",
        "primary_feasible_shot_count",
        "y_aux_consistent_count",
        "unique_feasible_candidates",
        "unique_feasible_count",
        "duplicate_feasible_hits",
        "unique_topK_hits",
        "unique_topK_count",
        "top_k_target",
    )
    for k in exact_keys:
        if k not in archived:
            errs.append(f"{tn}: archived spqc550 missing required key '{k}'")
        elif dec[k] != archived[k]:
            errs.append(f"{tn}: '{k}' mismatch (redecoded={dec[k]!r} != archived={archived[k]!r})")

    float_keys = (
        "feasible_rate_pct",
        "primary_feasible_rate_pct",
        "best_ising_energy",
        "exact_min_V_hat",
        "gap_to_opt",
        "continuous_regret",
    )
    for k in float_keys:
        if k not in archived:
            errs.append(f"{tn}: archived spqc550 missing float key '{k}'")
            continue
        dv, av = dec.get(k), archived.get(k)
        if (dv is None) != (av is None):
            errs.append(f"{tn}: '{k}' null mismatch (redecoded={dv!r} != archived={av!r})")
        elif dv is not None and abs(float(dv) - float(av)) > 1e-5:
            errs.append(f"{tn}: '{k}' mismatch (redecoded={dv} != archived={av})")

    db, ab = dec.get("best_feasible"), archived.get("best_feasible")
    if (db is None) != (ab is None):
        errs.append(f"{tn}: best_feasible null mismatch (redecoded={db!r} != archived={ab!r})")
    elif db is not None:
        for bk in ("formula", "pool_rank"):
            if db.get(bk) != ab.get(bk):
                errs.append(f"{tn}: best_feasible.{bk} mismatch ({db.get(bk)!r} != {ab.get(bk)!r})")
        for bk in ("V_hat", "regret", "ising_energy"):
            if abs(float(db[bk]) - float(ab[bk])) > 1e-5:
                errs.append(f"{tn}: best_feasible.{bk} mismatch ({db[bk]} != {ab[bk]})")

    dsamps = dec.get("samples", [])
    asamps = archived.get("samples", [])
    if len(dsamps) != len(asamps):
        errs.append(f"{tn}: samples length mismatch (redecoded={len(dsamps)} != archived={len(asamps)})")
        return errs

    row_exact = (
        "sample_idx",
        "kA",
        "kB",
        "kX",
        "one_hot_ok",
        "pool_ok",
        "y_aux_consistent",
        "y_aux_mismatch_count",
        "primary_feasible",
        "feasible",
        "formula",
        "pool_rank",
    )
    for idx, (dr, ar) in enumerate(zip(dsamps, asamps)):
        for rk in row_exact:
            if dr.get(rk) != ar.get(rk):
                errs.append(f"{tn}: samples[{idx}].{rk} mismatch ({dr.get(rk)!r} != {ar.get(rk)!r})")
        for rk in ("ising_energy", "V_hat", "regret"):
            dv, av = dr.get(rk), ar.get(rk)
            if (dv is None) != (av is None):
                errs.append(f"{tn}: samples[{idx}].{rk} null mismatch ({dv!r} != {av!r})")
            elif dv is not None and abs(float(dv) - float(av)) > 1e-5:
                errs.append(f"{tn}: samples[{idx}].{rk} mismatch ({dv} != {av})")
    return errs


def main():
    parser = argparse.ArgumentParser(description="SPQC-550 Hardware Execution & Bit-Exact Verification Harness")
    parser.add_argument("--dry-run", action="store_true", help="Pre-flight hardware gate only; writes ONLY to results/v2_spqc550_dryrun.json (0 cloud calls)")
    parser.add_argument("--run-spqc", action="store_true", help="Submit uncached tasks to real SPQC-550 quantum hardware")
    parser.add_argument("--verify-only", action="store_true", help="Re-verify saved SPQC-550 outputs from disk with strict SHA-256, Ising recovery & per-record equality checks")
    parser.add_argument("--export-track-a", action="store_true", help="Parse Track A table4 CSV into results/track_a_spqc_archive_summary.json (0 cloud calls)")
    parser.add_argument("--task-set", choices=["v2", "v3", "all"], default="v3", help="Task set to process ('v3' active tasks, or 'v2' historical checkpoint integrity audit)")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional subset of task names to process")
    parser.add_argument("--results-json", default=DEFAULT_RESULTS_JSON, help="Path to archived/live SPQC-550 results JSON")
    parser.add_argument("--dryrun-json", default=DEFAULT_DRYRUN_JSON, help="Path to isolated dry-run diagnostic JSON")
    parser.add_argument("--track-a-json", default=DEFAULT_TRACK_A_SUMMARY, help="Output path for Track A archive summary JSON")
    parser.add_argument("--ckpt-dir", default=CKPT_DIR, help="Checkpoint directory for raw spin .npy arrays")
    parser.add_argument("--poll-interval", type=int, default=5, help="CIMOptimizer polling interval in SECONDS (SDK default=6s, SDK min=5s)")
    args = parser.parse_args()

    if not any([args.dry_run, args.run_spqc, args.verify_only, args.export_track_a]):
        parser.print_help()
        return 0

    if args.export_track_a:
        ta_summary = export_track_a_archive_summary(out_path=args.track_a_json)
        print(
            f"[Track A Archive Exported] {_safe_relative_path(args.track_a_json)} "
            f"(rows={ta_summary['row_count']}, tasks={ta_summary['task_count']}, "
            f"csv_sha8={ta_summary['source_csv_sha256'][:8]})",
            flush=True,
        )
        if not any([args.dry_run, args.run_spqc, args.verify_only]):
            return 0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    payload = {}
    if os.path.exists(args.results_json):
        with open(args.results_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    payload.setdefault("tasks", {})
    prior_tasks = copy.deepcopy(payload["tasks"])
    prior_archived = copy.deepcopy(payload.get("historical_v2_archived_tasks", {}))

    if args.task_set == "v2":
        if args.run_spqc:
            raise RuntimeError(
                "Refusing --run-spqc for task_set='v2': archived v2 matrices belong to pre-fix formulations "
                "and must not be overwritten with current v3 builders (R11)."
            )
        print("\n=== [Historical v2 Checkpoint Integrity Audit] ===", flush=True)
        hist_failures = verify_historical_v2_checkpoints_integrity(prior_archived, args.ckpt_dir)
        if hist_failures:
            for msg in hist_failures:
                print(f"  [FAIL] {msg}", file=sys.stderr, flush=True)
            return 1
        print(f"\n[HISTORICAL-AUDIT PASSED] {len(prior_archived)} archived v2 checkpoint(s) intact.", flush=True)
        return 0

    tasks = build_tasks(task_set=args.task_set, task_filter=args.tasks)
    if not tasks:
        print("[ERROR] No tasks matched the requested task-set / task filter.", file=sys.stderr)
        return 1

    print("=== [Phase A] Pre-Flight SPQC-550 Hardware Gate Check ===", flush=True)
    staged_entries = {}
    for t in tasks:
        tn = t["task_name"]
        hw_info = verify_hardware_matrix(t["J"], tn)
        cert = q23.certify_global_ground_states(
            t["J"], t["meta"], t["df"], t["lookup"], task_kind=t["kind"], verify_zero_illegal=True
        )
        assert cert["ising_offset_C"] == hw_info["ising_offset_C"], (
            f"{tn}: cert offset {cert['ising_offset_C']} != hw offset {hw_info['ising_offset_C']}"
        )
        sa_dec, tb_dec = run_classical_gate(t)
        print(
            f"  [PASS] {tn:26s} spins={hw_info['spins']:2d} aux={t['aux_vars']:2d} "
            f"bw={hw_info['bit_width']} C={hw_info['ising_offset_C']} certE={cert['cert_min_ising_energy']} "
            f"legal_gs={cert['legal_ground_count']} illegal_gs={cert['illegal_ground_count']} "
            f"sha8={hw_info['sha256'][:8]} "
            f"SA={sa_dec['feasible_rate_pct']:5.1f}%({sa_dec['best_feasible']['formula']}#{sa_dec['best_feasible']['pool_rank']}) "
            f"Tabu={tb_dec['feasible_rate_pct']:5.1f}%({tb_dec['best_feasible']['formula']}#{tb_dec['best_feasible']['pool_rank']})",
            flush=True,
        )
        staged_entries[tn] = {
            "task_name": tn,
            "label": t["label"],
            "scope": t["scope"],
            "aux_vars": t["aux_vars"],
            "cert_min_ising_energy": cert["cert_min_ising_energy"],
            "cert_min_qubo_energy": cert["cert_min_qubo_energy"],
            "legal_ground_count": cert["legal_ground_count"],
            "illegal_ground_count": cert["illegal_ground_count"],
            "certification": {
                "cert_min_ising_energy": cert["cert_min_ising_energy"],
                "cert_min_qubo_energy": cert["cert_min_qubo_energy"],
                "ising_offset_C": cert["ising_offset_C"],
                "legal_ground_count": cert["legal_ground_count"],
                "illegal_ground_count": cert["illegal_ground_count"],
                "illegal_aux_tie_count": cert["illegal_aux_tie_count"],
                "ground_state_degeneracy": cert["ground_state_degeneracy"],
                "continuous_regret_top1": cert["continuous_regret_top1"],
                "continuous_regret_worst_gs": cert["continuous_regret_worst_gs"],
                "enumeration_method": cert["enumeration_method"],
                "M_max": cert.get("M_max"),
            },
            "hardware_spec": hw_info,
            "classical_SA": sa_dec,
            "classical_Tabu": tb_dec,
        }

    if args.dry_run and not args.run_spqc:
        dryrun_payload = {
            "mode": "dry_run",
            "task_set": args.task_set,
            "results_json_untouched": _safe_relative_path(args.results_json),
            "tasks": staged_entries,
        }
        _assert_no_sensitive_paths(dryrun_payload)
        os.makedirs(os.path.dirname(os.path.abspath(args.dryrun_json)), exist_ok=True)
        with open(args.dryrun_json, "w", encoding="utf-8") as f:
            json.dump(dryrun_payload, f, indent=2)
        print(
            f"\n[Dry-Run Complete] Diagnostic output written ONLY to {_safe_relative_path(args.dryrun_json)} "
            f"(0 cloud calls; {os.path.basename(args.results_json)} untouched).",
            flush=True,
        )
        return 0

    if args.verify_only:
        print("\n=== [Verify-Only] Strict SHA-256, Algebraic Recovery & Full Per-Record Verification ===", flush=True)
        failures = []
        for t in tasks:
            tn = t["task_name"]
            J = t["J"]
            cur_sha = staged_entries[tn]["hardware_spec"]["sha256"]
            cur_C = staged_entries[tn]["hardware_spec"]["ising_offset_C"]
            prev_entry = prior_tasks.get(tn)
            if not prev_entry:
                failures.append(f"{tn}: missing archived entry in {_safe_relative_path(args.results_json)}")
                continue
            archived_sha = prev_entry.get("hardware_spec", {}).get("sha256")
            if archived_sha != cur_sha:
                failures.append(
                    f"{tn}: matrix SHA-256 mismatch (built={cur_sha} != archived={archived_sha})"
                )
                continue
            spqc_block = prev_entry.get("spqc550", {})
            spqc_sha = spqc_block.get("matrix_sha256")
            if spqc_sha is not None and spqc_sha != cur_sha:
                failures.append(
                    f"{tn}: spqc550.matrix_sha256 mismatch (built={cur_sha} != archived={spqc_sha})"
                )
                continue
            if tn.startswith("v3_") and spqc_sha != cur_sha:
                failures.append(
                    f"{tn}: missing or mismatched spqc550.matrix_sha256 ({spqc_sha})"
                )
                continue
            archived_C = spqc_block.get("ising_offset_C", prev_entry.get("hardware_spec", {}).get("ising_offset_C"))
            if archived_C is not None and int(archived_C) != int(cur_C):
                failures.append(f"{tn}: ising_offset_C mismatch (built={cur_C} != archived={archived_C})")
                continue

            hash_path, legacy_path = _checkpoint_paths(tn, cur_sha, args.ckpt_dir)
            if os.path.exists(hash_path):
                raw_path = hash_path
                ckpt_mode = "hash-keyed"
            elif os.path.exists(legacy_path):
                raw_path = legacy_path
                ckpt_mode = "legacy-sha-verified"
            else:
                failures.append(
                    f"{tn}: missing raw spin checkpoint ({os.path.basename(hash_path)} or {os.path.basename(legacy_path)})"
                )
                continue

            raw_spins = np.load(raw_path)
            if raw_spins.ndim != 2 or raw_spins.shape[1] != J.shape[0]:
                failures.append(f"{tn}: raw_spins shape {raw_spins.shape} incompatible with J.shape={J.shape}")
                continue
            if not np.all(np.isin(raw_spins, [-1, 1])):
                failures.append(f"{tn}: raw_spins contains non-{{-1,+1}} entries")
                continue

            exp_file_sha = spqc_block.get("raw_spins_file_sha256")
            if exp_file_sha is not None and file_sha256(raw_path) != exp_file_sha:
                failures.append(f"{tn}: raw_spins_file_sha256 mismatch on {os.path.basename(raw_path)}")
                continue
            exp_arr_sha = spqc_block.get("raw_spins_int8_sha256")
            if exp_arr_sha is not None and spins_int8_sha256(raw_spins) != exp_arr_sha:
                failures.append(f"{tn}: raw_spins_int8_sha256 mismatch on {os.path.basename(raw_path)}")
                continue

            rec92 = verify_section92_ising_recovery(J, extra_spins=raw_spins)
            dec = decode_raw_spins(raw_spins, J, t["kind"], t["meta"], t["df"], t["lookup"])
            cmp_errs = _compare_decoded_with_archived(tn, dec, spqc_block)
            if cmp_errs:
                failures.extend(cmp_errs)
                continue

            print(
                f"  [VERIFIED:{ckpt_mode}] {tn:26s} sha={cur_sha[:12]}.. C={rec92['ising_offset_C']} "
                f"certE={staged_entries[tn]['cert_min_ising_energy']} bestE={dec['best_ising_energy']} "
                f"id_diff={rec92['qubo_ising_identity_max_diff']} "
                f"strict_feas={dec['feasible_shot_count']}/{dec['n_samples']} ({dec['feasible_rate_pct']}%) "
                f"prim_feas={dec['primary_feasible_shot_count']}/{dec['n_samples']} "
                f"aux_ok={dec['y_aux_consistent_count']}/{dec['n_samples']} "
                f"unique={dec['unique_feasible_candidates']} dup={dec['duplicate_feasible_hits']} "
                f"best={dec['best_feasible']}",
                flush=True,
            )

        if args.task_set in ("v3", "all") and prior_archived:
            print("\n=== [Verify-Only] Checking Preserved Historical v2 Archive Checkpoints ===", flush=True)
            failures.extend(verify_historical_v2_checkpoints_integrity(prior_archived, args.ckpt_dir))

        if failures:
            print("\n[VERIFY-ONLY FAILED]", file=sys.stderr, flush=True)
            for msg in failures:
                print(f"  [FAIL] {msg}", file=sys.stderr, flush=True)
            return 1
        print(f"\n[VERIFY-ONLY PASSED] All {len(tasks)} task(s) passed bit-exact SHA-256, algebraic Ising recovery & full per-record comparison.", flush=True)
        return 0

    if args.run_spqc:
        uid = os.environ.get("KAIWU_USER_ID", "").strip()
        code = os.environ.get("KAIWU_SDK_CODE", "").strip()
        if not uid or not code:
            raise RuntimeError("Set KAIWU_USER_ID and KAIWU_SDK_CODE environment variables before --run-spqc.")

        if args.task_set == "v3":
            hist_bucket = payload.setdefault("historical_v2_archived_tasks", {})
            for old_tn in list(payload["tasks"].keys()):
                if old_tn.startswith("apmcm26_v2_"):
                    if old_tn not in hist_bucket:
                        hist_bucket[old_tn] = copy.deepcopy(payload["tasks"][old_tn])
                    del payload["tasks"][old_tn]

        # Enforce hard credit-safety guard: at most 4 uncached submissions in a run
        uncached_names = []
        for t in tasks:
            tn = t["task_name"]
            cur_sha = staged_entries[tn]["hardware_spec"]["sha256"]
            hash_path, legacy_path = _checkpoint_paths(tn, cur_sha, args.ckpt_dir)
            prev_entry = prior_tasks.get(tn, {})
            prev_hw_sha = prev_entry.get("hardware_spec", {}).get("sha256")
            spqc_sha = prev_entry.get("spqc550", {}).get("matrix_sha256")
            cached_path = hash_path if os.path.exists(hash_path) else (legacy_path if os.path.exists(legacy_path) else None)
            is_cached = (
                cached_path is not None
                and prev_entry.get("spqc550")
                and prev_hw_sha == cur_sha
                and (spqc_sha is None or spqc_sha == cur_sha)
            )
            if not is_cached:
                uncached_names.append(tn)
        if len(uncached_names) > 4:
            raise RuntimeError(
                f"Refusing to submit {len(uncached_names)} tasks ({uncached_names}): "
                "max 4 submissions allowed to strictly preserve >= 5 SPQC-550 credits."
            )

        kw.license.init(uid, code)

        from kaiwu.cim import CIMOptimizer, TaskMode
        kw.common.CheckpointManager.save_dir = args.ckpt_dir

        print(f"\n=== [Phase B] Executing Tasks on Real Bose Quantum SPQC-550 ({len(tasks)} task(s)) ===", flush=True)
        for idx, t in enumerate(tasks, start=1):
            tn = t["task_name"]
            J = t["J"]
            cur_sha = staged_entries[tn]["hardware_spec"]["sha256"]
            hash_path, legacy_path = _checkpoint_paths(tn, cur_sha, args.ckpt_dir)

            prev_entry = prior_tasks.get(tn, {})
            prev_hw_sha = prev_entry.get("hardware_spec", {}).get("sha256")
            spqc_sha = prev_entry.get("spqc550", {}).get("matrix_sha256")
            cached_path = hash_path if os.path.exists(hash_path) else (legacy_path if os.path.exists(legacy_path) else None)

            if (
                cached_path is not None
                and prev_entry.get("spqc550")
                and prev_hw_sha == cur_sha
                and (spqc_sha is None or spqc_sha == cur_sha)
            ):
                print(f"  ({idx}/{len(tasks)}) Reusing hash-verified SPQC-550 cache for {tn} (sha8={cur_sha[:8]})", flush=True)
                entry = copy.deepcopy(prev_entry)
                entry.update(staged_entries[tn])
                payload["tasks"][tn] = entry
                _update_credit_accounting(payload)
                _assert_no_sensitive_paths(payload)
                with open(args.results_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                continue

            if prev_hw_sha and prev_hw_sha != cur_sha:
                print(f"  ({idx}/{len(tasks)}) Matrix SHA changed for {tn} ({prev_hw_sha[:8]} -> {cur_sha[:8]}); invalidating stale cache.", flush=True)

            print(f"  ({idx}/{len(tasks)}) Submitting/Fetching '{tn}' (spins={J.shape[0]}, bw={safe_ising_bit_width(J)}, sha8={cur_sha[:8]}) on SPQC-550...", flush=True)
            t0 = time.perf_counter()
            raw_sols = None
            opt = None
            for attempt in range(1, 6):
                try:
                    opt = CIMOptimizer(
                        task_name=tn,
                        wait=True,
                        interval=max(1, int(args.poll_interval)),
                        task_mode=TaskMode.OPTIMIZATION,
                        sample_number=10,
                    )
                    raw_sols = opt.solve(J, negtail_flip=True, sort_solutions=False)
                    break
                except Exception as exc:
                    msg_low = str(exc).lower()
                    if any(w in msg_low for w in ("quota", "credit", "balance", "insufficient", "余额")):
                        raise RuntimeError(f"SPQC-550 credit/quota alert for '{tn}': {exc}") from exc
                    if attempt == 5:
                        raise
                    print(f"    [Retry {attempt}/5] Transient network error for '{tn}': {exc.__class__.__name__}; waiting 10s...", flush=True)
                    time.sleep(10)
            dt = time.perf_counter() - t0
            if raw_sols is None or len(raw_sols) == 0:
                raise RuntimeError(f"SPQC-550 returned empty solution for {tn}")

            raw_arr = np.asarray(raw_sols, dtype=np.int8)
            np.save(hash_path, raw_arr)
            dec = decode_raw_spins(raw_arr, J, t["kind"], t["meta"], t["df"], t["lookup"])
            dec["matrix_sha256"] = cur_sha
            dec["raw_spins_file_sha256"] = file_sha256(hash_path)
            dec["raw_spins_int8_sha256"] = spins_int8_sha256(raw_arr)
            dec["ising_offset_C"] = staged_entries[tn]["hardware_spec"]["ising_offset_C"]
            dec["cloud_roundtrip_wall_sec"] = round(dt, 2)
            req_sec = max(1, int(args.poll_interval))
            dec["poll_interval_requested_sec"] = req_sec
            safe_opt_meta = _extract_safe_optimizer_meta(opt)
            eff_sec = safe_opt_meta.get("interval", max(5, req_sec)) if safe_opt_meta else max(5, req_sec)
            dec["poll_interval_effective_sec"] = int(eff_sec)
            dec["poll_interval_unit_note"] = (
                "Kaiwu SDK 1.4.4 CIMOptimizer interval is in seconds (default 6s, min 5s)."
            )
            if safe_opt_meta:
                dec["sdk_optimizer_meta"] = safe_opt_meta

            entry = payload["tasks"].get(tn, {})
            entry.update(staged_entries[tn])
            entry["spqc550"] = dec
            payload["tasks"][tn] = entry
            _update_credit_accounting(payload)

            # Save incrementally after every single real hardware call!
            _assert_no_sensitive_paths(payload)
            with open(args.results_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            bf = dec["best_feasible"]
            bf_str = f"{bf['formula']} (#{bf['pool_rank']}, V={bf['V_hat']})" if bf else "INFEASIBLE"
            print(
                f"  -> [DONE {dt:.1f}s] {tn}: strict_feas={dec['feasible_shot_count']}/{dec['n_samples']} "
                f"({dec['feasible_rate_pct']}%), unique={dec['unique_feasible_candidates']}, best={bf_str}, "
                f"unique Top-{dec['top_k_target']} hits={dec['unique_topK_count']} {dec['unique_topK_hits']}",
                flush=True,
            )

        print(
            f"\nAll {len(tasks)} SPQC-550 hardware task(s) processed and saved to {_safe_relative_path(args.results_json)}.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
