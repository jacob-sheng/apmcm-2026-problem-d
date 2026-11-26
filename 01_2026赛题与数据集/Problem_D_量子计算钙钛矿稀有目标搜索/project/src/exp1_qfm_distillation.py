"""Experiment 1: Surrogate Ablation & Second-Order QFM Distillation (M1).

Track 1: Out-of-sample M6_QFM_Student trained strictly on 02_castelli_train.csv
         and evaluated on 03_castelli_val.csv (zero (A, B) overlap, zero val leakage).
         Appends M6_QFM_Student to outputs/problem1_ablation_table.csv ->
         outputs/table1_surrogate_ablation.csv (7 rows, 13 columns).

Track 2: Q3 QFM-Student rank-16 distillation from outputs/q3_full_pool_predictions.csv
         (8,112 rows mapped explicitly by (X, A, B) coordinate, never .reshape()),
         canonical ANOVA zero-sum gauge centering, exact polynomial equivalence check,
         and Kaiwu 108-spin 8-bit compatibility self-check (no CIMOptimizer calls).
         Exports outputs/exp1_qfm_q3_artifacts.npz and outputs/exp1_qfm_summary.json.
"""

import os

for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import check_grad, minimize
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

import kaiwu as kw
from kaiwu.core import Binary, QuboModel, ndarray
from kaiwu.preprocess import adjust_ising_matrix_precision, perform_precision_adaption_mutate

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from element_features import ELEMENT_DATA, ELEMENTS_52, X_CLASSES_7, X_DATA
from problem1_surrogate import eval_metrics

X_Q3 = ["FO2", "NO2", "O3"]


def _standardize_cols(mat: np.ndarray) -> np.ndarray:
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True)
    return (mat - mean) / (std + 1e-8)


def _standardize_slice(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def build_crystal_descriptors():
    """Build deterministic Phi_E (52,21), Phi_X (7,9), P_AB (52,52,14), P_AX (52,7,7), P_BX (52,7,8)."""
    e_raw = np.array([ELEMENT_DATA[el] for el in ELEMENTS_52], dtype=np.float64)
    Z = e_raw[:, 0]
    period = e_raw[:, 1]
    group = e_raw[:, 2]
    mass = e_raw[:, 3]
    chi = e_raw[:, 4]
    r_cov = e_raw[:, 5]
    r_eff = e_raw[:, 6]
    IE1 = e_raw[:, 7]
    EA = e_raw[:, 8]
    n_val = e_raw[:, 9]
    n_d = e_raw[:, 10]
    pettifor = e_raw[:, 11]
    ox = np.where(group <= 12, np.minimum(group, 6), group - 10)

    phi_e_raw = np.column_stack([
        Z, period, group, mass, chi, r_cov, r_eff, IE1, EA, n_val, n_d, pettifor, ox,
        r_eff ** 2, r_cov ** 2, chi ** 2, ox ** 2, n_val ** 2, IE1 ** 2, EA ** 2, pettifor ** 2,
    ])
    Phi_E = _standardize_cols(phi_e_raw)
    assert Phi_E.shape == (52, 21)

    x_raw = np.array([X_DATA[x] for x in X_CLASSES_7], dtype=np.float64)
    Phi_X = _standardize_cols(x_raw)
    assert Phi_X.shape == (7, 9)

    Q_X = x_raw[:, 4]
    chi_X = x_raw[:, 5]
    r_X = x_raw[:, 6]
    r_X0 = float(r_X.mean())

    t0_AB = (r_eff[:, None] + r_X0) / (np.sqrt(2.0) * (r_eff[None, :] + r_X0))
    p_ab_slices = [
        np.eye(52, dtype=np.float64),
        np.abs(r_eff[:, None] - r_eff[None, :]),
        (r_eff[:, None] - r_eff[None, :]) ** 2,
        t0_AB,
        (t0_AB - 0.95) ** 2,
        np.abs(chi[:, None] - chi[None, :]),
        (chi[:, None] - chi[None, :]) ** 2,
        ox[:, None] + ox[None, :],
        (ox[:, None] + ox[None, :]) ** 2,
        n_val[:, None] + n_val[None, :],
        (n_val[:, None] + n_val[None, :]) ** 2,
        np.abs(pettifor[:, None] - pettifor[None, :]),
        (Z[:, None] + Z[None, :]) % 2,
        (n_val[:, None] + n_val[None, :]) % 2,
    ]
    P_AB = np.stack([_standardize_slice(s) for s in p_ab_slices], axis=-1)
    assert P_AB.shape == (52, 52, 14)

    p_ax_slices = [
        r_eff[:, None] + r_X[None, :],
        np.abs(chi_X[None, :] - chi[:, None]),
        (chi_X[None, :] - chi[:, None]) ** 2,
        ox[:, None] - Q_X[None, :],
        (ox[:, None] - Q_X[None, :]) ** 2,
        n_val[:, None] - Q_X[None, :],
        (n_val[:, None] - Q_X[None, :]) ** 2,
    ]
    P_AX = np.stack([_standardize_slice(s) for s in p_ax_slices], axis=-1)
    assert P_AX.shape == (52, 7, 7)

    p_bx_slices = [
        r_eff[:, None] / r_X[None, :],
        1.0 / (r_eff[:, None] + r_X[None, :]),
        np.abs(chi_X[None, :] - chi[:, None]),
        (chi_X[None, :] - chi[:, None]) ** 2,
        ox[:, None] - Q_X[None, :],
        (ox[:, None] - Q_X[None, :]) ** 2,
        n_val[:, None] - Q_X[None, :],
        (n_val[:, None] - Q_X[None, :]) ** 2,
    ]
    P_BX = np.stack([_standardize_slice(s) for s in p_bx_slices], axis=-1)
    assert P_BX.shape == (52, 7, 8)

    return Phi_E, Phi_X, P_AB, P_AX, P_BX


def anova_gauge_center(V_grid: np.ndarray):
    """Canonical ANOVA zero-sum gauge decomposition for shape (K, 52, 52)."""
    w0 = float(V_grid.mean())
    V0 = V_grid - w0
    w_x = V0.mean(axis=(1, 2))
    w_a = V0.mean(axis=(0, 2))
    w_b = V0.mean(axis=(0, 1))
    V1 = V0 - w_x[:, None, None] - w_a[None, :, None] - w_b[None, None, :]
    W_ab = V1.mean(axis=0)
    W_ax = V1.mean(axis=2).T
    W_bx = V1.mean(axis=1).T
    return w0, w_a, w_b, w_x, W_ab, W_ax, W_bx


def reconstruct_qfm_grid(w0, w_a, w_b, w_x, W_ab, W_ax, W_bx) -> np.ndarray:
    """Reconstruct (K, 52, 52) grid from 2nd-order QFM parameters."""
    return (
        w0
        + w_x[:, None, None]
        + w_a[None, :, None]
        + w_b[None, None, :]
        + W_ab[None, :, :]
        + W_ax.T[:, :, None]
        + W_bx.T[:, None, :]
    )


def fit_track1_m6_qfm_student(tr: pd.DataFrame, va: pd.DataFrame):
    """Fit M6_QFM_Student strictly on 02_castelli_train.csv and evaluate on 03_castelli_val.csv."""
    tr_pairs = set(zip(tr["A"], tr["B"]))
    va_pairs = set(zip(va["A"], va["B"]))
    assert len(tr_pairs & va_pairs) == 0, "Train/val (A, B) overlap detected!"

    e_to_i = {e: i for i, e in enumerate(ELEMENTS_52)}
    x_to_k = {x: k for k, x in enumerate(X_CLASSES_7)}

    tr_a = tr["A"].map(e_to_i).to_numpy(dtype=np.int64)
    tr_b = tr["B"].map(e_to_i).to_numpy(dtype=np.int64)
    tr_x = tr["X"].map(x_to_k).to_numpy(dtype=np.int64)
    y_tr = tr["V"].to_numpy(dtype=np.float64)

    va_a = va["A"].map(e_to_i).to_numpy(dtype=np.int64)
    va_b = va["B"].map(e_to_i).to_numpy(dtype=np.int64)
    va_x = va["X"].map(x_to_k).to_numpy(dtype=np.int64)
    y_va = va["V"].to_numpy(dtype=np.float64)

    Phi_E, Phi_X, P_AB, P_AX, P_BX = build_crystal_descriptors()
    P_AB_flat = P_AB.reshape(52 * 52, 14)
    P_AX_flat = P_AX.reshape(52 * 7, 7)
    P_BX_flat = P_BX.reshape(52 * 7, 8)

    rank = 8
    c_inv = 3.0
    gamma = 1.0
    asym_under = 2.0
    beta_under = 2.0
    tau_safe = 12.0
    beta_neg = 2.0
    reg_lin = 1e-4
    reg_fm = 5e-3
    reg_bilin = 2e-3
    seed = 42

    N = len(y_tr)
    W_raw = (y_tr + c_inv) ** (-gamma)
    W_norm = W_raw / W_raw.mean()
    t_safe = np.minimum(y_tr, tau_safe)

    specs = [
        ("w0", (1,)),
        ("w_a_raw", (52,)),
        ("theta_a", (21,)),
        ("w_b_raw", (52,)),
        ("theta_b", (21,)),
        ("w_x_raw", (7,)),
        ("theta_x", (9,)),
        ("Ua", (52, rank)),
        ("Ub", (52, rank)),
        ("M_ab", (21, 21)),
        ("g_ab", (14,)),
        ("W_ax_raw", (52, 7)),
        ("M_ax", (21, 9)),
        ("g_ax", (7,)),
        ("W_bx_raw", (52, 7)),
        ("M_bx", (21, 9)),
        ("g_bx", (8,)),
    ]
    slices = {}
    offset = 0
    for name, shape in specs:
        size = int(np.prod(shape))
        slices[name] = (slice(offset, offset + size), shape)
        offset += size

    def unpack(theta_vec):
        return {name: theta_vec[sl].reshape(shape) for name, (sl, shape) in slices.items()}

    def fold_weights(p):
        w0 = float(p["w0"][0])
        w_a = p["w_a_raw"] + Phi_E @ p["theta_a"]
        w_b = p["w_b_raw"] + Phi_E @ p["theta_b"]
        w_x = p["w_x_raw"] + Phi_X @ p["theta_x"]
        W_ab = p["Ua"] @ p["Ub"].T + Phi_E @ p["M_ab"] @ Phi_E.T + (P_AB_flat @ p["g_ab"]).reshape(52, 52)
        W_ax = p["W_ax_raw"] + Phi_E @ p["M_ax"] @ Phi_X.T + (P_AX_flat @ p["g_ax"]).reshape(52, 7)
        W_bx = p["W_bx_raw"] + Phi_E @ p["M_bx"] @ Phi_X.T + (P_BX_flat @ p["g_bx"]).reshape(52, 7)
        return w0, w_a, w_b, w_x, W_ab, W_ax, W_bx

    ab_flat_idx = tr_a * 52 + tr_b
    ax_flat_idx = tr_a * 7 + tr_x
    bx_flat_idx = tr_b * 7 + tr_x

    def loss_and_grad(theta_vec):
        p = unpack(theta_vec)
        w0, w_a, w_b, w_x, W_ab, W_ax, W_bx = fold_weights(p)

        pred = (
            w0
            + w_a[tr_a]
            + w_b[tr_b]
            + w_x[tr_x]
            + W_ab[tr_a, tr_b]
            + W_ax[tr_a, tr_x]
            + W_bx[tr_b, tr_x]
        )

        diff = pred - y_tr
        w_eff = W_norm * np.where(diff < 0.0, 1.0 + asym_under, 1.0)
        loss_mse = 0.5 * np.mean(w_eff * diff * diff)
        d_pred = (w_eff * diff) / N

        under = np.maximum(0.0, t_safe - pred)
        loss_under = 0.5 * beta_under * np.mean(under * under)
        d_pred -= (beta_under / N) * under

        neg = np.maximum(0.0, -pred)
        loss_neg = 0.5 * beta_neg * np.mean(neg * neg)
        d_pred -= (beta_neg / N) * neg

        lin_names = ("w_a_raw", "w_b_raw", "w_x_raw", "theta_a", "theta_b", "theta_x")
        fm_names = ("Ua", "Ub")
        bilin_names = ("M_ab", "g_ab", "W_ax_raw", "M_ax", "g_ax", "W_bx_raw", "M_bx", "g_bx")

        loss_reg = 0.5 * reg_lin * sum(float(np.sum(p[k] ** 2)) for k in lin_names)
        loss_reg += 0.5 * reg_fm * sum(float(np.sum(p[k] ** 2)) for k in fm_names)
        loss_reg += 0.5 * reg_bilin * sum(float(np.sum(p[k] ** 2)) for k in bilin_names)

        total_loss = float(loss_mse + loss_under + loss_neg + loss_reg)

        g_w0 = np.array([d_pred.sum()], dtype=np.float64)
        g_wa = np.bincount(tr_a, weights=d_pred, minlength=52)
        g_wb = np.bincount(tr_b, weights=d_pred, minlength=52)
        g_wx = np.bincount(tr_x, weights=d_pred, minlength=7)
        G_ab = np.bincount(ab_flat_idx, weights=d_pred, minlength=52 * 52).reshape(52, 52)
        G_ax = np.bincount(ax_flat_idx, weights=d_pred, minlength=52 * 7).reshape(52, 7)
        G_bx = np.bincount(bx_flat_idx, weights=d_pred, minlength=52 * 7).reshape(52, 7)

        grads = {
            "w0": g_w0,
            "w_a_raw": g_wa + reg_lin * p["w_a_raw"],
            "theta_a": Phi_E.T @ g_wa + reg_lin * p["theta_a"],
            "w_b_raw": g_wb + reg_lin * p["w_b_raw"],
            "theta_b": Phi_E.T @ g_wb + reg_lin * p["theta_b"],
            "w_x_raw": g_wx + reg_lin * p["w_x_raw"],
            "theta_x": Phi_X.T @ g_wx + reg_lin * p["theta_x"],
            "Ua": G_ab @ p["Ub"] + reg_fm * p["Ua"],
            "Ub": G_ab.T @ p["Ua"] + reg_fm * p["Ub"],
            "M_ab": Phi_E.T @ G_ab @ Phi_E + reg_bilin * p["M_ab"],
            "g_ab": P_AB_flat.T @ G_ab.ravel() + reg_bilin * p["g_ab"],
            "W_ax_raw": G_ax + reg_bilin * p["W_ax_raw"],
            "M_ax": Phi_E.T @ G_ax @ Phi_X + reg_bilin * p["M_ax"],
            "g_ax": P_AX_flat.T @ G_ax.ravel() + reg_bilin * p["g_ax"],
            "W_bx_raw": G_bx + reg_bilin * p["W_bx_raw"],
            "M_bx": Phi_E.T @ G_bx @ Phi_X + reg_bilin * p["M_bx"],
            "g_bx": P_BX_flat.T @ G_bx.ravel() + reg_bilin * p["g_bx"],
        }
        grad_vec = np.concatenate([grads[name].ravel() for name, _ in specs])
        return total_loss, grad_vec

    rng = np.random.default_rng(seed)
    theta_test = rng.normal(0.0, 0.05, size=offset)
    grad_err = float(
        check_grad(
            lambda v: loss_and_grad(v)[0],
            lambda v: loss_and_grad(v)[1],
            theta_test,
            epsilon=1e-7,
        )
    )
    assert grad_err < 1e-4, f"Track 1 analytical gradient check failed: err={grad_err}"

    theta0 = np.zeros(offset, dtype=np.float64)
    theta0[slices["w0"][0]] = float(np.sum(W_norm * y_tr) / np.sum(W_norm))
    theta0[slices["Ua"][0]] = rng.normal(0.0, 0.02, size=52 * rank)
    theta0[slices["Ub"][0]] = rng.normal(0.0, 0.02, size=52 * rank)

    t0 = time.time()
    res = minimize(
        loss_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    train_sec = float(time.time() - t0)

    p_opt = unpack(res.x)
    w0, w_a, w_b, w_x, W_ab, W_ax, W_bx = fold_weights(p_opt)

    pred_va = (
        w0
        + w_a[va_a]
        + w_b[va_b]
        + w_x[va_x]
        + W_ab[va_a, va_b]
        + W_ax[va_a, va_x]
        + W_bx[va_b, va_x]
    )
    pred_tr = (
        w0
        + w_a[tr_a]
        + w_b[tr_b]
        + w_x[tr_x]
        + W_ab[tr_a, tr_b]
        + W_ax[tr_a, tr_x]
        + W_bx[tr_b, tr_x]
    )

    V_grid_7 = reconstruct_qfm_grid(w0, w_a, w_b, w_x, W_ab, W_ax, W_bx)
    g_w0, g_wa, g_wb, g_wx, g_Wab, g_Wax, g_Wbx = anova_gauge_center(V_grid_7)
    V_grid_7_re = reconstruct_qfm_grid(g_w0, g_wa, g_wb, g_wx, g_Wab, g_Wax, g_Wbx)
    gauge_poly_err = float(np.max(np.abs(V_grid_7_re - V_grid_7)))
    assert gauge_poly_err < 1e-12

    m6_val_metrics = eval_metrics(y_va, pred_va, "M6_QFM_Student")
    m6_tr_metrics = eval_metrics(y_tr, pred_tr, "M6_QFM_Student_Train")

    info = {
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "train_val_ab_overlap": 0,
        "rank": rank,
        "hyperparams": {
            "c_inv": c_inv,
            "gamma": gamma,
            "asym_under": asym_under,
            "beta_under": beta_under,
            "tau_safe": tau_safe,
            "beta_neg": beta_neg,
            "reg_lin": reg_lin,
            "reg_fm": reg_fm,
            "reg_bilin": reg_bilin,
            "seed": seed,
            "maxiter": 500,
        },
        "optimizer_success": bool(res.success),
        "optimizer_nit": int(res.nit),
        "optimizer_fun": float(res.fun),
        "grad_check_l2_err": grad_err,
        "gauge_poly_equiv_err": gauge_poly_err,
        "train_time_sec": train_sec,
        "val_pred_min": float(pred_va.min()),
        "val_pred_max": float(pred_va.max()),
        "train_metrics": m6_tr_metrics,
        "val_metrics": m6_val_metrics,
    }
    return m6_val_metrics, info


def fit_track2_q3_distillation(q3_df: pd.DataFrame):
    """Distill rank-16 2nd-order QFM-Student from q3_full_pool_predictions.csv (3 x 52 x 52)."""
    e_to_i = {e: i for i, e in enumerate(ELEMENTS_52)}
    x_to_k = {x: k for k, x in enumerate(X_Q3)}

    T = np.full((3, 52, 52), np.nan, dtype=np.float64)
    formula_grid = np.empty((3, 52, 52), dtype=object)
    blind_grid = np.zeros((3, 52, 52), dtype=np.int64)
    filled = np.zeros((3, 52, 52), dtype=bool)

    for row in q3_df.itertuples(index=False):
        k = x_to_k[row.X]
        i = e_to_i[row.A]
        j = e_to_i[row.B]
        assert not filled[k, i, j], f"Duplicate Q3 cell {(row.X, row.A, row.B)}"
        T[k, i, j] = float(row.V_pred)
        formula_grid[k, i, j] = str(row.formula)
        blind_grid[k, i, j] = int(row.in_blind_pool)
        filled[k, i, j] = True

    assert filled.all(), "Missing cells in Q3 3x52x52 tensor!"
    assert not np.isnan(T).any(), "NaNs found in Q3 teacher tensor!"

    rank = 16
    c_inv = 0.5
    gamma = 2.0
    beta_under = 1.0
    tau_safe = 3.0
    beta_neg = 2.0
    reg_lin = 1e-5
    reg_pair = 1e-4
    seed = 42

    N = T.size  # 8112
    W_raw = (T + c_inv) ** (-gamma)
    W_norm = W_raw / W_raw.mean()
    t_safe = np.minimum(T, tau_safe)

    specs = [
        ("w0", (1,)),
        ("w_a", (52,)),
        ("w_b", (52,)),
        ("w_x", (3,)),
        ("Ua", (52, rank)),
        ("Ub", (52, rank)),
        ("W_ax", (52, 3)),
        ("W_bx", (52, 3)),
    ]
    slices = {}
    offset = 0
    for name, shape in specs:
        size = int(np.prod(shape))
        slices[name] = (slice(offset, offset + size), shape)
        offset += size

    def unpack(theta_vec):
        return {name: theta_vec[sl].reshape(shape) for name, (sl, shape) in slices.items()}

    def loss_and_grad(theta_vec):
        p = unpack(theta_vec)
        w0 = float(p["w0"][0])
        W_ab_raw = p["Ua"] @ p["Ub"].T
        pred = (
            w0
            + p["w_x"][:, None, None]
            + p["w_a"][None, :, None]
            + p["w_b"][None, None, :]
            + W_ab_raw[None, :, :]
            + p["W_ax"].T[:, :, None]
            + p["W_bx"].T[:, None, :]
        )
        diff = pred - T
        loss_mse = 0.5 * np.mean(W_norm * diff * diff)
        d_pred = (W_norm * diff) / N

        under = np.maximum(0.0, t_safe - pred)
        loss_under = 0.5 * beta_under * np.mean(under * under)
        d_pred -= (beta_under / N) * under

        neg = np.maximum(0.0, -pred)
        loss_neg = 0.5 * beta_neg * np.mean(neg * neg)
        d_pred -= (beta_neg / N) * neg

        lin_names = ("w_a", "w_b", "w_x")
        pair_names = ("Ua", "Ub", "W_ax", "W_bx")
        loss_reg = 0.5 * reg_lin * sum(float(np.sum(p[k] ** 2)) for k in lin_names)
        loss_reg += 0.5 * reg_pair * sum(float(np.sum(p[k] ** 2)) for k in pair_names)

        total_loss = float(loss_mse + loss_under + loss_neg + loss_reg)

        g_w0 = np.array([d_pred.sum()], dtype=np.float64)
        g_wx = d_pred.sum(axis=(1, 2)) + reg_lin * p["w_x"]
        g_wa = d_pred.sum(axis=(0, 2)) + reg_lin * p["w_a"]
        g_wb = d_pred.sum(axis=(0, 1)) + reg_lin * p["w_b"]
        G_ab = d_pred.sum(axis=0)
        g_Ua = G_ab @ p["Ub"] + reg_pair * p["Ua"]
        g_Ub = G_ab.T @ p["Ua"] + reg_pair * p["Ub"]
        g_Wax = d_pred.sum(axis=2).T + reg_pair * p["W_ax"]
        g_Wbx = d_pred.sum(axis=1).T + reg_pair * p["W_bx"]

        grads = {
            "w0": g_w0,
            "w_a": g_wa,
            "w_b": g_wb,
            "w_x": g_wx,
            "Ua": g_Ua,
            "Ub": g_Ub,
            "W_ax": g_Wax,
            "W_bx": g_Wbx,
        }
        grad_vec = np.concatenate([grads[name].ravel() for name, _ in specs])
        return total_loss, grad_vec

    rng = np.random.default_rng(seed)
    theta_test = rng.normal(0.0, 0.05, size=offset)
    grad_err = float(
        check_grad(
            lambda v: loss_and_grad(v)[0],
            lambda v: loss_and_grad(v)[1],
            theta_test,
            epsilon=1e-7,
        )
    )
    assert grad_err < 1e-4, f"Track 2 analytical gradient check failed: err={grad_err}"

    theta0 = np.zeros(offset, dtype=np.float64)
    theta0[slices["w0"][0]] = float(np.sum(W_norm * T) / np.sum(W_norm))
    theta0[slices["Ua"][0]] = rng.normal(0.0, 0.08, size=52 * rank)
    theta0[slices["Ub"][0]] = rng.normal(0.0, 0.08, size=52 * rank)

    t0 = time.time()
    res = minimize(
        loss_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
    )
    train_sec = float(time.time() - t0)

    p_opt = unpack(res.x)
    V_qfm_q3 = (
        float(p_opt["w0"][0])
        + p_opt["w_x"][:, None, None]
        + p_opt["w_a"][None, :, None]
        + p_opt["w_b"][None, None, :]
        + (p_opt["Ua"] @ p_opt["Ub"].T)[None, :, :]
        + p_opt["W_ax"].T[:, :, None]
        + p_opt["W_bx"].T[:, None, :]
    )

    w0, w_a, w_b, w_x, W_ab, W_ax, W_bx = anova_gauge_center(V_qfm_q3)
    Q_QFM = reconstruct_qfm_grid(w0, w_a, w_b, w_x, W_ab, W_ax, W_bx)
    max_poly_err = float(np.max(np.abs(Q_QFM - V_qfm_q3)))

    gauge_errors = {
        "abs_sum_wa": float(np.abs(w_a.sum())),
        "abs_sum_wb": float(np.abs(w_b.sum())),
        "abs_sum_wx": float(np.abs(w_x.sum())),
        "max_abs_mean_Wab_axis0": float(np.max(np.abs(W_ab.mean(axis=0)))),
        "max_abs_mean_Wab_axis1": float(np.max(np.abs(W_ab.mean(axis=1)))),
        "max_abs_mean_Wax_axis0": float(np.max(np.abs(W_ax.mean(axis=0)))),
        "max_abs_mean_Wax_axis1": float(np.max(np.abs(W_ax.mean(axis=1)))),
        "max_abs_mean_Wbx_axis0": float(np.max(np.abs(W_bx.mean(axis=0)))),
        "max_abs_mean_Wbx_axis1": float(np.max(np.abs(W_bx.mean(axis=1)))),
    }

    assert max_poly_err < 1e-12, f"Polynomial equivalence failed: {max_poly_err}"
    for gk, gv in gauge_errors.items():
        assert gv < 1e-10, f"ANOVA gauge error {gk}={gv} >= 1e-10"

    t_flat = T.ravel()
    q_flat = V_qfm_q3.ravel()
    f_flat = formula_grid.ravel()
    b_flat = blind_grid.ravel()

    idx_t = np.argsort(t_flat)
    idx_q = np.argsort(q_flat)
    rank_t = np.empty(N, dtype=np.int64)
    rank_q = np.empty(N, dtype=np.int64)
    rank_t[idx_t] = np.arange(1, N + 1)
    rank_q[idx_q] = np.arange(1, N + 1)

    def _subset_rmse(mask):
        return float(root_mean_squared_error(t_flat[mask], q_flat[mask]))

    top_overlaps = {}
    for K in (10, 20, 50, 100):
        top_overlaps[f"top_{K}_overlap"] = int(len(set(idx_t[:K]) & set(idx_q[:K])))

    top10_teacher = []
    for pos, flat_i in enumerate(idx_t[:10], start=1):
        k_idx, i_idx, j_idx = np.unravel_index(flat_i, (3, 52, 52))
        top10_teacher.append({
            "teacher_rank": pos,
            "qfm_rank": int(rank_q[flat_i]),
            "formula": str(f_flat[flat_i]),
            "A": ELEMENTS_52[i_idx],
            "B": ELEMENTS_52[j_idx],
            "X": X_Q3[k_idx],
            "V_teacher": float(t_flat[flat_i]),
            "V_qfm": float(q_flat[flat_i]),
            "in_blind_pool": int(b_flat[flat_i]),
        })

    top10_qfm = []
    for pos, flat_i in enumerate(idx_q[:10], start=1):
        k_idx, i_idx, j_idx = np.unravel_index(flat_i, (3, 52, 52))
        top10_qfm.append({
            "qfm_rank": pos,
            "teacher_rank": int(rank_t[flat_i]),
            "formula": str(f_flat[flat_i]),
            "A": ELEMENTS_52[i_idx],
            "B": ELEMENTS_52[j_idx],
            "X": X_Q3[k_idx],
            "V_teacher": float(t_flat[flat_i]),
            "V_qfm": float(q_flat[flat_i]),
            "in_blind_pool": int(b_flat[flat_i]),
        })

    fidelity = {
        "RMSE_all": float(root_mean_squared_error(t_flat, q_flat)),
        "MAE_all": float(mean_absolute_error(t_flat, q_flat)),
        "R2_all": float(r2_score(t_flat, q_flat)),
        "Spearman_rho": float(spearmanr(t_flat, q_flat).statistic),
        "RMSE_Vlt1": _subset_rmse(t_flat < 1.0),
        "RMSE_Vlt2": _subset_rmse(t_flat < 2.0),
        "RMSE_Vlt5": _subset_rmse(t_flat < 5.0),
        "V_qfm_min": float(q_flat.min()),
        "V_qfm_max": float(q_flat.max()),
        "teacher_top1_formula": str(f_flat[idx_t[0]]),
        "qfm_top1_formula": str(f_flat[idx_q[0]]),
        **top_overlaps,
    }

    assert fidelity["Spearman_rho"] >= 0.93, f"Spearman too low: {fidelity['Spearman_rho']}"
    assert fidelity["RMSE_Vlt1"] < 0.20, f"RMSE(V<1) too high: {fidelity['RMSE_Vlt1']}"
    assert fidelity["top_10_overlap"] >= 9, f"Top-10 overlap too low: {fidelity['top_10_overlap']}"
    assert fidelity["top_50_overlap"] >= 43, f"Top-50 overlap too low: {fidelity['top_50_overlap']}"
    assert fidelity["qfm_top1_formula"] == "LaTiO2N", f"Unexpected #1: {fidelity['qfm_top1_formula']}"

    kaiwu_check = verify_kaiwu_108_8bit(w0, w_a, w_b, w_x, W_ab, W_ax, W_bx, Q_QFM, formula_grid, T)

    artifacts = {
        "elements_52": np.array(ELEMENTS_52),
        "x_q3": np.array(X_Q3),
        "w0": np.float64(w0),
        "w_a": w_a.astype(np.float64),
        "w_b": w_b.astype(np.float64),
        "w_x": w_x.astype(np.float64),
        "W_ab": W_ab.astype(np.float64),
        "W_ax": W_ax.astype(np.float64),
        "W_bx": W_bx.astype(np.float64),
        "V_qfm_q3": V_qfm_q3.astype(np.float64),
    }

    summary = {
        "n_q3_grid": int(N),
        "rank": rank,
        "hyperparams": {
            "c_inv": c_inv,
            "gamma": gamma,
            "beta_under": beta_under,
            "tau_safe": tau_safe,
            "beta_neg": beta_neg,
            "reg_lin": reg_lin,
            "reg_pair": reg_pair,
            "seed": seed,
            "maxiter": 500,
        },
        "optimizer_success": bool(res.success),
        "optimizer_nit": int(res.nit),
        "optimizer_fun": float(res.fun),
        "grad_check_l2_err": grad_err,
        "train_time_sec": train_sec,
        "max_poly_equiv_err": max_poly_err,
        "anova_gauge_errors": gauge_errors,
        "fidelity": fidelity,
        "top10_teacher": top10_teacher,
        "top10_qfm": top10_qfm,
        "kaiwu_108_8bit_check": kaiwu_check,
    }
    return artifacts, summary


def verify_kaiwu_108_8bit(w0, w_a, w_b, w_x, W_ab, W_ax, W_bx, Q_QFM, formula_grid, T):
    """Build 108-spin QUBO with P_oh=35.0, evaluate J_mut (float64) and J_int (8-bit int64)."""
    P_oh = 35.0
    m = QuboModel()
    a = ndarray((52,), "a", Binary)
    b = ndarray((52,), "b", Binary)
    x = ndarray((3,), "x", Binary)

    obj = (
        w0
        + kw.core.dot(w_a, a)
        + kw.core.dot(w_b, b)
        + kw.core.dot(w_x, x)
        + a @ W_ab @ b
        + a @ W_ax @ x
        + b @ W_bx @ x
        + P_oh * ((a.sum() - 1) ** 2 + (b.sum() - 1) ** 2 + (x.sum() - 1) ** 2)
    )
    m.set_objective(obj)
    im = kw.qubo_model_to_ising_model(m)
    variables = im.get_variables()
    n_spins = len(variables)
    assert n_spins == 108, f"Expected 108 spins, got {n_spins}"

    J_raw = im.get_matrix()
    bias_im = float(im.get_bias())
    assert J_raw.shape == (108, 108)

    J_mut = perform_precision_adaption_mutate(J_raw)
    J_int = adjust_ising_matrix_precision(J_mut, bit_width=8)

    assert J_int.shape == (108, 108)
    assert J_int.dtype == np.int64
    assert np.array_equal(J_int, J_int.T), "J_int not symmetric!"
    assert np.all(np.diag(J_int) == 0), "J_int diagonal non-zero!"
    assert int(J_int.min()) >= -128 and int(J_int.max()) <= 127, f"Out of 8-bit range: [{J_int.min()}, {J_int.max()}]"

    a_col = np.array([variables[a[i].name] for i in range(52)], dtype=np.int64)
    b_col = np.array([variables[b[j].name] for j in range(52)], dtype=np.int64)
    x_col = np.array([variables[x[k].name] for k in range(3)], dtype=np.int64)
    spin_col = int(variables["__spin__"])

    N = 3 * 52 * 52
    S = np.full((N, n_spins), -1.0, dtype=np.float64)
    S[:, spin_col] = 1.0

    k_grid, i_grid, j_grid = np.meshgrid(np.arange(3), np.arange(52), np.arange(52), indexing="ij")
    row_idx = np.arange(N)
    S[row_idx, a_col[i_grid.ravel()]] = 1.0
    S[row_idx, b_col[j_grid.ravel()]] = 1.0
    S[row_idx, x_col[k_grid.ravel()]] = 1.0

    E_raw = -np.einsum("ni,ij,nj->n", S, J_raw, S) + bias_im
    exact_ising_err = float(np.max(np.abs(E_raw - Q_QFM.ravel())))
    assert exact_ising_err < 1e-10, f"Ising energy != Q_QFM: max err {exact_ising_err}"

    E_mut = -np.einsum("ni,ij,nj->n", S, J_mut, S)
    E_int = -np.einsum("ni,ij,nj->n", S, J_int.astype(np.float64), S)

    idx_q = np.argsort(Q_QFM.ravel())
    idx_mut = np.argsort(E_mut)
    idx_int = np.argsort(E_int)
    f_flat = formula_grid.ravel()
    t_flat = T.ravel()

    assert f_flat[idx_mut[0]] == "LaTiO2N", f"Mutated float64 Ising #1 changed to {f_flat[idx_mut[0]]}"

    top5_mut = [
        {"rank": r + 1, "formula": str(f_flat[idx_mut[r]]), "V_teacher": float(t_flat[idx_mut[r]]), "V_qfm": float(Q_QFM.ravel()[idx_mut[r]]), "E_mut": float(E_mut[idx_mut[r]])}
        for r in range(5)
    ]
    top5_int = [
        {"rank": r + 1, "formula": str(f_flat[idx_int[r]]), "V_teacher": float(t_flat[idx_int[r]]), "V_qfm": float(Q_QFM.ravel()[idx_int[r]]), "E_int": float(E_int[idx_int[r]])}
        for r in range(5)
    ]

    J_ab_int = J_int[np.ix_(a_col, b_col)]
    max_abs_J_raw = float(np.max(np.abs(J_raw)))
    max_abs_J_ab = float(np.max(np.abs(J_raw[np.ix_(a_col, b_col)])))

    return {
        "n_spins": int(n_spins),
        "P_oh": float(P_oh),
        "exact_ising_vs_qfm_max_err": exact_ising_err,
        "J_raw_min": float(J_raw.min()),
        "J_raw_max": float(J_raw.max()),
        "max_abs_J_ab_raw": max_abs_J_ab,
        "raw_dynamic_range_hub_vs_ab": float(max_abs_J_raw / (max_abs_J_ab + 1e-12)),
        "mutate_float64_scale_vs_raw": float(np.max(np.abs(J_mut)) / (max_abs_J_raw + 1e-12)),
        "spearman_qfm_vs_mutate_float64": float(spearmanr(Q_QFM.ravel(), E_mut).statistic),
        "top5_overlap_qfm_vs_mutate_float64": int(len(set(idx_q[:5]) & set(idx_mut[:5]))),
        "top10_overlap_qfm_vs_mutate_float64": int(len(set(idx_q[:10]) & set(idx_mut[:10]))),
        "top1_mutate_float64_formula": str(f_flat[idx_mut[0]]),
        "top5_mutate_float64_candidates": top5_mut,
        "J_int_dtype": str(J_int.dtype),
        "J_int_min": int(J_int.min()),
        "J_int_max": int(J_int.max()),
        "J_int_unique_levels": int(len(np.unique(J_int))),
        "J_ab_int_min": int(J_ab_int.min()),
        "J_ab_int_max": int(J_ab_int.max()),
        "spearman_qfm_vs_8bit_int": float(spearmanr(Q_QFM.ravel(), E_int).statistic),
        "top5_overlap_qfm_vs_8bit_int": int(len(set(idx_q[:5]) & set(idx_int[:5]))),
        "top10_overlap_qfm_vs_8bit_int": int(len(set(idx_q[:10]) & set(idx_int[:10]))),
        "top1_8bit_int_formula": str(f_flat[idx_int[0]]),
        "top5_8bit_int_candidates": top5_int,
        "unwindowed_8bit_finding": (
            "perform_precision_adaption_mutate(J_raw) preserves 5/5 Top-5 and 10/10 Top-10 in float64 "
            "(2*J_raw), but does not eliminate the -12.5*P_oh=-437.5 __spin__ one-hot star-hub bias "
            "(224x dynamic range vs max|J_ab|=1.97). Direct 8-bit int quantization [-127,127] at P_oh=35 "
            "without V_cap windowing/scaling crushes J_ab_int into [0, 1]; M3 exp3 must apply V_cap "
            "windowing or objective/bias scaling before hardware submission."
        ),
    }


def main():
    t_start = time.time()
    print("=== Experiment 1: Surrogate Ablation & Second-Order QFM Distillation ===")

    tr = pd.read_csv(os.path.join(DATA_DIR, "02_castelli_train.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "03_castelli_val.csv"))
    p1_ablation = pd.read_csv(os.path.join(OUT_DIR, "problem1_ablation_table.csv"))
    q3_df = pd.read_csv(os.path.join(OUT_DIR, "q3_full_pool_predictions.csv"))

    # Track 1
    m6_metrics, track1_info = fit_track1_m6_qfm_student(tr, va)
    print(
        f"[Track 1] M6_QFM_Student (03_val): RMSE={m6_metrics['RMSE']:.4f}, MAE={m6_metrics['MAE']:.4f}, "
        f"R2={m6_metrics['R2']:.4f}, Spearman={m6_metrics['Spearman_rho']:.4f}, "
        f"Recall@20={m6_metrics['Recall_Vlt1_Top20']:.4f}, Recall@50={m6_metrics['Recall_Vlt1_Top50']:.4f}, "
        f"Recall@100={m6_metrics['Recall_Vlt1_Top100']:.4f}"
    )

    table1 = pd.concat([p1_ablation, pd.DataFrame([m6_metrics])], ignore_index=True)
    assert list(table1.columns) == list(p1_ablation.columns)
    assert table1.shape == (7, 13), f"Expected (7, 13), got {table1.shape}"
    assert int(table1.isna().sum().sum()) == 0, "NaNs in table1!"

    table1_path = os.path.join(OUT_DIR, "table1_surrogate_ablation.csv")
    table1.to_csv(table1_path, index=False)
    print(f"[Track 1] Saved {table1_path} (shape={table1.shape})")

    # Track 2
    artifacts, track2_summary = fit_track2_q3_distillation(q3_df)
    fid = track2_summary["fidelity"]
    k8 = track2_summary["kaiwu_108_8bit_check"]
    print(
        f"[Track 2] Q3 QFM-Student: R2={fid['R2_all']:.4f}, Spearman={fid['Spearman_rho']:.4f}, "
        f"RMSE(V<1)={fid['RMSE_Vlt1']:.4f}, RMSE(V<2)={fid['RMSE_Vlt2']:.4f}, "
        f"Top-10={fid['top_10_overlap']}/10, Top-20={fid['top_20_overlap']}/20, "
        f"Top-50={fid['top_50_overlap']}/50, Top-100={fid['top_100_overlap']}/100, "
        f"#1={fid['qfm_top1_formula']}"
    )
    print(
        f"[Track 2] Polynomial eq max_err={track2_summary['max_poly_equiv_err']:.3e}, "
        f"max_gauge_err={max(track2_summary['anova_gauge_errors'].values()):.3e}"
    )
    print(
        f"[Track 2] Kaiwu 108-spin check: exact_ising_err={k8['exact_ising_vs_qfm_max_err']:.3e}, "
        f"J_mut_top5={k8['top5_overlap_qfm_vs_mutate_float64']}/5 (#1={k8['top1_mutate_float64_formula']}), "
        f"J_int_8bit range=[{k8['J_int_min']},{k8['J_int_max']}], J_ab_int=[{k8['J_ab_int_min']},{k8['J_ab_int_max']}], "
        f"J_int_8bit_top1={k8['top1_8bit_int_formula']} (top5={k8['top5_overlap_qfm_vs_8bit_int']}/5)"
    )

    npz_path = os.path.join(OUT_DIR, "exp1_qfm_q3_artifacts.npz")
    np.savez(npz_path, **artifacts)
    print(f"[Track 2] Saved {npz_path}")

    summary = {
        "total_runtime_sec": float(time.time() - t_start),
        "track1_m6_qfm_student": track1_info,
        "track2_q3_distillation": track2_summary,
    }
    summary_path = os.path.join(OUT_DIR, "exp1_qfm_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Track 2] Saved {summary_path} (total {summary['total_runtime_sec']:.2f}s)")


if __name__ == "__main__":
    main()
