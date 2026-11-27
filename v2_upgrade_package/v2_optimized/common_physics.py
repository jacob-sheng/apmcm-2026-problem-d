#!/usr/bin/env python3
"""
Common First-Principles Crystal-Chemistry Descriptors & Exact Band-Edge Physics
for 2026 APMCM Problem D (Castelli ABX3 Cubic Perovskites).

All elemental constants are textbook periodic-table quantities (CRC Handbook /
Pauling / Cordero / Shannon / NIST). Strictly NO external DFT labels are used.
Includes Bartel et al. (Science Advances 2019) tolerance factor tau and logistic
stability probability alongside Goldschmidt t, electron parity, closed-shell
charge neutrality, and the exact Butler-Ginley log-additive band-center model.
"""

import hashlib
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_data_dir(repo_root=None):
    """
    Dynamically resolve the official Castelli dataset directory ('date/').
    Search priority:
      1. Environment variable APMCM_DATA_DIR (if set and non-empty)
      2. <repo_root>/01_2026赛题与数据集/Problem_D_量子计算钙钛矿稀有目标搜索/date
      3. <repo_root>/2026 APMCM Problems/2026 APMCM Problem D/date
    Raises FileNotFoundError with searched paths and APMCM_DATA_DIR instructions if missing.
    """
    default_repo_root = os.path.dirname(BASE_DIR)
    root = os.path.abspath(repo_root) if repo_root is not None else default_repo_root

    searched = []
    env_dir = os.environ.get("APMCM_DATA_DIR", "").strip()
    if env_dir:
        cand_env = os.path.abspath(env_dir)
        searched.append(("APMCM_DATA_DIR", cand_env))
        if os.path.isdir(cand_env):
            return cand_env
        raise FileNotFoundError(
            f"APMCM_DATA_DIR is set to '{cand_env}', which does not exist or is not a directory. "
            "Set APMCM_DATA_DIR to the directory containing 02_castelli_train.csv and sha256_public.json, "
            "or unset APMCM_DATA_DIR to use repository auto-discovery."
        )

    candidates = [
        ("workspace_cn_dataset", os.path.join(root, "01_2026赛题与数据集", "Problem_D_量子计算钙钛矿稀有目标搜索", "date")),
        ("workspace_en_dataset", os.path.join(root, "2026 APMCM Problems", "2026 APMCM Problem D", "date")),
    ]
    if repo_root is None:
        candidates.append(
            ("package_legacy_dataset", os.path.join(BASE_DIR, "2026 APMCM Problems", "2026 APMCM Problem D", "date"))
        )

    for label, path in candidates:
        searched.append((label, path))
        if os.path.isdir(path):
            return os.path.abspath(path)

    searched_msg = "\n".join(f"  - [{lbl}] {p}" for lbl, p in searched)
    raise FileNotFoundError(
        "Could not locate the official Problem D dataset directory ('date/').\n"
        f"Searched paths:\n{searched_msg}\n"
        "Fix: set the environment variable APMCM_DATA_DIR to the folder containing "
        "'02_castelli_train.csv' and 'sha256_public.json' (e.g., export APMCM_DATA_DIR=/path/to/date)."
    )


def verify_dataset_checksums(data_dir=None):
    """
    Verify all official dataset files against sha256_public.json using BOTH:
      - raw-byte SHA-256 (exact file bytes on disk)
      - CRLF-normalized SHA-256 (accounting for LF <-> CRLF checkout conversion)
    Returns a summary dict with raw_bytes_matches, crlf_normalized_matches, total_files, and per-file details.
    """
    target_dir = os.path.abspath(data_dir) if data_dir is not None else DATA_DIR
    manifest_path = os.path.join(target_dir, "sha256_public.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing official checksum manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files = {}
    raw_matches = 0
    crlf_norm_matches = 0
    for fname, expected_sha in manifest.items():
        fpath = os.path.join(target_dir, fname)
        if not os.path.isfile(fpath):
            files[fname] = {
                "exists": False,
                "size_bytes": 0,
                "expected_sha256": expected_sha,
                "raw_sha256": None,
                "crlf_sha256": None,
                "lf_sha256": None,
                "raw_match": False,
                "crlf_exact_match": False,
                "crlf_normalized_match": False,
                "detected_eol": "MISSING",
            }
            continue

        with open(fpath, "rb") as fp:
            raw = fp.read()
        raw_sha = hashlib.sha256(raw).hexdigest()
        lf_bytes = raw.replace(b"\r\n", b"\n")
        crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
        lf_sha = hashlib.sha256(lf_bytes).hexdigest()
        crlf_sha = hashlib.sha256(crlf_bytes).hexdigest()

        has_crlf = b"\r\n" in raw
        has_bare_lf = b"\n" in raw.replace(b"\r\n", b"")
        if has_crlf and has_bare_lf:
            eol = "MIXED"
        elif has_crlf:
            eol = "CRLF"
        elif has_bare_lf:
            eol = "LF"
        else:
            eol = "NONE"

        raw_ok = (raw_sha == expected_sha)
        crlf_exact_ok = (crlf_sha == expected_sha)
        crlf_norm_ok = raw_ok or crlf_exact_ok or (lf_sha == expected_sha)
        if raw_ok:
            raw_matches += 1
        if crlf_norm_ok:
            crlf_norm_matches += 1

        files[fname] = {
            "exists": True,
            "size_bytes": len(raw),
            "expected_sha256": expected_sha,
            "raw_sha256": raw_sha,
            "crlf_sha256": crlf_sha,
            "lf_sha256": lf_sha,
            "raw_match": raw_ok,
            "crlf_exact_match": crlf_exact_ok,
            "crlf_normalized_match": crlf_norm_ok,
            "detected_eol": eol,
        }

    total = len(manifest)
    return {
        "data_dir": target_dir,
        "manifest_path": manifest_path,
        "total_files": total,
        "raw_bytes_matches": raw_matches,
        "crlf_normalized_matches": crlf_norm_matches,
        "all_raw_match": (raw_matches == total),
        "all_crlf_normalized_match": (crlf_norm_matches == total),
        "files": files,
    }


DATA_DIR = resolve_data_dir()

# (Z, mass, group, period, chi_pauling, ie1_eV, ea_eV, r_cov_A, r_ion6_A,
#  n_val, n_d, mendeleev, common_ox, closed_shell_ox)
ELEMENT_DATA = {
    "Li": (3,   6.941,   1, 2, 0.98,  5.392, 0.618, 1.28, 0.76,  1,  0,  1, (1,),          (1,)),
    "Be": (4,   9.012,   2, 2, 1.57,  9.323, 0.000, 0.96, 0.45,  2,  0, 67, (2,),          (2,)),
    "B":  (5,  10.811,  13, 2, 2.04,  8.298, 0.277, 0.84, 0.27,  3,  0, 72, (3,),          (3,)),
    "Na": (11, 22.990,   1, 3, 0.93,  5.139, 0.548, 1.66, 1.02,  1,  0,  2, (1,),          (1,)),
    "Mg": (12, 24.305,   2, 3, 1.31,  7.646, 0.000, 1.41, 0.72,  2,  0, 68, (2,),          (2,)),
    "Al": (13, 26.982,  13, 3, 1.61,  5.986, 0.433, 1.21, 0.54,  3,  0, 73, (3,),          (3,)),
    "Si": (14, 28.086,  14, 3, 1.90,  8.152, 1.389, 1.11, 0.40,  4,  0, 78, (4,),          (4,)),
    "K":  (19, 39.098,   1, 4, 0.82,  4.341, 0.501, 2.03, 1.38,  1,  0,  3, (1,),          (1,)),
    "Ca": (20, 40.078,   2, 4, 1.00,  6.113, 0.025, 1.76, 1.00,  2,  0,  7, (2,),          (2,)),
    "Sc": (21, 44.956,   3, 4, 1.36,  6.561, 0.188, 1.70, 0.75,  3,  1, 11, (3,),          (3,)),
    "Ti": (22, 47.867,   4, 4, 1.54,  6.828, 0.079, 1.60, 0.61,  4,  2, 43, (4, 3, 2),     (4,)),
    "V":  (23, 50.942,   5, 4, 1.63,  6.746, 0.525, 1.53, 0.54,  5,  3, 46, (5, 4, 3, 2),  (5,)),
    "Cr": (24, 51.996,   6, 4, 1.66,  6.767, 0.666, 1.39, 0.62,  6,  5, 49, (3, 6, 2),     (6,)),
    "Mn": (25, 54.938,   7, 4, 1.55,  7.434, 0.000, 1.39, 0.65,  7,  5, 52, (2, 3, 4, 7),  (7,)),
    "Fe": (26, 55.845,   8, 4, 1.83,  7.902, 0.151, 1.32, 0.65,  8,  6, 55, (2, 3),        ()),
    "Co": (27, 58.933,   9, 4, 1.88,  7.881, 0.662, 1.26, 0.65,  9,  7, 58, (2, 3),        ()),
    "Ni": (28, 58.693,  10, 4, 1.91,  7.640, 1.156, 1.24, 0.69, 10,  8, 61, (2, 3),        ()),
    "Cu": (29, 63.546,  11, 4, 1.90,  7.726, 1.235, 1.32, 0.73, 11, 10, 64, (1, 2),        (1,)),
    "Zn": (30, 65.380,  12, 4, 1.65,  9.394, 0.000, 1.22, 0.74, 12, 10, 69, (2,),          (2,)),
    "Ga": (31, 69.723,  13, 4, 1.81,  5.999, 0.430, 1.22, 0.62,  3, 10, 74, (3,),          (3,)),
    "Ge": (32, 72.630,  14, 4, 2.01,  7.899, 1.233, 1.20, 0.53,  4, 10, 79, (4, 2),        (4, 2)),
    "As": (33, 74.922,  15, 4, 2.18,  9.789, 0.814, 1.19, 0.46,  5, 10, 84, (5, 3),        (5, 3)),
    "Rb": (37, 85.468,   1, 5, 0.82,  4.177, 0.486, 2.20, 1.52,  1,  0,  4, (1,),          (1,)),
    "Sr": (38, 87.620,   2, 5, 0.95,  5.695, 0.052, 1.95, 1.18,  2,  0,  8, (2,),          (2,)),
    "Y":  (39, 88.906,   3, 5, 1.22,  6.217, 0.307, 1.90, 0.90,  3,  1, 12, (3,),          (3,)),
    "Zr": (40, 91.224,   4, 5, 1.33,  6.634, 0.426, 1.75, 0.72,  4,  2, 44, (4,),          (4,)),
    "Nb": (41, 92.906,   5, 5, 1.60,  6.759, 0.893, 1.64, 0.64,  5,  4, 47, (5, 4, 3),     (5,)),
    "Mo": (42, 95.950,   6, 5, 2.16,  7.092, 0.747, 1.54, 0.59,  6,  5, 50, (6, 5, 4),     (6,)),
    "Ru": (44, 101.07,   8, 5, 2.20,  7.361, 1.050, 1.46, 0.62,  8,  7, 56, (3, 4, 6),     ()),
    "Rh": (45, 102.91,   9, 5, 2.28,  7.459, 1.137, 1.42, 0.67,  9,  8, 59, (3, 4),        (3,)),
    "Pd": (46, 106.42,  10, 5, 2.20,  8.337, 0.562, 1.39, 0.86, 10, 10, 62, (2, 4),        (4,)),
    "Ag": (47, 107.87,  11, 5, 1.93,  7.576, 1.302, 1.45, 1.15, 11, 10, 65, (1, 2),        (1,)),
    "Cd": (48, 112.41,  12, 5, 1.69,  8.994, 0.000, 1.44, 0.95, 12, 10, 70, (2,),          (2,)),
    "In": (49, 114.82,  13, 5, 1.78,  5.786, 0.384, 1.42, 0.80,  3, 10, 75, (3, 1),        (3, 1)),
    "Sn": (50, 118.71,  14, 5, 1.96,  7.344, 1.112, 1.39, 0.69,  4, 10, 80, (4, 2),        (4, 2)),
    "Sb": (51, 121.76,  15, 5, 2.05,  8.608, 1.046, 1.39, 0.60,  5, 10, 85, (5, 3),        (5, 3)),
    "Te": (52, 127.60,  16, 5, 2.10,  9.010, 1.971, 1.38, 0.56,  6, 10, 90, (6, 4),        (6, 4)),
    "Cs": (55, 132.91,   1, 6, 0.79,  3.894, 0.472, 2.44, 1.67,  1,  0,  5, (1,),          (1,)),
    "Ba": (56, 137.33,   2, 6, 0.89,  5.212, 0.145, 2.15, 1.35,  2,  0,  9, (2,),          (2,)),
    "La": (57, 138.91,   3, 6, 1.10,  5.577, 0.470, 2.07, 1.03,  3,  1, 13, (3,),          (3,)),
    "Hf": (72, 178.49,   4, 6, 1.30,  6.825, 0.018, 1.75, 0.71,  4,  2, 45, (4,),          (4,)),
    "Ta": (73, 180.95,   5, 6, 1.50,  7.550, 0.322, 1.70, 0.64,  5,  3, 48, (5, 4),        (5,)),
    "W":  (74, 183.84,   6, 6, 2.36,  7.864, 0.816, 1.62, 0.60,  6,  4, 51, (6, 5, 4),     (6,)),
    "Re": (75, 186.21,   7, 6, 1.90,  7.833, 0.060, 1.51, 0.55,  7,  5, 54, (7, 6, 4),     (7,)),
    "Os": (76, 190.23,   8, 6, 2.20,  8.438, 1.078, 1.44, 0.63,  8,  6, 57, (4, 6, 8),     (8,)),
    "Ir": (77, 192.22,   9, 6, 2.20,  8.967, 1.564, 1.41, 0.68,  9,  7, 60, (3, 4),        (3,)),
    "Pt": (78, 195.08,  10, 6, 2.28,  8.959, 2.125, 1.36, 0.80, 10,  9, 63, (2, 4),        (4,)),
    "Au": (79, 196.97,  11, 6, 2.54,  9.226, 2.309, 1.36, 0.85, 11, 10, 66, (1, 3),        (1,)),
    "Hg": (80, 200.59,  12, 6, 2.00, 10.438, 0.000, 1.32, 1.02, 12, 10, 71, (2, 1),        (2,)),
    "Tl": (81, 204.38,  13, 6, 1.62,  6.108, 0.377, 1.45, 1.50,  3, 10, 76, (1, 3),        (1, 3)),
    "Pb": (82, 207.20,  14, 6, 2.33,  7.417, 0.364, 1.46, 1.19,  4, 10, 81, (2, 4),        (2, 4)),
    "Bi": (83, 208.98,  15, 6, 2.02,  7.286, 0.942, 1.48, 1.03,  5, 10, 86, (3, 5),        (3, 5)),
}

ANION_ATOMS = {
    "O": (15.999, 3.44, 13.618, 1.461, 1.40, -2, 6),
    "N": (14.007, 3.04, 14.534, 0.000, 1.46, -3, 5),
    "F": (18.998, 3.98, 17.423, 3.401, 1.33, -1, 7),
    "S": (32.065, 2.58, 10.360, 2.077, 1.84, -2, 6),
}

X_COMPOSITION = {
    "O3":  {"O": 3, "N": 0, "F": 0, "S": 0},
    "NO2": {"O": 2, "N": 1, "F": 0, "S": 0},
    "FO2": {"O": 2, "N": 0, "F": 1, "S": 0},
    "N2O": {"O": 1, "N": 2, "F": 0, "S": 0},
    "N3":  {"O": 0, "N": 3, "F": 0, "S": 0},
    "O2S": {"O": 2, "N": 0, "F": 0, "S": 1},
    "FNO": {"O": 1, "N": 1, "F": 1, "S": 0},
}


def _x_props(x_cls):
    comp = X_COMPOSITION[x_cls]
    mass = sum(c * ANION_ATOMS[k][0] for k, c in comp.items())
    chi = sum(c * ANION_ATOMS[k][1] for k, c in comp.items()) / 3.0
    ie1 = sum(c * ANION_ATOMS[k][2] for k, c in comp.items()) / 3.0
    ea = sum(c * ANION_ATOMS[k][3] for k, c in comp.items()) / 3.0
    r_x = sum(c * ANION_ATOMS[k][4] for k, c in comp.items()) / 3.0
    q_x = sum(c * ANION_ATOMS[k][5] for k, c in comp.items())
    n_val = sum(c * ANION_ATOMS[k][6] for k, c in comp.items())
    return {
        "X_mass": mass, "X_chi": chi, "X_ie1": ie1, "X_ea": ea,
        "X_r_ion": r_x, "X_q": q_x, "X_n_val": n_val,
        "X_n_O": comp["O"], "X_n_N": comp["N"], "X_n_F": comp["F"], "X_n_S": comp["S"],
        "X_chi_std": np.sqrt(sum(c * (ANION_ATOMS[k][1] - chi) ** 2 for k, c in comp.items()) / 3.0),
        "X_r_std": np.sqrt(sum(c * (ANION_ATOMS[k][4] - r_x) ** 2 for k, c in comp.items()) / 3.0),
    }


X_PROPS = {k: _x_props(k) for k in X_COMPOSITION}


def _min_charge_defect(ox_a, ox_b, q_x):
    if not ox_a or not ox_b:
        return 9.0, 0.0, 0.0
    best = 99.0
    best_pair = (ox_a[0], ox_b[0])
    for oa in ox_a:
        for ob in ox_b:
            d = abs(oa + ob + q_x)
            if d < best or (d == best and ob > best_pair[1]):
                best = d
                best_pair = (oa, ob)
    return float(best), float(best_pair[0]), float(best_pair[1])


def build_basic_v1_features(df):
    """Minimal elemental descriptor set matching teammate's v1 baseline roadmap."""
    rows = []
    for a, b, x, a_eq_b in zip(df["A"], df["B"], df["X"], df["a_eq_b"]):
        za, ma, ga, pa, chia, iea, eaa, rcova, riona, nva, nda, menda, _, _ = ELEMENT_DATA[a]
        zb, mb, gb, pb, chib, ieb, eab, rcovb, rionb, nvb, ndb, mendb, _, _ = ELEMENT_DATA[b]
        xp = X_PROPS[x]
        rx = xp["X_r_ion"]
        t_ion = (riona + rx) / (np.sqrt(2.0) * (rionb + rx))
        rows.append({
            "a_eq_b": int(a_eq_b),
            "A_Z": za, "A_group": ga, "A_period": pa, "A_chi": chia, "A_rion": riona, "A_nval": nva,
            "B_Z": zb, "B_group": gb, "B_period": pb, "B_chi": chib, "B_rion": rionb, "B_nval": nvb,
            "X_q": xp["X_q"], "X_chi": xp["X_chi"], "X_r_ion": rx,
            "X_n_O": xp["X_n_O"], "X_n_N": xp["X_n_N"], "X_n_F": xp["X_n_F"], "X_n_S": xp["X_n_S"],
            "t_ion": t_ion, "mu_oct": rionb / rx, "chi_diff_BA": chib - chia,
        })
    return pd.DataFrame(rows, index=df.index)


def build_v2_physics_features(df):
    """
    Rich v2 first-principles perovskite descriptors including Bartel (2019)
    SISSO tolerance factor tau + stability probability, Goldschmidt t,
    closed-shell charge neutrality, electron parity, and orbital descriptors.
    """
    rows = []
    for a, b, x, a_eq_b in zip(df["A"], df["B"], df["X"], df["a_eq_b"]):
        za, ma, ga, pa, chia, iea, eaa, rcova, riona, nva, nda, menda, oxa, csa = ELEMENT_DATA[a]
        zb, mb, gb, pb, chib, ieb, eab, rcovb, rionb, nvb, ndb, mendb, oxb, csb = ELEMENT_DATA[b]
        xp = X_PROPS[x]
        qx = xp["X_q"]
        rx = xp["X_r_ion"]
        chix = xp["X_chi"]

        chg_def, best_oa, best_ob = _min_charge_defect(oxa, oxb, qx)
        cs_def, cs_oa, cs_ob = _min_charge_defect(csa, csb, qx)

        z_x_tot = xp["X_n_O"] * 8 + xp["X_n_N"] * 7 + xp["X_n_F"] * 9 + xp["X_n_S"] * 16
        z_tot = za + zb + z_x_tot
        z_parity = z_tot % 2
        val_tot = nva + nvb + xp["X_n_val"]
        val_parity = val_tot % 2

        t_ion = (riona + rx) / (np.sqrt(2.0) * (rionb + rx))
        t_cov = (rcova + rx) / (np.sqrt(2.0) * (rcovb + rx))
        mu_oct = rionb / rx
        mu_a = riona / rx
        ratio_ab = riona / (rionb + 1e-6)

        # Bartel et al. (Science Advances 2019) tolerance factor tau & stability prob
        na_ox = max(1.0, best_oa)
        bartel_valid = int(ratio_ab > 1.05)
        if ratio_ab > 1.05:
            tau_bartel = (rx / rionb) - na_ox * (na_ox - ratio_ab / np.log(ratio_ab))
            # Bartel threshold tau < 4.18 predicts stable perovskite (logistic calibration)
            bartel_prob = float(1.0 / (1.0 + np.exp(1.25 * (tau_bartel - 4.18))))
        else:
            tau_bartel = 15.0
            bartel_prob = 0.0

        chim_a = 0.5 * (iea + eaa)
        chim_b = 0.5 * (ieb + eab)
        chim_x = 0.5 * (xp["X_ie1"] + xp["X_ea"])

        # Pauling ionicity index I = 1 - exp(-0.25*(chi_X - chi_M)^2)
        ionicity_bx = 1.0 - np.exp(-0.25 * (chix - chib) ** 2)
        ionicity_ax = 1.0 - np.exp(-0.25 * (chix - chia) ** 2)
        # d-shell configuration indicators (d0 / d10 favor wide-gap semiconductors)
        b_is_d0_d10 = int((gb in (3, 4, 5, 6, 12, 13, 14, 15)) and (len(csb) > 0))
        a_is_d0_d10 = int((ga in (1, 2, 3, 12, 13, 14, 15)) and (len(csa) > 0))

        rows.append({
            "a_eq_b": int(a_eq_b),
            "A_Z": za, "A_mass": ma, "A_group": ga, "A_period": pa,
            "A_chi": chia, "A_ie1": iea, "A_ea": eaa, "A_chim": chim_a,
            "A_rcov": rcova, "A_rion": riona, "A_nval": nva, "A_nd": nda, "A_mend": menda,
            "A_has_cs": int(len(csa) > 0), "A_max_ox": max(oxa), "A_min_ox": min(oxa),
            "A_is_d_block": int(3 <= ga <= 12), "A_is_alkali_earth": int(ga in (1, 2)),
            "A_is_d0_d10": a_is_d0_d10,
            "B_Z": zb, "B_mass": mb, "B_group": gb, "B_period": pb,
            "B_chi": chib, "B_ie1": ieb, "B_ea": eab, "B_chim": chim_b,
            "B_rcov": rcovb, "B_rion": rionb, "B_nval": nvb, "B_nd": ndb, "B_mend": mendb,
            "B_has_cs": int(len(csb) > 0), "B_max_ox": max(oxb), "B_min_ox": min(oxb),
            "B_is_d_block": int(3 <= gb <= 12), "B_is_alkali_earth": int(gb in (1, 2)),
            "B_is_d0_d10": b_is_d0_d10,
            **xp,
            "Z_tot": z_tot, "Z_parity": z_parity,
            "val_tot": val_tot, "val_parity": val_parity,
            "chg_defect": chg_def, "is_chg_balanced": int(chg_def == 0),
            "best_ox_A": best_oa, "best_ox_B": best_ob, "best_ox_diff": best_ob - best_oa,
            "cs_defect": cs_def, "is_cs_balanced": int(cs_def == 0),
            "cs_ox_A": cs_oa, "cs_ox_B": cs_ob,
            "t_ion": t_ion, "t_ion_dev": abs(t_ion - 1.0),
            "t_cov": t_cov, "mu_oct": mu_oct, "mu_a": mu_a,
            "r_ratio_AB": ratio_ab, "r_diff_AB": riona - rionb, "r_sum_AB": riona + rionb,
            "rcov_diff_AB": rcova - rcovb, "rcov_sum_AB": rcova + rcovb,
            "tau_bartel": float(np.clip(tau_bartel, -20.0, 20.0)),
            "bartel_valid": bartel_valid,
            "bartel_prob": bartel_prob,
            "ionicity_ax": ionicity_ax, "ionicity_bx": ionicity_bx,
            "ionicity_diff": ionicity_ax - ionicity_bx,
            "chi_diff_XA": chix - chia, "chi_diff_XB": chix - chib,
            "chi_diff_BA": chib - chia, "chi_abs_AB": abs(chib - chia),
            "chi_mean_AB": 0.5 * (chia + chib),
            "ie1_diff_BA": ieb - iea, "ie1_sum_AB": iea + ieb,
            "ea_diff_BA": eab - eaa,
            "mend_diff_AB": menda - mendb, "mend_sum_AB": menda + mendb,
            "nd_sum_AB": nda + ndb, "nd_diff_BA": ndb - nda,
            "group_sum_AB": ga + gb, "group_diff_BA": gb - ga,
            "period_sum_AB": pa + pb, "period_diff_BA": pb - pa,
            "mass_tot": ma + mb + xp["X_mass"],
            "chim_geom_proxy": (max(0.1, chim_a) * max(0.1, chim_b) * (chim_x ** 3)) ** 0.2,
            "a_est_lattice": 2.0 * (rionb + rx),
        })
    return pd.DataFrame(rows, index=df.index)


def fit_band_center_model(df_train):
    """Exact Butler-Ginley log-additive band center: ln((VBM+CBM)/2) = w0+wA+wB+wX."""
    X_oh = pd.get_dummies(df_train[["A", "B", "X"]], dtype=float)
    y_log = np.log(0.5 * (df_train["vbm"].values + df_train["cbm"].values))
    model = Ridge(alpha=1e-10, fit_intercept=True)
    model.fit(X_oh, y_log)
    return model, list(X_oh.columns)


def predict_band_center(model, cols, df):
    X_oh = pd.get_dummies(df[["A", "B", "X"]], dtype=float).reindex(columns=cols, fill_value=0.0)
    return np.exp(model.predict(X_oh))


def v_band_terms(gap, ec):
    vbm = ec + 0.5 * gap
    cbm = ec - 0.5 * gap
    v_gl = np.maximum(0.0, (1.4 - gap) / 0.5) ** 2
    v_gh = np.maximum(0.0, (gap - 3.1) / 0.5) ** 2
    v_vbm = np.maximum(0.0, (5.73 - vbm) / 0.5) ** 2
    v_cbm = np.maximum(0.0, (cbm - 4.50) / 0.5) ** 2
    return v_gl, v_gh, v_vbm, v_cbm


def exact_expected_hinge_sq(mu, sigma, threshold, sign=1.0, scale=0.5):
    """Exact E[max(0, sign*(Y - threshold)/scale)^2] for Y ~ N(mu, sigma^2)."""
    m = sign * (mu - threshold) / scale
    s = np.maximum(sigma / scale, 1e-6)
    z = m / s
    return (m ** 2 + s ** 2) * norm.cdf(z) + m * s * norm.pdf(z)


def compute_band_lower_bound(ec):
    g_min_zero = np.maximum.reduce([np.full_like(ec, 1.4), 2.0 * (5.73 - ec), 2.0 * (ec - 4.50)])
    feasible_window = np.maximum(0.0, 3.1 - g_min_zero)
    g_grid = np.linspace(0.0, 6.0, 121)[None, :]
    vgl, vgh, vvbm, vcbm = v_band_terms(g_grid, ec[:, None])
    v_min = np.min(vgl + vgh + vvbm + vcbm, axis=1)
    v_metal = np.sum(v_band_terms(np.zeros_like(ec), ec), axis=0)
    return g_min_zero, feasible_window, v_min, v_metal


def eval_surrogate_metrics(y_true, y_pred, name=""):
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    sp = float(spearmanr(y_true, y_pred).statistic)
    order = np.argsort(y_pred)
    n = len(y_true)
    mask05 = y_true < 0.5
    mask10 = y_true < 1.0
    n05 = max(1, int(mask05.sum()))
    n10 = max(1, int(mask10.sum()))
    rec50_05 = float(mask05[order[:50]].sum() / n05)
    rec75_05 = float(mask05[order[:75]].sum() / n05)
    rec100_05 = float(mask05[order[:100]].sum() / n05)
    rec100_10 = float(mask10[order[:100]].sum() / n10)
    ef50_05 = float(rec50_05 / (50.0 / n))
    ef100_10 = float(rec100_10 / (100.0 / n))
    top50_mean_v = float(np.mean(y_true[order[:50]]))
    return {
        "model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Spearman": round(sp, 4),
        "Rec@50(V<0.5)": round(rec50_05, 4),
        "Rec@75(V<0.5)": round(rec75_05, 4),
        "Rec@100(V<0.5)": round(rec100_05, 4),
        "EF@50(V<0.5)": round(ef50_05, 2),
        "Rec@100(V<1.0)": round(rec100_10, 4),
        "EF@100(V<1.0)": round(ef100_10, 2),
        "Top50_Mean_V": round(top50_mean_v, 4),
    }


def safe_ising_bit_width(ising_mat):
    import kaiwu as kw
    res = kw.calculate_ising_matrix_bit_width(ising_mat)
    val = res["precision"] if isinstance(res, dict) else res
    return int(val) if np.isfinite(val) else 999


def sort_ising_solutions(sols_array, ising_mat):
    """Sort Kaiwu solver spin rows ascending by exact Ising Hamiltonian energy E(s) = -s^T h s."""
    arr = np.asarray(sols_array, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    energies = -np.einsum("ij,jk,ik->i", arr, ising_mat, arr)
    order = np.argsort(energies)
    return arr[order]
