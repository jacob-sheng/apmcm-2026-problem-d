'''Periodic-table crystal-chemistry and leak-free OOF marginal features for ABX3 perovskites.'''

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# (Z, period, group, atomic_mass, pauling_chi, r_cov_A, r_eff_A, IE1_eV, EA_eV, n_val, n_d, pettifor)
ELEMENT_DATA = {
    "Li": (3, 2, 1, 6.94, 0.98, 1.28, 0.92, 5.392, 0.618, 1, 0, 12),
    "Be": (4, 2, 2, 9.012, 1.57, 0.96, 0.45, 9.323, -0.50, 2, 0, 77),
    "B":  (5, 2, 13, 10.81, 2.04, 0.84, 0.27, 8.298, 0.277, 3, 0, 86),
    "Na": (11, 3, 1, 22.99, 0.93, 1.66, 1.18, 5.139, 0.548, 1, 0, 11),
    "Mg": (12, 3, 2, 24.305, 1.31, 1.41, 0.89, 7.646, -0.40, 2, 0, 73),
    "Al": (13, 3, 13, 26.982, 1.61, 1.21, 0.535, 5.986, 0.433, 3, 0, 80),
    "Si": (14, 3, 14, 28.085, 1.90, 1.11, 0.40, 8.152, 1.390, 4, 0, 85),
    "K":  (19, 4, 1, 39.098, 0.82, 2.03, 1.51, 4.341, 0.501, 1, 0, 10),
    "Ca": (20, 4, 2, 40.078, 1.00, 1.76, 1.12, 6.113, 0.025, 2, 0, 16),
    "Sc": (21, 4, 3, 44.956, 1.36, 1.70, 0.87, 6.561, 0.188, 3, 1, 19),
    "Ti": (22, 4, 4, 47.867, 1.54, 1.60, 0.74, 6.828, 0.079, 4, 2, 51),
    "V":  (23, 4, 5, 50.942, 1.63, 1.53, 0.68, 6.746, 0.525, 5, 3, 54),
    "Cr": (24, 4, 6, 51.996, 1.66, 1.39, 0.62, 6.767, 0.666, 6, 5, 57),
    "Mn": (25, 4, 7, 54.938, 1.55, 1.39, 0.67, 7.434, -0.50, 7, 5, 60),
    "Fe": (26, 4, 8, 55.845, 1.83, 1.32, 0.65, 7.902, 0.151, 8, 6, 61),
    "Co": (27, 4, 9, 58.933, 1.88, 1.26, 0.61, 7.881, 0.663, 9, 7, 64),
    "Ni": (28, 4, 10, 58.693, 1.91, 1.24, 0.69, 7.640, 1.156, 10, 8, 67),
    "Cu": (29, 4, 11, 63.546, 1.90, 1.32, 0.73, 7.726, 1.236, 11, 10, 72),
    "Zn": (30, 4, 12, 65.38, 1.65, 1.22, 0.74, 9.394, -0.60, 12, 10, 76),
    "Ga": (31, 4, 13, 69.723, 1.81, 1.22, 0.62, 5.999, 0.430, 3, 10, 81),
    "Ge": (32, 4, 14, 72.630, 2.01, 1.20, 0.53, 7.899, 1.233, 4, 10, 84),
    "As": (33, 4, 15, 74.922, 2.18, 1.19, 0.46, 9.789, 0.805, 5, 10, 88),
    "Rb": (37, 5, 1, 85.468, 0.82, 2.20, 1.61, 4.177, 0.486, 1, 0, 9),
    "Sr": (38, 5, 2, 87.62, 0.95, 1.95, 1.26, 5.695, 0.052, 2, 0, 15),
    "Y":  (39, 5, 3, 88.906, 1.22, 1.90, 1.02, 6.217, 0.307, 3, 1, 25),
    "Zr": (40, 5, 4, 91.224, 1.33, 1.75, 0.84, 6.634, 0.426, 4, 2, 49),
    "Nb": (41, 5, 5, 92.906, 1.60, 1.64, 0.74, 6.759, 0.893, 5, 4, 53),
    "Mo": (42, 5, 6, 95.95, 2.16, 1.54, 0.65, 7.092, 0.747, 6, 5, 56),
    "Ru": (44, 5, 8, 101.07, 2.20, 1.46, 0.62, 7.361, 1.050, 8, 7, 62),
    "Rh": (45, 5, 9, 102.91, 2.28, 1.42, 0.60, 7.459, 1.137, 9, 8, 65),
    "Pd": (46, 5, 10, 106.42, 2.20, 1.39, 0.615, 8.337, 0.562, 10, 10, 69),
    "Ag": (47, 5, 11, 107.87, 1.93, 1.45, 1.15, 7.576, 1.302, 11, 10, 71),
    "Cd": (48, 5, 12, 112.41, 1.69, 1.44, 0.95, 8.994, -0.70, 12, 10, 75),
    "In": (49, 5, 13, 114.82, 1.78, 1.42, 0.80, 5.786, 0.300, 3, 10, 79),
    "Sn": (50, 5, 14, 118.71, 1.96, 1.39, 0.69, 7.344, 1.112, 4, 10, 83),
    "Sb": (51, 5, 15, 121.76, 2.05, 1.39, 0.76, 8.608, 1.047, 5, 10, 87),
    "Te": (52, 5, 16, 127.60, 2.10, 1.38, 0.97, 9.010, 1.971, 6, 10, 91),
    "Cs": (55, 6, 1, 132.91, 0.79, 2.44, 1.74, 3.894, 0.472, 1, 0, 8),
    "Ba": (56, 6, 2, 137.33, 0.89, 2.15, 1.42, 5.212, 0.145, 2, 0, 14),
    "La": (57, 6, 3, 138.91, 1.10, 2.07, 1.16, 5.577, 0.470, 3, 1, 33),
    "Hf": (72, 6, 4, 178.49, 1.30, 1.75, 0.83, 6.825, 0.018, 4, 2, 50),
    "Ta": (73, 6, 5, 180.95, 1.50, 1.70, 0.74, 7.550, 0.322, 5, 3, 52),
    "W":  (74, 6, 6, 183.84, 2.36, 1.62, 0.66, 7.864, 0.816, 6, 4, 55),
    "Re": (75, 6, 7, 186.21, 1.90, 1.51, 0.63, 7.833, 0.060, 7, 5, 58),
    "Os": (76, 6, 8, 190.23, 2.20, 1.44, 0.63, 8.438, 1.078, 8, 6, 63),
    "Ir": (77, 6, 9, 192.22, 2.20, 1.41, 0.68, 8.967, 1.564, 9, 7, 66),
    "Pt": (78, 6, 10, 195.08, 2.28, 1.36, 0.625, 8.959, 2.128, 10, 9, 68),
    "Au": (79, 6, 11, 196.97, 2.54, 1.36, 0.85, 9.226, 2.309, 11, 10, 70),
    "Hg": (80, 6, 12, 200.59, 2.00, 1.32, 1.02, 10.438, -0.50, 12, 10, 74),
    "Tl": (81, 6, 13, 204.38, 1.62, 1.45, 1.50, 6.108, 0.377, 3, 10, 78),
    "Pb": (82, 6, 14, 207.2, 1.87, 1.46, 1.19, 7.417, 0.357, 4, 10, 82),
    "Bi": (83, 6, 15, 208.98, 2.02, 1.48, 1.03, 7.286, 0.942, 5, 10, 86),
}

ELEMENTS_52 = sorted(ELEMENT_DATA.keys())

# (n_O, n_N, n_F, n_S, Q_X_formal, chi_X_avg, r_X_eff, Z_X_total, val_X_total)
X_DATA = {
    "O3":  (3, 0, 0, 0, 6, 3.440, 1.40, 24, 18),
    "NO2": (2, 1, 0, 0, 7, 3.307, 1.42, 23, 17),
    "FO2": (2, 0, 1, 0, 5, 3.620, 1.37, 25, 19),
    "N2O": (1, 2, 0, 0, 8, 3.173, 1.44, 22, 16),
    "N3":  (0, 3, 0, 0, 9, 3.040, 1.46, 21, 15),
    "O2S": (2, 0, 0, 1, 6, 3.153, 1.54, 32, 18),
    "FNO": (1, 1, 1, 0, 6, 3.487, 1.39, 24, 18),
}
X_CLASSES_7 = sorted(X_DATA.keys())


def build_intrinsic_features(df: pd.DataFrame) -> pd.DataFrame:
    '''Build deterministic crystal-chemistry features from (A, B, X, a_eq_b).'''
    elem_cols = ["Z", "period", "group", "mass", "chi", "r_cov", "r_eff", "IE1", "EA", "n_val", "n_d", "pettifor"]
    e_df = pd.DataFrame.from_dict(ELEMENT_DATA, orient="index", columns=elem_cols)

    a_feat = e_df.loc[df["A"]].reset_index(drop=True).add_prefix("A_")
    b_feat = e_df.loc[df["B"]].reset_index(drop=True).add_prefix("B_")

    x_cols = ["n_O", "n_N", "n_F", "n_S", "Q_X", "chi_X", "r_X", "Z_X", "val_X"]
    x_df = pd.DataFrame.from_dict(X_DATA, orient="index", columns=x_cols)
    x_feat = x_df.loc[df["X"]].reset_index(drop=True)

    out = pd.concat([a_feat, b_feat, x_feat], axis=1)
    out["a_eq_b"] = df["a_eq_b"].astype(float).values

    elem_to_idx = {el: i for i, el in enumerate(ELEMENTS_52)}
    x_to_idx = {xc: i for i, xc in enumerate(X_CLASSES_7)}
    out["A_idx"] = df["A"].map(elem_to_idx).values
    out["B_idx"] = df["B"].map(elem_to_idx).values
    out["X_idx"] = df["X"].map(x_to_idx).values

    out["goldschmidt_t"] = (out["A_r_eff"] + out["r_X"]) / (np.sqrt(2.0) * (out["B_r_eff"] + out["r_X"]))
    out["goldschmidt_t_cov"] = (out["A_r_cov"] + out["r_X"]) / (np.sqrt(2.0) * (out["B_r_cov"] + out["r_X"]))
    out["octahedral_mu"] = out["B_r_eff"] / out["r_X"]
    out["A_over_B_r"] = out["A_r_eff"] / out["B_r_eff"]
    out["r_sum"] = out["A_r_eff"] + out["B_r_eff"] + 2.0 * out["r_X"]
    out["r_diff_AB"] = out["A_r_eff"] - out["B_r_eff"]
    out["r_cov_diff_AB"] = out["A_r_cov"] - out["B_r_cov"]

    out["valence_AB_sum"] = out["A_n_val"] + out["B_n_val"]
    out["valence_mismatch"] = out["valence_AB_sum"] - out["Q_X"]
    out["abs_valence_mismatch"] = np.abs(out["valence_mismatch"])
    a_ox = np.where(out["A_group"] <= 12, np.minimum(out["A_group"], 6), out["A_group"] - 10)
    b_ox = np.where(out["B_group"] <= 12, np.minimum(out["B_group"], 6), out["B_group"] - 10)
    out["ox_sum"] = a_ox + b_ox
    out["ox_mismatch"] = out["ox_sum"] - out["Q_X"]
    out["abs_ox_mismatch"] = np.abs(out["ox_mismatch"])

    out["Z_total"] = out["A_Z"] + out["B_Z"] + out["Z_X"]
    out["electron_parity"] = out["Z_total"] % 2
    out["tot_val_electrons"] = out["A_n_val"] + out["B_n_val"] + out["val_X"]
    out["val_electron_parity"] = out["tot_val_electrons"] % 2

    out["chi_diff_AB"] = out["A_chi"] - out["B_chi"]
    out["chi_mean_AB"] = 0.5 * (out["A_chi"] + out["B_chi"])
    out["chi_X_minus_AB"] = out["chi_X"] - out["chi_mean_AB"]
    out["chi_X_minus_B"] = out["chi_X"] - out["B_chi"]
    out["IE1_diff_AB"] = out["A_IE1"] - out["B_IE1"]
    out["IE1_mean_AB"] = 0.5 * (out["A_IE1"] + out["B_IE1"])
    out["EA_diff_AB"] = out["A_EA"] - out["B_EA"]
    out["d_sum_AB"] = out["A_n_d"] + out["B_n_d"]
    out["pettifor_diff"] = out["A_pettifor"] - out["B_pettifor"]
    out["mass_total"] = out["A_mass"] + out["B_mass"]

    return out


MARGINAL_TARGETS = ["e_form", "is_semi", "gap", "vbm", "cbm", "a_lat", "volume", "V", "log1pV"]


def _compute_group_stats(ref_df: pd.DataFrame) -> dict:
    '''Compute marginal target means/stds from ref_df only.'''
    tmp = ref_df[["A", "B", "X", "e_form", "gap", "vbm", "cbm", "a_lat", "volume", "V"]].copy()
    tmp["is_semi"] = (tmp["gap"] > 0.0).astype(float)
    tmp["log1pV"] = np.log1p(tmp["V"].values)

    stats = {"global": {c: float(tmp[c].mean()) for c in MARGINAL_TARGETS}}
    for key_cols, tag in [
        (["A", "X"], "AX"),
        (["B", "X"], "BX"),
        (["A"], "A"),
        (["B"], "B"),
        (["X"], "X"),
    ]:
        agg = tmp.groupby(key_cols)[MARGINAL_TARGETS].agg(["mean", "std"])
        agg.columns = [f"marg_{tag}_{col}_{fn}" for col, fn in agg.columns]
        stats[tag] = agg
    return stats


def _apply_group_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    '''Map precomputed marginal stats onto df without touching df own labels.'''
    parts = []
    for key_cols, tag in [
        (["A", "X"], "AX"),
        (["B", "X"], "BX"),
        (["A"], "A"),
        (["B"], "B"),
        (["X"], "X"),
    ]:
        merged = df[key_cols].merge(stats[tag], on=key_cols, how="left")
        stat_cols = [c for c in merged.columns if c.startswith(f"marg_{tag}_")]
        sub = merged[stat_cols].copy()
        for col in stat_cols:
            base = col.replace(f"marg_{tag}_", "").rsplit("_", 1)[0]
            fill_val = stats["global"][base] if col.endswith("_mean") else 0.0
            sub[col] = sub[col].fillna(fill_val)
        parts.append(sub)
    return pd.concat(parts, axis=1)


def build_oof_marginal_features(train_df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    '''Compute leak-free GroupKFold (grouped by A|B pair) marginal stats for train_df.'''
    groups = (train_df["A"] + "_" + train_df["B"]).values
    gkf = GroupKFold(n_splits=n_splits)
    oof_df = None
    for tr_idx, va_idx in gkf.split(train_df, groups=groups):
        stats = _compute_group_stats(train_df.iloc[tr_idx])
        part = _apply_group_stats(train_df.iloc[va_idx], stats)
        if oof_df is None:
            oof_df = pd.DataFrame(index=train_df.index, columns=part.columns, dtype=float)
        oof_df.iloc[va_idx] = part.values
    return oof_df


def build_test_marginal_features(ref_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    '''Compute marginal stats from ref_df and map onto target_df.'''
    stats = _compute_group_stats(ref_df)
    return _apply_group_stats(target_df, stats)
