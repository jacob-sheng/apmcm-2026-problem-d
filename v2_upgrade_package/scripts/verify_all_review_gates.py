#!/usr/bin/env python3
"""
Independent Read-Only Review Gate Verifier (Section 9.3 & F01-F29).

Modes:
  --code-only   Verify dataset checksums, Q1 isolation/baselines, conformal splits,
                QUBO/HOBO/ANOVA global ground-state certifications, closed-loop &
                sensitivity suites, SPQC-550 hardware --verify-only & credit margin,
                PII/path leak scan, Track A archive consistency, and figures/schematics.
                Skips paper/main.typ & REPORT.md prose/numeric sync checks.
  (default)     Full mode: runs all --code-only checks PLUS paper/main.typ and
                v2_optimized/REPORT.md Section 8.6 phrasing & numeric sync gates.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_ROOT = os.path.dirname(PKG_DIR)
V2_DIR = os.path.join(PKG_DIR, "v2_optimized")
RES_DIR = os.path.join(V2_DIR, "results")
FIG_DIR = os.path.join(V2_DIR, "figures")
SCHEM_DIR = os.path.join(PKG_DIR, "paper", "schematics")
PAPER_TYP = os.path.join(PKG_DIR, "paper", "main.typ")
REPORT_MD = os.path.join(V2_DIR, "REPORT.md")

TABLE1_CSV = os.path.join(
    REPO_ROOT,
    "01_2026赛题与数据集",
    "Problem_D_量子计算钙钛矿稀有目标搜索",
    "project",
    "outputs",
    "table1_surrogate_ablation.csv",
)
TABLE4_CSV = os.path.join(
    REPO_ROOT,
    "01_2026赛题与数据集",
    "Problem_D_量子计算钙钛矿稀有目标搜索",
    "project",
    "outputs",
    "table4_spqc550_hardware_vs_classical.csv",
)

if V2_DIR not in sys.path:
    sys.path.insert(0, V2_DIR)

from common_physics import resolve_data_dir, verify_dataset_checksums

PATH_LEAK_RE = re.compile(
    r"([A-Za-z]:[/\\]+Users[/\\]+|[A-Za-z]:[/\\]+Codex|/Users/|/home/|/mnt/[a-zA-Z]/|@(?:[\w.-]+\.)?(edu\.cn|qq\.com|163\.com|gmail\.com|outlook\.com|126\.com|foxmail\.com))",
    re.I,
)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class GateRunner:
    def __init__(self):
        self.passed = []
        self.failed = []

    def check(self, name, fn):
        try:
            detail = fn()
            self.passed.append((name, detail or "OK"))
            print(f"[PASS] {name}: {detail or 'OK'}")
        except Exception as exc:
            self.failed.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")


def gate_dataset_checksums():
    data_dir = resolve_data_dir()
    csum = verify_dataset_checksums(data_dir)
    assert csum["total_files"] == 9, f"Expected 9 manifest files, got {csum['total_files']}"
    assert csum["crlf_normalized_matches"] == 9, (
        f"Expected 9/9 CRLF-normalized matches, got {csum['crlf_normalized_matches']}/9"
    )
    assert csum["all_crlf_normalized_match"] is True
    return (
        f"9/9 CRLF-normalized SHA-256 matched "
        f"(raw={csum['raw_bytes_matches']}/9, dir={os.path.basename(data_dir)})"
    )


def gate_q1_diagnostics():
    diag = _load_json(os.path.join(RES_DIR, "v2_q1_diagnostics.json"))

    iso = diag["outer_fold_isolation_test"]
    assert iso["passed"] is True, "outer_fold_isolation_test.passed is not True"
    assert iso["group_overlap"] == 0, f"group_overlap={iso['group_overlap']}"
    assert iso["max_feature_delta"] < 1e-12, f"max_feature_delta={iso['max_feature_delta']}"
    assert iso["max_pred_delta"] < 1e-12, f"max_pred_delta={iso['max_pred_delta']}"

    # Cross-check M4 & M5 against official table1_surrogate_ablation.csv
    assert os.path.isfile(TABLE1_CSV), f"Missing {TABLE1_CSV}"
    t1_by_model = {}
    with open(TABLE1_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t1_by_model[row["model"]] = row

    diag_by_model = {r["model"]: r for r in diag["baseline_comparison"]["models"]}
    for mkey in ("M4_PIMT_Physics_Only", "M5_PIMT_Hybrid_Ensemble"):
        assert mkey in t1_by_model, f"{mkey} missing from table1_surrogate_ablation.csv"
        assert mkey in diag_by_model, f"{mkey} missing from v2_q1_diagnostics.json"
        for json_k, csv_k in [("RMSE", "RMSE"), ("MAE", "MAE"), ("R2", "R2"), ("Spearman", "Spearman_rho")]:
            v_json = float(diag_by_model[mkey][json_k])
            v_csv = float(t1_by_model[mkey][csv_k])
            assert abs(v_json - v_csv) < 1e-3, f"{mkey}.{json_k}: json={v_json} vs csv={v_csv}"

    # Verify exported CSVs exist, row counts, and SHA-256
    exp_sha = diag["exported_csv_sha256"]
    for fn in ("v2_holdout_04_train02only.csv", "v2_holdout_04.csv"):
        p = os.path.join(RES_DIR, fn)
        assert os.path.isfile(p), f"Missing {fn}"
        with open(p, "r", encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1
        assert n_rows == 3703, f"{fn} row count {n_rows} != 3703"

    for fn, expected_hash in exp_sha.items():
        p = os.path.join(RES_DIR, fn)
        assert os.path.isfile(p), f"Missing exported CSV {fn}"
        actual_hash = _sha256_file(p)
        assert actual_hash == expected_hash, f"SHA-256 mismatch for {fn}: {actual_hash} != {expected_hash}"

    ab = diag["a_eq_b_sensitivity"]
    for k in ("holdout_04_02plus03", "holdout_04_02only", "q2_full_07", "q3_full_08"):
        assert k in ab, f"Missing a_eq_b_sensitivity.{k}"

    return (
        f"isolation max_pred_delta={iso['max_pred_delta']:.2e}, "
        f"M4/M5 synced, {len(exp_sha)}/{len(exp_sha)} CSV SHA-256 verified, "
        f"04 holdouts=3703 rows"
    )


def gate_conformal_and_physics():
    conf = _load_json(os.path.join(RES_DIR, "v2_deep_physics_conformal.json"))
    mc = conf["mondrian_conformal_03val"]
    assert mc["n_splits"] == 200, f"n_splits={mc['n_splits']}"
    assert mc["group_overlap"] == 0, f"group_overlap={mc['group_overlap']}"

    res_corr = conf["semiconductor_residual_correlation_03val"]
    assert res_corr["n_semiconductors"] > 0
    assert math.isfinite(res_corr["pearson_r"]) and math.isfinite(res_corr["spearman_rho"])

    hy = conf["holdout_04_expected_yield"]
    scope_txt = hy["definition_scope"].lower()
    assert "not unseen true-positive recall" in scope_txt, "holdout_04_expected_yield scope note missing recall disclaimer"
    assert 0 < hy["sum_P_hit_top30"] < hy["sum_P_hit_all"]

    return (
        f"200 splits (group_overlap=0), semi residual r={res_corr['pearson_r']:.4f}, "
        f"sum(P_hit)={hy['sum_P_hit_all']:.4f} scoped"
    )


def gate_qubo_certification():
    qubo = _load_json(os.path.join(RES_DIR, "v2_qubo_results.json"))
    certs = qubo["certification_summary"]
    assert len(certs) == 8, f"Expected 8 certified matrices, got {len(certs)}"

    for c in certs:
        mid = c["matrix_id"]
        assert c["illegal_ground_count"] == 0, f"{mid}: illegal_ground_count={c['illegal_ground_count']}"
        assert c["illegal_aux_tie_count"] == 0, f"{mid}: illegal_aux_tie_count={c['illegal_aux_tie_count']}"
        assert c["legal_ground_count"] >= 1, f"{mid}: legal_ground_count={c['legal_ground_count']}"
        assert c["n_primary_states_total"] > 0, f"{mid}: n_primary_states_total={c['n_primary_states_total']}"
        assert abs(c["declared_pool_min_V_hat"] - 0.252406) < 1e-4, f"{mid}: min V_hat={c['declared_pool_min_V_hat']}"
        assert c["continuous_regret_top1"] >= 0.0

    for k in ("Q3_full_08_Model3B_CompactANOVA", "Q3_blind_12_Model3B_CompactANOVA"):
        lvl = qubo["Q3"][k]["distinct_legal_energy_levels"]
        assert lvl >= 6, f"{k}: distinct_legal_energy_levels={lvl} < 6"

    ca = qubo["compression_attribution"]
    assert len(ca) >= 4, f"compression_attribution rows={len(ca)}"
    enc_joined = " | ".join(r["encoding"] for r in ca)
    for req in ("A-B Rosenberg", "A-X Rosenberg", "Model 3A-HOBO", "Model 3A-Comp"):
        assert req in enc_joined, f"Missing '{req}' in compression_attribution"

    cap_ret = qubo["c_cap_retention"]
    assert len(cap_ret) > 0, "c_cap_retention empty"
    for r in cap_ret:
        assert "top10_retained_lt_cap" in r and "top25_retained_lt_cap" in r

    aed = qubo["anova_error_decomposition"]
    for scope in ("Q3_full_08", "Q3_blind_12"):
        assert scope in aed
        for layer in ("layer_1_surrogate_vs_float_anova", "layer_2_float_anova_vs_8bit_ising", "layer_3_8bit_ising_vs_solvers"):
            assert layer in aed[scope], f"Missing {scope}.{layer}"

    return "8/8 matrices certified (0 illegal GS, 0 aux ties), Model 3B >= 6 levels, 4-row attribution & C_cap retention OK"


def gate_experiments():
    exp = _load_json(os.path.join(RES_DIR, "v2_experiments.json"))
    cl = exp["closed_loop_fmqa"]
    hg = cl["analytical_hypergeometric"]["final_expected_rare05"]
    assert abs(hg - 0.9070295) < 1e-5, f"analytical_hypergeometric={hg}"
    assert cl["n_seeds"] == 20, f"n_seeds={cl['n_seeds']}"

    fm_conv = cl["fm_lbfgs_convergence"]
    assert fm_conv["converged_pct"] == 100.0, f"converged_pct={fm_conv['converged_pct']}"
    assert fm_conv["total_fits"] == 300, f"total_fits={fm_conv['total_fits']}"

    qubo_aud = cl["per_round_qubo_audit"]["v2_Real_8Bit_QUBO"]
    assert qubo_aud["total_rounds"] == 300, f"total_rounds={qubo_aud['total_rounds']}"
    assert qubo_aud["exact_8bit_pct"] == 100.0, f"exact_8bit_pct={qubo_aud['exact_8bit_pct']}"
    assert qubo_aud["total_fallback_fills"] == 0, f"total_fallback_fills={qubo_aud['total_fallback_fills']}"

    eb_rows = exp["equal_budget_solver_topk"]["rows"]
    assert len(eb_rows) == 32, f"equal_budget_solver_topk rows={len(eb_rows)} != 32"
    assert all(r["n_runs"] == 50 for r in eb_rows), "Not all equal_budget rows have n_runs == 50"

    bw = exp["bitwidth_sweep"]
    bws = {int(r["bit_width"]): r for r in bw}
    assert set(bws.keys()) == {4, 5, 6, 7, 8, 10, 12}, f"bitwidth_sweep keys={sorted(bws.keys())}"
    assert bws[10]["v2_M_max"] == 17, f"bw=10 M_max={bws[10]['v2_M_max']}"
    assert bws[12]["v2_M_max"] == 71, f"bw=12 M_max={bws[12]['v2_M_max']}"
    assert bws[4]["v2_illegal_ground_count"] > 0

    pen = exp["penalty_sweep"]
    assert len(pen) == 10, f"penalty_sweep len={len(pen)} != 10"
    assert all(r["M_max"] == 4 for r in pen), "penalty_sweep M_max not fixed at 4"
    assert any(abs(r["alpha_P"] - 1.5) < 1e-6 for r in pen), "alpha_P=1.50 missing from penalty_sweep"

    mc = exp["monte_carlo_robustness"]
    assert mc["scales"] == [0.5, 1.0, 1.5, 2.0], f"MC scales={mc['scales']}"
    for mode in ("multi_scale_independent", "multi_scale_group_correlated_logit"):
        for inst in ("Q2_blind_11", "Q3_blind_12"):
            assert len(mc[mode][inst]) == 4

    return "20-seed closed-loop (hypergeom=0.9070295, 300/300 8-bit exact), 32 solver rows, 7 bitwidths, 10 M_max=4 penalty rows, 4 MC scales"


def gate_hardware_and_track_a():
    spqc = _load_json(os.path.join(RES_DIR, "v2_spqc550_results.json"))
    qubo = _load_json(os.path.join(RES_DIR, "v2_qubo_results.json"))
    cert_map = {c["matrix_id"]: c for c in qubo["certification_summary"]}

    expected_v3 = {
        "v3_q2_blind_12": "Q2_blind_11",
        "v3_q3_comp_18": "Q3_blind_12_Model3A_Comp",
        "v3_q3_hobo_29": "Q3_blind_12_Model3A_HOBO",
        "v3_q3_anova_25": "Q3_blind_12_Model3B_CompactANOVA",
    }
    tasks = spqc["tasks"]
    for tk, cert_key in expected_v3.items():
        assert tk in tasks, f"Missing v3 task {tk}"
        t = tasks[tk]
        assert t["hardware_spec"]["sha256"] == t["spqc550"]["matrix_sha256"], f"{tk} sha mismatch"
        assert t["cert_min_ising_energy"] == cert_map[cert_key]["cert_min_ising_energy"], f"{tk} cert_min mismatch"
        assert "y_aux_consistent_count" in t["spqc550"]

    cred = spqc["credit_accounting"]
    assert cred["estimated_remaining"] >= 5, f"estimated_remaining={cred['estimated_remaining']} < 5"
    assert cred["credit_safety_margin_satisfied"] is True

    # Invoke v2_spqc550_hardware.py --verify-only
    cmd = [sys.executable, os.path.join(V2_DIR, "v2_spqc550_hardware.py"), "--verify-only"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"v2_spqc550_hardware.py --verify-only failed:\n{res.stdout}\n{res.stderr}"

    # Scan all results/*.json, results/*.csv, v2_optimized/*.py, and requirements.txt for PII / local path leaks
    leaks = []
    scan_targets = []
    for fn in sorted(os.listdir(RES_DIR)):
        if fn.endswith((".json", ".csv")):
            scan_targets.append((f"results/{fn}", os.path.join(RES_DIR, fn)))
    for fn in sorted(os.listdir(V2_DIR)):
        if fn.endswith(".py"):
            scan_targets.append((f"v2_optimized/{fn}", os.path.join(V2_DIR, fn)))
    req_path = os.path.join(PKG_DIR, "requirements.txt")
    if os.path.isfile(req_path):
        scan_targets.append(("requirements.txt", req_path))

    for label, fpath in scan_targets:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        m = PATH_LEAK_RE.search(txt)
        if m:
            leaks.append(f"{label}: {m.group(0)}")
    assert not leaks, f"Sensitive path/PII leaks found in code/results: {leaks}"

    # Verify track_a_spqc_archive_summary.json against official table4 CSV
    ta = _load_json(os.path.join(RES_DIR, "track_a_spqc_archive_summary.json"))
    assert os.path.isfile(TABLE4_CSV), f"Missing {TABLE4_CSV}"
    with open(TABLE4_CSV, "r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert ta["row_count"] == len(csv_rows) == 18
    assert ta["task_count"] == 6
    for r_json, r_csv in zip(ta["rows"], csv_rows):
        assert r_json["task_name"] == r_csv["task_name"]
        assert r_json["solver"] == r_csv["solver"]
        assert int(r_json["n_spins"]) == int(r_csv["n_spins"])
        assert abs(float(r_json["feasible_rate"]) - float(r_csv["feasible_rate"])) < 1e-6
        assert r_json["best_formula"] == r_csv["best_formula"]

    return (
        f"4/4 v3 SPQC tasks verified via --verify-only, "
        f"credits remaining={cred['estimated_remaining']} (>=5), "
        f"0 leaks in results/, Track A 18/18 rows synced"
    )


def gate_figures_and_schematics():
    expected_figs = [
        "fig01_v1_vs_v2_surrogate_metrics.png",
        "fig02_residual_and_calibration.png",
        "fig03_closed_loop_fmqa_benchmark.png",
        "fig04_q2_v1_vs_v2_hardware_drift.png",
        "fig05_q3_qubit_compression_and_solvers.png",
        "fig06_sensitivity_and_mc_robustness.png",
        "fig07_spqc550_real_hardware_comparison.png",
        "fig08_electronic_phase_and_conformal_audit.png",
    ]
    for fn in expected_figs:
        p = os.path.join(FIG_DIR, fn)
        assert os.path.isfile(p), f"Missing figure {fn}"
        sz = os.path.getsize(p)
        assert sz > 10_000, f"Figure {fn} suspiciously small ({sz} B)"

    expected_schems = [
        "schematic01_pipeline_architecture",
        "schematic02_band_alignment_window",
        "schematic03_qubo_ising_compilation",
    ]
    for stem in expected_schems:
        svg_p = os.path.join(SCHEM_DIR, f"{stem}.svg")
        png_s = os.path.join(SCHEM_DIR, f"{stem}.png")
        png_f = os.path.join(FIG_DIR, f"{stem}.png")
        for p in (svg_p, png_s, png_f):
            assert os.path.isfile(p), f"Missing schematic {p}"
            assert os.path.getsize(p) > 5_000, f"Schematic {p} suspiciously small"
        with open(svg_p, "r", encoding="utf-8") as f:
            svg_txt = f.read()
        assert not CJK_RE.search(svg_txt), f"CJK character found in {svg_p}"

    # Verify generate_figures.py has zero hardcoded Track A arrays and uses cert_min_ising_energy
    gf_path = os.path.join(V2_DIR, "generate_figures.py")
    with open(gf_path, "r", encoding="utf-8") as f:
        gf_src = f.read()
    tree = ast.parse(gf_src)
    str_literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for forbidden_lit in ("AgTaO3", "TM\nQ2 Uncap\n(105 sp)", "apmcm26_v2_q3_submod_33"):
        assert forbidden_lit not in str_literals, f"Hardcoded legacy hardware literal found in generate_figures.py: {forbidden_lit!r}"
    assert "track_a_spqc_archive_summary.json" in gf_src
    assert "v2_spqc550_results.json" in gf_src
    assert "cert_min_ising_energy" in gf_src

    return "8/8 figures + 3/3 schematics (.svg+.png) present, zero hardcoded Track A data in generate_figures.py"


def gate_paper_and_report_sync():
    for doc_path in (PAPER_TYP, REPORT_MD):
        assert os.path.isfile(doc_path), f"Missing document: {doc_path}"

    with open(PAPER_TYP, "r", encoding="utf-8") as f:
        typ_txt = f.read()
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        rep_txt = f.read()

    assert not CJK_RE.search(typ_txt), "Chinese characters detected in paper/main.typ"

    doc_leaks = []
    leak_targets = [
        ("paper/main.typ", typ_txt),
        ("v2_optimized/REPORT.md", rep_txt),
    ]
    build_py = os.path.join(PKG_DIR, "paper", "build.py")
    if os.path.isfile(build_py):
        with open(build_py, "r", encoding="utf-8", errors="ignore") as f:
            leak_targets.append(("paper/build.py", f.read()))
    if os.path.isdir(SCHEM_DIR):
        for fn in sorted(os.listdir(SCHEM_DIR)):
            if fn.endswith(".svg"):
                with open(os.path.join(SCHEM_DIR, fn), "r", encoding="utf-8", errors="ignore") as f:
                    leak_targets.append((f"paper/schematics/{fn}", f.read()))
    for label, txt in leak_targets:
        m = PATH_LEAK_RE.search(txt)
        if m:
            doc_leaks.append(f"{label}: {m.group(0)}")
    assert not doc_leaks, f"Sensitive path/PII leaks found in paper/report/schematics: {doc_leaks}"

    forbidden_phrases = [
        "Theorem 0",
        "eliminates VBM/CBM regression error entirely",
        "3,969 unseen (A,B) pairs",
        "3,969 unseen $(A, B)$ pairs",
        "97.8%",
        "17.6/18",
        "44.0 times",
        "44x",
        "proving 3-body exactness on real quantum hardware",
        "all Top-25 blind candidates lie strictly below",
        "M_max in [6, 12]",
        "exact jump at alpha_P* = 1.50",
    ]
    for label, txt in (("paper/main.typ", typ_txt), ("v2_optimized/REPORT.md", rep_txt)):
        for ph in forbidden_phrases:
            assert ph.lower() not in txt.lower(), f"Forbidden Section 8.6 / F29 phrase {ph!r} found in {label}"

    # Cross-check key verified JSON numbers appear consistently in paper & report
    qubo = _load_json(os.path.join(RES_DIR, "v2_qubo_results.json"))
    exp = _load_json(os.path.join(RES_DIR, "v2_experiments.json"))
    spqc = _load_json(os.path.join(RES_DIR, "v2_spqc550_results.json"))

    cert_energies = [str(spqc["tasks"][k]["cert_min_ising_energy"]) for k in ("v3_q2_blind_12", "v3_q3_comp_18", "v3_q3_hobo_29", "v3_q3_anova_25")]
    hg_val = f"{exp['closed_loop_fmqa']['analytical_hypergeometric']['final_expected_rare05']:.4f}"

    for label, txt in (("paper/main.typ", typ_txt), ("v2_optimized/REPORT.md", rep_txt)):
        assert re.search(r"\b567\b", txt), f"{label} missing 567 (A,B) pairs"
        assert re.search(r"(?<!\d)(?:3,969|3969)(?!\d)", txt), f"{label} missing 3,969 rows"
        assert re.search(r"\b19\b", txt) and re.search(r"\b15\b", txt), f"{label} missing 19 vs 15 target distinction"
        assert hg_val in txt or "0.907" in txt, f"{label} missing hypergeometric baseline {hg_val}"
        assert "0.2524" in txt, f"{label} missing 0.252406 exact pool optimum"
        for ce in cert_energies:
            assert re.search(rf"(?<!\d){re.escape(ce)}(?!\d)", txt), f"{label} missing certified ground-state energy {ce}"
        for fr in ("30.0%", "90.0%", "20.0%"):
            assert fr in txt, f"{label} missing v3 hardware feasibility rate {fr}"

    return "paper/main.typ & REPORT.md pass Section 8.6 phrasing, JSON numeric sync, and 0 PII/path leaks"


def main():
    parser = argparse.ArgumentParser(description="Verify all Section 9.3 and F01-F29 review gates")
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Skip paper/main.typ and REPORT.md prose/numeric sync checks (use during M5A).",
    )
    args = parser.parse_args()

    print("=== 2026 APMCM Problem D Automated Review Gate Verifier ===")
    print(f"Mode: {'CODE-ONLY' if args.code_only else 'FULL (Code + Paper + Report)'}\n")

    runner = GateRunner()
    runner.check("Gate 1: Dataset Dual SHA-256 (F21, F24)", gate_dataset_checksums)
    runner.check("Gate 2: Q1 Isolation, M4/M5 Baselines & Holdouts (F05, F12, F25, F26)", gate_q1_diagnostics)
    runner.check("Gate 3: Conformal Splits, Residual Corr & Scoped P_hit (F06, F07, F10)", gate_conformal_and_physics)
    runner.check("Gate 4: Q2/Q3 Certification, ANOVA 3-Layer & Attribution (F01-F03, F18, F19)", gate_qubo_certification)
    runner.check("Gate 5: Closed-Loop 20-Seed & Sensitivity Suite (F04, F13-F17)", gate_experiments)
    runner.check("Gate 6: SPQC-550 --verify-only, Credits, PII & Track A (F08, F09, F22, F23)", gate_hardware_and_track_a)
    runner.check("Gate 7: Figures, Schematics & Zero Hardcoded Track A (F22, F23)", gate_figures_and_schematics)
    if not args.code_only:
        runner.check("Gate 8: Paper & REPORT.md Section 8.6 Sync (F10, F11, F27-F29)", gate_paper_and_report_sync)
    else:
        print("[SKIP] Gate 8: Paper & REPORT.md sync skipped (--code-only)")

    print(f"\n=== SUMMARY: {len(runner.passed)} passed, {len(runner.failed)} failed ===")
    return 1 if runner.failed else 0


if __name__ == "__main__":
    sys.exit(main())
