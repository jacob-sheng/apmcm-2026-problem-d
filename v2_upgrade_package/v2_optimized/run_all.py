#!/usr/bin/env python3
"""
Staged CLI Master Runner for v2_optimized (2026 APMCM Problem D).

Stages (strictly local/offline; NEVER invokes paid --run-spqc cloud calls):
  --verify-saved     Run dataset SHA-256 audit, check saved JSON/CSV artifacts,
                     and run v2_spqc550_hardware.py --verify-only.
  --retrain-q1       Run v1_baseline.py, v2_surrogate_q1.py, and
                     v2_deep_physics_conformal.py.
  --run-qubo         Run v2_qubo_q2_q3.py.
  --run-experiments  Run v2_experiments.py.
  --build-figures    Run generate_figures.py.
  --all              Run all offline stages end-to-end (retrain-q1 -> run-qubo
                     -> run-experiments -> build-figures -> verify-saved).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import pandas as pd

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(CUR_DIR, "results")
FIG_DIR = os.path.join(CUR_DIR, "figures")
PYTHON = sys.executable

if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from common_physics import verify_dataset_checksums


REQUIRED_JSONS = [
    "v1_baseline_metrics.json",
    "v2_q1_diagnostics.json",
    "v2_deep_physics_conformal.json",
    "v2_qubo_results.json",
    "v2_experiments.json",
    "v2_spqc550_results.json",
]

REQUIRED_CSVS = [
    "v2_val_predictions.csv",
    "v2_holdout_04.csv",
    "v2_holdout_top20_detailed.csv",
    "v2_q2_full_07.csv",
    "v2_q2_blind_11.csv",
    "v2_q3_full_08.csv",
    "v2_q3_blind_12.csv",
]

OPTIONAL_CSVS = [
    "v2_holdout_04_train02only.csv",
]


def _run_script(title, script_name, extra_args=None):
    cmd = [PYTHON, os.path.join(CUR_DIR, script_name)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'=' * 72}\n>>> {title} ({' '.join(os.path.basename(c) if i == 1 else c for i, c in enumerate(cmd))})\n{'=' * 72}", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=True)
    dt = time.perf_counter() - t0
    print(f"--- Finished {script_name} in {dt:.2f}s (exit={res.returncode}) ---", flush=True)
    return dt


def run_verify_saved():
    print(f"\n{'=' * 72}\n>>> Stage: Verify Saved Artifacts & Dataset Checksums (Offline)\n{'=' * 72}", flush=True)
    t0 = time.perf_counter()

    # 1. Official dataset checksum audit (raw-byte + CRLF-normalized)
    csum = verify_dataset_checksums()
    print(
        f"[Dataset Checksums] dir={csum['data_dir']}\n"
        f"  raw_bytes_matches={csum['raw_bytes_matches']}/{csum['total_files']}, "
        f"crlf_normalized_matches={csum['crlf_normalized_matches']}/{csum['total_files']}",
        flush=True,
    )
    for fname, info in csum["files"].items():
        status = "RAW+CRLF" if info["raw_match"] else ("CRLF-NORM" if info["crlf_normalized_match"] else "MISMATCH")
        print(f"    - {fname:32s} [{status:9s}] eol={info['detected_eol']:4s} size={info['size_bytes']}", flush=True)
    if not csum["all_crlf_normalized_match"]:
        raise RuntimeError(
            f"Dataset checksum audit failed: {csum['crlf_normalized_matches']}/{csum['total_files']} CRLF-normalized matches."
        )

    # 2. Check saved JSONs
    print("\n[Saved JSONs]", flush=True)
    for fn in REQUIRED_JSONS:
        p = os.path.join(RES_DIR, fn)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required result JSON: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or len(data) == 0:
            raise ValueError(f"Empty or invalid JSON object: {p}")
        print(f"  [OK] {fn:32s} keys={list(data.keys())[:5]}", flush=True)

    # 3. Check saved CSVs
    print("\n[Saved CSVs]", flush=True)
    for fn in REQUIRED_CSVS:
        p = os.path.join(RES_DIR, fn)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required result CSV: {p}")
        df = pd.read_csv(p)
        if len(df) == 0:
            raise ValueError(f"Empty CSV file: {p}")
        print(f"  [OK] {fn:32s} shape={df.shape}", flush=True)

    for fn in OPTIONAL_CSVS:
        p = os.path.join(RES_DIR, fn)
        if os.path.isfile(p):
            df = pd.read_csv(p)
            print(f"  [OK] {fn:32s} shape={df.shape} (optional/post-M2)", flush=True)
        else:
            print(f"  [INFO] {fn:30s} not present yet (created in M2)", flush=True)

    # 4. Check existing figures
    if os.path.isdir(FIG_DIR):
        pngs = sorted(f for f in os.listdir(FIG_DIR) if f.endswith(".png"))
        print(f"\n[Saved Figures] {len(pngs)} PNG(s) in {FIG_DIR}", flush=True)
        for fn in pngs:
            sz = os.path.getsize(os.path.join(FIG_DIR, fn))
            if sz == 0:
                raise ValueError(f"Zero-byte figure file: {fn}")
            print(f"  [OK] {fn:48s} ({sz:,} B)", flush=True)

    # 5. SPQC-550 offline hardware matrix & raw spin verification
    _run_script("SPQC-550 Offline Verification", "v2_spqc550_hardware.py", ["--verify-only"])
    print(f"\n[PASS] --verify-saved completed cleanly in {time.perf_counter() - t0:.2f}s (0 cloud calls).", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Staged CLI runner for 2026 APMCM Problem D (offline stages only; "
            "live SPQC-550 calls require explicit v2_spqc550_hardware.py --run-spqc)."
        )
    )
    parser.add_argument("--verify-saved", action="store_true", help="Verify dataset checksums, saved JSONs/CSVs, and SPQC-550 archives (no retraining or cloud calls)")
    parser.add_argument("--retrain-q1", action="store_true", help="Run v1_baseline.py, v2_surrogate_q1.py, and v2_deep_physics_conformal.py")
    parser.add_argument("--run-qubo", action="store_true", help="Run v2_qubo_q2_q3.py")
    parser.add_argument("--run-experiments", action="store_true", help="Run v2_experiments.py")
    parser.add_argument("--build-figures", action="store_true", help="Run generate_figures.py")
    parser.add_argument("--all", action="store_true", help="Run all offline stages end-to-end (never invokes --run-spqc)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.verify_saved, args.retrain_q1, args.run_qubo, args.run_experiments, args.build_figures, args.all]):
        parser.print_help()
        return 0

    t_all = time.perf_counter()
    do_q1 = args.all or args.retrain_q1
    do_qubo = args.all or args.run_qubo
    do_exp = args.all or args.run_experiments
    do_figs = args.all or args.build_figures
    do_verify = args.all or args.verify_saved

    if do_q1:
        _run_script("Stage 1a: v1 Baseline Benchmark", "v1_baseline.py")
        _run_script("Stage 1b: v2 Q1 Physics-Coupled Surrogate & Pool Scoring", "v2_surrogate_q1.py")
        _run_script("Stage 1c: v2 Deep Physics & Conformal Analysis", "v2_deep_physics_conformal.py")

    if do_qubo:
        _run_script("Stage 2: v2 Q2 & Q3 Hardware-Exact 8-Bit QUBO/HOBO (SA vs Tabu)", "v2_qubo_q2_q3.py")

    if do_exp:
        _run_script("Stage 3: Closed-Loop Acquisition & Sensitivity/Robustness Suite", "v2_experiments.py")

    if do_figs:
        _run_script("Stage 4: Publication-Grade Figure Generation (300 DPI)", "generate_figures.py")

    if do_verify:
        run_verify_saved()

    print(f"\nREQUESTED STAGES COMPLETED SUCCESSFULLY in {time.perf_counter() - t_all:.2f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
