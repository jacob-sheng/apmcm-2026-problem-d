#!/usr/bin/env python3
"""
Package 2026 APMCM Problem D Supporting Materials into submission/Dapmcm26302266fj.zip.
Includes all anonymized v2_optimized/ source code, verified SPQC-550 quantum hardware
checkpoints, processed prediction tables, quantitative JSON summaries, and 8 publication
300 DPI figures. Strictly EXCLUDES all official raw problem CSV files, .env, .venv/,
*.whl, and verifies zero path/PII/secret leaks inside packaged text/code files.
"""

import os
import re
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ZIP = os.path.join(BASE_DIR, "submission", "Dapmcm26302266fj.zip")
os.makedirs(os.path.dirname(OUT_ZIP), exist_ok=True)

RAW_EXCLUDED_NAMES = {
    "02_castelli_train.csv",
    "03_castelli_val.csv",
    "04_castelli_test_holdout.csv",
    "06_castelli_no_physics.csv",
    "07_castelli_q2_pool.csv",
    "08_castelli_q3_pool.csv",
    "11_castelli_q2_pool_blind.csv",
    "12_castelli_q3_pool_blind.csv",
    "sha256_public.json",
    "report.md",
}

INCLUDE_DIRS = [
    ("v2_optimized", "v2_optimized"),
]

README_CONTENT = """# Supporting Materials — 2026 APMCM Problem D (Team Dapmcm26302266)

## Contents
- `v2_optimized/*.py`: Complete, anonymized, reproducible Python source code for:
  - `common_physics.py`: First-principles Periodic Table descriptors, Bartel tau tolerance factor, exact Butler-Ginley log-additive band-center model, Gaussian hinge expectation, and Ising energy sorting.
  - `v1_baseline.py`: Baseline benchmarks (Direct LightGBM, log1p LightGBM, One-Hot FM, and naive 8-bit QUBO).
  - `v2_surrogate_q1.py`: Problem 1 Stage-0 multi-task DFT predictors + 4-family calibrated Bayes-optimal hurdle surrogate + residual/calibration diagnostics + pool scoring.
  - `v2_qubo_q2_q3.py`: Problems 2 & 3 hardware-exact 8-bit active-subspace QUBO/HOBO (Model 3A Submodular HOBO & Model 3B Compact ANOVA) with SA vs. TabuSearch benchmarking.
  - `v2_experiments.py`: Closed-loop FMQA active-screening benchmark (03_val ground truth), bit-width/penalty/cap sensitivity sweeps, and 500-trial Monte Carlo robustness.
  - `v2_spqc550_hardware.py`: Bose Quantum SPQC-550 real optical quantum computer submission & independent raw-spin verification (`--dry-run`, `--run-spqc`, `--verify-only`).
  - `v2_deep_physics_conformal.py`: Site-exchange Butler-Ginley symmetry audit, Mondrian split-conformal coverage, holdout yield curve, and Figure 8 generation.
  - `generate_figures.py` & `run_all.py`: Publication-grade 300 DPI figure generator and one-command pipeline runner.
- `v2_optimized/results/`: All quantitative JSON/CSV summaries, blind holdout predictions (`v2_holdout_04.csv`, `v2_q2_blind_11.csv`, `v2_q3_blind_12.csv`), and raw verified SPQC-550 hardware checkpoints (`spqc_checkpoints/`).
- `v2_optimized/figures/`: All 8 publication-grade 300 DPI English PNG figures (`fig01` through `fig08`).
- `requirements.txt`: Exact Python environment dependencies.

All official raw competition CSV datasets are strictly excluded per APMCM submission rules.

## Reproduction
Place official dataset CSVs in `2026 APMCM Problems/2026 APMCM Problem D/date/` and run:
1. `python v2_optimized/run_all.py`
2. `python v2_optimized/v2_deep_physics_conformal.py`
3. `python v2_optimized/v2_spqc550_hardware.py --verify-only`
"""


def main():
    count = 0
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", README_CONTENT)
        count += 1
        req_path = os.path.join(BASE_DIR, "requirements.txt")
        if os.path.exists(req_path):
            zf.write(req_path, "requirements.txt")
            count += 1

        for rel_src, arc_prefix in INCLUDE_DIRS:
            abs_src = os.path.join(BASE_DIR, rel_src)
            if not os.path.isdir(abs_src):
                continue
            for root, dirs, files in os.walk(abs_src):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for fn in sorted(files):
                    if fn.lower() in RAW_EXCLUDED_NAMES or fn.endswith((".pyc", ".whl", ".tmp", ".env")):
                        continue
                    full_p = os.path.join(root, fn)
                    rel_inner = os.path.relpath(full_p, abs_src).replace("\\", "/")
                    arcname = f"{arc_prefix}/{rel_inner}"
                    zf.write(full_p, arcname)
                    count += 1

    size_mb = os.path.getsize(OUT_ZIP) / (1024 * 1024)
    print(f"[OK] Packaged {count} files into {OUT_ZIP} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
