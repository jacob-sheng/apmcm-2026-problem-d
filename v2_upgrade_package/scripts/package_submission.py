#!/usr/bin/env python3
"""
Package 2026 APMCM Problem D Supporting Materials into submission/Dapmcm26302266fj.zip.
Includes all anonymized v2_optimized/ source code, verified SPQC-550 quantum hardware
checkpoints, processed prediction tables, quantitative JSON summaries, and all 11
publication 300 DPI PNGs (8 data figures + 3 conceptual schematics).
Strictly EXCLUDES all official raw problem CSV files, *dryrun*.json, *.npz, .env*,
.venv/, *.whl, and verifies zero path/PII/secret leaks and < 25 MB archive size.
"""

import os
import re
import sys
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ZIP = os.path.join(BASE_DIR, "submission", "Dapmcm26302266fj.zip")
os.makedirs(os.path.dirname(OUT_ZIP), exist_ok=True)

MAX_ZIP_BYTES = 25 * 1024 * 1024

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
    "v2_spqc550_dryrun.json",
    ".ds_store",
    "thumbs.db",
}

EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".whl", ".tmp", ".npz", ".env")
SCAN_SUFFIXES = (".py", ".json", ".csv", ".txt", ".md", ".typ")

PATH_LEAK_RE = re.compile(
    r"([A-Za-z]:[/\\]+Users[/\\]+|[A-Za-z]:[/\\]+Codex|"
    + "/" + "Users/"
    + "|" + "/" + "home/"
    + "|" + "/" + "mnt/[a-zA-Z]/|"
    + r"@(?:[\w.-]+\.)?(edu\.cn|qq\.com|163\.com|gmail\.com|outlook\.com|126\.com|foxmail\.com))",
    re.I,
)

INCLUDE_DIRS = [
    ("v2_optimized", "v2_optimized"),
    ("scripts", "scripts"),
]

README_CONTENT = """# Supporting Materials — 2026 APMCM Problem D (Team Dapmcm26302266)

## Contents
- `v2_optimized/*.py`: Complete, anonymized, reproducible Python source code for:
  - `common_physics.py`: Dynamic `resolve_data_dir()` (`APMCM_DATA_DIR` -> workspace official dataset folder), dual raw-bytes & CRLF-normalized SHA-256 audit (`verify_dataset_checksums()`), Periodic Table descriptors, Bartel tau tolerance factor, Butler-Ginley log-additive band-center Ridge, Gaussian hinge expectation, and Ising energy sorting.
  - `v1_baseline.py`: Baseline benchmarks (Direct LightGBM, log1p LightGBM, One-Hot FM, and naive 8-bit QUBO).
  - `v2_surrogate_q1.py`: Problem 1 outer `(A,B)`-GroupKFold base-isolated Stage-0 + 4-family calibrated Bayes-optimal hurdle surrogate (with pooled-OOF meta-layer), automated `+100` base-path isolation test, `M4/M5` baseline alignment, and dual `02-only` (`v2_holdout_04_train02only.csv`) & `02+03` refit (`v2_holdout_04.csv`) exports.
  - `v2_deep_physics_conformal.py`: Site-exchange Butler-Ginley audit, 200 `(A,B)`-group-disjoint Mondrian split-conformal calibration (`p_semi >= 0.20` strata), semiconductor residual correlation, deterministic `sum(P_hit)` analysis, and Figure 8 generation.
  - `v2_qubo_q2_q3.py`: Problems 2 & 3 hardware-exact 8-bit QUBO/HOBO (`Q2 Exact-8bit`, `Model 3A-Comp`, `Model 3A-HOBO`, `Model 3B-ANOVA`), `certify_global_ground_states()` full primary-variable enumeration (`0` illegal ground states), 3-layer ANOVA error decomposition, 4-row compression attribution, 3-subspace `C_cap` retention, and equal-budget 50-run SA vs. Tabu benchmarking.
  - `v2_experiments.py`: 20-seed closed-loop benchmark (analytical hypergeometric expectation, converged `L-BFGS-B` FM, classical sequential vs. hybrid Top-20 + 8-bit masked-QUBO acquisition), equal-budget Top-K, independent 4-12 bit `M_max(B)` sweep, fixed-`M_max=4` penalty sweep, and multi-scale (0.5-2.0 sigma) independent & group-correlated MC perturbation stability.
  - `v2_spqc550_hardware.py`: Bose Quantum `SPQC-550` submission & independent algebraic Ising recovery + per-record verification (`--dry-run` isolated, `--verify-only` bit-exact SHA-256 + record equality check, explicit `--run-spqc`).
  - `generate_figures.py` & `run_all.py`: Publication-grade 300 DPI figure generator and staged offline CLI runner.
- `v2_optimized/reference_baselines/`: Bundled archival reference CSV summaries (`table1_surrogate_ablation.csv`, `table4_spqc550_hardware_vs_classical.csv`) enabling completely standalone offline reproduction.
- `v2_optimized/results/`: All quantitative JSON/CSV summaries (`track_a_spqc_archive_summary.json`, `v2_holdout_04_train02only.csv`, `v2_holdout_04.csv`, `v2_q2_blind_11.csv`, `v2_q3_blind_12.csv`, etc.), and verified `SPQC-550` raw spin checkpoints (`spqc_checkpoints/`).
- `v2_optimized/figures/`: All 11 publication-grade 300 DPI English PNGs (`fig01` through `fig08` plus conceptual schematics `schematic01` through `schematic03`).
- `scripts/`: Standalone code-gate verifier (`verify_all_review_gates.py`) and submission packaging/audit scripts (`package_submission.py`, `audit_submission.py`).
- `requirements.txt`: Pinned Python environment dependencies and offline `kaiwu==1.4.4` installation notes.

All official raw competition CSV datasets, dry-run artifacts, `.npz` files, and credentials are strictly excluded per APMCM submission rules.

## Standalone Reproduction (Supporting Archive + Official Dataset Only)
Unpack this archive, install `requirements.txt` (with `kaiwu==1.4.4`), set `APMCM_DATA_DIR` to the official `date/` directory, and run:
1. `python v2_optimized/run_all.py --verify-saved`            (checksums, saved CSVs/JSONs & SPQC-550 record audit; 0 cloud calls)
2. `python scripts/verify_all_review_gates.py --code-only`   (all 7 live matrix, certification & data gates; 0 cloud calls)
3. `python v2_optimized/run_all.py --all`                     (full offline retraining, Q2/Q3 certification, experiments & figures)

*(In the full repository containing `paper/` and `submission/`, run `python scripts/verify_all_review_gates.py` and `python scripts/audit_submission.py --problem D --team apmcm26302266`.)*
"""


def _is_excluded(filename):
    low = filename.lower()
    if low in RAW_EXCLUDED_NAMES:
        return True
    if low.startswith(".env"):
        return True
    if "dryrun" in low and low.endswith(".json"):
        return True
    if low.endswith(EXCLUDED_SUFFIXES):
        return True
    return False


def main():
    count = 0
    leaks = []
    if PATH_LEAK_RE.search(README_CONTENT):
        leaks.append("README.md")

    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.md", README_CONTENT)
        count += 1
        req_path = os.path.join(BASE_DIR, "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path, "r", encoding="utf-8", errors="ignore") as f:
                if PATH_LEAK_RE.search(f.read()):
                    leaks.append("requirements.txt")
            zf.write(req_path, "requirements.txt")
            count += 1

        for rel_src, arc_prefix in INCLUDE_DIRS:
            abs_src = os.path.join(BASE_DIR, rel_src)
            if not os.path.isdir(abs_src):
                continue
            for root, dirs, files in os.walk(abs_src):
                dirs[:] = [d for d in sorted(dirs) if d not in ("__pycache__", "catboost_info")]
                for fn in sorted(files):
                    if _is_excluded(fn):
                        continue
                    full_p = os.path.join(root, fn)
                    rel_inner = os.path.relpath(full_p, abs_src).replace("\\", "/")
                    arcname = f"{arc_prefix}/{rel_inner}"
                    if fn.lower().endswith(SCAN_SUFFIXES):
                        with open(full_p, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read()
                        m = PATH_LEAK_RE.search(txt)
                        if m:
                            leaks.append(f"{arcname}: {m.group(0)}")
                    zf.write(full_p, arcname)
                    count += 1

    if leaks:
        os.remove(OUT_ZIP)
        print(f"[ERROR] Sensitive path/PII leaks detected; aborted archive creation: {leaks}", file=sys.stderr)
        return 1

    size_bytes = os.path.getsize(OUT_ZIP)
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes >= MAX_ZIP_BYTES:
        print(f"[ERROR] Archive {OUT_ZIP} ({size_mb:.2f} MB) exceeds 25 MB limit!", file=sys.stderr)
        return 1

    print(f"[OK] Packaged {count} files into {OUT_ZIP} ({size_mb:.2f} MB, 0 leaks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
