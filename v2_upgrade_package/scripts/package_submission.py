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
    "verify_all_review_gates.py",
    "audit_submission.py",
    "package_submission.py",
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

## Directory Layout
- `v2_optimized/`: Python source code for crystal-chemistry descriptors (`common_physics.py`), baseline benchmarks (`v1_baseline.py`), Problem 1 hurdle-Bayes surrogate (`v2_surrogate_q1.py`), conformal calibration (`v2_deep_physics_conformal.py`), Problem 2/3 8-bit QUBO/HOBO formulations and exact certification (`v2_qubo_q2_q3.py`), closed-loop and sensitivity benchmarks (`v2_experiments.py`), SPQC-550 hardware decoding (`v2_spqc550_hardware.py`), figure generation (`generate_figures.py`), and the offline runner (`run_all.py`).
- `v2_optimized/results/`: Exported prediction tables, JSON summaries, and saved SPQC-550 raw spin checkpoints (`spqc_checkpoints/`).
- `v2_optimized/figures/`: Figures and diagrams used in the paper.
- `v2_optimized/reference_baselines/`: Reference baseline summary CSVs.
- `scripts/verify_milp_qubo_crosscheck.py`: Independent Fortet-McCormick MILP verification using SciPy/HiGHS.
- `requirements.txt`: Python package versions.

Official competition CSV files and API credentials are excluded.

## Offline Reproduction
Install dependencies from `requirements.txt` (plus `kaiwu==1.4.4`), set `APMCM_DATA_DIR` to the official dataset folder (`date/`), and run:

1. `python v2_optimized/run_all.py --verify-saved` — verify dataset checksums, saved predictions, and SPQC-550 raw spin checkpoints (no cloud API calls).
2. `python scripts/verify_milp_qubo_crosscheck.py` — cross-check all 8 integer Ising ground-state energies via HiGHS MILP (`mip_gap = 0`).
3. `python v2_optimized/run_all.py --all` — run full offline surrogate training, QUBO/HOBO certification, benchmarks, and figure generation.
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
