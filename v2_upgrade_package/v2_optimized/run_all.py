#!/usr/bin/env python3
"""
One-Command Master Runner for v2_optimized (2026 APMCM Problem D).
Executes:
  1. v1_baseline.py      (Teammate v1 baseline benchmark)
  2. v2_surrogate_q1.py  (Upgraded v2 Q1 physics-coupled surrogate + diagnostics + pool scoring)
  3. v2_qubo_q2_q3.py    (Upgraded v2 8-bit exact QUBO/HOBO + SA vs Tabu benchmark)
  4. v2_experiments.py   (Closed-loop FMQA benchmark + multi-axis sensitivity & MC robustness)
  5. generate_figures.py (6 publication-grade 300 DPI English-only PNG figures)
"""

import os
import subprocess
import sys
import time

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

STEPS = [
    ("Stage 1: v1 Baseline Benchmark", "v1_baseline.py"),
    ("Stage 2: v2 Q1 Physics-Coupled Surrogate & Pool Scoring", "v2_surrogate_q1.py"),
    ("Stage 3: v2 Q2 & Q3 Hardware-Exact 8-Bit QUBO/HOBO (SA vs Tabu)", "v2_qubo_q2_q3.py"),
    ("Stage 4: Closed-Loop FMQA & Sensitivity/Robustness Suite", "v2_experiments.py"),
    ("Stage 5: Publication-Grade Figure Generation (300 DPI)", "generate_figures.py"),
]


def main():
    t_all = time.perf_counter()
    for title, script in STEPS:
        print(f"\n{'='*72}\n>>> {title} ({script})\n{'='*72}")
        t0 = time.perf_counter()
        res = subprocess.run([PYTHON, os.path.join(CUR_DIR, script)], check=True)
        print(f"--- Finished {script} in {time.perf_counter() - t0:.2f}s (exit={res.returncode}) ---")
    print(f"\nALL STAGES COMPLETED SUCCESSFULLY in {time.perf_counter() - t_all:.2f}s.")


if __name__ == "__main__":
    main()
