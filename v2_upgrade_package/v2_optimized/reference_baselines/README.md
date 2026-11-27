# Bundled Historical Reference Baselines (`v2_optimized/reference_baselines/`)

This directory bundles two small, team-generated archival summary CSV files so the standalone supporting archive (`Dapmcm26302266fj.zip`) can execute end-to-end offline verification (`scripts/verify_all_review_gates.py --code-only`) and full retraining (`v2_optimized/run_all.py --all`) using only the official contest dataset directory (`APMCM_DATA_DIR`), with zero external repository dependencies:

1. `table1_surrogate_ablation.csv`: Historical `02_train -> 03_val` benchmark metrics for `M4_PIMT_Physics_Only` and `M5_PIMT_Hybrid_Ensemble` (archived ablation summary; per-row `03_val` predictions were not archived in the baseline outputs folder, so near-target `MAE(V<1)` and `MAE(V<0.5)` for `M4`/`M5` are reported as `N/A`).
2. `table4_spqc550_hardware_vs_classical.csv`: Historical Track A (`SPQC-550` vs. SA/Tabu) execution summary across six legacy full-space Ising tasks, parsed cleanly by `v2_spqc550_hardware.py --export-track-a` into `v2_optimized/results/track_a_spqc_archive_summary.json`.

All files in this folder contain strictly anonymized model metrics and contain zero PII, local paths, or official raw competition CSV rows.
