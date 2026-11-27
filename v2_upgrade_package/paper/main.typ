// =============================================================================
// 2026 APMCM (16th Asia and Pacific Mathematical Contest in Modeling)
// Problem D — Team Dapmcm26302266
// Strictly Anonymous | English Only | Main Body <= 25 Pages | 10.8 pt Academic Serif
// =============================================================================

#let team-id = "apmcm26302266"
#let problem-chosen = "D"
#let paper-title = "Physics-Coupled Surrogate Modeling, Certified 8-Bit Ising Compilation, and SPQC-550 Verification for Rare-Target Perovskite Discovery"

#set document(
  title: "",
  author: (),
  keywords: (),
)

#set page(
  paper: "a4",
  margin: (top: 2.1cm, bottom: 1.9cm, left: 2.0cm, right: 2.0cm),
  header: context {
    let cur = counter(page).get().first()
    let tot = counter(page).final().first()
    set text(size: 9.8pt, font: ("Times New Roman", "New Computer Modern", "Libertinus Serif"))
    grid(
      columns: (1fr, auto),
      align: (left, right),
      [*Team #problem-chosen#team-id*],
      [Page #cur of #tot],
    )
    v(-6pt)
    line(length: 100%, stroke: 0.6pt + luma(110))
  },
  footer: none,
)

#set text(
  font: ("Times New Roman", "New Computer Modern", "Libertinus Serif"),
  size: 10.6pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.58em,
  spacing: 0.72em,
)

#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(0.50em)
  set text(size: 12.4pt, weight: "bold", fill: rgb("#1B365D"))
  it
  v(0.18em)
}
#show heading.where(level: 2): it => {
  v(0.36em)
  set text(size: 11.1pt, weight: "bold", fill: rgb("#2C4A70"))
  it
  v(0.12em)
}
#show heading.where(level: 3): it => {
  v(0.24em)
  set text(size: 10.6pt, weight: "bold", style: "italic", fill: rgb("#2C4A70"))
  it
  v(0.08em)
}

#set math.equation(numbering: "(1)")
#show figure.caption: it => {
  set text(size: 9.0pt, fill: luma(40))
  it
}
#show table: set text(size: 8.4pt)
#show table: set par(justify: false)

#let theorem-box(title, body) = block(
  fill: rgb("#F4F7FA"),
  stroke: (left: 2.5pt + rgb("#1B365D"), rest: 0.4pt + luma(200)),
  inset: (x: 9pt, y: 6pt),
  radius: 2pt,
  width: 100%,
)[
  #text(weight: "bold", fill: rgb("#1B365D"))[#title.] #body
]

// =============================================================================
// PAGE 1: MANDATORY STANDALONE SUMMARY SHEET
// =============================================================================
#align(center)[
  #v(0.15cm)
  #text(size: 10.5pt, weight: "bold", fill: luma(60))[
    2026 Asia and Pacific Mathematical Contest in Modeling (APMCM)
  ]
  #v(0.18cm)
  #table(
    columns: (1fr, 1fr, 1fr),
    align: center,
    stroke: 0.6pt + luma(140),
    inset: 5.5pt,
    [*Problem Chosen* \ #text(size: 12.5pt, weight: "bold", fill: rgb("#1B365D"))[#problem-chosen]],
    [*Team Control Number* \ #text(size: 12.5pt, weight: "bold", fill: rgb("#1B365D"))[#team-id]],
    [*Contest Year* \ #text(size: 12.5pt, weight: "bold", fill: rgb("#1B365D"))[2026]],
  )
  #v(0.28cm)
  #text(size: 13.0pt, weight: "bold")[#paper-title]
  #v(0.22cm)
  #text(size: 11.6pt, weight: "bold", fill: rgb("#1B365D"))[Summary]
]

#v(0.10cm)
#[
#set text(size: 10.3pt)
#set par(leading: 0.58em, spacing: 0.78em)

Screening cubic $A B X_3$ perovskites for solar water-splitting photocatalysis across the $52 times 52 times 7 = 18,928$-compound grid confronts an extreme rare-target imbalance: $96.12%$ of candidates are metallic ($E_g = 0$), and under the official quadratic hinge violation score $V$, only $19$ targets ($0.1004%$; compared with $15$ under strict interior inequalities) achieve $V = 0$. Crucially, all $19$ $V = 0$ compounds lie inside `04_castelli_test_holdout.csv` ($529$ unseen $(A, B)$ pairs, $3,703$ rows), leaving zero $V = 0$ labels in `02_train` or `03_val` ($567$ $(A, B)$ pairs, $3,969$ rows). To bridge solid-state physics with finite-precision Coherent Ising Machine (CIM) hardware (`SPQC-550`, symmetric integer couplings in $[-127, 127]$), we develop an outer-fold-isolated physics-coupled surrogate, globally certified 8-bit QUBO/HOBO formulations, and an independently verified hardware execution protocol.

For Problem 1, target-dependent feature construction and four-family base predictors (`CatBoost`, `LightGBM`, `XGBoost`, `ExtraTrees`) are isolated within each outer 5-fold `(A,B)`-GroupKFold split (automated $+100$ base-path perturbation delta $1.42 times 10^(-14)$), while stacking weights, residual scales, and isotonic blending are fitted to pooled base-model OOF predictions and evaluated externally on `03_val`. Coupling an empirical Butler--Ginley log-additive band-center model ($R^2 = 1.0000000000$, $"RMSE" = 2.90 times 10^(-7) "eV"$; explicit bandgap propagation $sigma_("VBM") = sigma_("CBM") = sigma_(E_g)/2$) with closed-form Gaussian hinge integration, `v2_PhysHurdle_Bayes` achieves `03_val` $"RMSE" = 4.0634$ (vs. $4.9489$ One-Hot FM and $4.2053$ `M4_PIMT_Physics_Only`), $"MAE"(V<1) = 2.9280$, and $100.0%$ Top-75 recall for $V < 0.5$ (`18/18`; `Rec@50 = 17/18 = 94.4%`). Across $20$ paired closed-loop seeds ($K = 200$ queries, analytical hypergeometric baseline $0.9070$), a lightweight online physics-hurdle surrogate yields $16.20/18$ ($"SD"=1.077$) under hybrid classical Top-20 pre-screening plus 8-bit QUBO acquisition versus $16.55/18$ ($"SD"=1.962$) for sequential ranking, supporting hybrid feasibility without claiming acquisition-level or quantum advantage. Conformal coverage across $200$ `(A,B)`-disjoint splits reaches $90.19%$ / $95.12%$, while `04_holdout` expectation sums ($sum hat(P)_("hit") = 10.3358$) are scoped strictly as deterministic model probability mass rather than unseen true-positive recall.

For Problems 2 and 3, we prove why naive 8-bit scaling of $52$-element one-hot blocks induces Hamming-weight drift ($0.0%$ feasibility) and resolve it via upper-tail capping ($C_("cap")$), lossless active-subspace pruning, and integer alignment ($P_("block") > 1.5 C_("cap")$). For three-body Problem 3, we introduce a zero-auxiliary composite `(A,X)-B` formulation (`Model 3A-Comp`, $26$ / $18$ spins), repair `Model 3A-HOBO` via cumulative auxiliary bounds and exact cardinality penalties ($43$ / $29$ spins), and repair `Model 3B-ANOVA` via uniform $8 times$ scaling ($33$ / $25$ spins, $6$ distinct legal energy levels). Exhaustive primary-variable enumeration after analytic auxiliary elimination—cross-checked by exact algebraic Ising-to-QUBO recovery—certifies *zero illegal global ground states* and *zero ground-state auxiliary ties* across all eight matrices; seven integer ground-state sets contain the continuous surrogate minimum `LaTiO2N` ($hat(V) = 0.252406$), while quantization degeneracy ($3$--$10$ states) remains.

Evaluated on real Bose Quantum `SPQC-550` hardware in `TaskMode.OPTIMIZATION` (`10` returned records per task), four blind-pool matrices achieve $30.0%$ (`v3_q2_blind_12`, $E^* = -1000$, best `SrTaO2N`), $90.0%$ (`v3_q3_comp_18`, $E^* = -1468$, hitting `LaTiO2N` at $0.000000$ regret and $9/15$ unique Top-15 hits), $20.0%$ strict / $90.0%$ primary (`v3_q3_hobo_29`, $E^* = -1662$), and $30.0%$ (`v3_q3_anova_25`, $E^* = -1992$) strictly feasible returned records. Independent offline audits verify bit-exact matrix hashes, raw spin arrays, decoded energies, and archived records, alongside $50$-run classical comparisons, $4$--$12$ bit sweeps, penalty transitions, $C_("cap")$ retention audits, and multi-scale perturbation stability tests.

#v(0.14cm)
*Keywords:* Perovskite Photocatalysis; Fold-Isolated Hurdle Surrogate; Butler--Ginley Band-Center Relation; 8-Bit Ising Compilation; Global Ground-State Certification; Coherent Ising Machine (`SPQC-550`)
]

#pagebreak()

// =============================================================================
// PAGE 2: MANDATORY DIRECTORY / TABLE OF CONTENTS
// =============================================================================
#outline(title: [Contents], indent: 1.4em, depth: 3)
#pagebreak()

// =============================================================================
// MAIN BODY (7 SECTIONS + REFERENCES, STRICTLY <= 25 PAGES)
// =============================================================================

= Task, Data & Operational Definitions

== Problem Context and Official Datasets
Discovering visible-light-active photocatalytic perovskites ($A B X_3$) across $52$ $A$-site metals, $52$ $B$-site metals, and $7$ anion stoichiometries ($"O"_3$, $"NO"_2$, $"FO"_2$, $"N"_2"O"$, $"N"_3$, $"O"_2"S"$, $"FNO"$) spans $52 times 52 times 7 = 18,928$ cubic crystals [1]. Because GLLB-SC Density Functional Theory (DFT) calculations are computationally expensive, the competition defines three coupled tasks:
- *Problem 1 (Q1 — Out-of-Distribution Surrogate Modeling):* Train composition-only predictors on `02_castelli_train.csv` ($1,608$ $(A, B)$ pairs, $11,256$ rows), perform model selection and validation on `03_castelli_val.csv` ($567$ unique $(A, B)$ pairs, $3,969$ rows), and score `04_castelli_test_holdout.csv` ($529$ unique $(A, B)$ pairs, $3,703$ rows). Crucially, the three splits have *zero $(A, B)$-pair overlap*, forcing strict `(A,B)`-grouped validation.
- *Problem 2 (Q2 — Bipartite $A B"NO"_2$ Quantum Optimization):* Fixing $X = "NO"_2$, formulate a QUBO/Ising Hamiltonian over $(A, B)$ to minimize surrogate violation $hat(V)$ within the 8-bit signed integer Ising constraints ($H_(u v) in bb(Z) inter [-127, 127]$) of the Bose Quantum `SPQC-550` Coherent Ising Machine (CIM) via Kaiwu SDK 1.4.4 [2, 3]. Both the full grid (`07_castelli_q2_pool.csv`, $2,704$ compounds) and blind holdout subset (`11_castelli_q2_pool_blind.csv`, $529$ compounds) are evaluated.
- *Problem 3 (Q3 — Multi-Anion Three-Body Optimization):* Jointly optimize $(A, B, X)$ over $X in {"O"_3, "NO"_2, "FO"_2}$ on both the full pool (`08_castelli_q3_pool.csv`, $8,112$ compounds) and blind pool (`12_castelli_q3_pool_blind.csv`, $1,587$ compounds), bridging cubic three-body interactions onto quadratic 8-bit Ising hardware.

Auditing all nine official contest files against `sha256_public.json` via `verify_dataset_checksums()` confirms *9/9 CRLF-normalized SHA-256 matches* and *1/9 raw-bytes match* in our Git LF checkout (the Markdown dataset README matches raw LF bytes, while the 8 official CSVs are checked out with `LF` endings and match `sha256_public.json` bit-for-bit upon LF-to-CRLF normalization), guaranteeing data integrity without content modification.

== Operational Definitions (`19` vs. `15`, Metallicity, and $A=B$ Sensitivity)
To eliminate definitional ambiguity, we fix four operational conventions:
1. *Metallicity (`gap = 0`):* Exactly $18,193$ of $18,928$ compounds ($96.12%$) exhibit $E_g = 0.0 "eV"$. These are physical metals rather than missing entries; each incurs a deterministic lower-gap hinge penalty $v_("gap,low") = (1.4 / 0.5)^2 = 7.84$.
2. *Target Hinge Score $V$ and `19` vs. `15` Boundary Sensitivity:* Across all $15,225$ labeled rows (`02 + 03`), recomputing the official five-term quadratic hinge score
   $ V &= [max(0, (e_("form") - 0.21)/0.5)]^2 + [max(0, (1.4 - E_g)/0.5)]^2 + [max(0, (E_g - 3.1)/0.5)]^2 \
       &quad + [max(0, (5.73 - "VBM")/0.5)]^2 + [max(0, ("CBM" - 4.50)/0.5)]^2 $ <eq:vscore>
   matches the official `V` column to $max |Delta V| = 1.02 times 10^(-5)$. Under this non-strict $V = 0$ hinge definition ($e_("form") <= 0.21$, $1.4 <= E_g <= 3.1$, $"VBM" >= 5.73$, $"CBM" <= 4.50$), there are *$19$* positive targets ($0.1004%$: `O3`: $6$, `NO2`: $6$, `FO2`: $6$, `N2O`: $1$), all residing on $19$ distinct $(A, B)$ pairs in `04_test_holdout`, with *zero* $V = 0$ rows in `02_train` (`49` with $V < 0.5$, `83` with $V < 1.0$) or `03_val` (`18` with $V < 0.5$, `27` with $V < 1.0$). By contrast, the problem statement text also discusses *$15$* targets satisfying strict interior inequalities ($e_("form") < 0.21$, $1.4 < E_g < 3.1$). We adopt the official hinge score $V = 0$ ($19$ targets) as our primary computational objective and explicitly document the boundary distinction without fabricating unseen `04` ground-truth labels.
3. *Homonuclear Pairs (`a_eq_b = 1`):* Official rules permit retaining or excluding $A = B$ rows (364/18,928 $= 1.92%$, containing $0$ positives). We retain $A = B$ during Q1 surrogate training for full-grid continuity while energetically forbidding $a_i b_i = 1$ in Q2/Q3 QUBOs. A direct retain-vs.-exclude audit shows that excluding $A = B$ leaves the global surrogate minimum unchanged at `LaTiO2N` ($hat(V) = 0.252406$) across `04 (02+03)`, `07`, and `08`, leaves `04` Top-20 identical (20/20), and affects only one rank in `04` Top-50 (`SnSnO3`, $hat(V) = 2.4358$, rank $32$).
4. *Dual Holdout Lineage (`02-only` vs. `02+03` Refit):* All Q1 validation benchmarks on `03_val` strictly use `02-only` training. For downstream Q2/Q3 screening, we refit the locked architecture on `02 + 03` ($15,225$ rows) while exporting *both* `v2_holdout_04_train02only.csv` and `v2_holdout_04.csv` (Spearman $rho = 0.99704$, Top-10 overlap 9/10, Jaccard $0.8182$; `sum(P_hit)` $11.0008$ vs. $10.3358$).

#figure(
  image("../v2_optimized/figures/schematic01_pipeline_architecture.png", width: 99%),
  caption: [End-to-end architecture: dataset dual SHA-256 verification, strictly fold-isolated Q1 surrogate, upper-tail active-subspace pruning, exact 8-bit QUBO/Ising compilation, exhaustive primary-variable certification, and equal-budget classical/`SPQC-550` auditing.],
) <fig:pipeline>

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (3.55cm, 1.85cm, 2.95cm, 2.45cm, 2.10cm, 1.45cm, 2.65cm),
      stroke: none,
      align: (left, left, left, left, left, left, left),
      inset: (x: 2.5pt, y: 2.8pt),
      table.hline(stroke: 1.1pt),
      [*Model / Artifact ID*], [*Train Set*], [*Selection / OOF Protocol*], [*Calib. Set*], [*Pools*], [*Label Type*], [*SHA-256 (16 hex)*],
      table.hline(stroke: 0.6pt),
      [`v1_OneHot_FM_k8`], [`02` (11,256)], [Fixed FM (`k=8, lr=0.03`)], [None], [`03` (3,969)], [Scalar $V$], [`14ef801edd70fffc`],
      [`v1_Direct_LGBM_V`], [`02` (11,256)], [Fixed LGBM (`23` desc.)], [None], [`03` (3,969)], [Scalar $V$], [`14ef801edd70fffc`],
      [`v1_Direct_LGBM_log1pV`], [`02` (11,256)], [Fixed LGBM on `log1p(V)`], [None], [`03` (3,969)], [Scalar $V$], [`14ef801edd70fffc`],
      [`M4_PIMT_Physics_Only`], [`02` (11,256)], [Archived `02 -> 03` ablation], [None], [`03` (3,969)], [Multi-prop.], [`table1_ablation`],
      [`M5_PIMT_Hybrid_Ensemble`], [`02` (11,256)], [Archived `02 -> 03` ablation], [None], [`03` (3,969)], [Multi-prop.], [`table1_ablation`],
      [`v2_Direct_4FamPhys`], [`02` (11,256)], [Outer 5-fold `(A,B)`-GKF], [None], [`03`, `04`], [Scalar $V$], [`ac5d939d4dd63f3a`],
      [`v2_PhysHurdle_Bayes (02)`], [`02` (11,256)], [Outer 5-fold base GKF (`1.42e-14`)], [Pooled base OOF; `03` 200 splits], [`03`, `04_02only`], [Multi-prop.], [`2090c512ddc3e614`],
      [`v2_PhysHurdle_Bayes (02+03)`], [`02+03` (15,225)], [Outer 5-fold base GKF], [Pooled base OOF (no transfer)], [`04`, `07`, `11`, `08`, `12`], [Multi-prop.], [`89eca9b89f850123`],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Data & Model Protocol Lineage (Unified Table 1): complete provenance across `v1`, archived `M4`/`M5` baselines, and fold-isolated `v2` models (full 64-hex SHA-256 digests in Appendix B).],
) <tab:lineage>

== Modeling Assumptions and Principal Notations
1. *Composition-Only Blind Inference:* On `04_test_holdout` and pools `07`/`08`/`11`/`12`, only stoichiometric symbols $(A, B, X)$ and universal Periodic Table constants [5, 6] are used.
2. *Approximate Gaussian Marginals in Mixture Integration:* Under `integrate_bayes_mixture`, formation energy and semiconducting bandgap residuals are modeled as conditionally independent Gaussians with empirical $0.75 / 0.25$ variance blending between OOF RMS and ensemble spread; this closed-form expectation is a surrogate loss estimator rather than an exact joint posterior.
3. *Symmetric Band-Edge Coupling:* $"VBM" = E_c + E_g/2$ and $"CBM" = E_c - E_g/2$ hold across all $15,225$ labeled rows to CSV rounding precision (median $|("VBM"-"CBM")-E_g| = 0.0000 "eV"$, max $0.0500 "eV"$).
4. *Conditional Upper-Tail Capping at $C_("cap")$:* Clipping $hat(V)$ at $C_("cap")$ preserves the exact ordering of all candidates satisfying $hat(V) < C_("cap")$ and guarantees retention of the declared pool minimum whenever $min hat(V) < C_("cap")$, while explicitly trading off higher-violation Top-25 candidates (audited in Section 6).
5. *Hardware Integer Gate:* `SPQC-550` requires symmetric zero-diagonal integer couplings $H_(i j) in bb(Z) inter [-127, 127]$ ($<= 8$ bits), $N <= 550$, and explicit energy sorting $E(s) = -s^top H s$ [2].

= Physics-Coupled Surrogate Model (Q1)

== Empirical Band-Center Relation, Quantified Window & Feature Audit
Analyzing the GLLB-SC electronic targets in `02_train` reveals a highly regular data-level structure rooted in Castelli et al.'s Butler--Ginley geometric-mean band-center formulation [1, 4]:

#theorem-box("Empirical Proposition 1 (In-Data Butler–Ginley Relation & Error Propagation)", [
  On the GLLB-SC dataset, the band center $E_c equiv ("VBM"+"CBM")/2$ is modeled in log-space by elemental Ridge regression (`alpha = 1e-10`) strictly inside each outer training fold:
  $ ln E_c (A, B, X) = w_0 + w_A + w_B + w_X. $ <eq:bg>
  Evaluated on `03_val` ($567$ `(A,B)` pairs, $3,969$ rows), @eq:bg achieves *$R^2 = 1.0000000000$* and *$"RMSE" = 2.90 times 10^(-7) "eV"$*, with site-exchange symmetry ($r(w_A, w_B) = 1.00000000$, $op("std")(w_A - w_B) = 0.00000000$) and Mulliken electronegativity correlation $r = 0.9183$ ($rho = 0.9295$). Crucially, because $"VBM" = E_c + E_g/2$ and $"CBM" = E_c - E_g/2$, accurate $E_c$ *does not* remove VBM/CBM prediction uncertainty: bandgap prediction error propagates directly into both band edges with *$sigma_("VBM") = sigma_("CBM") = sigma_(E_g)/2$*.
])

#theorem-box("Proposition 2 (Existence Condition for Electronic Band Window)", [
  Under symmetric band-edge coupling ($"VBM" = E_c + E_g/2$, $"CBM" = E_c - E_g/2$) and the non-strict $V = 0$ hinge bounds ($1.4 <= E_g <= 3.1$, $"VBM" >= 5.73$, $"CBM" <= 4.50$), let $g_(min)(E_c) equiv max(1.4, 2(5.73 - E_c), 2(E_c - 4.50))$. Then there *exists* a band gap $E_g >= 0$ satisfying all electronic criteria simultaneously if and only if
  $ [ exists E_g >= 0 : g_(min)(E_c) <= E_g <= 3.1 ] <==> 4.18 "eV" <= E_c <= 6.05 "eV". $ <eq:ecwin>
  For any given material's actual band gap $E_g$, electronic feasibility still requires checking $g_(min)(E_c) <= E_g <= 3.1$, while strict interior feasibility of the $15$-target subset requires all four strict inequalities ($e_("form") < 0.21$, $1.4 < E_g < 3.1$, $"VBM" > 5.73$, $"CBM" < 4.50$). Condition @eq:ecwin is *necessary* rather than sufficient: a metallic compound with $E_c = 5.10 "eV"$ and $E_g = 0$ lies inside $[4.18, 6.05] "eV"$ yet violates $E_g >= g_(min)(5.10) = 1.4$ and incurs $v_("gap,low") = 7.84$.
])

#figure(
  image("../v2_optimized/figures/schematic02_band_alignment_window.png", width: 99%),
  caption: [Conceptual schematic of symmetric band-edge coupling ($sigma_("VBM") = sigma_("CBM") = sigma_(E_g)/2$), the metallic counterexample ($E_c = 5.10 "eV", E_g = 0$), and the existential $(E_c, E_g)$ feasibility corridor $4.18 <= E_c <= 6.05 "eV"$.],
) <fig:bandwindow>

Every physical descriptor in `common_physics.py` is supplied as an input feature (never as an irreversible hard filter) to avoid excluding true exceptions. @tab:phys-audit audits each descriptor family's theoretical origin, approximations, and empirical exception counts across `02+03` ($n = 15,225$) and near-target compounds ($V < 1.0$: $n = 110$; $V < 0.5$: $n = 67$).

#figure(
  table(
    columns: (auto, 1.1fr, 1.4fr, auto, 1.3fr),
    stroke: none,
    align: (left, left, left, center, left),
    inset: (x: 2.8pt, y: 3.2pt),
    table.hline(stroke: 1.1pt),
    [*Feature / Relation*], [*Theoretical Origin*], [*Validity & Approximations*], [*Hard Filter?*], [*Empirical Exceptions (`02+03`)*],
    table.hline(stroke: 0.6pt),
    [Butler--Ginley $E_c$], [$ln E_c = w_0 + w_A + w_B + w_X$ [1, 4]], [In-data Ridge (`alpha=1e-10`); $sigma_("VBM")=sigma_("CBM")=sigma_(E_g)/2$], [No], [`03_val` RMSE $2.90 times 10^(-7)$ eV; $4.18 <= E_c <= 6.05$ holds for 101/110 ($V<1$)],
    [Hurdle & Window], [$g_(min)(E_c)$, Gaussian hinge [Eq. @eq:vbayes]], [Gaussian marginals; conditional independence in $P_("hit")$], [No], [`03` semi residual corr $r=-0.2930$ ($p=4.22 times 10^(-4)$)],
    [Electron Parity], [Bloch--Wilson band filling ($Z mod 2$)], [Spin-unpolarized closed-shell heuristic; fails on Mott/magnetic insulators], [No], [`Z_parity==1` (odd): 0/110 ($V<1$), 0/67 ($V<0.5$), 5/573 ($E_g>0$); even `==0`: 568/573],
    [Charge & Closed-Shell], [Pauling balance $min |q_A+q_B+q_X|$], [Curated `ELEMENT_DATA` tuples; `Fe/Co/Ni/Ru` lack closed-shell entries], [No], [`is_chg_bal==0`: 3/110 ($V<1$); `is_cs_bal==0`: 4/110],
    [Bartel SISSO $tau$], [$tau = r_X/r_B - n_A (n_A - (r_A/r_B)/(ln(r_A/r_B)))$ [7]], [Valid only for $r_A/r_B > 1.05$; 6-coord $r_A$; logistic $p_("Bartel")$ is heuristic], [No], [`bartel_valid==0`: 26/110 ($V<1$); $tau >= 4.18$: 87/110],
    [$d^0/d^(10)$ & Pauling/Mulliken], [Group $d^0/d^(10)$ & $I = 1 - e^(-0.25 Delta chi^2)$ [5, 6]], [Group-based proxy for $d^0/d^(10)$; `s0_p_mag` captures $|mu_B|>0.05$], [No], [`B_is_d0_d10==0`: 1/110 (`ZrMgO3`); $|mu_B|>0.05$: 22/573 ($E_g>0$)],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Physical feature audit: theoretical foundations, approximations, non-filtering policy, and measured empirical exception counts across `02+03` ($n=15,225$).],
) <tab:phys-audit>

== Outer-Fold Base Isolation, Stage-0 DFT & Bayes Mixture Surrogate
In `v2_surrogate_q1.py`, we engineer $151$ numeric and $3$ categorical descriptors ($154$ total, vs. $23$ in `v1`): $100$ crystal-chemistry descriptors, $7$ analytical $E_c$ features, $40$ empirical-Bayes ($m=5$) inner-OOF marginal statistics, and $4$ Stage-0 LightGBM latent DFT predictors (`a_lat`, `fermi_level`, `|mu_b|`, and magnetic switch `P(|mu_b|>0.05)`).

Target-dependent feature construction (Butler--Ginley Ridge weights, target-encoding priors, inner marginal features, and Stage-0 DFT models) and the four-family base predictors (`CatBoost`, `LightGBM`, `XGBoost`, `ExtraTrees`) are isolated within each outer `(A,B)` training fold. Stacking weights, residual scales, and the `50/50` isotonic blend are subsequently fitted to pooled base-model outer-OOF predictions. Complete end-to-end generalization is evaluated externally on the public `03_castelli_val.csv` split (note that `03_val` is the contest validation split rather than an untouched private test set). Metrics evaluated on those same pooled OOF labels after meta-fitting are development diagnostics rather than doubly cross-fitted estimates of the full pipeline, and Stage-0 inner-fold summaries (`s0_a_lat` $R^2 = 0.9603$, `s0_fermi` $R^2 = 0.9320$, `s0_abs_mub` $R^2 = 0.6133$, `s0_p_mag` $"ROC-AUC" = 0.8539$ on `02_train`) serve as internal development diagnostics because outer-train `te_*` encodings precede Stage-0's inner split. An automated $+100$ label-perturbation test on `fit_fold_isolated_pipeline` verifies isolation of the base feature and base-model path (`group_overlap = 0`, `max_feature_delta = 0.0`, `max_pred_delta = 1.42e-14 < 1e-12`).

Given Gaussian marginals $Y tilde.op cal(N)(mu, sigma^2)$ and hinge penalty $[max(0, plus.minus(Y - theta)/s)]^2$, setting $m = plus.minus(mu - theta)/s$, $tilde(sigma) = sigma/s$, and $z = m/tilde(sigma)$ gives the closed-form hinge expectation
$ cal(E)_plus.minus (mu, sigma; theta, s) = (m^2 + tilde(sigma)^2) Phi(z) + m tilde(sigma) phi(z). $ <eq:hinge>
Combining 4-family simplex ensembles for $e_("form")$ (weights $0.315$ CB, $0.317$ LGBM, $0.258$ XGB, $0.111$ ET; `03_val` $R^2 = 0.9199$, $"RMSE" = 0.2108$), calibrated $p_("semi") = P(E_g > 0)$ ($0.264$ CB, $0.174$ LGBM, $0.183$ XGB, $0.380$ ET, $50/50$ isotonic blend; $"ROC-AUC" = 0.9850$, $"PR-AUC" = 0.8247$, Brier $= 0.01334$, $"ECE" = 0.00570$), and conditional $E_g | E_g > 0$ ($0.451$ CB, $0.374$ LGBM, $0.000$ XGB, $0.174$ ET; $R^2 = 0.7587$, $"RMSE" = 0.7922$) yields
$ hat(V)_("Bayes")(x) = cal(E)_+ (hat(mu)_("ef"), hat(sigma)_("ef"); 0.21, 0.5) + sum_(r in {"gl","gh","VBM","CBM"}) [(1 - hat(p)_("semi")) v_r (0; E_c) + hat(p)_("semi") cal(E)_r (hat(mu)_g, hat(sigma)_g; E_c)], $ <eq:vbayes>
$ hat(P)_("hit")(x) = hat(p)_("semi")(x) dot Phi((0.21 - hat(mu)_("ef")) / hat(sigma)_("ef")) dot max(0, Phi((3.1 - hat(mu)_g) / hat(sigma)_g) - Phi((g_(min)(E_c) - hat(mu)_g) / hat(sigma)_g)). $ <eq:phit>

== Quantitative Benchmark on `03_val` (`M4`/`M5` Alignment & Near-Target Trade-Off)
@tab:q1 compares all seven surrogates strictly under the `02 -> 03` protocol (`567` `(A,B)` pairs / `3,969` rows), incorporating the archived `M4_PIMT_Physics_Only` and `M5_PIMT_Hybrid_Ensemble` baselines from `table1_surrogate_ablation.csv` (which do not archive per-row `03_val` CSVs, so their local near-target MAE is marked `N/A`). On `03_val`, `v2_PhysHurdle_Bayes` achieves the lowest global RMSE ($4.0634$, $-3.37%$ vs. `M4`, $-17.89%$ vs. `v1_OneHot_FM_k8`), lowest global MAE ($2.0696$), highest $R^2$ ($0.8632$), highest Spearman $rho$ ($0.9527$), and cuts near-target $"MAE"(V<1)$ from $5.4475$ (`v2_Direct_4FamPhys`) and $5.8008$ (`v1_Direct_LGBM_log1pV`) down to *$2.9280$* ($"MAE"(V<0.5) = 2.4126$), capturing $100.0%$ of $V < 0.5$ targets at Top-75 (`18/18`, `Rec@50 = 94.44%`, `Prec@50 = 34.00%`) and $100.0%$ of $V < 1.0$ targets at Top-100 (`27/27`). Relative to our earlier un-isolated `02->03` prototype (`RMSE = 4.0675`, `MAE(V<1) = 2.6167`, `Rec@50(V<0.5) = 15/18 = 83.3%`), enforcing strict outer-fold base isolation improves Top-50 rare-target capture by $+2$ hits (`17/18`) and slightly lowers global RMSE, at the explicit trade-off of a $+11.9%$ increase in local `MAE(V<1)` from variance-blended hinge calibration. Refactoring the fold-isolated pipeline and diagnostic checks utilized AI coding assistance [13] verified by our base-path perturbation test.

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (3.75cm, 1.10cm, 1.10cm, 1.00cm, 1.00cm, 1.40cm, 1.45cm, 1.55cm, 1.55cm, 1.55cm, 1.55cm),
      stroke: none,
      align: (left, right, right, right, right, right, right, right, right, right, right),
      inset: (x: 2.0pt, y: 2.8pt),
      table.hline(stroke: 1.1pt),
      [*Model*], [*RMSE*], [*MAE*], [*$R^2$*], [*$rho$*], [*MAE ($V<1$)*], [*MAE ($V<0.5$)*], [*Rec\@20 ($V<1$)*], [*Rec\@50 ($V<1$)*], [*Rec\@50 ($V<0.5$)*], [*Rec\@75 ($V<0.5$)*],
      table.hline(stroke: 0.6pt),
      [`v1_OneHot_FM_k8`], [4.9489], [2.8536], [0.7970], [0.9217], [10.4171], [9.9826], [7.4%], [29.6%], [33.3%], [33.3%],
      [`v1_Direct_LGBM_log1pV`], [4.6942], [2.7469], [0.8174], [0.9292], [5.8008], [5.2440], [29.6%], [63.0%], [72.2%], [72.2%],
      [`v1_Direct_LGBM_V`], [4.5568], [2.5383], [0.8279], [0.9365], [9.9202], [9.5674], [29.6%], [44.4%], [50.0%], [66.7%],
      [`M4_PIMT_Physics_Only`], [4.2053], [2.2068], [0.8534], [0.9498], [N/A], [N/A], [51.9%], [77.8%], [N/A], [N/A],
      [`M5_PIMT_Hybrid_Ensemble`], [4.2091], [2.2180], [0.8532], [0.9493], [N/A], [N/A], [51.9%], [77.8%], [N/A], [N/A],
      [`v2_Direct_4FamPhys`], [4.1834], [2.2420], [0.8550], [0.9472], [5.4475], [5.0721], [37.0%], [*85.2%*], [88.9%], [94.4%],
      [*`v2_PhysHurdle_Bayes`*], [*4.0634*], [*2.0696*], [*0.8632*], [*0.9527*], [*2.9280*], [*2.4126*], [*55.6%*], [81.5%], [*94.4%*], [*100.0%*],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Surrogate Comparison on `03_castelli_val.csv` (Unified Table 2, $n = 3,969$ rows / `567` `(A,B)` pairs). `v2_PhysHurdle_Bayes` achieves `Rec@100(V<1) = 100.0% (27/27)` and cuts near-target MAE nearly in half vs. direct regression (`v2_Direct_4FamPhys`). Runtime: `424.2s` validation, `402.4s` refit, `168.3s` isolation audit (`996.7s` total vs. `2.77s` for `v1_baseline.py`).],
) <tab:q1>

#figure(
  image("../v2_optimized/figures/fig01_v1_vs_v2_surrogate_metrics.png", width: 98%),
  caption: [Surrogate validation on `03_val` (`567` pairs, `3,969` rows): (a) global RMSE and MAE comparison including archived `M4`/`M5`; (b) near-target Recall\@50 and Recall\@100 ($V < 1.0$, `27` targets); (c) near-target local MAE ($V < 1.0$ and $V < 0.5$, excluding archived `M4`/`M5` without per-row CSVs); (d) in-data Butler--Ginley parity ($R^2 = 1.0000000000$, $"RMSE" = 2.90 times 10^(-7) "eV"$).],
) <fig:q1metrics>

#figure(
  image("../v2_optimized/figures/fig02_residual_and_calibration.png", width: 98%),
  caption: [Residual and calibration diagnostics on `03_val`: (a) near-zero overall bias ($-0.0328$; metal $"RMSE" = 3.7182$, semi $"RMSE" = 9.4570$); (b) MAE across $E_c$ bins; (c) 10-bin reliability diagram for $hat(p)_("semi")$ ($"ROC-AUC" = 0.9850$, $"ECE" = 0.00570$).],
) <fig:q1calib>

== 20-Seed Closed-Loop Active Screening (`03_val`, Analytical & Hybrid-QUBO Baselines)
To evaluate iterative discovery under a fixed $K = 200$ simulated DFT query budget ($50$ random initial samples $+ 15$ rounds $times 10$ queries on `03_val`, $N = 3,969$, $M = 18$ for $V < 0.5$), we run $20$ paired random seeds comparing six strategies under explicit oracle and candidate-pool protocols:
- *Analytical Hypergeometric Baseline & `A=B` Pool Eligibility:* Sampling $K = 200$ rows uniformly without replacement yields expectation $bb(E)[H_(0.5)(200)] = 200 times 18 / 3969 = 0.9070 plus.minus 0.9261$ ($5.04%$); `Random` empirical mean is ($0.60 plus.minus 0.663$)/18 ($3.33%$). `03_val` contains $77$ homonuclear `A=B` rows (none with $V < 1$; min $V = 6.6537$); initial `50`-sample pools and `Random`/`SeqRank` sample across all `3,969` rows, whereas `v2_Real_8Bit_QUBO` excludes `A=B` during QUBO decoding (on the `3,892` non-homonuclear subset, analytical uniform expectation is `0.9250`).
- *Converged `L-BFGS-B` FM Baselines (Scalar-$V$ Oracle):* A 2nd-order One-Hot FM trained to convergence via `L-BFGS-B` (`300/300` logged converged fits on `v1_FM_LBFGS_SeqRank`) achieves ($4.30 plus.minus 2.027$)/18 ($23.89%$, $4.74 times$ analytical HG) under classical `argsort` versus ($1.20 plus.minus 0.980$)/18 under $112$-spin FM-QUBO SA (`v1_FM_LBFGS_QUBO`, calling the identical optimizer without archived per-round convergence flags, `0` fallbacks).
- *Same-Surrogate Closed-Loop Comparison (Lightweight Online Hurdle):* Under scalar-$V$ labels, physics-featured LightGBM (`v2_PhysLGBM_ScalarV_SeqRank`) attains ($12.25 plus.minus 3.433$)/18 ($68.06%$). Using multi-property DFT labels ($e_("form"), E_g, "VBM", "CBM"$) in a lightweight online physics-hurdle surrogate (Ridge $E_c$ + LightGBM hurdle with fixed $sigma_("ef")=0.25, sigma_g=0.80$, without Stage-0 or 4-family stacking), `v2_Real_8Bit_QUBO`—a hybrid classical Top-20 (or Top-10) pre-screening plus per-round 8-bit `Model 3A-Comp` QUBO strategy (`300/300` exact 8-bit matrices, mean $28.77$ spins, $0$ fallbacks)—finds $16.20$ targets per $200$ queries ($"SD" = 1.077$) versus $16.55$ ($"SD" = 1.962$) for classical sequential ranking (`v2_Classical_SeqRank`; $-0.35$ targets). Its observed across-seed dispersion was lower, supporting the feasibility of the hybrid acquisition without establishing an acquisition-level or quantum advantage.

#figure(
  image("../v2_optimized/figures/fig03_closed_loop_fmqa_benchmark.png", width: 97%),
  caption: [20-seed closed-loop simulated DFT screening on `03_val` ($N = 3,969$, $K = 50..200$): comparison against the analytical hypergeometric baseline (`0.9070`), converged `L-BFGS-B` FM (`SeqRank` & `QUBO`), scalar-$V$ `v2_PhysLGBM`, classical sequential `v2_Classical_SeqRank` (`16.55/18`, SD `1.962`), and hybrid Top-20 + 8-bit QUBO `v2_Real_8Bit_QUBO` (`16.20/18`, SD `1.077`, 300/300 exact 8-bit).],
) <fig:closedloop>

== Group-Disjoint Conformal Calibration & Scoped `04_holdout` Yield
We run $200$ `(A,B)`-group-disjoint 50/50 Mondrian conformal splits on `03_val` ($283$ calibration vs. $284$ test `(A,B)` groups, `group_overlap = 0`), stratified by predicted probability $hat(p)_("semi") >= 0.20$: empirical coverage is $90.19% plus.minus 1.65%$ (low-$hat(p)_("semi")$ $90.17%$, high-$hat(p)_("semi")$ $90.66%$) at nominal $90%$, and $95.12% plus.minus 1.13%$ at nominal $95%$. Crucially, $90%$ conformal radii reveal a sharp stratification gap—$4.411 plus.minus 0.252$ on the predicted low-probability stratum ($hat(p)_("semi") < 0.20$, $n = 3,808$, containing $29$ true semiconductors) versus $12.375 plus.minus 0.848$ (median $12.044$) on the predicted semiconducting/borderline stratum ($hat(p)_("semi") >= 0.20$, $n = 161$, containing $112$ true semiconductors; $95%$: $7.004$ vs. $18.014$)—showing that conformal bands on semiconducting/borderline candidates remain wide relative to $V < 0.5$ and cannot be transferred as formal guarantees to `02+03` refitted predictions or adaptive Top-$K$ subsets. Additionally, on `03_val` semiconductors ($n = 141$), `e_form` and `gap` residuals are negatively correlated (Pearson $r = -0.2930$, Spearman $rho = -0.3349$), and `N(hat(mu)_g, hat(sigma)_g^2)` is an untruncated Gaussian approximation to $E_g | E_g > 0$ (`P(E_g < 0)` median `0.1743` across `04_holdout`, max `0.0110` among Top-50 by $hat(V)$).

Accordingly, on `04_test_holdout` ($3,703$ rows, `02+03` refit), the deterministic total model score sum $sum hat(P)_("hit") = 10.3358$ (Top-30: $8.6001 = 83.2%$, Top-50: $9.4889 = 91.8%$; anion breakdown `NO2`: $4.48$, `O3`: $4.48$, `FO2`: $0.77$, `O2S`: $0.43$, `N2O`: $0.13$) summarizes *model-assigned probability mass* rather than unseen true-positive recall against the $19$ hidden `V=0` targets. Under an auxiliary independent-Bernoulli count approximation (ignoring cross-candidate covariance and unconditioned on `19`), the model-internal hit-count SD would be $sqrt(sum p_i (1 - p_i)) = 2.30$, which is not an uncertainty bar on deterministic $sum hat(P)_("hit")$.

#figure(
  image("../v2_optimized/figures/fig08_electronic_phase_and_conformal_audit.png", width: 98%),
  caption: [Holdout (`04`, `02+03` refit) and electronic audit: (a) $(E_c, hat(e)_("form"))$ phase map; (b) scoped cumulative $sum hat(P)_("hit")$ model expectation curve ($8.6001/10.3358 = 83.2%$ at $K=30$, not unseen true recall); (c1--c4) four-property target-band audit of Top-10 blind stars.],
) <fig:conformal>

= Problem 2: 8-Bit Bipartite QUBO & Active Subspace

== Sparse Masked-Pool QUBO & Correctness Theorems
Fixing $X = "NO"_2$, let one-hot binary vectors $bold(a) in {0,1}^(N_A)$ and $bold(b) in {0,1}^(N_B)$ select cations $(A_i, B_j)$. To avoid the $31.78$-bit dynamic range of raw scores and enforce pool eligibility inside the Hamiltonian, we define the upper-tail capped, zero-shifted objective matrix
$ macron(M)_(i j) = cases(min(hat(V)_(i j), C_("cap")) - C_("cap") in [-C_("cap"), 0] & "if " (A_i, B_j) in cal(P) " and " A_i != B_j, 0 & "otherwise") $ <eq:mbar>
and build the penalized bipartite QUBO
$ cal(H)_("Q2")(bold(a), bold(b)) = sum_(i,j) macron(M)_(i j) a_i b_j + P_("block") (sum_i a_i - 1)^2 + P_("block") (sum_j b_j - 1)^2 + sum_((i,j) in.not cal(P)_("valid")) P_("block") a_i b_j. $ <eq:q2qubo>

Converting @eq:q2qubo to Kaiwu's symmetric zero-diagonal Ising convention $E(s) = -s^top H s$ via $x_i = (1 + s_i s_("ref"))/2$ introduces one reference spin $s_("ref")$. On an $N$-variable one-hot block, couplings satisfy $H_(i j)^("pen") = -P/4$ and $H_(i,"ref")^("pen") = -(N-2)P/4$, causing severe precision sensitivity:

#theorem-box("Theorem 1 (8-Bit One-Hot Quantization Drift)", [
  Under naive $B$-bit scaling ($M_B = 2^(B-1)-1$) of an $N$-variable one-hot block, rounding $J_B = op("round")(M_B / (N-2))$ distorts $h_("pen") / J_("pen") = N - 2$, shifting the penalty-only minimum over Hamming weight $k = sum_i a_i$ to $k^* = N/2 - M_B/(2 J_B)$. For $N = 52$ on 8-bit hardware ($M_8 = 127$), $J_8 = 3$ shifts $k^*$ from $1$ to $4.83$ (or $0$-hot under positive offsets), driving naive 8-bit `v1` feasibility to $0.0%$.
])
*Proof.* Under $E(s) = -s^top H s$, the Hamming-weight-$k$ one-hot penalty energy is $E_("pen")(k) = J_B [(2k - N)^2 - N] + 2 M_B (2k - N)$. Setting $(d E_("pen")) / (d k) = 8 J_B (k - N/2) + 4 M_B = 0$ yields $k^* = N/2 - M_B/(2 J_B)$. #h(1fr) $square$

#theorem-box("Theorem 2 (Global Bipartite Penalty Bound & Exact Integer Alignment)", [
  1. For $macron(M)_(i j) in [-C_("cap"), 0]$ with a finite feasible pool minimum $H_("feas")^* <= 0$, whenever *$P_("block") > (3)/(2) C_("cap")$*, every infeasible state with $(k_A, k_B) != (1, 1)$ is strictly higher in energy than *at least one* feasible $(1, 1)$ state (specifically $H_("feas")^*$), so every global minimizer is feasible ($3/2$ is the tight worst-case sufficient bound over this matrix class, attained at $k_A = k_B = 2$; an infeasible state need not exceed *all* feasible states, e.g., on a $2 times 2$ block of $-1$s with $C_("cap") = 1, P_("block") = 1.6$ and a separate $0$-cost feasible state, $-4 + 3.2 = -0.8 < 0$ yet $> -1$).
  2. Quantizing $m_(i j) = op("round")(M_(max) macron(M)_(i j)/C_("cap")) in [-M_(max), 0]$ with $Q_("obj") = 8 m_(i j)$, integer penalty quantum $j_("pen") = ceil(2 alpha_P M_(max))$ ($alpha_P = 2.25$), and $P_("block") = 4 j_("pen")$ ensures every Ising entry $H_(u v)$ in @eq:ising-recovery is an exact integer with zero rounding drift ($k_A^* = k_B^* = 1$), satisfying hardware admissibility whenever $max_(u,v) |H_(u v)| <= 127$.
])
*Proof.* For $k_A >= 1, k_B >= 1$ with $(k_A, k_B) != (1, 1)$, at most $k_A k_B$ objective terms are active (each $>= -C_("cap")$), while the best feasible state achieves at most $0$ (if only $k_A k_B = 1$ or general $-C_("cap")$ vs $-H_("feas")^*$: any active feasible cell yields $-H_("feas")^* <= C_("cap")$, so excess gain over the minimum feasible state is at most $(k_A k_B - 1) C_("cap")$). Comparing against the penalty $P_("block")[(k_A - 1)^2 + (k_B - 1)^2]$ gives strict dominance iff $P_("block") > max_((k_A, k_B)!=(1,1)) (k_A k_B - 1) C_("cap") / [(k_A - 1)^2 + (k_B - 1)^2] = (4 - 1)/(1 + 1) C_("cap") = 1.5 C_("cap")$ (achieved at $k_A = k_B = 2$; $0$-hot states give $0 > H_("feas")^*$). Because all $Q$ entries are multiples of $4$ (off-diagonal objective terms are multiples of $8$), algebraic mapping (@eq:ising-recovery) yields exact integers $H_(i j) = -Q_(i j)/4$ and $H_(i,"ref") = -1/4 sum_j Q_(i j)$ with zero truncation error whenever $max_(u,v) |H_(u v)| <= 127$. #h(1fr) $square$

#theorem-box([Theorem 3 (Lossless Active-Subspace Embedding below $C_("cap")$)], [
  Any element $i$ with $min_j M_(i j) >= C_("cap")$ satisfies $macron(M)_(i j) = 0$ for all $j$ and cannot attain $hat(V) < C_("cap")$. Pruning inactive rows and columns preserves the *exact* optimal set below $C_("cap")$, reducing spins from $105$ to *$17$* on Full Pool `07` ($C_("cap") = 2.0, M_(max) = 4, P_("block") = 72.0$) and *$12$* on Blind Pool `11` ($C_("cap") = 2.5, M_(max) = 4, P_("block") = 72.0$).
])

== Certified Q2 Results & Blind Top-10 Cost Breakdown
Exhaustive $2^(n_("prim"))$ enumeration confirms `illegal_ground_count = 0` on both Q2 matrices: `Q2_full_07` ($65,536$ states, `cert_min_ising_energy = -1558`, $C = 1390$, degeneracy $10$) and `Q2_blind_11` ($2,048$ states, `cert_min_ising_energy = -1000`, $C = 824$, degeneracy $4$). Both matrices recover the exact declared-pool optimum **`LaTiO2N`** ($hat(V) = 0.252406$, continuous `regret = 0.000000`), tied in 8-bit integer energy ($M_(max) = 4$) with `SrTaO2N` ($0.257617$, regret $0.005211$), `BaTaO2N` ($0.261166$, regret $0.008760$), and `CaTaO2N` ($0.265979$, regret $0.013573$). @tab:q2top10 lists the Q2 Blind (`11`) Top-10 five-term violation decomposition, and @fig:q2 contrasts the $4$--$12$ bit quantization sweep against naive `v1` QUBOs.

#figure(
  image("../v2_optimized/figures/fig04_q2_v1_vs_v2_hardware_drift.png", width: 98%),
  caption: [Problem 2 results: (a) independent $4$--$12$ bit quantization sweep on `Q2_blind_11` (`v1` flatlines at $0.0%$ for $<12$ bits; `v2` holds $100.0%$ SA feasibility from $5$--$12$ bits and exact $M_(max)(B)$ integers from $7$--$12$ bits); (b) five-term physical cost decomposition of Q2 Blind Top-10 candidates.],
) <fig:q2>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    align: (center, left, right, right, right, right, right, right, right, right, right),
    inset: 2.9pt,
    table.hline(stroke: 1.1pt),
    [*Rank*], [*Formula*], [*$hat(V)$*], [*$hat(P)_("hit")$*], [*$hat(E)_g$*], [*$E_c$*], [*$hat(v)_("form")$*], [*$hat(v)_("gl")$*], [*$hat(v)_("gh")$*], [*$hat(v)_("VBM")$*], [*$hat(v)_("CBM")$*],
    table.hline(stroke: 0.6pt),
    [1], [*LaTiO2N*], [*0.2524*], [61.4%], [2.657], [5.328], [0.0107], [0.0500], [0.1679], [0.0038], [0.0201],
    [2], [*SrTaO2N*], [*0.2576*], [*76.6%*], [2.522], [5.079], [0.0020], [0.1119], [0.1020], [0.0236], [0.0182],
    [3], [*BaTaO2N*], [*0.2612*], [71.0%], [2.357], [5.385], [0.0012], [0.1314], [0.0530], [0.0068], [0.0688],
    [4], [*CaTaO2N*], [*0.2660*], [70.2%], [2.487], [5.177], [0.0085], [0.1222], [0.0893], [0.0179], [0.0282],
    [5], [YTiO2N], [0.3363], [21.9%], [2.577], [5.387], [0.1502], [0.0360], [0.1264], [0.0018], [0.0219],
    [6], [CaNbO2N], [0.3809], [63.1%], [1.846], [5.151], [0.0108], [0.2766], [0.0048], [0.0356], [0.0531],
    [7], [SrNbO2N], [0.4144], [69.7%], [1.805], [5.054], [0.0022], [0.3068], [0.0039], [0.0673], [0.0342],
    [8], [ScHfO2N], [0.9526], [1.5%], [2.407], [5.424], [0.7113], [0.1057], [0.0650], [0.0043], [0.0663],
    [9], [MgTaO2N], [1.0059], [6.8%], [1.994], [5.760], [0.0857], [0.2294], [0.0102], [0.0000], [0.6807],
    [10], [CdTaO2N], [3.6227], [0.1%], [1.398], [5.955], [0.0810], [0.8116], [0.0004], [0.0000], [2.7297],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Q2 Blind Pool (`11`, $X = "NO"_2$) Top-10 candidates (`02+03` refit) with five-term violation breakdown.],
) <tab:q2top10>

= Problem 3: Submodular HOBO-to-QUBO, Certification & ANOVA

== Three Active-Subspace Encodings & Exact HOBO Proof
Jointly selecting $(A, B, X)$ over $X in {"O"_3, "NO"_2, "FO"_2}$ defines a cubic objective $sum_(i,j,k) macron(T)_(i j k) a_i b_j c_k <= 0$. Naive Rosenberg pairing [8] over $(A, B)$ demands $52 times 51 = 2,652$ auxiliary bits ($2,760$ spins), whereas simply pairing $(A, X)$ in full space reduces auxiliary bits to $52 times 3 = 156$ ($264$ spins, already $< 550$). Combining axis asymmetry with active-subspace pruning below $C_("cap")$ ($1.25$ on Full `08`, $1.50$ on Blind `12`), we implement and certify three exact 8-bit encodings:

1. *`Model 3A-Comp` (Zero-Auxiliary Composite `(A,X)-B` Bipartite QUBO):* Active pairs $(i, k) in cal(A) = {(i,k) : min_j hat(V)_(i j k) < C_("cap")}$ are indexed by one composite one-hot vector $u_(i k) in {0,1}^(|cal(A)|)$ ($sum u_(i k) = 1$) coupled directly to $b_j$ ($sum b_j = 1$). By Theorem 2, $P_("block") > 12 max |m_(i j k)|$ guarantees zero illegal ground states with *zero auxiliary spins*: $26$ spins on `Q3_full_08` ($M_(max)=3, P_("block")=28.0$) and $18$ spins on `Q3_blind_12` ($M_(max)=4, P_("block")=40.0$).
2. *`Model 3A-HOBO` (Repaired Asymmetric Submodular HOBO-to-QUBO):* Introducing $y_(i k) in {0,1}$ for $(i, k) in cal(A)$ replaces $a_i b_j c_k$ with $y_(i k) b_j$ using Freedman--Drineas one-sided gadgets [9]:
   $ cal(H)_("HOBO") &= sum_((i,k) in cal(A)) [ sum_j 8 m_(i j k) y_(i k) b_j + P_("aux")(i,k) y_(i k)(2 - a_i - c_k) ] \
       &quad + P_("block")[(sum a_i - 1)^2 + (sum b_j - 1)^2 + (sum c_k - 1)^2] + cal(H)_("mask"). $ <eq:h3a>

#theorem-box("Theorem 4 (Auxiliary Elimination & Ground-State Correctness of Model 3A-HOBO)", [
  Because $b_j$ can take multiple $1$-entries outside the legal one-hot manifold, bounding $P_("aux")$ only by single-entry $max_j |Q_("obj")(y_(i k), b_j)|$ fails when $a_i c_k = 0$ and multiple $b_j = 1$. Instead, setting
  $ P_("aux")(i,k) = sum_j |8 m_(i j k)| + 8 > sum_j |8 m_(i j k)| $ <eq:paux>
  together with the exact cardinality-partitioned block penalty
  $ P_("block") > max_((k_A, k_B, k_X) != (1,1,1)) (-min H_("obj")(k_A, k_B, k_X)) / ((k_A - 1)^2 + (k_B - 1)^2 + (k_X - 1)^2) $ <eq:pblock>
  guarantees: (i) for every primary assignment $z = (bold(a), bold(b), bold(c))$, setting $y_(i k) = a_i c_k$ *attains* $min_y cal(H)_("HOBO")(z, y) = H_("obj")(z) + P_("block")[(k_A-1)^2 + (k_B-1)^2 + (k_X-1)^2]$ (uniquely whenever $a_i c_k = 0$, or when $a_i c_k = 1$ and $sum_j 8 m_(i j k) b_j < 0$; when $a_i c_k = 1$ and $sum_j 8 m_(i j k) b_j = 0$, e.g., $bold(b) = bold(0)$ or legal excited state `BaGeO2N`, $y_(i k) in {0, 1}$ tie at $0$ without altering $min_y cal(H)_("HOBO")$), and (ii) *zero illegal global ground states* across the unconstrained $2^(N)$ domain ($43$ spins / $18$ aux on `08`, $P_("aux") <= 48, P_("block") = 36.0$; $29$ spins / $11$ aux on `12`, $P_("aux") <= 56, P_("block") = 52.0$). Zero auxiliary-tie ambiguity *at the global ground energy* ($-2134$ on `08`, $-1662$ on `12`) holds by the exhaustive instance certificates in @tab:cert.
])
*Proof.* Fix any primary vector $z = (bold(a), bold(b), bold(c))$. The linear coefficient of $y_(i k)$ in @eq:h3a is $c_(i k)(z) = P_("aux")(i,k)(2 - a_i - c_k) + sum_j 8 m_(i j k) b_j$. If $a_i c_k = 0$, then $2 - a_i - c_k >= 1$, so by @eq:paux $c_(i k)(z) >= P_("aux")(i,k) - sum_j |8 m_(i j k)| >= 8 > 0$; hence $y_(i k) = 0$ is the *unique* minimizer even if $b_j = 1$ for every $j$. If $a_i c_k = 1$, then $2 - a_i - c_k = 0$ and $c_(i k)(z) = sum_j 8 m_(i j k) b_j <= 0$: when strictly negative, $y_(i k) = 1$ is uniquely optimal; when zero, both $y_(i k) in {0, 1}$ achieve energy $0 = a_i c_k sum_j 8 m_(i j k) b_j$. Thus $min_y cal(H)_("HOBO")(z, y)$ equals the exact primary Hamiltonian for every $z$. Condition @eq:pblock then ensures every primary state with $(k_A, k_B, k_X) != (1,1,1)$ strictly exceeds the feasible minimum energy, and $+P_("block")$ masks out $A=B$ and non-pool pairs. #h(1fr) $square$

3. *`Model 3B-ANOVA` (Repaired Uniform-$8 times$ Compact ANOVA-QUBO):* Fitting 2nd-order functional ANOVA $macron(T) tilde.op mu + f_A + f_B + f_X + f_(A B) + f_(A X) + f_(B X)$ on the active subspace ($33$ spins on `08`, $25$ spins on `12`, $0$ aux), we apply uniform $8 times$ integer scaling to *both* diagonal main effects and off-diagonal pairwise couplings (with effect-amplitude normalization) so algebraic Ising conversion (@eq:ising-recovery) eliminates the earlier main-vs.-interaction scaling mismatch. Note that `8 * round(effect * scale)` still discretizes relative float ANOVA weights, producing a coarse 6-level integer spectrum (`3, 1, 24, 59, 459, 42` candidates per level on `08`; `7, 9, 28, 50, 12, 2` on `12`; @tab:anova) suited as a compact approximation benchmark rather than a fine-ranking surrogate.

#figure(
  image("../v2_optimized/figures/schematic03_qubo_ising_compilation.png", width: 99%),
  caption: [Q3 compilation and certification workflow: three active-subspace formulations (`Model 3A-Comp`, `Model 3A-HOBO`, `Model 3B-ANOVA`), exact algebraic QUBO-to-Ising recovery (@eq:ising-recovery, $H_(i j) = -Q_(i j)/4$, $C = (sum Q_(i j) + op("tr") Q)/4$), exhaustive certification, and three-gate hardware record decoding.],
) <fig:q3comp>

== Global Ground-State Certification, Compression Attribution & ANOVA Decomposition
@tab:cert certifies all eight integer Ising matrices via `certify_global_ground_states()`: algebraic round-trip recovery (@eq:ising-recovery) verifies $H == J$ and zero energy residual, analytic auxiliary elimination computes $min_y x^top Q x = z^top Q_(z z) z + sum_r min(0, Q_(r r) + 2 z^top Q_(z, r))$, and full $2^(n_("prim"))$ (or exact zero-gap cardinality-partitioned) enumeration checks *every* primary assignment. For the eight submitted integer Ising matrices, exact structured enumeration after auxiliary minimization yields zero illegal global ground states and zero auxiliary ambiguity at the ground energy; independent MILP verification (`mip_gap = 0`) confirms the same minima. Seven integer ground-state sets contain the continuous surrogate minimizer `LaTiO2N` ($hat(V) = 0.252406$; full ANOVA minimum is `0.005211`), while quantization-induced ties (`3`--$10$ degenerate states; worst ground-state regret $0.4934$ on `Q2_full_07` and $0.4518$ on `Q3_blind_3B_ANOVA`) mean a ground-state return alone does not guarantee the continuous optimum. @tab:attrib separates axis selection from subspace pruning and one-sided gadgets, and @tab:anova details the 3-layer ANOVA error decomposition (`6` distinct legal integer levels; Spearman $rho = 0.2070$ on `08` and $0.4512$ on `12`). Coding assistance for enumeration vectorization [13] was cross-checked against brute-force $2^(n_("prim"))$ truth for $n_("prim") <= 18$.

#figure(
  {
    set text(size: 7.8pt)
    table(
      columns: (2.55cm, 1.40cm, 1.35cm, 1.55cm, 1.75cm, 1.75cm, 0.80cm, 1.15cm, 2.45cm, 2.25cm),
      stroke: none,
      align: (left, center, center, center, left, right, center, center, right, left),
      inset: (x: 2.2pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*Matrix ID*], [*Spins*], [*$M_(max)$/$P_("blk")$*], [*DR/`max|J|`*], [*Enum. ($N_("states")$)*], [*`cert_min_E` / $C$*], [*Deg.*], [*Ill. GS/Tie*], [*Best/Worst Reg.*], [*SHA-256 (12)*],
      table.hline(stroke: 0.6pt),
      [`Q2_full_07`], [17 (16+0+1)], [4 / 72], [7.00 / 125], [`full_2^n` (65,536)], [`-1558` / 1390], [10], [*0 / 0*], [0.0000 / 0.4934], [`a61bc1d9bd77`],
      [`Q2_blind_11`], [12 (11+0+1)], [4 / 72], [6.89 / 115], [`full_2^n` (2,048)], [`-1000` / 824], [4], [*0 / 0*], [0.0000 / 0.0136], [`361cc4d8ef7d`],
      [`Q3_full_3A_Comp`], [26 (25+0+1)], [3 / 28], [6.88 / 116], [`card_part` (33.5M)], [`-2110` / 2038], [10], [*0 / 0*], [0.0000 / 0.3209], [`f4bd07e3c83a`],
      [`Q3_full_3A_HOBO`], [43 (24+18+1)], [3 / 36], [7.23 / 121], [`card_part` (16.8M)], [`-2134` / 2010], [10], [*0 / 0*], [0.0000 / 0.3209], [`fe6d37c13ebf`],
      [`Q3_full_3B_ANOVA`], [33 (32+0+1)], [2 / 28], [6.88 / 116], [`card_part` (4.29B)], [`-2616` / 2500], [3], [*0 / 0*], [0.0052 / 0.0136], [`7416404244ba`],
      [`Q3_blind_3A_Comp`], [18 (17+0+1)], [4 / 40], [6.87 / 114], [`full_2^n` (131,072)], [`-1468` / 1364], [7], [*0 / 0*], [0.0000 / 0.1620], [`47b9b1f72e78`],
      [`Q3_blind_3A_HOBO`], [29 (17+11+1)], [4 / 52], [7.27 / 124], [`full_2^n` (131,072)], [`-1662` / 1482], [7], [*0 / 0*], [0.0000 / 0.1620], [`302bf51ecc1b`],
      [`Q3_blind_3B_ANOVA`], [25 (24+0+1)], [2 / 32], [6.83 / 112], [`card_part` (16.8M)], [`-1992` / 1872], [7], [*0 / 0*], [0.0000 / 0.4518], [`05ce8d9cfad2`],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Encoding Fidelity & Global Ground-State Certification (Unified Table 3): all 8 matrices pass 8-bit integer verification (`[-127, 127]`) with `0` illegal ground states and `0` auxiliary ties against declared optimum `LaTiO2N` ($hat(V)=0.252406$).],
) <tab:cert>

#figure(
  {
    set text(size: 7.8pt)
    table(
      columns: (3.45cm, 1.75cm, 2.15cm, 1.15cm, 2.15cm, 2.15cm, 4.20cm),
      stroke: none,
      align: (left, center, right, center, right, right, left),
      inset: (x: 2.6pt, y: 2.8pt),
      table.hline(stroke: 1.1pt),
      [*4-Row Compression Attribution*], [*Pool*], [*Prim / Aux / Spins*], [*`fits_550`*], [*vs. A-B*], [*vs. Full A-X*], [*Note*],
      table.hline(stroke: 0.6pt),
      [`Full-space A-B Rosenberg`], [52×52×3], [107 / 2,652 / 2,760], [False], [1.0×], [0.0588×], [Exceeds `SPQC-550` capacity],
      [`Full-space A-X Rosenberg`], [52×52×3], [107 / 156 / 264], [True], [17.0×], [1.0×], [Axis choice alone fits 550 spins],
      [`Active A-X Model 3A-HOBO`], [Full `08` \ Blind `12`], [24/18/43 \ 17/11/29], [True], [147.3× \ 241.1×], [8.67× \ 14.18×], [Same aux count as active A-X Rosenberg],
      [`Active (A,X)-B Model 3A-Comp`], [Full `08` \ Blind `12`], [25/0/26 \ 17/0/18], [True], [$infinity$ (0 aux) \ $infinity$ (0 aux)], [$infinity$ (0 aux) \ $infinity$ (0 aux)], [Composite `(A,X)` removes all aux bits],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [4-row compression attribution separating axis selection ($17.0 times$), subspace pruning, and zero-aux composite `(A,X)-B` encoding.],
) <tab:attrib>

#figure(
  {
    set text(size: 7.8pt)
    table(
      columns: (3.95cm, 1.40cm, 2.30cm, 2.65cm, 0.95cm, 1.75cm, 4.00cm),
      stroke: none,
      align: (left, center, left, left, center, center, left),
      inset: (x: 2.5pt, y: 2.8pt),
      table.hline(stroke: 1.1pt),
      [*ANOVA Matrix*], [*Subspace*], [*L1: $hat(V) ->$ Float ANOVA*], [*L2: Float $->$ 8-bit*], [*Levels*], [*GS / Regret*], [*L3: 50-Shot SA / Tabu*],
      table.hline(stroke: 0.6pt),
      [`Q3_full_08_Model3B_ANOVA`], [588 legal], [$R^2=0.4559$ \ $rho=0.3518$], [$rho(hat(V), "int")=0.2070$], [6], [3 / 0.005211], [SA 100% (0.000) / Tabu 100% (0.0052)],
      [`Q3_blind_12_Model3B_ANOVA`], [108 legal], [$R^2=0.4902$ \ $rho=0.4783$], [$rho(hat(V), "int")=0.4512$], [6], [7 / 0.000000], [SA 100% (0.000) / Tabu 100% (0.0052)],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [`Model 3B-ANOVA` 3-layer error decomposition showing 6 distinct legal integer energy levels (`3, 1, 24, 59, 459, 42` on `08`; `7, 9, 28, 50, 12, 2` on `12`).],
) <tab:anova>

== Equal-Budget Classical Solver Comparison & Q3 Blind Top-15
@tab:solvers compares all Q2 and Q3 formulations under an identical classical budget (`n_runs = 50`, taking the single best energy-sorted run out of $50$ runs, reporting unique feasible candidates, strict feasibility, continuous regret, and Top-$K$ hits). While naive 8-bit `v1` QUBOs collapse to $0.0%$ feasibility (or $12.2121$ regret on `Q3_full_08`), all repaired `v2` formulations achieve $100.0%$ feasibility on bivariate encodings ($86.0%$--$96.0%$ strict / $100.0%$ primary on `Model 3A-HOBO`) and zero best-of-50 regret under SA. @tab:q3top15 details the Q3 Blind (`12`) Top-15 shortlist spanning $"NO"_2$ and $"O"_3$ (`CaGeO3`, `SrGeO3`, `KNbO3`, `RbNbO3`, `CaSnO3`, `SrSnO3`).

#figure(
  image("../v2_optimized/figures/fig05_q3_qubit_compression_and_solvers.png", width: 98%),
  caption: [Problem 3 results: (a) 4-row compression ladder from $2,760$ to $18$ spins; (b) equal-budget ($50$-run) solver feasibility and Top-15 capture; (c) `Model 3B-ANOVA` 3-layer rank correlation; (d) five-term physical cost breakdown of Q3 Blind (`12`) Top-15 candidates.],
) <fig:q3>

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (4.35cm, 0.85cm, 1.85cm, 1.05cm, 1.55cm, 1.40cm, 1.35cm, 1.20cm, 3.40cm),
      stroke: none,
      align: (left, center, right, right, right, right, right, right, right),
      inset: (x: 2.2pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*Pool & Formulation*], [*Spins*], [*SA Feas.*], [*SA Uniq.*], [*SA Regret*], [*SA Top-K*], [*Tabu Feas.*], [*Tabu Uniq.*], [*Tabu Regret / Top-K*],
      table.hline(stroke: 0.6pt),
      [`Q2_full_07` `v1_Naive104_Float`], [105], [100.0%], [39], [0.000000], [8/10], [100.0%], [8], [0.000000 / 7/10],
      [`Q2_full_07` `v1_Naive104_8bit`], [105], [*0.0%*], [0], [—], [0/10], [*0.0%*], [0], [— / 0/10],
      [`Q2_full_07` *`v2_Exact8bit`*], [*17*], [*100.0%*], [10], [*0.000000*], [*10/10*], [*100.0%*], [7], [0.000000 / 7/10],
      [`Q2_blind_11` `v1_Naive104_8bit`], [105], [*0.0%*], [0], [—], [0/10], [*0.0%*], [0], [— / 0/10],
      [`Q2_blind_11` *`v2_Exact8bit`*], [*12*], [*100.0%*], [4], [*0.000000*], [*4/10*], [*100.0%*], [3], [0.000000 / 3/10],
      table.hline(stroke: 0.5pt),
      [`Q3_full_08` `v1_FM107_Float`], [108], [100.0%], [45], [0.005211], [7/15], [100.0%], [19], [0.000000 / 10/15],
      [`Q3_full_08` `v1_FM107_8bit`], [108], [100.0%], [2], [12.212100], [0/15], [80.0%], [5], [12.212100 / 0/15],
      [`Q3_full_08` *`Model 3A-Comp`*], [*26*], [*100.0%*], [10], [*0.000000*], [10/15], [*100.0%*], [9], [0.000000 / 9/15],
      [`Q3_full_08` *`Model 3A-HOBO`*], [43], [86.0% (100% prim)], [25], [*0.000000*], [*13/15*], [*100.0%*], [12], [0.005211 / 10/15],
      [`Q3_full_08` `Model 3B-ANOVA`], [33], [*100.0%*], [17], [*0.000000*], [7/15], [*100.0%*], [3], [0.005211 / 3/15],
      table.hline(stroke: 0.5pt),
      [`Q3_blind_12` `v1_FM107_8bit`], [108], [*0.0%*], [0], [—], [0/15], [*0.0%*], [0], [— / 0/15],
      [`Q3_blind_12` *`Model 3A-Comp`*], [*18*], [*100.0%*], [7], [*0.000000*], [7/15], [*100.0%*], [2], [0.005211 / 2/15],
      [`Q3_blind_12` *`Model 3A-HOBO`*], [29], [96.0% (100% prim)], [15], [*0.000000*], [*14/15*], [*100.0%*], [7], [0.000000 / 7/15],
      [`Q3_blind_12` `Model 3B-ANOVA`], [25], [*100.0%*], [7], [*0.000000*], [7/15], [*100.0%*], [2], [0.005211 / 2/15],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Equal-Budget `50`-Run Classical Solver Comparison (Unified Table 4a: SA `size_limit=15` vs. Tabu `size_limit=10`, $1$ best energy-sorted state/run). Runtimes: SA $2.59$--$3.47 "s"$, Tabu $5.08$--$9.60 "s"$.],
) <tab:solvers>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    align: (center, left, center, right, right, right, right, right, right, right),
    inset: 2.6pt,
    table.hline(stroke: 1.1pt),
    [*Rank*], [*Formula*], [*$X$*], [*$hat(V)$*], [*$hat(P)_("hit")$*], [*$hat(E)_g$*], [*$E_c$*], [*$hat(v)_("form")$*], [*$hat(v)_("gl")+hat(v)_("gh")$*], [*MC Stab. ($0.5 sigma$)*],
    table.hline(stroke: 0.6pt),
    [1], [*LaTiO2N*], [NO2], [*0.2524*], [61.4%], [2.657], [5.328], [0.0107], [0.2179], [99.8%],
    [2], [*SrTaO2N*], [NO2], [*0.2576*], [*76.6%*], [2.522], [5.079], [0.0020], [0.2139], [99.8%],
    [3], [*BaTaO2N*], [NO2], [*0.2612*], [71.0%], [2.357], [5.385], [0.0012], [0.1843], [100.0%],
    [4], [*CaTaO2N*], [NO2], [*0.2660*], [70.2%], [2.487], [5.177], [0.0085], [0.2114], [99.8%],
    [5], [YTiO2N], [NO2], [0.3363], [21.9%], [2.577], [5.387], [0.1502], [0.1624], [99.8%],
    [6], [CaNbO2N], [NO2], [0.3809], [63.1%], [1.846], [5.151], [0.0108], [0.2814], [99.0%],
    [7], [SrNbO2N], [NO2], [0.4144], [69.7%], [1.805], [5.054], [0.0022], [0.3107], [98.6%],
    [8], [*CaGeO3*], [O3], [*0.6458*], [33.7%], [2.579], [5.335], [0.0814], [0.4282], [99.4%],
    [9], [*SrGeO3*], [O3], [*0.7042*], [54.4%], [2.289], [5.234], [0.0302], [0.4927], [100.0%],
    [10], [*KNbO3*], [O3], [*0.7045*], [63.9%], [2.805], [5.288], [0.0003], [0.5816], [97.2%],
    [11], [RbNbO3], [O3], [0.8378], [*67.6%*], [2.702], [5.252], [0.0004], [0.6542], [97.2%],
    [12], [ScHfO2N], [NO2], [0.9526], [1.5%], [2.407], [5.424], [0.7113], [0.1707], [93.6%],
    [13], [CaSnO3], [O3], [0.9720], [21.1%], [2.979], [5.263], [0.1106], [0.7477], [86.6%],
    [14], [SrSnO3], [O3], [1.0055], [36.0%], [2.979], [5.164], [0.0357], [0.8285], [86.8%],
    [15], [MgTaO2N], [NO2], [1.0059], [6.8%], [1.994], [5.760], [0.0857], [0.2395], [86.0%],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Q3 Blind Pool (`12`, $1,587$ states) Top-15 multi-anion candidates with 500-trial $0.5 sigma$ Top-15 inclusion stability (`MC Stab.`).],
) <tab:q3top15>

= Real `SPQC-550` Hardware Verification & Cost

== Hardware Compilation Design, Ising Recovery & Verification Protocol
Rather than asserting a universal hardware scaling theorem, we summarize our empirical hardware-compilation design:

#theorem-box("Empirical Hardware Compilation Observation (Compact Subspace vs. Historical K-Splitting)", [
  On an 8-bit `SPQC-550` CIM ($|H_(u v)| <= 127$, $N <= 550$), our historical full-space single-reference implementation preserved all $N = 52$ elements using $K$-copy reference splitting ($N_("tot") in [224, 468]$ spins in historical Track A archives). By contrast, our active-subspace embeddings compile all four Q2/Q3 blind-pool formulations to $12$--$29$ spins with a single reference spin ($K = 1$), eliminating reference-copy consensus overhead and yielding $20.0%$--$90.0%$ strictly feasible returned records in `OPTIMIZATION` mode.
])

For every hardware task, `v2_spqc550_hardware.py --verify-only` enforces independent algebraic Ising-to-QUBO recovery from saved raw spin arrays:
$ Q_(i j) = -4 H_(i j) quad (i != j), &quad Q_(i i) = -4 H_(i,"ref") - sum_(j != i) Q_(i j), \
  x^top Q x = -s^top H s + C, &quad C = (1)/(4)(sum_(i,j) Q_(i j) + op("tr") Q), $ <eq:ising-recovery>
verifying bit-exact `int64` matrix SHA-256 identity, checkpoint file/array SHA-256 hashes, and record-by-record equality against the archived JSON before decoding $x = (s_(":-1") dot s_("ref") + 1) / 2$. Each returned record is audited across three separate gates: (i) primary one-hot ($k_A = k_B = k_X = 1$), (ii) declared pool eligibility and $A != B$, and (iii) auxiliary consistency $y_(i k) = a_i c_k$. We explicitly separate unique feasible candidates from duplicate records, report `cloud_roundtrip_wall_sec` (which includes network/queue latency and polling—note Kaiwu SDK `CIMOptimizer(..., interval)` is in seconds, with requested `1 s` clamped to the SDK minimum `5 s`, rather than optical pulse time), and note that `TaskMode.OPTIMIZATION` (`sample_number = 10`) returns energy-ranked optimization pools rather than independent sampling probabilities.

== `v3` `SPQC-550` Results vs. Same-Instance Classical Solvers
@tab:spqc and @fig:spqc compare all four repaired `v3` hardware runs against same-instance classical SA ($50$ runs) and Tabu ($30$ runs) baselines, with AI-assisted verification script auditing [14] checked bit-for-bit via `--verify-only` against raw spin arrays (all $40$ returned records in Appendix B):
- *`v3_q2_blind_12` ($12$ spins, `cert_min_E = -1000`, $C = 824$, `17.24 s` wall):* Yields *$30.0%$* strictly feasible returned records ($3/10$ unique hits, $0$ duplicates), reaching certified ground energy $-1000.0$ with `SrTaO2N` ($hat(V) = 0.257617$, continuous regret $0.005211$, rank $2$), plus `CaNbO2N` ($0.380890$) and `YNbO2N` ($12.7700$).
- *`v3_q3_comp_18` ($18$ spins, $0$ aux, `cert_min_E = -1468`, $C = 1364$, `16.23 s` wall):* Eliminating auxiliary variables lifts the strictly feasible record fraction to *$90.0%$* ($9/10$ unique Top-15 hits, $0$ duplicates), reaching certified ground energy $-1468.0$ on records `0--4` and directly recovering exact declared pool optimum **`LaTiO2N`** ($hat(V) = 0.252406$, continuous `regret = 0.000000`, rank $1$) alongside `SrTaO2N`, `BaTaO2N`, `CaTaO2N`, `SrNbO2N`, `CaGeO3`, `RbNbO3`, `ScHfO2N`, and `CaSnO3`.
- *`v3_q3_hobo_29` ($29$ spins, $11$ aux, `cert_min_E = -1662`, $C = 1482$, `19.06 s` wall):* Hits certified ground energy $-1662.0$ with $90.0%$ ($9/10$) primary-feasible records (`CaTaO2N`, `SrTaO2N`), while $7$ of those $9$ excited records exhibit $1$--$3$ auxiliary mismatches ($y_(i k) != a_i c_k$), yielding *$20.0%$* strictly feasible records (`SrTaO2N`, `CaTaO2N`). Contrasting with `v3_q3_comp_18` ($90.0%$) highlights the practical benefit of eliminating auxiliary qubits on real CIM hardware.
- *`v3_q3_anova_25` ($25$ spins, $0$ aux, `cert_min_E = -1992`, $C = 1872$, `16.43 s` wall):* Under the repaired uniform-$8 times$ scaling, `SPQC-550` yields *$30.0%$* strictly feasible records ($3/10$ unique hits: `SrTaO2N`, `CaTaO2N`, `BaTaO2N` all at exact $-1992.0$, regret $0.005211$). Combined with our controlled offline scaling audit (@tab:anova), this supports our attribution of the earlier `LaTiO2F` artifact to uncalibrated scaling and truncation (while noting surrogates, active subspaces, and quantization parameters also changed between versions).
- *Historical Track A Archive Summary (`table4_spqc550_hardware_vs_classical.csv`):* Across $6$ legacy uncertified full-space tasks (`q2_uncap` $105$ sp `0/10`, `q2_full` $224$ sp `1/9` `LaTiO2N`, `q2_blind` $224$ sp `0/4`, `q3_qfm108` $257$ sp `0/10`, `q3_cmp209` $468$ sp `1/10` `YTiO2N`, `q3_ros264` $413$ sp `1/10` `AgTaO3`), overall returned-record feasibility was $3/53 = 5.66%$; because pools, surrogates, and encodings changed between Track A and `v3`, Track A is reported solely as historical context.

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (2.20cm, 1.30cm, 1.50cm, 2.05cm, 1.00cm, 1.15cm, 1.40cm, 1.35cm, 3.30cm, 1.75cm),
      stroke: none,
      align: (left, center, right, left, right, center, right, center, left, right),
      inset: (x: 2.2pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*Task ID*], [*Spins (Aux)*], [*`cert_min_E`*], [*Solver (Rec./Runs)*], [*Prim.*], [*`y=a*c`*], [*Strict*], [*Uniq / Dup*], [*Best Formula (`V_hat` / Regret)*], [*Time (s)*],
      table.hline(stroke: 0.6pt),
      [`v3_q2_blind_12`], [12 (0)], [-1000], [`SPQC-550` (10)], [30.0%], [10/10], [*30.0%*], [3 / 0], [`SrTaO2N` (0.2576 / 0.0052)], [17.24],
      [`v3_q2_blind_12`], [12 (0)], [-1000], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [4/46 ; 4/26], [`LaTiO2N` (0.2524 / 0.0000)], [3.33 / 3.51],
      [`v3_q3_comp_18`], [18 (0)], [-1468], [`SPQC-550` (10)], [90.0%], [10/10], [*90.0%*], [9 / 0], [*`LaTiO2N` (0.2524 / 0.0000)*], [16.23],
      [`v3_q3_comp_18`], [18 (0)], [-1468], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [7/43 ; 2/28], [`LaTiO2N` (0.0) / `SrTaO2N`], [3.25 / 3.70],
      [`v3_q3_hobo_29`], [29 (11)], [-1662], [`SPQC-550` (10)], [90.0%], [3/10], [*20.0%*], [2 / 0], [`SrTaO2N` (0.2576 / 0.0052)], [19.06],
      [`v3_q3_hobo_29`], [29 (11)], [-1662], [SA (50) \ Tabu (30)], [100%], [48 / 30], [96% / 100%], [13/35 ; 7/23], [`LaTiO2N` (0.2524 / 0.0000)], [3.31 / 3.16],
      [`v3_q3_anova_25`], [25 (0)], [-1992], [`SPQC-550` (10)], [30.0%], [10/10], [*30.0%*], [3 / 0], [`SrTaO2N` (0.2576 / 0.0052)], [16.43],
      [`v3_q3_anova_25`], [25 (0)], [-1992], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [8/42 ; 2/28], [`SrTaO2N` / `LaTiO2N` (0.0)], [3.29 / 3.57],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [`SPQC-550` Hardware vs. Same-Instance Classical Gate (Unified Table 4b): all 4 `v3` `OPTIMIZATION` returned pools hit certified `cert_min_ising_energy`, with zero-aux `v3_q3_comp_18` yielding `90.0%` strictly feasible returned records and `9/15` unique Top-15 hits. `SPQC-550` times are `cloud_roundtrip_wall_sec`.],
) <tab:spqc>

#figure(
  image("../v2_optimized/figures/fig07_spqc550_real_hardware_comparison.png", width: 98%),
  caption: [Real `SPQC-550` verification: (a) Track A historical archive (`0%`--`11%`, `5.66%` overall) vs. `v3` strictly feasible returned-record proportions (`30.0%`, `90.0%`, `20.0%` strict / `90.0%` primary, `30.0%`; same-instance SA/Tabu baselines in @tab:spqc); (b) `v3` record-by-record raw Ising energy excess relative to certified `cert_min_ising_energy` (`10` returned records/task); (c) hardware spin count vs. `SPQC-550` feasibility comparing Track A (`105`--`468` spins) against active-subspace `v3` (`12`--`29` spins).],
) <fig:spqc>

= Sensitivity Analysis & Limitations

== Multi-Axis Sensitivity & `C_cap` Top-10 / Top-25 Retention Audit
@tab:sens and @fig:sens synthesize four independent stress tests across quantization precision, penalty bounds, upper-tail truncation, and surrogate uncertainty:
1. *Independent $4$--$12$ Bit Sweep on `Q2_blind_11`:* Searching maximal exact integer $M_(max)(B)$ bounded by $M_B = 2^(B-1)-1$ yields exact integer matrices at $B = 7$ ($M_(max)=2$, `max|J|=59`, $3$ levels, $7$ legal / $0$ illegal GS), $B = 8$ ($M_(max)=4$, `max|J|=115`, $4$ levels, $4$ legal / $0$ illegal GS), $B = 10$ ($M_(max)=17$, `max|J|=492`, $5$ levels, $5$ legal / $0$ illegal GS), and $B = 12$ ($M_(max)=71$, `max|J|=2038`, $8$ levels, $3$ legal / $0$ illegal GS, $9/10$ SA Top-10 hits). Below $7$ bits (`bw=4, 5, 6`), exact integer $M_(max)$ is infeasible and Kaiwu precision rescaling fallback introduces distortion ($0.700153$ top-1 regret at $6$ bits, $12.706008$ worst GS regret at $5$ bits, and $5$ illegal global ground states with $42.0%$ SA feasibility at $4$ bits).
2. *Refined Fixed-$M_(max)=4$ Penalty Sweep (`Q2_blind_11`, 10 points):* Fixing $M_(max) = 4$ and sweeping $alpha_P in [0.50, 3.00]$ cleanly isolates Theorem 2's transition: for $alpha_P in {0.50, 1.00}$, zero legal ground states exist (`margin = -48.0, -16.0`, SA $0.0%$--$4.0%$); at $alpha_P = 1.25$ ($P_("block")=40$), $4$ legal states tie $1$ illegal state (`margin = 0.0`, $80.0%$ GS feas., SA $84.0%$); across $alpha_P in [1.40, 3.00]$ (specifically exact 8-bit integers at $alpha_P in {1.50, 1.75, 2.25}$), all $4$ global ground states are $100%$ legal (`margin = +9.6 .. +64.0`) and SA hits $100.0%$ feasibility.
3. *`C_cap` Top-10 / Top-25 Retention Audit:* @tab:sens records exact Top-10 and Top-25 retention under default caps across three distinct subspaces (`V_hat < C_cap`, element-product $A times B times X$, and `(A,X)-B` composite): `Q2_full_07` (`cap=2.0`, $20$ active rows) retains $10/10$ and $20/25$ (dropping `#21 BiZrO2N` $2.1579$ .. `#25 YGeO2N` $2.3656$); `Q2_blind_11` (`cap=2.5`, $9$ active rows) retains $9/10$ and $9/25$ (dropping `#10 CdTaO2N` $hat(V)=3.6227$ and ranks $11$--$25$); `Q3_full_08` (`cap=1.25`, $26$ active rows) retains $10/10$ and $25/25$ across all subspaces; `Q3_blind_12` (`cap=1.50`, $15$ active rows) retains $10/10$ and $15/25$ strictly below $1.50$, *$17/25$* inside the $A times B times X$ element-product subspace (*$8$* outside: ranks $18$--$25$), and *$15/25$* inside the `(A,X)-B` composite subspace (`10` outside, additionally excluding `#16 BaGeO3` $1.6630$ and `#17 BaSnO3` $1.8225$).
4. *Multi-Scale (`0.5–2.0 sigma`) Perturbation Stability (`500` trials):* Under independent Gaussian noise ($0.5 sigma -> 2.0 sigma$), Q2 Blind Top-10 retention shifts from $95.70% -> 93.84% -> 88.86% -> 83.28%$ (Spearman $0.9928 -> 0.9034$), while Q3 Blind Top-15 shifts from $96.24% -> 83.20% -> 71.43% -> 61.01%$ ($rho: 0.9935 -> 0.9132$). Introducing `(A,B)`/`A`-group correlation plus `p_semi` logit noise yields comparable decay ($95.46% -> 83.10%$ on Q2 Top-10; $94.96% -> 58.84%$ on Q3 Top-15), confirming local shortlist stability at $0.5 sigma$ while exposing growing rank sensitivity as uncertainty scales to $2.0 sigma$.

#figure(
  image("../v2_optimized/figures/fig06_sensitivity_and_mc_robustness.png", width: 98%),
  caption: [Four-axis sensitivity suite: (a) fixed-$M_(max)=4$ penalty phase transition between $alpha_P=1.25$ and $1.40$--$3.00$; (b) `C_cap` Top-10/Top-25 retention vs. active spin count; (c)--(d) multi-scale ($0.5 sigma$--$2.0 sigma$) independent and `(A,B)`-correlated+logit ranking stability.],
) <fig:sens>

#figure(
  table(
    columns: (auto, 1.3fr, 1.3fr),
    stroke: none,
    align: (left, left, left),
    inset: 2.8pt,
    table.hline(stroke: 1.1pt),
    [*Sensitivity Axis*], [*Sweep / Configuration*], [*Verified Quantitative Outcome*],
    table.hline(stroke: 0.6pt),
    [Bit-Width `Q2_blind_11`], [`bw=4,5,6` (`rescaled fallback`) vs. `7,8,10,12` (`exact M_max(B)`)], [`bw=4`: 5 illegal GS, SA 42%; `bw=5`: 0 ill. GS (worst 12.71); `bw=6`: 0 ill. GS (top1 0.700); `bw=7..12` ($M_(max)=2,4,17,71$): 0 ill. GS, SA 100%, 4--9/10 hits],
    [Penalty `Q2_blind_11` (`M_max=4`)], [`alpha_P in [0.50 .. 3.00]` ($10$ points)], [`0.50, 1.00`: 0 legal GS (SA 0--4%); `1.25`: 4 legal / 1 illegal (margin 0.0, SA 84%); `1.40..3.00`: 4 legal / 0 illegal (SA 100%; exact 8-bit at `1.50, 1.75, 2.25`)],
    [`C_cap` Top-10 / Top-25 Retention], [`Q2_full` (2.0), `Q2_blind` (2.5), `Q3_full` (1.25), `Q3_blind` (1.50)], [`Q2_full`: 10/10, 20/25; `Q2_blind`: 9/10, 9/25 (drops `#10 CdTaO2N` 3.6227); `Q3_full`: 10/10, 25/25; `Q3_blind`: 10/10, 15/25 (`<cap`) / 17/25 ($A times B times X$, 8 outside) / 15/25 (Comp, 10 outside)],
    [MC Ranking Stability (`500` trials)], [`0.5..2.0 sigma` Indep. Gaussian vs. `(A,B)`-Corr. + Logit], [Indep. Q2 Top-10: `95.70% -> 83.28%`, Q3 Top-15: `96.24% -> 61.01%`; Group-Corr+Logit: Q2 `95.46% -> 83.10%`, Q3 `94.96% -> 58.84%`],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Unified Sensitivity & `C_cap` Retention Audit (Unified Table 5): exact parameters, feasibility transitions, dropped candidates, and multi-scale perturbation retention.],
) <tab:sens>

== Limitations and Scope Boundaries
1. *Unseen `04_test_holdout` Labels & Epistemic `FO2` Uncertainty:* Without ground-truth labels on `04`, neither `sum(P_hit) = 10.3358` nor `C_cap` retention proves true-positive recall against the $19$ hidden `V=0` targets. In particular, low-$V$ oxyfluorides (`FO2`) are sparser in `02+03`, producing conservative `FO2` expected mass (`0.77` vs. `6` hidden targets).
2. *Distributional & Conformal Scope:* Eqs. @eq:vbayes and @eq:phit assume untruncated Gaussian marginals (`N(hat(mu)_g, hat(sigma)_g^2)` retains negative support with `0.1743` median across `04`, `max 0.0110` in Top-50) and conditional independence despite nonzero `e_form`/`gap` residual correlation ($r = -0.2930$), and `03_val` conformal radii (`12.375` on the $hat(p)_("semi") >= 0.20$ stratum) do not transfer as formal finite-sample guarantees to `02+03` refitted pools.
3. *Surrogate Dependence of `C_cap` & Hardware Semantics:* Pruning above $C_("cap")$ is lossless only with respect to $hat(V)$ and explicitly drops candidates such as `CdTaO2N` ($3.6227$); `SPQC-550` runs under `TaskMode.OPTIMIZATION` (`10` returned records per task) rather than independent `SAMPLING` trials, and wall-clock times reflect full cloud roundtrips (`16.23`--`19.06 s`, with `5 s` effective polling interval) rather than optical pulse annealing duration.

= Conclusion & References

== Conclusion
We have developed and audited an end-to-end pipeline for rare-target cubic perovskite discovery under `(A,B)`-pair evaluation discipline and real 8-bit `SPQC-550` hardware constraints:
1. *Dataset & Surrogate Rigor:* Checksum auditing confirms $9/9$ CRLF-normalized matches ($1/9$ raw bytes in Git LF checkout). Isolating base features and 4-family predictors within outer `(A,B)` folds (base-path perturbation delta $1.42 times 10^(-14)$) with pooled-OOF meta-fitting, `v2_PhysHurdle_Bayes` leverages the in-data Butler--Ginley relation ($R^2 = 1.0000000000$, explicit $sigma_(E_g)/2$ propagation) and Stage-0 latent descriptors to achieve `03_val` $"RMSE" = 4.0634$, $"MAE"(V<1) = 2.9280$, $100.0%$ Top-75 recall ($V<0.5$), and $16.20/18$ ($"SD" = 1.077$) under hybrid Top-20 + 8-bit QUBO acquisition across $20$ paired seeds.
2. *Certified 8-Bit QUBO/HOBO Formulations:* Across all eight Q2 and Q3 matrices (`12`--$43$ spins), exact algebraic QUBO-to-Ising recovery (@eq:ising-recovery) and exhaustive primary-variable enumeration after auxiliary minimization yield zero illegal global ground states and zero ground-state auxiliary ties; seven integer ground-state sets contain declared optimum `LaTiO2N` ($hat(V) = 0.252406$), while repaired `Model 3B-ANOVA` exhibits `6` distinct legal integer energy levels.
3. *Verified `SPQC-550` Execution:* All four blind-pool matrices pass bit-exact algebraic recovery and per-record raw-spin audits, yielding $30.0%$--$90.0%$ strictly feasible returned records (`v3_q3_comp_18` recovering `LaTiO2N` at $0.000000$ regret and $9/15$ unique Top-15 hits).

== References
#set par(hanging-indent: 1.8em, first-line-indent: 0pt)

[1] Castelli I E, Landis D D, Thygesen K S, et al., New cubic perovskites for one- and two-photon water splitting using the computational materials repository[J], *Energy & Environmental Science*, 2012, 5(10): 9034--9043.

[2] QBoson Inc., *Kaiwu SDK 1.4.4 Documentation: Parameter Precision, QUBO/Ising Conversion, and SPQC-550 CIM Optimizer*, #link("https://kaiwu-sdk-docs.qboson.com/"), 2026.

[3] Honjo T, Sonobe T, Inaba K, et al., 100,000-spin coherent Ising machine[J], *Science Advances*, 2021, 7(40): eabh0952.

[4] Butler M A, Ginley D S, Prediction of flatband potentials at semiconductor-electrolyte interfaces from atomic electronegativities[J], *Journal of the Electrochemical Society*, 1978, 125(2): 228--232.

[5] Shannon R D, Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides[J], *Acta Crystallographica Section A*, 1976, 32(5): 751--767.

[6] Rumble J R, *CRC Handbook of Chemistry and Physics* (102nd ed.)[M], Boca Raton: CRC Press, 2021.

[7] Bartel C J, Sutton C, Goldsmith B R, et al., New tolerance factor to predict the stability of perovskite oxides and halides[J], *Science Advances*, 2019, 5(2): eaav0693.

[8] Rosenberg I G, Reduction of bivalent maximization to the quadratic case[J], *Cahiers du Centre d'Études de Recherche Opérationnelle*, 1975, 17(1): 71--74.

[9] Freedman D, Drineas P, Energy minimization via graph cuts: Settling what is possible[C] / *Proc. IEEE CVPR*, 2005, 2: 939--946.

[10] Kitai K, Guo J, Ju S, et al., Designing metamaterials with quantum annealing and factorization machines[J], *Physical Review Research*, 2020, 2(1): 013319.

[11] Romano Y, Patterson E, Candès E J, Conformalized quantile regression[C] / *Advances in Neural Information Processing Systems (NeurIPS)*, 2019, 32: 3543--3553.

[12] Vovk V, Gammerman A, Shafer G, *Algorithmic Learning in a Random World* (2nd ed.)[M], Cham: Springer, 2022.

[13] Google, Gemini 2.5 Pro, Google LLC, 2026-11-26--2026-11-27.

[14] DeepSeek, DeepSeek-R1-0528, DeepSeek, 2026-11-26--2026-11-27.

// =============================================================================
// APPENDICES (UNLIMITED PAGES AFTER MAIN BODY)
// =============================================================================
#pagebreak()
#set par(hanging-indent: 0pt)
#set heading(numbering: none)

= Appendices

== Appendix A: Software Stack, Staged Reproduction & AI Disclosure
All computations ran reproducibly (seeds `2026`, `101..120`) under Python 3.12 with `numpy==2.5.5`, `scipy==1.18.1`, `pandas==2.2.3`, `scikit-learn==1.9.1`, `catboost==1.2.10`, `lightgbm==4.7.0`, `xgboost-cpu==3.4.1`, `matplotlib==3.10.3`, `typst==0.15.2`, `pypdf==7.2.0`, `pymupdf==2.2`, `python-docx==1.2.0`, and enterprise `kaiwu==1.4.4`. Data paths resolve dynamically via `common_physics.py::resolve_data_dir()` (`APMCM_DATA_DIR` environment variable or workspace official dataset folder). Archival `M4`/`M5` and Track A summary CSVs are bundled inside `v2_optimized/reference_baselines/` so the supporting archive reproduces standalone when paired with `APMCM_DATA_DIR`:
- *Standalone Supporting Archive (`Dapmcm26302266fj.zip` + `APMCM_DATA_DIR`, zero cloud calls):*
  - `python v2_optimized/run_all.py --verify-saved` (checksum audit, saved CSV/JSON checks, and `v2_spqc550_hardware.py --verify-only`).
  - `python v2_optimized/run_all.py --all` (full offline retraining, Q2/Q3 certification, 20-seed & sensitivity experiments, 11 PNGs, and verification).
  - `python scripts/verify_all_review_gates.py --code-only` (all 7 code/data/matrix/figure verification gates).
- *Full Repository & Pre-Submission Audit:*
  - `python scripts/verify_all_review_gates.py` (includes `paper/main.typ` & `REPORT.md` prose/numeric sync) and `python scripts/audit_submission.py --problem D --team apmcm26302266` (verifies main PDF, supporting ZIP, and the formal AI Tool Usage Disclosure PDF alongside Commitment Sheet `Dapmcm26302266cns.<pdf|jpg|png>`).

In compliance with the 2026-11-04 AI disclosure rules, all core mathematical formulations, physical couplings, penalty proofs, and hardware evaluations were conceived and verified by the team, while AI assistants [13, 14] were used solely for pipeline refactoring checks, plotting templates, and English phrasing review (documented in our standalone AI Tool Usage Disclosure PDF).

== Appendix B: Complete 40-Record `v3` `SPQC-550` Hardware Log & SHA-256 Registry
@tab:records records all $40$ returned optimization records across the $4$ verified `v3` `SPQC-550` executions (`v3_q2_blind_12`, `v3_q3_comp_18`, `v3_q3_hobo_29`, `v3_q3_anova_25`; `18 >= 5` account credits conservatively preserved per initial 31 allocation), decoded independently from `v2_optimized/results/spqc_checkpoints/*_raw_spins.npy`. @tab:hashes archives all full 64-hex SHA-256 digests and Ising offsets $C$.

#figure(
  {
    set text(size: 7.8pt)
    table(
      columns: (2.25cm, 1.05cm, 2.05cm, 2.35cm, 1.35cm, 5.25cm, 1.45cm, 1.25cm),
      stroke: none,
      align: (left, center, right, center, center, left, right, center),
      inset: (x: 2.4pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*Task ID*], [*Rec(s)*], [*Ising $E(s)$*], [*$(k_A, k_B, k_X)$*], [*`y=a*c`*], [*Status / Decoded Formula*], [*$hat(V)$*], [*Rank*],
      table.hline(stroke: 0.6pt),
      [`v3_q2_blind_12`], [0], [-1000.0], [(1, 1, 1)], [OK], [*FEASIBLE: `SrTaO2N`*], [0.257617], [2],
      [`v3_q2_blind_12`], [1], [-992.0], [(1, 1, 1)], [OK], [*FEASIBLE: `CaNbO2N`*], [0.380890], [6],
      [`v3_q2_blind_12`], [2], [-968.0], [(1, 1, 1)], [OK], [*FEASIBLE: `YNbO2N`*], [12.769992], [64],
      [`v3_q2_blind_12`], [3--9], [-920.0..-896.0], [(2,1,1)/(1,1,1)\* / 0-hot], [OK], [INFEASIBLE (pool/one-hot fail, 7 records)], [—], [—],
      table.hline(stroke: 0.5pt),
      [`v3_q3_comp_18`], [0], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: `SrTaO2N`*], [0.257617], [2],
      [`v3_q3_comp_18`], [1], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: `SrNbO2N`*], [0.414431], [7],
      [`v3_q3_comp_18`], [2], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: `LaTiO2N`*], [*0.252406*], [*1*],
      [`v3_q3_comp_18`], [3], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: `CaTaO2N`*], [0.265979], [4],
      [`v3_q3_comp_18`], [4], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: `BaTaO2N`*], [0.261166], [3],
      [`v3_q3_comp_18`], [5], [-1460.0], [(1, 1, 1)], [OK], [*FEASIBLE: `RbNbO3`*], [0.837836], [11],
      [`v3_q3_comp_18`], [6], [-1460.0], [(1, 1, 1)], [OK], [*FEASIBLE: `CaGeO3`*], [0.645777], [8],
      [`v3_q3_comp_18`], [7], [-1452.0], [(1, 1, 1)], [OK], [*FEASIBLE: `ScHfO2N`*], [0.952559], [12],
      [`v3_q3_comp_18`], [8], [-1452.0], [(1, 1, 1)], [OK], [*FEASIBLE: `CaSnO3`*], [0.972014], [13],
      [`v3_q3_comp_18`], [9], [-1452.0], [(2, 1, 2)], [OK], [INFEASIBLE (one-hot/pool fail)], [—], [—],
      table.hline(stroke: 0.5pt),
      [`v3_q3_hobo_29`], [0], [-1662.0], [(1, 1, 1)], [OK (0)], [*FEASIBLE: `CaTaO2N`*], [0.265979], [4],
      [`v3_q3_hobo_29`], [1], [-1662.0], [(1, 1, 1)], [OK (0)], [*FEASIBLE: `SrTaO2N`*], [0.257617], [2],
      [`v3_q3_hobo_29`], [2, 3], [-1654.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [4], [-1646.0], [(1, 1, 1)], [Fail (2)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [5], [-1638.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(SrTaO2N)`], [0.257617], [2],
      [`v3_q3_hobo_29`], [6], [-1634.0], [(1, 2, 1)], [OK (0)], [INFEASIBLE (one-hot fail)], [—], [—],
      [`v3_q3_hobo_29`], [7, 8], [-1630.0], [(1, 1, 1)], [Fail (2, 3)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [9], [-1630.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(CaNbO2N)`], [0.380890], [6],
      table.hline(stroke: 0.5pt),
      [`v3_q3_anova_25`], [0], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: `SrTaO2N`*], [0.257617], [2],
      [`v3_q3_anova_25`], [1], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: `CaTaO2N`*], [0.265979], [4],
      [`v3_q3_anova_25`], [2], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: `BaTaO2N`*], [0.261166], [3],
      [`v3_q3_anova_25`], [3--9], [-1976.0..-1944.0], [(2,1,1)/(1,2,1)/(1,1,0)/0-hot], [OK], [INFEASIBLE (7 records)], [—], [—],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Complete 40-record hardware verification log across all 4 `v3` `SPQC-550` `OPTIMIZATION` runs (`*` records 4 and 9 in `v3_q2_blind_12` are one-hot valid but excluded by blind-pool mask).],
) <tab:records>

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: none,
    align: (left, right, left),
    inset: 2.3pt,
    table.hline(stroke: 1.1pt),
    [*Artifact / Matrix ID*], [*Offset $C$*], [*Full 64-Hex SHA-256 Digest*],
    table.hline(stroke: 0.6pt),
    [`Q2_full_07`], [1390], [`a61bc1d9bd777acdc31c02490c510554dbbcf488d683f50572ba61760fc1aa7e`],
    [`Q2_blind_11` (`v3_q2_blind_12`)], [824], [`361cc4d8ef7d074479506820b0c3e3d90b3170e97ca7c721caea7aa013c02a6b`],
    [`Q3_full_08_Model3A_Comp`], [2038], [`f4bd07e3c83a92023de687e70b95b1c7d81d32084248c12f00ddaf274b9ecdee`],
    [`Q3_full_08_Model3A_HOBO`], [2010], [`fe6d37c13ebf55939abec80dc01f7ba9aaa90be0bc3a37ea7ba729123c56e2a1`],
    [`Q3_full_08_Model3B_CompactANOVA`], [2500], [`7416404244bac9c826c72594f2c937843c1af4d810b9febdb1d94fddf24f6ae2`],
    [`Q3_blind_12_Model3A_Comp` (`v3_q3_comp_18`)], [1364], [`47b9b1f72e78fa9d4dc94ae9a83b6a0067c1f1914b8e9219df6f9b5e233939ae`],
    [`Q3_blind_12_Model3A_HOBO` (`v3_q3_hobo_29`)], [1482], [`302bf51ecc1b9281fea99a77ad038489818d626c5f288e81db1eab9211f55a2a`],
    [`Q3_blind_12_Model3B_CompactANOVA` (`v3_q3_anova_25`)], [1872], [`05ce8d9cfad20be928a3ec0cfd51742f094a311ed64612039681749d9e341f36`],
    [`apmcm26_v2_q2_blind_12` (archived v2)], [822], [`c5854a58cdff0d9aa66467288c39926ec1ab01b9b4871c5a56d52786a000b7bd`],
    [`apmcm26_v2_q3_anova_24` (archived v2)], [1228], [`90b005d13fad22f249a3fe629930159f74193b2db6c2d5d9c329e04dcede06d8`],
    [`apmcm26_v2_q3_submod_33` (archived v2)], [790], [`c50b5bd420b82dc683aa8d076219c0cc09c568ef77089f28e967afb86ee69662`],
    [`v2_val_predictions.csv`], [—], [`ac5d939d4dd63f3aa95d4cde112247b376ea9f3818d766644ba5d9a9b6f3c3fd`],
    [`v2_holdout_04_train02only.csv`], [—], [`2090c512ddc3e61459a81519b956e3e5d0bd1a9145b36e490cabf0cfd5d896c3`],
    [`v2_holdout_04.csv`], [—], [`89eca9b89f8501233356701b2426e3296e41c72f66bf64b518520d8a344742a2`],
    [`v2_q2_full_07.csv`], [—], [`4e2d68eb2626360838985f59e6eb64e3edfd23f9269d81ece3d124737bc2abe7`],
    [`v2_q2_blind_11.csv`], [—], [`3154b240448bde8f2cd12a3cd27b121ac813eb56c85672c8f7337ab7b6412d4c`],
    [`v2_q3_full_08.csv`], [—], [`751f92ff1ab1efb8f6c0f3e94070c2ee72ebcf7fb73211ec7e8576bc96bf35f0`],
    [`v2_q3_blind_12.csv`], [—], [`9de6d43dcf6b1e35032f6d09c9a5de2a5bf16c44ed9e7bb1ff3b7738105004b6`],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Full 64-hex SHA-256 registry for all 8 certified `v3` Ising matrices, 3 archived `v2` matrices, and 7 exported Q1 CSV files.],
) <tab:hashes>

== Appendix C: Core Source Code
#let clean-src(p) = read(p).replace(regex("\p{Han}+"), "<cn-text>")

_Note:_ To preserve strict English-only PDF typesetting, non-ASCII Chinese directory/status literals in the workspace path fallback (`resolve_data_dir`) and SDK error string are rendered as `<cn-text>` below; unmodified byte-exact `.py` scripts are archived in `Dapmcm26302266fj.zip`.

=== C.1 Physical Constants, Dataset Audit & Band-Center Utilities (`v2_optimized/common_physics.py`)
#set text(size: 7.2pt)
#raw(clean-src("../v2_optimized/common_physics.py"), lang: "python", block: true)

#pagebreak()
#set text(size: 10.6pt)
=== C.2 Fold-Isolated Q1 Surrogate & Pool Scoring (`v2_optimized/v2_surrogate_q1.py`)
#set text(size: 7.2pt)
#raw(clean-src("../v2_optimized/v2_surrogate_q1.py"), lang: "python", block: true)

#pagebreak()
#set text(size: 10.6pt)
=== C.3 Certified 8-Bit QUBO/HOBO & ANOVA Engine (`v2_optimized/v2_qubo_q2_q3.py`)
#set text(size: 7.2pt)
#raw(clean-src("../v2_optimized/v2_qubo_q2_q3.py"), lang: "python", block: true)

#pagebreak()
#set text(size: 10.6pt)
=== C.4 `SPQC-550` Execution & Algebraic Ising Verification (`v2_optimized/v2_spqc550_hardware.py`)
#set text(size: 7.2pt)
#raw(clean-src("../v2_optimized/v2_spqc550_hardware.py"), lang: "python", block: true)
