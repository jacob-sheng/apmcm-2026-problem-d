// =============================================================================
// 2026 APMCM (16th Asia and Pacific Mathematical Contest in Modeling)
// Problem D — Team Dapmcm26302266
// Strictly Anonymous | English Only | Main Body <= 25 Pages | 11 pt Academic Serif
// =============================================================================

#let team-id = "apmcm26302266"
#let problem-chosen = "D"
#let paper-title = "Physics-Coupled Surrogate Modeling and Active-Subspace 8-Bit Ising Optimization for Rare-Target Perovskite Discovery"

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
    set text(size: 9.6pt, font: ("Times New Roman", "New Computer Modern", "Libertinus Serif"))
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
  size: 11.0pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.59em,
  spacing: 0.73em,
)

#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(0.54em)
  set text(size: 12.8pt, weight: "bold", fill: rgb("#1B365D"))
  it
  v(0.20em)
}
#show heading.where(level: 2): it => {
  v(0.38em)
  set text(size: 11.4pt, weight: "bold", fill: rgb("#2C4A70"))
  it
  v(0.14em)
}
#show heading.where(level: 3): it => {
  v(0.26em)
  set text(size: 11.0pt, weight: "bold", style: "italic", fill: rgb("#2C4A70"))
  it
  v(0.08em)
}

#set math.equation(numbering: "(1)", supplement: [Eq.])
#show figure: set block(breakable: false, above: 0.80em, below: 0.80em)
#show figure.caption: it => {
  set text(size: 9.1pt, fill: luma(40))
  it
}
#show table: set text(size: 8.4pt)
#show table: set par(justify: false)

#let theorem-box(title, body) = block(
  fill: rgb("#F4F7FA"),
  stroke: (left: 2.5pt + rgb("#1B365D"), rest: 0.4pt + luma(200)),
  inset: (x: 9.5pt, y: 6.2pt),
  radius: 2pt,
  width: 100%,
  breakable: false,
)[
  #text(weight: "bold", fill: rgb("#1B365D"))[#title.] #body
]

// =============================================================================
// PAGE 1: MANDATORY STANDALONE SUMMARY SHEET
// =============================================================================
#align(center)[
  #v(0.08cm)
  #text(size: 10.5pt, weight: "bold", fill: luma(60))[
    2026 Asia and Pacific Mathematical Contest in Modeling (APMCM)
  ]
  #v(0.15cm)
  #table(
    columns: (1fr, 1fr, 1fr),
    align: center,
    stroke: 0.6pt + luma(140),
    inset: 5.5pt,
    [*Problem Chosen* \ #text(size: 12.5pt, weight: "bold", fill: rgb("#1B365D"))[#problem-chosen]],
    [*Team Control Number* \ #text(size: 12.5pt, weight: "bold", fill: rgb("#1B365D"))[#team-id]],
    [*Contest Year* \ #text(size: 12.5pt, weight: "bold", fill: rgb("#1B365D"))[2026]],
  )
  #v(0.22cm)
  #text(size: 13.0pt, weight: "bold")[#paper-title]
  #v(0.16cm)
  #text(size: 11.6pt, weight: "bold", fill: rgb("#1B365D"))[Summary]
]

#v(0.06cm)
#[
#set text(size: 10.2pt)
#set par(leading: 0.56em, spacing: 0.70em)

Cubic $A B X_3$ perovskites for solar-driven water splitting must simultaneously satisfy thermodynamic stability, a visible-light band gap, and straddling redox band edges across $52 times 52 times 7 = 18,928$ candidate compositions. Two physical obstacles dominate this search: $96.12%$ of the grid consists of zero-gap metals ($E_g = 0$), and only $19$ materials ($0.1004%$ under the official non-strict hinge score $V = 0$, or $15$ under strict interior inequalities) meet every requirement—all located within the unlabeled test split. To couple composition-only machine learning with physical coherent Ising machine (CIM) hardware, we develop a physics-coupled hurdle-Bayes surrogate and a family of globally certified 8-bit Quadratic Unconstrained Binary Optimization (QUBO) encodings.

For *Problem 1*, we prevent cross-composition target leakage by grouping all cross-validation splits by $(A,B)$ cation pairs. Rather than fitting a single regressor across the abrupt metal-to-semiconductor discontinuity, we decompose prediction into three physical channels—formation energy, calibrated semiconducting probability, and conditional band gap—trained across four tree-ensemble families within each outer fold. We exploit the empirical log-additive band-center law of GLLB-SC perovskites ($"RMSE" = 2.90 times 10^(-7) "eV"$, $R^2 > 1 - 10^(-12)$) and propagate half the band-gap uncertainty ($sigma_(E_g) slash 2$) into each band edge. Across $567$ unseen validation $(A,B)$ pairs ($3,969$ compounds), the resulting surrogate achieves $"RMSE" = 4.0634$, cuts near-target $"MAE"(V < 1)$ from $5.4475$ to $2.9280$, and recovers $94.4%$ ($17 slash 18$) of $V < 0.5$ validation compounds within the top $50$ predictions ($100.0%$, $18 slash 18$, within the top $75$). Group-disjoint Mondrian conformal calibration attains $90.19%$ and $95.12%$ empirical coverage at nominal $90%$ and $95%$ levels.

For *Problems 2 and 3*, we prove analytically why direct 8-bit quantization of $52$-element one-hot constraints fails on SPQC-550 hardware ($H_(u v) in bb(Z) inter [-127, 127]$): rounding the reference-spin field shifts the continuous penalty minimum from $k = 1$ to $k^* = 4.83$ ($0.0%$ empirical feasibility). We resolve this precision bottleneck through upper-tail truncation at $C_("cap")$, a tight bipartite penalty bound ($P_("block") > (3 slash 2) C_("cap")$), exact $4 times slash 8 times$ integer coupling alignment, and lossless active-subspace pruning—compressing Problem 2 to $17$ spins (full pool) and $12$ spins (blind pool). For three-body $(A,B,X)$ selection in Problem 3, we compare a zero-auxiliary composite $(A,X)$--$B$ QUBO ($26$ and $18$ spins), an asymmetric submodular higher-order reduction ($43$ and $29$ spins), and a uniformly scaled second-order ANOVA QUBO ($33$ and $25$ spins). Exhaustive primary-space enumeration and exact mixed-integer linear programming confirm zero infeasible global ground states across all eight compiled matrices, identifying $"LaTiO"_2"N"$ ($hat(V) = 0.252406$) as the global minimizer.

On the Bose Quantum SPQC-550 optical CIM ($10$ optimization records per task), our four blind-pool Hamiltonians achieve $30.0%$, $90.0%$, $20.0%$ ($90.0%$ primary-feasible), and $30.0%$ strictly feasible returned records; in a single hardware run, the $18$-spin composite encoding captures $9$ distinct top-$15$ multi-anion candidates, including $"LaTiO"_2"N"$ at zero regret. Finally, in $20$-seed iterative discovery under a $200$-query budget ($0.9070$ random expectation), hybrid pre-screening with per-round 8-bit QUBO optimization captures $16.20 slash 18$ rare targets ($"SD" = 1.077$), matching sequential surrogate ranking ($16.55 slash 18$, $"SD" = 1.962$) with tighter cross-seed dispersion.

#v(0.08cm)
*Keywords:* Perovskite Photocatalysis; Physics-Coupled Hurdle Surrogate; Butler--Ginley Band Center; 8-Bit QUBO/Ising Compilation; Submodular Reduction; Coherent Ising Machine (SPQC-550)
]

#pagebreak()

// =============================================================================
// PAGE 2: MANDATORY DIRECTORY / TABLE OF CONTENTS
// =============================================================================
#outline(title: [Contents], indent: 1.4em, depth: 2)
#pagebreak()

// =============================================================================
// MAIN BODY (8 SECTIONS + REFERENCES, STRICTLY <= 25 PAGES)
// =============================================================================

= Introduction and Problem Analysis

== Background: Combinatorial Screening of Water-Splitting Perovskites
Cubic perovskites of stoichiometry $A B X_3$ offer a versatile crystal scaffold for solar-driven photocatalytic water splitting, because independent substitution on the $A$ cation site, $B$ cation site, and $X$ anion sublattice permits fine tuning of thermodynamic stability and electronic band structure [1]. Yet high-fidelity Density Functional Theory (DFT) calculations with the GLLB-SC functional remain computationally expensive. Independent enumeration of $52$ metal cations on the $A$ and $B$ sites across $7$ anionic classes ($"O"_3$, $"NO"_2$ [oxynitride $A B "O"_2"N"$], $"FO"_2$ [oxyfluoride $A B "O"_2"F"$], $"N"_2"O"$, $"N"_3$, $"O"_2"S"$, and $"FNO"$) spans a full factorial grid of $52 times 52 times 7 = 18,928$ compositions. Because exhaustive first-principles screening does not scale to broader multicomponent chemical spaces, we pair a physics-coupled composition surrogate $hat(V)(A, B, X)$ with Quadratic Unconstrained Binary Optimization (QUBO) on a specialized Coherent Ising Machine (CIM) [2, 3].

== Problem Restatement and Three Core Bottlenecks
The competition comprises three progressively coupled modeling stages:
- *Problem 1 (Out-of-Distribution Surrogate Construction):* Train a composition-only model $hat(V)(A,B,X)$ on `02_castelli_train.csv` ($11,256$ rows), perform model selection on `03_castelli_val.csv` ($3,969$ rows), and predict the target violation degree $V$ across `04_castelli_test_holdout.csv` ($3,703$ rows), where all DFT physical quantities are withheld.
- *Problem 2 (Single-Anion Bipartite QUBO Optimization):* Refit the surrogate on the combined labeled splits (`02` + `03`, $15,225$ rows), fix $X = "NO"_2$, and select a cation pair $(A, B)$ minimizing $hat(V)(A, B mid(|) X="NO"_2)$ on both the full candidate grid (`07_castelli_q2_pool.csv`, $2,704$ rows) and blind holdout subset (`11_castelli_q2_pool_blind.csv`, $529$ rows) within the 8-bit integer coupling limits of the Bose Quantum SPQC-550 CIM.
- *Problem 3 (Multi-Anion Three-Body QUBO Optimization):* Jointly optimize $(A, B, X)$ across $X in {"O"_3, "NO"_2, "FO"_2}$ on the full multi-anion pool (`08_castelli_q3_pool.csv`, $8,112$ rows) and blind pool (`12_castelli_q3_pool_blind.csv`, $1,587$ rows), reducing cubic three-body interactions into quadratic 8-bit Ising Hamiltonians and explaining candidate violation costs item by item.

An examination of the dataset structure and CIM hardware constraints reveals three mathematical bottlenecks that defeat standard off-the-shelf pipelines:
1. *Extreme Zero-Inflation and Disjoint Cation Pairs:* Exactly $96.12%$ of the $18,928$ candidates are zero-gap metals ($E_g = 0$), creating a sharp cliff in $V$. Furthermore, the training, validation, and blind test splits partition the $52 times 52 = 2,704$ cation pairs with *zero $(A, B)$ overlap* ($1,608$, $567$, and $529$ pairs, respectively). Ordinary random cross-validation leaks $(A, B)$ identity across anion variants and severely overstates accuracy.
2. *Precision Collapse of One-Hot Constraints on 8-Bit Ising Hardware:* In zero-diagonal Ising form $E(s) = -s^top H s$, a $52$-element one-hot constraint requires a reference-spin local field $(N - 2) = 50$ times stronger than pairwise spin couplings. Mapping both this $50:1$ structural ratio and raw continuous violation costs into signed 8-bit integers ($[-127, 127]$) rounds the pairwise penalty so coarsely that feasible $(1,1)$ selections cease to be ground states.
3. *Cubic Interaction Explosion vs. Auxiliary Spin Fragility:* Standard Rosenberg quadratization [8] over $(A, B)$ introduces $52 times 51 = 2,652$ auxiliary variables ($2,760$ spins), exceeding the $550$-spin capacity of SPQC-550. Even compact higher-order reductions introduce auxiliary-consistency constraints ($y_(i k) = a_i c_k$) that can fracture into infeasible excited states during optical annealing.

== Overview of Our Modeling Framework
@fig:pipeline outlines how our framework addresses these three bottlenecks in sequence. First, $(A, B)$-grouped cross-validation and a multi-task hurdle architecture separate metallic screening from semiconducting band-edge integration via an exact in-data Butler--Ginley band-center law. Second, upper-tail truncation at $C_("cap")$ enables lossless active-subspace pruning and zero-drift integer Ising compilation. Third, three complementary Problem 3 encodings—led by a zero-auxiliary composite $(A,X)$--$B$ QUBO—are benchmarked via exact ground-state certification, physical SPQC-550 hardware runs, and $20$-seed closed-loop active discovery.

#figure(
  image("../v2_optimized/figures/schematic01_pipeline_architecture.png", width: 98%),
  caption: [Overview of the discovery framework: $(A,B)$-grouped physics-coupled hurdle-Bayes surrogate modeling (Problem 1), upper-tail active-subspace pruning and exact 8-bit Ising compilation (Problems 2 & 3), global ground-state certification, and SPQC-550 hardware plus closed-loop verification.],
) <fig:pipeline>

= Data Structure, Assumptions, and Notations

== Dataset Architecture and Split Discipline
Across the $18,928$-material grid, the training split (`02_castelli_train.csv`, $1,608$ $(A,B)$ pairs, $11,256$ rows) and validation split (`03_castelli_val.csv`, $567$ pairs, $3,969$ rows) report both composition and GLLB-SC DFT properties, whereas the blind test set (`04_castelli_test_holdout.csv`, $529$ pairs, $3,703$ rows) and optimization pools (`07`, `08`, `11`, `12`) provide composition only. All Problem 1 validation benchmarks train exclusively on split `02` and evaluate on split `03`. For Problems 2 and 3, we refit the locked surrogate architecture on the combined labeled set (`02` + `03`, $15,225$ rows); blind-set predictions before and after refitting agree closely (Spearman $rho = 0.9970$, sharing $9$ of the top $10$ compositions).

== Target Violation Degree and Physical Edge Conventions
Catalytic suitability is governed by the five-term quadratic hinge violation score:
$ V &= [max(0, (e_("form") - 0.21)/0.5)]^2 + [max(0, (1.4 - E_g)/0.5)]^2 + [max(0, (E_g - 3.1)/0.5)]^2 \
    &quad + [max(0, (5.73 - "VBM")/0.5)]^2 + [max(0, ("CBM" - 4.50)/0.5)]^2. $ <eq:vscore>
Direct evaluation of @eq:vscore across all $15,225$ labeled records reproduces the tabulated `V` column within $max |Delta V| = 1.02 times 10^(-5)$ (6-decimal CSV rounding). Three physical and boundary conventions are fixed throughout:
1. *Physical Metallicity ($E_g = 0$):* All $18,193$ entries with `gap_gllbsc = 0` ($96.12%$) represent gapless metallic phases rather than missing values; each incurs the deterministic lower-gap penalty $v_("gap,low") = (1.4 slash 0.5)^2 = 7.84$ in @eq:vscore.
2. *Boundary Convention ($19$ Non-Strict vs. $15$ Strict Targets):* Under the non-strict hinge condition $V = 0$ ($e_("form") <= 0.21$, $1.4 <= E_g <= 3.1$, $"VBM" >= 5.73$, $"CBM" <= 4.50$), the $18,928$-compound space contains $19$ target materials ($0.1004%$: $6$ each in $"O"_3$, $"NO"_2$, and $"FO"_2$, plus $1$ in $"N"_2"O"$), whereas $15$ compounds satisfy the strict interior inequalities noted in the problem background ($e_("form") < 0.21$, $1.4 < E_g < 3.1$, $"VBM" > 5.73$, $"CBM" < 4.50$). None of the $19$ $V = 0$ materials appear in splits `02` or `03` (which contain $49$ and $18$ near-targets with $V < 0.5$, and $83$ and $27$ with $V < 1.0$, respectively); all $19$ occupy distinct $(A,B)$ pairs inside split `04`. Throughout this paper, we model and optimize the official continuous hinge objective $V$ (@eq:vscore).
3. *Homonuclear Compositions ($A = B$):* Drawing $A$ and $B$ from the same $52$-metal pool yields $364$ homonuclear records ($1.92%$, none with $V = 0$). We retain $A = B$ entries during Problem 1 training to preserve full-factorial structure, while energetically masking $A = B$ pairs in Problems 2 and 3. Excluding versus retaining $A = B$ leaves the surrogate global optimum unchanged at $"LaTiO"_2"N"$ ($hat(V) = 0.252406$), preserves all top $20$ blind-test candidates ($20 slash 20$), and affects no rank above $32$ ($"SnSnO"_3$, $hat(V) = 2.4358$).

== Model Assumptions and Principal Notations
Our formulation rests on four explicit assumptions:
1. *Composition-Only Blind Inference:* All predictions on unlabeled splits depend solely on stoichiometric identity $(A, B, X)$ and universal atomic or ionic constants [5, 6], preventing any DFT feature leakage.
2. *Symmetric Band-Edge Coupling:* Across all $15,225$ labeled GLLB-SC records, valence and conduction band edges satisfy $"VBM" = E_c + E_g slash 2$ and $"CBM" = E_c - E_g slash 2$ around the band center $E_c equiv ("VBM" + "CBM") slash 2$ to CSV rounding precision (median discrepancy $0.0000 "eV"$, maximum $0.0500 "eV"$).
3. *Gaussian Marginal Risk Integration:* Over semiconducting states, formation-energy and band-gap prediction residuals are approximated by Gaussian marginals calibrated from out-of-fold residuals and ensemble spread, providing a closed-form risk estimator for @eq:vscore.
4. *Upper-Tail Objective Truncation:* Candidates with predicted violation above a cutoff $C_("cap")$ well above the pool minimum are irrelevant to minimum-violation search; clipping $hat(V)$ at $C_("cap")$ preserves exact rankings among all candidates satisfying $hat(V) < C_("cap")$.

@tab:symbols lists the principal mathematical symbols used across Sections 3--7.

#figure(
  table(
    columns: (2.7cm, 9.5cm, 4.3cm),
    stroke: none,
    align: (left, left, left),
    inset: (x: 4pt, y: 3.0pt),
    table.hline(stroke: 1.1pt),
    [*Symbol*], [*Physical or Mathematical Meaning*], [*Domain / Unit*],
    table.hline(stroke: 0.6pt),
    [$A, B, X$], [$A$-site metal ($52$), $B$-site metal ($52$), and $X$-site anion class ($7$)], [Discrete indices],
    [$e_("form"), E_g$], [DFT formation energy (tabulated as eV) and fundamental band gap], [$"eV/atom"$, $"eV"$],
    [$"VBM", "CBM", E_c$], [Valence band maximum, conduction band minimum, and band center], [$"eV"$ (vs. vacuum)],
    [$V, hat(V)$], [Ground-truth and surrogate-predicted 5-term quadratic hinge violation], [$[0, infinity)$, dimensionless],
    [$hat(p)_("semi"), hat(P)_("hit")$], [Calibrated probability $P(E_g > 0)$ and model-predicted target-hit score], [$[0, 1]$],
    [$C_("cap"), macron(M)_(i j)$], [Upper-tail truncation threshold and shifted non-positive QUBO cost], [$C_("cap") > 0$, $[-C_("cap"), 0]$],
    [$bold(a), bold(b), bold(c); y_(i k)$], [Primary one-hot selection vectors and one-sided auxiliary binary variables], [${0, 1}^N$; ${0, 1}$],
    [$P_("block"), P_("aux"), P_("forbid")$], [One-hot cardinality penalty, auxiliary gadget weight, and mask penalty], [$bb(R)_+$ / exact integers],
    [$Q, H, C$], [QUBO matrix, zero-diagonal symmetric Ising matrix, and energy offset], [$H_(u v) in bb(Z) inter [-127, 127]$],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Summary of principal notations across surrogate modeling and QUBO/Ising compilation.],
) <tab:symbols>

= Problem 1: Physics-Coupled Hurdle-Bayes Surrogate Model

== Empirical Band-Center Law and the Electronic Feasibility Corridor
Independent regression on VBM and CBM overlooks an exact structural constraint of the Castelli et al. GLLB-SC perovskite dataset [1, 4]. Under the Butler--Ginley model, neutral crystal band centers follow the geometric mean of constituent Mulliken electronegativities—a linear additive law in logarithmic space:

#theorem-box("Proposition 1 (In-Data Log-Additive Band-Center Law & Error Propagation)", [
  On the GLLB-SC perovskite dataset, fitting a log-additive elemental Ridge regression (penalty $lambda = 10^(-10)$) strictly within the training split for the band center $E_c equiv ("VBM" + "CBM") slash 2$,
  $ ln E_c (A, B, X) = w_0 + w_A + w_B + w_X, $ <eq:bg>
  achieves $"RMSE" = 2.90 times 10^(-7) "eV"$ ($R^2 > 1 - 10^(-12)$, i.e., $1.0000000000$ to $10$ decimals) across all $3,969$ validation compounds ($567$ unseen $(A, B)$ pairs), with exact cation site-exchange symmetry ($r(w_A, w_B) = 1.0000$) and strong correlation with Mulliken electronegativity ($r = 0.9183$, Spearman $rho = 0.9295$). Because $"VBM" = E_c + E_g slash 2$ and $"CBM" = E_c - E_g slash 2$, however, exact knowledge of $E_c$ *does not* remove band-edge uncertainty: any error in $E_g$ propagates symmetrically into both band edges with standard deviation $sigma_("VBM") = sigma_("CBM") = sigma_(E_g) slash 2$.
])

Substituting $"VBM" = E_c + E_g slash 2$ and $"CBM" = E_c - E_g slash 2$ into the redox bounds reduces three electronic conditions ($E_g$, VBM, CBM) to a two-dimensional wedge in the $(E_c, E_g)$ plane (@fig:bandwindow):

#theorem-box("Proposition 2 (Necessary Band-Center Corridor for Photocatalytic Water Splitting)", [
  Under symmetric band-edge coupling and the non-strict $V = 0$ electronic bounds ($1.4 <= E_g <= 3.1$, $"VBM" >= 5.73$, $"CBM" <= 4.50$), define the minimum required band gap at band center $E_c$ as $g_(min)(E_c) equiv max(1.4, 2(5.73 - E_c), 2(E_c - 4.50))$. Then there *exists* a band gap $E_g >= 0$ satisfying all electronic conditions simultaneously if and only if
  $ [ exists E_g >= 0 : g_(min)(E_c) <= E_g <= 3.1 ] <==> 4.18 "eV" <= E_c <= 6.05 "eV". $ <eq:ecwin>
  For a specific crystal with actual band gap $E_g$, electronic feasibility requires $g_(min)(E_c) <= E_g <= 3.1$ (and strict inequalities for the $15$-target interior subset). Thus @eq:ecwin is *necessary* rather than sufficient: a metallic crystal at $E_c = 5.10 "eV"$ with $E_g = 0$ lies well within $[4.18, 6.05] "eV"$ yet fails $E_g >= g_(min)(5.10) = 1.4 "eV"$, incurring $v_("gap,low") = 7.84$.
])

#figure(
  image("../v2_optimized/figures/schematic02_band_alignment_window.png", width: 98%),
  caption: [Symmetric band-edge coupling ($sigma_("VBM") = sigma_("CBM") = sigma_(E_g) slash 2$), the metallic false-positive trap ($E_c = 5.10 "eV", E_g = 0$), and the $(E_c, E_g)$ feasibility corridor $4.18 <= E_c <= 6.05 "eV"$.],
) <fig:bandwindow>

To capture crystal-chemistry regularities without discarding quantum-mechanical exceptions, every domain heuristic—Goldschmidt and Bartel tolerance factors [7], Pauling charge balance, Bloch--Wilson electron-count parity, and $d^0 slash d^(10)$ cation configuration—is supplied strictly as a soft input feature rather than an irreversible pre-filter. @tab:phys-audit summarizes each descriptor family and its empirical exception counts across `02` + `03` ($n = 15,225$). For example, $26$ of the $110$ near-target compounds ($V < 1.0$) fall outside the nominal $r_A slash r_B > 1.05$ domain of Bartel's tolerance factor, and $"ZrMgO"_3$ violates the $d^0 slash d^(10)$ rule while maintaining low violation.

#figure(
  table(
    columns: (auto, 1.1fr, 1.35fr, auto, 1.35fr),
    stroke: none,
    align: (left, left, left, center, left),
    inset: (x: 2.8pt, y: 3.2pt),
    table.hline(stroke: 1.1pt),
    [*Descriptor Family*], [*Physical Origin*], [*Modeling Role & Approximation*], [*Hard Filter?*], [*Observed Exceptions (`02+03`, $n=15,225$)*],
    table.hline(stroke: 0.6pt),
    [Butler--Ginley $E_c$], [$ln E_c = w_0 + w_A + w_B + w_X$ [1, 4]], [Fold-internal Ridge; $sigma_("VBM")=sigma_("CBM")=sigma_(E_g) slash 2$], [No], [Validation RMSE $2.90 times 10^(-7)$ eV; $101 slash 110$ ($V<1$) lie in $[4.18, 6.05]$ eV],
    [Corridor Threshold], [Lower wedge $g_(min)(E_c) <= E_g <= 3.1$], [Couples band-gap and redox-edge penalties], [No], [Validation semiconductor residual correlation $r=-0.2930$ ($p=4.22 times 10^(-4)$)],
    [Electron Parity], [Bloch--Wilson filling ($Z mod 2$)], [Closed-shell heuristic; breaks in magnetic/Mott phases], [No], [Odd $Z$ parity: $0 slash 110$ ($V<1$), $0 slash 67$ ($V<0.5$), $5 slash 573$ ($E_g>0$)],
    [Valence & Shell], [Pauling balance $min |q_A+q_B+q_X|$], [Tabulated oxidation states; Fe, Co, Ni, Ru open shells], [No], [Charge imbalance: $3 slash 110$ ($V<1$); open-shell mismatch: $4 slash 110$],
    [Bartel Factor $tau$], [$tau = r_X slash r_B - n_A (n_A - (r_A slash r_B) slash ln(r_A slash r_B))$ [7]], [Defined for $r_A slash r_B > 1.05$; 6-coordinate $r_A$ proxy], [No], [Outside $r_A slash r_B$ domain: $26 slash 110$ ($V<1$); $tau >= 4.18$ (nominal non-perovskite): $87 slash 110$],
    [$d^0 slash d^(10)$ & Ionicity], [Cation $d^0 slash d^(10)$ & Pauling $1 - e^(-0.25 Delta chi^2)$ [5, 6]], [Periodic-group proxy; auxiliary $|mu_B|>0.05$ predictor], [No], [Non-$d^0 slash d^(10)$ $B$-site: $1 slash 110$ ($"ZrMgO"_3$); $|mu_B|>0.05$: $22 slash 573$ ($E_g>0$)],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Physical descriptor families, approximations, and empirical exception counts across `02` + `03`.],
) <tab:phys-audit>

== Fold-Isolated Hurdle Architecture and Expected Hinge Integration
Because $96.12%$ of perovskites are gapless metals while $3.88%$ open a finite semiconductor gap, fitting a single regression ensemble directly to scalar $V$ blurs the metal--semiconductor transition and pulls near-zero predictions upward toward $7.84$. We therefore construct a *Physics-Coupled Hurdle-Bayes (PHB)* surrogate using $154$ composition-only inputs: $3$ categorical site identifiers $(A, B, X)$, $100$ elemental and octahedral descriptors, $7$ analytical band-center features from @eq:bg, $40$ empirical-Bayes ($m = 5$) smoothed marginal statistics, and $4$ auxiliary structural and magnetic predictors (lattice constant, Fermi level, $|mu_B|$, and magnetic-ordering probability).

To respect the disjoint $(A, B)$ split structure, we use outer $5$-fold $(A, B)$-grouped cross-validation. Every target-dependent transformation—log-band-center regression, empirical-Bayes marginals, auxiliary DFT predictors, and four base tree ensembles (CatBoost, LightGBM, XGBoost, and ExtraTrees)—is fitted strictly within each outer training fold [13], preventing cross-pair target leakage. Convex stacking weights, predictive standard deviations ($hat(sigma)_("ef"), hat(sigma)_g$, blending $0.75$ out-of-fold residual scale with $0.25$ inter-family spread), and a $50 slash 50$ isotonic probability calibration are then fitted on pooled base-model out-of-fold predictions across three physical channels:
1. *Formation Energy ($hat(mu)_("ef"), hat(sigma)_("ef")$):* Four-family ensemble ($0.315$ CatBoost, $0.317$ LightGBM, $0.258$ XGBoost, $0.111$ ExtraTrees), attaining validation $R^2 = 0.9199$ and $"RMSE" = 0.2108 "eV/atom"$.
2. *Semiconducting Hurdle Probability ($hat(p)_("semi") = P(E_g > 0)$):* Calibrated classifier ($0.264$ CatBoost, $0.174$ LightGBM, $0.183$ XGBoost, $0.380$ ExtraTrees), reaching validation $"ROC-AUC" = 0.9850$, $"PR-AUC" = 0.8247$, Brier score $0.01334$, and $"ECE" = 0.00570$.
3. *Conditional Semiconducting Band Gap ($hat(mu)_g, hat(sigma)_g$ for $E_g mid(|) E_g > 0$):* Trained solely on $E_g > 0$ crystals ($0.451$ CatBoost, $0.374$ LightGBM, $0.000$ XGBoost, $0.174$ ExtraTrees), yielding validation $R^2 = 0.7587$ and $"RMSE" = 0.7922 "eV"$.

For a Gaussian random variable $Y tilde.op cal(N)(mu, sigma^2)$ entering a one-sided quadratic hinge $[max(0, plus.minus(Y - theta) slash s)]^2$, setting $m = plus.minus(mu - theta) slash s$, $tilde(sigma) = sigma slash s$, and $z = m slash tilde(sigma)$ yields the closed-form expectation
$ cal(E)_plus.minus (mu, sigma; theta, s) = (m^2 + tilde(sigma)^2) Phi(z) + m tilde(sigma) phi(z), $ <eq:hinge>
where $Phi$ and $phi$ are the standard normal CDF and PDF. Let $v_r (0; E_c)$ denote the four deterministic electronic hinge terms at $E_g = 0$ ($"VBM" = "CBM" = E_c$), and let $cal(E)_r (hat(mu)_g, hat(sigma)_g; E_c)$ denote their Gaussian semiconducting expectations—namely $cal(E)_- (hat(mu)_g, hat(sigma)_g; 1.4, 0.5)$ for `gl`, $cal(E)_+ (hat(mu)_g, hat(sigma)_g; 3.1, 0.5)$ for `gh`, $cal(E)_- (E_c + hat(mu)_g slash 2, hat(sigma)_g slash 2; 5.73, 0.5)$ for `VBM`, and $cal(E)_+ (E_c - hat(mu)_g slash 2, hat(sigma)_g slash 2; 4.50, 0.5)$ for `CBM`. Marginalizing over the metallic and semiconducting branches yields our risk predictor $hat(V)_("Bayes")$ and companion target-satisfaction score $hat(P)_("hit")$:
$ hat(V)_("Bayes")(x) = cal(E)_+ (hat(mu)_("ef"), hat(sigma)_("ef"); 0.21, 0.5) + sum_(r in {"gl","gh","VBM","CBM"}) [(1 - hat(p)_("semi")) v_r (0; E_c) + hat(p)_("semi") cal(E)_r (hat(mu)_g, hat(sigma)_g; E_c)], $ <eq:vbayes>
$ hat(P)_("hit")(x) = hat(p)_("semi")(x) dot Phi((0.21 - hat(mu)_("ef")) / hat(sigma)_("ef")) dot max(0, Phi((3.1 - hat(mu)_g) / hat(sigma)_g) - Phi((g_(min)(E_c) - hat(mu)_g) / hat(sigma)_g)). $ <eq:phit>

== Validation Benchmark and Calibration Diagnostics on `03_val`
@tab:q1 compares seven surrogate models on the external validation split (`03_castelli_val.csv`, $3,969$ compounds across $567$ unseen $(A,B)$ pairs): a one-hot factorization machine, direct scalar LightGBM models on $V$ and $ln(1+V)$, two physics-informed multi-task baselines (`PIMT Physics-Only` and `PIMT Hybrid`), a four-family direct scalar ensemble (`Direct 4-Family`), and our `Physics-Hurdle Bayes (PHB)` surrogate.

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (3.45cm, 1.15cm, 1.15cm, 1.12cm, 1.12cm, 1.42cm, 1.52cm, 1.48cm, 1.48cm, 1.48cm, 1.48cm),
      stroke: none,
      align: (left, right, right, right, right, right, right, right, right, right, right),
      inset: (x: 2.2pt, y: 2.8pt),
      table.hline(stroke: 1.1pt),
      [*Surrogate Model*], [*RMSE*], [*MAE*], [*$R^2$*], [*$rho$*], [*MAE ($V<1$)*], [*MAE ($V<0.5$)*], [*Rec\@20 ($V<1$)*], [*Rec\@50 ($V<1$)*], [*Rec\@50 ($V<0.5$)*], [*Rec\@75 ($V<0.5$)*],
      table.hline(stroke: 0.6pt),
      [One-Hot FM ($k=8$)], [4.9489], [2.8536], [0.7970], [0.9217], [10.4171], [9.9826], [7.4%], [29.6%], [33.3%], [33.3%],
      [Direct LightGBM ($ln(1+V)$)], [4.6942], [2.7469], [0.8174], [0.9292], [5.8008], [5.2440], [29.6%], [63.0%], [72.2%], [72.2%],
      [Direct LightGBM ($V$)], [4.5568], [2.5383], [0.8279], [0.9365], [9.9202], [9.5674], [29.6%], [44.4%], [50.0%], [66.7%],
      [PIMT (Physics Only)], [4.2053], [2.2068], [0.8534], [0.9498], [N/A], [N/A], [51.9%], [77.8%], [N/A], [N/A],
      [PIMT (Hybrid Ensemble)], [4.2091], [2.2180], [0.8532], [0.9493], [N/A], [N/A], [51.9%], [77.8%], [N/A], [N/A],
      [Direct 4-Family Ensemble], [4.1834], [2.2420], [0.8550], [0.9472], [5.4475], [5.0721], [37.0%], [*85.2%*], [88.9%], [94.4%],
      [*Physics-Hurdle Bayes (PHB)*], [*4.0634*], [*2.0696*], [*0.8632*], [*0.9527*], [*2.9280*], [*2.4126*], [*55.6%*], [81.5%], [*94.4%*], [*100.0%*],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Out-of-pair validation benchmark on `03_castelli_val.csv` ($3,969$ compounds, $567$ unseen $(A,B)$ pairs; trained solely on split `02`). PHB attains $100.0%$ recall of $V < 1.0$ targets ($27 slash 27$) at Top-100 (`N/A`: local $V < 0.5$ slice not logged in summary multi-task baseline).],
) <tab:q1>

@tab:q1, @fig:q1metrics, and @fig:q1calib highlight two key findings. First, even with identical $154$-feature tree ensembles, direct scalar regression (`Direct 4-Family`) incurs $"MAE"(V < 1) = 5.4475$ because metallic leaf averages pull low-violation predictions upward. Separating metallic classification from conditional band-gap integration cuts near-target $"MAE"(V < 1)$ from $5.4475$ to $2.9280$ ($"MAE"(V < 0.5) = 2.4126$), lowers overall RMSE to $4.0634$, and captures $94.4%$ ($17 slash 18$) of $V < 0.5$ validation compounds within the top $50$ predictions ($100.0%$, $18 slash 18$, within the top $75$). Second, residual diagnostics (@fig:q1calib) confirm negligible global bias ($-0.0328$) and tight probability calibration ($"ECE" = 0.00570$); residuals are naturally wider on true semiconductors ($"RMSE" = 9.4570$, where finite band gaps span $V in [0, 80]$) than on metals ($"RMSE" = 3.7182$).

#figure(
  image("../v2_optimized/figures/fig01_v1_vs_v2_surrogate_metrics.png", width: 96%),
  caption: [Validation performance on split `03`: (a) global RMSE and MAE across seven surrogates; (b) near-target recall at Top-50 and Top-100 ($V < 1.0$, $27$ targets); (c) local error in the catalytic regime ($V < 1.0$ and $V < 0.5$); (d) parity plot of the fold-internal Butler--Ginley band-center law.],
) <fig:q1metrics>

#figure(
  image("../v2_optimized/figures/fig02_residual_and_calibration.png", width: 96%),
  caption: [Diagnostic audit on split `03`: (a) residual distributions separated by metallic ($"RMSE" = 3.7182$) and semiconducting ($"RMSE" = 9.4570$) phases; (b) local MAE across band-center $E_c$ windows; (c) 10-bin reliability curve for $hat(p)_("semi")$ ($"ROC-AUC" = 0.9850$, $"ECE" = 0.00570$).],
) <fig:q1calib>

== Conformal Uncertainty Quantification and Blind Holdout (`04`) Screening
To assess distribution-free uncertainty, we evaluate $200$ group-disjoint $50 slash 50$ Mondrian conformal splits on `03_val` ($283$ calibration vs. $284$ test $(A,B)$ pairs per split) [11, 12], stratified at $hat(p)_("semi") >= 0.20$. Empirical coverage closely matches nominal levels at $90.19% plus.minus 1.65%$ (nominal $90%$) and $95.12% plus.minus 1.13%$ (nominal $95%$). At $90%$ confidence, conformal half-widths equal $4.411 plus.minus 0.252$ on the predicted metallic stratum ($hat(p)_("semi") < 0.20$, $n = 3,808$) versus $12.375 plus.minus 0.848$ on the predicted semiconducting stratum ($hat(p)_("semi") >= 0.20$, $n = 161$), showing that point rankings effectively concentrate rare semiconductors into the top $1%$--$2%$ of a pool even though individual semiconducting intervals remain wide relative to $V < 0.5$.

On the blind test set (`04_test_holdout`, $3,703$ compounds; refitted on `02` + `03`), high-scoring candidates cluster inside the $[4.18, 6.05] "eV"$ band-center corridor (@fig:conformal), led by oxynitrides ($"LaTiO"_2"N"$ at $hat(V)=0.2524$, $"SrTaO"_2"N"$ at $0.2576$), oxides ($"CaGeO"_3$ at $0.6458$, $"SrGeO"_3$ at $0.7042$), oxysulfides ($"SnZrO"_2"S"$ at $0.6774$, $"NaTaO"_2"S"$ at $0.7225$), dinitrides ($"YTaON"_2$ at $1.3855$), and oxyfluorides ($"BaInO"_2"F"$ at $1.9959$). Across all $3,703$ blind compositions, @eq:phit assigns a total predicted hit mass of $sum hat(P)_("hit") = 10.3358$ ($"NO"_2$: $4.48$, $"O"_3$: $4.48$, $"FO"_2$: $0.77$, $"O"_2"S"$: $0.43$, $"N"_2"O"$: $0.13$, $"FNO"$: $0.05$, $"N"_3$: $0.00$), with $83.2%$ ($8.6001$) concentrated in the top $30$ candidates ($0.81%$ of the pool) and $91.8%$ ($9.4889$) in the top $50$. Because ground-truth labels on split `04` are withheld for official scoring, $sum hat(P)_("hit")$ represents surrogate-assigned probability mass rather than empirical recall against the $19$ hidden targets.

#figure(
  image("../v2_optimized/figures/fig08_electronic_phase_and_conformal_audit.png", width: 96%),
  caption: [Blind holdout (`04_test_holdout`, `02+03` refit) electronic screening: (a) predicted $(E_c, hat(e)_("form"))$ phase map highlighting the $[4.18, 6.05] "eV"$ corridor; (b) cumulative model-assigned hit mass $sum hat(P)_("hit")$ vs. screening budget ($83.2%$ in Top-30); (c1--c4) four-property target-window audit of top blind candidates.],
) <fig:conformal>

= Problem 2: Active-Subspace Bipartite QUBO under 8-Bit Precision

== Exact QUBO-Ising Bijection and Full-Grid 8-Bit Quantization Drift
With the anion site fixed at $X = "NO"_2$, Problem 2 seeks a cation pair $(A_i, B_j)$ minimizing $hat(V)_(i j) = hat(V)(A_i, B_j, "NO"_2)$. Let binary one-hot vectors $bold(a) in {0, 1}^(N_A)$ and $bold(b) in {0, 1}^(N_B)$ select the $A$- and $B$-site elements. On the SPQC-550 CIM (Kaiwu SDK 1.4.4 [2]), any symmetrized QUBO $x^top Q x$ ($x in {0, 1}^n$) maps bijectively to a homogeneous, symmetric, zero-diagonal Ising Hamiltonian $E(s) = -s^top H s$ over $s = (s_1, dots, s_n, s_("ref"))^top in {-1, +1}^(n+1)$ via $x_i = (1 + s_i s_("ref")) slash 2$:
$ H_(i j) = -(1)/(4) Q_(i j) quad (i != j), &quad H_(i, "ref") = -(1)/(4) sum_(j=1)^n Q_(i j), \
  x^top Q x = -s^top H s + C, &quad C = (1)/(4)(sum_(i,j=1)^n Q_(i j) + op("tr") Q), $ <eq:ising-recovery>
with inverse recovery $Q_(i j) = -4 H_(i j)$ ($i != j$) and $Q_(i i) = -4 H_(i, "ref") - sum_(j != i) Q_(i j)$.

Under @eq:ising-recovery, an $N$-element one-hot penalty $P(sum_(i=1)^N a_i - 1)^2$ yields pairwise couplings $H_(i j)^("pen") = -P slash 4$ ($i != j$) and reference-spin couplings $H_(i, "ref")^("pen") = -(N - 2) P slash 4$. Preserving the one-hot minimum at Hamming weight $k = 1$ requires maintaining the exact ratio $H_(i, "ref")^("pen") slash H_(i j)^("pen") = N - 2$. When couplings are rounded to signed 8-bit integers ($[-127, 127]$) over $N = 52$ elements, this ratio breaks down:

#theorem-box("Theorem 1 (8-Bit Quantization Drift of Large One-Hot Blocks)", [
  Consider an $N$-variable one-hot penalty mapped to signed $B$-bit Ising integers bounded by $M_B = 2^(B-1) - 1$. Scaling the largest coupling $(N - 2) P slash 4$ to $M_B$ rounds the pairwise spin coupling to $J_B = op("round")(M_B slash (N - 2))$, shifting the continuous stationary point of the quantized penalty energy over Hamming weight $k = sum_(i=1)^N a_i$ from $k = 1$ to
  $ k^* = N / 2 - M_B / (2 J_B). $ <eq:drift>
  For $N = 52$ elements on 8-bit hardware ($M_8 = 127$), $J_8 = op("round")(127 slash 50) = 3$ shifts the continuous stationary point to $k^* = 4.83$ (discrete integer penalty minimum $k = 5$). Feasible $1$-hot selections are thus energetically destabilized before objective costs are added, accounting for the $0.0%$ feasibility observed in full-grid 8-bit QUBO runs.
])
*Proof.* By spin-flip symmetry $E(-s) = E(s)$, fix $s_("ref") = +1$. Any binary vector of Hamming weight $k$ places $k$ spins at $+1$ and $N - k$ at $-1$, so $sum_i s_i = 2k - N$ and $sum_(i != j) s_i s_j = (2k - N)^2 - N$. With quantized couplings $H_(i j) = -J_B$ and $H_(i, "ref") = -M_B$, the Ising penalty energy is $E_("pen")(k) = J_B [(2k - N)^2 - N] + 2 M_B (2k - N)$. Setting $d E_("pen") slash d k = 8 J_B (k - N slash 2) + 4 M_B = 0$ yields $k^* = N slash 2 - M_B slash (2 J_B)$. For $N = 52$ and $M_8 = 127$ ($J_8 = 3$), $k^* = 26 - 127 slash 6 = 4.833$, whose discrete integer minimizer is $k = 5$. #h(1fr) $square$

== Upper-Tail Truncation, Tight Penalty Bounds, and Lossless Pruning
Theorem 1 dictates two design principles for exact 8-bit compilation: bound the objective dynamic range and reduce the active block size $N$ without losing the optimal perovskite. Distinguishing between mediocre crystals at $hat(V) = 15$ and poor metals at $hat(V) = 90$ expends limited integer dynamic range without aiding minimum-violation search. Given an upper-tail cutoff $C_("cap")$ well above the pool minimum, we shift all valid non-homonuclear pairs into $[-C_("cap"), 0]$:
$ macron(M)_(i j) = cases(min(hat(V)_(i j), C_("cap")) - C_("cap") in [-C_("cap"), 0] & "if " (A_i, B_j) in cal(P) " and " A_i != B_j, 0 & "otherwise,") $ <eq:mbar>
and formulate the masked bipartite QUBO Hamiltonian
$ cal(H)_("Q2")(bold(a), bold(b)) = sum_(i,j) macron(M)_(i j) a_i b_j + P_("block") (sum_i a_i - 1)^2 + P_("block") (sum_j b_j - 1)^2 + sum_((i,j) in.not cal(P)_("valid")) P_("forbid") a_i b_j, $ <eq:q2qubo>
where $cal(P)_("valid") = {(i,j) : (A_i, B_j) in cal(P) "and" A_i != B_j}$ and $P_("forbid") >= P_("block")$. This non-positive shift yields a tight uniform penalty bound and lossless active-subspace pruning:

#theorem-box("Theorem 2 (Uniform Bipartite Penalty Sufficiency & Exact Integer Alignment)", [
  1. Let $macron(M)_(i j) in [-C_("cap"), 0]$ have minimum valid-pair cost $m^* equiv min_((i,j) in cal(P)_("valid")) macron(M)_(i j) in [-C_("cap"), 0]$, and let $P_("forbid") >= P_("block")$. Setting $P_("block") > (3 slash 2) C_("cap")$ *suffices* uniformly across the matrix class so that every non-one-hot state $(k_A, k_B) != (1, 1)$ has strictly higher energy than $m^*$, guaranteeing that every global minimizer of @eq:q2qubo is strictly feasible ($3 slash 2$ is tight over the matrix class at $k_A = k_B = 2$).
  2. Quantizing $m_(i j) = op("round")(M_(max) macron(M)_(i j) slash C_("cap")) in [-M_(max), 0]$ with symmetrized bilinear weights $Q_("obj") = 8 m_(i j)$, integer penalty quantum $j_("pen") = ceil(2 alpha_P M_(max))$ ($alpha_P = 2.25$), $P_("block") = 4 j_("pen")$, and $P_("forbid") = 8 ceil(j_("pen") slash 2)$ ensures every Ising entry $H_(u v)$ in @eq:ising-recovery is an *exact integer* with zero rounding drift ($k_A^* = k_B^* = 1$) whenever $max_(u,v) |H_(u v)| <= 127$.
])
*Proof.* If $k_A = 0$ or $k_B = 0$, no bilinear term $a_i b_j$ is active and $(k_A - 1)^2 + (k_B - 1)^2 >= 1$, giving $cal(H)_("Q2") >= P_("block") > 0 >= m^*$. For $k_A >= 1, k_B >= 1$ with $(k_A, k_B) != (1, 1)$, each of the $k_A k_B$ active pairs either belongs to $cal(P)_("valid")$ (contributing at least $m^*$) or incurs $+P_("forbid") >= P_("block") > 0 >= m^*$. Since $m^* >= -C_("cap")$, total pair cost is at least $k_A k_B m^* = m^* + (k_A k_B - 1) m^* >= m^* - (k_A k_B - 1) C_("cap")$. Hence $cal(H)_("Q2") - m^* >= P_("block")[(k_A - 1)^2 + (k_B - 1)^2] - (k_A k_B - 1) C_("cap") > 0$ whenever $P_("block") > C_("cap") max_((k_A, k_B) != (1,1)) (k_A k_B - 1) slash [(k_A - 1)^2 + (k_B - 1)^2] = (3 slash 2) C_("cap")$ at $k_A = k_B = 2$. Under $Q_("obj") = 8 m_(i j)$, $P_("block") = 4 j_("pen")$, and $P_("forbid") = 8 ceil(j_("pen") slash 2)$, every symmetrized QUBO off-diagonal entry ($4 m_(i j)$, $4 j_("pen")$, or $4 ceil(j_("pen") slash 2)$) and row sum is a multiple of $4$, making $H_(i j)$ and $H_(i, "ref")$ in @eq:ising-recovery exact integers for every integer $j_("pen")$. #h(1fr) $square$

Note that global optimality only requires infeasible states to exceed $m^*$ rather than *every* feasible state (for instance, on a $2 times 2$ block of $-1$s with $C_("cap") = 1$, $P_("block") = 1.6$, and a separate $0$-cost valid pair, $-4 + 3.2 = -0.8 < 0$ yet $-0.8 > -1 = m^*$).

#theorem-box([Theorem 3 (Lossless Active-Subspace Reduction below $C_("cap")$)], [
  Any $A$-site element $i$ with $min_j hat(V)_(i j) >= C_("cap")$ (or $B$-site element $j$ with $min_i hat(V)_(i j) >= C_("cap")$) has identically zero objective entries $macron(M)_(i j) = 0$ in @eq:mbar and cannot participate in any solution satisfying $hat(V) < C_("cap")$. Removing all such inactive rows and columns preserves the *exact* optimal set below $C_("cap")$, compressing total Ising spins from $105$ to $17$ on Full Pool `07` ($C_("cap") = 2.0, M_(max) = 4, P_("block") = P_("forbid") = 72$) and $12$ on Blind Pool `11` ($C_("cap") = 2.5, M_(max) = 4, P_("block") = P_("forbid") = 72$).
])

== Certified Problem 2 Results and Item-by-Item Cost Breakdown
Complete enumeration across all $2^(16) = 65,536$ primary states of `Q2_full_07` ($17$ spins, $max |H_(u v)| = 125$) and all $2^(11) = 2,048$ primary states of `Q2_blind_11` ($12$ spins, $max |H_(u v)| = 115$) confirms zero infeasible ground states at exact Ising ground energies $-1558$ (offset $C = 1390$, degeneracy $10$) and $-1000$ (offset $C = 824$, degeneracy $4$). Both matrices recover the continuous surrogate minimizer $"LaTiO"_2"N"$ ($hat(V) = 0.252406$, continuous regret $0.000000$), tied at the quantized 8-bit ground level ($M_(max) = 4$) with $"SrTaO"_2"N"$ ($0.257617$), $"BaTaO"_2"N"$ ($0.261166$), and $"CaTaO"_2"N"$ ($0.265979$). @fig:q2(a) confirms that active-subspace QUBO maintains $100.0%$ simulated-annealing feasibility from $5$ to $12$ bits and exact integer couplings from $7$ to $12$ bits, whereas full-grid QUBO flatlines at $0.0%$ below $12$ bits.

@tab:q2top10 and @fig:q2(b) explain the physical violation costs item by item across the top $10$ blind candidates in `11_castelli_q2_pool_blind.csv`:
- *Ranks 1--4 ($"LaTiO"_2"N"$, $"SrTaO"_2"N"$, $"BaTaO"_2"N"$, $"CaTaO"_2"N"$):* Closed-shell $d^0$ transition metals ($"Ti"^(4+), "Ta"^(5+)$) paired with large electropositive $A$-cations place $E_c$ squarely at $5.08$--$5.39 "eV"$ and yield favorable formation energies ($hat(mu)_("ef") <= 0.023 "eV/atom"$, $hat(v)_("form") <= 0.011$). Their residual expected cost ($hat(V) approx 0.25$--$0.27$) arises almost solely from band-gap predictive spread ($hat(sigma)_g approx 0.58 "eV"$) overlapping the $1.4$ and $3.1 "eV"$ window edges ($hat(v)_("gl") + hat(v)_("gh") approx 0.18$--$0.22$), supporting high target probabilities ($hat(P)_("hit") = 61.4%$--$76.6%$).
- *Ranks 5--7 ($"YTiO"_2"N"$, $"CaNbO"_2"N"$, $"SrNbO"_2"N"$):* Smaller $"Y"^(3+)$ induces octahedral distortion penalty in $"YTiO"_2"N"$ ($hat(v)_("form") = 0.1502$), whereas the lower $4d$ orbital energy of $"Nb"^(5+)$ narrows predicted gaps to $1.81$--$1.85 "eV"$, elevating lower-gap penalty ($hat(v)_("gl") approx 0.28$--$0.31$).
- *Ranks 8--10 ($"ScHfO"_2"N"$, $"MgTaO"_2"N"$, $"CdTaO"_2"N"$):* $"ScHfO"_2"N"$ incurs severe formation instability ($hat(v)_("form") = 0.7113$), while more electronegative $"Mg"^(2+)$ and $"Cd"^(2+)$ shift $E_c$ down toward $5.76$--$5.96 "eV"$, violating the conduction-band $"H"^+$ slash $"H"_2$ reduction limit ($hat(v)_("CBM") = 0.6807$ and $2.7297$).

#figure(
  image("../v2_optimized/figures/fig04_q2_v1_vs_v2_hardware_drift.png", width: 96%),
  caption: [Problem 2 results: (a) independent $4$--$12$ bit quantization sweep on `Q2_blind_11` contrasting full-grid 8-bit failure ($0.0%$ feasibility below $12$ bits) against active-subspace exact-integer compilation; (b) five-term physical violation breakdown of the Q2 Blind Top-10 candidates.],
) <fig:q2>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    align: (center, left, right, right, right, right, right, right, right, right, right),
    inset: (x: 3.0pt, y: 3.6pt),
    table.hline(stroke: 1.1pt),
    [*Rank*], [*Formula*], [*$hat(V)$*], [*$hat(P)_("hit")$*], [*$hat(E)_g$*], [*$E_c$*], [*$hat(v)_("form")$*], [*$hat(v)_("gl")$*], [*$hat(v)_("gh")$*], [*$hat(v)_("VBM")$*], [*$hat(v)_("CBM")$*],
    table.hline(stroke: 0.6pt),
    [1], [*$"LaTiO"_2"N"$*], [*0.2524*], [61.4%], [2.657], [5.328], [0.0107], [0.0500], [0.1679], [0.0038], [0.0201],
    [2], [*$"SrTaO"_2"N"$*], [*0.2576*], [*76.6%*], [2.522], [5.079], [0.0020], [0.1119], [0.1020], [0.0236], [0.0182],
    [3], [*$"BaTaO"_2"N"$*], [*0.2612*], [71.0%], [2.357], [5.385], [0.0012], [0.1314], [0.0530], [0.0068], [0.0688],
    [4], [*$"CaTaO"_2"N"$*], [*0.2660*], [70.2%], [2.487], [5.177], [0.0085], [0.1222], [0.0893], [0.0179], [0.0282],
    [5], [$"YTiO"_2"N"$], [0.3363], [21.9%], [2.577], [5.387], [0.1502], [0.0360], [0.1264], [0.0018], [0.0219],
    [6], [$"CaNbO"_2"N"$], [0.3809], [63.1%], [1.846], [5.151], [0.0108], [0.2766], [0.0048], [0.0356], [0.0531],
    [7], [$"SrNbO"_2"N"$], [0.4144], [69.7%], [1.805], [5.054], [0.0022], [0.3068], [0.0039], [0.0673], [0.0342],
    [8], [$"ScHfO"_2"N"$], [0.9526], [1.5%], [2.407], [5.424], [0.7113], [0.1057], [0.0650], [0.0043], [0.0663],
    [9], [$"MgTaO"_2"N"$], [1.0059], [6.8%], [1.994], [5.760], [0.0857], [0.2294], [0.0102], [0.0000], [0.6807],
    [10], [$"CdTaO"_2"N"$], [3.6227], [0.1%], [1.398], [5.955], [0.0810], [0.8116], [0.0004], [0.0000], [2.7297],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Item-by-item violation decomposition of the Q2 Blind Pool (`11`, $X = "NO"_2$) Top-10 candidates.],
) <tab:q2top10>

= Problem 3: Multi-Anion Three-Body Reduction and Certification

== Three Quadratic Formulations for Cubic $(A, B, X)$ Optimization
Joint selection over $X in {"O"_3, "NO"_2, "FO"_2}$ defines a cubic objective $sum_(i,j,k) macron(T)_(i j k) a_i b_j c_k$ over one-hot vectors $bold(a) in {0,1}^(N_A)$, $bold(b) in {0,1}^(N_B)$, and $bold(c) in {0,1}^3$, with shifted non-positive tensor $macron(T)_(i j k) = min(hat(V)_(i j k), C_("cap")) - C_("cap") <= 0$ ($C_("cap") = 1.25$ on Pool `08`, $1.50$ on Blind Pool `12`) and quantized integer tensor $m_(i j k) = op("round")(M_(max) macron(T)_(i j k) slash C_("cap")) in [-M_(max), 0]$. Because the three dimensions are strongly asymmetric ($52 times 52 times 3$), pairing $(A, B)$ in full space demands $52 times 51 = 2,652$ auxiliary bits ($2,760$ spins), whereas pairing $(A, X)$ requires only $52 times 3 = 156$ auxiliary bits ($264$ spins, already below the 550-spin SPQC-550 limit). Combining axis asymmetry with active-subspace pruning (@fig:q3comp) yields three 8-bit formulations that trade auxiliary spin count against exactness:

1. *Zero-Auxiliary Composite $(A,X)$--$B$ QUBO (`Model 3A-Comp`):* Instead of maintaining separate $bold(a)$ and $bold(c)$ registers coupled by auxiliary penalties, we index all active pairs $(i, k) in cal(A) = {(i,k) : min_j hat(V)_(i j k) < C_("cap")}$ by a single composite one-hot vector $u_(i k) in {0, 1}^(|cal(A)|)$ ($sum_((i,k) in cal(A)) u_(i k) = 1$) coupled directly to $b_j$. This folds the cubic objective into an exact bipartite QUBO with *zero auxiliary spins*: $26$ spins on `Q3_full_08` ($M_(max) = 3, P_("block") = 28$) and $18$ spins on `Q3_blind_12` ($M_(max) = 4, P_("block") = 40$).
2. *Asymmetric Submodular HOBO Reduction (`Model 3A-HOBO`):* Keeping explicit $bold(a), bold(b), bold(c)$ registers, we introduce binary gadgets $y_(i k) in {0, 1}$ for $(i, k) in cal(A)$. Because $8 m_(i j k) <= 0$ is submodular, Freedman--Drineas one-sided gadgets [9] linearize $a_i b_j c_k$ via
   $ cal(H)_("HOBO")(z, y) &= sum_((i,k) in cal(A)) [ sum_j 8 m_(i j k) y_(i k) b_j + P_("aux")(i,k) y_(i k)(2 - a_i - c_k) ] \
       &quad + P_("block")[(sum a_i - 1)^2 + (sum b_j - 1)^2 + (sum c_k - 1)^2] + H_("mask")(z), $ <eq:h3a>
   with $P_("block") = 4 j_("pen")$ and eligibility mask $H_("mask")(z) equiv sum_((i,j) in.not cal(P)_(A B)) P_("forbid") a_i b_j$ ($P_("forbid") = 8 ceil(j_("pen") slash 2) >= P_("block")$: $40$ vs. $36$ on `08`, $56$ vs. $52$ on `12`). Note that a two-body $(A,B)$ mask is exact for Blind Pool `12` because contest splits are partitioned strictly by $(A,B)$ cation pairs across all three anions.
3. *Uniformly Scaled Second-Order ANOVA QUBO (`Model 3B-ANOVA`):* Spanning the element sets of the top-$K$ legal pool candidates ($K = 35$ on Pool `08`, $33$ spins; $K = 25$ on Blind Pool `12`, $25$ spins; $0$ auxiliary spins), we fit a second-order functional ANOVA expansion $macron(T)_(i j k) approx mu + f_A (i) + f_B (j) + f_X (k) + f_(A B)(i,j) + f_(A X)(i,k) + f_(B X)(j,k)$ and apply uniform $8 times$ integer scaling across both linear main effects and pairwise couplings ($M_(max) = 2$), forming a $6$-level pairwise approximation benchmark.

In `Model 3A-HOBO`, setting $P_("aux")(i,k)$ only above the single largest entry $max_j |8 m_(i j k)|$ fails outside the one-hot manifold whenever $a_i c_k = 0$ and multiple $b_j = 1$ turn on simultaneously. Let $H_("obj")(z) equiv sum_((i,k) in cal(A)) sum_j 8 m_(i j k) a_i b_j c_k$, and let $min H_("obj")(k_A, k_B, k_X)$ denote its minimum over primary vectors of cardinalities $(k_A, k_B, k_X)$:

#theorem-box("Theorem 4 (Global Auxiliary Elimination & Ground-State Exactness of Model 3A-HOBO)", [
  With row-sum auxiliary penalty
  $ P_("aux")(i,k) = sum_j |8 m_(i j k)| + 8 > sum_j |8 m_(i j k)| $ <eq:paux>
  and cardinality-partitioned block penalty
  $ P_("block") > max_((k_A, k_B, k_X) != (1,1,1)) (-min H_("obj")(k_A, k_B, k_X)) / ((k_A - 1)^2 + (k_B - 1)^2 + (k_X - 1)^2), $ <eq:pblock>
  every primary assignment $z = (bold(a), bold(b), bold(c)) in {0, 1}^(n_("prim"))$ satisfies: (i) $y_(i k) = a_i c_k$ attains the exact auxiliary minimum
  $ min_y cal(H)_("HOBO")(z, y) = H_("obj")(z) + P_("block")[(k_A-1)^2 + (k_B-1)^2 + (k_X-1)^2] + H_("mask")(z), $ <eq:hoboelim>
  and (ii) @eq:h3a has *zero infeasible global ground states* across the unconstrained $2^N$ spin space ($43$ spins on `08`, $P_("aux") <= 48, P_("block") = 36$; $29$ spins on `12`, $P_("aux") <= 56, P_("block") = 52$), with zero auxiliary-tie ambiguity at the certified ground energy.
])
*Proof.* Fix any $z = (bold(a), bold(b), bold(c))$. In @eq:h3a, $y_(i k)$ enters linearly with coefficient $c_(i k)(z) = P_("aux")(i,k)(2 - a_i - c_k) + sum_j 8 m_(i j k) b_j$. If $a_i c_k = 0$, then $2 - a_i - c_k >= 1$, so @eq:paux gives $c_(i k)(z) >= P_("aux")(i,k) - sum_j |8 m_(i j k)| >= 8 > 0$, making $y_(i k) = 0$ the unique minimizer even when $b_j = 1$ for all $j$. If $a_i c_k = 1$, then $2 - a_i - c_k = 0$ and $c_(i k)(z) = sum_j 8 m_(i j k) b_j <= 0$: when strictly negative, $y_(i k) = 1$ is uniquely optimal; when zero, both $y_(i k) in {0,1}$ give energy $0 = a_i c_k sum_j 8 m_(i j k) b_j$. Because the block penalties and $H_("mask")(z)$ depend only on $z$, minimizing over $y$ preserves $H_("mask")(z)$ verbatim in @eq:hoboelim, and @eq:pblock together with $H_("mask")(z) >= 0$ forces every non-one-hot or masked primary state strictly above the legal ground energy. #h(1fr) $square$

#figure(
  image("../v2_optimized/figures/schematic03_qubo_ising_compilation.png", width: 98%),
  caption: [Problem 3 compilation and verification workflow: active-subspace encodings (`Model 3A-Comp`, `Model 3A-HOBO`, `Model 3B-ANOVA`), exact algebraic QUBO-Ising bijection (@eq:ising-recovery), global ground-state certification, and spin decoding.],
) <fig:q3comp>

== Exact Ground-State Certification, Spin Compression, and Error Attribution
@tab:cert reports global ground-state certification across all eight integer Ising matrices. Recovering $Q$ algebraically from the 8-bit couplings $H$ (@eq:ising-recovery), eliminating auxiliary variables via $min_y x^top Q x = z^top Q_(z z) z + sum_r min(0, Q_(r r) + 2 z^top Q_(z, r))$, and evaluating every primary assignment via vectorized $2^(n_("prim"))$ enumeration [13] or zero-gap cardinality partitioning confirms *zero infeasible ground states* and *zero ground-energy auxiliary ties* on all eight matrices. An independent Fortet--McCormick mixed-integer linear programming verification script [15], solved to zero optimality gap in HiGHS, reproduces the identical ground-state energy on every instance. Seven of the eight integer ground-state sets include $"LaTiO"_2"N"$ ($hat(V) = 0.252406$; on `Q3_full_3B_ANOVA`, the best ground state is $"SrTaO"_2"N"$ at $hat(V) = 0.257617$, regret $0.005211$), with $3$--$10$ quantized ground-state ties per matrix.

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (2.50cm, 1.58cm, 1.32cm, 1.55cm, 1.75cm, 1.75cm, 0.78cm, 1.15cm, 2.27cm, 2.25cm),
      stroke: none,
      align: (left, center, center, center, left, right, center, center, right, left),
      inset: (x: 2.2pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*Matrix ID*], [*Spins*], [*$M_(max)$/$P_("blk")$*], [*DR / `max|H|`*], [*Cert. Method*], [*`cert_min_E` / $C$*], [*Deg.*], [*Ill. / Tie*], [*Best/Worst Reg.*], [*Digest (12 hex)*],
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
  caption: [Global ground-state certification across all eight 8-bit Ising matrices (`[-127, 127]`), verified identically by primary-space enumeration and Fortet--McCormick MILP.],
) <tab:cert>

@tab:attrib separates the sources of spin compression in Problem 3: switching quadratization from $(A, B)$ to $(A, X)$ yields a $17.0 times$ auxiliary reduction ($2,760 -> 264$ spins), active-subspace pruning lowers spins to $43 slash 29$ (`Model 3A-HOBO`), and composite $(A,X)$--$B$ indexing (`Model 3A-Comp`) removes all auxiliary bits ($26 slash 18$ spins). @tab:anova decomposes `Model 3B-ANOVA` across three stages: second-order float projection ($R^2 = 0.4559$ on `08`, $0.4902$ on `12`), 8-bit rounding ($6$ distinct legal energy levels; Spearman $rho = 0.2070$ on `08`, $0.4512$ on `12`), and heuristic solver search.

#figure(
  {
    set text(size: 7.8pt)
    table(
      columns: (3.45cm, 1.75cm, 2.15cm, 1.15cm, 2.15cm, 2.15cm, 4.20cm),
      stroke: none,
      align: (left, center, right, center, right, right, left),
      inset: (x: 2.6pt, y: 2.8pt),
      table.hline(stroke: 1.1pt),
      [*Encoding Stage*], [*Domain*], [*Prim / Aux / Spins*], [*`<= 550`?*], [*Aux Reduction vs. A-B*], [*Aux Reduction vs. Full A-X*], [*Structural Mechanism*],
      table.hline(stroke: 0.6pt),
      [Full-space `A-B` Rosenberg], [52×52×3], [107 / 2,652 / 2,760], [No], [1.0×], [0.0588×], [Exceeds SPQC-550 spin limit],
      [Full-space `A-X` Rosenberg], [52×52×3], [107 / 156 / 264], [Yes], [17.0×], [1.0×], [Axis asymmetry alone fits 550 spins],
      [Active `A-X` (`Model 3A-HOBO`)], [Full `08` \ Blind `12`], [24/18/43 \ 17/11/29], [Yes], [147.3× \ 241.1×], [8.67× \ 14.18×], [Subspace pruning below $C_("cap")$],
      [Composite `(A,X)-B` (`3A-Comp`)], [Full `08` \ Blind `12`], [25/0/26 \ 17/0/18], [Yes], [$infinity$ (0 aux) \ $infinity$ (0 aux)], [$infinity$ (0 aux) \ $infinity$ (0 aux)], [Bipartite `(A,X)` merges cubic terms],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Four-stage spin compression attribution separating axis selection ($17.0 times$), active-subspace pruning, and zero-auxiliary composite indexing.],
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
      [*ANOVA Instance*], [*Subspace*], [*Stage 1: Float ANOVA*], [*Stage 2: 8-Bit Int*], [*Levels*], [*GS / Regret*], [*Stage 3: 50-Run SA / Tabu*],
      table.hline(stroke: 0.6pt),
      [`Q3_full_08_Model3B_ANOVA`], [588 legal], [$R^2=0.4559$ \ $rho=0.3518$], [$rho(hat(V), "int")=0.2070$], [6], [3 / 0.005211], [SA 100% (0.000) / Tabu 100% (0.0052)],
      [`Q3_blind_12_Model3B_ANOVA`], [108 legal], [$R^2=0.4902$ \ $rho=0.4783$], [$rho(hat(V), "int")=0.4512$], [6], [7 / 0.000000], [SA 100% (0.000) / Tabu 100% (0.0052)],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Three-stage approximation and quantization breakdown of `Model 3B-ANOVA` (`3, 1, 24, 59, 459, 42` legal states per level on `08`; `7, 9, 28, 50, 12, 2` on `12`).],
) <tab:anova>

== Classical Solver Benchmarks and Multi-Anion Candidate Shortlist
Under an equalized $50$-run classical budget (simulated annealing [SA] and tabu search [Tabu] in Kaiwu SDK [2], retaining the lowest-energy state per run; @fig:q3 and @tab:solvers), unpruned 8-bit baselines fail ($0.0%$ feasibility on `Q2_full_07`, `Q2_blind_11`, and `Q3_blind_12`, or $12.2121$ regret on `Q3_full_08`), whereas all active-subspace zero-auxiliary encodings achieve $100.0%$ feasibility (`Model 3A-HOBO` attains $86.0%$--$96.0%$ strict auxiliary feasibility and $100.0%$ primary feasibility) and $0.000000$ best-of-50 continuous regret under SA.

#figure(
  image("../v2_optimized/figures/fig05_q3_qubit_compression_and_solvers.png", width: 96%),
  caption: [Problem 3 summary: (a) spin compression from $2,760$ to $18$ spins; (b) equal-budget ($50$-run) SA and Tabu feasibility and Top-15 capture; (c) `Model 3B-ANOVA` rank correlation across stages; (d) five-term physical violation breakdown of the Q3 Blind (`12`) Top-15 shortlist.],
) <fig:q3>

#figure(
  {
    set text(size: 7.5pt)
    table(
      columns: (4.25cm, 0.80cm, 2.80cm, 0.90cm, 1.45cm, 1.30cm, 1.20cm, 1.10cm, 2.85cm),
      stroke: none,
      align: (left, center, right, right, right, right, right, right, right),
      inset: (x: 2.0pt, y: 2.2pt),
      table.hline(stroke: 1.1pt),
      [*Pool & Encoding*], [*Spins*], [*SA Feas.*], [*SA Uniq.*], [*SA Regret*], [*SA Top-K*], [*Tabu Feas.*], [*Tabu Uniq.*], [*Tabu Regret / Top-K*],
      table.hline(stroke: 0.6pt),
      [`Q2_full_07` Full-Grid Float], [105], [100.0%], [39], [0.000000], [8/10], [100.0%], [8], [0.000000 / 7/10],
      [`Q2_full_07` Full-Grid 8-Bit], [105], [*0.0%*], [0], [—], [0/10], [*0.0%*], [0], [— / 0/10],
      [`Q2_full_07` *Active Exact 8-Bit*], [*17*], [*100.0%*], [10], [*0.000000*], [*10/10*], [*100.0%*], [7], [0.000000 / 7/10],
      [`Q2_blind_11` Full-Grid 8-Bit], [105], [*0.0%*], [0], [—], [0/10], [*0.0%*], [0], [— / 0/10],
      [`Q2_blind_11` *Active Exact 8-Bit*], [*12*], [*100.0%*], [4], [*0.000000*], [*4/10*], [*100.0%*], [3], [0.000000 / 3/10],
      table.hline(stroke: 0.5pt),
      [`Q3_full_08` Full-FM Float], [108], [100.0%], [45], [0.005211], [7/15], [100.0%], [19], [0.000000 / 10/15],
      [`Q3_full_08` Full-FM 8-Bit], [108], [100.0%], [2], [12.212100], [0/15], [80.0%], [5], [12.212100 / 0/15],
      [`Q3_full_08` *`Model 3A-Comp`*], [*26*], [*100.0%*], [10], [*0.000000*], [10/15], [*100.0%*], [9], [0.000000 / 9/15],
      [`Q3_full_08` *`Model 3A-HOBO`*], [43], [86.0% (100% prim)], [25], [*0.000000*], [*13/15*], [*100.0%*], [12], [0.005211 / 10/15],
      [`Q3_full_08` `Model 3B-ANOVA`], [33], [*100.0%*], [17], [*0.000000*], [7/15], [*100.0%*], [3], [0.005211 / 3/15],
      table.hline(stroke: 0.5pt),
      [`Q3_blind_12` Full-FM 8-Bit], [108], [*0.0%*], [0], [—], [0/15], [*0.0%*], [0], [— / 0/15],
      [`Q3_blind_12` *`Model 3A-Comp`*], [*18*], [*100.0%*], [7], [*0.000000*], [7/15], [*100.0%*], [2], [0.005211 / 2/15],
      [`Q3_blind_12` *`Model 3A-HOBO`*], [29], [96.0% (100% prim)], [15], [*0.000000*], [*14/15*], [*100.0%*], [7], [0.000000 / 7/15],
      [`Q3_blind_12` `Model 3B-ANOVA`], [25], [*100.0%*], [7], [*0.000000*], [7/15], [*100.0%*], [2], [0.005211 / 2/15],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Equal-budget $50$-run classical solver comparison across Problem 2 and Problem 3 encodings (runtimes: SA $2.59$--$3.47 "s"$, Tabu $5.08$--$9.60 "s"$).],
) <tab:solvers>

@tab:q3top15 decomposes all five physical violation components across the top $15$ multi-anion candidates in Blind Pool `12` (Full Pool `08` shares the identical global optimum $"LaTiO"_2"N"$ at $hat(V) = 0.2524$, while additionally admitting labeled-pair candidates $"ScTiO"_2"N"$ [$0.4489$], $"SnTiO"_3$ [$0.4618$], and $"ScZrO"_2"N"$ [$0.5733$] into its top $10$). Oxynitrides ($"NO"_2$) claim the top $7$ blind ranks ($hat(V) = 0.2524$--$0.4144$) because hybridized $"N" 2p$ states elevate the VBM relative to pure oxides, centering $hat(E)_g$ inside $2.3$--$2.7 "eV"$ without compromising formation stability. Six pure oxides ($"CaGeO"_3$, $"SrGeO"_3$, $"KNbO"_3$, $"RbNbO"_3$, $"CaSnO"_3$, $"SrSnO"_3$; ranks $8$--$11$, $13$--$14$) follow at $hat(V) = 0.6458$--$1.0055$: all six exhibit strong thermodynamic stability ($hat(v)_("form") <= 0.1106$), while incurring moderate band-gap edge ($hat(v)_("gl") + hat(v)_("gh") approx 0.43$--$0.83$) and CBM ($hat(v)_("CBM") approx 0.08$--$0.13$) costs.

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
      stroke: none,
      align: (center, left, center, right, right, right, right, right, right, right, right, right),
      inset: (x: 2.4pt, y: 3.4pt),
      table.hline(stroke: 1.1pt),
      [*Rank*], [*Formula*], [*$X$*], [*$hat(V)$*], [*$hat(P)_("hit")$*], [*$hat(E)_g$*], [*$hat(v)_("form")$*], [*$hat(v)_("gl")$*], [*$hat(v)_("gh")$*], [*$hat(v)_("VBM")$*], [*$hat(v)_("CBM")$*], [*MC ($0.5 sigma$)*],
      table.hline(stroke: 0.6pt),
      [1], [*$"LaTiO"_2"N"$*], [$"NO"_2$], [*0.2524*], [61.4%], [2.657], [0.0107], [0.0500], [0.1679], [0.0038], [0.0201], [99.8%],
      [2], [*$"SrTaO"_2"N"$*], [$"NO"_2$], [*0.2576*], [*76.6%*], [2.522], [0.0020], [0.1119], [0.1020], [0.0236], [0.0182], [99.8%],
      [3], [*$"BaTaO"_2"N"$*], [$"NO"_2$], [*0.2612*], [71.0%], [2.357], [0.0012], [0.1314], [0.0530], [0.0068], [0.0688], [100.0%],
      [4], [*$"CaTaO"_2"N"$*], [$"NO"_2$], [*0.2660*], [70.2%], [2.487], [0.0085], [0.1222], [0.0893], [0.0179], [0.0282], [99.8%],
      [5], [$"YTiO"_2"N"$], [$"NO"_2$], [0.3363], [21.9%], [2.577], [0.1502], [0.0360], [0.1264], [0.0018], [0.0219], [99.8%],
      [6], [$"CaNbO"_2"N"$], [$"NO"_2$], [0.3809], [63.1%], [1.846], [0.0108], [0.2766], [0.0048], [0.0356], [0.0531], [99.0%],
      [7], [$"SrNbO"_2"N"$], [$"NO"_2$], [0.4144], [69.7%], [1.805], [0.0022], [0.3068], [0.0039], [0.0673], [0.0342], [98.6%],
      [8], [*$"CaGeO"_3$*], [$"O"_3$], [*0.6458*], [33.7%], [2.579], [0.0814], [0.3052], [0.1230], [0.0238], [0.1123], [99.4%],
      [9], [*$"SrGeO"_3$*], [$"O"_3$], [*0.7042*], [54.4%], [2.289], [0.0302], [0.4547], [0.0380], [0.0546], [0.1267], [100.0%],
      [10], [*$"KNbO"_3$*], [$"O"_3$], [*0.7045*], [63.9%], [2.805], [0.0003], [0.2932], [0.2884], [0.0291], [0.0936], [97.2%],
      [11], [$"RbNbO"_3$], [$"O"_3$], [0.8378], [*67.6%*], [2.702], [0.0004], [0.4520], [0.2022], [0.0522], [0.1310], [97.2%],
      [12], [$"ScHfO"_2"N"$], [$"NO"_2$], [0.9526], [1.5%], [2.407], [0.7113], [0.1057], [0.0650], [0.0043], [0.0663], [93.6%],
      [13], [$"CaSnO"_3$], [$"O"_3$], [0.9720], [21.1%], [2.979], [0.1106], [0.2783], [0.4694], [0.0309], [0.0829], [86.6%],
      [14], [$"SrSnO"_3$], [$"O"_3$], [1.0055], [36.0%], [2.979], [0.0357], [0.3643], [0.4643], [0.0595], [0.0819], [86.8%],
      [15], [$"MgTaO"_2"N"$], [$"NO"_2$], [1.0059], [6.8%], [1.994], [0.0857], [0.2294], [0.0102], [0.0000], [0.6807], [86.0%],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Five-term physical cost decomposition of the Q3 Blind Pool (`12`, $1,587$ states) Top-15 multi-anion candidates with $0.5 sigma$ Monte Carlo Top-15 inclusion stability.],
) <tab:q3top15>

= Quantum Hardware Execution and Closed-Loop Discovery

== Compact Single-Reference Compilation and Hardware Decoding
Physical CIM execution highlights the contrast between full-grid reference chains and active-subspace compilation. In earlier full-grid reference-chain encodings ($105$--$468$ spins), preserving all $52$ elements within $[-127, 127]$ required splitting $s_("ref")$ into multiple ferromagnetically coupled copies, yielding only $3$ feasible solutions out of $53$ returned records ($5.66%$) across six hardware tasks. Active-subspace pruning instead compiles all four blind-pool Hamiltonians into $12$--$29$ spins with a single reference spin ($K = 1$), avoiding reference-chain frustration altogether.

For each SPQC-550 task, we apply the inverse bijection of @eq:ising-recovery directly to the saved 8-bit matrix $H$ [14], decode each returned spin vector $s in {-1, +1}^(n+1)$ via $x_i = (1 + s_i s_("ref")) slash 2$ ($i = 1, dots, n$), sort records by exact Ising energy $E(s) = -s^top H s$, and check (i) primary one-hot cardinality ($k_A = k_B = k_X = 1$), (ii) blind-pool membership and $A != B$, and (iii) auxiliary consistency ($y_(i k) = a_i c_k$). Note that SPQC-550 runs in optimization mode (returning $10$ lowest-energy optical states per task rather than independent Boltzmann samples), and wall-clock times ($16.23$--$19.06 "s"$, subject to a $5 "s"$ minimum SDK polling interval) measure full cloud round-trip latency rather than optical pulse evolution time.

== Experimental Verification on the Bose Quantum SPQC-550 CIM
@tab:spqc and @fig:spqc compare the four blind-pool SPQC-550 hardware runs against same-instance classical SA ($50$ runs) and Tabu ($30$ runs), with all $40$ individual returned records detailed in Appendix B:
- *Q2-Blind (`12` spins, $E^* = -1000$):* SPQC-550 attains the certified ground energy $-1000.0$ and returns $30.0%$ ($3 slash 10$) strictly feasible returned records with zero duplicates, recovering $"SrTaO"_2"N"$ ($hat(V) = 0.2576$, regret $0.0052$, rank $2$), $"CaNbO"_2"N"$ ($0.3809$, rank $6$), and $"YNbO"_2"N"$ ($12.7700$, rank $64$).
- *Q3-Comp (`18` spins, $0$ aux, $E^* = -1468$):* Removing auxiliary spins markedly improves optical convergence: $90.0%$ ($9 slash 10$) of returned records are strictly feasible, all $9$ are distinct Top-15 candidates spanning $"NO"_2$ and $"O"_3$ ($"LaTiO"_2"N"$, $"SrTaO"_2"N"$, $"BaTaO"_2"N"$, $"CaTaO"_2"N"$, $"SrNbO"_2"N"$, $"CaGeO"_3$, $"RbNbO"_3$, $"ScHfO"_2"N"$, $"CaSnO"_3$), and records `0`--`4` reach the exact certified ground energy $-1468.0$—recovering $"LaTiO"_2"N"$ at $0.000000$ regret.
- *Q3-HOBO (`29` spins, $11$ aux, $E^* = -1662$):* Optical annealing reaches the certified ground state $-1662.0$ on records `0` and `1` ($"CaTaO"_2"N"$, $"SrTaO"_2"N"$) and satisfies primary $(A, B, X)$ one-hot constraints on $90.0%$ ($9 slash 10$) of records, but $7$ low-lying excited states exhibit $1$--$3$ auxiliary flips ($y_(i k) != a_i c_k$), yielding $20.0%$ strict feasibility. The contrast between this $20.0%$ strict rate and the $90.0%$ rate of `Q3-Comp` demonstrates the practical benefit of eliminating auxiliary qubits on physical CIM hardware.
- *Q3-ANOVA (`25` spins, $0$ aux, $E^* = -1992$):* Uniform $8 times$ scaling reaches the certified ground energy $-1992.0$ across three distinct oxynitrides ($"SrTaO"_2"N"$, $"CaTaO"_2"N"$, $"BaTaO"_2"N"$; $30.0%$ strict feasibility, regret $0.0052$).

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (2.25cm, 1.25cm, 1.45cm, 2.05cm, 1.00cm, 1.15cm, 1.35cm, 1.35cm, 3.45cm, 1.70cm),
      stroke: none,
      align: (left, center, right, left, right, center, right, center, left, right),
      inset: (x: 2.2pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*Instance*], [*Spins (Aux)*], [*`cert_min_E`*], [*Solver (Rec./Runs)*], [*Prim.*], [*`y=a*c`*], [*Strict*], [*Uniq / Dup*], [*Best Candidate ($hat(V)$ / Regret)*], [*Wall (s)*],
      table.hline(stroke: 0.6pt),
      [Q2-Blind], [12 (0)], [-1000], [SPQC-550 (10)], [30.0%], [10/10], [*30.0%*], [3 / 0], [$"SrTaO"_2"N"$ (0.2576 / 0.0052)], [17.24],
      [Q2-Blind], [12 (0)], [-1000], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [4/46 ; 4/26], [$"LaTiO"_2"N"$ (0.2524 / 0.0000)], [3.33 / 3.51],
      [Q3-Comp], [18 (0)], [-1468], [SPQC-550 (10)], [90.0%], [10/10], [*90.0%*], [9 / 0], [*$"LaTiO"_2"N"$ (0.2524 / 0.0000)*], [16.23],
      [Q3-Comp], [18 (0)], [-1468], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [7/43 ; 2/28], [$"LaTiO"_2"N"$ (0.0000) / $"SrTaO"_2"N"$], [3.25 / 3.70],
      [Q3-HOBO], [29 (11)], [-1662], [SPQC-550 (10)], [90.0%], [3/10], [*20.0%*], [2 / 0], [$"SrTaO"_2"N"$ (0.2576 / 0.0052)], [19.06],
      [Q3-HOBO], [29 (11)], [-1662], [SA (50) \ Tabu (30)], [100%], [48 / 30], [96% / 100%], [13/35 ; 7/23], [$"LaTiO"_2"N"$ (0.2524 / 0.0000)], [3.31 / 3.16],
      [Q3-ANOVA], [25 (0)], [-1992], [SPQC-550 (10)], [30.0%], [10/10], [*30.0%*], [3 / 0], [$"SrTaO"_2"N"$ (0.2576 / 0.0052)], [16.43],
      [Q3-ANOVA], [25 (0)], [-1992], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [8/42 ; 2/28], [$"LaTiO"_2"N"$ (0.0000) / $"SrTaO"_2"N"$], [3.29 / 3.57],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [SPQC-550 optical CIM results vs. same-instance classical solvers on the four blind-pool matrices (SPQC-550 wall times include cloud queueing and $5 "s"$ SDK polling latency; raw task IDs in Appendix B).],
) <tab:spqc>

#figure(
  image("../v2_optimized/figures/fig07_spqc550_real_hardware_comparison.png", width: 96%),
  caption: [Hardware verification on SPQC-550: (a) earlier full-grid reference-chain encodings ($5.66%$ overall feasibility) vs. active-subspace blind-pool runs ($20.0%$--$90.0%$ strict feasibility); (b) per-record Ising energy gap $E(s) - E^*$ across all $40$ returned hardware states; (c) feasible record rate vs. total Ising spin count.],
) <fig:spqc>

== Closed-Loop Active Discovery under a Limited DFT Budget
To address the engineering question posed in the problem statement—how efficiently rare targets can be recovered under a strict DFT budget—we run $20$ paired active-learning campaigns on the validation pool ($N = 3,969$, containing $M = 18$ compounds with $V < 0.5$) [10]. Each campaign starts from $50$ random DFT evaluations and acquires $10$ candidates per round over $15$ rounds (total budget $K = 200$, or $5.04%$ of the pool):
- *Random Sampling Baseline:* Uniform sampling without replacement gives an analytical hypergeometric expectation of $bb(E)[H_(0.5)(200)] = 200 times 18 slash 3969 = 0.9070$ hits ($"SD" = 0.9261$; empirical 20-seed mean $0.60 slash 18$).
- *Factorization Machine Baselines:* A second-order one-hot FM trained to convergence via L-BFGS-B on scalar $V$ labels captures $4.30 plus.minus 2.03$ targets under greedy ranking ($300 slash 300$ converged fits), whereas solving its unpruned $112$-spin QUBO via SA drops to $1.20 plus.minus 0.98$ targets.
- *Physics-Hurdle Sequential vs. Hybrid 8-Bit QUBO Acquisition:* With an online variant of our physics-coupled hurdle surrogate (fold-refitted Butler--Ginley $E_c$ plus LightGBM hurdle channels), classical greedy ranking captures $16.55 slash 18$ rare targets ($"SD" = 1.962$, an $18.2 times$ gain over random sampling), while a hybrid strategy combining classical Top-20 pre-screening with per-round 8-bit `Model 3A-Comp` QUBO optimization ($300 slash 300$ exact 8-bit matrices averaging $28.77$ spins) captures $16.20 slash 18$ targets ($"SD" = 1.077$; @fig:closedloop). Thus, while the discovery speedup stems primarily from the physics-coupled hurdle representation rather than QUBO acquisition over greedy sorting, the hybrid 8-bit QUBO loop executes reliably across all $300$ rounds with tighter cross-seed dispersion.

#figure(
  image("../v2_optimized/figures/fig03_closed_loop_fmqa_benchmark.png", width: 95%),
  caption: [20-seed closed-loop active discovery on split `03` ($N = 3,969$, budget $K in [50, 200]$): analytical hypergeometric baseline ($0.9070$), converged L-BFGS-B FM baselines, scalar-$V$ LightGBM, sequential hurdle ranking ($16.55 slash 18$), and hybrid Top-20 + 8-bit QUBO acquisition ($16.20 slash 18$, $"SD" = 1.077$).],
) <fig:closedloop>

= Sensitivity Analysis and Model Evaluation

== Four-Axis Sensitivity and Perturbation Stress Tests
@fig:sens and @tab:sens evaluate stability across four modeling and compilation parameters:
1. *Quantization Bit-Width ($B in [4, 12]$ on `Q2_blind_11`):* Exact integer coupling without precision fallback holds from $B = 7$ ($M_(max) = 2$, $max |H_(u v)| = 59$, $0$ infeasible GS) to $B = 12$ ($M_(max) = 71$, $max |H_(u v)| = 2038$, $9 slash 10$ SA Top-10 capture). Below $7$ bits, fallback rescaling distorts objective resolution ($0.7002$ top-1 regret at $6$ bits, $12.71$ worst regret at $5$ bits, and $5$ infeasible ground states at $4$ bits).
2. *Penalty Multiplier Phase Transition ($alpha_P in [0.50, 3.00]$ at fixed $M_(max) = 4$):* Variation of $alpha_P$ on `Q2_blind_11` verifies Theorem 2's analytical threshold (@fig:sens(a)): for $alpha_P in {0.50, 1.00}$, infeasible states dominate ($0$ legal ground states); at $alpha_P = 1.25$ ($P_("block") = 40$), $4$ legal states tie $1$ infeasible state (energy margin $0.0$, SA $84.0%$); for all $alpha_P >= 1.40$ (including exact 8-bit points $1.50, 1.75, 2.25$), all $4$ ground states are $100%$ legal (energy margin $+9.6$ to $+64.0$) and SA reaches $100.0%$ feasibility.
3. *Upper-Tail Cutoff $C_("cap")$ Retention Across Subspaces:* At our default cutoffs, Top-10 retention is $10 slash 10$ on `Q2_full_07`, `Q3_full_08`, and `Q3_blind_12`, and $9 slash 10$ on `Q2_blind_11` (omitting only Rank 10 $"CdTaO"_2"N"$ at $hat(V) = 3.6227$). At Top-25 on `Q3_blind_12` ($C_("cap") = 1.50$, $15$ active rows below cutoff), the Cartesian $A times B times X$ domain of `Model 3A-HOBO` retains $17 slash 25$ candidates, `Model 3A-Comp` retains $15 slash 25$ (excluding $"BaGeO"_3$ and $"BaSnO"_3$ above cutoff), and the Top-25 subspace of `Model 3B-ANOVA` retains $24 slash 25$ legal candidates (excluding homonuclear $"SnSnO"_3$); raising $C_("cap")$ to $2.50$ ($21$ active rows) expands retention to $23 slash 25$ (`HOBO`) and $21 slash 25$ (`Comp`) at higher spin counts (@fig:sens(b)).
4. *Surrogate Uncertainty Perturbations ($0.5 sigma$--$2.0 sigma$, $500$ Monte Carlo trials):* Perturbing $(hat(mu)_("ef"), hat(mu)_g, hat(p)_("semi"))$ (@fig:sens(c)--(d)) shows high shortlist stability at $0.5 sigma$: independent Gaussian noise retains $95.70%$ (Q2 Top-10) and $96.24%$ (Q3 Top-15), decaying to $83.28%$ and $61.01%$ at $2.0 sigma$; $(A,B)$-correlated plus logit noise similarly shifts retention from $95.46%$ to $83.10%$ (Q2) and from $94.96%$ to $58.84%$ (Q3).

#figure(
  image("../v2_optimized/figures/fig06_sensitivity_and_mc_robustness.png", width: 92%),
  caption: [Four-axis sensitivity suite: (a) penalty multiplier phase transition at fixed $M_(max) = 4$; (b) Pareto trade-off between $C_("cap")$ Top-10/Top-25 retention and active spin count; (c)--(d) shortlist retention under $0.5 sigma$--$2.0 sigma$ independent and $(A,B)$-correlated surrogate perturbations.],
) <fig:sens>

#figure(
  table(
    columns: (auto, 1.25fr, 1.45fr),
    stroke: none,
    align: (left, left, left),
    inset: 2.8pt,
    table.hline(stroke: 1.1pt),
    [*Sensitivity Dimension*], [*Parameter Sweep*], [*Key Quantitative Finding*],
    table.hline(stroke: 0.6pt),
    [Quantization Bit-Width (`Q2_blind_11`)], [$B in {4, 5, 6, 7, 8, 10, 12}$], [$B <= 6$: fallback distortion ($5$ illegal GS at $B=4$); $B in [7, 12]$ ($M_(max) in [2, 71]$): $0$ illegal GS, $100%$ SA feasibility, $4$--$9 slash 10$ Top-10 hits],
    [Penalty Ratio $alpha_P$ ($M_(max)=4$)], [$alpha_P in [0.50, 3.00]$ ($10$ points)], [$alpha_P <= 1.0$: $0$ legal GS; $alpha_P = 1.25$: boundary tie (margin $0$); $alpha_P >= 1.40$: $100%$ legal GS & $100%$ SA feasibility],
    [Truncation Threshold $C_("cap")$], [`Q2` ($2.0, 2.5$), `Q3` ($1.25, 1.50, 2.50$)], [Top-10: $10 slash 10$ (`Q2_full`, `Q3_full`, `Q3_blind`), $9 slash 10$ (`Q2_blind`); `Q3_blind` Top-25 at $1.50$: `15/25` (`Comp`), `17/25` (`HOBO`), `24/25` (`ANOVA`); at $2.50$: `23/25` (`HOBO`), `21/25` (`Comp`)],
    [Surrogate Noise (`500` MC trials)], [$0.5 sigma$--$2.0 sigma$ Indep. & Group-Corr.], [Indep. $0.5 sigma -> 2.0 sigma$: Q2 Top-10 $95.70% -> 83.28%$, Q3 Top-15 $96.24% -> 61.01%$; Group-Corr.: Q2 $95.46% -> 83.10%$, Q3 $94.96% -> 58.84%$],
    table.hline(stroke: 1.1pt),
  ),
  caption: [Summary of multi-axis sensitivity and perturbation stability experiments.],
) <tab:sens>

== Model Strengths, Limitations, and Practical Scope
*Core Strengths:*
1. *Physics-Consistent Surrogate Decoupling:* Our hurdle-Bayes decomposition separates metallic screening from conditional band-gap regression and embeds the in-data Butler--Ginley band-center law ($sigma_("VBM") = sigma_("CBM") = sigma_(E_g) slash 2$), halving near-target prediction error under strict $(A,B)$-pair out-of-distribution validation.
2. *Mathematically Certified 8-Bit Hardware Compilation:* Theorems 1--4 establish closed-form penalty bounds, zero-drift integer coupling alignment, and exact primary-space ground-state certificates across all eight QUBO and HOBO instances without heuristic dynamic-range tuning.
3. *Zero-Auxiliary Three-Body Scalability on Quantum Hardware:* Composite $(A,X)$--$B$ indexing removes auxiliary consistency constraints altogether, raising strict hardware feasibility on SPQC-550 from $20.0%$ (`Q3-HOBO`) to $90.0%$ (`Q3-Comp`) and recovering $9$ distinct top-$15$ perovskites in a single hardware run.

*Limitations and Scope Boundaries:*
1. *Withheld Ground Truth on `04_test_holdout` and Oxyfluoride Sparsity:* Because DFT labels on split `04` are unseen, neither $sum hat(P)_("hit") = 10.3358$ nor $C_("cap")$ retention measures empirical recall against the $19$ hidden $V = 0$ targets. Scarce low-$V$ $"FO"_2$ examples in `02` + `03` also yield conservative oxyfluoride probability mass ($0.77$ vs. $6$ hidden targets).
2. *Gaussian Marginal Approximation and Conformal Scope:* @eq:vbayes and @eq:phit neglect the modest negative residual correlation between $e_("form")$ and $E_g$ on semiconductors ($r = -0.2930$) and use untruncated normal marginals ($P(E_g < 0)$ median $0.1743$ across `04`, but $<= 0.0110$ among the top $50$ candidates); likewise, `03_val` conformal radii do not transfer as formal finite-sample guarantees after refitting on `02` + `03`.
3. *Surrogate-Relative Pruning and Cloud Execution Semantics:* Active-subspace pruning below $C_("cap")$ is lossless with respect to $hat(V)$ rather than true DFT $V$, and SPQC-550 optimization runs return $10$ lowest-energy optical states per cloud round-trip ($16.23$--$19.06 "s"$ latency) rather than independent thermodynamic samples.

= Conclusion and References

== Conclusion
We have developed an end-to-end physics-coupled machine learning and 8-bit quantum optimization framework for rare-target $A B X_3$ photocatalyst discovery:
1. *Out-of-Pair Surrogate Modeling (Problem 1):* Under $(A,B)$-grouped validation, coupling a calibrated metal--semiconductor hurdle classifier with the log-additive Butler--Ginley band-center law lowers validation RMSE to $4.0634$, reduces $"MAE"(V < 1)$ to $2.9280$, and recovers $100.0%$ of $V < 0.5$ targets within the top $75$ candidates ($94.4%$ at top $50$).
2. *Certified 8-Bit QUBO Compilation (Problems 2 & 3):* Upper-tail truncation, tight bipartite and submodular penalty bounds, and active-subspace pruning compress all eight Hamiltonians to $12$--$43$ spins with zero infeasible global ground states, identifying $"LaTiO"_2"N"$ ($hat(V) = 0.252406$) alongside top tantalate, niobate, germanate, and stannate candidates.
3. *Optical CIM Execution & Active Screening:* On the Bose Quantum SPQC-550 CIM, our four blind-pool Hamiltonians yield $20.0%$--$90.0%$ strictly feasible returned records—highlighted by the $18$-spin composite encoding recovering $"LaTiO"_2"N"$ at zero regret and $9 slash 15$ top candidates—while $20$-seed closed-loop discovery captures $16.20 slash 18$ rare targets within $200$ queries.

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

[15] OpenAI, ChatGPT (Web Version), OpenAI, 2026-11-26--2026-11-27.

// =============================================================================
// APPENDICES (UNLIMITED PAGES AFTER MAIN BODY)
// =============================================================================
#pagebreak()
#set par(hanging-indent: 0pt)
#set heading(numbering: none)

= Appendices

== Appendix A: Computational Environment, Reproducibility, and AI Disclosure
All numerical experiments were executed reproducibly (fixed seeds `2026` and `101`--`120`) under Python 3.12 using `numpy==2.5.5`, `scipy==1.18.1`, `pandas==2.2.3`, `scikit-learn==1.9.1`, `catboost==1.2.10`, `lightgbm==4.7.0`, `xgboost-cpu==3.4.1`, `matplotlib==3.10.3`, `typst==0.15.2`, `pypdf==7.2.0`, `pymupdf==2.2`, `python-docx==1.2.0`, and `kaiwu==1.4.4`. Dataset integrity across all nine official files in `sha256_public.json` was verified prior to modeling (`9/9` CRLF-normalized SHA-256 matches; `1/9` raw-byte match on the README alongside LF line-ending checkout on the eight CSVs). Key offline reproduction entry points included in `Dapmcm26302266fj.zip` (with `APMCM_DATA_DIR` pointing to the official dataset folder, requiring zero cloud API calls) are:
- `python v2_optimized/run_all.py --verify-saved` — verifies dataset checksums, exported Q1 CSVs (`v2_PhysHurdle_Bayes`), ground-state certificates, and saved SPQC-550 raw spin checkpoints (`v2_spqc550_hardware.py --verify-only`).
- `python scripts/verify_milp_qubo_crosscheck.py` — solves independent Fortet--McCormick MILP linearizations via SciPy/HiGHS (`mip_gap = 0`) across all eight integer Ising matrices.
- `python v2_optimized/run_all.py --all` — executes full offline surrogate training, QUBO/HOBO certification, 20-seed closed-loop & sensitivity benchmarks, and figure generation.

*AI Tool Usage Disclosure:* In accordance with the 2026-11-04 contest AI policy, all mathematical models, physical derivations, and SPQC-550 hardware experiments were conceived and executed by the team. AI assistants [13--15] were used strictly as auxiliary tools for Python pipeline refactoring, visualization/Typst formatting, English language polishing, and manuscript/mathematical review (all adopted adjustments and MILP cross-checks were independently implemented and verified by the team, as detailed in our standalone AI usage disclosure document).

== Appendix B: Complete 40-Record SPQC-550 Hardware Log and Digest Registry
@tab:records documents all $40$ optimization records returned across the four blind-pool SPQC-550 tasks (`v3_q2_blind_12`, `v3_q3_comp_18`, `v3_q3_hobo_29`, `v3_q3_anova_25`; `18` of `31` initial account credits preserved), decoded directly from the archived raw spin arrays. @tab:hashes lists the SHA-256 digests and Ising offsets $C$ for all certified matrices and exported prediction files.

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
      [`v3_q2_blind_12`], [0], [-1000.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"SrTaO"_2"N"$*], [0.257617], [2],
      [`v3_q2_blind_12`], [1], [-992.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"CaNbO"_2"N"$*], [0.380890], [6],
      [`v3_q2_blind_12`], [2], [-968.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"YNbO"_2"N"$*], [12.769992], [64],
      [`v3_q2_blind_12`], [3--9], [-920.0.. -896.0], [(2,1,1)/(1,1,1)\* / 0-hot], [OK], [INFEASIBLE (pool/one-hot fail, 7 records)], [—], [—],
      table.hline(stroke: 0.5pt),
      [`v3_q3_comp_18`], [0], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"SrTaO"_2"N"$*], [0.257617], [2],
      [`v3_q3_comp_18`], [1], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"SrNbO"_2"N"$*], [0.414431], [7],
      [`v3_q3_comp_18`], [2], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"LaTiO"_2"N"$*], [*0.252406*], [*1*],
      [`v3_q3_comp_18`], [3], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"CaTaO"_2"N"$*], [0.265979], [4],
      [`v3_q3_comp_18`], [4], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"BaTaO"_2"N"$*], [0.261166], [3],
      [`v3_q3_comp_18`], [5], [-1460.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"RbNbO"_3$*], [0.837836], [11],
      [`v3_q3_comp_18`], [6], [-1460.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"CaGeO"_3$*], [0.645777], [8],
      [`v3_q3_comp_18`], [7], [-1452.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"ScHfO"_2"N"$*], [0.952559], [12],
      [`v3_q3_comp_18`], [8], [-1452.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"CaSnO"_3$*], [0.972014], [13],
      [`v3_q3_comp_18`], [9], [-1452.0], [(2, 1, 2)], [OK], [INFEASIBLE (one-hot/pool fail)], [—], [—],
      table.hline(stroke: 0.5pt),
      [`v3_q3_hobo_29`], [0], [-1662.0], [(1, 1, 1)], [OK (0)], [*FEASIBLE: $"CaTaO"_2"N"$*], [0.265979], [4],
      [`v3_q3_hobo_29`], [1], [-1662.0], [(1, 1, 1)], [OK (0)], [*FEASIBLE: $"SrTaO"_2"N"$*], [0.257617], [2],
      [`v3_q3_hobo_29`], [2, 3], [-1654.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [4], [-1646.0], [(1, 1, 1)], [Fail (2)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [5], [-1638.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(SrTaO2N)`], [0.257617], [2],
      [`v3_q3_hobo_29`], [6], [-1634.0], [(1, 2, 1)], [OK (0)], [INFEASIBLE (one-hot fail)], [—], [—],
      [`v3_q3_hobo_29`], [7, 8], [-1630.0], [(1, 1, 1)], [Fail (2, 3)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [9], [-1630.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(CaNbO2N)`], [0.380890], [6],
      table.hline(stroke: 0.5pt),
      [`v3_q3_anova_25`], [0], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"SrTaO"_2"N"$*], [0.257617], [2],
      [`v3_q3_anova_25`], [1], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"CaTaO"_2"N"$*], [0.265979], [4],
      [`v3_q3_anova_25`], [2], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: $"BaTaO"_2"N"$*], [0.261166], [3],
      [`v3_q3_anova_25`], [3--9], [-1976.0.. -1944.0], [(2,1,1)/(1,2,1)/(1,1,0)/0-hot], [OK], [INFEASIBLE (7 records)], [—], [—],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [Complete 40-record SPQC-550 hardware log across all four blind-pool tasks (`*` records 4 and 9 in `v3_q2_blind_12` satisfy one-hot constraints on elements outside the blind pair set).],
) <tab:records>

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: none,
    align: (left, right, left),
    inset: 2.3pt,
    table.hline(stroke: 1.1pt),
    [*Artifact / Matrix ID*], [*Offset $C$*], [*SHA-256 Digest*],
    table.hline(stroke: 0.6pt),
    [`Q2_full_07`], [1390], [`a61bc1d9bd777acdc31c02490c510554dbbcf488d683f50572ba61760fc1aa7e`],
    [`Q2_blind_11` (`v3_q2_blind_12`)], [824], [`361cc4d8ef7d074479506820b0c3e3d90b3170e97ca7c721caea7aa013c02a6b`],
    [`Q3_full_08_Model3A_Comp`], [2038], [`f4bd07e3c83a92023de687e70b95b1c7d81d32084248c12f00ddaf274b9ecdee`],
    [`Q3_full_08_Model3A_HOBO`], [2010], [`fe6d37c13ebf55939abec80dc01f7ba9aaa90be0bc3a37ea7ba729123c56e2a1`],
    [`Q3_full_08_Model3B_CompactANOVA`], [2500], [`7416404244bac9c826c72594f2c937843c1af4d810b9febdb1d94fddf24f6ae2`],
    [`Q3_blind_12_Model3A_Comp` (`v3_q3_comp_18`)], [1364], [`47b9b1f72e78fa9d4dc94ae9a83b6a0067c1f1914b8e9219df6f9b5e233939ae`],
    [`Q3_blind_12_Model3A_HOBO` (`v3_q3_hobo_29`)], [1482], [`302bf51ecc1b9281fea99a77ad038489818d626c5f288e81db1eab9211f55a2a`],
    [`Q3_blind_12_Model3B_CompactANOVA` (`v3_q3_anova_25`)], [1872], [`05ce8d9cfad20be928a3ec0cfd51742f094a311ed64612039681749d9e341f36`],
    [`v2_val_predictions.csv`], [—], [`ac5d939d4dd63f3aa95d4cde112247b376ea9f3818d766644ba5d9a9b6f3c3fd`],
    [`v2_holdout_04_train02only.csv`], [—], [`2090c512ddc3e61459a81519b956e3e5d0bd1a9145b36e490cabf0cfd5d896c3`],
    [`v2_holdout_04.csv`], [—], [`89eca9b89f8501233356701b2426e3296e41c72f66bf64b518520d8a344742a2`],
    [`v2_q2_full_07.csv`], [—], [`4e2d68eb2626360838985f59e6eb64e3edfd23f9269d81ece3d124737bc2abe7`],
    [`v2_q2_blind_11.csv`], [—], [`3154b240448bde8f2cd12a3cd27b121ac813eb56c85672c8f7337ab7b6412d4c`],
    [`v2_q3_full_08.csv`], [—], [`751f92ff1ab1efb8f6c0f3e94070c2ee72ebcf7fb73211ec7e8576bc96bf35f0`],
    [`v2_q3_blind_12.csv`], [—], [`9de6d43dcf6b1e35032f6d09c9a5de2a5bf16c44ed9e7bb1ff3b7738105004b6`],
    table.hline(stroke: 1.1pt),
  ),
  caption: [SHA-256 registry for the eight certified 8-bit Ising matrices and seven exported prediction CSVs.],
) <tab:hashes>

== Appendix C: Core Source Code
#let clean-src(p) = read(p).replace(regex("\p{Han}+"), "<cn-text>")

_Note:_ Complete runnable source scripts and auxiliary modules are archived in `Dapmcm26302266fj.zip`.

=== C.1 Physical Constants & Band-Center Utilities (`v2_optimized/common_physics.py`)
#set text(size: 7.2pt)
#raw(clean-src("../v2_optimized/common_physics.py"), lang: "python", block: true)

#pagebreak()
#set text(size: 11.0pt)
=== C.2 Physics-Coupled Hurdle Surrogate (`v2_optimized/v2_surrogate_q1.py`)
#set text(size: 7.2pt)
#raw(clean-src("../v2_optimized/v2_surrogate_q1.py"), lang: "python", block: true)

#pagebreak()
#set text(size: 11.0pt)
=== C.3 Active-Subspace 8-Bit QUBO/HOBO & ANOVA Engine (`v2_optimized/v2_qubo_q2_q3.py`)
#set text(size: 7.2pt)
#raw(clean-src("../v2_optimized/v2_qubo_q2_q3.py"), lang: "python", block: true)

#pagebreak()
#set text(size: 11.0pt)
=== C.4 SPQC-550 Execution & Algebraic Ising Recovery (`v2_optimized/v2_spqc550_hardware.py`)
#set text(size: 7.2pt)
#raw(clean-src("../v2_optimized/v2_spqc550_hardware.py"), lang: "python", block: true)
