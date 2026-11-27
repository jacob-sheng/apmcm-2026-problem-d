// =============================================================================
// 2026 APMCM (第 16 届亚太地区大学生数学建模竞赛) Problem D — Team Dapmcm26302266
// 队内中文对照审阅版（仅供队内审阅，赛氪正式提交请使用英文正稿 submission/Dapmcm26302266.pdf）
// =============================================================================

#let team-id = "apmcm26302266"
#let problem-chosen = "D"
#let paper-title-zh = "物理耦合代理建模、可认证 8-Bit 伊辛编译与 SPQC-550 光量子真机核验：钙钛矿稀有目标搜索"
#let paper-title-en = "Physics-Coupled Surrogate Modeling, Certified 8-Bit Ising Compilation, and SPQC-550 Verification for Rare-Target Perovskite Discovery"

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
    set text(size: 8.8pt, font: ("Times New Roman", "SimSun"))
    grid(
      columns: (1fr, auto),
      align: (left, right),
      [*Team #problem-chosen#team-id · 中文对照审阅版（仅供队内审阅）*],
      [第 #cur 页 / 共 #tot 页],
    )
    v(-6pt)
    line(length: 100%, stroke: 0.6pt + luma(110))
  },
  footer: none,
)

#set text(
  font: ("Times New Roman", "SimSun"),
  size: 10.5pt,
  lang: "zh",
)

#set par(
  justify: true,
  leading: 0.60em,
  spacing: 0.72em,
)

#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(0.55em)
  set text(size: 12.6pt, weight: "bold", font: ("Times New Roman", "SimHei"), fill: rgb("#1B365D"))
  it
  v(0.18em)
}
#show heading.where(level: 2): it => {
  v(0.38em)
  set text(size: 11.2pt, weight: "bold", font: ("Times New Roman", "SimHei"), fill: rgb("#2C4A70"))
  it
  v(0.12em)
}
#show heading.where(level: 3): it => {
  v(0.24em)
  set text(size: 10.6pt, weight: "bold", font: ("Times New Roman", "SimHei"), fill: rgb("#2C4A70"))
  it
  v(0.08em)
}

#set math.equation(numbering: "(1)", supplement: [式])
#show figure.caption: it => {
  set text(size: 9.0pt, fill: luma(40))
  it
}
#show table: set text(size: 8.3pt)
#show table: set par(justify: false)

#let theorem-box(title, body) = block(
  fill: rgb("#F4F7FA"),
  stroke: (left: 2.5pt + rgb("#1B365D"), rest: 0.4pt + luma(200)),
  inset: (x: 9pt, y: 6.5pt),
  radius: 2pt,
  width: 100%,
  breakable: false,
)[
  #text(weight: "bold", font: ("Times New Roman", "SimHei"), fill: rgb("#1B365D"))[#title。] #body
]

// =============================================================================
// 第 1 页：摘要页（与英文正稿 1:1 对应）
// =============================================================================
#align(center)[
  #v(0.10cm)
  #text(size: 10.5pt, weight: "bold", fill: luma(60))[
    2026 亚太地区大学生数学建模竞赛（APMCM）· 中文对照审阅版
  ]
  #v(0.16cm)
  #table(
    columns: (1fr, 1fr, 1fr),
    align: center,
    stroke: 0.6pt + luma(140),
    inset: 5.5pt,
    [*所选赛题（Problem）* \ #text(size: 12.2pt, weight: "bold", fill: rgb("#1B365D"))[#problem-chosen]],
    [*参赛队号（Team Control No.）* \ #text(size: 12.2pt, weight: "bold", fill: rgb("#1B365D"))[#team-id]],
    [*竞赛年份* \ #text(size: 12.2pt, weight: "bold", fill: rgb("#1B365D"))[2026]],
  )
  #v(0.24cm)
  #text(size: 13.0pt, weight: "bold", font: ("Times New Roman", "SimHei"))[#paper-title-zh]
  #v(0.06cm)
  #text(size: 9.2pt, style: "italic", fill: luma(70))[（#paper-title-en）]
  #v(0.20cm)
  #text(size: 11.6pt, weight: "bold", font: ("Times New Roman", "SimHei"), fill: rgb("#1B365D"))[摘要]
]

#v(0.08cm)
#[
#set text(size: 10.1pt)
#set par(leading: 0.58em, spacing: 0.72em)

在 $52 times 52 times 7 = 18,928$ 种立方相 $A B X_3$ 钙钛矿空间中筛选可见光光解水催化剂面临极端的稀有目标不平衡：$96.12%$ 的晶体呈金属性（$E_g = 0$），在官方铰链违约分 $V$ 下仅有 $19$ 个目标满足 $V = 0$（占比 $0.1004%$；严格内点不等式口径为 $15$ 个），且这 $19$ 个正例全部位于盲测集 `04_test_holdout`（$529$ 个未见 $(A,B)$ 对）中。针对真实相干伊辛机 `SPQC-550` 硬件（对称 8-bit 整数耦合 $[-127, 127]$），本文构建了一套“外层折基模型隔离的物理耦合代理 + 全域认证 8-bit QUBO/HOBO 编译 + 位级真机复核”全流程框架。

针对问题一，我们将目标相关特征工程与四家族基预测器（`CatBoost`、`LightGBM`、`XGBoost`、`ExtraTrees`）严格隔离于每个外层 5 折 `(A,B)`-GroupKFold 训练折内（`+100` 基路径标签扰动差 $1.42 times 10^(-14)$），随后在汇总基模型 OOF 预测上拟合堆叠权重、残差尺度与等渗校准，并在外部验证集 `03_val`（$567$ 对、$3,969$ 行）上评估。将经验 Butler--Ginley 带心关系（$R^2 = 1.0000000000$，$"RMSE" = 2.90 times 10^(-7) "eV"$；带隙误差传播 $sigma_("VBM") = sigma_("CBM") = sigma_(E_g) slash 2$）与高斯铰链闭式积分耦合，模型 `v2_PhysHurdle_Bayes` 在 `03_val` 上取得 $"RMSE" = 4.0634$（优于 `M4_PIMT_Physics_Only` 的 $4.2053$）、近目标 $"MAE"(V<1) = 2.9280$、前 50 召回 `Rec@50(V<0.5) = 94.4%` 及前 75 全部召回 `Rec@75(V<0.5) = 100.0%`（18/18）。在 $20$ 个配对种子的闭环主动筛选（预算 $K = 200$，解析超几何基准期望 $0.9070$）中，另一套轻量在线 Ridge--LightGBM hurdle 代理在古典 Top-20 预筛 + 逐轮 8-bit QUBO 采集下命中 16.20/18（$"SD"=1.077$），对比同代理古典顺序排序 16.55/18（$"SD"=1.962$），验证了混合 QUBO 采集的工程可行性而不宣称采集层优势。200 次 `(A,B)`-组不交保形拆分覆盖率达 $90.19%$ / $95.12%$，`04_holdout` 上的 $sum hat(P)_("hit") = 10.3358$ 严格界定为确定性模型概率质量总和而非未见真阳性召回。

针对问题二与问题三，我们证明了 52 维独热块在 8-bit 硬件直接缩放下的汉明重量漂移机理（可行率 $0.0%$），并通过上尾截断（$C_("cap")$）、无损活性子空间剪枝与整数量化对齐予以消解。针对三体问题三，本文提出零辅助比特的复合配对编码 `Model 3A-Comp`（$26$ / $18$ 自旋），利用行累计辅助罚界与含掩码全局消元恒等式修复 `Model 3A-HOBO`（$43$ / $29$ 自旋），并采用主效/交互项统一 $8 times$ 缩放修复 `Model 3B-ANOVA`（$33$ / $25$ 自旋，$6$ 个可区分合法能级）。经解析消去辅助变量后的主变量穷举认证与独立 HiGHS MILP 交叉核验（`mip_gap = 0`），全部 $8$ 个矩阵均实现**零非法全局基态**与**零基态辅助变量歧义**（7/8 矩阵整数基态集合包含连续代理最优 LaTiO2N，$hat(V) = 0.252406$；量化退化度 $3$--$10$ 个态）。

在博智量子 `SPQC-550` 光量子真机（`OPTIMIZATION` 模式，每任务返回 $10$ 条记录）上，4 组盲池任务分别取得 $30.0%$（`v3_q2_blind_12`，$E^*=-1000$）、$90.0%$（`v3_q3_comp_18`，$E^*=-1468$，零遗憾直击 LaTiO2N 并命中 9/15 个 Top-15 唯一候选）、$20.0%$ 严格 / $90.0%$ 主可行（`v3_q3_hobo_29`，$E^*=-1662$）和 $30.0%$（`v3_q3_anova_25`，$E^*=-1992$）严格可行记录比例，并与原始自旋文件逐比特复核一致。

*关键词：* 钙钛矿光催化；外层折隔离 hurdle 代理；Butler--Ginley 带心关系；8-Bit 伊辛编译；全域基态认证；相干伊辛机（`SPQC-550`）
]

#pagebreak()

// =============================================================================
// 第 2 页：目录
// =============================================================================
#outline(title: [目录], indent: 1.4em, depth: 3)
#pagebreak()

// =============================================================================
// 正文（第 1–7 章）
// =============================================================================

= 赛题背景、官方数据集校验与操作口径界定

== 问题背景与官方数据集划分
在 $52$ 种 $A$ 位金属、$52$ 种 $B$ 位金属与 $7$ 种阴离子配比（$"O"_3$、$"NO"_2$、$"FO"_2$、$"N"_2"O"$、$"N"_3$、$"O"_2"S"$、$"FNO"$）构成的 $52 times 52 times 7 = 18,928$ 种立方相 $A B X_3$ 钙钛矿中发现可见光光解水催化剂具有高昂的 GLLB-SC 密度泛函理论（DFT）计算代价 [1]。竞赛据此设置三项递进任务：
- *问题一（Q1 — 跨组分泛化代理建模）：* 基于仅含化学式的特征在 `02_castelli_train.csv`（$1,608$ 个 $(A,B)$ 对，$11,256$ 行）上训练，在 `03_castelli_val.csv`（$567$ 个独特 $(A,B)$ 对，$3,969$ 行）上选型验证，并对 `04_castelli_test_holdout.csv`（$529$ 个未见 $(A,B)$ 对，$3,703$ 行）评分。三份划分之间 $(A,B)$ 配对严格零重叠，强制要求按 `(A,B)` 分组交叉验证。
- *问题二（Q2 — 二体 $A B"NO"_2$ 量子优化）：* 固定 $X = "NO"_2$，将寻找代理违约分 $hat(V)$ 极小值转化为面向玻色量子 `SPQC-550` 相干伊辛机（CIM，Kaiwu SDK 1.4.4 [2, 3]）的 8-bit 有符号整数 Ising 矩阵（$H_(u v) in bb(Z) inter [-127, 127]$），覆盖全网格 `07_castelli_q2_pool.csv`（$2,704$ 条）与盲测池 `11_castelli_q2_pool_blind.csv`（$529$ 条）。
- *问题三（Q3 — 多阴离子三体优化）：* 在 $X in {"O"_3, "NO"_2, "FO"_2}$ 上联合优化 $(A, B, X)$，覆盖全池 `08_castelli_q3_pool.csv`（$8,112$ 条）与盲池 `12_castelli_q3_pool_blind.csv`（$1,587$ 条），将三次交互映射至二次 8-bit 伊辛硬件。

对全部 $9$ 个官方文件与 `sha256_public.json` 执行 `verify_dataset_checksums()` 双校验：在 Git LF 检出环境下，*CRLF 归一化后 SHA-256 匹配数为 9/9*，*原始字节匹配数为 1/9*（Markdown 数据说明文件按 LF 原字节一致，其余 $8$ 个 CSV 经 LF 转 CRLF 后与官方校验和逐比特一致），确保数据零篡改。

== 关键操作口径界定（`19` vs. `15`、金属态与 $A=B$ 敏感性）
为消除语义歧义，全文固定以下四条操作准则：
1. *金属相（`gap = 0`）：* 全网格中恰有 $18,193$ / $18,928$（$96.12%$）个化合物满足 $E_g = 0.0 "eV"$。这是真实金属相而非缺失值，其下带隙违约惩罚项确定为 $v_("gap,low") = (1.4 slash 0.5)^2 = 7.84$。
2. *五项分段铰链分 $V$ 与 `19` vs. `15` 边界差异：* 在全部 $15,225$ 条有标签样本（`02 + 03`）上复算官方五项二次铰链总分
   $ V &= [max(0, (e_("form") - 0.21)/0.5)]^2 + [max(0, (1.4 - E_g)/0.5)]^2 + [max(0, (E_g - 3.1)/0.5)]^2 \
       &quad + [max(0, (5.73 - "VBM")/0.5)]^2 + [max(0, ("CBM" - 4.50)/0.5)]^2 $ <eq:vscore>
   与官方 `V` 列误差仅 $max |Delta V| = 1.02 times 10^(-5)$。在此非严格 $V = 0$ 口径（$e_("form") <= 0.21$、$1.4 <= E_g <= 3.1$、$"VBM" >= 5.73$、$"CBM" <= 4.50$）下，全空间共有 *$19$* 个正目标（$0.1004%$：`O3` $6$、`NO2` $6$、`FO2` $6$、`N2O` $1$），全部落在 `04_test_holdout` 的 $19$ 个不同 $(A,B)$ 对中，`02_train`（$V<0.5$ 有 49 个，$V<1.0$ 有 83 个）与 `03_val`（$V<0.5$ 有 18 个，$V<1.0$ 有 27 个）均不含 $V=0$ 样本；赛题文字另述及满足严格不等式（$e_("form") < 0.21, 1.4 < E_g < 3.1$）的 *$15$* 个目标。本文以官方评价函数 $V = 0$（$19$ 目标）为主计算目标，显式标注口径差异而不虚构未见 `04` 标签。
3. *同核配对（`a_eq_b = 1`）：* 规则允许保留或剔除 $A = B$ 行（共 364/18,928 $= 1.92%$，含 $0$ 个正例）。我们在 Q1 代理训练中保留以维持网格连续性，在 Q2/Q3 QUBO 哈密顿量中对 $a_i b_i = 1$ 施加硬惩罚禁用。对照审计显示剔除 $A = B$ 后，四池代理最优均保持为 LaTiO2N（$hat(V) = 0.252406$），`04` Top-20 完全重合（20/20），Top-50 仅改变第 32 位 SnSnO3（$hat(V) = 2.4358$）。
4. *双盲测谱系（`02-only` vs. `02+03` 重训）：* 所有 `03_val` 性能比较严格基于 `02-only` 训练；为服务下游盲池筛选，锁定超参后在 `02 + 03`（$15,225$ 行）重训，同时导出 `v2_holdout_04_train02only.csv` 与 `v2_holdout_04.csv`（Spearman $rho = 0.99704$，Top-10 重合 9/10，Jaccard $0.8182$；$sum hat(P)_("hit")$ 分别为 $11.0008$ vs. $10.3358$）。

#figure(
  image("../v2_optimized/figures/schematic01_pipeline_architecture.png", width: 99%),
  caption: [端到端总体技术路线：数据集双 SHA-256 核验、外层折基模型隔离 + Pooled-OOF 元层拟合 Q1 代理、上尾活性子空间剪枝、8-bit QUBO/Ising 精确编译、主变量穷举认证及同预算经典/`SPQC-550` 核验。],
) <fig:pipeline>

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (3.45cm, 1.75cm, 3.00cm, 2.45cm, 2.05cm, 1.40cm, 2.55cm),
      stroke: none,
      align: (left, left, left, left, left, left, left),
      inset: (x: 2.5pt, y: 2.8pt),
      table.hline(stroke: 1.1pt),
      [*模型 / 产物 ID*], [*训练集*], [*选型 / OOF 协议*], [*校准集*], [*推理池*], [*标签类型*], [*SHA-256 (前16位)*],
      table.hline(stroke: 0.6pt),
      [`v1_OneHot_FM_k8`], [`02` (11,256)], [固定 FM (`k=8, lr=0.03`)], [无], [`03` (3,969)], [标量 $V$], [`14ef801edd70fffc`],
      [`v1_Direct_LGBM_V`], [`02` (11,256)], [固定 LGBM (23 描述符)], [无], [`03` (3,969)], [标量 $V$], [`14ef801edd70fffc`],
      [`v1_Direct_LGBM_log1pV`], [`02` (11,256)], [对 `log1p(V)` 拟合 LGBM], [无], [`03` (3,969)], [标量 $V$], [`14ef801edd70fffc`],
      [`M4_PIMT_Physics_Only`], [`02` (11,256)], [历史 `02 -> 03` 消融归档], [无], [`03` (3,969)], [多物性], [`table1_ablation`],
      [`M5_PIMT_Hybrid_Ensemble`], [`02` (11,256)], [历史 `02 -> 03` 消融归档], [无], [`03` (3,969)], [多物性], [`table1_ablation`],
      [`v2_Direct_4FamPhys`], [`02` (11,256)], [外层 5 折 `(A,B)`-GKF], [无], [`03`, `04`], [标量 $V$], [`ac5d939d4dd63f3a`],
      [`v2_PhysHurdle_Bayes (02)`], [`02` (11,256)], [外层 5 折基隔离 (`1.42e-14`)], [Pooled OOF; `03` 200拆分], [`03`, `04_02only`], [多物性], [`2090c512ddc3e614`],
      [`v2_PhysHurdle_Bayes (02+03)`], [`02+03` (15,225)], [外层 5 折基隔离], [Pooled OOF（无迁移保证）], [`04`, `07`, `11`, `08`, `12`], [多物性], [`89eca9b89f850123`],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [数据与模型协议谱系表（统一表 1）：完整记录 `v1`、历史 `M4`/`M5` 基线与外层隔离 `v2` 模型来源（全 64 位 SHA-256 见附录 B）。],
) <tab:lineage>

== 基本假设与符号说明
1. *盲测仅用化学式特征：* 在 `04_test_holdout` 及筛选池 `07`/`08`/`11`/`12` 上仅使用元素符号 $(A,B,X)$ 与教科书周期表常量 [5, 6]。
2. *混合积分的高斯边缘近似：* 在 `integrate_bayes_mixture` 中，生成能 $e_("form")$ 与半导体带隙残差假定为条件独立高斯分布，方差按 0.75/0.25 加权 OOF RMS 与集成标准差，属代理损失近似而非精确联合后验。
3. *带边对称耦合：* $"VBM" = E_c + E_g slash 2$ 与 $"CBM" = E_c - E_g slash 2$ 在全部 $15,225$ 条有标签数据上成立至 CSV 舍入精度（中位偏差 $0.0000 "eV"$，最大 $0.0500 "eV"$）。
4. *条件上尾截断 $C_("cap")$：* 将 $hat(V)$ 截断至 $C_("cap")$ 严格保序所有 $hat(V) < C_("cap")$ 候选并保证保留池最小值（只要 $min hat(V) < C_("cap")$），同时显式权衡高违约区 Top-25 截断损失（详见第 6 章）。
5. *硬件整数约束：* `SPQC-550` 要求零对角对称整数耦合 $H_(i j) in bb(Z) inter [-127, 127]$（$<= 8$ bits）、总比特数 $N <= 550$ 及按 $E(s) = -s^top H s$ 排序解码 [2]。

= 问题一：物理耦合代理模型与不确定性审计

== 经验带心关系、定量窗口与物理特征例外审计
对 `02_train` 中 GLLB-SC 电子结构目标的分析揭示了极具规律的数据内结构，源于 Castelli 等人采用的 Butler--Ginley 几何平均带心公式 [1, 4]：

#theorem-box("经验命题 1（数据内 Butler–Ginley 带心关系与误差传播）", [
  在 GLLB-SC 数据集上，带心 $E_c equiv ("VBM"+"CBM") slash 2$ 在外层每个训练折内经对数空间 Ridge 回归（`alpha = 1e-10`）建模：
  $ ln E_c (A, B, X) = w_0 + w_A + w_B + w_X. $ <eq:bg>
  在外部验证集 `03_val`（$567$ 个 `(A,B)` 对，$3,969$ 行）上，@eq:bg 达到 *$R^2 = 1.0000000000$* 与 *$"RMSE" = 2.90 times 10^(-7) "eV"$*，具有高度位点交换对称性（$r(w_A, w_B) = 1.00000000$，$op("std")(w_A - w_B) = 0.00000000$）并与 Mulliken 电负性显著相关（$r = 0.9183, rho = 0.9295$）。然而，由于 $"VBM" = E_c + E_g slash 2$ 且 $"CBM" = E_c - E_g slash 2$，带心极准*并不会*消除 VBM/CBM 的预测误差：带隙预测不确定性仍按 *$sigma_("VBM") = sigma_("CBM") = sigma_(E_g) slash 2$* 直接传播至两个带边。
])

#theorem-box("命题 2（电子带边可行窗口的存在性充要条件）", [
  在带边对称耦合（$"VBM" = E_c + E_g slash 2, "CBM" = E_c - E_g slash 2$）与官方 $V = 0$ 非严格带边约束（$1.4 <= E_g <= 3.1, "VBM" >= 5.73, "CBM" <= 4.50$）下，记 $g_(min)(E_c) equiv max(1.4, 2(5.73 - E_c), 2(E_c - 4.50))$。则*存在*带隙 $E_g >= 0$ 同时满足全部电子学条件，当且仅当
  $ [ exists E_g >= 0 : g_(min)(E_c) <= E_g <= 3.1 ] <==> 4.18 "eV" <= E_c <= 6.05 "eV". $ <eq:ecwin>
  对具体材料而言，电子学可行仍需检验实际带隙是否满足 $g_(min)(E_c) <= E_g <= 3.1$（而 15 目标子集要求 4 条严格不等式）。@eq:ecwin 仅为*必要非充分*条件：例如 $E_c = 5.10 "eV"$ 的金属化合物（$E_g = 0$）虽落入 $[4.18, 6.05] "eV"$，却因 $E_g < g_(min)(5.10) = 1.4$ 必然承担 $v_("gap,low") = 7.84$。
])

#figure(
  image("../v2_optimized/figures/schematic02_band_alignment_window.png", width: 99%),
  caption: [对称带边耦合机制（$sigma_("VBM") = sigma_("CBM") = sigma_(E_g) slash 2$）、金属相假阳性反例（$E_c = 5.10 "eV", E_g = 0$）及存在量词化的 $(E_c, E_g)$ 可行走廊 $4.18 <= E_c <= 6.05 "eV"$。],
) <fig:bandwindow>

`common_physics.py` 中所有物理描述符仅作为输入特征喂给学习器（从不用作不可逆硬过滤），以防错杀例外材料。表 2 审计了每类特征在 `02+03`（$15,225$ 行）及近目标样本（$V < 1.0$: $110$ 条；$V < 0.5$: $67$ 条）上的实测例外。

#figure(
  table(
    columns: (auto, 1.1fr, 1.4fr, auto, 1.3fr),
    stroke: none,
    align: (left, left, left, center, left),
    inset: (x: 2.8pt, y: 3.2pt),
    table.hline(stroke: 1.1pt),
    [*特征 / 物理关系*], [*理论来源*], [*有效性与近似假设*], [*硬过滤?*], [*`02+03` 实测例外计数*],
    table.hline(stroke: 0.6pt),
    [Butler--Ginley $E_c$], [$ln E_c = w_0 + w_A + w_B + w_X$ [1, 4]], [数据内 Ridge (`alpha=1e-10`)；$sigma_("VBM")=sigma_("CBM")=sigma_(E_g) slash 2$], [否], [`03_val` RMSE $2.90 times 10^(-7)$ eV；101/110 ($V<1$) 满足 $4.18 <= E_c <= 6.05$],
    [Hurdle 与可行窗口], [$g_(min)(E_c)$ 与高斯铰链 [@eq:vbayes]], [高斯边缘；$P_("hit")$ 条件独立近似], [否], [`03` 半导体残差相关 $r=-0.2930$ ($p=4.22 times 10^(-4)$)],
    [电子奇偶性], [Bloch--Wilson 满带规则 ($Z mod 2$)], [非自旋极化闭壳层启发；对莫特/磁性绝缘体失效], [否], [`Z_parity==1` (奇电子): 0/110 ($V<1$), 0/67 ($V<0.5$), 5/573 ($E_g>0$)],
    [电价平衡与闭壳层], [Pauling 电价最小残差 $min |q_A+q_B+q_X|$], [人工整理 `ELEMENT_DATA`；`Fe/Co/Ni/Ru` 缺闭壳层], [否], [`is_chg_bal==0`: 3/110 ($V<1$)；`is_cs_bal==0`: 4/110],
    [Bartel 容忍因子 $tau$], [$tau = r_X slash r_B - n_A (n_A - (r_A slash r_B) slash ln(r_A slash r_B))$ [7]], [仅 $r_A slash r_B > 1.05$ 有效；$p_("Bartel")$ 属启发式], [否], [`bartel_valid==0`: 26/110 ($V<1$)；$tau >= 4.18$: 87/110],
    [$d^0 slash d^(10)$ 与电负性], [族序代理与 $I = 1 - e^(-0.25 Delta chi^2)$ [5, 6]], [族序代理构型；`s0_p_mag` 刻画 $|mu_B|>0.05$], [否], [`B_is_d0_d10==0`: 1/110 (ZrMgO3)；$|mu_B|>0.05$: 22/573 ($E_g>0$)],
    table.hline(stroke: 1.1pt),
  ),
  caption: [物理特征审计表：理论来源、近似条件、非硬过滤原则及在 `02+03`（$n=15,225$）上的实测例外计数。],
) <tab:phys-audit>

== 外层折基模型隔离、Stage-0 隐 DFT 与 Bayes 混合代理
我们在 `v2_surrogate_q1.py` 中构造 $154$ 维特征（$100$ 个晶体化学描述符、$7$ 个 $E_c$ 解析特征、$40$ 个经验 Bayes 内层边缘统计与 $4$ 个 Stage-0 LightGBM 隐含 DFT 预测：`a_lat`, `fermi_level`, `|mu_b|`, `P(|mu_b|>0.05)`）。目标依赖特征构建与四家族基模型（`CatBoost`、`LightGBM`、`XGBoost`、`ExtraTrees`）完全隔离于每个外层 `(A,B)` 训练折内；随后的单纯形权重、残差尺度与 50/50 等渗校准在汇总的基模型 outer-OOF 上拟合，而端到端泛化在外部 `03_castelli_val.csv`（竞赛公开验证集）上评价。同一汇总 OOF 标签上的元层指标及 Stage-0 内折指标（`s0_a_lat` $R^2=0.9603$、`s0_fermi` $R^2=0.9320$、`s0_abs_mub` $R^2=0.6133$、`s0_p_mag` AUC $=0.8539$）均明确归类为内部开发诊断。`+100` 基路径标签扰动测试验证基特征与基模型路径完全零泄漏（`max_feature_delta = 0.0`, `max_pred_delta = 1.42e-14 < 1e-12`）。

设高斯边缘 $Y tilde.op cal(N)(mu, sigma^2)$，记 $m = plus.minus(mu - theta) slash s$、$tilde(sigma) = sigma slash s$、$z = m slash tilde(sigma)$，可得铰链平方期望闭式解：
$ cal(E)_plus.minus (mu, sigma; theta, s) = (m^2 + tilde(sigma)^2) Phi(z) + m tilde(sigma) phi(z). $ <eq:hinge>
联立生成能（`03_val` $R^2=0.9199, "RMSE"=0.2108$）、半导体开关概率（$"ROC-AUC"=0.9850, "PR-AUC"=0.8247, "ECE"=0.00570$）及条件带隙（$R^2=0.7587, "RMSE"=0.7922$），导出混合预期违约分与命中打分：
$ hat(V)_("Bayes")(x) = cal(E)_+ (hat(mu)_("ef"), hat(sigma)_("ef"); 0.21, 0.5) + sum_(r in {"gl","gh","VBM","CBM"}) [(1 - hat(p)_("semi")) v_r (0; E_c) + hat(p)_("semi") cal(E)_r (hat(mu)_g, hat(sigma)_g; E_c)], $ <eq:vbayes>
$ hat(P)_("hit")(x) = hat(p)_("semi")(x) dot Phi((0.21 - hat(mu)_("ef")) / hat(sigma)_("ef")) dot max(0, Phi((3.1 - hat(mu)_g) / hat(sigma)_g) - Phi((g_(min)(E_c) - hat(mu)_g) / hat(sigma)_g)). $ <eq:phit>

== `03_val` 定量基准对比（含 `M4`/`M5` 与近目标取舍）
@tab:q1 在统一 `02 -> 03` 协议（$567$ 对 / $3,969$ 行）下比较 7 个代理，纳入随包归档的 `M4_PIMT_Physics_Only` 与 `M5_PIMT_Hybrid_Ensemble`。`v2_PhysHurdle_Bayes` 取得最低全局 $"RMSE" = 4.0634$、最低 $"MAE" = 2.0696$、最高 $R^2 = 0.8632$ 与最高 Spearman $rho = 0.9527$，并将近目标 $"MAE"(V<1)$ 从直接回归的 $5.4475$ 降至 *$2.9280$*（$"MAE"(V<0.5)=2.4126$），`Rec@50(V<0.5) = 94.44%`（17/18），`Rec@75(V<0.5) = 100.0%`（18/18），`Rec@100(V<1) = 100.0%`（27/27）。相较早期未外层隔离原型（`RMSE = 4.0675`, `MAE(V<1) = 2.6167`, `Rec@50(V<0.5) = 15/18`），本次外层基隔离版本在多步修订后将 Top-50 稀有命中提高 $+2$ 个、全局 RMSE 微降，同时伴随局部 `MAE(V<1)` 上升 $+11.9%$（报告为整体修订观测取舍，未做单因子剥离消融）。代码管线重构使用了 AI 编程辅助 [13] 并通过扰动自检。

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (3.55cm, 1.05cm, 1.05cm, 1.00cm, 1.00cm, 1.45cm, 1.65cm, 1.55cm, 1.55cm, 1.55cm, 1.55cm),
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
  caption: [`03_castelli_val.csv` 代理性能统一对比表（统一表 2，$n = 3,969$ 行 / $567$ 个 `(A,B)` 对；总运行时间 `996.7s`，对比 `v1` 为 `2.77s`）。],
) <tab:q1>

#figure(
  image("../v2_optimized/figures/fig01_v1_vs_v2_surrogate_metrics.png", width: 98%),
  caption: [`03_val` 验证表现（567 对、3,969 行）：(a) 全局 RMSE/MAE（含归档 `M4`/`M5`）；(b) 近目标 `V<1.0` 召回率；(c) 近目标局部 MAE（`M4`/`M5` 无逐行 CSV 故不列入）；(d) 数据内 Butler--Ginley 带心一致性验证。],
) <fig:q1metrics>

#figure(
  image("../v2_optimized/figures/fig02_residual_and_calibration.png", width: 98%),
  caption: [`03_val` 残差与概率校准审计：(a) 近零总偏差（$-0.0328$；金属 RMSE $=3.7182$，半导体 $=9.4570$）；(b) $E_c$ 分箱误差剖面；(c) $hat(p)_("semi")$ 十箱校准曲线（ROC-AUC $=0.9850$，ECE $=0.00570$）。],
) <fig:q1calib>

== 20 种子闭环主动筛选基准（`03_val`，解析基线与混合 QUBO）
在 $K = 200$ 预算（$50$ 随机初始 + $15$ 轮 $times 10$ 次查询，$N=3,969, M=18$ for $V<0.5$）下运行 20 个配对种子对比：
- *解析超几何基准与 `A=B` 资格口径：* 无放回抽样期望 $bb(E)[H_(0.5)(200)] = 200 times 18 slash 3969 = 0.9070$（$"SD" = 0.9261, 5.04%$）；`Random` 实测均值 `0.60/18`（$"SD"=0.663$）。`03_val` 含 $77$ 条同核 `A=B` 行（最低 $V=6.6537$，0 正例），全集初始池与 `Random`/`SeqRank` 在 $3,969$ 行抽取，而 `v2_Real_8Bit_QUBO` 解码禁用 `A=B`（非同核 3,892 子集解析期望 $0.9250$）。
- *`L-BFGS-B` 收敛 FM 基线：* 二阶 One-Hot FM 在 `v1_FM_LBFGS_SeqRank` 上记录 300/300 次收敛（4.30/18），对比 $112$ 自旋 `v1_FM_LBFGS_QUBO`（1.20/18，调用同优化器，逐轮收敛标志未另归档，0 回退）。
- *同代理混合采集对照（轻量在线 hurdle）：* 在相同轻量在线物理 hurdle 代理（使用多物性标签 $e_("form"), E_g, "VBM", "CBM"$，Ridge $E_c$ + LightGBM hurdle，不含 Stage-0 与 4 家族堆叠）下，古典 Top-20 预筛 + 逐轮 8-bit `Model 3A-Comp` 混合采集 `v2_Real_8Bit_QUBO`（300/300 轮精确 8-bit，均值 $28.77$ 自旋，0 回退）命中 *16.20/18*（$"SD"=1.077$），对比同代理顺序排序 `v2_Classical_SeqRank` *16.55/18*（$"SD"=1.962$）。该结果支撑混合采集工程可行性，不宣称采集层或量子优势。

#figure(
  image("../v2_optimized/figures/fig03_closed_loop_fmqa_benchmark.png", width: 97%),
  caption: [20 种子闭环筛选（`03_val`, $K=50..200$）：包含解析超几何基准（0.9070）、`L-BFGS-B` FM（`SeqRank` 300/300 收敛；`QUBO` 逐轮标志未归档）、标量 `v2_PhysLGBM`、古典顺序 `v2_Classical_SeqRank`（16.55/18, SD 1.962）与混合 8-bit `v2_Real_8Bit_QUBO`（16.20/18, SD 1.077）。],
) <fig:closedloop>

== 组不交保形校准与 `04_holdout` 作用域限定期望
在 `03_val` 上执行 200 次 `(A,B)`-组不交 50/50 拆分（`group_overlap = 0`），按预测概率分层 $hat(p)_("semi") >= 0.20$ 划分：实测经验覆盖率达 $90.19% plus.minus 1.65%$（名义 90%）和 $95.12% plus.minus 1.13%$（名义 95%）。但 90% 保形半宽在两层呈现陡峭差异：低概率预测层（$hat(p)_("semi") < 0.20$，$n=3,808$，含 $29$ 真半导体）半径 $4.411 plus.minus 0.252$，而半导体/边界预测层（$hat(p)_("semi") >= 0.20$，$n=161$，含 $112$ 真半导体）半径高达 $12.375 plus.minus 0.848$（中位数 $12.044$），表明保形区间在候选带仍较宽且不能正式外推至 `02+03` 重训集或自适应 Top-$K$ 集合。此外 `03_val` 半导体（$n=141$）存在残差负相关（$r=-0.2930, rho=-0.3349$），且无截断高斯条件 $E_g$ 在 `04` 上保有负尾概率（全集均值 0.1743，Top-50 最大 0.0110）。

在 `04_test_holdout`（`02+03` 重训）上，确定性模型总打分 $sum hat(P)_("hit") = 10.3358$（Top-30 占 8.6001 $=83.2%$，Top-50 占 9.4889 $=91.8%$）仅刻画模型概率质量总和而非针对 19 个隐藏正例的真阳性召回；模型内部独立 Bernoulli 标准差 $sqrt(sum p_i(1-p_i)) = 2.30$ 并非确定性求和 $10.3358$ 的置信区间。

#figure(
  image("../v2_optimized/figures/fig08_electronic_phase_and_conformal_audit.png", width: 98%),
  caption: [`04_holdout` 相图与作用域限定统计：(a) $(E_c, hat(e)_("form"))$ 相分布；(b) 累计模型得分 $sum hat(P)_("hit")$（Top-30 占 8.6001/10.3358 $=83.2%$，非真实召回）；(c1--c4) Top-10 盲测星标四项电子指标审计。],
) <fig:conformal>

= 问题二：8-Bit 二部图 QUBO 与活性子空间

== 稀疏掩码惩罚 QUBO 与三条正确性定理
固定 $X = "NO"_2$，构建上尾截断零平移矩阵 $macron(M)_(i j) = min(hat(V)_(i j), C_("cap")) - C_("cap") in [-C_("cap"), 0]$（合法池内且 $A_i != B_j$，否则为 $0$）及惩罚 QUBO：
$ cal(H)_("Q2")(bold(a), bold(b)) = sum_(i,j) macron(M)_(i j) a_i b_j + P_("block") (sum_i a_i - 1)^2 + P_("block") (sum_j b_j - 1)^2 + sum_((i,j) in.not cal(P)_("valid")) P_("block") a_i b_j. $ <eq:q2qubo>
通过参考自旋 $s_("ref")$ 映射 $x_i = (1 + s_i s_("ref")) slash 2$ 转为对称 Ising 矩阵 $E(s) = -s^top H s$ 后，产生以下三个核心理论结论：

#theorem-box("定理 1（8-Bit 独热块量化漂移机理）", [
  在 $N$ 元独热块朴素 $B$-bit 量化（$M_B = 2^(B-1)-1$）中，舍入 $J_B = op("round")(M_B slash (N-2))$ 破坏 $h_("pen") slash J_("pen") = N-2$ 比例，将纯罚函数曲线按汉明重量 $k$ 的连续松弛驻点漂移至 $k^* = N slash 2 - M_B slash (2 J_B)$。对 $N=52, M_8=127$，$J_8=3$ 对应连续驻点 $k^* = 4.83$（整数汉明重量罚函数极小值 $k=5$），解释了实验上 `v1` 8-bit 可行率坍塌至 $0.0%$ 的现象。
])
*证明。* 在 $E(s)=-s^top H s$ 下，汉明重量 $k$ 的独热罚能量为 $E_("pen")(k) = J_B[(2k-N)^2-N] + 2M_B(2k-N)$，令其对 $k$ 导数为零即得连续驻点 $k^* = N slash 2 - M_B slash (2 J_B)$（当 $N=52, J_8=3$ 时整数极小在 $k=5$）。 #h(1fr) $square$

#theorem-box("定理 2（二部图惩罚充分界与零误差整数对齐）", [
  1. 设 $macron(M)_(i j) in [-C_("cap"), 0]$ 且存在有限合法池最小值 $m^* equiv H_("feas")^* in [-C_("cap"), 0]$，矩阵类充分界 *$P_("block") > (3 slash 2) C_("cap")$* 统一保证任意非法态 $(k_A, k_B) != (1,1)$ 的能量严格高于*至少一个*合法 $(1,1)$ 态（即 $m^*$），故全局最优点必合法（$3 slash 2$ 在 $k_A=k_B=2$ 处取紧，系矩阵类统一充分界而非单实例必要条件；且无需高于所有合法态，如在 $2 times 2$ 全 $-1$ 块搭配 $C_("cap")=1, P_("block")=1.6$ 与另设 $0$ 成本合法态上，非法态能量 $-4+3.2=-0.8 < 0$ 但仍高于 $-1$）。
  2. 令 $m_(i j) = op("round")(M_(max) macron(M)_(i j) slash C_("cap")) in [-M_(max), 0]$、$Q_("obj") = 8 m_(i j)$、$j_("pen") = ceil(2 alpha_P M_(max))$（$alpha_P = 2.25$）及 $P_("block") = 4 j_("pen")$，则只要 $max |H_(u v)| <= 127$，代数映射 @eq:ising-recovery 所得所有 $H_(u v)$ 均为精确整数且无量化漂移。
])
*证明。* 若为 $0$-hot 态（$k_A=0$ 或 $k_B=0$），零个目标项激活而 $(k_A-1)^2+(k_B-1)^2 >= 1$，故 $cal(H)_("invalid") >= P_("block") > 0 >= m^*$（当 $m^*=0$ 时亦严格成立）。若 $k_A >= 1, k_B >= 1$ 且 $(k_A, k_B) != (1,1)$，因至多激活 $k_A k_B$ 项且每项满足 $macron(M)_(i j) >= m^*$（合法）或施加 $+P_("block") > 0 >= m^*$（非法），目标部分至少为 $k_A k_B m^* >= m^* - (k_A k_B - 1) C_("cap")$。比较罚项 $P_("block")[(k_A-1)^2+(k_B-1)^2]$ 即得在 $P_("block") > 1.5 C_("cap")$（极值于 $k_A=k_B=2$）下 $cal(H)_("invalid") > m^*$。因 $Q$ 全为 $4$ 的整数倍，映射到 $H_(i j) = -Q_(i j) slash 4$ 无截断误差。 #h(1fr) $square$

#theorem-box("定理 3（C_cap 下界无损活性子空间嵌入）", [
  任意满足 $min_j M_(i j) >= C_("cap")$ 的元素 $i$ 恒有 $macron(M)_(i j) = 0$，必无法达到 $hat(V) < C_("cap")$。剪枝非活性行列可无损保留全部 $hat(V) < C_("cap")$ 最优集合，将自旋数由 $105$ 分别无损压缩至 `Q2_full_07` 的 *$17$*（$C_("cap")=2.0$）与 `Q2_blind_11` 的 *$12$*（$C_("cap")=2.5$）。
])

== 认证 Q2 结果与盲测 Top-10 违约分解
全域 $2^(n_("prim"))$ 穷举验证 `Q2_full_07`（$65,536$ 态，$E^*=-1558, C=1390$，退化度 10）与 `Q2_blind_11`（$2,048$ 态，$E^*=-1000, C=824$，退化度 4）均为 `illegal_ground_count = 0`，均包含连续池最优 LaTiO2N（$hat(V) = 0.252406$，regret $0.0$）。@fig:q2 与 @tab:q2top10 分别展示 4–12 bit 量化扫描与 Q2 Blind Top-10 成本分解。

#figure(
  image("../v2_optimized/figures/fig04_q2_v1_vs_v2_hardware_drift.png", width: 98%),
  caption: [问题二定量结果：(a) `Q2_blind_11` 独立 4–12 bit 扫描（`v1` 在 $<12$ bit 全塌缩为 0%；`v2` 自 5–12 bit 保持 100% 可行且 7–12 bit 满足精确整数条件）；(b) Q2 Blind Top-10 五项违约分分解。],
) <fig:q2>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    align: (center, left, right, right, right, right, right, right, right, right, right),
    inset: 2.9pt,
    table.hline(stroke: 1.1pt),
    [*排名*], [*化学式*], [*$hat(V)$*], [*$hat(P)_("hit")$*], [*$hat(E)_g$*], [*$E_c$*], [*$hat(v)_("form")$*], [*$hat(v)_("gl")$*], [*$hat(v)_("gh")$*], [*$hat(v)_("VBM")$*], [*$hat(v)_("CBM")$*],
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
  caption: [Q2 Blind 池（`11`, $X="NO"_2$）Top-10 候选五项违约分分解（`02+03` 重训）。],
) <tab:q2top10>

= 问题三：次模 HOBO-to-QUBO 降阶、认证与 ANOVA 分解

== 三套活性子空间编码与完整含掩码消元定理
在 $X in {"O"_3, "NO"_2, "FO"_2}$ 上联合优化 $(A, B, X)$ 形成三次目标。利用轴不对称性与 $C_("cap")$ 剪枝（`08` 为 $1.25$，`12` 为 $1.50$），构建并认证三套 8-bit 编码：
1. *`Model 3A-Comp`（零辅助复合 `(A,X)-B` 二部图 QUBO）：* 将活性 $(A,X)$ 配对作为单一独热向量 $u_(i k)$ 直接耦合 $b_j$，依据定理 2 在零辅助变量下实现精确编码（`08` 为 26 自旋、`12` 为 18 自旋）。
2. *`Model 3A-HOBO`（修复非对称单侧降阶 HOBO）：* 引入辅助变量 $y_(i k)$ 并以 Freedman--Drineas 器件 [9] 构建
   $ cal(H)_("HOBO")(z, y) &= sum_((i,k) in cal(A)) [ sum_j 8 m_(i j k) y_(i k) b_j + P_("aux")(i,k) y_(i k)(2 - a_i - c_k) ] \
       &quad + P_("block")[(sum a_i - 1)^2 + (sum b_j - 1)^2 + (sum c_k - 1)^2] + H_("mask")(z), $ <eq:h3a>
   其中掩码罚项 $H_("mask")(z) equiv sum_((i,j) in.not cal(P)_(A B)) P_("forbid") a_i b_j$（$P_("forbid") = 8 ceil(j_("pen") slash 2) >= P_("block")$：`08` 上 $40$ vs. $36$，`12` 上 $56$ vs. $52$）禁止 $A=B$ 与缺位配对。

#theorem-box("定理 4（Model 3A-HOBO 全域辅助消元与基态正确性）", [
  采用行绝对值累计辅助罚界
  $ P_("aux")(i,k) = sum_j |8 m_(i j k)| + 8 > sum_j |8 m_(i j k)| $ <eq:paux>
  与精确分块基数惩罚界
  $ P_("block") > max_((k_A, k_B, k_X) != (1,1,1)) (-min H_("obj")(k_A, k_B, k_X)) / ((k_A - 1)^2 + (k_B - 1)^2 + (k_X - 1)^2), $ <eq:pblock>
  可保证：(i) 对全 $2^(n_("prim"))$ 空间内任意主赋值 $z$，取 $y_(i k) = a_i c_k$ 即可*达到*全局消元极小：
  $ min_y cal(H)_("HOBO")(z, y) = H_("obj")(z) + P_("block")[(k_A-1)^2 + (k_B-1)^2 + (k_X-1)^2] + H_("mask")(z) $ <eq:hoboelim>
  （当 $a_i c_k=0$ 或交互和为负时 $y_(i k)$ 唯一；若 $a_i c_k=1$ 且交互和恰为 $0$，如激发合法态 BaGeO2N，允许 $y_(i k) in {0,1}$ 在零能处平局而不改最小能量）；(ii) 全 $2^N$ 空间*零非法全局基态*（`08` 43 自旋、`12` 29 自旋），且在全局基态能级处（$-2134$ 与 $-1662$）辅助变量均零歧义（见 @tab:cert）。
])
*证明。* 当 $a_i c_k = 0$ 时，$2-a_i-c_k >= 1$，由 @eq:paux 得 $y_(i k)$ 一次项系数 $c_(i k)(z) >= 8 > 0$，故即使所有 $b_j=1$ 亦唯一取 $y_(i k)=0$；当 $a_i c_k = 1$ 时系数 $<= 0$，$y_(i k)=1$ 必达到最小值 $a_i c_k sum_j 8 m_(i j k) b_j$。因块罚项与 $H_("mask")(z)$ 均与 $y$ 无关，对 $y$ 最小化原封不动保留 $H_("mask")(z)$（已在真实矩阵上经 `(Ge,Ge,O3)` $=+40$ 与 `(Ba,Hf,NO2)` $=+56$ 掩码态验证），进而结合 @eq:pblock 保证全空间零非法全局基态。 #h(1fr) $square$

3. *`Model 3B-ANOVA`（统一 $8 times$ 缩放紧致 ANOVA-QUBO）：* 在 Top-$K$ 活性子空间拟合二阶函数 ANOVA（`08` $33$ 自旋、`12` $25$ 自旋，$0$ 辅助），对主效对角与交叉非对角统一执行 $8 times$ 整数量化，消除尺度失配并呈 6 个清晰合法整数能级。

#figure(
  image("../v2_optimized/figures/schematic03_qubo_ising_compilation.png", width: 99%),
  caption: [Q3 编译与认证流程：三套活性子空间 QUBO 编码、精确代数恢复（@eq:ising-recovery）、主变量穷举认证与三门禁真机解码。],
) <fig:q3comp>

== 全域基态认证、独立 MILP 交叉核验与三层分解
表 5 汇报 8 矩阵全证书结果：结构化精确穷举与独立 Fortet/McCormick MILP 线性化（`scipy.optimize.milp` HiGHS，`mip_gap = 0`，实现于 `scripts/verify_milp_qubo_crosscheck.py` 并归档 `v2_milp_crosscheck.json`，系基于外部审稿建议 [15] 由团队独立实现并运行验证）在全部 $8$ 个矩阵上得出完全一致的全局最小值，零非法全局基态与零基态辅助歧义。其中 7 个矩阵的整数基态集包含连续代理最优 LaTiO2N（full ANOVA 最佳基态 regret 为 `0.005211`，对应 SrTaO2N $hat(V)=0.257617$）。表 6 与表 7 分别呈压缩四行归因与三层 ANOVA 误差分解。

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (2.50cm, 1.58cm, 1.32cm, 1.55cm, 1.75cm, 1.75cm, 0.78cm, 1.15cm, 2.27cm, 2.25cm),
      stroke: none,
      align: (left, center, center, center, left, right, center, center, right, left),
      inset: (x: 2.2pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*矩阵 ID*], [*自旋*], [*$M_(max)$/$P_("blk")$*], [#strong[DR /] \ #strong[`max|J|`]], [*穷举方式*], [*`cert_min_E` / $C$*], [*退化*], [*非法 GS/Tie*], [*最优/最差 Reg.*], [*SHA-256 (12)*],
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
  caption: [编码保真度与全域基态认证汇总表（统一表 3）：8/8 整数矩阵通过 8-bit、零非法基态、零辅助歧义及 HiGHS MILP（`mip_gap=0`）核验。],
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
      [*四行压缩归因*], [*网格规模*], [*主/辅/总自旋*], [*<=550?*], [*对比 A-B*], [*对比全 A-X*], [*说明*],
      table.hline(stroke: 0.6pt),
      [`Full-space A-B Rosenberg`], [52×52×3], [107 / 2,652 / 2,760], [否], [1.0×], [0.0588×], [远超 `SPQC-550` 容量上限],
      [`Full-space A-X Rosenberg`], [52×52×3], [107 / 156 / 264], [是], [17.0×], [1.0×], [仅换轴即可降至 550 自旋内],
      [`Active A-X Model 3A-HOBO`], [Full `08` \ Blind `12`], [24/18/43 \ 17/11/29], [是], [147.3× \ 241.1×], [8.67× \ 14.18×], [活性剪枝进一步压缩辅助比特],
      [`Active (A,X)-B Model 3A-Comp`], [Full `08` \ Blind `12`], [25/0/26 \ 17/0/18], [是], [$infinity$ (0 aux) \ $infinity$ (0 aux)], [$infinity$ (0 aux) \ $infinity$ (0 aux)], [复合配对彻底消除辅助比特],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [四行自旋压缩归因表：剥离换轴降维（$17.0 times$）、子空间剪枝与复合配对零辅助贡献。],
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
      [*ANOVA 矩阵*], [*子空间*], [*L1: 浮点拟合*], [*L2: 浮点->8bit*], [*能级*], [*基态/Regret*], [*L3: 50 次 SA / Tabu*],
      table.hline(stroke: 0.6pt),
      [`Q3_full_08_Model3B_ANOVA`], [588 合法], [$R^2=0.4559$ \ $rho=0.3518$], [$rho(hat(V), "int")=0.2070$], [6], [3 / 0.005211], [SA 100% (0.000) / Tabu 100% (0.0052)],
      [`Q3_blind_12_Model3B_ANOVA`], [108 合法], [$R^2=0.4902$ \ $rho=0.4783$], [$rho(hat(V), "int")=0.4512$], [6], [7 / 0.000000], [SA 100% (0.000) / Tabu 100% (0.0052)],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [`Model 3B-ANOVA` 三层误差分解与 6 个可区分合法能级（`08`: `3, 1, 24, 59, 459, 42`；`12`: `7, 9, 28, 50, 12, 2`）。],
) <tab:anova>

== 同预算经典求解器对比与 Q3 Blind Top-15
@tab:solvers 与 @fig:q3 展示同预算（$50$ 次独立运行）SA 与 Tabu 性能，@tab:q3top15 列出 Q3 Blind 跨阴离子 Top-15 清单。

#figure(
  image("../v2_optimized/figures/fig05_q3_qubit_compression_and_solvers.png", width: 98%),
  caption: [问题三结果：(a) 从 2,760 到 18 自旋的四阶梯压缩；(b) 同预算（50 次）经典求解可行率与 Top-15 捕获；(c) ANOVA 三层相关性；(d) Q3 Blind Top-15 五项分解。],
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
      [*池与编码*], [*自旋*], [*SA 可行率*], [*SA* \ *唯一*], [*SA Regret*], [*SA* \ *Top-K*], [*Tabu* \ *可行*], [*Tabu* \ *唯一*], [*Tabu Regret / Top-K*],
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
  caption: [同预算 50 次经典求解对比表（统一表 4a：SA `size_limit=15` vs. Tabu `size_limit=10`）。],
) <tab:solvers>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    align: (center, left, center, right, right, right, right, right, right, right),
    inset: 2.6pt,
    table.hline(stroke: 1.1pt),
    [*排名*], [*化学式*], [*$X$*], [*$hat(V)$*], [*$hat(P)_("hit")$*], [*$hat(E)_g$*], [*$E_c$*], [*$hat(v)_("form")$*], [*$hat(v)_("gl")+hat(v)_("gh")$*], [*MC 稳定率 ($0.5 sigma$)*],
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
  caption: [Q3 Blind 池（`12`, $1,587$ 个合法组合）Top-15 多阴离子候选与 $0.5 sigma$ 扰动 Top-15 纳入率（`MC 稳定率`）。],
) <tab:q3top15>

= 真实 `SPQC-550` 光量子真机核验

== 硬件编译设计、代数 Ising 恢复与全记录三门禁核验
我们汇总本研究的实证编译对照观察，而不宣称普适硬件缩放定理：

#theorem-box("实证硬件编译观察（紧致活性子空间 vs. 历史 K-拷贝拆分）", [
  在 8-bit `SPQC-550` CIM（$|H_(u v)| <= 127, N <= 550$）上，早期全空间实现需依赖参考自旋 $K$-拷贝拆分维持 $N=52$ 元素（Track A 达 $224$--$468$ 自旋）；相比之下，活性子空间编码将四组盲池问题均压缩至 $12$--$29$ 自旋与单参考自旋（$K=1$），在 `OPTIMIZATION` 模式下获得 $20.0%$--$90.0%$ 严格可行返回记录。
])

`v2_spqc550_hardware.py --verify-only` 从保存原始自旋出发执行独立代数 QUBO-to-Ising 恢复：
$ Q_(i j) = -4 H_(i j) quad (i != j), &quad Q_(i i) = -4 H_(i,"ref") - sum_(j != i) Q_(i j), \
  x^top Q x = -s^top H s + C, &quad C = (1)/(4)(sum_(i,j) Q_(i j) + op("tr") Q), $ <eq:ising-recovery>
强制校验 `int64` 矩阵 SHA-256、`.npy` 文件/数组 SHA-256、有限浮点及逐条记录全字段一致（含 `kAX`）。所有返回记录需过三门禁：(i) 独热约束（$k_A=k_B=k_X=1$）；(ii) 申报池内且 $A != B$；(iii) 辅助一致性 $y_(i k) = a_i c_k$。同时明确区分 `cloud_roundtrip_wall_sec`（含网络队列及 Kaiwu SDK 以秒为单位的 `interval`，请求 1s 实为 SDK 最小 5s）与真实光脉冲退火时间，并说明 `OPTIMIZATION` 返回按能量排序的优化候选集而非独立同分布采样。

== `v3` `SPQC-550` 实测结果对比同实例经典基线
@tab:spqc 与 @fig:spqc 汇总 4 组修复后 `v3` 真机实测（全 40 条返回记录见附录 B）：
- *`v3_q2_blind_12`（12 自旋，$E^*=-1000$，$C=824$，`17.24 s`）：* 30.0% 严格可行记录（3/10 唯一命中，0 重复），触达基态 $-1000.0$（SrTaO2N, $hat(V)=0.257617$, regret `0.005211`），外加 CaNbO2N 与 YNbO2N。
- *`v3_q3_comp_18`（18 自旋，0 辅助，$E^*=-1468$，$C=1364$，`16.23 s`）：* 该无辅助复合实例当前批次取得 *90.0%* 严格可行返回比例（9/10 唯一 Top-15 命中，0 重复），记录 `0..4` 触达基态 $-1468.0$ 并零遗憾击中全场第 1 名 *LaTiO2N*（$hat(V)=0.252406$，regret $0.000000$）。
- *`v3_q3_hobo_29`（29 自旋，11 辅助，$E^*=-1662$，$C=1482$，`19.06 s`）：* 命中基态 $-1662.0$，主变量可行 90.0%（9/10），但 7 条激发记录出现 $1$--$3$ 处辅助不一致，严格可行 20.0%（SrTaO2N、CaTaO2N），客观体现去除辅助比特对当前批次真机可行率的现实意义（而不构成本批次单次因果定理）。
- *`v3_q3_anova_25`（25 自旋，0 辅助，$E^*=-1992$，$C=1872$，`16.43 s`）：* 取得 30.0% 严格可行记录（3/10 均在 $-1992.0$）。
- *Track A 历史归档参考（`table4_spqc550_hardware_vs_classical.csv`）：* 6 组未认证历史全空间任务总体可行率为 3/53 $= 5.66%$，因代理、池与编码均有跨版本变更，仅作历史背景呈报。

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (2.20cm, 1.30cm, 1.50cm, 2.05cm, 1.00cm, 1.15cm, 1.40cm, 1.35cm, 3.30cm, 1.75cm),
      stroke: none,
      align: (left, center, right, left, right, center, right, center, left, right),
      inset: (x: 2.2pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*任务 ID*], [*自旋 (Aux)*], [*`cert_min_E`*], [*求解器 (记录/次)*], [*主可行*], [*`y=a*c`*], [*严格可行*], [*唯一 / 重复*], [*最佳化学式 (`V_hat` / Regret)*], [*耗时 (s)*],
      table.hline(stroke: 0.6pt),
      [`v3_q2_blind_12`], [12 (0)], [-1000], [`SPQC-550` (10)], [30.0%], [10/10], [*30.0%*], [3 / 0], [SrTaO2N (0.2576 / 0.0052)], [17.24],
      [`v3_q2_blind_12`], [12 (0)], [-1000], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [4/46 ; 4/26], [LaTiO2N (0.2524 / 0.0000)], [3.33 / 3.51],
      [`v3_q3_comp_18`], [18 (0)], [-1468], [`SPQC-550` (10)], [90.0%], [10/10], [*90.0%*], [9 / 0], [*LaTiO2N (0.2524 / 0.0000)*], [16.23],
      [`v3_q3_comp_18`], [18 (0)], [-1468], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [7/43 ; 2/28], [LaTiO2N (0.0) / SrTaO2N], [3.25 / 3.70],
      [`v3_q3_hobo_29`], [29 (11)], [-1662], [`SPQC-550` (10)], [90.0%], [3/10], [*20.0%*], [2 / 0], [SrTaO2N (0.2576 / 0.0052)], [19.06],
      [`v3_q3_hobo_29`], [29 (11)], [-1662], [SA (50) \ Tabu (30)], [100%], [48 / 30], [96% / 100%], [13/35 ; 7/23], [LaTiO2N (0.2524 / 0.0000)], [3.31 / 3.16],
      [`v3_q3_anova_25`], [25 (0)], [-1992], [`SPQC-550` (10)], [30.0%], [10/10], [*30.0%*], [3 / 0], [SrTaO2N (0.2576 / 0.0052)], [16.43],
      [`v3_q3_anova_25`], [25 (0)], [-1992], [SA (50) \ Tabu (30)], [100%], [50 / 30], [100%], [8/42 ; 2/28], [SrTaO2N / LaTiO2N (0.0)], [3.29 / 3.57],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [`SPQC-550` 真机实测与同实例经典对照表（统一表 4b）：全部 4 组 `v3` 任务均击中认证基态 `cert_min_ising_energy`，真机耗时为云端往返 Wall Time。],
) <tab:spqc>

#figure(
  image("../v2_optimized/figures/fig07_spqc550_real_hardware_comparison.png", width: 98%),
  caption: [真实 `SPQC-550` 真机验证：(a) Track A 历史归档（0%--11%，合计 5.66%）对比 `v3` 严格可行记录比例；(b) 逐条返回记录相对认证基态的能量残差（10 条/任务）；(c) 硬件自旋数 vs. 可行率全景对比。],
) <fig:spqc>

= 多轴敏感性分析与方法局限边界

== 四轴敏感性实验与真实构建器四域 $C_("cap")$ 留存审计
@tab:sens 与 @fig:sens 给出四维独立压力测试：
1. *独立 4–12 Bit 位宽扫描（`Q2_blind_11`）：* $B=7, 8, 10, 12$ 对应 $M_(max) = 2, 4, 17, 71$ 均满足精确整数无漂移（$0$ 非法基态、SA 100% 可行）；$B <= 6$ 触发 Kaiwu 重缩放回退，$B=4$ 出现 $5$ 个非法基态（SA 可行率 42.0%）。
2. *固定 $M_(max)=4$ 罚系数相变扫描（10 点）：* $alpha_P <= 1.00$ 零合法基态；$alpha_P = 1.25$ 边界处 4 合法与 1 非法态平局（SA 84.0%）；$alpha_P in [1.40, 3.00]$ 全部实现 4 合法 / 0 非法基态与 100.0% SA 可行率。
3. *构建器四域 $C_("cap")$ Top-10 / Top-25 留存审计：* 直接从四套构建器 `metadata` 提取活性集合：`Q2_full_07`（$C_("cap")=2.0$）10/10 与 20/25 `<cap`（24/25 元素积）；`Q2_blind_11`（$C_("cap")=2.5$）9/10 与 9/25（截除 `#10` CdTaO2N）；`Q3_full_08`（$C_("cap")=1.25$）四域全 25/25；`Q3_blind_12`（默认 $C_("cap")=1.50$）保留 10/10 与 15/25 `<cap`、*17/25* 于 `Model 3A-HOBO` $A times B times X$ 域（8 个在外）、*15/25* 于 `Model 3A-Comp` 复合域（10 个在外），以及 *24/25* 合法候选于 `Model 3B-ANOVA` Top-25 域（仅排除 $A=B$ 的 `#22` SnSnO3）。当 `Q3_blind_12` 取 `cap=2.5` 时，真实构建器计数 $21$ 条 `<2.5` 活跃行（含 SnSnO3 激活 `Sn`/`O3`），从而合法 SnGeO3（$2.5236$）进入 HOBO（23/25）与 Comp（21/25）真实编码域。
4. *多尺度（$0.5$--$2.0 sigma$）蒙特卡洛扰动稳定性（500 次）：* 独立高斯下 Q2 Blind Top-10 由 $95.70%$ 降至 $83.28%$，Q3 Top-15 由 $96.24%$ 降至 $61.01%$；引入 `(A,B)` 组相关 + `p_semi` logit 扰动后分别为 $95.46% -> 83.10%$ 与 $94.96% -> 58.84%$。

#figure(
  image("../v2_optimized/figures/fig06_sensitivity_and_mc_robustness.png", width: 98%),
  caption: [四轴敏感性套件：(a) 固定 $M_(max)=4$ 罚系数相变；(b) $C_("cap")$ 帕累托留存与自旋数；(c)--(d) 独立高斯与组相关+logit 多尺度排名稳定性。],
) <fig:sens>

#figure(
  table(
    columns: (auto, 1.3fr, 1.3fr),
    stroke: none,
    align: (left, left, left),
    inset: 2.8pt,
    table.hline(stroke: 1.1pt),
    [*敏感性维度*], [*扫描 / 配置范围*], [*实测核验结论*],
    table.hline(stroke: 0.6pt),
    [位宽扫描 `Q2_blind_11`], [`bw=4,5,6` 回退 vs. `7,8,10,12` 精确], [`bw=4`: 5 非法 GS, SA 42%；`bw=7..12` ($M_(max)=2,4,17,71$): 0 非法 GS, SA 100%, 4--9/10 命中],
    [罚系数 `Q2_blind_11` (`M_max=4`)], [`alpha_P in [0.50 .. 3.00]` ($10$ 点)], [`0.50, 1.00`: 0 合法；`1.25`: 4 合法 / 1 非法（margin=0.0, SA 84%）；`1.40..3.00`: 4 合法 / 0 非法（SA 100%）],
    [$C_("cap")$ Top-10 / Top-25 留存], [`Q2_full` (2.0), `Q2_blind` (2.5), `Q3_full` (1.25), `Q3_blind` (1.50)], [`Q2_full`: 10/10, 20/25；`Q2_blind`: 9/10, 9/25；`Q3_full`: 四域全 25/25；`Q3_blind`: 15/25 (`<cap`) / 17/25 (HOBO) / 15/25 (Comp) / 24/25 (ANOVA)；`cap=2.5` 为 23/25 HOBO, 21/25 Comp],
    [MC 排名稳定性 (500 次)], [`0.5..2.0 sigma` 独立 vs. 组相关+Logit], [独立 Q2 Top-10: `95.70% -> 83.28%`, Q3 Top-15: `96.24% -> 61.01%`；组相关+Logit: Q2 `95.46% -> 83.10%`, Q3 `94.96% -> 58.84%`],
    table.hline(stroke: 1.1pt),
  ),
  caption: [敏感性与 $C_("cap")$ 留存统一汇总表（统一表 5）。],
) <tab:sens>

== 局限性与适用边界
1. *盲测 `04` 无真值标签与氟氧化物稀疏性：* 在未见真实标签下，$sum hat(P)_("hit") = 10.3358$ 与 $C_("cap")$ 留存率均不构成对 19 个隐藏正例真召回率的证明；`02+03` 中低 $V$ 氟氧化物（`FO2`）样本稀疏，使其模型预期质量偏保守（$0.77$ vs. $6$ 个隐藏目标）。
2. *分布假设与保形外推边界：* @eq:vbayes 与 @eq:phit 采用未截断高斯近似（$cal(N)(hat(mu)_g, hat(sigma)_g^2)$ 在 `04` 上存在负带隙尾部中位数 $0.1743$，Top-50 最大 $0.0110$）及条件独立假设；`03_val` 半导体保形半径（$12.375$）不能作为 `02+03` 重训结果的有限样本覆盖保证。
3. *截断依赖性与硬件返回语义：* $C_("cap")$ 剪枝仅相对代理分 $hat(V)$ 无损，会显式舍弃如 CdTaO2N（$3.6227$）的高分候选；`SPQC-550` 报告为 `OPTIMIZATION`（$10$ 条/任务）排序返回记录池，耗时为云端全流程墙上时间（`16.23`--`19.06 s`，有效轮询间隔 5s）。

= 结论与参考文献

== 结论
本文针对 `2026 APMCM Problem D` 构建并严格核验了从代理学习、8-bit Ising 编译认证到 `SPQC-550` 光量子真机实测的全流程方案：
1. *数据集与代理建模严谨性：* 全部 $9$ 个官方数据集通过 CRLF 归一化 SHA-256 校验（9/9；Git LF 原始字节 1/9）。将基础特征与四家族预测器隔离于外层 `(A,B)` 折内（基路径扰动差 $1.42 times 10^(-14)$）配合 pooled-OOF 元层拟合，`v2_PhysHurdle_Bayes` 结合数据内 Butler--Ginley 关系（$R^2=1.0000000000$，显式 $sigma_(E_g) slash 2$ 传播）与 Stage-0 特征在 `03_val` 上取得 $"RMSE"=4.0634$、$"MAE"(V<1)=2.9280$ 及 $100.0%$ Top-75 召回（$V<0.5$）。此外，一套独立的轻量在线 Ridge--LightGBM hurdle 代理在 20 种子闭环基准中经混合 Top-20 + 8-bit QUBO 采集命中 `16.20/18`（$"SD"=1.077$），对比顺序排序 `16.55/18`（$"SD"=1.962$），该实验不宣称 QUBO 采集的平均命中数优势。
2. *可认证 8-Bit QUBO/HOBO 编译：* 全部 $8$ 个 Q2/Q3 矩阵（$12$--$43$ 自旋）经代数恢复（@eq:ising-recovery）、含 $H_("mask")(z)$ 的主变量精确穷举及独立 HiGHS MILP 交叉核验（`mip_gap = 0`），均实现零非法全局基态与零基态辅助歧义；7/8 矩阵整数基态集合包含连续代理最优 LaTiO2N（$hat(V)=0.252406$），修复版 `Model 3B-ANOVA` 呈 $6$ 个合法整数能级。
3. *真实 `SPQC-550` 硬件核验：* 4 组盲池矩阵均通过位级原始自旋与全字段复核，取得 *$20.0%$--$90.0%$* 严格可行返回记录比例（`v3_q3_comp_18` 以零遗憾击中 LaTiO2N 并捕获 9/15 个 Top-15 唯一候选）。

== 参考文献
#set par(hanging-indent: 1.8em, first-line-indent: 0pt)

[1] Castelli I E, Landis D D, Thygesen K S, et al., New cubic perovskites for one- and two-photon water splitting using the computational materials repository[J], *Energy & Environmental Science*, 2012, 5(10): 9034--9043. \
[2] QBoson Inc., *Kaiwu SDK 1.4.4 Documentation: Parameter Precision, QUBO/Ising Conversion, and SPQC-550 CIM Optimizer*, #link("https://kaiwu-sdk-docs.qboson.com/"), 2026. \
[3] Honjo T, Sonobe T, Inaba K, et al., 100,000-spin coherent Ising machine[J], *Science Advances*, 2021, 7(40): eabh0952. \
[4] Butler M A, Ginley D S, Prediction of flatband potentials at semiconductor-electrolyte interfaces from atomic electronegativities[J], *Journal of the Electrochemical Society*, 1978, 125(2): 228--232. \
[5] Shannon R D, Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides[J], *Acta Crystallographica Section A*, 1976, 32(5): 751--767. \
[6] Rumble J R, *CRC Handbook of Chemistry and Physics* (102nd ed.)[M], Boca Raton: CRC Press, 2021. \
[7] Bartel C J, Sutton C, Goldsmith B R, et al., New tolerance factor to predict the stability of perovskite oxides and halides[J], *Science Advances*, 2019, 5(2): eaav0693. \
[8] Rosenberg I G, Reduction of bivalent maximization to the quadratic case[J], *Cahiers du Centre d'Études de Recherche Opérationnelle*, 1975, 17(1): 71--74. \
[9] Freedman D, Drineas P, Energy minimization via graph cuts: Settling what is possible[C] / *Proc. IEEE CVPR*, 2005, 2: 939--946. \
[10] Kitai K, Guo J, Ju S, et al., Designing metamaterials with quantum annealing and factorization machines[J], *Physical Review Research*, 2020, 2(1): 013319. \
[11] Romano Y, Patterson E, Candès E J, Conformalized quantile regression[C] / *Advances in Neural Information Processing Systems (NeurIPS)*, 2019, 32: 3543--3553. \
[12] Vovk V, Gammerman A, Shafer G, *Algorithmic Learning in a Random World* (2nd ed.)[M], Cham: Springer, 2022. \
[13] Google, Gemini 2.5 Pro, Google LLC, 2026-11-26--2026-11-27. \
[14] DeepSeek, DeepSeek-R1-0528, 深度求索 (DeepSeek), 2026-11-26--2026-11-27. \
[15] OpenAI, ChatGPT（网页端论文、数学论证与验证器审阅会话，未指定固定子版本号）, OpenAI, 2026-11-26--2026-11-27.

// =============================================================================
// 附录 A & B（审阅版附硬件全记录与 SHA-256，完整源码附于英文正稿附录 C 与支撑包）
// =============================================================================
#pagebreak()
#set par(hanging-indent: 0pt)
#set heading(numbering: none)

= 附录（审阅版含软件复现命令、AI 披露与完整硬件日志）

== 附录 A：软件环境、独立复现入口与 AI 披露说明
全部计算在 Python 3.12 环境（随机种子 `2026`、`101..120`）下运行，核心依赖包括 `numpy==2.5.5`、`scipy==1.18.1`、`pandas==2.2.3`、`scikit-learn==1.9.1`、`catboost==1.2.10`、`lightgbm==4.7.0`、`xgboost-cpu==3.4.1`、`matplotlib==3.10.3`、`typst==0.15.2`、`pypdf==7.2.0`、`pymupdf==2.2`、`python-docx==1.2.0` 及企业版 `kaiwu==1.4.4`。数据路径由 `common_physics.py::resolve_data_dir()` 动态解析；历史 `M4`/`M5` 与 Track A 对照表已打包入 `v2_optimized/reference_baselines/`。
- *独立支撑压缩包（`Dapmcm26302266fj.zip` + `APMCM_DATA_DIR`，0 云端扣费）：*
  - `python v2_optimized/run_all.py --verify-saved`（数据集校验和、保存结果及 `v2_spqc550_hardware.py --verify-only`）；
  - `python scripts/verify_milp_qubo_crosscheck.py`（SciPy/HiGHS 8 矩阵 MILP 核验，`mip_gap = 0`）；
  - `python scripts/verify_all_review_gates.py --code-only`（全部 7 项代码/数据/矩阵/MILP/图表门禁）；
  - `python v2_optimized/run_all.py --all`（完整离线重算、认证与绘图）。
- *完整仓库与提交前全审：* `python scripts/verify_all_review_gates.py` 与 `python scripts/audit_submission.py --problem D --team apmcm26302266`。

遵循 2026-11-04 竞赛 AI 披露规定，全部建模推导与真机运行由团队执行；AI 助手 `[13–15]` 用于代码管道重构、绘图与 Typst 排版、英文润色及多轮审阅/数学反例/验证器核查（所有采纳的定理修订、MILP 实现与合成负向验证均由团队手工复算验证，详载于独立提交的 `AI工具使用详情.pdf`）。（注：全量 Python 源码见英文正式提交稿附录 C 及 `Dapmcm26302266fj.zip`。）

== 附录 B：完整 40 条 `v3` `SPQC-550` 真机返回日志与 SHA-256 注册表

#figure(
  {
    set text(size: 7.7pt)
    table(
      columns: (2.25cm, 1.05cm, 2.05cm, 2.35cm, 1.35cm, 5.25cm, 1.45cm, 1.25cm),
      stroke: none,
      align: (left, center, right, center, center, left, right, center),
      inset: (x: 2.4pt, y: 2.5pt),
      table.hline(stroke: 1.1pt),
      [*任务 ID*], [*记录*], [*Ising $E(s)$*], [*$(k_A, k_B, k_X)$*], [*`y=a*c`*], [*状态 / 解码化学式*], [*$hat(V)$*], [*池排名*],
      table.hline(stroke: 0.6pt),
      [`v3_q2_blind_12`], [0], [-1000.0], [(1, 1, 1)], [OK], [*FEASIBLE: SrTaO2N*], [0.257617], [2],
      [`v3_q2_blind_12`], [1], [-992.0], [(1, 1, 1)], [OK], [*FEASIBLE: CaNbO2N*], [0.380890], [6],
      [`v3_q2_blind_12`], [2], [-968.0], [(1, 1, 1)], [OK], [*FEASIBLE: YNbO2N*], [12.769992], [64],
      [`v3_q2_blind_12`], [3--9], [-920.0..-896.0], [(2,1,1)/(1,1,1)\* / 0-hot], [OK], [INFEASIBLE (池掩码/独热失败，7条)], [—], [—],
      table.hline(stroke: 0.5pt),
      [`v3_q3_comp_18`], [0], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: SrTaO2N*], [0.257617], [2],
      [`v3_q3_comp_18`], [1], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: SrNbO2N*], [0.414431], [7],
      [`v3_q3_comp_18`], [2], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: LaTiO2N*], [*0.252406*], [*1*],
      [`v3_q3_comp_18`], [3], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: CaTaO2N*], [0.265979], [4],
      [`v3_q3_comp_18`], [4], [-1468.0], [(1, 1, 1)], [OK], [*FEASIBLE: BaTaO2N*], [0.261166], [3],
      [`v3_q3_comp_18`], [5], [-1460.0], [(1, 1, 1)], [OK], [*FEASIBLE: RbNbO3*], [0.837836], [11],
      [`v3_q3_comp_18`], [6], [-1460.0], [(1, 1, 1)], [OK], [*FEASIBLE: CaGeO3*], [0.645777], [8],
      [`v3_q3_comp_18`], [7], [-1452.0], [(1, 1, 1)], [OK], [*FEASIBLE: ScHfO2N*], [0.952559], [12],
      [`v3_q3_comp_18`], [8], [-1452.0], [(1, 1, 1)], [OK], [*FEASIBLE: CaSnO3*], [0.972014], [13],
      [`v3_q3_comp_18`], [9], [-1452.0], [(2, 1, 2)], [OK], [INFEASIBLE (独热/池失败)], [—], [—],
      table.hline(stroke: 0.5pt),
      [`v3_q3_hobo_29`], [0], [-1662.0], [(1, 1, 1)], [OK (0)], [*FEASIBLE: CaTaO2N*], [0.265979], [4],
      [`v3_q3_hobo_29`], [1], [-1662.0], [(1, 1, 1)], [OK (0)], [*FEASIBLE: SrTaO2N*], [0.257617], [2],
      [`v3_q3_hobo_29`], [2, 3], [-1654.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [4], [-1646.0], [(1, 1, 1)], [Fail (2)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [5], [-1638.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(SrTaO2N)`], [0.257617], [2],
      [`v3_q3_hobo_29`], [6], [-1634.0], [(1, 2, 1)], [OK (0)], [INFEASIBLE (独热失败)], [—], [—],
      [`v3_q3_hobo_29`], [7, 8], [-1630.0], [(1, 1, 1)], [Fail (2, 3)], [`AUX_INCONSISTENT(CaTaO2N)`], [0.265979], [4],
      [`v3_q3_hobo_29`], [9], [-1630.0], [(1, 1, 1)], [Fail (1)], [`AUX_INCONSISTENT(CaNbO2N)`], [0.380890], [6],
      table.hline(stroke: 0.5pt),
      [`v3_q3_anova_25`], [0], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: SrTaO2N*], [0.257617], [2],
      [`v3_q3_anova_25`], [1], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: CaTaO2N*], [0.265979], [4],
      [`v3_q3_anova_25`], [2], [-1992.0], [(1, 1, 1)], [OK], [*FEASIBLE: BaTaO2N*], [0.261166], [3],
      [`v3_q3_anova_25`], [3--9], [-1976.0..-1944.0], [(2,1,1)/(1,2,1)/(1,1,0)/0-hot], [OK], [INFEASIBLE (7条)], [—], [—],
      table.hline(stroke: 1.1pt),
    )
  },
  caption: [全部 4 组 `v3` `SPQC-550` 真机返回 40 条记录明细（`*` `v3_q2_blind_12` 第 4、9 条满足独热但不在盲池内）。],
) <tab:records>

#figure(
  table(
    columns: (auto, auto, 1fr),
    stroke: none,
    align: (left, right, left),
    inset: 2.3pt,
    table.hline(stroke: 1.1pt),
    [*产物 / 矩阵 ID*], [*常数 $C$*], [*完整 64 位十六进制 SHA-256 摘要*],
    table.hline(stroke: 0.6pt),
    [`Q2_full_07`], [1390], [`a61bc1d9bd777acdc31c02490c510554dbbcf488d683f50572ba61760fc1aa7e`],
    [`Q2_blind_11` (`v3_q2_blind_12`)], [824], [`361cc4d8ef7d074479506820b0c3e3d90b3170e97ca7c721caea7aa013c02a6b`],
    [`Q3_full_08_Model3A_Comp`], [2038], [`f4bd07e3c83a92023de687e70b95b1c7d81d32084248c12f00ddaf274b9ecdee`],
    [`Q3_full_08_Model3A_HOBO`], [2010], [`fe6d37c13ebf55939abec80dc01f7ba9aaa90be0bc3a37ea7ba729123c56e2a1`],
    [`Q3_full_08_Model3B_CompactANOVA`], [2500], [`7416404244bac9c826c72594f2c937843c1af4d810b9febdb1d94fddf24f6ae2`],
    [`Q3_blind_12_Model3A_Comp` (`v3_q3_comp_18`)], [1364], [`47b9b1f72e78fa9d4dc94ae9a83b6a0067c1f1914b8e9219df6f9b5e233939ae`],
    [`Q3_blind_12_Model3A_HOBO` (`v3_q3_hobo_29`)], [1482], [`302bf51ecc1b9281fea99a77ad038489818d626c5f288e81db1eab9211f55a2a`],
    [`Q3_blind_12_Model3B_CompactANOVA` (`v3_q3_anova_25`)], [1872], [`05ce8d9cfad20be928a3ec0cfd51742f094a311ed64612039681749d9e341f36`],
    [`apmcm26_v2_q2_blind_12` (历史 v2)], [822], [`c5854a58cdff0d9aa66467288c39926ec1ab01b9b4871c5a56d52786a000b7bd`],
    [`apmcm26_v2_q3_anova_24` (历史 v2)], [1228], [`90b005d13fad22f249a3fe629930159f74193b2db6c2d5d9c329e04dcede06d8`],
    [`apmcm26_v2_q3_submod_33` (历史 v2)], [790], [`c50b5bd420b82dc683aa8d076219c0cc09c568ef77089f28e967afb86ee69662`],
    [`v2_val_predictions.csv`], [—], [`ac5d939d4dd63f3aa95d4cde112247b376ea9f3818d766644ba5d9a9b6f3c3fd`],
    [`v2_holdout_04_train02only.csv`], [—], [`2090c512ddc3e61459a81519b956e3e5d0bd1a9145b36e490cabf0cfd5d896c3`],
    [`v2_holdout_04.csv`], [—], [`89eca9b89f8501233356701b2426e3296e41c72f66bf64b518520d8a344742a2`],
    [`v2_q2_full_07.csv`], [—], [`4e2d68eb2626360838985f59e6eb64e3edfd23f9269d81ece3d124737bc2abe7`],
    [`v2_q2_blind_11.csv`], [—], [`3154b240448bde8f2cd12a3cd27b121ac813eb56c85672c8f7337ab7b6412d4c`],
    [`v2_q3_full_08.csv`], [—], [`751f92ff1ab1efb8f6c0f3e94070c2ee72ebcf7fb73211ec7e8576bc96bf35f0`],
    [`v2_q3_blind_12.csv`], [—], [`9de6d43dcf6b1e35032f6d09c9a5de2a5bf16c44ed9e7bb1ff3b7738105004b6`],
    table.hline(stroke: 1.1pt),
  ),
  caption: [全部 8 个认证 `v3` 矩阵、3 个历史 `v2` 矩阵及 7 份 Q1 导出 CSV 的完整 64 位 SHA-256 注册表。],
) <tab:hashes>
