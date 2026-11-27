# 2026 APMCM Problem D — v3 Remediated Engineering & Modeling Report

> **Team Control Number**: `apmcm26302266` | **Problem**: D (Application of Special Purpose Quantum Computer to Rare-Target Search in Perovskite Candidate Materials)  
> **Deliverable Root**: `v2_optimized/` | **Bundled Reference Summaries**: `v2_optimized/reference_baselines/`  
> **Data Integrity**: Dual SHA-256 audit (`9/9` CRLF-normalized matches and `1/9` raw-bytes match in Git LF checkout against `sha256_public.json`; all 8 LF CSVs match byte-for-byte after LF->CRLF normalization). Strictly trained and validated ONLY on official `02_castelli_train.csv` and `03_castelli_val.csv`; zero external DFT labels used for `04_castelli_test_holdout.csv` or blind pools `11` / `12`.

---

## 1. 基线对比、口径校准与核心工程修复总览

### 1.1 官方数据拓扑与操作口径（`19` vs. `15` 与 `A=B` 敏感性）
- **全空间划分**：$52 \times 52 \times 7 = 18,928$ 条候选记录中，`gap = 0` 的金属相占 `18,193` 条（`96.12%`），`a_eq_b = 1` 的同核配对占 `364` 条（`1.92%`，无正例）。`02_train` 含 `1,608` 个 `(A,B)` 对（`11,256` 行），`03_val` 含 **`567` 个 `(A,B)` 对（`3,969` 行）**，`04_holdout` 含 **`529` 个 `(A,B)` 对（`3,703` 行）**，三者按有序 `(A,B)` 对严格无重叠。
- **`19` vs. `15` 正例口径界定**：官方评分函数采用非严格分段二次铰链 $V = \sum_{m=1}^5 [\max(0, \delta_m / 0.5)]^2$（边界 $e_{\text{form}} \le 0.21\,\text{eV/atom}$、$1.4 \le E_g \le 3.1\,\text{eV}$、$\text{VBM} \ge 5.73\,\text{eV}$、$\text{CBM} \le 4.50\,\text{eV}$），在此官方 $V=0$ 口径下共有 **`19`** 个隐藏正例（`0.1004%`：`O3`: 6, `NO2`: 6, `FO2`: 6, `N2O`: 1，全部位于 `04_holdout`）；而赛题正文对严格内点不等式（$e_{\text{form}} < 0.21$, $1.4 < E_g < 3.1$）另述 **`15`** 个目标。本工程以官方评分 $V=0$ 为主计算目标，并显式区分严格内点口径而不虚构未见 `04` 标签。
- **`A=B` 保留/排除敏感性**：`02+03` 全训练集保留 `A=B` 化合物，Q2/Q3 QUBO 哈密顿量中主动施加 `+P_block` 硬排除。保留或排除 `A=B` 不改变四池代理最优（`04 (02+03)`、`07`、`08` 最小 $\hat{V}$ 均保持 `LaTiO2N = 0.252406`；`04` Top-20 重合 `20/20`，Top-50 仅第 32 位 `SnSnO3 = 2.4358`）。

### 1.2 核心修复与架构升级（R01–R15 口径校准版）
1. **Q1 外层分组基模型隔离 + Pooled-OOF 元层（[`v2_surrogate_q1.py`](v2_surrogate_q1.py), [`v2_deep_physics_conformal.py`](v2_deep_physics_conformal.py)）**：
   - 将带中心 Ridge 权重、目标编码先验、内层边缘特征、Stage-0 辅助 DFT 预测器（`a_lat`, `fermi_level`, `|mu_b|`, `P(|mu_b|>0.05)`）及 4 家族基模型全部放入 **5-fold `(A,B)`-GroupKFold 外层训练折内部**；自动化 `+100` 基路径扰动自检验证 `fit_fold_isolated_pipeline`（`group_overlap = 0`, `max_feature_delta = 0.0`, **`max_pred_delta = 1.42e-14 < 1e-12`**）。随后的 simplex 权重、残差尺度 $\sigma$ 与 `50/50` Isotonic 校准在汇总的基模型 outer-OOF 上拟合（相应回评及 Stage-0 内层值为内部开发诊断），完整端到端性能在外部公开验证集 `03_val` 上检验。相对第一轮原型，本版 `03_val` 全局 `RMSE` 微降至 `4.0634`（`-0.10%`），`Rec@50(V<0.5)` 由 `15/18` 升至 **`17/18 (94.4%)`**（`Rec@75=18/18`），而近目标 `MAE(V<1)` 同步上升 `+11.9%` 至 `2.9280`（仍远低于直接回归 `5.4475`）。
   - 同协议纳入已随包打包至 `reference_baselines/` 的历史最佳基线 **`M4_PIMT_Physics_Only`** 与 **`M5_PIMT_Hybrid_Ensemble`**，同时导出 `02-only`（`v2_holdout_04_train02only.csv`）与 `02+03 refit`（`v2_holdout_04.csv`）双预测清单（`Spearman = 0.99704`, `Top-10 重合 = 9/10`）。
   - 将带中心 $E_c$ 定性为基于 GLLB-SC 数据的**经验 Butler–Ginley 对数加性关系**（`03_val` 上 $R^2=1.0000000000, \text{RMSE}=2.90\times 10^{-7}\,\text{eV}$），明确指出带隙误差仍以 $\sigma_{\text{VBM}}=\sigma_{\text{CBM}}=\sigma_{E_g}/2$ 传播至带边；电子窗口明确表述为 $[\exists E_g \ge 0:\, g_{\min}(E_c) \le E_g \le 3.1] \iff 4.18 \le E_c \le 6.05\,\text{eV}$。
   - 保形校准采用 **200 次 `(A,B)`-组不交 50/50 拆分**（`group_overlap = 0`），按**预测概率分层** $\hat{p}_{\text{semi}} \ge 0.20$ 划分：**低概率层**（$\hat{p}_{\text{semi}} < 0.20, n=3,808$，含 `29` 个真实半导体，90% 半径 `4.411`）vs **预测半导体/边界层**（$\hat{p}_{\text{semi}} \ge 0.20, n=161$，含 `112` 个真实半导体，90% 半径 `12.375`）；明确 `sum(P_hit) = 10.3358`（Top-30 `8.6001 = 83.2%`、Top-50 `9.4889 = 91.8%`）为**确定性模型概率质量总和，而非未见 `04` 真实正例召回**（`2.30` 仅为独立 Bernoulli 近似下的假设命中数模型内标准差，非 `10.3358` 的置信区间），并报告未截断高斯条件 $E_g$ 在 `04` 上的负尾概率（全池中位数 `0.1743`，Top-50 最大 `0.0110`）及 `03_val` 141 个半导体残差相关性（`r = -0.2930, rho = -0.3349`）。
2. **Q2/Q3 整数 QUBO/HOBO 数学修复与全域基态证书（[`v2_qubo_q2_q3.py`](v2_qubo_q2_q3.py)）**：
   - 修复原 `Model 3A-HOBO` 一侧 gadget 非法基态漏洞：采用累计行和辅助罚项 $P_{\text{aux}}(i,k) > \sum_j |Q_{\text{obj}}(y_{ik}, b_j)|$（`P_aux <= 48/56`）与精确基数分块块罚项 $P_{\text{block}}$，并新增 **零辅助复合配对二部图 QUBO `Model 3A-Comp`**（`26` / `18` 自旋）。Theorem 4 明确 $y_{ik}=a_i c_k$ 对所有主赋值取得极小（`a_i c_k=1` 且激活奖励为 `0` 的合法非基态如 `BaGeO2N` 允许 `y` 零能简并），而全部 8 个实例在**全局基态能级**处均为零辅助歧义。
   - 修复 `Model 3B-ANOVA` 缩放不一致缺陷（对角/非对角统一 `8 * round(...)` 与效果幅值归一化，消除了不同系数尺度差但仍保留量化离散效应），得到 **`6` 个不同合法整数能级**（Full `08` 占有数 `[3, 1, 24, 59, 459, 42]`，Blind `12` `[7, 9, 28, 50, 12, 2]`）并执行三层误差分解。
   - 实现 `certify_global_ground_states()`（解析消去辅助变量 + 代数 QUBO-Ising 恢复核验 + $2^{n_{\text{primary}}}$ 全枚举或零间隙分块基数枚举），对全部 **8 个矩阵** 认证 **`illegal_ground_count = 0`** 与 **`illegal_aux_tie_count = 0`**；7 个矩阵基态集合**包含**连续代理最优 `LaTiO2N`（需配合最差基态 regret 如 Q2 full `0.4934` / Q3 blind ANOVA `0.4518` 解读）。
3. **20 种子闭环混合采集、同预算实验与 `SPQC-550` 真机验证（[`v2_experiments.py`](v2_experiments.py), [`v2_spqc550_hardware.py`](v2_spqc550_hardware.py)）**：
   - 闭环筛选升级至 **20 配对种子**（`N=3,969`，含 `77` 条均 $V \ge 6.6537$ 的 `A=B` 行）：解析超几何基准 `E[H(200)] = 0.9070`（非同核子集为 `0.9250`）；`v1_FM_LBFGS_SeqRank` 记录 `300/300` 次收敛（`v1_FM_LBFGS_QUBO` 调用同优化器，逐轮收敛标志未另行归档）；在相同**轻量在线物理 hurdle 代理**（多物性标签 `e_form, gap, vbm, cbm` + Ridge $E_c$ + LightGBM，固定 $\sigma_{\text{ef}}=0.25, \sigma_g=0.80$）下，古典 Top-20 预筛 + 逐轮 8-bit `Model 3A-Comp` 混合策略 `v2_Real_8Bit_QUBO` 命中 **`16.20/18`（SD `1.077`）**，对比同代理顺序排序 `v2_Classical_SeqRank` **`16.55/18`（SD `1.962`）**，支持混合采集可行性而不宣称采集层或量子优势。
   - 完成全部 4 组修复后 `v3` 真机任务（`TaskMode.OPTIMIZATION`, `10` 条返回记录/任务；Kaiwu `interval` 单位为秒，请求 `1s` 由 SDK 钳制为有效最小 `5s`），`--verify-only` 强制校验矩阵 SHA-256、`.npy` 文件/数组 SHA-256 及逐条解码字段完全一致，并严格保留 **`18 >= 5`** 账号额度。

---

## 2. 统一核心定量对比表（与 `v2_optimized/results/` 完全同步）

### 表 1：Q1 代理模型在 `03_castelli_val.csv`（`567` `(A,B)` 对 / `3,969` 行）上的验证表现

| 模型 ID | RMSE ↓ | MAE ↓ | $R^2$ ↑ | Spearman $\rho$ ↑ | MAE ($V<1$) ↓ | MAE ($V<0.5$) ↓ | Rec@20 ($V<1$) ↑ | Rec@50 ($V<1$) ↑ | Rec@100 ($V<1$) ↑ | EF@50 ($V<1$) ↑ | Rec@50 ($V<0.5$) ↑ | Rec@75 ($V<0.5$) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `v1_OneHot_FM_k8` | 4.9489 | 2.8536 | 0.7970 | 0.9217 | 10.4171 | 9.9826 | 7.41% (2/27) | 29.63% (8/27) | 37.04% (10/27) | 23.52× | 33.33% (6/18) | 33.33% (6/18) |
| `v1_Direct_LGBM_log1pV` | 4.6942 | 2.7469 | 0.8174 | 0.9292 | 5.8008 | 5.2440 | 29.63% (8/27) | 62.96% (17/27) | 74.07% (20/27) | 49.98× | 72.22% (13/18) | 72.22% (13/18) |
| `v1_Direct_LGBM_V` | 4.5568 | 2.5383 | 0.8279 | 0.9365 | 9.9202 | 9.5674 | 29.63% (8/27) | 44.44% (12/27) | 62.96% (17/27) | 35.28× | 50.00% (9/18) | 66.67% (12/18) |
| `M4_PIMT_Physics_Only` *(已归档表1)* | 4.2053 | 2.2068 | 0.8534 | 0.9498 | N/A | N/A | 51.85% (14/27) | 77.78% (21/27) | 100.00% (27/27) | 61.74× | N/A | N/A |
| `M5_PIMT_Hybrid_Ensemble` *(已归档表1)* | 4.2091 | 2.2180 | 0.8532 | 0.9493 | N/A | N/A | 51.85% (14/27) | 77.78% (21/27) | 100.00% (27/27) | 61.74× | N/A | N/A |
| `v2_Direct_4FamPhys` *(外层隔离)* | 4.1834 | 2.2420 | 0.8550 | 0.9472 | 5.4475 | 5.0721 | 37.04% (10/27) | **85.19% (23/27)** | 92.59% (25/27) | **67.62×** | 88.89% (16/18) | 94.44% (17/18) |
| **`v2_PhysHurdle_Bayes` *(外层隔离)*** | **4.0634** | **2.0696** | **0.8632** | **0.9527** | **2.9280** | **2.4126** | **55.56% (15/27)** | 81.48% (22/27) | **100.00% (27/27)** | 64.68× | **94.44% (17/18)** | **100.00% (18/18)** |

> **子模型与诊断指标（`03_val`）**：Stage-0 内折 `a_lat R2=0.9603`, `fermi R2=0.9320`, `|mu_b| R2=0.6133`, `P(|mu_b|>0.05) ROC-AUC=0.8539`；Stage-1/2 `e_form R2=0.9199 (RMSE=0.2108)`, `P(gap>0) ROC-AUC=0.9850 (PR-AUC=0.8247, Brier=0.01334, ECE=0.00570)`, `gap|semi R2=0.7587 (RMSE=0.7922)`；Mondrian 200 次 `(A,B)`-组不交 50/50 保形拆分覆盖率 `90.19% ± 1.65%`（预测分层 $\hat{p}_{\text{semi}}<0.20$ 半径 `4.411` vs $\ge 0.20$ `12.375`）与 `95.12% ± 1.13%`（`7.004` vs `18.014`）。

### 表 2：Q2 & Q3 全部 8 个认证整数 Ising 矩阵保真度与同预算（`50` runs）经典求解表现

> 声明评价池理论代理最优（`07`, `11`, `08`, `12`）：**`LaTiO2N`（$\hat{V} = 0.252406$）**；#2 `SrTaO2N`（`0.257617`）；#3 `BaTaO2N`（`0.261166`）；#4 `CaTaO2N`（`0.265979`）；#5 `YTiO2N`（`0.336278`）。

| 矩阵 ID | 自旋 (`prim+aux+1`) | $M_{\max}$ / $P_{\text{block}}$ | 位宽 / DR / `max\|J\|` | `cert_min_E` / `C` | 合法基态 (退化) | 非法基态 / Aux Ties | 基态最小 / 最差 Regret | SA `50`-Run 可行 / Top-K | Tabu `50`-Run 可行 / Top-K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Q2_full_07` | 17 (`16+0+1`) | 4 / 72.0 | 8 / 7.00 / 125 | `-1558` / `1390` | 10 | **0 / 0** | `0.000000` / `0.493444` | 100.0% / 10/10 | 100.0% / 7/10 |
| `Q2_blind_11` (`v3_q2_blind_12`) | 12 (`11+0+1`) | 4 / 72.0 | 8 / 6.89 / 115 | `-1000` / `824` | 4 | **0 / 0** | `0.000000` / `0.013573` | 100.0% / 4/10 | 100.0% / 3/10 |
| `Q3_full_08_Model3A_Comp` | 26 (`25+0+1`) | 3 / 28.0 | 8 / 6.88 / 116 | `-2110` / `2038` | 10 | **0 / 0** | `0.000000` / `0.320906` | 100.0% / 10/15 | 100.0% / 9/15 |
| `Q3_full_08_Model3A_HOBO` | 43 (`24+18+1`) | 3 / 36.0 (`P_aux<=48`) | 8 / 7.23 / 121 | `-2134` / `2010` | 10 | **0 / 0** | `0.000000` / `0.320906` | 86.0% strict (100% prim) / 13/15 | 100.0% / 10/15 |
| `Q3_full_08_Model3B_CompactANOVA` | 33 (`32+0+1`) | 2 / 28.0 | 8 / 6.88 / 116 | `-2616` / `2500` | 3 (`6` 能级) | **0 / 0** | `0.005211` / `0.013573` | 100.0% / 7/15 | 100.0% / 3/15 |
| `Q3_blind_12_Model3A_Comp` (`v3_q3_comp_18`) | 18 (`17+0+1`) | 4 / 40.0 | 8 / 6.87 / 114 | `-1468` / `1364` | 7 | **0 / 0** | `0.000000` / `0.162025` | 100.0% / 7/15 | 100.0% / 2/15 |
| `Q3_blind_12_Model3A_HOBO` (`v3_q3_hobo_29`) | 29 (`17+11+1`) | 4 / 52.0 (`P_aux<=56`) | 8 / 7.27 / 124 | `-1662` / `1482` | 7 | **0 / 0** | `0.000000` / `0.162025` | 96.0% strict (100% prim) / 14/15 | 100.0% / 7/15 |
| `Q3_blind_12_Model3B_CompactANOVA` (`v3_q3_anova_25`) | 25 (`24+0+1`) | 2 / 32.0 | 8 / 6.83 / 112 | `-1992` / `1872` | 7 (`6` 能级) | **0 / 0** | `0.000000` / `0.451804` | 100.0% / 7/15 | 100.0% / 2/15 |

> **四行压缩归因表（F18）**：(1) 全空间 `A-B` Rosenberg：`2,760` 自旋（`2,652` aux, 超 550 上限）；(2) 全空间 `A-X` Rosenberg：`264` 自旋（`156` aux, 已可上机，`17.0×`）；(3) 活性子空间 `A-X` `Model 3A-HOBO`：`43` (`18` aux, `147.33×` vs A-B / `8.67×` vs full A-X) 与 `29` (`11` aux, `241.09×` / `14.18×`)；(4) 活性子空间复合 `(A,X)-B` `Model 3A-Comp`：`26` / `18` 自旋（`0` aux, 无穷大压缩比）。

### 表 3：真实 `SPQC-550` 光学相干伊辛机（`v3` 4 组任务，`OPTIMIZATION` 10 条返回记录/任务）vs. 同实例经典对照

| 真机任务 ID | 自旋 (`Aux`) | 认证基态 `cert_min_E` | 求解器 (`Rec./Runs`) | 主变量可行率 | `y=a*c` 一致率 | 严格可行记录比例 | 唯一 / 重复命中 | 最佳解码候选 (`V_hat` / `regret`) | Top-K 命中 | 耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `v3_q2_blind_12` | 12 (0) | `-1000` | `SPQC-550` (10) | `30.0%` | `10/10` | **`30.0%`** | `3 / 0` | `SrTaO2N` (`0.257617` / `0.005211`) | `2/10` | `17.24` (cloud) |
| `v3_q2_blind_12` | 12 (0) | `-1000` | `SA` (50) / `Tabu` (30) | `100% / 100%` | `50 / 30` | `100% / 100%` | `4/46`, `4/26` | `LaTiO2N` (`0.252406` / `0.0`) | `4/10 / 4/10` | `3.331 / 3.508` |
| `v3_q3_comp_18` | 18 (0) | `-1468` | `SPQC-550` (10) | `90.0%` | `10/10` | **`90.0%`** | `9 / 0` | **`LaTiO2N` (`0.252406` / `0.0`)** | **`9/15`** | `16.23` (cloud) |
| `v3_q3_comp_18` | 18 (0) | `-1468` | `SA` (50) / `Tabu` (30) | `100% / 100%` | `50 / 30` | `100% / 100%` | `7/43`, `2/28` | `LaTiO2N` (`0.0`), `SrTaO2N` (`0.005211`) | `7/15 / 2/15` | `3.245 / 3.701` |
| `v3_q3_hobo_29` | 29 (11) | `-1662` | `SPQC-550` (10) | `90.0%` | `3/10` | **`20.0%`** | `2 / 0` | `SrTaO2N` (`0.257617` / `0.005211`) | `2/15` | `19.06` (cloud) |
| `v3_q3_hobo_29` | 29 (11) | `-1662` | `SA` (50) / `Tabu` (30) | `100% / 100%` | `48 / 30` | `96.0% / 100%` | `13/35`, `7/23` | `LaTiO2N` (`0.0`) / `LaTiO2N` (`0.0`) | `11/15 / 7/15` | `3.305 / 3.159` |
| `v3_q3_anova_25` | 25 (0) | `-1992` | `SPQC-550` (10) | `30.0%` | `10/10` | **`30.0%`** | `3 / 0` | `SrTaO2N` (`0.257617` / `0.005211`) | `3/15` | `16.43` (cloud) |
| `v3_q3_anova_25` | 25 (0) | `-1992` | `SA` (50) / `Tabu` (30) | `100% / 100%` | `50 / 30` | `100% / 100%` | `8/42`, `2/28` | `SrTaO2N` (`0.005211`) / `LaTiO2N` (`0.0`) | `7/15 / 2/15` | `3.285 / 3.570` |

> **硬件额度与历史归档**：初始 `31` 次，Track A 历史 6 组 CSV 任务消耗 `6` 次（总 `53` 样本仅 `3` 次可行 = `5.66%`），旧 `v2` 归档消耗 `3` 次，`v3` 消耗 `4` 次，**剩余 `18 >= 5` 次**，全部 4 组 `v3` 任务均通过 `v2_spqc550_hardware.py --verify-only` 位级 SHA-256 核验。

### 表 4：20 种子闭环筛选（`03_val`, `K=200`）与四维敏感性汇总

- **20 种子闭环模拟 DFT 筛选**（`N=3,969`, 含 `77` 条 `A=B`, `M=18` for `V<0.5`）：解析超几何基准 `E[H_0.5(200)] = 0.9070`（去 `A=B` 为 `0.9250`）；`Random`: `0.60 ± 0.663 / 18`；`v1_FM_LBFGS_QUBO` (`0` 回退): `1.20 ± 0.980 / 18`；`v1_FM_LBFGS_SeqRank` (`300/300` 收敛): `4.30 ± 2.027 / 18`；`v2_PhysLGBM_ScalarV_SeqRank`: `12.25 ± 3.433 / 18`；同**轻量在线多物性 hurdle 代理**（`e_form, gap, vbm, cbm` + Ridge `E_c` + LightGBM）下，**`v2_Real_8Bit_QUBO`** (Top-20 预筛 + 逐轮 8-bit `Model 3A-Comp` QUBO, `300/300` 精确 8-bit, `0` 回退) 命中 **`16.20 ± 1.077 / 18`**，对比 **`v2_Classical_SeqRank`**: **`16.55 ± 1.962 / 18`**。
- **独立 `4–12` bit 扫描 (`Q2_blind_11`)**：`bw=4..6` 触发 Kaiwu 重缩放回退（`bw=4` 产生 `5` 个非法基态，`42.0%` 可行）；`bw=7` (`M_max=2`)、`bw=8` (`M_max=4`)、`bw=10` (`M_max=17`)、`bw=12` (`M_max=71`) 均通过精确整型路径且 `0` 非法基态、`100.0%` SA 可行率。
- **固定 `M_max=4` 罚系数相变扫描 (`Q2_blind_11`, 10 点)**：`alpha_P <= 1.00` 全域 `0` 合法基态；`alpha_P = 1.25` 处于边界（`4` 合法 / `1` 非法基态, `margin = 0.0`, SA `84.0%`）；`alpha_P in [1.40, 3.00]` 全部达到 `4` 合法 / `0` 非法基态与 `100.0%` SA 可行率（其中 `1.50, 1.75, 2.25` 为精确 8-bit 整数）。
- **`C_cap` Top-10 / Top-25 保留与截断损失**：`Q2_blind_11` (`cap=2.5`) 保留 `9/10` / `9/25`（截去第 10 名 `CdTaO2N = 3.6227` 等）；`Q3_blind_12` (`cap=1.5`) 按 `< cap` 口径保留 `10/10` / `15/25`，按 $A \times B \times X$ 元素积子空间保留 **`17/25`**（**`8`** 个在子空间外：排名 `18–25`），按 `(A,X)-B` 复合子空间保留 **`15/25`**（`10` 个在外，另排除 `#16 BaGeO3 = 1.6630` 与 `#17 BaSnO3 = 1.8225`）。
- **多尺度 (`0.5–2.0 sigma`) 蒙特卡洛排名稳定性 (`500` trials)**：独立高斯模式下 `Q2_blind_11` Top-10 留存率由 `95.70%` 渐变至 `83.28%`，`Q3_blind_12` Top-15 由 `96.24%` 渐变至 `61.01%`；`(A,B)`-组相关 + `p_semi` logit 扰动下分别为 `95.46% -> 83.10%` 与 `94.96% -> 58.84%`。

---

## 3. 11 张 300 DPI 可视化资产清单（`v2_optimized/figures/`）

- **3 张概念架构示意图（`paper/schematics/*.svg` + `v2_optimized/figures/schematic*.png`）**：
  1. `schematic01_pipeline_architecture.png` — 端到端数据核验、外层隔离代理、活性子空间编译、全域证书与真机核验全流程架构图
  2. `schematic02_band_alignment_window.png` — 对称带边耦合（$\sigma_{\text{VBM}}=\sigma_{\text{CBM}}=\sigma_{E_g}/2$）与 $(E_c, E_g)$ 可行窗口存在量词及金属反例示意图
  3. `schematic03_qubo_ising_compilation.png` — Q3 三种活性子空间 QUBO 编码、代数 QUBO-Ising 映射与全域基态认证逻辑图
- **8 张全自动数据驱动定量图（严格由 `results/*.json|csv` 生成，零硬编码数组）**：
  4. `fig01_v1_vs_v2_surrogate_metrics.png` — Q1 基线对比（含 `M4/M5` 与 `MAE(V<1)`/`MAE(V<0.5)`）及 `E_c` 经验带心验证
  5. `fig02_residual_and_calibration.png` — `03_val` 金属/半导体残差分布、带心误差剖面与 10-bin 可靠性校准曲线
  6. `fig03_closed_loop_fmqa_benchmark.png` — 20 种子闭环筛选（含解析超几何基准、`L-BFGS-B` FM 与真实逐轮 8-bit QUBO）
  7. `fig04_q2_v1_vs_v2_hardware_drift.png` — `4–12` bit 独立重编码漂移扫描与 Q2 Blind Top-10 五项物理成本分解
  8. `fig05_q3_qubit_compression_and_solvers.png` — 4 行 Q3 自旋压缩归因阶梯与 Q3 Blind Top-15 跨阴离子分解
  9. `fig06_sensitivity_and_mc_robustness.png` — 固定-$M_{\max}=4$ 罚系数相变、`C_cap` 帕累托保留与多尺度 (`0.5–2.0 sigma`) MC 稳定性
  10. `fig07_spqc550_real_hardware_comparison.png` — 4 组 `v3` `SPQC-550` 任务 vs 同实例 SA/Tabu 对比与认证基态能谱
  11. `fig08_electronic_phase_and_conformal_audit.png` — `04_holdout` 相图、作用域限定 `sum(P_hit)` 曲线与 200 次组不交保形覆盖率

---

## 4. 离线复现入口与交件合规说明

### 4.1 数据路径发现与分阶段 CLI 复现（WSL2 `.venv`）
`common_physics.py::resolve_data_dir()` 按优先级自动解析官方 `date/` 目录：`APMCM_DATA_DIR` 环境变量 → 工作区 `01_2026赛题与数据集/Problem_D_量子计算钙钛矿稀有目标搜索/date/` → 备选相对路径。在仓库根目录运行（全部离线且零云端扣费）：

```bash
# 1. 仅核验数据集 SHA-256、已冻结 JSON/CSV 与 SPQC-550 原始自旋档案（推荐，0 云端调用）
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python v2_upgrade_package/v2_optimized/run_all.py --verify-saved

# 2. 全流程离线重算（Q1 外层隔离训练 -> Q2/Q3 全域认证 -> 20 种子/敏感性 -> 绘图 -> 核验）
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python v2_upgrade_package/v2_optimized/run_all.py --all

# 3. 自动化门禁、论文编译、支撑压缩包与匿名提交审计
python v2_upgrade_package/scripts/verify_all_review_gates.py
python v2_upgrade_package/paper/build.py --out v2_upgrade_package/submission/Dapmcm26302266.pdf
python v2_upgrade_package/scripts/package_submission.py
python v2_upgrade_package/scripts/audit_submission.py --problem D --team apmcm26302266 --allow-missing-cns
```

### 4.2 交件产物与 2026-11-04 AI 使用合规
- **正式提交产物 (`v2_upgrade_package/submission/`)**：
  - `Dapmcm26302266.pdf`（正文含参考文献严格 $\le 25$ 页，第 1 页摘要，第 2 页目录，纯英文且零身份信息/本地路径）；
  - `Dapmcm26302266fj.zip`（含自编代码、`scripts/` 审计门禁、`reference_baselines/` 历史对比表、结果、11 张 PNG 及说明，支持仅配官方 `APMCM_DATA_DIR` 独立完整运行；严格排除官方原始 CSV、`*dryrun*.json`、`*.npz` 与 `.env*`，体积 $< 25\,\text{MB}$）；
  - `AI工具使用详情.pdf` 与 `AI工具使用详情说明_本队草稿.docx`（按 2026-11-04 规定匿名填报并与正文参考文献 `[13]–[14]` 一一对应；已纳入 `audit_submission.py` 默认必检清单，不打入 `fj.zip`）。
  - **交件前最后人工操作**：赛题官方承诺书（`Dapmcm26302266cns.<pdf|jpg|png>`）由全体队员线下签名扫描后上传至竞赛系统，切勿上传至公开 GitHub。
