# 2026 APMCM Problem D 中文精读与建模路线图

- **英文标题**：*Application of Special Purpose Quantum Computer to Rare-Target Search in Perovskite Candidate Materials*
- **中文参考标题**：专用量子计算机在钙钛矿候选材料稀有目标搜索中的应用
- **官方附件与平台**：
  - `2026 APMCM Problem D.pdf`（共 6 页，已生成于当前目录）
  - `date/` 文件夹（10 个文件：`02_castelli_train.csv`、`03_castelli_val.csv`、`04_castelli_test_holdout.csv`、`06_castelli_no_physics.csv`、`07_castelli_q2_pool.csv`、`08_castelli_q3_pool.csv`、`11_castelli_q2_pool_blind.csv`、`12_castelli_q3_pool_blind.csv`、`README_数据集说明.md`、`sha256_public.json`）
  - **玻色量子（Qboson）Kaiwu SDK 文档**：`https://kaiwu-sdk-docs.qboson.com/zh/latest/`
  - **本目录可直接参考的获奖范文**：`03_往年真题与优秀论文拆解/2024_D题优秀论文_云计算资源预测与QUBO量子求解.pdf`

---

## 一、物理背景、数据结构与重大审题陷阱

1. **候选材料空间**：立方钙钛矿 $\text{ABX}_3$ 全因子无缺失网格，共 $52(\text{A}) \times 52(\text{B}) \times 7(\text{X}) = 18,928$ 种候选组合（其中 A 位与 B 位来自同一组 52 种元素，保留 $\text{A}=\text{B}$ 组合；X 位共 7 类：`O3, NO2, FO2, N2O, N3, O2S, FNO`）。
2. **稀有目标与违规度 $V$ 定义**：同时满足四项光电催化物理指标（形成能 `e_form < 0.21 eV`、带隙 `1.4 < Eg < 3.1 eV`、价带顶 `VBM > 5.73 eV`、导带底 `CBM < 4.50 eV`）的完美样本在整个 18,928 空间中仅有 **19 条（阳性率约 0.1004%，极度稀有目标！）**。目标违规度 $V$ 定义为 5 个平方惩罚项之和：
   $$V = v_{\text{form}} + v_{\text{gap\_low}} + v_{\text{gap\_high}} + v_{\text{vbm}} + v_{\text{cbm}}$$
   其中每一项形如 $[\max(0, \text{越界偏差}/0.5)]^2$，$V = 0$ 即完全达标。
3. **重大审题陷阱（题面第 2 页底部黑体 Note）**：
   - **`gap_gllbsc = 0` 表示材料呈金属性（无带隙 Metallicity），绝不是缺失值（Missing Value）！** 全数据集中多达 `18,193` 条记录（占 `96.12%`）该值为 `0`！若在数据清洗时误将其当作缺失值插补或删除，全盘皆输！

---

## 二、三道子问题逐问精读与 QUBO 建模路线

### Question 1：违规度代理模型 $\hat{V}(A, B, X)$ 的构建与盲测集预测
- **题目要求**：
  1. 在 `02_castelli_train.csv`（11,256 行）和 `03_castelli_val.csv`（3,969 行）上严格按公式计算真实目标违规度 $V$；
  2. 构建机器学习代理模型 $\hat{V}(A, B, X)$（注意盲测集仅提供组成元素/非DFT先验特征，切勿使用测试集不存在的 DFT 泄漏列，需对照 `04_castelli_test_holdout.csv` 与 `06_castelli_no_physics.csv` 的字段一致性！）；
  3. 输出对 `04_castelli_test_holdout.csv`（3,703 行）的预测结果并定量评估（同时关注整体回归误差 RMSE/MAE 与低 $V$ 稀有候选区间的排序召回率 Top-$K$ Recall / Spearman $\rho$）。
- **推荐模型**：
  - **两阶段/双模型策略**：
    1. **高精度预测器（用于 Q1 精度冲刺）**：融合元素周期表物化描述符（电负性、原子半径、价电子数、容忍因子 Goldschmidt Tolerance Factor $t$、八面体因子 $\mu$）+ **CatBoost / LightGBM / 深度残差表格网络**（可采用 $\log(1+V)$ 变换或样本加权缓解长尾分布与低 $V$ 稀有区域欠拟合）；
    2. **可精确二阶代数化的可解释代理模型（专为 Q2/Q3 转 QUBO 设计！）**：构建基于 One-hot 编码的**二阶因子分解机（Factorization Machine, FM）/ 成对交互二次多项式回归**：
       $$\hat{V}(\mathbf{z}_A, \mathbf{z}_B, \mathbf{z}_X) = c_0 + \mathbf{h}_A^\top \mathbf{z}_A + \mathbf{h}_B^\top \mathbf{z}_B + \mathbf{h}_X^\top \mathbf{z}_X + \mathbf{z}_A^\top \mathbf{J}_{AB} \mathbf{z}_B + \mathbf{z}_A^\top \mathbf{J}_{AX} \mathbf{z}_X + \mathbf{z}_B^\top \mathbf{z}_X \cdots$$
       因为二元 0-1 变量满足 $z_i^2 = z_i$，**二阶交互模型天然就是精确的二次型 $\mathbf{z}^\top \mathbf{Q}_0 \mathbf{z}$，无需高阶降阶辅助比特，可直接无损写入量子计算机！**

### Question 2：固定 `X = NO2` 下的 `(A, B)` 单点选择 QUBO 建模与求解
- **题目要求**：合并 `02+03` 训练 $\hat{V}$，固定 `X = NO2`，在候选池 `07_castelli_q2_pool.csv`（2,704 行；或盲测池 `11_castelli_q2_pool_blind.csv`，529 行）中选择最优 `(A, B)` 最小化 $\hat{V}(a, b \mid X=\text{NO2})$，必须表述为 **QUBO（Quadratic Unconstrained Binary Optimization）** 形式，满足专用量子计算机（SPQC，如相干伊辛机 CIM）硬件系数比特精度约束，逐项解释目标函数与惩罚项成本。
- **推荐 QUBO 构造**：
  - 设二值独热向量 $\mathbf{a} \in \{0,1\}^{N_A}$、$\mathbf{b} \in \{0,1\}^{N_B}$（对于 $52+52=104$ 个量子比特，远在相干光量子计算机百~千比特容量内！）；
  - 唯一性约束（One-hot Constraint）：$\left(\sum_{i} a_i - 1\right)^2 + \left(\sum_{j} b_j - 1\right)^2$；
  - 若候选池存在禁止组合 $(i,j) \notin \mathcal{P}$，加入禁止对惩罚 $\lambda_{\text{forbid}} \sum_{(i,j)\notin \mathcal{P}} a_i b_j$；
  - 总哈密顿量：
    $$\min_{\mathbf{z}\in\{0,1\}^{N_A+N_B}} H(\mathbf{z}) = \mathbf{a}^\top \mathbf{W} \mathbf{b} + \mathbf{u}^\top \mathbf{a} + \mathbf{v}^\top \mathbf{b} + P_A\left(\sum_i a_i - 1\right)^2 + P_B\left(\sum_j b_j - 1\right)^2$$
  - 使用 `kaiwu` SDK 执行矩阵动态范围缩放与硬件系数精度量化（Precision Reducer / Int8 映射），对比模拟退火（SA）、禁忌搜索与 CIM 求解器结果及精确穷举最优解的一致性！

### Question 3：X 位三选一 + `(A, B)` 双自由度联合 QUBO 优化
- **题目要求**：在 `X ∈ {O3, NO2, FO2}`（3 选 1）与 `(A, B)` 联合空间（`08_castelli_q3_pool.csv` 8,112 行，或 `12_castelli_q3_pool_blind.csv` 1,587 行）中联合搜索最优 `(a, b, x)` 最小化 $\hat{V}(a,b,x)$。
- **推荐 QUBO 构造**：
  - 引入 3 比特 $\mathbf{x} = (x_1, x_2, x_3)^\top \in \{0,1\}^3$，总比特数仅 $52 + 52 + 3 = 107$（或盲测池 $23 + 23 + 3 = 49$ 比特）；
  - 注意若存在三体交叉项 $w_{ijk} a_i b_j x_k$（三次项 HUBO），可通过 **Rosenberg 平方降阶法**（引入辅助二值变量 $y = x_k \cdot a_i$ 或利用 `kaiwu.qubo` 高阶转二阶算子）降为标准二阶 QUBO，并定量讨论辅助比特开销 vs 二阶成对近似 $\mathbf{J}_{AB} + \mathbf{J}_{AX} + \mathbf{J}_{BX}$ 的精度权衡——这一点正是专用量子计算机论文最核心的加分项！
