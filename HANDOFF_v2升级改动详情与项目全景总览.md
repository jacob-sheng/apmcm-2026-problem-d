# 2026 APMCM Problem D — v2 深度升级改动详情与项目全景交接文档

> **队伍编号**：`apmcm26302266`（本科生组，D 题）  
> **交接定位**：本仓库原有目录（`01_2026赛题与数据集/.../project/`，提交 `f984b4d`）**未做任何覆盖或修改**；所有 v2 深度升级源码、真实 `SPQC-550` 光量子真机实测结果、8 张 300 DPI 纯英文论文配图、以及编译脱敏完成的 41 页英文终稿论文均独立放置在 `v2_upgrade_package/` 目录中，方便直接审阅、对比、复现或合并提交。  
> **合规承诺**：全套代码、数据表、图表、PDF 元数据均已通过自动化隐私与密钥扫描（**0 密钥/Token、0 个人/学校/路径信息、0 原始官方 CSV 混入、0 外部 DFT 真值泄露**）。

---

## 一、项目目前为止已完成的全景里程碑

截至目前，整个 2026 APMCM Problem D 工程已形成 **“第一性原理物理规律 → 多任务校准贝叶斯代理 → 闭环主动筛选与共形不确定性量化 → 8-bit 硬件精确 QUBO/HOBO 数学理论 → 真实 Bose Quantum `SPQC-550` 双路线光量子实测 → 41 页英文终稿论文 + 支撑材料打包审计”** 的完整闭环：

| 阶段 | 已完成核心工作 | 核心产出文件（位于 `v2_upgrade_package/`） | 当前状态 |
| :--- | :--- | :--- | :---: |
| **0. 数据完整性与物理规律审计** | 校验 9/9 官方文件 SHA-256；发现并证明 **Theorem 0**：精确 Butler–Ginley 对数加性带心律（`03_val` $R^2=\mathbf{1.0000000000}$，$\text{RMSE}=2.90\times 10^{-7}\,\text{eV}$）及严格 $A\leftrightarrow B$ 位点交换对称性（$r(w_A,w_B)=1.00000000$）；导出闭式可行带心窗口 $4.18<E_c<6.05\,\text{eV}$ 与 Bloch–Wilson 电子奇偶律（`Z_parity==0` 100% 捕获 $V<1$）。 | `v2_optimized/common_physics.py`<br>`v2_optimized/v2_deep_physics_conformal.py` | 已完成 |
| **1. 同口径 v1 基线复现** | 依据初期路线图复现 `v1_OneHot_FM_k8`、`v1_Direct_LGBM_V`、`v1_Direct_LGBM_log1pV` 及朴素 104/107 变量 8-bit QUBO，定量暴露 One-Hot FM 低召回（33.3%）、朴素 8-bit 漂移 0% 可行、朴素 Rosenberg 2,760 自旋超限三大瓶颈。 | `v2_optimized/v1_baseline.py`<br>`v2_optimized/results/v1_baseline_metrics.json` | 已完成 |
| **2. 问题一（Q1）v2 代理升级** | 新增 **Stage-0 辅助 DFT 多任务预测器**（严格 5-fold `(A,B)`-GroupKFold OOF：`a_lat` $R^2=0.9626$、`fermi_level` $R^2=0.9381$、磁矩开关 `P(|mu_b|>0.05)` ROC-AUC $=0.8598$）、Bartel $\tau$ 容忍因子、**4 家族异质集成**（CatBoost + LightGBM + XGBoost + ExtraTrees 单纯形加权 + Isotonic 校准）及**闭式高斯铰链期望混合积分**。完成 `03_val` 残差/校准诊断与 `04_test_holdout`（3,703 条）全量打分。 | `v2_optimized/v2_surrogate_q1.py`<br>`v2_optimized/results/v2_q1_diagnostics.json`<br>`v2_optimized/results/v2_holdout_04.csv` | 已完成 |
| **3. 闭环 FMQA 与共形统计分析** | 在 `03_val`（3,969 真实标签）上完成 5 种子闭环主动筛选基准（5% DFT 预算发现 **97.8%** 稀有目标，为传统 FMQA 的 **5.87×**）；完成 200 次置换 **Mondrian 分层共形预测**（89.95% / 95.02% 精准覆盖）与盲测集 Goldilocks 相图/预期收益曲线。 | `v2_optimized/v2_experiments.py`<br>`v2_optimized/v2_deep_physics_conformal.py`<br>`v2_optimized/results/v2_deep_physics_conformal.json` | 已完成 |
| **4. 问题二 & 三（Q2/Q3）8-bit QUBO/HOBO 理论与多求解器基准** | 提出并证明 **Theorems 1–4**（8-bit 漂移定理、全局惩罚下界 $\alpha_P^*=1.50$ 与精确整比例对齐、无损活性子空间剪枝+盲池硬掩码约束、非对称单侧子模 HOBO→QUBO 降阶 126.3×–204.0× 辅变量压缩）；完成 Kaiwu SA vs TabuSearch 双求解器横向评测与四维敏感性/500 次蒙特卡洛鲁棒性检验。 | `v2_optimized/v2_qubo_q2_q3.py`<br>`v2_optimized/results/v2_qubo_results.json`<br>`v2_optimized/results/v2_experiments.json` | 已完成 |
| **5. 真实 `SPQC-550` 光量子真机双轨实测** | 严控消耗 **3 次**云端真机配额，将我方 3 组紧致精确 8-bit 盲池矩阵（12 / 24 / 33 自旋）提交至 **Bose Quantum `SPQC-550`** 真机，独立解码复核全部 30 条原始自旋样本；并与你之前的 6 组全空间 $K$-分裂真机任务组成 **9 任务双轨真机标度律对照（Theorem 5）**。 | `v2_optimized/v2_spqc550_hardware.py`<br>`v2_optimized/results/v2_spqc550_results.json`<br>`v2_optimized/results/spqc_checkpoints/` | 已完成 |
| **6. 论文级配图、终稿 PDF 与支撑包交付** | 生成 **8 张 300 DPI 纯英文高清多联图**（逐张视觉核查无重叠无乱码）；基于 Typst 排版编译 **41 页纯英文终稿论文 `Dapmcm26302266.pdf`**（Page 1 独立 Summary、Page 2 Contents、正文 15 页 $\le 25$ 页、附录含 30 次真机日志与核心源码）及 **`Dapmcm26302266fj.zip`**，通过匿名/元数据/无中文自动化审计。 | `v2_optimized/figures/fig01`–`fig08.png`<br>`paper/main.typ`<br>`submission/Dapmcm26302266.pdf`<br>`submission/Dapmcm26302266fj.zip` | 已完成 |

---

## 二、所有细节改动清单（改了什么 & 为什么这么改）

### 2.1 算法与物理建模层面的关键升级（对比 `v1` 及队友 `f984b4d`）
1. **发现并嵌入精确 Butler–Ginley 几何平均带心律（Theorem 0）**：
   - 队友 `f984b4d` 单独回归 `VBM` 和 `CBM`；我们发现 $\ln\frac{\text{VBM}+\text{CBM}}{2} = w_0+w_A+w_B+w_X$ 在未见 `(A,B)` 对上精度达 **$R^2=1.0000000000$（$\text{RMSE}=2.90\times 10^{-7}\,\text{eV}$）** 且严格满足 $w_A \equiv w_B$。由此消除了 `VBM/CBM` 回归误差，将四维电子约束坍缩为单变量带隙可行区间 $g_{\min}(E_c) \le E_g \le 3.1$。
2. **闭式高斯铰链期望混合积分 + 4 家族异质集成（加入 `ExtraTrees`）**：
   - 相比点预测直接回代公式（存在 Jensen 不等式低估偏差），我们在 `v2_surrogate_q1.py` 中对金属（$E_g=0$）与半导体（$E_g \sim \mathcal{N}(\hat{\mu}_g, \hat{\sigma}_g^2)$）做精确闭式期望积分 $\mathbb{E}[\max(0, \pm(Y-\theta)/0.5)^2]$，并引入非提升随机树 `ExtraTrees` 与 CatBoost/LightGBM/XGBoost 做 OOF 单纯形加权（`ExtraTrees` 在 `P(gap>0)` 中自动获得 48.9% 权重）。
   - **实测效果（`03_val`）**：相比队友最优 `M5_PIMT_Hybrid_Ensemble`（`RMSE=4.209`, `MAE=2.218`, $R^2=0.8532$, $\rho=0.9493$, `p_semi AUC=0.9809`），v2 将指标进一步提升至 **`RMSE=4.0675`（-3.4%）、`MAE=2.0735`（-6.5%）、$R^2=\mathbf{0.8629}$、$\rho=\mathbf{0.9532}$、`p_semi ROC-AUC = 0.9849`、`PR-AUC = 0.8193`、`ECE = 0.00580`**，同时保持 **`Rec@75(V<0.5) = 100.0%`** 与 **`Rec@100(V<1.0) = 100.0%`**。
3. **修复两大底层工程暗坑（已写入论文与代码）**：
   - **坑 1（Kaiwu 1.4.4 SA 输出无序）**：实测确认 `SimulatedAnnealingOptimizer.solve(H)` 返回解池为内部堆序而非 Ising 能量升序（而 `TabuSearchOptimizer` 已排序）。我们在 `common_physics.py` 新增 `sort_ising_solutions()` 按精确能量 $E(s)=-s^\top H s$ 升序重排，统一了 v1/v2 与 SA/Tabu 的基态评价口径。
   - **坑 2（盲测池非盲对硬掩码）**：在 `11`/`12` 盲测池中，属于 train/val 的 `(A,B)` 对若仅置零而不加罚项，可能干扰低能态分布。我们在 `v2_qubo_q2_q3.py` 中对所有非盲池对及 $A=B$ 对在 QUBO 哈密顿量内部直接施加 $+P_{\text{block}} a_i b_j$ 硬掩码，使 Q2 盲测 SA/Tabu 8-bit 可行率双双跃升至 **100.0%**。
4. **双轨真机实测与三体精确降阶的真机证伪（核心亮点）**：
   - 将你的 **Track A（全空间 $K$-副本参考分裂，224–468 自旋，零预筛）** 与我方 **Track B（无损活性子空间精确 8-bit，12–33 自旋）** 统一建模为 **Theorem 5**，并在真实 `SPQC-550` 芯片上跑完 3 组 Track B 任务，获得以下关键对照数据：

### 2.2 真实 Bose Quantum `SPQC-550` 9 任务双轨对照总表

| 路线与 `SPQC-550` 任务名 | 搜索池 | 自旋数 ↓ | 位宽 | **真机可行率 (样本)** ↑ | 真机命中最佳候选 | 池内排名 | 真机耗时 |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| Track A（队友）`q2_uncap` | Q2 Full (`07`) | 105 | 8 | 0.0% (0/10) | INFEASIBLE (8-bit 漂移) | — | 29.4s |
| Track A（队友）`q2_full` ($K=120$) | Q2 Full (`07`) | 224 | 8 | 11.1% (1/9) | `LaTiO2N` | **#1** | 29.1s |
| Track A（队友）`q2_blind` ($K=120$) | Q2 Blind (`11`) | 224 | 8 | **0.0% (0/4)** | INFEASIBLE | — | 21.3s |
| Track A（队友）`q3_qfm108` ($K=150$) | Q3 Full (`08`) | 257 | 8 | 0.0% (0/10) | INFEASIBLE | — | 30.4s |
| Track A（队友）`q3_cmp209` ($K=260$) | Q3 Full (`08`) | 468 | 8 | 10.0% (1/10) | `YTiO2N` | **#2** | 27.8s |
| Track A（队友）`q3_ros264` ($K=150$) | Q3 Full (`08`) | 413 | 8 | 10.0% (1/10) | `AgTaO3` | #21 | 37.8s |
| **Track B（v2）`apmcm26_v2_q2_blind_12`** | **Q2 Blind (`11`)** | **12 (-94.6%)** | **8** | **30.0% (3/10)** | **`SrTaO2N` + `CaTaO2N` + `YTiO2N`** | **#1, #2, #6** | 82.7s |
| Track B（v2）`apmcm26_v2_q3_anova_24` | Q3 Blind (`12`) | 24 (-94.9%) | 8 | 20.0% (2/10) | `LaTiO2F` *(二阶投影伪影)* | #236 | 42.9s |
| **Track B（v2）`apmcm26_v2_q3_submod_33`** | **Q3 Blind (`12`)** | **33 (-92.0%)** | **7** | **40.0% (4/10)** | **`CaTaO2N` (4/4 精确基态!)** | **#2** | **12.1s** |

> **关键科学价值**：
> 1. 紧致编码将真机自旋压缩 **12.5×–18.7×**，把真实 `SPQC-550` 盲测可行率从 **0%–11.1%** 提升至 **30.0%–40.0%**，且双方真机结果共同指向 `SrTaO2N / CaTaO2N / LaTiO2N / YTiO2N` 同一批氮氧化物明星材料，形成极强交叉验证！
> 2. 24 自旋 `ANOVA` 在真机上命中自身基态 $E=-1308.0$ 却解出金属 `LaTiO2F`（因二阶近似 $R^2=0.5044$ 丢掉三体耦合），而 33 自旋 `Submodular HOBO` 无损保留三体交互，在真机上以 **40.0%** 可行率（12.1s）四次精准命中 `#2 CaTaO2N`，用真机实验直接证明了三体精确降阶的必要性。

---

## 三、`v2_upgrade_package/` 文件树与用途速查

```text
v2_upgrade_package/
├── requirements.txt                           # 精确 Python 依赖列表
├── paper/
│   ├── main.typ                               # 41 页英文终稿论文 Typst 源码（含 6 定理 + 8 图 + 7 表 + 附录）
│   └── build.py                               # 一键编译 PDF + 自动抹除 PDF 隐私元数据脚本
├── scripts/
│   ├── audit_submission.py                    # 赛前硬门禁审计脚本（页眉/元数据/中文字符/匿名/原始CSV检查）
│   └── package_submission.py                  # 一键重打包 Dapmcm26302266fj.zip（自动剔除官方原始 CSV）
├── submission/
│   ├── Dapmcm26302266.pdf                     # 编译完成、元数据脱敏、可直接审阅提交的 41 页英文终稿论文
│   └── Dapmcm26302266fj.zip                   # 6.65 MB 完整支撑材料压缩包（已通过 0 错误合规审计）
└── v2_optimized/
    ├── REPORT.md                              # 论文写作详尽数据交接表与逐图解读手册
    ├── common_physics.py                      # 元素物理常量、Bartel tau、Butler-Ginley 带心、能量排序工具
    ├── v1_baseline.py                         # v1 同口径基线复现脚本
    ├── v2_surrogate_q1.py                     # Q1 Stage-0 DFT + 4家族校准贝叶斯代理 + 残差/校准诊断 + 全池打分
    ├── v2_qubo_q2_q3.py                       # Q2/Q3 精确 8-bit 活性子空间 QUBO/HOBO（Model 3A/3B）+ SA vs Tabu
    ├── v2_experiments.py                      # 闭环 FMQA 主动筛选 + 位宽/惩罚/截断敏感性 + 500次蒙特卡洛鲁棒性
    ├── v2_spqc550_hardware.py                 # SPQC-550 真机提交与独立复核脚本（--dry-run / --run-spqc / --verify-only）
    ├── v2_deep_physics_conformal.py           # Theorem 0 对称性审计 + Mondrian 共形预测 + Goldilocks 相图/收益曲线
    ├── generate_figures.py                    # 批量生成 300 DPI 纯英文论文级配图（fig01–fig07）
    ├── run_all.py                             # 一键全流程复现入口
    ├── figures/                               # 8 张 300 DPI 纯英文 PNG 高清配图（fig01–fig08）
    └── results/                               # 所有 JSON 指标、盲测预测 CSV 及 spqc_checkpoints/ 真机原始自旋
```

---

## 四、队友一键复现 & 赛前最终提交 Checklist

### 4.1 本地一键验证（零真机配额消耗）
将官方 `date/` 数据集放在本地工作区后，在 `v2_upgrade_package/` 下可执行：
```bash
# 1. 复现全部本地建模、QUBO/HOBO、闭环 FMQA、敏感性与图表
python v2_optimized/run_all.py
python v2_optimized/v2_deep_physics_conformal.py

# 2. 零配额独立复核已保存的 30 次 SPQC-550 真实光量子硬件采样
python v2_optimized/v2_spqc550_hardware.py --verify-only

# 3. 重新编译论文 PDF（并自动脱敏元数据）与打包支撑材料
python paper/build.py --out submission/Dapmcm26302266.pdf
python scripts/package_submission.py
```

### 4.2 最终上传赛氪前仅剩的 2 项人工动作
1. **签署承诺书**：将全员手签后的承诺书扫描/拍照命名为 **`Dapmcm26302266cns.pdf`**（或 `.jpg`/`.png`）放入 `submission/`，运行 `python scripts/audit_submission.py --problem D --team apmcm26302266` 确认 **0 errors**。
2. **AI 工具使用说明（赛氪 50212 号通知）**：按本仓库 `00_竞赛规则与必读指南/AI工具使用详情说明_记录模板.docx` 如实填写并导出 **`AI工具使用详情.pdf`** 一并提交。
