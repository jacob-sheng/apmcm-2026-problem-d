// =============================================================================
// 2026 APMCM — Problem D Formal AI Usage Disclosure Form (AI工具使用详情.pdf)
// Strictly Anonymous | Team Control Number: apmcm26302266 | Problem Chosen: D
// =============================================================================

#set document(
  title: "",
  author: (),
  keywords: (),
)

#set page(
  paper: "a4",
  margin: (top: 2.0cm, bottom: 2.0cm, left: 2.0cm, right: 2.0cm),
  header: context {
    let cur = counter(page).get().first()
    let tot = counter(page).final().first()
    set text(size: 9pt, font: ("Times New Roman", "SimSun"))
    grid(
      columns: (1fr, auto),
      align: (left, right),
      [*Team Dapmcm26302266 — AI工具使用详情*],
      [Page #cur of #tot],
    )
    v(-5pt)
    line(length: 100%, stroke: 0.5pt + luma(130))
  },
  footer: none,
)

#set text(
  font: ("Times New Roman", "SimSun"),
  size: 10.5pt,
  lang: "zh",
)

#set par(leading: 0.62em, spacing: 0.75em)

#align(center)[
  #text(size: 16pt, weight: "bold", fill: rgb("#1B365D"))[AI工具使用详情说明]
  #v(0.25em)
  #text(size: 10.5pt)[
    *参赛队号（Team Control Number）：*`apmcm26302266` #h(2.0em)
    *所选赛题（Problem Chosen）：*`D`
  ]
]

#v(0.35em)
#line(length: 100%, stroke: 0.8pt + rgb("#1B365D"))

== 一、所用 AI 工具清单与版本信息

#table(
  columns: (auto, auto, 1.3fr, 1.2fr, 1.3fr),
  stroke: 0.6pt + luma(140),
  align: (center, center, left, left, center),
  inset: 5.5pt,
  [*序号*], [*工具名称*], [*具体版本 / 模型型号*], [*开发机构 / 公司*], [*使用日期*],
  [1], [Gemini], [Gemini 2.5 Pro], [Google], [2026-11-26 ～ 2026-11-27],
  [2], [DeepSeek], [DeepSeek-R1-0528], [深度求索（DeepSeek）], [2026-11-26 ～ 2026-11-27],
  [3], [ChatGPT], [ChatGPT 网页端审阅会话（未指定子版本）], [OpenAI], [2026-11-26 ～ 2026-11-27],
)

#v(0.25em)
== 二、AI 工具辅助环节、关键提示词（Prompts）与人工核验修改记录

#{
  set text(size: 8.6pt)
  set par(leading: 0.50em)
  table(
    columns: (0.95fr, 1.15fr, 1.60fr, 1.55fr, 2.25fr),
    stroke: 0.6pt + luma(140),
    align: (left, left, left, left, left),
    inset: 4.0pt,
    [*对应论文位置*], [*使用阶段与用途*], [*关键提示词（Prompt 摘要）*], [*AI 输出概要*], [*团队人工核验与采纳/修改说明*],
    [Section 2.2--2.3],
    [Q1 外层隔离基代理与诊断脚本重构 [13]],
    [重构 5-fold (A,B)-GroupKFold 外层隔离特征管道、200 次组不交保形拆分及残差相关性脚本],
    [输出 Stage-0 外层隔离框架、+100 基路径扰动自检与残差/校准绘图结构],
    [团队独立推导 Butler--Ginley 对称关系与 Bayes 混合积分，严格区分 phase1 (`02->03`) 与 phase2 (`02+03` 无外部带标测试集)，实测 `max_pred_delta = 1.42e-14`],

    [Section 3--4],
    [Q2/Q3 整数 QUBO/HOBO 认证与误差分解 [13]],
    [核查 `Model 3A-HOBO` 累计辅助罚界、`Model 3B-ANOVA` 统一 8x 缩放及穷举认证逻辑],
    [提供解析 $y$ 消元、分块基数枚举与三层 ANOVA 分解代码骨架],
    [团队独立证明累计罚界与代数 QUBO-Ising 恢复公式，全量穷举认证 8 矩阵（`0` 非法基态、`0` 基态辅助歧义）并核对 ANOVA 6 能级占有数],

    [Section 5--6],
    [SPQC-550 真机核验与多轴敏感性整理 [14]],
    [梳理 `--verify-only` SHA-256 严格核验、`4--12` bit 重编码及多尺度蒙特卡洛扰动脚本],
    [生成哈希比对、唯一候选/重复记录分离及多尺度稳定率模板],
    [团队独立执行 SPQC-550 `OPTIMIZATION` 任务并逐条解码全部 40 条返回自旋记录，修正 Kaiwu `interval` 单位（秒）与 4 套构建器域统计],

    [Section 1--7 & App.],
    [多轮成果审阅、数学反例定位与离线数值/MILP 复核建议 [15]],
    [对仓库、真机记录与论文稿开展端到端法医审阅，定位量词/边界错漏、验证器盲区与过强结论],
    [指出 pooled-OOF 口径、`Theorem 4` 消元等式遗漏 $H_"mask"$（反例 `GeGeO3`/`BaHfNO2`）、`cap=2.5` 域分叉（`SnGeO3` `23/21`）、`NaN`/`kAX` 漏检及独立 MILP 交叉核验建议],
    [*采纳*：团队逐条复算反例后在 `Theorem 2/4` 补全 $H_"mask"$ 与 `0`-hot 证明，改用构建器元数据重导 `C_cap` 表，独立编写 `verify_milp_qubo_crosscheck.py` 验证 8 矩阵 HiGHS MILP (`mip_gap=0`) 并加固防篡改合成夹具；*未采纳*：未为已准确收窄的主张追加无谓全量重训或额外付费真机调用],
  )
}

#v(0.3em)
== 三、正文标注与团队责任声明

1. *正文标注对应关系*：论文第 2.3 节（Q1 重构 [13]）、第 4.2 节（穷举向量化 [13] 与 MILP 外部复核建议 [15]）、第 5.2 节（真机核验审计 [14]）及附录 A（排版润色、审阅与交叉核验披露 [13--15]）均与参考文献 `[13]–[15]` 一一对应。
2. *责任与真实性声明*：本参赛队（`apmcm26302266`）确认：所有核心建模思想、物理耦合设计、云端 `SPQC-550` 实测数据以及团队自主运行的整数穷举和 HiGHS MILP 核验结果均为真实客观产物；AI 工具除代码调试与英文润色外，还实际参与了多轮论文与验证器审阅、反例发现及复核建议（[15]），团队对上述所有工具使用及采纳后的人工推导、源码实现与验证结果负全部责任。
