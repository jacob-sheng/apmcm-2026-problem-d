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
)

#v(0.3em)
== 二、AI 工具辅助环节、关键提示词（Prompts）与人工核验修改记录

#{
  set text(size: 9.2pt)
  set par(leading: 0.55em)
  table(
    columns: (1.05fr, 1.25fr, 1.65fr, 1.55fr, 2.15fr),
    stroke: 0.6pt + luma(140),
    align: (left, left, left, left, left),
    inset: 4.5pt,
    [*对应论文位置*], [*使用阶段与用途*], [*关键提示词（Prompt 摘要）*], [*AI 输出概要*], [*团队人工核验与修改说明*],
    [Section 2.2--2.3],
    [Q1 外层隔离基代理与诊断脚本调试],
    [重构 5-fold (A,B)-GroupKFold 外层隔离特征管道、200 次组不交保形拆分及残差相关性统计脚本],
    [输出 Stage-0 外层隔离框架、+100 基路径扰动隔离自检模板及残差/校准绘图结构],
    [团队独立推导 Butler--Ginley 带心对称关系与 Bayes 混合积分公式，界定外层基模型隔离与 pooled-OOF 元层回评诊断口径，实跑验证基路径 max_pred_delta = 1.42e-14 并逐项核对 03_val 指标 [13]],

    [Section 3--4],
    [Q2/Q3 整数 QUBO/HOBO 认证与误差分解调试],
    [核查 Model 3A-HOBO 累计辅助罚界、Model 3B-ANOVA 统一 8x 缩放及主变量穷举认证逻辑],
    [提供解析 y 消元、分块基数枚举与三层 ANOVA 误差分解汇总表生成代码骨架],
    [团队独立完成累计罚界与代数 QUBO-Ising 恢复公式证明，全量认证 8 个整数 Ising 矩阵（0 个非法基态、0 个基态辅助歧义），修正了 Proposition 2 与 Theorem 2/4 边界条件并逐项核对 regret [13]],

    [Section 5--6],
    [SPQC-550 真机核验与多轴敏感性实验梳理],
    [整理 --verify-only SHA-256 严格核验、4--12 bit 独立重编码及多尺度蒙特卡洛扰动脚本],
    [生成哈希比对、唯一候选/重复记录分离及多尺度稳定率统计输出模板],
    [团队独立执行云端 SPQC-550 OPTIMIZATION 任务，逐条解码并硬比对全部 40 条真机返回自旋记录，修正 Kaiwu interval 单位（秒）与三套子空间 Top-25 保留口径 [14]],

    [Section 1--7 & App.],
    [英文学术表达润色与 Typst 排版规范检查],
    [对照 2026 APMCM 匿名规范与结论边界要求，润色英文摘要、命题前提与 Typst 图表排版],
    [优化英文学术句式、明确经验关系与假设边界，规范统一对比表及示意图版式],
    [团队逐句核查所有数学公式、300 DPI 图表编号（Fig 1--11）、参考文献 [13][14] 及匿名规定，并通过全部自动化门禁、防篡改与独立支持包测试 [13][14]],
  )
}

#v(0.4em)
== 三、正文标注与独立性声明

1. *正文标注对应关系*：论文正文第 2.3 节（Q1 代理管道重构与诊断）、第 4.2 节（穷举认证向量化核查）、第 5.2 节（真机核验脚本审计）及附录 A（英文润色与排版检查）已使用文献上标 `[13]`、`[14]` 明确标注，并与论文参考文献列表 `[13]` Google Gemini 2.5 Pro、`[14]` DeepSeek-R1-0528 一一对应。
2. *独立性声明*：本参赛队（`apmcm26302266`）郑重承诺，论文的核心数学模型构建、算法设计与推导证明、实验设计以及全部科学结论均由队员独立完成；所有 AI 工具仅作为编程辅助与语言检查工具使用，全部输出均经过团队严格的人工复核与独立代码验证，团队对递交材料的真实性、准确性与学术规范性负完全责任。
