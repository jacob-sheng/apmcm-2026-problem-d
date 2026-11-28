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
  margin: (top: 1.75cm, bottom: 1.65cm, left: 1.85cm, right: 1.85cm),
  header: context {
    let cur = counter(page).get().first()
    let tot = counter(page).final().first()
    set text(size: 8.8pt, font: ("Times New Roman", "SimSun"), fill: luma(85))
    grid(
      columns: (1fr, auto),
      align: (left, right),
      [*Team Dapmcm26302266 — AI工具使用详情*],
      [Page #cur of #tot],
    )
    v(-5pt)
    line(length: 100%, stroke: 0.5pt + rgb("#CBD6E6"))
  },
  footer: none,
)

#set text(
  font: ("Times New Roman", "SimSun"),
  size: 10pt,
  lang: "zh",
  fill: luma(25),
)

#set par(leading: 0.60em, spacing: 0.65em, justify: true)

#show heading.where(level: 2): it => {
  v(0.55em)
  block(
    width: 100%,
    below: 0.45em,
    stroke: (left: 2.8pt + rgb("#1B365D")),
    inset: (left: 7pt, top: 1pt, bottom: 1pt),
  )[
    #text(font: ("Times New Roman", "Microsoft YaHei"), size: 10.8pt, weight: "bold", fill: rgb("#1B365D"))[#it.body]
  ]
}

// -----------------------------------------------------------------------------
// Title Banner & Metadata Bar
// -----------------------------------------------------------------------------
#align(center)[
  #text(font: ("Times New Roman", "Microsoft YaHei"), size: 16pt, weight: "bold", fill: rgb("#1B365D"))[
    AI工具使用详情说明
  ]
  #v(0.15em)
  #text(size: 9pt, fill: luma(95))[
    2026 Asia and Pacific Mathematical Contest in Modeling (APMCM) · 人工智能工具规范使用披露表
  ]
]

#v(0.35em)
#block(
  width: 100%,
  fill: rgb("#F4F7FB"),
  stroke: 0.6pt + rgb("#CBD6E6"),
  inset: (x: 10pt, y: 6.5pt),
  radius: 3pt,
)[
  #set text(size: 9.2pt)
  #grid(
    columns: (auto, 1fr, auto),
    align: (left, center, right),
    [#text(font: ("Times New Roman", "Microsoft YaHei"), weight: "bold", fill: rgb("#1B365D"))[参赛队号（Team Control No.）：]`apmcm26302266`],
    [#text(font: ("Times New Roman", "Microsoft YaHei"), weight: "bold", fill: rgb("#1B365D"))[所选赛题（Problem）：]`D`],
    [#text(font: ("Times New Roman", "Microsoft YaHei"), weight: "bold", fill: rgb("#1B365D"))[正文引用编号：]`[13]`--`[15]`],
  )
]

// -----------------------------------------------------------------------------
// Section 1: AI Tool Inventory
// -----------------------------------------------------------------------------
== 一、所用 AI 工具清单与版本信息

#{
  set text(size: 9.3pt)
  show table.cell.where(y: 0): set text(font: ("Times New Roman", "Microsoft YaHei"), weight: "bold", fill: rgb("#1B365D"))
  table(
    columns: (1.2cm, 2.3cm, 1.35fr, 1.15fr, 1.25fr),
    stroke: (_, y) => (
      top: if y == 0 { 1.1pt + rgb("#1B365D") } else if y == 1 { 0.6pt + rgb("#1B365D") } else { 0.35pt + luma(215) },
      bottom: 0.35pt + luma(215),
    ),
    fill: (_, y) => if y == 0 { rgb("#EDF2F8") } else if calc.even(y) { rgb("#FAFBFD") } else { white },
    align: (center, center, left, left, center),
    inset: (x: 6pt, y: 5pt),
    [*序号*], [*工具名称*], [*具体版本 / 模型型号*], [*开发机构 / 公司*], [*使用日期*],
    [1], [*Gemini*], [Gemini 2.5 Pro], [Google LLC], [2026-11-26 ～ 2026-11-27],
    [2], [*DeepSeek*], [DeepSeek-R1-0528], [深度求索（DeepSeek）], [2026-11-26 ～ 2026-11-27],
    [3], [*ChatGPT*], [ChatGPT (Web Version)], [OpenAI], [2026-11-26 ～ 2026-11-27],
    table.hline(stroke: 1.1pt + rgb("#1B365D")),
  )
}

// -----------------------------------------------------------------------------
// Section 2: Code Collaboration & Assistance Log
// -----------------------------------------------------------------------------
== 二、AI 工具代码协作与辅助环节记录

#{
  set text(size: 8.6pt)
  set par(leading: 0.50em, justify: true)
  show table.cell.where(y: 0): set text(font: ("Times New Roman", "Microsoft YaHei"), weight: "bold", fill: rgb("#1B365D"))
  table(
    columns: (0.95fr, 1.15fr, 1.55fr, 1.50fr, 2.15fr),
    stroke: (_, y) => (
      top: if y == 0 { 1.1pt + rgb("#1B365D") } else if y == 1 { 0.6pt + rgb("#1B365D") } else { 0.35pt + luma(215) },
      bottom: 0.35pt + luma(215),
    ),
    fill: (_, y) => if y == 0 { rgb("#EDF2F8") } else if calc.even(y) { rgb("#FAFBFD") } else { white },
    align: (left, left, left, left, left),
    inset: (x: 4.8pt, y: 4.8pt),
    [*对应论文位置*], [*使用阶段与协作用途*], [*关键提示词（Prompt 摘要）*], [*AI 输出概要*], [*团队人工核验与独立实现说明*],

    [Section 3.2 &\ Appendix A],
    [特征工程管道组织与统计绘图代码协作 `[13]`],
    [协助整理 5 折 $(A,B)$ 阳离子对分组交叉验证管道、保形校准统计及 Matplotlib 多子图学术排版代码],
    [生成模块化交叉验证接口、校准曲线统计与多面板可视化绘图脚手架代码],
    [核心 Butler--Ginley 对数带心定律、两阶段高斯铰链解析积分及 $154$ 维物理特征均由团队独立推导设计，人工逐项核对分布外预测一致性],

    [Section 5.2 &\ Appendix A],
    [QUBO/HOBO 向量化枚举代码协作 `[13]`],
    [协助优化主变量空间 $2^(n_("prim"))$ 向量化穷举与分块基数遍历的 NumPy 矩阵运算实现],
    [提供辅助变量解析消元后的批量矩阵求值与基态统计代码结构],
    [团队独立完成 8-bit 量化漂移定理与紧罚系数界（Theorem 1--4）数学证明，自主运行并核验全部 $8$ 个伊辛矩阵实现零非法全局基态],

    [Section 6.1 &\ Appendix A],
    [真机返回态离线校验与敏感性脚本协作 `[14]`],
    [协助编写 SPQC-550 返回自旋记录的 SHA-256 完整性校验、去重统计及蒙特卡洛扰动批处理脚本],
    [输出离线检查点哈希比对、唯一候选/重复态汇总及多尺度扰动实验脚本框架],
    [团队独立提交玻色量子 SPQC-550 优化任务，依代数双射公式逆向还原 QUBO 并逐条解码核验全部 $40$ 条原始自旋记录],

    [Section 5.2,\ 1--8 & App. A],
    [英文学术表达润色与独立 MILP 交叉验证代码协作 `[15]`],
    [辅助英文长句语法与 Typst 排版润色，核查定理边界表述并协作编写独立 MILP 线性化核验脚本],
    [给出英文学术措辞优化建议、公式边界核对意见及 SciPy/HiGHS 混合整数规划（MILP）验证代码骨架],
    [团队逐句对照原意审校数学符号与实验数值，独立实现并运行 `verify_milp_qubo_crosscheck.py`（$8$ 矩阵 `mip_gap = 0`），与穷举基态能量完全一致],

    table.hline(stroke: 1.1pt + rgb("#1B365D")),
  )
}

// -----------------------------------------------------------------------------
// Section 3: Citation Mapping & Academic Integrity Declaration
// -----------------------------------------------------------------------------
== 三、正文标注映射与学术独立性声明

#block(
  width: 100%,
  fill: rgb("#F7F9FC"),
  stroke: (left: 2.8pt + rgb("#1B365D"), rest: 0.5pt + rgb("#D5DFEC")),
  inset: (x: 10pt, y: 7.5pt),
  radius: (right: 2.5pt),
)[
  #set text(size: 9.2pt)
  #set par(leading: 0.58em, spacing: 0.45em)
  1. *正文引用对应关系*：英文正稿 `Section 3.2`（分组交叉验证管道代码整理 `[13]`）、`Section 5.2`（向量化穷举 `[13]` 与独立 MILP 交叉验证脚本 `[15]`）、`Section 6.1`（真机离线解码校验脚本 `[14]`）及 `Appendix A`（代码重构、绘图/Typst 排版与语言润色汇总说明 `[13–15]`）均已明确标注，并与文末 `References [13]–[15]` 严格一一对应。
  2. *独立性与真实性声明*：本参赛队（`apmcm26302266`）郑重承诺：论文的全部核心数学模型构建、物理机理推导、QUBO/HOBO 编译算法设计、玻色量子 `SPQC-550` 真机实验及结果分析均由参赛队员独立完成；AI 工具仅用于上述 Python 代码工程协作、可视化与排版辅助、独立核验脚本编写及英文语法润色。团队已对所有辅助代码与文字逐行人工复核验证，并对提交作品的真实性、准确性与学术规范性负全部责任。
]
