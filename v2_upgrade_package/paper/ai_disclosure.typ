// 2026 APMCM Problem D — AI Usage Disclosure (AI工具使用详情.pdf)

#set document(
  title: "",
  author: (),
  keywords: (),
)

#set page(
  paper: "a4",
  margin: (top: 2.0cm, bottom: 2.0cm, left: 2.1cm, right: 2.1cm),
  header: context {
    let cur = counter(page).get().first()
    let tot = counter(page).final().first()
    set text(size: 9pt, font: ("Times New Roman", "SimSun"))
    grid(
      columns: (1fr, auto),
      align: (left, right),
      [Team Dapmcm26302266],
      [Page #cur of #tot],
    )
    v(-5pt)
    line(length: 100%, stroke: 0.5pt + black)
  },
  footer: none,
)

#set text(
  font: ("Times New Roman", "SimSun"),
  size: 10.5pt,
  lang: "zh",
  fill: black,
)

#set par(leading: 0.65em, spacing: 0.7em, justify: true)

#show heading.where(level: 2): it => {
  v(0.75em)
  block(below: 0.45em)[
    #text(font: ("Times New Roman", "Microsoft YaHei"), size: 11pt, weight: "bold")[#it.body]
  ]
}

#align(center)[
  #text(font: ("Times New Roman", "Microsoft YaHei"), size: 15pt, weight: "bold")[AI工具使用详情说明]
  #v(0.35em)
  #text(size: 10.5pt)[
    参赛队号（Team Control Number）：apmcm26302266 #h(2.2em) 所选赛题（Problem Chosen）：D
  ]
]

#v(0.25em)
#line(length: 100%, stroke: 0.6pt + black)

== 一、所用 AI 工具基本信息

#{
  set text(size: 9.8pt)
  show table.cell.where(y: 0): set text(font: ("Times New Roman", "Microsoft YaHei"), weight: "bold")
  table(
    columns: (1.2cm, 2.3cm, 1.35fr, 1.15fr, 1.25fr),
    stroke: none,
    align: (center, center, left, left, center),
    inset: (x: 5pt, y: 5pt),
    table.hline(stroke: 1.0pt),
    [序号], [工具名称], [模型版本], [开发机构], [使用日期],
    table.hline(stroke: 0.5pt),
    [1], [Gemini], [Gemini 2.5 Pro], [Google], [2026-11-26 ～ 2026-11-27],
    [2], [DeepSeek], [DeepSeek-R1-0528], [深度求索（DeepSeek）], [2026-11-26 ～ 2026-11-27],
    [3], [ChatGPT], [ChatGPT (Web Version)], [OpenAI], [2026-11-26 ～ 2026-11-27],
    table.hline(stroke: 1.0pt),
  )
}

== 二、具体辅助环节与人工核对记录

#{
  set text(size: 9pt)
  set par(leading: 0.52em, justify: true)
  show table.cell.where(y: 0): set text(font: ("Times New Roman", "Microsoft YaHei"), weight: "bold")
  table(
    columns: (1.0fr, 1.15fr, 1.55fr, 1.45fr, 1.85fr),
    stroke: none,
    align: (left, left, left, left, left),
    inset: (x: 4.5pt, y: 5pt),
    table.hline(stroke: 1.0pt),
    [对应章节], [辅助用途], [提问概要], [输出内容], [人工修改与核对],
    table.hline(stroke: 0.5pt),

    [Section 3.2,\ Appendix A],
    [数据处理与绘图代码协作 \[13\]],
    [整理分组交叉验证特征计算流程与 Matplotlib 多子图排版代码],
    [生成特征统计函数与多面板图表绘制代码],
    [核对特征计算公式与数据维度，补充带心回归及期望积分核心代码],

    [Section 5.2,\ Appendix A],
    [组合穷举代码加速 \[13\]],
    [将主变量状态遍历改写为 NumPy 矩阵批量运算以缩短耗时],
    [给出分批矩阵求值与能量统计代码写法],
    [代入小规模算例比对结果，确认与逐状态枚举输出一致],

    [Section 6.1,\ Appendix A],
    [实验结果整理脚本 \[14\]],
    [编写硬件返回结果的离线读取、去重统计与参数扫描汇总脚本],
    [输出自旋记录读取、哈希核对与表格导出代码],
    [逐条核对 40 条原始返回记录的解码化学式与伊辛能量值],

    [Section 5.2,\ 全文及 App. A],
    [英文语法润色与 MILP 验算脚本 \[15\]],
    [检查英文初稿语法表达，协助编写基于 SciPy HiGHS 的混合整数规划验算代码],
    [给出长句语法修改建议及 MILP 线性化求解示例],
    [人工通读审校英文定稿，本地运行独立 MILP 验算脚本核对 8 个矩阵最优值],

    table.hline(stroke: 1.0pt),
  )
}

== 三、备注说明

1. 文中涉及上述代码协作与英文润色之处已在 Section 3.2 \[13\]、Section 5.2 \[13, 15\]、Section 6.1 \[14\] 及 Appendix A \[13--15\] 标注，并列入文末 References。
2. 本文的核心建模思路、理论推导、QUBO/HOBO 编码设计与 SPQC-550 光量子真机实验均由本队独立完成，所有辅助代码与文字均经队员人工测试与核实。
