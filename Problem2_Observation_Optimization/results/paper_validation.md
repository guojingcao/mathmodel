# 问题二论文材料校验记录

结论：**PASS**。

## 1. 数值来源校验

- 布局、检测点数和交会角三类均值逐项回查 `group_statistics.csv`。
- 相对随机布局的配对差及区间逐项回查 `paired_layout_comparisons.csv`。
- 本文布局相对 DOP 基线的配对差及区间逐项回查 `paired_dop_comparison.csv`。
- 三线表按三位小数四舍五入；正文中的核心数字与表格采用相同口径。
- 420 次实验结果、布局选择结果和原始观测记录均未因论文整理而改动。

## 2. LaTeX 结构校验

- 使用 XeLaTeX 对 `paper/compile_check.tex` 连续编译两遍。
- 正文、5 张 PDF 图和 5 张三线表均成功载入，共形成 7 页检查稿。
- 编译日志中无 LaTeX Error、Undefined control sequence、未解析交叉引用或 Overfull box。

## 3. 问题一至问题二衔接校验

- 问题二只负责选择检测点，随后调用问题一接口计算可行域 $\Omega$，没有复制或重建定位模型。
- 布局评价仍采用问题一给出的面积、直径、安全覆盖圆和区域质心指标。
- 问题一两站实验使用原始交会角 $\alpha\in[0,180^\circ]$；问题二多站诊断使用条件角 $\widetilde\gamma=\min(\gamma,180^\circ-\gamma)$，正文已明确区分。
- DOP 仅为诊断基线；最终布局目标仍为面积外包、直径外包、交会角组成的字典序。
- 未引入 PSO、遗传算法、概率地图或其他未经实验支持的复杂算法。

更完整的接口、参数与数据口径映射见 `problem1_problem2_consistency.md`。
