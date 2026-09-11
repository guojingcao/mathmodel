# Problem2_Observation_Optimization

问题二工程严格实现：**选择检测点布局 → 调用问题一定位模型 → 计算 Ω 可行区域 → 分指标评价布局**。不重建点定位模型，不使用任意加权目标，不把DOP作为最终优化目标，不含PSO、遗传算法、概率地图或任务调度。

## 运行

```powershell
cd D:\My_MathModeling_Project
python -m unittest discover -s Problem1_Localization/tests -v
python -m unittest discover -s Problem2_Observation_Optimization/tests -v
python Problem2_Observation_Optimization/main.py
python Problem2_Observation_Optimization/validate.py
```

依赖Python≥3.10、NumPy、Matplotlib以及相邻的 `Problem1_Localization` 和 `CUMCM_Visualization`。默认生成24个候选点，贪心选择6点，运行30次配对重复。快速检查可用：

```powershell
python Problem2_Observation_Optimization/main.py --repetitions 2 --evaluation-resolution 0.28125 --output Problem2_Observation_Optimization/results/smoke --no-plots
```

自定义输出不会覆盖正式结果；`validate.py` 复核默认正式目录。

## 模块的数学、设计、验证和数据

|模块|数学模型对应|为什么这样设计|如何验证正确|输出实验数据|
|---|---|---|---|---|
|candidate_points.py|圆形场地内有限候选集与相邻移动约束|将连续布局转成可解释、可审计的有限枚举|点数、半径、唯一性、场地和步长边界测试|候选坐标、路线可行布尔值|
|localization.py|问题一的角域交集 Ω|唯一适配入口，避免复制后产生两套定位逻辑|对象类型、角度、真值包含和问题一指标一致性测试|RegionResult、面积/直径/圆/质心指标、测得角|
|geometry_metric.py|面积、直径、条件交会角、DOP诊断|Ω指标为主，几何量只帮助解释|正交、同向、反向共线及秩亏解析测试|面积和直径数值界、γmin、DOP|
|layout_optimizer.py|有限候选集上的逐点字典序贪心|直接最小化Ω面积外包，不混合量纲|可复现性、候选枚举数、无重复、嵌套面积测试|每轮入选轨迹、每个候选的完整评价|
|compare_layout.py|固定4点的方法对照|同候选集、同真值、同旋转、同误差才能公平归因|逐重复输入配对和真值可行性复核|随机/均匀/优化/DOP基线全指标|
|point_number.py|嵌套增加角域约束|保持前n点与误差不变，验证集合收缩|超出问题一上下界的单调性违反计数|2至6点面积、直径、圆半径和误差|
|angle_analysis.py|交会角对两站区域定位的影响|固定距离与误差，隔离角度因素|五档角度配对输入、实际均值和区间|30°至120°所有Ω指标|
|report.py|重复实验统计与配对比较|不靠单次场景或预设结论评价优劣|明细复算、自举区间和唯一ID检查|统计CSV、配对CSV、审计JSON、分析Markdown|
|plot_problem2.py|五类论文图|复用既有CUMCM风格，保留原始重复轨迹|三格式完整性、600dpi、PDF回读与视觉复核|5张PDF/SVG/PNG及诊断JSON|

## 主要交付

- `MODEL.md`：模型继承、字典序目标、贪心过程、DOP边界及实验假设。
- `results/problem2_summary.csv`：420次定位明细。
- `results/optimization_trace.csv`：从1至6点的入选收敛数据。
- `results/candidate_evaluations.csv`：每轮全部候选点评分。
- `results/selected_layouts.json`：候选集、本文布局和DOP布局坐标/编号。
- `results/observations.jsonl`：每次实验的问题一完整输入。
- `results/group_statistics.csv`、`paired_layout_comparisons.csv`：分组与配对结果。
- `results/paired_dop_comparison.csv`：本文Ω布局相对DOP诊断基线的直接配对差。
- `results/problem2_analysis.md`：由实际数据自动生成的论文分析。
- `results/validation.json`、`unit_tests.txt`：验证证据。
- `results/visual_review.md`：PNG与PDF回读的逐图视觉复核记录。
- `paper/problem2_section.tex`：可直接纳入论文的问题二正文。
- `paper/problem2_tables.tex`：由正式结果整理的五张三线表。
- `paper/compile_check.tex`：正文与三线表的独立编译检查入口。
- `results/problem1_problem2_consistency.md`：问题一、问题二模型和数据口径衔接检查。
- `results/paper_validation.md`：正文数值、LaTeX编译和模型衔接的最终校验记录。
- 图件位于 `../origin_figures/Problem2_Observation_Optimization/`，每图PDF、SVG及600dpi PNG。

问题一面积是网格内外包中点；直径是内包近似并带外包上界；覆盖半径使用安全外包最小圆。完整定义见问题一 `MODEL.md`。本工程的2000 m相邻移动上限是可配置实验条件，不替代官方速度、时间或障碍物规则。
