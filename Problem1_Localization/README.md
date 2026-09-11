# Problem1_Localization

问题一既定模型的独立工程实现：**有界测向误差角域交集 + 自适应二维网格**。不重新设计模型，不含在线搜索、覆盖调度或模拟器通信。

## 运行

环境：Python≥3.10，NumPy、Matplotlib。绘图直接调用相邻目录中现有的 `CUMCM_Visualization`，不修改共享库。

```powershell
cd D:\My_MathModeling_Project
python -m pip install -r Problem1_Localization/requirements.txt
python -m unittest discover -s Problem1_Localization/tests -v
python Problem1_Localization/main.py
python Problem1_Localization/validate.py
```

本工作区依赖已存在，无须重新安装。默认30次配对重复×3类实验×5个水平，共450次；初始200×200格，边界精度0.03515625 m。

```powershell
# 小规模快速复现，写到独立目录以保留正式结果
python Problem1_Localization/main.py --repetitions 2 --resolution 0.140625 --output Problem1_Localization/results/smoke --no-plots
# 增加样本或改变网格，另指定输出路径
python Problem1_Localization/main.py --repetitions 100 --grid 200 --resolution 0.017578125 --output Problem1_Localization/results/refined --figures origin_figures/Problem1_Localization_refined
```

同一输出目录重跑会更新该目录内的同名结果。主结果位于本工程 `results/`；全部图件遵循工作区统一规则，位于 `../origin_figures/Problem1_Localization/`，每图PDF、SVG、PNG（600dpi）。保持工程与 `CUMCM_Visualization` 相邻即可迁移运行。

## 调用求解器

在本工程目录执行：

```python
from config import GridConfig
from model.feasible_region import calculate_feasible_region
from model.metrics import calculate_metrics

points = [[-1000, 0], [0, -1000]]
angles = [0, 90]
valid_points = calculate_feasible_region(points, angles, error=1)
region = calculate_feasible_region(points, angles, error=1,
    config=GridConfig(initial_grid=200, resolution=0.03515625), return_details=True)
metrics = calculate_metrics(region, true_position=[0, 0])
print(metrics)
```

方位角从正x轴逆时针计量，单位度；距离m，面积m²。求解不需要真值，`true_position` 仅用于离线误差指标。支持0≤error<90°。

## 各模块：数学对应、设计原因、验证及实验数据

|模块|数学模型对应|为什么这样设计|如何验证正确|输出数据|
|---|---|---|---|---|
|geometry.py|圆周角距、测向角域、半平面与凸包|正确处理跨零度和正向射线|原始atan2与半平面随机点交叉检验；零误差退化测试|方向角、约束布尔值、凸包|
|feasible_region.py|连续角域交集的自适应网格包络|格心不满足不代表整格无解，边界必须细分|窄域、空域、圆边界、平行测向、解析参考和收敛|valid_points、内外凸包、叶格、状态、格宽、计算量|
|metrics.py|区域面积、直径、最小圆、面积质心|避免等权点均值与误用直径圆|正方形/等边三角形解析值，小点集穷举圆|面积/直径/圆半径及上下界、质心、定位误差|
|experiment_number.py|增加约束后集合嵌套收缩|保持已有站与误差不变才能归因于数量|检查真值可行及超出离散界的单调性违反|2至6站所有指标和原始观测|
|experiment_error.py|放宽角误差界后集合嵌套扩大|固定测得角度，区分误差界与实际噪声|真实注入误差≤最小界；区间化单调性检查|五档误差界指标、log-log描述性斜率|
|experiment_angle.py|交会几何条件对定位稳定性的影响|等距离与配对误差消除混杂|同一重复输入核对，实际均值及配对95%区间|五档交会角误差、区域指标和配对差|
|report.py|重复实验统计与配对比较|不把样本均值当普适规律|明细可复算，区间跨0不强行作差异结论|分组统计CSV、配对CSV、审计JSON、分析报告|
|plot_results.py|角域几何和控制变量实验的可视化|CUMCM统一风格、局部放大、小误差不夸大|导出完整性、字体诊断、600dpi与视觉复核|五张PDF/SVG/PNG、示例区域NPZ|

## 主要文件

- `MODEL.md`：可用于论文整理的完整数学模型、包络推导、质心误差界和局限。
- `results/experiment_summary.csv`：450行逐次实验指标，含要求的8个字段以及数值上下界。
- `results/observations.jsonl`：逐次真值、站位、测得角度、实际注入误差与误差界。
- `results/group_statistics.csv`：15个实验水平的均值、标准差、自举区间。
- `results/paired_angle_comparisons.csv`：各交会角相对90°的配对差。
- `results/analysis_report.md`：自动生成的结果分析，只报告实际数据。
- `results/experiment_audit.json`：真值可行性、反单调检查和最大离散界宽。
- `results/example_region.npz`：图2的有效点、内外凸包和叶格，单位m。
- `results/run_manifest.json`：种子、网格参数、Python/NumPy版本与源代码SHA-256。
- `results/run.log`：每次求解和各层网格日志。
- `results/validation.json`：默认结果集的完整验证记录，包括450组参考几何、配对输入、源代码哈希及PNG分辨率。
- `results/grid_convergence.csv`：同一场景的五级细分收敛数据。
- `results/unit_tests.txt`：12项单元测试的执行记录。

`validate.py` 复核默认 `results/` 和默认图目录；自定义输出实验可通过其中相同的校验函数进一步复核。使用半平面多边形裁剪的参考解只存在于测试代码中，不替换生产自适应网格求解器。

`area` 是面积界中点；`diameter` 是内包近似值，真实直径不大于 `diameter_upper`；`cover_radius` 是能覆盖外包的保守最小圆半径，真实最小半径不小于 `cover_radius_lower`。切勿将这些有限精度指标称为解析精确解。

`empty` 表示保格流程已排除所有网格；`unresolved` 表示仍有潜在区域但没有采样到有效点，需要细化或分析退化情形，不能当作无解。站点方向未定义；零面积区域不具有通常的面积质心。实现的几何数值保证受双精度舍入容差限制，详见 MODEL.md。
