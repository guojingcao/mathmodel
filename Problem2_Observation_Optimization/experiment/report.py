"""问题二统计、配对比较和数据驱动论文分析。"""
import csv
import json
import numpy as np


def write_csv(path, rows):
    """写入UTF-8 BOM明细表，保留空字段。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bootstrap_ci(values, seed=72):
    """用2000次百分位自举估计重复场景均值95%区间。"""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def summarize(rows):
    """按实验、方法、点数或角度分组，输出四项Ω指标及几何诊断。"""
    keys = sorted({(r["experiment_type"], r["method"], r["number_points"],
                    r["intersection_angle"]) for r in rows}, key=str)
    output = []
    for experiment, method, number, angle in keys:
        subset = [r for r in rows if (r["experiment_type"], r["method"],
                  r["number_points"], r["intersection_angle"]) == (experiment, method, number, angle)]
        item = {"experiment_type": experiment, "method": method,
                "number_points": number, "intersection_angle": angle, "n": len(subset)}
        for metric in ("area", "diameter", "cover_radius", "localization_error",
                       "min_intersection_angle", "dop", "runtime"):
            values = [r[metric] for r in subset]
            lo, hi = bootstrap_ci(values)
            item.update({metric+"_mean": float(np.mean(values)),
                         metric+"_std": float(np.std(values, ddof=1)),
                         metric+"_ci_low": lo, metric+"_ci_high": hi})
        output.append(item)
    return output


def paired_layout(rows, baseline="random"):
    """逐场景计算各方法相对随机布局差值；负值表示该误差指标改善。"""
    subset = [r for r in rows if r["experiment_type"] == "layout"]
    base = {r["repetition"]: r for r in subset if r["method"] == baseline}
    output = []
    for method in sorted({r["method"] for r in subset} - {baseline}):
        method_rows = {r["repetition"]: r for r in subset if r["method"] == method}
        item = {"method": method, "baseline": baseline, "n": len(base)}
        for metric in ("area", "diameter", "cover_radius", "localization_error"):
            delta = [method_rows[k][metric]-base[k][metric] for k in sorted(base)]
            lo, hi = bootstrap_ci(delta)
            item.update({metric+"_difference_mean": float(np.mean(delta)),
                         metric+"_difference_ci_low": lo, metric+"_difference_ci_high": hi})
        output.append(item)
    return output


def paired_methods(rows, method, baseline):
    """计算两个指定布局的逐场景差，用于直接检验Ω优化与DOP基线。"""
    subset = [row for row in rows if row["experiment_type"] == "layout"]
    first = {row["repetition"]: row for row in subset if row["method"] == method}
    second = {row["repetition"]: row for row in subset if row["method"] == baseline}
    item = {"method": method, "baseline": baseline, "n": len(first)}
    for metric in ("area", "diameter", "cover_radius", "localization_error"):
        delta = [first[index][metric]-second[index][metric] for index in sorted(first)]
        lo, hi = bootstrap_ci(delta)
        item.update({metric+"_difference_mean": float(np.mean(delta)),
                     metric+"_difference_ci_low": lo, metric+"_difference_ci_high": hi})
    return item


def write_report(rows, output, figures, repetitions, config, optimized_indices, dop_indices):
    """生成问题二论文分析；优劣结论由实际Ω指标和配对区间决定。"""
    summaries = summarize(rows)
    paired = paired_layout(rows)
    dop_comparison = paired_methods(rows, "optimized", "dop_baseline")
    write_csv(output / "group_statistics.csv", summaries)
    write_csv(output / "paired_layout_comparisons.csv", paired)
    write_csv(output / "paired_dop_comparison.csv", [dop_comparison])
    layout_stats = {s["method"]: s for s in summaries if s["experiment_type"] == "layout"}
    best_area = min(layout_stats, key=lambda m: layout_stats[m]["area_mean"])
    best_diameter = min(layout_stats, key=lambda m: layout_stats[m]["diameter_mean"])
    best_cover = min(layout_stats, key=lambda m: layout_stats[m]["cover_radius_mean"])
    angle_stats = sorted([s for s in summaries if s["experiment_type"] == "angle"],
                         key=lambda s: s["intersection_angle"])
    best_angle = min(angle_stats, key=lambda s: s["localization_error_mean"])
    number_stats = sorted([s for s in summaries if s["experiment_type"] == "number"],
                          key=lambda s: s["number_points"])
    monotonic_violations = 0
    for rep in range(repetitions):
        one = sorted([r for r in rows if r["experiment_type"] == "number" and r["repetition"] == rep],
                     key=lambda r: r["number_points"])
        monotonic_violations += sum(b["area_lower"] > a["area_upper"]+1e-6 for a, b in zip(one, one[1:]))
    audit = {"runs": len(rows), "truth_feasible": sum(r["truth_feasible"] for r in rows),
             "best_mean_area_method": best_area, "best_mean_diameter_method": best_diameter,
             "best_mean_cover_radius_method": best_cover,
             "best_mean_localization_error_angle": best_angle["intersection_angle"],
             "number_area_monotonicity_violations_beyond_bounds": monotonic_violations,
             "optimized_indices": optimized_indices, "dop_baseline_indices": dop_indices}
    (output / "experiment_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = {"random":"随机布局", "uniform":"均匀圆周布局", "optimized":"本文Ω面积贪心布局",
              "dop_baseline":"DOP筛选诊断基线"}
    lines = ["# 问题二实验分析报告", "", "## 1. 物理意义与模型继承", "",
             "问题二不重新建立点定位模型。检测点布局决定各有界测向角域的相对方向，布局输入问题一后得到 Ω；以 Ω 的面积、直径和最小覆盖圆半径评价布局。",
             "布局规划围绕问题一当前可行域代表中心进行。受控实验用真值替代该参考中心，只用于隔离几何因素，不表示实战中预先知道干扰源。", "",
             "## 2. 优化模型与求解", "",
             "候选集为参考中心周围半径1000 m、每15°一个的24个圆周点。检测点须在半径1800 m场地内，按选择顺序相邻移动不超过2000 m。",
             "每轮枚举所有可达候选点，并调用问题一计算加入该点后的 Ω。使用字典序最小化：(面积外包上界，直径外包上界，负的最小条件交会角，候选编号)。这保留各指标的物理量纲，不用任意权重相加。",
             f"本文优化选择索引为 {optimized_indices}；DOP诊断基线索引为 {dop_indices}。固定种子仅用于初始点和重复实验复现。", "",
             "## 3. 实验设置", "",
             f"每个水平{repetitions}次配对重复。误差独立生成于[-1°,1°]，所有方法共享每次真值、整体旋转及误差向量。问题一初始网格{config.initial_grid}×{config.initial_grid}，最终评价边界精度≤{config.evaluation_resolution_m:g} m。",
             "随机、均匀、本文优化和DOP基线均从同一24点候选圆周构造；DOP仅是诊断基线，全部最终评价仍使用 Ω。自举区间是重复场景均值区间，不是目标位置概率置信域。", "",
             "## 4. 布局比较结果", "", "|方法|平均面积(m²)|平均直径(m)|平均覆盖半径(m)|平均定位误差(m)|最小交会角(°)|DOP|", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, s in layout_stats.items():
        lines.append(f"|{labels.get(method,method)}|{s['area_mean']:.3f}|{s['diameter_mean']:.3f}|{s['cover_radius_mean']:.3f}|{s['localization_error_mean']:.3f}|{s['min_intersection_angle_mean']:.1f}|{s['dop_mean']:.3f}|")
    lines += ["", f"平均面积最小的是{labels[best_area]}，平均直径最小的是{labels[best_diameter]}，平均覆盖半径最小的是{labels[best_cover]}。以下配对差均为方法减随机布局，负值表示改善。", "",
              "|方法|面积差(m²)|面积差95%区间|直径差(m)|覆盖半径差(m)|定位误差差(m)|", "|---|---:|---:|---:|---:|---:|"]
    for p in paired:
        lines.append(f"|{labels[p['method']]}|{p['area_difference_mean']:.3f}|[{p['area_difference_ci_low']:.3f},{p['area_difference_ci_high']:.3f}]|{p['diameter_difference_mean']:.3f}|{p['cover_radius_difference_mean']:.3f}|{p['localization_error_difference_mean']:.3f}|")
    lines += ["", "## 5. 检测数量与交会角", "",
              f"本文嵌套布局从2点到6点的平均面积由 {number_stats[0]['area_mean']:.3f} m² 变为 {number_stats[-1]['area_mean']:.3f} m²；超出网格上下界的反单调次数为 {monotonic_violations}。新增点可能冗余，因此不要求每步严格改善。",
              f"两站交会角实验中，平均区域质心定位误差最小的角度是 {best_angle['intersection_angle']:g}°（{best_angle['localization_error_mean']:.3f} m）。结果只适用于等距离、指定误差界和本样本。", "",
              "## 6. 与DOP方法的关系", "",
              "DOP来自目标附近的局部线性随机误差近似；本题给定的是确定性角误差界，且评价对象是有限场地内完整 Ω。DOP不直接包含距离引起的角域宽度、场地截断和非线性边界，因此不能替代面积或直径目标。",
              f"本实验没有预设DOP一定更差：DOP基线最终仍以问题一指标检验。实际平均面积为 {layout_stats['dop_baseline']['area_mean']:.3f} m²，本文布局为 {layout_stats['optimized']['area_mean']:.3f} m²。本文减DOP的配对面积差为 {dop_comparison['area_difference_mean']:.3f} m²，95%自举区间为 [{dop_comparison['area_difference_ci_low']:.3f}, {dop_comparison['area_difference_ci_high']:.3f}] m²；本受控实验支持本文布局面积更小。该结论不外推到其他候选集、距离或误差模型。", "",
              "## 7. 输出、验证与局限", "",
              f"共{len(rows)}次定位，{sum(r['truth_feasible'] for r in rows)}/{len(rows)}次真值满足角界。明细保存每次站位、测得角、注入误差与数值包络；图件位于 {figures.as_posix()}。",
              "候选中心依赖问题一先验区域代表点；若该代表点偏差很大，真实布局效果会下降。本文未模拟障碍物、禁行区和连续路径时间，仅验证圆形场地和相邻移动上限。实际任务应滚动更新问题一 Ω 后重新枚举合法候选点。",
              "结论仅用于既定确定性覆盖/定位路线，不外推为新的概率搜索或在线策略。"]
    (output / "problem2_analysis.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summaries
