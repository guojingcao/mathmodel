"""统计、配对自举及报告生成；只输出实际数据支持的结论。"""
import csv
import json
import numpy as np

SETTINGS = {"number": "number_of_points", "error": "angle_error", "angle": "intersection_angle"}


def group_rows(rows, kind):
    """按单因素实验的横轴变量分组，返回有序字典。"""
    field = SETTINGS[kind]
    values = sorted({r[field] for r in rows if r["experiment_type"] == kind})
    return {v: [r for r in rows if r["experiment_type"] == kind and r[field] == v] for v in values}


def bootstrap_mean_ci(values, seed=41):
    """对重复实验均值给出 2000 次百分位自举 95% 区间，不假设误差正态。"""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def write_csv(path, rows):
    """以 UTF-8 BOM 输出可由中文 Excel 打开的明细 CSV，空值保留为空。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows, output, figures, repetitions, config):
    """自动生成分组汇总、配对趋势检验与论文分析；不预设 90° 必为样本最优。"""
    summaries = []
    for kind in SETTINGS:
        for setting, subset in group_rows(rows, kind).items():
            item = {"experiment_type": kind, "setting": setting, "n": len(subset)}
            for metric in ("area", "diameter", "cover_radius", "localization_error"):
                values = [r[metric] for r in subset]
                lo, hi = bootstrap_mean_ci(values)
                item.update({metric + "_mean": float(np.mean(values)), metric + "_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                             metric + "_ci_low": lo, metric + "_ci_high": hi})
            item["area_bound_gap_mean"] = float(np.mean([r["area_upper"] - r["area_lower"] for r in subset]))
            summaries.append(item)
    write_csv(output / "group_statistics.csv", summaries)
    paired = []
    angles = group_rows(rows, "angle")
    reference = {r["repetition"]: r for r in angles[90]}
    for setting, subset in angles.items():
        if setting == 90:
            continue
        delta = [r["localization_error"] - reference[r["repetition"]]["localization_error"] for r in subset]
        lo, hi = bootstrap_mean_ci(delta)
        paired.append({"angle": setting, "comparison": "angle minus 90", "mean_error_difference": float(np.mean(delta)),
                       "paired_ci_low": lo, "paired_ci_high": hi})
    write_csv(output / "paired_angle_comparisons.csv", paired)
    violations = {}
    for kind, sign in (("number", -1), ("error", 1)):
        count = 0
        for rep in range(repetitions):
            subset = sorted([r for r in rows if r["experiment_type"] == kind and r["repetition"] == rep], key=lambda r: r[SETTINGS[kind]])
            for a, b in zip(subset, subset[1:]):
                # 检验的是违反单调性且超出离散区间的情况，不拿近似中点代替定理。
                count += int(b["area_lower"] > a["area_upper"] + 1e-6) if sign < 0 else int(b["area_upper"] < a["area_lower"] - 1e-6)
        violations[kind] = count
    angle_summaries = [s for s in summaries if s["experiment_type"] == "angle"]
    winner = min(angle_summaries, key=lambda s: s["localization_error_mean"])
    errors = [s for s in summaries if s["experiment_type"] == "error"]
    exponent = float(np.polyfit(np.log([s["setting"] for s in errors]), np.log([s["area_mean"] for s in errors]), 1)[0])
    audit = {"runs": len(rows), "truth_feasible_count": sum(r["truth_feasible"] for r in rows),
             "monotonicity_violations_beyond_grid_bounds": violations,
             "max_area_bound_gap": max(r["area_upper"] - r["area_lower"] for r in rows),
             "max_diameter_bound_gap": max(r["diameter_upper"] - r["diameter"] for r in rows),
             "max_radius_bound_gap": max(r["cover_radius"] - r["cover_radius_lower"] for r in rows),
             "angle_mean_error_best": winner["setting"], "area_error_loglog_slope": exponent}
    (output / "experiment_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 问题一实验分析报告", "", "## 1. 模型及适用边界", "",
             "采用已经确定的确定性有界误差定位模型：圆盘与各站测向角域的交集。没有引入搜索策略或概率后验。真值仅供离线检验，不参与网格筛选。",
             "有限 valid_points 不是连续 Ω；内外凸包给出数值包络。面积记录界中点，直径记录内包值并附外包上界；cover_radius 使用外包最小圆，因此安全但略保守。数学推导见 ../MODEL.md。", "",
             "## 2. 参数与控制变量", "",
             f"每组 {repetitions} 次配对重复，共 {len(rows)} 次；圆盘半径 {config.radius:g} m，初始 {config.initial_grid}×{config.initial_grid} 格，边界精度 ≤{config.resolution:g} m。",
             "各站距真值 1000 m；真值在半径 300 m 圆盘内均匀生成，场景整体随机旋转。此为受控场景，不代表所有可能场地。",
             "数量实验固定前 n 站及误差，逐次加入观测；误差界实验固定三站和测得角度，真实注入误差≤0.2°，仅改变误差界；交会角实验使用两站和同一组配对角度误差，界为±1°。",
             "表中 CI 是独立重复场景均值的百分位自举95%区间，不是干扰源位置的概率置信区间。界误差与实际测量噪声严格区分。", "",
             "## 3. 分组统计", "", "|实验|设置|平均面积(m²)|平均直径(m)|平均覆盖半径(m)|平均定位误差(m)|", "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"|{s['experiment_type']}|{s['setting']:g}|{s['area_mean']:.3f}|{s['diameter_mean']:.3f}|{s['cover_radius_mean']:.3f}|{s['localization_error_mean']:.3f}|")
    lines += ["", "## 4. 实际规律与解释", "",
              f"图3：嵌套加入约束后，区域不可能在集合意义上扩大；超出网格误差界的面积反单调次数为 {violations['number']}。这不能推广为新增任意站都严格改善，冗余观测可能无改善。",
              f"图4：固定测量后放宽误差界，区域应单调扩大；超出网格误差界的反单调次数为 {violations['error']}。五点均值的 log-log 描述性斜率为 {exponent:.3f}，不将有限区间拟合当作普适幂律。",
              f"图5：本批样本中平均定位误差最小的交会角为 {winner['setting']:g}°（{winner['localization_error_mean']:.3f} m）。90°是否优于其他角度以配对差及区间判断，不要求每个样本均在90°最优。",
              "小误差、等距离的局部线性化给出均方位置误差与 1/sin²(α) 成正比，解释接近直角通常更稳定；该近似不替代实际角域求解。", "",
              "|交会角减90°|平均定位误差差(m)|配对95%区间(m)|", "|---:|---:|---:|"]
    for p in paired:
        lines.append(f"|{p['angle']}|{p['mean_error_difference']:.4f}|[{p['paired_ci_low']:.4f}, {p['paired_ci_high']:.4f}]|")
    lines += ["", "区间跨0时不足以确认差异；多组比较为探索性，未做多重性校正。概率分布仅用于生成验证数据，不改变确定性可行域的含义。", "",
              "## 5. 数值验证与数据追溯", "",
              f"{audit['truth_feasible_count']}/{len(rows)} 组真值满足全部原始角度约束。最大面积上下界差 {audit['max_area_bound_gap']:.6f} m²；最大直径界差 {audit['max_diameter_bound_gap']:.6f} m；最大覆盖半径界差 {audit['max_radius_bound_gap']:.6f} m。",
              "单元测试另验跨零度、相反射线、平行测向、不相容观测、窄域不漏检、解析面积/质心/直径/覆盖圆和细分收敛。完整输入见 observations.jsonl；版本、参数及哈希见 run_manifest.json。",
              "面积质心数值误差界见 CSV 的 centroid_error_bound；区域有界不意味着其直径圆一定覆盖区域。等边三角形反例说明最小覆盖圆需独立计算。", "",
              "## 6. 图件与用途", "",
              "图1展示单站方向角及±1°界；图2左图为全局交会几何，右图放大真实可行域及其数值内外包。图3、图4、图5分别对应以上三项实验。",
              f"图件遵循 CUMCM_Visualization，输出目录：{figures.as_posix()}；每张都有 PDF、SVG、600 dpi PNG。",
              "图件可用于论文排版，但结论仅限本受控实验；正式论文需保留参数、界误差、样本量和适用条件。"]
    (output / "analysis_report.md").write_text("\n\n".join(lines[:1]) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")
    return summaries
