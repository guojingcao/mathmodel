"""随机、均匀、Ω面积贪心及DOP诊断基线的公平候选集比较。"""
from experiment.common import scenario_randomness, transform_layout, uniform_indices, evaluate


def run(config, repetitions, seed, candidates, optimized, dop_layout):
    """固定四点；每次方法共享真值、整体旋转和误差向量。"""
    rows, records = [], []
    count = 4
    uniform = candidates[uniform_indices(len(candidates), count)]
    for repetition in range(repetitions):
        target, rotation, errors, rng = scenario_randomness(seed, repetition)
        random = candidates[rng.choice(len(candidates), count, replace=False)]
        layouts = {"random": random, "uniform": uniform,
                   "optimized": optimized[:count], "dop_baseline": dop_layout[:count]}
        for method, template in layouts.items():
            row, raw = evaluate("layout", method, repetition,
                transform_layout(template, target, rotation), target, errors[:count], config)
            rows.append(row); records.append(raw)
    return rows, records
