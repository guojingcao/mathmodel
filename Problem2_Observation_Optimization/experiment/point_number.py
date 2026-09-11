"""在同一贪心选择序列上评价2至6个检测点。"""
from experiment.common import scenario_randomness, transform_layout, evaluate


def run(config, repetitions, seed, optimized):
    """前n个点嵌套不变，隔离检测点数量的影响。"""
    rows, records = [], []
    for repetition in range(repetitions):
        target, rotation, errors, _ = scenario_randomness(seed, repetition)
        for number in range(2, 7):
            points = transform_layout(optimized[:number], target, rotation)
            row, raw = evaluate("number", "optimized", repetition, points,
                                target, errors[:number], config)
            rows.append(row); records.append(raw)
    return rows, records
