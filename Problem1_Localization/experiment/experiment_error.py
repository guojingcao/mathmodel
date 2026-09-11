"""误差界敏感性：固定测得角度，仅放宽容许误差界。"""
from experiment.common import make_scenario, evaluate


def run(config, repetitions, seed):
    """真实注入误差 ≤0.2°，五个误差界都有效；该设计研究界宽而非换噪声。"""
    rows, records = [], []
    for repetition in range(repetitions):
        target, points, angles, errors = make_scenario(seed, repetition, [0, 90, 210], .2)
        for bound in (.2, .5, 1.0, 1.5, 2.0):
            row, record = evaluate("error", repetition, bound, target, points, angles,
                                   errors, bound, 90, config)
            rows.append(row)
            records.append(record)
    return rows, records
