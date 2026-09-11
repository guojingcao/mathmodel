"""交会角实验：等观测距离、两站、同一重复配对误差。"""
from experiment.common import make_scenario, evaluate


def run(config, repetitions, seed):
    """变化的是由真值到两站向量的夹角，不混淆整体朝向与交会几何。"""
    rows, records = [], []
    for repetition in range(repetitions):
        for intersection in (30, 45, 60, 90, 120):
            target, points, angles, errors = make_scenario(seed, repetition, [0, intersection])
            row, record = evaluate("angle", repetition, intersection, target, points, angles,
                                   errors, 1.0, intersection, config)
            rows.append(row)
            records.append(record)
    return rows, records
