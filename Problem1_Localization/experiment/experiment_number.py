"""检测数量实验：嵌套加入约束，不重新抽取已有站的误差。"""
from experiment.common import make_scenario, evaluate


def run(config, repetitions, seed):
    """比较 2~6 站；同一重复前 n 站完全一致，因此连续可行集必然嵌套。"""
    rows, records = [], []
    for repetition in range(repetitions):
        target, points, angles, errors = make_scenario(seed, repetition, [0, 90, 210, 45, 135, 280])
        for n in (2, 3, 4, 5, 6):
            row, record = evaluate("number", repetition, n, target, points[:n], angles[:n],
                                   errors[:n], 1.0, 90, config)
            rows.append(row)
            records.append(record)
    return rows, records
