"""固定两站距离，只改变交会角。"""
import numpy as np
from experiment.common import scenario_randomness, transform_layout, evaluate


def run(config, repetitions, seed):
    """同一重复共享真值、整体旋转和两次测向误差，检验实际Ω质心误差。"""
    rows, records = [], []
    radius = config.observation_radius
    for repetition in range(repetitions):
        target, rotation, errors, _ = scenario_randomness(seed, repetition)
        for angle in (30, 45, 60, 90, 120):
            radians = np.deg2rad([0, angle])
            template = radius*np.column_stack([np.cos(radians), np.sin(radians)])
            points = transform_layout(template, target, rotation)
            row, raw = evaluate("angle", "controlled", repetition, points,
                                target, errors[:2], config, angle)
            rows.append(row); records.append(raw)
    return rows, records
