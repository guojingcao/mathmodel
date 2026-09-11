# 问题4 机器人程序使用说明（全向 + 定向干扰源）

`robot4.py` 是问题4 的模拟器对接程序。

## 模型结构

```
保证层  —— 三角网格覆盖(边长 a<=1000m) + 逐频道三角形覆盖证书
状态层  —— 位置可行域(半平面交) + 频道状态机
调度层  —— 几何冗余过滤 + 受限顺路清除 + 批量补测(方位鲁棒) + 2-opt 清除
```

## 定向源带来的三个关键机制

1. **三角网格覆盖**：用边长 ≤1000m 的三角网格覆盖半径 1800m 圆盘并向圆外延伸，
   使边界处"朝外辐射"的定向源也落在某个含外侧顶点的三角形内。
   验证：网格最大边长 1000.0m；圆盘内 20000 采样点 **0% 未被三角形覆盖**。
2. **逐频道三角形覆盖证书**：若某三角形三个顶点对该频道均返回 `no_signal`，
   则该三角形内不可能存在该频道干扰源（顶点凸包 + 1000m 保证接收）；
   当覆盖圆盘的全部三角形都被证伪时，该频道**退役**（确定不存在）。
3. **方位鲁棒补测与归航**：
   - `no_signal` **不用于排除位置**（可能是定向盲区），只用于三角证书；
   - 补测点可能在盲区 → 依次尝试 8 个方位；
   - 清除失败 → **对每条已有示向各做一次"沿示向二分归航"**
     （边界源可能只有个别方位稳健，单条示向会跑偏）。

## 运行

```powershell
python robot4.py --robot-id 你的参赛队号
# 可选: --base-url / --log-file / --mesh-stats
```

`--mesh-stats` 只打印网格统计（点数/三角形数/需证伪数）后退出，不连模拟器。

## 离线自检

```powershell
python selfcheck4.py 20            # 20 案例, 用默认网格余量
python selfcheck4.py 20 500        # 指定网格外扩余量(米)
```

自检会先验证网格覆盖性与证书距离条件，再跑蒙特卡洛（定向比例 0%/50%/100%）。

## 网格参数灵敏度消融

`python sensitivity4.py 10` 扫描「网格边长 a × 外扩余量」。实测（每格 10 案例）：

| a(m) | 余量(m) | 点数 | 圆盘覆盖 | 50% 定向清除率 | 移动(m) | 估计时间(s) |
|---|---|---|---|---|---|---|
| 900 | 500 | 19 | 83.0% | 99.1% | 24618 | 6638 |
| 900 | **800** | **31** | **100%** | **100%** | **37023** | **10125** |
| 900 | 1000 | 37 | 100% | 100% | 41818 | 11627 |
| 950 | 500 | 19 | 91.0% | 100% | 25702 | 6895 |
| 950 | 800 | 31 | 100% | 100% | 40296 | 10754 |
| 1000 | 500/800 | 19 | 97.4% | 100% | 28104 | 7403 |
| 1000 | 1000 | 31 | 100% | 100% | 41015 | 11136 |
| 1000 | 1200 | 37 | 100% | 100% | 48015 | 13020 |

**结论**：
1. **19 点配置圆盘覆盖只有 83~97.4%**（边界薄环空洞）→ 会偶发漏检（对应 99.1~99.3%）；
2. **100% 覆盖至少需要 31 点**；
3. 在 100% 覆盖的配置中，**a=900 / 余量=800 最快**（路径较短），故选为默认。

25 案例确认（a=900/余量800）：三档定向比例 **全部 100% 清除**，估计时间 9493~10079 s。

## 结构化 JSON 日志

每次运行会写两个文件：

- `robot4_log_<时间戳>.jsonl` —— 逐条动作日志（请求/响应/accepted/virtual_time_s）；
- `robot4_log_<时间戳>.summary.json` —— **结构化 JSON 汇总**：

```json
{
  "team_no": "...", "base_url": "...",
  "config": {"mesh_a": 900.0, "mesh_margin": 800.0, "on_way_delta": 300.0, "r_clear": 20.0},
  "final_virtual_time_s": ..., "measure_count": ...,
  "clear_attempt_count": ..., "clear_success_count": ..., "clear_failure_count": ...,
  "movement_distance_m": ...,
  "robot": {"mesh_points": 31, "mesh_triangles": 42, "cover_triangles": 42,
            "cleared_count": ..., "found_channels": ..., "excluded_channels": ...,
            "channels": {"1": {"state": "cleared", "bearings": 2, "near": false, "no_signal_points": 5}, ...}}
}
```

论文中「被清除干扰源个数」应取 `clear_success_count`（不是 `clear_attempt_count`）。

## 调试工具

- `diag4.py`：统计漏检源的位置/指向/可测网格点数；
- `diag4b.py`：追踪单个漏检源的失败环节（定位估计误差、二分归航误差）。


## 日志（问题3 / 问题4 统一）

每次实机运行会在**本程序目录下的 `logs/`** 生成两个文件：

- `p4_log_<时间戳>.jsonl` —— 逐条动作日志（`/enter` `/measure` `/clear` `/exit` 的请求、响应、`accepted`、`virtual_time_s`）；
- `p4_log_<时间戳>.summary.json` —— **结构化 JSON 汇总**（两问同 schema）：

```json
{
  "problem": 4,
  "team_no": "...", "base_url": "...",
  "config": { "...": "..." },
  "final_virtual_time_s": 0.0,
  "movement_distance_m": 0.0,
  "measure_count": 0,
  "clear_attempt_count": 0,
  "clear_success_count": 0,
  "clear_failure_count": 0,
  "robot": {
    "cleared_count": 0,
    "found_channels": 0, "excluded_channels": 0,
    "channels": { "1": { "state": "cleared", "bearings": 2 } }
  }
}
```

论文中「被清除干扰源个数」取 `clear_success_count`（不是 `clear_attempt_count`）；
「平均定位清除时间」= `final_virtual_time_s / clear_success_count`。

> 日志文件默认**不入 git**（见根目录 `.gitignore`），只保留在本地 `logs/` 便于分析。
