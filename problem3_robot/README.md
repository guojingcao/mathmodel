# 问题3 机器人程序使用说明

`robot.py` 是问题3（全向干扰源自动定位与清除）的模拟器对接程序，按《汇总版》最终模型设计实现。

## 模型结构（三层 + 清除执行）

```
保证层   —— 多环覆盖(原点+正六边形7点) + 有界角域定位(半平面交可行域)
状态层   —— 频道状态更新(待排查/已发现/可清除/已清除/已排除) + 可行域维护
调度层   —— 动态任务队列(已定位频道→清除任务, 就近调度执行)
清除执行 —— 最小覆盖圆≤20m保证清除 / 归航精定位
```

**已按消融实验结论，删除 DOP、极小极大、概率排序等复杂模块**（保留覆盖保证 + 任务队列调度）。

- **保证覆盖**：原点 + 半径 1200m 圆周上 6 个正六边形顶点（共 7 个搜索点），最坏覆盖距离约 969m < 1000m，保证任何全向源至少被一个搜索点测到。
- **蛇形扫描**：相邻搜索点用相反频道顺序（1→20 / 20→1），避免跨点频道切换；已发现频道仍继续测量（获取免费交会）。
- **可行域**：每次示向度 → ±1° 楔形（两个半平面），联合可行域 = 半平面交（凸多边形）。
- **清除判据**：可行域最小覆盖圆半径 ≤ 20m 时在圆心清除（保证成功）；返回 `near`（≤5m）直接清除。
- **定位兜底**：最小二乘交会 → 垂直补测（仅交会角≥25° 时采用）→ 沿示向度二分归航。
- **任务队列调度**：每个已定位频道生成一个清除任务，最近邻生成初始清除路径，再用 **2-opt 局部交换**缩短总移动距离。
- **终止**：20 个频道全部"已排除"或"已清除"后调用 `/exit`。

## 运行前

1. 在模拟器中**在线登录**，启动"**问题3演练测试**"或"**问题3正式测试**"，等待倒计时结束、界面提示机器狗接口就绪。

## 运行

参赛队号通过命令行参数传入（不写死在代码里）：

```bash
python robot.py --robot-id 你的参赛队号
```

也可用环境变量，或指定接口地址/端口：

```bash
# 环境变量
set ROBOT_ID=你的参赛队号   # Windows
python robot.py

# 指定接口地址(模拟器修改了端口时)
python robot.py --robot-id 你的队号 --base-url http://127.0.0.1:2026
```

参数说明：

| 参数 | 说明 | 默认 |
|---|---|---|
| `--robot-id` | 参赛队号（必填，或用环境变量 `ROBOT_ID`） | 无 |
| `--base-url` | 模拟器接口地址 | `http://127.0.0.1:2026` |
| `--arena-id` | 竞技场ID | `default` |
| `--log-file` | 本地行为日志输出路径 | `robot_log_<时间戳>.jsonl` |

程序会依次：`/enter` → 7 点蛇形扫描 → 定位 → 任务队列就近清除 → `/exit`，并在控制台打印过程。

**本地日志**：程序会把每条动作（`/enter`、`/measure`、`/clear`、`/exit`）的请求、响应、`accepted`、`virtual_time_s` 记录为 JSON Lines 文件（`.jsonl`），末尾附汇总（总动作数、最终虚拟时刻、检测/清除次数）。演练结束后可用它核对定位效果与耗时。

## 关键实现细节

- **HTTP 客户端**：串行发送，逐条等待响应；同时检查 HTTP 状态码与 `accepted`；网络异常复用原 `request_id` 重试。
- **幂等**：每个新动作使用新 `request_id`。
- **时间控制**：读取 `/enter` 返回的 `remaining_real_duration_s`，不硬编码 1200s。

## 离线自检（不连模拟器）

`selfcheck.py` 用等价仿真环境 mock 掉 HTTP 层，验证策略逻辑：

```bash
python selfcheck.py
```

结果：**1000 随机案例清除比例 100.000%（1000/1000 全清），平均移动距离约 21284m**。

`diag_robot.py` 用于定位漏清源（调试用）。



## 日志（问题3 / 问题4 统一）

每次实机运行会在**本程序目录下的 `logs/`** 生成两个文件：

- `p3_log_<时间戳>.jsonl` —— 逐条动作日志（`/enter` `/measure` `/clear` `/exit` 的请求、响应、`accepted`、`virtual_time_s`）；
- `p3_log_<时间戳>.summary.json` —— **结构化 JSON 汇总**（两问同 schema）：

```json
{
  "problem": 3,
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
