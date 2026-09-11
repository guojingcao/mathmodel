# 2026 国赛 B 题模拟器最小通信测试

本目录只验证问题 3 演练模拟器的 HTTP+JSON 通信链路，不包含问题 1–4 的建模、定位或搜索算法。

## 准备

1. 安装 Python 3.10 或更高版本和 `requests`：

   ```powershell
   python -m pip install requests
   ```

2. 打开 `simulator_client.py`，在文件顶部填写当前登录参赛队号：

   ```python
   ROBOT_ID = "在这里填写参赛队号"
   ```

3. 在模拟器中在线登录，启动“问题 3 演练测试”，等待 5 秒倒计时结束，直到界面显示机器狗接口已经就绪。

默认地址是 `http://127.0.0.1:2026`。模拟器只监听本机回环地址，测试程序必须和模拟器运行在同一台电脑上。

## 运行

在本目录执行：

```powershell
python test_connection.py
```

程序严格串行发送以下动作：

1. `POST /enter`
2. `POST /measure`，位置 `(0, 0)`，频道 `1`
3. `POST /measure`，位置 `(100, 0)`，频道 `2`
4. `POST /clear`，位置 `(100, 0)`，频道 `2`
5. `POST /exit`

只有 `/enter` 返回 HTTP 200 且 `accepted=true` 后，程序才发送后续动作。若某个动作在有限重试后仍没有完整 HTTP 响应，程序不会发送新的动作，避免前一动作状态未知时破坏串行约束。收到 HTTP 429 后也会停止发送新动作。

若你已经确认界面显示接口就绪，但仍无法通信，可运行：

```powershell
python test_connection.py --interface-open
```

该参数只影响最终分类：确认接口开放后仍没有任何 HTTP 响应，结果为 `FAIL`。

## 输出与日志

每个动作都会打印：

- `request_id`
- HTTP 状态
- `accepted`
- `virtual_time_s`
- `measure_result` 或 `clear_result`（适用时）
- 完整响应 JSON

每次 HTTP 尝试都追加写入 `logs/connectivity_test.jsonl`。记录包含请求 JSON、响应 JSON、异常和尝试次数。重复运行会保留旧日志；需要单独归档时，请在运行前手工重命名旧文件。

## request_id 与重试

每个新动作由 UUID 生成新的 `request_id`。只有发生 `ConnectionError` 或 `Timeout` 时，同一动作才最多自动重试一次；重试间隔为 1 秒，并复用同一个 `request_id` 以及同一份已经编码好的 UTF-8 JSON 请求体。HTTP 400、409、415、429 和 `accepted=false` 不会被自动重试。

## 结果判定

- `PASS`：`/enter`、至少一次 `/measure`、`/exit` 都返回 HTTP 200 且 `accepted=true`。
- `PARTIAL`：已经收到模拟器 HTTP 响应，但必要动作未全部成功，或被业务规则/协议检查拒绝。
- `FAIL`：使用 `--interface-open` 明确确认演练接口已开放后，仍无法取得任何 HTTP 响应或合法通信。
- `NOT_READY`：没有收到 HTTP 响应，同时未声明接口已开放。此时请先在模拟器中开始问题 3 演练测试，并等待 5 秒倒计时结束。
- `SETUP_REQUIRED`：尚未填写 `ROBOT_ID`，程序不会发送请求。

`/enter` 成功后，程序读取并打印 `remaining_real_duration_s`，不会假设本局一定剩余 1200 秒。

## 常见状态

- HTTP 400：JSON 结构、缺失字段、类型、标识符、频道或坐标不合法。
- HTTP 409：同一 `request_id` 被用于不同内容，或出现并发动作。
- HTTP 415：`Content-Type` 或 `Content-Encoding` 不符合协议。
- HTTP 429：触发连接/无效流量保护，或本局幂等记录达到上限。
- HTTP 200 且 `accepted=false`：JSON 可解析，但模拟器业务状态拒绝动作、存在未知字段，或 `arena_id`/`robot_id` 不匹配。

所有成功动作必须同时满足 HTTP 200 和 `accepted is True`，不能只检查其中一项。
