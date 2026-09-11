"""执行官方协议的最小通信链路测试。"""

from __future__ import annotations

import argparse
import sys

from simulator_client import BASE_URL, LOG_PATH, ROBOT_ID, ActionResult, SimulatorClient


PLACEHOLDER_ROBOT_ID = "在这里填写参赛队号"


def classify(
    results: list[ActionResult], *, interface_confirmed_open: bool
) -> str:
    enter_ok = any(r.endpoint == "/enter" and r.ok for r in results)
    measure_ok = any(r.endpoint == "/measure" and r.ok for r in results)
    exit_ok = any(r.endpoint == "/exit" and r.ok for r in results)
    if enter_ok and measure_ok and exit_ok:
        return "PASS"
    if any(r.received_http_response for r in results):
        return "PARTIAL"
    if interface_confirmed_open:
        return "FAIL"
    return "NOT_READY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2026 国赛 B 题模拟器 HTTP+JSON 最小连通性测试"
    )
    parser.add_argument(
        "--interface-open",
        action="store_true",
        help="仅在你已确认问题3演练接口开放后使用；无 HTTP 响应将判为 FAIL。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not ROBOT_ID.strip() or ROBOT_ID == PLACEHOLDER_ROBOT_ID:
        print("SETUP_REQUIRED: 请先在 simulator_client.py 顶部填写真实参赛队号 ROBOT_ID。")
        print("尚未发送任何 HTTP 请求，也未对通信链路作失败判断。")
        return 2

    print(f"模拟器地址: {BASE_URL}")
    print(f"日志文件: {LOG_PATH}")
    print("请求将严格串行执行；网络异常时同一动作最多自动重试 1 次。")

    results: list[ActionResult] = []
    with SimulatorClient() as client:
        enter_result = client.enter()
        results.append(enter_result)

        if enter_result.ok:
            remaining = (enter_result.response_json or {}).get(
                "remaining_real_duration_s"
            )
            print(f"remaining_real_duration_s: {remaining}")
            if not isinstance(remaining, (int, float)) or isinstance(remaining, bool):
                print("警告: /enter 成功响应缺少有效的 remaining_real_duration_s。")

            communication_uncertain = False
            rate_limited = False
            for action in (
                lambda: client.measure((0, 0), 1),
                lambda: client.measure((100, 0), 2),
                lambda: client.clear((100, 0), 2),
            ):
                result = action()
                results.append(result)
                if not result.received_http_response:
                    communication_uncertain = True
                    print("未收到完整 HTTP 响应；为保持严格串行，不再发送新动作。")
                    break
                if result.http_status == 429:
                    rate_limited = True
                    print("收到 HTTP 429；为避免异常高频请求，不再发送新动作。")
                    break

            if not communication_uncertain and not rate_limited:
                results.append(client.exit())
        else:
            print("/enter 未成功 accepted=true，后续动作序列不执行。")

        outcome = classify(
            results, interface_confirmed_open=args.interface_open
        )
        client.write_summary(outcome, results)

    print("=" * 72)
    print(f"最终判断: {outcome}")
    if outcome == "PASS":
        print("/enter、至少一次 /measure、/exit 均为 HTTP 200 且 accepted=true。")
    elif outcome == "PARTIAL":
        print("能够连接模拟器，但至少一个必要动作被业务规则或协议检查拒绝。")
    elif outcome == "FAIL":
        print("已按命令行参数确认接口开放，但仍未获得任何 HTTP 响应或合法通信。")
    else:
        print("未确认演练接口开放，因此不把 Connection refused 判为代码失败。")
        print("请先在模拟器中开始问题3演练测试，并等待5秒倒计时结束。")

    return {"PASS": 0, "PARTIAL": 1, "FAIL": 1, "NOT_READY": 2}[outcome]


if __name__ == "__main__":
    sys.exit(main())

