"""2026 国赛 B 题模拟器的最小 HTTP+JSON 客户端。"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


# 按官方要求，只修改这里即可；不要在动作函数中写死参赛队号。
ROBOT_ID = "202604004013"

BASE_URL = "http://127.0.0.1:2026"
ARENA_ID = "default"
LOG_PATH = Path(__file__).resolve().parent / "logs" / "connectivity_test.jsonl"

# 自动重试不是无限重试。一次新动作最多额外重试 1 次，间隔 1 秒。
# 重试时复用同一个 request_id 和完全相同的 UTF-8 JSON 请求体字节。
MAX_NETWORK_RETRIES = 1
NETWORK_RETRY_DELAY_S = 1.0
CONNECT_TIMEOUT_S = 2.0
READ_TIMEOUT_S = 5.0


HTTP_MESSAGES = {
    400: "请求 JSON、字段类型、标识符、频道或坐标不合法；修正后可复用该 request_id。",
    409: "同一 request_id 被用于不同动作/内容，或不同动作被并发发送。",
    415: "Content-Type 或 Content-Encoding 不受支持。",
    429: "触发连接/无效流量保护或本局幂等记录达到上限；本客户端不会自动重试该 HTTP 响应。",
}


@dataclass(frozen=True)
class ActionResult:
    """一次逻辑动作的最终结果；可能没有收到 HTTP 响应。"""

    endpoint: str
    request_id: str
    request_json: dict[str, Any]
    attempts: int
    http_status: int | None = None
    accepted: bool | None = None
    response_json: dict[str, Any] | None = None
    raw_response_text: str | None = None
    error_kind: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.http_status == 200 and self.accepted is True

    @property
    def received_http_response(self) -> bool:
        return self.http_status is not None


class SimulatorClient:
    """串行执行官方定义的四类动作。"""

    def __init__(
        self,
        robot_id: str = ROBOT_ID,
        base_url: str = BASE_URL,
        log_path: Path | str = LOG_PATH,
        max_network_retries: int = MAX_NETWORK_RETRIES,
    ) -> None:
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.log_path = Path(log_path)
        self.max_network_retries = max(0, int(max_network_retries))

        self._request_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._session = requests.Session()
        # 模拟器只监听本机回环地址，不应经过系统代理。
        self._session.trust_env = False
        self._headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "identity",
        }

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "SimulatorClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def enter(self) -> ActionResult:
        return self._new_action("/enter")

    def measure(self, position: tuple[float, float], channel: int) -> ActionResult:
        return self._new_action("/measure", position=position, channel=channel)

    def clear(self, position: tuple[float, float], channel: int) -> ActionResult:
        return self._new_action("/clear", position=position, channel=channel)

    def exit(self) -> ActionResult:
        return self._new_action("/exit")

    def write_summary(self, outcome: str, results: list[ActionResult]) -> None:
        self._append_log(
            {
                "record_type": "summary",
                "outcome": outcome,
                "action_count": len(results),
                "request_ids": [result.request_id for result in results],
                "logged_at_unix_ms": time.time_ns() // 1_000_000,
            }
        )

    def _new_action(
        self,
        endpoint: str,
        *,
        position: tuple[float, float] | None = None,
        channel: int | None = None,
    ) -> ActionResult:
        request_id = f"{endpoint.removeprefix('/')}-{uuid.uuid4()}"
        payload: dict[str, Any] = {
            "arena_id": ARENA_ID,
            "robot_id": self.robot_id,
            "request_id": request_id,
        }
        if position is not None:
            x, y = position
            payload["position"] = {"x": x, "y": y}
        if channel is not None:
            payload["channel"] = channel

        # 只序列化一次。后续网络重试直接复用 body，确保内容逐字节相同。
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return self._post_serial(endpoint, payload, body)

    def _post_serial(
        self,
        endpoint: str,
        payload: dict[str, Any],
        body: bytes,
    ) -> ActionResult:
        # 即使调用方误用多线程，这把锁也保证不会并发发送不同动作。
        with self._request_lock:
            return self._post_with_bounded_network_retry(endpoint, payload, body)

    def _post_with_bounded_network_retry(
        self,
        endpoint: str,
        payload: dict[str, Any],
        body: bytes,
    ) -> ActionResult:
        total_attempts = 1 + self.max_network_retries
        request_id = str(payload["request_id"])

        for attempt in range(1, total_attempts + 1):
            try:
                response = self._session.post(
                    self.base_url + endpoint,
                    data=body,
                    headers=self._headers,
                    timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
                )
            except requests.exceptions.ConnectionError as exc:
                result = ActionResult(
                    endpoint=endpoint,
                    request_id=request_id,
                    request_json=payload,
                    attempts=attempt,
                    error_kind="ConnectionError",
                    error_message=str(exc),
                )
                self._log_attempt(result, attempt)
                if attempt < total_attempts:
                    self._print_network_retry(result, attempt, total_attempts)
                    time.sleep(NETWORK_RETRY_DELAY_S)
                    continue
                self._print_result(result)
                print("请先在模拟器中开始问题3演练测试，并等待5秒倒计时结束。")
                return result
            except requests.exceptions.Timeout as exc:
                result = ActionResult(
                    endpoint=endpoint,
                    request_id=request_id,
                    request_json=payload,
                    attempts=attempt,
                    error_kind="Timeout",
                    error_message=str(exc),
                )
                self._log_attempt(result, attempt)
                if attempt < total_attempts:
                    self._print_network_retry(result, attempt, total_attempts)
                    time.sleep(NETWORK_RETRY_DELAY_S)
                    continue
                self._print_result(result)
                return result

            response_json: dict[str, Any] | None
            raw_text: str | None = None
            try:
                parsed = response.json()
                response_json = parsed if isinstance(parsed, dict) else None
                if response_json is None:
                    raw_text = response.text
            except requests.exceptions.JSONDecodeError:
                response_json = None
                raw_text = response.text

            accepted_value = (
                response_json.get("accepted") if response_json is not None else None
            )
            accepted = accepted_value if isinstance(accepted_value, bool) else None
            result = ActionResult(
                endpoint=endpoint,
                request_id=request_id,
                request_json=payload,
                attempts=attempt,
                http_status=response.status_code,
                accepted=accepted,
                response_json=response_json,
                raw_response_text=raw_text,
                error_kind=(
                    "InvalidJSONResponse" if response_json is None else None
                ),
                error_message=(
                    "响应体不是 JSON 对象。" if response_json is None else None
                ),
            )
            self._log_attempt(result, attempt)
            self._print_result(result)
            return result

        raise AssertionError("unreachable")

    def _log_attempt(self, result: ActionResult, attempt: int) -> None:
        self._append_log(
            {
                "record_type": "http_attempt",
                "endpoint": result.endpoint,
                "request_id": result.request_id,
                "attempt": attempt,
                "request_json": result.request_json,
                "http_status": result.http_status,
                "accepted": result.accepted,
                "virtual_time_s": (
                    result.response_json.get("virtual_time_s")
                    if result.response_json is not None
                    else None
                ),
                "measure_result": (
                    result.response_json.get("measure_result")
                    if result.response_json is not None
                    else None
                ),
                "clear_result": (
                    result.response_json.get("clear_result")
                    if result.response_json is not None
                    else None
                ),
                "response_json": result.response_json,
                "raw_response_text": result.raw_response_text,
                "error_kind": result.error_kind,
                "error_message": result.error_message,
                "logged_at_unix_ms": time.time_ns() // 1_000_000,
            }
        )

    def _append_log(self, record: dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self._log_lock:
            with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")

    @staticmethod
    def _print_network_retry(
        result: ActionResult, attempt: int, total_attempts: int
    ) -> None:
        print(
            f"[{result.endpoint}] {result.error_kind}；"
            f"将在 1 秒后进行第 {attempt + 1}/{total_attempts} 次尝试。"
        )
        print(f"重试继续使用 request_id: {result.request_id}")

    @staticmethod
    def _print_result(result: ActionResult) -> None:
        response = result.response_json or {}
        print("=" * 72)
        print(f"动作: {result.endpoint}")
        print(f"request_id: {result.request_id}")
        print(f"HTTP状态: {result.http_status if result.http_status is not None else '无响应'}")
        print(f"accepted: {result.accepted}")
        print(f"virtual_time_s: {response.get('virtual_time_s')}")
        if result.endpoint == "/measure":
            print(f"measure_result: {response.get('measure_result')}")
        if result.endpoint == "/clear":
            print(f"clear_result: {response.get('clear_result')}")
        print("完整响应JSON:")
        if result.response_json is not None:
            print(json.dumps(result.response_json, ensure_ascii=False, indent=2))
        else:
            print("null")
            if result.raw_response_text is not None:
                print(f"原始响应体: {result.raw_response_text}")
        if result.error_kind:
            print(f"异常: {result.error_kind}: {result.error_message}")
        if result.http_status in HTTP_MESSAGES:
            print(f"说明: {HTTP_MESSAGES[result.http_status]}")
        elif result.http_status is not None and result.http_status != 200:
            print("说明: 收到非 200 HTTP 状态，动作不能判为成功。")
        if result.http_status == 200 and result.accepted is False:
            print("说明: 模拟器业务规则拒绝了该动作，虚拟时钟不推进。")
        if result.http_status == 200 and result.accepted is None:
            print("说明: 响应缺少布尔型 accepted 字段，不能判为成功。")
