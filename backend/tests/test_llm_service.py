"""LLM 流式调用测试（llm_service.stream_chat / stream_round1_with_retry，不连外部服务）

流式解析与首轮重试逻辑从 chat_service 下沉到资源层 llm_service，测试随函数迁移：
SSE 行解析、HTTP 错误显式化、瞬时错误退避重试，均直接测本层函数，
mock httpx.stream / patch 本层 stream_chat，无需真实 DeepSeek。
"""
import sys

import httpx
import pytest

sys.path.insert(0, "/app")

from services import llm_service as ls

from config import settings

# 工具 schema 占位：流式解析不校验 schema 内容，仅透传进 payload（层间依赖抽象——
# 资源层不持有 chat_service 的具体工具 schema，测试同理）
_FAKE_TOOLS = [{"type": "function", "function": {"name": "hybrid_retrieve", "parameters": {}}}]


def _stream_lines_resp(lines, status_code=200):
    """把 SSE data 行包装成 stream_chat 需要的 httpx 响应对象

    status_code 可指定非 2xx：stream_chat 现做 resp.raise_for_status()，
    fake 须提供 status_code 属性（默认 200 兼容既有用例）。
    """
    class _Resp:
        def iter_lines(self):
            return iter(lines)

        def raise_for_status(self):
            # 镜像 httpx.Response.raise_for_status：非 2xx 抛 HTTPStatusError（供调用方 except）
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"HTTP {self.status_code}", request=None, response=None)

    # class 体内不构成闭包（class 作用域不捕获函数参数），须在类定义后经函数作用域赋值
    _Resp.status_code = status_code
    class _Stream:
        def __enter__(self):
            return _Resp()
        def __exit__(self, *a):
            return False
    return _Stream()


def test_stream_chat_parses_usage_and_delta(monkeypatch):
    """stream_chat：SSE 行解析——content/reasoning 增量 + 流末 usage chunk（不取 delta 报错）"""
    lines = [
        'data: {"choices":[{"delta":{"content":"你"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"思考"}}]}',
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
        'data: [DONE]',
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _stream_lines_resp(lines))

    events = list(ls.stream_chat([{"role": "user", "content": "问题"}]))
    assert events[0] == {"type": "content", "content": "你"}
    assert events[1] == {"type": "reasoning", "content": "思考"}
    assert events[2]["type"] == "usage"
    assert events[2]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_stream_chat_parses_tool_calls(monkeypatch):
    """stream_chat：delta.tool_calls 按 index 分片累加，finish_reason=="tool_calls" 触发 flush"""
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"hybrid_retrieve","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"query\\":\\""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"工资发放日"}}]},"finish_reason":"tool_calls"}]}',
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}',
        'data: [DONE]',
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _stream_lines_resp(lines))

    events = list(ls.stream_chat([{"role": "user", "content": "问题"}], tools=_FAKE_TOOLS))
    tc = next(e for e in events if e["type"] == "tool_call")
    call = tc["tool_calls"][0]
    assert call["id"] == "call_1", "tool_call id 应完整（首个分片携带）"
    assert call["function"]["name"] == "hybrid_retrieve"
    assert call["function"]["arguments"] == '{"query":"工资发放日', "arguments 应按分片顺序拼接"
    usage = next(e for e in events if e["type"] == "usage")
    assert usage["usage"]["total_tokens"] == 8


def test_stream_chat_tool_call_flush_on_done(monkeypatch):
    """stream_chat：个别实现不返回 finish_reason 时，[DONE] 后兜底 flush tool_calls"""
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"hybrid_retrieve","arguments":"{\\"query\\":\\"a\\"}"}}]}}]}',
        'data: [DONE]',
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _stream_lines_resp(lines))

    events = list(ls.stream_chat([{"role": "user", "content": "问题"}], tools=_FAKE_TOOLS))
    tc = next(e for e in events if e["type"] == "tool_call")
    assert tc["tool_calls"][0]["function"]["arguments"] == '{"query":"a"}'


def test_stream_chat_http_error_raises(monkeypatch):
    """stream_chat：HTTP 非 2xx（401/429/500）显式抛异常，不再静默空返回（前端可见明确 error）"""
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _stream_lines_resp([], status_code=401))
    with pytest.raises(httpx.HTTPStatusError):
        list(ls.stream_chat([{"role": "user", "content": "问题"}]))


def test_stream_round1_retry_transient(monkeypatch):
    """首轮瞬时错误（429）且未 yield 任何事件：整体退避重试一次，成功后续流"""
    from types import SimpleNamespace
    calls = {"n": 0}

    def _boom_then_ok(messages, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.HTTPStatusError(
                "429 限流", request=object(), response=SimpleNamespace(status_code=429)
            )
        yield {"type": "content", "content": "OK"}
        yield {"type": "usage", "usage": {"total_tokens": 1}}

    monkeypatch.setattr(settings, "chat_llm_max_attempts", 2)
    monkeypatch.setattr(settings, "chat_llm_retry_backoff_seconds", 0)
    monkeypatch.setattr(ls, "stream_chat", _boom_then_ok)
    events = list(ls.stream_round1_with_retry([{"role": "user", "content": "问题"}]))
    assert calls["n"] == 2, "总调用次数上限 2 = 首次失败后重试一次"
    assert [e["type"] for e in events] == ["content", "usage"]


def test_stream_round1_no_retry_on_401(monkeypatch):
    """客户端错误（401）非瞬时：不可重试（重试无意义且可能计费），立即抛给上层"""
    from types import SimpleNamespace
    calls = {"n": 0}

    def _boom_401(messages, tools=None):
        calls["n"] += 1
        raise httpx.HTTPStatusError(
            "401 未授权", request=object(), response=SimpleNamespace(status_code=401)
        )

    monkeypatch.setattr(settings, "chat_llm_max_attempts", 2)
    monkeypatch.setattr(ls, "stream_chat", _boom_401)
    with pytest.raises(httpx.HTTPStatusError):
        list(ls.stream_round1_with_retry([{"role": "user", "content": "问题"}]))
    assert calls["n"] == 1, "401 非瞬时错误不应重试"


def test_stream_round1_no_retry_after_yield(monkeypatch):
    """已 yield 事件后出错不可整体重试：重试会事件序错乱/重复，即使 5xx 也直接抛"""
    from types import SimpleNamespace
    calls = {"n": 0}

    def _yield_then_boom(messages, tools=None):
        calls["n"] += 1
        yield {"type": "content", "content": "部分内容"}
        raise httpx.HTTPStatusError(
            "500 服务端错误", request=object(), response=SimpleNamespace(status_code=500)
        )

    monkeypatch.setattr(settings, "chat_llm_max_attempts", 2)
    monkeypatch.setattr(ls, "stream_chat", _yield_then_boom)
    got = []
    with pytest.raises(httpx.HTTPStatusError):
        for ev in ls.stream_round1_with_retry([{"role": "user", "content": "问题"}]):
            got.append(ev)
    assert calls["n"] == 1, "已 yield 事件后即使 5xx 也不重试"
    assert [e["type"] for e in got] == ["content"]


def test_stream_round1_max_attempts_follows_config(monkeypatch):
    """总调用次数 = 配置 chat_llm_max_attempts（含首次）：配置 5、持续 429 时恰好调用 5 次"""
    from types import SimpleNamespace
    calls = {"n": 0}

    def _always_429(messages, tools=None):
        calls["n"] += 1
        raise httpx.HTTPStatusError(
            "429 限流", request=object(), response=SimpleNamespace(status_code=429)
        )

    monkeypatch.setattr(settings, "chat_llm_max_attempts", 5)
    monkeypatch.setattr(settings, "chat_llm_retry_backoff_seconds", 0)
    monkeypatch.setattr(ls, "stream_chat", _always_429)
    with pytest.raises(httpx.HTTPStatusError):
        list(ls.stream_round1_with_retry([{"role": "user", "content": "问题"}]))
    assert calls["n"] == 5, "总调用次数应等于配置值（含首次）"
