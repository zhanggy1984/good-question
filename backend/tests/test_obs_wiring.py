"""obs_sdk 观测旁路单测（§11.3 gq llm_call 覆盖：stream_chat 包装层）。

直通路径（_obs_sdk() 返 None，即未装/未 init）由现有 LLM 流测试在 mock httpx 下天然覆盖
（94 个用例全绿即为无观测时调用语义不变的证明），本文件只验证**启用观测**分支：
- 正常流末：record_llm ok + usage token 透传 + duration 计算
- 流中异常：record_llm error 先记再抛（§2.4 前提）、error_type 按异常分型
全程 monkeypatch _obs_sdk/_stream_chat_http，不触真实 HTTP、不依赖 sdk 安装。
"""
import httpx
import pytest

from services import llm_service
from services.llm_service import stream_chat


class _FakeObs:
    """假 obs_sdk：只记录 record_llm 调用（无 init 状态校验）。"""

    def __init__(self):
        self.calls = []

    def record_llm(self, model, status, *, duration_ms, error_type=None,
                   error_msg=None, usage=None):
        self.calls.append({
            "model": model, "status": status, "duration_ms": duration_ms,
            "error_type": error_type, "error_msg": error_msg, "usage": usage,
        })


def test_obs_records_ok_with_usage(monkeypatch):
    fake = _FakeObs()
    monkeypatch.setattr(llm_service, "_obs_sdk", lambda: fake)

    def _fake_stream(messages, tools=None):
        yield {"type": "content", "content": "hi"}
        yield {"type": "usage", "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                          "total_tokens": 15}}

    monkeypatch.setattr(llm_service, "_stream_chat_http", _fake_stream)
    out = list(stream_chat([{"role": "user", "content": "hi"}]))
    assert [e["type"] for e in out] == ["content", "usage"], "观测不得吞/改事件"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["status"] == "ok"
    assert call["usage"]["total_tokens"] == 15, "usage 应从流末 chunk 透传"
    assert call["model"] == llm_service.settings.deepseek_model
    assert call["duration_ms"] is not None and call["duration_ms"] >= 0
    assert call["error_type"] is None and call["error_msg"] is None


def test_obs_records_error_then_raises(monkeypatch):
    """流中裸异常：先记 error 再抛（§2.4 前提；消费端以 llm_call error 红显）。"""
    fake = _FakeObs()
    monkeypatch.setattr(llm_service, "_obs_sdk", lambda: fake)

    def _boom(messages, tools=None):
        yield {"type": "content", "content": "x"}
        raise httpx.TimeoutException("上游超时")

    monkeypatch.setattr(llm_service, "_stream_chat_http", _boom)
    with pytest.raises(httpx.TimeoutException):
        list(stream_chat([]))
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["status"] == "error"
    assert call["error_type"] == "TIMEOUT", "httpx 超时应分型 TIMEOUT"
    assert call["error_msg"] == "上游超时"
    assert call["usage"] is None


def test_obs_untouched_on_http_status_error(monkeypatch):
    """非 2xx（resp.raise_for_status 抛 HTTPStatusError）→ HTTP_<code> 分型。"""
    fake = _FakeObs()
    monkeypatch.setattr(llm_service, "_obs_sdk", lambda: fake)

    import httpx as _hx

    def _boom(messages, tools=None):
        yield {"type": "content", "content": "y"}
        resp = _hx.Response(429, request=_hx.Request("POST", "http://x/chat"))
        raise _hx.HTTPStatusError("rate limited", request=resp.request, response=resp)

    monkeypatch.setattr(llm_service, "_stream_chat_http", _boom)
    with pytest.raises(_hx.HTTPStatusError):
        list(stream_chat([]))
    assert fake.calls[0]["status"] == "error"
    assert fake.calls[0]["error_type"] == "HTTP_429"
