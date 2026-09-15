"""obs_sdk 观测旁路单测（§11.3 gq llm_call 覆盖：stream_chat 包装层）。

直通路径（_obs_sdk() 返 None，即未装/未 init）由现有 LLM 流测试在 mock httpx 下天然覆盖
（94 个用例全绿即为无观测时调用语义不变的证明），本文件只验证**启用观测**分支：
- 流穷尽：record_llm ok + usage token 透传 + duration 计算
- 流中异常：record_llm error 先记再抛（§2.4 前提）、error_type 按异常分型
- 流中途被弃用（客户端断连 → gen.close()）：record_llm ok 仍须记账（§13.0 #3 无黑洞）
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
    assert call["error_type"] == "llm_timeout", "httpx 超时应分型 llm_timeout（平台白名单词）"
    assert call["error_msg"] == "上游超时"
    assert call["usage"] is None


def test_obs_records_ok_on_midstream_close(monkeypatch):
    """客户端中途断连（chat.py 的 gen.close()）：ok 收口仍须记账。

    GeneratorExit 落在 yield 点且承 BaseException，except Exception 接不住 ⇒ 收口若写在
    try 之后会静默全丢。**真机对照**（2026-09-15 gq 容器）：截断驱动的请求只有 request、
    零 llm_call；完整 drain 的请求 request + llm_call ok —— 本用例即该对照的单测复现。
    """
    fake = _FakeObs()
    monkeypatch.setattr(llm_service, "_obs_sdk", lambda: fake)

    def _fake_stream(messages, tools=None):
        yield {"type": "content", "content": "hello"}
        yield {"type": "usage", "usage": {"total_tokens": 15}}

    monkeypatch.setattr(llm_service, "_stream_chat_http", _fake_stream)
    gen = stream_chat([])
    assert next(gen)["content"] == "hello", "只驱动一步 = 断连现场"
    gen.close()  # 等价于 chat.py _stream_with_disconnect_check 的 finally 分支

    assert len(fake.calls) == 1, "一次调用一条账（既不能丢，也不能 finally 重复记）"
    call = fake.calls[0]
    assert call["status"] == "ok", "调用已成功（HTTP 200 且已产出增量）⇒ ok，不是 error"
    assert call["usage"] is None, "断连早于 usage chunk ⇒ usage 空，但状态仍为 ok"


def test_obs_untouched_on_http_status_error(monkeypatch):
    """非 2xx（resp.raise_for_status 抛 HTTPStatusError）→ 仅 429 单列，其余归 llm_other。"""
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
    assert fake.calls[0]["error_type"] == "llm_rate_limit"


# 平台错误分类白名单（§4.3 L1+L2 词表）。llm_call 的 error_type 若落在此集合外，
# 平台不产生回流候选 ⇒ 值域卫生是本表唯一护栏。
_PLATFORM_ERR_WHITELIST = {
    "llm_timeout", "llm_rate_limit", "llm_connection", "llm_context_exceeded",
    "llm_empty_response", "llm_parse_error", "llm_other",
    "llm_interface_business", "external_non_llm", "db_error", "redis_error",
}


def _status_error(code: int) -> httpx.HTTPStatusError:
    resp = httpx.Response(code, request=httpx.Request("POST", "http://x/chat"))
    return httpx.HTTPStatusError("boom", request=resp.request, response=resp)


@pytest.mark.parametrize("exc,expected", [
    (httpx.TimeoutException("上游超时"), "llm_timeout"),
    (httpx.ConnectError("连接失败"), "llm_connection"),
    (_status_error(429), "llm_rate_limit"),
    (_status_error(500), "llm_other"),
    (_status_error(401), "llm_other"),
    (ValueError("别的东西"), "llm_other"),
])
def test_llm_error_type_maps_into_platform_whitelist(exc, expected):
    """每条分支都必须产出白名单内的词（原实现透出 HTTP_{code}/TIMEOUT/NETWORK，均不在册）。"""
    got = llm_service.llm_error_type(exc)
    assert got == expected
    assert got in _PLATFORM_ERR_WHITELIST, f"{got} 不在平台白名单，平台不会据此产生回流候选"
