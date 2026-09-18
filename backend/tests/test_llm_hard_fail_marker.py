"""请求级 LLM 硬失败标记单测（七环回流 B 方案 · 改动点 A）。

背景：SSE 接口恒返 200，LLM 异常被吞成 error 帧后生成器正常结束 ⇒ root 记 ok ⇒ 平台
按「故障已被业务吸收」把回流候选切掉（环② 断）。故业务层须在失败出口置位，观测中间件
出口据此记 error。

**mock 边界必须落在 `_stream_chat_http`**：若按既有 `test_obs_wiring.py` 的思路只 monkeypatch
更外层的 `llm_stream_chat`，置位点（`stream_chat` 的 except）根本不执行 ⇒ 用例只验了撤销、
没验置位，形同虚设。

三条核心语义：
- 最终失败 ⇒ 置位（否则环② 断，本机制无意义）
- **重试后成功 ⇒ 必须仍未置位**（否则「首次 429 → 重试成功」被误标 error，是假红）
- 上下文缺失 ⇒ 打日志暴露（否则置位静默空转，表现成「trace 记 ok」——正是要治的病）
"""
import logging
from types import SimpleNamespace

import httpx
import pytest
from services import llm_service
from services.llm_service import stream_chat, stream_round1_with_retry
from utils.trace import llm_health_var


class _FakeObs:
    """假 obs_sdk：只记录 record_llm 调用（无 init 状态校验）。"""

    def __init__(self):
        self.calls = []

    def record_llm(self, model, status, *, duration_ms, error_type=None,
                   error_msg=None, usage=None):
        self.calls.append({"status": status, "error_type": error_type})


@pytest.fixture
def health():
    """模拟 obs 中间件：请求进入时置入可变 dict，退出时复位。"""
    h: dict = {"hard_fail": False, "error_type": None}
    # 变量名避开 `token`：pre-commit 的 secrets 检查按名匹配，会把 contextvar 的
    # reset handle 误判成凭据（同族误报已记 memory；改名是既有处置惯例）
    cv_handle = llm_health_var.set(h)
    yield h
    llm_health_var.reset(cv_handle)


def _status_error(code: int) -> httpx.HTTPStatusError:
    resp = httpx.Response(code, request=httpx.Request("POST", "http://x/chat"))
    return httpx.HTTPStatusError("boom", request=resp.request, response=resp)


def test_mark_on_stream_chat_error(monkeypatch, health):
    """最终失败 ⇒ 置位，error_type 取自异常分型（平台白名单词）。"""
    monkeypatch.setattr(llm_service, "_obs_sdk", lambda: _FakeObs())

    def _boom(messages, tools=None):
        yield {"type": "content", "content": "x"}
        raise httpx.TimeoutException("上游超时")

    monkeypatch.setattr(llm_service, "_stream_chat_http", _boom)
    with pytest.raises(httpx.TimeoutException):
        list(stream_chat([]))

    assert health["hard_fail"] is True, "最终失败必须置位，否则 root 记 ok、环② 断"
    assert health["error_type"] == "llm_timeout"


def test_retry_then_success_must_stay_ok(monkeypatch, health):
    """首次 429 → 重试成功：用户拿到完整回答，root 必须仍为 ok（撤销生效）。

    这是「撤销」这一维的**唯一**直接判据：只标记不撤销会把这条正常请求记成 error。
    用默认 chat_llm_max_attempts（=2，首次失败后重试 1 次），退避 0.5s。
    """
    monkeypatch.setattr(llm_service, "_obs_sdk", lambda: _FakeObs())
    state = {"n": 0}

    def _flaky(messages, tools=None):
        state["n"] += 1
        if state["n"] == 1:
            raise _status_error(429)
        yield {"type": "content", "content": "完整回答"}

    monkeypatch.setattr(llm_service, "_stream_chat_http", _flaky)
    out = list(stream_round1_with_retry([]))

    assert state["n"] == 2, "首次 429 应触发一次重试"
    assert [e["type"] for e in out] == ["content"]
    assert health["hard_fail"] is False, "重试成功 ⇒ 交付未降级，不得标记（假红）"
    assert health["error_type"] is None


def test_retry_exhausted_keeps_mark(monkeypatch, health):
    """重试耗尽后仍失败 ⇒ 保留最后一次的标记（用户最终拿到的是兜底话术）。"""
    fake = _FakeObs()
    monkeypatch.setattr(llm_service, "_obs_sdk", lambda: fake)

    def _always_429(messages, tools=None):
        raise _status_error(429)
        yield  # pragma: no cover —— 使本函数成为生成器

    monkeypatch.setattr(llm_service, "_stream_chat_http", _always_429)
    with pytest.raises(httpx.HTTPStatusError):
        list(stream_round1_with_retry([]))

    assert health["hard_fail"] is True
    assert health["error_type"] == "llm_rate_limit"
    assert len(fake.calls) == 2, "每次付费尝试各记 1 条（失败尝试不掩盖）"


def test_mark_without_context_is_loud(monkeypatch, caplog):
    """上下文缺失（中间件未置入）⇒ 打日志，不静默空转。

    静默空转的后果与不修本机制相同（trace 记 ok），故必须能被发现；本用例同时锁住
    「不得抛异常打断业务」——观测边带故障不拦服务。
    """
    monkeypatch.setattr(llm_service, "_obs_sdk", lambda: _FakeObs())

    def _boom(messages, tools=None):
        raise httpx.TimeoutException("上游超时")
        yield  # pragma: no cover

    monkeypatch.setattr(llm_service, "_stream_chat_http", _boom)
    with caplog.at_level(logging.ERROR), pytest.raises(httpx.TimeoutException):
        list(stream_chat([]))

    assert any("llm_health 上下文缺失" in r.getMessage() for r in caplog.records), \
        "置位丢失必须 fail-loud，否则表现为「trace 记 ok」而无人察觉"


class _FakeEndObs:
    """假 obs_sdk 的 end_request 侧：只记收口调用。"""

    def __init__(self):
        self.ends = []

    def end_request(self, status, *, error_type=None, error_msg=None, input=None):
        self.ends.append({"status": status, "error_type": error_type, "input": input})


def _request_with(input_value):
    return SimpleNamespace(state=SimpleNamespace(obs_input=input_value))


def test_obs_end_hard_fail_records_error():
    """出口分支：LLM 硬失败 ⇒ 记 error（root 终态由这一句决定，环② 的入口）。"""
    import main as app_main

    obs = _FakeEndObs()
    app_main._obs_end(obs, SimpleNamespace(status_code=200),
                      _request_with({"content": "问题"}), {"hard_fail": True,
                                                          "error_type": "llm_timeout"})

    assert len(obs.ends) == 1
    assert obs.ends[0]["status"] == "error", "SSE 恒 200，状态码判不出，必须靠 health 记 error"
    assert obs.ends[0]["error_type"] == "llm_timeout"
    assert obs.ends[0]["input"] == {"content": "问题"}, "error 路径也要带现场，否则建不出簇"


def test_obs_end_normal_request_stays_ok():
    """反例（不可省）：正常请求仍记 ok —— 防「一律记 error」式的过度修。

    只验正例的话，「硬失败记 error」与「所有请求都记 error」两种实现在断言上不可区分。
    """
    import main as app_main

    obs = _FakeEndObs()
    app_main._obs_end(obs, SimpleNamespace(status_code=200),
                      _request_with({"content": "问题"}), {"hard_fail": False,
                                                          "error_type": None})

    assert obs.ends[0]["status"] == "ok"
    assert obs.ends[0]["error_type"] is None
