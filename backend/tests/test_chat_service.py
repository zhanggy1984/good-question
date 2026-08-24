"""聊天服务纯函数测试（规则意图分类 + 二期 function calling agent loop，不连外部服务）

chat_service 依赖链较深（database/models），但导入仅需 sqlalchemy 等已装依赖；
pymilvus 由 conftest 条件 stub 兜底，宿主机可离线运行。
契约逻辑测试通过 monkeypatch 隔离外部依赖（子进程/httpx/DB/检索器），无需真实服务。
"""
import json
import sys

import httpx

sys.path.insert(0, "/app")

import services.chat_service as cs
from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from services.chat_service import (
    _build_messages,
    _classify_intent,
    _is_low_confidence,
    _is_smalltalk,
)


def test_build_messages_fc_system():
    """二期 system prompt：工具规则版，无 {context} 占位符（检索结果经 tool 消息回传）"""
    messages = _build_messages("摘要", [], "问题")
    assert messages[0]["role"] == "system"
    assert "工具使用规则" in messages[0]["content"]
    assert "{context}" not in messages[0]["content"], "system prompt 不应再有 context 占位符"
    assert "hybrid_retrieve" in messages[0]["content"]


def test_build_messages_structure():
    """messages 结构：system + history（user/assistant）+ 末尾 user 问题"""
    history = [HumanMessage(content="旧问题"), AIMessage(content="旧回答")]
    messages = _build_messages("摘要", history, "新问题")
    assert len(messages) == 4
    assert messages[1] == {"role": "user", "content": "旧问题"}
    assert messages[2] == {"role": "assistant", "content": "旧回答"}
    assert messages[3] == {"role": "user", "content": "新问题"}


def test_is_low_confidence_boundaries():
    """置信档边界：精排最高分落在 [LOW, 低置信阈值) 判低置信，其余判否"""
    low = settings.similarity_threshold_low
    high = settings.rerank_low_confidence_threshold
    assert low < high
    assert _is_low_confidence(None) is False               # 精排失败/降级
    assert _is_low_confidence(low - 0.01) is False         # 低于 LOW：文档无关（_rerank 已返回空）
    assert _is_low_confidence(low) is True                 # 左闭：等于 LOW 判低置信
    assert _is_low_confidence((low + high) / 2) is True    # 区间中段
    assert _is_low_confidence(high) is False               # 右开：等于高阈值判正常
    assert _is_low_confidence(high + 0.1) is False         # 高于高阈值：正常


def test_confidence_band_three_band():
    """二期置信档三档：none / low / high（tool_call result 与监控日志用）"""
    low = settings.similarity_threshold_low
    high = settings.rerank_low_confidence_threshold
    assert cs._confidence_band(None) == "none"
    assert cs._confidence_band(low - 0.01) == "none"
    assert cs._confidence_band((low + high) / 2) == "low"
    assert cs._confidence_band(high) == "high"
    assert cs._confidence_band(high + 0.1) == "high"


def test_is_smalltalk_boundaries():
    """闲聊粗判边界：纯问候/纯闲聊判 True；含查询意图（即便带问候前缀/感谢尾缀）判 False"""
    assert _is_smalltalk("你好") is True
    assert _is_smalltalk("您好，请问在吗") is True
    assert _is_smalltalk("hi") is True
    assert _is_smalltalk("你是谁") is True
    assert _is_smalltalk("你能做什么") is True          # 身份闲聊含"什么"仍判闲聊
    assert _is_smalltalk("在吗") is True
    assert _is_smalltalk("谢谢") is True
    assert _is_smalltalk("工资发放日是几号") is False
    assert _is_smalltalk("你好，工资发放日是几号") is False  # 带问候前缀的查询仍判非闲聊
    assert _is_smalltalk("工资是几号发，谢谢") is False      # 带感谢尾缀的查询仍判非闲聊
    assert _is_smalltalk("请事假怎么请") is False
    assert _is_smalltalk("帮我总结一下") is False
    assert _is_smalltalk("今天心情怎么样") is True   # 口语寒暄整句（含"怎么"仍判闲聊）
    assert _is_smalltalk("这钱啥时候到账") is False  # 领域词/口语疑问词判查询
    assert _is_smalltalk("") is False
    assert _is_smalltalk("   ") is False


def test_classify_intent_three_way():
    """规则意图分类三档：smalltalk（身份/问候/寒暄）｜query（疑问/查询动词）｜unknown（无法识别）"""
    assert _classify_intent("你是谁") == "smalltalk"         # 身份闲聊优先于疑问词
    assert _classify_intent("你能做什么") == "smalltalk"     # 含"什么"但身份闲聊
    assert _classify_intent("你好") == "smalltalk"
    assert _classify_intent("在吗") == "smalltalk"
    assert _classify_intent("最近怎么样") == "smalltalk"
    assert _classify_intent("今天心情怎么样") == "smalltalk"  # 口语寒暄变体（整句正则命中）
    assert _classify_intent("最近咋样") == "smalltalk"        # 口语"咋"变体
    assert _classify_intent("你最近咋样") == "smalltalk"      # 人称前缀在时间词前
    assert _classify_intent("你咋了") == "smalltalk"          # 口语闲聊 vs 疑问词"咋"
    assert _classify_intent("你呢") == "smalltalk"            # 闲聊反问，直接识别不靠 history
    assert _classify_intent("谢谢") == "smalltalk"
    assert _classify_intent("你好，工资发放日是几号") == "query"  # 问候前缀不覆盖查询
    assert _classify_intent("工资是几号发，谢谢") == "query"       # 感谢尾缀不覆盖查询
    assert _classify_intent("今天心情怎么样，工资几号发") == "query"  # 寒暄+查询整句不误伤
    assert _classify_intent("Docker 的常用命令有哪些") == "query"
    assert _classify_intent("这钱啥时候到账") == "query"       # 领域词/口语疑问词防滑向 unknown
    assert _classify_intent("帮我总结一下") == "query"
    assert _classify_intent("Docker") == "unknown"          # 无闲聊词也无查询标记
    assert _classify_intent("") == "unknown"


def test_classify_intent_history_fallback():
    """unknown 且命中明确回指词才回看 history 归队；非回指 unknown 不归队"""
    from types import SimpleNamespace
    def human(t): return SimpleNamespace(type="human", content=t)
    # 回指词 + 前一轮事实查询 → 归队 query（延续追问）
    assert _classify_intent("就这个", history=[human("工资发放日是几号")]) == "query"
    # 回指词 + 前一轮闲聊 → 归队 smalltalk（延续寒暄）
    assert _classify_intent("还有呢", history=[human("最近怎么样")]) == "smalltalk"
    # 无 history 时 unknown 保持 unknown（不归队）
    assert _classify_intent("就这个") == "unknown"
    # 非回指词 unknown 不归队（"天气"不是回指词，前一轮是 query 也不归队）
    assert _classify_intent("天气", history=[human("工资发放日是几号")]) == "unknown"
    # "你呢"是闲聊反问，直接识别为 smalltalk（跨话题不归队到 query）
    assert _classify_intent("你呢", history=[human("工资发放日是几号")]) == "smalltalk"
    # 当前句本身有明确查询意图，history 不覆盖明确分类
    assert _classify_intent("工资几号发", history=[human("最近怎么样")]) == "query"


# ════════ 评测契约逻辑测试（2.0 契约改造）════════

def test_git_sha_success(monkeypatch):
    """_git_sha：git 子进程可用时取到短 sha（进程内缓存，首次调用后不再 spawn）"""
    monkeypatch.delenv("GIT_SHA", raising=False)  # 环境变量优先逻辑不应干扰子进程路径
    cs._git_sha.cache_clear()
    class _Ok:
        stdout = "abc123\n"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Ok())
    assert cs._git_sha() == "abc123"
    cs._git_sha.cache_clear()


def test_git_sha_failure_returns_empty(monkeypatch):
    """_git_sha：子进程异常（容器内无 .git）时尽力返回空串，不抛错"""
    monkeypatch.delenv("GIT_SHA", raising=False)
    cs._git_sha.cache_clear()
    def _boom(*a, **k):
        raise FileNotFoundError("no git")
    monkeypatch.setattr("subprocess.run", _boom)
    assert cs._git_sha() == ""
    cs._git_sha.cache_clear()


def test_git_sha_env_priority(monkeypatch):
    """_git_sha：构建注入的 GIT_SHA 环境变量优先——即使 git 子进程失败（镜像内无 .git）也返回注入值"""
    monkeypatch.setenv("GIT_SHA", "10ece5f")
    cs._git_sha.cache_clear()
    def _boom(*a, **k):
        raise FileNotFoundError("no git")
    monkeypatch.setattr("subprocess.run", _boom)
    assert cs._git_sha() == "10ece5f"
    cs._git_sha.cache_clear()


def test_knowledge_version_filters_by_library(monkeypatch):
    """_knowledge_version：必须按 library_id 过滤（多库不串库）；空库返回空串"""
    from datetime import datetime
    calls = []

    class _Query:
        def filter(self, *a, **k):
            calls.append(a)
            return self
        def scalar(self):
            return datetime(2026, 8, 24, 0, 52, 0)

    class _Db:
        def query(self, *a, **k):
            return _Query()
        def close(self):
            pass

    monkeypatch.setattr(cs, "SessionLocal", lambda: _Db())
    assert cs._knowledge_version(7) == "20260824005200"
    assert calls, "查询链未走 filter（会漏掉 library 过滤，多库串库）"
    expr = calls[0][0]
    assert expr.left.key == "library_id", "filter 条件不是按 library_id"
    assert expr.right.value == 7

    class _QueryNone:
        def filter(self, *a, **k):
            return self
        def scalar(self):
            return None

    class _DbNone:
        def query(self, *a, **k):
            return _QueryNone()
        def close(self):
            pass

    monkeypatch.setattr(cs, "SessionLocal", lambda: _DbNone())
    assert cs._knowledge_version(7) == ""


def _stream_lines_resp(lines):
    """把 SSE data 行包装成 _stream_deepseek 需要的 httpx 响应对象"""
    class _Resp:
        def iter_lines(self):
            return iter(lines)
    class _Stream:
        def __enter__(self):
            return _Resp()
        def __exit__(self, *a):
            return False
    return _Stream()


def test_stream_deepseek_parses_usage_and_delta(monkeypatch):
    """_stream_deepseek：SSE 行解析——content/reasoning 增量 + 流末 usage chunk（不取 delta 报错）"""
    lines = [
        'data: {"choices":[{"delta":{"content":"你"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"思考"}}]}',
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
        'data: [DONE]',
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _stream_lines_resp(lines))

    events = list(cs._stream_deepseek([{"role": "user", "content": "问题"}]))
    assert events[0] == {"type": "content", "content": "你"}
    assert events[1] == {"type": "reasoning", "content": "思考"}
    assert events[2]["type"] == "usage"
    assert events[2]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_stream_deepseek_parses_tool_calls(monkeypatch):
    """_stream_deepseek：delta.tool_calls 按 index 分片累加，finish_reason=="tool_calls" 触发 flush"""
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"hybrid_retrieve","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"query\\":\\""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"工资发放日"}}]},"finish_reason":"tool_calls"}]}',
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}',
        'data: [DONE]',
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _stream_lines_resp(lines))

    events = list(cs._stream_deepseek([{"role": "user", "content": "问题"}], tools=[cs.RETRIEVE_TOOL_SCHEMA]))
    tc = next(e for e in events if e["type"] == "tool_call")
    call = tc["tool_calls"][0]
    assert call["id"] == "call_1", "tool_call id 应完整（首个分片携带）"
    assert call["function"]["name"] == "hybrid_retrieve"
    assert call["function"]["arguments"] == '{"query":"工资发放日', "arguments 应按分片顺序拼接"
    usage = next(e for e in events if e["type"] == "usage")
    assert usage["usage"]["total_tokens"] == 8


def test_stream_deepseek_tool_call_flush_on_done(monkeypatch):
    """_stream_deepseek：个别实现不返回 finish_reason 时，[DONE] 后兜底 flush tool_calls"""
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"hybrid_retrieve","arguments":"{\\"query\\":\\"a\\"}"}}]}}]}',
        'data: [DONE]',
    ]
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _stream_lines_resp(lines))

    events = list(cs._stream_deepseek([{"role": "user", "content": "问题"}], tools=[cs.RETRIEVE_TOOL_SCHEMA]))
    tc = next(e for e in events if e["type"] == "tool_call")
    assert tc["tool_calls"][0]["function"]["arguments"] == '{"query":"a"}'


def test_execute_retrieve_tool_structure(monkeypatch):
    """_execute_retrieve_tool：包装 HybridRetriever，返回 context/sources/source_count/max_score/confidence_band

    max_score 来自 rerank（numpy float32），必须转原生 float——json.dumps 不认 float32，
    否则 tool_call SSE 事件与 JSON 日志序列化崩溃（曾线上复现）。
    """
    import numpy as np
    from types import SimpleNamespace
    chunk = SimpleNamespace(
        content="内容内容内容",
        metadata={"document_name": "测试.md", "heading_path": ["标题"], "chunk_index": 1, "total_chunks": 3},
    )

    class _Retriever:
        max_rerank_score = np.float32(0.9)  # 模拟 rerank 返回的 numpy 标量
        def __init__(self, *a, **k):
            pass
        def invoke(self, q):
            return [chunk]

    monkeypatch.setattr(cs, "HybridRetriever", _Retriever)
    r = cs._execute_retrieve_tool(7, "问题")
    assert r["source_count"] == 1
    assert r["confidence_band"] == "high"
    assert isinstance(r["max_score"], float), "max_score 应转原生 float（numpy float32 不可 JSON 序列化）"
    json.dumps({"source_count": r["source_count"], "max_score": r["max_score"]})  # 序列化不抛
    assert r["sources"][0]["document_name"] == "测试.md"
    assert "测试.md" in r["context"] and "内容内容内容" in r["context"]


def test_json_log_outputs_json_line(monkeypatch):
    """_json_log：logger.info 输出含 kind/ts 的 JSON 行（结构化监控日志）"""
    captured = []

    class _Rec:
        def info(self, msg, *a, **k):
            captured.append(msg)

    monkeypatch.setattr(cs, "logger", _Rec())
    cs._json_log("tool_decision", rule_agree=True, total_tokens=7)
    assert len(captured) == 1
    data = json.loads(captured[0])
    assert data["kind"] == "tool_decision"
    assert data["rule_agree"] is True
    assert data["total_tokens"] == 7
    assert isinstance(data["ts"], int)


# ════════ 二期 agent loop 测试（stream_chat 外部依赖全隔离）════════

# 预置 tool 结果：检索命中（1 条）与检索空
_TOOL_RESULT_HIT = {
    "context": "内容内容内容",
    "sources": [{"document_name": "测试.md", "heading_path": ["标题"], "chunk_content": "内容内容内容",
                 "chunk_index": 1, "total_chunks": 3}],
    "source_count": 1,
    "max_score": 0.9,
    "confidence_band": "high",
}
_TOOL_RESULT_EMPTY = {
    "context": "", "sources": [], "source_count": 0, "max_score": None, "confidence_band": "none",
}

_TOOL_CALL_EVENT = {"type": "tool_call", "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "hybrid_retrieve", "arguments": '{"query": "工资发放日"}'}}]}


def _patch_chat_pipeline(monkeypatch, tool_result=None, round1=None, round2=None):
    """stream_chat 外部依赖全隔离：DB/上下文/持久化/检索工具/LLM 流式（按轮次返回事件）

    round1=第一轮（带 tools）事件、round2=第二轮事件；tools 区分轮次（第二轮不带 tools）。
    """
    class _FakeSession:
        id = 1
        library_id = 7

    class _FakeDb:
        def query(self, *a, **k):
            return self
        def filter(self, *a, **k):
            return self
        def first(self):
            return _FakeSession()
        def close(self):
            pass

    def _fake_stream(messages, tools=None):
        return iter(round1 if tools else (round2 if round2 is not None else round1))

    monkeypatch.setattr(cs, "SessionLocal", lambda: _FakeDb())
    monkeypatch.setattr(cs, "_build_context", lambda db, s: ("摘要", []))
    monkeypatch.setattr(cs, "_save_messages", lambda db, s, q, a, src: 1)
    monkeypatch.setattr(cs, "_compress_memory", lambda db, s: None)
    monkeypatch.setattr(cs, "_stream_deepseek", _fake_stream)
    monkeypatch.setattr(cs, "_execute_retrieve_tool", lambda library_id, query: tool_result)
    monkeypatch.setattr(cs, "_json_log", lambda *a, **k: None)


def test_stream_chat_no_retrieve_direct_answer(monkeypatch):
    """LLM 决定不检索（信任直接回答）：meta → token → usage → done，无 tool_call/sources"""
    _patch_chat_pipeline(monkeypatch, round1=[
        {"type": "content", "content": "你好，我是文档问答助手，可以基于文档库中的内容为你查找、总结或理解文档信息并解答相关问题。"},
        {"type": "usage", "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}},
    ])
    events = list(cs.stream_chat(1, "你好"))
    types = [t for t, _ in events]
    assert types == ["meta", "token", "usage", "done"], f"事件序应为直接回答路径，实际 {types}"
    assert "tool_call" not in types and "sources" not in types
    tok = next(d for t, d in events if t == "token")
    assert "文档问答助手" in tok["content"]
    usage = next(d for t, d in events if t == "usage")
    assert usage["total_tokens"] == 10


def test_stream_chat_retrieve_hit(monkeypatch):
    """LLM 决定检索且命中：meta → tool_call → sources → token(第二轮) → usage → done"""
    _patch_chat_pipeline(
        monkeypatch,
        tool_result=_TOOL_RESULT_HIT,
        round1=[_TOOL_CALL_EVENT],
        round2=[
            {"type": "content", "content": "工资发放日为每月 10 号。"},
            {"type": "usage", "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        ],
    )
    events = list(cs.stream_chat(1, "工资发放日是几号"))
    types = [t for t, _ in events]
    assert types == ["meta", "tool_call", "sources", "token", "usage", "done"], f"实际 {types}"
    tc = next(d for t, d in events if t == "tool_call")
    for k in ("id", "name", "args", "result", "status", "ts"):
        assert k in tc, f"tool_call 缺字段 {k}"
    assert tc["name"] == "hybrid_retrieve"
    assert tc["status"] == "ok"
    assert tc["result"]["source_count"] == 1
    assert tc["result"]["confidence_band"] == "high"
    assert "source_count" in tc["result"], "tool_call result 应含 source_count（验证脚本依赖）"
    src = next(d for t, d in events if t == "sources")
    assert src["sources"][0]["document_name"] == "测试.md"
    tok = next(d for t, d in events if t == "token")
    assert "每月 10 号" in tok["content"]


def test_stream_chat_usage_merged_across_rounds(monkeypatch):
    """多轮调用的 usage 合并为一个 usage 事件（7 + 3 = 10）"""
    _patch_chat_pipeline(
        monkeypatch,
        tool_result=_TOOL_RESULT_HIT,
        round1=[_TOOL_CALL_EVENT,
                {"type": "usage", "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}],
        round2=[{"type": "content", "content": "答案"},
                {"type": "usage", "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}],
    )
    events = list(cs.stream_chat(1, "工资发放日是几号"))
    usages = [d for t, d in events if t == "usage"]
    assert len(usages) == 1, "usage 应合并为单个事件（done 前统一发出）"
    assert usages[0]["total_tokens"] == 10


def test_stream_chat_retrieve_empty_query_uses_not_found(monkeypatch):
    """LLM 决定检索但空 + 事实查询：tool_call → token(未找到) → usage → done，第二轮不调（防幻觉）"""
    _patch_chat_pipeline(monkeypatch, tool_result=_TOOL_RESULT_EMPTY, round1=[_TOOL_CALL_EVENT])

    def _fake_stream(messages, tools=None):
        if tools:
            return iter([_TOOL_CALL_EVENT])
        raise AssertionError("空结果 + 事实查询不应调第二轮 LLM")

    monkeypatch.setattr(cs, "_stream_deepseek", _fake_stream)
    events = list(cs.stream_chat(1, "工资发放日是几号"))
    types = [t for t, _ in events]
    assert types == ["meta", "tool_call", "token", "usage", "done"], f"实际 {types}"
    assert "sources" not in types
    tok = next(d for t, d in events if t == "token")
    assert tok["content"] == cs._NOT_FOUND_ANSWER


def test_stream_chat_retrieve_empty_unknown_uses_clarify(monkeypatch):
    """LLM 决定检索但空 + 意图不明（unknown）：走澄清话术而非"未找到"，第二轮不调"""
    _patch_chat_pipeline(monkeypatch, tool_result=_TOOL_RESULT_EMPTY, round1=[_TOOL_CALL_EVENT])

    def _fake_stream(messages, tools=None):
        if tools:
            return iter([_TOOL_CALL_EVENT])
        raise AssertionError("unknown 空结果不应调第二轮 LLM")

    monkeypatch.setattr(cs, "_stream_deepseek", _fake_stream)
    events = list(cs.stream_chat(1, "Docker"))  # 无闲聊词也无查询标记 → unknown
    tok = next(d for t, d in events if t == "token")
    assert tok["content"] == cs._UNKNOWN_ANSWER
    assert tok["content"] != cs._NOT_FOUND_ANSWER


def test_stream_chat_retrieve_empty_smalltalk_second_round(monkeypatch):
    """寒暄却被 LLM 检索且空（模型异常，低频）：第二轮 LLM 自然引导，事件序 tool_call → token(第二轮)"""
    _patch_chat_pipeline(
        monkeypatch,
        tool_result=_TOOL_RESULT_EMPTY,
        round1=[_TOOL_CALL_EVENT],
        round2=[
            {"type": "content", "content": "你好，我是文档问答助手，有什么可以帮你？"},
            {"type": "usage", "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}},
        ],
    )
    events = list(cs.stream_chat(1, "你好"))  # smalltalk
    types = [t for t, _ in events]
    assert types == ["meta", "tool_call", "token", "usage", "done"], f"实际 {types}"
    tok = next(d for t, d in events if t == "token")
    assert "文档问答助手" in tok["content"]


# ════════ 三期 F3 规则否决权测试（LLM 决定不检索但规则判该查 → 强制检索）════════


def test_stream_chat_rule_override_hit(monkeypatch):
    """F3 否决命中：rule=query + LLM 不检 → 强制检索 → tool_call(rule_override) + sources + 第二轮补答"""
    _patch_chat_pipeline(
        monkeypatch,
        tool_result=_TOOL_RESULT_HIT,
        round1=[
            {"type": "content", "content": "工资发放日为每月 10 号。"},  # LLM 首轮不检直接答
            {"type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        ],
        round2=[
            {"type": "content", "content": "基于文档，工资发放日为每月 10 号。[来源1]"},
            {"type": "usage", "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}},
        ],
    )
    events = list(cs.stream_chat(1, "工资发放日是几号"))
    types = [t for t, _ in events]
    # 首轮 token 已流出无法撤回 → 补答拼接在后
    assert types == ["meta", "token", "tool_call", "sources", "token", "usage", "done"], f"实际 {types}"
    tc = next(d for t, d in events if t == "tool_call")
    assert tc["status"] == "rule_override", f"否决时 tool_call.status 应为 rule_override，实际 {tc['status']}"
    assert tc["result"]["source_count"] == 1
    toks = [d["content"] for t, d in events if t == "token"]
    assert toks[0] == "工资发放日为每月 10 号。"   # 首轮 LLM 直接答（已流出）
    assert "基于文档" in toks[1]                    # 第二轮基于 context 重答
    assert "sources" in types


def test_stream_chat_rule_override_empty_query_uses_not_found(monkeypatch):
    """F3 否决但检索空 + query：tool_call(rule_override) → token(未找到)，第二轮不调（防空 context 编造）"""
    _patch_chat_pipeline(
        monkeypatch,
        tool_result=_TOOL_RESULT_EMPTY,
        round1=[{"type": "content", "content": "工资发放日为每月 10 号。"},
                {"type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}],
    )

    def _fake_stream(messages, tools=None):
        if tools:
            return iter([{"type": "content", "content": "工资发放日为每月 10 号。"}])
        raise AssertionError("否决后检索空 + query 不应调第二轮 LLM")

    monkeypatch.setattr(cs, "_stream_deepseek", _fake_stream)
    events = list(cs.stream_chat(1, "工资发放日是几号"))
    types = [t for t, _ in events]
    assert types == ["meta", "token", "tool_call", "token", "usage", "done"], f"实际 {types}"
    assert "sources" not in types
    toks = [d["content"] for t, d in events if t == "token"]
    assert toks[0] == "工资发放日为每月 10 号。"   # 首轮
    assert toks[1] == cs._NOT_FOUND_ANSWER         # 固定话术，防空 context 再编造


def test_stream_chat_rule_override_empty_unknown_uses_clarify(monkeypatch):
    """F3 否决但检索空 + unknown：tool_call(rule_override) → token(澄清) 而非"未找到" """
    _patch_chat_pipeline(
        monkeypatch,
        tool_result=_TOOL_RESULT_EMPTY,
        round1=[{"type": "content", "content": "Docker。"},
                {"type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}],
    )

    def _fake_stream(messages, tools=None):
        if tools:
            return iter([{"type": "content", "content": "Docker。"}])
        raise AssertionError("否决后检索空 + unknown 不应调第二轮 LLM")

    monkeypatch.setattr(cs, "_stream_deepseek", _fake_stream)
    events = list(cs.stream_chat(1, "Docker"))  # 无闲聊词也无查询标记 → unknown
    toks = [d["content"] for t, d in events if t == "token"]
    assert toks[-1] == cs._UNKNOWN_ANSWER
    assert toks[-1] != cs._NOT_FOUND_ANSWER


def test_stream_chat_rule_override_disabled(monkeypatch):
    """F3 开关关闭：rule=query + LLM 不检 → 不否决，信任 LLM 直接答（无 tool_call）"""
    monkeypatch.setattr(cs.settings, "rule_override_enabled", False)
    _patch_chat_pipeline(
        monkeypatch,
        round1=[{"type": "content", "content": "工资发放日为每月 10 号。"},
                {"type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}],
    )
    events = list(cs.stream_chat(1, "工资发放日是几号"))
    types = [t for t, _ in events]
    assert types == ["meta", "token", "usage", "done"], f"实际 {types}"
    assert "tool_call" not in types and "sources" not in types


def test_stream_chat_rule_override_smalltalk_not_triggered(monkeypatch):
    """F3 范围排除：smalltalk + LLM 不检 → 不否决，无 tool_call（现有直接答路径不受否决影响）"""
    _patch_chat_pipeline(
        monkeypatch,
        round1=[{"type": "content", "content": "你好，我是文档问答助手。"},
                {"type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}],
    )
    events = list(cs.stream_chat(1, "你好"))  # smalltalk
    types = [t for t, _ in events]
    assert types == ["meta", "token", "usage", "done"], f"实际 {types}"
    assert "tool_call" not in types


def test_stream_chat_rule_override_log_field(monkeypatch):
    """tool_decision 日志含 rule_override：否决=True、非否决=False（监控可分辨"否决修正"vs"不一致未处理"）"""
    captured = []

    def _log(kind, **fields):
        captured.append({"kind": kind, **fields})

    _patch_chat_pipeline(
        monkeypatch,
        tool_result=_TOOL_RESULT_EMPTY,
        round1=[{"type": "content", "content": "工资发放日为每月 10 号。"},
                {"type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}],
    )
    monkeypatch.setattr(cs, "_json_log", _log)
    list(cs.stream_chat(1, "工资发放日是几号"))  # rule=query + LLM 不检 → 否决
    td = next(x for x in captured if x["kind"] == "tool_decision")
    assert td["rule_override"] is True
    assert td["rule_agree"] is False          # 否决时规则与 LLM 必然不一致
    assert td["llm_decided"] is False         # LLM 本意不检索
    assert td["tool"] == "hybrid_retrieve"    # 实际执行了检索
    # 非否决路径：smalltalk 不触发
    captured.clear()
    list(cs.stream_chat(1, "你好"))
    td2 = next(x for x in captured if x["kind"] == "tool_decision")
    assert td2["rule_override"] is False
    assert td2["rule_agree"] is True
    # unknown 触发否决：rule_agree 也应为 False（unknown 属于"该查"集合；
    # 若沿用二期 query-only 公式会误报 True，无法区分"否决修正"vs"不一致未处理"）
    captured.clear()
    list(cs.stream_chat(1, "Docker"))  # unknown + LLM 不检 → 否决
    td3 = next(x for x in captured if x["kind"] == "tool_decision")
    assert td3["rule_override"] is True
    assert td3["rule_agree"] is False


def test_is_non_doc_question():
    """_is_non_doc_question：纯计算/当前时间/通用常识 → True；文档类问题 → False（不误伤）"""
    assert cs._is_non_doc_question("17 乘以 23 等于多少")
    assert cs._is_non_doc_question("1+1等于几")
    assert cs._is_non_doc_question("计算 17*23")
    assert cs._is_non_doc_question("今天是星期几")
    assert cs._is_non_doc_question("今天天气怎么样")
    assert cs._is_non_doc_question("圆周率是多少")
    # 文档类（演示库考勤/工资），不应豁免
    assert not cs._is_non_doc_question("工资发放日是几号")
    assert not cs._is_non_doc_question("请事假需要提前几天申请")
    assert not cs._is_non_doc_question("加班费怎么算")
    assert not cs._is_non_doc_question("帮我总结一下文档讲了什么")


def test_stream_chat_rule_override_skipped_for_calc(monkeypatch):
    """F3 豁免：纯计算题 + LLM 不检 → 不否决，直接答（无 tool_call，避免"先答再补未找到"）"""
    _patch_chat_pipeline(
        monkeypatch,
        round1=[{"type": "content", "content": "17 × 23 = 391。"},
                {"type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}],
    )
    events = list(cs.stream_chat(1, "17 乘以 23 等于多少"))
    types = [t for t, _ in events]
    assert types == ["meta", "token", "usage", "done"], f"实际 {types}"
    assert "tool_call" not in types and "sources" not in types
    tok = next(d for t, d in events if t == "token")
    assert "391" in tok["content"]


def test_stream_chat_retrieve_empty_calc_second_round(monkeypatch):
    """检索空 + 纯计算题（LLM 检但空）：不走"未找到"，第二轮 LLM 自然作答（豁免非文档问题）"""
    _patch_chat_pipeline(
        monkeypatch,
        tool_result=_TOOL_RESULT_EMPTY,
        round1=[_TOOL_CALL_EVENT],
        round2=[{"type": "content", "content": "17 × 23 = 391。"},
                {"type": "usage", "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}],
    )
    events = list(cs.stream_chat(1, "17 乘以 23 等于多少"))
    types = [t for t, _ in events]
    assert types == ["meta", "tool_call", "token", "usage", "done"], f"实际 {types}"
    tok = next(d for t, d in events if t == "token")
    assert "391" in tok["content"]
    assert tok["content"] != cs._NOT_FOUND_ANSWER


def test_stream_chat_rule_override_error(monkeypatch):
    """F3 否决但检索工具抛异常：tool_call status=rule_override_error + 空结果走固定话术，第二轮不调"""
    _patch_chat_pipeline(
        monkeypatch,
        round1=[{"type": "content", "content": "工资发放日为每月 10 号。"},
                {"type": "usage", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}],
    )

    def _boom(library_id, query):
        raise RuntimeError("milvus down")

    def _fake_stream(messages, tools=None):
        if tools:
            return iter([{"type": "content", "content": "工资发放日为每月 10 号。"}])
        raise AssertionError("否决检索失败 + query 不应调第二轮 LLM")

    monkeypatch.setattr(cs, "_execute_retrieve_tool", _boom)
    monkeypatch.setattr(cs, "_stream_deepseek", _fake_stream)
    events = list(cs.stream_chat(1, "工资发放日是几号"))
    tc = next(d for t, d in events if t == "tool_call")
    assert tc["status"] == "rule_override_error"
    toks = [d["content"] for t, d in events if t == "token"]
    assert toks[-1] == cs._NOT_FOUND_ANSWER


def test_contracts_manifest_endpoint():
    """GET /api/contracts：声明 LLM 评测接口与场景清单（平台自动发现用）"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.contracts import router as contracts_router

    app = FastAPI()
    app.include_router(contracts_router)
    client = TestClient(app)

    r = client.get("/contracts")
    assert r.status_code == 200
    data = r.json()
    assert data["agent"] == "good-question"
    assert data["contract_version"] == "1.0"
    chat_iface = next(i for i in data["interfaces"] if i["name"] == "chat")
    assert chat_iface["contract_type"] == "sse"
    assert chat_iface["llm"] is True
    tags = {s["tag"] for s in data["scenes"]}
    assert {"greeting", "doc_qa", "no_hit", "summarize"} <= tags
