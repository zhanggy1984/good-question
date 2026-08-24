"""聊天服务纯函数测试（两级置信档 + 评测契约逻辑，不连外部服务）

chat_service 依赖链较深（database/models），但导入仅需 sqlalchemy 等已装依赖；
pymilvus 由 conftest 条件 stub 兜底，宿主机可离线运行。
契约逻辑测试通过 monkeypatch 隔离外部依赖（子进程/httpx/DB/检索器），无需真实服务。
"""
import httpx
import sys

sys.path.insert(0, "/app")

import services.chat_service as cs
from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from services.chat_service import LOW_CONFIDENCE_HINT, _build_messages, _is_low_confidence


def test_build_messages_low_confidence_injects_hint():
    """低置信档：system prompt 追加 LOW_CONFIDENCE_HINT，提示 LLM 相关性存疑、不足则如实回答"""
    messages = _build_messages("摘要", [], "上下文", "问题", low_confidence=True)
    assert messages[0]["role"] == "system"
    assert LOW_CONFIDENCE_HINT in messages[0]["content"]


def test_build_messages_normal_without_hint():
    """正常档：不注入低置信提示"""
    messages = _build_messages("摘要", [], "上下文", "问题", low_confidence=False)
    assert LOW_CONFIDENCE_HINT not in messages[0]["content"]


def test_build_messages_structure():
    """messages 结构：system + history（user/assistant）+ 末尾 user 问题"""
    history = [HumanMessage(content="旧问题"), AIMessage(content="旧回答")]
    messages = _build_messages("摘要", history, "上下文", "新问题")
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


# ════════ 评测契约逻辑测试（2.0 契约改造）════════

def test_git_sha_success(monkeypatch):
    """_git_sha：git 子进程可用时取到短 sha（进程内缓存，首次调用后不再 spawn）"""
    cs._git_sha.cache_clear()
    class _Ok:
        stdout = "abc123\n"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Ok())
    assert cs._git_sha() == "abc123"
    cs._git_sha.cache_clear()


def test_git_sha_failure_returns_empty(monkeypatch):
    """_git_sha：子进程异常（容器内无 .git）时尽力返回空串，不抛错"""
    cs._git_sha.cache_clear()
    def _boom(*a, **k):
        raise FileNotFoundError("no git")
    monkeypatch.setattr("subprocess.run", _boom)
    assert cs._git_sha() == ""
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


def test_stream_deepseek_parses_usage_and_delta(monkeypatch):
    """_stream_deepseek：SSE 行解析——content/reasoning 增量 + 流末 usage chunk（不取 delta 报错）"""
    lines = [
        'data: {"choices":[{"delta":{"content":"你"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"思考"}}]}',
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
        'data: [DONE]',
    ]
    class _Resp:
        def iter_lines(self):
            return iter(lines)
    class _Stream:
        def __enter__(self):
            return _Resp()
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _Stream())

    events = list(cs._stream_deepseek([{"role": "user", "content": "问题"}]))
    assert events[0] == {"type": "content", "content": "你"}
    assert events[1] == {"type": "reasoning", "content": "思考"}
    assert events[2]["type"] == "usage"
    assert events[2]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def _patch_chat_pipeline(monkeypatch, docs=None, max_score=None, llm_events=None):
    """stream_chat 外部依赖全隔离：DB/检索器/上下文/持久化/LLM 流式"""
    from types import SimpleNamespace

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

    class _FakeRetriever:
        max_rerank_score = max_score
        def __init__(self, *a, **k):
            pass
        def invoke(self, q):
            return docs or []

    monkeypatch.setattr(cs, "SessionLocal", lambda: _FakeDb())
    monkeypatch.setattr(cs, "HybridRetriever", _FakeRetriever)
    monkeypatch.setattr(cs, "_build_context", lambda db, s: ("摘要", []))
    monkeypatch.setattr(cs, "_save_messages", lambda db, s, q, a, src: 1)
    monkeypatch.setattr(cs, "_compress_memory", lambda db, s: None)
    monkeypatch.setattr(cs, "_stream_deepseek", lambda messages: iter(llm_events or []))


def test_stream_chat_event_sequence_no_docs(monkeypatch):
    """无检索结果：meta → reasoning/token → usage → done，不推 tool_call/sources"""
    _patch_chat_pipeline(monkeypatch, llm_events=[
        {"type": "reasoning", "content": "思考"},
        {"type": "content", "content": "答案"},
        {"type": "usage", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ])
    events = list(cs.stream_chat(1, "问题"))
    types = [t for t, _ in events]
    assert types[0] == "meta"
    assert types[-1] == "done"
    assert "usage" in types and types.index("usage") < types.index("done")
    assert "tool_call" not in types and "sources" not in types

    meta = events[0][1]
    for k in ("agent", "model", "interface", "contract_version", "git_sha", "knowledge_version", "ts"):
        assert k in meta
    assert isinstance(meta["ts"], int)

    for t, d in events:
        if t in ("reasoning", "token"):
            assert d["content"] == d["delta"]
            assert isinstance(d["ts"], int)


def test_stream_chat_tool_call_when_docs(monkeypatch):
    """有检索结果：meta → tool_call → sources → ... → usage → done，tool_call 结构完整"""
    from types import SimpleNamespace
    doc = SimpleNamespace(
        metadata={"document_name": "测试.md", "heading_path": ["标题"], "chunk_index": 1, "total_chunks": 3},
        page_content="内容内容内容",
    )
    _patch_chat_pipeline(monkeypatch, docs=[doc], max_score=0.9, llm_events=[
        {"type": "content", "content": "答案"},
        {"type": "usage", "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}},
    ])
    events = list(cs.stream_chat(1, "问题"))
    types = [t for t, _ in events]
    assert types[0] == "meta" and types[-1] == "done"
    assert "tool_call" in types and "sources" in types
    tc = next(d for t, d in events if t == "tool_call")
    for k in ("id", "name", "args", "result", "status", "ts"):
        assert k in tc
    assert tc["name"] == "hybrid_retrieve"
    assert tc["status"] == "ok"
    assert tc["result"]["source_count"] == 1


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
