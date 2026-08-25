"""聊天问答缓存模块测试（chat_cache）：key/get/set/flush/重放/降级，不连真实 Redis

用内存版 fake client 替代真实连接覆盖核心逻辑；Redis 连接异常降级用抛错 client 模拟。
"""
import json
from types import SimpleNamespace

import services.chat_cache as cc


class _FakeRedis:
    """内存版 Redis 客户端（get/set/scan_iter/delete），测试替代真实连接"""
    def __init__(self):
        self._d = {}

    def get(self, key):
        return self._d.get(key)

    def set(self, key, value, ex=None):
        self._d[key] = value

    def scan_iter(self, match="", count=100):
        prefix = match[:-1] if match.endswith("*") else match
        return iter([k for k in self._d if k.startswith(prefix)])

    def delete(self, *keys):
        n = 0
        for k in keys:
            if k in self._d:
                del self._d[k]
                n += 1
        return n


def _reset(monkeypatch):
    """清熔断状态（异常降级测试会设置 _DISABLED_UNTIL，避免污染其他用例）"""
    monkeypatch.setattr(cc, "_DISABLED_UNTIL", 0.0)
    monkeypatch.setattr(cc.settings, "chat_cache_enabled", True)


def _payload(**overrides):
    p = {
        "decided_retrieve": True, "rule_override": False, "query": "q",
        "tool_status": "ok",
        "tool_result": {"source_count": 1, "max_score": 0.9, "confidence_band": "high"},
        "sources": [],
        "reasoning_round1": "思", "reasoning_round2": "考",
        "intent": "query", "non_doc_question": False,
        "answer": "答",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    p.update(overrides)
    return p


def test_key_stable_and_library_isolated(monkeypatch):
    monkeypatch.setattr(cc.settings, "deepseek_model", "deepseek-chat")
    k1 = cc._key(7, "工资几号发")
    assert cc._key(7, "工资几号发") == k1          # 同库同问题 → 同 key
    assert cc._key(8, "工资几号发") != k1          # 不同库 → 隔离
    assert cc._key(7, " 工资几号发 ") == k1        # 首尾空白归一，不影响命中
    assert k1.startswith("goodq:chat:7:")
    # 换模型 → key 变，旧模型缓存自然失效（config 支持只改 DEEPSEEK_MODEL 切换）
    monkeypatch.setattr(cc.settings, "deepseek_model", "deepseek-v4-pro")
    assert cc._key(7, "工资几号发") != k1


def test_set_get_roundtrip(monkeypatch):
    _reset(monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(cc, "_client", lambda: fake)
    cc.set_cached(7, "工资几号发", _payload())
    got = cc.get_cached(7, "工资几号发")
    assert got is not None
    assert got["v"] == cc._CACHE_VERSION, "写入应带结构版本"
    assert got["answer"] == "答"
    assert got["usage"]["total_tokens"] == 7


def test_get_miss_returns_none(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setattr(cc, "_client", lambda: _FakeRedis())
    assert cc.get_cached(7, "不存在的问题") is None


def test_get_version_mismatch_returns_none(monkeypatch):
    _reset(monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(cc, "_client", lambda: fake)
    fake._d[cc._key(7, "q")] = json.dumps({"v": 999, "answer": "旧格式"}, ensure_ascii=False)
    assert cc.get_cached(7, "q") is None, "版本不兼容的旧缓存应视为未命中"


def test_redis_error_degrade_and_circuit_break(monkeypatch):
    _reset(monkeypatch)

    class _Boom:
        def get(self, *a, **k):
            raise ConnectionError("redis down")

    _factory = type("R", (), {"from_url": staticmethod(lambda *a, **k: _Boom())})
    monkeypatch.setattr(cc, "_redis_lib", SimpleNamespace(Redis=_factory))
    assert cc.get_cached(7, "q") is None, "Redis 异常应静默降级返回 None，不影响主链路"
    assert cc._DISABLED_UNTIL > 0, "异常后应触发熔断"

    # 熔断期内不再尝试连接：即便 from_url 恢复正常，_client 也返回 None
    _factory2 = type("R", (), {"from_url": staticmethod(lambda *a, **k: _FakeRedis())})
    monkeypatch.setattr(cc, "_redis_lib", SimpleNamespace(Redis=_factory2))
    assert cc.get_cached(7, "q") is None, "熔断期内不应再尝试 Redis"


def test_flush_library_scans_prefix(monkeypatch):
    _reset(monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(cc, "_client", lambda **k: fake)
    cc.set_cached(7, "a", _payload())
    cc.set_cached(7, "b", _payload())
    cc.set_cached(8, "a", _payload())
    cc.flush_library(7)
    assert len(fake._d) == 1, "库 7 的缓存应全部清空，库 8 保留"
    assert cc.get_cached(8, "a") is not None, "库 8 的缓存不受影响"


def test_flush_library_not_blocked_by_circuit_break(monkeypatch):
    """熔断只拦高频读路径（get/set）；flush 是低频管理操作，始终尝试连接

    Redis 恢复后下次文档上传即可清掉残留缓存，而不是干等 TTL 到期。
    """
    _reset(monkeypatch)
    monkeypatch.setattr(cc, "_DISABLED_UNTIL", 10 ** 12)  # 模拟熔断中（1 年内不可恢复）
    fake = _FakeRedis()
    _factory = type("R", (), {"from_url": staticmethod(lambda *a, **k: fake)})
    monkeypatch.setattr(cc, "_redis_lib", SimpleNamespace(Redis=_factory))
    assert cc.get_cached(7, "q") is None, "熔断期内读路径不应尝试 Redis"
    # 直接往 fake 造数据（set 受熔断拦截，这里绕过），验证 flush 仍能真实连上并清掉
    fake._d[cc._key(7, "q")] = json.dumps({"v": cc._CACHE_VERSION, "answer": "残留"}, ensure_ascii=False)
    cc.flush_library(7)
    assert fake._d == {}, "flush 不受熔断限制，Redis 恢复后下次上传即可清残留缓存"


def test_replay_events_retrieved_hit():
    value = _payload(reasoning_round1="思考", reasoning_round2="作答", answer="答案是 10 号")
    events = list(cc.replay_events(value))
    types = [t for t, _ in events]
    # 事件序与真实流程一致：reasoning(首轮，逐字符分片可能多条) → tool_call → sources →
    # reasoning(次轮) → token* → usage。断言相对位置而非固定下标（分片条数不定）
    assert types[0] == "reasoning", f"首事件应为首轮 reasoning，实际 {types}"
    assert types[-1] == "usage"
    # 相对位置：首轮 reasoning 在 tool_call 前、次轮 reasoning 在 sources 后（真实流程位置）
    tool_call_idx = types.index("tool_call")
    reasoning_idx = [i for i, t in enumerate(types) if t == "reasoning"]
    assert min(reasoning_idx) < tool_call_idx, "首轮思考应在 tool_call 前"
    assert tool_call_idx < types.index("sources") < types.index("token")
    assert max(reasoning_idx) > types.index("sources"), "次轮思考应在 sources 后"
    tc = dict(next(d for t, d in events if t == "tool_call"))
    assert tc["name"] == "hybrid_retrieve"
    assert tc["args"]["query"] == "q"
    assert tc["result"]["source_count"] == 1
    assert tc["result"]["confidence_band"] == "high"
    # intent/non_doc_question 透传：前端据其决定"已检索但空"提示，缓存命中路径必须与真实一致
    assert tc["intent"] == "query"
    assert tc["non_doc_question"] is False
    assert "".join(d["content"] for t, d in events if t == "reasoning") == "思考作答"
    assert "".join(d["content"] for t, d in events if t == "token") == "答案是 10 号"
    usage = dict(events[-1][1])
    assert usage["cached"] is True, "命中时 usage 应带 cached 标记（计费统计排除）"
    assert usage["total_tokens"] == 7
    # ts 严格单调递增：逐字符分片同一毫秒完成，复用独立 _ts() 会重复，重放必须 +1 递增
    ts = [d["ts"] for _, d in events]
    assert all(a < b for a, b in zip(ts, ts[1:])), "重放事件 ts 应严格单调递增"


def test_replay_events_empty_hit_smalltalk_intent_passthrough():
    """空检索 + smalltalk/non_doc 意图：tool_call 透传 intent，前端据此不误显示"未找到"提示

    缓存命中"非文档问题 + 空检索"的答案时，若 tool_call 缺 intent/non_doc_question，
    前端 ChatView 判断（intent!=='smalltalk' 且 !non_doc_question）会误触发 empty 提示。
    """
    value = _payload(
        tool_result={"source_count": 0, "max_score": None, "confidence_band": "none"},
        sources=[], reasoning_round1="", reasoning_round2="",
        answer="你好，我是文档问答助手。",
        intent="smalltalk", non_doc_question=True,
    )
    events = list(cc.replay_events(value))
    tc = next(d for t, d in events if t == "tool_call")
    assert tc["intent"] == "smalltalk", "空命中 smalltalk 意图必须透传，防前端误显 empty 提示"
    assert tc["non_doc_question"] is True


def test_replay_events_not_found_empty_retrieve():
    """检索空（source_count=0）：重放 tool_call + token(固定话术)，不发 sources"""
    value = _payload(
        tool_result={"source_count": 0, "max_score": None, "confidence_band": "none"},
        sources=[], reasoning_round1="", reasoning_round2="",
        answer="根据当前文档库的内容，未找到与您问题直接相关的信息。",
    )
    events = list(cc.replay_events(value))
    types = [t for t, _ in events]
    assert types[0] == "tool_call" and types[-1] == "usage"
    assert "sources" not in types
    assert events[0][1]["result"]["source_count"] == 0


def test_replay_events_direct_answer_no_tool():
    """LLM 直接答路径（未检索/非文档问题）：无 tool_call/sources，仅 token + usage"""
    value = _payload(decided_retrieve=False, rule_override=False, tool_result=None, tool_status=None)
    events = list(cc.replay_events(value))
    types = [t for t, _ in events]
    # 直接答路径：reasoning* → token* → usage，无 tool_call/sources（reasoning 可非空）
    assert "reasoning" in types and "token" in types and types[-1] == "usage"
    assert "tool_call" not in types and "sources" not in types
    assert events[-1][1]["cached"] is True
