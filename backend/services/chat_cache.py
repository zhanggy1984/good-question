"""DeepSeek 问答缓存（Redis）：新会话首句 + 同库 + 同问题的完整问答缓存重放。

计费动机：评测/示例/跨用户重复问同一问题时，每次都会 1~2 轮流式调用 DeepSeek。
缓存命中后重放 SSE 事件流，跳过 LLM，命中即省整轮计费。

关键约束（方向性决策，见方案评审）：
- 只缓存"空上下文"的问答（新会话首句）。多轮对话上下文每次不同，命中率≈0，
  查了也是白查；key 不含上下文维度，非空 history 由调用方直接跳过（不查不写），
  避免多轮问答污染无上下文缓存导致张冠李戴。
- 不做语义缓存/规划缓存：RAG 场景 embedding 相似≠答案可复用（"工资几号发" vs
  "报销几号发" 结构相似但答案不同），近似命中会给错答案；规划缓存与精确缓存命中
  场景完全重叠，是同场景次优解。
- 缓存是纯优化：Redis 连接失败/超时静默降级为无缓存，绝不影响主链路。
- 答案绑定库（key 含 library_id）；文档更新后由 document_service 上传成功时
  flush_library 清库缓存，TTL 兜底（默认 2h）。
- 命中时 usage 事件带 cached 标记，计费统计须排除命中请求，避免 token 虚高。
"""
import hashlib
import json
import logging
import time
from typing import Any, Iterator

from config import settings

logger = logging.getLogger("native_rag")

try:
    import redis as _redis_lib
except ImportError:  # 宿主无 redis 依赖时降级为无缓存（容器内 Dockerfile 已装）
    _redis_lib = None

# 缓存结构版本：重放/落库逻辑演进时 +1，旧缓存自然过期不兼容
_CACHE_VERSION = 2

# 连接失败熔断：Redis 不可用时静默降级，_DISABLE_COOLDOWN_SECONDS 内不再尝试
# 连接（避免每次请求都白等 socket timeout，也避免反复打错误日志）
_DISABLED_UNTIL: float = 0.0
_DISABLE_COOLDOWN_SECONDS = 60

# 单例 Redis 客户端：redis-py 的 Redis 对象内部自带连接池，复用同一实例即复用连接池。
# 每次请求都 from_url 新建连接是热路径浪费；熔断置空，冷却期后按需重建。
_client_instance: Any | None = None


def _disable(e: Exception) -> None:
    """Redis 异常熔断：记录降级日志，冷却期内不再尝试连接，丢弃坏连接实例"""
    global _DISABLED_UNTIL, _client_instance
    _DISABLED_UNTIL = time.time() + _DISABLE_COOLDOWN_SECONDS
    _client_instance = None
    logger.warning(
        "[chat.cache] Redis 不可用，缓存降级为关闭（%ss 后重试）: %s",
        _DISABLE_COOLDOWN_SECONDS, e,
    )


def _client(ignore_circuit_break: bool = False) -> Any | None:
    """惰性取 Redis 客户端（单例复用连接池）；配置关闭 / 未装依赖 / 熔断期内返回 None

    ignore_circuit_break：管理操作（flush_library）不受熔断限制——文档上传低频，
    始终尝试连接，确保 Redis 恢复后下次上传能清掉残留缓存（TTL 仅兜底）。
    """
    global _client_instance
    if not settings.chat_cache_enabled or _redis_lib is None:
        return None
    if not ignore_circuit_break and time.time() < _DISABLED_UNTIL:
        return None
    if _client_instance is None:
        _client_instance = _redis_lib.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    return _client_instance


def _key(library_id: int, question: str) -> str:
    """缓存 key：库隔离 + 模型 + 问题 sha256。

    不含会话/用户（跨会话共享才有命中价值）；仅存空上下文问答，故 key 无上下文维度。
    必须含 model：config 支持"切换模型只改 DEEPSEEK_MODEL"，换模型后旧答案要自然失效。
    """
    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:16]
    return f"goodq:chat:{library_id}:{settings.deepseek_model}:{digest}"


def get_cached(library_id: int, question: str) -> dict | None:
    """查询缓存：命中返回 value dict（含版本校验），未命中/Redis 不可用返回 None"""
    client = _client()
    if client is None:
        return None
    try:
        raw = client.get(_key(library_id, question))
    except Exception as e:
        _disable(e)
        return None
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if value.get("v") != _CACHE_VERSION:
        return None
    return value


def set_cached(library_id: int, question: str, payload: dict) -> None:
    """写入缓存（TTL 由配置控制）。payload 不含版本字段，内部补 v。"""
    client = _client()
    if client is None:
        return
    value = {"v": _CACHE_VERSION, **payload}
    try:
        client.set(
            _key(library_id, question),
            json.dumps(value, ensure_ascii=False),
            ex=settings.chat_cache_ttl_seconds,
        )
    except Exception as e:
        _disable(e)


def flush_library(library_id: int) -> None:
    """清空某库的全部问答缓存（文档上传/重新解析成功后调用，堵 TTL 窗口内旧答案）

    不受熔断限制（ignore_circuit_break=True）：低频管理操作，始终尝试连接，
    Redis 恢复后下次上传即可清掉残留缓存（而不是等 TTL 到期）。
    """
    client = _client(ignore_circuit_break=True)
    if client is None:
        return
    try:
        keys = list(client.scan_iter(match=f"goodq:chat:{library_id}:*", count=100))
        if keys:
            client.delete(*keys)
            logger.info("[chat.cache] 清库缓存 library=%s 清掉 %s 条", library_id, len(keys))
    except Exception as e:
        _disable(e)


def replay_events(value: dict) -> Iterator[tuple[str, dict]]:
    """缓存命中时重放 SSE 事件（事件序与 stream_chat 真实流程对齐，前端/评测契约零改动）。

    yield (event_type, data)；meta 由调用方在查缓存前发出，done 由调用方落库后发出。
    事件顺序与真实流程一致：
      reasoning(首轮决策思考) → tool_call → sources → reasoning(次轮作答思考) → token* → usage(cached)
    缓存区分首/次轮 reasoning（set_cached 时按 full_reasoning 前缀快照拆分），命中路径
    的事件序因此与真实流程一致；tool_call 透传 intent / non_doc_question——前端据这两
    个字段精确决定"已检索但空"提示是否显示（smalltalk 问候、non_doc 计算豁免的空命中
    走 LLM 自然答，缓存命中重放时也不该误提示"未找到"）。
    reasoning/token 按字符分片重放，保留打字机效果与 delta 语义（前端拼接后全文一致）。
    ts 从当前毫秒起严格单调递增（+1/事件）：逐字符分片可能同一毫秒完成，复用独立
    _ts() 会产出重复时间戳，评测契约若校验 ts 严格递增会异常。
    """
    ts_next = int(time.time() * 1000)

    def _next_ts() -> int:
        nonlocal ts_next
        ts_next += 1
        return ts_next

    decided = value["decided_retrieve"]
    override = value["rule_override"]
    tr = value["tool_result"]
    for ch in value.get("reasoning_round1") or []:
        yield ("reasoning", {"content": ch, "delta": ch, "ts": _next_ts()})
    if decided or override:
        yield ("tool_call", {
            "id": f"retrieve-{_next_ts()}",
            "name": "hybrid_retrieve",
            "args": {"query": value["query"]},
            "result": (
                {k: tr[k] for k in ("source_count", "max_score", "confidence_band")}
                if tr else {}
            ),
            "status": value.get("tool_status") or "ok",
            # 意图透传与真实 tool_call 事件一致，防前端误判"已检索但空"提示（见函数 docstring）
            "intent": value.get("intent"),
            "non_doc_question": value.get("non_doc_question"),
            "ts": _next_ts(),
        })
        if tr and tr["source_count"] > 0:
            yield ("sources", {"sources": value["sources"], "ts": _next_ts()})
    for ch in value.get("reasoning_round2") or []:
        yield ("reasoning", {"content": ch, "delta": ch, "ts": _next_ts()})
    for ch in value["answer"]:
        yield ("token", {"content": ch, "delta": ch, "ts": _next_ts()})
    usage = dict(value["usage"])
    usage["cached"] = True
    yield ("usage", {**usage, "ts": _next_ts()})
