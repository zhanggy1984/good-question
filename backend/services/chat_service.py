"""聊天业务逻辑：会话管理 + RAG 链 + 对话记忆压缩 + SSE 流式生成"""
import json
import logging
import re
import time
from functools import lru_cache

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.orm import Session

from config import settings
from database import SessionLocal
from models import ChatMessage, ChatSession
from schemas.common import Page
from services.llm_service import get_llm
from services.retrieval_service import HybridRetriever
from utils.exceptions import ForbiddenError, NotFoundError

logger = logging.getLogger("native_rag")

# 记忆压缩配置
MAX_MESSAGES_BEFORE_COMPRESS = 20  # >10 轮（20 条）触发压缩
KEEP_RECENT_MESSAGES = 6           # 保留最近 3 轮（6 条）
TITLE_MAX_CHARS = 30

# 二期 function calling：检索由 LLM 自主决定。system prompt 无 {context} 占位符——
# 检索结果由工具执行器经 tool 消息回传，模型基于 tool 结果作答（见 stream_chat agent loop）。
SYSTEM_PROMPT = """你是「好问」文档问答助手，基于文档库内容回答问题。

工具使用规则：
1. 用户询问文档库中的事实/信息/规则/流程，或要求总结文档内容时，必须先调用 hybrid_retrieve 工具检索相关资料。
2. 仅问候、寒暄、自我介绍等与文档无关的对话可直接回答，不调用工具。
3. 收到检索结果后严格基于结果内容回答，可用 [来源N] 标注引用片段；不得编造结果中不存在的事实。
4. 检索结果为空时，如实回答“文档中未找到相关信息”，不要编造“找到”或具体事实。
5. 若检索结果标注低置信，说明相关性存疑；不足以支撑回答时如实说明，不要勉强作答。

对话历史摘要（早期对话已压缩，供参考）：
{summary}"""

# hybrid_retrieve 工具定义（DeepSeek function calling 用）；query 由 LLM 自主生成
RETRIEVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "hybrid_retrieve",
        "description": (
            "在文档库中检索与用户问题相关的资料。"
            "用户询问文档库中的事实、规则、流程、条款，或要求总结文档内容时，必须先调用本工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于检索的查询词，通常是用户问题的原文或核心关键词",
                },
            },
            "required": ["query"],
        },
    },
}

# agent loop 循环上限：第一轮带 tools 判断是否检索，命中后第二轮作答不再传 tools（防再循环）
MAX_TOOL_ROUNDS = 2

# 未命中固定话术：事实类查询在 docs 空时不调 LLM（实测空 context 下 DeepSeek 稳定编造
# "合理答案"，如编造工资发放日为 10 号），直接如实回答；闲聊类请求才交给 LLM 走引导话术
_NOT_FOUND_ANSWER = (
    "根据当前文档库的内容，未找到与您问题直接相关的信息。"
    "您可以换个问法再试，或确认该问题是否属于文档库涵盖的范围。"
)

# 意图无法识别（unknown）的澄清话术：区别于 query 的"未找到"——unknown 不是"没检索到答案"
# 而是"没听懂意图"，措辞引导澄清而非断言文档无此内容。同样不调 LLM、不编造（防幻觉不变）。
_UNKNOWN_ANSWER = (
    "抱歉，我还没完全理解您的问题。"
    "请换个说法，或告诉我您想了解文档库中哪方面的信息。"
)

# 轻量规则意图分类词表（仅用于 docs 空时区分"引导话术"与"如实未找到"）
# 身份类闲聊：最特定（"你是谁/你能做什么"），即便含疑问词也判闲聊——优先于查询意图
_IDENTITY_SMALLTALK = (
    "你是谁", "你叫什么", "你能做什么", "你会什么", "你是什么", "你是干嘛的",
    "自我介绍", "介绍一下你自己", "介绍下你自己",
)
# 查询意图标记：疑问词 / 疑问号 / 查询动词。命中判查询——防编造的最关键闸门
# （query 优先于一般闲聊：带问候前缀的查询如"你好，工资几号发"必须判查询，
#   绝不能交给 LLM 在空 context 下编造）
_QUERY_MARKERS = (
    "？", "?", "什么", "怎么", "如何", "为什么", "几", "多少", "哪些", "何时",
    "哪里", "是不是", "能否", "可以吗", "有没有", "是否", "查", "找", "帮",
    "告诉", "解释", "总结", "说明", "写", "列出", "推荐",
    # 领域词（演示文档场景）与口语疑问词：减少冷门话术滑向 unknown
    "到账", "发放", "报销", "请假", "安装", "部署", "命令", "配置",
    "多久", "几天", "啥时候", "咋",
)
# 寒暄整句（正则 fullmatch）：覆盖"最近/今天 + 心情/状态/过得 + 怎么样/咋样"等口语变体，
# 含"怎么/咋"但整句是寒暄则非查询。优先级放在 query 之前——关键约束是 fullmatch 整句：
# "今天心情怎么样，工资几号发"（带查询）不命中 → 继续走 query。
# 原固定短语清单（"最近怎么样/今天怎么样/最近在忙什么/吃了没/干嘛呢/忙不忙"）全部被下列模式覆盖。
_CASUAL_PATTERNS = (
    # 人称前缀可选：覆盖"你最近怎么样/您最近咋样"（人称在时间词前）与无人称"最近怎么样"
    re.compile(r"^(你|您)?(最近|今天|这两天|这段时间)(心情|状态|过得|身体)?怎么样[啊呀呢嘛!！？?\s]*$"),
    re.compile(r"^(你|您)?(最近|今天|这两天|这段时间)(心情|状态|过得|身体)?咋样[啊呀呢嘛!！？?\s]*$"),
    re.compile(r"^(最近|这两天)?(在)?忙(什么|不忙)[啊呀呢嘛!！？?\s]*$"),
    re.compile(r"^(你|您)(现在)?(在)?(干嘛|干啥|忙什么|做什么|咋了|怎么了)呢?[啊呀?！?\s]*$"),
    re.compile(r"^你呢?[啊呀?！?\s]*$"),
    re.compile(r"^吃了(没|吗)[啊呀?！?\s]*$"),
)
# 明确回指词：unknown 时仅命中这些才回看 history 归队（"就这个/还有呢"延续上一轮意图）。
# 收紧条件避免跨话题短词（如"你呢"）被归错队——"你呢"已是闲聊反问，由 _CASUAL_PATTERNS 直接识别。
_REFERENTIAL_WORDS = (
    "就这个", "还有呢", "然后呢", "再说说", "再详细点", "这个呢", "那个呢",
)
# 一般闲聊：问候 / 感谢 / 道别。仅当无查询意图时才判闲聊
_SOCIAL_SMALLTALK = (
    "你好", "您好", "嗨", "哈喽", "嗨喽", "hello", "hi", "在吗", "在不在",
    "早上好", "中午好", "下午好", "晚上好",
    "谢谢", "感谢", "辛苦你了", "谢谢你",
    "再见", "拜拜", "回头聊", "下次聊",
)


def _classify_intent(text: str, history: list | None = None) -> str:
    """轻量规则意图分类：smalltalk / query / unknown（docs 空时决定走 LLM 还是固定话术）

    优先级：身份闲聊 > 寒暄整句 > 查询意图 > 一般闲聊 > unknown（回看历史最近一条 user）。
    - 身份闲聊最特定，即便含疑问词（"你能做什么"）也判闲聊，走 LLM 引导话术；
    - 寒暄整句（"最近怎么样/今天心情怎么样"）含"怎么"但整句匹配、非查询，优先于 query；
    - 查询意图优先于一般闲聊：带问候前缀的查询（"你好，工资几号发"）必须判查询，
      避免交给 LLM 在空 context 下编造；
    - unknown（既非闲聊也非明确查询）保守判非闲聊 → 走澄清话术（同样不调 LLM，防编造）；
      history 可选：unknown 且命中明确回指词（"就这个/还有呢"）时回看最近一条 user 消息
      的意图归队延续上一轮；"你呢"等闲聊反问已由 _CASUAL_PATTERNS 直接识别，不靠 history
      归队（避免跨话题短词被归错队）。
    抽成纯函数便于单元测试覆盖边界（见 tests/test_chat_service.py）。
    """
    t = text.lower().strip()
    if not t:
        return "unknown"
    if any(k in t for k in _IDENTITY_SMALLTALK):
        return "smalltalk"
    if any(p.fullmatch(t) for p in _CASUAL_PATTERNS):
        return "smalltalk"
    if any(q in t for q in _QUERY_MARKERS):
        return "query"
    if any(k in t for k in _SOCIAL_SMALLTALK):
        return "smalltalk"
    if history and any(w in t for w in _REFERENTIAL_WORDS):
        prev = _last_user_intent(history, current_text=t)
        if prev in ("smalltalk", "query"):
            return prev
    return "unknown"


def _last_user_intent(history: list, current_text: str) -> str | None:
    """回看最近一条 user 消息的意图（跳过当前问题本身），供 unknown 回指延续"""
    for msg in reversed(history):
        if getattr(msg, "type", None) != "human":
            continue
        prev = (getattr(msg, "content", "") or "").strip().lower()
        if not prev or prev == current_text:
            continue
        return _classify_intent(prev)
    return None


def _is_smalltalk(text: str, history: list | None = None) -> bool:
    """是否闲聊意图（_classify_intent 的布尔化，兼容调用点与旧测试）"""
    return _classify_intent(text, history) == "smalltalk"


# ════════ 会话 CRUD ════════

def create_session(db: Session, user_id: int, library_id: int) -> ChatSession:
    """创建会话（绑定文档库）"""
    session = ChatSession(user_id=user_id, library_id=library_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.debug("[session.create] 创建会话 id=%s user=%s library=%s", session.id, user_id, library_id)
    return session


def list_sessions(
    db: Session, user_id: int, library_id: int | None, page: int, page_size: int
) -> Page:
    """当前用户的会话列表（可按库过滤，按更新时间倒序）"""
    query = db.query(ChatSession).filter(ChatSession.user_id == user_id)
    if library_id:
        query = query.filter(ChatSession.library_id == library_id)
    total = query.count()
    items = (
        query.order_by(ChatSession.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return Page(items=items, total=total, page=page, page_size=page_size)


def _get_owned_session(db: Session, session_id: int, user_id: int) -> ChatSession:
    """查询会话并校验归属（会话隔离）"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session is None:
        raise NotFoundError("会话不存在")
    if session.user_id != user_id:
        raise ForbiddenError("无权访问该会话")
    return session


def delete_session(db: Session, session_id: int, user_id: int) -> None:
    """删除会话（仅所有者）"""
    session = _get_owned_session(db, session_id, user_id)
    db.delete(session)
    db.commit()


def get_session_detail(db: Session, session_id: int, user_id: int):
    """会话详情 + 历史消息"""
    session = _get_owned_session(db, session_id, user_id)
    # 同 created_at 秒级排序不稳定，用自增主键保证插入顺序（见 _build_context 注释）
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.id.asc())
        .all()
    )
    return session, messages


# ════════ 对话记忆 ════════

def _build_context(db: Session, session: ChatSession):
    """构建 prompt 所需：summary + 最近 N 条消息（history）"""
    # 用自增主键 id 排序而非 created_at：同一次 commit 写入的 user+assistant
    # 两条消息 created_at 同为秒级，desc 排序不稳定会打乱历史顺序（导致 LLM 串味）
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.id.desc())
        .limit(KEEP_RECENT_MESSAGES)
        .all()
    )
    messages.reverse()
    history = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in messages
    ]
    return session.summary or "", history


def _compress_memory(db: Session, session: ChatSession) -> None:
    """超过 10 轮：旧消息 + 现有摘要压缩为新摘要（保留最近 3 轮原文）"""
    # 同 created_at 秒级排序不稳定，用自增主键保证插入顺序（见 _build_context 注释）
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.id.asc())
        .all()
    )
    if len(messages) <= MAX_MESSAGES_BEFORE_COMPRESS:
        return
    old = messages[:-KEEP_RECENT_MESSAGES]
    dialogue = "\n".join(
        f"{'用户' if m.role == 'user' else '助手'}: {m.content}" for m in old
    )
    compress_prompt = f"""请将以下对话内容压缩为一段简洁的中文摘要，保留关键信息和上下文。
已有的历史摘要：{session.summary or '无'}

对话内容：
{dialogue}

请输出压缩后的摘要（不超过 200 字）："""
    try:
        llm = get_llm(streaming=False)
        new_summary = llm.invoke(compress_prompt).content.strip()
        session.summary = new_summary
        db.commit()
        logger.debug("[memory] 会话 %s 记忆已压缩", session.id)
    except Exception as e:
        logger.warning("[memory] 压缩失败: %s", e)


# ════════ RAG 链 ════════

def _format_docs(docs) -> str:
    """检索结果格式化为 prompt 的 context（带来源编号）"""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        src = f"[来源{i}] {meta.get('document_name', '未知文档')}"
        heading = [h for h in (meta.get("heading_path") or []) if h]
        if heading:
            src += f" > {' > '.join(heading)}"
        parts.append(f"{src}\n{doc.content}")
    return "\n\n---\n\n".join(parts)


def _build_messages(summary: str, history: list, question: str) -> list[dict]:
    """构建 DeepSeek API 的 messages（system + history + 末尾 user 问题）

    二期 function calling：system prompt 不含检索 context（由 hybrid_retrieve 的
    tool 结果经 tool 消息回传），第一轮与第二轮都以此骨架起步，第二轮再追加
    assistant(tool_calls) + tool 消息。
    """
    system = SYSTEM_PROMPT.format(summary=summary)
    messages = [{"role": "system", "content": system}]
    for m in history:
        role = "user" if m.type == "human" else "assistant"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": question})
    return messages


def _is_low_confidence(max_score: float | None) -> bool:
    """低置信判定：精排最高分落在 [similarity_threshold_low, rerank_low_confidence_threshold) 区间

    低于 LOW 判"文档无关"（_rerank 已返回空，此处恒为 False）；等于/高于低置信阈值判正常。
    抽成纯函数便于单元测试覆盖边界（见 tests/test_chat_service.py）。
    """
    return (
        max_score is not None
        and max_score >= settings.similarity_threshold_low
        and max_score < settings.rerank_low_confidence_threshold
    )


@lru_cache(maxsize=1)
def _git_sha() -> str:
    """尽力取当前 git commit sha（进程内缓存，仅首次 spawn 子进程；容器内可能无 .git）

    每次 chat 请求都 spawn 一次 git 子进程不划算，commit 在进程生命周期内不变，
    故用 lru_cache 缓存一次。
    """
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except Exception:
        return ""


def _knowledge_version(library_id: int) -> str:
    """知识库版本标识：取该会话所在文档库的最近上传时间（尽力），失败返回空串。

    评测契约要求 knowledge_version 用于黄金答案时效校验；以库级最新文档时间为锚。
    必须按 library_id 过滤：多库部署下取全局 max 会串库，知识版本标识失去意义。
    """
    try:
        from models.document import Document
        from sqlalchemy import func
        db = SessionLocal()
        try:
            v = (
                db.query(func.max(Document.created_at))
                .filter(Document.library_id == library_id)
                .scalar()
            )
            return v.strftime("%Y%m%d%H%M%S") if v else ""
        finally:
            db.close()
    except Exception:
        return ""


def _tool_call_event(tool_acc: dict) -> dict:
    """把按 index 累积的 tool_calls 组装成 yield 事件（arguments 保持 JSON 字符串，不 parse）"""
    calls = [
        {
            "id": item["id"],
            "type": "function",
            "function": {"name": item["name"], "arguments": item["arguments"]},
        }
        for item in tool_acc.values()
    ]
    return {"type": "tool_call", "tool_calls": calls}


def _stream_deepseek(messages: list[dict], tools: list[dict] | None = None):
    """直连 DeepSeek 流式调用，解析 reasoning_content / content / tool_calls / usage

    yield {"type": "reasoning"|"content"|"tool_call"|"usage", ...}
    - reasoning/content：增量文本
    - tool_call：LLM 决定调用工具时（finish_reason=="tool_calls"）一次性 flush 完整参数
      （tool_calls 按 delta 分片按 index 累加，arguments 为 JSON 字符串）
    - usage：流末 usage chunk（include_usage 开启后返回，choices 为空），data 为 usage 对象
    """
    payload = {
        "model": settings.deepseek_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        # 评测契约要求透出真实 token 消耗；DeepSeek 流式默认不返回 usage，须显式开启
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools  # 不设 tool_choice，默认 auto，由 LLM 自主决定
    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.stream(
        "POST",
        f"{settings.deepseek_base_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    ) as resp:
        tool_acc: dict = {}  # index -> {"id", "name", "arguments"}；arguments 增量拼接
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
                # 流末 usage chunk：choices 为空但有 usage 字段（须在取 delta 前判断）
                if chunk.get("usage"):
                    yield {"type": "usage", "usage": chunk["usage"]}
                    continue
                choice = chunk["choices"][0]
                delta = choice["delta"]
                finish_reason = choice.get("finish_reason")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            # tool_calls 按 index 分片累加（id/name 首个分片携带，arguments 增量拼接）
            if "tool_calls" in delta:
                for tc in delta.get("tool_calls", []):
                    item = tool_acc.setdefault(
                        tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.get("id"):
                        item["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        item["name"] = fn["name"]
                    if fn.get("arguments"):
                        item["arguments"] += fn["arguments"]
            if "reasoning_content" in delta and delta["reasoning_content"]:
                yield {"type": "reasoning", "content": delta["reasoning_content"]}
            if "content" in delta and delta["content"]:
                yield {"type": "content", "content": delta["content"]}
            # 结束信号：finish_reason=="tool_calls" 的 chunk 通常带着最后一个 tool_calls
            # 分片（须先累积再 flush，保证参数完整）
            if finish_reason == "tool_calls" and tool_acc:
                yield _tool_call_event(tool_acc)
                tool_acc = {}
        # [DONE] 兜底：个别实现不返回 finish_reason 也能累积到 tool_calls
        if tool_acc:
            yield _tool_call_event(tool_acc)


# ════════ 二期 function calling 支撑 ════════

def _ts() -> int:
    """unix ms（评测契约要求，agent 侧生成）"""
    return int(time.time() * 1000)


def _json_log(kind: str, **fields) -> None:
    """结构化 JSON 日志：tool 决策与请求汇总各打一行 JSON，供请求分析监控采集"""
    logger.info(json.dumps({"ts": _ts(), "kind": kind, **fields}, ensure_ascii=False))


def _confidence_band(max_score: float | None) -> str:
    """检索置信档三档：none（无分/低于 LOW 视为无关）｜low（[LOW, 低置信阈值) 相关性存疑）｜high"""
    if max_score is None or max_score < settings.similarity_threshold_low:
        return "none"
    if max_score < settings.rerank_low_confidence_threshold:
        return "low"
    return "high"


def _execute_retrieve_tool(library_id: int, query: str) -> dict:
    """执行 hybrid_retrieve 工具：检索 + 组装 LLM 上下文与 SSE sources

    返回 dict：context（[来源N] 格式化，供第二轮 LLM 的 tool 消息）、sources（前端引用卡片）、
    source_count / max_score / confidence_band（tool_call 事件 result 与监控日志用）
    """
    retriever = HybridRetriever(library_id=library_id)
    chunks = retriever.invoke(query)
    max_score = retriever.max_rerank_score
    sources = [
        {
            "document_name": c.metadata.get("document_name", "未知文档"),
            "heading_path": [h for h in (c.metadata.get("heading_path") or []) if h],
            "chunk_content": c.content[:200],
            "chunk_index": c.metadata.get("chunk_index"),
            "total_chunks": c.metadata.get("total_chunks"),
        }
        for c in chunks
    ]
    return {
        "context": _format_docs(chunks),
        "sources": sources,
        "source_count": len(chunks),
        # rerank 返回 numpy float32，进 SSE tool_call result 与 JSON 日志须转原生 float（json.dumps 不认 float32）
        "max_score": float(max_score) if max_score is not None else None,
        "confidence_band": _confidence_band(max_score),
    }


# ════════ 流式聊天 ════════

def stream_chat(session_id: int, user_content: str):
    """SSE 生成器：yield (event_type, data_dict)

    二期 function calling 编排：LLM 第一轮带 hybrid_retrieve 工具自主决定是否检索，
    命中则经 tool 消息回传结果、第二轮作答；检索空走规则三路兜底（防幻觉）。

    使用独立 db session（避免请求 db 生命周期/跨请求状态问题）。
    事件顺序（二期）：
      - 不检索：meta → reasoning*/token* → usage → done
      - 检索命中：meta → reasoning* → tool_call → sources → reasoning*/token* → usage → done
      - 检索空 + query/unknown：meta → reasoning* → tool_call → token(固定话术) → usage → done
    LLM 失败时 yield error。

    评测契约（§5.1）对齐（保留事件名，前端零改动）：
    - meta / tool_call / usage 事件与全事件 ts 字段，供评测平台采集
    - 平台侧 field_map 把 token→answer 映射，reasoning/token 的 data 兼容读 content 或 delta
    """
    db = SessionLocal()
    try:
        t_start = time.time()
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session is None:
            yield ("error", {"message": "会话不存在"})
            return

        # 0. meta：环境快照（评测契约；模型名去前缀归一，git_sha 尽力取）
        yield ("meta", {
            "agent": "good-question",
            "model": settings.deepseek_model,
            "interface": "/api/chat",
            "contract_version": "1.0",
            "git_sha": _git_sha(),
            "knowledge_version": _knowledge_version(session.library_id),
            "ts": _ts(),
        })

        summary, history = _build_context(db, session)
        llm_rounds = 0
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        sources: list = []
        full_answer = ""
        full_reasoning = ""

        def _merge_usage(usage: dict) -> None:
            """多轮调用的 token 消耗合并进 usage_total（done 前统一发出一个 usage 事件）"""
            for k in usage_total:
                usage_total[k] += usage.get(k, 0) or 0

        # 1. 第一轮：LLM 带 hybrid_retrieve 工具，自主决定是否检索
        messages = _build_messages(summary, history, user_content)
        tool_calls = None
        try:
            for ev in _stream_deepseek(messages, tools=[RETRIEVE_TOOL_SCHEMA]):
                llm_rounds = 1
                if ev["type"] == "usage":
                    _merge_usage(ev["usage"])
                elif ev["type"] == "reasoning":
                    full_reasoning += ev["content"]
                    yield ("reasoning", {"content": ev["content"], "delta": ev["content"], "ts": _ts()})
                elif ev["type"] == "tool_call":
                    tool_calls = ev["tool_calls"]
                else:  # content：LLM 直接回答（决定不检索或未命中前先输出思考文本）
                    full_answer += ev["content"]
                    yield ("token", {"content": ev["content"], "delta": ev["content"], "ts": _ts()})
        except Exception as e:
            logger.error("[chat] LLM 第一轮流式失败: %s", e)
            yield ("error", {"message": "LLM 调用失败，请稍后重试"})
            return

        decided_retrieve = tool_calls is not None
        result = None
        if decided_retrieve:
            # 2. LLM 决定检索：执行工具，检索完成后再发 tool_call（result 带真实结果）
            try:
                args = json.loads(tool_calls[0]["function"]["arguments"] or "{}")
                query = args.get("query") or user_content
                result = _execute_retrieve_tool(session.library_id, query)
                status = "ok"
            except Exception as e:
                logger.warning("[chat] 检索工具执行失败: %s", e)
                query = user_content
                result = None
                status = "error"
            yield ("tool_call", {
                "id": f"retrieve-{_ts()}",
                "name": "hybrid_retrieve",
                "args": {"query": query},
                "result": (
                    {k: result[k] for k in ("source_count", "max_score", "confidence_band")}
                    if result else {}
                ),
                "status": status,
                "ts": _ts(),
            })

            # 3. 命中：推 sources + 第二轮 LLM 基于 tool 结果作答；空：规则三路兜底
            if result and result["source_count"] > 0:
                sources = result["sources"]
                yield ("sources", {"sources": sources, "ts": _ts()})
                # DeepSeek tool 轮次必须回传 assistant 的 tool_calls + reasoning_content，否则报错
                messages += [
                    {"role": "assistant", "content": "", "reasoning_content": full_reasoning,
                     "tool_calls": tool_calls},
                    {"role": "tool", "tool_call_id": tool_calls[0]["id"],
                     "content": json.dumps(
                         {"context": result["context"], "confidence_band": result["confidence_band"]},
                         ensure_ascii=False)},
                ]
                try:
                    for ev in _stream_deepseek(messages):  # 第二轮不带 tools，防再循环
                        llm_rounds = 2
                        if ev["type"] == "usage":
                            _merge_usage(ev["usage"])
                        elif ev["type"] == "reasoning":
                            full_reasoning += ev["content"]
                            yield ("reasoning", {"content": ev["content"], "delta": ev["content"], "ts": _ts()})
                        elif ev["type"] == "tool_call":
                            continue
                        else:
                            full_answer += ev["content"]
                            yield ("token", {"content": ev["content"], "delta": ev["content"], "ts": _ts()})
                except Exception as e:
                    logger.error("[chat] LLM 第二轮流式失败: %s", e)
                    yield ("error", {"message": "LLM 调用失败，请稍后重试"})
                    return
            else:
                # 检索空 → 规则三路兜底（用户确认）：query 如实"未找到"、unknown 澄清、smalltalk 引导
                intent = _classify_intent(user_content, history)
                if intent != "smalltalk":
                    answer = _NOT_FOUND_ANSWER if intent == "query" else _UNKNOWN_ANSWER
                    full_answer = answer
                    yield ("token", {"content": answer, "delta": answer, "ts": _ts()})
                else:
                    # 寒暄却被 LLM 检索且空（模型行为异常，低频）：第二轮 LLM 自然引导，不传 tools
                    messages += [
                        {"role": "assistant", "content": "", "reasoning_content": full_reasoning,
                         "tool_calls": tool_calls},
                        {"role": "tool", "tool_call_id": tool_calls[0]["id"],
                         "content": json.dumps(
                             {"context": "（本次未检索到相关文档内容）", "confidence_band": "none"},
                             ensure_ascii=False)},
                    ]
                    try:
                        for ev in _stream_deepseek(messages):
                            llm_rounds = 2
                            if ev["type"] == "usage":
                                _merge_usage(ev["usage"])
                            elif ev["type"] == "reasoning":
                                full_reasoning += ev["content"]
                                yield ("reasoning", {"content": ev["content"], "delta": ev["content"], "ts": _ts()})
                            elif ev["type"] == "tool_call":
                                continue
                            else:
                                full_answer += ev["content"]
                                yield ("token", {"content": ev["content"], "delta": ev["content"], "ts": _ts()})
                    except Exception as e:
                        logger.error("[chat] LLM 引导轮流式失败: %s", e)
                        yield ("error", {"message": "LLM 调用失败，请稍后重试"})
                        return

        # 4. 监控日志：tool 决策（规则分类器仅对比，不干预——用户确认"信任 LLM"）
        rule_intent = _classify_intent(user_content, history)
        _json_log(
            "tool_decision",
            session_id=session_id,
            iteration=1,
            llm_decided=decided_retrieve,
            tool="hybrid_retrieve" if decided_retrieve else None,
            source_count=(result or {}).get("source_count"),
            max_score=(result or {}).get("max_score"),
            confidence_band=(result or {}).get("confidence_band"),
            rule_intent=rule_intent,
            rule_agree=(rule_intent == "query") == decided_retrieve,
            llm_rounds=llm_rounds,
            total_tokens=usage_total["total_tokens"],
            duration_ms=int((time.time() - t_start) * 1000),
        )

        # 5. usage（多轮合并为一个）+ 保存消息 + 记忆压缩
        yield ("usage", {**usage_total, "ts": _ts()})
        assistant_id = _save_messages(db, session, user_content, full_answer, sources)
        _compress_memory(db, session)
        _json_log(
            "chat_request",
            session_id=session_id,
            llm_rounds=llm_rounds,
            total_tokens=usage_total["total_tokens"],
            decided_retrieve=decided_retrieve,
            duration_ms=int((time.time() - t_start) * 1000),
        )

        # 6. done
        yield ("done", {"message_id": assistant_id, "ts": _ts()})
    finally:
        db.close()


def _save_messages(
    db: Session, session: ChatSession, user_content: str, answer: str, sources: list
) -> int:
    """保存用户 + 助手消息，更新 message_count 并生成标题（一次 commit）"""
    db.add(ChatMessage(session_id=session.id, role="user", content=user_content))
    msg = ChatMessage(
        session_id=session.id, role="assistant", content=answer, sources_json=sources
    )
    db.add(msg)
    session.message_count += 2
    # 首条消息生成标题（截取前 N 字）
    if not session.title:
        session.title = user_content[:TITLE_MAX_CHARS]
    db.commit()
    db.refresh(msg)
    return msg.id
