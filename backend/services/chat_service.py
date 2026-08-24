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

SYSTEM_PROMPT = """你是文档问答助手，基于提供的文档内容回答问题。

回答要求：
1. 依据提供的文档内容回答，可从文档合理推断和总结，不必逐字照搬
2. 回答中可用 [来源N] 标注引用的文档片段
3. 只有文档完全未涉及该问题时，才回答"文档中未找到相关信息"
4. 不得编造文档中不存在的具体事实
5. 若请求与文档无关（如问候、自我介绍、闲聊），且未检索到参考内容时，请自然礼貌地
   说明你是文档问答助手、可解答文档库内问题，并邀请用户提问；不要编造"文档找到/查到"等表述。
   注意：检索未命中不等于请求与文档无关——若问题是在询问文档内容（如查找条款、总结资料），
   即使未检索到相关文档，也应走第 3 条如实回答"文档中未找到相关信息"，不得使用本条的问候话术。
   打招呼场景可参考如下开场话术（可微调措辞，务必完整通顺）：
   "你好，我是文档问答助手，可以基于文档库中的内容为你查找、总结或理解文档信息并解答相关问题。请问有什么可以帮你的吗？"

对话历史摘要（早期对话已压缩，供参考）：
{summary}

参考资料（带来源编号）：
{context}"""

# 低置信兜底提示：检索到内容但相关性存疑时追加到 system prompt，
# 让 LLM 不基于边缘相关片段勉强作答/编造（见 stream_chat 的 low_confidence 分支）
LOW_CONFIDENCE_HINT = (
    "\n\n注意：本次检索到的资料与问题相关性不确定。"
    "若这些资料不足以支撑回答，请直接回答“文档中未找到相关信息”，"
    "不要基于相关性存疑的资料勉强作答，更不要编造。"
)

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


def _build_messages(
    summary: str, history: list, context: str, question: str, low_confidence: bool = False
) -> list[dict]:
    """构建 DeepSeek API 的 messages（system + history + human）；低置信档追加兜底提示"""
    system = SYSTEM_PROMPT.format(summary=summary, context=context)
    if low_confidence:
        system += LOW_CONFIDENCE_HINT
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


def _stream_deepseek(messages: list[dict]):
    """直连 DeepSeek 流式调用，解析 reasoning_content 与 content

    yield {"type": "reasoning"|"content"|"usage", ...}
    - reasoning/content：增量文本
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
                delta = chunk["choices"][0]["delta"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if "reasoning_content" in delta and delta["reasoning_content"]:
                yield {"type": "reasoning", "content": delta["reasoning_content"]}
            if "content" in delta and delta["content"]:
                yield {"type": "content", "content": delta["content"]}


# ════════ 流式聊天 ════════

def stream_chat(session_id: int, user_content: str):
    """SSE 生成器：yield (event_type, data_dict)

    使用独立 db session（避免请求 db 生命周期/跨请求状态问题）。
    事件顺序：meta → sources/tool_call → reasoning*/token* → usage → done（LLM 失败时 yield error）

    评测契约（§5.1）对齐（路径 A：保留原有事件名，前端零改动）：
    - 新增 meta / tool_call / usage 事件与全事件 ts 字段，供评测平台采集
    - 平台侧 field_map 把 token→answer 映射，reasoning/token 的 data 兼容读 content 或 delta
    """
    db = SessionLocal()
    try:
        t_start = time.time()
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session is None:
            yield ("error", {"message": "会话不存在"})
            return

        def _ts() -> int:
            """unix ms（评测契约要求，agent 侧生成）"""
            return int(time.time() * 1000)

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

        # 1. 混合检索（dense + 稀疏 → RRF 融合 → rerank）
        retriever = HybridRetriever(library_id=session.library_id)
        docs = retriever.invoke(user_content)

        # 两级置信档：最高分 < LOW 判"文档无关"返回空；落在 [LOW, 低置信阈值) 判"相关性存疑"，
        # 检索结果照常保留（不误杀），但提示 LLM 相关性存疑、不足以回答则如实说未找到
        max_score = retriever.max_rerank_score
        low_confidence = _is_low_confidence(max_score)
        logger.info(
            "[chat] 检索置信档 max_score=%s low_confidence=%s",
            "无" if max_score is None else round(max_score, 3), low_confidence,
        )

        # 2. 仅在检索到结果时推送 sources + tool_call（无结果时不展示空引用来源）
        sources = []
        if docs:
            sources = [
                {
                    "document_name": d.metadata.get("document_name", "未知文档"),
                    "heading_path": [h for h in (d.metadata.get("heading_path") or []) if h],
                    "chunk_content": d.content[:200],
                    "chunk_index": d.metadata.get("chunk_index"),
                    "total_chunks": d.metadata.get("total_chunks"),
                }
                for d in docs
            ]
            # 检索外显为标准 tool_call（评测契约；sources 保留给前端）
            yield ("tool_call", {
                "id": f"retrieve-{_ts()}",
                "name": "hybrid_retrieve",
                "args": {"query": user_content},
                "result": {"source_count": len(docs), "sources": sources},
                "status": "ok",
                "ts": _ts(),
            })
            yield ("sources", {"sources": sources, "ts": _ts()})
        logger.info("[chat] 检索完成 耗时=%.2fs 结果=%s，开始 LLM 流式", time.time() - t_start, len(docs))

        # 3. 构建上下文
        summary, history = _build_context(db, session)
        context = _format_docs(docs)
        if not docs:
            # 未检索到任何参考内容：显式告知 LLM，避免空 context 下幻觉"文档找到"
            context = "（本次未检索到相关文档内容）"
            # 防幻觉加固：事实类查询未命中直接走固定话术，不交给 LLM——实测空 context 下
            # DeepSeek 会稳定编造"合理答案"（如编造工资发放日为每月 10 号）。仅闲聊类请求
            # 继续走 LLM（SYSTEM_PROMPT 有专门的引导话术）。意图分类分三路：
            # query → 如实"未找到"；unknown（无法确定意图）→ 澄清引导（同样不调 LLM，
            #   防编造不变，但措辞是"没听懂"而非"没找到"）；smalltalk → 走 LLM 引导话术。
            intent = _classify_intent(user_content, history)
            if intent != "smalltalk":
                answer = _NOT_FOUND_ANSWER if intent == "query" else _UNKNOWN_ANSWER
                yield ("token", {"content": answer, "delta": answer, "ts": _ts()})
                # 未调 LLM 无真实 token 消耗，合成 0 计数对齐评测契约（usage 必须在 done 前）
                yield ("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "ts": _ts()})
                assistant_id = _save_messages(db, session, user_content, answer, sources)
                _compress_memory(db, session)
                yield ("done", {"message_id": assistant_id, "ts": _ts()})
                return

        # 4. 流式调用 DeepSeek（解析 reasoning_content 思考过程 + content 回答 + usage）
        messages = _build_messages(summary, history, context, user_content, low_confidence=low_confidence)
        full_answer = ""
        full_reasoning = ""
        usage: dict | None = None
        try:
            for ev in _stream_deepseek(messages):
                if ev["type"] == "usage":
                    # 透传真实 token 消耗（评测契约必选字段）
                    usage = ev["usage"]
                    yield ("usage", {**ev["usage"], "ts": _ts()})
                elif ev["type"] == "reasoning":
                    full_reasoning += ev["content"]
                    yield ("reasoning", {"content": ev["content"], "delta": ev["content"], "ts": _ts()})
                else:
                    full_answer += ev["content"]
                    yield ("token", {"content": ev["content"], "delta": ev["content"], "ts": _ts()})
        except Exception as e:
            logger.error("[chat] LLM 流式失败: %s", e)
            yield ("error", {"message": "LLM 调用失败，请稍后重试"})
            return

        # 5. 保存消息（含标题生成）+ 记忆压缩
        assistant_id = _save_messages(db, session, user_content, full_answer, sources)
        _compress_memory(db, session)

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
