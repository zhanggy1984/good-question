"""聊天业务逻辑：会话管理 + RAG 链 + 对话记忆压缩 + SSE 流式生成"""
import json
import logging
import time
from functools import lru_cache

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        parts.append(f"{src}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_chain():
    """LCEL RAG 链：prompt + 流式 LLM + 字符串输出"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    return prompt | get_llm(streaming=True) | StrOutputParser()


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

        # 1. 混合检索（dense + BM25 → RRF 融合 → rerank）
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
                    "chunk_content": d.page_content[:200],
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
