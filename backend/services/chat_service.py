"""聊天业务逻辑：会话管理 + RAG 链 + 对话记忆压缩 + SSE 流式生成"""
import json
import logging
import time

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

对话历史摘要（早期对话已压缩，供参考）：
{summary}

参考资料（带来源编号）：
{context}"""


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


def _build_messages(summary: str, history: list, context: str, question: str) -> list[dict]:
    """构建 DeepSeek API 的 messages（system + history + human）"""
    system = SYSTEM_PROMPT.format(summary=summary, context=context)
    messages = [{"role": "system", "content": system}]
    for m in history:
        role = "user" if m.type == "human" else "assistant"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": question})
    return messages


def _stream_deepseek(messages: list[dict]):
    """直连 DeepSeek 流式调用，解析 reasoning_content 与 content

    yield {"type": "reasoning"|"content", "content": str}
    """
    payload = {
        "model": settings.deepseek_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
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
                delta = json.loads(raw)["choices"][0]["delta"]
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
    事件顺序：sources → token* → done（LLM 失败时 yield error）
    """
    db = SessionLocal()
    try:
        t_start = time.time()
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session is None:
            yield ("error", {"message": "会话不存在"})
            return

        # 1. 混合检索（语义 + ES → rerank，低相似度已被过滤）
        retriever = HybridRetriever(library_id=session.library_id)
        docs = retriever.invoke(user_content)

        # 2. 仅在检索到结果时推送 sources（无结果时不展示空引用来源）
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
            yield ("sources", {"sources": sources})
        logger.info("[chat] 检索完成 耗时=%.2fs 结果=%s，开始 LLM 流式", time.time() - t_start, len(docs))

        # 3. 构建上下文
        summary, history = _build_context(db, session)
        context = _format_docs(docs)

        # 4. 流式调用 DeepSeek（解析 reasoning_content 思考过程 + content 回答）
        messages = _build_messages(summary, history, context, user_content)
        full_answer = ""
        full_reasoning = ""
        try:
            for ev in _stream_deepseek(messages):
                if ev["type"] == "reasoning":
                    full_reasoning += ev["content"]
                    yield ("reasoning", {"content": ev["content"]})
                else:
                    full_answer += ev["content"]
                    yield ("token", {"content": ev["content"]})
        except Exception as e:
            logger.error("[chat] LLM 流式失败: %s", e)
            yield ("error", {"message": "LLM 调用失败，请稍后重试"})
            return

        # 5. 保存消息（含标题生成）+ 记忆压缩
        assistant_id = _save_messages(db, session, user_content, full_answer, sources)
        _compress_memory(db, session)

        # 6. done
        yield ("done", {"message_id": assistant_id})
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
