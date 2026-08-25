"""聊天 API（SSE 流式）"""
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from middleware.auth import get_current_user
from models import User
from services import chat_service

logger = logging.getLogger("native_rag")
router = APIRouter()


class ChatRequest(BaseModel):
    """聊天请求"""
    content: str = Field(min_length=1, max_length=4000, description="用户问题")


async def _stream_with_disconnect_check(gen, is_disconnected):
    """迭代 stream_chat 生成器，客户端断开时停止生成

    yield (event_type, data)；is_disconnected 为可 await 的断连判断（request.is_disconnected）。
    断连后 return + gen.close()：触发 stream_chat 的 finally（db.close / httpx 流关闭），
    终止对 DeepSeek 的调用，不再烧 token 与连接资源。正常迭代结束 close 是 no-op。
    """
    try:
        for event, data in gen:
            if await is_disconnected():
                return
            yield event, data
    finally:
        gen.close()


@router.post("/chat/{session_id}")
def chat(
    session_id: int,
    body: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """发送消息，SSE 流式返回（sources → token* → done）"""
    # 用请求 db 校验会话归属（会话隔离）
    chat_service._get_owned_session(db, session_id, current_user.id)
    logger.debug("[chat] 入参 session_id=%s content=%.50s", session_id, body.content)

    async def event_stream():
        # stream_chat 内部使用独立 db session；客户端断开即停止生成（断连检测）
        gen = chat_service.stream_chat(session_id, body.content)
        async for event, data in _stream_with_disconnect_check(gen, request.is_disconnected):
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁止 nginx 缓冲
        },
    )
