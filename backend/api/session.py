"""聊天会话 API"""
import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import get_db
from middleware.auth import get_current_user
from models import User
from schemas.common import Page
from schemas.session import (
    ChatMessageResponse,
    SessionCreate,
    SessionDetailResponse,
    SessionResponse,
)
from services import chat_service

logger = logging.getLogger("native_rag")
router = APIRouter()


@router.get("", response_model=Page[SessionResponse])
def list_sessions(
    library_id: int | None = Query(None, description="按文档库过滤"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """当前用户的会话列表"""
    result = chat_service.list_sessions(db, current_user.id, library_id, page, page_size)
    logger.debug("[session.list] 出参 total=%s", result.total)
    return result


@router.post("", response_model=SessionResponse, status_code=201)
def create_session(
    body: SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建会话（绑定文档库）"""
    logger.debug("[session.create] 入参 library_id=%s", body.library_id)
    return chat_service.create_session(db, current_user.id, body.library_id)


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """会话详情 + 历史消息"""
    session, messages = chat_service.get_session_detail(db, session_id, current_user.id)
    result = SessionDetailResponse.model_validate(session)
    result.messages = [ChatMessageResponse.model_validate(m) for m in messages]
    return result


@router.delete("/{session_id}", status_code=204)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除会话（仅所有者）"""
    logger.debug("[session.delete] 入参 session_id=%s", session_id)
    chat_service.delete_session(db, session_id, current_user.id)
