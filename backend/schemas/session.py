"""聊天会话相关请求/响应模型"""
from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """创建会话请求"""
    library_id: int = Field(gt=0, description="绑定的文档库")


class SessionResponse(BaseModel):
    """会话信息响应"""
    id: int
    library_id: int
    title: str | None
    summary: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatMessageResponse(BaseModel):
    """会话消息响应"""
    id: int
    session_id: int
    role: str
    content: str
    sources_json: dict | list | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionDetailResponse(SessionResponse):
    """会话详情（含历史消息）"""
    messages: list[ChatMessageResponse]
