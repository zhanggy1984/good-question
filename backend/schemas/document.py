"""文档相关请求/响应模型"""
from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """文档信息响应"""
    id: int
    library_id: int
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    processed_chunks: int
    chunk_size: int
    overlap_token: int
    status: str
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentStatusResponse(BaseModel):
    """文档处理状态响应（含进度）"""
    status: str
    chunk_count: int
    processed_chunks: int
    error_message: str | None


class ChunkResponse(BaseModel):
    """文档 chunk 响应"""
    id: int
    chunk_index: int
    content: str
    token_count: int
    metadata_json: dict

    model_config = {"from_attributes": True}
