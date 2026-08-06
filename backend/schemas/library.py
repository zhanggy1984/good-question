"""文档库相关请求/响应模型"""
from datetime import datetime

from pydantic import BaseModel, Field


class LibraryCreate(BaseModel):
    """新增文档库请求"""
    name: str = Field(min_length=1, max_length=200, description="文档库名称")
    description: str | None = Field(default=None, max_length=2000, description="描述")


class LibraryResponse(BaseModel):
    """文档库信息响应"""
    id: int
    name: str
    description: str | None
    created_by: int | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
