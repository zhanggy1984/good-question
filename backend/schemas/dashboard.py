"""仪表盘相关响应模型"""
from pydantic import BaseModel


class DashboardResponse(BaseModel):
    """仪表盘统计响应"""
    library_count: int
    document_count: int
    chunk_count: int
