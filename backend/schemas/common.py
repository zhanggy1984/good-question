"""通用分页模型"""
from typing import Generic, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """统一分页响应结构"""
    items: List[T]
    total: int
    page: int
    page_size: int
