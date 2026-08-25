"""文档库业务逻辑"""
import logging

from sqlalchemy import func
from sqlalchemy.orm import Session

from models import DocumentLibrary
from schemas.common import Page
from utils.exceptions import NotFoundError

logger = logging.getLogger("native_rag")


def list_libraries(db: Session, page: int, page_size: int) -> Page:
    """分页查询文档库列表（按创建时间倒序）"""
    total = db.query(func.count(DocumentLibrary.id)).scalar() or 0
    items = (
        db.query(DocumentLibrary)
        .order_by(DocumentLibrary.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return Page(items=items, total=total, page=page, page_size=page_size)


def get_library(db: Session, library_id: int) -> DocumentLibrary:
    """查询文档库详情"""
    lib = db.query(DocumentLibrary).filter(DocumentLibrary.id == library_id).first()
    if lib is None:
        raise NotFoundError("文档库不存在")
    return lib


def create_library(
    db: Session, name: str, description: str | None, created_by: int
) -> DocumentLibrary:
    """创建文档库"""
    lib = DocumentLibrary(name=name, description=description, created_by=created_by)
    db.add(lib)
    db.commit()
    db.refresh(lib)
    logger.debug("[library.create] 创建文档库 id=%s name=%s", lib.id, name)
    return lib


def delete_library(db: Session, library_id: int) -> None:
    """删除文档库

    MySQL 外键 ON DELETE CASCADE 自动级联删除 documents 与 chunks；
    Milvus 的库数据（library_id 过滤）单独清理。
    """
    lib = get_library(db, library_id)

    # 先清理 Milvus 库数据，再删 MySQL
    import logging
    from services import vector_store_service
    logger = logging.getLogger("native_rag")
    try:
        vector_store_service.delete_library_collection(library_id)
    except Exception as e:
        logger.warning("[library.delete] Milvus 清理失败: %s", e)

    db.delete(lib)
    db.commit()

    # 删库后清该库问答缓存：旧答案（含已删库全量）在 TTL 窗口内重放会误导，必须立即失效
    from services.chat_cache import flush_library
    flush_library(library_id)

    logger.debug("[library.delete] 删除文档库 id=%s name=%s", library_id, lib.name)
