"""仪表盘业务逻辑"""
from sqlalchemy import func
from sqlalchemy.orm import Session

from models import Chunk, Document, DocumentLibrary


def get_stats(db: Session) -> dict:
    """统计文档库/文档/chunk 数量"""
    return {
        "library_count": db.query(func.count(DocumentLibrary.id)).scalar() or 0,
        "document_count": db.query(func.count(Document.id)).scalar() or 0,
        "chunk_count": db.query(func.count(Chunk.id)).scalar() or 0,
    }
