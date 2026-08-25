"""文档 API"""
import logging

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from middleware.auth import get_admin_user, get_current_user
from models import Chunk, Document, DocumentLibrary, User
from schemas.common import Page
from schemas.document import ChunkResponse, DocumentResponse, DocumentStatusResponse
from services import document_service, library_service
from utils.exceptions import NotFoundError, ValidationError

logger = logging.getLogger("native_rag")
router = APIRouter()

ALLOWED_TYPES = {"pdf", "docx", "txt", "md"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.get("/libraries/{library_id}/documents", response_model=Page[DocumentResponse])
def list_documents(
    library_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """文档列表（登录用户可见，admin 可管理）"""
    library_service.get_library(db, library_id)  # 校验库存在
    total = db.query(func.count(Document.id)).filter(Document.library_id == library_id).scalar() or 0
    items = (
        db.query(Document)
        .filter(Document.library_id == library_id)
        .order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return Page(items=items, total=total, page=page, page_size=page_size)


@router.post("/libraries/{library_id}/documents", response_model=DocumentResponse, status_code=201)
def upload_document(
    library_id: int,
    file: UploadFile = File(...),
    chunk_size: int = Form(1024),
    overlap_token: int = Form(102),
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """上传文档并触发后台处理（仅 admin），可按文档配置切分参数"""
    logger.debug(
        "[document.upload] 入参 library_id=%s filename=%s chunk_size=%s overlap_token=%s",
        library_id, file.filename, chunk_size, overlap_token,
    )

    library_service.get_library(db, library_id)  # 校验库存在

    file_type = (file.filename or "").rsplit(".", 1)[-1].lower()
    if file_type not in ALLOWED_TYPES:
        raise ValidationError(f"不支持的文件类型 .{file_type}，仅支持 pdf/docx/txt/md")

    # 校验切分参数（纯参数校验，先于落盘，失败不产生垃圾文件）
    if not (128 <= chunk_size <= 8192):
        raise ValidationError(f"chunk_size 需在 128~8192 之间，当前 {chunk_size}")
    if not (0 <= overlap_token < chunk_size):
        raise ValidationError(f"overlap_token 需在 0~{chunk_size - 1} 之间，当前 {overlap_token}")

    # 保存文件并创建记录
    stored_path, _, file_size = document_service.save_upload_file(file, library_id)
    if file_size > MAX_FILE_SIZE:
        import os
        os.remove(stored_path)
        raise ValidationError("文件超过 50MB 限制")

    doc = document_service.create_document(
        db, library_id, file.filename, stored_path, file_type, file_size, admin.id,
        chunk_size=chunk_size, overlap_token=overlap_token,
    )

    # 提交后台处理
    document_service.start_process(doc.id)

    logger.debug("[document.upload] 出参 document_id=%s status=processing", doc.id)
    return doc


@router.get("/documents/{document_id}/chunks", response_model=Page[ChunkResponse])
def list_document_chunks(
    document_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """文档的 chunk 分页列表（登录用户可看）"""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise NotFoundError("文档不存在")
    total = db.query(func.count(Chunk.id)).filter(Chunk.document_id == document_id).scalar() or 0
    items = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return Page(items=items, total=total, page=page, page_size=page_size)


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def document_status(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查询文档处理状态（前端轮询）"""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise NotFoundError("文档不存在")
    return DocumentStatusResponse(
        status=doc.status,
        chunk_count=doc.chunk_count,
        processed_chunks=doc.processed_chunks,
        error_message=doc.error_message,
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """删除文档（仅 admin）"""
    logger.debug("[document.delete] 入参 document_id=%s", document_id)
    document_service.delete_document(db, document_id)


@router.post("/documents/{document_id}/reprocess", response_model=DocumentResponse)
def reprocess_document(
    document_id: int,
    chunk_size: int = Form(None),
    overlap_token: int = Form(None),
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """重新处理文档（failed/ready 均可）：清旧向量与 chunks 后重跑管线，可选覆盖切分参数（仅 admin）"""
    logger.debug(
        "[document.reprocess] 入参 document_id=%s chunk_size=%s overlap_token=%s",
        document_id, chunk_size, overlap_token,
    )
    doc = document_service.reprocess_document(db, document_id, chunk_size, overlap_token)
    logger.debug("[document.reprocess] 出参 status=%s", doc.status)
    return doc
