"""文档库 API"""
import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import get_db
from middleware.auth import get_admin_user, get_current_user
from models import User
from schemas.common import Page
from schemas.library import LibraryCreate, LibraryResponse
from services import library_service

logger = logging.getLogger("native_rag")
router = APIRouter()


@router.get("", response_model=Page[LibraryResponse])
def list_libraries(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """文档库列表（所有登录用户）"""
    result = library_service.list_libraries(db, page, page_size)
    logger.debug("[library.list] 出参 page=%s page_size=%s total=%s", page, page_size, result.total)
    return result


@router.get("/{library_id}", response_model=LibraryResponse)
def get_library(
    library_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """文档库详情（所有登录用户）"""
    return library_service.get_library(db, library_id)


@router.post("", response_model=LibraryResponse, status_code=201)
def create_library(
    body: LibraryCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """新增文档库（仅 admin）"""
    logger.debug("[library.create] 入参 name=%s", body.name)
    lib = library_service.create_library(db, body.name, body.description, admin.id)
    logger.debug("[library.create] 出参 id=%s", lib.id)
    return lib


@router.delete("/{library_id}", status_code=204)
def delete_library(
    library_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    """删除文档库（仅 admin）"""
    logger.debug("[library.delete] 入参 library_id=%s", library_id)
    library_service.delete_library(db, library_id)
