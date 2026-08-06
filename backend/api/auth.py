"""认证 API"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from middleware.auth import get_current_user
from models import User
from schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from services import auth_service

logger = logging.getLogger("native_rag")
router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """注册新用户"""
    return auth_service.register(db, body.username, body.password)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """登录，返回 JWT access token"""
    token = auth_service.authenticate(db, body.username, body.password)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return current_user
