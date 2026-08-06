"""JWT 认证与角色校验依赖"""
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from database import get_db
from models import User
from utils.exceptions import ForbiddenError, UnauthorizedError
from utils.security import decode_access_token

# auto_error=False：header 缺失或格式错误时返回 None，由业务层统一处理
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """从 Authorization header 解析 JWT，返回当前用户"""
    if credentials is None:
        raise UnauthorizedError("未登录")
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise UnauthorizedError("登录已过期，请重新登录")
    user = db.query(User).filter(User.id == int(payload.get("sub", 0))).first()
    if user is None:
        raise UnauthorizedError("用户不存在")
    return user


def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """校验当前用户是否为 admin，否则 403"""
    if current_user.role != "admin":
        raise ForbiddenError("仅管理员可执行此操作")
    return current_user
