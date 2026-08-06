"""认证业务逻辑"""
import logging

from sqlalchemy.orm import Session

from config import settings
from models import User
from utils.exceptions import ConflictError, InvalidCredentialsError
from utils.security import create_access_token, hash_password, verify_password

logger = logging.getLogger("native_rag")


def register(db: Session, username: str, password: str) -> User:
    """注册新用户（默认角色 user）"""
    logger.debug("[auth.register] 入参 username=%s", username)
    if db.query(User).filter(User.username == username).first():
        raise ConflictError("用户名已存在")
    user = User(username=username, password_hash=hash_password(password), role="user")
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.debug("[auth.register] 出参 user_id=%s", user.id)
    return user


def authenticate(db: Session, username: str, password: str) -> str:
    """验证用户名密码，成功返回 JWT token"""
    logger.debug("[auth.login] 入参 username=%s", username)
    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, user.password_hash):
        raise InvalidCredentialsError("用户名或密码错误")
    token = create_access_token(subject=str(user.id), role=user.role)
    logger.debug("[auth.login] 出参 用户=%s 签发成功", username)
    return token


def seed_admin(db: Session) -> None:
    """启动时创建预设 admin 账号（仅当不存在时）"""
    if db.query(User).filter(User.username == settings.admin_username).first():
        return
    admin = User(
        username=settings.admin_username,
        password_hash=hash_password(settings.admin_password),
        role="admin",
    )
    db.add(admin)
    db.commit()
    logger.info("[seed] 已创建预设 admin 账号: %s", settings.admin_username)
