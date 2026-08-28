"""认证业务逻辑"""
import logging

from sqlalchemy.orm import Session

from config import settings
from models import User
from utils import rate_limit
from utils.exceptions import ConflictError, InvalidCredentialsError, TooManyRequestsError
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
    # 登录防爆破：连续失败超限临时锁定（Redis 不可用时 fail-open 放行，见 rate_limit）
    if not rate_limit.check_allowed(username):
        raise TooManyRequestsError("登录尝试过于频繁，请稍后再试")
    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, user.password_hash):
        rate_limit.record_failure(username)
        raise InvalidCredentialsError("用户名或密码错误")
    rate_limit.reset(username)
    token = create_access_token(subject=str(user.id), role=user.role)
    logger.debug("[auth.login] 出参 用户=%s 签发成功", username)
    return token


def seed_admin(db: Session) -> None:
    """启动时创建预设 admin 账号；已存在时按 .env 配置轮换密码哈希。

    轮换语义：fail-fast（_validate_secrets）已保证 .env 是强随机值；若 DB 哈希与 .env 密码
    不一致（旧版 admin123 等弱密码已在存量库落库），启动即更新哈希——升级部署后旧弱密码不再可登录。
    """
    admin = db.query(User).filter(User.username == settings.admin_username).first()
    if admin is None:
        admin = User(
            username=settings.admin_username,
            password_hash=hash_password(settings.admin_password),
            role="admin",
        )
        db.add(admin)
        db.commit()
        logger.info("[seed] 已创建预设 admin 账号: %s", settings.admin_username)
        return
    if not verify_password(settings.admin_password, admin.password_hash):
        admin.password_hash = hash_password(settings.admin_password)
        db.commit()
        logger.warning("[seed] admin 密码已按 .env 更新（存量弱密码轮换）")
