"""SQLAlchemy 数据库连接与会话管理"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

# 延迟建 engine/session：宿主机离线单测（无 MYSQL_PASSWORD）import 本模块不抛错；
# 首次实际建连时才触发 config.database_url 的密码校验（fail-fast，防缺配静默连错）。
# create_engine 本身惰性，真正的 TCP 连接发生在首次 SessionLocal()，校验前置到 URL 构造即可拦截缺配。
_engine = None
_session_factory = None


def get_engine():
    """返回全局 SQLAlchemy engine（首次调用创建；密码空时 database_url 抛 RuntimeError）"""
    global _engine
    if _engine is None:
        # 连接池配置：pool_pre_ping 检测断连，pool_recycle 定期回收避免 MySQL wait_timeout 断开
        _engine = create_engine(
            settings.database_url,   # 空密码在此抛错（见 config.database_url 校验）
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
    return _engine


def _make_session():
    """首次调用创建 sessionmaker，返回一个新 session"""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _session_factory()


# 兼容旧调用方式：SessionLocal() 语义不变（首次调用触发 engine 创建）
SessionLocal = _make_session


class Base(DeclarativeBase):
    """所有 ORM 模型的基类"""


def get_db():
    """FastAPI 依赖：请求级数据库会话，请求结束自动关闭"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
