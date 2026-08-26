"""会话过期清理：定时 sweep + 惰性清理（过期判定完全落在 DB 侧，避免时区错位）

清理规则：最后活跃时间（chat_sessions.updated_at，随存消息/压缩自动刷新）
超过保留期（chat_retention_days）即过期。物理硬删除会话行，
chat_messages 由外键 ON DELETE CASCADE 级联删，无需显式删消息。

判定统一用 MySQL 侧 NOW() - INTERVAL :days DAY，Python 不生成 cutoff 时间：
容器 Python 时区与 MySQL 时区若不一致（生产/换时区即触发），datetime.now() 与
DB 存储的 updated_at 比较会系统性错位。参考 customer-service 的 SessionCleaner
做法（判据完全落在 MySQL 侧，避免 aware/naive 混比及时区错位）。

双机制：
- 定时 sweep（SessionCleaner）：后台 asyncio 循环分批删除，随应用生命周期启停
- 惰性清理：chat_service 查询入口（详情/聊天/列表）发现过期会话即时回收
"""
import asyncio
import logging
import time

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from config import settings

logger = logging.getLogger("native_rag")

# 批间让步时长（秒）：长事务连删会挤压正常写路径，主动 sleep 让出事务窗口
_BATCH_YIELD_SECONDS = 0.05

# 清理索引名（alembic 0004 迁移创建；缺失时分批删除仍正确，仅可能全表扫）
_CLEANUP_INDEX = "idx_updated_at"


def expired_session_condition():
    """过期的 SQL 判定表达式：updated_at >= now() - INTERVAL n DAY（DB 侧 NOW()，时区无关）"""
    # days 参数化（与 is_session_expired / sweep 写法一致），不拼接 SQL
    return func.now() - text("INTERVAL :days DAY").bindparams(days=settings.chat_retention_days)


def is_session_expired(db: Session, session_id: int) -> bool:
    """会话是否过期：判定完全落在 MySQL 侧（NOW() 与 updated_at 同库时区），
    惰性清理共用；严格小于 cutoff（边界不删）"""
    row = db.execute(
        text(
            "SELECT updated_at < NOW() - INTERVAL :days DAY AS expired "
            "FROM chat_sessions WHERE id = :id"
        ),
        {"days": settings.chat_retention_days, "id": session_id},
    ).first()
    return bool(row and row[0])


def delete_session_by_id(db: Session, session_id: int) -> None:
    """物理删除单个会话（messages 由外键 CASCADE 级联），幂等（不存在则删 0 行）"""
    db.execute(text("DELETE FROM chat_sessions WHERE id = :id"), {"id": session_id})
    db.commit()


class SessionCleaner:
    """定时 sweep：后台循环分批删除过期会话，随应用生命周期启停"""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """启动后台清理循环（幂等）；启动时检查清理索引，缺失仅告警不阻塞"""
        if self._task is None:
            self._warn_if_index_missing()
            self._task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """停止后台清理循环（应用关闭时调用）"""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _cleanup_loop(self) -> None:
        """周期执行 sweep，异常吞掉保证循环不退出（下次周期重试）"""
        while True:
            await asyncio.sleep(settings.chat_cleanup_interval_seconds)
            try:
                # sweep 是同步分批删除（含批间 sleep），丢线程池执行，避免阻塞事件循环（SSE 流式）
                total = await asyncio.to_thread(self.sweep)
                if total:
                    logger.info("event=chat_cleanup_summary total=%s", total)
            except Exception as exc:
                logger.error("event=chat_cleanup_error error=%s", exc)

    def sweep(self, db: Session | None = None) -> int:
        """分批删除过期会话，返回删除总数；db 缺省时自建会话（便于测试注入）"""
        own_session = db is None
        if db is None:
            from database import SessionLocal  # 延迟导入避免模块级循环依赖
            db = SessionLocal()
        try:
            total = 0
            while True:
                n = db.execute(
                    text(
                        "DELETE FROM chat_sessions "
                        "WHERE updated_at < NOW() - INTERVAL :days DAY "
                        "ORDER BY id LIMIT :batch"
                    ),
                    {
                        "days": settings.chat_retention_days,
                        "batch": settings.chat_cleanup_batch_size,
                    },
                ).rowcount
                db.commit()
                total += n
                if n < settings.chat_cleanup_batch_size:
                    break
                time.sleep(_BATCH_YIELD_SECONDS)
            return total
        finally:
            if own_session:
                db.close()

    def _warn_if_index_missing(self) -> None:
        """清理索引缺失检查（只读 information_schema，缺失仅告警不阻塞）：
        迁移 0004 未跑时分批删除仍正确（功能无碍），但可能全表扫，告警提示运维"""
        try:
            from database import SessionLocal  # 延迟导入避免模块级循环依赖
            db = SessionLocal()
            try:
                row = db.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.statistics "
                        "WHERE table_schema = DATABASE() AND table_name = 'chat_sessions' "
                        "AND index_name = :index"
                    ),
                    {"index": _CLEANUP_INDEX},
                ).scalar()
                if not row:
                    logger.warning(
                        "event=chat_cleanup_index_missing 清理索引 %s 缺失，"
                        "请执行 alembic upgrade head（缺失时分批删除仍正确，仅可能全表扫）",
                        _CLEANUP_INDEX,
                    )
            finally:
                db.close()
        except Exception as exc:
            logger.warning("event=chat_cleanup_index_check_error error=%s", exc)
