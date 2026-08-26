"""chat_sessions.updated_at 加索引（会话过期清理按最后活跃时间分批删除，防全表扫）

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-26
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """加 idx_updated_at：清理 DELETE WHERE updated_at < cutoff 走索引而非全表扫"""
    op.create_index("idx_updated_at", "chat_sessions", ["updated_at"])


def downgrade() -> None:
    """回滚：删除索引"""
    op.drop_index("idx_updated_at", table_name="chat_sessions")
