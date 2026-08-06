"""documents 表新增 chunk_size / overlap_token 列（按文档维度配置切分参数）

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """新增 chunk_size / overlap_token 列（已有数据默认 1024 / 102）"""
    op.add_column(
        "documents",
        sa.Column("chunk_size", sa.Integer(), nullable=False, server_default="1024"),
    )
    op.add_column(
        "documents",
        sa.Column("overlap_token", sa.Integer(), nullable=False, server_default="102"),
    )


def downgrade() -> None:
    """回滚：删除两列"""
    op.drop_column("documents", "overlap_token")
    op.drop_column("documents", "chunk_size")
