"""documents 表新增 processed_chunks 列，用于大文档处理进度上报

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """新增 processed_chunks 列（已有数据默认 0）"""
    op.add_column(
        "documents",
        sa.Column(
            "processed_chunks",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    """回滚：删除该列"""
    op.drop_column("documents", "processed_chunks")
