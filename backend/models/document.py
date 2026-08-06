"""文档模型"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("idx_library", "library_id"),
        Index("idx_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    library_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("document_libraries.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 已向量化/写入的 chunk 数（大文档处理期间实时更新，前端用于进度展示）
    processed_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 切分配置：按文档维度设置 chunk 大小与重叠 token 数（上传时传入，默认 1024/102）
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1024)
    overlap_token: Mapped[int] = mapped_column(Integer, nullable=False, default=102)
    status: Mapped[str] = mapped_column(
        Enum("processing", "ready", "failed", name="document_status_enum"),
        nullable=False,
        default="processing",
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    uploaded_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    library: Mapped["DocumentLibrary"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
