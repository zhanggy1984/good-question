"""一次性迁移脚本：从 MySQL chunks 表重灌 Milvus（替代 ChromaDB + ES）

ChromaDB/ES 是派生索引，MySQL chunks 表存了全文 + metadata_json 完整源数据，
故迁移只需从 MySQL 读出所有 chunk，重新向量化写入 Milvus，旧存储直接废弃。

用法：
    docker exec -it rag-backend bash
    cd /app && python scripts/migrate_to_milvus.py

幂等：add_chunks 用 upsert（主键 {document_id}_{chunk_index}），可安全重跑。
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import SessionLocal  # noqa: E402
from models import Chunk  # noqa: E402
from services import vector_store_service  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("native_rag")

# MySQL 读取分批大小：避免一次性加载全量 chunk 到内存
READ_BATCH = 500


def migrate() -> int:
    """逐库迁移：读取 MySQL chunks → 分批向量化写入 Milvus 对应 partition，返回总 chunk 数"""
    db = SessionLocal()
    total = 0
    try:
        library_ids = [r[0] for r in db.query(Chunk.library_id).distinct().all()]
        logger.info("发现 %s 个文档库待迁移：%s", len(library_ids), library_ids)
        for library_id in library_ids:
            t0 = time.time()
            offset = 0
            while True:
                rows = (
                    db.query(Chunk)
                    .filter(Chunk.library_id == library_id)
                    .order_by(Chunk.id)
                    .offset(offset)
                    .limit(READ_BATCH)
                    .all()
                )
                if not rows:
                    break
                chunks = []
                for c in rows:
                    # metadata_json 为完整溯源；缺关键字段时从表字段兜底
                    meta = dict(c.metadata_json or {})
                    meta.setdefault("document_id", c.document_id)
                    meta.setdefault("chunk_index", c.chunk_index)
                    meta.setdefault("library_id", c.library_id)
                    chunks.append({"content": c.content, "metadata": meta})
                vector_store_service.add_chunks(library_id, chunks)
                total += len(rows)
                offset += len(rows)
            logger.info(
                "迁移完成 library=%s chunk数=%s 耗时=%.1fs",
                library_id, offset, time.time() - t0,
            )
        logger.info("迁移全部完成，共 %s chunk", total)
        return total
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
