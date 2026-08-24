"""文档处理编排：抽取 → 清洗 → 切片 → 向量化(Milvus) → MySQL

文档处理是耗时操作，在后台线程执行，不阻塞 API。
DB 会话在线程内独立创建（SQLAlchemy session 非线程安全）。
"""
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sqlalchemy.orm import Session

from config import settings
from database import SessionLocal
from models import Chunk, Document
from utils import chunker, mineru_extractor, text_cleaner

logger = logging.getLogger("native_rag")

# 后台处理线程池：多 worker 避免单文档（尤其大 PDF）卡住阻塞所有后续上传。
# 向量化是 CPU 密集操作，3 个并发在"隔离阻塞"与"CPU 竞争"间取平衡。
_executor = ThreadPoolExecutor(max_workers=3)


def save_upload_file(upload_file, library_id: int) -> tuple[str, str, int]:
    """保存上传文件到磁盘，返回 (存储路径, 文件类型, 文件大小)

    路径: {UPLOAD_DIR}/{library_id}/{uuid}_{原始文件名}
    """
    filename = upload_file.filename or "unnamed"
    suffix = Path(filename).suffix.lower().lstrip(".")
    storage_dir = Path(settings.upload_dir) / str(library_id)
    storage_dir.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
    stored_path = storage_dir / stored_name

    # 分块写盘：避免 50MB 上限文件整体读入内存（并发上传时有 OOM 风险）
    size = 0
    with stored_path.open("wb") as out:
        while chunk := upload_file.file.read(1024 * 1024):
            out.write(chunk)
            size += len(chunk)

    return str(stored_path), suffix, size


def create_document(
    db: Session, library_id: int, filename: str, file_path: str, file_type: str,
    file_size: int, uploaded_by: int, chunk_size: int = 1024, overlap_token: int = 102,
) -> Document:
    """创建文档记录（status=processing，含切分配置）"""
    doc = Document(
        library_id=library_id,
        filename=filename,
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        status="processing",
        uploaded_by=uploaded_by,
        chunk_size=chunk_size,
        overlap_token=overlap_token,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    logger.debug(
        "[document.create] 创建文档 id=%s filename=%s chunk_size=%s overlap=%s",
        doc.id, filename, chunk_size, overlap_token,
    )
    return doc


def start_process(document_id: int) -> None:
    """提交后台处理任务"""
    _executor.submit(process_document, document_id)


def _document_exists(db: Session, document_id: int) -> bool:
    """文档是否仍存在（删除中断检查）

    文档被删除后处理线程应尽快释放：每阶段完成时检查一次，若已删除直接 return。
    相比全局任务注册表，逐阶段查 DB 开销低，且天然与删除事务一致。
    """
    return db.query(Document.id).filter(Document.id == document_id).scalar() is not None


def process_document(document_id: int) -> None:
    """后台执行完整处理管线（抽取 → 清洗 → 切片 → 向量化 → MySQL/Milvus）"""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            return
        logger.info("[document.process] 开始处理文档 id=%s %s", doc.id, doc.filename)

        # 阶段 1：文本抽取（MinerU，>200 页自动分流本地 CLI，失败降级轻量方案）
        t0 = time.time()
        raw_text = mineru_extractor.extract_text(doc.file_path, doc.file_type)
        logger.info("[document.process] 阶段1 抽取完成 耗时=%.1fs 字符数=%s", time.time() - t0, len(raw_text))
        if not _document_exists(db, document_id):
            logger.info("[document.process] 文档已删除，中止处理 id=%s", document_id)
            return

        # 阶段 2：文本预清洗
        t0 = time.time()
        cleaned = text_cleaner.clean_text(raw_text)
        if not cleaned:
            raise ValueError("文档内容为空，可能为扫描件或加密 PDF")
        logger.info("[document.process] 阶段2 清洗完成 耗时=%.1fs 字符数=%s", time.time() - t0, len(cleaned))
        if not _document_exists(db, document_id):
            logger.info("[document.process] 文档已删除，中止处理 id=%s", document_id)
            return

        # 阶段 3：智能切片（按文档配置 chunk_size/overlap_token，超 MAX_CHUNKS 截断）
        t0 = time.time()
        chunks = chunker.chunk_text(
            cleaned, doc.id, doc.library_id, doc.filename,
            chunk_size=doc.chunk_size, overlap_token=doc.overlap_token,
        )
        logger.info(
            "[document.process] 阶段3 切片完成 耗时=%.1fs chunk数=%s chunk_size=%s overlap=%s",
            time.time() - t0, len(chunks), doc.chunk_size, doc.overlap_token,
        )
        if not _document_exists(db, document_id):
            logger.info("[document.process] 文档已删除，中止处理 id=%s", document_id)
            return

        # 阶段 4：向量化并写入 Milvus（分批处理，on_progress 逐批把进度写库，前端轮询可见）
        t0 = time.time()
        from services import vector_store_service

        def _update_progress(written: int, total: int) -> None:
            # 逐批落库进度；文档已被删除则不再写，避免无谓 commit
            if _document_exists(db, document_id):
                doc.processed_chunks = written
                db.commit()

        processed = vector_store_service.add_chunks(doc.library_id, chunks, on_progress=_update_progress)
        if not _document_exists(db, document_id):
            logger.info("[document.process] 文档已删除，中止处理 id=%s", document_id)
            return
        logger.info(
            "[document.process] 阶段4 向量化完成 耗时=%.1fs 已处理=%s/%s",
            time.time() - t0, processed, len(chunks),
        )

        # 阶段 5：存 MySQL chunks 表（Milvus 的 dense 向量 + 稀疏向量已在阶段 4 写入）
        t0 = time.time()
        for c in chunks:
            db.add(Chunk(
                document_id=doc.id,
                library_id=doc.library_id,
                chunk_index=c["metadata"]["chunk_index"],
                content=c["content"],
                token_count=c["metadata"]["token_count"],
                metadata_json=c["metadata"],
            ))
        db.flush()
        logger.info("[document.process] 阶段5 MySQL 写入完成 耗时=%.1fs chunk数=%s", time.time() - t0, len(chunks))

        # 阶段 6：更新文档状态
        doc.status = "ready"
        doc.chunk_count = len(chunks)
        db.commit()
        logger.info("[document.process] 完成文档 id=%s 共 %s chunk", doc.id, len(chunks))

    except Exception as e:
        db.rollback()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            # 清理已写入的向量数据，保持存储一致（MySQL 为准）
            try:
                from services import vector_store_service
                vector_store_service.delete_by_document(document_id)
            except Exception:
                pass
            doc.status = "failed"
            doc.error_message = str(e)[:2000]
            db.commit()
        logger.error("[document.process] 处理失败 document_id=%s: %s", document_id, e, exc_info=True)
    finally:
        db.close()


def delete_document(db: Session, document_id: int) -> None:
    """删除文档：MySQL 级联 chunks + Milvus 向量 + 磁盘文件"""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        from utils.exceptions import NotFoundError
        raise NotFoundError("文档不存在")

    file_path = doc.file_path

    # 先清理 Milvus 向量，再删 MySQL
    from services import vector_store_service
    try:
        vector_store_service.delete_by_document(document_id)
    except Exception as e:
        logger.warning("[document.delete] Milvus 清理失败: %s", e)

    db.delete(doc)  # 外键 CASCADE 删除 chunks
    db.commit()

    # 删除磁盘文件
    try:
        Path(file_path).unlink(missing_ok=True)
    except OSError as e:
        logger.warning("[document.delete] 删除文件失败 %s: %s", file_path, e)

    logger.debug("[document.delete] 删除文档 id=%s", document_id)
