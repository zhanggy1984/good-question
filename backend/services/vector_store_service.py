"""向量库服务：LangChain Chroma + ChromaDB Server

每个文档库一个 collection（library_{id}），chunk id 用 {document_id}_{chunk_index}。
"""
import logging
import time
from collections.abc import Callable
from functools import lru_cache

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from config import settings
from services.embedding_service import get_embeddings

logger = logging.getLogger("native_rag")

# 向量化分批大小：避免大文档一次性全量 embed 导致 CPU/内存峰值
BATCH_SIZE = 64


def _get_chroma_settings() -> ChromaSettings:
    """连接 ChromaDB Server 的客户端配置"""
    return ChromaSettings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        chroma_server_host=settings.chroma_host,
        chroma_server_http_port=settings.chroma_port,
        anonymized_telemetry=False,
    )


@lru_cache(maxsize=64)
def _get_collection(library_id: int) -> Chroma:
    """获取/创建某库的向量 collection"""
    collection_name = f"library_{library_id}"
    collection = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        client_settings=_get_chroma_settings(),
    )
    return collection


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB 不支持空 list 作为 metadata 值（如无标题文档的 heading_path=[]），清洗为空字符串列表"""
    return {k: ([""] if v == [] else v) for k, v in meta.items()}


def add_chunks(
    library_id: int,
    chunks: list[dict],
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """批量写入 chunk 到 ChromaDB（自动向量化），分批处理

    分批目的：大文档数百/上千 chunk 一次性 embed 会打满 CPU、撑高内存，
    分批可平滑资源占用，并逐批打进度日志供排查。

    on_progress(written, total)：每批写入完成后回调一次，供上层把进度写库
    （前端轮询展示"处理中 N 段"）；None 则只打日志。
    返回实际写入的 chunk 数。
    """
    if not chunks:
        return 0
    collection = _get_collection(library_id)
    total = len(chunks)
    written = 0
    start = time.time()
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        metadatas = [_sanitize_metadata(c["metadata"]) for c in batch]
        ids = [f"{c['metadata']['document_id']}_{c['metadata']['chunk_index']}" for c in batch]
        collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        written += len(batch)
        if on_progress:
            on_progress(written, total)
        logger.info(
            "[vector_store] 向量化进度 %s/%s（批 %s/%s）",
            written, total, i // BATCH_SIZE + 1, (total + BATCH_SIZE - 1) // BATCH_SIZE,
        )
    logger.info(
        "[vector_store] 写入完成 %s chunk 到 collection=%s 耗时=%.1fs",
        total, collection._collection_name, time.time() - start,
    )
    return written


def delete_by_document(library_id: int, document_id: int) -> None:
    """删除某文档的全部向量（按 metadata.document_id 过滤）"""
    collection = _get_collection(library_id)
    collection.delete(where={"document_id": document_id})


def delete_library_collection(library_id: int) -> None:
    """删除整个文档库的 collection"""
    from langchain_chroma import Chroma

    collection_name = f"library_{library_id}"
    # 通过 client 删除 collection
    chroma_client = _get_collection(library_id)._client
    try:
        chroma_client.delete_collection(collection_name)
        logger.debug("[vector_store] 已删除 collection=%s", collection_name)
    except Exception as e:
        logger.warning("[vector_store] 删除 collection 失败: %s", e)
    _get_collection.cache_clear()


def similarity_search(library_id: int, query: str, k: int = 3) -> list[tuple]:
    """语义检索，返回 [(Document, score)]（Task 6 混合检索用）"""
    collection = _get_collection(library_id)
    return collection.similarity_search_with_score(query, k=k)
