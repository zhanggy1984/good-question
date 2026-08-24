"""Milvus 向量库服务门面（RAG 检索迁移一期：内部换 LlamaIndex，对外接口不变）

对外接口 add_chunks / hybrid_search / delete_by_document / delete_library_collection
的签名与返回形状与迁移前一致，调用方（document_service / library_service /
retrieval_service / migrate 脚本）零改动。

内部实现迁移到 LlamaIndex（见 llama_store.py）：
- schema / 索引 / 稀疏路（BGE-M3 学习稀疏，替代服务端 BM25）全由 MilvusVectorStore 管理
- 库隔离：partition p_{library_id} → library_id 字段 MetadataFilters（expr kwarg 在
  0.12.52 被忽略，只认 query.filters）
- 幂等：LlamaIndex add 是 insert 非 upsert，add 前先按 document_id 删旧数据
- 可见性：insert 用 force_flush=True，delete 后显式 flush（Milvus 延迟可见）
"""
import logging
import time

from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)

from services import llama_store
from services.embedding_service import embed_texts

logger = logging.getLogger("native_rag")

BATCH_SIZE = 64


def add_chunks(
    library_id: int,
    chunks: list[dict],
    on_progress=None,
) -> int:
    """批量写入 chunk 到 Milvus（自动向量化），分批处理

    分批目的：大文档数百/上千 chunk 一次性 embed 会打满 CPU、撑高内存，
    分批可平滑资源占用。幂等：LlamaIndex add 是 insert 非 upsert，add 前先按
    涉及的 document_id 删除旧数据（delete 后 flush），保证可安全重跑。

    on_progress(written, total)：每批写入完成后回调一次，供上层把进度写库
    （前端轮询展示"处理中 N 段"）；None 则只打日志。
    返回实际写入的 chunk 数。
    """
    if not chunks:
        return 0
    store = llama_store._get_store()  # 首次调用触发 collection 自愈（旧 schema 重建）
    total = len(chunks)
    written = 0
    start = time.time()

    # 幂等：Milvus insert 不允许重复主键，重跑前删同文档旧数据（delete 后必须 flush）
    doc_ids = {str(c["metadata"]["document_id"]) for c in chunks}
    for doc_id in doc_ids:
        store.delete(ref_doc_id=doc_id)
    if doc_ids:
        llama_store.flush()

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        embeddings = embed_texts(texts)
        nodes = [
            llama_store.chunk_to_node(c, library_id, emb)
            for c, emb in zip(batch, embeddings)
        ]
        store.add(nodes, force_flush=True)
        written += len(batch)
        if on_progress:
            on_progress(written, total)
        logger.info(
            "[vector_store] 写入进度 %s/%s（批 %s/%s）",
            written, total, i // BATCH_SIZE + 1, (total + BATCH_SIZE - 1) // BATCH_SIZE,
        )
    logger.info(
        "[vector_store] 写入完成 %s chunk 到 collection=%s 耗时=%.1fs",
        total, llama_store.COLLECTION_NAME, time.time() - start,
    )
    return written


def delete_by_document(document_id: int) -> None:
    """删除某文档的全部向量（按顶层 document_id 字段，跨全 collection，与库无关）"""
    store = llama_store._get_store()
    store.delete(ref_doc_id=str(document_id))
    llama_store.flush()  # Milvus delete 延迟可见，flush 后才立即生效


def delete_library_collection(library_id: int) -> None:
    """删除整个文档库的全部 chunk（按 library_id 字段过滤）"""
    store = llama_store._get_store()
    store.client.delete(
        collection_name=llama_store.COLLECTION_NAME,
        filter=f"library_id == {library_id}",
    )
    llama_store.flush()


def hybrid_search(
    library_id: int,
    query: str,
    dense_vec: list[float],
    dense_k: int = 5,
    bm25_k: int = 5,
    limit: int = 6,
) -> list[dict]:
    """Milvus 混合检索：dense + 稀疏双路召回 + RRF 融合

    LlamaIndex 0.12.52 的 hybrid 模式无双路独立 k：AnnSearchRequest 双路 limit 与
    最终 limit 均为 similarity_top_k（VectorStoreQuery.sparse_top_k 字段未被使用），
    故 similarity_top_k=limit 保总条数；dense_k/bm25_k 保留签名兼容调用方。
    库隔离走 MetadataFilters（kwargs 的 expr 被 _prepare_before_search 忽略）。
    混合检索异常时降级为纯 dense 语义检索（不阻塞主链路）；Milvus 整体不可用时
    抛异常，由上层异常处理兜底。
    """
    store = llama_store._get_store()
    filters = MetadataFilters(
        filters=[MetadataFilter(key="library_id", value=library_id)]
    )
    try:
        result = store.query(
            VectorStoreQuery(
                query_embedding=dense_vec,
                query_str=query,
                mode=VectorStoreQueryMode.HYBRID,
                similarity_top_k=limit,
                sparse_top_k=limit,
                filters=filters,
            )
        )
    except Exception as e:
        logger.warning("[vector_store] 混合检索失败，降级纯语义检索: %s", e)
        result = store.query(
            VectorStoreQuery(
                query_embedding=dense_vec,
                query_str=query,
                mode=VectorStoreQueryMode.DEFAULT,
                similarity_top_k=limit,
                filters=filters,
            )
        )
    return [llama_store.node_to_hit(n) for n in result.nodes]
