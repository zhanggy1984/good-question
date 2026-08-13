"""Milvus 向量库服务：统一语义检索（dense）+ BM25 全文检索（sparse）

单 collection `rag_chunks` + 每文档库一个 partition（p_{library_id}）实现库隔离：
- dense: jina-embeddings-v2-base-zh 768 维，embedding_service 输出已 L2 归一化（归一化后 IP == COSINE），index 用 IP 内积
- sparse: Milvus 内置 FunctionType.BM25，由 text 字段服务端自动分词生成稀疏向量，应用层不传
"""
import logging
import threading
import time

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from config import settings
from services.embedding_service import embed_texts

logger = logging.getLogger("native_rag")

COLLECTION_NAME = "rag_chunks"
TEXT_MAX_LENGTH = 65535  # 1024-token chunk 中文约 3k 字符，65535 富余
BATCH_SIZE = 64


def _partition_name(library_id: int) -> str:
    """库 partition 命名"""
    return f"p_{library_id}"


_client: MilvusClient | None = None
_client_lock = threading.Lock()
# collection/partition 创建锁：文档处理线程池并发首个请求时串行化 check-then-act，避免重复创建竞态
_ensure_lock = threading.Lock()


def get_client() -> MilvusClient:
    """懒加载 Milvus 客户端（单例，线程安全）

    文档处理线程池（3 worker）与请求线程并发首次调用时，双重检查锁保证只建一个连接。
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MilvusClient(uri=settings.milvus_uri)
                logger.info("[milvus] 已连接 %s", settings.milvus_uri)
    return _client


def _build_schema(dim: int):
    """构建 collection schema（含 BM25 function；analyzer 为 collection 级永久配置，建后不可改）"""
    schema = get_client().create_schema(enable_dynamic_field=False)
    schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(field_name="document_id", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="library_id", datatype=DataType.INT64)
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=TEXT_MAX_LENGTH,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
        enable_match=True,
    )
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="sparse_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    schema.add_function(
        Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse_bm25"],
            function_type=FunctionType.BM25,
        )
    )
    return schema


def _build_index_params():
    """索引：dense 用 HNSW/IP（向量已在 embedding_service L2 归一化，IP 等价于 COSINE），sparse 用 BM25 倒排"""
    index_params = get_client().prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse_bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    return index_params


def ensure_collection() -> None:
    """确保 collection 存在（幂等懒创建，线程安全）；dense 维度取自 embedding 模型实际输出，避免与配置漂移

    处理线程池（3 worker）并发首次写入时，双重检查 + 锁保证只有一个线程执行创建；
    对"重复创建"异常幂等处理，防止并发下文档处理被误标 failed。
    """
    client = get_client()
    if client.has_collection(COLLECTION_NAME):
        return
    with _ensure_lock:
        if client.has_collection(COLLECTION_NAME):
            return
        dim = len(embed_texts(["维度探测"])[0])
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=_build_schema(dim),
                index_params=_build_index_params(),
            )
            logger.info("[milvus] 已创建 collection=%s dim=%s", COLLECTION_NAME, dim)
        except Exception as e:
            # 另一线程已创建（create 幂等冲突），经确认存在则视为成功
            if not client.has_collection(COLLECTION_NAME):
                raise
            logger.warning("[milvus] collection 创建冲突已幂等处理: %s", e)


def ensure_partition(library_id: int) -> None:
    """确保某库 partition 存在（幂等，线程安全）"""
    client = get_client()
    pname = _partition_name(library_id)
    if client.has_partition(COLLECTION_NAME, pname):
        return
    with _ensure_lock:
        if client.has_partition(COLLECTION_NAME, pname):
            return
        try:
            client.create_partition(COLLECTION_NAME, pname)
            logger.info("[milvus] 已创建 partition=%s", pname)
        except Exception as e:
            if not client.has_partition(COLLECTION_NAME, pname):
                raise
            logger.warning("[milvus] partition 创建冲突已幂等处理: %s", e)


def add_chunks(
    library_id: int,
    chunks: list[dict],
    on_progress=None,
) -> int:
    """批量写入 chunk 到 Milvus（自动向量化），分批处理

    分批目的：大文档数百/上千 chunk 一次性 embed 会打满 CPU、撑高内存，
    分批可平滑资源占用。upsert 幂等（主键 pk={document_id}_{chunk_index}），可安全重跑。

    on_progress(written, total)：每批写入完成后回调一次，供上层把进度写库
    （前端轮询展示"处理中 N 段"）；None 则只打日志。
    返回实际写入的 chunk 数。
    """
    if not chunks:
        return 0
    client = get_client()
    ensure_collection()
    ensure_partition(library_id)
    pname = _partition_name(library_id)

    total = len(chunks)
    written = 0
    start = time.time()
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        embeddings = embed_texts(texts)
        rows = [
            {
                "pk": f"{c['metadata']['document_id']}_{c['metadata']['chunk_index']}",
                "document_id": c["metadata"]["document_id"],
                "chunk_index": c["metadata"]["chunk_index"],
                "library_id": library_id,
                "text": c["content"],
                "dense": embeddings[j],
                "metadata": c["metadata"],
            }
            for j, c in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, data=rows, partition_name=pname)
        written += len(batch)
        if on_progress:
            on_progress(written, total)
        logger.info(
            "[vector_store] 写入进度 %s/%s（批 %s/%s）",
            written, total, i // BATCH_SIZE + 1, (total + BATCH_SIZE - 1) // BATCH_SIZE,
        )
    logger.info(
        "[vector_store] 写入完成 %s chunk 到 collection=%s partition=%s 耗时=%.1fs",
        total, COLLECTION_NAME, pname, time.time() - start,
    )
    return written


def delete_by_document(document_id: int) -> None:
    """删除某文档的全部向量（按 document_id 过滤，跨全 collection，与库无关）"""
    client = get_client()
    if not client.has_collection(COLLECTION_NAME):
        return
    client.delete(collection_name=COLLECTION_NAME, filter=f"document_id == {document_id}")


def delete_library_collection(library_id: int) -> None:
    """删除整个文档库的 partition"""
    client = get_client()
    pname = _partition_name(library_id)
    if client.has_collection(COLLECTION_NAME) and client.has_partition(COLLECTION_NAME, pname):
        client.drop_partition(COLLECTION_NAME, pname)
        logger.info("[vector_store] 已删除 partition=%s", pname)


def _hit_to_dict(hit) -> dict:
    """Milvus 命中结果转 dict（供检索层转 Document）"""
    entity = hit["entity"]
    return {
        "document_id": entity.get("document_id"),
        "chunk_index": entity.get("chunk_index"),
        "text": entity.get("text", ""),
        "metadata": entity.get("metadata", {}),
    }


def hybrid_search(
    library_id: int,
    query: str,
    dense_vec: list[float],
    dense_k: int = 5,
    bm25_k: int = 5,
    limit: int = 6,
) -> list[dict]:
    """Milvus 混合检索：dense + BM25 双路召回 + RRF 融合

    默认候选数仅供未显式指定时兜底；实际检索由 HybridRetriever.candidate_k 控制
    （默认 3+3 双路 → RRF 融合取 6，与替换前语义 3 + ES 3 相当，不放大 Rerank 瓶颈）。
    BM25/混合检索异常时降级为纯 dense 语义检索（不阻塞主链路）；
    Milvus 整体不可用时抛异常，由上层异常处理兜底。
    """
    client = get_client()
    ensure_collection()
    ensure_partition(library_id)
    pname = _partition_name(library_id)

    # ef（efSearch）为查询时 HNSW 动态候选列表：默认偏保守，显式设 128 提升语义召回。
    # 属运行期查询参数，不影响索引、可随时回退；候选仅 6 个，ef=128 足够且开销可忽略。
    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense",
        param={"metric_type": "IP", "ef": 128},
        limit=dense_k,
    )
    bm25_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_bm25",
        param={"metric_type": "BM25"},
        limit=bm25_k,
    )
    try:
        results = client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[dense_req, bm25_req],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["document_id", "chunk_index", "text", "metadata"],
            partition_names=[pname],
        )
    except Exception as e:
        logger.warning("[vector_store] 混合检索失败，降级纯语义检索: %s", e)
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[dense_vec],
            anns_field="dense",
            search_params={"metric_type": "IP", "ef": 128},
            limit=limit,
            output_fields=["document_id", "chunk_index", "text", "metadata"],
            partition_names=[pname],
        )
    return [_hit_to_dict(hit) for hit in results[0]]
