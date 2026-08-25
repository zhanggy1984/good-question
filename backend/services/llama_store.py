"""LlamaIndex Milvus 存储适配（RAG 检索迁移一期的 LlamaIndex 专属管道）

本模块是 LlamaIndex 专属管道的唯一隔离层：MilvusVectorStore 懒加载单例 + node↔hit
转换纯函数。vector_store_service 门面层在此之上做业务逻辑（分批/进度/降级/幂等），
检索与 SSE 编排层不直接接触 LlamaIndex 类型。

关键设计（Step 0 PoC 验证，见 scripts/poc_llamaindex.py）：
- text_key=None：读回走 metadata_dict_to_node（从 _node_content JSON 完整还原
  text/metadata/数组/int），避免 schema 无 text 列时 text=None 触发
  TextNode ValidationError（string_type）。
- node.ref_doc_id（relationships[SOURCE]）：node_to_metadata_dict 把顶层 document_id
  字段写为 node.ref_doc_id（未设则 "None"），store.delete(ref_doc_id) 按该字段删整个
  文档——delete_by_document 语义与旧接口一致。
- add 必须 force_flush=True：否则 insert 延迟可见（stats row_count=0，~0.4s 后才可查）。
- delete 后必须 flush：否则删除延迟可见（delete_count=1 但立即 query 仍见旧数据）。
- 库隔离：query 用 filters=MetadataFilters（kwargs 的 expr 被 _prepare_before_search
  忽略，只认 query.filters/doc_ids/node_ids）。
- 稀疏路：BGE-M3 学习稀疏（enable_sparse=True + sparse_embedding_function），
  字段名 sparse_embedding 由 MilvusVectorStore 硬编码；dense 索引用 HNSW/IP。
- 旧 collection（服务端 BM25 + pk 主键 + enable_dynamic_field=False）schema 不兼容，
  _get_store 检测到缺 sparse_embedding 字段即 drop，由 LlamaIndex 重建（数据可从
  MySQL 重灌，示例项目无丢失）。
"""
import asyncio
import logging
import threading
from functools import lru_cache

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores.milvus import IndexManagement, MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from pymilvus import MilvusClient

from config import settings
from services.embedding_service import embed_texts

logger = logging.getLogger("native_rag")

COLLECTION_NAME = "rag_chunks"


def _ensure_event_loop() -> None:
    """确保当前线程存在 asyncio 事件循环

    MilvusVectorStore 构造内部创建 pymilvus grpc async channel，依赖当前线程有事件
    循环（uvloop.get_event_loop 无 loop 时抛 RuntimeError）。文档处理在 FastAPI
    ThreadPoolExecutor 线程跑（默认无 loop），migrate/PoC 在主线程（有 loop）——
    故这里兜底为任意线程绑定一个 loop，保证 lru_cache 单例首次构造不在工作线程炸掉。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())


@lru_cache(maxsize=1)
def _get_sparse_fn():
    """BGE-M3 稀疏编码器单例（构造即加载 bge-m3 模型，~2.3GB，缓存于 model_cache volume）"""
    return BGEM3SparseEmbeddingFunction()


def _get_milvus_client() -> MilvusClient:
    """轻量 Milvus 客户端（schema 探测/重建用）"""
    return MilvusClient(uri=settings.milvus_uri)


_store_lock = threading.Lock()
_cached_store: MilvusVectorStore | None = None


def _build_store() -> MilvusVectorStore:
    """实际构造 MilvusVectorStore（仅首次由 _get_store 锁保护下调用）

    旧版 collection（服务端 BM25）与 0.12.52 学习稀疏模型不兼容，检测到缺
    sparse_embedding 字段即 drop 重建（数据可从 MySQL 重灌）。
    """
    _ensure_event_loop()
    client = _get_milvus_client()
    if client.has_collection(COLLECTION_NAME):
        schema = client.describe_collection(COLLECTION_NAME)
        fields = [f["name"] for f in schema.get("fields", [])]
        if "sparse_embedding" not in fields:
            logger.warning(
                "[llama_store] 检测到旧版 collection=%s（无 sparse_embedding 字段），drop 重建",
                COLLECTION_NAME,
            )
            client.drop_collection(COLLECTION_NAME)
    # dense 维度取自 embedding 模型实际输出，避免与配置漂移
    dim = len(embed_texts(["维度探测"])[0])
    store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection_name=COLLECTION_NAME,
        dim=dim,
        embedding_field="dense",
        # text_key=None：读回走 _node_content 完整还原，避免 schema 无 text 列时
        # text=None 触发 TextNode ValidationError（见模块 docstring）
        text_key=None,
        output_fields=["*"],
        enable_sparse=True,
        sparse_embedding_function=_get_sparse_fn(),
        hybrid_ranker="RRFRanker",
        index_config={"index_type": "HNSW", "M": 16, "efConstruction": 200},
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        overwrite=False,
    )
    logger.info("[llama_store] MilvusVectorStore 就绪 collection=%s dim=%s", COLLECTION_NAME, dim)
    return store


def _get_store() -> MilvusVectorStore:
    """MilvusVectorStore 线程安全懒加载单例

    不用 lru_cache：它不串行化并发 cache-miss（thundering herd），文档处理在
    3 线程 ThreadPoolExecutor 并发首次上传会同时构造 MilvusVectorStore，collection
    创建（CREATE_IF_NOT_EXISTS 非原子）可能竞争失败。模块级锁 + double-checked
    保证只有一个线程真正构造，其余复用。
    """
    global _cached_store
    if _cached_store is not None:
        return _cached_store
    with _store_lock:
        if _cached_store is None:
            _cached_store = _build_store()
    return _cached_store


def flush() -> None:
    """强制 flush：Milvus 的 insert/delete 均为延迟可见，flush 后才立即反映到查询

    add 用 force_flush=True 已覆盖；delete 需显式调用本函数（MilvusVectorStore.delete
    内部不 flush）。delete_by_document / delete_library_collection 删除后必须调用。

    显式 ensure loop：当前走同步 client 本不需要，但作为 llama_store 的公共入口，
    与 _get_store() 保持同一约束（构造/操作不得依赖调用线程已有事件循环），防御未来改 async。
    """
    _ensure_event_loop()
    _get_store().client.flush(COLLECTION_NAME)


def ensure_loaded() -> None:
    """启动时加载 rag_chunks collection：Milvus 重启后 collection 不自动 load，
    不 load 时检索会报 "collection not loaded"。失败由调用方（lifespan）兜底告警。

    collection 可能尚不存在（首启未上传任何文档），先 has_collection 探再 load。
    """
    client = _get_milvus_client()
    if client.has_collection(COLLECTION_NAME):
        client.load_collection(COLLECTION_NAME)
        logger.info("[llama_store] Milvus collection 已加载: %s", COLLECTION_NAME)


def chunk_to_node(chunk: dict, library_id: int, embedding: list[float]) -> TextNode:
    """chunk dict（{content, metadata}）→ TextNode

    - node_id = {document_id}_{chunk_index}：与旧 upsert 主键格式一致，保证幂等主键语义
    - ref_doc_id = str(document_id)：node_to_metadata_dict 把顶层 document_id 字段写为
      node.ref_doc_id（未设则 "None"），store.delete(ref_doc_id) 按该字段删整个文档。
      metadata 里的 document_id(int) 仍保留在 _node_content，消费侧读 int 不变。
    - embedding 显式注入：embedding_service 已 L2 归一化（Milvus IP 前提），不依赖
      LlamaIndex 自动 embed（其 FastEmbedEmbedding 不归一化）。
    """
    md = {**chunk["metadata"], "library_id": library_id}
    node = TextNode(text=chunk["content"], metadata=md, embedding=embedding)
    node.node_id = f"{md['document_id']}_{md['chunk_index']}"
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=str(md["document_id"]))
    return node


def node_to_hit(node) -> dict:
    """LlamaIndex 检索结果 node → dict（vector_store_service.hybrid_search 返回形状不变）

    返回形状与旧 _hit_to_dict 对齐：document_id/chunk_index/text/metadata。
    """
    md = node.metadata or {}
    return {
        "document_id": md.get("document_id"),
        "chunk_index": md.get("chunk_index"),
        "text": node.text or "",
        "metadata": md,
    }
