"""检索/向量存储门面测试（mock LlamaIndex store，不连 Milvus）

迁移后 vector_store_service 门面走 llama_store._get_store() 懒加载单例，
测试 patch 该单例为 Mock，覆盖分批/幂等/降级/库隔离；HybridRetriever 链路 mock 到底层。
"""
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, "/app")

from llama_index.core.vector_stores.types import VectorStoreQueryMode  # noqa: E402

from services.llama_store import COLLECTION_NAME  # noqa: E402
from services.retrieval_service import HybridRetriever, _hits_to_chunks  # noqa: E402
from services.retrieval_types import RetrievedChunk  # noqa: E402
from services.vector_store_service import (  # noqa: E402
    add_chunks,
    delete_by_document,
    delete_library_collection,
    hybrid_search,
)


def _mock_store():
    store = Mock()
    store.client = Mock()
    return store


def test_hits_to_chunks():
    """Milvus 命中 dict → RetrievedChunk：metadata 缺 document_id/chunk_index 时从顶层补齐"""
    hits = [
        {"document_id": 1, "chunk_index": 3, "text": "hello", "metadata": {"document_name": "a.md", "chunk_index": 3}},
        {"document_id": 2, "chunk_index": 0, "text": "world", "metadata": {}},
    ]
    chunks = _hits_to_chunks(hits)
    assert len(chunks) == 2
    assert chunks[0].content == "hello"
    assert chunks[0].metadata["document_name"] == "a.md"
    assert chunks[0].metadata["document_id"] == 1
    assert chunks[1].metadata["chunk_index"] == 0


def test_hybrid_retriever_invoke_propagates_max_rerank_score():
    """HybridRetriever.invoke 链路：embed → hybrid_search → rerank_chunks → max_rerank_score 透出"""
    with patch("services.retrieval_service.embed_query", return_value=[0.1, 0.2]), \
            patch("services.retrieval_service.vector_store_service.hybrid_search",
                  return_value=[{"text": "内容", "metadata": {"document_id": 1, "chunk_index": 0}}]), \
            patch("services.retrieval_service.rerank.rerank_chunks",
                  return_value=([RetrievedChunk(content="内容", metadata={"document_id": 1})], 0.4)):
        retriever = HybridRetriever(library_id=1)
        result = retriever.invoke("查询")
    assert retriever.max_rerank_score == 0.4
    assert len(result) == 1
    assert result[0].content == "内容"


def test_hybrid_search_library_filter():
    """库隔离：filters 按 library_id 构造（LlamaIndex 只认 query.filters，忽略 kwargs 的 expr）"""
    store = _mock_store()
    result = Mock()
    result.nodes = []
    store.query.return_value = result
    with patch("services.llama_store._get_store", return_value=store):
        out = hybrid_search(3, "查询", [0.1, 0.2])
    assert out == []
    q = store.query.call_args[0][0]
    assert q.filters.filters[0].key == "library_id"
    assert q.filters.filters[0].value == 3
    assert q.mode == VectorStoreQueryMode.HYBRID


def test_hybrid_search_fallback_to_dense():
    """hybrid 检索异常 → 降级纯 dense（mode=DEFAULT），不阻塞主链路"""
    store = _mock_store()
    result = Mock()
    result.nodes = []
    store.query.side_effect = [RuntimeError("mock 混合检索失败"), result]
    with patch("services.llama_store._get_store", return_value=store):
        out = hybrid_search(1, "查询", [0.1, 0.2])
    assert out == []
    assert store.query.call_count == 2
    assert store.query.call_args_list[1][0][0].mode == VectorStoreQueryMode.DEFAULT


def test_add_chunks_batches_and_reports_progress():
    """add_chunks 分批写入 + on_progress 回调（64 一批，尾批不足 64）+ 幂等先删同文档"""
    store = _mock_store()
    chunks = [
        {"content": f"内容{i}", "metadata": {"document_id": 1, "chunk_index": i}}
        for i in range(70)
    ]
    with patch("services.llama_store._get_store", return_value=store), \
            patch("services.llama_store.flush"), \
            patch("services.vector_store_service.embed_texts",
                  side_effect=lambda texts: [[0.1] * 4 for _ in texts]):
        progress = []
        written = add_chunks(1, chunks, on_progress=lambda w, t: progress.append((w, t)))
    assert written == 70
    assert store.add.call_count == 2  # 64 + 6
    assert progress == [(64, 70), (70, 70)]
    # 幂等：LlamaIndex add 是 insert 非 upsert，add 前按 document_id 删旧数据
    store.delete.assert_called_once_with(ref_doc_id="1")


def test_add_chunks_empty_returns_zero():
    """空 chunk 列表：直接返回 0，不触碰 store"""
    store = _mock_store()
    with patch("services.llama_store._get_store", return_value=store):
        assert add_chunks(1, []) == 0
    store.add.assert_not_called()


def test_delete_by_document():
    """delete_by_document：store.delete(ref_doc_id) + flush（Milvus 删除延迟可见）"""
    store = _mock_store()
    with patch("services.llama_store._get_store", return_value=store), \
            patch("services.llama_store.flush") as flush:
        delete_by_document(9)
    store.delete.assert_called_once_with(ref_doc_id="9")
    flush.assert_called_once()


def test_delete_library_collection():
    """delete_library_collection：按 library_id 字段过滤删除 + flush"""
    store = _mock_store()
    with patch("services.llama_store._get_store", return_value=store), \
            patch("services.llama_store.flush") as flush:
        delete_library_collection(3)
    store.client.delete.assert_called_once()
    _, kwargs = store.client.delete.call_args
    assert kwargs["collection_name"] == COLLECTION_NAME
    assert kwargs["filter"] == "library_id == 3"
    flush.assert_called_once()
