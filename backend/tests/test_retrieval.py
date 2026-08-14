"""检索/向量存储工具纯函数测试（不连外部服务，mock Milvus 客户端）"""
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, "/app")

from langchain_core.documents import Document

from services.retrieval_service import HybridRetriever, _milvus_hits_to_docs, _rerank
from services.vector_store_service import _hit_to_dict, add_chunks, hybrid_search


def test_milvus_hits_to_docs():
    hits = [
        {"document_id": 1, "chunk_index": 3, "text": "hello", "metadata": {"document_name": "a.md", "chunk_index": 3}},
        {"document_id": 2, "chunk_index": 0, "text": "world", "metadata": {}},
    ]
    docs = _milvus_hits_to_docs(hits)
    assert len(docs) == 2
    assert docs[0].page_content == "hello"
    assert docs[0].metadata["document_id"] == 1
    assert docs[0].metadata["document_name"] == "a.md"


def test_milvus_hits_to_docs_ensures_ids():
    # metadata 无 document_id/chunk_index 时从顶层补齐（Milvus 冗余字段兜底）
    hits = [{"document_id": 9, "chunk_index": 4, "text": "x", "metadata": {}}]
    docs = _milvus_hits_to_docs(hits)
    assert docs[0].metadata["document_id"] == 9
    assert docs[0].metadata["chunk_index"] == 4


def test_rerank_orders_by_score():
    """rerank 正路径：按分数倒序取 top_k"""
    docs = [Document(page_content="low"), Document(page_content="high")]

    class _FakeReranker:
        def predict(self, pairs, batch_size=16, show_progress_bar=False):
            return [0.1, 0.9]

    with patch("services.retrieval_service._get_reranker", return_value=_FakeReranker()):
        result, max_score = _rerank("查询", docs, top_k=1)
    assert result[0].page_content == "high"
    assert max_score == 0.9


def test_rerank_fallback_on_failure():
    """rerank 关键异常路径：精排失败时降级取前 top_k（不阻塞主链路）"""
    docs = [Document(page_content=f"doc{i}") for i in range(5)]
    with patch("services.retrieval_service._get_reranker", side_effect=RuntimeError("mock 失败")):
        result, max_score = _rerank("查询", docs, top_k=3)
    assert len(result) == 3
    assert result == docs[:3]
    assert max_score is None


def test_rerank_low_score_returns_empty():
    """rerank 关键判定：最高分低于低阈值时判定文档无关，返回空"""
    docs = [Document(page_content="unrelated")]

    class _LowScoreReranker:
        def predict(self, pairs, batch_size=16, show_progress_bar=False):
            return [0.05]

    with patch("services.retrieval_service._get_reranker", return_value=_LowScoreReranker()):
        result, max_score = _rerank("查询", docs, top_k=1)
    assert result == []
    assert max_score == 0.05


def test_rerank_mid_score_returns_docs():
    """低置信档：最高分落在 [LOW, 低置信阈值) 之间时，文档照常返回（保召回），分数透出"""
    docs = [Document(page_content="边缘相关")]

    class _MidScoreReranker:
        def predict(self, pairs, batch_size=16, show_progress_bar=False):
            return [0.4]

    with patch("services.retrieval_service._get_reranker", return_value=_MidScoreReranker()):
        result, max_score = _rerank("查询", docs, top_k=1)
    assert len(result) == 1
    assert result[0].page_content == "边缘相关"
    assert max_score == 0.4


def test_hybrid_retriever_propagates_max_rerank_score():
    """HybridRetriever 将 rerank 精排最高分透出到 max_rerank_score（chat 层识别低置信档的依据）"""
    docs = [Document(page_content="内容")]

    with patch("services.retrieval_service.embed_query", return_value=[0.1, 0.2]), \
            patch("services.retrieval_service.vector_store_service.hybrid_search",
                  return_value=[{"text": "内容", "metadata": {}}]), \
            patch("services.retrieval_service._rerank", return_value=(docs, 0.4)):
        retriever = HybridRetriever(library_id=1)
        result = retriever._get_relevant_documents("查询")
    assert retriever.max_rerank_score == 0.4
    assert result == docs


def test_hit_to_dict():
    """Milvus 命中实体转 dict（纯函数）"""
    hit = {"entity": {"document_id": 1, "chunk_index": 2, "text": "正文", "metadata": {"document_name": "a.md"}}}
    result = _hit_to_dict(hit)
    assert result == {"document_id": 1, "chunk_index": 2, "text": "正文", "metadata": {"document_name": "a.md"}}


def test_hybrid_search_fallback_to_dense():
    """hybrid_search 关键异常路径：BM25/混合检索异常时降级纯 dense 检索，不阻塞主链路"""
    client = Mock()
    client.hybrid_search.side_effect = RuntimeError("mock 混合检索失败")
    client.search.return_value = [[{"entity": {"document_id": 1, "chunk_index": 0, "text": "x", "metadata": {}}}]]
    with patch("services.vector_store_service.get_client", return_value=client), \
            patch("services.vector_store_service.ensure_collection"), \
            patch("services.vector_store_service.ensure_partition"):
        result = hybrid_search(1, "查询", [0.1, 0.2])
    assert len(result) == 1
    assert result[0]["document_id"] == 1
    client.search.assert_called_once()


def test_add_chunks_batches_and_reports_progress():
    """add_chunks 分批写入 + on_progress 回调（64 一批，尾批不足 64）"""
    client = Mock()
    chunks = [
        {"content": f"内容{i}", "metadata": {"document_id": 1, "chunk_index": i}}
        for i in range(70)
    ]
    with patch("services.vector_store_service.get_client", return_value=client), \
            patch("services.vector_store_service.ensure_collection"), \
            patch("services.vector_store_service.ensure_partition"), \
            patch("services.vector_store_service.embed_texts",
                  side_effect=lambda texts: [[0.1] * 4 for _ in texts]):
        progress = []
        written = add_chunks(1, chunks, on_progress=lambda w, t: progress.append((w, t)))
    assert written == 70
    assert client.upsert.call_count == 2  # 64 + 6
    assert progress == [(64, 70), (70, 70)]
