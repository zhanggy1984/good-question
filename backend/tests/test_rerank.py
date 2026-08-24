"""rerank 模块纯函数测试（三档判定 + 精排原始分透出，方案 R6；需容器内 llama-index-core）

关键守护：SentenceTransformerRerank 的 node.score 必须是 CrossEncoder 原始分（未归一化），
否则 0.20/0.50 置信档阈值语义漂移。以下用 Fake 精排器固定分数断言透出值。
"""
import sys
from unittest.mock import patch

sys.path.insert(0, "/app")

from llama_index.core.schema import NodeWithScore  # noqa: E402

from config import settings  # noqa: E402
from services.rerank import rerank_chunks  # noqa: E402
from services.retrieval_types import RetrievedChunk  # noqa: E402


def _fake_reranker(scores):
    """模拟 LlamaIndex SentenceTransformerRerank：postprocess_nodes 直接给 node.score 赋原始分并降序"""

    class _Fake:
        def postprocess_nodes(self, nodes, query_str=None):
            for node, s in zip(nodes, scores):
                node.score = s
            return sorted(nodes, key=lambda n: n.score if n.score is not None else -1.0, reverse=True)

    return _Fake()


def _chunks(n=3):
    return [
        RetrievedChunk(content=f"片段{i}", metadata={"document_id": i, "chunk_index": i})
        for i in range(n)
    ]


def test_rerank_orders_by_score_and_preserves_raw_score():
    """正路径：按原始分降序取 top_k，score 透出原始 CrossEncoder 分（未归一化，R6 守护）"""
    with patch("services.rerank._get_reranker", return_value=_fake_reranker([0.4, 0.9, 0.1])):
        result, max_score = rerank_chunks("查询", _chunks(), top_k=2)
    assert [c.content for c in result] == ["片段1", "片段0"]
    assert result[0].score == 0.9  # 原始分，未归一化
    assert result[1].score == 0.4
    assert max_score == 0.9


def test_rerank_low_score_returns_empty():
    """低分档：最高分 < similarity_threshold_low → 文档无关，返回空（保最高分透出）"""
    low = settings.similarity_threshold_low
    with patch("services.rerank._get_reranker", return_value=_fake_reranker([low - 0.01])):
        result, max_score = rerank_chunks("查询", _chunks(1), top_k=1)
    assert result == []
    assert max_score == low - 0.01


def test_rerank_mid_score_keeps_docs():
    """低置信档：最高分落在 [LOW, 低置信阈值) → 文档照常返回，分数透出（保召回不误杀）"""
    mid = (settings.similarity_threshold_low + settings.rerank_low_confidence_threshold) / 2
    with patch("services.rerank._get_reranker", return_value=_fake_reranker([mid])):
        result, max_score = rerank_chunks("查询", _chunks(1), top_k=1)
    assert len(result) == 1
    assert result[0].content == "片段0"
    assert max_score == mid


def test_rerank_fallback_on_failure():
    """降级：精排异常 → 取前 top_k，score=None，max_score=None（不阻塞主链路）"""
    with patch("services.rerank._get_reranker", side_effect=RuntimeError("mock 失败")):
        result, max_score = rerank_chunks("查询", _chunks(5), top_k=3)
    assert [c.content for c in result] == ["片段0", "片段1", "片段2"]
    assert result[0].score is None
    assert max_score is None


def test_rerank_empty_input():
    """空候选：直接返回空与 None"""
    result, max_score = rerank_chunks("查询", [])
    assert result == []
    assert max_score is None
