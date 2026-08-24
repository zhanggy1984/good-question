"""embedding_service L2 归一化测试（纯函数，不加载模型）

归一化是 Milvus IP 检索的前提（归一化后 IP == COSINE），属核心逻辑，必须覆盖。
"""
import math
import sys

sys.path.insert(0, "/app")

from services.embedding_service import _l2_normalize


def test_l2_normalize_single_unit_length():
    """单向量归一化后模长为 1"""
    v = _l2_normalize([3.0, 4.0])
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2)
    assert abs(norm - 1.0) < 1e-6


def test_l2_normalize_batch_shapes():
    """批量归一化保持形状且每行模长为 1"""
    vecs = _l2_normalize([[3.0, 4.0], [0.0, 5.0]])
    assert len(vecs) == 2
    assert len(vecs[0]) == 2
    for v in vecs:
        assert abs(math.sqrt(v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-6


def test_l2_normalize_zero_vector_safe():
    """零向量兜底不除零、不变"""
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_llama_fastembed_delegates_to_normalized_exports(monkeypatch):
    """LlamaFastEmbed 委托 embed_query/embed_texts（归一化唯一出口），不引入未归一化实现

    Milvus IP 检索前提是 L2 归一化；若 LlamaIndex 侧改用自带的未归一化 FastEmbed，
    排序语义会漂移。本测试守护委托关系不被改坏。
    """
    import services.embedding_service as es

    calls = {"query": None, "texts": None}
    monkeypatch.setattr(es, "embed_query", lambda q: (calls.__setitem__("query", q), [0.1])[1])
    monkeypatch.setattr(es, "embed_texts", lambda ts: (calls.__setitem__("texts", ts), [[0.2]])[1])

    adapter = es.LlamaFastEmbed()
    assert adapter._get_query_embedding("问题") == [0.1]
    assert calls["query"] == "问题"
    assert adapter._get_text_embedding("正文") == [0.2]
    assert calls["texts"] == ["正文"]
    # 批量委托透传 embed_texts 结果（mock 固定单向量），并确认文本原样传入
    assert adapter._get_text_embeddings(["a", "b"]) == [[0.2]]
    assert calls["texts"] == ["a", "b"]
