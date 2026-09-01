"""检索/向量存储门面测试（mock LlamaIndex store，不连 Milvus）

迁移后 vector_store_service 门面走 llama_store._get_store() 懒加载单例，
测试 patch 该单例为 Mock，覆盖分批/幂等/降级/库隔离；HybridRetriever 链路 mock 到底层。
"""
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, "/app")

from llama_index.core.vector_stores.types import VectorStoreQueryMode  # noqa: E402

from config import settings  # noqa: E402
from services.llama_store import COLLECTION_NAME  # noqa: E402
from services import retrieval_service as rs  # noqa: E402
from services.retrieval_service import HybridRetriever, _hits_to_chunks, execute_retrieve_tool  # noqa: E402
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


def test_hybrid_search_extra_filters_appended():
    """章节扩充过滤：extra_filters 追加为 AND 等值过滤（与 library_id 并列），不覆盖库隔离"""
    store = _mock_store()
    result = Mock()
    result.nodes = []
    store.query.return_value = result
    with patch("services.llama_store._get_store", return_value=store):
        out = hybrid_search(3, "查询", [0.1, 0.2], extra_filters={"document_id": 1, "section_id": "1:0"})
    assert out == []
    q = store.query.call_args[0][0]
    pairs = {(f.key, f.value) for f in q.filters.filters}
    assert ("library_id", 3) in pairs, "extra_filters 不得覆盖库隔离过滤"
    assert ("document_id", 1) in pairs
    assert ("section_id", "1:0") in pairs


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


# ════════ 章节级扩充（#5）：按 (document_id, section_id) 回查兄弟 chunk 合并 context ════════


def _chunk(text, doc_id, chunk_index, section_id):
    return RetrievedChunk(
        content=text,
        metadata={"document_id": doc_id, "section_id": section_id, "chunk_index": chunk_index},
    )


def test_expand_section_context_merges_siblings():
    """top-3 涉及的 section 兄弟 chunk 合并进 context（按 chunk_index 去重），顺序 top 在前"""
    top = [
        _chunk("A", 1, 0, "1:0"),
        _chunk("B", 1, 2, "1:0"),
    ]
    sibling_hits = [
        {"document_id": 1, "chunk_index": 1, "text": "sib1",
         "metadata": {"document_id": 1, "section_id": "1:0", "chunk_index": 1}},
        {"document_id": 1, "chunk_index": 3, "text": "sib3",
         "metadata": {"document_id": 1, "section_id": "1:0", "chunk_index": 3}},
    ]
    with patch("services.retrieval_service.embed_query", return_value=[0.1]), \
            patch("services.retrieval_service.vector_store_service.hybrid_search",
                  return_value=sibling_hits) as hs:
        retriever = HybridRetriever(library_id=1)
        merged = retriever._expand_section_context("查询", top)
    assert [c.content for c in merged] == ["A", "B", "sib1", "sib3"], "top 在前，兄弟 chunk 按命中顺序追加"
    # 回查按 document_id 过滤（str：Milvus 标量字段存字符串，传 int 生成不了匹配表达式），
    # limit 用章节上限；相邻节扩展在应用层完成（命中节 ±ADJACENT 并入），不在查询层过滤 section
    _, kwargs = hs.call_args
    assert kwargs["extra_filters"] == {"document_id": "1"}
    assert kwargs["limit"] == 20


def test_expand_section_context_skips_without_section_id():
    """旧数据无 section_id 时天然退化：不扩充，原样返回 top（不查 Milvus）"""
    top = [_chunk("A", 1, 0, None)]
    with patch("services.retrieval_service.embed_query") as eq, \
            patch("services.retrieval_service.vector_store_service.hybrid_search") as hs:
        retriever = HybridRetriever(library_id=1)
        out = retriever._expand_section_context("查询", top)
    assert out is top, "无 section_id 应原样返回"
    eq.assert_not_called()
    hs.assert_not_called()


def test_expand_section_context_respects_total_cap(monkeypatch):
    """扩充总量上限：合并结果截断到 _SECTION_EXPAND_TOTAL，防大 section 撑爆 prompt"""
    monkeypatch.setattr("services.retrieval_service._SECTION_EXPAND_TOTAL", 3)
    top = [_chunk("A", 1, 0, "1:0"), _chunk("B", 1, 1, "1:0")]
    sibling_hits = [
        {"document_id": 1, "chunk_index": 2, "text": "sib2",
         "metadata": {"document_id": 1, "section_id": "1:0", "chunk_index": 2}},
        {"document_id": 1, "chunk_index": 3, "text": "sib3",
         "metadata": {"document_id": 1, "section_id": "1:0", "chunk_index": 3}},
    ]
    with patch("services.retrieval_service.embed_query", return_value=[0.1]), \
            patch("services.retrieval_service.vector_store_service.hybrid_search",
                  return_value=sibling_hits):
        retriever = HybridRetriever(library_id=1)
        merged = retriever._expand_section_context("查询", top)
    assert len(merged) == 3, "扩充后应受总量上限截断"


# ════════ 检索工具结果组装（由 chat_service 下沉，随函数迁移） ════════


def test_execute_retrieve_tool_structure(monkeypatch):
    """execute_retrieve_tool：包装 HybridRetriever，返回 context/sources/source_count/max_score/confidence_band

    max_score 来自 rerank（numpy float32），必须转原生 float——json.dumps 不认 float32，
    否则 tool_call SSE 事件与 JSON 日志序列化崩溃（曾线上复现）。
    """
    import json
    import numpy as np
    from types import SimpleNamespace
    chunk = SimpleNamespace(
        content="内容内容内容",
        metadata={"document_name": "测试.md", "heading_path": ["标题"], "chunk_index": 1, "total_chunks": 3},
    )

    class _Retriever:
        max_rerank_score = np.float32(0.9)  # 模拟 rerank 返回的 numpy 标量
        top_hits = []  # 章节扩充前未设置时退化为 invoke 结果（与 HybridRetriever.__init__ 契约一致）
        def __init__(self, *a, **k):
            pass
        def invoke(self, q):
            return [chunk]

    monkeypatch.setattr(rs, "HybridRetriever", _Retriever)
    r = rs.execute_retrieve_tool(7, "问题")
    assert r["source_count"] == 1
    assert r["confidence_band"] == "high"
    assert isinstance(r["max_score"], float), "max_score 应转原生 float（numpy float32 不可 JSON 序列化）"
    json.dumps({"source_count": r["source_count"], "max_score": r["max_score"]})  # 序列化不抛
    assert r["sources"][0]["document_name"] == "测试.md"
    assert "测试.md" in r["context"] and "内容内容内容" in r["context"]


def test_execute_retrieve_tool_sources_all_chunks_with_expanded_flag(monkeypatch):
    """章节扩充后：sources 携带全部 context chunk（含扩充），expanded 标记区分精排/扩充。

    编号与 context 的 [来源N] 一一对应（引用可全量核对）；精排 top-3 expanded=False，
    扩充 chunk expanded=True（前端降权标"补充上下文"）。
    """
    from types import SimpleNamespace
    top1 = SimpleNamespace(content="A", metadata={"document_name": "a.md", "chunk_index": 0})
    top2 = SimpleNamespace(content="B", metadata={"document_name": "a.md", "chunk_index": 1})
    expanded = [top1, top2, SimpleNamespace(content="C", metadata={"document_name": "a.md", "chunk_index": 2})]

    class _Retriever:
        def __init__(self, *a, **k):
            self.max_rerank_score = 0.8
            self.top_hits = [top1, top2]

        def invoke(self, q):
            return expanded

    monkeypatch.setattr(rs, "HybridRetriever", _Retriever)
    r = rs.execute_retrieve_tool(7, "问题")
    assert r["source_count"] == 3, "source_count 应取扩充后 context 量（与 [来源N] 编号一致）"
    assert len(r["sources"]) == 3, "sources 应携带全部 context chunk（含扩充），编号一一对应"
    assert [s["chunk_index"] for s in r["sources"]] == [0, 1, 2]
    assert [s["expanded"] for s in r["sources"]] == [False, False, True], "精排 False、扩充 True"


def test_confidence_band_three_band():
    """二期置信档三档：none / low / high（tool_call result 与监控日志用）"""
    low = settings.similarity_threshold_low
    high = settings.rerank_low_confidence_threshold
    assert rs._confidence_band(None) == "none"
    assert rs._confidence_band(low - 0.01) == "none"
    assert rs._confidence_band((low + high) / 2) == "low"
    assert rs._confidence_band(high) == "high"
    assert rs._confidence_band(high + 0.1) == "high"
