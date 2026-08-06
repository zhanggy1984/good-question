"""检索工具纯函数测试（不连外部服务）"""
import sys
sys.path.insert(0, "/app")

from langchain_core.documents import Document

from services.retrieval_service import _deduplicate, _es_hits_to_docs


def test_dedup_by_document_and_chunk():
    d1 = Document(page_content="a", metadata={"document_id": 1, "chunk_index": 0})
    d2 = Document(page_content="a2", metadata={"document_id": 1, "chunk_index": 0})
    d3 = Document(page_content="b", metadata={"document_id": 2, "chunk_index": 0})
    result = _deduplicate([d1, d2, d3])
    # d2 与 d1 同 document_id+chunk_index 被去重，保留首个
    assert len(result) == 2
    assert result[0].page_content == "a"


def test_dedup_preserves_different_chunks():
    d1 = Document(page_content="a", metadata={"document_id": 1, "chunk_index": 0})
    d2 = Document(page_content="b", metadata={"document_id": 1, "chunk_index": 1})
    assert len(_deduplicate([d1, d2])) == 2


def test_es_hits_to_docs():
    hits = [
        {"document_id": 1, "text": "hello", "metadata": {"chunk_index": 3, "document_name": "a.md"}},
        {"document_id": 2, "text": "world", "metadata": {"chunk_index": 0}},
    ]
    docs = _es_hits_to_docs(hits)
    assert len(docs) == 2
    assert docs[0].page_content == "hello"
    assert docs[0].metadata["document_id"] == 1
    assert docs[0].metadata["document_name"] == "a.md"


def test_es_hits_to_docs_ensures_document_id():
    # metadata 无 document_id 时从顶层补齐
    hits = [{"document_id": 9, "text": "x", "metadata": {}}]
    docs = _es_hits_to_docs(hits)
    assert docs[0].metadata["document_id"] == 9
