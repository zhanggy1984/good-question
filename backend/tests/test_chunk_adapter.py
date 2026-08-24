"""node→RetrievedChunk 接缝保真测试：LlamaIndex Node → node_to_hit → _hits_to_chunks → RetrievedChunk

守护迁移接缝：检索结果的 metadata（document_id int / heading_path 数组 / token_count）
经 LlamaIndex 动态字段往返 + 门面转换后不丢、类型不变。
"""
import sys

sys.path.insert(0, "/app")

from services import llama_store  # noqa: E402
from services.retrieval_service import _hits_to_chunks  # noqa: E402


def test_node_to_retrieved_chunk_metadata_preserved():
    """metadata 保真：document_id 为 int、heading_path 数组、token_count int 全保留"""
    node = llama_store.chunk_to_node(
        {
            "content": "正文内容",
            "metadata": {
                "document_id": 9, "chunk_index": 2,
                "document_name": "x.md", "heading_path": ["甲", "乙"],
                "token_count": 12, "source_type": "paragraph",
            },
        },
        library_id=7, embedding=[0.1],
    )
    hit = llama_store.node_to_hit(node)
    chunks = _hits_to_chunks([hit])
    assert len(chunks) == 1
    c = chunks[0]
    assert c.content == "正文内容"
    assert c.metadata["document_id"] == 9  # int 类型不变
    assert c.metadata["chunk_index"] == 2
    assert c.metadata["heading_path"] == ["甲", "乙"]
    assert c.metadata["token_count"] == 12
    assert c.metadata["library_id"] == 7
    assert c.score is None  # 检索接缝阶段尚未精排
