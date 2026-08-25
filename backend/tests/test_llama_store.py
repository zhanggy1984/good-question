"""llama_store 纯函数测试（node↔hit 转换，不连 Milvus；LlamaIndex 类型已在容器内）

守护迁移接缝：node_id={document_id}_{chunk_index} 与旧 upsert 主键一致、
ref_doc_id=document_id（delete_by_document 语义不变）、library_id 注入 metadata。
"""
import sys

import pytest

sys.path.insert(0, "/app")

from services import llama_store  # noqa: E402


def test_chunk_to_node_sets_ids_and_library():
    """chunk → TextNode：node_id/ref_doc_id/library_id 与迁移前主键、删除语义对齐"""
    node = llama_store.chunk_to_node(
        {"content": "正文", "metadata": {"document_id": 42, "chunk_index": 1, "document_name": "a.md"}},
        library_id=7, embedding=[0.1, 0.2],
    )
    assert node.node_id == "42_1"  # 与旧 upsert 主键格式一致（幂等语义）
    assert node.ref_doc_id == "42"  # relationships[SOURCE] 派生，delete(ref_doc_id) 按此删整文档
    assert node.metadata["library_id"] == 7
    assert node.metadata["document_id"] == 42  # int 保留在 _node_content，消费侧读 int 不变
    assert node.text == "正文"
    assert node.embedding == [0.1, 0.2]  # 显式注入，不依赖 LlamaIndex 自动 embed


def test_node_to_hit_shape():
    """node → dict：document_id/chunk_index/text/metadata 与旧 _hit_to_dict 形状对齐"""
    node = llama_store.chunk_to_node(
        {
            "content": "正文",
            "metadata": {"document_id": 42, "chunk_index": 1, "document_name": "a.md", "heading_path": ["标题"]},
        },
        library_id=7, embedding=[0.1],
    )
    hit = llama_store.node_to_hit(node)
    assert hit["document_id"] == 42
    assert hit["chunk_index"] == 1
    assert hit["text"] == "正文"
    assert hit["metadata"]["document_name"] == "a.md"
    assert hit["metadata"]["heading_path"] == ["标题"]


def test_ensure_event_loop_binds_in_worker_thread():
    """工作线程（Python 3.11+ 无事件循环）调用后 get_event_loop 可用

    守护 MilvusVectorStore 构造前提（AsyncMilvusClient grpc aio 需线程有 loop）：
    FastAPI 文档处理在 ThreadPoolExecutor 线程跑，无 loop 时首次 _get_store 构造必炸。
    """
    import asyncio
    import threading

    result = {}

    def _worker():
        try:
            asyncio.get_event_loop()
            result["had_loop"] = True
        except RuntimeError:
            result["had_loop"] = False
        llama_store._ensure_event_loop()
        try:
            asyncio.get_event_loop()
            result["ok"] = True
        except RuntimeError:
            result["ok"] = False

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert result["had_loop"] is False, "子线程不应预置事件循环（Python 3.11+）"
    assert result["ok"] is True, "_ensure_event_loop 应为无 loop 线程兜底绑定事件循环"


def test_flush_delegates_to_client_flush(monkeypatch):
    """flush 委托 _get_store().client.flush——删除后必须显式 flush 才立即可见"""
    class _FakeClient:
        def flush(self, name):
            self.called_with = name

    class _FakeStore:
        client = _FakeClient()

    monkeypatch.setattr(llama_store, "_get_store", lambda: _FakeStore())
    llama_store.flush()
    assert _FakeStore.client.called_with == llama_store.COLLECTION_NAME


def test_ensure_loaded_skips_when_collection_missing(monkeypatch):
    """collection 不存在（首启未传文档）时跳过 load，不抛错

    守护 ensure_loaded 的分支：Milvus 重启后 collection 不自动 load，
    不 load 检索会报 "collection not loaded"；但首启集合不存在时 has_collection 探为 False 应静默跳过。
    """
    class _FakeClient:
        def __init__(self):
            self.loaded = False

        def has_collection(self, name):
            return False

        def load_collection(self, name):
            self.loaded = True
            self.loaded_name = name

    fake = _FakeClient()
    monkeypatch.setattr(llama_store, "_get_milvus_client", lambda: fake)
    llama_store.ensure_loaded()
    assert fake.loaded is False


def test_ensure_loaded_loads_existing_collection(monkeypatch):
    """collection 已存在（重启后未自动 load）时调用 load_collection(COLLECTION_NAME)"""
    class _FakeClient:
        def __init__(self):
            self.loaded = False

        def has_collection(self, name):
            return True

        def load_collection(self, name):
            self.loaded = True
            self.loaded_name = name

    fake = _FakeClient()
    monkeypatch.setattr(llama_store, "_get_milvus_client", lambda: fake)
    llama_store.ensure_loaded()
    assert fake.loaded is True
    assert fake.loaded_name == llama_store.COLLECTION_NAME


# ════════ #9 embedding 维度 fail-fast 校验（换模型后旧 collection 维度不匹配提前报错）════════


def test_raise_on_dim_mismatch_raises():
    """dense 维度与当前模型输出不一致：抛 EmbeddingDimensionMismatchError，错误信息含两侧维度"""
    fields = [
        {"name": "dense", "params": {"dim": 1024}},
        {"name": "sparse_embedding", "params": {}},
    ]
    with pytest.raises(llama_store.EmbeddingDimensionMismatchError) as ei:
        llama_store._raise_on_dim_mismatch(fields, 768)
    assert "1024" in str(ei.value) and "768" in str(ei.value)


def test_raise_on_dim_mismatch_passes():
    """dense 维度与模型一致：不抛"""
    llama_store._raise_on_dim_mismatch([{"name": "dense", "params": {"dim": 768}}], 768)


def test_raise_on_dim_mismatch_missing_dense_ignored():
    """schema 无 dense 字段（异常/不完整 schema）：静默通过，不误抛"""
    llama_store._raise_on_dim_mismatch([{"name": "pk", "params": {}}], 768)


def test_ensure_dimension_match_fails_fast_on_mismatch(monkeypatch):
    """启动校验：collection 存在 + 维度不一致 → fail-fast 抛错（不做静默 drop，丢向量需用户决定）"""
    class _FakeClient:
        def has_collection(self, name):
            return True

        def describe_collection(self, name):
            return {"fields": [
                {"name": "dense", "params": {"dim": 1024}},
                {"name": "sparse_embedding", "params": {}},
            ]}

    monkeypatch.setattr(llama_store, "_get_milvus_client", lambda: _FakeClient())
    monkeypatch.setattr(llama_store, "embed_texts", lambda texts: [[0.0] * 768])
    with pytest.raises(llama_store.EmbeddingDimensionMismatchError):
        llama_store.ensure_dimension_match()


def test_ensure_dimension_match_skips_missing_collection(monkeypatch):
    """collection 不存在（首启未上传文档）：跳过校验不抛错"""
    class _FakeClient:
        def has_collection(self, name):
            return False

        def describe_collection(self, name):
            raise AssertionError("collection 不存在不应执行 describe")

    monkeypatch.setattr(llama_store, "_get_milvus_client", lambda: _FakeClient())
    llama_store.ensure_dimension_match()  # 不抛即通过
