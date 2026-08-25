"""文档库服务测试：删库后清该库问答缓存（防 TTL 窗口内旧答案重放）"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/app")

from models import DocumentLibrary


class _FakeDb:
    """模拟 Session：仅 delete_document 路径需要的 delete + commit"""

    def __init__(self):
        self.deleted = None
        self.commits = 0

    def delete(self, row):
        self.deleted = row

    def commit(self):
        self.commits += 1


def test_delete_library_flushes_cache(monkeypatch):
    """删库后清该库问答缓存 + Milvus 库数据，MySQL 删除落库"""
    import services.chat_cache as chat_cache
    import services.library_service as lib_svc
    from services import vector_store_service

    lib = SimpleNamespace(id=7, name="测试库")
    db = _FakeDb()

    flushed = []
    monkeypatch.setattr(lib_svc, "get_library", lambda db_, library_id: lib)
    monkeypatch.setattr(vector_store_service, "delete_library_collection", lambda library_id: None)
    monkeypatch.setattr(chat_cache, "flush_library", lambda library_id: flushed.append(library_id))

    lib_svc.delete_library(db, 7)

    assert db.deleted is lib
    assert db.commits == 1
    assert flushed == [7], "删库后应立即清该库问答缓存，而非等 TTL 兜底"


def test_delete_library_milvus_failure_not_blocking(monkeypatch):
    """Milvus 清理失败不阻断删库：仍删 MySQL + 清问答缓存（与 delete_document 降级一致）"""
    import services.chat_cache as chat_cache
    import services.library_service as lib_svc
    from services import vector_store_service

    lib = SimpleNamespace(id=7, name="测试库")
    db = _FakeDb()

    monkeypatch.setattr(lib_svc, "get_library", lambda db_, library_id: lib)
    monkeypatch.setattr(
        vector_store_service,
        "delete_library_collection",
        lambda library_id: (_ for _ in ()).throw(RuntimeError("milvus down")),
    )
    flushed = []
    monkeypatch.setattr(chat_cache, "flush_library", lambda library_id: flushed.append(library_id))

    lib_svc.delete_library(db, 7)

    assert db.deleted is lib, "Milvus 清理失败不应阻断删库（MySQL 是事实源）"
    assert db.commits == 1
    assert flushed == [7], "Milvus 清理失败仍应清问答缓存"
