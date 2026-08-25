"""文档服务测试：上传文件流式写盘 + 阶段5 overlap 填值逻辑"""
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, "/app")

from models import Document


class FakeUploadFile:
    """模拟 FastAPI UploadFile：带 filename，file 支持分块 read"""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.file = BytesIO(content)


def test_save_upload_file_streaming(tmp_path, monkeypatch):
    """分块写盘：按 library_id 分目录保存，返回正确的类型和大小（非整读内存）"""
    from config import settings
    from services import document_service

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    content = b"a" * (2 * 1024 * 1024)  # 2MB，跨多个 1MB 分块
    path, suffix, size = document_service.save_upload_file(
        FakeUploadFile("big.pdf", content), library_id=5
    )

    assert suffix == "pdf"
    assert size == len(content)
    assert Path(path).exists()
    assert Path(path).stat().st_size == size
    # 按库分目录：{upload_dir}/{library_id}/
    assert Path(path).parent == tmp_path / "5"
    # 文件名带原始名，避免覆盖
    assert Path(path).name.endswith("big.pdf")


def test_fill_overlap_ids_index_to_id():
    """overlap_prev_chunk_index → DB id 映射；index 为 None 的 chunk 不填"""
    from services.document_service import _fill_overlap_ids

    rows = [
        SimpleNamespace(id=11, chunk_index=0, metadata_json={"overlap_prev_chunk_index": None}),
        SimpleNamespace(id=12, chunk_index=1, metadata_json={"overlap_prev_chunk_index": 0}),
        SimpleNamespace(id=13, chunk_index=2, metadata_json={"overlap_prev_chunk_index": 1}),
        SimpleNamespace(id=14, chunk_index=3, metadata_json={"overlap_prev_chunk_index": None}),
    ]
    _fill_overlap_ids(rows)
    assert "overlap_prev_chunk_id" not in rows[0].metadata_json
    assert rows[1].metadata_json["overlap_prev_chunk_id"] == 11
    assert rows[2].metadata_json["overlap_prev_chunk_id"] == 12
    assert "overlap_prev_chunk_id" not in rows[3].metadata_json


def test_fill_overlap_ids_order_independent():
    """映射与输入顺序无关（按 chunk_index 建索引，DB flush 顺序不保证有序）"""
    from services.document_service import _fill_overlap_ids

    rows = [
        SimpleNamespace(id=3, chunk_index=1, metadata_json={"overlap_prev_chunk_index": 0}),
        SimpleNamespace(id=2, chunk_index=0, metadata_json={"overlap_prev_chunk_index": None}),
    ]
    _fill_overlap_ids(rows)
    assert rows[0].metadata_json["overlap_prev_chunk_id"] == 2


def test_fill_overlap_ids_single_chunk():
    """单个 chunk 无前驱 index，不填"""
    from services.document_service import _fill_overlap_ids

    rows = [SimpleNamespace(id=1, chunk_index=0, metadata_json={"overlap_prev_chunk_index": None})]
    _fill_overlap_ids(rows)
    assert "overlap_prev_chunk_id" not in rows[0].metadata_json


# ---------- reprocess_document（重新处理/重试） ----------


class FakeDoc:
    """模拟 Document 行：足够 reprocess_document 读写属性"""

    def __init__(self, status="ready", chunk_size=1024, overlap_token=102):
        self.id = 1
        self.status = status
        self.chunk_size = chunk_size
        self.overlap_token = overlap_token
        self.error_message = None
        self.chunk_count = 10
        self.processed_chunks = 10


class FakeQuery:
    """链式 query 桩：query().filter().first()/delete()"""

    def __init__(self, db, model):
        self.db = db
        self.model = model

    def filter(self, *a, **k):
        return self

    def first(self):
        # Document 返回预置行；Chunk（bulk delete）不调 first
        return self.db.doc if self.model is Document else None

    def delete(self):
        self.db.deleted_chunks = True
        return 1


class FakeDb:
    """模拟 Session：Document 查询 + Chunk bulk delete + commit"""

    def __init__(self, doc):
        self.doc = doc
        self.commits = 0
        self.deleted_chunks = False

    def query(self, model):
        return FakeQuery(self, model)

    def commit(self):
        self.commits += 1


def test_reprocess_rejects_processing():
    """processing 状态拒绝重试（防并发，DB 层快速失败）"""
    from models import Document
    from services.document_service import reprocess_document
    from utils.exceptions import ValidationError

    db = FakeDb(FakeDoc(status="processing"))
    with pytest.raises(ValidationError):
        reprocess_document(db, 1)
    assert db.commits == 0  # 拒绝路径不落库


def test_reprocess_rejects_not_found():
    """文档不存在抛 NotFoundError"""
    from models import Document
    from services.document_service import reprocess_document
    from utils.exceptions import NotFoundError

    with pytest.raises(NotFoundError):
        reprocess_document(FakeDb(None), 999)


def test_reprocess_validates_params():
    """切分参数越界拒绝：chunk_size 越界 / overlap 相对新 chunk_size 越界"""
    from models import Document
    from services.document_service import reprocess_document
    from utils.exceptions import ValidationError

    with pytest.raises(ValidationError):
        reprocess_document(FakeDb(FakeDoc()), 1, chunk_size=100)  # <128
    with pytest.raises(ValidationError):
        reprocess_document(FakeDb(FakeDoc()), 1, chunk_size=9000)  # >8192
    with pytest.raises(ValidationError):
        # chunk_size 更新为 512 后，overlap 600 应相对新值拒绝
        reprocess_document(FakeDb(FakeDoc()), 1, chunk_size=512, overlap_token=600)


def test_reprocess_reset_and_submit(monkeypatch):
    """ready 重试：清 Milvus + 清 chunks + 重置状态 + commit + start_process（含参数覆盖）"""
    import services.vector_store_service as vss
    from models import Document
    from services import document_service
    from services.document_service import reprocess_document

    deleted = []
    monkeypatch.setattr(vss, "delete_by_document", lambda doc_id: deleted.append(doc_id))
    started = []
    monkeypatch.setattr(document_service, "start_process", lambda i: started.append(i))

    doc = FakeDoc(status="ready", chunk_size=1024, overlap_token=100)
    db = FakeDb(doc)
    reprocess_document(db, 1, chunk_size=2048, overlap_token=200)

    assert deleted == [1]             # 清 Milvus 旧向量
    assert db.deleted_chunks is True  # 清 MySQL chunks
    assert doc.status == "processing"
    assert doc.error_message is None
    assert doc.chunk_count == 0
    assert doc.processed_chunks == 0
    assert doc.chunk_size == 2048     # 参数覆盖生效
    assert doc.overlap_token == 200
    assert db.commits == 1
    assert started == [1]             # 提交后台管线


def test_reprocess_rejects_processing_in_memory(monkeypatch):
    """内存层占用时拒绝（DB status 通过但 _try_acquire_processing 拒绝）：不落库、不启动

    DB status 检查有事务隔离窗口，两个并发 reprocess 可能同时读到非 processing；只有
    内存 set 原子占用是最终裁决。DB 层通过、内存层拒绝的场景必须同样走 ValidationError。
    """
    import services.document_service as ds
    from services.document_service import reprocess_document
    from utils.exceptions import ValidationError

    db = FakeDb(FakeDoc(status="ready"))  # DB 层 ready，仅内存层拦截
    monkeypatch.setattr(ds, "_try_acquire_processing", lambda doc_id: False)
    started = []
    monkeypatch.setattr(ds, "start_process", lambda i: started.append(i))
    with pytest.raises(ValidationError):
        reprocess_document(db, 1)
    assert db.commits == 0, "拒绝路径不落库"
    assert db.deleted_chunks is False, "拒绝路径不执行重置"
    assert started == [], "拒绝路径不提交后台管线"


def test_reprocess_milvus_cleanup_failure_not_blocking(monkeypatch):
    """Milvus 清理失败不阻断重跑：仍清 MySQL chunks + 重置状态 + commit + start_process

    与 delete_document 一致：Milvus 先删、失败 warning 降级；MySQL 是事实源，chunks
    不清会导致重跑重复插入。
    """
    import services.vector_store_service as vss
    from services import document_service
    from services.document_service import reprocess_document

    monkeypatch.setattr(vss, "delete_by_document", lambda doc_id: (_ for _ in ()).throw(RuntimeError("milvus down")))
    started = []
    monkeypatch.setattr(document_service, "start_process", lambda i: started.append(i))

    doc = FakeDoc(status="ready")
    db = FakeDb(doc)
    reprocess_document(db, 1)

    assert db.deleted_chunks is True, "Milvus 失败仍应清 MySQL chunks（事实源）"
    assert doc.status == "processing"
    assert db.commits == 1
    assert started == [1], "Milvus 清理失败不应阻断后台重跑"


# ---------- 防并发：_try_acquire_processing / process_document acquire+finally ----------


def test_processing_acquire_rejects_duplicate():
    """同 id 二次占用被拒（原子 check-and-add，upload 与 reprocess 统一受益）"""
    import services.document_service as ds
    with ds._processing_lock:
        ds._processing_ids.clear()
    try:
        assert ds._try_acquire_processing(1) is True
        assert ds._try_acquire_processing(1) is False, "已占用 id 二次占用应被拒"
        assert ds._try_acquire_processing(2) is True, "不同 id 互不影响"
    finally:
        with ds._processing_lock:
            ds._processing_ids.clear()


def test_process_acquire_fail_keeps_owner_mark(monkeypatch):
    """acquire 失败提前 return 不误清他人占用标记 + 不创建会话——防并发不被击穿

    回归场景：线程 A 处理 id=1（_processing_ids 含 1），线程 B 同 id acquire 失败提前
    return；若 B 的 finally 执行 discard(1)，会把 A 的占用标记清掉，随后线程 C 可再次
    占用 id=1，内存层防并发就此失效（并发双跑写 Chunk 重复）。
    同时 SessionLocal 不应被调用：acquire 失败路径创建会话是连接池泄漏（会话无人 close）。
    """
    import services.document_service as ds
    with ds._processing_lock:
        ds._processing_ids.add(1)  # 模拟 A 已占用
    try:
        created = []
        monkeypatch.setattr(
            ds, "SessionLocal",
            lambda: (created.append(1), SimpleNamespace(close=lambda: None))[1],
        )
        ds.process_document(1)
        assert 1 in ds._processing_ids, "acquire 失败路径误清他人占用标记——防并发失效"
        assert created == [], "acquire 失败提前 return 不应创建会话（防连接池泄漏）"
    finally:
        with ds._processing_lock:
            ds._processing_ids.discard(1)


class _FailDb:
    """异常路径专用 FakeDb：query(Document) 返预置 doc，rollback/commit/close 记录"""

    def __init__(self, doc):
        self.doc = doc
        self.commits = 0
        self.closed = False

    def query(self, model):
        if model is Document:
            return SimpleNamespace(filter=lambda *a, **k: SimpleNamespace(first=lambda: self.doc))
        raise AssertionError(f"异常路径不应查询 {model}")

    def rollback(self):
        pass

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def test_process_document_releases_processing_after_failure(monkeypatch):
    """异常路径结束 finally 释放占用 + 失败清理（rollback/delete_by_document/status=failed）"""
    import services.document_service as ds
    from services import vector_store_service

    doc = SimpleNamespace(id=1, filename="x.txt", file_path="/tmp/x.txt", file_type="txt",
                          library_id=7, status="processing", chunk_size=1024,
                          overlap_token=100, error_message=None)
    db = _FailDb(doc)
    with ds._processing_lock:
        ds._processing_ids.clear()
    monkeypatch.setattr(ds, "SessionLocal", lambda: db)
    monkeypatch.setattr(ds.mineru_extractor, "extract_text",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("解析失败")))
    deleted = []
    monkeypatch.setattr(vector_store_service, "delete_by_document", lambda doc_id: deleted.append(doc_id))

    ds.process_document(1)

    assert 1 not in ds._processing_ids, "异常路径 finally 应释放占用，可再次处理"
    assert doc.status == "failed" and "解析失败" in doc.error_message
    assert deleted == [1], "失败应清理已写入向量（MySQL 为准）"
    assert db.commits >= 1
    assert db.closed is True
