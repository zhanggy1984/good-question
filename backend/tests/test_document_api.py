"""文档读接口 admin-only 权限回归：非 admin 403 / admin 放行（TestClient + 依赖覆盖，不连真实 DB）

P0 加固：list_documents / list_document_chunks / document_status 依赖 get_current_user → get_admin_user。
守护"共享知识库"授权边界——普通用户 403（走真实 get_admin_user 逻辑）、admin 可读、无 token 401。
get_db 被覆盖为内存替身，授权校验不依赖真实数据库。
"""
import sys
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, "/app")

import pytest
from fastapi.testclient import TestClient

from main import app
from database import get_db
from middleware.auth import get_current_user
from models import Chunk, Document, DocumentLibrary

# 三个读接口路径（P0 加固对象；app 挂载在 /api 前缀下）
READ_PATHS = (
    "/api/libraries/1/documents",
    "/api/documents/1/chunks",
    "/api/documents/1/status",
)


class _User:
    """内存 User 替身：role/id 足够授权校验（不落库）"""

    def __init__(self, id: int, role: str):
        self.id = id
        self.role = role


class _FakeDB:
    """按查询模型返回预设数据的 db 替身：覆盖三个读接口的 query 链
    （filter/first/scalar/all/order_by/offset/limit；func.count 表达式不在 data 中 → 返回 0）"""

    def __init__(self, data: dict):
        self._data = data

    def query(self, model):
        rows = self._data.get(model, [])

        class _Query:
            def filter(self, *a, **k):
                return self

            def order_by(self, *a, **k):
                return self

            def offset(self, n):
                return self

            def limit(self, n):
                return self

            def first(self):
                return rows[0] if rows else None

            def scalar(self):
                return len(rows)

            def all(self):
                return rows

        return _Query()

    def close(self):
        pass


def _doc_rows():
    return [
        SimpleNamespace(
            id=1, library_id=1, filename="员工考勤管理制度.md", file_type="md",
            file_size=1024, chunk_count=6, processed_chunks=6, chunk_size=1024,
            overlap_token=102, status="ready", error_message=None,
            created_at=datetime(2026, 1, 1),
        )
    ]


def _chunk_rows():
    return [
        SimpleNamespace(
            id=1, chunk_index=0, content="事假须提前 3 个工作日提出申请",
            token_count=20, metadata_json={},
        )
    ]


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: _FakeDB({
        DocumentLibrary: [SimpleNamespace(id=1, name="演示知识库", description="", created_by=1)],
        Document: _doc_rows(),
        Chunk: _chunk_rows(),
    })
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("path", READ_PATHS)
def test_read_apis_forbidden_for_regular_user(client, path):
    """普通用户访问文档读接口应 403（真实 get_admin_user 授权逻辑，非 mock 授权）"""
    app.dependency_overrides[get_current_user] = lambda: _User(id=2, role="user")
    r = client.get(path)
    assert r.status_code == 403


@pytest.mark.parametrize("path", READ_PATHS)
def test_read_apis_ok_for_admin(client, path):
    """admin 访问文档读接口应 200 且返回结构化数据"""
    app.dependency_overrides[get_current_user] = lambda: _User(id=1, role="admin")
    r = client.get(path)
    assert r.status_code == 200
    assert r.json() is not None


@pytest.mark.parametrize("path", READ_PATHS)
def test_read_apis_unauthorized_without_token(client, path):
    """无 token 访问文档读接口应 401（真实 bearer 解析：无 header → 未登录）"""
    r = client.get(path)  # 不覆盖 get_current_user → 走真实认证链路
    assert r.status_code == 401
