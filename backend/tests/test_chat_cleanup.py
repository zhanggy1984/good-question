"""会话过期清理测试：定时 sweep + 惰性清理（不连外部服务，monkeypatch 隔离 DB）

清理规则：最后活跃时间（chat_sessions.updated_at）超过保留期即过期，物理删除，
messages 由 DB 外键 CASCADE 级联。过期判定统一落在 DB 侧（NOW() - INTERVAL），
Python 不生成 cutoff（避免 Python/DB 时区错位）。测试通过注入 fake db /
monkeypatch 判定函数，无需真实 MySQL。
"""
import sys
from datetime import datetime

import pytest

sys.path.insert(0, "/app")

from config import settings
from services import chat_service as cs
from services.chat_cleanup import (
    SessionCleaner,
    delete_session_by_id,
    expired_session_condition,
    is_session_expired,
)
from utils.exceptions import ForbiddenError, NotFoundError


# ---------- 过期判定（DB 侧 NOW()，时区无关） ----------


def test_is_session_expired_db_now():
    """过期判定完全在 DB 侧：SQL 含 NOW() - INTERVAL，传 days 参数（非 Python cutoff）"""
    class _FakeDB:
        def __init__(self, expired):
            self._expired = expired
            self.called = None
        def execute(self, stmt, params):
            self.called = (str(stmt), params)
            return self
        def first(self):
            return (self._expired,)

    db = _FakeDB(True)
    assert is_session_expired(db, 42) is True
    assert "NOW() - INTERVAL" in db.called[0].upper()
    assert db.called[1] == {"days": settings.chat_retention_days, "id": 42}
    assert is_session_expired(_FakeDB(False), 42) is False


def test_is_session_expired_no_row_returns_false():
    """会话行不存在（已被删）→ 视为不过期（不重复删）"""
    class _NoRow:
        def execute(self, stmt, params):
            return self
        def first(self):
            return None

    assert is_session_expired(_NoRow(), 42) is False


def test_expired_session_condition_uses_db_now():
    """列表过滤表达式用 DB 侧 NOW()，无 Python cutoff"""
    expr = expired_session_condition()
    rendered = str(expr).upper()
    assert "NOW()" in rendered or "NOW" in rendered
    assert "INTERVAL" in rendered


# ---------- 单会话删除 ----------


def test_delete_session_by_id_sql_and_commit():
    """单会话删除：正确 SQL + 一次 commit"""
    class _FakeDB:
        def __init__(self):
            self.called = None
            self.committed = False
        def execute(self, stmt, params):
            self.called = (str(stmt), params)
        def commit(self):
            self.committed = True

    db = _FakeDB()
    delete_session_by_id(db, 42)
    assert db.committed
    assert "DELETE FROM chat_sessions" in db.called[0]
    assert db.called[1] == {"id": 42}


# ---------- 定时 sweep ----------


def test_sweep_batch_loop(monkeypatch):
    """分批删除：满批续删、不满批退出，每批 commit，SQL 用 DB 侧 NOW() 判定"""
    monkeypatch.setattr(settings, "chat_cleanup_batch_size", 500)

    class _Result:
        def __init__(self, n):
            self.rowcount = n

    class _FakeDB:
        def __init__(self, counts):
            self._counts = list(counts)
            self.commits = 0
            self.calls = []
        def execute(self, stmt, params):
            self.calls.append(str(stmt))
            return _Result(self._counts.pop(0))
        def commit(self):
            self.commits += 1
        def close(self):
            pass

    db = _FakeDB([500, 300])
    assert SessionCleaner().sweep(db) == 800
    assert db.commits == 2
    assert "DELETE FROM chat_sessions" in db.calls[0]
    assert "NOW() - INTERVAL" in db.calls[0].upper()


def test_sweep_removes_nothing_when_no_expired(monkeypatch):
    """无过期会话：零删除，一次查询即退出"""
    monkeypatch.setattr(settings, "chat_cleanup_batch_size", 500)

    class _Result:
        rowcount = 0

    class _FakeDB:
        def __init__(self):
            self.commits = 0
        def execute(self, stmt, params):
            return _Result()
        def commit(self):
            self.commits += 1
        def close(self):
            pass

    assert SessionCleaner().sweep(_FakeDB()) == 0


# ---------- 惰性清理：会话归属校验（详情/删除路径） ----------


class _FakeQuery:
    """fake query 链：filter -> first 返回预设会话"""

    def __init__(self, session):
        self._session = session

    def filter(self, *args):  # noqa: ANN002
        return self

    def first(self):
        return self._session


class _FakeDB:
    """fake db：query 返回预设会话，execute 记录删除 id"""

    def __init__(self, session):
        self._session = session
        self.deleted_ids = []

    def query(self, model):  # noqa: ANN001
        return _FakeQuery(self._session)

    def execute(self, stmt, params):
        self.deleted_ids.append(params["id"])
        return None

    def commit(self):
        pass

    def close(self):
        pass


class _Session:
    """最小会话桩：仅承载归属与最后活跃字段"""

    def __init__(self, session_id, user_id, updated_at):
        self.id = session_id
        self.user_id = user_id
        self.updated_at = updated_at


def test_get_owned_session_expired_deleted(monkeypatch):
    """过期会话访问：物理删除 + 视作不存在（抛 NotFound）"""
    monkeypatch.setattr(cs, "is_session_expired", lambda db, sid: True)
    db = _FakeDB(_Session(7, 1, datetime.now()))
    with pytest.raises(NotFoundError):
        cs._get_owned_session(db, 7, 1)
    assert db.deleted_ids == [7]


def test_get_owned_session_not_expired_returns(monkeypatch):
    """保留期内会话：正常返回，不删除"""
    monkeypatch.setattr(cs, "is_session_expired", lambda db, sid: False)
    db = _FakeDB(_Session(7, 1, datetime.now()))
    session = cs._get_owned_session(db, 7, 1)
    assert session.id == 7
    assert db.deleted_ids == []


def test_get_owned_session_forbidden_unchanged(monkeypatch):
    """非归属用户访问：仍抛 Forbidden（惰性清理不改变权限语义）"""
    monkeypatch.setattr(cs, "is_session_expired", lambda db, sid: False)
    db = _FakeDB(_Session(7, 1, datetime.now()))
    with pytest.raises(ForbiddenError):
        cs._get_owned_session(db, 7, 2)
    assert db.deleted_ids == []


# ---------- 惰性清理：聊天入口 ----------


def test_stream_chat_expired_yields_error(monkeypatch):
    """过期会话聊天：惰性删除 + 首个事件即 error（删除后立即返回，不进入后续流程）"""
    monkeypatch.setattr(cs, "is_session_expired", lambda db, sid: True)
    deleted = []
    monkeypatch.setattr(cs, "delete_session_by_id", lambda db, sid: deleted.append(sid))
    monkeypatch.setattr(cs, "SessionLocal", lambda: _FakeDB(_Session(7, 1, datetime.now())))

    events = list(cs.stream_chat(7, "问题"))
    assert events and events[0][0] == "error"
    assert events[0][1]["message"] == "会话不存在"
    assert deleted == [7]


# ---------- 惰性清理：列表入口 ----------


def test_list_sessions_excludes_expired(monkeypatch):
    """列表查询排除过期会话：过滤条件含 updated_at >= 过期表达式"""
    monkeypatch.setattr(cs, "expired_session_condition", expired_session_condition)

    class _FakeListQuery:
        def __init__(self):
            self.filters = []
        def filter(self, *conds):  # noqa: ANN002
            self.filters.extend(conds)
            return self
        def count(self):
            return 0
        def order_by(self, *args):  # noqa: ANN002
            return self
        def offset(self, n):
            return self
        def limit(self, n):
            return self
        def all(self):
            return []

    fake_q = _FakeListQuery()

    class _FakeListDB:
        def query(self, model):  # noqa: ANN001
            return fake_q

    cs.list_sessions(_FakeListDB(), 1, None, 1, 10)
    assert len(fake_q.filters) == 2, "user_id + 过期过滤两个条件"
    cond = fake_q.filters[1]
    assert "updated_at" in str(cond) and ">=" in str(cond)
