"""auth_service.authenticate 登录限流集成测试：锁定 429 / 失败计数 / 成功清零 / 异常链 429

rate_limit 模块本身在 test_login_rate_limit 有单测；这里守护 authenticate 的限流集成点：
- 锁定时抛 TooManyRequestsError（服务层），登录接口经统一异常处理器返回 429（HTTP 层）
- 密码错误 / 用户不存在 → record_failure 计数
- 成功登录 → reset 清零 + 签发 token
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "/app")

import pytest
from fastapi.testclient import TestClient

from config import settings
from main import app
from database import get_db
from services import auth_service
from utils.exceptions import InvalidCredentialsError, TooManyRequestsError


class _User:
    def __init__(self, id=1, role="admin", password_hash="hash"):
        self.id = id
        self.role = role
        self.username = "admin"
        self.password_hash = password_hash


class _Q:
    def filter(self, *a, **k):
        return self

    def first(self):
        return None


class _Db:
    def query(self, model):
        return _Q()

    def close(self):
        pass


class _QUser:
    def filter(self, *a, **k):
        return self

    def first(self):
        return _User(id=7, role="user")


class _DbUser:
    def query(self, model):
        return _QUser()

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _mock_auth(monkeypatch):
    """隔离密码校验与签发自定义，专注测限流集成"""
    monkeypatch.setattr(auth_service, "verify_password", lambda p, h: p == "correct")
    monkeypatch.setattr(auth_service, "create_access_token", lambda subject, role: "signed-result")


def test_authenticate_locked_raises_429(monkeypatch):
    """锁定时 authenticate 抛 TooManyRequestsError（HTTP 层映射 429）"""
    monkeypatch.setattr(auth_service.rate_limit, "check_allowed", lambda u: False)
    with pytest.raises(TooManyRequestsError):
        auth_service.authenticate(_Db(), "admin", "correct")


def test_authenticate_wrong_password_records_failure(monkeypatch):
    """密码错误 → record_failure 计数 + 抛 InvalidCredentialsError"""
    calls = []
    monkeypatch.setattr(auth_service.rate_limit, "check_allowed", lambda u: True)
    monkeypatch.setattr(auth_service.rate_limit, "record_failure", lambda u: calls.append(u))
    with pytest.raises(InvalidCredentialsError):
        auth_service.authenticate(_DbUser(), "admin", "wrong")
    assert calls == ["admin"]


def test_authenticate_user_missing_records_failure(monkeypatch):
    """用户不存在 → record_failure 计数 + 抛 InvalidCredentialsError"""
    calls = []
    monkeypatch.setattr(auth_service.rate_limit, "check_allowed", lambda u: True)
    monkeypatch.setattr(auth_service.rate_limit, "record_failure", lambda u: calls.append(u))
    with pytest.raises(InvalidCredentialsError):
        auth_service.authenticate(_Db(), "ghost", "correct")  # query().first() → None
    assert calls == ["ghost"]


def test_authenticate_success_resets_and_returns_token(monkeypatch):
    """成功登录 → rate_limit.reset 清零 + 返回签发结果（签发入参正确）"""
    resets = []
    captured = {}
    monkeypatch.setattr(auth_service.rate_limit, "check_allowed", lambda u: True)
    monkeypatch.setattr(auth_service.rate_limit, "reset", lambda u: resets.append(u))
    monkeypatch.setattr(
        auth_service, "create_access_token",
        lambda subject, role: captured.update(subject=subject, role=role) or "mocked",
    )
    result = auth_service.authenticate(_DbUser(), "admin", "correct")
    assert result == "mocked"
    assert captured == {"subject": "7", "role": "user"}, "签发应使用用户 id 与角色"
    assert resets == ["admin"]


def test_login_api_returns_429_when_locked(monkeypatch):
    """HTTP 层：锁定时登录接口返回 429 TOO_MANY_REQUESTS（统一异常处理器链路）"""
    app.dependency_overrides[get_db] = lambda: _Db()
    monkeypatch.setattr(auth_service.rate_limit, "check_allowed", lambda u: False)
    try:
        r = TestClient(app).post("/api/auth/login", json={"username": "admin", "password": "x"})
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "TOO_MANY_REQUESTS"
    finally:
        app.dependency_overrides.clear()


# ════════ seed_admin 存量密码轮换（升级防弱密码裸奔） ════════


def test_seed_admin_rotates_password_when_env_changed(monkeypatch):
    """.env 密码与 DB 哈希不一致 → 轮换哈希（升级部署后旧 admin123 等弱密码不可再登录）"""
    state = {}
    admin_obj = SimpleNamespace(password_hash="old-weak-hash")

    class _Q:
        def filter(self, *a, **k):
            return self

        def first(self):
            return admin_obj

    class _Db:
        def query(self, model):
            return _Q()

        def commit(self):
            state["committed"] = True

    monkeypatch.setattr(auth_service, "verify_password", lambda p, h: False)
    monkeypatch.setattr(auth_service, "hash_password", lambda p: f"rotated-{p}")
    auth_service.seed_admin(_Db())
    assert admin_obj.password_hash == f"rotated-{settings.admin_password}"
    assert state.get("committed") is True


def test_seed_admin_keeps_password_when_matching(monkeypatch):
    """.env 密码与 DB 哈希一致 → 保持不动（不重复写库）"""
    state = {}
    admin_obj = SimpleNamespace(password_hash="matching-hash")

    class _Q:
        def filter(self, *a, **k):
            return self

        def first(self):
            return admin_obj

    class _Db:
        def query(self, model):
            return _Q()

        def commit(self):
            state["committed"] = True

    monkeypatch.setattr(auth_service, "verify_password", lambda p, h: True)
    monkeypatch.setattr(auth_service, "hash_password", lambda p: "should-not-use")
    auth_service.seed_admin(_Db())
    assert admin_obj.password_hash == "matching-hash"
    assert state.get("committed") is None, "密码一致时不应轮换/写库"
