"""安全工具测试：密码哈希与 JWT（纯本地）"""
import sys
sys.path.insert(0, "/app")

from utils.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify():
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h)


def test_wrong_password_rejected():
    h = hash_password("secret123")
    assert not verify_password("wrongpass", h)


def test_invalid_hash_returns_false():
    assert not verify_password("secret123", "not-a-valid-hash")


def test_token_roundtrip():
    token = create_access_token("42", "admin")
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "42"
    assert payload["role"] == "admin"


def test_invalid_token_returns_none():
    assert decode_access_token("garbage.token.here") is None
