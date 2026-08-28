"""安全响应头中间件 + CORS 收紧测试（HTTP 层，TestClient 不连真实 DB）"""
import sys

sys.path.insert(0, "/app")

import pytest
from fastapi.testclient import TestClient

from main import app


def test_security_headers_present():
    """所有 API 响应都应带 4 个安全响应头（nosniff / XFO / Referrer-Policy / CSP）"""
    r = TestClient(app).get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "SAMEORIGIN"
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "font-src 'self' data:" in csp, "font-src 应与 nginx 页面层保持一致"


def test_cors_allows_whitelist_origin():
    """CORS 收紧：白名单来源允许，但绝不放行 *"""
    r = TestClient(app).options(
        "/api/auth/login",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_non_whitelist_origin():
    """CORS 收紧：白名单外来源不放行（无 allow-origin 且非 *）"""
    r = TestClient(app).options(
        "/api/auth/login",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    allow = r.headers.get("access-control-allow-origin")
    assert allow is None or allow != "*", "不应向任意来源开放跨域"
