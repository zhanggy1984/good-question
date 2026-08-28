"""config.Settings 连接配置测试（P3.2 共享 infra：MySQL 凭据仅 env 注入）

守护两条迁移接缝：
1. 无 MYSQL_PASSWORD 环境变量（密码空）时 database_url 抛 RuntimeError（fail-fast，防缺配静默连错库）；
2. URL 经 MYSQL_USER/MYSQL_PASSWORD 环境变量构造共享 infra 独立账号（不再用 root + MYSQL_ROOT_PASSWORD）。

经 env 注入验证而非直接构造 Settings(**kwargs)：既贴近真实运行路径，也避免凭据扫描器误报。
"""
import sys
from pathlib import Path

# 兼容容器（/app，扁平布局）与宿主机（backend/）两种布局：父目录插 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from config import Settings, settings

_TEST_VALUE = "a-test-only-value"


def test_database_url_fails_fast_when_password_missing(monkeypatch):
    """未注入 MYSQL_PASSWORD（回落到空默认）时访问 database_url 抛错"""
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    # _env_file=None：禁用 .env 读取，避免宿主机本地 dev 密码干扰本用例
    s = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="MYSQL_PASSWORD"):
        _ = s.database_url


def test_database_url_uses_env_injected_shared_account(monkeypatch):
    """MYSQL_PASSWORD 注入后，URL 用共享 infra 独立账号构造"""
    monkeypatch.setenv("MYSQL_USER", "native_rag_user")
    monkeypatch.setenv("MYSQL_PASSWORD", _TEST_VALUE)
    monkeypatch.setenv("MYSQL_DATABASE", "native_rag")
    monkeypatch.setenv("MYSQL_HOST", "mysql")
    monkeypatch.setenv("MYSQL_PORT", "3306")
    s = Settings()
    assert s.database_url == (
        f"mysql+pymysql://native_rag_user:{_TEST_VALUE}@mysql:3306/native_rag?charset=utf8mb4"
    )


# ── 安全默认值 fail-fast（main._validate_secrets）──


def test_validate_secrets_rejects_weak_defaults(monkeypatch):
    """JWT/Admin 密钥为占位默认值时应 fail-fast（禁止裸奔启动）"""
    from main import _validate_secrets
    monkeypatch.setattr(settings, "jwt_secret_key", "change-me")
    monkeypatch.setattr(settings, "admin_password", "admin123")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        _validate_secrets()


def test_validate_secrets_rejects_empty(monkeypatch):
    """Admin 密钥为空时也应 fail-fast（缺配视为未配置）"""
    from main import _validate_secrets
    monkeypatch.setattr(settings, "jwt_secret_key", "a-strong-random-jwt-key")
    monkeypatch.setattr(settings, "admin_password", "")
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        _validate_secrets()


def test_validate_secrets_rejects_template_residue_pattern(monkeypatch):
    """JWT 含模板残留特征（change-in-production / secret-key）也应 fail-fast：
    历史模板值（如 native-rag-jwt-secret-key-change-in-production）不在精确弱值集合里，
    但含明显占位特征，属本次补漏的弱值检测。"""
    from main import _validate_secrets
    monkeypatch.setattr(settings, "jwt_secret_key", "native-rag-jwt-secret-key-change-in-production")
    monkeypatch.setattr(settings, "admin_password", "a-strong-admin-pass")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        _validate_secrets()


def test_validate_secrets_accepts_real_values(monkeypatch):
    """强随机值通过校验（不应抛）"""
    from main import _validate_secrets
    monkeypatch.setattr(settings, "jwt_secret_key", "a-strong-random-jwt-key")
    monkeypatch.setattr(settings, "admin_password", "a-strong-admin-pass")
    _validate_secrets()  # 不应抛
