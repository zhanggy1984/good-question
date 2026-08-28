"""登录防爆破限流（Redis 计数）：同一账号连续失败超限即临时锁定，窗口过期自动重置。

与 chat_cache 的熔断降级语义不同——缓存是纯优化可静默降级，限流是安全链路：
Redis 故障时 fail-open（放行登录）+ error 告警，绝不把"限流器不可用"变成全员锁死
（那本身就是 DoS）。安全兜底靠 bcrypt + JWT，Redis 计数只是防爆破加速器。
"""
import logging
import time
from typing import Any

from config import settings

logger = logging.getLogger("native_rag")

try:
    import redis as _redis_lib
except ImportError:  # 宿主无 redis 依赖时限流失效（fail-open，与容器内 Dockerfile 已装一致）
    _redis_lib = None

# 单例 Redis 客户端：redis-py 对象自带连接池，复用同一实例即复用连接池
_client_instance: Any | None = None

# Redis 故障冷却：告警后短暂跳过重试，避免每次登录都白等 socket timeout
_DISABLED_UNTIL: float = 0.0
_DISABLE_COOLDOWN_SECONDS = 30


def _client() -> Any | None:
    """惰性取单例 Redis 客户端（复刻 chat_cache._client；无熔断降级，异常由调用方 fail-open）"""
    global _client_instance
    if _redis_lib is None:
        return None
    if time.time() < _DISABLED_UNTIL:
        return None
    if _client_instance is None:
        _client_instance = _redis_lib.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    return _client_instance


def _key(username: str) -> str:
    """限流 key：按账号计数（防定向爆破）；前缀隔离，便于 scan 排查"""
    return f"goodq:login_fail:{username}"


def _fault(e: Exception) -> None:
    """Redis 异常：冷却期内不再尝试 + error 告警。调用方需 fail-open（放行登录）"""
    global _DISABLED_UNTIL, _client_instance
    _DISABLED_UNTIL = time.time() + _DISABLE_COOLDOWN_SECONDS
    _client_instance = None
    logger.error("[login.ratelimit] Redis 访问异常，登录限流失效（fail-open）: %s", e)


def check_allowed(username: str) -> bool:
    """该账号是否允许继续尝试登录（连续失败达 login_fail_max 返回 False）。

    fail-open：Redis 不可用时放行（登录走 bcrypt/JWT 正常校验，仅失去防爆破保护）。
    """
    client = _client()
    if client is None:
        return True
    try:
        return int(client.get(_key(username)) or 0) < settings.login_fail_max
    except Exception as e:
        _fault(e)
        return True


def record_failure(username: str) -> None:
    """记录一次登录失败；窗口是滑动 TTL（每次失败刷新），过期自动重置计数"""
    client = _client()
    if client is None:
        # 冷却期内静默（fail-open 语义已由 _client() 表达），避免每次失败登录刷 ERROR；
        # 冷却期结束首探仍不可用时由 _fault 重新告警。
        if time.time() >= _DISABLED_UNTIL:
            logger.error("[login.ratelimit] Redis 不可用，登录限流失效（fail-open）")
        return
    try:
        pipe = client.pipeline()
        pipe.incr(_key(username))
        pipe.expire(_key(username), settings.login_fail_window_seconds)
        pipe.execute()
    except Exception as e:
        _fault(e)


def reset(username: str) -> None:
    """登录成功清空失败计数"""
    client = _client()
    if client is None:
        return
    try:
        client.delete(_key(username))
    except Exception as e:
        _fault(e)
