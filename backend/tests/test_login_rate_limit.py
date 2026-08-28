"""登录防爆破限流测试（rate_limit）：锁定/重置/窗口/fail-open，不连真实 Redis

对齐 test_chat_cache 模式：内存 fake client + monkeypatch 替换 _client()；
Redis 连接异常用抛错 client 模拟 fail-open（限流器故障不锁死登录）。
"""
from types import SimpleNamespace

import utils.rate_limit as rl


class _FakeRedis:
    """内存版 Redis（get/incr/expire/delete/pipeline），测试替代真实连接"""
    def __init__(self):
        self._d = {}
        self._ttl = {}

    def get(self, key):
        return self._d.get(key)

    def incr(self, key):
        self._d[key] = int(self._d.get(key, 0)) + 1
        return self._d[key]

    def expire(self, key, seconds):
        self._ttl[key] = seconds
        return True

    def delete(self, key):
        return 1 if self._d.pop(key, None) is not None else 0

    def pipeline(self):
        return _FakePipeline(self)


class _FakePipeline:
    """内存版 pipeline：累积操作，execute 时批量应用到 store"""
    def __init__(self, store):
        self._store = store
        self._ops = []

    def incr(self, key):
        self._ops.append(("incr", key))
        return self

    def expire(self, key, seconds):
        self._ops.append(("expire", key, seconds))
        return self

    def execute(self):
        for op in self._ops:
            if op[0] == "incr":
                self._store.incr(op[1])
            else:
                self._store.expire(op[1], op[2])
        return []


def _reset(monkeypatch):
    """清 Redis 故障冷却状态（fail-open 测试会设置 _DISABLED_UNTIL，避免污染其他用例）"""
    monkeypatch.setattr(rl, "_DISABLED_UNTIL", 0.0)
    monkeypatch.setattr(rl.settings, "login_fail_max", 5)


def test_lock_after_max_failures(monkeypatch):
    """连续失败达 login_fail_max 次后锁定：check_allowed 返回 False"""
    _reset(monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(rl, "_client", lambda: fake)
    assert rl.check_allowed("bob") is True, "未失败前应放行"
    for _ in range(rl.settings.login_fail_max):
        rl.record_failure("bob")
    assert rl.check_allowed("bob") is False, "达到上限后应锁定"


def test_check_allowed_below_max(monkeypatch):
    """失败次数低于上限时仍放行（如 max=5，失败 4 次第 5 次仍可试）"""
    _reset(monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(rl, "_client", lambda: fake)
    for _ in range(rl.settings.login_fail_max - 1):
        rl.record_failure("bob")
    assert rl.check_allowed("bob") is True


def test_success_reset_clears(monkeypatch):
    """登录成功 reset 后计数清零，账号恢复可登录"""
    _reset(monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(rl, "_client", lambda: fake)
    for _ in range(rl.settings.login_fail_max):
        rl.record_failure("bob")
    assert rl.check_allowed("bob") is False
    rl.reset("bob")
    assert rl.check_allowed("bob") is True, "reset 后应恢复"


def test_window_ttl_set_on_failure(monkeypatch):
    """每次失败设置窗口 TTL（滑动窗口），窗口过期由 Redis 自动清计数"""
    _reset(monkeypatch)
    fake = _FakeRedis()
    monkeypatch.setattr(rl, "_client", lambda: fake)
    rl.record_failure("bob")
    assert fake._ttl.get(rl._key("bob")) == rl.settings.login_fail_window_seconds, \
        "失败必须设置窗口 TTL，否则计数永不失效"


def test_redis_error_fail_open(monkeypatch):
    """Redis 异常时 fail-open：check_allowed 放行、record_failure 不抛（防限流器故障锁死登录）"""
    _reset(monkeypatch)

    class _Boom:
        def get(self, *a, **k):
            raise ConnectionError("redis down")
        def pipeline(self, *a, **k):
            raise ConnectionError("redis down")
        def delete(self, *a, **k):
            raise ConnectionError("redis down")

    _factory = type("R", (), {"from_url": staticmethod(lambda *a, **k: _Boom())})
    monkeypatch.setattr(rl, "_redis_lib", SimpleNamespace(Redis=_factory))
    assert rl.check_allowed("bob") is True, "Redis 异常应 fail-open 放行"
    rl.record_failure("bob")  # 不应抛
    rl.reset("bob")           # 不应抛
    assert rl._DISABLED_UNTIL > 0, "异常后应进入冷却期（避免反复白等 socket timeout）"


def test_cooldown_skips_redis(monkeypatch):
    """冷却期内 _client 返回 None，不再尝试连接（fail-open 继续放行）"""
    _reset(monkeypatch)
    monkeypatch.setattr(rl, "_DISABLED_UNTIL", 10 ** 12)  # 模拟冷却中
    assert rl._client() is None
    assert rl.check_allowed("bob") is True, "冷却期内应 fail-open"
