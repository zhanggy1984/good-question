"""api.chat._stream_with_disconnect_check 测试（#7 后端断连检测）

客户端停止生成 → request.is_disconnected()=True → 停止迭代并 gen.close()，
终止对 DeepSeek 的调用（触发 stream_chat 的 finally：db.close / httpx 流关闭），
不再烧 token 与连接资源。正常迭代结束 close 是 no-op。

生成器无 .closed 属性，close() 是否被调用用 finally 块打点验证。
"""
import asyncio
import sys

sys.path.insert(0, "/app")

from api.chat import _stream_with_disconnect_check  # noqa: E402


def test_disconnect_stops_before_first_yield():
    """首事件即断连：wrapper 不产出任何事件就 return，并关闭底层生成器"""
    async def _run():
        events = []
        closed = {"n": 0}

        def gen():
            try:
                yield "token", {"content": "a"}
                yield "token", {"content": "b"}
            finally:
                closed["n"] += 1

        async def is_disconnected():
            return True

        async for ev, data in _stream_with_disconnect_check(gen(), is_disconnected):
            events.append((ev, data))
        return events, closed

    events, closed = asyncio.run(_run())
    assert events == [], "断连时不应再向客户端产出事件"
    assert closed["n"] == 1, "断连后应 close 生成器（触发 stream_chat 清理）"


def test_disconnect_mid_stream():
    """事件流中途断连：已 yield 的事件保留，后续不再迭代，生成器被关闭"""
    async def _run():
        events = []
        checked = {"n": 0}
        closed = {"n": 0}

        def gen():
            try:
                yield "token", {"content": "a"}
                yield "token", {"content": "b"}
                yield "token", {"content": "c"}
            finally:
                closed["n"] += 1

        async def is_disconnected():
            checked["n"] += 1
            return checked["n"] >= 2  # 第二个事件检查时断连

        async for ev, data in _stream_with_disconnect_check(gen(), is_disconnected):
            events.append((ev, data))
        return events, closed

    events, closed = asyncio.run(_run())
    assert [e for e, _ in events] == ["token"], "第二个事件检查断连，仅第一个事件已产出"
    assert closed["n"] == 1, "中途断连应 close 生成器"


def test_normal_iteration_yields_all_and_closes():
    """客户端未断连：全部事件迭代完，finally 里 close 对已结束生成器是 no-op"""
    async def _run():
        events = []
        closed = {"n": 0}

        def gen():
            try:
                yield "token", {"content": "a"}
                yield "usage", {"total": 1}
            finally:
                closed["n"] += 1

        async def is_disconnected():
            return False

        async for ev, data in _stream_with_disconnect_check(gen(), is_disconnected):
            events.append((ev, data))
        return events, closed

    events, closed = asyncio.run(_run())
    assert [e for e, _ in events] == ["token", "usage"], "未断连应迭代完所有事件"
    assert closed["n"] == 1, "正常结束后 close 同样调用（对已结束生成器是 no-op）"
