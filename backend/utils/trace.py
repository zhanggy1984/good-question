"""链路追踪：请求级 traceId 注入日志（T8 网关接入）。

网关 api-gateway 生成 X-Request-ID 头透传至此；若直连后端（不经网关），
中间件自动生成 uuid 兜底。日志 Filter 从 contextvar 读取，避免并发请求串号。
"""
import contextvars
import logging

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class TraceIdFilter(logging.Filter):
    """把当前请求的 trace_id 注入每条日志记录（日志格式占位符 %(trace_id)s）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = trace_id_var.get()
        return True


def install() -> None:
    """把 TraceIdFilter 装到 root logger 及其所有 handler。

    logger 的 filter 只作用于"该 logger 自己发出的记录"：子 logger 传播到 root
    handler 时不经过 root logger 的 filter；handler 的 filter 在 emit 前应用，
    覆盖所有最终落到该 handler 的记录。root logger + handler 双挂。
    """
    root = logging.getLogger()
    root.addFilter(TraceIdFilter())
    for _h in root.handlers:
        _h.addFilter(TraceIdFilter())
