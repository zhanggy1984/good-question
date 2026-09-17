"""链路追踪：请求级 traceId 注入日志（T8 网关接入）。

网关 api-gateway 生成 X-Request-ID 头透传至此；若直连后端（不经网关），
中间件自动生成 uuid 兜底。日志 Filter 从 contextvar 读取，避免并发请求串号。
"""
import contextvars
import logging

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

# 请求级 LLM 硬失败标记。SSE 接口恒返 200（HTTP 码判不出「LLM 挂了、用户拿到兜底话术」），
# 平台判定侧只认 root 终态 ⇒ 业务层必须在失败出口置位，观测中间件出口据此把 root 记 error，
# 否则真实故障会被当作「已被业务吸收」切掉回流候选。
# ⚠️ 持可变 dict 而非标量：置位发生在深层调用里，标量赋值传不回中间件持有的那个上下文；
# dict 按引用跨上下文副本共享，写才可见。
llm_health_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "llm_health", default=None
)


def mark_llm_hard_fail(error_type: str) -> None:
    """置位本请求的 LLM 硬失败（只取首次，error_type 用平台白名单词）。

    写侧 fail-loud：上下文缺失时置位是静默空转——那样「标记丢了」会表现成「trace 记 ok」，
    正是本机制要治的病，故必须打日志暴露而不是默默返回。
    """
    health = llm_health_var.get()
    if health is None:
        logging.getLogger("native_rag").error(
            "[obs] llm_health 上下文缺失，LLM 硬失败标记丢失（error_type=%s）；"
            "生产环境出现即中间件未按序置入 context", error_type,
        )
        return
    if not health["hard_fail"]:
        health["hard_fail"] = True
        health["error_type"] = error_type


def unmark_llm_hard_fail() -> None:
    """撤销标记：本次失败确定要重试时调用，避免「重试成功」被误记为 LLM 硬失败。

    上下文缺失时无事可撤（mark 同样什么也没做），静默返回即可，不重复告警。
    """
    health = llm_health_var.get()
    if health is None:
        return
    health["hard_fail"] = False
    health["error_type"] = None


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
