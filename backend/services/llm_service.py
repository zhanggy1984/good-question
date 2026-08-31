"""LLM 服务：DeepSeek（OpenAI 兼容协议）

通过 langchain-openai 的 ChatOpenAI 封装，模型名从 .env 读取，可切换。
"""
import json
import logging
import re
import time
import uuid
from functools import lru_cache

import httpx
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger("native_rag")


@lru_cache(maxsize=2)
def get_llm(streaming: bool = False) -> ChatOpenAI:
    """获取 DeepSeek LLM 实例（streaming 区分流式/非流式，各缓存一份）"""
    logger.debug("[llm] 初始化 ChatOpenAI model=%s streaming=%s", settings.deepseek_model, streaming)
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        streaming=streaming,
        temperature=0.3,
        max_retries=2,
        timeout=120,
    )


REWRITE_PROMPT = """你是文档检索辅助助手。将用户的问题改写为更利于文档检索的查询语句。

要求：
1. 保持原问题的完整语义，不要只提取关键词堆砌
2. 补充同义词和相关概念
3. 口语化表述改为书面语
4. 输出一句完整、自然的检索查询（不超过 40 字），不要解释

用户问题：{question}
检索查询："""


# 显式缓存字典（比 lru_cache 可观测：命中/未命中均有日志）
_rewrite_cache: dict[str, str] = {}


def rewrite_query(question: str) -> str:
    """改写用户问题为利于检索的查询（口语化→规范化、补同义词），失败返回原问题

    显式缓存：相同问题不重复调用 LLM（减少延迟与成本），命中时打日志便于观测。
    """
    if question in _rewrite_cache:
        logger.debug("[llm] query 改写命中缓存: %s", question[:20])
        return _rewrite_cache[question]
    try:
        llm = get_llm(streaming=False)
        result = llm.invoke(REWRITE_PROMPT.format(question=question)).content.strip()
        result = result if result else question
    except Exception as e:
        logger.warning("[llm] query 改写失败，使用原问题: %s", e)
        result = question
    _rewrite_cache[question] = result
    return result


# ═══════════ 流式对话调用（由 chat_service 下沉，资源层职责） ═══════════


_TOOL_CALLS_OPEN = "<tool_calls>"


def _parse_content_tool_calls(text: str) -> list[dict] | None:
    """解析 DeepSeek 偶发渲染进 content 的工具调用 XML，返回结构化 tool_calls 列表。

    背景：标准 function calling 走 delta.tool_calls 结构化字段；但 DeepSeek 偶发（实测 ~1/7
    概率）把工具调用直接渲染为 content 文本（<tool_calls><invoke name="X"><parameter
    name="Y">Z</parameter></invoke></tool_calls>），不携带结构化 tool_calls。若不识别，
    这段 XML 会被当作普通回答流式输出给用户（前端显示原始 XML），且污染 Redis 缓存。
    这里解析成与结构化路径一致的 tool_calls 结构，由调用方走正常检索流程。

    无法解析返回 None（调用方兜底原样输出，不吞用户内容）。
    """
    m = re.search(r"<invoke name=\"([^\"]+)\">(.*?)</invoke>", text, re.S)
    if not m:
        return None
    name = m.group(1).strip()
    args_xml = m.group(2)
    args: dict = {}
    for pm in re.finditer(r"<parameter name=\"([^\"]+)\"[^>]*>(.*?)</parameter>", args_xml, re.S):
        args[pm.group(1).strip()] = pm.group(2)
    if not name or not args:
        return None
    # 伪 id：DeepSeek 只校验 assistant tool_calls 与 tool 消息 tool_call_id 一致，
    # 不校验 id 来源（已实测 call_<hex> 格式被接受），供第二轮 tool 消息回传
    return [{"id": "call_" + uuid.uuid4().hex, "type": "function",
             "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]


def _tool_call_event(tool_acc: dict) -> dict:
    """把按 index 累积的 tool_calls 组装成 yield 事件（arguments 保持 JSON 字符串，不 parse）"""
    calls = [
        {
            "id": item["id"],
            "type": "function",
            "function": {"name": item["name"], "arguments": item["arguments"]},
        }
        for item in tool_acc.values()
    ]
    return {"type": "tool_call", "tool_calls": calls}


def stream_chat(messages: list[dict], tools: list[dict] | None = None):
    """直连 DeepSeek 流式调用，解析 reasoning_content / content / tool_calls / usage

    yield {"type": "reasoning"|"content"|"tool_call"|"usage", ...}
    - reasoning/content：增量文本
    - tool_call：LLM 决定调用工具时（finish_reason=="tool_calls"）一次性 flush 完整参数
      （tool_calls 按 delta 分片按 index 累加，arguments 为 JSON 字符串）
    - usage：流末 usage chunk（include_usage 开启后返回，choices 为空），data 为 usage 对象
    """
    payload = {
        "model": settings.deepseek_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.3,
        # 思考过程开关：deepseek-chat 默认不返回 reasoning_content，开启 thinking 才输出
        #（前端"💭 思考过程"折叠块依赖 reasoning 事件）。deepseek-reasoner 恒返回，不受影响。
        # 条件构建避免传 "thinking": None 被 API 拒绝。
        "stream_options": {"include_usage": True},
    }
    if settings.deepseek_thinking_enabled:
        payload["thinking"] = {"type": "enabled"}
    if tools:
        payload["tools"] = tools  # 不设 tool_choice，默认 auto，由 LLM 自主决定
    headers = {
        "Authorization": f"Bearer {settings.deepseek_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.stream(
        "POST",
        f"{settings.deepseek_base_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    ) as resp:
        # HTTP 错误显式化：非 2xx 响应体是 JSON 错误（非 SSE），若不检查会静默零 yield、
        # 被当成"LLM 空返回"而空白收尾（实测 401/429 路径）。抛异常后由各调用点统一
        # except 接住 → yield error 事件 → 前端明确提示（区别于静默空白）。
        resp.raise_for_status()
        tool_acc: dict = {}  # index -> {"id", "name", "arguments"}；arguments 增量拼接
        content_buf = ""  # content 累积：检测 DeepSeek 偶发的工具调用 XML 渲染（见 _parse_content_tool_calls）
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
                # 流末 usage chunk：choices 为空但有 usage 字段（须在取 delta 前判断）
                if chunk.get("usage"):
                    yield {"type": "usage", "usage": chunk["usage"]}
                    continue
                choice = chunk["choices"][0]
                delta = choice["delta"]
                finish_reason = choice.get("finish_reason")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            # tool_calls 按 index 分片累加（id/name 首个分片携带，arguments 增量拼接）
            if "tool_calls" in delta:
                for tc in delta.get("tool_calls", []):
                    item = tool_acc.setdefault(
                        tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.get("id"):
                        item["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        item["name"] = fn["name"]
                    if fn.get("arguments"):
                        item["arguments"] += fn["arguments"]
            if "reasoning_content" in delta and delta["reasoning_content"]:
                yield {"type": "reasoning", "content": delta["reasoning_content"]}
            if "content" in delta and delta["content"]:
                # 工具调用 XML 渲染检测：DeepSeek 偶发把 tool_calls 渲染进 content（而非
                # delta.tool_calls 结构化字段）。流式分片下 <tool_calls> 可能被拆成多片
                #（实测 < 与 tool_calls> 分 chunk 到达），故用"前缀相容"判断而非完整串
                # 匹配。逐块处理 content_buf：普通文本输出、闭合 XML 块解析为 tool_call 事件、
                # 未闭合的 XML 渲染前缀保留累积等后续 chunk。普通文本每次消费后清空
                # content_buf——否则流结束兜底会重复输出完整回答（实测 judge 因"重复表述"
                # 扣分到 80，教训：buf 生命周期必须与文本流一致）。
                content_buf += delta["content"]
                while True:
                    m = re.search(r"<tool_calls>.*?</tool_calls>", content_buf, re.S)
                    if m:
                        if m.start() > 0:
                            yield {"type": "content", "content": content_buf[:m.start()]}
                        parsed = _parse_content_tool_calls(m.group(0))
                        if parsed is not None:
                            yield {"type": "tool_call", "tool_calls": parsed}
                        else:
                            # 解析失败不吞输出：兜底原样 yield（保守，宁可显示原文也不丢内容）
                            yield {"type": "content", "content": m.group(0)}
                        content_buf = content_buf[m.end():]
                        continue
                    stripped = content_buf.lstrip()
                    # 未闭合的 XML 渲染前缀：与 <tool_calls> 前缀相容即保留累积
                    #（含完整 <tool_calls> 开头，或 <tool_calls> 被拆分片如 "<"）
                    if stripped.startswith("<tool_calls>") or _TOOL_CALLS_OPEN.startswith(stripped):
                        break
                    # 完整 <tool_calls> 子串出现在 buf 中但未闭合（前有普通文本的跨界场景）
                    open_pos = content_buf.rfind("<tool_calls>")
                    if open_pos != -1 and "</tool_calls>" not in content_buf[open_pos:]:
                        if open_pos > 0:
                            yield {"type": "content", "content": content_buf[:open_pos]}
                        content_buf = content_buf[open_pos:]
                        break
                    # 纯普通文本：全量输出并清空（不留给流结束兜底，避免重复）
                    if content_buf:
                        yield {"type": "content", "content": content_buf}
                    content_buf = ""
                    break
            # 结束信号：finish_reason=="tool_calls" 的 chunk 通常带着最后一个 tool_calls
            # 分片（须先累积再 flush，保证参数完整）
            if finish_reason == "tool_calls" and tool_acc:
                yield _tool_call_event(tool_acc)
                tool_acc = {}
        # [DONE] 兜底：个别实现不返回 finish_reason 也能累积到 tool_calls
        if tool_acc:
            yield _tool_call_event(tool_acc)
        # 工具调用 XML 未闭合兜底：DeepSeek 中断/截断时 <tool_calls> 前缀不吞输出（原样 yield）
        if content_buf:
            yield {"type": "content", "content": content_buf}


def _is_transient_http_error(e: Exception) -> bool:
    """瞬时错误判定：429 限流 / 5xx 服务端错误可重试，401 等客户端错误不可

    raise_for_status 抛 httpx.HTTPStatusError；response 为 None（测试 fake）时
    取不到状态码，视为不可重试（宁可少重试一次也不重复计费）。
    """
    if isinstance(e, httpx.HTTPStatusError):
        resp = getattr(e, "response", None)
        code = resp.status_code if resp is not None else None
        return code == 429 or (code is not None and code >= 500)
    return False


def stream_round1_with_retry(messages: list[dict], tools: list[dict] | None = None):
    """第一轮 LLM 流式：瞬时错误（429/5xx）且未流出任何事件时退避重试

    仅限首轮整体重试：此阶段尚未 yield 任何 token/sources，重试不会事件序错乱或
    重复；第二轮失败时首轮事件已发出，无法整体重试，直接抛给上层报错。
    总调用次数上限 = 配置 chat_llm_max_attempts（含首次），默认 2 = 首次失败后重试 1 次；
    退避秒数由配置控制（默认 0.5s）。每次重试都是完整付费调用，配置请克制。
    tools 由调用方（控制层）传入，本层不持有具体工具 schema——层间依赖抽象。
    """
    max_attempts = max(1, settings.chat_llm_max_attempts)
    for attempt in range(1, max_attempts + 1):
        yielded_any = False
        try:
            for ev in stream_chat(messages, tools=tools):
                yielded_any = True
                yield ev
            return
        except Exception as e:
            if attempt >= max_attempts or yielded_any or not _is_transient_http_error(e):
                raise
            logger.warning(
                "[llm] LLM 首轮瞬时错误，退避后重试（第 %s/%s 次）: %s",
                attempt, max_attempts, e,
            )
            time.sleep(settings.chat_llm_retry_backoff_seconds)
