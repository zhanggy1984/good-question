"""LLM 服务：DeepSeek（OpenAI 兼容协议）

通过 langchain-openai 的 ChatOpenAI 封装，模型名从 .env 读取，可切换。
"""
import logging
from functools import lru_cache

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
