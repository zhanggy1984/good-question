"""聊天服务纯函数测试（两级置信档相关，不连外部服务）

chat_service 依赖链较深（database/models），但导入仅需 sqlalchemy 等已装依赖；
pymilvus 由 conftest 条件 stub 兜底，宿主机可离线运行。
"""
import sys

sys.path.insert(0, "/app")

from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from services.chat_service import LOW_CONFIDENCE_HINT, _build_messages, _is_low_confidence


def test_build_messages_low_confidence_injects_hint():
    """低置信档：system prompt 追加 LOW_CONFIDENCE_HINT，提示 LLM 相关性存疑、不足则如实回答"""
    messages = _build_messages("摘要", [], "上下文", "问题", low_confidence=True)
    assert messages[0]["role"] == "system"
    assert LOW_CONFIDENCE_HINT in messages[0]["content"]


def test_build_messages_normal_without_hint():
    """正常档：不注入低置信提示"""
    messages = _build_messages("摘要", [], "上下文", "问题", low_confidence=False)
    assert LOW_CONFIDENCE_HINT not in messages[0]["content"]


def test_build_messages_structure():
    """messages 结构：system + history（user/assistant）+ 末尾 user 问题"""
    history = [HumanMessage(content="旧问题"), AIMessage(content="旧回答")]
    messages = _build_messages("摘要", history, "上下文", "新问题")
    assert len(messages) == 4
    assert messages[1] == {"role": "user", "content": "旧问题"}
    assert messages[2] == {"role": "assistant", "content": "旧回答"}
    assert messages[3] == {"role": "user", "content": "新问题"}


def test_is_low_confidence_boundaries():
    """置信档边界：精排最高分落在 [LOW, 低置信阈值) 判低置信，其余判否"""
    low = settings.similarity_threshold_low
    high = settings.rerank_low_confidence_threshold
    assert low < high
    assert _is_low_confidence(None) is False               # 精排失败/降级
    assert _is_low_confidence(low - 0.01) is False         # 低于 LOW：文档无关（_rerank 已返回空）
    assert _is_low_confidence(low) is True                 # 左闭：等于 LOW 判低置信
    assert _is_low_confidence((low + high) / 2) is True    # 区间中段
    assert _is_low_confidence(high) is False               # 右开：等于高阈值判正常
    assert _is_low_confidence(high + 0.1) is False         # 高于高阈值：正常
