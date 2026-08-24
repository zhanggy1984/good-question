"""检索结果接缝类型：LlamaIndex（Node）与手写编排（SSE/置信档）之间的唯一契约

迁移后检索返回 `list[RetrievedChunk]` 而非 LangChain Document，避免 LlamaIndex 类型
泄漏到 chat/SSE 层。字段对齐消费侧读法（`.content`/`.metadata`），score 承载精排分。
"""
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """一次检索精排后的结果片段

    - content：片段文本（现 Document.page_content 的读法迁移到 .content）
    - metadata：溯源信息（document_name / heading_path / chunk_index / total_chunks / library_id 等）
    - score：本片段的 rerank 精排原始分；None 表示精排失败走降级
    """

    content: str
    metadata: dict = field(default_factory=dict)
    score: float | None = None
