"""智能切片：结构感知 + LlamaIndex SentenceSplitter

- chunk 大小与重叠 token 数按文档维度配置（默认 1024 / 102），上传时前端可设置
- Markdown 按标题层级分节，chunk 带 heading_path 元数据（溯源）
- 其他格式直接切分
"""
import logging
import re
from functools import lru_cache

from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer

from config import settings

logger = logging.getLogger("native_rag")

# 单文档 chunk 数上限：防止极端大文件（扫描件/超大 PDF）拖垮向量化与存储
MAX_CHUNKS = 2000


@lru_cache(maxsize=1)
def _get_tokenizer():
    """BGE 模型 tokenizer（用于精确 token 计数），模块级缓存"""
    return AutoTokenizer.from_pretrained(settings.embedding_model_name)


def _count_tokens(text: str) -> int:
    """按 BGE tokenizer 统计 token 数"""
    return len(_get_tokenizer().encode(text))


def _split_markdown_sections(md_text: str) -> list[dict]:
    """按标题层级切分 Markdown，返回 [{heading_path, content}]"""
    sections = []
    heading_stack: list[tuple[int, str]] = []
    current: dict | None = None

    for line in md_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # 维护标题栈：同层或更高层标题出栈
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            if current is not None:
                sections.append(current)
            current = {"heading_path": [t for _, t in heading_stack], "content": ""}
        else:
            if current is None:
                current = {"heading_path": [], "content": ""}
            current["content"] += line + "\n"

    if current is not None:
        sections.append(current)
    return sections


def _infer_source_type(content: str) -> str:
    """粗判内容类型，用于溯源展示"""
    stripped = content.strip()
    if stripped.startswith(("|", "+")) and "|" in stripped:
        return "table"
    if any(l.lstrip().startswith(("- ", "* ", "1. ")) for l in stripped.splitlines()[:5]):
        return "list"
    if stripped.startswith(("def ", "class ", "import ", "```")):
        return "code"
    return "paragraph"


def chunk_text(
    text: str,
    document_id: int,
    library_id: int,
    document_name: str,
    chunk_size: int = 1024,
    overlap_token: int = 102,
) -> list[dict]:
    """将清洗后全文切片，返回 chunk 元信息列表

    返回每个 chunk: {content, metadata}
    metadata 含 document_id/document_name/library_id/chunk_index/total_chunks/
    heading_path/source_type/token_count 等，同时用于 MySQL 和 Milvus。
    """
    # 换 LlamaIndex SentenceSplitter（替换 RecursiveCharacterTextSplitter）：
    # 主分隔符取段落（\n\n），超长段内再按中文句末标点切（secondary_chunking_regex）。
    # 注意：SentenceSplitter 内部用 re.findall 提取匹配块（非 split 切分），正则必须是
    # 「非分隔符块 + 可选分隔符」形态（默认 [^,\.;]+[,\.;]? 即此语义）；若只写 [。！？；\n]
    # 会 findall 出纯标点、chunk 内容丢失。tokenizer 契约是 Callable[[str], list]
    # （SentenceSplitter._token_size 对返回值做 len()），返回 token ids 而非 int。
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_token,
        separator="\n\n",
        secondary_chunking_regex=r"[^。！？；\n]+[。！？；]?",
        tokenizer=lambda t: _get_tokenizer().encode(t),
    )

    results: list[dict] = []

    # Markdown 按标题分节，否则整篇递归切
    sections = _split_markdown_sections(text) if text.lstrip().startswith("#") else [
        {"heading_path": [], "content": text}
    ]

    for section in sections:
        chunks = splitter.split_text(section["content"])
        for content in chunks:
            content = content.strip()
            if not content:
                continue
            results.append({
                "content": content,
                "metadata": {
                    "document_id": document_id,
                    "document_name": document_name,
                    "library_id": library_id,
                    "heading_path": section["heading_path"],
                    "source_type": _infer_source_type(content),
                    "token_count": _count_tokens(content),
                },
            })

    # 上限保护：超限截断（保留前 MAX_CHUNKS 个），防止极端大文件拖垮后续向量化/存储
    if len(results) > MAX_CHUNKS:
        logger.warning(
            "[chunker] 文档 %s 切片 %s 个 chunk 超上限 %s，截断保留前 %s 个",
            document_name, len(results), MAX_CHUNKS, MAX_CHUNKS,
        )
        results = results[:MAX_CHUNKS]

    # 补充 chunk 序号信息
    total = len(results)
    for i, r in enumerate(results):
        r["metadata"]["chunk_index"] = i
        r["metadata"]["total_chunks"] = total

    logger.info(
        "[chunker] 文档 %s 切片完成 %s 个 chunk（总 token 约 %s）",
        document_name, len(results), sum(r["metadata"]["token_count"] for r in results),
    )
    return results
