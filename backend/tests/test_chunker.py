"""切片器测试（mock tokenizer，不下载模型）"""
import sys
from unittest.mock import patch

sys.path.insert(0, "/app")

from utils import chunker


class FakeTokenizer:
    """假 tokenizer：每字符算 1 个 token"""

    def encode(self, text: str):
        return list(text)


def test_chunk_text_structure():
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text(
            "微服务架构是一种将应用程序划分为小型独立服务的设计模式。", 1, 1, "test.md"
        )
        assert len(chunks) >= 1
        meta = chunks[0]["metadata"]
        assert meta["document_id"] == 1
        assert meta["library_id"] == 1
        assert meta["document_name"] == "test.md"
        assert meta["chunk_index"] == 0
        assert meta["total_chunks"] == len(chunks)


def test_markdown_heading_splits():
    md = "# 第一章\n内容一\n## 二级标题\n内容二"
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        sections = chunker._split_markdown_sections(md)
        assert len(sections) == 2
        assert sections[0]["heading_path"] == ["第一章"]
        assert sections[1]["heading_path"] == ["第一章", "二级标题"]


def test_plain_text_heading_empty():
    text = "没有标题的普通文本段落"
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text(text, 1, 1, "a.txt")
        assert chunks[0]["metadata"]["heading_path"] == []


def test_max_chunks_truncation(monkeypatch):
    """超 MAX_CHUNKS 上限时截断保留前 N 个，且 chunk 序号连续、total_chunks 一致"""
    monkeypatch.setattr(chunker, "MAX_CHUNKS", 10)
    # 含句号分隔的文本，按 1024 token/chunk 会切出远超 10 个 chunk
    big_text = "段落。内容。" * 5000
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text(big_text, 1, 1, "big.txt")
        assert len(chunks) == 10
        metas = [c["metadata"] for c in chunks]
        assert [m["chunk_index"] for m in metas] == list(range(10))
        assert metas[-1]["total_chunks"] == 10
        # 截断后内容首尾应保留前段内容
        assert chunks[0]["content"].startswith("段落。")
