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


def test_heading_after_lead_paragraph_triggers_structure():
    """MinerU 常见形态：开头是导语/表格，标题在后面——也应走结构感知切分"""
    text = "本手册介绍公司薪酬制度。\n# 第一章 薪酬结构\n基本工资按月发放。"
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text(text, 1, 1, "guide.md")
        assert any(c["metadata"]["heading_path"] for c in chunks), \
            "标题在中间的文档应走结构切分，而不是退化为无 heading_path"


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


def test_section_page_range_cross_page():
    """跨页 section：heading_stack 全局延续（跨页二级 level=2）、page_range 覆盖、标记剥离"""
    md = (
        "@@PAGE:1@@\n# 第一章\n第一页内容\n"
        "@@PAGE:2@@\n第二页内容\n"
        "@@PAGE:3@@\n## 二级\n第三页内容"
    )
    sections = chunker._split_markdown_sections(md)
    assert len(sections) == 2
    assert sections[0]["heading_path"] == ["第一章"]
    assert sections[0]["page_range"] == [1, 2]  # 正文跨页，页范围扩大
    assert sections[1]["heading_path"] == ["第一章", "二级"]
    assert sections[1]["heading_level"] == 2  # 上一页的一级标题仍是祖先
    assert sections[1]["page_range"] == [3, 3]
    assert "@@PAGE" not in sections[0]["content"]
    assert "@@PAGE" not in sections[1]["content"]


def test_chunk_text_page_markers():
    """跨页 section 合并成单 chunk（语义连续优先），page_range 覆盖两页、标记不残留"""
    text = (
        "@@PAGE:1@@\n# 第一章\n第一页的正文内容。\n"
        "@@PAGE:2@@\n第二页的正文内容。"
    )
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text(text, 1, 1, "doc.pdf")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_range"] == [1, 2]
    assert "第一页的正文内容。" in chunks[0]["content"]
    assert "第二页的正文内容。" in chunks[0]["content"]
    assert "@@PAGE" not in chunks[0]["content"], "页标记不得残留进 content"


def test_plain_text_with_page_markers():
    """无标题但含页标记（纯文本型 PDF）：单 section、标记剥离、page_range 正确"""
    text = "@@PAGE:1@@\n第一页内容\n@@PAGE:2@@\n第二页内容"
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text(text, 1, 1, "doc.pdf")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_range"] == [1, 2]
    assert chunks[0]["metadata"]["heading_path"] == []
    assert "@@PAGE" not in chunks[0]["content"]


def test_plain_text_page_range_zero():
    """无页标记（TXT/MD/DOCX）→ page_range=[0,0]，与旧 page_number=0 语义一致"""
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text("无标题文本", 1, 1, "a.txt")
    assert chunks[0]["metadata"]["page_range"] == [0, 0]


def test_overlap_prev_chunk_index():
    """同 section 内非首 chunk 填前一个 chunk 的 index；首个 chunk 为 None"""
    long_text = "这是一段很长的正文。" * 300  # > 1024 token，单 section 切多 chunk
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text(long_text, 1, 1, "a.txt")
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["overlap_prev_chunk_index"] is None
    for i in range(1, len(chunks)):
        assert chunks[i]["metadata"]["overlap_prev_chunk_index"] == i - 1


def test_overlap_prev_chunk_index_cross_section():
    """跨 section 首 chunk 无 overlap（各 section 独立切分）"""
    text = "# 第一章\n" + "第一章正文。" * 200 + "\n# 第二章\n" + "第二章正文。" * 200
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text(text, 1, 1, "a.md")
    sec2_start = next(
        i for i, c in enumerate(chunks) if c["metadata"]["heading_path"] == ["第二章"]
    )
    assert sec2_start > 0
    assert chunks[1]["metadata"]["overlap_prev_chunk_index"] == 0  # 第一章内重叠
    assert chunks[sec2_start]["metadata"]["overlap_prev_chunk_index"] is None  # 跨 section


def test_heading_level():
    """heading_level：一级标题=1、嵌套二级=2"""
    md = "# 一级\n内容一\n## 二级\n内容二"
    sections = chunker._split_markdown_sections(md)
    assert sections[0]["heading_level"] == 1
    assert sections[1]["heading_level"] == 2


def test_plain_text_heading_level_zero():
    """纯文本无标题 → heading_level=0"""
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        chunks = chunker.chunk_text("无标题文本", 1, 1, "a.txt")
    assert chunks[0]["metadata"]["heading_level"] == 0


def test_content_hash():
    """content_hash：同 content 稳定、长度 8、不同 content 不同"""
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        h1 = chunker.chunk_text("相同的文本内容。", 1, 1, "a.txt")[0]["metadata"]["content_hash"]
        h2 = chunker.chunk_text("相同的文本内容。", 1, 1, "a.txt")[0]["metadata"]["content_hash"]
        h3 = chunker.chunk_text("不同的文本内容。", 1, 1, "a.txt")[0]["metadata"]["content_hash"]
    assert h1 == h2
    assert len(h1) == 8
    assert h1 != h3


def test_splitter_field():
    """splitter：结构切分=heading_aware，纯文本=sentence_splitter"""
    with patch.object(chunker, "_get_tokenizer", return_value=FakeTokenizer()):
        struct = chunker.chunk_text("# 标题\n正文内容", 1, 1, "a.md")
        plain = chunker.chunk_text("无标题的普通文本", 1, 1, "a.txt")
    assert struct[0]["metadata"]["splitter"] == "heading_aware"
    assert plain[0]["metadata"]["splitter"] == "sentence_splitter"
