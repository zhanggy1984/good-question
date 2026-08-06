"""文本预清洗与内容规范化

在切片/向量化前对抽取的文本做规范化处理，提升后续检索与向量化质量：
- 去 BOM、统一换行符
- 全角英文字母/数字/符号转半角（NFKC 规范化）
- 移除控制字符
- 压缩空白与空行
"""
import re
import unicodedata


def clean_text(text: str) -> str:
    """预清洗：规范化文本，保持段落结构（\n）"""
    if not text:
        return ""

    # 1. 去 BOM
    text = text.lstrip("﻿")

    # 2. 统一换行符（\r\n / \r → \n）
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. 全角字符转半角（NFKC：ＡＢＣ→ABC、１２３→123、＠→@）
    #    注意：NFKC 会把中文标点（，。！？等全角形式）一并转半角，破坏中文排版，
    #    故跳过中文标点集合，只对字母/数字/符号规范化。
    _CHINESE_PUNCT = set("，。！？；：、（）《》「」『』【】“”‘’—…")
    text = "".join(
        unicodedata.normalize("NFKC", ch) if ch not in _CHINESE_PUNCT else ch
        for ch in text
    )

    # 4. 移除控制字符（保留换行和制表符）
    text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")

    # 5. 全角空格转半角（NFKC 已处理，这里兜底）
    text = text.replace("　", " ")

    # 6. 压缩连续空行为最多两个换行（保留段落分隔）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. 行尾多余空白
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)

    return text.strip()
