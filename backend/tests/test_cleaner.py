"""文本预清洗纯函数测试"""
import sys
sys.path.insert(0, "/app")

from utils.text_cleaner import clean_text


def test_remove_control_chars():
    assert clean_text("a\x00b") == "ab"


def test_collapse_blank_lines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_strip_whitespace():
    assert clean_text("  hello  ") == "hello"


def test_fullwidth_space_to_half():
    assert clean_text("a　b") == "a b"


def test_trim_trailing_spaces():
    assert clean_text("hello   \nworld") == "hello\nworld"


def test_empty_text():
    assert clean_text("") == ""


def test_normal_text_unchanged():
    assert clean_text("微服务架构设计指南") == "微服务架构设计指南"


def test_fullwidth_to_halfwidth():
    assert clean_text("ＡＢＣ１２３＠") == "ABC123@"


def test_chinese_punctuation_preserved():
    # 中文全角标点应保留（NFKC 不影响中文标点）
    assert clean_text("你好，世界。") == "你好，世界。"


def test_unify_newlines():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_remove_bom():
    assert clean_text("﻿开头文本") == "开头文本"
