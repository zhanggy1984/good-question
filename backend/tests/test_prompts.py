"""提示词外置回归测试（prompts/*.md 与代码的绑定守卫）。

为什么需要：本仓 4 个提示词模板已从 .py 内联字符串迁到 backend/prompts/*.md，
chat_service.py 在 import 期用 load_prompt() 绑成模块级常量。外置的红利是
「改文案不动代码」，代价是**多了一处会静默失效的接缝**：有人改一句话时把正文
拷回 .py、或模板文件被清空/改名，代码照样跑，只是外置失效——没有任何测试会红。
本文件守这条接缝。（4 个 agent 仓里，cs 有 test_prompt_guard.py，本仓原先缺。）

第二类断言守的是 `.format()` 的**前提**：prompts/__init__.py 声明「两个带占位符
的模板除各自那一对花括号外正文再无 { }，故 .format() 安全」。这个前提一旦被破坏
（比如正文里加了个 JSON 示例 `{"a": 1}`），.format() 会抛 KeyError/IndexError，
或在恰好同名时静默替换坏输出——都不是测试能自动发现的。故把前提本身写成断言。
"""
import sys
from pathlib import Path

# 兼容容器（/app，扁平布局）与宿主机（backend/）两种布局：父目录插 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from prompts import load_prompt
from services import chat_service as cs

# 模板名 → 该模板正文里唯一的占位符名（None = 无占位符，原样使用）
_PLACEHOLDER_TEMPLATES = {
    "chat_system": "summary",
    "override_context": "context",
}
_PLAIN_TEMPLATES = ("retrieve_tool_description", "retrieve_query_description")


def _prompt_file(name: str) -> Path:
    """模板文件路径：以 prompts 包的实际位置为准，不手写相对路径。"""
    import prompts

    return Path(prompts.__file__).parent / f"{name}.md"


@pytest.mark.parametrize("name", sorted(_PLACEHOLDER_TEMPLATES) + list(_PLAIN_TEMPLATES))
def test_prompt_file_non_empty(name):
    assert load_prompt(name).strip(), f"{name}.md 为空——文案被清空了"


@pytest.mark.parametrize("name", sorted(_PLACEHOLDER_TEMPLATES) + list(_PLAIN_TEMPLATES))
def test_load_prompt_reads_verbatim(name):
    """load_prompt 只读取、不插值：正文须与文件逐字一致。"""
    assert load_prompt(name) == _prompt_file(name).read_text(encoding="utf-8")


def test_module_constants_bound_to_files():
    """模块级常量必须来自模板文件，而不是被重新内联的字面量。

    这条是外置本身的自证：若有人把文案拷回 chat_service.py 写成字面量，
    .md 就成了死文件（改文案又不动代码了），此断言转红。
    """
    assert cs.SYSTEM_PROMPT == _prompt_file("chat_system").read_text(encoding="utf-8")
    assert cs._OVERRIDE_CONTEXT_PROMPT == _prompt_file("override_context").read_text(encoding="utf-8")
    assert cs.RETRIEVE_TOOL_SCHEMA["function"]["description"] == load_prompt("retrieve_tool_description")
    assert (
        cs.RETRIEVE_TOOL_SCHEMA["function"]["parameters"]["properties"]["query"]["description"]
        == load_prompt("retrieve_query_description")
    )


@pytest.mark.parametrize("name,placeholder", sorted(_PLACEHOLDER_TEMPLATES.items()))
def test_placeholder_template_has_no_stray_braces(name, placeholder):
    """除唯一占位符外，正文不得再有花括号——这是 .format() 安全的前提。"""
    body = load_prompt(name)
    assert body.count("{" + placeholder + "}") == 1, f"{name}.md 应恰有一个 {{{placeholder}}}"
    stripped = body.replace("{" + placeholder + "}", "")
    assert "{" not in stripped and "}" not in stripped, (
        f"{name}.md 正文含额外花括号，.format() 会炸或静默替换坏输出"
    )


@pytest.mark.parametrize("name,placeholder", sorted(_PLACEHOLDER_TEMPLATES.items()))
def test_placeholder_template_formats(name, placeholder):
    """占位符能被填充，且填充后占位符消失、正文其余部分不变。"""
    body = load_prompt(name)
    filled = body.format(**{placeholder: "SENTINEL-VALUE"})
    assert "SENTINEL-VALUE" in filled
    assert "{" + placeholder + "}" not in filled
    assert filled.replace("SENTINEL-VALUE", "{" + placeholder + "}") == body


def test_plain_templates_need_no_formatting():
    """无占位符的两个模板不含花括号，调用方原样使用（不经 .format()）。"""
    for name in _PLAIN_TEMPLATES:
        body = load_prompt(name)
        assert "{" not in body and "}" not in body, f"{name}.md 无占位符，不应含花括号"


def test_tool_schema_names_unchanged():
    """工具名是 LLM function calling 的协议面，改它对模型是破坏性变更，单列一条守。"""
    assert cs.RETRIEVE_TOOL_SCHEMA["function"]["name"] == "hybrid_retrieve"
