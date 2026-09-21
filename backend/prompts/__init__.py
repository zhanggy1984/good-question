"""提示词模板集中存放与加载。

每个模板一个纯文本文件（.md），正文即全部内容——改文案不必动代码。

**与 cs/cc 的差别**：本目录 4 个模板里，2 个（chat_system / override_context）
**带 {占位符}**，由调用方 .format() 填充；另 2 个是 RETRIEVE_TOOL_SCHEMA
的中文文案，无占位符、原样使用。故 load_prompt 只负责读取、**不做任何插值**——
插值时机与参数由调用方决定，不把各自的上下文知识塞进加载器。

这 2 个带占位符的模板经 AST 校验确认：除各自那一对花括号外正文再无 { }，
故 .format() 安全、无需转义（cs 的 intent_system.md 因含 22 个字面大括号被迫改用
string.Template，gq 无此问题）。

放在 backend/ 下，随 Dockerfile 的 `COPY . .` 进镜像（WORKDIR /app），无需改 Dockerfile。
"""
from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """读取提示词模板正文。

    Args:
        name: 模板名（不含 .md 后缀），如 "chat_system"。

    Returns:
        模板正文，原样返回、不做插值；带占位符的模板由调用方自行 .format()。
    """
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")
