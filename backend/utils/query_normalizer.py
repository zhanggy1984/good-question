"""检索 query 规则化去噪与清洗（确定性，无 LLM）

从 chat_service 下沉（四层重构）：query 清洗是纯函数基础设施，归 utils 层。
只消除对稀疏检索/切词有害的确定性噪音，不改语义、不删实体——区别于 LLM 改写
（历史实测 LLM 改写收益趋零且每检索多 2-4s 延迟，见 retrieval_service.py 注释，已回滚）。
覆盖 LLM 生成的 query 与 F3 否决路径的原文 query（clean_query 统一入口）。
"""
import re

# 检索 query 清洗参数：上限（防 LLM 把整段对话/上下文照抄进 query）；BGE 模型 512 token
# 上限内 400 字安全，且给"整段制度条款引用"留足空间（200 会切断完整句子）
_QUERY_MAX_LEN = 400
# 截断时保留的最小长度：前缀内最后一个句末断点过靠前（< 此值）说明整段无标点/连写，
# 此时按断点截会丢信息，退化为硬截保信息量
_QUERY_MIN_KEEP = 64
# 句末标点（中英）与换行：超长 query 优先在此断，避免切断完整句子破坏检索语义
_QUERY_BOUNDARY_RE = re.compile(r"[。！？；.!?;\n]")

# 全角 → 半角：仅全角数字（０-９）与全角英文字母（Ａ-Ｚ ａ-ｚ）转半角。
# 不转中文标点：文档 chunk 入库保留全角标点，query 转半角标点在稀疏检索处 token 错位，
# 会丢标点侧的匹配权重（BGE-M3 稀疏对 token 精确匹配敏感）。
def _to_halfwidth(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if (0xFF10 <= code <= 0xFF19
                or 0xFF21 <= code <= 0xFF3A
                or 0xFF41 <= code <= 0xFF5A):
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)

# emoji / 杂项符号 / 变体选择符：无检索价值，只增噪音
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # 扩展象形文字/表情符号
    "\U00002600-\U000027BF"   # 杂项符号/装饰符号
    "\U0001F900-\U0001F9FF"   # 补充符号与象形文字扩展
    "\\uFE0F"            # 变体选择符
    "]+"
)

# 口语客套前缀/后缀（^/$ 锚定，完整词，不单删"请/帮"等可能为实义的单字）
# 按词长降序排列：正则 alternation 左优先，长词先匹配
_CASUAL_PREFIX_WORDS = (
    "麻烦你帮我看看", "麻烦您帮我看看", "麻烦帮我看看",
    "麻烦你帮我", "麻烦您帮我", "麻烦帮我", "麻烦问一下", "麻烦问下",
    "请问一下", "帮我查一下", "帮我看看", "帮忙查一下", "帮忙看看",
    "我想问一下", "我想问下", "想问一下", "想问下", "想咨询一下",
    "咨询一下", "帮忙查", "帮忙", "帮我查", "帮我", "请问", "麻烦",
    "我想问", "想问", "想咨询", "劳驾",
)
_CASUAL_SUFFIX_WORDS = (
    "谢谢啦", "谢谢你", "辛苦啦", "辛苦你了", "谢谢", "感谢",
    "多谢", "辛苦了", "麻烦你了", "拜托啦", "拜托了",
)
_CASUAL_PREFIX_RE = re.compile("^(?:" + "|".join(_CASUAL_PREFIX_WORDS) + ")")
_CASUAL_SUFFIX_RE = re.compile("(?:" + "|".join(_CASUAL_SUFFIX_WORDS) + ")$")

# 客套剥离后可能残留的首尾标点/空白（如"工资几号发，谢谢"剥"谢谢"后剩尾部逗号）
_EDGE_NOISE_RE = re.compile(r"^[，,。.、：:；;!！?？~·\s]+|[，,。.、：:；;!！?？~·\s]+$")
# 连续空白压缩（多个空格撑乱切词）
_COLLAPSE_WS_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """规则化去噪检索 query：全角数字/字母转半角、去 emoji/客套、压冗余标点

    只做确定性清洗，不改语义、不删实体。剥离后为空时回退原文，保证检索 query 非空
    （空 query 直接拖垮召回）。
    """
    q = _to_halfwidth(query or "")
    q = _EMOJI_RE.sub("", q)
    q = _CASUAL_PREFIX_RE.sub("", q)
    q = _CASUAL_SUFFIX_RE.sub("", q)
    q = _EDGE_NOISE_RE.sub("", q)
    q = _COLLAPSE_WS_RE.sub(" ", q).strip()
    return q or query  # 剥空回退原文，防空 query 拖垮召回


def clean_query(query: str, fallback: str) -> str:
    """清洗 LLM 生成的检索 query：规则化去噪、空回退原文、超长按句末标点截断

    LLM 可能输出空串/整段对话/超长串，脏 query 直接拖垮召回。先 normalize_query 做
    确定性去噪（全半角/emoji/客套/冗余标点），再截断。上限 _QUERY_MAX_LEN 字，超限时
    取前缀，优先在最后一个句末标点处截断（不切断完整句子）；若断点过靠前
    （< _QUERY_MIN_KEEP）则硬截，保证信息量优先。
    """
    q = (query or "").strip()
    if not q:
        q = fallback.strip()
    q = normalize_query(q)
    if len(q) <= _QUERY_MAX_LEN:
        return q
    prefix = q[:_QUERY_MAX_LEN]
    matches = list(_QUERY_BOUNDARY_RE.finditer(prefix))
    if matches and matches[-1].end() >= _QUERY_MIN_KEEP:
        return prefix[: matches[-1].end()].strip()
    return prefix
