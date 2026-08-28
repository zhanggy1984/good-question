"""检索 query 清洗测试（utils/query_normalizer.py，随函数从 chat_service 迁移）

query 清洗下沉到 utils 基础设施层：全角转半角、去 emoji/客套、压冗余标点、
超长按句末标点截断。纯函数测试，不依赖外部服务。
"""
import sys

sys.path.insert(0, "/app")

from utils import query_normalizer as qn  # noqa: E402


def test_clean_query():
    """清洗 LLM 生成的 query：去空白、空回退原文、超长按句末标点截断"""
    assert qn.clean_query("  工资发放日  ", "问题原文") == "工资发放日"
    assert qn.clean_query("", "问题原文") == "问题原文"
    assert qn.clean_query("   ", "问题原文") == "问题原文"
    assert qn.clean_query(None, "问题原文") == "问题原文"
    assert qn.clean_query("真实查询", "问题原文") == "真实查询"  # 非空优先于 fallback
    # 超长含句末标点：在最后一个断点处截断，不切断完整句子
    long_with_boundary = "员工工资发放规定。" * 60  # 540 字，每 9 字一个断点
    cut = qn.clean_query(long_with_boundary, "问题原文")
    assert len(cut) <= qn._QUERY_MAX_LEN
    assert cut.endswith("。"), "应在句末标点处截断，不得切断完整句子"
    # 超长无标点连写：硬截到上限保信息量
    long_no_boundary = "查" * (qn._QUERY_MAX_LEN + 100)
    assert len(qn.clean_query(long_no_boundary, "问题原文")) == qn._QUERY_MAX_LEN
    # 规则化去噪先于截断：客套/全角数字在超长判定前已被清洗
    assert qn.clean_query("请问１２３条", "问题原文") == "123条"


def test_normalize_query():
    """规则化去噪：全角数字/字母转半角、去 emoji/客套、压冗余标点；剥空回退原文"""
    # 全角数字/英文字母 → 半角；中文标点保留（避免稀疏检索 token 错位）
    assert qn.normalize_query("员工１２３条，ＡＢＣ方案") == "员工123条，ABC方案"
    assert qn.normalize_query("第１条制度") == "第1条制度"
    # emoji / 变体选择符去除
    assert qn.normalize_query("工资几号发😀") == "工资几号发"
    assert qn.normalize_query("缺勤怎么办👍") == "缺勤怎么办"
    # 口语客套前缀（完整词锚定，不单删"请/帮"等可能为实义的单字）
    assert qn.normalize_query("请问事假提前几天申请") == "事假提前几天申请"
    assert qn.normalize_query("帮我查一下Docker常用命令") == "Docker常用命令"
    assert qn.normalize_query("麻烦你帮我看看考勤制度") == "考勤制度"
    assert qn.normalize_query("我想问下年假有几天") == "年假有几天"
    # 口语客套后缀（含前导逗号一并清理）
    assert qn.normalize_query("工资几号发，谢谢") == "工资几号发"
    assert qn.normalize_query("缺勤怎么处理辛苦啦") == "缺勤怎么处理"
    # 连续空白压缩
    assert qn.normalize_query("请假  提前  几天") == "请假 提前 几天"
    # 剥空回退原文（防空 query 拖垮召回）
    assert qn.normalize_query("请问。谢谢") == "请问。谢谢"
