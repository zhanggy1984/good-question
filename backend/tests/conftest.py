"""pytest 全局夹具：宿主机未装 pymilvus 时注入同名 stub，保证测试可离线收集/运行。

容器内有真实 pymilvus，这里仅当 import 不到时才注入假模块（不覆盖真实实现）。
只补被 retrieval/chat 模块 import 的 6 个名字，跑不了真实 Milvus 调用——涉及
真实查询的用例仍需在容器内（docker exec rag-backend）执行。
"""
import sys
import types


def _install_pymilvus_stub() -> None:
    """构造器接受任意参数（AnnSearchRequest 在 try 块外构造，构造不抛错才能测到降级逻辑）"""
    _mod = types.ModuleType("pymilvus")

    def _stub_init(self, *args, **kwargs):  # noqa: ANN001
        pass

    for _name in (
        "AnnSearchRequest",
        "DataType",
        "Function",
        "FunctionType",
        "MilvusClient",
        "RRFRanker",
    ):
        setattr(_mod, _name, type(_name, (), {"__init__": _stub_init}))
    sys.modules["pymilvus"] = _mod


try:
    import pymilvus  # noqa: F401
except ImportError:
    _install_pymilvus_stub()
