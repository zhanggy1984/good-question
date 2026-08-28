"""后端四层架构依赖守护：控制层/能力层/资源层单向依赖 + 反向禁止

分层按"业务语义"切（用户拍板）：
- 控制层：chat_service（编排：意图理解→会话状态→SSE 事件流组装）
- 能力层：document/library/auth/dashboard/retrieval（有业务语义的操作）
- 资源层：llm/embedding/rerank/vector_store/llama_store/chat_cache/chat_cleanup/retrieval_types
  （无业务语义的基础设施，向外部库/存储/模型抽象）

守护规则：
1. 任何 services 层不得 import api.*（下层绝不依赖上层）
2. 资源层不得 import 控制层/能力层（只允许依赖资源层自身 + models/utils/config 等基础设施）
3. 能力层不得 import 控制层（可依赖资源层）
"""
import ast
from pathlib import Path

SERVICES_DIR = Path(__file__).resolve().parent.parent / "services"

# 层归属（按业务语义：有无业务语义）
CONTROL = {"chat_service"}
CAPABILITY = {
    "document_service", "library_service", "auth_service",
    "dashboard_service", "retrieval_service",
}
RESOURCE = {
    "llm_service", "embedding_service", "rerank", "vector_store_service",
    "llama_store", "chat_cache", "chat_cleanup", "retrieval_types",
}
ALL_SERVICES = CONTROL | CAPABILITY | RESOURCE


def _service_files():
    for f in sorted(SERVICES_DIR.glob("*.py")):
        if f.name != "__init__.py":
            yield f


def _imported_service_modules(path: Path) -> list[str]:
    """提取文件里 import 的 services 内部模块名（from services import X / from services.X import Y）"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "services":
                if len(parts) >= 2:
                    found.append(parts[1])
                else:
                    found.extend(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")
                if root[0] == "services" and len(root) >= 2:
                    found.append(root[1])
    return found


def _imports_api(path: Path) -> bool:
    """是否 import 了 api 层（反向依赖信号）"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "api" or a.name.startswith("api.") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "api" or node.module.startswith("api."):
                return True
    return False


def test_services_never_import_api():
    """反向依赖禁止：services 层不得 import api.*"""
    offenders = [f.name for f in _service_files() if _imports_api(f)]
    assert not offenders, f"services 层反向依赖 api 层: {offenders}"


def test_resource_layer_does_not_import_upper_layers():
    """资源层只能依赖资源层自身 + 基础设施，不得 import 控制层/能力层"""
    upper = CONTROL | CAPABILITY
    offenders = []
    for name in sorted(RESOURCE):
        f = SERVICES_DIR / f"{name}.py"
        if not f.exists():
            continue
        bad = [m for m in _imported_service_modules(f) if m in upper]
        if bad:
            offenders.append(f"{name} -> {bad}")
    assert not offenders, f"资源层 import 上层: {offenders}"


def test_capability_layer_does_not_import_control_layer():
    """能力层可以依赖资源层，但不得 import 控制层"""
    offenders = []
    for name in sorted(CAPABILITY):
        f = SERVICES_DIR / f"{name}.py"
        if not f.exists():
            continue
        bad = [m for m in _imported_service_modules(f) if m in CONTROL]
        if bad:
            offenders.append(f"{name} -> {bad}")
    assert not offenders, f"能力层 import 控制层: {offenders}"


def test_all_services_modules_accounted():
    """services/ 下所有业务模块都被分到某层（新服务未归层会在此暴露，强制补归）"""
    existing = {f.stem for f in _service_files()}
    unassigned = existing - ALL_SERVICES
    assert not unassigned, f"services 模块未归层: {unassigned}"
