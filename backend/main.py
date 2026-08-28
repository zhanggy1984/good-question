"""Native RAG - FastAPI 应用入口"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from database import SessionLocal
from services import auth_service
from utils.exceptions import AppError
from utils.trace import install, trace_id_var

# 日志配置（[%(trace_id)s]：链路追踪，值来自中间件注入的 X-Request-ID，见 utils/trace.py）
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: [%(trace_id)s] %(message)s",
)
install()
logger = logging.getLogger("native_rag")


# 常见弱/占位密钥：命中即禁止启动（安全配置必须显式写入强随机值）
_WEAK_SECRETS = {
    "", "change-me", "change_me", "admin123", "generate_a_random_string_here",
    "sk-xxx", "sk-your-api-key-here", "change_me_admin",
}
# 模板残留特征（子串命中即弱）：覆盖 "xxx-change-in-production" / "xxx-your-api-key" 等历史模板值。
# 刻意不用 secret-key/secret_key 等过泛词——强随机密钥名可能天然含 secret，避免误杀；
# 历史模板值 native-rag-jwt-secret-key-change-in-production 已被 change-in-production 子串覆盖。
_WEAK_PATTERNS = (
    "change-in-production", "change_in_production",
    "your-api-key", "generate-a-random", "generate_a_random",
)


def _is_weak(secret: str) -> bool:
    """是否弱/占位密钥：精确命中弱值集合，或含模板残留特征子串（大小写不敏感）"""
    val = (secret or "").strip().lower()
    if val in _WEAK_SECRETS:
        return True
    return any(p in val for p in _WEAK_PATTERNS)


def _validate_secrets() -> None:
    """启动前校验安全配置：JWT/Admin 密钥禁占位默认值（fail-fast，防缺配裸奔）。

    放 lifespan 而非 Settings 构造：不破坏"无 .env 也能 import"的测试/工具链路，
    服务真实启动必经此校验，fail-fast 语义在启动边界达成。
    """
    for name, val in (
        ("JWT_SECRET_KEY", settings.jwt_secret_key),
        ("ADMIN_PASSWORD", settings.admin_password),
    ):
        if _is_weak(val):
            raise RuntimeError(f"{name} 未配置或为默认占位值，禁止启动（请在 .env 写入强随机值）")
    if not settings.deepseek_api_key:
        logger.warning("DEEPSEEK_API_KEY 未配置，聊天功能不可用（服务仍可启动）")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化资源，关闭时清理"""
    logger.info("[lifespan] Native RAG 启动中...")
    logger.info("[lifespan] DeepSeek 模型: %s", settings.deepseek_model)

    # 安全配置 fail-fast：JWT/Admin 密钥为空或占位值禁止启动（防缺配裸奔）；
    # DeepSeek key 空仅告警：聊天不可用但服务可启动，兼容离线/测试场景。
    _validate_secrets()

    # 初始化预设 admin 账号
    db = SessionLocal()
    try:
        auth_service.seed_admin(db)
    finally:
        db.close()

    # 预热模型（embedding/rerank），避免首次提问等待模型加载
    try:
        from services.embedding_service import get_embeddings
        from services.rerank import _get_reranker
        get_embeddings()
        _get_reranker()
        logger.info("[lifespan] 模型预热完成")
    except Exception as e:
        logger.warning("[lifespan] 模型预热失败（不影响启动）: %s", e)

    # 加载 Milvus collection（rag_chunks）：Milvus 重启后 collection 不自动 load，
    # 不 load 检索会报错。维度校验失败（换 embedding 模型）强制停机提示重灌——
    # 维度不匹配到插入/检索时才炸难以排查，必须 fail-fast；其他失败仅告警，不影响启动。
    try:
        from services.llama_store import (
            EmbeddingDimensionMismatchError,
            ensure_dimension_match,
            ensure_loaded,
        )
        ensure_dimension_match()
        ensure_loaded()
    except EmbeddingDimensionMismatchError as e:
        logger.error("[lifespan] %s", e)
        raise  # 维度不匹配必须停机重灌，不静默降级
    except Exception as e:
        logger.warning("[lifespan] Milvus collection 加载失败（不影响启动）: %s", e)

    # 会话过期清理：定时 sweep 随应用启停（受开关控制，异常可一键停）
    from services.chat_cleanup import SessionCleaner
    chat_cleaner = SessionCleaner()
    if settings.chat_cleanup_enabled:
        chat_cleaner.start()
        logger.info(
            "[lifespan] 会话过期清理已启动（保留 %s 天，间隔 %ss）",
            settings.chat_retention_days,
            settings.chat_cleanup_interval_seconds,
        )

    yield
    if settings.chat_cleanup_enabled:
        await chat_cleaner.stop()
    logger.info("[lifespan] Native RAG 关闭")


app = FastAPI(
    title="Native RAG",
    description="基于 RAG 的文档问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 白名单（settings.cors_origins）：前端经 nginx 同源反代本不需 CORS，白名单仅兜底
# 直连开发；* 通配 + credentials 是无效组合，且生产收紧是安全基线
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """链路追踪：取网关透传的 X-Request-ID（无则生成 uuid），写入 contextvar 供日志
    filter 使用，并在响应头回传（经网关时网关会隐藏后端重复头，无副作用）。"""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    trace_id_var.set(rid)
    response = await call_next(request)
    response.headers.setdefault("X-Request-ID", rid)
    return response


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """安全响应头：nosniff / X-Frame-Options / Referrer-Policy / CSP。

    放后端不放 nginx：8080 与 8089 均映射到宿主，后端一处覆盖两个入口，避免重复头。
    CSP 需前端 build 产物实测，若控制台报 blocked 再放宽对应指令（勿删整条 CSP）。
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; font-src 'self' data:; connect-src 'self'",
    )
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """统一业务异常处理：返回 {error: {code, message}}"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """参数校验异常：统一为 {error: {code, message}} 格式"""
    first_error = exc.errors()[0] if exc.errors() else {}
    message = first_error.get("msg", "参数校验失败")
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", "message": message}},
    )


# 注册 API 路由
from api import auth as auth_api  # noqa: E402
from api import chat as chat_api  # noqa: E402
from api import contracts as contracts_api  # noqa: E402
from api import dashboard as dashboard_api  # noqa: E402
from api import document as document_api  # noqa: E402
from api import library as library_api  # noqa: E402
from api import session as session_api  # noqa: E402

app.include_router(auth_api.router, prefix="/api/auth", tags=["auth"])
app.include_router(dashboard_api.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(library_api.router, prefix="/api/libraries", tags=["libraries"])
app.include_router(document_api.router, prefix="/api", tags=["documents"])
app.include_router(session_api.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(chat_api.router, prefix="/api", tags=["chat"])
app.include_router(contracts_api.router, prefix="/api", tags=["contracts"])


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "service": "Native RAG"}
