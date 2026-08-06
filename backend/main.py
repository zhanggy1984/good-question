"""Native RAG - FastAPI 应用入口"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from database import SessionLocal
from services import auth_service
from utils.exceptions import AppError

# 日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("native_rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化资源，关闭时清理"""
    logger.info("[lifespan] Native RAG 启动中...")
    logger.info("[lifespan] DeepSeek 模型: %s", settings.deepseek_model)

    # 初始化预设 admin 账号
    db = SessionLocal()
    try:
        auth_service.seed_admin(db)
    finally:
        db.close()

    # 预热模型（embedding/rerank），避免首次提问等待模型加载
    try:
        from services.embedding_service import get_embeddings
        from services.retrieval_service import _get_reranker
        get_embeddings()
        _get_reranker()
        logger.info("[lifespan] 模型预热完成")
    except Exception as e:
        logger.warning("[lifespan] 模型预热失败（不影响启动）: %s", e)

    yield
    logger.info("[lifespan] Native RAG 关闭")


app = FastAPI(
    title="Native RAG",
    description="基于 RAG 的文档问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件（开发环境前后端分离时需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "service": "Native RAG"}
