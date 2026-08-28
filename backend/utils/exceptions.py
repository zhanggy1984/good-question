"""统一业务异常与错误码

约定错误响应格式：
    {"error": {"code": "NOT_FOUND", "message": "资源不存在"}}
"""
from typing import Optional


class AppError(Exception):
    """业务异常基类：携带 HTTP 状态码与错误码"""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: Optional[str] = None):
        self.message = message or ""
        super().__init__(self.message)


class UnauthorizedError(AppError):
    """未认证（401）"""
    status_code = 401
    code = "UNAUTHORIZED"


class ForbiddenError(AppError):
    """无权限（403）"""
    status_code = 403
    code = "FORBIDDEN"


class NotFoundError(AppError):
    """资源不存在（404）"""
    status_code = 404
    code = "NOT_FOUND"


class ConflictError(AppError):
    """资源冲突（409），如用户名已存在"""
    status_code = 409
    code = "CONFLICT"


class InvalidCredentialsError(AppError):
    """用户名或密码错误（401）"""
    status_code = 401
    code = "INVALID_CREDENTIALS"


class ValidationError(AppError):
    """业务参数校验失败（400），区别于 FastAPI 参数校验 422"""
    status_code = 400
    code = "VALIDATION_ERROR"


class TooManyRequestsError(AppError):
    """请求过于频繁（429），如登录防爆破锁定"""
    status_code = 429
    code = "TOO_MANY_REQUESTS"
