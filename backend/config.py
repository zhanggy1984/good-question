"""核心配置，从 .env 文件读取所有环境变量"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MySQL（共享 infra：用独立账号 native_rag_user，避免 root 跨库权限）
    mysql_root_password: str = "change_me"   # 兼容保留，不再用于连接
    mysql_user: str = "native_rag_user"
    mysql_password: str = ""                 # 仅由环境变量 MYSQL_PASSWORD 注入（共享 infra 凭据不落代码）
    mysql_database: str = "native_rag"
    mysql_host: str = "mysql"
    mysql_port: int = 3306

    # Milvus（统一语义检索 + BGE-M3 稀疏混合检索存储）
    milvus_host: str = "milvus"
    milvus_port: int = 19530

    @property
    def milvus_uri(self) -> str:
        """构建 Milvus 连接地址"""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    # DeepSeek LLM
    deepseek_api_key: str = "sk-xxx"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # 聊天问答缓存（Redis）：相同问题 + 空上下文（新会话首句）命中时重放 SSE，省 DeepSeek 计费。
    # 文档更新后由 document_service 上传成功时清库缓存，TTL 兜底（默认 24h）。
    redis_url: str = "redis://redis:6379/2"   # 共享 Redis db index 2（隔离规范）
    chat_cache_enabled: bool = True
    chat_cache_ttl_seconds: int = 86400

    # Embedding（默认与 .env.example 一致；换模型须保证向量维度与 Milvus collection 一致）
    embedding_model_name: str = "jinaai/jina-embeddings-v2-base-zh"
    embedding_device: str = "cpu"
    # FastEmbed 模型缓存目录（挂载 volume 持久化，避免容器重启丢模型）
    fastembed_cache_dir: str = "/root/.cache/fastembed"

    # Rerank
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cpu"
    # 无结果判定阈值：rerank 最高分低于此值才判定"文档无关"（返回空，不发 sources）
    # 绝对阈值经实测不可靠（相关 chunk 可能低分），改为相对排序 + 此低阈值兜底
    similarity_threshold_low: float = 0.20
    # 低置信判定阈值：最高分落在 [similarity_threshold_low, 此值) 视为"低置信"——
    # 检索结果照常返回（保召回，避免重蹈绝对阈值误杀覆辙），但 chat 层会提示 LLM
    # 相关性存疑、不足以回答则如实说"未找到"，防止基于边缘相关片段编造答案
    rerank_low_confidence_threshold: float = 0.50

    # F3 规则否决权开关：LLM 决定不检索但规则判该查（query/unknown）时强制检索，防直接编造。
    # 可设 RULE_OVERRIDE_ENABLED=false 临时关闭回滚到"信任 LLM"（二期行为）
    rule_override_enabled: bool = True

    # Admin
    admin_username: str = "admin"
    admin_password: str = "admin123"

    # JWT
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # 文件上传
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50

    # MinerU：留空用本地，填入 Token 走官方 API
    mineru_api_token: str = ""

    @property
    def database_url(self) -> str:
        """构建 SQLAlchemy 连接字符串"""
        if not self.mysql_password:
            raise RuntimeError(
                "缺少 MYSQL_PASSWORD 环境变量（共享 infra MySQL 密码，见 .env.example）"
            )
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
