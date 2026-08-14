"""核心配置，从 .env 文件读取所有环境变量"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MySQL
    mysql_root_password: str = "change_me"
    mysql_database: str = "native_rag"
    mysql_host: str = "mysql"
    mysql_port: int = 3306

    # Milvus（统一语义检索 + BM25 全文检索存储）
    milvus_host: str = "milvus"
    milvus_port: int = 19530

    @property
    def milvus_uri(self) -> str:
        """构建 Milvus 连接地址"""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    # DeepSeek LLM
    deepseek_api_key: str = "sk-xxx"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-pro"

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
        return (
            f"mysql+pymysql://root:{self.mysql_root_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
