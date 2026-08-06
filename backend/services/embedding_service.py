"""Embedding 服务：BAAI/bge-large-zh-v1.5 via FastEmbed（ONNX 运行时）

选择 FastEmbed 而非 sentence-transformers：
- ONNX 运行时，不依赖 torch，避免 CUDA 大包下载（官方源 403 不可访问）
- 体积小、加载快，效果与 sentence-transformers 的 BGE 接近
"""
import logging
from functools import lru_cache

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger("native_rag")


class FastEmbedBGE(Embeddings):
    """LangChain Embeddings 封装：FastEmbed 加载 BGE 模型"""

    def __init__(self, model_name: str):
        # cache_dir 指向持久化 volume，避免容器重启后模型缓存丢失
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=settings.fastembed_cache_dir
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档片段（numpy.float32 转原生 float，兼容 ChromaDB）"""
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """向量化单个查询"""
        return [float(x) for x in next(self._model.embed([text]))]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """懒加载 embedding 模型（首次加载从 HF 镜像下载 ONNX 模型）"""
    logger.info("[embedding] 加载 FastEmbed 模型 %s", settings.embedding_model_name)
    return FastEmbedBGE(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化文档片段"""
    return get_embeddings().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """向量化用户查询"""
    return get_embeddings().embed_query(text)
