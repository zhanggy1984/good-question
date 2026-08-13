"""Embedding 服务：jinaai/jina-embeddings-v2-base-zh via FastEmbed（ONNX 运行时）

选择 FastEmbed 而非 sentence-transformers：
- ONNX 运行时，不依赖 torch，避免 CUDA 大包下载（官方源 403 不可访问）
- 体积小、加载快，效果与 sentence-transformers 的 BGE 接近
"""
import logging
from functools import lru_cache

import numpy as np
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
        """批量向量化文档片段（numpy.float32 转原生 float，兼容 Milvus）"""
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """向量化单个查询"""
        return [float(x) for x in next(self._model.embed([text]))]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """懒加载 embedding 模型（首次加载从 HF 镜像下载 ONNX 模型）"""
    logger.info("[embedding] 加载 FastEmbed 模型 %s", settings.embedding_model_name)
    return FastEmbedBGE(settings.embedding_model_name)


def _l2_normalize(vectors):
    """L2 归一化向量（单条或批量）

    Milvus dense index 用 IP 内积，前提是向量已归一化（归一化后 IP == COSINE，
    排序不变）。同时消除未归一化时内积对长文本（大模长）的偏差。
    归一化在公共出口统一做，保证写入（embed_texts）与检索（embed_query）两端一致。
    """
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # 零向量兜底，避免除零
    return (arr / norms).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化文档片段（输出已 L2 归一化，供 Milvus IP 检索）"""
    return _l2_normalize(get_embeddings().embed_documents(texts))


def embed_query(text: str) -> list[float]:
    """向量化用户查询（输出已 L2 归一化，供 Milvus IP 检索）"""
    return _l2_normalize(get_embeddings().embed_query(text))
