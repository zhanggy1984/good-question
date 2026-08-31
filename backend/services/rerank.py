"""精排服务：BGE-Reranker（LlamaIndex SentenceTransformerRerank）TOP-3 + 两级置信档判定

迁移自 retrieval_service._rerank：替换手写 CrossEncoder 调用为 LlamaIndex 后置处理器，
返回类型从 (Document 列表, 最高分) 改为 (RetrievedChunk 列表, 最高分)，两级置信档语义原样保留。

分数口径（关键，见方案 R6）：SentenceTransformerRerank 内部用 CrossEncoder.predict 并直接
赋值 node.score，不归一化——0.20/0.50 阈值语义与迁移前一致。test_rerank 断言原始分守护。
"""
import logging
from functools import lru_cache

from llama_index.core.schema import NodeWithScore, TextNode

from config import settings
from services.retrieval_types import RetrievedChunk

logger = logging.getLogger("native_rag")


@lru_cache(maxsize=2)
def _get_reranker(top_n: int = 3):
    """懒加载 LlamaIndex BGE-Reranker 后置处理器（首次从 HF 镜像下载模型）

    top_n 参数化（默认 3，总结/列举类扩召回传 6）：SentenceTransformerRerank 在
    postprocess_nodes 内部按 self.top_n 截断，rerank_chunks 的 top_k 只在截断后切片，
    想扩召回必须同步放大 top_n。lru_cache 按 top_n 分键（并发安全，不复用实例改属性）。
    """
    from llama_index.core.postprocessor import SentenceTransformerRerank

    logger.info("[rerank] 加载模型 %s (top_n=%s)", settings.rerank_model_name, top_n)
    return SentenceTransformerRerank(
        model=settings.rerank_model_name,
        top_n=top_n,
        device="cpu",
    )


def _chunks_to_nodes(chunks: list[RetrievedChunk]) -> list[NodeWithScore]:
    """RetrievedChunk → NodeWithScore（喂给 LlamaIndex 后置处理器）"""
    return [
        NodeWithScore(
            node=TextNode(text=c.content, metadata=c.metadata),
            score=1.0,  # 占位分，postprocess_nodes 会用精排分覆盖
        )
        for c in chunks
    ]


def _nodes_to_chunks(nodes: list[NodeWithScore]) -> list[RetrievedChunk]:
    """NodeWithScore → RetrievedChunk（node.score 承载精排原始分）"""
    return [
        RetrievedChunk(content=n.node.text, metadata=n.node.metadata, score=n.score)
        for n in nodes
    ]


def rerank_chunks(
    query: str, chunks: list[RetrievedChunk], top_k: int = 3
) -> tuple[list[RetrievedChunk], float | None]:
    """BGE-Reranker 精排：返回 (top-k 片段, 精排最高分)，供调用方判断置信档

    绝对分数不可靠（实测最相关 chunk 可能只得 0.27 分），故不用绝对阈值过滤（避免误杀）：
    - 相对排序取 top-k 照常返回；
    - 仅当最高分低于 similarity_threshold_low（默认 0.2）才判定"文档无关"返回空（保最高分透出）；
    - 精排异常降级：直接取前 top_k，score=None（最高分归 None，置信档判定见 chat 层）。
    """
    if not chunks:
        return [], None
    try:
        reranker = _get_reranker(top_k)
        nodes = _chunks_to_nodes(chunks)
        ranked = reranker.postprocess_nodes(nodes, query_str=query)
        # 内部已按 node.score 降序，此处再排序幂等兜底；score 为 CrossEncoder 原始分
        ranked = sorted(ranked, key=lambda n: n.score if n.score is not None else -1.0, reverse=True)
        if not ranked:
            return [], None
        max_score = ranked[0].score
        # 无结果判定：最高分也低于低阈值（真正无关，宁缺毋滥）
        if max_score is None or max_score < settings.similarity_threshold_low:
            return [], max_score
        return _nodes_to_chunks(ranked[:top_k]), max_score
    except Exception as e:
        # rerank 失败降级：直接取前 top_k（score=None，max_score 归 None）
        logger.warning("[rerank] 精排失败，降级取前 %s: %s", top_k, e)
        return chunks[:top_k], None
