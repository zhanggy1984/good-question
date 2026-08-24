"""混合检索器：Milvus hybrid_search（dense 语义 + BGE-M3 稀疏，RRF 融合）→ BGE-Reranker 精排 TOP-3

RAG 迁移一期后为普通类（去掉 LangChain BaseRetriever 继承）：
- 类名 HybridRetriever / invoke() / max_rerank_score 保留，调用方（chat_service）零改动
- 返回 list[RetrievedChunk]（.content/.metadata/.score），不再返回 LangChain Document

检索范围限定在 session 绑定的 library_id（library_id 字段 MetadataFilters 过滤），跨库隔离。
Milvus 不可用时抛异常，由 chat_service 现有异常处理兜底。
"""
import logging
import time

from config import settings
from services import rerank
from services import vector_store_service
from services.embedding_service import embed_query
from services.retrieval_types import RetrievedChunk

logger = logging.getLogger("native_rag")


def _hits_to_chunks(hits: list[dict]) -> list[RetrievedChunk]:
    """Milvus 命中结果（node_to_hit dict）转 RetrievedChunk（metadata 携带溯源信息）"""
    chunks = []
    for h in hits:
        meta = dict(h.get("metadata") or {})
        meta.setdefault("document_id", h.get("document_id"))
        meta.setdefault("chunk_index", h.get("chunk_index"))
        chunks.append(RetrievedChunk(content=h["text"], metadata=meta))
    return chunks


class HybridRetriever:
    """混合检索器：dense 语义 + BGE-M3 稀疏 → RRF 融合 → rerank"""

    def __init__(self, library_id: int, candidate_k: int = 3):
        self.library_id = library_id
        self.candidate_k = candidate_k  # 单路召回候选数（语义+稀疏各 3，控制 rerank 候选总量）
        self.max_rerank_score: float | None = None  # 本次检索 rerank 精排最高分（invoke 后读取）

    def invoke(self, query: str) -> list[RetrievedChunk]:
        # 0. 直接用原 query 检索，不再 LLM 改写：
        #    改写需每次检索前多一次 LLM 调用（2-4s），是 sources 事件前的主要延迟；
        #    且历史实测改写检索结果与原 query 几乎一致（双路 vs 单路结论），收益趋零。
        #    未来若需规范化改写，可重新启用 llm_service.rewrite_query。
        search_query = query

        # 1. Milvus 混合检索：dense + 稀疏双路召回 + RRF 融合（服务端完成）
        #    RRF 已融合去重，无需应用层再按 document_id+chunk_index 去重。
        #    候选总量 = limit = candidate_k*2（默认 6）。注意：LlamaIndex 0.12.52 hybrid 模式
        #    无双路独立 k（双路与最终 limit 均为 similarity_top_k），dense_k/bm25_k 保留签名兼容。
        t_start = time.time()
        dense_vec = embed_query(search_query)
        hits = vector_store_service.hybrid_search(
            self.library_id, search_query, dense_vec,
            dense_k=self.candidate_k, bm25_k=self.candidate_k, limit=self.candidate_k * 2,
        )
        chunks = _hits_to_chunks(hits)
        logger.info(
            "[retrieve] Milvus 混合检索 耗时=%.2fs 命中=%s",
            time.time() - t_start, len(chunks),
        )

        # 2. Rerank 精排（用原 query 打分）
        t0 = time.time()
        result, max_score = rerank.rerank_chunks(search_query, chunks, top_k=3)
        self.max_rerank_score = max_score
        logger.info(
            "[retrieve] rerank 耗时=%.2fs 候选=%s 结果=%s 最高分=%s",
            time.time() - t0, len(chunks), len(result),
            "无" if max_score is None else round(max_score, 3),
        )
        return result
