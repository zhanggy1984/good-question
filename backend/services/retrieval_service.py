"""混合检索器：Milvus hybrid_search（dense 语义 + BM25 全文，RRF 融合）→ BGE-Reranker 精排 TOP-3

检索范围限定在 session 绑定的 library_id（partition 隔离），跨库隔离。
Milvus 不可用时抛异常，由 chat_service 现有异常处理兜底。
"""
import logging
import time
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import settings
from services import vector_store_service
from services.embedding_service import embed_query

logger = logging.getLogger("native_rag")


def _milvus_hits_to_docs(hits: list[dict]) -> list[Document]:
    """Milvus 命中结果转 LangChain Document（metadata 携带溯源信息）"""
    docs = []
    for h in hits:
        meta = dict(h.get("metadata") or {})
        meta.setdefault("document_id", h.get("document_id"))
        meta.setdefault("chunk_index", h.get("chunk_index"))
        docs.append(Document(page_content=h["text"], metadata=meta))
    return docs


@lru_cache(maxsize=1)
def _get_reranker():
    """懒加载 BGE-Reranker（CrossEncoder），首次从 HF 镜像下载模型"""
    from sentence_transformers import CrossEncoder

    logger.info("[rerank] 加载模型 %s", settings.rerank_model_name)
    return CrossEncoder(settings.rerank_model_name, device="cpu")


def _rerank(query: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    """BGE-Reranker 精排：相对排序取 top-k，仅当最高分极低才判定无关

    实测发现 rerank 绝对分数不可靠（最相关 chunk 可能只得 0.27 分），
    故不再用绝对阈值过滤（避免误杀），改为：相对排序取 top-k，
    仅当最高分低于 similarity_threshold_low（默认 0.2）才判定"文档无关"返回空。
    """
    try:
        reranker = _get_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        ranked = sorted(
            zip(docs, scores), key=lambda x: x[1], reverse=True
        )
        # 无结果判定：最高分也低于低阈值（真正无关，宁缺毋滥）
        if ranked and ranked[0][1] < settings.similarity_threshold_low:
            return []
        return [doc for doc, _ in ranked[:top_k]]
    except Exception as e:
        # rerank 失败降级：直接取合并后的前 top_k
        logger.warning("[rerank] 精排失败，降级取前 %s: %s", top_k, e)
        return docs[:top_k]


class HybridRetriever(BaseRetriever):
    """混合检索器：dense 语义 + BM25 全文 → RRF 融合 → rerank"""

    library_id: int = Field(description="文档库 id")
    candidate_k: int = Field(default=3, description="单路召回候选数（语义+BM25 各 3，控制 rerank 候选总量）")

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        # 0. 直接用原 query 检索，不再 LLM 改写：
        #    改写需每次检索前多一次 LLM 调用（2-4s），是 sources 事件前的主要延迟；
        #    且历史实测改写检索结果与原 query 几乎一致（双路 vs 单路结论），收益趋零。
        #    未来若需规范化改写，可重新启用 llm_service.rewrite_query。
        search_query = query

        # 1. Milvus 混合检索：dense + BM25 双路召回 + RRF 融合（服务端完成）
        #    RRF 已融合去重，无需应用层再按 document_id+chunk_index 去重。
        #    候选总量 = limit = candidate_k*2（默认 6，与替换前语义3+ES3 相当），不放大 Rerank 瓶颈。
        t_start = time.time()
        dense_vec = embed_query(search_query)
        hits = vector_store_service.hybrid_search(
            self.library_id, search_query, dense_vec,
            dense_k=self.candidate_k, bm25_k=self.candidate_k, limit=self.candidate_k * 2,
        )
        docs = _milvus_hits_to_docs(hits)
        logger.info(
            "[retrieve] Milvus 混合检索 耗时=%.2fs 命中=%s",
            time.time() - t_start, len(docs),
        )

        # 2. Rerank 精排（用原 query 打分）
        t0 = time.time()
        result = _rerank(search_query, docs, top_k=3)
        logger.info(
            "[retrieve] rerank 耗时=%.2fs 候选=%s 结果=%s",
            time.time() - t0, len(docs), len(result),
        )
        return result
