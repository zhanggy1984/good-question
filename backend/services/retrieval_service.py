"""混合检索器：语义 TOP-3 + ES 全文 TOP-3 → 去重 → BGE-Reranker 精排 TOP-2

检索范围限定在 session 绑定的 library_id，跨库隔离。
ES 不可用时降级为纯语义检索（不阻塞主链路）。
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import settings
from services import vector_store_service
from utils.es_index import es_index

logger = logging.getLogger("native_rag")


def _es_hits_to_docs(hits: list[dict]) -> list[Document]:
    """ES 命中结果转 LangChain Document（metadata 携带溯源信息）"""
    docs = []
    for h in hits:
        meta = dict(h.get("metadata") or {})
        meta.setdefault("document_id", h["document_id"])
        docs.append(Document(page_content=h["text"], metadata=meta))
    return docs


def _deduplicate(docs: list[Document]) -> list[Document]:
    """按 document_id + chunk_index 去重，保留首个"""
    seen = set()
    result = []
    for doc in docs:
        key = (
            doc.metadata.get("document_id"),
            doc.metadata.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


@lru_cache(maxsize=1)
def _get_reranker():
    """懒加载 BGE-Reranker（CrossEncoder），首次从 HF 镜像下载模型"""
    from sentence_transformers import CrossEncoder

    logger.info("[rerank] 加载模型 %s", settings.rerank_model_name)
    return CrossEncoder(settings.rerank_model_name, device="cpu")


def _rerank(query: str, docs: list[Document], top_k: int = 2) -> list[Document]:
    """BGE-Reranker 精排：相对排序取 top-k，仅当最高分极低才判定无关

    实测发现 rerank 绝对分数不可靠（最相关 chunk 可能只得 0.27 分），
    故不再用绝对阈值过滤（避免误杀），改为：相对排序取 top-k，
    仅当最高分低于 similarity_threshold_low（默认 0.3）才判定"文档无关"返回空。
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
    """混合检索器：query 改写 + 语义 + ES 全文 → rerank"""

    library_id: int = Field(description="文档库 id")
    candidate_k: int = Field(default=3, description="单路召回候选数（语义+ES 各 3，控制 rerank 候选总量）")

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        # 0. 直接用原 query 检索，不再 LLM 改写：
        #    改写需每次检索前多一次 LLM 调用（2-4s），是 sources 事件前的主要延迟；
        #    且历史实测改写检索结果与原 query 几乎一致（双路 vs 单路结论），收益趋零。
        #    未来若需规范化改写，可重新启用 llm_service.rewrite_query。
        search_query = query

        # 1+2. 语义检索（ChromaDB）与 ES 全文检索并发执行
        #    两路独立网络 I/O，串行需 2s+（语义 1.7 + ES 0.5），并发只耗时较长一路。
        #    ES 失败降级为空（不阻塞主链路）。
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sem = ex.submit(
                vector_store_service.similarity_search,
                self.library_id, search_query, k=self.candidate_k,
            )

            def _es_search():
                try:
                    return _es_hits_to_docs(
                        es_index.search(self.library_id, search_query, k=self.candidate_k)
                    )
                except Exception as e:
                    logger.warning("[retrieve] ES 检索失败: %s", e)
                    return []

            f_es = ex.submit(_es_search)
            pairs = f_sem.result()
            t_sem = time.time()
            es_docs = f_es.result()
            t_es = time.time()
        logger.info("[retrieve] 语义检索 耗时=%.2fs 命中=%s", t_sem - t_start, len(pairs))
        logger.info("[retrieve] ES 检索 耗时=%.2fs 命中=%s", t_es - t_start, len(es_docs))
        semantic_docs = [doc for doc, _ in pairs]

        # 3. 合并去重
        merged = _deduplicate(semantic_docs + es_docs)

        # 4. Rerank 精排（用改写后的规范 query 打分）
        t0 = time.time()
        result = _rerank(search_query, merged, top_k=3)
        logger.info(
            "[retrieve] 合并+rerank 耗时=%.2fs 候选=%s 结果=%s",
            time.time() - t0, len(merged), len(result),
        )
        return result
