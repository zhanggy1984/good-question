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

# 章节级扩充：单 section 回查上限 + 扩充后 context 总量上限（防大 section 撑爆 prompt）
_SECTION_EXPAND_LIMIT = 20
_SECTION_EXPAND_TOTAL = 6


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
    """混合检索器：dense 语义 + BGE-M3 稀疏 → RRF 融合 → rerank → 章节级扩充"""

    def __init__(self, library_id: int, candidate_k: int = 3):
        self.library_id = library_id
        self.candidate_k = candidate_k  # 单路召回候选数（语义+稀疏各 3，控制 rerank 候选总量）
        self.max_rerank_score: float | None = None  # 本次检索 rerank 精排最高分（invoke 后读取）
        self.top_hits: list[RetrievedChunk] = []  # rerank top-3（章节扩充前的精确小节，sources 用）

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
        self.top_hits = result
        logger.info(
            "[retrieve] rerank 耗时=%.2fs 候选=%s 结果=%s 最高分=%s",
            time.time() - t0, len(chunks), len(result),
            "无" if max_score is None else round(max_score, 3),
        )

        # 3. 章节级扩充：对精排 top-3 涉及的每个唯一 section 补齐兄弟 chunk，合并作 context。
        #    长文档内答案常横跨同章节多个 chunk，top-3 只覆盖片段上下文不全；扩充后
        #    LLM context 覆盖整节，sources 仍精确指向 top-3 小节。旧数据无 section_id 时
        #    天然退化（不扩充）。扩充失败降级保留 top-3，不阻塞主链路。
        if not result:
            return result
        return self._expand_section_context(search_query, result)

    def _expand_section_context(self, query: str, top: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """按 (document_id, section_id) 回查同章节兄弟 chunk，合并进 context（带总量上限）"""
        sections = {
            (c.metadata.get("document_id"), c.metadata.get("section_id"))
            for c in top
            if c.metadata.get("document_id") is not None and c.metadata.get("section_id") is not None
        }
        if not sections:
            return top
        dense_vec = embed_query(query)
        merged = list(top)
        seen = {(c.metadata.get("document_id"), c.metadata.get("chunk_index")) for c in top}
        for doc_id, sid in sections:
            try:
                hits = vector_store_service.hybrid_search(
                    self.library_id, query, dense_vec,
                    limit=_SECTION_EXPAND_LIMIT,
                    extra_filters={"document_id": doc_id, "section_id": sid},
                )
            except Exception as e:
                logger.warning("[retrieve] 章节扩充失败（降级保留 top）: %s", e)
                continue
            for c in _hits_to_chunks(hits):
                key = (c.metadata.get("document_id"), c.metadata.get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    merged.append(c)
        # 总量上限：扩充后 context 不超上限，防大 section 撑爆 prompt
        return merged[:_SECTION_EXPAND_TOTAL]


# ════════ 检索工具结果组装（由 chat_service 下沉，能力层职责：LLM 上下文 + SSE sources） ════════


def _format_docs(docs) -> str:
    """检索结果格式化为 prompt 的 context（带来源编号）"""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        src = f"[来源{i}] {meta.get('document_name', '未知文档')}"
        heading = [h for h in (meta.get("heading_path") or []) if h]
        if heading:
            src += f" > {' > '.join(heading)}"
        parts.append(f"{src}\n{doc.content}")
    return "\n\n---\n\n".join(parts)


def _confidence_band(max_score: float | None) -> str:
    """检索置信档三档：none（无分/低于 LOW 视为无关）｜low（[LOW, 低置信阈值) 相关性存疑）｜high"""
    if max_score is None or max_score < settings.similarity_threshold_low:
        return "none"
    if max_score < settings.rerank_low_confidence_threshold:
        return "low"
    return "high"


def execute_retrieve_tool(library_id: int, query: str) -> dict:
    """执行 hybrid_retrieve 工具：检索 + 组装 LLM 上下文与 SSE sources

    返回 dict：context（[来源N] 格式化，供第二轮 LLM 的 tool 消息）、sources（前端引用卡片）、
    source_count / max_score / confidence_band（tool_call 事件 result 与监控日志用）
    """
    retriever = HybridRetriever(library_id=library_id)
    chunks = retriever.invoke(query)  # 章节扩充后的 context（含同节兄弟 chunk）
    max_score = retriever.max_rerank_score
    # sources 保持精排 top-3 精确小节：章节扩充只扩大喂 LLM 的 context，不稀释引用精度
    top_hits = retriever.top_hits or chunks
    sources = [
        {
            "document_name": c.metadata.get("document_name", "未知文档"),
            "heading_path": [h for h in (c.metadata.get("heading_path") or []) if h],
            "chunk_content": c.content[:200],
            "chunk_index": c.metadata.get("chunk_index"),
            "total_chunks": c.metadata.get("total_chunks"),
            "page_range": c.metadata.get("page_range") or [0, 0],
        }
        for c in top_hits
    ]
    return {
        "context": _format_docs(chunks),
        "sources": sources,
        "source_count": len(chunks),  # 与 context 的 [来源N] 编号一致（LLM 视角的上下文量）
        # rerank 返回 numpy float32，进 SSE tool_call result 与 JSON 日志须转原生 float（json.dumps 不认 float32）
        "max_score": float(max_score) if max_score is not None else None,
        "confidence_band": _confidence_band(max_score),
    }
