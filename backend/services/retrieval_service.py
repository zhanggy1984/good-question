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

# 章节级扩充：同文档回查候选上限 + 扩充后 context 总量上限 + 命中节相邻扩展节数。
# 总量上限防大文档撑爆 prompt；相邻扩展解决「条款/规则跨节问答」：用户问某节时，相邻
# 约束条款（如违约责任）是独立 section，仅同节回查永远取不到（实测 3172 违约 absent）。
_SECTION_EXPAND_LIMIT = 20
_SECTION_EXPAND_TOTAL = 6
_SECTION_EXPAND_ADJACENT = 2

# 总结/列举/概述类意图词表：命中时扩召回，覆盖多 section（默认 top-3 只覆盖最相关小节，
# 总结类答案若只 feed 少数 section 会漏掉其他章节条款，judge 判 factuality 吃亏）
_SUMMARY_INTENT_WORDS = ("总结", "汇总", "概括", "介绍", "列出", "列举", "说明", "有哪些", "全部", "整体", "主要")


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

    def __init__(self, library_id: int, candidate_k: int = 3, top_k: int = 3):
        self.library_id = library_id
        self.candidate_k = candidate_k  # 单路召回候选数（语义+稀疏各 3，控制 rerank 候选总量）
        self.top_k = top_k  # rerank 精排进 context 的数量（总结/列举类调高，覆盖多 section）
        self.max_rerank_score: float | None = None  # 本次检索 rerank 精排最高分（invoke 后读取）
        self.top_hits: list[RetrievedChunk] = []  # rerank 精排结果（历史字段：同源模式下 execute_retrieve_tool 直接读 invoke 返回值）

    def invoke(self, query: str, expand: bool = True) -> list[RetrievedChunk]:
        # expand=False 关闭章节扩充：execute_retrieve_tool 默认关闭（context 与 sources 同源、
        # 编号一一对应，回答引用必有卡片）。扩充逻辑保留，需相邻约束条款时按需开启。
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
        result, max_score = rerank.rerank_chunks(search_query, chunks, top_k=self.top_k)
        self.max_rerank_score = max_score
        self.top_hits = result
        logger.info(
            "[retrieve] rerank 耗时=%.2fs 候选=%s 结果=%s 最高分=%s",
            time.time() - t0, len(chunks), len(result),
            "无" if max_score is None else round(max_score, 3),
        )

        # 3. 章节级扩充：按 document 回查，对 top-3 命中的 section 补齐同节及相邻节 chunk。
        #    条款/规则类文档每节独立 section，用户问某节时相邻约束条款（如违约责任）在另一节，
        #    仅同节回查取不到 → 命中节前后各 _SECTION_EXPAND_ADJACENT 节一并并入。sources
        #    仍精确指向 top-3 小节。旧数据无 section_id 时天然退化（不扩充）；扩充失败降级
        #    保留 top-3，不阻塞主链路。
        if not result or not expand:
            return result
        return self._expand_section_context(search_query, result)

    def _expand_section_context(self, query: str, top: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """按 document 回查同文档 chunk，扩充命中 section 的相邻 section，合并作 context。

        从「同 section 回查」升级为「同 document + 相邻 section」的原因：条款/协议/规则类
        文档每章节是独立 section（section_id=doc:序），top-3 命中「保密范围/期限」节时，
        同节兄弟 chunk 只有它自己，「六、违约责任」这类相邻约束条款永远进不了 context
        （实测 3172：context 无「违约」字样，prompt 无法弥补）。按 document 回查 + 命中节
        前后各 _SECTION_EXPAND_ADJACENT 个相邻 section 一并并入后，约束条款进入 LLM 视野。
        旧数据无 section_id 时天然退化（不扩充）；扩充失败降级保留 top，不阻塞主链路。
        """
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
        # 相邻扩充只针对主文档（rerank 最高分 chunk 所属文档，top[0]）。多文档命中时若每个
        # 文档都做 document 级扩充，次要文档会互相膨胀挤掉关键节（实测 3160「违约金」context
        # 被考勤文档 5 节污染）。主文档的相邻约束条款才是用户当前问题真正需要的上下文。
        primary_doc = top[0].metadata.get("document_id")
        for doc_id in {d for d, _ in sections}:
            if doc_id != primary_doc:
                continue
            try:
                hits = vector_store_service.hybrid_search(
                    self.library_id, query, dense_vec,
                    limit=_SECTION_EXPAND_LIMIT,
                    # Milvus 标量字段 document_id 存字符串，LlamaIndex MetadataFilter 传 int
                    # 生成不了匹配表达式（实测 int=0 命中、str=全命中）→ 必须转 str
                    extra_filters={"document_id": str(doc_id)},
                )
            except Exception as e:
                logger.warning("[retrieve] 章节扩充失败（降级保留 top）: %s", e)
                continue
            doc_chunks = _hits_to_chunks(hits)
            if not doc_chunks:
                continue
            # 章节序列：按 section 内最小 chunk_index 排序，得到文档内的节顺序
            sec_map: dict[str, list[RetrievedChunk]] = {}
            for c in doc_chunks:
                sec_map.setdefault(c.metadata.get("section_id") or "", []).append(c)
            section_order = sorted(
                sec_map.items(),
                key=lambda kv: min(c.metadata.get("chunk_index") or 0 for c in kv[1]),
            )
            doc_sids = {sid for d, sid in sections if d == doc_id}
            hit_pos = {i for i, (sid, _) in enumerate(section_order) if sid in doc_sids}
            if not hit_pos:
                continue
            # 命中节自身 + 前后各 ADJACENT 个相邻节
            expand_pos = set()
            for p in hit_pos:
                expand_pos.update(
                    range(max(0, p - _SECTION_EXPAND_ADJACENT),
                          min(len(section_order), p + _SECTION_EXPAND_ADJACENT + 1))
                )
            for p in sorted(expand_pos):
                for c in section_order[p][1]:
                    key = (c.metadata.get("document_id"), c.metadata.get("chunk_index"))
                    if key not in seen:
                        seen.add(key)
                        merged.append(c)
        if len(merged) <= _SECTION_EXPAND_TOTAL:
            return merged
        # 总量上限内截断：命中节优先，其余按距命中 chunk 近者优先（只丢最远相邻节，不丢关键约束）
        def _dist(c: RetrievedChunk) -> int:
            did = c.metadata.get("document_id")
            cidx = c.metadata.get("chunk_index") or 0
            if (did, c.metadata.get("section_id")) in sections:
                return -1
            return min(
                (abs(cidx - (h.metadata.get("chunk_index") or 0))
                 for h in top if h.metadata.get("document_id") == did),
                default=999,
            )
        merged.sort(key=_dist)
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


def _is_summary_intent(user_question: str | None, query: str) -> bool:
    """总结/列举/概述类意图检测。

    用**原始用户问题**（user_question）而非 LLM 改写的 query：chat 链路第一轮 LLM 自主决定
    调工具时传的 query 常把"总结/介绍"这类意图词丢在 prompt 里（如 3162 的
    query="考勤管理制度主要内容"），意图词仍留在 user_content，故两者都查防漏判。
    """
    text = f"{user_question or ''} {query}"
    return any(w in text for w in _SUMMARY_INTENT_WORDS)


def execute_retrieve_tool(library_id: int, query: str, user_question: str | None = None) -> dict:
    """执行 hybrid_retrieve 工具：检索 + 组装 LLM 上下文与 SSE sources

    返回 dict：context（[来源N] 格式化，供第二轮 LLM 的 tool 消息）、sources（前端引用卡片）、
    source_count / max_score / confidence_band（tool_call 事件 result 与监控日志用）

    user_question 传入原始用户问题：总结/列举类问题扩召回（candidate_k/top_k 3→6），
    普通问答行为不变（沿用默认 top-3）。

    context 与 sources 同源（rerank 精排结果，不做章节扩充）：普通问答 top-3、
    总结类 top-6，[来源N] 编号与卡片一一对应，回答引用必有出处。关闭章节扩充
    避免弱相关的相邻节进入 context 导致回答发散、引用无卡片（实测体验断裂）。
    扩充逻辑保留在 HybridRetriever.invoke(expand=True)，需相邻约束条款时按需开启。
    """
    summary = _is_summary_intent(user_question, query)
    retriever = HybridRetriever(
        library_id=library_id,
        candidate_k=6 if summary else 3,
        top_k=6 if summary else 3,
    )
    chunks = retriever.invoke(query, expand=False)  # 精排结果：普通 top-3 / 总结类 top-6
    max_score = retriever.max_rerank_score
    # context 与 sources 同源：sources 按 chunks 编号构建，回答 [来源N] 引用必有卡片。
    sources = [
        {
            "document_name": c.metadata.get("document_name", "未知文档"),
            "heading_path": [h for h in (c.metadata.get("heading_path") or []) if h],
            "chunk_content": c.content[:200],
            "chunk_index": c.metadata.get("chunk_index"),
            "total_chunks": c.metadata.get("total_chunks"),
            "page_range": c.metadata.get("page_range") or [0, 0],
        }
        for c in chunks
    ]
    return {
        "context": _format_docs(chunks),
        "sources": sources,
        # 命中信号 + 来源数：与卡片数一致（普通 3 / 总结类 6）。source_count>0 是
        # chat_service/chat_cache 判断"检索命中"与缓存重放 sources 的开关。
        "source_count": len(chunks),
        # rerank 返回 numpy float32，进 SSE tool_call result 与 JSON 日志须转原生 float（json.dumps 不认 float32）
        "max_score": float(max_score) if max_score is not None else None,
        "confidence_band": _confidence_band(max_score),
    }
