"""Elasticsearch 全文检索封装

单 index `rag_chunks`，通过 library_id filter 实现库隔离。
中文分词用 IK 插件（ik_max_word 索引 / ik_smart 查询）。
"""
import logging

from elasticsearch import Elasticsearch

from config import settings

logger = logging.getLogger("native_rag")

INDEX_NAME = "rag_chunks"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "long"},
            "document_id": {"type": "long"},
            "library_id": {"type": "long"},
            "text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
            },
            # 只存不索引（溯源信息以 MySQL chunks 表为准）
            "metadata": {"type": "object", "enabled": False},
        }
    }
}


class ESIndex:
    """ES 全文索引管理"""

    def __init__(self):
        self._client: Elasticsearch | None = None

    def get_client(self) -> Elasticsearch:
        """懒加载 ES 客户端"""
        if self._client is None:
            self._client = Elasticsearch(settings.es_url, request_timeout=30)
        return self._client

    def ensure_index(self) -> None:
        """确保 rag_chunks 索引存在（不存在则创建）"""
        client = self.get_client()
        if not client.indices.exists(index=INDEX_NAME):
            client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
            logger.info("[es] 已创建索引 %s", INDEX_NAME)

    def bulk_add_chunks(self, chunks: list[dict]) -> None:
        """批量写入 chunk（上传低频操作，写入后立即 refresh 保证可检索）"""
        if not chunks:
            return
        client = self.get_client()
        operations = []
        for c in chunks:
            operations.append({"index": {"_index": INDEX_NAME, "_id": c["_id"]}})
            operations.append({
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "library_id": c["library_id"],
                "text": c["text"],
                "metadata": c.get("metadata", {}),
            })
        resp = client.bulk(operations=operations, refresh=True)
        if resp.get("errors"):
            logger.error("[es] bulk 写入存在错误: %s", resp.get("errors"))

    def search(self, library_id: int, query: str, k: int = 3) -> list[dict]:
        """全文检索（精确优先，宽松补足）

        先要求所有查询词命中（operator=and，精确），
        结果不足 k 时用任一查询词命中（operator=or）补足到 k。
        """
        client = self.get_client()

        def _run(operator: str) -> list[dict]:
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"text": {"query": query, "operator": operator}}}
                        ],
                        "filter": [{"term": {"library_id": library_id}}],
                    }
                },
                "size": k,
            }
            resp = client.search(index=INDEX_NAME, body=body)
            return [
                {
                    "chunk_id": h["_source"]["chunk_id"],
                    "document_id": h["_source"]["document_id"],
                    "text": h["_source"]["text"],
                    "metadata": h["_source"].get("metadata", {}),
                }
                for h in resp["hits"]["hits"]
            ]

        # 精确优先：所有查询词都必须出现
        hits = _run("and")
        if len(hits) >= k:
            return hits

        # 结果不足 k：用宽松（or）补足
        loose = _run("or")
        seen = {h["chunk_id"] for h in hits}
        for h in loose:
            if len(hits) >= k:
                break
            if h["chunk_id"] not in seen:
                hits.append(h)
                seen.add(h["chunk_id"])
        return hits

    def delete_by_document(self, document_id: int) -> None:
        """删除某文档的全部 chunk"""
        client = self.get_client()
        client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"term": {"document_id": document_id}}},
            refresh=True,
        )

    def delete_by_library(self, library_id: int) -> None:
        """删除某库的全部 chunk"""
        client = self.get_client()
        client.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"term": {"library_id": library_id}}},
            refresh=True,
        )


es_index = ESIndex()
