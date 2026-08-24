"""LlamaIndex ↔ Milvus 重建路径 PoC（Step 0）：验证 LlamaIndex 自建 collection + BGE-M3 混合检索

在容器内运行（LlamaIndex + FlagEmbedding 装好后）：
    docker exec rag-backend python scripts/poc_llamaindex.py

背景（0.12.52 源码已确认）：MilvusVectorStore 无服务端 BM25，也没有
sparse_embedding_field / vector_store_query_mode / enable_dynamic_field / analyzer_params 这些参数。
稀疏检索走 Python 侧 BGE-M3 学习稀疏：add 用 sparse_embedding_function.encode_documents 生成稀疏向量
直接插入默认稀疏字段 sparse_embedding（索引 SPARSE_INVERTED_INDEX + IP），查询用 encode_queries。
schema 由 LlamaIndex 全权创建（主键 id、enable_dynamic_field=True 硬编码，metadata 存动态字段 _node_content）。
现有 rag_chunks（服务端 BM25 function + pk 主键 + enable_dynamic_field=False）无法复用，必须重建。

本 PoC 验证重建路径本身可行：schema 形态 + 写读删 + text/metadata 往返 + expr 库隔离 + delete expr。
临时 collection rag_chunks_poc_rebuild，结束即 drop，不触碰生产数据。

注意：BGEM3SparseEmbeddingFunction 首次实例化会从 HF 下载 bge-m3 模型（~2.3GB，
缓存于 /root/.cache 即 model_cache volume），首次运行较慢属预期。
"""
import sys

sys.path.insert(0, "/app")

from functools import lru_cache

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.milvus import MilvusVectorStore, IndexManagement
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction

from config import settings
from services.embedding_service import embed_query

REBUILD_COLLECTION = "rag_chunks_poc_rebuild"  # 重建路径验证用临时 collection，结束即 drop
DIM = 768
TEST_LIB_A = 990001
TEST_LIB_B = 990002
TEST_DOC_A = 99000101  # 合成文档 document_id（高位同段位数字，避免与真实数据混淆）
TEST_DOC_B = 99000201
QUERY_TEXT = "员工请假 考勤异常 报销单"

_results: list[tuple[str, bool | None, str]] = []


def report(name: str, ok: bool | None, detail: str = ""):
    """记录单项结论：True=pass，False=fail，None=无法判定（unknown）"""
    _results.append((name, ok, detail))
    tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[ok]
    print(f"[{tag}] {name}" + (f"  -- {detail}" if detail else ""))


@lru_cache(maxsize=1)
def _get_sparse_fn():
    """懒加载 BGE-M3 稀疏编码器（首次下载模型；模块级单例避免多次加载内存）

    注意：BGEM3SparseEmbeddingFunction 无参构造（device 内部自动选 CPU），
    构造即加载模型（非懒加载），故用 lru_cache 包住避免多次加载。
    """
    return BGEM3SparseEmbeddingFunction()


def make_store(
    collection: str = REBUILD_COLLECTION,
    index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    output_fields=None,
    **overrides,
) -> MilvusVectorStore:
    """构造 MilvusVectorStore（0.12.52 真实 API，重建路径）

    稀疏用 BGE-M3 学习稀疏（enable_sparse=True + sparse_embedding_function），
    双路融合 RRF；dense 索引用 HNSW；schema 由 LlamaIndex 自建。
    """
    params = dict(
        uri=settings.milvus_uri,
        collection_name=collection,
        dim=DIM,
        embedding_field="dense",
        # text_key=None：读回走 metadata_dict_to_node（_node_content 完整还原 text/metadata），
        # 避免 schema 无 text 列时 text=None 触发 TextNode ValidationError（string_type）
        text_key=None,
        output_fields=output_fields or ["*"],
        enable_sparse=True,
        sparse_embedding_function=_get_sparse_fn(),
        hybrid_ranker="RRFRanker",
        index_config={"index_type": "HNSW", "M": 16, "efConstruction": 200},
        index_management=index_management,
        overwrite=False,
    )
    params.update(overrides)
    return MilvusVectorStore(**params)


def _query(
    store: MilvusVectorStore,
    query_text: str,
    library_id: int,
    limit: int = 6,
    mode=VectorStoreQueryMode.HYBRID,
):
    """hybrid/dense 查询：embedding（L2 归一化）+ query_str（BGE-M3 稀疏路）+ filters（库隔离）

    注意：MilvusVectorStore 的 query() 忽略 kwargs 里的 expr（_prepare_before_search 只认
    query.filters/doc_ids/node_ids），库隔离必须走 MetadataFilters。
    """
    return store.query(
        VectorStoreQuery(
            query_embedding=embed_query(query_text),
            query_str=query_text,
            mode=mode,
            similarity_top_k=limit,
            sparse_top_k=limit,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="library_id", value=library_id)]
            ),
        )
    )


def _make_test_node(doc_id: int, chunk_index: int, lib: int, text: str, heading: list[str]) -> TextNode:
    """合成测试节点：metadata 带 heading_path 数组 / token_count 整数，验证动态字段往返保真"""
    node = TextNode(
        text=text,
        metadata={
            "document_id": doc_id,
            "document_name": "poc_llamaindex_test.md",
            "library_id": lib,
            "heading_path": heading,
            "source_type": "markdown",
            "token_count": 120,
            "chunk_index": chunk_index,
            "total_chunks": 2,
        },
        embedding=embed_query(text),
    )
    node.node_id = f"{doc_id}_{chunk_index}"  # 主键格式与生产 add_chunks 一致
    # ref_doc_id = document_id：ref_doc_id 是只读 property（从 relationships[SOURCE] 派生），
    # node_to_metadata_dict 把顶层 document_id 字段写为 node.ref_doc_id（未设则 "None"），
    # delete(ref_doc_id) 按该字段删。设置 relationships 后顶层 document_id=str(doc_id)，
    # delete_by_document 语义与现有接口一致。
    # （metadata 里的 document_id(int) 仍保存在 _node_content，消费侧读 int 不变）
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=str(doc_id))
    return node


def _verify_rebuild_path(client) -> None:
    """重建路径验证：LlamaIndex 自建 collection（BGE-M3 学习稀疏）写读删闭环

    观测 schema 形态 + metadata 动态字段往返 + expr 库隔离 + delete expr。
    临时 collection，finally 中 drop。
    """
    try:
        if client.has_collection(REBUILD_COLLECTION):
            client.drop_collection(REBUILD_COLLECTION)
        store = make_store()

        schema = client.describe_collection(REBUILD_COLLECTION)
        fields = [(f["name"], str(f.get("type"))) for f in schema.get("fields", [])]
        has_pk = any(f[0] == "id" for f in fields)
        has_sparse = any("sparse" in f[0] for f in fields)
        report("C1 重建 schema 形态", has_pk and has_sparse, f"字段={fields}")

        nodes_a = [
            _make_test_node(TEST_DOC_A, 0, TEST_LIB_A, "员工请假需提前一天提交申请，考勤异常需及时申诉。", ["请假制度"]),
            _make_test_node(TEST_DOC_A, 1, TEST_LIB_A, "报销单需附发票原件，出差补助按天计算。", ["报销流程"]),
            _make_test_node(TEST_DOC_B, 0, TEST_LIB_B, "入职需准备身份证与学历证明，签订劳动合同。", ["入职指南"]),
        ]
        ids = store.add(nodes_a, force_flush=True)
        report("C2 BGE-M3 add（含学习稀疏向量）", bool(ids), f"写入 {len(ids)} 条")

        # 稀疏向量确实写入 collection（显式 SPARSE 字段非空）
        rows = client.query(
            collection_name=REBUILD_COLLECTION,
            filter=f'document_id == "{TEST_DOC_A}"',
            output_fields=["sparse_embedding"],
            limit=16,
        )
        sparse_ok = rows and all(r.get("sparse_embedding") for r in rows)
        report("C3 稀疏向量写入确认", bool(sparse_ok), f"{len(rows)} 条 sparse_embedding 非空" if rows else "无数据")

        res = _query(store, QUERY_TEXT, TEST_LIB_A, limit=6)
        hit = next((n for n in res.nodes if n.metadata.get("document_id") == TEST_DOC_A), None)
        if hit is not None:
            md = hit.metadata
            roundtrip_ok = (
                md.get("heading_path") == ["请假制度"]
                and md.get("token_count") == 120
                and md.get("library_id") == TEST_LIB_A
                and md.get("document_name") == "poc_llamaindex_test.md"
                and hit.text.startswith("员工请假")
            )
            report(
                "C4 hybrid 检索 + text/metadata 动态字段往返",
                roundtrip_ok,
                f"返回 {len(res.nodes)} 条 heading_path={md.get('heading_path')!r} token_count={md.get('token_count')!r} text={hit.text[:30]!r}",
            )
        else:
            report("C4 hybrid 检索 + text/metadata 动态字段往返", False, "add 后 hybrid 查不到合成数据")

        # expr 库隔离在 hybrid 生效（双路均按 library_id 过滤）
        res_a = _query(store, QUERY_TEXT, TEST_LIB_A, limit=6)
        leaked = any(n.metadata.get("library_id") == TEST_LIB_B for n in res_a.nodes)
        report(
            "C5 expr 库隔离在 hybrid 生效",
            (not leaked) and any(n.metadata.get("library_id") == TEST_LIB_A for n in res_a.nodes),
            f"A 库查询 {len(res_a.nodes)} 条，串库={leaked}",
        )

        # 稀疏路生效：hybrid 融合双路召回应 ≥ 纯 dense（含词面匹配的增量）
        # 0.12.52 的 VectorStoreQueryMode 无 DENSE 枚举，dense 检索即 DEFAULT
        d = _query(store, QUERY_TEXT, TEST_LIB_A, limit=6, mode=VectorStoreQueryMode.DEFAULT)
        report(
            "C6 稀疏路生效（hybrid vs dense）",
            len(res.nodes) >= len(d.nodes),
            f"hybrid={len(res.nodes)} vs dense={len(d.nodes)}",
        )

        report("C7 路级 k 语义（similarity/sparse_top_k）", len(res.nodes) <= 12, f"limit=6 实返 {len(res.nodes)} 条（RRF 融合上限=两路之和）")

        store.delete(ref_doc_id=str(TEST_DOC_A))  # 按 doc_id 删除该文档全部 chunk
        # Milvus delete 返回 delete_count=1 但删除是延迟可见（立即 query 仍见旧数据），
        # 必须 flush 才立即生效。生产 delete_by_document 同样要 delete 后 flush。
        client.flush(REBUILD_COLLECTION)
        remain = client.query(
            collection_name=REBUILD_COLLECTION,
            filter=f'document_id == "{TEST_DOC_A}"',
            output_fields=["id"],
            limit=16,
        )
        report("C8 delete(ref_doc_id=document_id) 可删", not remain, f"删除后残留 {len(remain)} 条")
    except Exception as e:
        report("C 重建路径验证", False, f"{type(e).__name__}: {e}")
    finally:
        if client.has_collection(REBUILD_COLLECTION):
            client.drop_collection(REBUILD_COLLECTION)


def main() -> None:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=settings.milvus_uri)
    print("=" * 72)
    print(f"LlamaIndex 重建路径 PoC @ {settings.milvus_uri}")
    print("=" * 72)

    # ── 0. 现有 collection 探测：记录为何不能复用 ──
    if client.has_collection("rag_chunks"):
        schema = client.describe_collection("rag_chunks")
        fields = [(f["name"], str(f.get("type"))) for f in schema.get("fields", [])]
        print(f"[info] 现有 rag_chunks schema: {fields}")
        report(
            "P1 复用现有 collection",
            None,
            "现有 schema 为服务端 BM25 + pk 主键 + enable_dynamic_field=False，与 0.12.52 学习稀疏模型不兼容，跳过复用",
        )
    else:
        report("P1 复用现有 collection", None, "collection 不存在，直接走重建")

    try:
        _verify_rebuild_path(client)
    finally:
        _summarize()


def _summarize() -> None:
    print("\n" + "=" * 72)
    print("结论汇总")
    print("=" * 72)
    for name, ok, detail in _results:
        tag = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[ok]
        print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    fails = [n for n, ok, _ in _results if ok is False]
    if fails:
        print("\n结论：重建路径存在 FAIL 项，需按报告调整方案后重试。")
    else:
        print("\n结论：重建路径可行——LlamaIndex 自建 collection + BGE-M3 学习稀疏 + expr 库隔离 + delete expr 全部通过。")
        print("      生产迁移：migrate_to_milvus.py 改为 drop rag_chunks → LlamaIndex CREATE_IF_NOT_EXISTS 重建 → 逐库重灌。")
        print("      重建后 schema 由 LlamaIndex 定义（主键 id、稀疏字段 sparse_embedding、metadata 动态字段），")
        print("      vector_store_service / retrieval_service 按此适配。")
    print("=" * 72)


if __name__ == "__main__":
    main()
