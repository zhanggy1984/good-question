"""embedding_service L2 归一化测试（纯函数，不加载模型）

归一化是 Milvus IP 检索的前提（归一化后 IP == COSINE），属核心逻辑，必须覆盖。
"""
import math

from services.embedding_service import _l2_normalize


def test_l2_normalize_single_unit_length():
    """单向量归一化后模长为 1"""
    v = _l2_normalize([3.0, 4.0])
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2)
    assert abs(norm - 1.0) < 1e-6


def test_l2_normalize_batch_shapes():
    """批量归一化保持形状且每行模长为 1"""
    vecs = _l2_normalize([[3.0, 4.0], [0.0, 5.0]])
    assert len(vecs) == 2
    assert len(vecs[0]) == 2
    for v in vecs:
        assert abs(math.sqrt(v[0] ** 2 + v[1] ** 2) - 1.0) < 1e-6


def test_l2_normalize_zero_vector_safe():
    """零向量兜底不除零、不变"""
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]
