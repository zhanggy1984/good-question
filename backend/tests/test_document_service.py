"""文档服务测试：上传文件流式写盘"""
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, "/app")


class FakeUploadFile:
    """模拟 FastAPI UploadFile：带 filename，file 支持分块 read"""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.file = BytesIO(content)


def test_save_upload_file_streaming(tmp_path, monkeypatch):
    """分块写盘：按 library_id 分目录保存，返回正确的类型和大小（非整读内存）"""
    from config import settings
    from services import document_service

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    content = b"a" * (2 * 1024 * 1024)  # 2MB，跨多个 1MB 分块
    path, suffix, size = document_service.save_upload_file(
        FakeUploadFile("big.pdf", content), library_id=5
    )

    assert suffix == "pdf"
    assert size == len(content)
    assert Path(path).exists()
    assert Path(path).stat().st_size == size
    # 按库分目录：{upload_dir}/{library_id}/
    assert Path(path).parent == tmp_path / "5"
    # 文件名带原始名，避免覆盖
    assert Path(path).name.endswith("big.pdf")
