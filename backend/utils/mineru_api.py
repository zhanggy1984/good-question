"""MinerU 官方 API 客户端（精准解析，需 Token）

调用流程：获取上传链接 → PUT 上传本地文件 → 轮询解析结果 → 下载 zip → 解压取 full.md
参考：https://mineru.net/apiManage/docs
"""
import io
import logging
import time
import zipfile
from pathlib import Path

import requests

from config import settings

logger = logging.getLogger("native_rag")

BASE_URL = "https://mineru.net"
POLL_INTERVAL = 3
POLL_TIMEOUT = 300  # 5 分钟


def extract_with_api(file_path: str, token: str) -> str:
    """上传本地文件并解析，返回 Markdown 文本"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    # 1. 获取批量上传链接（files 需 name 字段）
    resp = requests.post(
        f"{BASE_URL}/api/v4/file-urls/batch",
        headers=headers,
        json={"files": [{"name": Path(file_path).name}]},
        timeout=30,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise ValueError(f"MinerU API 初始化失败: {data.get('msg', data)}")
    batch_id = data["data"]["batch_id"]
    file_url = data["data"]["file_urls"][0]

    # 2. PUT 上传文件（无 Content-Type）
    with open(file_path, "rb") as f:
        up = requests.put(file_url, data=f, timeout=120)
    up.raise_for_status()

    # 3. 轮询解析结果
    result_url = f"{BASE_URL}/api/v4/extract-results/batch/{batch_id}"
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
        r = requests.get(result_url, headers=headers, timeout=30)
        rd = r.json()
        if rd.get("code") != 0:
            raise ValueError(f"MinerU API 查询失败: {rd.get('msg', rd)}")
        results = rd["data"].get("extract_result") or []
        if not results:
            continue
        result = results[0]
        state = result.get("state")
        if state == "done":
            full_zip_url = (
                result.get("full_zip_url")
                or result.get("zip_url")
                or result.get("result_url")
            )
            if not full_zip_url:
                raise ValueError(f"MinerU API 完成但无下载地址: {result}")
            return _download_markdown(full_zip_url)
        if state == "failed":
            raise ValueError(f"MinerU API 解析失败: {result.get('err_msg', result)}")
    raise ValueError("MinerU API 解析超时")


def _download_markdown(zip_url: str) -> str:
    """下载结果 zip，解压读取 full.md"""
    resp = requests.get(zip_url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith("full.md") or n == "full.md"]
        if not names:
            raise ValueError("MinerU 结果 zip 中无 full.md")
        return zf.read(names[0]).decode("utf-8")


def is_api_enabled() -> bool:
    """是否配置了 MinerU API Token"""
    return bool(settings.mineru_api_token)
