# -*- coding: utf-8 -*-
"""一键生成示例数据：登录 → 重建"示例知识库" → 上传 4 份示例文档 → 轮询就绪 → 打印场景问题清单

幂等可重跑：每次运行会删除已存在的"示例知识库"并重建（chat_sessions/documents/chunks 外键
CASCADE 级联清理，Milvus 库 partition 一并删除），保证客观可复现。

示例文档用 .md（抽取走明文读取，无需 MinerU），只需 MySQL + Milvus + backend 容器。

运行（需服务已启动，对 http://localhost）：
    python test-data/seed_example.py

凭据从环境变量读取（默认 admin/admin123），避免脚本内硬编码：
    RAG_ADMIN_USER / RAG_ADMIN_PASS
"""
import json
import os
import sys
import time
import urllib.request
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 统一 UTF-8，避免 Windows 控制台 GBK 乱码

BASE = "http://localhost"
LIB_NAME = "示例知识库"

# 示例文档：与 README"八、示例场景"一一对应，文件名即上传后的文档名
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
DEMO_DOCS = [
    "员工考勤管理制度.md",
    "Docker环境安装部署手册.md",
    "产品发布上线流程规范.md",
    "客户数据保密协议.md",
]

# 场景问题清单（照 README 示例场景设计，供复制到前端聊天页提问）
SCENARIOS = [
    ("场景 1 · 事实问答", "公司规定请事假需要提前几天申请？",
     "答案命中考勤制度，带 [来源N] 引用；溯源卡片显示“员工考勤管理制度 > 请假管理”标题路径"),
    ("场景 2 · 技术检索", "Docker 的常用命令有哪些？",
     "答案列出手册中的命令，答案逐字流式返回、思考过程折叠展示"),
    ("场景 3 · 无命中兜底", "工资发放日是几号？",
     "文档未写，如实回答“文档中未找到相关信息”，不编造、不出现空引用卡片"),
    ("场景 4 · 跨文档总结", "分别用一句话总结四份文档的核心内容",
     "答案基于多份文档总结，溯源卡片含多份文档来源"),
    ("场景 5 · 闲聊引导", "你好",
     "助手自我介绍并引导提问，不检索、不编造“文档查到”"),
    ("场景 6 · 多轮追问", "请事假扣工资吗？→ 年假有几天？→ 缺勤会怎样？",
     "连续追问上下文连贯；提问超过 10 轮后早期对话自动压缩为摘要，不丢早期上下文"),
]

USER = os.environ.get("RAG_ADMIN_USER", "admin")
PASS = os.environ.get("RAG_ADMIN_PASS", "admin123")


def req(method, path, auth=None, data=None, files=None, timeout=180):
    """HTTP 请求封装：支持 JSON body 与 multipart 文件上传，返回 (status, raw_text)"""
    headers = {}
    if auth:
        headers["Authorization"] = "Bearer " + auth
    body = None
    if files is not None:
        # multipart/form-data：chunk_size/overlap_token 表单字段 + file 文件字段
        boundary = uuid.uuid4().hex
        with open(files, "rb") as f:
            content = f.read()
        filename = os.path.basename(files)
        body = b"--" + boundary.encode() + b"\r\n"
        body += b'Content-Disposition: form-data; name="chunk_size"\r\n\r\n1024\r\n'
        body += b"--" + boundary.encode() + b"\r\n"
        body += b'Content-Disposition: form-data; name="overlap_token"\r\n\r\n102\r\n'
        body += b"--" + boundary.encode() + b"\r\n"
        body += (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
            + b"Content-Type: text/markdown\r\n\r\n"
            + content
            + b"\r\n"
        )
        body += b"--" + boundary.encode() + b"--\r\n"
        headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
    elif data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode()
            return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def login():
    """登录返回 access_token"""
    code, raw = req("POST", "/api/auth/login", data={"username": USER, "password": PASS})
    if code != 200:
        print(f"!! 登录失败（{code}）: {raw[:200]}", file=sys.stderr)
        sys.exit(1)
    print(f"[1] 登录 OK（{USER}）")
    return json.loads(raw)["access_token"]


def reset_library(auth):
    """幂等清理：删除已存在的“示例知识库”（外键 CASCADE 级联清理会话/文档/向量）"""
    code, raw = req("GET", "/api/libraries?page=1&page_size=100", auth=auth)
    libs = json.loads(raw).get("items") or []
    old = [lb for lb in libs if lb["name"] == LIB_NAME]
    for lb in old:
        code, _ = req("DELETE", f"/api/libraries/{lb['id']}", auth=auth)
        print(f"[2] 删除旧库 id={lb['id']}（{lb['name']}），status={code}")
    if not old:
        print("[2] 无旧示例库，直接新建")


def create_library(auth):
    """创建示例库并返回 library_id"""
    code, raw = req("POST", "/api/libraries", auth=auth,
                    data={"name": LIB_NAME, "description": "一键示例数据：4 份中文文档，支撑 6 个示例场景"})
    if code != 201:
        print(f"!! 建库失败（{code}）: {raw[:300]}", file=sys.stderr)
        sys.exit(1)
    lib = json.loads(raw)
    print(f"[3] 建库 OK id={lib['id']} name={lib['name']}")
    return lib["id"]


def upload_docs(auth, library_id):
    """逐个上传示例文档，返回 {文件名: 文档id}"""
    doc_ids = {}
    for name in DEMO_DOCS:
        path = os.path.join(EXAMPLES_DIR, name)
        if not os.path.exists(path):
            print(f"!! 示例文档缺失: {path}", file=sys.stderr)
            sys.exit(1)
        code, raw = req("POST", f"/api/libraries/{library_id}/documents", auth=auth, files=path)
        if code != 201:
            print(f"!! 上传 {name} 失败（{code}）: {raw[:300]}", file=sys.stderr)
            sys.exit(1)
        doc = json.loads(raw)
        doc_ids[name] = doc["id"]
        print(f"[4] 上传 OK {name} -> id={doc['id']}")
    return doc_ids


def wait_ready(auth, doc_ids):
    """轮询所有文档至 ready；任一 failed 则以非 0 退出"""
    pending = set(doc_ids.values())
    for i in range(60):  # 最多 60 × 3s = 180s
        for doc_id in list(pending):
            code, raw = req("GET", f"/api/documents/{doc_id}/status", auth=auth)
            st = json.loads(raw)
            status = st.get("status")
            if status == "failed":
                print(f"!! 文档 {doc_id} 处理失败: {st.get('error_message')}", file=sys.stderr)
                sys.exit(1)
            if status == "ready":
                pending.discard(doc_id)
                print(f"[5] 就绪 doc_id={doc_id} chunks={st.get('chunk_count')}")
        if not pending:
            return
        print(f"[5] 等待处理中…（剩余 {len(pending)} 个文档，processed={st.get('processed_chunks')}/{st.get('chunk_count')}）")
        time.sleep(3)
    print(f"!! 处理超时，未就绪: {pending}", file=sys.stderr)
    sys.exit(1)


def print_scenarios():
    """打印示例场景问题清单（复制到前端聊天页提问）"""
    print("\n" + "=" * 68)
    print("示例就绪。请在浏览器打开 http://localhost，以 admin 登录，")
    print("进入“聊天问答”选择文档库【示例知识库】，新建会话后复制以下问题提问：")
    print("=" * 68)
    for title, question, expect in SCENARIOS:
        print(f"\n{title}")
        print(f"  提问：{question}")
        print(f"  预期：{expect}")


def main():
    auth = login()
    reset_library(auth)
    library_id = create_library(auth)
    doc_ids = upload_docs(auth, library_id)
    wait_ready(auth, doc_ids)
    print_scenarios()


if __name__ == "__main__":
    main()
