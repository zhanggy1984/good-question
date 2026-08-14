# -*- coding: utf-8 -*-
"""端到端验证：登录→建库→上传→轮询就绪（验证 Milvus 写入链路）"""
import json, os, sys, time, uuid, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 统一 UTF-8，避免 Windows 控制台 GBK 乱码

BASE = "http://localhost"

# 登录凭据从环境变量读取（本地开发默认 admin/admin123），避免脚本内硬编码
USER = os.environ.get("RAG_ADMIN_USER", "admin")
PASS = os.environ.get("RAG_ADMIN_PASS", "admin123")


def req(method, path, auth=None, data=None, files=None):
    headers = {}
    if auth:
        headers["Authorization"] = "Bearer " + auth
    body = None
    if files is not None:
        boundary = uuid.uuid4().hex
        with open(files, "rb") as f:
            content = f.read()
        body = b"--" + boundary.encode() + b"\r\n"
        body += b'Content-Disposition: form-data; name="file"; filename="test.md"\r\n'
        body += b"Content-Type: text/markdown\r\n\r\n" + content + b"\r\n"
        body += b"--" + boundary.encode() + b"--\r\n"
        headers["Content-Type"] = "multipart/form-data; boundary=" + boundary
    elif data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            raw = resp.read().decode()
            return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# 1. 登录
_, raw = req("POST", "/api/auth/login", data={"username": USER, "password": PASS})
auth = json.loads(raw)["access_token"]
print("[1] 登录 OK")

# 2. 建库
code, raw = req("POST", "/api/libraries", auth=auth, data={"name": "Milvus验证库", "description": "端到端"})
print("[2] 建库:", code, raw[:200])
lib = json.loads(raw)
lib_id = lib["id"]

# 3. 上传
code, raw = req("POST", f"/api/libraries/{lib_id}/documents", auth=auth, files="test-data/milvus升级验证文档.md")
print("[3] 上传:", code, raw[:300])
doc = json.loads(raw)
doc_id = doc["id"]

# 4. 轮询状态
for i in range(40):
    code, raw = req("GET", f"/api/documents/{doc_id}/status", auth=auth)
    st = json.loads(raw)
    print(f"   轮询 {i+1}: status={st['status']} processed={st.get('processed_chunks')} chunks={st.get('chunk_count')}")
    if st["status"] in ("ready", "failed"):
        if st["status"] == "failed":
            print("!! 处理失败:", st.get("error_message"))
        break
    time.sleep(3)
print("[4] 文档最终状态:", st["status"])
