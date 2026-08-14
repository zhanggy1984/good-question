# -*- coding: utf-8 -*-
"""旧数据迁移验证：登录→查库→对旧库提问→读 SSE sources（验证 MySQL 迁移 + Milvus 重灌后检索链路）"""
import json, os, re, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost"

# 登录凭据从环境变量读取（本地开发默认 admin/admin123），避免脚本内硬编码
USER = os.environ.get("RAG_ADMIN_USER", "admin")
PASS = os.environ.get("RAG_ADMIN_PASS", "admin123")


def req(method, path, auth=None, data=None, timeout=180):
    headers = {}
    if auth:
        headers["Authorization"] = "Bearer " + auth
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    return urllib.request.urlopen(r, timeout=timeout)


# 1. 登录
auth = json.loads(req("POST", "/api/auth/login", data={"username": USER, "password": PASS}).read())["access_token"]
print("[1] 登录 OK")

# 2. 查库列表，确认旧库存在
libraries = json.loads(req("GET", "/api/libraries", auth=auth).read())
items = libraries["items"] if isinstance(libraries, dict) else libraries
for lib in items:
    print(f"   库 id={lib['id']} name={lib['name']}")
if not any(l["id"] == 5 for l in items):
    print("!! 未找到旧库 id=5"); sys.exit(1)

# 3. 建会话（绑定旧库 id=5）
sess = json.loads(req("POST", "/api/sessions", auth=auth, data={"library_id": 5}).read())
sid = sess["id"]
print(f"[2] 会话创建 OK id={sid}")

# 4. 提问（基于旧库 README.md / guide.md 内容），读 SSE 事件流
resp = req("POST", f"/api/chat/{sid}", auth=auth, data={"content": "AI agent 项目如何安装依赖和环境配置？"})
buf = ""
for raw in resp:
    buf += raw.decode("utf-8", "ignore")

print("[3] SSE 原始事件流（前 2000 字符）:")
print(buf[:2000])

# 5. 解析 sources 事件（SSE 格式：event: sources\ndata: {json}\n\n）
m = re.search(r"event: sources\r?\ndata: (\{.*?\})\r?\n\r?\n", buf, re.DOTALL)
print("\n[4] sources 事件存在:", m is not None)
if m:
    srcs = json.loads(m.group(1))["sources"]
    print(f"    溯源数量: {len(srcs)}")
    for s in srcs[:3]:
        print(f"    - {s.get('document_name')} | heading={s.get('heading_path')} | chunk={s.get('chunk_index')}")
else:
    print("    sources 未匹配，原始片段:", buf[:300])
