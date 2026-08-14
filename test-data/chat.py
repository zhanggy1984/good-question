# -*- coding: utf-8 -*-
"""chat 完整链路验证：登录→建会话→提问→读 SSE 事件"""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 统一 UTF-8，避免 Windows 控制台 GBK 乱码

BASE = "http://localhost"

# 登录凭据从环境变量读取（本地开发默认 admin/admin123），避免脚本内硬编码
USER = os.environ.get("RAG_ADMIN_USER", "admin")
PASS = os.environ.get("RAG_ADMIN_PASS", "admin123")

def req(method, path, auth=None, data=None, timeout=120):
    headers = {}
    if auth: headers["Authorization"] = "Bearer " + auth
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    return urllib.request.urlopen(r, timeout=timeout)

auth = json.loads(req("POST", "/api/auth/login", data={"username": USER, "password": PASS}).read())["access_token"]
print("[1] 登录 OK")

sess = json.loads(req("POST", "/api/sessions", auth=auth, data={"library_id": 1}).read())
sid = sess["id"]
print("[2] 会话创建 OK id=", sid)

print("[3] 提问（SSE 事件流）...")
resp = req("POST", f"/api/chat/{sid}", auth=auth, data={"content": "如何安装docker环境"}, timeout=180)
buf = ""
for raw in resp:
    buf += raw.decode("utf-8", "ignore")
events = buf.split("data:")
print("[4] SSE 原始事件片段（前 600 字符）:")
print(buf[:600])
