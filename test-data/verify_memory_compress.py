# -*- coding: utf-8 -*-
"""多轮记忆压缩端到端验证：超过 10 轮触发 _compress_memory，旧消息压成摘要

机制（chat_service.py）：
- MAX_MESSAGES_BEFORE_COMPRESS=20：消息记录 >20 条（11 轮 = 22 条）时触发
- KEEP_RECENT_MESSAGES=6：保留最近 3 轮原文，更早消息 + 现有摘要 → LLM 压缩为新摘要（≤200字）
- 压缩发生在 stream_chat done 前；前端始终显示全部历史，压缩只影响后端传给 LLM 的 history

验证路径：
1. 一个会话连发 11 轮（3 轮实质文档查询 + 8 轮寒暄凑数），第 11 轮后压缩应触发
2. 查 DB chat_sessions.summary 非空（压缩成功）
3. 再发 1 轮"我们最早聊的话题"——模型应基于摘要回述"事假/请假"，证明摘要承载早期记忆
"""
import json, os, re, subprocess, sys, urllib.request
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost"
USER = os.environ.get("RAG_ADMIN_USER", "admin")
PASS = os.environ.get("RAG_ADMIN_PASS", "admin123")

def req(method, path, auth=None, data=None, timeout=180):
    headers = {}
    if auth: headers["Authorization"] = "Bearer " + auth
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(BASE + path, data=body, headers=headers, method=method)
    return urllib.request.urlopen(r, timeout=timeout)

def chat_sse(auth, sid, content):
    """发问题并解析 SSE，返回 (token 全文, 是否出现 tool_call)"""
    resp = req("POST", f"/api/chat/{sid}", auth=auth, data={"content": content}, timeout=180)
    buf = resp.read().decode("utf-8", "ignore")
    token_parts, tool_called, error = [], False, False
    for block in buf.split("\n\n"):
        ev_name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if ev_name == "error":
            error = True
            continue
        if ev_name == "tool_call":
            tool_called = True
        elif ev_name == "token" and data:
            try:
                token_parts.append(json.loads(data).get("content", ""))
            except Exception:
                pass
    return "".join(token_parts), tool_called, error

def mysql_summary(sid):
    """docker exec 查 chat_sessions.summary（密码从项目根 .env 读）"""
    env = {}
    for line in (Path(__file__).resolve().parent.parent / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    pw = env.get("MYSQL_ROOT_PASSWORD", "")
    db = env.get("MYSQL_DATABASE", "native_rag")
    out = subprocess.run(
        ["docker", "exec", "rag-mysql", "mysql", "-uroot", f"-p{pw}", "-N", "-D", db, "-e",
         f"SELECT summary FROM chat_sessions WHERE id={sid}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    # 只取 stdout：stderr 恒有 "password on command line" warning，fallback 会造成假阳性；
    # text=True 默认按 GBK 解码，UTF-8 中文会崩，须显式 encoding="utf-8"
    return out.stdout.strip()

auth = json.loads(req("POST", "/api/auth/login", data={"username": USER, "password": PASS}).read())["access_token"]
libs = json.loads(req("GET", "/api/libraries", auth=auth).read()).get("items") or []
assert libs, "无可用文档库"
# 优先选"示例知识库"（含员工考勤管理制度.md，实质问题可命中）；libs[0] 未必是它
lib = next((x for x in libs if x.get("name") == "示例知识库"), libs[0])
lib_id = lib["id"]
sid = json.loads(req("POST", "/api/sessions", auth=auth, data={"library_id": lib_id}).read())["id"]
print(f"新会话 sid={sid}（文档库 {lib_id}「{lib.get('name')}」）\n")

qs = [
    "公司规定请事假需要提前几天申请？",  # r1 实质（应检索命中）
    "那病假呢？",                         # r2 实质（应检索命中）
    "加班费怎么算？",                     # r3 实质（应检索命中）
] + ["你好"] * 8                          # r4-r11 寒暄凑数（不检索，快）

for i, q in enumerate(qs, 1):
    text, tool_called, error = chat_sse(auth, sid, q)
    print(f"第 {i:2d} 轮 [{q[:14]:<14}] tool_call={tool_called!s:<5} err={error} 回答={text[:40]!r}")
    if error:
        print(f"   !! 第 {i} 轮出现 error，终止"); sys.exit(1)

# 11 轮 = 22 条消息 > 20，压缩应已触发
summary = mysql_summary(sid)
print(f"\n压缩触发后 chat_sessions.summary: {summary[:120]!r}")
print(f"summary 非空: {bool(summary)}")

# 再发 1 轮：验证摘要承载早期记忆（早期"事假"信息应能回述）
text, tool_called, error = chat_sse(auth, sid, "我们刚才聊了什么？我最早问的是哪个话题？")
print(f"\n记忆回述轮回答: {text[:200]!r}")
print(f"回述提到请假相关: {'事假' in text or '病假' in text or '请假' in text}")

ok = bool(summary) and not error and ("事假" in text or "病假" in text or "请假" in text)
print("\n" + ("压缩验证通过" if ok else "压缩验证失败"))
sys.exit(0 if ok else 1)
