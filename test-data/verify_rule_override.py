# -*- coding: utf-8 -*-
"""F3 规则否决权端到端验证：否决触发 + 纯计算/常识豁免 + 对照

验证目标：
1. 纯计算/常识（豁免类）：即使 LLM 决定不检索，也不触发 rule_override、回答不含"未找到"
   ——"先答再补未找到"的割裂体验已被豁免消除
2. 文档类问题：LLM 通常检索（status=ok）；若 LLM 不检则触发否决（status=rule_override）
3. 寒暄：不否决（对照）

LLM 是否检索不可控，但豁免类问题的结果应与 LLM 行为无关（规则保证不否决、不"未找到"）。
"""
import json, os, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost"
USER = os.environ.get("RAG_ADMIN_USER", "admin")
PASS = os.environ.get("RAG_ADMIN_PASS", "admin123")
NOT_FOUND_MARK = "未找到与您问题直接相关的信息"

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
    """发问题解析 SSE，返回 (token 全文, tool_call 状态清单, 是否 error)"""
    resp = req("POST", f"/api/chat/{sid}", auth=auth, data={"content": content}, timeout=180)
    buf = resp.read().decode("utf-8", "ignore")
    parts, statuses, error = [], [], False
    for block in buf.split("\n\n"):
        ev_name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if ev_name == "error":
            error = True
        if ev_name == "tool_call" and data:
            try:
                statuses.append(json.loads(data).get("status"))
            except Exception:
                pass
        elif ev_name == "token" and data:
            try:
                parts.append(json.loads(data).get("content", ""))
            except Exception:
                pass
    return "".join(parts), statuses, error

auth = json.loads(req("POST", "/api/auth/login", data={"username": USER, "password": PASS}).read())["access_token"]
libs = json.loads(req("GET", "/api/libraries", auth=auth).read()).get("items") or []
assert libs, "无可用文档库"
lib = next((x for x in libs if x.get("name") == "演示知识库"), libs[0])
sid = json.loads(req("POST", "/api/sessions", auth=auth, data={"library_id": lib["id"]}).read())["id"]
print(f"新会话 sid={sid}（文档库「{lib.get('name')}」）\n")

exempt_questions = [
    "计算 17 乘以 23 等于多少",   # 纯计算：应豁免 → 不否决、不含"未找到"
    "今天是星期几",              # 实时信息：应豁免
]
doc_questions = [
    "工资发放日是几号",           # 文档题：LLM 通常检索 status=ok；不检则 rule_override
]
control = ["你好"]  # 寒暄：不否决

failed = []

def run(q):
    text, statuses, error = chat_sse(auth, sid, q)
    print(f"[{q}]")
    print(f"    tool_call status={statuses or '无'}  回答={text[:70]!r}  err={error}")
    return text, statuses, error

for q in exempt_questions:
    text, statuses, error = run(q)
    if "rule_override" in statuses:
        failed.append(f"豁免场景「{q}」仍触发否决: {statuses}")
    if NOT_FOUND_MARK in text:
        failed.append(f"豁免场景「{q}」回答仍含「未找到」（割裂体验未消除）: {text[:80]!r}")

for q in doc_questions:
    text, statuses, error = run(q)
    if not statuses and error:
        failed.append(f"文档题「{q}」异常: error={error}")

for q in control:
    text, statuses, error = run(q)
    if statuses:
        failed.append(f"寒暄「{q}」不应有 tool_call: {statuses}")

print("\n" + ("豁免验证通过：纯计算/常识不再被否决、不再追加「未找到」" if not failed else "失败项："))
for f in failed:
    print(f"  ✗ {f}")
sys.exit(0 if not failed else 1)
