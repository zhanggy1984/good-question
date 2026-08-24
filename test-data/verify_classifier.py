# -*- coding: utf-8 -*-
"""规则意图分类器端到端验证：SSE 链路确认运行中服务行为
场景1 "最近怎么样"（寒暄）→ 应走 LLM 引导话术（usage > 0，token 非固定话术）
场景2 "工资发放日是几号"（事实查询）→ 固定话术，不调 LLM（usage 全 0）
"""
import json, os, sys, urllib.request
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
    """发问题并解析 SSE，返回 (事件类型列表, token 全文, usage dict)

    契约 v1.0：event:/data: 成对，JSON 无 type 字段，事件类型在 event: 行。
    """
    resp = req("POST", f"/api/chat/{sid}", auth=auth, data={"content": content}, timeout=180)
    buf = resp.read().decode("utf-8", "ignore")
    types, token_parts, usage = [], [], None
    for block in buf.split("\n\n"):
        ev_name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if ev_name is None and data is None:
            continue
        if ev_name == "done":
            types.append("done")
            continue
        if not data:
            continue
        try:
            ev = json.loads(data)
        except Exception:
            continue
        if ev_name == "token":
            token_parts.append(ev.get("content", ""))
        elif ev_name == "usage":
            usage = ev  # data 行本身即 {prompt_tokens, completion_tokens, total_tokens, ts}
        if ev_name:
            types.append(ev_name)
    return types, "".join(token_parts), usage

auth = json.loads(req("POST", "/api/auth/login", data={"username": USER, "password": PASS}).read())["access_token"]
libs = json.loads(req("GET", "/api/libraries", auth=auth).read()).get("items") or []
assert libs, "无可用文档库"
lib_id = libs[0]["id"]

def check(name, content, expect_llm, expect_text=None):
    sid = json.loads(req("POST", "/api/sessions", auth=auth, data={"library_id": lib_id}).read())["id"]
    types, text, usage = chat_sse(auth, sid, content)
    llm_called = usage and (usage.get("total_tokens") or 0) > 0
    text_ok = (expect_text is None) or (expect_text in text)
    status = "PASS" if (llm_called == expect_llm and text_ok) else "FAIL"
    print(f"[{status}] {name}: llm_called={llm_called} text_ok={text_ok} usage={usage}")
    print(f"    token 前 80 字: {text[:80]!r}")
    print(f"    事件序: {'→'.join(types)}")
    return status == "PASS"

ok = True
ok &= check("场景1 寒暄'最近怎么样'→走LLM", "最近怎么样", expect_llm=True)
ok &= check("场景2 事实查询'工资发放日是几号'→固定话术", "工资发放日是几号",
            expect_llm=False, expect_text="未找到与您问题直接相关的信息")
ok &= check("场景3 unknown'天气'→澄清话术(非未找到)", "天气",
            expect_llm=False, expect_text="还没完全理解您的问题")
print("\n" + ("全部通过" if ok else "存在失败"))
sys.exit(0 if ok else 1)
