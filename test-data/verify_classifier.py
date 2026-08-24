# -*- coding: utf-8 -*-
"""二期 function calling 编排端到端验证：SSE 链路确认运行中服务行为

二期所有请求第一轮必调 LLM（判断是否检索），usage 恒 > 0；tool_call 由 LLM 决定才出现。
场景1 "最近怎么样"（寒暄）→ 期望 LLM 决定不检索，直接答（无 tool_call）
场景2 "工资发放日是几号"（事实查询）→ 期望 LLM 决定检索（tool_call 出现）；命中则答案+溯源，
       空则"未找到"固定话术（规则三路兜底）
场景3 "天气"（意图不明）→ 期望 LLM 决策；若检索空则走澄清话术（"还没完全理解"）
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
    """发问题并解析 SSE，返回 (事件类型列表, token 全文, usage dict, 是否出现 tool_call)

    契约 v1.0：event:/data: 成对，JSON 无 type 字段，事件类型在 event: 行。
    """
    resp = req("POST", f"/api/chat/{sid}", auth=auth, data={"content": content}, timeout=180)
    buf = resp.read().decode("utf-8", "ignore")
    types, token_parts, usage, tool_called = [], [], None, False
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
        if ev_name == "tool_call":
            tool_called = True
        elif ev_name == "token":
            token_parts.append(ev.get("content", ""))
        elif ev_name == "usage":
            usage = ev
        if ev_name:
            types.append(ev_name)
    return types, "".join(token_parts), usage, tool_called

auth = json.loads(req("POST", "/api/auth/login", data={"username": USER, "password": PASS}).read())["access_token"]
libs = json.loads(req("GET", "/api/libraries", auth=auth).read()).get("items") or []
assert libs, "无可用文档库"
lib_id = libs[0]["id"]

def check(name, content, expect_tool=None, expect_text=None):
    """expect_tool: None=不限制 LLM 决策；True/False 断言 tool_call 是否出现"""
    sid = json.loads(req("POST", "/api/sessions", auth=auth, data={"library_id": lib_id}).read())["id"]
    types, text, usage, tool_called = chat_sse(auth, sid, content)
    llm_called = usage and (usage.get("total_tokens") or 0) > 0
    text_ok = (expect_text is None) or (expect_text in text)
    tool_ok = (expect_tool is None) or (tool_called == expect_tool)
    status = "PASS" if (llm_called and tool_ok and text_ok and "error" not in types) else "FAIL"
    print(f"[{status}] {name}: tool_called={tool_called} llm_called={llm_called} "
          f"tool_ok={tool_ok} text_ok={text_ok} usage={usage}")
    print(f"    token 前 80 字: {text[:80]!r}")
    print(f"    事件序: {'→'.join(types)}")
    return status == "PASS"

ok = True
# 场景1 寒暄：LLM 应判"与文档无关"不检索，直接答（system prompt 规则 2）
ok &= check("场景1 寒暄'最近怎么样'→不检索直接答", "最近怎么样", expect_tool=False)
# 场景2 事实查询：LLM 应决定检索（system prompt 规则 1）；空则"未找到"固定话术
ok &= check("场景2 事实查询'工资发放日是几号'→LLM决定检索", "工资发放日是几号", expect_tool=True)
# 场景3 意图不明：LLM 决策；检索空应走澄清话术（与"未找到"区分）
ok &= check("场景3 unknown'天气'→检索空则澄清话术", "天气", expect_text="还没完全理解")
print("\n" + ("全部通过" if ok else "存在失败"))
sys.exit(0 if ok else 1)
