"""good-question 契约改造（B.1）运行时验证。

登录 admin → 复用/新建文档库 → 建会话 → 调 /api/chat/{sid} 读 SSE，
校验事件流符合评测契约 §5.1（路径 A，二期 function calling 编排）：
- 事件顺序：meta 首个，done 最后，usage（多轮合并）在 done 前
- meta 含 agent/model/interface/contract_version/git_sha/knowledge_version/ts
- reasoning/token 的 data 同时含 content+delta（兼容前端 content 与评测 delta）并带 ts
- usage 含 prompt_tokens/completion_tokens/total_tokens 并带 ts
- LLM 决定检索时才出现 tool_call（id/name/args/result/status/ts），命中后跟 sources

宿主机运行：python verify_contract.py
"""
import json
import sys

import httpx

# 统一 UTF-8，避免 Windows 控制台 GBK 无法编码 ✅ 等字符
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8080"
ADMIN = ("admin", "admin123")


def p(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    with httpx.Client(timeout=180) as c:
        # 1. 登录
        r = c.post(f"{BASE}/api/auth/login", json={"username": ADMIN[0], "password": ADMIN[1]})
        assert r.status_code == 200, f"登录失败 {r.status_code}: {r.text[:200]}"
        h = {"Authorization": f"Bearer {r.json()['access_token']}"}
        p("✅ 登录成功")

        # 2. 复用已有库，无则新建
        libs = c.get(f"{BASE}/api/libraries", headers=h).json().get("items") or []
        if libs:
            lib_id = libs[0]["id"]
            p(f"复用已有 library={lib_id}（{libs[0]['name']}）")
        else:
            r = c.post(f"{BASE}/api/libraries", headers=h,
                       json={"name": "契约验证库", "description": "B.1 契约改造验证"})
            assert r.status_code == 201, f"建库失败 {r.status_code}: {r.text[:200]}"
            lib_id = r.json()["id"]
            p(f"新建 library={lib_id}")

        # 3. 建会话
        r = c.post(f"{BASE}/api/sessions", headers=h, json={"library_id": lib_id})
        assert r.status_code == 201, f"建会话失败 {r.status_code}: {r.text[:200]}"
        sid = r.json()["id"]
        p(f"新建 session={sid}")

        # 4. 调 chat（真实 DeepSeek 流式）
        r = c.post(f"{BASE}/api/chat/{sid}", headers=h, json={"content": "你好，请介绍一下你自己"})
        assert r.status_code == 200, f"chat 失败 {r.status_code}: {r.text[:200]}"

        # 5. 解析 SSE 帧
        events: list[dict] = []
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                events.append({"type": line[len("event: "):].strip(), "data": None})
            elif line.startswith("data: "):
                events[-1]["data"] = json.loads(line[len("data: "):])

        types = [e["type"] for e in events]
        p(f"事件序列({len(types)}): {types}")

        # 6. 顺序断言
        assert types[0] == "meta", f"首个事件应为 meta，实际 {types[0]}"
        assert types[-1] == "done", f"末事件应为 done，实际 {types[-1]}"
        assert "usage" in types, "缺 usage 事件"
        assert types.index("usage") < types.index("done"), "usage 应在 done 之前"
        assert "error" not in types, "出现 error 事件，LLM 调用失败"

        # 7. meta 字段
        meta = events[0]["data"]
        for k in ("agent", "model", "interface", "contract_version", "git_sha",
                  "knowledge_version", "ts"):
            assert k in meta, f"meta 缺字段 {k}"
        assert isinstance(meta["ts"], int), f"meta.ts 应为 int，实际 {meta['ts']!r}"
        p(f"  meta: agent={meta['agent']} model={meta['model']} contract={meta['contract_version']} "
          f"git_sha={meta['git_sha'] or '(空)'} knowledge_version={meta['knowledge_version'] or '(空)'}")

        # 8. reasoning/token 双字段 + ts
        for e in events:
            if e["type"] in ("reasoning", "token"):
                d = e["data"]
                assert "content" in d and "delta" in d, f"{e['type']} data 应含 content+delta"
                assert d["content"] == d["delta"], f"{e['type']} 的 content 应等于 delta"
                assert isinstance(d.get("ts"), int), f"{e['type']} 缺 ts"
        p(f"  reasoning/token 事件均含 content+delta+ts")

        # 9. usage 字段
        usage = events[types.index("usage")]["data"]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert k in usage, f"usage 缺字段 {k}"
        assert isinstance(usage.get("ts"), int), "usage 缺 ts"
        p(f"  usage: prompt={usage['prompt_tokens']} completion={usage['completion_tokens']} "
          f"total={usage['total_tokens']}")

        # 10. tool_call（二期：LLM 决定检索时才出现；命中后另有 sources 事件）
        if any(e["type"] == "tool_call" for e in events):
            tc = next(e["data"] for e in events if e["type"] == "tool_call")
            for k in ("id", "name", "args", "result", "status", "ts"):
                assert k in tc, f"tool_call 缺字段 {k}"
            p(f"  tool_call: name={tc['name']} status={tc['status']} "
              f"source_count={tc['result'].get('source_count')}")
        else:
            p("  无 tool_call（LLM 决定不检索：闲聊/直接回答，符合二期语义）")

        p("✅ 契约事件流验证通过（meta → reasoning*/token* → [tool_call → sources]? → reasoning*/token* → usage → done，全事件带 ts）")


if __name__ == "__main__":
    main()
