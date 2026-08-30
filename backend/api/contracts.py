"""4.3 标准契约清单端点（平台定标准，agent 适配）。

统一 `GET /api/contracts`（公开无鉴权），声明本 agent 的 LLM 评测接口、场景清单与
**驱动契约（contract 段，manifest v2）**。平台脚手架读此端点做接口自动发现（决策
#55/#56）与 adapter 生成（{{input.*}}/{{auth.*}}/{{prepare.*}} 占位符由平台渲染）。
llm=false 为辅助接口（登录等），只登记不进 agent_interface。contract 段是平台驱动本
agent 的权威声明，改动需与平台 seed 快照保持同构（discover 会对比漂移）。
"""
from fastapi import APIRouter

router = APIRouter(prefix="/contracts", tags=["contracts"])

MANIFEST = {
    "agent": "good-question",
    "contract_version": "2.0",
    "interfaces": [
        {"name": "chat", "path": "/api/chat/{session_id}", "method": "POST",
         "contract_type": "sse", "llm": True,
         "description": "知识问答（SSE 流式，LLM 自主决定是否检索，检索经 tool_call/sources 事件回传；token 事件经 field_map 映射 answer）"},
        {"name": "login", "path": "/api/auth/login", "method": "POST",
         "llm": False, "description": "会话鉴权（辅助接口）"},
    ],
    "scenes": [
        {"tag": "greeting", "description": "问候与闲聊"},
        {"tag": "doc_qa", "description": "文档检索问答"},
        {"tag": "no_hit", "description": "无命中/意图不明兜底（检索空时如实回复或引导澄清）"},
        {"tag": "summarize", "description": "文档内容总结"},
    ],
    "contract": {
        "type": "sse", "timeout": 180,
        "prepare": [
            {"name": "login", "method": "POST", "path": "/api/auth/login",
             "body": {"username": "{{auth.username}}", "password": "{{auth.password}}"},
             "extract": {"token": "access_token"}},
            {"name": "session", "method": "POST", "path": "/api/sessions",
             "headers": {"Authorization": "Bearer {{prepare.login.token}}"},
             "body": {"library_id": 3},
             "extract": {"id": "id"}},
        ],
        "request": {
            "path": "/api/chat/{{prepare.session.id}}", "method": "POST",
            "headers": {"Authorization": "Bearer {{prepare.login.token}}",
                        "Content-Type": "application/json"},
            "body": {"content": "{{input.content}}", "stream": True},
        },
        "sse": {"field_map": {"token": "answer"}},  # 终答事件名是 token 非 answer
    },
}


@router.get("", summary="标准契约清单")
async def contracts() -> dict:
    return MANIFEST
