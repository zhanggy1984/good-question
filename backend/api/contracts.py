"""4.3 标准契约清单端点（平台定标准，agent 适配）。

统一 `GET /api/contracts`（公开无鉴权），声明本 agent 的 LLM 评测接口与场景清单。
平台脚手架读此端点做接口自动发现（决策 #55/#56）。llm=false 为辅助接口（登录等），
只登记不进 agent_interface。interfaces[].path 为业务路径，与平台 seed_data 一致。
"""
from fastapi import APIRouter

router = APIRouter(prefix="/contracts", tags=["contracts"])

MANIFEST = {
    "agent": "good-question",
    "contract_version": "1.0",
    "interfaces": [
        {"name": "chat", "path": "/api/chat/{session_id}", "method": "POST",
         "contract_type": "sse", "llm": True,
         "description": "知识问答（SSE 流式，token 事件经 field_map 映射 answer）"},
        {"name": "login", "path": "/api/auth/login", "method": "POST",
         "llm": False, "description": "会话鉴权（辅助接口）"},
    ],
    "scenes": [
        {"tag": "greeting", "description": "问候与闲聊"},
        {"tag": "doc_qa", "description": "文档检索问答"},
        {"tag": "no_hit", "description": "无命中/意图不明兜底（不调 LLM，如实回复或引导澄清）"},
        {"tag": "summarize", "description": "文档内容总结"},
    ],
}


@router.get("", summary="标准契约清单")
async def contracts() -> dict:
    return MANIFEST
