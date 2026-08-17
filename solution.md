# native-rag 技术方案（v2）

## Context

基于 PRD 需求，构建一个多用户 RAG 文档问答系统。用户上传文档到文档库，系统自动完成内容抽取→清洗→切片→向量化，用户可在绑定了文档库的聊天会话中提问，系统通过混合检索（语义 + ES 全文 → rerank）召回相关内容，由 DeepSeek 生成带溯源的流式回答。

## 关键决策摘要

| 决策项 | 选型 | 说明 |
|--------|------|------|
| 后端框架 | **FastAPI + LangChain 1.x** | LangChain 负责 AI 管线（切片/检索/LLM/流式），FastAPI 负责 HTTP 和路由 |
| 文档抽取 | **MinerU 3.4.4**（pipeline backend + torch CPU） | 结构化解析 PDF/DOCX，失败降级 PyMuPDF/python-docx |
| Embedding | **FastEmbed**（ONNX）加载 **bge-small-zh-v1.5**（512 维） | 无 torch 依赖；sentence-transformers 因 torch 官方源 403 放弃 |
| Rerank | **sentence-transformers CrossEncoder** 加载 bge-reranker-v2-m3 | 替代 FlagEmbedding（旧版且未装），同款模型 |
| 全文检索 | **Elasticsearch 8.13**（IK 中文分词，docker-compose 服务） | 替代 rank_bm25/jieba 内存索引，可无缝升级 ES 集群 / K8s（ECK） |
| 向量库 | **ChromaDB Server**（Docker 独立服务） | 通过 langchain-chroma 的 HttpClient 连接 |
| 关系库 | **MySQL 8.0**（Docker 独立服务） | SQLAlchemy + Alembic |
| 前端 | Vue 3 + Naive UI | SSE 流式渲染 |
| 部署 | Docker Compose **六服务** | mysql / chromadb / elasticsearch / backend / **nginx（前端静态文件 + 反向代理）** |
| 权限 | admin 管理文档库和文档 + 聊天；普通用户浏览 + 聊天，自行注册 | |

### 权限矩阵

| 操作 | admin | 普通用户 |
|------|-------|----------|
| 注册/登录 | ✅ | ✅（自行注册） |
| 查看仪表盘 | ✅ | ✅ |
| 浏览文档库列表 | ✅ | ✅（选库聊天需要） |
| 浏览文档列表 | ✅ | ✅（了解库内有哪些文档） |
| 创建/删除文档库 | ✅ | ❌ |
| 上传/删除文档 | ✅ | ❌ |
| 创建/删除聊天会话 | ✅ | ✅（仅自己的会话） |
| 聊天问答 | ✅ | ✅（仅自己的会话） |

### 账号管理

- **admin**：系统首次启动时根据 `.env` 中的 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 自动创建（若已存在则跳过种子）
- **普通用户**：通过 `/register` 页面自行注册，无需 admin 干预

---

## 1. 项目结构

```
native-rag/
├── docker-compose.yml
├── .env
├── .env.example
├── init.sql                          # MySQL 建表 DDL（不含种子数据）
├── backend/
│   ├── Dockerfile                    # Python 运行环境（无前端构建阶段）
│   ├── requirements.txt
│   ├── alembic.ini
│   ├── alembic/
│   │   └── versions/
│   ├── main.py                       # FastAPI 入口 + lifespan（启动时初始化模型/ES index/admin种子）
│   ├── config.py                     # pydantic-settings 读取 .env
│   ├── models/                       # SQLAlchemy ORM（6 张表，同 v1）
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── document_library.py
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── chat_session.py
│   │   └── chat_message.py
│   ├── schemas/                      # Pydantic 请求/响应
│   │   └── ...
│   ├── api/                          # FastAPI 路由（薄层）
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── dashboard.py
│   │   ├── library.py
│   │   ├── document.py
│   │   ├── session.py
│   │   └── chat.py
│   ├── services/                     # 业务逻辑（含 LangChain 组件）
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── dashboard_service.py
│   │   ├── library_service.py
│   │   ├── document_service.py       # 文档处理编排（MinerU → 清洗 → 切片 → 向量化）
│   │   ├── embedding_service.py      # LangChain HuggingFaceBgeEmbeddings 封装
│   │   ├── vector_store_service.py   # LangChain Chroma 封装 + collection 管理
│   │   ├── retrieval_service.py      # 混合检索器 HybridRetriever（LangChain BaseRetriever 子类）
│   │   ├── llm_service.py            # LangChain ChatOpenAI 封装（DeepSeek，模型名从 .env 读取）
│   │   └── chat_service.py           # RAG 链构建（LCEL）+ 对话记忆 + SSE 流式回调
│   ├── middleware/
│   │   └── auth.py                   # JWT + admin 校验 FastAPI Depends
│   └── utils/
│       ├── __init__.py
│       ├── mineru_extractor.py       # MinerU 封装（调用 magic-pdf）
│       ├── text_cleaner.py           # 文本预清洗
│       └── es_index.py               # ES 全文检索（ensure_index/bulk/search/delete）
├── elasticsearch/
│   └── Dockerfile                    # ES 官方镜像 + IK 中文分词插件
├── frontend/                         # Vue 3 前端（开发用 Vite，生产用 nginx）
│   ├── Dockerfile                    # 多阶段：node 编译 → nginx 服务静态文件
│   ├── nginx.conf                    # nginx 配置（gzip + /api 反向代理 + SPA fallback）
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── main.ts
│       ├── App.vue
│       ├── router/index.ts
│       ├── api/                      # Axios 封装
│       ├── stores/                   # Pinia 状态管理
│       ├── views/                    # 页面组件
│       ├── components/               # 通用组件
│       └── utils/sse.ts              # SSE 流式接收 + 断连重试
└── data/
    └── uploads/                      # 上传的原始文档（仅这一个本地持久化目录）
```

---

## 2. Chunk 元数据设计

每个 chunk 在 MySQL 和 ChromaDB 中都需要保存元数据，便于溯源、维护和增量更新。

### 2.1 元数据字段定义

```json
{
    // ── 来源定位 ──
    "document_id": 123,
    "document_name": "2024年度技术报告.pdf",
    "library_id": 1,
    "chunk_index": 5,
    "total_chunks": 20,

    // ── 结构溯源 ──
    "heading_path": ["第三章 技术架构", "3.2 微服务治理"],
    "heading_level": 2,
    "page_number": 42,
    "source_type": "paragraph",

    // ── 内容特征 ──
    "token_count": 480,
    "char_start": 15200,
    "char_end": 16750,
    "content_hash": "sha256:a1b2c3d4...",

    // ── 处理溯源 ──
    "created_at": "2026-08-04T10:30:00Z",
    "splitter": "heading_aware",
    "overlap_prev_chunk_id": null
}
```

### 2.2 字段说明

| 字段 | 类型 | 用途 |
|------|------|------|
| `document_id` | int | 关联 MySQL documents 表，向量库中按此删除/过滤 |
| `document_name` | str | 溯源时直接展示，无需回查 MySQL |
| `library_id` | int | 冗余，方便 ChromaDB 按库过滤（`where={"library_id": 1}`） |
| `chunk_index` | int | 文档内排序，溯源时展示位置 |
| `total_chunks` | int | 方便前端展示"第 5/20 片段" |
| `heading_path` | list[str] | 层级标题路径，溯源时展示"文档 → 第三章 → 3.2 微服务治理" |
| `heading_level` | int | 标题层级（1=一级标题，0=无标题段落），便于按重要性排序 |
| `page_number` | int | PDF 的页码，DOCX/MD 填 0 |
| `source_type` | enum | `heading` / `paragraph` / `table` / `list` / `code`，便于前端区分展示 |
| `token_count` | int | token 数，便于统计和监控 |
| `char_start/end` | int | 在清洗后全文中的字符偏移，便于定位原文 |
| `content_hash` | str | SHA256 前 8 位，用于增量更新时对比变更 |
| `splitter` | str | 记录用哪种策略切的（`heading_aware` / `recursive_tts`），便于排查质量问题 |
| `overlap_prev_chunk_id` | int\|null | 和前一个 chunk 有重叠时记录其 ID，便于追踪重叠关系 |

### 2.3 MySQL chunks 表更新

```sql
CREATE TABLE chunks (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    document_id   BIGINT NOT NULL,
    library_id    BIGINT NOT NULL,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    token_count   INT NOT NULL DEFAULT 0,
    metadata_json JSON NOT NULL,          -- 上述全部元数据
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_document (document_id),
    INDEX idx_library (library_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (library_id) REFERENCES document_libraries(id) ON DELETE CASCADE
);
```

### 2.4 溯源展示示例

前端 SourceCard 组件展示：

```
┌──────────────────────────────────────┐
│ 📄 2024年度技术报告.pdf   第 5/20 片段 │
│ 📍 第三章 技术架构 > 3.2 微服务治理     │
│ 📃 第 42 页                          │
│                                      │
│ 微服务治理平台提供统一的服务注册与发现  │
│ 机制，支持同城双活和异地多活部署方案...  │
└──────────────────────────────────────┘
```

---

## 3. MinerU 文档抽取

### 3.1 选型理由

- MinerU (magic-pdf) 对中文 PDF 的结构化解析能力远超 PyMuPDF
- 输出 Markdown 保留标题层级、表格、列表结构 —— 这正是按结构切分的理想格式
- 一道工具覆盖 PDF + DOCX，减少依赖碎片

### 3.2 各格式抽取方案

| 格式 | 工具 | 说明 |
|------|------|------|
| PDF | MinerU `magic-pdf` | 输出结构化 Markdown（含标题、表格、公式） |
| DOCX | MinerU `magic-pdf` | 同上 |
| DOC | LibreOffice headless → TXT → 读文本 | MinerU 不直接支持 .doc，先转 TXT 再处理 |
| TXT | 直接读取 | UTF-8/GBK 编码检测 |
| MD | 直接读取 | 原生标题结构可直接用于切分 |

### 3.3 MinerU 调用封装（`utils/mineru_extractor.py`）

```python
# 伪代码示意
from magic_pdf.pipe.UNIPipe import UNIPipe

def extract_to_markdown(file_path: str, file_type: str) -> dict:
    """
    返回:
    {
        "full_text": "清洗后的全文文本",
        "markdown": "结构化 MD（用于按标题切分）",
        "metadata": {
            "pages": 50,
            "title": "检测到的标题",
            "toc": [{"level": 1, "title": "第一章", "page": 3}, ...]
        }
    }
    """
    if file_type == "pdf":
        # MinerU UNIPipe 处理
        pipe = UNIPipe(pdf_bytes=..., ...)
        pipe.pipe_classify()
        pipe.pipe_parse()
        pdf_info = pipe.pipe_mk_uni_markdown(...)
        return {
            "full_text": pdf_info.get_text(),
            "markdown": pdf_info.get_markdown(),
            "metadata": { ... }
        }
    elif file_type == "docx":
        # 类似流程
        ...
```

### 3.4 MinerU 模型文件

MinerU 首次运行会下载模型文件，约 2GB。在 Docker 镜像中通过预下载挂载到 `/root/.magic-pdf/` 来避免每次重启下载。

---

## 4. LangChain 管线架构

### 4.1 LangChain 职责划分

不再单独建 `rag/` 目录，LangChain 组件直接集成在 `services/` 中，减少无意义的层级。

```
┌─ FastAPI（HTTP 层）──────────────────┐
│  - 路由注册                           │
│  - JWT 鉴权注入（Depends）            │
│  - 请求参数校验（Pydantic）            │
│  - SSE StreamingResponse              │
│  - 全局异常处理（exception_handler）   │
└──────────┬───────────────────────────┘
           │ 调用
┌──────────▼───────────────────────────┐
│  services/*（业务 + LangChain 组件）   │
│  - 事务管理（SQLAlchemy）              │
│  - LangChain 组件：                    │
│    · chunking（LangChain splitter）    │
│    · HybridRetriever（BaseRetriever）  │
│    · ConversationMemory（自定义）      │
│    · AsyncStreamingCallback（自定义）   │
│  - 业务编排（LCEL 链组装）             │
└──────────────────────────────────────┘
```

### 4.2 使用的 LangChain 核心组件

| LangChain 组件 | 对应包 | 在本项目中用途 |
|----------------|--------|---------------|
| `RecursiveCharacterTextSplitter` | `langchain-text-splitters` | 无法按结构切分时的回退策略 |
| `HuggingFaceBgeEmbeddings` | `langchain-huggingface` | BGE 模型加载与向量化 |
| `Chroma` (VectorStore) | `langchain-chroma` | 向量库读写（HttpClient 连 ChromaDB Server） |
| `ChatOpenAI` | `langchain-openai` | DeepSeek LLM（OpenAI 兼容协议） |
| `BaseRetriever` | `langchain-core` | 自定义混合检索器的基类 |
| `ChatPromptTemplate` | `langchain-core` | 构建问答 prompt |
| `RunnablePassthrough` / `RunnableLambda` | `langchain-core` | LCEL 链组装 |
| `AsyncIteratorCallbackHandler` | `langchain-core` | LLM 流式 token 回调 |
| `StrOutputParser` | `langchain-core` | 输出解析 |
| `Document` | `langchain-core` | 统一文档/Chunk 抽象 |

### 4.3 RAG 链（LCEL 构建，位于 `services/chat_service.py`）

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

class RAGChainBuilder:
    """构建 RAG 问答链"""

    SYSTEM_PROMPT = """你是文档问答助手。根据提供的文档内容回答问题。
如果文档中没有相关信息，直接说"文档中未找到相关信息"，不要编造。

可用文档片段：
{context}"""

    def build(self, retriever, llm):
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        chain = (
            {
                "context": itemgetter("question") | retriever | self._format_docs,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def _format_docs(self, docs: list[Document]) -> str:
        """将检索到的 Document 格式化为 prompt 中的 context"""
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            src = f"[来源{i}] {meta['document_name']}"
            if meta.get("heading_path"):
                src += f" > {' > '.join(meta['heading_path'])}"
            parts.append(f"{src}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
```

### 4.4 自定义混合检索器（位于 `services/retrieval_service.py`）

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class HybridRetriever(BaseRetriever):
    """混合检索器：语义 TOP-3 + ES 全文 TOP-3 → Rerank TOP-2"""

    vector_store: Chroma          # LangChain Chroma wrapper
    es_search: ESMixin            # ES 全文检索封装
    embeddings: HuggingFaceBgeEmbeddings
    reranker: FlagReranker        # BGE-Reranker

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        # 1. 语义检索 TOP-3
        semantic_docs = self.vector_store.similarity_search(query, k=3)

        # 2. ES 全文检索 TOP-3（IK 分词，精确词/专有名词）
        es_docs = self.es_search.search(library_id=self.library_id, query=query, k=3)

        # 3. 合并去重（按 chunk_id）
        merged = self._deduplicate(semantic_docs + es_docs)

        # 4. Rerank 精排 TOP-2
        ranked = self.reranker.rerank(query, merged, top_k=2)

        return ranked
```

### 4.5 对话记忆管理（位于 `services/chat_service.py`）

不使用 LangChain 自带的 ConversationBufferMemory（已逐渐废弃），而是自定义实现，更精确控制"10轮压缩 + 保留3轮"。

```python
class ConversationMemory:
    """自定义对话记忆管理器"""

    MAX_ROUNDS_BEFORE_COMPRESS = 10   # 超过10轮触发压缩
    KEEP_RECENT_ROUNDS = 3            # 保留最近3轮

    def __init__(self, session_id: int, summary: str, llm):
        self.session_id = session_id
        self.summary = summary
        self.llm = llm  # LangChain ChatOpenAI，用于生成摘要

    def add_messages(self, user_msg: str, assistant_msg: str):
        """添加一轮对话，必要时触发压缩"""
        ...

    def get_context_for_prompt(self) -> list:
        """
        返回构建 prompt 时用的历史消息列表，
        格式: [("摘要", summary), ("human", msg1), ("ai", msg2), ...]
        用于 LangChain ChatPromptTemplate 的 MessagesPlaceholder
        """
        ...

    async def _compress(self, all_messages: list):
        """调用 LLM 将旧消息 + 现有摘要压缩为新摘要"""
        compress_prompt = """请将以下对话内容压缩为一段简洁的摘要..."""
        ...
```

---

## 5. Docker Compose 部署（六服务）

> **镜像体积**：backend 镜像含 PyTorch + MinerU(2GB) + BGE(1.3GB) + Reranker(1GB) + LibreOffice，预计 **8-12GB**。首次 `docker compose build` 耗时较长。nginx 镜像仅几十 MB，前端变更无需重建 backend。ES 服务默认 heap 1G（可调大），实际占用约 2GB。

### 访问流程

```
用户 → http://localhost:80
         ↓
       nginx (frontend 服务)
         ├── /api/*  → proxy_pass → backend:8080
         └── /*      → Vue 静态文件 (index.html fallback)
```

用户只需访问 **`http://localhost`**（80 端口），nginx 自动区分前端页面和 API 请求。

```yaml
version: '3.8'
services:

  mysql:
    image: mysql:8.0
    container_name: rag-mysql
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: ${MYSQL_DATABASE:-native_rag}
    volumes:
      - mysql_data:/var/lib/mysql
      - ./init.sql:/docker-entrypoint-initdb.d/01-init.sql
    ports:
      - "${MYSQL_PORT:-3306}:3306"
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      retries: 5

  chromadb:
    image: chromadb/chroma:latest
    container_name: rag-chromadb
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma
      - ANONYMIZED_TELEMETRY=FALSE
    ports:
      - "${CHROMA_PORT:-8000}:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v2/heartbeat"]
      interval: 10s
      retries: 5

  elasticsearch:
    build: ./elasticsearch           # 官方镜像 + IK 分词插件
    container_name: rag-es
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false   # 单机内网，关闭认证
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    volumes:
      - es_data:/usr/share/elasticsearch/data
    ports:
      - "${ES_PORT:-9200}:9200"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9200/_cluster/health"]
      interval: 10s
      timeout: 10s
      retries: 10

  backend:
    build: ./backend
    container_name: rag-backend
    env_file:
      - .env
    volumes:
      - uploads_data:/app/data/uploads
      - model_cache:/root/.cache/huggingface
      - mineru_models:/root/.magic-pdf
    # 仅内网暴露，不映射宿主机端口（由 nginx 代理访问）
    expose:
      - "8080"
    depends_on:
      mysql:
        condition: service_healthy
      chromadb:
        condition: service_healthy
      elasticsearch:
        condition: service_healthy

  nginx:
    build: ./frontend
    container_name: rag-nginx
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  mysql_data:
  chroma_data:
  es_data:
  uploads_data:
  model_cache:
  mineru_models:
```

### Elasticsearch Dockerfile（官方镜像 + IK 插件）

```dockerfile
FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.0
# 安装 IK 中文分词插件（版本必须与 ES 版本严格匹配）
RUN elasticsearch-plugin install --batch \
    https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.13.0/elasticsearch-analysis-ik-8.13.0.zip
```

### Backend Dockerfile（仅 Python）

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# 系统依赖（MinerU 的 OpenGL 依赖 + LibreOffice for .doc）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libreoffice-headless \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Frontend Dockerfile（node 编译 + nginx 服务）

```dockerfile
# 阶段1：编译 Vue
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# 阶段2：nginx 服务静态文件
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### nginx.conf

```nginx
server {
    listen 80;
    server_name localhost;

    # gzip 压缩（减少传输体积，尤其是大 HTML/JS 文件）
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;

    # /api/ 反向代理到 backend
    location /api/ {
        proxy_pass http://backend:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        # SSE 流式传输必须关闭缓冲
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # 其他所有请求 → Vue SPA
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;   # Vue History 模式 fallback
    }
}
```

### 环境变量（.env）—— 完整版

```bash
# ── MySQL ──
MYSQL_ROOT_PASSWORD=change_me
MYSQL_DATABASE=native_rag
MYSQL_HOST=mysql
MYSQL_PORT=3306

# ── ChromaDB ──
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# ── DeepSeek LLM（切换模型只改 DEEPSEEK_MODEL）──
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-pro          # 可切换为 deepseek-chat / deepseek-reasoner 等

# ── Embedding 模型（本地）──
EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5
EMBEDDING_DEVICE=cpu

# ── Rerank 模型（本地）──
RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANK_DEVICE=cpu

# ── Admin 预设账号 ──
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change_me_admin

# ── JWT ──
JWT_SECRET_KEY=generate_a_random_string_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# ── 文件上传 ──
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=50
```

---

## 6. 数据库设计（MySQL）

### 6.1 users
| 列 | 类型 | 说明 |
|----|------|------|
| id | BIGINT AUTO_INCREMENT PK | |
| username | VARCHAR(50) UNIQUE NOT NULL | |
| password_hash | VARCHAR(255) NOT NULL | bcrypt |
| role | ENUM('admin','user') NOT NULL DEFAULT 'user' | |
| created_at | DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP | |

### 6.2 document_libraries
| 列 | 类型 | 说明 |
|----|------|------|
| id | BIGINT AUTO_INCREMENT PK | |
| name | VARCHAR(200) NOT NULL | |
| description | TEXT | |
| created_by | BIGINT FK → users.id | |
| created_at | DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP | |
| updated_at | DATETIME ON UPDATE CURRENT_TIMESTAMP | |

### 6.3 documents
| 列 | 类型 | 说明 |
|----|------|------|
| id | BIGINT AUTO_INCREMENT PK | |
| library_id | BIGINT FK → document_libraries.id NOT NULL | |
| filename | VARCHAR(500) NOT NULL | 原始文件名 |
| file_path | VARCHAR(1000) NOT NULL | 存储路径 |
| file_type | VARCHAR(20) NOT NULL | pdf/docx/doc/txt/md |
| file_size | BIGINT NOT NULL | 字节数 |
| chunk_count | INT NOT NULL DEFAULT 0 | |
| chunk_size | INT NOT NULL DEFAULT 1024 | 切分参数（token），上传时按文档配置，默认 1024 |
| overlap_token | INT NOT NULL DEFAULT 102 | 重叠 token 数，默认 102（chunk_size 的 10%） |
| status | ENUM('processing','ready','failed') NOT NULL DEFAULT 'processing' | |
| error_message | TEXT | |
| uploaded_by | BIGINT FK → users.id | |
| created_at | DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP | |

### 6.4 chunks
详见 2.3 节完整 DDL，metadata_json 字段内容见 2.1 节。

### 6.5 chat_sessions
| 列 | 类型 | 说明 |
|----|------|------|
| id | BIGINT AUTO_INCREMENT PK | |
| user_id | BIGINT FK → users.id NOT NULL | |
| library_id | BIGINT FK → document_libraries.id NOT NULL | |
| title | VARCHAR(200) | 首条问题截取生成 |
| summary | TEXT | 压缩摘要 |
| message_count | INT NOT NULL DEFAULT 0 | **冗余字段，用于快速判断是否超过 10 轮（>20 条消息）触发压缩，避免每次 count(*) 查询** |
| created_at | DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP | |
| updated_at | DATETIME ON UPDATE CURRENT_TIMESTAMP | |

### 6.6 chat_messages
| 列 | 类型 | 说明 |
|----|------|------|
| id | BIGINT AUTO_INCREMENT PK | |
| session_id | BIGINT FK → chat_sessions.id NOT NULL | |
| role | ENUM('user','assistant') NOT NULL | |
| content | TEXT NOT NULL | |
| sources_json | JSON | assistant 消息的溯源信息 |
| created_at | DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP |

---

## 7. API 设计

全部以 `/api` 为前缀，除 auth 外均需 JWT Bearer Token。**所有列表接口统一分页参数 `page`（默认 1）/ `page_size`（默认 20，最大 100）**，返回格式：

```json
{ "items": [...], "total": 150, "page": 1, "page_size": 20 }
```

### 7.1 认证

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/auth/register | 注册（username + password），返回 user 信息 |
| POST | /api/auth/login | 登录，返回 `{ access_token, token_type: "bearer" }` |
| GET | /api/auth/me | 当前用户信息（含 role） |

### 7.2 仪表盘

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/dashboard | `{ library_count, document_count, chunk_count }` |

### 7.3 文档库（admin 写，全员读，分页）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/libraries?page=1&page_size=20 | 分页列表 |
| GET | /api/libraries/{id} | 详情 |
| POST | /api/libraries | 新增（admin） |
| DELETE | /api/libraries/{id} | 删除（admin），级联删文档/chunks/ChromaDB 向量/ES 数据 |

### 7.4 文档（admin 写，全员读，分页）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/libraries/{library_id}/documents?page=1&page_size=20 | 分页列表 |
| GET | /api/documents/{id}/status | 文档处理状态 `{ status, chunk_count, error_message }` |
| POST | /api/libraries/{library_id}/documents | 上传（multipart/form-data）（admin），返回 document 对象（status=processing）。**表单额外支持 `chunk_size`（默认 1024，范围 128~8192）与 `overlap_token`（默认 102，范围 0~chunk_size-1），按文档维度配置切分** |
| DELETE | /api/documents/{id} | 删除文档 + 所有 chunks + ChromaDB 向量（admin） |

**文档状态轮询**：前端上传文档后，用 `GET /api/documents/{id}/status` 轮询（建议 2 秒间隔），status 变为 `ready` 或 `failed` 后停止，展示结果。

### 7.5 聊天会话（分页）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/sessions?library_id=&page=1&page_size=20 | 当前用户在该库的会话列表 |
| POST | /api/sessions | 创建会话 `{ library_id }`，title 暂为空 |
| DELETE | /api/sessions/{id} | 删除会话（仅所有者） |
| GET | /api/sessions/{id} | 会话详情 + 历史消息 |

**会话标题自动生成规则**：
- 首条用户问题作为标题
- 中文按字符截取前 30 字，超出加 `...`
- 英文按空格分词后截取前 15 词
- 生成时机：首条 assistant 消息写入后异步更新 title 字段

### 7.6 聊天（SSE 流式）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/chat/{session_id} | `{ content }`，返回 SSE 流 |

**SSE 事件顺序**（检索完后先推 sources，再推 token）：

```
event: sources
data: {"sources": [{"document_name": "xxx.pdf", "chunk_content": "...", "heading_path": ["第三章"], "page_number": 42, "chunk_index": 5, "total_chunks": 20}]}

event: token
data: {"content": "根"}

event: token
data: {"content": "据"}

...

event: done
data: {"message_id": 123}
```

前端收到 `sources` 后立即渲染溯源卡片，后续 token 逐字追加到答案区域。

**SSE 断连处理**：前端检测到连接异常关闭（答案不完整）时，显示"连接中断"提示 + "重试"按钮。重试时重新 POST 同一条消息（后台通过 session 历史判断幂等）。

---

## 8. 全局异常处理

统一错误响应格式：

```json
{
    "error": {
        "code": "DOCUMENT_NOT_FOUND",
        "message": "文档不存在或已删除"
    }
}
```

| 场景 | HTTP 状态码 | code |
|------|------------|------|
| 参数校验失败 | 422 | VALIDATION_ERROR |
| 未登录 | 401 | UNAUTHORIZED |
| 无权限 | 403 | FORBIDDEN |
| 资源不存在 | 404 | NOT_FOUND |
| LLM API 不可用 | 502 | LLM_UNAVAILABLE |
| ChromaDB 不可用 | 502 | VECTOR_DB_UNAVAILABLE |
| 文件解析失败 | 400 | PARSE_FAILED |
| 文件格式不支持 | 400 | UNSUPPORTED_FORMAT |
| 内部错误 | 500 | INTERNAL_ERROR |

FastAPI 通过 `@app.exception_handler` 全局注册，service 层抛自定义异常，由 handler 统一转换为上述格式。

---

## 9. 前端（Vue 3 + Naive UI）

> **开发 vs 生产**：开发时 `npm run dev` 启动 Vite dev server（`:5173`），通过 Vite proxy 转发 `/api` 到 backend；生产时 nginx 编译 Vue 并统一服务在 `:80`，`/api` 反向代理到 backend。前端代码无需感知环境差异。

### 9.1 开发环境配置

开发时 Vue dev server（`localhost:5173`）和 FastAPI（`localhost:8080`）跨端口，Vite 配置 proxy：

```typescript
// vite.config.ts
export default defineConfig({
    server: {
        port: 5173,
        proxy: {
            '/api': {
                target: 'http://localhost:8080',
                changeOrigin: true,
            },
        },
    },
});
```

这样前端 `fetch("/api/xxx")` 自动代理到后端，无需 CORS 配置。生产环境无此问题（FastAPI 直接服务静态文件）。

### 9.2 SSE 接收

```typescript
// utils/sse.ts
async function* streamChat(sessionId: number, content: string): AsyncGenerator<SSEEvent> {
    const response = await fetch(`/api/chat/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ content }),
    });
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    // 按 SSE 协议解析 event: / data: 行...
    // 连接异常时抛出 StreamError，上层捕获后显示"重试"按钮
}
```

### 9.3 分页处理

- 文档库列表、文档列表、会话列表均使用 Naive UI 的 `n-pagination` 组件
- API 返回 `total` 用于计算总页数

---

## 10. MinerU 风险与降级策略

### 风险

| 风险项 | 说明 |
|--------|------|
| API 不稳定 | `magic-pdf` 的 Python API（UNIPipe 等）仍在快速迭代，接口可能变化 |
| 模型体积大 | MinerU 自身模型约 2GB，加上 BGE（1.3GB）+ Reranker（1GB）+ PyTorch，总模型体积约 **5GB** |
| 解析失败 | 扫描件 PDF、加密 PDF、损坏文件可能导致 MinerU 解析失败 |

### 应对

1. **版本锁定**：`requirements.txt` 中锁定 `magic-pdf==x.x.x` 精确版本，每次升级需验证
2. **降级策略**：MinerU 解析失败时自动回退到轻量方案——PDF 用 PyMuPDF、DOCX 用 python-docx、DOC 用 LibreOffice
3. **镜像体积**：Docker 镜像预计 **8-12GB**，在 README 中明确标注，首次 `docker compose up` 需预留时间拉取/构建
4. **模型预热**：Dockerfile 中 `RUN python -c "from magic_pdf...; from sentence_transformers..."` 预下载模型，避免容器启动时下载超时

---

## 11. ChromaDB 向量清理细节

### 删除文档时的向量清理

```python
# services/vector_store_service.py
def delete_document_vectors(self, library_id: int, document_id: int):
    """从 ChromaDB 中删除指定文档的所有向量"""
    collection = self.client.get_collection(f"library_{library_id}")
    collection.delete(where={"document_id": document_id})
```

### 删除文档库时的向量清理

```python
def delete_library_collection(self, library_id: int):
    """删除整个文档库对应的 ChromaDB Collection"""
    self.client.delete_collection(f"library_{library_id}")
```

ChromaDB 的 `delete(where=...)` 按 metadata 过滤删除，创建 chunk 时把 `document_id` 写入向量 metadata 即可。不需要遍历 chunk ID。

---

## 12. 关键依赖（Python）

```
# ── Web 框架 ──
fastapi
uvicorn[standard]
python-multipart
aiofiles

# ── LangChain 全家桶（1.x；langchain-community 无 1.x，未使用）──
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-chroma                # Chroma vector store 集成
langchain-openai                # ChatOpenAI (DeepSeek 兼容)

# ── 文档处理 ──
mineru[pipeline]                # MinerU 3.x（pipeline backend）
pymupdf                         # MinerU 降级方案：PDF
python-docx                     # MinerU 降级方案：DOCX

# ── 向量 & 模型 ──
fastembed                       # Embedding（ONNX，bge-small-zh-v1.5）
transformers                    # chunker 精确 token 计数
sentence-transformers           # Rerank（CrossEncoder 加载 bge-reranker-base，.env 可换模型）
# torch / torchvision：Dockerfile 单独从官方 CPU 源安装（固定 2.13.0 / 0.28.0）

# ── 全文检索 ──
elasticsearch                   # ES 官方 Python 客户端

# ── 数据库 ──
sqlalchemy
alembic
pymysql
chromadb-client                 # ChromaDB HTTP 客户端

# ── 认证 ──
python-jose[cryptography]
bcrypt

# ── 配置 ──
python-dotenv
pydantic-settings

# ── 异步 HTTP ──
httpx
```

---

## 13. 实现状态与方案差异（截至 2026-08-05：含 chunk 配置化 + 串味修复 + 1024 重传验证）

> 本系统按上述方案实施，但落地过程中因环境限制（国内网络、包源可用性）和用户决策做了一些调整。**方案正文保留原始设计**，实际实现以本表为准。

### 13.1 技术栈差异

| 项 | 方案设计 | 实际实现 | 原因 |
|----|----------|----------|------|
| LangChain | 0.3.x | **1.x**（core 1.5.3 等） | 用户指定用 1.x |
| langchain-community | 使用 | **移除** | 无 1.x 版本，代码不需要 |
| Embedding 框架 | sentence-transformers | **FastEmbed**（ONNX 运行时） | PyTorch 官方源 403 无法下载 CPU torch（后虽恢复，但已切换） |
| Embedding 模型 | bge-large-zh-v1.5 (1024维) | **jina-embeddings-v2-base-zh** (768维) | 先试 bge-small-zh 后换 jina（multilingual-e5-large 下载源被墙） |
| Rerank 框架 | FlagEmbedding | **sentence-transformers CrossEncoder** | FlagEmbedding 旧版兼容性差 |
| Rerank 模型 | bge-reranker-v2-m3 | **bge-reranker-base** | v2-m3（2.27B）在 CPU 上单对打分约 27s，换成 base（279M）后约 5s，仍用 CrossEncoder 加载，精度略降可接受（.env 配置） |
| 文档抽取 | MinerU（magic-pdf） | **MinerU 3.4.4，本地 CLI + 官方 API 可切换**（`.env` 配 `MINERU_API_TOKEN` 走 API，留空走本地） | MinerU 已更名；本地慢但隐私，API 快但文档上传云端 |
| torch | 隐含 CUDA 版 | **torch 2.13.0+cpu + torchvision 0.28.0**（Dockerfile 固定） | 避免 CUDA 大包；torchvision 需匹配版本 |
| .doc 支持 | LibreOffice 转换 | **暂不支持** | LibreOffice 未解决 |

### 13.2 已完成的实现要点

| Task | 内容 | 状态 |
|------|------|------|
| Task 1 | 骨架 + 六服务 docker-compose | ✅ 完成 |
| Task 2 | 数据库模型 + Alembic 迁移 | ✅ 完成（表结构由 Alembic 管理） |
| Task 3 | 认证（注册/登录/JWT/admin 种子） | ✅ 完成 |
| Task 4 | 文档库管理 + 仪表盘 | ✅ 完成 |
| Task 5 | 文档上传管线（MinerU→切片→FastEmbed→ChromaDB→ES） | ✅ 完成（MD/TXT/PDF/DOCX 四格式验证） |
| Task 6 | 聊天问答（会话/混合检索/SSE 流式/记忆压缩） | ✅ 完成 |
| Task 7 | 前端页面（登录/仪表盘/文档库/文档/聊天 + SSE 流式渲染 + 溯源卡片） | ✅ 完成（浏览器端到端验证） |
| Task 8 | 联调与验收（含 SSE 断连恢复实测 + README） | ✅ 完成 |
| Task 9 | 按文档维度配置 chunk（模型+migration+chunker 参数化+API 校验+前端表单） | ✅ 完成（pytest 5 passed，真实上传验证 512/50 与默认 1024/102 均生效） |
| Task 10 | 多轮串味 bug 修复 + 检索速度收敛（candidate_k 4→3）+ 1024 重传 | ✅ 完成（串味验证通过；rerank 仍是速度瓶颈，见 13.4 第 13 条） |

### 13.3 检索与体验优化（Task 7 后用户反馈 + 精度专项）

#### 检索精度专项（5/6 问题通过，vs 改进前 2 失败 + 1 漏召回）

| 优化 | 实现 |
|------|------|
| **embedding 升级** | bge-small-zh(512) → **jina-embeddings-v2-base-zh(768)**（FastEmbed 支持、可国内下载） |
| **候选池** | 语义/ES 单路各 **TOP-3**（`candidate_k=3`，semantic 3 + ES 3，去重后约 4-5 候选）。曾放大到 TOP-8 提升召回，但 rerank 按候选数线性耗时，为控速收敛到 3 |
| **query 改写** | 曾用 LLM 规范化问题（口语化→书面语）并双路检索，实测改写收益趋零且每次多 2-4s，**已停用**，直接用原 query |
| **rerank 策略** | 绝对阈值(0.75)实测不可靠（相关 chunk 可能只得 0.27 分），改为**相对排序 top-3 + 低分兜底**(`SIMILARITY_THRESHOLD_LOW` 默认 0.2，最高分低于此才判定无关) |
| **rerank 输入** | 只用 `doc.page_content` 纯正文（heading 等元数据在 metadata 单独存，不拼进打分文本） |
| **检索 top-3** | rerank 后取 top-3（原 top-2，复杂问题信息更足） |
| **chunk 粒度 1024** | 按文档维度配置（上传时设置），**默认 1024 tokens / 102 重叠**。256/512 曾因切得过碎、短 chunk 在 rerank 下被打压而质量差，改为 1024 且可配置 |

#### 体验优化

| 优化 | 实现 |
|------|------|
| **删除二次确认** | 文档删除（NPopconfirm）、文档库/会话删除（dialog），全部二次确认 |
| **ES 查询调松** | `operator: and` 精确优先，结果不足 k 时用 `or` 宽松补足 |
| **无结果不发 sources** | 检索为空时不推送 sources 事件，LLM 答"文档中未找到" |
| **思考过程展示** | 后端 httpx 直连 DeepSeek 解析 `reasoning_content`，SSE 单独推 `reasoning` 事件；前端折叠"💭 思考过程" |
| **前端权限恢复** | 刷新后恢复 user 信息（AppLayout fetchMe）+ isAdmin 用 computed |

#### 按文档维度配置 chunk（新增功能）

| 项 | 实现 |
|----|------|
| **数据模型** | `documents` 表新增 `chunk_size`（默认 1024）/ `overlap_token`（默认 102），Alembic migration `0003` 回填旧文档为 1024/102 |
| **上传 API** | `POST /api/libraries/{id}/documents` 表单新增 `chunk_size`（128~8192）/ `overlap_token`（0~chunk_size-1），校验失败抛 `ValidationError` 且先清理已落盘文件 |
| **chunker 参数化** | `chunk_text(..., chunk_size=1024, overlap_token=102)` 替代模块常量 `CHUNK_TOKENS`/`OVERLAP_TOKENS`，保留 `MAX_CHUNKS=2000` 上限 |
| **前端表单** | 文档列表页上传按钮旁新增两个 `NInputNumber`（chunk 长度 128~8192 / 重叠 token 0~chunk_size-1），`uploadDocument(libraryId, file, chunkSize, overlapToken)` 写入 FormData |

**动机**：实测 512-token chunk 检索质量明显差于 1024——切得碎导致短 chunk 在 BGE-Reranker 下被打压（Q3 中 README 的 38-175 token 短 chunk 打分仅 0.04，被低分兜底误杀返回空）。故改为按文档维度配置，默认回到 1024。已上传文档不重切。

> **前端构建坑**：本地改动前端后必须手动 `npm run build` 再 `docker compose build nginx`。本次曾因只 `docker compose build nginx`（`COPY dist/` 被 Docker 层缓存 CACHED）导致页面未更新，且 `grep` dist 产物无新增字符串，最终重新 `npm run build` 才解决。

#### 多轮串味 bug 修复 + 1024 数据重传验证（2026-08-05）

- **串味修复**：见 13.4 第 12 条（created_at 秒级排序 → id 排序），三处（历史构建/记忆压缩/会话详情）统一修复
- **验证结果**：同一 session 连续问「AI Agent → flink → JAVA」，第三问无串味、回答各自命中对应文档；保存的消息顺序严格 user/assistant 交替
- **1024 重传**：删除 library_5 旧 512 数据，用默认 1024/102 重新上传 6 个文档（含中文文件名 `flink原理问题.txt`），均处理成功；upload_big_test.pdf 由 882 chunks（512）变为 375 chunks（1024）

### 13.4 关键实现决策（踩坑记录）

1. **SSE 生成器内用独立 db session**：FastAPI 请求级 db 在 StreamingResponse 生成器多次 commit 后状态不稳定（标题/message_count 不生效），改为 `stream_chat` 内部 `SessionLocal()`。
2. **torchvision 版本必须匹配 torch**：pip 从官方 CPU 源误解析 torchvision 到 0.1.6（2017 年版），导致 rerank 的 `InterpolationMode` 缺失。固定 `torch 2.13.0 + torchvision 0.28.0`（对应规律 `torchvision 0.x = torch 主版本 + 15`）。
3. **MinerU 3.x 的 pipeline backend 依赖 torch**：`mineru` 基础包不含 torch，需 `mineru[pipeline]` extra + 单独装 CPU torch。
4. **MinerU 本地通过 CLI subprocess 调用**（`mineru -p file -o outdir -b pipeline`），解析输出 markdown；需 opencv 系统库（libxcb 等）。
5. **MinerU 官方 API 接入要点**：`/api/v4/file-urls/batch` 请求体需 `files:[{"name": 文件名}]`（非 filename）；上传后轮询 `/api/v4/extract-results/batch/{batch_id}`，取 `data.extract_result[0].state` 与 `full_zip_url`。
6. **ES 检索策略**：`and`（精确）优先，不足 k 时 `or`（宽松）补足，避免查询词多时无结果。
7. **FastEmbed 缓存持久化**：`cache_dir` 指向 `/root/.cache/fastembed`（挂载 volume），避免容器重启丢模型。
8. **前端 Naive UI 组件需显式 import**：Vue 3 按需注册，`n-data-table` 等未 import 的组件不渲染；`useMessage` 在懒加载组件找不到 provider，改用 `createDiscreteApi`。
9. **FastEmbed 模型下载源被墙**：`intfloat/multilingual-e5-large` 的模型文件在 storage.googleapis.com（Qdrant），国内不可访问且 HF 缺 fastembed ONNX 文件 → 改用 `jina-embeddings-v2-base-zh`（走 HF 镜像可下载）。
10. **rerank 绝对分数不可靠**：实测最相关 chunk（"同步通信使用 REST API 或 gRPC"）可能只得 0.27 分，绝对阈值过滤会误杀。改为相对排序 + 低分兜底（`SIMILARITY_THRESHOLD_LOW`）。
11. **DeepSeek reasoning_content 需低层解析**：LangChain ChatOpenAI 拿不到 `reasoning_content`，改用 httpx 直连 DeepSeek 流式接口，解析 `delta.reasoning_content`（SSE 单独推 `reasoning` 事件）。
12. **多轮对话串味根因是 created_at 秒级排序不稳定**：`_save_messages` 同一次 commit 写入 user+assistant 两条消息，`created_at` 同为秒级，`order_by(created_at.desc())` 排序不稳定导致历史消息顺序错乱（assistant 跑到 user 前面），LLM 把上一轮话题当当前问题回答（实测第二问「flink 原理」答出「AI Agent」）。**修复：历史/压缩/详情统一按自增主键 `id` 排序**（`id` 严格按插入顺序）。
13. **rerank 是检索延迟绝对瓶颈**：语义(0.3-0.5s)+ES(0.4-0.7s) 并发可忽略，rerank 占检索 99% 时间。bge-reranker-base 在 CPU 上对 1024-token 长文本对打分**每对约 5s**，5 候选即约 25s。候选数无法压到 4 以下（语义 3 + ES 3 去重后仍有 5），且 chunk 越大打分越慢。进一步压速只能换更小模型/GPU/截断（截断已被实测否定：被截掉的部分可能恰是相关核心，排序不稳定）。

---

## 14. 验证计划

1. **启动**: `docker compose up -d`，六服务均 healthy，访问 `http://localhost` 确认前端页面可打开
2. **模型就绪**: backend 启动日志确认 BGE + Reranker + MinerU 模型加载成功
3. **认证**: 注册 → 登录 → admin 预设账号登录 → 注册用户尝试删除文档库被拒（403）
4. **文档库 CRUD**: admin 创建/删除库，删除后确认 ChromaDB collection 已清理、MySQL 级联删除、ES 数据已清理
5. **文档上传全链路**: 上传 PDF(DOCX/DOC/TXT/MD) → processing → 轮询 status 端点确认变 ready → chunks 表有数据 → ChromaDB 可检索 → ES 可检索
6. **MinerU 降级**：准备一个加密/损坏 PDF，确认回退到 PyMuPDF 处理
7. **聊天**: 创建会话 → 提问 → SSE 先收到 sources 事件（溯源卡片出现）→ 再收到 token 流（逐字渲染）→ 答案完整
8. **SSE 断连恢复**：强制关闭 SSE 连接，确认前端显示"连接中断"和重试按钮
9. **混合检索验证**: 同一问题的语义和 ES 召回结果对比，确认有互补效应
10. **记忆压缩**: 连续对话 >10 轮，验证 summary 字段更新且后续回答仍能关联早期上下文
11. **会话隔离**: 用户 A 无法通过 API 直接访问用户 B 的会话
12. **分页**: 文档库/文档/会话列表确认分页参数生效
13. **前端**: 科技简约风格、暗色主题、断连重试 UI、溯源卡片正确展示
14. **按文档配置 chunk**: 上传时传 `chunk_size=512&overlap_token=50` → 文档记录保存该值 → 就绪后 chunk 分布符合配置；不传时回落默认 1024/102；越界参数（如 chunk_size=100 或 overlap>=chunk_size）返回 422/ValidationError 且不残留垃圾文件
15. **多轮对话不串味**: 同一 session 连续问 2-3 个不相关问题，回答各自命中对应文档、不混入上一轮话题；会话详情消息顺序严格 user/assistant 交替
16. **rerank 速度基线**: 实测语义+ES 并发约 0.7s、rerank 每候选约 5s（CPU，1024-token 对），检索总耗时约 19-31s——作为优化前后对比基线

---

## 15. 实施任务拆解

### 任务依赖关系

```
Task 1: 项目骨架搭建
  ↓
Task 2: 数据库模型与迁移
  ↓
Task 3: 认证系统（用户注册/登录/JWT/admin种子）
  ↓
Task 4: 文档库管理 API ─────────────────┐
Task 5: 文档上传处理管线（MinerU→切片→向量化→BM25）
  ↓                                      │
Task 6: 聊天问答系统（RAG链/SSE流式/记忆压缩）
  ↓                                      │
Task 7: 前端页面开发 ←───────────────────┘（依赖 Task 3~6 API 就绪）
  ↓
Task 8: 联调与验收
```

---

### Task 1: 项目骨架搭建

**目标**：搭建可启动的空项目框架，`docker compose up` 能看到五个服务都 running。

**内容**：
- 创建 `docker-compose.yml`（mysql + chromadb + backend + nginx 四个服务定义 + volumes）
- 创建 `.env` 和 `.env.example`（包含方案中全部环境变量）
- 创建 `init.sql`（6 张表的 DDL，暂不包含种子数据）
- 创建 `backend/` 目录结构（所有 `__init__.py`、空 `main.py`、`config.py`）
- 创建 `backend/Dockerfile`（python:3.11-slim + 系统依赖 + requirements.txt）
- 创建 `backend/requirements.txt`（固定版本号，包含方案第 12 节全部依赖）
- 创建 `frontend/` 目录结构（Vite + Vue 3 + Naive UI 脚手架）
- 创建 `frontend/Dockerfile`（node 编译 + nginx 服务）
- 创建 `frontend/nginx.conf`（gzip + /api 反代 + SPA fallback + SSE buffering off）
- 创建 `frontend/package.json`（vue、vue-router、pinia、naive-ui、axios、vite）
- 创建 `frontend/vite.config.ts`（proxy `/api` → `localhost:8080`）
- `main.py` 仅包含一个 `/api/health` 端点返回 `{"status": "ok"}`，验证 backend 可达

**验收标准**：
- `docker compose up -d` 五个服务全部 healthy
- `curl http://localhost/api/health` 返回 `{"status": "ok"}`
- `curl http://localhost` 返回 nginx 默认页或 Vue 空白页（非 502/404）

---

### Task 2: 数据库模型与迁移

**目标**：6 张表的 ORM 模型 + Alembic 迁移脚本就绪，init.sql 与模型一致。

**内容**：
- 创建 6 个 SQLAlchemy 模型文件：
  - `models/user.py`（id, username, password_hash, role, created_at）
  - `models/document_library.py`（id, name, description, created_by FK, created_at, updated_at）
  - `models/document.py`（id, library_id FK, filename, file_path, file_type, file_size, chunk_count, status, error_message, uploaded_by FK, created_at）
  - `models/chunk.py`（id, document_id FK, library_id FK, chunk_index, content, token_count, metadata_json JSON, created_at）
  - `models/chat_session.py`（id, user_id FK, library_id FK, title, summary, message_count, created_at, updated_at）
  - `models/chat_message.py`（id, session_id FK, role ENUM, content, sources_json JSON, created_at）
- `models/__init__.py` 导出所有模型 + Base
- `config.py` 用 pydantic-settings 读取 `.env`，构建 `DATABASE_URL`
- `main.py` 中初始化 SQLAlchemy engine + sessionmaker
- Alembic 初始化（`alembic init`），生成初始迁移脚本
- 同步更新 `init.sql` 确保 DDL 与 ORM 模型一致

**验收标准**：
- `alembic upgrade head` 在 MySQL 中成功创建 6 张表
- `init.sql` 首次部署时自动建表成功
- 能用 Python REPL 插入/查询各表数据

---

### Task 3: 认证系统

**目标**：用户可注册、登录、获取 token，admin 账号自动种子，权限校验生效。

**内容**：
- `schemas/auth.py`：RegisterRequest、LoginRequest、TokenResponse、UserResponse
- `services/auth_service.py`：
  - `register(username, password)` → 创建 user（bcrypt hash）
  - `login(username, password)` → 验证密码，返回 JWT token
  - `get_current_user(token)` → 解码 JWT，查询用户
  - `seed_admin()` → lifespan 中调用，若 admin 不存在则从 .env 创建
- `api/auth.py`：POST /register、POST /login、GET /me
- `middleware/auth.py`：
  - `get_current_user`（Depends，从 Authorization header 解析 JWT）
  - `get_admin_user`（Depends，校验 role=admin，否则 403）
- JWT 用 `python-jose`，过期时间从 `.env` 读取
- 全局注册异常处理（401/403）

**验收标准**：
- 注册新用户 → 登录 → 拿到 access_token
- `/api/auth/me` 带 token 返回正确用户信息，不带 token 返回 401
- admin 种子：首次启动后 admin 账号可登录
- 用普通用户 token 调用 admin 接口返回 403

---

### Task 4: 文档库管理 API

**目标**：文档库 CRUD 完成，admin 可管理，普通用户只读，分页生效。

**内容**：
- `schemas/library.py`：LibraryCreate、LibraryUpdate、LibraryResponse、LibraryListResponse
- `services/library_service.py`：
  - `list_libraries(page, page_size)` → 分页查询
  - `get_library(id)` → 详情
  - `create_library(name, description, created_by)` → 创建
  - `delete_library(id)` → 级联删除 documents + chunks + ChromaDB collection + ES 数据
- `api/library.py`：GET /api/libraries、GET /api/libraries/{id}、POST /api/libraries、DELETE /api/libraries/{id}
- POST/DELETE 路由注入 `get_admin_user` Depends
- GET 路由所有登录用户可访问
- 仪表盘 API 一并完成（`services/dashboard_service.py` + `api/dashboard.py`）

**验收标准**：
- admin 创建文档库 → 成功；查询列表 → 可见
- 普通用户创建文档库 → 403
- admin 删除文档库 → 成功；级联删除 MySQL 中关联数据
- 分页：`?page=2&page_size=5` 正确返回

---

### Task 5: 文档上传处理管线

**目标**：上传文档 → 自动完成抽取/清洗/切片/向量化/写入 ES → 可检索状态。

**内容**：

**5a. 文档上传 API**
- `schemas/document.py`：DocumentResponse、DocumentStatusResponse
- `api/document.py`：POST /api/libraries/{id}/documents（multipart/form-data）、GET /api/documents/{id}/status、DELETE /api/documents/{id}、GET /api/libraries/{id}/documents（分页）
- 上传后保存文件到 `data/uploads/{library_id}/{uuid}_{filename}`，创建 document 记录 status=processing，返回 document 对象

**5b. 文档处理编排**（`services/document_service.py`）
- 接收上传文件路径，编排完整处理流水线：
  1. 文本抽取 → `utils/mineru_extractor.py`
  2. 文本清洗 → `utils/text_cleaner.py`
  3. 智能切片 → LangChain `RecursiveCharacterTextSplitter` + 结构感知切分逻辑
  4. 向量化 → `services/embedding_service.py`（批量 embed）
  5. 存入 ChromaDB → `services/vector_store_service.py`（collection=library_{id}）
  6. 存入 MySQL chunks 表（含完整 metadata_json）
  7. 写入 ES → `utils/es_index.py`（bulk 写入 text + chunk_id/document_id/library_id）
  8. 更新 document.status=ready / chunk_count
- 全程 try/except，失败则 document.status=failed + error_message；ES 写入失败单独捕获并标记（避免 MySQL 有、ES 无）

**5c. MinerU 集成**（`utils/mineru_extractor.py`）
- 封装 MinerU API，输出 `{"full_text": ..., "markdown": ..., "metadata": {...}}`
- 降级策略：MinerU 失败时 PDF → PyMuPDF、DOCX → python-docx

**5d. Embedding 服务**（`services/embedding_service.py`）
- 加载 BAAI/bge-large-zh-v1.5，提供 `embed_texts(texts: list[str])` 和 `embed_query(text: str)` 方法
- lifespan 中加载模型（首次下载约 1.3GB）

**5e. 向量库服务**（`services/vector_store_service.py`）
- 通过 langchain-chroma + HttpClient 连接 ChromaDB Server
- 提供 `add_chunks(chunks)`, `delete_by_document(doc_id)`, `delete_collection(lib_id)`, `similarity_search(query, k)` 方法

**5f. ES 全文检索**（`utils/es_index.py`）
- 官方 `elasticsearch` 客户端连接 rag-es（配置在 .env：ES_HOST/ES_PORT）
- 启动时确保 index `rag_chunks` 存在（mapping：text 用 ik_max_word/ik_smart，metadata enabled=false）
- 提供 `ensure_index()`, `bulk_add_chunks(chunks)`, `search(library_id, query, k)`, `delete_by_document(document_id)`, `delete_by_library(library_id)` 方法
- 单 index + library_id filter 实现库隔离
- ES 不可用时 `search` 返回空，不影响语义检索主链路

**验收标准**：
- 上传 PDF → status 从 processing 变 ready → chunks 表有数据 → curl status 端点正常
- ChromaDB 中可检索到该文档的向量
- ES 全文搜索（IK 分词）可命中关键词
- 上传损坏 PDF → status=failed + error_message 非空
- MinerU 降级验证：无 MinerU 环境时 PyMuPDF 可正常抽取
- 删除文档 → MySQL chunks 清除 + ChromaDB 向量清除 + ES 数据清除

---

### Task 6: 聊天问答系统

**目标**：用户创建会话 → 提问 → SSE 流式返回答案（带溯源）→ 多轮对话记忆压缩。

**内容**：

**6a. 会话管理 API**
- `schemas/session.py`：SessionCreate、SessionResponse
- `services/chat_service.py` 中实现会话 CRUD：
  - `create_session(user_id, library_id)` → 创建
  - `list_sessions(user_id, library_id, page, page_size)` → 分页
  - `delete_session(session_id, user_id)` → 仅所有者
  - `get_session(session_id)` → 会话详情 + 历史消息
- `api/session.py`：对应路由
- 首条 assistant 回复后异步调用 LLM 生成标题（中文截 30 字）

**6b. LLM 服务**（`services/llm_service.py`）
- LangChain `ChatOpenAI` 封装，参数从 .env 读取：
  - `base_url` = DEEPSEEK_BASE_URL
  - `api_key` = DEEPSEEK_API_KEY
  - `model` = DEEPSEEK_MODEL（可配置，支持 deepseek-chat / deepseek-reasoner / deepseek-v4-pro 等）
- 提供 `get_llm(streaming=True)` 和 `get_llm(streaming=False)` 两种模式
- 提供 `generate_summary(messages)` → 用于记忆压缩

**6c. 混合检索器**（`services/retrieval_service.py`）
- 继承 LangChain `BaseRetriever`，实现 `_get_relevant_documents`：
  1. ChromaDB 语义检索 TOP-3
  2. ES 全文检索 TOP-3（IK 分词，按 library_id filter）
  3. 按 chunk_id 去重
  4. BGE-Reranker 精排 TOP-2
- Reranker 在 lifespan 中加载模型
- 检索范围限定为 session 绑定的 library_id
- ES 不可用时降级为纯语义检索（不阻塞主链路）

**6d. RAG 链**（`services/chat_service.py` 中实现）
- LCEL 构建：
  ```
  {context: retriever, question, history} → prompt → llm → StrOutputParser
  ```
- prompt 包含：system 指令 + context（带来源编号）+ history + human question
- 通过 `astream_events()` 或自定义 callback 实现流式输出

**6e. SSE 聊天端点**（`api/chat.py`）
- POST /api/chat/{session_id}，body: `{ content }`
- 流程：
  1. 校验 session 归属 + library 存在
  2. 获取 context（summary + 最近 3 轮消息）
  3. 调用 retriever 检索
  4. 先推送 `event: sources`（含溯源信息）
  5. 调用 LLM stream
  6. 逐 token 推送 `event: token`
  7. 推送 `event: done`
  8. 保存 user + assistant 消息到 MySQL（含 sources_json）
  9. 更新 session.message_count += 2
  10. 触发记忆压缩检查
- `StreamingResponse` + `text/event-stream` content-type

**6f. 对话记忆**（`services/chat_service.py` 中 `ConversationMemory` 类）
- 压缩触发：message_count > 20（>10 轮）
- 压缩逻辑：取最近 6 条消息保留，其余 + 现有 summary → 调用 LLM → 新 summary
- 上下文构建：summary（如有）+ 最近 6 条消息 → 填入 prompt 的 MessagesPlaceholder
- message_count 冗余字段加速判断

**验收标准**：
- 创建会话 → 提问 → SSE 流逐 token 返回 → done 事件收到
- sources 事件在第一个 token 之前收到，前端能渲染溯源卡片
- 切换 DEEPSEEK_MODEL 后重启，使用新模型回答
- 连续对话 11 轮 → 查看 DB 中 summary 字段已更新 → 后续回答仍能引用早期上下文
- ES 全文 + 语义检索有互补（验证同一个 query 两种方式召回不同 chunk）
- 不同 library 之间检索隔离（session 绑 library A，不会检索到 library B 的内容）

---

### Task 7: 前端页面开发

**目标**：完整的单页应用，科技简约暗色风格，所有功能页面可用。

**内容**：

**7a. 基础框架**
- `main.ts`：创建 app，注册 router + Pinia + Naive UI
- `App.vue`：Naive UI `n-config-provider`（暗色主题）+ `<router-view>`
- `router/index.ts`：路由定义 + 导航守卫（未登录跳 /login）
- `stores/auth.ts`：登录状态、token、用户信息、登录/登出方法
- `api/index.ts`：Axios 实例（baseURL、自动带 Authorization header、401 拦截跳登录）

**7b. 页面开发**（按顺序）
1. `LoginView.vue` — 登录表单，成功后跳转 /
2. `RegisterView.vue` — 注册表单，成功后跳转 /login
3. `AppLayout.vue` — 全局布局：左侧 `n-menu` 导航 + 右侧 `router-view`，根据 role 显示/隐藏管理菜单
4. `DashboardView.vue` — 3 个统计卡片（`n-card` + `n-statistic`），调用 /api/dashboard
5. `LibraryListView.vue` — `n-data-table` 分页列表，admin 可见新建/删除按钮，`n-modal` 新建表单
6. `DocumentListView.vue` — 文档列表（分页），admin 可见上传按钮（`n-upload`），状态列显示 processing/ready/failed，点击 ready 文档可查看 chunks 预览
7. `ChatView.vue` — 核心页面：

**7c. ChatView 子组件**
- `SessionList.vue`：左侧会话列表，新建/切换/删除会话
- `ChatMessage.vue`：单条消息气泡，assistant 消息嵌入 `SourceCard.vue`（文档名、标题路径、页码、chunk 序号/总数）
- `ChatInput.vue`：底部输入框 + 发送按钮，发送中禁用

**7d. SSE 流式接收**（`utils/sse.ts`）
```typescript
async function* streamChat(sessionId, content): AsyncGenerator<SSEEvent>
// 解析 SSE 协议，yield { type: 'sources' | 'token' | 'done', data }
// 捕获网络异常 → 抛出 StreamError → 上层显示"连接中断,重试"按钮
```

**7e. 状态管理**
- `stores/chat.ts`：当前会话、消息列表、流式接收中状态、断连状态

**验收标准**：
- 所有页面渲染正常，暗色主题统一
- 注册 → 登录 → 看到仪表盘统计
- admin 可见管理菜单，普通用户不可见
- 上传文档 → 文档列表 status 自动更新（轮询）
- 聊天：提问后 sources 卡片先出现 → 逐字流式追加答案 → 溯源信息正确
- 断网模拟：SSE 断开后显示"连接中断"和重试按钮
- 浏览器刷新后登录状态保持（token 存 localStorage）

---

### Task 8: 联调与验收

**目标**：端到端全流程跑通，覆盖全部验收场景。

**内容**：
- 准备测试文档（每个格式各一份：PDF、DOCX、DOC、TXT、MD，中文内容）
- 按第 13 节验证计划逐条执行
- 修复联调中发现的 bug
- 性能检查：
  - 文档处理耗时（千字级文档应在 30 秒内完成）
  - 问答首 token 延迟（应在 3 秒内）
  - SSE 流无卡顿
- 补充 README.md：
  - 项目简介
  - 快速启动（`cp .env.example .env` → 编辑配置 → `docker compose up -d`）
  - 默认 admin 账号说明
  - 模型切换说明
  - 镜像体积说明

**验收标准**：
- 13 条验证计划全部通过
- README 按步骤操作可成功启动并完成一次问答
- 无未处理的异常崩溃
