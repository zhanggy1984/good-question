# 不懂就问

多用户 RAG 文档问答系统。你上传文档到"文档库"，系统自动完成 **内容抽取 → 清洗 → 切片 → 向量化 → 索引**，之后你可以在聊天中提问，系统从文档里"查"出相关内容，交给大模型生成**带来源标注、流式返回**的答案。

> **一句话理解 RAG**：RAG（Retrieval-Augmented Generation，检索增强生成）= 先从你的私有文档里检索出最相关的片段，再让大模型基于这些片段回答。这样大模型就能回答"它的训练数据里根本没有"的私有内容，且答案有出处、可溯源、不凭空编造。

本 README 分两大部分，**第一部分讲"这个系统能做什么"（功能），第二部分讲"它内部怎么实现的"（架构与技术）**。完全不了解本项目的人，按顺序读即可。

---

## 目录

1. [项目概览](#1-项目概览)
2. [第一部分 · 功能介绍](#第一部分--功能介绍)
   - [2.1 角色与权限](#21-角色与权限)
   - [2.2 端到端使用流程](#22-端到端使用流程)
   - [2.3 功能清单](#23-功能清单)
   - [2.4 检索质量是怎么保证的](#24-检索质量是怎么保证的)
3. [第二部分 · 架构与技术实现](#第二部分--架构与技术实现)
   - [3.1 系统架构总览](#31-系统架构总览)
   - [3.2 技术栈选型](#32-技术栈选型)
   - [3.3 各服务职责](#33-各服务职责)
   - [3.4 文档处理管线](#34-文档处理管线)
   - [3.5 混合检索管线](#35-混合检索管线)
   - [3.6 LLM 流式生成](#36-llm-流式生成)
   - [3.7 数据模型](#37-数据模型)
   - [3.8 多轮对话记忆](#38-多轮对话记忆)
4. [快速启动](#4-快速启动)
5. [配置说明](#5-配置说明)
6. [项目目录结构](#6-项目目录结构)
7. [开发与测试](#7-开发与测试)
8. [已知限制与优化方向](#8-已知限制与优化方向)

---

## 1. 项目概览

| 维度 | 说明 |
|------|------|
| 产品形态 | Web 应用（浏览器访问），前端 + 后端 + 多个数据服务 |
| 核心功能 | 文档库管理、文档上传自动处理、带溯源的智能问答 |
| 支持格式 | PDF / DOCX / TXT / Markdown |
| 技术栈 | FastAPI + LangChain 1.x + Vue 3 + MySQL + Milvus + DeepSeek |
| 部署方式 | Docker Compose 一键启动（7 个服务） |
| 交互方式 | 聊天问答 SSE 流式输出（文字逐字出现）+ 溯源卡片 |
| 数据安全 | 多用户隔离：每个人的会话只能看到自己绑定文档库的内容 |

---

## 第一部分 · 功能介绍

### 2.1 角色与权限

系统内置两种角色：**管理员（admin）** 与 **普通用户（user）**。二者登录后**看到的界面不同、能做的操作不同**——简单说：**admin 负责"喂内容"，普通用户负责"用内容"**。

#### 管理员（admin）—— 系统管理 + 内容运营

登录后**全部菜单可见**（仪表盘 / 文档库 / 聊天问答），负责建库、传文档、维护内容：

- 📊 **仪表盘**：查看文档库总数、文档总数、片段总数三个统计。
- 📚 **文档库管理**：创建 / 删除文档库（删除时级联清理文档、切片、向量、全文索引）。
- 📄 **文档管理**：上传 / 删除文档；上传时可为每篇文档设置切分参数（chunk 长度 / 重叠 token），并可查看切片预览。
- 💬 **聊天问答**：与普通用户相同。
- 管理员账号：系统首次启动时根据 `.env` 的 `ADMIN_USERNAME` / `ADMIN_PASSWORD` 自动创建。

#### 普通用户（user）—— 只问答，不管理

登录后**只看到"聊天问答"一个菜单**，仪表盘、文档库菜单均不显示；即使手动改 URL 访问 `/libraries`、`/` 等页面，也会被前端路由守卫重定向回聊天页。

- 💬 **聊天问答**：使用管理员创建好的文档库进行问答（可创建 / 删除自己的会话，会话归属后端强制隔离）。
- 🚫 **不可做**：创建 / 删除文档库、上传 / 删除文档——API 层面同样返回 403（后端用 `get_admin_user` 依赖校验）。
- 注册方式：在登录页点"注册"自助创建账号，无需管理员审批。
- 会话隔离：通过 API 也**无法访问他人会话**（后端强制校验归属）。

> 对比一句话：**admin 有"管理菜单 + 聊天"**，**普通用户只有"聊天"**。

### 2.2 端到端使用流程

```
① 管理员登录 → ② 创建文档库 → ③ 在库里上传文档（可设置切分参数）
   → ④ 等文档状态变"就绪" → ⑤ 进入聊天页，选文档库、新建会话
   → ⑥ 提问 → ⑦ 答案流式返回 + 溯源卡片 → ⑧ 多轮追问
```

**管理员视角的典型操作：**

1. **创建文档库**（如"技术知识库"）—— 一个库就像一个大文件夹，文档按库组织。
2. **上传文档** —— 上传前**可设置该文档的切分参数**：
   - **chunk 长度**（默认 1024）：文档被切成多少 token 一段。越大每段信息越完整，但检索时越"粗"；越小越精细但可能切碎。
   - **重叠 token**（默认 102）：相邻两段之间重叠多少 token，避免一句关键内容恰好被切断在边界。
   - 不同文档可以用不同配置（比如大 PDF 用 1024，短小说明用 512），系统按每篇文档自己的配置切分。
3. **查看处理进度** —— 上传后文档状态从"处理中"变成"就绪"，处理期间显示"已处理 N 段"实时进度。
4. **点开文档** —— 可预览这篇文档切出来的所有片段（chunk），确认切分效果。

**普通用户视角：** 登录后**直接落在聊天页**（只显示"聊天问答"一个菜单）→ 选一个文档库 → 新建会话 → 提问，就能获得基于该库文档的答案。文档库是管理员提前建好的，普通用户无需、也无法管理。

### 2.3 功能清单

| 功能 | 说明 |
|------|------|
| 🔐 **用户认证** | 注册、登录（JWT Token）、登出；登录状态刷新后保持 |
| 📊 **仪表盘**（admin） | 文档库总数、文档总数、切片总数三个统计卡片 |
| 📚 **文档库管理**（admin） | 创建/删除文档库，删除时级联清理文档、切片、向量、全文索引 |
| 📄 **文档管理**（admin） | 上传（多格式）、删除；上传时按文档配置 chunk 长度与重叠 token |
| 📃 **文档状态轮询** | 上传后前端自动轮询处理进度，处理中显示已处理片段数，完成变"就绪" |
| 🔍 **Chunk 预览** | 点开文档查看所有切分片段的原文与 token 数，便于检查切分质量 |
| 💬 **聊天问答** | 绑定文档库的会话制问答，SSE 流式逐字输出 |
| 📎 **溯源卡片** | 每条回答带来源：文档名、标题路径、片段序号、片段内容预览 |
| 💭 **思考过程** | DeepSeek 的思考过程单独折叠展示（"💭 思考过程"） |
| 🧠 **多轮记忆** | 连续对话保留上下文；超过 10 轮自动把早期对话压缩成摘要，长对话不丢上下文也不爆 prompt |
| 🚫 **未命中处理** | 文档里确实没有相关内容时，如实回答"文档中未找到相关信息"，不编造 |
| 🔌 **断连恢复** | SSE 连接中断时前端提示并可重试 |

### 2.4 检索质量是怎么保证的

系统用**三种检索手段 + 一道精排**保证"查得准"：

1. **语义检索（向量）**：理解"意思"——"怎么搭环境"和"环境安装步骤"是同一意思，靠向量相似度匹配。
2. **全文检索（关键词）**：精确匹配——专有名词、代码标识符、文件名这些语义检索常失灵的词，靠 Milvus BM25（内置中文分词）精确命中。
3. **精排（Rerank）**：把两路召回的结果合并去重后，用一个专门的"相关性打分模型"逐段重新排序，取最相关的 top-3 给大模型。

两路召回互补（语义找"相近的意思"，全文找"精确的词"），精排兜底质量。这是本系统检索质量的核心设计。Milvus 将两路召回合一（`hybrid_search` + RRF），检索存储统一、可横向扩展。

---

## 第二部分 · 架构与技术实现

### 3.1 系统架构总览

```
                    ┌────────────────────────────────────────────────┐
                    │                用户浏览器 (Vue 3)               │
                    │   登录/仪表盘/文档库/文档/聊天(SSE 流式)          │
                    └──────────────────────┬─────────────────────────┘
                                           │ http://localhost:80
                    ┌──────────────────────▼─────────────────────────┐
                    │              nginx (rag-nginx)                  │
                    │   静态文件服务 + /api 反向代理 + SSE 关缓冲      │
                    └──────────────────────┬─────────────────────────┘
                                           │ /api/*  → backend:8080
                    ┌──────────────────────▼─────────────────────────┐
                    │           backend (FastAPI, rag-backend)        │
                    │  认证 │ 文档处理管线 │ 混合检索 │ 聊天流式        │
                    └───┬─────────┬──────────┬──────────┬─────────────┘
                        │         │          │          │
              ┌─────────▼──┐ ┌────▼─────────▼──┐ ┌─────▼────────────┐
              │ MySQL      │ │ Milvus Standalone│ │ DeepSeek(外部)   │
              │ 关系数据    │ │ 语义(dense)+BM25 │ │ 大模型           │
              │ 文档/用户/  │ │ 混合检索         │ │ OpenAI 兼容流式  │
              │ 会话/消息   │ │ 依赖 etcd+MinIO  │ │                  │
              └────────────┘ └─────────────────┘ └──────────────────┘
```

一次问答的完整链路：

```
用户提问
  → backend 混合检索（Milvus dense + BM25 → RRF 融合 → Rerank 精排 top-3）
  → 组装 prompt（检索片段 + 会话历史）
  → 直连 DeepSeek 流式生成（解析思考过程 + 正文）
  → 前端按 SSE 事件顺序渲染：先溯源卡片 → 再逐字答案 → 完成
```

### 3.2 技术栈选型

| 层 | 技术 | 选型理由 |
|----|------|----------|
| Web 框架 | **FastAPI + Uvicorn** | Python 异步性能好，SSE 流式天然支持，Pydantic 校验 |
| AI 管线 | **LangChain 1.x**（core/text-splitters/openai） | 切片、检索器、LLM 封装标准化，1.x 版本更稳定 |
| 前端 | **Vue 3 + Naive UI + Pinia** | 组合式 API，SSE 流式渲染友好，组件库齐全 |
| 文档解析 | **MinerU 3.4**（本地 CLI / 官方 API 可切换） | 中文 PDF/DOCX 结构化解析能力强，输出 Markdown 保留标题层级 |
| 向量模型 | **jina-embeddings-v2-base-zh**（768 维，FastEmbed ONNX） | 中文语义强、可国内下载；ONNX 免 torch 依赖 |
| 精排模型 | **bge-reranker-v2-m3**（sentence-transformers CrossEncoder） | 在候选片段间做相关性排序，比向量匹配更准（多语言，较旧 v1-base 更准） |
| 检索存储 | **Milvus 2.5 Standalone** | dense 语义 + BM25 全文一体；单 collection + partition 库隔离；可横向扩展 |
| 关系库 | **MySQL 8.0 + SQLAlchemy + Alembic** | 文档/用户/会话/消息持久化，迁移管理 |
| 大模型 | **DeepSeek**（OpenAI 兼容协议，模型可切换） | 流式输出 + reasoning_content 思考过程 |
| 部署 | **Docker Compose**（7 服务） | 一键启动，数据卷持久化，健康检查依赖编排 |

### 3.3 各服务职责

| 服务 | 容器名 | 职责 |
|------|--------|------|
| **backend** | rag-backend | 唯一业务服务：认证、文档处理、检索、聊天；8080 端口 |
| **nginx** | rag-nginx | 前端静态文件 + `/api` 反向代理 + SSE 关缓冲；80 端口（对外唯一入口） |
| **mysql** | rag-mysql | 持久化用户/文档/会话/消息/切片元数据 |
| **etcd** | rag-etcd | Milvus 元数据存储 |
| **minio** | rag-minio | Milvus 对象存储 |
| **milvus** | rag-milvus | dense 语义 + BM25 全文混合检索（standalone，可扩分布式） |
| **attu** | rag-attu | Milvus 可视化管理 UI（collection/partition/数据浏览/查询）；8000 端口 |

> 用户只需访问 `http://localhost`（nginx 80 端口），无需直接接触任何后端端口。
> Milvus 可视化管理通过 `http://localhost:8000`（Attu，连接地址预填 `milvus:19530`）。

### 3.4 文档处理管线

上传文档后，backend 在**后台线程**异步执行（不阻塞上传响应），分 6 个阶段：

```
文件落盘
  → ① 文本抽取   MinerU 结构化解析（PDF/DOCX），失败降级 PyMuPDF/python-docx
  → ② 文本清洗   去乱码、规范空白（text_cleaner）
  → ③ 智能切片   RecursiveCharacterTextSplitter，按该文档的 chunk_size/overlap_token
                 Markdown 按标题分节；上限 MAX_CHUNKS=2000 防极端大文件
  → ④ 向量化     jina 模型分批嵌入（每批 64 段），连同 BM25 全文一并写入 Milvus（进度逐批写库）
  → ⑤ 写 MySQL   chunks 表（完整元数据，为源数据；Milvus 统一承担语义与全文检索）
  → ⑥ 标记就绪   更新文档状态为 ready，记录 chunk_count
```

关键实现点：

- **切分配置按文档维度**：每篇文档在 `documents` 表里存自己的 `chunk_size` / `overlap_token`（上传时设置，默认 1024 / 102），切分阶段从文档记录读取。已上传文档不会自动重切。
- **每阶段检查文档是否被删除**：文档被删除后处理线程尽快释放，不产生"孤儿"数据。
- **进度可见**：向量化是耗时阶段，每批写入后把进度写库，前端轮询即可看到"处理中 N 段"。
- **失败处理**：任何阶段失败 → 文档标记 failed + 错误信息；已写入 Milvus 的向量自动清理，保证存储一致。

### 3.5 混合检索管线

```
提问
  → ① 混合检索   Milvus hybrid_search：dense 语义 top-3 + BM25 全文 top-3
                 （同一 collection partition 内，服务端 RRF 融合去重，取 top-6 候选）
  → ② Rerank 精排  BGE-Reranker 对融合后的候选逐段打分，按分数倒序取 top-3
  → ③ 无结果判定  最高分 < 0.2（SIMILARITY_THRESHOLD_LOW）才判定"文档无关"，返回空
```

设计决策（踩过的坑）：

- **不设绝对阈值过滤**：实测 Rerank 的绝对分数不可靠——最相关的片段可能只得 0.27 分，用绝对阈值会误杀（曾导致某问题明明检索到了正确内容却返回空）。因此改为**相对排序**（取分数最高的 top-3），仅当最高分也低到离谱才判定无关。
- **Rerank 输入只用正文**：打分时只用 chunk 正文（`page_content`），标题层级等元数据单独存在 metadata 里，不拼进打分文本。
- **不用 LLM 改写问题**：曾用大模型先把口语化问题规范化再检索，实测改写收益趋零、还每次多花 2-4 秒，已停用，直接用原问题检索。
- **BM25 路降级**：BM25 路异常时自动降级为纯 dense 语义检索，不阻塞主链路。

### 3.6 LLM 流式生成

- **直连 DeepSeek**：不用 LangChain 的 ChatOpenAI 流式（它拿不到 DeepSeek 的 `reasoning_content` 思考过程），改为 `httpx` 直连 DeepSeek 流式接口，逐行解析 SSE。
- **两种内容流**：
  - `reasoning_content`（思考过程）→ 单独推 `reasoning` 事件，前端折叠展示
  - `content`（正式答案）→ 推 `token` 事件，前端逐字渲染
- **SSE 事件顺序**：

```
event: sources   →  首次检索完成后，推溯源卡片数据（文档名/标题路径/片段序号/片段预览）
event: reasoning →  DeepSeek 思考过程（可能多段）
event: token     →  正式答案，逐字推送（多段）
event: done      →  完成，携带消息 id
event: error     →  LLM 调用失败（异常兜底）
```

- **无结果时不推 sources**：检索为空时不发送空引用卡片，大模型如实回答"文档中未找到"。
- **回答要求写死在 system prompt**：必须依据文档回答、用 `[来源N]` 标注引用、不得编造文档中不存在的具体事实。

### 3.7 数据模型

6 张核心表（SQLAlchemy + Alembic 迁移管理，`init.sql` 仅建库）：

| 表 | 关键字段 | 说明 |
|----|----------|------|
| `users` | username, password_hash(bcrypt), role | 用户（admin / user） |
| `document_libraries` | name, description, created_by | 文档库 |
| `documents` | library_id, filename, file_path, file_type, file_size, chunk_count, **chunk_size, overlap_token**, status, error_message, uploaded_by | 文档 + 切分配置 |
| `chunks` | document_id, chunk_index, content, token_count, metadata_json(JSON) | 切片，metadata_json 存完整溯源元数据 |
| `chat_sessions` | user_id, library_id, title, summary, message_count | 聊天会话（title=首问截取，summary=压缩摘要） |
| `chat_messages` | session_id, role, content, sources_json | 聊天消息（assistant 消息带溯源） |

**Chunk 元数据**（`metadata_json`，同时用于溯源展示与维护）：`document_id` / `document_name` / `library_id` / `chunk_index` / `total_chunks` / `heading_path`（标题路径）/ `source_type` / `token_count` 等。

### 3.8 多轮对话记忆

- **上下文构建**：每次提问时，把该会话最近 6 条消息（3 轮）+ 摘要（如有）+ 本次检索片段一起发给大模型。
- **记忆压缩**：超过 10 轮（20 条消息）后，把早期消息 + 现有摘要交给大模型压成一段新摘要，只保留最近 3 轮原文。长对话既不丢早期上下文，也不会让 prompt 无限膨胀。
- **消息顺序用自增主键排序**：历史消息按 `id`（插入顺序）而非 `created_at`（秒级时间戳）排序——同一次写入的 user/assistant 两条消息 `created_at` 相同，按时间排序会乱序，曾导致多轮问答"串味"（第二问答出第一问的内容），用 `id` 排序后彻底解决。

---

## 4. 快速启动

```bash
# ① 复制环境配置模板
cp .env.example .env

# ② 编辑 .env，至少填写：
#    - DEEPSEEK_API_KEY        （必填，DeepSeek API Key）
#    - ADMIN_PASSWORD          （必改，管理员密码）
#    - JWT_SECRET_KEY          （建议改，JWT 签名密钥）
#    - MINERU_API_TOKEN        （可选；填了走 MinerU 官方 API，留空用本地解析）

# ③ 启动（首次构建较久，见下方说明）
docker compose up -d

# ④ 访问
open http://localhost
```

**默认账号**（首次启动自动创建，强烈建议改密码）：

- 管理员：`ADMIN_USERNAME` / `ADMIN_PASSWORD`（.env.example 默认 `admin` / `change_me_admin`）
- 普通用户：登录页点"去注册"自助注册

> ⚠️ **镜像体积与资源**：backend 镜像含 PyTorch(CPU) + MinerU + 模型依赖，约 **8-12GB**；首次构建会下载模型和依赖，耗时数十分钟。模型缓存挂载在 volume，重启不重复下载。Milvus Standalone + etcd + MinIO 内存占用约 2-4GB，请确保宿主机内存充足（建议 ≥ 8GB）。

## 5. 配置说明

关键环境变量（完整见 `.env.example`）：

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（**必填**） | - |
| `DEEPSEEK_MODEL` | LLM 模型名 | `deepseek-v4-pro` |
| `EMBEDDING_MODEL_NAME` | 向量模型（FastEmbed 支持） | `jinaai/jina-embeddings-v2-base-zh` |
| `RERANK_MODEL_NAME` | 精排模型（多语言，较 v1-base 更准；CPU 下更慢） | `BAAI/bge-reranker-v2-m3` |
| `SIMILARITY_THRESHOLD_LOW` | 无结果兜底阈值：rerank 最高分低于此值才判"文档无关"返回空 | `0.20` |
| `MILVUS_HOST/PORT` | Milvus 连接地址 | `milvus` / `19530` |
| `ATTU_PORT` | Attu 可视化 UI 端口 | `8000` |
| `MINERU_API_TOKEN` | MinerU 官方 API Token（空=本地解析） | 空 |
| `HF_ENDPOINT` | HuggingFace 下载镜像（国内加速） | `https://hf-mirror.com` |
| `ADMIN_USERNAME/PASSWORD` | 管理员账号 | `admin` / `change_me_admin` |
| `JWT_SECRET_KEY` | JWT 签名密钥（生产必须改） | `change-me` |

**模型切换**：改 `.env` 里对应模型名 → `docker compose up -d --force-recreate backend` 重启 backend 生效。模型文件缓存于 volume，切换后首次会重新下载。

## 6. 项目目录结构

```
native-rag/
├── docker-compose.yml        # 7 服务编排（mysql/etcd/minio/milvus/attu/backend/nginx）
├── .env / .env.example       # 环境配置
├── init.sql                  # 建库脚本（表结构由 Alembic 迁移管理）
├── backend/                  # FastAPI + LangChain 后端
│   ├── main.py               # 入口 + lifespan（模型预热/admin 种子）
│   ├── config.py             # pydantic-settings 读取 .env
│   ├── api/                  # 路由层（薄）：auth/dashboard/library/document/session/chat
│   ├── services/             # 业务逻辑：document_service(处理管线) / retrieval_service(混合检索)
│   │                         #          chat_service(问答/记忆/流式) / embedding_service / vector_store_service / llm_service
│   ├── scripts/              # migrate_to_milvus.py：从 MySQL chunks 幂等重灌 Milvus（升级迁移）
│   ├── models/               # SQLAlchemy ORM（6 张表）
│   ├── schemas/              # Pydantic 请求/响应
│   ├── middleware/           # JWT 鉴权 Depends
│   ├── utils/                # mineru_extractor / text_cleaner / chunker
│   └── alembic/              # 数据库迁移
├── frontend/                 # Vue 3 + Naive UI 前端
│   ├── Dockerfile            # 多阶段：node 编译 → nginx 服务
│   ├── nginx.conf            # gzip + /api 反代 + SPA fallback + SSE 关缓冲
│   └── src/
│       ├── api/              # Axios 封装
│       ├── stores/           # Pinia（auth/chat）
│       ├── views/            # 页面（登录/注册/仪表盘/文档库/文档/chunk详情/聊天）
│       ├── components/       # 布局 + 溯源卡片等
│       └── utils/sse.ts      # SSE 流式接收 + 断连重试
├── test-data/                # Milvus 迁移验证脚本（chat 完整链路 / e2e 写入 / 旧数据验证）
└── data/                     # 上传文件存储（uploads/）
```

**后端代码分层**：`api/`（路由，薄）→ `services/`（业务逻辑）→ `models/` + `utils/`（数据与基础设施）。新增功能一般改 `api/` + `services/` 即可。

## 7. 开发与测试

```bash
# 后端开发（热重载）
cd backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8080

# 前端开发（Vite dev server，代理 /api 到 8080）
cd frontend && npm install && npm run dev
```

**生产构建前端**（改了前端代码后**必须先 build**，否则 nginx 里是旧页面）：

```bash
cd frontend && npm run build
cd .. && docker compose build nginx && docker compose up -d nginx
```

**单元测试**（核心逻辑纯函数测试，不依赖外部服务）：

```bash
docker exec -it rag-backend bash    # 进 backend 容器
cd /app && python -m pytest tests/ -v
```

**Milvus 数据迁移 / 迁移验证**（从旧版检索存储升级，或验证写入/检索链路）：

```bash
# ① 从 MySQL chunks 幂等重灌 Milvus（可重复执行，主键 upsert 不产生重复数据）
docker exec -it rag-backend bash
cd /app && python scripts/migrate_to_milvus.py

# ② 迁移验证脚本（项目根目录，对 http://localhost 执行；登录凭据默认读环境变量 RAG_ADMIN_USER / RAG_ADMIN_PASS）
python test-data/e2e.py                # 端到端：登录→建库→上传→轮询就绪（验证 Milvus 写入链路）
python test-data/chat.py               # chat 完整链路：登录→建会话→提问→读 SSE 事件
python test-data/verify_old_data.py    # 旧数据迁移后检索链路验证（MySQL 迁移 + Milvus 重灌）
```

**常用运维命令**：

```bash
docker compose up -d                          # 启动全部
docker compose up -d --force-recreate backend # 改后端代码后重启 backend
docker compose logs -f backend                # 看后端日志（排查检索/处理问题）
docker compose ps                             # 查看服务健康状态
```

## 8. 已知限制与优化方向

**如实说明当前已知的性能与边界问题：**

1. **Rerank 是检索延迟的绝对瓶颈**：Milvus 混合检索（dense + BM25，RRF 融合）约 0.1s 可忽略，但 Rerank 精排在 CPU 上对候选文本对打分是主要耗时（每段数秒，chunk 越大越慢），当前 `bge-reranker-v2-m3` 比旧 v1-base 更准但也更重。候选数由 Milvus `hybrid_search` 的 `limit` 控制（默认 6）。
   - 可选方向：换更小的 rerank 模型、GPU 推理、或更激进的候选控制（均已评估，各有取舍，尚未落地）。
2. **`.doc` 格式暂不支持**：MinerU 不支持 `.doc`（老版 Word），需先另存为 `.docx` 再上传。
3. **本地 MinerU 解析较慢**：大 PDF 处理耗时取决于文档复杂度；如需加速可配置 `MINERU_API_TOKEN` 走官方 API（代价是文档内容上传到云端）。
4. **模型全 CPU 推理**：embedding 与 rerank 均为 CPU，重负载场景可考虑 GPU 容器。
5. **无 LLM 缓存**：相同的问与答会重复调用 DeepSeek（计费）。高频率场景可加缓存层。
6. **Milvus 中文分词弱于原 ES IK**：内置 `chinese` analyzer 对领域词/自定义词典支持有限，专业术语精确命中可能略降（Milvus 自定义词典规划在 v3.0）。
