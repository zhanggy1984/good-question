# 不懂就问（Native RAG）

> **多用户 RAG 文档问答系统**：文档上传即自动抽取 → 清洗 → 切片 → 向量化，之后基于文档库提问，大模型用**带来源标注、流式返回**的答案作答。私有知识有出处、可溯源、不编造，会话与文档库双隔离。

本系统是**契约对齐的 RAG 演示项目**：7 服务一键启动、nginx 80 端口为唯一入口、Milvus 统一混合检索（dense + BM25 + RRF 融合 + Rerank 精排）、两级置信档防幻觉、SSE 事件流对齐评测契约 v1.0（`meta`/`tool_call`/`usage` + 全事件 `ts`）、45 项单元测试 + 契约运行时验证脚本开箱即用。

---

## 目录

- [一、项目简介：解决什么痛点](#一项目简介解决什么痛点)
- [二、业务价值：给谁带来什么](#二业务价值给谁带来什么)
- [三、技术闪光点](#三技术闪光点)
- [四、系统架构](#四系统架构)
- [五、技术栈一览](#五技术栈一览)
- [六、快速开始（3 步跑起来）](#六快速开始3-步跑起来)
- [七、配置说明](#七配置说明)
- [八、演示场景（一键造数）](#八演示场景一键造数)
- [九、目录结构](#九目录结构)
- [十、测试与验收](#十测试与验收)
- [十一、开发指南](#十一开发指南)
- [十二、常见问题](#十二常见问题)
- [十三、已知限制与优化方向](#十三已知限制与优化方向)

---

## 一、项目简介：解决什么痛点

用大模型回答"你私有的文档"这件事，常见的坑有四个：

- **私有文档问不到**：公司政策、产品手册、合同条款散落在文档里，大模型训练数据里根本没有——直接问必然瞎编；
- **答案没出处**：就算答了，也说不出依据哪段原文，无法核验、无法追责；
- **知识无法隔离**：多人共用一套知识库，A 问到的内容串到 B 的库里，出了事找不到是谁泄露的；
- **等不起**：文档多、上下文大，普通请求要几十秒才回，用户没有耐心。

本系统针对这四个痛点，提供四个核心能力：

| 能力 | 实现 | 对应痛点 |
|------|------|---------|
| **私有文档知识化** | 上传即自动抽取 → 清洗 → 切片 → 向量化，问答只基于你上传的内容 | 私有文档问不到 |
| **带溯源问答** | 混合检索（语义 + BM25 + Rerank）+ DeepSeek 生成，答案用 `[来源N]` 标注引用 | 答案没出处 |
| **多用户隔离** | 文档库 + 会话双隔离，归属后端强制校验，普通用户 API 层也拿不到他人数据 | 知识无法隔离 |
| **流式体验** | SSE 逐字返回，思考过程单独折叠展示，首包秒回、不干等 | 等不起 |

> **一句话理解 RAG**：RAG（Retrieval-Augmented Generation，检索增强生成）= 先从你的私有文档里检索出最相关的片段，再让大模型基于这些片段回答。这样大模型就能回答"它的训练数据里根本没有"的私有内容，且答案有出处、可溯源、不凭空编造。

---

## 二、业务价值：给谁带来什么

### 对管理员（内容运营，admin）
- **建库传文档**：一个文档库一个主题，文档按库组织；每篇文档可**独立设置切分参数**（chunk 长度 / 重叠 token），大 PDF 用 1024、短说明用 512，按文档自身配置切分；
- **进度可见**：上传后异步处理，处理中实时显示"已处理 N 段"，完成变"就绪"，点开可预览全部切片确认切分质量；
- **级联清理**：删除文档库时文档、切片、向量、全文索引一并清理，不留脏数据。

### 对普通用户（user）
- **只问答，不管理**：登录后只看到"聊天问答"，仪表盘、文档库菜单均不显示，改 URL 也会被前端路由守卫重定向；
- **会话制问答**：绑定一个文档库，可创建/删除多个会话，会话之间完全隔离；
- **看得见的过程**：答案带溯源卡片（文档名 / 标题路径 / 片段序号 / 片段预览）+ 思考过程折叠展示，不是黑盒结论；
- **如实回答**：文档里没有的，明确说"文档中未找到相关信息"，绝不编造。

### 对评测 / 验收方
- **接口自动发现**：公开 `GET /api/contracts` 声明本 agent 的 LLM 接口与场景清单，平台脚手架直接读取，无需人工配置；
- **契约对齐**：SSE 事件流严格对齐评测契约（`meta`/`tool_call`/`usage` + 全事件 `ts`），`reasoning`/`token` 双字段兼容前端与评测侧，`usage` 透传真实 token 消耗；
- **一键验证**：`python verify_contract.py` 运行时验证完整事件流，字段缺失、顺序错误立即暴露。

---

## 三、技术闪光点

### 1. Milvus 统一混合检索：双存储合一
检索存储从 **ChromaDB（语义）+ Elasticsearch（全文）双存储**，迁移为 **Milvus 2.5 Standalone 单存储**：dense 语义 + BM25 全文在同一 collection 内 `hybrid_search`，服务端 RRF 融合去重，partition 级做文档库隔离，可横向扩展。配套 `scripts/migrate_to_milvus.py` 幂等重灌脚本（主键 upsert，可重复执行）与 `test-data/verify_old_data.py` 迁移验证，升级不丢失数据。

### 2. 两级置信档：防幻觉，也不误杀
实测 Rerank 的**绝对分数不可靠**——最相关的片段可能只得 0.27 分，用绝对阈值会误杀（曾导致"明明检索到了却返回空"）。因此：

- **相对排序取 top-3**，不做绝对阈值过滤；
- **最高分 < 0.2**（`SIMILARITY_THRESHOLD_LOW`）判"文档无关"，返回空、不发空引用，LLM 如实说"未找到"；
- **最高分 ∈ [0.2, 0.5)**（`RERANK_LOW_CONFIDENCE_THRESHOLD`）判"相关性存疑"——检索结果**照常保留**（保住召回、不误杀低分相关片段），但在 system prompt 追加提示，让 LLM"不足以回答就如实说未找到"，杜绝基于边缘相关片段编造。

### 3. 直连 DeepSeek 流式：思考过程 + 真实 token
不用 LangChain `ChatOpenAI` 的流式（它拿不到 DeepSeek 的 `reasoning_content` 思考过程），改为 `httpx` **直连 DeepSeek 流式接口**逐行解析 SSE：

- `reasoning_content`（思考过程）与 `content`（正式答案）分流推送，前端把思考过程单独折叠展示；
- `stream_options.include_usage` 显式开启，透传**真实 token 消耗**（默认不返回），供计费与评测。

### 4. SSE 事件流对齐评测契约（meta / tool_call / usage + ts）
`/api/chat/{session_id}` 的 SSE 事件序对齐评测契约 v1.0（路径 A：保留原事件名，前端零改动）：

```
event: meta       环境快照（agent/model/interface/contract_version/git_sha/knowledge_version/ts）
event: tool_call  检索外显为标准工具调用（name=hybrid_retrieve，仅检索到结果时）
event: sources    溯源卡片数据（仅检索到结果时）
event: reasoning  DeepSeek 思考过程（content + delta + ts，多段）
event: token      正式答案逐字推送（content + delta + ts，多段）
event: usage      真实 token 消耗（prompt/completion/total_tokens + ts，done 之前）
event: done       完成，携带 message_id
event: error      LLM 调用失败兜底
```

- **全事件带 `ts`**（unix 毫秒），`reasoning`/`token` 的 data 同时含 `content` + `delta`，前端读 `content`、评测平台经 field_map 读 `delta`，两侧都兼容；
- `meta` 中 `knowledge_version` 以**该会话文档库**最近上传时间为锚（按 `library_id` 过滤，多库部署不串库）；
- 公开 `GET /api/contracts` 声明接口与 4 个场景清单，平台自动发现；
- 前端 `sse.ts` 只消费 `sources`/`reasoning`/`token`/`done`/`error`，新增的 `meta`/`tool_call`/`usage` 事件自然忽略——路径 A，前端无需改动；
- **断连恢复**：`sse.ts` 网络中断抛 `StreamError`，上层提示用户重试，已渲染内容不丢失。

### 5. MinerU 结构化解析 + 失败降级
中文 PDF / DOCX 用 **MinerU 3.4.4** 结构化解析，输出 Markdown 并保留标题层级（标题路径用于溯源展示）；解析失败自动降级 PyMuPDF / python-docx。本地解析 / 官方 API（`MINERU_API_TOKEN`）可切换，大 PDF 赶时间可走官方 API。

### 6. 多轮记忆压缩：长对话不爆 prompt
- 上下文 = 最近 3 轮原文 + 压缩摘要 + 本次检索片段；
- 超过 10 轮（20 条消息）自动把早期对话压成一段 ≤200 字摘要，只保留最近 3 轮原文——既不丢早期上下文，也不让 prompt 无限膨胀；
- 历史消息按**自增主键 `id`** 排序而非 `created_at`（秒级时间戳排序不稳定，曾导致多轮问答"串味"——第二问答出第一问的内容），用 `id` 排序后彻底解决。

### 7. 多用户会话隔离
会话归属后端强制校验：普通用户即使改 URL / 直接调 API，也拿不到不属于自己的会话（403）。检索范围限定在会话绑定库的 partition 内，跨库天然隔离。

### 8. 幂等异步文档管线：失败不残留
上传后后台线程异步处理（不阻塞上传响应），6 个阶段：`落盘 → ①抽取 → ②清洗 → ③切片 → ④向量化 → ⑤写库 → ⑥就绪`。关键保证：

- **按文档维度切分**：每篇文档在 `documents` 表存自己的 `chunk_size` / `overlap_token`，切分阶段从文档记录读取；
- **进度逐批写库**：向量化分批写入 Milvus，逐批更新 `processed_chunks`，前端轮询可见"处理中 N 段"；
- **失败自动清理**：任何阶段失败 → 文档标记 `failed` + 错误信息，已写入 Milvus 的向量自动清理，存储保持一致；
- **删除及时释放**：每阶段检查文档是否被删，不留"孤儿"数据；
- **上限保护**：`MAX_CHUNKS=2000` 防极端大文件打爆内存。

---

## 四、系统架构

```mermaid
graph TB
    subgraph 前端
        WEB["Vue3 + Naive UI<br/>（chat/dashboard/library/document）"]
        NGINX["nginx :80<br/>静态服务 + /api 反代 + SSE 关缓冲（唯一对外入口）"]
    end
    subgraph 应用层
        API["backend FastAPI :8080<br/>认证 / 文档管线 / 混合检索 / 聊天流式"]
    end
    subgraph AI 服务
        DS["DeepSeek LLM<br/>生成 / 思考 / 记忆压缩"]
        EMBED["FastEmbed jina-embeddings-v2-base-zh<br/>768 维（CPU）"]
        RERANK["BGE-Reranker v2-m3<br/>精排（CPU）"]
    end
    subgraph 数据层
        MYSQL[(MySQL 8<br/>用户/文档/切片/会话/消息)]
        MILVUS[(Milvus 2.5 Standalone<br/>dense 语义 + BM25 全文<br/>依赖 etcd + MinIO)]
        ATTU["Attu :8000<br/>Milvus 可视化管理 UI"]
    end

    WEB --> NGINX
    NGINX --> API
    API -- SSE 流式 --> WEB
    API --> MYSQL
    API --> MILVUS
    API --> DS
    API --> EMBED
    API --> RERANK
    API -.辅助.-> ATTU
```

**一次问答的完整链路**：

```
用户提问
  → ① 混合检索   Milvus hybrid_search：dense 语义 top-3 + BM25 全文 top-3
                 （同一 partition 内，服务端 RRF 融合去重，取 top-6 候选）
  → ② Rerank 精排 BGE-Reranker 对候选逐段打分，按分数倒序取 top-3
  → ③ 置信分级   最高分 < 0.2 → 判"文档无关"返回空；
                 最高分 ∈ [0.2, 0.5) → 判"相关性存疑"，结果照常进 LLM 但追加兜底提示
  → ④ 组装 prompt（检索片段 + 最近 3 轮历史 + 摘要）
  → ⑤ 直连 DeepSeek 流式生成（reasoning_content 思考 + content 正文 + usage）
  → ⑥ SSE 事件序：meta → tool_call/sources? → reasoning*/token* → usage → done
  → ⑦ 前端按序渲染：溯源卡片 → 思考折叠 → 逐字答案 → 完成
```

---

## 五、技术栈一览

| 层 | 技术 | 说明 |
|----|------|------|
| 后端 | Python 3.11 + FastAPI + Uvicorn | async/await，OpenAPI 自动文档，SSE 流式原生支持 |
| AI 管线 | LangChain 1.x（core/text-splitters/openai） | 切片、检索器、LLM 封装标准化，1.x 更稳定 |
| 前端 | Vue 3 + Naive UI + Pinia | 组合式 API，SSE 流式渲染友好 |
| 文档解析 | MinerU 3.4.4（pipeline backend） | 中文 PDF/DOCX 结构化解析，失败降级 PyMuPDF/python-docx |
| 向量模型 | jina-embeddings-v2-base-zh（768 维，FastEmbed ONNX） | 中文语义强、可国内下载，ONNX 免 torch 依赖 |
| 精排模型 | BGE-Reranker v2-m3（CrossEncoder） | 候选间相关性排序，比向量匹配更准 |
| 检索存储 | Milvus 2.5 Standalone | dense 语义 + BM25 全文一体，RRF 融合，partition 库隔离 |
| 关系库 | MySQL 8.0 + SQLAlchemy 2 + Alembic | 6 张表，迁移统一管理 |
| 大模型 | DeepSeek（OpenAI 兼容协议，模型可切换） | 流式输出 + reasoning_content 思考过程 + usage |
| 部署 | Docker Compose（7 服务） | 一键启动，数据卷持久化，健康检查编排 |

---

## 六、快速开始（3 步跑起来）

> 前置：Docker Desktop（Linux 容器）、Python 3.11（跑宿主机验证脚本用）。

### 第 1 步：配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填写：
#   DEEPSEEK_API_KEY=sk-xxx        # DeepSeek API Key（问答必需）
#   ADMIN_PASSWORD=xxx             # 管理员密码（务必修改）
#   JWT_SECRET_KEY=xxx             # JWT 签名密钥（建议修改）
#   MINERU_API_TOKEN=              # 可选；留空用本地解析，填了走 MinerU 官方 API
```

### 第 2 步：一键启动全栈（7 个服务）

```bash
docker compose up -d
# 首次含镜像构建（torch CPU + MinerU 依赖，体积 8-12GB、耗时数十分钟）
docker compose ps                 # mysql/etcd/minio/milvus/attu/backend/nginx 全 healthy
curl localhost:8080/api/health    # {"status":"ok","service":"Native RAG"}
```

### 第 3 步：访问

```bash
open http://localhost          # 浏览器前端（nginx 80 端口）
open http://localhost:8000     # Attu：Milvus 可视化管理 UI
```

**默认账号**（首次启动自动创建，强烈建议改密码）：

| 角色 | 账号 | 密码 |
|------|------|------|
| 管理员 | `ADMIN_USERNAME`（.env，默认 `admin`） | `ADMIN_PASSWORD`（.env，默认 `change_me_admin`） |
| 普通用户 | 登录页点"去注册"自助注册 | — |

> ⚠️ **资源占用**：backend 镜像含 PyTorch(CPU) + MinerU + 模型，约 8-12GB；Milvus Standalone + etcd + MinIO 内存占用约 2-4GB，建议宿主机内存 ≥ 8GB。模型缓存挂载在 volume，重启不重复下载。

---

## 七、配置说明

关键环境变量（完整见 `.env.example`，每项含注释）：

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（**必填**） | - |
| `DEEPSEEK_MODEL` | LLM 模型名（切换只改这一项） | `deepseek-v4-pro` |
| `EMBEDDING_MODEL_NAME` | 向量模型（FastEmbed 支持） | `jinaai/jina-embeddings-v2-base-zh` |
| `RERANK_MODEL_NAME` | 精排模型（多语言，较 v1-base 更准；CPU 更慢） | `BAAI/bge-reranker-v2-m3` |
| `SIMILARITY_THRESHOLD_LOW` | 无结果兜底阈值：rerank 最高分低于此值才判"文档无关"返回空 | `0.20` |
| `RERANK_LOW_CONFIDENCE_THRESHOLD` | 低置信阈值：最高分落在 [LOW, 此值) 判"相关性存疑"，LLM 被提示如实回答 | `0.50` |
| `MILVUS_HOST/PORT` | Milvus 连接地址 | `milvus` / `19530` |
| `ATTU_PORT` | Attu 可视化管理 UI 端口 | `8000` |
| `MINERU_API_TOKEN` | MinerU 官方 API Token（空 = 本地解析） | 空 |
| `HF_ENDPOINT` | HuggingFace 下载镜像（国内加速） | `https://hf-mirror.com` |
| `ADMIN_USERNAME/PASSWORD` | 管理员账号 | `admin` / `change_me_admin` |
| `JWT_SECRET_KEY` | JWT 签名密钥（生产必须改） | `change-me` |

**模型切换**：改 `.env` 里对应模型名 → `docker compose up -d --build backend` 重启生效（换模型须保证向量维度与 Milvus collection 一致）。模型文件缓存于 volume，切换后首次会重新下载。

---

## 八、演示场景（一键造数）

> 演示数据**客观可复现**：`test-data/seed_demo.py` 一键重建"演示知识库"（删旧库 → 上传 4 份中文文档 → 等就绪 → 打印问题清单）。演示文档用 `.md`（抽取走明文读取，无需 MinerU），容器就绪即可秒级造数。

### 一键造数

```bash
docker compose up -d                   # 7 服务就绪（首次构建较久）
python test-data/seed_demo.py          # 登录 → 重建演示库 → 上传 4 份文档 → 等就绪 → 打印问题清单
open http://localhost                   # admin 登录 → 聊天页选"演示知识库" → 复制场景问题提问
```

幂等可重跑：重复执行会删除已存在的演示库并重建（会话/文档/向量级联清理），结果一致。

> ⚠️ 演示/验证脚本默认以 `admin / admin123` 登录（`config.py` 默认值）；若已修改 `ADMIN_PASSWORD`，请用环境变量 `RAG_ADMIN_USER` / `RAG_ADMIN_PASS` 覆盖后运行。

### 6 个演示场景

| 场景 | 提问（复制即用） | 预期 | 观看点 |
|------|------------------|------|--------|
| **1 · 事实问答** | 公司规定请事假需要提前几天申请？ | 答案命中考勤制度，带 `[来源N]` | 溯源卡片显示"员工考勤管理制度 > 请假管理"标题路径 |
| **2 · 技术检索** | Docker 的常用命令有哪些？ | 答案列出手册中的命令 | 答案逐字流式 + 思考过程折叠 |
| **3 · 无命中兜底** | 工资发放日是几号？ | 文档未写，如实回答"文档中未找到相关信息" | **不出现空引用卡片**（两级置信档下不编造） |
| **4 · 跨文档总结** | 分别用一句话总结四份文档的核心内容 | 答案基于多份文档总结 | 溯源卡片含多份文档来源 |
| **5 · 闲聊引导** | 你好 | 助手自我介绍并引导提问 | 问候不检索、不编造"文档查到" |
| **6 · 多轮追问** | 请事假扣工资吗？→ 年假有几天？→ 缺勤会怎样？ | 连续追问上下文连贯 | 提问超 10 轮自动压缩记忆，长对话不串味 |

### 评测场景（契约声明）

`GET /api/contracts` 声明的 4 个评测场景是**行为契约**（平台据此出题），与上述业务演示一一对应：

| 契约标签 | 对应演示 |
|---------|---------|
| `greeting` 问候闲聊 | 场景 5 |
| `doc_qa` 文档检索问答 | 场景 1 / 2 / 6 |
| `no_hit` 无命中兜底 | 场景 3 |
| `summarize` 文档总结 | 场景 4 |

**验收演示**（对运行中的服务）：

```bash
python verify_contract.py           # 契约事件流验证（meta → tool_call? → reasoning/token → usage → done）
python test-data/e2e.py             # 端到端：登录→建库→上传→轮询就绪
python test-data/chat.py            # chat 完整链路：登录→建会话→提问→读 SSE 事件
```

---

## 九、目录结构

```
good-question/
├── docker-compose.yml        # 7 服务编排（mysql/etcd/minio/milvus/attu/backend/nginx）
├── .env / .env.example       # 环境配置（.env.example 每项含注释）
├── init.sql                  # 建库脚本（表结构由 Alembic 迁移管理）
├── prd.txt                   # PRD 需求（原始需求）
├── solution.md               # 技术方案（v2 归档存根，检索实现以本 README 为准）
├── verify_contract.py        # SSE 契约运行时验证（宿主机直接运行）
├── backend/                  # FastAPI + LangChain 后端
│   ├── main.py               # 入口 + lifespan（admin 种子 / 模型预热 / 路由注册）
│   ├── config.py             # pydantic-settings 读取 .env
│   ├── api/                  # 路由层（薄）：auth/dashboard/library/document/session/chat/contracts
│   │   └── contracts.py      # GET /api/contracts 契约清单（评测平台自动发现）
│   ├── services/             # 业务逻辑：document(管线) / retrieval(混合检索) / chat(问答/记忆/流式)
│   │                         #          dashboard / library / auth / embedding / vector_store / llm
│   ├── scripts/              # migrate_to_milvus.py：从 MySQL chunks 幂等重灌 Milvus
│   ├── models/               # SQLAlchemy ORM（6 张表）
│   ├── schemas/              # Pydantic 请求/响应
│   ├── middleware/           # JWT 鉴权 Depends
│   ├── utils/                # mineru_extractor / mineru_api / text_cleaner / chunker / security / exceptions
│   ├── alembic/              # 数据库迁移（versions/：0001~0003）
│   ├── tests/                # 单元测试（45 个测试函数，pytest 已内置镜像）
│   ├── requirements.txt      # 生产依赖
│   └── requirements-dev.txt  # 测试依赖（pytest，已装进镜像）
├── frontend/                 # Vue 3 + Naive UI 前端
│   ├── Dockerfile            # 多阶段：node 编译 → nginx 服务
│   ├── nginx.conf            # gzip + /api 反代 + SPA fallback + SSE 关缓冲
│   └── src/
│       ├── api/              # Axios 封装（auth/library/document/session/chat）
│       ├── stores/           # Pinia（auth/chat）
│       ├── views/            # 7 个页面（登录/注册/仪表盘/文档库/文档/chunk详情/聊天）
│       ├── components/       # 布局 + 溯源卡片 + 会话列表等
│       └── utils/sse.ts      # SSE 流式接收（解析 event/data 块）
├── test-data/                # seed_demo.py（一键演示数据）+ demo/（4 份演示文档）
│                             # + 迁移验证脚本（e2e / chat / verify_old_data）+ 验证文档
└── data/                     # 上传文件存储（uploads/）
```

**后端代码分层**：`api/`（路由，薄）→ `services/`（业务逻辑）→ `models/` + `utils/`（数据与基础设施）。新增功能一般改 `api/` + `services/` 即可。

---

## 十、测试与验收

| 层 | 内容 | 说明 |
|----|------|------|
| 单元测试 | `backend/tests/` 45 个测试函数 | 安全 / 清洗 / 切片 / 文档服务 / embedding / 检索 / 聊天服务（含两级置信档边界），纯函数级，不依赖外部服务 |
| 契约验证 | `verify_contract.py` | 运行时读真实 SSE 事件流，断言事件序（meta 首、usage 在 done 前）与字段完整（meta/tool_call/usage/reasoning/token/ts） |
| 迁移验证 | `test-data/e2e.py` / `chat.py` / `verify_old_data.py` | 登录→建库→上传→就绪 / chat 完整链路读 SSE / 旧数据重灌后检索验证 |
| 前端构建 | `npm run build` | 改了前端代码必须先 build，否则 nginx 里是旧页面 |

**pytest 已内置 Docker 镜像**（无需临时安装）：

```bash
docker exec -it rag-backend bash
cd /app && pytest tests/ -v          # 容器内开箱即用
```

**宿主机跑单元测试**：

```bash
cd backend && pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -v
```

**契约验证 / 迁移验证**（需服务已启动，对 `http://localhost`）：

```bash
python verify_contract.py           # 契约事件流验证
python test-data/e2e.py             # 端到端：登录→建库→上传→轮询就绪
python test-data/chat.py            # chat 完整链路：登录→建会话→提问→读 SSE
python test-data/verify_old_data.py # 旧数据迁移后检索链路验证
```

---

## 十一、开发指南

### 环境

```bash
docker compose up -d                # 起中间件（mysql/milvus 等）
cd backend && pip install -r requirements.txt -r requirements-dev.txt
uvicorn main:app --reload --port 8080   # 宿主机热重载 API（不依赖 Docker 内 backend）
```

### 改后端代码

- 后端源码 **COPY 进镜像**（`docker-compose.yml` 未挂载源码卷），改代码必须重建：

```bash
docker compose up -d --build backend   # 重新构建并重启 backend
```

### 改前端代码

```bash
cd frontend && npm run build
cd .. && docker compose build nginx && docker compose up -d nginx
```

### 跑测试 / 验收

```bash
docker exec -it rag-backend bash && cd /app && pytest tests/ -v   # 容器内单测
python verify_contract.py                                          # 宿主机契约验证
```

### 新增 API

- models → schemas → services → `api/` 路由 → 注册到 `main.py` → 补单元测试；
- **契约优先**：对外接口先出方案，确认后再实现（本项目架构约束）；LLM 相关接口如需评测平台发现，同步登记进 `api/contracts.py` 的 MANIFEST。

### 数据库变更

- 改模型后 `alembic revision --autogenerate -m "描述"` → `alembic upgrade head`；
- 容器启动时 `CMD` 会自动执行 `alembic upgrade head`，无需手动迁移。

### 编码规范（约定）

- 4 空格缩进、阿里 Java 规范思想；注释写"为什么"；中文注释、英文标识符；
- 新增接口/消费者打印入参出参（debug 级）；核心逻辑（Service 业务分支）必须单测覆盖；
- **pre-commit 校验**：提交新特性缺核心单测会被 test-coverage hook 拦截，写代码就配套测试。

---

## 十二、常见问题

| 现象 | 处理 |
|------|------|
| 首次提问很慢 | 模型首次加载（embedding/rerank）需几秒；已在 lifespan 中预热，重启后首次提问前加载 |
| `DEEPSEEK_API_KEY` 无效 | 检查 `.env` 的 `DEEPSEEK_API_KEY`；`verify_contract.py` 会直接暴露 LLM 调用失败 |
| 前端 80 端口白屏 | `frontend/` 未构建 dist：`cd frontend && npm run build && docker compose build nginx && docker compose up -d nginx` |
| 改了后端代码不生效 | 后端源码在镜像内，需 `docker compose up -d --build backend`（仅 `--force-recreate` 不会加载新代码） |
| Milvus 不可用 | `docker compose up -d` 后等 milvus healthy；可开 `http://localhost:8000`（Attu）看 collection/数据 |
| 端口冲突 80/8080/3306/19530 | 改 `.env` 对应端口（compose 会引用），或停掉占用进程 |
| 上传的 `.doc` 不支持 | MinerU 不支持老版 `.doc`，先另存为 `.docx` 再上传 |
| 本地解析太慢 | 大 PDF 可配置 `MINERU_API_TOKEN` 走官方 API（代价是文档内容上传云端） |
| 检索结果异常 | 重跑 `python test-data/verify_old_data.py` 验证写入/检索链路；数据污染可重跑 `migrate_to_milvus.py` 幂等重灌 |
| 答案像"编造" | 两级置信档已兜底低相关；若仍出现，检查该文档是否已"就绪"、问题是否在库内容之外 |

---

## 十三、已知限制与优化方向

**如实说明当前已知的性能与边界问题：**

1. **Rerank 是检索延迟的绝对瓶颈**：Milvus 混合检索（dense + BM25，RRF 融合）约 0.1s 可忽略，但 Rerank 精排在 CPU 上对候选文本对打分是主要耗时（每段数秒，chunk 越大越慢），当前 `bge-reranker-v2-m3` 比旧 v1-base 更准但也更重。候选数由 Milvus `hybrid_search` 的 `limit` 控制（默认 6）。
   - 可选方向：换更小的 rerank 模型、GPU 推理、或更激进的候选控制（均已评估，各有取舍，尚未落地）。
2. **模型全 CPU 推理**：embedding 与 rerank 均为 CPU，重负载场景可考虑 GPU 容器。
3. **无 LLM 缓存**：相同的问与答会重复调用 DeepSeek（计费）。高频率场景可加缓存层。
4. **Milvus 中文分词弱于原 ES IK**：内置 `chinese` analyzer 对领域词/自定义词典支持有限，专业术语精确命中可能略降（Milvus 自定义词典规划在 v3.0）。

---

## 文档索引

- **PRD 需求**：[prd.txt](prd.txt)（原始功能需求）
- **技术方案**：[solution.md](solution.md)（v2 归档存根；检索存储已演进为 Milvus，实现细节以本 README 四/五章为准）
- **SSE 契约验证**：[verify_contract.py](verify_contract.py)（契约事件流运行时断言）
- **一键演示数据**：[test-data/seed_demo.py](test-data/seed_demo.py)（重建"演示知识库"并打印 6 个场景问题清单）
- **Milvus 迁移配套**：[backend/scripts/migrate_to_milvus.py](backend/scripts/migrate_to_milvus.py)、[test-data/](test-data/)（e2e / chat / verify_old_data + 升级验证文档）
