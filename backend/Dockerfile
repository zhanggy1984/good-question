FROM python:3.11-slim
WORKDIR /app

# 系统依赖：MinerU 的 opencv(cv2) 需要 X11 库；Debian 13 (trixie) 包名
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglx-mesa0 libglib2.0-0t64 libsm6 libxext6 libxrender1 \
        libxcb1 libxcb-shm0 libxcb-render0 \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 先装 torch CPU 版 + 匹配的 torchvision（MinerU/rerank 依赖）
# 版本严格匹配：torch 2.13 ↔ torchvision 0.28（避免 pip 误解析到远古版本）
RUN pip install --no-cache-dir "torch==2.13.0" "torchvision==0.28.0" \
    --index-url https://download.pytorch.org/whl/cpu \
    --timeout 180 --retries 15

# Python 依赖（清华 pip 源；大包网络波动时用超时+重试兜底）
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 180 --retries 15

COPY . .

EXPOSE 8080
# 启动时先执行数据库迁移（表结构由 Alembic 统一管理），再启动 API
CMD ["sh", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8080"]
