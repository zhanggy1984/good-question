-- Native RAG 数据库初始化脚本
-- 注意：本文件只负责建库。表结构统一由 Alembic 迁移管理（backend 启动时执行 alembic upgrade head）。
-- MYSQL_DATABASE 环境变量已保证 native_rag 库存在，此处 CREATE DATABASE 为幂等保险。

CREATE DATABASE IF NOT EXISTS native_rag
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE native_rag;
