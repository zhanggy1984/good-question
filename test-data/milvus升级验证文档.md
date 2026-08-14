# 环境搭建指南

## 安装步骤

1. 下载安装包，双击运行安装程序。
2. 安装完成后，打开终端，输入 `pip install docker-compose` 验证环境。
3. 配置文件位于 `/etc/app/config.yaml`，默认端口是 8080。

## 常见问题

- 如果启动失败，检查防火墙是否放行 8080 端口。
- 数据库连接串格式：`mysql://root:password@localhost:3306/native_rag`。
- 查看日志文件：`/var/log/app/app.log`。

## 性能调优

- 将 `MAX_WORKERS` 环境变量设为 8，可提升并发处理能力。
- Redis 缓存建议设置 5 分钟过期时间，减少数据库压力。
