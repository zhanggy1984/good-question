# Docker 环境安装部署手册

## 一、Docker 简介

Docker 是一个开源的应用容器引擎，让开发者可以把应用及其依赖打包进一个轻量级、可移植的容器中，然后在任何支持 Docker 的环境里运行。容器之间相互隔离，解决了"在我机器上能跑"的环境一致性问题。

Docker 的核心概念包括镜像（Image）、容器（Container）和仓库（Registry）：

- 镜像：只读的模板，定义了运行一个应用所需的全部环境与代码。
- 容器：镜像的运行实例，可以启动、停止、删除。
- 仓库：集中存放镜像的地方，如 Docker Hub 或私有仓库。

## 二、安装 Docker

### Ubuntu 系统安装

在 Ubuntu 上安装 Docker 的推荐步骤如下：

1. 更新软件源：`sudo apt-get update`。
2. 安装依赖包：`sudo apt-get install -y apt-transport-https ca-certificates curl`。
3. 添加 Docker 官方 GPG 密钥并配置软件源。
4. 安装 Docker 引擎：`sudo apt-get install -y docker-ce`。
5. 启动并设置开机自启：`sudo systemctl enable --now docker`。
6. 验证安装：`docker --version`，并运行 `sudo docker run hello-world` 确认工作正常。

### CentOS 系统安装

CentOS 7 及以上版本安装 Docker：

1. 卸载旧版本（如有）：`sudo yum remove docker docker-client docker-common`。
2. 安装依赖：`sudo yum install -y yum-utils device-mapper-persistent-data lvm2`。
3. 配置 Docker 软件源：`sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo`。
4. 安装 Docker 引擎：`sudo yum install -y docker-ce docker-ce-cli containerd.io`。
5. 启动服务：`sudo systemctl start docker && sudo systemctl enable docker`。

### Windows 系统安装

Windows 需安装 Docker Desktop：

1. 从官网下载 Docker Desktop 安装包。
2. 双击安装，勾选使用 WSL 2 作为后端（推荐）。
3. 安装完成后启动 Docker Desktop，等待引擎状态变为 Running。
4. 在 PowerShell 中运行 `docker --version` 验证。

## 三、常用命令

以下命令在日常使用中频率最高：

- `docker pull 镜像名`：从仓库拉取镜像。
- `docker images`：查看本地已有镜像。
- `docker run 镜像名`：启动一个容器；常用参数 `-d`（后台运行）、`-p 宿主机端口:容器端口`（端口映射）、`--name 容器名`（命名）、`-v 宿主机目录:容器目录`（挂载卷）。
- `docker ps`：查看运行中的容器，`-a` 查看包括已停止的所有容器。
- `docker stop 容器ID` / `docker start 容器ID`：停止 / 启动容器。
- `docker rm 容器ID`：删除容器。
- `docker rmi 镜像ID`：删除镜像。
- `docker build -t 镜像名:标签 .`：基于当前目录的 Dockerfile 构建镜像。
- `docker push 镜像名`：把镜像推送到远程仓库。
- `docker logs 容器ID`：查看容器日志。

## 四、镜像仓库配置

- 国内可直接使用阿里云镜像加速器，在 Docker 配置中填写加速器地址可大幅提升拉取速度。
- 企业内部可搭建私有仓库（如 Harbor 或自建 Registry），推送与拉取通过 `docker push/pull 仓库地址/镜像名` 完成。
- 使用私有仓库前，需在 `/etc/docker/daemon.json` 中配置 `insecure-registries`（非 HTTPS 仓库）或登录认证。

## 五、常见问题

- **权限不足**：当前用户不在 docker 组，报 `permission denied`。把用户加入 docker 组：`sudo usermod -aG docker $USER`，重新登录生效。
- **容器无法访问外网**：检查宿主机的 DNS 配置，必要时在 `daemon.json` 中设置 `dns` 参数。
- **端口被占用**：启动报 `port is already allocated`，改用其他宿主机端口或停掉占用容器。
- **镜像拉取慢**：配置国内镜像加速器，或改用公司内网代理。
