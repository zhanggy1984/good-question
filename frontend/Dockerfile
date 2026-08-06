# ── 说明 ──
# 前端产物（dist/）在宿主机构建（npm run build），避免 Docker 内 npm ci 网络不稳定。
# 前端代码变更流程：npm run build → docker compose build nginx → up -d nginx

FROM nginx:latest
COPY dist/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
