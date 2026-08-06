FROM docker.elastic.co/elasticsearch/elasticsearch:8.13.0

# 安装 IK 中文分词插件
# 官方下载源 get.infini.cloud（GitHub release 下载在国内不稳定）
# 注意：版本号必须与 ES 主版本严格匹配
RUN elasticsearch-plugin install --batch \
    https://get.infini.cloud/elasticsearch/analysis-ik/8.13.0
