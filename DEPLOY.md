# 部署步骤（服务器 starxn）

假设线上目录为 `/opt/ocsjs-ai-answer-service`（以实际 `docker compose` 所在目录为准，先 `docker inspect` 确认挂载来源）。

## 0. 先备份

```bash
cd /opt/ocsjs-ai-answer-service
cp -a app.py config.py utils.py Dockerfile docker-compose.yml /root/ocs-backup-$(date +%Y%m%d-%H%M)/
docker compose ps
```

## 1. 放新文件

需要新增/覆盖：`kb.py`、`embedder.py`、`app.py`、`config.py`、`requirements.txt`、
`docker-compose.yml`、`Dockerfile`、`templates/dashboard.html`、`docs/knowledge-base.md`、
`knowledge/`（目录本身）。

`.env` 追加（密钥沿用已有的 OPENAI_API_KEY，不写入仓库）：

```ini
KB_ENABLED=True
KB_DIR=/app/knowledge
KB_INDEX_PATH=/app/knowledge/.index/kb.npz
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_BASE=<现有网关地址>
# EMBEDDING_API_KEY 留空则复用 OPENAI_API_KEY
```

## 2. 重建镜像并起容器

```bash
docker compose build --no-cache
docker compose up -d
docker compose logs --tail=50
```

首次启动若 `knowledge/` 为空，日志会出现「知识库目录没有可用文件」，属正常——放文件后再 rebuild。

## 3. 放知识库文件并建索引

```bash
mkdir -p knowledge
# 放入 qa.jsonl / *.md 等
curl -s -X POST "http://127.0.0.1:5000/api/kb/rebuild?token=<ACCESS_TOKEN>"
curl -s "http://127.0.0.1:5000/api/kb/status"
```

## 4. 验证

```bash
# 检索命中的实际分数（调阈值用）
curl -s "http://127.0.0.1:5000/api/kb/search?q=中国的首都是哪个城市&token=<ACCESS_TOKEN>"
# 真实答题
curl -s "http://127.0.0.1:5000/api/search?title=中国的首都是哪个城市&type=single&options=A.上海%0AB.北京&token=<ACCESS_TOKEN>"
# 外网
# 把 <你的域名或IP> 换成你自己的入口
curl -s "https://<你的域名或IP>/api/health"
```

## 5. 回滚

```bash
cd /opt/ocsjs-ai-answer-service
cp -a /root/ocs-backup-*/app.py /root/ocs-backup-*/config.py /root/ocs-backup-*/docker-compose.yml .
docker compose up -d --build
```
或直接把 `.env` 里 `KB_ENABLED=False` 后 `docker compose up -d`（行为等同原版，但需重启容器）。
