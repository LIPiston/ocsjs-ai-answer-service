# 知识库（bge-m3 检索增强答题）

给题库加的一层「先查资料、再作答」能力。默认开启，不改任何原有接口的请求/响应格式（只在答案里多一个可选的 `kb` 字段）。

## 两种答题模式

| 模式 | 触发条件 | 行为 |
|------|----------|------|
| `kb` | 命中**问答型**条目，且综合分 ≥ `KB_ANSWER_THRESHOLD`（默认 0.72） | 直接返回库内答案，**不调用 LLM**，快且稳定 |
| `rag` | 只命中普通文本（md/txt 等资料），或问答条目分数不够 | 把检索到的片段拼进 system prompt，让 LLM 依据资料作答 |
| `none` | 没有超过 `KB_MIN_SCORE`（默认 0.35）的片段 | 和原版一样纯 LLM 直答 |

嵌入服务挂掉时整个知识库会自动跳过，退化为纯 LLM，接口仍返回 200。

## 知识库文件放哪

宿主机 `./knowledge` 目录（容器内 `/app/knowledge`）。支持的格式：

| 格式 | 说明 |
|------|------|
| `.jsonl` / `.json` | **问答对**，每个对象含 `question`/`answer`，可选 `options`、`type`。这是命中率最高、最省 token 的格式 |
| `.csv` | 同上，表头用 `题目,答案` 或 `question,answer` |
| `.md` / `.markdown` / `.txt` | 按标题和空行切块，作为 `rag` 资料 |

问答对示例（`knowledge/qa.jsonl`）：

```json
{"question":"中国的首都是哪个城市？","answer":"北京","type":"single","options":"A. 上海\nB. 北京"}
{"question":"地球是太阳系中第三颗行星。","answer":"正确","type":"judgement"}
{"question":"以下哪些是中国的一线城市？","answer":"A#B#C#D","type":"multiple"}
```

多选题答案用 `#` 分隔（和 OCS 期望的格式一致）。子目录会被递归读取。

加完文件后 **不用重启容器**，调用一次重建接口即可：

```bash
curl -X POST "http://127.0.0.1:5000/api/kb/rebuild?token=<ACCESS_TOKEN>"
```

索引写在 `knowledge/.index/kb.npz`（单个文件，加载时整体读入内存并归一化）。

## 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/kb/status` | GET | 知识库状态：块数、模型、建库时间、错误 |
| `/api/kb/rebuild` | POST | 重新扫描并向量化（需 token） |
| `/api/kb/search?q=...` | GET/POST | 只检索不答题，返回命中片段与分数，用于调阈值（需 token） |
| `/api/health` | GET | 多了 `kb_enabled` / `kb_ready` / `kb_chunks` / `embedding_model` |
| `/api/stats` | GET | 多了 `kb` 段 |

## 环境变量

```ini
KB_ENABLED=True                 # 总开关，False 则完全等于原版行为
KB_DIR=/app/knowledge
KB_INDEX_PATH=/app/knowledge/.index/kb.npz
KB_INDEX_AUTO=True              # 启动时若索引缺失，后台自动建库（不阻塞首个请求）

EMBEDDING_MODEL=BAAI/bge-m3     # 走 OpenAI 兼容 /v1/embeddings
EMBEDDING_API_BASE=https://<你的网关地址>/v1
EMBEDDING_API_KEY=...           # 留空则复用 OPENAI_API_KEY
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DIMENSIONS=           # 留空不传；有些网关支持降维

KB_TOP_K=5                      # 每次取前几块
KB_MIN_SCORE=0.35               # 低于此余弦分直接丢弃
KB_ANSWER_THRESHOLD=0.72        # 问答条目高于此分直接作答，不调 LLM
KB_KEYWORD_WEIGHT=0.15          # 2-gram 字面重合加权，补语义检索的短板
KB_CHUNK_TARGET=600             # 普通文本切块目标字符数
KB_CONTEXT_MAX_CHARS=4000       # 塞给 LLM 的资料上限
```

## 调参建议

- 命中率低（该答的直接走了 LLM）：先看 `/api/kb/search` 的 `top_score`。分数在 0.6 左右说明相似但没过线，把 `KB_ANSWER_THRESHOLD` 降到 0.65。
- 答错了库外的题：`KB_MIN_SCORE` 抬高到 0.45~0.5，减少误召回。
- 知识库很大（几万块以上）：内存占用会线性增长（每块 1024 维 float32 ≈ 4 KB），`KB_TOP_K` 可以保持 5，但要注意容器内存上限。

## 设计取舍

- **不引入向量数据库**：几百到几万块规模用内存矩阵乘法足够，省掉一个常驻服务，适合 1.6G 内存的小机器。
- **索引是文件 + mtime 缓存**：重建后无需重启进程，也不用重启容器。
- **嵌入失败不致命**：知识库只是增强层，不能成为答题的单点故障。
- **问答型条目单独建模**：这类条目本身带答案，命中即可跳过一次 LLM 调用，既省钱又避免模型「想太多」。
