# 知识库文件放这里

- 问答对：`qa.jsonl` / `*.json` / `*.csv`（含 question/answer 字段）→ 命中直接作答，不调 LLM
- 资料：`*.md` / `*.txt` → 作为 RAG 上下文交给 LLM

加完文件后执行：

```bash
curl -X POST "http://127.0.0.1:5000/api/kb/rebuild?token=<ACCESS_TOKEN>"
```

细节见 `docs/knowledge-base.md`。
