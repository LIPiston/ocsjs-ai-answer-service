# AI题库服务API文档

## 概述

AI题库服务是一个基于OpenAI API的问题解答服务，专为[OCS (Online Course Script)](https://github.com/ocsjs/ocsjs)设计，可以通过AI自动回答题目。此服务实现了与OCS AnswererWrapper兼容的API接口，方便用户将AI能力整合到OCS题库搜索中。

## 接口详情

### 搜索接口

**URL**: `/api/search`

**方法**: `GET` 或 `POST`

**参数**:

| 参数名   | 类型   | 必填 | 说明                                                     |
|---------|--------|------|----------------------------------------------------------|
| title   | string | 是   | 题目内容                                                 |
| type    | string | 否   | 题目类型 (single-单选, multiple-多选, judgement-判断, completion-填空) |
| options | string | 否   | 选项内容，通常是A、B、C、D选项的文本                       |

**成功响应**:

```json
{
  "code": 1,
  "question": "问题内容",
  "answer": "AI生成的答案"
}
```

**失败响应**:

```json
{
  "code": 0,
  "msg": "错误信息"
}
```

### 知识库状态接口
**URL**: `/api/kb/status`
**方法**: `GET`

```json
{
  "enabled": true,
  "ready": true,
  "kb_dir": "/app/knowledge",
  "index_path": "/app/knowledge/.index/kb.npz",
  "chunks": 1280,
  "model": "BAAI/bge-m3",
  "built_at": 1794700000.0,
  "top_k": 5,
  "min_score": 0.35,
  "answer_threshold": 0.72,
  "error": ""
}
```

### 重建知识库接口
**URL**: `/api/kb/rebuild`
**方法**: `POST`（需 token）

重新扫描 `KB_DIR`，切块并调用 bge-m3 重新向量化，写回索引文件。不重启容器即可生效。

```json
{"success": true, "chunks": 1280, "dim": 1024, "model": "BAAI/bge-m3", "elapsed_seconds": 42.3}
```

### 知识库检索接口
**URL**: `/api/kb/search`
**方法**: `GET` 或 `POST`（需 token）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| q      | string | 是 | 查询内容（也可用 title） |
| top_k  | int | 否 | 返回条数，默认取配置 |

```json
{
  "success": true,
  "query": "中国的首都是哪个城市？",
  "mode": "kb",
  "hits": [
    {"score": 0.91, "cosine": 0.88, "kind": "qa", "source": "qa.jsonl",
     "heading": "", "text": "问：...
答：北京", "answer": "北京", "question": "..."}
  ]
}
```

`hits[].score` 是余弦分加字面重合加权后的综合分，用来对照 `KB_MIN_SCORE` / `KB_ANSWER_THRESHOLD` 调参。

### 答题响应的 kb 字段

`/api/search` 在启用知识库时，成功响应会多一个可选的 `kb` 字段，原有 `code` / `question` / `answer` 不变：

```json
{"code": 1, "question": "...", "answer": "北京",
 "kb": {"mode": "kb", "source": "qa.jsonl", "score": 0.91}}
```

`mode` 取值：

- `kb`：命中问答型条目，答案直接来自知识库，没有调用大模型
- `rag`：检索到相关资料，交给大模型作答（`sources` 为前 3 个来源文件）
- `none` / 无该字段：纯大模型直答

### 健康检查接口

**URL**: `/api/health`

**方法**: `GET`

**响应**:

```json
{
  "status": "ok",
  "message": "AI题库服务运行正常",
  "version": "1.0.0",
  "cache_enabled": true,
  "model": "gpt-3.5-turbo"
}
```

### 缓存清理接口

**URL**: `/api/cache/clear`

**方法**: `POST`

**响应**:

```json
{
  "success": true,
  "message": "缓存已清除"
}
```

### 统计信息接口

**URL**: `/api/stats`

**方法**: `GET`

**响应**:

```json
{
  "version": "1.0.0",
  "uptime": 1621234567.89,
  "model": "gpt-3.5-turbo",
  "cache_enabled": true,
  "cache_size": 123
}
```

## OCS配置示例

在OCS的自定义题库配置中添加如下配置：

```json
[
  {
    "name": "AI智能题库",
    "homepage": "https://github.com/yourusername/ai-answer-service",
    "url": "http://localhost:5000/api/search",
    "method": "post",
    "type": "GM_xmlhttpRequest",
    "contentType": "json",
    "data": {
      "title": "${title}",
      "type": "${type}",
      "options": "${options}"
    },
    "handler": "return (res)=> res.code === 1 && res.answer ? [res.question, res.answer] : undefined"
  }
]
```

> ⚠️ `type` 必须是 `GM_xmlhttpRequest`。部分网课站点（如智慧树考试页）的 CSP `connect-src` 不含 `localhost`，页面 `fetch` 会被浏览器直接拦截并报 `TypeError: Failed to fetch`；`GM_xmlhttpRequest` 由脚本管理器发请求，不受页面 CSP 限制。使用前需在脚本 `@connect` 中允许 `localhost`（或安装官方全域名通用版本）。

## 安全设置

如果你想增加安全性，可以在`.env`文件中设置访问令牌：

```
ACCESS_TOKEN=your_secret_token_here
```

设置后，所有API请求都需要包含此令牌，可以通过以下两种方式之一传递：

1. HTTP头部: `X-Access-Token: your_secret_token_here`
2. URL参数: `?token=your_secret_token_here`

## 注意事项

1. **多选题答案格式**: 对于多选题，OCS期望的答案格式是用`#`分隔的选项，例如`A#B#C`。本服务会自动处理这个格式，将OpenAI返回的多选答案转换为此格式。

2. **API请求限制**: 注意OpenAI API有使用限制和费用。确保你的账户有足够的额度来处理预期的请求量。

3. **网络连接**: 确保部署此服务的服务器能够访问OpenAI API（api.openai.com）。某些地区可能需要代理服务。

4. **题库域名**: 根据OCS文档说明，需要将题库配置中`homepage`以及`url`所涉及到的域名，在脚本头部元信息`@connect`中新增，否则无法请求到数据。