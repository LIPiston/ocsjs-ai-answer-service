# -*- coding: utf-8 -*-
"""
配置文件
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

# 基础配置
class Config:
    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    # OpenAI API Base URL，如果不设置则使用默认值
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 安全配置（可选）
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", None)
    
    # 响应配置
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    
    # 缓存配置
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
    CACHE_EXPIRATION = int(os.getenv("CACHE_EXPIRATION", 86400))  # 默认缓存24小时
    # ---------- 知识库（可选题库外挂） ----------
    # 总开关：关闭时行为和原版完全一致（纯 LLM 直答）
    KB_ENABLED = os.getenv("KB_ENABLED", "True").lower() == "true"
    # 知识库目录，容器内挂载点；支持 .md/.markdown/.txt/.jsonl/.json/.csv
    KB_DIR = os.getenv("KB_DIR", "/app/knowledge")
    # 向量索引文件（由 /api/kb/rebuild 生成）
    KB_INDEX_PATH = os.getenv("KB_INDEX_PATH", "/app/knowledge/.index/kb.npz")
    # 向量模型：默认 bge-m3（走 OpenAI 兼容 /v1/embeddings）
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    # 嵌入接口地址与密钥；未单独配置时复用对话模型那套（同一个网关）
    EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE") or OPENAI_API_BASE
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or OPENAI_API_KEY
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
    EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", 60))
    # 有些网关支持降维（bge-m3 原生 1024），留空表示不传
    EMBEDDING_DIMENSIONS = os.getenv("EMBEDDING_DIMENSIONS")
    # 检索参数
    KB_TOP_K = int(os.getenv("KB_TOP_K", 5))
    KB_MIN_SCORE = float(os.getenv("KB_MIN_SCORE", 0.35))
    # 命中问答型条目且分数超过此值，直接返回库内答案，不调 LLM
    KB_ANSWER_THRESHOLD = float(os.getenv("KB_ANSWER_THRESHOLD", 0.72))
    # 关键词（2-gram 重叠）加权系数，补一点字面匹配
    KB_KEYWORD_WEIGHT = float(os.getenv("KB_KEYWORD_WEIGHT", 0.15))
    # 单个文本块目标长度（字符）
    KB_CHUNK_TARGET = int(os.getenv("KB_CHUNK_TARGET", 600))
    # 启动时自动建索引（索引不存在或 KB_INDEX_AUTO 目录更新时才建）
    KB_INDEX_AUTO = os.getenv("KB_INDEX_AUTO", "True").lower() == "true"
    # 走知识库时允许追加给 LLM 的上下文最大字符数
    KB_CONTEXT_MAX_CHARS = int(os.getenv("KB_CONTEXT_MAX_CHARS", 4000))
