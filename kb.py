# -*- coding: utf-8 -*-
"""
知识库模块

把本地知识库文件切块、用 bge-m3（OpenAI 兼容 /v1/embeddings）向量化，
存成单个 .npz 索引，检索时用余弦相似度 + 关键词加权。

设计取舍：
- 不引入向量数据库：知识库规模是几百到几万 chunk，内存里一次矩阵乘法足够，
  省掉一个常驻服务，适合 1.6G 内存的小机器。
- 索引是磁盘文件 + 进程内缓存，文件 mtime 变了才重新加载，重建索引不必重启容器。
- 嵌入失败不致命：本轮检索退化为纯 LLM 直答，接口照常返回。
"""
import base64
import csv
import io
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger('ai_answer_service.kb')

SUPPORTED_SUFFIXES = ('.md', '.markdown', '.txt', '.jsonl', '.json', '.csv')
QUESTION_KEYS = ('question', 'title', 'q', '题目', '问题', '题干')
ANSWER_KEYS = ('answer', 'a', '答案', '正确答案')
OPTIONS_KEYS = ('options', 'option', '选项')
TYPE_KEYS = ('type', 'question_type', '题型')


def _first_value(record: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for key in keys:
        for real_key in (key, key.lower(), key.upper()):
            if real_key in record:
                value = record[real_key]
                if isinstance(value, (list, tuple)):
                    value = '\n'.join(str(v) for v in value)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    return text
    return None


def _clean(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _split_markdown(text: str, target: int = 600, overlap: int = 100) -> List[Tuple[str, str]]:
    """按标题与空行切块，返回 [(标题路径, 块内容)]"""
    heading = ''
    blocks: List[Tuple[str, str]] = []
    buffer = ''
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            if buffer.strip():
                blocks.append((heading, buffer.strip()))
                buffer = ''
            heading = stripped.lstrip('#').strip()
            continue
        if not stripped:
            if buffer.strip():
                blocks.append((heading, buffer.strip()))
                buffer = ''
            continue
        buffer += line + '\n'
    if buffer.strip():
        blocks.append((heading, buffer.strip()))

    chunks: List[Tuple[str, str]] = []
    for head, body in blocks:
        if len(body) <= target:
            chunks.append((head, body))
            continue
        start = 0
        while start < len(body):
            piece = body[start:start + target]
            chunks.append((head, piece))
            if start + target >= len(body):
                break
            start += target - overlap
    return [(h, _clean(c)) for h, c in chunks if _clean(c)]


def _records_from_jsonl(text: str) -> List[Dict[str, Any]]:
    records = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
        elif isinstance(item, list):
            records.extend(i for i in item if isinstance(i, dict))
    return records


def _chunks_from_records(records: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """问答型记录 → 结构化 chunk（带答案，可命中后直接回答）"""
    chunks: List[Dict[str, Any]] = []
    for record in records:
        question = _first_value(record, QUESTION_KEYS)
        answer = _first_value(record, ANSWER_KEYS)
        if not question or not answer:
            continue
        options = _first_value(record, OPTIONS_KEYS)
        qtype = _first_value(record, TYPE_KEYS)
        text = f"问：{question}"
        if options:
            text += f"\n选项：{options}"
        text += f"\n答：{answer}"
        chunks.append({
            'text': text,
            'answer': answer,
            'question': question,
            'options': options or '',
            'qtype': qtype or '',
            'heading': '',
            'source': source,
            'kind': 'qa',
        })
    return chunks


def _chunks_from_csv(text: str, source: str) -> List[Dict[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(text))
        records = [row for row in reader]
    except Exception:
        return []
    return _chunks_from_records(records, source)


def build_chunks(kb_dir: str) -> List[Dict[str, Any]]:
    """遍历知识库目录，产出统一结构的 chunk 列表"""
    chunks: List[Dict[str, Any]] = []
    if not os.path.isdir(kb_dir):
        return chunks
    for root, dirs, files in os.walk(kb_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for name in sorted(files):
            if name.startswith('.'):
                continue
            suffix = os.path.splitext(name)[1].lower()
            if suffix not in SUPPORTED_SUFFIXES:
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, kb_dir)
            try:
                raw = open(path, 'r', encoding='utf-8', errors='ignore').read()
            except OSError as exc:
                logger.warning(f"读取知识库文件失败 {rel}: {exc}")
                continue
            raw = _clean(raw)
            if not raw:
                continue

            if suffix in ('.jsonl', '.json', '.csv'):
                if suffix == '.csv':
                    structured = _chunks_from_csv(raw, rel)
                else:
                    structured = _chunks_from_records(_records_from_jsonl(raw), rel)
                if structured:
                    chunks.extend(structured)
                    continue
                # 不是问答型记录，当普通文本处理
                chunks.extend({
                    'text': body,
                    'answer': None,
                    'question': None,
                    'options': '',
                    'qtype': '',
                    'heading': heading,
                    'source': rel,
                    'kind': 'passage',
                } for heading, body in _split_markdown(raw))
                continue

            chunks.extend({
                'text': body,
                'answer': None,
                'question': None,
                'options': '',
                'qtype': '',
                'heading': heading,
                'source': rel,
                'kind': 'passage',
            } for heading, body in _split_markdown(raw))
    return chunks


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def tokenize(text: str) -> List[str]:
    """轻量分词：英文/数字按词，中文按 2-gram，够做关键词加权"""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9_]{2,}', text)
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class KnowledgeBase:
    """知识库：索引文件 + 进程内缓存 + 检索"""

    def __init__(self, kb_dir: str, index_path: str, embedder, top_k: int = 5,
                 min_score: float = 0.35, answer_threshold: float = 0.72,
                 keyword_weight: float = 0.15, chunk_target: int = 600):
        self.kb_dir = kb_dir
        self.index_path = index_path
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.answer_threshold = answer_threshold
        self.keyword_weight = keyword_weight
        self.chunk_target = chunk_target

        self._lock = threading.Lock()
        self._vectors: Optional[np.ndarray] = None
        self._chunks: List[Dict[str, Any]] = []
        self._token_sets: List[set] = []
        self._loaded_mtime: float = 0.0
        self._model: str = ''
        self._built_at: float = 0.0
        self._error: str = ''

    # ---------- 索引读写 ----------

    def _load(self) -> bool:
        if not os.path.exists(self.index_path):
            self._error = '索引文件不存在，请先 POST /api/kb/rebuild'
            return False
        mtime = os.path.getmtime(self.index_path)
        if self._vectors is not None and mtime == self._loaded_mtime:
            return True
        try:
            with np.load(self.index_path, allow_pickle=False) as data:
                vectors = np.asarray(data['vectors'], dtype=np.float32)
                # meta 存成 utf-8 字节，避免 numpy 的 unicode/object 数组在某些版本上需要 pickle
                raw_meta = data['meta'].tobytes().decode('utf-8')
                meta = json.loads(raw_meta)
        except Exception as exc:
            self._error = f'索引读取失败: {exc}'
            logger.error(self._error)
            return False
        chunks = meta.get('chunks') or []
        if len(chunks) != vectors.shape[0]:
            self._error = f'索引损坏：chunks={len(chunks)} vectors={vectors.shape[0]}'
            logger.error(self._error)
            return False
        self._vectors = _normalize(vectors)
        self._chunks = chunks
        self._token_sets = [set(tokenize(c.get('text', ''))) for c in chunks]
        self._model = meta.get('model', '')
        self._built_at = float(meta.get('built_at') or 0)
        self._loaded_mtime = mtime
        self._error = ''
        return True

    def _save(self, vectors: np.ndarray, chunks: List[Dict[str, Any]], model: str) -> None:
        os.makedirs(os.path.dirname(self.index_path) or '.', exist_ok=True)
        meta = {
            'version': 1,
            'model': model,
            'built_at': time.time(),
            'count': len(chunks),
            'dim': int(vectors.shape[1]) if vectors.size else 0,
            'chunks': chunks,
        }
        # np.savez_compressed 会给不以 .npz 结尾的文件名补后缀，所以临时文件也必须是 .npz
        tmp = self.index_path + '.tmp.npz'
        meta_bytes = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode('utf-8'), dtype=np.uint8)
        np.savez_compressed(tmp, vectors=vectors.astype(np.float32), meta=meta_bytes)
        os.replace(tmp, self.index_path)

    # ---------- 建库 ----------

    def rebuild(self) -> Dict[str, Any]:
        start = time.time()
        chunks = build_chunks(self.kb_dir)
        if not chunks:
            self._error = f'知识库目录没有可用文件: {self.kb_dir}'
            return {'success': False, 'message': self._error, 'chunks': 0}
        texts = [c['text'] for c in chunks]
        try:
            vectors = self.embedder.embed(texts)
        except Exception as exc:
            self._error = f'嵌入失败: {exc}'
            logger.error(self._error, exc_info=True)
            return {'success': False, 'message': self._error, 'chunks': len(chunks)}
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunks):
            self._error = '嵌入数量与 chunk 数量不一致'
            return {'success': False, 'message': self._error, 'chunks': len(chunks)}
        with self._lock:
            self._save(vectors, chunks, getattr(self.embedder, 'model', ''))
            self._vectors = None
            self._loaded_mtime = 0.0
            self._load()
        elapsed = round(time.time() - start, 2)
        logger.info(f"知识库重建完成：{len(chunks)} 块，耗时 {elapsed}s")
        return {
            'success': True,
            'chunks': len(chunks),
            'dim': int(vectors.shape[1]),
            'model': getattr(self.embedder, 'model', ''),
            'elapsed_seconds': elapsed,
        }

    # ---------- 检索 ----------

    def ready(self) -> bool:
        with self._lock:
            return self._load()

    def status(self) -> Dict[str, Any]:
        ready = self.ready()
        return {
            'enabled': True,
            'ready': ready,
            'kb_dir': self.kb_dir,
            'index_path': self.index_path,
            'chunks': len(self._chunks) if ready else 0,
            'model': self._model,
            'built_at': self._built_at,
            'top_k': self.top_k,
            'min_score': self.min_score,
            'answer_threshold': self.answer_threshold,
            'error': self._error if not ready else '',
        }

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not query.strip() or not self.ready():
            return []
        top_k = top_k or self.top_k
        try:
            query_vec = np.asarray(self.embedder.embed([query]), dtype=np.float32)
        except Exception as exc:
            logger.warning(f"检索嵌入失败，本次跳过知识库: {exc}")
            self._error = f'检索嵌入失败: {exc}'
            return []
        if query_vec.ndim != 2 or query_vec.shape[0] != 1:
            return []
        query_vec = _normalize(query_vec)[0]
        cosine = self._vectors @ query_vec

        tokens = set(tokenize(query))
        hits: List[Dict[str, Any]] = []
        for idx in np.argsort(-cosine)[:max(top_k * 4, top_k)]:
            score = float(cosine[idx])
            if score < self.min_score:
                continue
            overlap = 0.0
            if tokens and self._token_sets[idx]:
                overlap = len(tokens & self._token_sets[idx]) / len(tokens)
            final = score + self.keyword_weight * overlap
            chunk = self._chunks[int(idx)]
            hits.append({
                'score': round(final, 4),
                'cosine': round(score, 4),
                'kind': chunk.get('kind', 'passage'),
                'source': chunk.get('source', ''),
                'heading': chunk.get('heading', ''),
                'text': chunk.get('text', ''),
                'answer': chunk.get('answer'),
                'question': chunk.get('question'),
            })
        hits.sort(key=lambda h: h['score'], reverse=True)
        return hits[:top_k]

    def lookup(self, question: str, question_type: str = '', options: str = '') -> Dict[str, Any]:
        """
        为一道题找知识库依据。

        返回 {'answer': str|None, 'context': str, 'hits': [...], 'mode': 'kb'|'rag'|'none'}
        - 命中问答型条目且分数够高 → 直接给答案（mode=kb）
        - 只命中普通文本 → 作为上下文交给 LLM（mode=rag）
        """
        query = question if not options else f"{question}\n{options}"
        hits = self.search(query)
        if not hits:
            return {'answer': None, 'context': '', 'hits': [], 'mode': 'none'}
        best = hits[0]
        if best.get('answer') and best['score'] >= self.answer_threshold:
            return {'answer': best['answer'], 'context': best['text'], 'hits': hits, 'mode': 'kb'}
        context = '\n\n'.join(
            f"[{h['source']}{(' #' + h['heading']) if h['heading'] else ''}] {h['text']}"
            for h in hits
        )
        return {'answer': None, 'context': context, 'hits': hits, 'mode': 'rag'}
