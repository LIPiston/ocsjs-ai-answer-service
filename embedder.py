# -*- coding: utf-8 -*-
"""
嵌入客户端

走 OpenAI 兼容的 /v1/embeddings（bge-m3 等），批量、带重试。
刻意只依赖 openai 包：服务里已经用它调对话模型。
"""
import logging
import time
from typing import List, Optional, Sequence

import numpy as np
import openai

logger = logging.getLogger('ai_answer_service.embedder')


class Embedder:
    def __init__(self, api_key: str, base_url: str, model: str = 'BAAI/bge-m3',
                 batch_size: int = 32, timeout: float = 60.0, max_retries: int = 2,
                 dimensions: Optional[int] = None):
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self.dimensions = dimensions
        kwargs = {'api_key': api_key, 'base_url': base_url, 'timeout': timeout}
        if max_retries:
            kwargs['max_retries'] = max_retries
        self._client = openai.OpenAI(**kwargs)

    def _one_batch(self, batch: Sequence[str]) -> List[List[float]]:
        kwargs = {'model': self.model, 'input': list(batch)}
        if self.dimensions:
            kwargs['dimensions'] = self.dimensions
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.embeddings.create(**kwargs)
                # 某些网关不保证顺序，按 index 排回去
                items = sorted(resp.data, key=lambda d: getattr(d, 'index', 0))
                return [item.embedding for item in items]
            except Exception as exc:  # noqa: BLE001 - 网关错误类型不统一
                last_error = exc
                if attempt < self.max_retries:
                    wait = 1.5 ** attempt
                    logger.warning(f"嵌入请求失败（第 {attempt + 1} 次），{wait:.1f}s 后重试: {exc}")
                    time.sleep(wait)
        raise RuntimeError(f'嵌入请求持续失败: {last_error}')

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            out.extend(self._one_batch(batch))
        matrix = np.asarray(out, dtype=np.float32)
        if matrix.ndim != 2:
            raise RuntimeError(f'嵌入返回维度异常: {matrix.shape}')
        return matrix

    def ping(self) -> bool:
        try:
            self.embed(['ping'])
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"嵌入服务不可用: {exc}")
            return False
