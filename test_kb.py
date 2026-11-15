# -*- coding: utf-8 -*-
"""
kb.py 离线自测：用确定性假嵌入器验证切块 / 建库 / 检索 / 命中逻辑。
不访问网络，不需要真实 bge-m3。
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kb import KnowledgeBase, build_chunks, tokenize  # noqa: E402


class FakeEmbedder:
    """把文本的 2-gram 哈希到一个 256 维向量，语义相近的文本向量相近。"""

    model = 'fake-bge-m3'
    DIM = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode('utf-8')).hexdigest()[:8], 16)
                out[i, h % self.DIM] += 1.0
        return out


def main():
    workdir = tempfile.mkdtemp(prefix='ocs-kb-test-')
    kb_dir = os.path.join(workdir, 'knowledge')
    os.makedirs(os.path.join(kb_dir, 'sub'), exist_ok=True)

    # 1. 普通 markdown（带标题、长段落）
    with open(os.path.join(kb_dir, '马原.md'), 'w', encoding='utf-8') as f:
        f.write('# 马克思主义基本原理\n\n## 物质与意识\n\n'
                '物质决定意识，意识对物质具有能动的反作用。这是辩证唯物主义的基本观点。\n\n'
                '## 实践与认识\n\n实践是认识的基础，认识对实践具有反作用。' + '补充说明。' * 200)

    # 2. jsonl 问答对
    with open(os.path.join(kb_dir, 'sub', 'qa.jsonl'), 'w', encoding='utf-8') as f:
        for item in [
            {'question': '中国的首都是哪个城市？', 'answer': '北京', 'type': 'single',
             'options': 'A. 上海\nB. 北京\nC. 广州'},
            {'question': '地球是太阳系中第三颗行星。', 'answer': '正确', 'type': 'judgement'},
            {'question': '以下哪些是中国的一线城市？', 'answer': 'A#B#C#D', 'type': 'multiple',
             'options': 'A. 北京\nB. 上海\nC. 广州\nD. 深圳'},
        ]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 3. csv 问答对
    with open(os.path.join(kb_dir, 'sub', 'qa.csv'), 'w', encoding='utf-8') as f:
        f.write('题目,答案\n《红楼梦》的作者是谁？,曹雪芹\n水的化学式是什么？,H2O\n')

    # 4. 不支持的后缀应被忽略
    with open(os.path.join(kb_dir, 'ignore.pdf'), 'w', encoding='utf-8') as f:
        f.write('不应被读取')

    chunks = build_chunks(kb_dir)
    kinds = {}
    for c in chunks:
        kinds[c['kind']] = kinds.get(c['kind'], 0) + 1
    print(f"[1] 切块: 共 {len(chunks)} 块, 分布 {kinds}")
    assert kinds.get('qa', 0) == 5, kinds
    assert kinds.get('passage', 0) >= 2, kinds
    assert not any(c['source'].endswith('.pdf') for c in chunks)

    index_path = os.path.join(kb_dir, '.index', 'kb.npz')
    kb = KnowledgeBase(kb_dir=kb_dir, index_path=index_path, embedder=FakeEmbedder(),
                       top_k=3, min_score=0.05, answer_threshold=0.5)

    print("[2] 索引不存在时 status:", kb.status()['ready'], kb.status()['error'])
    assert kb.status()['ready'] is False

    result = kb.rebuild()
    print("[3] 建库:", result)
    assert result['success'] and result['chunks'] == len(chunks)
    assert os.path.exists(index_path), '索引文件未生成'

    # 模拟新进程：重新实例化，从磁盘加载
    kb2 = KnowledgeBase(kb_dir=kb_dir, index_path=index_path, embedder=FakeEmbedder(),
                        top_k=3, min_score=0.05, answer_threshold=0.5)
    status = kb2.status()
    print("[4] 重新加载:", {k: status[k] for k in ('ready', 'chunks', 'model')})
    assert status['ready'] and status['chunks'] == len(chunks)
    assert status['model'] == 'fake-bge-m3'

    # 问答型命中：应直接给答案
    hit = kb2.lookup('中国的首都是哪个城市？')
    print("[5] 问答命中:", hit['mode'], repr(hit['answer']), hit['hits'][0]['score'])
    assert hit['mode'] == 'kb' and hit['answer'] == '北京', hit
    assert hit['hits'][0]['source'].endswith('qa.jsonl')

    # 判断题
    hit2 = kb2.lookup('地球是太阳系中第三颗行星。', 'judgement')
    print("[6] 判断题命中:", hit2['mode'], repr(hit2['answer']))
    assert hit2['mode'] == 'kb' and hit2['answer'] == '正确'

    # csv 来源
    hit3 = kb2.lookup('《红楼梦》的作者是谁？')
    print("[7] csv 命中:", hit3['mode'], repr(hit3['answer']), hit3['hits'][0]['source'])
    assert hit3['answer'] == '曹雪芹'

    # 普通文本 → 走 rag 上下文
    hit4 = kb2.lookup('物质和意识的关系是什么')
    print("[8] 段落检索:", hit4['mode'], '上下文长度', len(hit4['context']))
    assert hit4['mode'] in ('rag', 'kb')
    if hit4['mode'] == 'rag':
        assert hit4['answer'] is None and hit4['context']

    # 完全无关 → none（阈值过滤）
    kb3 = KnowledgeBase(kb_dir=kb_dir, index_path=index_path, embedder=FakeEmbedder(),
                        top_k=3, min_score=0.99, answer_threshold=0.9)
    miss = kb3.lookup('量子色动力学的渐近自由')
    print("[9] 高阈值下的无关问题:", miss['mode'], len(miss['hits']))
    assert miss['mode'] == 'none'

    # 索引损坏不应崩溃
    with open(index_path, 'wb') as f:
        f.write(b'not a npz')
    kb4 = KnowledgeBase(kb_dir=kb_dir, index_path=index_path, embedder=FakeEmbedder())
    st = kb4.status()
    print("[10] 损坏索引:", st['ready'], st['error'][:40])
    assert st['ready'] is False and st['error']
    assert kb4.lookup('中国的首都')['mode'] == 'none'

    # 空目录重建
    empty = tempfile.mkdtemp(prefix='ocs-kb-empty-')
    kb5 = KnowledgeBase(kb_dir=empty, index_path=os.path.join(empty, '.index', 'kb.npz'),
                        embedder=FakeEmbedder())
    r = kb5.rebuild()
    print("[11] 空目录重建:", r['success'], r['message'][:40])
    assert r['success'] is False

    shutil.rmtree(workdir, ignore_errors=True)
    shutil.rmtree(empty, ignore_errors=True)
    print("\n全部断言通过 ✅")


if __name__ == '__main__':
    main()
