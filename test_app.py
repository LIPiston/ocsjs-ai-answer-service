# -*- coding: utf-8 -*-
"""
app.py 集成自测：Flask test client + 假嵌入器 + 假 LLM。

覆盖：
1. 知识库命中 → 直接返回答案，不调 LLM
2. 知识库只命中段落 → 把资料拼进 system prompt，调 LLM
3. 知识库无命中 → 纯 LLM 直答（与原版一致）
4. /api/kb/status、/api/kb/search、/api/kb/rebuild 接口
5. 嵌入服务故障时降级为纯 LLM，不 500
"""
import json
import os
import sys
import tempfile

os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['OPENAI_API_BASE'] = 'http://127.0.0.1:1/v1'
os.environ['KB_ENABLED'] = 'True'
os.environ['KB_INDEX_AUTO'] = 'False'
os.environ['ENABLE_CACHE'] = 'False'
os.environ['LOG_LEVEL'] = 'WARNING'
# 假嵌入器不是真语义模型，阈值调低以便测通 RAG 分支（真实 bge-m3 用默认 0.35）
os.environ['KB_MIN_SCORE'] = '0.05'
os.environ['KB_ANSWER_THRESHOLD'] = '0.8'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import kb as kb_module  # noqa: E402
from test_kb import FakeEmbedder  # noqa: E402


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs['messages']
        system = messages[0]['content']

        class Msg:
            content = '上海' if '马克思主义' not in system else '物质决定意识'

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


class FakeClient:
    def __init__(self):
        self.chat = type('Chat', (), {})()
        self.chat.completions = FakeCompletions()


def build_app(tmpdir):
    kb_dir = os.path.join(tmpdir, 'knowledge')
    os.makedirs(kb_dir, exist_ok=True)
    with open(os.path.join(kb_dir, 'qa.jsonl'), 'w', encoding='utf-8') as f:
        f.write(json.dumps({'question': '中国的首都是哪个城市？', 'answer': '北京',
                            'type': 'single', 'options': 'A. 上海\nB. 北京'}, ensure_ascii=False) + '\n')
    with open(os.path.join(kb_dir, '马原.md'), 'w', encoding='utf-8') as f:
        f.write('# 马克思主义\n\n## 物质与意识\n\n物质决定意识，意识对物质具有能动的反作用。\n')

    os.environ['KB_DIR'] = kb_dir
    os.environ['KB_INDEX_PATH'] = os.path.join(kb_dir, '.index', 'kb.npz')

    import importlib
    import config as config_module
    importlib.reload(config_module)
    import app as app_module
    importlib.reload(app_module)

    app_module.kb.embedder = FakeEmbedder()
    fake_client = FakeClient()
    app_module.client = fake_client
    return app_module, fake_client


def main():
    tmpdir = tempfile.mkdtemp(prefix='ocs-kb-app-')
    app_module, fake_client = build_app(tmpdir)
    client = app_module.app.test_client()
    kb = app_module.kb

    # 建库
    r = client.post('/api/kb/rebuild')
    print('[1] rebuild:', r.status_code, r.get_json())
    assert r.status_code == 200 and r.get_json()['success']

    # 健康检查带知识库信息
    r = client.get('/api/health')
    health = r.get_json()
    print('[2] health:', {k: health[k] for k in ('status', 'kb_enabled', 'kb_ready', 'kb_chunks')})
    assert health['kb_enabled'] and health['kb_ready'] and health['kb_chunks'] > 0

    # status 接口
    r = client.get('/api/kb/status')
    st = r.get_json()
    print('[3] status:', {k: st[k] for k in ('ready', 'chunks', 'model')})
    assert st['ready']

    # 检索接口
    r = client.get('/api/kb/search', query_string={'q': '中国的首都是哪个城市？'})
    sr = r.get_json()
    print('[4] search:', sr['mode'], sr['hits'][0]['answer'], sr['hits'][0]['score'])
    assert sr['success'] and sr['mode'] == 'kb' and sr['hits'][0]['answer'] == '北京'

    # 知识库命中 → 不调 LLM
    fake_client.chat.completions.calls.clear()
    r = client.get('/api/search', query_string={
        'title': '中国的首都是哪个城市？', 'type': 'single', 'options': 'A. 上海\nB. 北京'})
    res = r.get_json()
    print('[5] 命中直答:', res)
    assert res['code'] == 1 and res['answer'] == '北京'
    assert res.get('kb', {}).get('mode') == 'kb'
    assert len(fake_client.chat.completions.calls) == 0, '命中时不应调用 LLM'

    # 只命中段落 → 走 RAG，LLM 收到资料
    fake_client.chat.completions.calls.clear()
    r = client.get('/api/search', query_string={'title': '物质与意识的关系是什么', 'type': 'completion'})
    res = r.get_json()
    calls = fake_client.chat.completions.calls
    system_prompt = calls[0]['messages'][0]['content'] if calls else ''
    print('[6] RAG:', res['code'], res.get('kb'), '| prompt 含资料:', '物质决定意识' in system_prompt)
    assert res['code'] == 1 and len(calls) == 1
    assert '知识库检索到的资料' in system_prompt and '物质决定意识' in system_prompt
    assert res.get('kb', {}).get('mode') == 'rag'

    # 无关问题 → 纯 LLM，system prompt 无资料段
    fake_client.chat.completions.calls.clear()
    r = client.get('/api/search', query_string={'title': '请计算 2 的 10 次方', 'type': 'completion'})
    res = r.get_json()
    calls = fake_client.chat.completions.calls
    system_prompt = calls[0]['messages'][0]['content'] if calls else ''
    print('[7] 纯 LLM:', res['code'], res.get('kb'), '| prompt 含资料:', '知识库检索到的资料' in system_prompt)
    assert res['code'] == 1 and len(calls) == 1
    assert '知识库检索到的资料' not in system_prompt
    assert res.get('kb') is None

    # 不带 type/options 也能用
    r = client.get('/api/search', query_string={'title': '中国的首都是哪个城市？'})
    print('[8] 仅 title:', r.get_json()['answer'])
    assert r.get_json()['code'] == 1

    # 空题目 → 400 逻辑（code 0）
    r = client.get('/api/search', query_string={'title': ''})
    print('[9] 空题目:', r.get_json())
    assert r.get_json()['code'] == 0

    # 嵌入服务故障 → 降级纯 LLM，接口仍 200/code 1
    class BrokenEmbedder:
        model = 'broken'

        def embed(self, texts):
            raise RuntimeError('embedding gateway down')

    good = kb.embedder
    kb.embedder = BrokenEmbedder()
    fake_client.chat.completions.calls.clear()
    r = client.get('/api/search', query_string={'title': '中国的首都是哪个城市？', 'type': 'single'})
    res = r.get_json()
    print('[10] 嵌入故障降级:', r.status_code, res['code'], res['answer'], '| 调 LLM:', len(fake_client.chat.completions.calls))
    assert r.status_code == 200 and res['code'] == 1 and len(fake_client.chat.completions.calls) == 1
    kb.embedder = good

    # stats 带知识库信息
    r = client.get('/api/stats', query_string={'token': ''})
    stats = r.get_json()
    print('[11] stats kb:', stats.get('kb'))
    assert stats.get('kb', {}).get('chunks', 0) > 0

    # 首页和文档页仍可用
    print('[12] / ->', client.get('/').status_code, '| /docs ->', client.get('/docs').status_code,
          '| /dashboard ->', client.get('/dashboard').status_code)
    assert client.get('/').status_code == 200
    assert client.get('/docs').status_code == 200
    assert client.get('/dashboard').status_code == 200

    print('\n集成自测全部通过 ✅')


if __name__ == '__main__':
    main()
