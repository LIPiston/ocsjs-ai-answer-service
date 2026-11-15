# -*- coding: utf-8 -*-
"""
EduBrain AI - 智能题库系统
基于 OpenAI API 的智能题库服务，提供兼容 OCS 接口的智能答题功能
作者：Lynn
版本：1.1.0
"""
from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS
import os
import time
import logging
import openai
import json
from datetime import datetime

from config import Config
from utils import SimpleCache, format_answer_for_ocs, parse_question_and_options, extract_answer
from embedder import Embedder
from kb import KnowledgeBase

# 配置日志
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ai_answer_service')

# 初始化应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 初始化缓存
cache = SimpleCache(Config.CACHE_EXPIRATION) if Config.ENABLE_CACHE else None

# 验证OpenAI API密钥
if not Config.OPENAI_API_KEY:
    logger.critical("未设置OpenAI API密钥，请在.env文件中配置OPENAI_API_KEY")
    raise ValueError("请设置环境变量OPENAI_API_KEY")

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key=Config.OPENAI_API_KEY,
    base_url=Config.OPENAI_API_BASE
)

# 问答记录存储（实际应用中可以使用数据库）
qa_records = []
MAX_RECORDS = 100  # 最多保存100条记录
start_time = time.time()

# ---------- 知识库（可选，KB_ENABLED） ----------
kb = None
if Config.KB_ENABLED:
    _embedder = Embedder(
        api_key=Config.EMBEDDING_API_KEY,
        base_url=Config.EMBEDDING_API_BASE,
        model=Config.EMBEDDING_MODEL,
        batch_size=Config.EMBEDDING_BATCH_SIZE,
        timeout=Config.EMBEDDING_TIMEOUT,
        dimensions=int(Config.EMBEDDING_DIMENSIONS) if Config.EMBEDDING_DIMENSIONS else None,
    )
    kb = KnowledgeBase(
        kb_dir=Config.KB_DIR,
        index_path=Config.KB_INDEX_PATH,
        embedder=_embedder,
        top_k=Config.KB_TOP_K,
        min_score=Config.KB_MIN_SCORE,
        answer_threshold=Config.KB_ANSWER_THRESHOLD,
        keyword_weight=Config.KB_KEYWORD_WEIGHT,
        chunk_target=Config.KB_CHUNK_TARGET,
    )
    logger.info(f"知识库已启用: dir={Config.KB_DIR} model={Config.EMBEDDING_MODEL}")

    if Config.KB_INDEX_AUTO:
        import threading as _threading

        def _auto_index():
            try:
                if kb.ready():
                    st = kb.status()
                    logger.info(f"知识库索引已加载: {st['chunks']} 块 ({st['model']})")
                else:
                    logger.info("知识库索引缺失，后台开始建库…")
                    result = kb.rebuild()
                    logger.info(f"后台建库结果: {result}")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"后台建库失败（服务继续以纯 LLM 模式运行）: {exc}", exc_info=True)

        _threading.Thread(target=_auto_index, name='kb-autoindex', daemon=True).start()

def verify_access_token(request):
    """验证访问令牌（如果配置了的话）"""
    if Config.ACCESS_TOKEN:
        token = request.headers.get('X-Access-Token') or request.args.get('token')
        if not token or token != Config.ACCESS_TOKEN:
            return False
    return True

@app.route('/api/search', methods=['GET', 'POST'])
def search():
    """
    处理OCS发送的搜索请求，使用OpenAI API生成答案
    GET请求: 从URL参数获取问题
    POST请求: 从请求体获取问题
    
    参数:
        title: 问题内容
        type: 问题类型 (single-单选, multiple-多选, judgement-判断, completion-填空)
        options: 选项内容
        
    返回:
        成功: {'code': 1, 'question': '问题', 'answer': 'AI生成的答案'}
        失败: {'code': 0, 'msg': '错误信息'}
    """
    start_time = time.time()
    
    # 验证访问令牌（如果配置了的话）
    if not verify_access_token(request):
        return jsonify({
            'code': 0,
            'msg': '无效的访问令牌'
        }), 403
    
    try:
        # 根据请求方法获取问题内容
        if request.method == 'GET':
            question = request.args.get('title', '')
            question_type = request.args.get('type', '')
            options = request.args.get('options', '')
        else:  # POST
            content_type = request.headers.get('Content-Type', '')
            
            if 'application/json' in content_type:
                data = request.get_json()
                question = data.get('title', '')
                question_type = data.get('type', '')
                options = data.get('options', '')
            else:
                # 处理表单数据
                question = request.form.get('title', '')
                question_type = request.form.get('type', '')
                options = request.form.get('options', '')
        
        # 记录接收到的问题
        logger.info(f"接收到问题: '{question[:50]}...' (类型: {question_type})")
        
        # 如果没有提供问题，返回错误
        if not question:
            logger.warning("未提供问题内容")
            return jsonify({
                'code': 0,
                'msg': '未提供问题内容'
            })
        
        # 检查缓存中是否有此问题的答案
        if Config.ENABLE_CACHE:
            cached_answer = cache.get(question, question_type, options)
            if cached_answer:
                logger.info(f"从缓存获取答案 (耗时: {time.time() - start_time:.2f}秒)")
                return jsonify(format_answer_for_ocs(question, cached_answer))

        # 先查知识库（可选）：命中问答条目直接作答，否则把检索到的资料交给 LLM
        kb_result = {'mode': 'none', 'answer': None, 'context': '', 'hits': []}
        if kb is not None:
            try:
                kb_result = kb.lookup(question, question_type, options)
            except Exception as exc:  # noqa: BLE001 - 知识库故障不能拖垮答题
                logger.warning(f"知识库检索异常，降级为纯 LLM: {exc}", exc_info=True)

        if kb_result['mode'] == 'kb' and kb_result['answer']:
            answer = extract_answer(kb_result['answer'], question_type, options)
            source = kb_result['hits'][0] if kb_result['hits'] else {}
            logger.info(
                f"知识库命中直接作答 (score={source.get('score')}, source={source.get('source')}) "
                f"耗时 {time.time() - start_time:.2f}s"
            )
            if Config.ENABLE_CACHE:
                cache.set(question, answer, question_type, options)
            qa_records.append({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'type': question_type,
                'options': options,
                'answer': answer,
                'source': f"kb:{source.get('source', '')}",
            })
            if len(qa_records) > MAX_RECORDS:
                qa_records.pop(0)
            resp = format_answer_for_ocs(question, answer)
            resp['kb'] = {'mode': 'kb', 'source': source.get('source', ''), 'score': source.get('score')}
            return jsonify(resp)

        # 构建发送给OpenAI的提示
        prompt = parse_question_and_options(question, options, question_type)

        system_prompt = "你是一个专业的考试答题助手。请直接回答答案，不要解释。选择题只回答选项的内容(如：地球)；多选题用#号分隔答案,只回答选项的内容(如中国#世界#地球)；判断题只回答: 正确/对/true/√ 或 错误/错/false/×；填空题直接给出答案。"
        if kb_result['context']:
            context = kb_result['context'][:Config.KB_CONTEXT_MAX_CHARS]
            system_prompt += (
                "\n\n下面是从题库知识库检索到的资料。规则：\n"
                "1. 若资料中能直接找到题目所问的事实（尤其是具体的数字、年限、比例、金额、条数），"
                "必须严格照抄资料里的数值，不要自己推断或改写。\n"
                "2. 资料与题目无关时忽略它，凭你的知识作答。\n"
                "3. 答选择题时只输出选项内容本身，不要带字母前缀。\n"
                "4. 不要在答案中引用资料原文或标注来源。\n"
                f"----\n{context}\n----"
            )

        # 调用OpenAI API
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        # 获取AI生成的答案
        ai_answer = response.choices[0].message.content.strip()
        
        # 处理答案格式
        processed_answer = extract_answer(ai_answer, question_type, options)
        
        # 保存到缓存
        if Config.ENABLE_CACHE:
            cache.set(question, processed_answer, question_type, options)
        
        # 保存问答记录
        current_time = datetime.now()
        qa_records.append({
            'time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': current_time.isoformat(),
            'question': question,
            'type': question_type,
            'options': options,
            'answer': processed_answer,
            'source': 'kb:rag' if kb_result['mode'] == 'rag' else 'llm',
        })
        if len(qa_records) > MAX_RECORDS:
            qa_records.pop(0)
        
        # 记录处理时间
        process_time = time.time() - start_time
        logger.info(f"问题处理完成 (耗时: {process_time:.2f}秒, 知识库={kb_result['mode']})")
        
        # 返回符合OCS格式的响应
        payload = format_answer_for_ocs(question, processed_answer)
        if kb_result['mode'] != 'none':
            payload['kb'] = {
                'mode': kb_result['mode'],
                'sources': [h.get('source') for h in kb_result['hits'][:3]],
                'top_score': kb_result['hits'][0].get('score') if kb_result['hits'] else None,
            }
        return jsonify(payload)
    
    except Exception as e:
        # 记录异常
        logger.error(f"处理问题时发生错误: {str(e)}", exc_info=True)
        
        # 捕获所有异常并返回错误信息
        return jsonify({
            'code': 0,
            'msg': f'发生错误: {str(e)}'
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    payload = {
        'status': 'ok',
        'message': 'AI题库服务运行正常',
        'version': '1.2.0',
        'cache_enabled': Config.ENABLE_CACHE,
        'model': Config.OPENAI_MODEL,
        'kb_enabled': kb is not None,
    }
    if kb is not None:
        status = kb.status()
        payload['kb_ready'] = status['ready']
        payload['kb_chunks'] = status['chunks']
        payload['embedding_model'] = status['model'] or Config.EMBEDDING_MODEL
    return jsonify(payload)


@app.route('/api/kb/status', methods=['GET'])
def kb_status():
    """知识库状态"""
    if kb is None:
        return jsonify({'enabled': False, 'message': '知识库未启用（KB_ENABLED=false）'})
    return jsonify(kb.status())


@app.route('/api/kb/rebuild', methods=['POST'])
def kb_rebuild():
    """重建知识库索引：重新扫描 KB_DIR，切块并调用 bge-m3 重新向量化"""
    if not verify_access_token(request):
        return jsonify({'success': False, 'message': '无效的访问令牌'}), 403
    if kb is None:
        return jsonify({'success': False, 'message': '知识库未启用（KB_ENABLED=false）'}), 400
    result = kb.rebuild()
    return jsonify(result), (200 if result.get('success') else 500)


@app.route('/api/kb/search', methods=['GET', 'POST'])
def kb_search():
    """知识库检索（调试用）：返回命中的资料块与分数"""
    if not verify_access_token(request):
        return jsonify({'success': False, 'message': '无效的访问令牌'}), 403
    if kb is None:
        return jsonify({'success': False, 'message': '知识库未启用（KB_ENABLED=false）'}), 400
    if request.method == 'GET':
        query = request.args.get('q', '') or request.args.get('title', '')
        top_k = request.args.get('top_k')
    else:
        data = request.get_json(silent=True) or {}
        query = data.get('q', '') or data.get('title', '')
        top_k = data.get('top_k')
    if not query:
        return jsonify({'success': False, 'message': '缺少查询内容 q'}), 400
    try:
        top_k = int(top_k) if top_k else None
    except (TypeError, ValueError):
        top_k = None
    hits = kb.search(query, top_k)
    return jsonify({
        'success': True,
        'query': query,
        'hits': hits,
        'mode': ('kb' if hits and hits[0].get('answer') and hits[0]['score'] >= Config.KB_ANSWER_THRESHOLD
                 else ('rag' if hits else 'none')),
    })

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """清除缓存接口"""
    # 验证访问令牌
    if not verify_access_token(request):
        return jsonify({
            'success': False,
            'message': '无效的访问令牌'
        }), 403
    
    if not Config.ENABLE_CACHE:
        return jsonify({
            'success': False,
            'message': '缓存未启用'
        })
    
    cache.clear()
    return jsonify({
        'success': True,
        'message': '缓存已清除'
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取服务统计信息"""
    # 验证访问令牌
    if not verify_access_token(request):
        return jsonify({
            'success': False,
            'message': '无效的访问令牌'
        }), 403
    
    stats = {
        'version': '1.2.0',
        'uptime': time.time() - start_time,
        'model': Config.OPENAI_MODEL,
        'cache_enabled': Config.ENABLE_CACHE,
        'cache_size': len(cache.cache) if Config.ENABLE_CACHE else 0,
        'qa_records_count': len(qa_records)
    }
    if kb is not None:
        status = kb.status()
        stats['kb'] = {
            'ready': status['ready'],
            'chunks': status['chunks'],
            'embedding_model': status['model'] or Config.EMBEDDING_MODEL,
            'built_at': status['built_at'],
        }
    
    return jsonify(stats)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """仪表盘 - 显示问答记录和系统状态"""
    uptime_seconds = time.time() - start_time
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    uptime_str = f"{days}天{hours}小时{minutes}分钟"
    
    status = kb.status() if kb is not None else None
    return render_template(
        'dashboard.html',
        version="1.2.0",
        cache_enabled=Config.ENABLE_CACHE,
        cache_size=len(cache.cache) if Config.ENABLE_CACHE else 0,
        model=Config.OPENAI_MODEL,
        uptime=uptime_str,
        records=qa_records,
        kb_enabled=kb is not None,
        kb_ready=bool(status and status['ready']),
        kb_chunks=status['chunks'] if status else 0,
        kb_model=(status['model'] or Config.EMBEDDING_MODEL) if status else '',
    )

@app.route('/', methods=['GET'])
def index():
    """首页 - 显示Web界面"""
    return render_template('index.html')

@app.route('/docs', methods=['GET'])
def docs():
    """API文档页面"""
    with open('api_docs.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用markdown库将文档转换为HTML（需要安装：pip install markdown）
    try:
        import markdown
        html_content = markdown.markdown(content, extensions=['tables'])
        
        return f"""
        <html>
            <head>
                <title>AI题库服务 - API文档</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    code {{ background: #e0e0e0; padding: 2px 4px; border-radius: 3px; }}
                    pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; }}
                    th {{ background-color: #f4f4f4; }}
                </style>
            </head>
            <body>
                <div class="container">
                    {html_content}
                </div>
            </body>
        </html>
        """
    except ImportError:
        # 如果没有安装markdown库，则返回纯文本
        return f"""
        <html>
            <head>
                <title>AI题库服务 - API文档</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #333; }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>AI题库服务 - API文档</h1>
                    <pre>{content}</pre>
                </div>
            </body>
        </html>
        """

if __name__ == '__main__':
    # 开启应用
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)