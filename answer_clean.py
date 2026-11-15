# -*- coding: utf-8 -*-
"""
答案清洗（用于 extract_answer 的补丁段）

OCS 的答案匹配有三条路：精确匹配选项文本 → 相似度 > 0.6 → ABCD 字母序列。
像「B.两周」「A.1941年」这种「字母+内容」混合体既不是纯字母，也不等于选项文本，
风险最高（IContains 匹配可能因「两周」通过，也可能因「B.」前缀失败）。
所以单选题统一清洗成纯选项内容，判断题统一成 正确/错误。
"""

import re

# 选项前缀：A. A、 A． A) A） (A) A: 等
_OPTION_PREFIX = re.compile(
    r'^\s*[（(]?\s*([A-Ha-h])\s*[)）]?\s*[.、．:：)）]?\s*',
)

_JUDGE_TRUE = {'正确', '对', 'true', 't', '√', '是', 'right', 'a'}
_JUDGE_FALSE = {'错误', '错', 'false', 'f', '×', 'x', '否', 'wrong', 'b'}


def _strip_wrapping(text: str) -> str:
    """去掉整体包裹的引号、括号、markdown 加粗等"""
    t = text.strip()
    for _ in range(3):
        before = t
        t = t.strip()
        # 去掉 markdown 强调
        t = re.sub(r'^\*{1,3}(.+?)\*{1,3}$', r'\1', t, flags=re.S)
        # 去掉成对引号
        for pair in ('""', "''", '“”', '‘’', '《》', '「」', '『』'):
            if len(t) > 2 and t[0] == pair[0] and t[-1] == pair[1]:
                t = t[1:-1]
        # 去掉成对括号（但不拆掉答案本身的括号，如「（一）」）
        if len(t) > 2 and t[0] == '(' and t[-1] == ')':
            t = t[1:-1]
        t = t.strip().rstrip('。.;；,，')
        if t == before:
            break
    return t


def _strip_option_letter(text: str, options: str) -> str:
    """
    去掉答案开头的选项字母。

    仅当能确认选项里存在该字母对应的选项文本、或被去字母后内容与之相符时才去，
    避免把「A型题」里的 A 误删。
    """
    t = _strip_wrapping(text)
    m = _OPTION_PREFIX.match(t)
    if not m:
        return t
    letter = m.group(1).upper()
    rest = t[m.end():].strip()

    # 排除：「A型」「B超」这类，字母后紧跟汉字且不是选项内容的开头
    if not rest:
        # 只剩一个字母 → 保持原样（可能是纯字母答案）
        return t

    if options:
        # 在选项串里定位该字母对应的选项内容
        labeled = re.search(
            rf'(?:^|[\n\r])\s*{letter}\s*[.、．:：)）]?\s*(.+?)(?=(?:[\n\r]\s*[A-H]\s*[.、．:：)）]?)|$)',
            options, re.I | re.S)
        if labeled:
            opt_text = _strip_wrapping(labeled.group(1))
            # 去掉字母后的内容与选项文本前缀吻合 → 安全去掉字母
            if opt_text and (rest[:6] == opt_text[:6] or opt_text[:6] in rest):
                return rest

    # 没有选项串可依据时，保守处理：字母后紧跟标点（.、．)等）才认为是选项前缀
    if re.match(rf'^[A-Ha-h]\s*[.、．:：)）]', t):
        return rest
    if re.match(rf'^[（(]\s*[A-Ha-h]\s*[)）]', t):
        return rest
    return t


def _normalize_judgement(text: str) -> str:
    """判断题归一成「正确」/「错误」；无法确定则原样返回"""
    t = _strip_wrapping(text)
    low = t.lower()
    if low in _JUDGE_TRUE:
        return '正确'
    if low in _JUDGE_FALSE:
        return '错误'
    # 形如「正确（√）」「错误：不得…」→ 取首词判断
    head = re.split(r'[（({\[：:，,。.\s]', low)[0]
    if head in _JUDGE_TRUE:
        return '正确'
    if head in _JUDGE_FALSE:
        return '错误'
    if low.startswith(('正确', '对', '√')):
        return '正确'
    if low.startswith(('错误', '错', '×', '不对')):
        return '错误'
    return t


def _split_multi_letters(text: str) -> str:
    """多选题答案统一成 A#B#C 形式"""
    t = _strip_wrapping(text).upper()
    # 已经是 A#B 形式
    if '#' in t:
        parts = [p.strip() for p in t.split('#') if p.strip()]
        letters = []
        for p in parts:
            m = re.match(r'^[（(]?\s*([A-H])\s*[)）]?$', p)
            if m:
                letters.append(m.group(1))
        if letters:
            return '#'.join(sorted(set(letters)))
        return '#'.join(parts)
    # 形如 "ABC" / "A B C" / "A、B、C"
    found = re.findall(r'\b([A-H])\b', t)
    if not found:
        found = re.findall(r'([A-H])', t)
    if found:
        return '#'.join(sorted(set(found)))
    return t


def parse_options(options: str) -> dict:
    """
    从选项串里解析出 {字母: 选项内容}。

    支持 "A.张三 B.李四"、多行 "A、xx\\nB、yy"、"(A) xx" 等常见写法。
    """
    if not options:
        return {}
    text = options.replace('\r', '\n')
    # 行内多个选项排在一起时先拆开：在「字母+分隔符」前断行
    text = re.sub(r'(?<!^)\s*(?=[（(]?[A-H]\s*[.、．:：)）]\s*\S)', '\n', text)
    result = {}
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^[（(]?\s*([A-Ha-h])\s*[)）]?\s*[.、．:：)）]?\s*(.+)$', line)
        if m:
            letter = m.group(1).upper()
            content = _strip_wrapping(m.group(2))
            if content and letter not in result:
                result[letter] = content
    return result


def _letter_to_text(answer: str, options: str) -> str:
    """答案是纯字母（或 A#B）时，映射回选项内容"""
    t = _strip_wrapping(answer)
    mapped = parse_options(options)
    if not mapped:
        return t

    # 多字母形如 B / ABC / A#B
    letters = [x for x in re.split(r'[^A-Ha-h]+', t.upper()) if x]
    if letters and ''.join(letters) == t.replace('#', '').replace(' ', '').upper():
        texts = [mapped.get(l) for l in letters]
        if all(texts):
            return '#'.join(texts) if len(texts) > 1 else texts[0]
    return t


def clean_answer(ai_response: str, question_type: str, options: str = '') -> str:
    """
    按题型清洗答案，提高 OCS 匹配成功率。

    - single / 单选：去选项字母前缀，得到纯选项内容
    - judgement / 判断：归一成 正确 / 错误
    - multiple / 多选：统一成 A#B#C
    - completion / 填空：去包裹引号与句末标点
    """
    if not ai_response:
        return ''
    qtype = (question_type or '').lower().strip()

    if qtype in ('judgement', 'judge', 'true_false', 'tf', '判断', '判断题'):
        return _normalize_judgement(ai_response)

    if qtype in ('multiple', 'multi', '多选', '多选题'):
        return _split_multi_letters(ai_response)

    if qtype in ('single', 'radio', '单选', '单选题', ''):
        cleaned = _strip_option_letter(ai_response, options)
        cleaned = _strip_wrapping(cleaned)
        cleaned = _letter_to_text(cleaned, options)
        return cleaned or ai_response.strip()

    if qtype in ('completion', 'fill', '填空', '填空题'):
        return _strip_wrapping(ai_response)

    # 未知题型：至少去掉包裹符号
    return _strip_wrapping(ai_response)
