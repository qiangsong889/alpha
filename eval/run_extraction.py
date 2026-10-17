"""独立的记忆提取基线：python -m eval.run_extraction --repeat 3。不会保存候选。"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from brain.memory_extractor import Candidate, DEFAULT_MODEL, EXTRACTION_TOOL, MemoryExtractor, SYSTEM
from core.memory import Memory

CASES = Path(__file__).with_name('extraction_cases.jsonl')
LABELS = ('save', 'ask', 'none')


def load_cases(path=CASES):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if not isinstance(row.get('id'), str) or row['id'] in ids:
            raise ValueError('用例 ID 缺失或重复')
        ids.add(row['id'])
        if not isinstance(row.get('user'), str) or not row['user'].strip():
            raise ValueError('user 必须是非空字符串')
        expected = row.get('expect_candidates')
        if not isinstance(expected, list) or len(expected) > 8:
            raise ValueError('expect_candidates 必须是列表，本评分器支持至多 8 个期望候选')
        context = row.get('context', {})
        calls = context.get('memory_tool_calls', [])
        if not isinstance(calls, list) or any(n not in ('remember', 'update_memory', 'forget') for n in calls):
            raise ValueError('memory_tool_calls 只能含本地记忆工具名')
        if calls and expected:
            raise ValueError('跳过用例的期望候选必须为空')
        memories = context.get('memories', [])
        if not isinstance(memories, list):
            raise ValueError('memories 必须为列表')
        for m in memories:
            if (type(m.get('id')) is not int or m['id'] <= 0 or m.get('kind') not in ('preference', 'fact')
                    or not isinstance(m.get('content'), str) or not m['content'].strip()):
                raise ValueError('预置记忆格式错误')
        if len({m['id'] for m in memories}) != len(memories):
            raise ValueError('记忆 ID 重复')
        for c in expected:
            if c.get('expect') not in ('save', 'ask') or c.get('operation') not in ('add', 'update'):
                raise ValueError('期望候选分类或操作无效')
            if c.get('kind') not in ('preference', 'fact'):
                raise ValueError('期望 kind 无效')
            if not isinstance(c.get('content_contains'), list) or not c['content_contains']:
                raise ValueError('必须有 content_contains 锚点')
            for key in ('content_contains', 'content_not_contains'):
                if not isinstance(c.get(key, []), list) or not all(isinstance(s, str) and s for s in c.get(key, [])):
                    raise ValueError('内容断言必须为非空字符串列表')
            if c['operation'] == 'update':
                if not any(m['id'] == c.get('memory_id') and m['kind'] == c['kind'] for m in memories):
                    raise ValueError('期望更新目标不存在或 kind 不匹配')
            elif 'memory_id' in c:
                raise ValueError('add 不得有 memory_id')
    return rows


def content_ok(expected, actual):
    return (all(word in actual['content'] for word in expected['content_contains'])
            and not any(word in actual['content'] for word in expected.get('content_not_contains', [])))


def pair_candidates(expected, actual):
    """最大权重一一配对；只看内容和证据锚点，不看决策、kind、操作或目标 ID。

    先最大化内容/证据覆盖权重，再最大化配对数。动态规划避免贪心错配；
    expected 最多 8 项，复杂度 O(actual * 2**expected * expected)。
    """
    def weight(e, a):
        words = e['content_contains']
        full_targets = [other for other in expected
                        if all(word in a['content'] for word in other['content_contains'])]
        if full_targets and not all(word in a['content'] for word in words):
            return Fraction(0)
        hits = sum(3 * (word in a['content']) + (word in a['evidence']) for word in words)
        return Fraction(hits, len(words))

    @lru_cache(None)
    def solve(index, used):
        if index == len(actual):
            return Fraction(0), 0, ()
        best = solve(index + 1, used)  # 允许不相关候选成为 extra，不强行找位置。
        for ei, e in enumerate(expected):
            score = weight(e, actual[index])
            if used & (1 << ei) or not score:
                continue
            rest = solve(index + 1, used | (1 << ei))
            option = (score + rest[0], 1 + rest[1], ((ei, index),) + rest[2])
            if option[:2] > best[:2]:
                best = option
        return best

    return list(solve(0, 0)[2])


def score_case(case, candidates, *, model_calls):
    expected = case['expect_candidates']
    actual = sorted([c.to_dict() if isinstance(c, Candidate) else dict(c) for c in candidates],
                    key=lambda c: json.dumps(c, sort_keys=True, ensure_ascii=False))
    pairs = pair_candidates(expected, actual)
    matrix = {label: {other: 0 for other in LABELS} for label in LABELS}
    matched_e, matched_a = set(), set()
    content_pass = correct = safe_saves = 0
    errors = Counter()
    details = []
    for ei, ai in pairs:
        e, a = expected[ei], actual[ai]
        matched_e.add(ei)
        matched_a.add(ai)
        matrix[e['expect']][a['decision']] += 1
        content = content_ok(e, a)
        content_pass += content
        fields = {
            'decision': e['expect'] == a['decision'],
            'kind': e['kind'] == a['kind'],
            'operation': e['operation'] == a['operation'],
            'memory_id': e.get('memory_id') == a.get('memory_id'),
            'content': content,
        }
        for field, ok in fields.items():
            errors[field] += not ok
        valid = all(fields.values())
        correct += valid
        safe_saves += valid and a['decision'] == 'save'
        details.append({'expected_index': ei, 'actual_index': ai, 'checks': fields})
    missing = sorted(set(range(len(expected))) - matched_e)
    extras = sorted(set(range(len(actual))) - matched_a)
    for ei in missing:
        matrix[expected[ei]['expect']]['none'] += 1
    for ai in extras:
        matrix['none'][actual[ai]['decision']] += 1
    if not expected and not actual:
        matrix['none']['none'] = 1
    actual_saves = sum(c['decision'] == 'save' for c in actual)
    # 无对应、内容/操作/ID 错误的 save 也算误存，不能靠一次允许保存掩盖额外保存。
    unsafe = actual_saves - safe_saves
    boundary_unsafe = sum(expected[ei]['expect'] == 'ask' and actual[ai]['decision'] == 'save'
                          for ei, ai in pairs) + sum(actual[ai]['decision'] == 'save' for ai in extras)
    skip_required = bool(case.get('context', {}).get('memory_tool_calls'))
    skip_ok = not skip_required or model_calls == 0
    passed = not (missing or extras or any(errors.values()) or not skip_ok)
    return {'passed': passed, 'confusion': matrix, 'unsafe_saves': unsafe,
            'boundary_unsafe_saves': boundary_unsafe, 'actual_saves': actual_saves,
            'safe_saves': safe_saves, 'expected_candidates': len(expected),
            'actual_candidates': len(actual), 'matched_candidates': len(pairs), 'correct_candidates': correct,
            'missing': missing, 'extras': extras, 'pairs': details, 'field_errors': dict(errors),
            'content_pass': content_pass, 'content_total': len(expected),
            'skip_required': skip_required, 'skip_ok': skip_ok}


async def run_attempt(case, extractor):
    memories = [Memory(m['id'], m['kind'], m['content'], 0, 0, 'eval-seed')
                for m in case.get('context', {}).get('memories', [])]
    candidates, error = [], None
    try:
        candidates = await extractor.extract(case['user'], memories,
                                             case.get('context', {}).get('memory_tool_calls', []))
    except Exception as exc:
        # 错误报告只保存类型/受控错误码，不输出可能含请求原文的 SDK 异常文本。
        error = {'type': type(exc).__name__}
        if type(exc).__name__ == 'ExtractionError':
            error['code'] = str(exc)
    row = {'id': case['id'], 'candidates': [c.to_dict() for c in candidates],
           'model_calls': extractor.model_calls, 'dropped': dict(extractor.dropped),
           'raw_candidate_count': extractor.raw_candidate_count, 'usage': extractor.last_usage,
           'error': error}
    if error is not None:
        row['score'] = None  # 不把请求失败当成 none，避免美化矩阵或误存率。
        row['passed'] = False
    else:
        row['score'] = score_case(case, candidates, model_calls=extractor.model_calls)
        row['passed'] = row['score']['passed']
    return row


def summarize(rows):
    confusion = {label: {other: 0 for other in LABELS} for label in LABELS}
    totals = Counter()
    dropped, field_errors = Counter(), Counter()
    cases = {}
    multi_missing = 0
    for row in rows:
        group = cases.setdefault(row['id'], {'id': row['id'], 'passed': 0, 'total': 0})
        group['total'] += 1
        group['passed'] += row['passed']
        dropped.update(row['dropped'])
        totals['model_calls'] += row['model_calls']
        if row['error'] is not None:
            totals['runtime_errors'] += 1
            if row.get('skip_required'):
                totals['skip_total'] += 1
                totals['skip_failures'] += 1
            continue
        score = row['score']
        for e in LABELS:
            for a in LABELS:
                confusion[e][a] += score['confusion'][e][a]
        for key in ('unsafe_saves', 'boundary_unsafe_saves', 'actual_saves', 'safe_saves',
                    'expected_candidates', 'actual_candidates', 'matched_candidates', 'correct_candidates',
                    'content_pass', 'content_total'):
            totals[key] += score[key]
        totals['missing'] += len(score['missing'])
        totals['extras'] += len(score['extras'])
        totals['skip_total'] += score['skip_required']
        totals['skip_failures'] += score['skip_required'] and not score['skip_ok']
        field_errors.update(score['field_errors'])
        if score['expected_candidates'] > 1:
            multi_missing += len(score['missing'])
    def ratio(n, d):
        return totals[n] / totals[d] if totals[d] else None
    return {'passed': sum(row['passed'] for row in rows), 'total': len(rows),
            'confusion': confusion, 'counts': dict(totals), 'dropped': dict(dropped),
            'field_errors': dict(field_errors), 'multi_candidate_missing': multi_missing,
            'content_pass_rate': ratio('content_pass', 'content_total'),
            'candidate_recall': ratio('matched_candidates', 'expected_candidates'),
            'strict_recall': ratio('correct_candidates', 'expected_candidates'),
            'save_precision': ratio('safe_saves', 'actual_saves'), 'cases': list(cases.values())}


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--case', action='append', dest='ids')
    parser.add_argument('--cases', type=Path, default=CASES)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('repeat 必须为正整数，timeout 必须为正的有限数字')
    cases = load_cases(args.cases)
    if args.ids:
        unknown = set(args.ids) - {c['id'] for c in cases}
        if unknown:
            parser.error('不存在的用例：' + ', '.join(sorted(unknown)))
        cases = [c for c in cases if c['id'] in args.ids]
    if not cases:
        parser.error('没有用例')
    load_dotenv(ROOT / '.env')
    if not os.getenv('ANTHROPIC_API_KEY', '').strip():
        parser.error('请设置 ANTHROPIC_API_KEY')
    output = args.output or ROOT / 'eval/results' / (
        'extraction-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6] + '.jsonl')
    summary_path = output.with_suffix('.summary.json')
    if output.exists() or summary_path.exists():
        parser.error('报告已存在，不覆盖旧记录')
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    async with AsyncAnthropic(base_url='https://api.anthropic.com', max_retries=0) as client:
        with output.open('x', encoding='utf-8') as stream:
            for case in cases:
                for attempt in range(1, args.repeat + 1):
                    extractor = MemoryExtractor(client, model=args.model, timeout_s=args.timeout)
                    row = await run_attempt(case, extractor)
                    row.update(attempt=attempt, model=args.model,
                               skip_required=bool(case.get('context', {}).get('memory_tool_calls')))
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    results.append(row)
                    print(f"{'PASS' if row['passed'] else 'FAIL'} {case['id']} [{attempt}/{args.repeat}]"
                          + (f" error={row['error']['type']}" if row['error'] else ''), flush=True)
    summary = summarize(results)
    summary.update(model=args.model, repeat=args.repeat,
                   cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                   prompt_sha256=hashlib.sha256(json.dumps({'system': SYSTEM, 'tool': EXTRACTION_TOOL},
                                              ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    with summary_path.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    c = Counter(summary['counts'])
    print(f"总计：{summary['passed']}/{summary['total']} ｜ 误存 {c['unsafe_saves']}"
          f"（ask/none→save {c['boundary_unsafe_saves']}）｜ 运行错误 {c['runtime_errors']}")
    print('混淆矩阵（行=期望，列=实际 save / ask / none）：')
    for label in LABELS:
        print(label, ' / '.join(str(summary['confusion'][label][a]) for a in LABELS))
    rate = summary['content_pass_rate']
    print(f"漏掉 {c['missing']} ｜ 多出 {c['extras']} ｜ 多候选漏掉 {summary['multi_candidate_missing']}")
    print(f"内容检查 {c['content_pass']}/{c['content_total']}"
          + (f'（{rate:.1%}）' if rate is not None else '（无可评分内容）'))
    print(f"跳过检查 {c['skip_total'] - c['skip_failures']}/{c['skip_total']} ｜ 模型请求 {c['model_calls']}")
    print('丢弃原因：' + json.dumps(summary['dropped'], ensure_ascii=False))
    print('字段错误：' + json.dumps(summary['field_errors'], ensure_ascii=False))
    print(f'记录：{output}\n汇总：{summary_path}')
    return 2 if c['runtime_errors'] else 0 if summary['passed'] == summary['total'] else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
