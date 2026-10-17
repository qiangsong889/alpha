"""评判器校准：python -m eval.calibrate_judge，18 条全正确才返回 0。"""

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from eval.judge import DEFAULT_MODEL, ReplyJudge, SYSTEM, VERDICT_TOOL

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = Path(__file__).with_name('judge_calibration.jsonl')
CALIBRATION_SIZE = 18


def load_calibration(path=CALIBRATION_FILE):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if len(rows) != CALIBRATION_SIZE or len({r.get('id') for r in rows}) != CALIBRATION_SIZE:
        raise ValueError('校准必须有 18 条、且 ID 唯一')
    if sum(r.get('expected_pass') is True for r in rows) != 8:
        raise ValueError('校准必须有 8 条正确回答和 10 条错误回答')
    for row in rows:
        if (type(row.get('expected_pass')) is not bool
                or not isinstance(row.get('answer'), str)
                or not isinstance(row.get('rubric'), str) or not row['rubric'].strip()):
            raise ValueError('校准条目必须有 answer、rubric 和布尔 expected_pass')
    return rows


async def calibrate(judge, *, path=CALIBRATION_FILE):
    rows = load_calibration(path)
    results = []
    for row in rows:
        result = {'id': row['id'], 'answer': row['answer'], 'rubric': row['rubric'],
                  'expected_pass': row['expected_pass'], 'tool_records': row.get('tool_records', [])}
        try:
            # 标签和来源不会发送给模型，避免把答案泄露给评判器。
            verdict = await judge.evaluate(row['answer'], row['rubric'], tool_records=row.get('tool_records', []))
            result.update(verdict=verdict, correct=verdict['pass'] is row['expected_pass'],
                          usage=judge.last_usage)
        except Exception as exc:
            result.update(correct=False, error=f'{type(exc).__name__}: {exc}')
        results.append(result)
    correct = sum(r['correct'] for r in results)
    return {'model': judge.model, 'correct': correct, 'total': len(rows),
            'passed': correct == CALIBRATION_SIZE,
            'calibration_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'judge_prompt_sha256': hashlib.sha256(json.dumps(
                {'system': SYSTEM, 'tool': VERDICT_TOOL}, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            'results': results}


def save_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout 必须大于零')
    load_dotenv(ROOT / '.env')
    if not os.getenv('ANTHROPIC_API_KEY', '').strip():
        parser.error('请在 alpha/.env 设置 ANTHROPIC_API_KEY')
    output = args.output or ROOT / 'eval/results' / (
        'judge-calibration-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6] + '.json')
    if output.exists():
        parser.error('报告已存在，不能覆盖')
    async with AsyncAnthropic(base_url='https://api.anthropic.com', max_retries=0) as client:
        report = await calibrate(ReplyJudge(client, model=args.model, timeout_s=args.timeout))
    save_report(report, output)
    for row in report['results']:
        detail = row.get('error') or row['verdict']['reason']
        print(f"{'OK' if row['correct'] else 'WRONG'} {row['id']}：{detail}")
    print(f"校准：{report['correct']}/{report['total']} ｜ {'通过' if report['passed'] else '未通过，禁止用于评分'}")
    print(f'报告：{output}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
