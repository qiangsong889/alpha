"""控制台：令牌和 Host 检查、只读数据库、密钥不出现在响应里、总览和设置的计算。不启动真实进程，不调用模型。"""

import asyncio
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from console.data import ConsoleData, Unavailable, open_readonly, start_of_today
from console.server import COOKIE, CSRF_HEADER, create_app
from core.archive import ArchiveStore
from core.costs import CostLedger
from core.runtime import ProcessStatus, snapshot, write_status
from core.settings import BRAIN, VOICE, check_settings

TOKEN = 'test-token'
SECRETS = {'HA_TOKEN': 'ha-secret-value', 'ANTHROPIC_API_KEY': 'sk-ant-secret-value',
           'DASHSCOPE_API_KEY': 'ds-secret-value', 'TYPESAFE_API_KEY': 'ts-secret-value'}
ENV = {'HA_URL': 'http://ha.local:8123', 'QWEN_WORKSPACE_ID': 'ws-1', **SECRETS}
NOW = time.time()


def env_text(extra=None):
    return ''.join(f'{key}={value}\n' for key, value in {**ENV, **(extra or {})}.items())


class Fixture:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.env_file = root / '.env'
        self.env_file.write_text(env_text())
        self.web = root / 'web'
        (self.web / 'dist' / 'assets').mkdir(parents=True)
        (self.web / 'dist' / 'index.html').write_text('<!doctype html><div id="root"></div>')
        (self.web / 'dist' / 'assets' / 'app.js').write_text('console.log(1)')
        (root / 'secret.txt').write_text('outside dist')
        self.data = ConsoleData(env_file=self.env_file, archive_db=root / 'archive.db', costs_db=root / 'costs.db',
                                operations_db=root / 'operations.db', voice_status=root / 'run' / 'voice.json',
                                clock=lambda: NOW)
        self.brain = ProcessStatus('unreachable', detail='连不上大脑')

    async def brain_status(self, url):
        self.brain_url = url
        return self.brain

    def client(self, *, login=True):
        app = create_app(token=TOKEN, data=self.data, brain_status=self.brain_status, web_dir=self.web)
        client = TestClient(app, base_url='http://127.0.0.1:8770', follow_redirects=False)
        if login:
            response = client.get('/login', params={'token': TOKEN})
            assert response.status_code == 303, response.text
        return client

    def cleanup(self):
        self.directory.cleanup()


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.addCleanup(self.fixture.cleanup)

    def test_everything_requires_login_including_reads_and_the_page(self):
        client = self.fixture.client(login=False)
        for path in ('/api/overview', '/api/settings', '/', '/settings', '/assets/app.js'):
            self.assertEqual(client.get(path).status_code, 401, path)
        self.assertEqual(client.get('/login', params={'token': 'wrong'}).status_code, 403)
        self.assertEqual(client.get('/login').status_code, 403)

    def test_login_sets_strict_httponly_cookie_and_drops_token_from_url(self):
        client = self.fixture.client(login=False)
        response = client.get('/login', params={'token': TOKEN})
        self.assertEqual((response.status_code, response.headers['location']), (303, '/'))
        cookie = response.headers['set-cookie'].lower()
        self.assertIn('httponly', cookie)
        self.assertIn('samesite=strict', cookie)
        self.assertNotIn(TOKEN, cookie)  # cookie 的值不是登录令牌
        self.assertEqual(client.get('/api/overview').status_code, 200)

    def test_a_forged_cookie_is_rejected(self):
        client = self.fixture.client(login=False)
        client.cookies.set(COOKIE, TOKEN)
        self.assertEqual(client.get('/api/overview').status_code, 401)

    def test_other_hosts_are_rejected_even_with_a_valid_cookie(self):
        client = self.fixture.client()
        for host in ('evil.example', 'evil.example:8770', '127.0.0.1.evil.example', '[::1]:8770'):
            self.assertEqual(client.get('/api/overview', headers={'host': host}).status_code, 400, host)
        self.assertEqual(client.get('/api/overview', headers={'host': 'localhost:8770'}).status_code, 200)

    def test_state_changing_requests_need_same_origin_and_custom_header(self):
        client = self.fixture.client()
        good = {'origin': 'http://127.0.0.1:8770', CSRF_HEADER: '1'}
        self.assertEqual(client.post('/api/anything').status_code, 403)
        self.assertEqual(client.post('/api/anything', headers={**good, 'origin': 'http://evil.example'}).status_code, 403)
        self.assertEqual(client.post('/api/anything', headers={'origin': good['origin']}).status_code, 403)
        # 通过检查后才轮到路由：D3 没有任何写入接口。
        self.assertEqual(client.post('/api/anything', headers=good).status_code, 405)

    def test_security_headers_and_no_store(self):
        client = self.fixture.client()
        for path in ('/api/overview', '/', '/missing-before-login'):
            headers = client.get(path).headers
            self.assertIn("script-src 'self'", headers['content-security-policy'])
            self.assertIn("frame-ancestors 'none'", headers['content-security-policy'])
            self.assertEqual(headers['referrer-policy'], 'no-referrer')
            self.assertEqual(headers['x-content-type-options'], 'nosniff')
            self.assertEqual(headers['cache-control'], 'no-store')
        self.assertNotEqual(client.get('/assets/app.js').headers.get('cache-control'), 'no-store')

    def test_frontend_routes_fall_back_to_index_and_cannot_escape_dist(self):
        client = self.fixture.client()
        self.assertIn('root', client.get('/settings').text)
        self.assertEqual(client.get('/assets/app.js').text, 'console.log(1)')
        self.assertNotIn('outside dist', client.get('/../secret.txt').text)
        self.assertNotIn('outside dist', client.get('/%2e%2e/secret.txt').text)
        # 编码过的点和斜杠不会被客户端规范化，解码后才变成 ../，必须靠服务端的目录检查挡住。
        # secret.txt 在 dist 上面两层。
        self.assertNotIn('outside dist', client.get('/%2e%2e/%2e%2e/secret.txt').text)
        self.assertNotIn('outside dist', client.get('/assets%2f..%2f..%2f..%2fsecret.txt').text)
        self.assertEqual(client.get('/api/nope').status_code, 404)

    def test_responses_never_contain_secret_values(self):
        voice = check_settings(VOICE, ENV)
        write_status(self.fixture.data.voice_status, snapshot(voice, started_at=NOW - 60, now=NOW))
        self.fixture.brain = ProcessStatus('running', snapshot(check_settings(BRAIN, ENV), started_at=NOW - 60,
                                                               now=NOW, model='claude-haiku-5-5'))
        client = self.fixture.client()
        for path in ('/api/overview', '/api/settings'):
            text = client.get(path).text
            for secret in SECRETS.values():
                self.assertNotIn(secret, text, path)


class ReadOnlyTests(unittest.TestCase):
    def test_writes_fail_and_missing_files_are_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'archive.db'
            ArchiveStore(path).close()
            db = open_readonly(path)
            with self.assertRaises(sqlite3.OperationalError):
                db.execute("INSERT INTO sessions VALUES ('s', 1, 1, 0, NULL)")
            db.close()
            missing = Path(directory) / 'missing.db'
            with self.assertRaises(Unavailable):
                open_readonly(missing)
            self.assertFalse(missing.exists())


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.addCleanup(self.fixture.cleanup)
        self.data = self.fixture.data

    def test_missing_databases_are_unavailable_not_zero(self):
        overview = self.data.overview(self.fixture.brain)
        self.assertFalse(overview['today']['available'])
        self.assertFalse(overview['costs']['available'])
        self.assertFalse(overview['attention']['unknown_operations']['available'])
        self.assertEqual(overview['processes'][BRAIN]['state'], 'unreachable')
        self.assertEqual(overview['processes'][VOICE]['state'], 'not_started')

    def test_today_turns_and_costs_keep_known_estimate_and_to_verify_apart(self):
        today, yesterday = NOW, start_of_today(NOW) - 60
        archive = ArchiveStore(self.data.archive_db)
        archive.record_turn(turn_id='a' * 32, session_id='s1', user_text='开灯', answer_text='好',
                            status='ok', model='m', brain_ms=1000, created_at=today)
        archive.record_turn(turn_id='b' * 32, session_id='s1', user_text='关灯', answer_text=None,
                            status='error', model='m', brain_ms=3000, error='Timeout', created_at=today)
        archive.record_turn(turn_id='c' * 32, session_id='s0', user_text='昨天', answer_text='好',
                            status='ok', model='m', brain_ms=1, created_at=yesterday)
        archive.mark_delivery('a' * 32, 'played')
        archive.close()
        ledger = CostLedger(self.data.costs_db)
        ledger.db.executemany(
            'INSERT INTO cost_events (id, component, model, region, usage, usd, reason, updated_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)', [
                ('1', 'claude', 'm', 'r', '{}', '0.010000', '', today),
                ('2', 'tts', 'm', 'r', '{}', '0.002000', '', today),
                ('3', 'asr', 'm', 'r', '{}', '0.500000', 'embedded_asr_billing_unverified', today),
                ('4', 'claude', 'm', 'r', '{}', '0', 'request_started_no_final_usage', today),
                ('5', 'claude', 'm', 'r', '{}', '9.000000', '', yesterday),
            ])
        ledger.db.commit()
        ledger.close()
        overview = self.data.overview(self.fixture.brain)
        turns, costs = overview['today'], overview['costs']
        self.assertEqual((turns['turns'], turns['sessions'], turns['errors'], turns['avg_brain_ms']), (2, 1, 1, 2000))
        self.assertEqual(turns['delivery'], {'played': 1, 'not_reported': 1})
        self.assertEqual(Decimal(costs['known_usd']), Decimal('0.012'))       # 不含参考估算和昨天
        self.assertEqual(Decimal(costs['estimate_usd']), Decimal('0.5'))
        self.assertEqual(costs['to_verify'], 1)                               # 缺最终用量，不能当成零
        self.assertEqual(Decimal(costs['known_per_turn_usd']), Decimal('0.006'))
        self.assertEqual(costs['components']['claude']['to_verify'], 1)

    def test_unknown_operations_are_counted_for_attention(self):
        db = sqlite3.connect(self.data.operations_db)
        db.execute('CREATE TABLE operations (id TEXT, turn_id TEXT, name TEXT, arguments TEXT, status TEXT, '
                   'result TEXT, created_at REAL, updated_at REAL)')
        db.executemany('INSERT INTO operations VALUES (?, ?, ?, ?, ?, NULL, 0, 0)',
                       [('1', 't', 'light', '{}', 'unknown'), ('2', 't', 'light', '{}', 'succeeded')])
        db.commit()
        db.close()
        self.assertEqual(self.data.overview(self.fixture.brain)['attention']['unknown_operations']['count'], 1)

    def test_overview_process_status_drops_per_setting_values(self):
        write_status(self.data.voice_status, snapshot(check_settings(VOICE, ENV), started_at=NOW - 60, now=NOW,
                                                      input='asr', connection='connected'))
        voice = self.data.overview(self.fixture.brain)['processes'][VOICE]
        self.assertEqual((voice['state'], voice['input'], voice['connection']), ('running', 'asr', 'connected'))
        self.assertNotIn('settings', voice)


class SettingsViewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.addCleanup(self.fixture.cleanup)
        self.data = self.fixture.data

    def items(self, brain=None):
        view = self.data.settings(brain or self.fixture.brain)
        return view, {row['name']: row for row in view['items']}

    def test_saved_values_sources_and_metadata(self):
        self.fixture.env_file.write_text(env_text({'QWEN_TTS_VOICE': 'Cherry', 'QWEN_SILENCE_MS': '50',
                                                   'OMNI_API_KEY': 'old'}))
        view, items = self.items()
        self.assertEqual((items['QWEN_TTS_VOICE']['saved'], items['QWEN_TTS_VOICE']['source']), ('Cherry', 'env'))
        self.assertEqual((items['QWEN_REGION']['saved'], items['QWEN_REGION']['source']), ('singapore', 'default'))
        self.assertIn('QWEN_SILENCE_MS', items['QWEN_SILENCE_MS']['invalid'])
        self.assertIs(items['HA_TOKEN']['saved'], True)
        self.assertIsNone(items['HA_TOKEN']['default'])
        self.assertFalse(items['HA_URL']['editable'])
        self.assertNotIn('INTERRUPT_MS', items)                      # 旧路径不出现在页面上
        self.assertEqual(view['unrecognized_keys'], ['OMNI_API_KEY'])
        self.assertTrue(any('QWEN_SILENCE_MS' in e for e in view['saved_problems'][VOICE]['errors']))
        self.assertEqual(view['saved_problems'][BRAIN]['errors'], [])

    def test_states_per_process(self):
        # 语音按旧的 .env 启动，之后 .env 里改了音色。
        write_status(self.data.voice_status, snapshot(check_settings(VOICE, ENV), started_at=NOW - 60, now=NOW))
        self.fixture.env_file.write_text(env_text({'QWEN_TTS_VOICE': 'Cherry'}))
        view, items = self.items()
        voice_row = items['QWEN_TTS_VOICE']
        self.assertEqual(voice_row['state'], 'pending_restart')
        self.assertEqual(voice_row['processes'], [{'process': VOICE, 'state': 'pending_restart', 'effective': 'Ethan'}])
        self.assertEqual(items['QWEN_REGION']['state'], 'applied')
        self.assertEqual(items['ANTHROPIC_MODEL']['state'], 'unknown')  # 大脑连不上：不知道它在用什么
        self.assertEqual(items['ALPHA_VOICE']['state'], 'next_start')   # 只有按需运行的 CLI 读取
        # 两个进程都读的参数：一个已生效、一个不知道 → 不知道。
        self.assertEqual({row['process']: row['state'] for row in items['JEV_INPUT_USD_PER_MTOK']['processes']},
                         {BRAIN: 'unknown', VOICE: 'applied'})
        self.assertEqual(items['JEV_INPUT_USD_PER_MTOK']['state'], 'unknown')
        self.assertNotEqual(view['processes'][VOICE]['saved_fingerprint'],
                            view['processes'][VOICE]['running_fingerprint'])

    def test_brain_url_comes_from_saved_env(self):
        self.fixture.env_file.write_text(env_text({'ALPHA_BRAIN_URL': 'http://127.0.0.1:9999'}))
        self.fixture.client().get('/api/overview')
        self.assertEqual(self.fixture.brain_url, 'http://127.0.0.1:9999')


class BrainStatusTests(unittest.TestCase):
    def test_unreachable_brain_is_not_reported_as_stopped(self):
        from apps.console import fetch_brain_status
        status = asyncio.run(fetch_brain_status('http://127.0.0.1:9'))
        self.assertEqual(status.state, 'unreachable')


if __name__ == '__main__':
    unittest.main()
