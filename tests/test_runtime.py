"""运行状态：心跳文件、过期判断、已保存与正在生效的比较、大脑的 /status。不启动真实进程。"""

import asyncio
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import brain.agent as agent
from brain import server
from core.runtime import (STALE_AFTER_S, AlreadyRunning, single_instance, StatusWriter, compare_settings, effective_settings,
                          read_status_file, saved_report, snapshot, write_status)
from core.settings import BRAIN, VOICE, check_settings

BASE = {
    'HA_URL': 'http://ha.local:8123', 'HA_TOKEN': 'ha-secret', 'ANTHROPIC_API_KEY': 'sk-ant',
    'DASHSCOPE_API_KEY': 'ds-secret', 'QWEN_WORKSPACE_ID': 'ws-1', 'TYPESAFE_API_KEY': 'ts-secret',
}


class StatusFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'run' / 'voice.json'
        self.report = check_settings(VOICE, BASE)

    def write(self, *, state='running', heartbeat=1000.0):
        write_status(self.path, snapshot(self.report, started_at=900.0, state=state, now=heartbeat,
                                         connection='connected'))

    def test_states_allow_not_knowing(self):
        self.assertEqual(read_status_file(self.path, now=1000).state, 'not_started')
        self.write()
        running = read_status_file(self.path, now=1001)
        self.assertEqual((running.state, running.data['connection']), ('running', 'connected'))
        self.assertEqual(read_status_file(self.path, now=1000 + STALE_AFTER_S + 1).state, 'stale')
        self.write(state='stopped')
        self.assertEqual(read_status_file(self.path, now=5000).state, 'stopped')
        self.path.write_text('{half')
        self.assertEqual(read_status_file(self.path, now=1000).state, 'unreadable')
        self.path.write_text(json.dumps({'schema': 99}))
        self.assertEqual(read_status_file(self.path, now=1000).state, 'unreadable')

    def test_write_is_atomic_private_and_contains_no_secret_values(self):
        self.write()
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ['voice.json'])  # 没有遗留临时文件
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        text = self.path.read_text()
        for secret in ('ds-secret', 'ts-secret', 'ha-secret'):
            self.assertNotIn(secret, text)
        data = json.loads(text)
        self.assertIs(data['settings']['DASHSCOPE_API_KEY'], True)

    def test_effective_settings_only_include_what_the_process_reads(self):
        voice = effective_settings(self.report)
        self.assertIn('QWEN_ALPHA_INPUT', voice)
        self.assertNotIn('ALPHA_CONTEXT_HARD_TOKENS', voice)
        self.assertNotIn('INTERRUPT_MS', voice)  # 旧路径


class StatusWriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_reflects_runtime_changes_and_writes_stopped_on_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'voice.json'
            runtime = {'connection': 'starting'}
            writer = StatusWriter(path, check_settings(VOICE, BASE), lambda: runtime, interval_s=0.01)
            task = asyncio.create_task(writer.run())
            await asyncio.sleep(0.02)
            self.assertEqual(json.loads(path.read_text())['connection'], 'starting')
            runtime['connection'] = 'reconnecting'
            await asyncio.sleep(0.03)
            self.assertEqual(json.loads(path.read_text())['connection'], 'reconnecting')
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(json.loads(path.read_text())['state'], 'stopped')


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env_file = Path(self.directory.name) / '.env'

    def running(self, env):
        data = snapshot(check_settings(VOICE, env), started_at=1, now=10)
        return read_status_file(self._write(data), now=11)

    def _write(self, data):
        path = Path(self.directory.name) / 'voice.json'
        write_status(path, data)
        return path

    def test_saved_comes_from_the_file_not_the_readers_environment(self):
        self.env_file.write_text('QWEN_TTS_VOICE=Cherry\n')
        with patch.dict(os.environ, {'QWEN_TTS_VOICE': 'Ethan'}):
            self.assertEqual(saved_report(VOICE, self.env_file).values['QWEN_TTS_VOICE'], 'Cherry')

    def test_applied_pending_restart_and_unknown(self):
        self.env_file.write_text(''.join(f'{k}={v}\n' for k, v in BASE.items()) + 'QWEN_TTS_VOICE=Cherry\n')
        saved = saved_report(VOICE, self.env_file)
        rows = {row.name: row for row in compare_settings(saved, self.running(BASE))}
        self.assertEqual(rows['QWEN_TTS_VOICE'].state, 'pending_restart')
        self.assertEqual((rows['QWEN_TTS_VOICE'].saved, rows['QWEN_TTS_VOICE'].effective), ('Cherry', 'Ethan'))
        self.assertEqual(rows['QWEN_ALPHA_COLLECTION_SECONDS'].state, 'applied')  # 2.0 和 JSON 里的 2.0
        stopped = read_status_file(Path(self.directory.name) / 'missing.json', now=1)
        self.assertTrue(all(row.state == 'unknown' for row in compare_settings(saved, stopped)))


class SingleInstanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.lock = Path(self.directory.name) / 'run' / 'voice.lock'

    def test_second_holder_is_refused_with_the_owner_pid_and_lock_is_released_on_exit(self):
        with single_instance(self.lock):
            with self.assertRaises(AlreadyRunning) as error:
                with single_instance(self.lock):
                    self.fail('不应该拿到锁')
            self.assertEqual(str(error.exception), str(os.getpid()))
        with single_instance(self.lock):  # 正常退出后可以再拿到
            pass

    def test_lock_is_released_when_the_holder_is_killed(self):
        # 被 kill -9 的进程来不及做任何清理，锁也必须自动释放。
        holder = subprocess.Popen([sys.executable, '-c', (
            'import sys, time; from pathlib import Path; from core.runtime import single_instance\n'
            'with single_instance(Path(sys.argv[1])):\n'
            '    print("held", flush=True); time.sleep(60)'), str(self.lock)],
            stdout=subprocess.PIPE, text=True, cwd=Path(__file__).resolve().parent.parent)
        self.addCleanup(holder.kill)
        self.assertEqual(holder.stdout.readline().strip(), 'held')
        with self.assertRaises(AlreadyRunning) as error:
            with single_instance(self.lock):
                pass
        self.assertEqual(str(error.exception), str(holder.pid))
        holder.kill()
        holder.wait(5)
        with single_instance(self.lock):
            pass


class BrainStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_before_start_and_after_start(self):
        brain = agent.AlphaBrain()
        self.assertEqual(brain.status(), {'process': BRAIN, 'state': 'starting'})
        brain._started = True
        brain.settings_report = check_settings(BRAIN, BASE)
        brain.started_at, brain.catalog_checked_at, brain.model = 100.0, 120.0, 'claude-haiku-5-5'
        brain.ha_catalog, brain.personal_tools = None, [{'name': 'list_items'}]
        status = brain.status()
        self.assertEqual(status['state'], 'running')
        self.assertEqual(status['settings_fingerprint'], brain.settings_report.fingerprint)
        self.assertEqual(status['components']['home_assistant'], {'state': 'unavailable', 'checked_at': 120.0})
        self.assertEqual(status['components']['personal']['state'], 'available')
        self.assertEqual(status['components']['archive']['state'], 'disabled')
        self.assertNotIn('ha-secret', json.dumps(status))
        self.assertNotIn('QWEN_ALPHA_INPUT', status['settings'])  # 只有大脑会读的参数

    async def test_status_endpoint_returns_brain_snapshot(self):
        with patch.object(server, 'brain', type('B', (), {'status': lambda self: {'state': 'running'}})()):
            self.assertEqual(await server.status(), {'state': 'running'})


if __name__ == '__main__':
    unittest.main()
