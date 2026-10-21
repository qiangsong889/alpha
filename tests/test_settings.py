"""参数清单：默认值、类型校验、参数之间的关系、按进程检查、指纹、未识别参数。不读取真实 .env。"""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import brain.agent as agent
from core.settings import (BRAIN, LEGACY, PERSONAL, REGISTRY, SETTINGS, VOICE, SettingsError,
                           _convert, check_settings, setting, unrecognized_keys)

BASE = {
    'HA_URL': 'http://ha.local:8123', 'HA_TOKEN': 'ha-secret', 'ANTHROPIC_API_KEY': 'sk-ant',
    'DASHSCOPE_API_KEY': 'ds', 'QWEN_WORKSPACE_ID': 'ws-1', 'TYPESAFE_API_KEY': 'ts',
}


class RegistryTests(unittest.TestCase):
    def test_every_default_passes_its_own_validation(self):
        for item in SETTINGS:
            with self.subTest(name=item.name):
                if item.default is None or item.default == '':
                    continue
                raw = ('1' if item.default else '0') if item.kind == 'bool' else str(item.default)
                self.assertEqual(_convert(item, raw), item.default)

    def test_readers_are_known_and_secrets_are_never_editable(self):
        for item in SETTINGS:
            self.assertTrue(item.readers, item.name)
            self.assertLessEqual(item.readers, {BRAIN, VOICE, PERSONAL, 'cli', LEGACY}, item.name)
            if item.secret:
                self.assertFalse(item.editable, item.name)

    def test_defaults_match_previous_hard_coded_values(self):
        # 迁移前散落在代码里的默认值；这一步只是换地方存放，不能改变行为。
        expected = {
            'ANTHROPIC_MODEL': 'claude-sonnet-5', 'REMINDER_TIMEZONE': 'America/Los_Angeles',
            'ALPHA_AUTO_MEMORY': False, 'ALPHA_CONTEXT_IDLE_MIN': 30.0,
            'ALPHA_CONTEXT_COMPACT_TOKENS': 15000, 'ALPHA_CONTEXT_HARD_TOKENS': 40000,
            'QWEN_ALPHA_INPUT': 'asr', 'QWEN_ALPHA_SPEAKER': 'tts', 'QWEN_REGION': 'singapore',
            'QWEN_TTS_VOICE': 'Ethan', 'QWEN_VOICE': 'Tina', 'QWEN_ALPHA_SUBMIT_MODE': 'jev',
            'QWEN_ALPHA_COLLECTION_SECONDS': 2.0, 'QWEN_ALPHA_CONTINUATION_SECONDS': 5.0,
            'QWEN_ALPHA_JEV_INTERRUPTIONS': True, 'QWEN_ALPHA_UNCERTAIN_INTERRUPTION': 'hold',
            'JEV_PROVIDER': 'typesafe', 'JEV_TIMEOUT_MS': 300.0, 'JEV_MIN_CONFIDENCE': 0.8,
            'QWEN_SESSION_SECONDS': 1800.0, 'ALPHA_BRAIN_URL': 'http://127.0.0.1:8765',
            'QWEN_VAD': 'server_vad', 'QWEN_SILENCE_MS': 800, 'QWEN_ALPHA_LOCAL_VAD': False,
            'QWEN_ALPHA_VAD_PRE_MS': 400, 'QWEN_ALPHA_VAD_MODE': 3, 'QWEN_ALPHA_VAD_ONSET_MS': 80,
            'QWEN_ALPHA_VAD_MIN_RMS': 0.003, 'QWEN_ALPHA_VAD_DEBUG': False,
            'JEV_INPUT_USD_PER_MTOK': 0.042, 'ALPHA_VOICE': 'Meijia',
        }
        for name, value in expected.items():
            self.assertEqual(setting(name, {}), value, name)


class ReadTests(unittest.TestCase):
    def test_unset_and_blank_use_default_and_values_are_typed(self):
        self.assertEqual(setting('ALPHA_CONTEXT_HARD_TOKENS', {'ALPHA_CONTEXT_HARD_TOKENS': '  '}), 40000)
        self.assertEqual(setting('ALPHA_CONTEXT_HARD_TOKENS', {'ALPHA_CONTEXT_HARD_TOKENS': '50000'}), 50000)
        self.assertIs(setting('ALPHA_AUTO_MEMORY', {'ALPHA_AUTO_MEMORY': ' 1 '}), True)
        self.assertEqual(setting('QWEN_ALPHA_INPUT', {'QWEN_ALPHA_INPUT': 'ASR'}), 'asr')

    def test_invalid_values_name_the_setting_and_reason(self):
        cases = {
            'ALPHA_AUTO_MEMORY': 'true',          # 以前会被悄悄当成 False
            'QWEN_ALPHA_INPUT': 'asrr',
            'ALPHA_CONTEXT_HARD_TOKENS': '1.5',
            'QWEN_SILENCE_MS': '100',
            'JEV_MIN_CONFIDENCE': 'nan',
            'REMINDER_TIMEZONE': 'Mars/Base',
        }
        for name, raw in cases.items():
            with self.subTest(name=name), self.assertRaises(SettingsError) as error:
                setting(name, {name: raw})
            self.assertIn(name, str(error.exception))

    def test_unregistered_name_is_a_programming_error(self):
        with self.assertRaises(KeyError):
            setting('NOT_A_SETTING', {})


class CheckTests(unittest.TestCase):
    def test_valid_configuration_has_no_errors_for_either_process(self):
        for process in (BRAIN, VOICE):
            report = check_settings(process, BASE)
            self.assertEqual(report.errors, [], process)

    def test_errors_only_for_settings_the_process_reads(self):
        env = {**BASE, 'QWEN_ALPHA_INPUT': 'bad', 'ALPHA_CONTEXT_HARD_TOKENS': 'bad'}
        brain, voice = check_settings(BRAIN, env), check_settings(VOICE, env)
        self.assertTrue(any('ALPHA_CONTEXT_HARD_TOKENS' in e for e in brain.errors))
        self.assertFalse(any('QWEN_ALPHA_INPUT' in e for e in brain.errors))
        self.assertTrue(any('QWEN_ALPHA_INPUT' in e for e in voice.errors))
        self.assertFalse(any('ALPHA_CONTEXT_HARD_TOKENS' in e for e in voice.errors))

    def test_required_credentials_per_process(self):
        brain = check_settings(BRAIN, {**BASE, 'HA_TOKEN': '', 'ANTHROPIC_API_KEY': ''})
        self.assertEqual(sum('必需' in e for e in brain.errors), 2)
        voice = check_settings(VOICE, {**BASE, 'DASHSCOPE_API_KEY': ''})
        self.assertTrue(any('DASHSCOPE_API_KEY' in e for e in voice.errors))
        self.assertEqual(check_settings(VOICE, {**BASE, 'HA_TOKEN': ''}).errors, [])

    def test_relations_between_settings(self):
        limits = check_settings(BRAIN, {**BASE, 'ALPHA_CONTEXT_COMPACT_TOKENS': '50000'})
        self.assertTrue(any('必须小于' in e for e in limits.errors))
        gate = check_settings(VOICE, {**BASE, 'QWEN_ALPHA_INPUT': 'omni', 'QWEN_ALPHA_LOCAL_VAD': '1',
                                      'QWEN_VAD': 'semantic_vad'})
        self.assertTrue(any('QWEN_VAD=server_vad' in e for e in gate.errors))

    def test_warnings_do_not_block_startup(self):
        voice = check_settings(VOICE, {**BASE, 'JEV_PROVIDER': 'vercel', 'QWEN_ALPHA_SPEAKER': 'omni'})
        self.assertEqual(voice.errors, [])
        self.assertTrue(any('AI_GATEWAY_API_KEY' in w for w in voice.warnings))
        self.assertTrue(any('改写原文' in w for w in voice.warnings))
        brain = check_settings(BRAIN, {**BASE, 'ANTHROPIC_MODEL': 'claude-future-9'})
        self.assertTrue(any('价格表' in w for w in brain.warnings))

    def test_fingerprint_is_per_process_and_never_contains_secret_values(self):
        base_brain, base_voice = check_settings(BRAIN, BASE), check_settings(VOICE, BASE)
        voice_change = {**BASE, 'QWEN_TTS_VOICE': 'Cherry'}
        self.assertEqual(check_settings(BRAIN, voice_change).fingerprint, base_brain.fingerprint)
        self.assertNotEqual(check_settings(VOICE, voice_change).fingerprint, base_voice.fingerprint)
        rotated = {**BASE, 'HA_TOKEN': 'another-secret'}
        self.assertEqual(check_settings(BRAIN, rotated).fingerprint, base_brain.fingerprint)
        self.assertNotEqual(check_settings(BRAIN, {**BASE, 'HA_TOKEN': ''}).fingerprint, base_brain.fingerprint)

    def test_public_values_hide_secrets(self):
        public = check_settings(BRAIN, BASE).public_values()
        self.assertIs(public['HA_TOKEN'], True)
        self.assertNotIn('ha-secret', repr(public))


class UnrecognizedTests(unittest.TestCase):
    def test_reports_env_keys_missing_from_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('HA_URL=http://x\nOMNI_API_KEY=old\n# 注释\nGOOGLE_TTS_API_KEY=old\n')
            self.assertEqual(unrecognized_keys(path), ['GOOGLE_TTS_API_KEY', 'OMNI_API_KEY'])
        self.assertIn('INTERRUPT_MS', REGISTRY)  # 旧路径登记了，不会被误报


class BrainStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_brain_lists_all_its_problems_and_ignores_voice_only_mistakes(self):
        env = {**BASE, 'ALPHA_CONTEXT_HARD_TOKENS': 'lots', 'ANTHROPIC_API_KEY': '',
               'QWEN_ALPHA_INPUT': 'bad'}
        with patch.dict(agent.os.environ, env, clear=True), patch.object(agent, 'load_dotenv'):
            with self.assertRaises(SystemExit) as error:
                await agent.AlphaBrain().start()
        message = str(error.exception)
        self.assertIn('ALPHA_CONTEXT_HARD_TOKENS', message)
        self.assertIn('ANTHROPIC_API_KEY', message)   # 一次列出全部问题
        self.assertNotIn('QWEN_ALPHA_INPUT', message)  # 语音专用的参数由语音自己检查
        self.assertNotIn('必须小于', message)           # 用到了不合法参数的关系规则不下结论


if __name__ == '__main__':
    unittest.main()
