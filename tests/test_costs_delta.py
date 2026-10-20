from unittest import TestCase

from core.costs import CumulativeUsage


class CumulativeUsageTests(TestCase):
    def test_duration_differences(self):
        meter = CumulativeUsage()
        self.assertEqual([meter.delta({'duration': n})['duration'] for n in (41, 43, 45)], [41, 2, 2])

    def test_explicit_reset_starts_from_zero(self):
        meter = CumulativeUsage()
        meter.delta({'duration': 45})
        meter.reset()
        self.assertEqual(meter.delta({'duration': 5}), {'duration': 5})

    def test_counter_reset_without_reconnect_warns(self):
        meter = CumulativeUsage()
        meter.delta({'duration': 45})
        with self.assertLogs('alpha.costs', level='WARNING') as logs:
            self.assertEqual(meter.delta({'duration': 5}), {'duration': 5})
        self.assertEqual(len(logs.output), 1)
        self.assertIn('duration', logs.output[0])

    def test_nested_fields(self):
        meter = CumulativeUsage()
        meter.delta({'output_tokens_details': {'audio_tokens': 734}, 'input_tokens_details': {'text_tokens': 2}})
        self.assertEqual(meter.delta({'output_tokens_details': {'audio_tokens': 742},
                                      'input_tokens_details': {'text_tokens': 3}}),
                         {'output_tokens_details': {'audio_tokens': 8}, 'input_tokens_details': {'text_tokens': 1}})

    def test_non_numeric_fields_and_booleans_are_preserved(self):
        meter = CumulativeUsage()
        value = {'unit': 'seconds', 'complete': True, 'other': None, 'tags': ['asr'], 'duration': 1.5}
        meter.delta(value)
        self.assertEqual(meter.delta({**value, 'duration': 2.25}), {**value, 'duration': .75})

    def test_missing_fields_do_not_destroy_baseline_and_new_fields_are_raw(self):
        meter = CumulativeUsage()
        meter.delta({'duration': 41, 'details': {'a': 3, 'b': 7}})
        self.assertEqual(meter.delta({}), {})
        self.assertEqual(meter.delta({'details': {'a': 4}}), {'details': {'a': 1}})
        self.assertEqual(meter.delta({'duration': 43, 'details': {'b': 9, 'new': 2}}),
                         {'duration': 2, 'details': {'b': 2, 'new': 2}})

    def test_input_and_returned_values_do_not_mutate_baseline(self):
        meter = CumulativeUsage()
        original = {'details': {'tokens': 5}}
        result = meter.delta(original)
        original['details']['tokens'] = 999
        result['details']['tokens'] = 777
        self.assertEqual(meter.delta({'details': {'tokens': 7}}), {'details': {'tokens': 2}})
