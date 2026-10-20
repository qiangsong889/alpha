import unittest

import numpy as np

from voice.segmenter import FRAME, RATE, Segmenter, SegmentReady, SpeechStarted


def pcm(frames, voice=False):
    return np.full(frames * FRAME, 6000 if voice else 0, dtype='<i2').tobytes()


def detector(frame):
    return .9 if np.any(frame != 0) else .01


class SegmenterTests(unittest.TestCase):
    def ready(self, events):
        return [e for e in events if isinstance(e, SegmentReady)]

    def test_pause_splits_two_segments_with_correct_times_and_preroll(self):
        segmenter = Segmenter(detector)
        audio = pcm(20) + pcm(20, True) + pcm(32) + pcm(25, True) + pcm(25)
        events = segmenter.feed(audio) + segmenter.flush()
        started = [e for e in events if isinstance(e, SpeechStarted)]
        ready = self.ready(events)
        self.assertEqual(len(started), 2)
        self.assertEqual(len(ready), 2)
        self.assertEqual(len({e.seg_id for e in started}), 2)
        self.assertAlmostEqual(started[0].speech_start_s, .640)
        self.assertAlmostEqual(started[0].confirmed_s, .800)
        self.assertAlmostEqual(ready[0].start_s, .240)
        self.assertAlmostEqual(ready[0].end_s, 1.888)
        self.assertAlmostEqual(ready[0].end_s - ready[0].speech_end_s, .608)
        self.assertTrue(all(r.reason == 'silence' for r in ready))
        self.assertEqual([e.seg_id for e in started], [e.seg_id for e in ready])

    def test_noise_under_100ms_does_not_even_start(self):
        segmenter = Segmenter(detector)
        self.assertEqual(segmenter.feed(pcm(30) + pcm(3, True) + pcm(40)) + segmenter.flush(), [])

    def test_15_seconds_cuts_without_losing_or_duplicating_samples(self):
        segmenter = Segmenter(detector)
        audio = pcm(500, True)  # 16秒连续说话。
        events = segmenter.feed(audio) + segmenter.flush()
        ready = self.ready(events)
        self.assertEqual(len(ready), 2)
        self.assertEqual(ready[0].reason, 'max_duration')
        self.assertLessEqual(ready[0].end_s, 15 + FRAME / RATE)
        self.assertEqual(b''.join(r.pcm for r in ready), audio)
        starts = [e for e in events if isinstance(e, SpeechStarted)]
        self.assertFalse(starts[0].continuation)
        self.assertTrue(starts[1].continuation)

    def test_flush_and_fragmented_input_do_not_send_padding(self):
        audio = np.full(6407, 6000, dtype='<i2').tobytes()
        segmenter = Segmenter(detector)
        events = []
        for offset in range(0, len(audio), 74):
            events.extend(segmenter.feed(audio[offset:offset + 74]))
        events.extend(segmenter.flush())
        ready = self.ready(events)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].pcm, audio)
        self.assertEqual(ready[0].end_s, 6407 / RATE)

    def test_short_continuation_does_not_leave_an_unmatched_start(self):
        segmenter = Segmenter(detector)
        events = segmenter.feed(pcm(471, True)) + segmenter.flush()
        self.assertEqual(len([e for e in events if isinstance(e, SpeechStarted)]), 1)
        self.assertEqual(len(self.ready(events)), 1)

    def test_invalid_audio_probability_and_settings(self):
        with self.assertRaises(ValueError):
            Segmenter(detector).feed(b'x')
        with self.assertRaises(ValueError):
            Segmenter(lambda _: float('nan')).feed(pcm(1))
        for kwargs in ({'max_s': .01}, {'silence_ms': 0}, {'threshold': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Segmenter(detector, **kwargs)
