import unittest
from eval.text_numbers import contains_equivalent, normalize_numbers
from eval.run_extraction import content_ok


class NumberComparisonTests(unittest.TestCase):
    def test_both_directions_and_boundaries(self):
        for a, b in [('二十六度', '26度'), ('二十四度', '24度'),
                     ('百分之三十', '30%'), ('负二点五度', '-2.5度'),
                     ('每周三', '每周3'), ('２６度', '26度')]:
            self.assertTrue(contains_equivalent(a, b), (a, b))
            self.assertTrue(contains_equivalent(b, a), (b, a))
        for text in ('126度', '26.5度', '二百二十六度', '负二十六度'):
            self.assertFalse(contains_equivalent(text, '二十六度'))
        self.assertEqual(normalize_numbers('二十六度'), '26度')

    def test_required_and_forbidden_use_same_normalization(self):
        self.assertTrue(content_ok({'content_contains': ['二十六度']}, {'content': '空调26度'}))
        self.assertFalse(content_ok({'content_contains': [], 'content_not_contains': ['二十六度']}, {'content': '空调26度'}))
