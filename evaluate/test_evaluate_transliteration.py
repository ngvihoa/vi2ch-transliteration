import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate_transliteration.py")
SPEC = importlib.util.spec_from_file_location("evaluate_transliteration", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MetricHelpersTest(unittest.TestCase):
    def test_edit_counts(self):
        counts = MODULE.edit_counts("天地人", "天大仁人")
        self.assertEqual(counts.distance, 2)
        self.assertEqual(counts.substitutions, 1)
        self.assertEqual(counts.insertions, 1)
        self.assertEqual(counts.deletions, 0)

    def test_edit_counts_deletion(self):
        counts = MODULE.edit_counts("天地人", "天人")
        self.assertEqual(counts.distance, 1)
        self.assertEqual(counts.deletions, 1)

    def test_news_fscore(self):
        self.assertEqual(MODULE.news_fscore("天地", "天地"), 1.0)
        self.assertEqual(MODULE.news_fscore("天地", "人和"), 0.0)
        self.assertAlmostEqual(MODULE.news_fscore("天地人", "天人"), 0.8)

    def test_normalization(self):
        self.assertEqual(MODULE.normalize_han(" 天 地，\n"), "天地，")
        self.assertEqual(
            MODULE.normalize_han(" 天 地，\n", ignore_punctuation=True), "天地"
        )


if __name__ == "__main__":
    unittest.main()
