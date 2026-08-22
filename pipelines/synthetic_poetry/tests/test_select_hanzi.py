import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import select_hanzi as module


class SelectHanziTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "candidate_set_id": "ipa_n_example",
            "candidates": [
                {
                    "pinyin": "da2", "ipa": "ta", "score": 0.05,
                    "example_hanzi": ["达"],
                },
                {
                    "pinyin": "ta3", "ipa": "tʰa", "score": 0.08,
                    "example_hanzi": ["塔"],
                },
            ],
        }

    def test_reference_character_is_preferred_and_auditable(self):
        selected = module.select_candidates(
            self.row,
            {"da2": "达"},
            {"da2": {"达", "妲"}, "ta3": {"塔"}},
            Counter({"达": 100, "妲": 1, "塔": 50}),
            3,
        )
        self.assertEqual(selected[0]["char"], "达")
        self.assertEqual(selected[0]["provenance"], "xinhua_english_reference")
        self.assertFalse(selected[0]["requires_review"])

    def test_corpus_character_does_not_require_phonetic_review(self):
        selected = module.select_candidates(
            self.row,
            {},
            {"da2": {"达", "妲"}, "ta3": {"塔"}},
            Counter({"达": 100, "妲": 1, "塔": 50}),
            2,
        )
        self.assertEqual(selected[0]["provenance"], "corpus_frequency")
        self.assertFalse(selected[0]["requires_review"])

    def test_two_syllable_pinyin_produces_two_hanzi(self):
        row = {
            "candidate_set_id": "ipa_n_stop",
            "candidates": [{
                "pinyin": "mo4 te4", "pinyin_syllables": ["mo4", "te4"],
                "ipa": "mo tʰɤ", "ipa_syllables": ["mo", "tʰɤ"],
                "score": 0.1, "coda_strategy": "expanded_t",
            }],
        }
        selected = module.select_candidates(
            row,
            {"mo4": "莫", "te4": "特"},
            {"mo4": {"莫"}, "te4": {"特"}},
            Counter({"莫": 100, "特": 100}),
            2,
        )
        self.assertEqual(selected[0]["char"], "莫特")
        self.assertEqual(selected[0]["pinyin_syllables"], ["mo4", "te4"])
        self.assertEqual(selected[0]["mapping_length"], 2)

    def test_line_output_uses_top_candidate(self):
        pinyin_lines = [
            {
                "line_id": "sample_0001", "work": "Sample", "form": "luc_bat",
                "line_role": "luc", "text": "ta",
                "pinyin_syllables": [
                    {"text": "ta", "source_ipa": "ta33", "candidate_set_id": "ipa_n_example"}
                ],
            }
        ]
        hanzi_rows = [
            {
                "candidate_set_id": "ipa_n_example",
                "candidates": [
                    {
                        "char": "达", "pinyin": "da2", "selection_score": 0.1,
                        "provenance": "xinhua_english_reference", "requires_review": False,
                    }
                ],
            }
        ]
        lines = module.build_line_rows(pinyin_lines, hanzi_rows)
        self.assertEqual(lines[0]["hanzi"], "达")
        self.assertEqual(lines[0]["pinyin"], "da2")
        self.assertEqual(lines[0]["label_quality"], "synthetic_silver")
        self.assertFalse(lines[0]["requires_review"])


if __name__ == "__main__":
    unittest.main()
