import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import map_ipa_to_pinyin as module


class MapIpaToPinyinTests(unittest.TestCase):
    def test_split_vietnamese_ipa(self):
        onset, rhyme = module.split_vietnamese_ipa("tɕam")
        self.assertEqual(onset, ("tɕ",))
        self.assertEqual(rhyme, ("a", "m"))

        onset, rhyme = module.split_vietnamese_ipa("kwi")
        self.assertEqual(onset, ("k", "w"))
        self.assertEqual(rhyme, ("i",))

    def test_feature_distance_prefers_similar_phones(self):
        self.assertEqual(module.phoneme_distance("a", "a"), 0.0)
        self.assertLess(module.phoneme_distance("a", "ə"), module.phoneme_distance("a", "i"))
        self.assertLess(module.phoneme_distance("tɕ", "tɕʰ"), module.phoneme_distance("tɕ", "m"))

    def test_tone_distance_uses_contour(self):
        self.assertLess(module.tone_distance("33", "35"), module.tone_distance("33", "51"))
        self.assertGreater(module.tone_distance("3g5", "35"), module.tone_distance("35", "35"))

    def test_rank_candidates_returns_explainable_components(self):
        inventory = [
            {
                "pinyin": "zhan1", "base": "zhan", "tone_number": 1,
                "tone_chao": "55", "variants": [{"ipa": "ʈʂan", "onset": ("ʈʂ",), "rhyme": ("a", "n")}],
                "hanzi_count": 3, "example_hanzi": ["詹"],
            },
            {
                "pinyin": "ma1", "base": "ma", "tone_number": 1,
                "tone_chao": "55", "variants": [{"ipa": "ma", "onset": ("m",), "rhyme": ("a",)}],
                "hanzi_count": 2, "example_hanzi": ["妈"],
            },
        ]
        candidates = module.rank_candidates("tɕan", "33", inventory, 1)
        self.assertEqual(candidates[0]["pinyin"], "zhan1")
        self.assertIn("onset_distance", candidates[0])
        self.assertIn("rhyme_distance", candidates[0])

    def test_build_outputs_deduplicates_pronunciations(self):
        row = {
            "line_id": "sample_0001", "work": "Sample", "form": "luc_bat",
            "line_role": "luc", "text": "ta ta", "phonemization": {"dialect": "n"},
            "ipa_syllables": [
                {"text": "ta", "ipa": "ta33", "ipa_segments": "ta", "tone_chao": "33"},
                {"text": "ta", "ipa": "ta33", "ipa_segments": "ta", "tone_chao": "33"},
            ],
        }
        inventory = [
            {
                "pinyin": "da1", "base": "da", "tone_number": 1,
                "tone_chao": "55", "variants": [{"ipa": "ta", "onset": ("t",), "rhyme": ("a",)}],
                "hanzi_count": 1, "example_hanzi": ["搭"],
            }
        ]
        candidate_rows, line_rows = module.build_outputs([row], inventory, 1)
        self.assertEqual(len(candidate_rows), 1)
        self.assertEqual(candidate_rows[0]["occurrences"], 2)
        self.assertEqual(
            line_rows[0]["pinyin_syllables"][0]["candidate_set_id"],
            line_rows[0]["pinyin_syllables"][1]["candidate_set_id"],
        )


if __name__ == "__main__":
    unittest.main()
