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

        onset, rhyme = module.split_vietnamese_ipa("tɕaːj")
        self.assertEqual(onset, ("tɕ",))
        self.assertEqual(rhyme, ("a", "i"))

    def test_feature_distance_prefers_similar_phones(self):
        self.assertEqual(module.phoneme_distance("a", "a"), 0.0)
        self.assertLess(module.phoneme_distance("a", "ə"), module.phoneme_distance("a", "i"))
        self.assertLess(module.phoneme_distance("tɕ", "tɕʰ"), module.phoneme_distance("tɕ", "m"))

    def test_predictable_labial_glide_does_not_penalize_mo(self):
        self.assertEqual(module.normalize_mandarin_rhyme(("m",), ("w", "o")), ("o",))

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

    def test_trai_prefers_zhai_after_glide_normalization(self):
        inventory = [
            {
                "pinyin": "zhai1", "base": "zhai", "tone_number": 1,
                "tone_chao": "55", "variants": [{"ipa": "ʈʂai", "onset": ("ʈʂ",), "rhyme": ("a", "i")}],
                "hanzi_count": 4, "example_hanzi": ["斋"],
            },
            {
                "pinyin": "zhan1", "base": "zhan", "tone_number": 1,
                "tone_chao": "55", "variants": [{"ipa": "ʈʂan", "onset": ("ʈʂ",), "rhyme": ("a", "n")}],
                "hanzi_count": 4, "example_hanzi": ["詹"],
            },
        ]
        candidates = module.rank_candidates("tɕaːj", "33", inventory, 2)
        self.assertEqual(candidates[0]["pinyin"], "zhai1")

    def test_stop_coda_can_expand_to_two_pinyin_syllables(self):
        inventory = [
            {
                "pinyin": "mo4", "base": "mo", "tone_number": 4,
                "tone_chao": "51", "variants": [{"ipa": "mo", "onset": ("m",), "rhyme": ("o",)}],
                "hanzi_count": 4, "example_hanzi": ["莫"],
            },
            {
                "pinyin": "nong3", "base": "nong", "tone_number": 3,
                "tone_chao": "214", "variants": [{"ipa": "nʊŋ", "onset": ("n",), "rhyme": ("ʊ", "ŋ")}],
                "hanzi_count": 2, "example_hanzi": ["侬"],
            },
            {
                "pinyin": "te4", "base": "te", "tone_number": 4,
                "tone_chao": "51", "variants": [{"ipa": "tʰɤ", "onset": ("tʰ",), "rhyme": ("ɤ",)}],
                "hanzi_count": 5, "example_hanzi": ["特"],
            },
        ]
        candidate = module.rank_candidates("mot", "21", inventory, 1)[0]
        self.assertEqual(candidate["pinyin_syllables"], ["mo4", "te4"])
        self.assertEqual(candidate["mapping_length"], 2)

    def test_sample_regressions_keep_rhyme_and_expose_weak_matches(self):
        inventory = [
            {
                "pinyin": "long2", "base": "long", "tone_number": 2,
                "tone_chao": "35", "variants": [{"ipa": "lʊŋ", "onset": ("l",), "rhyme": ("ʊ", "ŋ")}],
                "hanzi_count": 5, "example_hanzi": ["龙"],
            },
            {
                "pinyin": "di2", "base": "di", "tone_number": 2,
                "tone_chao": "35", "variants": [{"ipa": "ti", "onset": ("t",), "rhyme": ("i",)}],
                "hanzi_count": 5, "example_hanzi": ["迪"],
            },
            {
                "pinyin": "mo2", "base": "mo", "tone_number": 2,
                "tone_chao": "35", "variants": [{"ipa": "mo", "onset": ("m",), "rhyme": ("o",)}],
                "hanzi_count": 5, "example_hanzi": ["模"],
            },
            {
                "pinyin": "te4", "base": "te", "tone_number": 4,
                "tone_chao": "51", "variants": [{"ipa": "tʰɤ", "onset": ("tʰ",), "rhyme": ("ɤ",)}],
                "hanzi_count": 5, "example_hanzi": ["特"],
            },
        ]
        self.assertEqual(module.rank_candidates("lɔŋ", "32", inventory, 1)[0]["pinyin"], "long2")
        weak = module.rank_candidates("thɨ", "24", inventory, 1)[0]
        self.assertGreater(weak["score"], 0.35)
        expanded = module.rank_candidates("zot", "45", inventory, 1)[0]
        self.assertEqual(expanded["pinyin_syllables"][-1], "te4")
        self.assertEqual(expanded["mapping_length"], 2)

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
