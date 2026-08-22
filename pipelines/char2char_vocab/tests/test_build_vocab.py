import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_vocab.py"
SPEC = importlib.util.spec_from_file_location("build_vocab", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildVocabTests(unittest.TestCase):
    def test_only_standalone_glosses_are_extracted(self):
        self.assertEqual(
            ["một", "đơn"],
            MODULE.lexical_glosses("một/đơn/một cái (mạo từ)/toàn bộ; tất cả"),
        )

    def test_parser_maps_meaning_not_han_viet_sound(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dict.u8"
            source.write_text(
                "一 一 [yi1] /một/đơn/\n"
                "歿 殁 [mo4] /kết thúc/chết/\n",
                encoding="utf-8",
            )
            evidence, _, _ = MODULE.parse_cvdict(source)
            self.assertIn("一", evidence["một"])
            self.assertNotIn("殁", evidence["một"])

    def test_vocab_is_one_to_many_and_curated_candidate_is_first(self):
        sources = {
            "cvdict_standalone_gloss": {"một": Counter({"壹": 2, "一": 1})},
            "kvietnamese_hanzi_reading": {"một": Counter({"沒": 1})},
            "hanviet_reading": {"một": Counter({"歿": 1})},
        }
        vocab, _ = MODULE.build_vocab(sources, {"một": ["一"]})
        self.assertEqual(["一", "壹"], vocab["một"])

    def test_reading_evidence_does_not_rank_or_create_candidates(self):
        sources = {
            "cvdict_standalone_gloss": {"từ": Counter({"文": 1, "字": 1})},
            "kvietnamese_hanzi_reading": {"từ": Counter({"文": 1, "辭": 1})},
            "hanviet_reading": {"rốt": Counter({"卒": 1})},
        }
        vocab, rows = MODULE.build_vocab(sources, {})
        self.assertEqual(["字", "文"], vocab["từ"])
        self.assertEqual(["𠀗"], vocab["rốt"])
        rốt = next(row for row in rows if row["token"] == "rốt")
        self.assertEqual(["卒"], [item["char"] for item in rốt["filtered_phonetic_candidates"]])
        self.assertTrue(rốt["uses_placeholder"])

    def test_kvietnamese_excludes_nom_only_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "kvietnamese.json"
            source.write_text(
                '{"昆": ["con"], "𡥵": ["con"], "𡳝": ["rốt"]}', encoding="utf-8"
            )
            evidence, stats = MODULE.parse_kvietnamese(source, {"昆"})
            self.assertEqual(Counter({"昆": 1}), evidence["con"])
            self.assertEqual(Counter(), evidence["rốt"])
            self.assertEqual(2, stats["nom_or_unverified_chars_excluded"])


if __name__ == "__main__":
    unittest.main()
