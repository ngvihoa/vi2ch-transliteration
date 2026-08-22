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
            evidence, _ = MODULE.parse_cvdict(source)
            self.assertIn("一", evidence["một"])
            self.assertNotIn("殁", evidence["một"])

    def test_vocab_is_one_to_many_and_curated_candidate_is_first(self):
        evidence = {"một": Counter({"壹": 2, "一": 1})}
        vocab, _ = MODULE.build_vocab(evidence, {"một": ["一"]})
        self.assertEqual(["一", "壹"], vocab["một"])


if __name__ == "__main__":
    unittest.main()
