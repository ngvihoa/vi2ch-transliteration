import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "map_raw_poems.py"
SPEC = importlib.util.spec_from_file_location("map_raw_poems", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MapRawPoemsTests(unittest.TestCase):
    def test_tokenize_normalizes_case_and_ignores_punctuation(self):
        self.assertEqual(["một", "trai"], MODULE.tokenize("  Một, trai! "))

    def test_mapping_selects_top_candidate_and_preserves_slots(self):
        chars, mappings = MODULE.map_tokens(
            ["một", "trai", "lạ"],
            {"một": ["一", "壹"], "trai": ["𠀗"]},
        )
        self.assertEqual(["一", "𠀗", "𠀗"], chars)
        self.assertEqual(3, len(mappings))
        self.assertEqual("placeholder_in_vocab", mappings[1]["method"])
        self.assertEqual("placeholder_oov", mappings[2]["method"])

    def test_reads_raw_directory_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.vi.txt"
            source.write_text("một trai\n\n", encoding="utf-8")
            rows = MODULE.map_poem_directory(Path(directory), {"một": ["一"], "trai": ["𠀗"]})
            self.assertEqual(1, len(rows))
            self.assertEqual(["一", "𠀗"], rows[0]["target_chars"])
            self.assertEqual(2, rows[0]["target_count"])
            self.assertTrue(rows[0]["length_preserved"])


if __name__ == "__main__":
    unittest.main()
