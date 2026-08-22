import tempfile
import unittest
from pathlib import Path

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import phonemize_poems as module


class PhonemizePoemsTests(unittest.TestCase):
    def test_parse_chao_transcription(self):
        self.assertEqual(module.parse_transcription("tɕam33"), ("tɕam", "33", "ok"))
        self.assertEqual(module.parse_transcription("ŋa3g5"), ("ŋa", "3g5", "ok"))
        self.assertEqual(module.parse_transcription("[linux]"), (None, None, "unrecognized"))

    def test_build_rows_preserves_metadata_and_marks_partial_lines(self):
        source = [
            {
                "line_id": "sample_0001",
                "include_in_benchmark": True,
                "syllables": ["trăm", "linux"],
            },
            {
                "line_id": "sample_0002",
                "include_in_benchmark": False,
                "syllables": ["title"],
            },
        ]
        rows = module.build_ipa_rows(
            source,
            {"trăm": "tɕam33", "linux": "[linux]"},
            {"name": "fake"},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["schema_version"], "poem-ipa-v1")
        self.assertEqual(rows[0]["phonemization_status"], "partial")
        self.assertEqual(rows[0]["ipa"], "tɕam33 <?>")
        self.assertEqual(rows[0]["ipa_syllables"][0]["tone_chao"], "33")

    def test_pronunciation_override_is_auditable(self):
        source = [
            {
                "line_id": "sample_0001",
                "include_in_benchmark": True,
                "syllables": ["qùi"],
            }
        ]
        rows = module.build_ipa_rows(
            source,
            {"quỳ": "kwi32"},
            {"name": "fake"},
            {"qùi": "quỳ"},
        )
        item = rows[0]["ipa_syllables"][0]
        self.assertEqual(item["phonemized_as"], "quỳ")
        self.assertEqual(item["normalization"], "override")
        self.assertEqual(item["ipa"], "kwi32")

    def test_run_vphon_batches_unique_syllables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "vPhon.py"
            fake.write_text(
                "import sys\nfor line in sys.stdin:\n print('x33' if line.strip() else '')\n",
                encoding="utf-8",
            )
            (root / "rules.py").write_text("", encoding="utf-8")
            result = module.run_vphon(["b", "a", "b"], fake, "n")
            self.assertEqual(result, {"a": "x33", "b": "x33"})

    def test_report_lists_failed_syllables(self):
        source = [{"include_in_benchmark": True}]
        rows = [
            {
                "phonemization_status": "partial",
                "ipa_syllables": [
                    {
                        "text": "trăm",
                        "phonemized_as": "trăm",
                        "normalization": "identity",
                        "status": "ok",
                    },
                    {
                        "text": "linux",
                        "phonemized_as": "linux",
                        "normalization": "identity",
                        "status": "unrecognized",
                    },
                ],
            }
        ]
        report = module.build_report(source, rows, {"name": "fake"})
        self.assertEqual(report["partial_lines"], 1)
        self.assertEqual(report["syllable_status_counts"], {"ok": 1, "unrecognized": 1})
        self.assertEqual(report["failed_syllables"], [{"text": "linux", "occurrences": 1}])


if __name__ == "__main__":
    unittest.main()
