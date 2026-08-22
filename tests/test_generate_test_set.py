import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_test-set.py"
SPEC = importlib.util.spec_from_file_location("generate_test_set", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GenerateTestSetTests(unittest.TestCase):
    def test_luc_bat_classification(self):
        result = MODULE.classify_lines(
            "luc_bat",
            ["một hai ba bốn năm sáu", "một hai ba bốn năm sáu bảy tám", "quá ngắn"],
        )
        self.assertEqual([item[0] for item in result], ["luc", "bat", "irregular"])
        self.assertFalse(result[-1][2])

    def test_embedded_seven_syllable_poem_uses_context(self):
        lines = [
            "thi rằng",
            "một hai ba bốn năm sáu bảy",
            "một hai ba bốn năm sáu bảy",
            "một hai ba bốn năm sáu bảy",
            "một hai ba bốn năm sáu bảy",
            "một hai ba bốn năm sáu",
            "một hai ba bốn năm sáu bảy",
        ]
        result = MODULE.classify_lines("luc_bat_mixed", lines)
        self.assertEqual(result[0][0], "section_marker")
        self.assertTrue(all(item[0] == "embedded_that_ngon" for item in result[1:5]))
        self.assertEqual(result[-1][1], "needs_review")

    def test_heading_candidate_is_excluded(self):
        result = MODULE.classify_lines(
            "that_ngon_mixed",
            ["du cổ tự", "một hai ba bốn năm sáu bảy"],
        )
        self.assertEqual(result[0][0], "heading_candidate")
        self.assertFalse(result[0][2])
        self.assertTrue(result[1][2])

    def test_write_outputs_produces_valid_jsonl_and_report(self):
        manifest = [{"file": "sample.txt", "work": "Sample", "form": "luc_bat"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            (raw / "sample.txt").write_text(
                "Một hai ba bốn năm sáu\nMột hai ba bốn năm sáu bảy tám\n",
                encoding="utf-8",
            )
            records = MODULE.build_records(manifest, raw)
            lines_path, review_path, report_path = MODULE.write_outputs(records, root / "out")

            lines = [json.loads(line) for line in lines_path.read_text(encoding="utf-8").splitlines()]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["text"], "một hai ba bốn năm sáu")
            self.assertEqual(report["benchmark_lines"], 2)
            self.assertEqual(review_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
