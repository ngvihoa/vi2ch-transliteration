import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import quality_gate_poems as module


def decoded_line(scores=(0.1, 0.2), fallback=False, rank=1):
    syllables = [
        {
            "char": char,
            "pinyin": pinyin,
            "selection_score": score,
            "candidate_rank": rank,
            "requires_review": fallback and index == 0,
            "changed_from_greedy": rank > 1,
        }
        for index, (char, pinyin, score) in enumerate(zip("达南", ("da2", "nan2"), scores))
    ]
    return {
        "schema_version": "poem-hanzi-decoded-v1",
        "line_id": "sample_0001",
        "work": "Sample",
        "form": "free_verse",
        "line_role": "free",
        "vi": "ta nam",
        "hanzi": "达南",
        "pinyin": "da2 nan2",
        "requires_review": fallback,
        "changed_from_greedy": rank > 1,
        "syllables": syllables,
    }


def source_line():
    return {
        "line_id": "sample_0001",
        "source_file": "sample.txt",
        "source_line_no": 1,
        "original_text": "ta nam",
        "text": "ta nam",
        "syllable_count": 2,
        "duplicate_within_work": False,
    }


class QualityGatePoemsTests(unittest.TestCase):
    def test_clean_line_becomes_release_candidate(self):
        scored, candidates, review = module.build_outputs(
            [decoded_line()], [source_line()], 0.22, 0.35, 3
        )
        self.assertEqual(len(scored), 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(review, [])
        self.assertEqual(scored[0]["quality_gate"]["status"], "candidate")
        self.assertEqual(scored[0]["source_file"], "sample.txt")

    def test_gate_emits_all_applicable_review_reasons(self):
        metrics, reasons = module.assess_line(decoded_line((0.3, 0.4), True, 4), 0.22, 0.35, 3)
        self.assertEqual(
            reasons,
            [
                "fallback_hanzi",
                "high_average_selection_score",
                "high_syllable_selection_score",
                "deep_decoder_candidate",
            ],
        )
        self.assertEqual(metrics["fallback_syllables"], 1)
        self.assertEqual(metrics["max_candidate_rank"], 4)

    def test_build_outputs_rejects_inconsistent_hanzi(self):
        row = decoded_line()
        row["hanzi"] = "错误"
        with self.assertRaisesRegex(ValueError, "Hanzi text mismatch"):
            module.build_outputs([row], [source_line()], 0.22, 0.35, 3)

    def test_report_counts_overlapping_reasons(self):
        scored, candidates, review = module.build_outputs(
            [decoded_line((0.3, 0.4), True, 4)], [source_line()], 0.22, 0.35, 3
        )
        report = module.build_report(scored, candidates, review, 0.22, 0.35, 3)
        self.assertEqual(report["candidate_lines"], 0)
        self.assertEqual(report["review_lines"], 1)
        self.assertEqual(report["review_reason_counts"]["fallback_hanzi"], 1)


if __name__ == "__main__":
    unittest.main()
