import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import decode_poem_hanzi as module


class DecodePoemHanziTests(unittest.TestCase):
    def test_luc_bat_rhyme_edges(self):
        lines = [
            {
                "work": "Sample", "form": "luc_bat", "line_role": "luc",
                "pinyin_syllables": [{} for _ in range(6)],
            },
            {
                "work": "Sample", "form": "luc_bat", "line_role": "bat",
                "pinyin_syllables": [{} for _ in range(8)],
            },
            {
                "work": "Sample", "form": "luc_bat", "line_role": "luc",
                "pinyin_syllables": [{} for _ in range(6)],
            },
        ]
        self.assertEqual(
            module.build_rhyme_edges(lines),
            [((0, 5), (1, 5)), ((1, 7), (2, 5))],
        )

    def test_song_that_rhyme_edges(self):
        roles = [("that", 7), ("that", 7), ("luc", 6), ("bat", 8), ("that", 7)]
        lines = [
            {
                "work": "Sample", "form": "song_that_luc_bat", "line_role": role,
                "pinyin_syllables": [{} for _ in range(length)],
            }
            for role, length in roles
        ]
        self.assertEqual(
            module.build_rhyme_edges(lines),
            [
                ((0, 6), (1, 4)), ((1, 6), (2, 5)),
                ((2, 5), (3, 5)), ((3, 7), (4, 4)),
            ],
        )

    def test_viterbi_can_trade_local_score_for_rhyme(self):
        left, right = (0, 0), (1, 0)
        options = {
            left: [
                {"selection_score": 0.0, "rhyme": "x"},
                {"selection_score": 0.03, "rhyme": "y"},
            ],
            right: [{"selection_score": 0.0, "rhyme": "y"}],
        }
        distance = lambda a, b: 0.0 if a["rhyme"] == b["rhyme"] else 1.0
        selected = module.optimize_path([left, right], options, 0.12, distance)
        self.assertEqual(selected, {left: 1, right: 0})

    def test_decode_keeps_unlinked_syllables_greedy(self):
        lines = [
            {
                "line_id": "sample_0001", "work": "Sample", "form": "that_ngon_mixed",
                "line_role": "that_ngon", "text": "ta",
                "pinyin_syllables": [
                    {"text": "ta", "source_ipa": "ta33", "candidate_set_id": "set1"}
                ],
            }
        ]
        candidate_rows = [
            {
                "candidate_set_id": "set1",
                "candidates": [
                    {
                        "char": "达", "pinyin": "da2", "ipa": "ta",
                        "selection_score": 0.1, "provenance": "xinhua_english_reference",
                        "requires_review": False,
                    },
                    {
                        "char": "大", "pinyin": "da4", "ipa": "ta",
                        "selection_score": 0.2, "provenance": "corpus_frequency_fallback",
                        "requires_review": True,
                    },
                ],
            }
        ]
        selections, edges = module.decode(
            lines, candidate_rows, 0.12, lambda left, right: 0.0
        )
        decoded = module.build_decoded_lines(lines, candidate_rows, selections)
        self.assertEqual(edges, [])
        self.assertEqual(decoded[0]["hanzi"], "达")
        self.assertFalse(decoded[0]["changed_from_greedy"])


if __name__ == "__main__":
    unittest.main()
