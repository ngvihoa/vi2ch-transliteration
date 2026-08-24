import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "split_poem_csvs.py"
SPEC = importlib.util.spec_from_file_location("split_poem_csvs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, prefix: str, count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("vi", "ch"))
        writer.writeheader()
        for index in range(count):
            writer.writerow({"vi": f"{prefix}-vi-{index}", "ch": f"{prefix}-ch-{index}"})


class SplitPoemCsvsTests(unittest.TestCase):
    def test_default_allocation_keeps_all_three_splits(self):
        self.assertEqual((8, 1, 1), MODULE.allocate_split_sizes(10, (0.8, 0.1, 0.1)))
        self.assertEqual((1, 1, 1), MODULE.allocate_split_sizes(3, (0.8, 0.1, 0.1)))

    def test_rejects_too_small_genre(self):
        with self.assertRaisesRegex(ValueError, "at least 3 rows"):
            MODULE.allocate_split_sizes(2, (0.8, 0.1, 0.1))

    def test_rejects_invalid_ratios(self):
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            MODULE.validate_ratios((0.8, 0.2, 0.1))
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.validate_ratios((float("nan"), 0.5, 0.5))

    def test_splits_each_genre_before_merging_without_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "genre-a.csv"
            second = root / "genre-b.csv"
            write_csv(first, "a", 10)
            write_csv(second, "b", 20)

            combined, counts = MODULE.build_dataset(
                [second, first], (0.8, 0.1, 0.1), seed=42
            )

            self.assertEqual({"train": 8, "test": 1, "val": 1}, counts["genre-a.csv"])
            self.assertEqual({"train": 16, "test": 2, "val": 2}, counts["genre-b.csv"])
            self.assertEqual(30, len(combined["clean"]))
            self.assertEqual([24, 3, 3], [len(combined[name]) for name in MODULE.SPLIT_NAMES])
            self.assertEqual("a-vi-0", combined["clean"][0]["vi"])
            self.assertEqual("b-vi-19", combined["clean"][-1]["vi"])

            all_pairs = {
                (row["vi"], row["ch"])
                for name in MODULE.SPLIT_NAMES
                for row in combined[name]
            }
            self.assertEqual(30, len(all_pairs))
            self.assertEqual(
                {(row["vi"], row["ch"]) for row in combined["clean"]},
                all_pairs,
            )

    def test_same_seed_is_reproducible(self):
        rows = [{"vi": str(index), "ch": str(index)} for index in range(12)]
        first = MODULE.split_rows(rows, (0.8, 0.1, 0.1), seed=7)
        second = MODULE.split_rows(rows, (0.8, 0.1, 0.1), seed=7)
        self.assertEqual(first, second)

    def test_rejects_wrong_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text("source,target\na,b\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected CSV header"):
                MODULE.read_poem_csv(path)


if __name__ == "__main__":
    unittest.main()
