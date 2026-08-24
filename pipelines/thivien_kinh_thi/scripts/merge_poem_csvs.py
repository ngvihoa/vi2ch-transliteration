#!/usr/bin/env python3
"""Merge per-poem Vietnamese/Chinese CSV files into one vi/cn CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "raw-collections" / "poetry-collecions"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "kinhthi.csv"
EXPECTED_HEADER = ["vi", "cn"]
LEGACY_HEADER = ["vi", "ch"]


class MergeError(RuntimeError):
    """Raised when an input CSV is malformed or cannot be merged safely."""


def discover_inputs(input_dir: Path, output: Path) -> list[Path]:
    """Return deterministic input paths, excluding the output itself."""
    if not input_dir.is_dir():
        raise MergeError(f"Thư mục input không tồn tại: {input_dir}")

    output_resolved = output.resolve()
    paths = [
        path
        for path in sorted(input_dir.glob("*.csv"))
        if path.resolve() != output_resolved
    ]
    if not paths:
        raise MergeError(f"Không tìm thấy file CSV trong {input_dir}")
    return paths


def merge_csv_files(input_paths: list[Path], output: Path) -> tuple[int, int]:
    """Validate and merge inputs, replacing output only after full success."""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    row_count = 0

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(EXPECTED_HEADER)

            for input_path in input_paths:
                with input_path.open("r", encoding="utf-8", newline="") as source:
                    reader = csv.reader(source)
                    header = next(reader, None)
                    if header not in (EXPECTED_HEADER, LEGACY_HEADER):
                        raise MergeError(
                            f"Header không hợp lệ trong {input_path}: "
                            f"cần {EXPECTED_HEADER}, nhận {header}"
                        )

                    for line_number, row in enumerate(reader, start=2):
                        if len(row) != 2:
                            raise MergeError(
                                f"{input_path}:{line_number} cần đúng 2 cột, "
                                f"nhận {len(row)}"
                            )
                        vi, cn = row
                        if not vi.strip() or not cn.strip():
                            raise MergeError(
                                f"{input_path}:{line_number} có cột vi/cn rỗng"
                            )
                        writer.writerow(row)
                        row_count += 1

            target.flush()
            os.fsync(target.fileno())

        os.replace(temporary, output)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise

    return len(input_paths), row_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gom các CSV từng bài Kinh Thi thành một CSV vi/cn duy nhất."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = discover_inputs(args.input_dir, args.output)
    file_count, row_count = merge_csv_files(input_paths, args.output)
    print(f"Đã gom {file_count} file / {row_count} cặp câu vào {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
