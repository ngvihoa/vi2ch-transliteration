#!/usr/bin/env python3
"""Merge per-poem Phú CSV files into one CSV using an atomic write."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "raw-collections" / "poetry-collecions" / "phu"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "phu.csv"
EXPECTED_HEADER = ["vi", "ch"]


class MergeError(RuntimeError):
    pass


def discover_inputs(input_dir: Path, output: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise MergeError(f"Thư mục input không tồn tại: {input_dir}")
    output_resolved = output.resolve()
    paths = [
        path for path in sorted(input_dir.glob("*.csv"))
        if path.resolve() != output_resolved
    ]
    if not paths:
        raise MergeError(f"Không tìm thấy CSV trong {input_dir}")
    return paths


def merge_csv_files(input_paths: list[Path], output: Path) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    row_count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(EXPECTED_HEADER)
            for input_path in input_paths:
                with input_path.open(encoding="utf-8", newline="") as source:
                    reader = csv.reader(source)
                    header = next(reader, None)
                    if header != EXPECTED_HEADER:
                        raise MergeError(f"Header không hợp lệ trong {input_path}: {header}")
                    for line_number, row in enumerate(reader, start=2):
                        if len(row) != 2:
                            raise MergeError(
                                f"{input_path}:{line_number} cần 2 cột, nhận {len(row)}"
                            )
                        if not row[0].strip() or not row[1].strip():
                            raise MergeError(f"{input_path}:{line_number} có cột rỗng")
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
    parser = argparse.ArgumentParser(description="Gom CSV từng bài Phú thành một CSV vi/ch.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files, rows = merge_csv_files(discover_inputs(args.input_dir, args.output), args.output)
    print(f"Đã gom {files} file / {rows} cặp câu vào {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
