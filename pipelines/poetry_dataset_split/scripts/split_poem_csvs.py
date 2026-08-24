#!/usr/bin/env python3
"""Split every poem genre independently, then merge the resulting splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = PIPELINE_ROOT / "input"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "outputs"
FIELDNAMES = ("vi", "ch")
SPLIT_NAMES = ("train", "test", "val")


def validate_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} ratios")
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("All split ratios must be finite and greater than zero")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0 (received {sum(ratios):g})")


def allocate_split_sizes(row_count: int, ratios: Sequence[float]) -> tuple[int, ...]:
    """Allocate rows proportionally while keeping every enabled split non-empty."""
    validate_ratios(ratios)
    if row_count < len(ratios):
        raise ValueError(
            f"Need at least {len(ratios)} rows to populate every split; found {row_count}"
        )

    quotas = [row_count * ratio for ratio in ratios]
    sizes = [int(quota) for quota in quotas]
    undistributed = row_count - sum(sizes)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (quotas[index] - sizes[index], ratios[index], -index),
        reverse=True,
    )
    for index in order[:undistributed]:
        sizes[index] += 1

    for empty_index, size in enumerate(sizes):
        if size:
            continue
        donor_index = max(range(len(sizes)), key=lambda index: (sizes[index], ratios[index]))
        sizes[donor_index] -= 1
        sizes[empty_index] = 1
    return tuple(sizes)


def read_poem_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDNAMES):
            raise ValueError(
                f"{path}: expected CSV header {','.join(FIELDNAMES)!r}, "
                f"found {reader.fieldnames!r}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{path}:{line_number}: row has more than two columns")
            cleaned = {field: (row[field] or "").strip() for field in FIELDNAMES}
            if not any(cleaned.values()):
                continue
            if not all(cleaned.values()):
                raise ValueError(f"{path}:{line_number}: both vi and ch must be non-empty")
            rows.append(cleaned)
    return rows


def genre_seed(seed: int, filename: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{filename}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_rows(
    rows: Sequence[dict[str, str]], ratios: Sequence[float], seed: int
) -> dict[str, list[dict[str, str]]]:
    sizes = allocate_split_sizes(len(rows), ratios)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    result: dict[str, list[dict[str, str]]] = {}
    start = 0
    for name, size in zip(SPLIT_NAMES, sizes):
        result[name] = shuffled[start : start + size]
        start += size
    return result


def build_dataset(
    input_paths: Iterable[Path], ratios: Sequence[float], seed: int
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
    validate_ratios(ratios)
    combined = {name: [] for name in SPLIT_NAMES}
    counts: dict[str, dict[str, int]] = {}

    for path in sorted(input_paths, key=lambda item: item.name):
        rows = read_poem_csv(path)
        try:
            split = split_rows(rows, ratios, genre_seed(seed, path.name))
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        counts[path.name] = {name: len(split[name]) for name in SPLIT_NAMES}
        for name in SPLIT_NAMES:
            combined[name].extend(split[name])

    # Avoid blocks of adjacent genres in the merged datasets.
    for index, name in enumerate(SPLIT_NAMES):
        random.Random(seed + index).shuffle(combined[name])
    return combined, counts


def atomic_write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split poem genre CSVs independently and merge train/test/val outputs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ratios = (args.train_ratio, args.test_ratio, args.val_ratio)
    input_paths = sorted(args.input_dir.glob("*.csv"))
    if not input_paths:
        raise SystemExit(f"No CSV files found in {args.input_dir}")

    try:
        combined, counts = build_dataset(input_paths, ratios, args.seed)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    for name in SPLIT_NAMES:
        atomic_write_csv(args.output_dir / f"poem.{name}.csv", combined[name])

    for filename, split_counts in counts.items():
        print(
            f"{filename}: "
            + ", ".join(f"{name}={split_counts[name]}" for name in SPLIT_NAMES)
        )
    print(
        "Combined: "
        + ", ".join(f"{name}={len(combined[name])}" for name in SPLIT_NAMES)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
