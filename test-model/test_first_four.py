#!/usr/bin/env python3
"""Run VietHanBERT on the first four non-empty rows of a CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CSV = SCRIPT_DIR / "that-ngon-bat-cu.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_MODEL_ID = "noah-nguyen-297/VietHanBERT-vi2cn-v1"
DEFAULT_REVISION = "10da3a6a9911120cb85da3471337ff001a675245"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run greedy VietHanBERT inference and evaluation on only the first "
            "four non-empty vi/cn rows of a CSV file."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hugging Face branch, tag, or commit (default: pinned model commit)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not args.csv.is_file():
        parser.error(f"CSV file does not exist: {args.csv}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Reuse the canonical evaluation implementation so decoding and metrics stay
    # identical to a full test-set run.
    evaluation_dir = PROJECT_DIR / "evaluate"
    sys.path.insert(0, str(evaluation_dir))
    from evaluate_transliteration import main as evaluate_main

    evaluator_args = [
        "--model-id",
        args.model_id,
        "--test-file",
        str(args.csv),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--limit",
        "4",
        "--log-every",
        "1",
    ]
    if args.revision:
        evaluator_args.extend(("--revision", args.revision))
    return evaluate_main(evaluator_args)


if __name__ == "__main__":
    raise SystemExit(main())
