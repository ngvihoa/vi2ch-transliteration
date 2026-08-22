#!/usr/bin/env python3
"""Apply auditable quality gates to decoded synthetic poetry lines."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_DECODED = PIPELINE_ROOT / "outputs" / "poem_hanzi_decoded.jsonl"
DEFAULT_SOURCE = PIPELINE_ROOT / "outputs" / "poem_lines.jsonl"
DEFAULT_SCORED = PIPELINE_ROOT / "outputs" / "poem_quality_scored.jsonl"
DEFAULT_CANDIDATES = PIPELINE_ROOT / "outputs" / "poem_release_candidates.jsonl"
DEFAULT_REVIEW = PIPELINE_ROOT / "outputs" / "poem_quality_review.jsonl"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "quality_report.json"
DEFAULT_MAX_AVERAGE_SCORE = 0.22
DEFAULT_MAX_SYLLABLE_SCORE = 0.35
DEFAULT_MAX_CANDIDATE_RANK = 3


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def index_unique(rows: list[dict[str, object]], name: str) -> dict[str, dict[str, object]]:
    indexed = {}
    for row in rows:
        line_id = row.get("line_id")
        if not isinstance(line_id, str) or not line_id:
            raise ValueError(f"Missing line_id in {name}")
        if line_id in indexed:
            raise ValueError(f"Duplicate line_id {line_id!r} in {name}")
        indexed[line_id] = row
    return indexed


def validate_line(decoded: dict[str, object], source: dict[str, object]) -> None:
    line_id = decoded["line_id"]
    syllables = decoded.get("syllables")
    if not isinstance(syllables, list) or not syllables:
        raise ValueError(f"Decoded line {line_id} has no syllables")
    if len(syllables) != source.get("syllable_count"):
        raise ValueError(f"Syllable count mismatch for {line_id}")
    if decoded.get("vi") != source.get("text"):
        raise ValueError(f"Vietnamese text mismatch for {line_id}")
    if "".join(str(item.get("char", "")) for item in syllables) != decoded.get("hanzi"):
        raise ValueError(f"Hanzi text mismatch for {line_id}")
    if " ".join(str(item.get("pinyin", "")) for item in syllables) != decoded.get("pinyin"):
        raise ValueError(f"Pinyin text mismatch for {line_id}")


def assess_line(
    decoded: dict[str, object],
    max_average_score: float,
    max_syllable_score: float,
    max_candidate_rank: int,
) -> tuple[dict[str, object], list[str]]:
    syllables = decoded["syllables"]
    scores = [float(item["selection_score"]) for item in syllables]
    ranks = [int(item["candidate_rank"]) for item in syllables]
    fallback_count = sum(bool(item["requires_review"]) for item in syllables)
    changed_count = sum(bool(item["changed_from_greedy"]) for item in syllables)
    average_score = sum(scores) / len(scores)
    maximum_score = max(scores)
    maximum_rank = max(ranks)

    reasons = []
    if fallback_count:
        reasons.append("fallback_hanzi")
    if average_score > max_average_score:
        reasons.append("high_average_selection_score")
    if maximum_score > max_syllable_score:
        reasons.append("high_syllable_selection_score")
    if maximum_rank > max_candidate_rank:
        reasons.append("deep_decoder_candidate")

    metrics = {
        "average_selection_score": round(average_score, 6),
        "max_selection_score": round(maximum_score, 6),
        "fallback_syllables": fallback_count,
        "fallback_rate": round(fallback_count / len(syllables), 6),
        "decoder_changed_syllables": changed_count,
        "decoder_changed_rate": round(changed_count / len(syllables), 6),
        "max_candidate_rank": maximum_rank,
    }
    return metrics, reasons


def build_outputs(
    decoded_rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    max_average_score: float,
    max_syllable_score: float,
    max_candidate_rank: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    sources = index_unique(source_rows, "source lines")
    index_unique(decoded_rows, "decoded lines")
    scored = []
    candidates = []
    review = []

    for decoded in decoded_rows:
        line_id = decoded["line_id"]
        if line_id not in sources:
            raise ValueError(f"Missing source metadata for {line_id}")
        source = sources[line_id]
        validate_line(decoded, source)
        metrics, reasons = assess_line(
            decoded, max_average_score, max_syllable_score, max_candidate_rank
        )
        status = "candidate" if not reasons else "needs_review"
        row = {
            **decoded,
            "schema_version": "poem-quality-v2",
            "source_file": source["source_file"],
            "source_line_no": source["source_line_no"],
            "original_text": source["original_text"],
            "duplicate_within_work": source["duplicate_within_work"],
            "quality_gate": {
                "status": status,
                "reasons": reasons,
                "metrics": metrics,
            },
        }
        scored.append(row)
        (candidates if status == "candidate" else review).append(row)
    return scored, candidates, review


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = round((len(values) - 1) * fraction)
    return round(sorted(values)[index], 6)


def build_report(
    scored: list[dict[str, object]],
    candidates: list[dict[str, object]],
    review: list[dict[str, object]],
    max_average_score: float,
    max_syllable_score: float,
    max_candidate_rank: int,
) -> dict[str, object]:
    reason_counts = Counter(
        reason for row in review for reason in row["quality_gate"]["reasons"]
    )
    averages = [row["quality_gate"]["metrics"]["average_selection_score"] for row in scored]
    maximums = [row["quality_gate"]["metrics"]["max_selection_score"] for row in scored]
    total = len(scored)
    return {
        "schema_version": "quality-report-v2",
        "label_quality": "synthetic_silver",
        "gate": "deterministic-line-quality-v2",
        "input_lines": total,
        "candidate_lines": len(candidates),
        "candidate_rate": round(len(candidates) / total, 6) if total else 0.0,
        "review_lines": len(review),
        "review_rate": round(len(review) / total, 6) if total else 0.0,
        "thresholds": {
            "max_average_selection_score": max_average_score,
            "max_syllable_selection_score": max_syllable_score,
            "max_candidate_rank": max_candidate_rank,
            "fallback_syllables_allowed": 0,
        },
        "review_reason_counts": dict(sorted(reason_counts.items())),
        "average_selection_score_distribution": {
            "p50": percentile(averages, 0.5),
            "p90": percentile(averages, 0.9),
            "p95": percentile(averages, 0.95),
            "p99": percentile(averages, 0.99),
            "max": max(averages, default=0.0),
        },
        "max_selection_score_distribution": {
            "p50": percentile(maximums, 0.5),
            "p90": percentile(maximums, 0.9),
            "p95": percentile(maximums, 0.95),
            "p99": percentile(maximums, 0.99),
            "max": max(maximums, default=0.0),
        },
        "duplicate_source_lines": sum(bool(row["duplicate_within_work"]) for row in scored),
        "decoder_changed_lines": sum(bool(row["changed_from_greedy"]) for row in scored),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    _atomic_write_text(path, content + ("\n" if content else ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score decoded synthetic poetry and create a human-review queue."
    )
    parser.add_argument("--decoded", type=Path, default=DEFAULT_DECODED)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--scored-output", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-average-score", type=float, default=DEFAULT_MAX_AVERAGE_SCORE)
    parser.add_argument("--max-syllable-score", type=float, default=DEFAULT_MAX_SYLLABLE_SCORE)
    parser.add_argument("--max-candidate-rank", type=int, default=DEFAULT_MAX_CANDIDATE_RANK)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_average_score < 0 or args.max_syllable_score < 0:
        raise ValueError("Score thresholds must be non-negative")
    if args.max_candidate_rank < 1:
        raise ValueError("--max-candidate-rank must be positive")
    decoded_rows = read_jsonl(args.decoded)
    source_rows = read_jsonl(args.source)
    scored, candidates, review = build_outputs(
        decoded_rows,
        source_rows,
        args.max_average_score,
        args.max_syllable_score,
        args.max_candidate_rank,
    )
    report = build_report(
        scored,
        candidates,
        review,
        args.max_average_score,
        args.max_syllable_score,
        args.max_candidate_rank,
    )
    _write_jsonl(args.scored_output, scored)
    _write_jsonl(args.candidates_output, candidates)
    _write_jsonl(args.review_output, review)
    _atomic_write_text(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Quality-gated {report['input_lines']} lines; "
        f"candidates {report['candidate_lines']} ({report['candidate_rate']:.1%}); "
        f"needs review {report['review_lines']}."
    )
    print(args.scored_output)
    print(args.candidates_output)
    print(args.review_output)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
