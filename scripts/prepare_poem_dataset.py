#!/usr/bin/env python3
"""Prepare a normalized, auditable poetry dataset from the raw text files.

This script intentionally stops before phonemization.  Its outputs are the stable
input layer for a later Vietnamese -> IPA -> Pinyin/Hanzi pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "poem_manifest.json"
DEFAULT_RAW_DIR = PROJECT_ROOT / "raw-collections" / "poem"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset"


@dataclass(frozen=True)
class LineRecord:
    schema_version: str
    line_id: str
    source_file: str
    work: str
    form: str
    source_line_no: int
    original_text: str
    text: str
    syllables: list[str]
    syllable_count: int
    line_role: str
    status: str
    include_in_benchmark: bool
    review_reasons: list[str]
    duplicate_within_work: bool


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding source punctuation."""
    return " ".join(unicodedata.normalize("NFC", text).strip().split()).lower()


def _seven_syllable_runs(counts: list[int]) -> set[int]:
    """Return zero-based indexes belonging to runs of at least four 7-word lines."""
    members: set[int] = set()
    start = 0
    while start < len(counts):
        if counts[start] != 7:
            start += 1
            continue
        end = start + 1
        while end < len(counts) and counts[end] == 7:
            end += 1
        if end - start >= 4:
            members.update(range(start, end))
        start = end
    return members


def classify_lines(form: str, texts: list[str]) -> list[tuple[str, str, bool, list[str]]]:
    """Classify lines using metre plus local context.

    Returns tuples of (role, status, include_in_benchmark, review_reasons).
    """
    counts = [len(text.split()) for text in texts]
    embedded_seven = _seven_syllable_runs(counts) if form == "luc_bat_mixed" else set()
    results: list[tuple[str, str, bool, list[str]]] = []

    for index, count in enumerate(counts):
        if not texts[index]:
            results.append(("blank", "excluded", False, ["blank_line"]))
            continue

        if form == "luc_bat":
            if count == 6:
                results.append(("luc", "valid", True, []))
            elif count == 8:
                results.append(("bat", "valid", True, []))
            else:
                results.append(("irregular", "needs_review", False, ["unexpected_syllable_count"]))

        elif form == "luc_bat_mixed":
            if count == 6:
                results.append(("luc", "valid", True, []))
            elif count == 8:
                results.append(("bat", "valid", True, []))
            elif index in embedded_seven:
                results.append(("embedded_that_ngon", "valid", True, []))
            elif count <= 3 and index + 1 < len(counts) and index + 1 in embedded_seven:
                results.append(("section_marker", "excluded", False, ["prose_cue_before_embedded_poem"]))
            else:
                results.append(("irregular", "needs_review", False, ["unexpected_syllable_count"]))

        elif form == "song_that_luc_bat":
            roles = {6: "luc", 7: "that", 8: "bat"}
            if count in roles:
                results.append((roles[count], "valid", True, []))
            else:
                results.append(("irregular", "needs_review", False, ["unexpected_syllable_count"]))

        elif form == "that_ngon_mixed":
            if count == 7:
                results.append(("that_ngon", "valid", True, []))
            elif count <= 4:
                results.append(("heading_candidate", "excluded", False, ["short_heading_candidate"]))
            else:
                results.append(("irregular", "needs_review", False, ["unexpected_syllable_count"]))

        else:
            results.append(("unknown", "needs_review", False, ["unknown_poetic_form"]))

    return results


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, list):
        raise ValueError("Manifest must be a JSON array")
    required = {"file", "work", "form"}
    for index, item in enumerate(manifest):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"Manifest item {index} must contain {sorted(required)}")
    return manifest


def build_records(manifest: Iterable[dict[str, str]], raw_dir: Path) -> list[LineRecord]:
    records: list[LineRecord] = []
    used_ids: set[str] = set()

    for item in manifest:
        source_path = raw_dir / item["file"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing poem source: {source_path}")

        original_lines = source_path.read_text(encoding="utf-8").splitlines()
        texts = [normalize_text(line) for line in original_lines]
        classifications = classify_lines(item["form"], texts)
        duplicate_counts = Counter(text for text in texts if text)
        stem = Path(item["file"]).name.split(".", 1)[0]

        for line_no, (original, text, classification) in enumerate(
            zip(original_lines, texts, classifications), start=1
        ):
            line_id = f"{stem}_{line_no:04d}"
            if line_id in used_ids:
                raise ValueError(f"Duplicate generated line id: {line_id}")
            used_ids.add(line_id)
            role, status, include, reasons = classification
            syllables = text.split() if text else []
            records.append(
                LineRecord(
                    schema_version="poem-lines-v1",
                    line_id=line_id,
                    source_file=item["file"],
                    work=item["work"],
                    form=item["form"],
                    source_line_no=line_no,
                    original_text=original,
                    text=text,
                    syllables=syllables,
                    syllable_count=len(syllables),
                    line_role=role,
                    status=status,
                    include_in_benchmark=include,
                    review_reasons=list(reasons),
                    duplicate_within_work=bool(text and duplicate_counts[text] > 1),
                )
            )
    return records


def build_report(records: list[LineRecord]) -> dict[str, object]:
    by_work: dict[str, dict[str, object]] = {}
    for work in dict.fromkeys(record.work for record in records):
        subset = [record for record in records if record.work == work]
        by_work[work] = {
            "source_file": subset[0].source_file,
            "form": subset[0].form,
            "total_lines": len(subset),
            "benchmark_lines": sum(record.include_in_benchmark for record in subset),
            "status_counts": dict(sorted(Counter(record.status for record in subset).items())),
            "role_counts": dict(sorted(Counter(record.line_role for record in subset).items())),
            "syllable_count_distribution": {
                str(key): value
                for key, value in sorted(Counter(record.syllable_count for record in subset).items())
            },
            "duplicate_line_occurrences": sum(record.duplicate_within_work for record in subset),
        }

    return {
        "schema_version": "poem-report-v1",
        "total_lines": len(records),
        "benchmark_lines": sum(record.include_in_benchmark for record in records),
        "excluded_lines": sum(record.status == "excluded" for record in records),
        "needs_review_lines": sum(record.status == "needs_review" for record in records),
        "works": by_work,
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


def write_outputs(records: list[LineRecord], output_dir: Path) -> tuple[Path, Path, Path]:
    lines_path = output_dir / "poem_lines.jsonl"
    review_path = output_dir / "poem_review.jsonl"
    report_path = output_dir / "poem_report.json"

    serialized = [json.dumps(asdict(record), ensure_ascii=False) for record in records]
    review = [
        json.dumps(asdict(record), ensure_ascii=False)
        for record in records
        if record.status != "valid"
    ]
    _atomic_write_text(lines_path, "\n".join(serialized) + "\n")
    _atomic_write_text(review_path, "\n".join(review) + ("\n" if review else ""))
    _atomic_write_text(
        report_path,
        json.dumps(build_report(records), ensure_ascii=False, indent=2) + "\n",
    )
    return lines_path, review_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare normalized poem sources as an auditable JSONL dataset."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = build_records(load_manifest(args.manifest), args.raw_dir)
    paths = write_outputs(records, args.output_dir)
    report = build_report(records)
    print(
        f"Processed {report['total_lines']} lines; "
        f"included {report['benchmark_lines']}, "
        f"excluded {report['excluded_lines']}, "
        f"needs review {report['needs_review_lines']}."
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
