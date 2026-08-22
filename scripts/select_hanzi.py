#!/usr/bin/env python3
"""Select auditable Hanzi candidates for ranked Pinyin pronunciations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PINYIN_CANDIDATES = PROJECT_ROOT / "dataset" / "pinyin_candidates.jsonl"
DEFAULT_PINYIN_LINES = PROJECT_ROOT / "dataset" / "poem_pinyin.jsonl"
DEFAULT_HANZI_CANDIDATES = PROJECT_ROOT / "dataset" / "hanzi_candidates.jsonl"
DEFAULT_HANZI_LINES = PROJECT_ROOT / "dataset" / "poem_hanzi.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "dataset" / "hanzi_report.json"
DEFAULT_REFERENCE = PROJECT_ROOT / "resources" / "xinhua_english_reference.json"
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "raw-collections" / "cn-vi"
DEFAULT_DEPS = PROJECT_ROOT / "tools" / "pinyin-python"
DEFAULT_LOCK = SCRIPT_DIR / "pinyin.lock.json"

WEIGHTS = {"phonetic": 0.75, "reference_pool": 0.15, "frequency": 0.10}


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


def activate_and_verify_dependencies(deps: Path, lock_path: Path) -> dict[str, object]:
    if not deps.is_dir():
        raise FileNotFoundError(
            f"Pinyin dependencies not found at {deps}. Run: "
            "python -m pip install --target tools/pinyin-python -r requirements-pinyin.txt"
        )
    sys.path.insert(0, str(deps))
    with lock_path.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    for package, expected in lock["packages"].items():
        if version(package) != expected:
            raise RuntimeError(f"Dependency {package} does not match {expected}")
    return lock


def load_reference(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        resource = json.load(handle)
    readings = resource.get("readings")
    if not isinstance(readings, dict) or not all(
        isinstance(pinyin, str) and isinstance(chars, str) and chars
        for pinyin, chars in readings.items()
    ):
        raise ValueError(f"Invalid transliteration reference: {path}")
    metadata = {key: value for key, value in resource.items() if key != "readings"}
    metadata["reading_count"] = len(readings)
    metadata["character_count"] = len(set("".join(readings.values())))
    return readings, metadata


def build_reverse_pinyin() -> dict[str, set[str]]:
    from pypinyin.constants import PINYIN_DICT
    from pypinyin.contrib.tone_convert import to_tone3

    reverse: dict[str, set[str]] = defaultdict(set)
    for codepoint, readings in PINYIN_DICT.items():
        char = chr(codepoint)
        if not "\u4e00" <= char <= "\u9fff":
            continue
        for reading in readings.split(","):
            numbered = to_tone3(reading, neutral_tone_with_five=True, v_to_u=True)
            if numbered[-1:] in {"1", "2", "3", "4"}:
                reverse[numbered].add(char)
    return reverse


def count_corpus_characters(corpus_dir: Path) -> tuple[Counter[str], list[str]]:
    paths = sorted(corpus_dir.glob("*.cn"))
    if not paths:
        raise FileNotFoundError(f"No .cn frequency corpus files found in {corpus_dir}")
    counts: Counter[str] = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), ""):
                counts.update(char for char in chunk if "\u4e00" <= char <= "\u9fff")
    return counts, [str(path.relative_to(PROJECT_ROOT)) for path in paths]


def frequency_penalty(count: int, maximum: int) -> float:
    if maximum <= 0:
        return 1.0
    return 1.0 - math.log1p(count) / math.log1p(maximum)


def select_candidates(
    row: dict[str, object],
    reference: dict[str, str],
    reverse: dict[str, set[str]],
    frequencies: Counter[str],
    top_n: int,
) -> list[dict[str, object]]:
    maximum = max(frequencies.values(), default=0)
    best_by_char: dict[str, dict[str, object]] = {}

    for pinyin_rank, pinyin_candidate in enumerate(row["candidates"], start=1):
        pinyin = pinyin_candidate["pinyin"]
        reference_chars = set(reference.get(pinyin, ""))
        corpus_chars = {char for char in reverse.get(pinyin, set()) if frequencies[char] > 0}
        options = [(char, "xinhua_english_reference") for char in reference_chars]
        options.extend(
            (char, "corpus_frequency_fallback")
            for char in corpus_chars - reference_chars
        )
        if not options:
            options.extend(
                (char, "unseen_dictionary_fallback")
                for char in pinyin_candidate.get("example_hanzi", [])
                if "\u4e00" <= char <= "\u9fff"
            )

        for char, provenance in options:
            pool_penalty = 0.0 if provenance == "xinhua_english_reference" else 1.0
            char_frequency_penalty = frequency_penalty(frequencies[char], maximum)
            selection_score = (
                WEIGHTS["phonetic"] * pinyin_candidate["score"]
                + WEIGHTS["reference_pool"] * pool_penalty
                + WEIGHTS["frequency"] * char_frequency_penalty
            )
            candidate = {
                "char": char,
                "pinyin": pinyin,
                "pinyin_rank": pinyin_rank,
                "ipa": pinyin_candidate["ipa"],
                "phonetic_score": pinyin_candidate["score"],
                "selection_score": round(selection_score, 6),
                "reference_pool_penalty": pool_penalty,
                "frequency_penalty": round(char_frequency_penalty, 6),
                "corpus_frequency": frequencies[char],
                "provenance": provenance,
                "requires_review": provenance != "xinhua_english_reference",
            }
            previous = best_by_char.get(char)
            if previous is None or (
                candidate["selection_score"], -candidate["corpus_frequency"], candidate["pinyin"]
            ) < (
                previous["selection_score"], -previous["corpus_frequency"], previous["pinyin"]
            ):
                best_by_char[char] = candidate

    return sorted(
        best_by_char.values(),
        key=lambda item: (item["selection_score"], -item["corpus_frequency"], item["char"]),
    )[:top_n]


def build_candidate_rows(
    pinyin_rows: list[dict[str, object]],
    reference: dict[str, str],
    reverse: dict[str, set[str]],
    frequencies: Counter[str],
    top_n: int,
) -> list[dict[str, object]]:
    output = []
    for row in pinyin_rows:
        candidates = select_candidates(row, reference, reverse, frequencies, top_n)
        if not candidates:
            raise RuntimeError(f"No Hanzi candidates for {row['candidate_set_id']}")
        output.append(
            {
                "schema_version": "hanzi-candidates-v1",
                "candidate_set_id": row["candidate_set_id"],
                "source_dialect": row["source_dialect"],
                "source_ipa_segments": row["source_ipa_segments"],
                "source_tone_chao": row["source_tone_chao"],
                "source_forms": row["source_forms"],
                "occurrences": row["occurrences"],
                "candidates": candidates,
            }
        )
    return output


def build_line_rows(
    pinyin_lines: list[dict[str, object]], hanzi_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    lookup = {row["candidate_set_id"]: row["candidates"] for row in hanzi_rows}
    output = []
    for line in pinyin_lines:
        selections = []
        for syllable in line["pinyin_syllables"]:
            selected = lookup[syllable["candidate_set_id"]][0]
            selections.append(
                {
                    "text": syllable["text"],
                    "source_ipa": syllable["source_ipa"],
                    "candidate_set_id": syllable["candidate_set_id"],
                    "char": selected["char"],
                    "pinyin": selected["pinyin"],
                    "selection_score": selected["selection_score"],
                    "provenance": selected["provenance"],
                    "requires_review": selected["requires_review"],
                }
            )
        output.append(
            {
                "schema_version": "poem-hanzi-v1",
                "label_quality": "synthetic_silver",
                "line_id": line["line_id"],
                "work": line["work"],
                "form": line["form"],
                "line_role": line["line_role"],
                "vi": line["text"],
                "hanzi": "".join(item["char"] for item in selections),
                "pinyin": " ".join(item["pinyin"] for item in selections),
                "requires_review": any(item["requires_review"] for item in selections),
                "syllables": selections,
            }
        )
    return output


def build_report(
    candidate_rows: list[dict[str, object]],
    line_rows: list[dict[str, object]],
    reference_metadata: dict[str, object],
    corpus_files: list[str],
    frequencies: Counter[str],
    top_n: int,
    dependency_lock: dict[str, object],
) -> dict[str, object]:
    selected = [row["candidates"][0] for row in candidate_rows]
    reference_sets = sum(item["provenance"] == "xinhua_english_reference" for item in selected)
    reference_occurrences = sum(
        row["occurrences"]
        for row in candidate_rows
        if row["candidates"][0]["provenance"] == "xinhua_english_reference"
    )
    total_occurrences = sum(row["occurrences"] for row in candidate_rows)
    return {
        "schema_version": "hanzi-report-v1",
        "label_quality": "synthetic_silver",
        "candidate_sets": len(candidate_rows),
        "line_count": len(line_rows),
        "syllable_occurrences": total_occurrences,
        "top_n": top_n,
        "weights": WEIGHTS,
        "reference_selected_sets": reference_sets,
        "reference_selected_set_rate": round(reference_sets / len(candidate_rows), 6),
        "reference_selected_occurrences": reference_occurrences,
        "reference_selected_occurrence_rate": round(reference_occurrences / total_occurrences, 6),
        "fallback_selected_sets": len(candidate_rows) - reference_sets,
        "lines_requiring_review": sum(row["requires_review"] for row in line_rows),
        "reference": reference_metadata,
        "frequency_corpus": {
            "files": corpus_files,
            "total_hanzi": sum(frequencies.values()),
            "unique_hanzi": len(frequencies),
        },
        "dependencies": dependency_lock,
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
    _atomic_write_text(path, "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Hanzi for ranked Pinyin transliterations.")
    parser.add_argument("--pinyin-candidates", type=Path, default=DEFAULT_PINYIN_CANDIDATES)
    parser.add_argument("--pinyin-lines", type=Path, default=DEFAULT_PINYIN_LINES)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_HANZI_CANDIDATES)
    parser.add_argument("--lines-output", type=Path, default=DEFAULT_HANZI_LINES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")
    dependency_lock = activate_and_verify_dependencies(args.deps, args.lock)
    reference, reference_metadata = load_reference(args.reference)
    reverse = build_reverse_pinyin()
    frequencies, corpus_files = count_corpus_characters(args.corpus_dir)
    pinyin_rows = read_jsonl(args.pinyin_candidates)
    pinyin_lines = read_jsonl(args.pinyin_lines)
    candidate_rows = build_candidate_rows(
        pinyin_rows, reference, reverse, frequencies, args.top_n
    )
    line_rows = build_line_rows(pinyin_lines, candidate_rows)
    report = build_report(
        candidate_rows, line_rows, reference_metadata, corpus_files,
        frequencies, args.top_n, dependency_lock,
    )
    _write_jsonl(args.candidates_output, candidate_rows)
    _write_jsonl(args.lines_output, line_rows)
    _atomic_write_text(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Selected Hanzi for {report['syllable_occurrences']} syllable occurrences; "
        f"reference coverage {report['reference_selected_occurrence_rate']:.1%}; "
        f"review lines {report['lines_requiring_review']}."
    )
    print(args.candidates_output)
    print(args.lines_output)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
