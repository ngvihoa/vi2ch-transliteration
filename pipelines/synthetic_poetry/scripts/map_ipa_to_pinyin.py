#!/usr/bin/env python3
"""Rank attested Mandarin Pinyin syllables for Vietnamese IPA syllables."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_INPUT = PIPELINE_ROOT / "outputs" / "poem_ipa.jsonl"
DEFAULT_CANDIDATES = PIPELINE_ROOT / "outputs" / "pinyin_candidates.jsonl"
DEFAULT_LINES = PIPELINE_ROOT / "outputs" / "poem_pinyin.jsonl"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "pinyin_report.json"
DEFAULT_DEPS = PROJECT_ROOT / "tools" / "pinyin-python"
DEFAULT_LOCK = SCRIPT_DIR / "pinyin.lock.json"

TONE_MARKS = str.maketrans("", "", "˥˦˧˨˩")
MANDARIN_TONES = {1: "55", 2: "35", 3: "214", 4: "51"}
WEIGHTS = {"onset": 0.35, "rhyme": 0.45, "tone": 0.20}
VI_ONSETS = tuple(
    sorted(
        ("tɕ", "tʰ", "kw", "ɓ", "c", "j", "ɗ", "ɣ", "h", "k", "l", "m", "n", "ŋ", "ɲ", "f", "p", "r", "ʂ", "t", "v", "x", "s", "z", "ʈ", "w"),
        key=len,
        reverse=True,
    )
)

VOWEL_FEATURES = {
    "i": (0.0, 0.0, 0), "y": (0.0, 0.0, 1), "ɨ": (0.0, 0.5, 0),
    "u": (0.0, 1.0, 1), "ʊ": (0.2, 1.0, 1), "e": (0.35, 0.0, 0),
    "ɛ": (0.65, 0.0, 0), "ə": (0.5, 0.5, 0), "ɤ": (0.45, 1.0, 0),
    "o": (0.35, 1.0, 1), "ɔ": (0.65, 1.0, 1), "a": (1.0, 0.5, 0),
    "ɚ": (0.5, 0.5, 0),
}
CONSONANT_FEATURES = {
    "p": (0.0, 0.0, 0, 0), "pʰ": (0.0, 0.0, 0, 1), "ɓ": (0.0, 0.0, 1, 0),
    "m": (0.0, 0.6, 1, 0), "f": (0.05, 0.4, 0, 0), "v": (0.05, 0.4, 1, 0),
    "t": (0.35, 0.0, 0, 0), "tʰ": (0.35, 0.0, 0, 1), "ɗ": (0.35, 0.0, 1, 0),
    "n": (0.35, 0.6, 1, 0), "s": (0.35, 0.4, 0, 0), "z": (0.35, 0.4, 1, 0),
    "l": (0.35, 0.75, 1, 0), "r": (0.35, 0.9, 1, 0), "ɹ": (0.35, 0.9, 1, 0),
    "ts": (0.35, 0.2, 0, 0), "tsʰ": (0.35, 0.2, 0, 1),
    "ʈ": (0.5, 0.0, 0, 0), "ʂ": (0.5, 0.4, 0, 0), "ʐ": (0.5, 0.4, 1, 0),
    "ɻ": (0.5, 0.9, 1, 0), "ʈʂ": (0.5, 0.2, 0, 0), "ʈʂʰ": (0.5, 0.2, 0, 1),
    "c": (0.65, 0.0, 0, 0), "ɲ": (0.65, 0.6, 1, 0), "j": (0.65, 0.9, 1, 0),
    "tɕ": (0.65, 0.2, 0, 0), "tɕʰ": (0.65, 0.2, 0, 1), "ɕ": (0.65, 0.4, 0, 0),
    "k": (0.85, 0.0, 0, 0), "kʰ": (0.85, 0.0, 0, 1), "ɣ": (0.85, 0.4, 1, 0),
    "x": (0.85, 0.4, 0, 0), "ŋ": (0.85, 0.6, 1, 0),
    "h": (1.0, 0.4, 0, 0), "w": (0.8, 0.9, 1, 0), "ɥ": (0.65, 0.9, 1, 0),
}
PHONEMES = tuple(sorted(set(VOWEL_FEATURES) | set(CONSONANT_FEATURES), key=len, reverse=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activate_and_verify_dependencies(deps: Path, lock_path: Path) -> dict[str, object]:
    if not deps.is_dir():
        raise FileNotFoundError(
            f"Pinyin dependencies not found at {deps}. Run: "
            "python -m pip install --target tools/pinyin-python "
            "-r pipelines/synthetic_poetry/requirements-pinyin.txt"
        )
    sys.path.insert(0, str(deps))
    with lock_path.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    for package, expected in lock["packages"].items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f"Dependency {package}: expected {expected}, got {actual}")
    for relative, expected in lock["files"].items():
        actual = _sha256(deps / relative)
        if actual != expected:
            raise RuntimeError(f"Dependency file {relative} does not match its lock hash")
    return lock


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


def clean_ipa(text: str) -> str:
    return (
        text.translate(TONE_MARKS)
        .replace("ː", "")
        .replace("̯", "")
        .replace("̩", "")
        .replace("͡", "")
        .replace("ʰ", "ʰ")
    )


def tokenize_ipa(text: str) -> tuple[str, ...]:
    text = clean_ipa(text)
    tokens: list[str] = []
    index = 0
    while index < len(text):
        match = next((symbol for symbol in PHONEMES if text.startswith(symbol, index)), None)
        if match is None:
            index += 1
            continue
        tokens.append(match)
        index += len(match)
    return tuple(tokens)


def split_vietnamese_ipa(segments: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    onset_text = next((onset for onset in VI_ONSETS if segments.startswith(onset)), "")
    if onset_text == "kw":
        onset_tokens = ("k", "w")
    else:
        onset_tokens = tokenize_ipa(onset_text)
    return onset_tokens, tokenize_ipa(segments[len(onset_text):])


def phoneme_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if left in VOWEL_FEATURES and right in VOWEL_FEATURES:
        lh, lb, lr = VOWEL_FEATURES[left]
        rh, rb, rr = VOWEL_FEATURES[right]
        return 0.4 * abs(lh - rh) + 0.4 * abs(lb - rb) + 0.2 * (lr != rr)
    if left in CONSONANT_FEATURES and right in CONSONANT_FEATURES:
        lp, lm, lv, la = CONSONANT_FEATURES[left]
        rp, rm, rv, ra = CONSONANT_FEATURES[right]
        return 0.35 * abs(lp - rp) + 0.35 * abs(lm - rm) + 0.15 * (lv != rv) + 0.15 * (la != ra)
    return 1.0


def sequence_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left and not right:
        return 0.0
    previous = [float(index) for index in range(len(right) + 1)]
    for row, left_phone in enumerate(left, start=1):
        current = [float(row)]
        for column, right_phone in enumerate(right, start=1):
            current.append(
                min(
                    previous[column] + 1.0,
                    current[column - 1] + 1.0,
                    previous[column - 1] + phoneme_distance(left_phone, right_phone),
                )
            )
        previous = current
    return min(1.0, previous[-1] / max(len(left), len(right)))


def _resample_tone(tone: str) -> tuple[float, float, float]:
    values = [(int(digit) - 1) / 4 for digit in re.findall(r"[1-5]", tone)]
    if not values:
        return (0.5, 0.5, 0.5)
    if len(values) == 1:
        return (values[0],) * 3
    middle_position = (len(values) - 1) / 2
    lower = int(middle_position)
    upper = min(lower + 1, len(values) - 1)
    fraction = middle_position - lower
    middle = values[lower] * (1 - fraction) + values[upper] * fraction
    return values[0], middle, values[-1]


def tone_distance(vietnamese: str, mandarin: str) -> float:
    left = _resample_tone(vietnamese)
    right = _resample_tone(mandarin)
    contour = sum(abs(a - b) for a, b in zip(left, right)) / 3
    glottal_penalty = 0.10 if "g" in vietnamese else 0.0
    return min(1.0, contour + glottal_penalty)


def build_inventory() -> list[dict[str, object]]:
    from pinyin_to_ipa import pinyin_to_ipa
    from pypinyin.constants import PINYIN_DICT
    from pypinyin.contrib.tone_convert import to_initials, to_tone3

    hanzi: dict[str, set[str]] = defaultdict(set)
    for codepoint, readings in PINYIN_DICT.items():
        for reading in readings.split(","):
            numbered = to_tone3(reading, neutral_tone_with_five=True, v_to_u=True)
            if numbered[-1:] in {"1", "2", "3", "4"}:
                hanzi[numbered].add(chr(codepoint))

    inventory: list[dict[str, object]] = []
    for pinyin in sorted(hanzi):
        base, tone_number = pinyin[:-1], int(pinyin[-1])
        has_initial = bool(to_initials(base, strict=True))
        variants = []
        for raw_variant in pinyin_to_ipa(pinyin):
            raw_parts = [clean_ipa(part) for part in raw_variant]
            if has_initial:
                onset = tokenize_ipa(raw_parts[0])
                rhyme = tuple(phone for part in raw_parts[1:] for phone in tokenize_ipa(part))
            else:
                onset = ()
                rhyme = tuple(phone for part in raw_parts for phone in tokenize_ipa(part))
            variants.append(
                {
                    "ipa": "".join(raw_parts),
                    "onset": onset,
                    "rhyme": rhyme,
                }
            )
        inventory.append(
            {
                "pinyin": pinyin,
                "base": base,
                "tone_number": tone_number,
                "tone_chao": MANDARIN_TONES[tone_number],
                "variants": variants,
                "hanzi_count": len(hanzi[pinyin]),
                "example_hanzi": sorted(hanzi[pinyin])[:10],
            }
        )
    return inventory


def rank_candidates(
    ipa_segments: str, source_tone: str, inventory: list[dict[str, object]], top_k: int
) -> list[dict[str, object]]:
    source_onset, source_rhyme = split_vietnamese_ipa(ipa_segments)
    ranked = []
    for target in inventory:
        best = None
        for variant in target["variants"]:
            onset = sequence_distance(source_onset, variant["onset"])
            rhyme = sequence_distance(source_rhyme, variant["rhyme"])
            tone = tone_distance(source_tone, target["tone_chao"])
            score = WEIGHTS["onset"] * onset + WEIGHTS["rhyme"] * rhyme + WEIGHTS["tone"] * tone
            result = (score, onset, rhyme, tone, variant["ipa"])
            if best is None or result < best:
                best = result
        score, onset, rhyme, tone, matched_ipa = best
        ranked.append(
            {
                "pinyin": target["pinyin"],
                "base": target["base"],
                "tone_number": target["tone_number"],
                "tone_chao": target["tone_chao"],
                "ipa": matched_ipa,
                "score": round(score, 6),
                "onset_distance": round(onset, 6),
                "rhyme_distance": round(rhyme, 6),
                "tone_distance": round(tone, 6),
                "hanzi_count": target["hanzi_count"],
                "example_hanzi": target["example_hanzi"],
            }
        )
    return heapq.nsmallest(top_k, ranked, key=lambda item: (item["score"], -item["hanzi_count"], item["pinyin"]))


def candidate_set_id(dialect: str, segments: str, tone: str) -> str:
    digest = hashlib.sha256(f"{dialect}|{segments}|{tone}".encode()).hexdigest()[:16]
    return f"ipa_{dialect}_{digest}"


def build_outputs(
    ipa_rows: list[dict[str, object]], inventory: list[dict[str, object]], top_k: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pronunciations: dict[tuple[str, str, str], dict[str, object]] = {}
    line_rows = []
    for row in ipa_rows:
        dialect = row["phonemization"]["dialect"]
        mapped_syllables = []
        for item in row["ipa_syllables"]:
            key = (dialect, item["ipa_segments"], item["tone_chao"])
            set_id = candidate_set_id(*key)
            entry = pronunciations.setdefault(
                key,
                {
                    "schema_version": "pinyin-candidates-v1",
                    "candidate_set_id": set_id,
                    "source_dialect": dialect,
                    "source_ipa_segments": item["ipa_segments"],
                    "source_tone_chao": item["tone_chao"],
                    "source_forms": set(),
                    "occurrences": 0,
                },
            )
            entry["source_forms"].add(item["text"])
            entry["occurrences"] += 1
            mapped_syllables.append(
                {
                    "text": item["text"],
                    "source_ipa": item["ipa"],
                    "candidate_set_id": set_id,
                }
            )
        line_rows.append(
            {
                "schema_version": "poem-pinyin-v1",
                "line_id": row["line_id"],
                "work": row["work"],
                "form": row["form"],
                "line_role": row["line_role"],
                "text": row["text"],
                "pinyin_syllables": mapped_syllables,
            }
        )

    candidate_rows = []
    for key, entry in sorted(pronunciations.items(), key=lambda item: item[1]["candidate_set_id"]):
        entry["source_forms"] = sorted(entry["source_forms"])
        entry["candidates"] = rank_candidates(key[1], key[2], inventory, top_k)
        candidate_rows.append(entry)
    return candidate_rows, line_rows


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = round((len(values) - 1) * fraction)
    return round(sorted(values)[index], 6)


def build_report(
    ipa_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    inventory: list[dict[str, object]],
    top_k: int,
    lock: dict[str, object],
) -> dict[str, object]:
    top_scores = [row["candidates"][0]["score"] for row in candidate_rows]
    return {
        "schema_version": "pinyin-report-v1",
        "source_lines": len(ipa_rows),
        "output_lines": len(ipa_rows),
        "source_syllable_occurrences": sum(len(row["ipa_syllables"]) for row in ipa_rows),
        "unique_source_pronunciations": len(candidate_rows),
        "mandarin_inventory_size": len(inventory),
        "top_k": top_k,
        "weights": WEIGHTS,
        "mandarin_tone_contours": {str(key): value for key, value in MANDARIN_TONES.items()},
        "top1_score_distribution": {
            "min": min(top_scores), "p50": percentile(top_scores, 0.5),
            "p90": percentile(top_scores, 0.9), "p99": percentile(top_scores, 0.99),
            "max": max(top_scores),
        },
        "dependencies": lock,
    }


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    _atomic_write_text(path, "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map Vietnamese IPA to top-k attested Mandarin Pinyin syllables.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--lines-output", type=Path, default=DEFAULT_LINES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    lock = activate_and_verify_dependencies(args.deps, args.lock)
    ipa_rows = read_jsonl(args.input)
    inventory = build_inventory()
    candidate_rows, line_rows = build_outputs(ipa_rows, inventory, args.top_k)
    report = build_report(ipa_rows, candidate_rows, inventory, args.top_k, lock)
    _atomic_write_jsonl(args.candidates_output, candidate_rows)
    _atomic_write_jsonl(args.lines_output, line_rows)
    _atomic_write_text(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Mapped {report['source_syllable_occurrences']} syllable occurrences across "
        f"{report['unique_source_pronunciations']} pronunciations to an inventory of "
        f"{report['mandarin_inventory_size']} attested Pinyin readings."
    )
    print(args.candidates_output)
    print(args.lines_output)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
