#!/usr/bin/env python3
"""Build a Vietnamese -> Hanzi 1:N lexical vocabulary from CVDICT."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_DICTIONARY = PROJECT_ROOT / "raw-collections" / "CVDICT.u8"
DEFAULT_HANVIET = PROJECT_ROOT / "raw-collections" / "hanviet.csv"
DEFAULT_KVIETNAMESE = PROJECT_ROOT / "raw-collections" / "kVietnamese.json"
DEFAULT_CURATED = PIPELINE_ROOT / "resources" / "curated_vocab.json"
DEFAULT_VOCAB = PIPELINE_ROOT / "outputs" / "vocab.json"
DEFAULT_EVIDENCE = PIPELINE_ROOT / "outputs" / "vocab_evidence.jsonl"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "vocab_report.json"
ENTRY_PATTERN = re.compile(r"^(\S+)\s+(\S+)\s+\[[^]]*\]\s+/(.*)/$")
ANNOTATION_PATTERN = re.compile(r"\([^)]*\)|\[[^]]*\]")
WORD_PATTERN = re.compile(r"^[^\W\d_]+$", re.UNICODE)


def is_hanzi(char: str) -> bool:
    return len(char) == 1 and "\u3400" <= char <= "\u9fff"


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().lower()


def lexical_glosses(raw_senses: str) -> list[str]:
    """Return only standalone one-token glosses; never split a phrase into tokens."""
    glosses = []
    for raw_sense in raw_senses.split("/"):
        sense = normalize(ANNOTATION_PATTERN.sub("", raw_sense))
        for fragment in re.split(r"[;,]", sense):
            gloss = normalize(fragment)
            if WORD_PATTERN.fullmatch(gloss):
                glosses.append(gloss)
    return glosses


def parse_cvdict(
    path: Path,
) -> tuple[dict[str, Counter[str]], set[str], dict[str, int]]:
    evidence: dict[str, Counter[str]] = defaultdict(Counter)
    inventory: set[str] = set()
    stats = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stats["lines"] += 1
            match = ENTRY_PATTERN.match(line.strip())
            if not match:
                continue
            stats["parsed_entries"] += 1
            traditional = match.group(1)
            simplified = match.group(2)
            inventory.update(char for char in (traditional, simplified) if is_hanzi(char))
            if not is_hanzi(simplified):
                stats["non_single_hanzi_entries"] += 1
                continue
            glosses = lexical_glosses(match.group(3))
            stats["standalone_gloss_occurrences"] += len(glosses)
            for gloss in glosses:
                evidence[gloss][simplified] += 1
    return evidence, inventory, dict(stats)


def parse_hanviet(
    path: Path,
) -> tuple[dict[str, Counter[str]], set[str], dict[str, int]]:
    evidence: dict[str, Counter[str]] = defaultdict(Counter)
    inventory: set[str] = set()
    stats = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stats["rows"] += 1
            char = row.get("char", "").strip()
            if not is_hanzi(char):
                stats["invalid_char_rows"] += 1
                continue
            inventory.add(char)
            try:
                readings = ast.literal_eval(row.get("hanviet", ""))
            except (SyntaxError, ValueError):
                stats["invalid_reading_rows"] += 1
                continue
            if not isinstance(readings, list):
                stats["invalid_reading_rows"] += 1
                continue
            for reading in readings:
                token = normalize(reading) if isinstance(reading, str) else ""
                if WORD_PATTERN.fullmatch(token):
                    evidence[token][char] += 1
                    stats["reading_occurrences"] += 1
    return evidence, inventory, dict(stats)


def parse_kvietnamese(
    path: Path, allowed_hanzi: set[str]
) -> tuple[dict[str, Counter[str]], dict[str, int]]:
    evidence: dict[str, Counter[str]] = defaultdict(Counter)
    stats = Counter()
    resource = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(resource, dict):
        raise ValueError(f"Expected an object in {path}")
    for char, readings in resource.items():
        stats["entries"] += 1
        # kVietnamese also contains purpose-built Nom glyphs. Only reuse a
        # character already attested by the Chinese dictionaries.
        if char not in allowed_hanzi:
            stats["nom_or_unverified_chars_excluded"] += 1
            continue
        if not isinstance(readings, list):
            stats["invalid_entries"] += 1
            continue
        for reading in readings:
            token = normalize(reading) if isinstance(reading, str) else ""
            if WORD_PATTERN.fullmatch(token):
                evidence[token][char] += 1
                stats["reading_occurrences"] += 1
    return evidence, dict(stats)


def load_curated(path: Path) -> dict[str, list[str]]:
    resource = json.loads(path.read_text(encoding="utf-8"))
    mappings = resource.get("tokens", {})
    if not isinstance(mappings, dict):
        raise ValueError("curated tokens must be an object")
    normalized = {}
    for token, chars in mappings.items():
        key = normalize(token)
        if not key or not isinstance(chars, list) or not chars or not all(is_hanzi(char) for char in chars):
            raise ValueError(f"Invalid curated mapping: {token!r} -> {chars!r}")
        normalized[key] = list(dict.fromkeys(chars))
    return normalized


def build_vocab(
    evidence_by_source: dict[str, dict[str, Counter[str]]],
    curated: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    vocab = {}
    rows = []
    source_order = ("cvdict_standalone_gloss", "kvietnamese_hanzi_reading", "hanviet_reading")
    tokens = set(curated)
    for evidence in evidence_by_source.values():
        tokens.update(evidence)
    for token in sorted(tokens):
        chars = list(curated.get(token, []))
        for source in source_order:
            counts = evidence_by_source[source].get(token, Counter())
            chars.extend(sorted(counts, key=lambda char: (-counts[char], char)))
        chars = list(dict.fromkeys(chars))
        vocab[token] = chars
        rows.append({
            "schema_version": "lexical-vocab-evidence-v1",
            "token": token,
            "candidates": [
                {
                    "char": char,
                    "sources": (["curated"] if char in curated.get(token, []) else [])
                        + [
                            source for source in source_order
                            if evidence_by_source[source].get(token, Counter())[char]
                        ],
                    "evidence_counts": {
                        source: evidence_by_source[source].get(token, Counter())[char]
                        for source in source_order
                        if evidence_by_source[source].get(token, Counter())[char]
                    },
                }
                for char in chars
            ],
        })
    return vocab, rows


def build_report(
    vocab: dict[str, list[str]], parser_stats: dict[str, dict[str, int]],
    evidence_by_source: dict[str, dict[str, Counter[str]]], curated: dict[str, list[str]]
) -> dict[str, object]:
    candidate_counts = Counter(len(chars) for chars in vocab.values())
    ambiguous = sum(count for size, count in candidate_counts.items() if size > 1)
    return {
        "schema_version": "lexical-vocab-report-v1",
        "mapping_contract": "one_vietnamese_token_to_one_or_more_hanzi_candidates",
        "sources_by_priority": [
            "curated",
            "raw-collections/CVDICT.u8",
            "raw-collections/kVietnamese.json (verified Hanzi only)",
            "raw-collections/hanviet.csv",
        ],
        "excluded_sources": [
            "raw-collections/chinese-vietnamese.csv",
            "raw-collections/cn-vi/",
            "pipelines/synthetic_poetry/",
            "vocab_old.json",
        ],
        "vietnamese_tokens": len(vocab),
        "total_token_hanzi_pairs": sum(len(chars) for chars in vocab.values()),
        "ambiguous_tokens": ambiguous,
        "ambiguous_token_rate": round(ambiguous / len(vocab), 6) if vocab else 0.0,
        "curated_tokens": len(curated),
        "tokens_by_source": {
            source: len(evidence) for source, evidence in evidence_by_source.items()
        },
        "candidate_count_distribution": dict(sorted(candidate_counts.items())),
        "parsers": parser_stats,
        "caveats": [
            "Only standalone one-token CVDICT glosses are accepted; multiword glosses are not decomposed.",
            "kVietnamese characters absent from the CVDICT/hanviet Hanzi inventory are excluded as Nom or unverified glyphs.",
            "kVietnamese and hanviet are reading evidence, so their candidates rank below curated and semantic CVDICT candidates.",
        ],
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a semantic Vietnamese -> Hanzi 1:N vocab.")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--hanviet", type=Path, default=DEFAULT_HANVIET)
    parser.add_argument("--kvietnamese", type=Path, default=DEFAULT_KVIETNAMESE)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--vocab-output", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cvdict, cvdict_inventory, cvdict_stats = parse_cvdict(args.dictionary)
    hanviet, hanviet_inventory, hanviet_stats = parse_hanviet(args.hanviet)
    kvietnamese, kvietnamese_stats = parse_kvietnamese(
        args.kvietnamese, cvdict_inventory | hanviet_inventory
    )
    evidence_by_source = {
        "cvdict_standalone_gloss": cvdict,
        "kvietnamese_hanzi_reading": kvietnamese,
        "hanviet_reading": hanviet,
    }
    curated = load_curated(args.curated)
    vocab, evidence_rows = build_vocab(evidence_by_source, curated)
    report = build_report(
        vocab,
        {"cvdict": cvdict_stats, "kvietnamese": kvietnamese_stats, "hanviet": hanviet_stats},
        evidence_by_source,
        curated,
    )
    atomic_write(args.vocab_output, json.dumps(vocab, ensure_ascii=False, indent=2) + "\n")
    atomic_write(
        args.evidence_output,
        "\n".join(json.dumps(row, ensure_ascii=False) for row in evidence_rows) + "\n",
    )
    atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Built {report['vietnamese_tokens']} tokens / "
        f"{report['total_token_hanzi_pairs']} token-Hanzi pairs from approved lexical sources."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
