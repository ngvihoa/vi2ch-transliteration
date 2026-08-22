#!/usr/bin/env python3
"""Apply the ranked lexical vocab directly to raw Vietnamese poem files."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_POEM_DIR = PROJECT_ROOT / "raw-collections" / "poem"
DEFAULT_VOCAB = PROJECT_ROOT / "pipelines" / "char2char_vocab" / "outputs" / "vocab.json"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "poem_char2char.jsonl"
DEFAULT_REVIEW = PIPELINE_ROOT / "outputs" / "poem_char2char_review.jsonl"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "poem_char2char_report.json"
PLACEHOLDER = "𠀗"
TOKEN_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).strip().lower().split())


def tokenize(text: str) -> list[str]:
    """Extract Unicode letter runs without treating punctuation as tokens."""
    return TOKEN_PATTERN.findall(normalize_text(text))


def load_vocab(path: Path) -> dict[str, list[str]]:
    resource = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(resource, dict):
        raise ValueError(f"Expected an object in {path}")
    for token, candidates in resource.items():
        if not isinstance(token, str) or not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Invalid vocab entry: {token!r} -> {candidates!r}")
        if not all(isinstance(char, str) and len(char) == 1 for char in candidates):
            raise ValueError(f"Candidates must be single characters: {token!r}")
    return resource


def map_tokens(tokens: list[str], vocab: dict[str, list[str]]) -> tuple[list[str], list[dict[str, object]]]:
    target_chars = []
    mappings = []
    for position, token in enumerate(tokens):
        candidates = vocab.get(token, [PLACEHOLDER])
        selected = candidates[0] if candidates else PLACEHOLDER
        if selected == PLACEHOLDER:
            method = "placeholder_in_vocab" if token in vocab else "placeholder_oov"
        else:
            method = "ranked_candidate"
        target_chars.append(selected)
        mappings.append({
            "position": position,
            "token": token,
            "char": selected,
            "method": method,
            "candidates": candidates,
        })
    return target_chars, mappings


def map_poem_directory(poem_dir: Path, vocab: dict[str, list[str]]) -> list[dict[str, object]]:
    rows = []
    for path in sorted(poem_dir.glob("*.vi.txt")):
        for line_no, original in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = normalize_text(original)
            if not text:
                continue
            tokens = tokenize(text)
            target_chars, mappings = map_tokens(tokens, vocab)
            if len(tokens) != len(target_chars):
                raise AssertionError("Token/target alignment invariant was violated")
            placeholder_positions = [
                index for index, char in enumerate(target_chars) if char == PLACEHOLDER
            ]
            rows.append({
                "schema_version": "raw-poem-char2char-v1",
                "line_id": f"{path.name.split('.', 1)[0]}_{line_no:04d}",
                "source_file": path.name,
                "source_line_no": line_no,
                "source_text": original,
                "normalized_text": text,
                "source_tokens": tokens,
                "target_chars": target_chars,
                "target_text": "".join(target_chars),
                "mapping": mappings,
                "token_count": len(tokens),
                "target_count": len(target_chars),
                "length_preserved": len(tokens) == len(target_chars),
                "placeholder_positions": placeholder_positions,
                "placeholder_count": len(placeholder_positions),
                "status": "contains_placeholder" if placeholder_positions else "mapped",
            })
    return rows


def build_report(rows: list[dict[str, object]]) -> dict[str, object]:
    file_counts = Counter(row["source_file"] for row in rows)
    total_tokens = sum(row["token_count"] for row in rows)
    placeholders = sum(row["placeholder_count"] for row in rows)
    return {
        "schema_version": "raw-poem-char2char-report-v1",
        "input_source": "raw-collections/poem/*.vi.txt",
        "vocab_source": "pipelines/char2char_vocab/outputs/vocab.json",
        "placeholder": PLACEHOLDER,
        "lines": len(rows),
        "tokens": total_tokens,
        "mapped_tokens": total_tokens - placeholders,
        "placeholder_tokens": placeholders,
        "mapped_token_rate": round((total_tokens - placeholders) / total_tokens, 6) if total_tokens else 0.0,
        "fully_mapped_lines": sum(row["status"] == "mapped" for row in rows),
        "lines_with_placeholder": sum(row["status"] == "contains_placeholder" for row in rows),
        "invalid_alignment_lines": sum(not row["length_preserved"] for row in rows),
        "lines_by_source_file": dict(sorted(file_counts.items())),
        "selection": "top_ranked_vocab_candidate",
        "uses_training_corpus": False,
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
    parser = argparse.ArgumentParser(description="Map raw poems with the ranked char2char vocab.")
    parser.add_argument("--poem-dir", type=Path, default=DEFAULT_POEM_DIR)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vocab = load_vocab(args.vocab)
    rows = map_poem_directory(args.poem_dir, vocab)
    review = [row for row in rows if row["status"] == "contains_placeholder"]
    report = build_report(rows)
    atomic_write(args.output, "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    atomic_write(
        args.review_output,
        "\n".join(json.dumps(row, ensure_ascii=False) for row in review) + ("\n" if review else ""),
    )
    atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Mapped {report['lines']} raw poem lines / {report['tokens']} tokens; "
        f"placeholder rate {1 - report['mapped_token_rate']:.1%}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
