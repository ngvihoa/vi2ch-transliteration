"""Vocabulary construction and coverage reporting."""

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .tokenization import tokenize_han, tokenize_vi


SPECIAL_VI = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
SPECIAL_HAN = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}


def build_vocabularies(data: Iterable[Mapping[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    """Build source and target vocabularies in frequency order."""
    vi_counter: Counter[str] = Counter()
    han_counter: Counter[str] = Counter()

    for item in data:
        vi_counter.update(tokenize_vi(item["vi"]))
        han_counter.update(tokenize_han(item["cn"]))

    vocab_vi = dict(SPECIAL_VI)
    vocab_han = dict(SPECIAL_HAN)
    for token, _ in vi_counter.most_common():
        vocab_vi.setdefault(token, len(vocab_vi))
    for token, _ in han_counter.most_common():
        vocab_han.setdefault(token, len(vocab_han))
    return vocab_vi, vocab_han


def save_vocabularies(
    vocab_vi: Mapping[str, int],
    vocab_han: Mapping[str, int],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save both vocabularies and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vi_path = output_dir / "vocab_vi.json"
    han_path = output_dir / "vocab_han.json"

    for path, vocab in ((vi_path, vocab_vi), (han_path, vocab_han)):
        with path.open("w", encoding="utf-8") as file:
            json.dump(vocab, file, ensure_ascii=False, indent=2)
    return vi_path, han_path


def calculate_coverage(
    data: Iterable[Mapping[str, str]],
    vocab: Mapping[str, int],
    tokenize: Callable[[str], list[str]] = tokenize_vi,
) -> dict[str, object]:
    """Calculate token and unique-token coverage for Vietnamese text."""
    all_words: Counter[str] = Counter()
    covered_words: Counter[str] = Counter()
    missing_words: Counter[str] = Counter()

    for item in data:
        for word in tokenize(item.get("vi", "")):
            all_words[word] += 1
            if word in vocab:
                covered_words[word] += 1
            else:
                missing_words[word] += 1

    total_tokens = all_words.total()
    covered_tokens = covered_words.total()
    covered_unique = set(all_words).intersection(vocab)
    return {
        "total_tokens": total_tokens,
        "covered_tokens": covered_tokens,
        "token_coverage": covered_tokens / total_tokens * 100 if total_tokens else 0,
        "total_unique_words": len(all_words),
        "covered_unique_words": len(covered_unique),
        "unique_coverage": len(covered_unique) / len(all_words) * 100 if all_words else 0,
        "all_words": all_words,
        "covered_words": covered_words,
        "missing_words": missing_words,
    }
