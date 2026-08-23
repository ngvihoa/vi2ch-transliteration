"""Tokenization utilities for Vietnamese-to-Han transliteration."""

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


_ASCII_PUNCTUATION = re.compile(r'([!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~])')
_CJK_PUNCTUATION = re.compile(r"([。！？；：，、（）「」『』“”‘’…])")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Unicode and collapse consecutive whitespace."""
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize_vi(text: str) -> list[str]:
    """Split Vietnamese text on whitespace and isolate punctuation."""
    text = normalize_text(text)
    text = _ASCII_PUNCTUATION.sub(r" \1 ", text)
    text = _CJK_PUNCTUATION.sub(r" \1 ", text)
    return text.split()


def tokenize_han(text: str) -> list[str]:
    """Tokenize Han text at character level, ignoring whitespace."""
    return list(normalize_text(text).replace(" ", ""))


class VietHanTokenizer:
    """Vocabulary-backed tokenizer for Vietnamese source and Han target text."""

    def __init__(self, vi_vocab_path: str | Path, han_vocab_path: str | Path) -> None:
        with Path(vi_vocab_path).open("r", encoding="utf-8") as file:
            self.vi_vocab = json.load(file)
        with Path(han_vocab_path).open("r", encoding="utf-8") as file:
            self.han_vocab = json.load(file)

        self.vi_id2token = {int(value): key for key, value in self.vi_vocab.items()}
        self.han_id2token = {int(value): key for key, value in self.han_vocab.items()}

        self.vi_pad_id = self.vi_vocab["<PAD>"]
        self.vi_unk_id = self.vi_vocab["<UNK>"]
        self.vi_cls_id = self.vi_vocab["<CLS>"]
        self.vi_sep_id = self.vi_vocab["<SEP>"]

        self.han_pad_id = self.han_vocab["<PAD>"]
        self.han_unk_id = self.han_vocab["<UNK>"]
        self.han_bos_id = self.han_vocab["<BOS>"]
        self.han_eos_id = self.han_vocab["<EOS>"]

    normalize_text = staticmethod(normalize_text)
    tokenize_vi = staticmethod(tokenize_vi)
    tokenize_han = staticmethod(tokenize_han)

    def encode_vi(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.vi_vocab.get(token, self.vi_unk_id) for token in tokenize_vi(text)]
        if add_special_tokens:
            ids = [self.vi_cls_id, *ids, self.vi_sep_id]
        return ids

    def decode_vi(self, ids: Iterable[int]) -> str:
        special_ids = {self.vi_pad_id, self.vi_cls_id, self.vi_sep_id}
        tokens = [
            self.vi_id2token.get(int(idx), "<UNK>")
            for idx in ids
            if int(idx) not in special_ids
        ]

        text = " ".join(tokens)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"\s+([。！？；：，、）】」』])", r"\1", text)
        return re.sub(r"([（【「『])\s+", r"\1", text)

    def encode_han(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids = [self.han_vocab.get(token, self.han_unk_id) for token in tokenize_han(text)]
        if add_bos:
            ids.insert(0, self.han_bos_id)
        if add_eos:
            ids.append(self.han_eos_id)
        return ids

    def decode_han(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.han_pad_id, self.han_bos_id, self.han_eos_id}
        tokens = []
        for idx in ids:
            idx = int(idx)
            if skip_special_tokens and idx in special_ids:
                continue
            tokens.append(self.han_id2token.get(idx, "<UNK>"))
        return "".join(tokens)

    @property
    def vi_vocab_size(self) -> int:
        return len(self.vi_vocab)

    @property
    def han_vocab_size(self) -> int:
        return len(self.han_vocab)
