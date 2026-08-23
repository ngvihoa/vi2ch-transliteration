"""Dataset loading, encoding, and batch collation."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenization import VietHanTokenizer


def load_json(path: str | Path) -> Any:
    """Load a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


class VietHanDataset(Dataset):
    """Encode Vietnamese-Han records for teacher-forced training."""

    def __init__(
        self,
        data: Sequence[Mapping[str, str]],
        tokenizer: VietHanTokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        source_ids = self.tokenizer.encode_vi(item["vi"])[: self.max_source_length]
        target_ids = self.tokenizer.encode_han(item["cn"])[: self.max_target_length]
        return {
            "input_ids": torch.tensor(source_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(target_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(target_ids[1:], dtype=torch.long),
        }


class VietHanCollator:
    """Pad encoded records and create their source attention masks."""

    def __init__(self, tokenizer: VietHanTokenizer) -> None:
        self.vi_pad_id = tokenizer.vi_pad_id
        self.han_pad_id = tokenizer.han_pad_id

    def __call__(self, batch: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_src_len = max(len(item["input_ids"]) for item in batch)
        max_tgt_len = max(len(item["decoder_input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        decoder_input_ids = []
        labels = []

        for item in batch:
            src = item["input_ids"]
            tgt = item["decoder_input_ids"]
            label = item["labels"]
            src_pad_len = max_src_len - len(src)
            tgt_pad_len = max_tgt_len - len(tgt)

            input_ids.append(torch.cat((src, torch.full((src_pad_len,), self.vi_pad_id))))
            attention_mask.append(torch.cat((torch.ones_like(src), torch.zeros(src_pad_len, dtype=torch.long))))
            decoder_input_ids.append(torch.cat((tgt, torch.full((tgt_pad_len,), self.han_pad_id))))
            labels.append(torch.cat((label, torch.full((tgt_pad_len,), -100))))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "decoder_input_ids": torch.stack(decoder_input_ids),
            "labels": torch.stack(labels),
        }


def create_data_loader(
    data: Sequence[Mapping[str, str]],
    tokenizer: VietHanTokenizer,
    *,
    batch_size: int,
    max_source_length: int = 128,
    max_target_length: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a dataset, collator, and data loader with one call."""
    dataset = VietHanDataset(data, tokenizer, max_source_length, max_target_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=VietHanCollator(tokenizer),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
