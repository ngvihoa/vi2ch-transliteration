"""Greedy decoding and generation evaluation helpers."""

from typing import Iterable, Mapping, Sequence

import torch

from .tokenization import VietHanTokenizer


@torch.no_grad()
def generate_han(
    model,
    tokenizer: VietHanTokenizer,
    text: str,
    device: torch.device | str,
    *,
    max_source_length: int = 128,
    max_length: int = 128,
) -> str:
    """Greedily generate Han characters for one Vietnamese input."""
    model.eval()
    source_ids = tokenizer.encode_vi(text)[:max_source_length]
    input_ids = torch.tensor([source_ids], dtype=torch.long, device=device)
    attention_mask = (input_ids != tokenizer.vi_pad_id).long()
    decoder_ids = torch.tensor([[tokenizer.han_bos_id]], dtype=torch.long, device=device)

    for _ in range(max_length):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids,
        )
        next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        decoder_ids = torch.cat((decoder_ids, next_token), dim=1)
        if next_token.item() == tokenizer.han_eos_id:
            break
    return tokenizer.decode_han(decoder_ids[0].tolist())


def generate_dataset(
    model,
    tokenizer: VietHanTokenizer,
    data: Sequence[Mapping[str, str]],
    device: torch.device | str,
    *,
    max_source_length: int = 128,
    max_length: int = 128,
    log_every: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Generate predictions and return sources, references, and predictions."""
    sources = []
    references = []
    predictions = []
    for index, item in enumerate(data, start=1):
        sources.append(item["vi"])
        references.append(item["cn"])
        predictions.append(
            generate_han(
                model,
                tokenizer,
                item["vi"],
                device,
                max_source_length=max_source_length,
                max_length=max_length,
            )
        )
        if log_every and index % log_every == 0:
            print(f"Processed {index}/{len(data)}")
    return sources, references, predictions


def character_bleu(predictions: Iterable[str], references: Iterable[str]):
    """Compute SacreBLEU after splitting each sentence into characters."""
    import sacrebleu

    char_predictions = [" ".join(prediction) for prediction in predictions]
    char_references = [" ".join(reference) for reference in references]
    return sacrebleu.corpus_bleu(char_predictions, [char_references])
