#!/usr/bin/env python3
"""Evaluate VietHanBERT on a Vietnamese-to-Han parallel CSV test set.

The primary metrics are character based because the model generates an
unsegmented Han-character sequence.  The script downloads a published model
from Hugging Face, performs batched greedy decoding, and writes both aggregate
metrics and row-level predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TEST_FILE = (
    PROJECT_DIR / "pipelines/poetry_dataset_split/outputs/poem.test.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_MODEL_ID = "noah-nguyen-297/VietHanBERT-vi2cn-v1"

# ASCII and common CJK punctuation. Whitespace is excluded separately.
_PUNCTUATION_RE = re.compile(
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"
    r"。！？；：，、（）【】「」『』〈〉《》“”‘’…—·]"
)


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def distance(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def normalize_han(text: str, *, ignore_punctuation: bool = False) -> str:
    """NFC-normalize Han text and remove whitespace.

    Punctuation is retained for strict evaluation.  It can be removed for the
    auxiliary lexical-only view of the metrics.
    """
    normalized = "".join(unicodedata.normalize("NFC", text).split())
    if ignore_punctuation:
        normalized = _PUNCTUATION_RE.sub("", normalized)
    return normalized


def edit_counts(reference: str, prediction: str) -> EditCounts:
    """Return one optimal Levenshtein alignment's S/D/I counts."""
    # Each cell stores (distance, substitutions, deletions, insertions).
    previous = [(index, 0, index, 0) for index in range(len(reference) + 1)]
    for pred_char in prediction:
        current = [(previous[0][0] + 1, 0, 0, previous[0][3] + 1)]
        for ref_index, ref_char in enumerate(reference, start=1):
            if ref_char == pred_char:
                diagonal = previous[ref_index - 1]
            else:
                cell = previous[ref_index - 1]
                diagonal = (cell[0] + 1, cell[1] + 1, cell[2], cell[3])

            cell = current[ref_index - 1]
            deletion = (cell[0] + 1, cell[1], cell[2] + 1, cell[3])
            cell = previous[ref_index]
            insertion = (cell[0] + 1, cell[1], cell[2], cell[3] + 1)
            # Deterministic tie-breaking: match/substitute, delete, insert.
            current.append(min((diagonal, deletion, insertion), key=lambda x: x[0]))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return EditCounts(substitutions, deletions, insertions)


def lcs_length(first: str, second: str) -> int:
    """Length of the longest common subsequence using O(min(m, n)) memory."""
    if len(first) < len(second):
        first, second = second, first
    previous = [0] * (len(second) + 1)
    for first_char in first:
        current = [0]
        for index, second_char in enumerate(second, start=1):
            if first_char == second_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def news_fscore(reference: str, prediction: str) -> float:
    """NEWS shared-task top-1 fuzzy F-score based on character LCS."""
    denominator = len(reference) + len(prediction)
    if denominator == 0:
        return 1.0
    return 2.0 * lcs_length(reference, prediction) / denominator


def load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"vi", "cn"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns {sorted(required)}: {path}")
        rows = [
            {"vi": row["vi"].strip(), "cn": row["cn"].strip()}
            for row in reader
            if row.get("vi", "").strip() and row.get("cn", "").strip()
        ]
    return rows[:limit] if limit is not None else rows


def resolve_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_bundle(model_id: str, revision: str | None, device):
    """Load config/vocab/weights from HF with the repository's model classes."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "Missing model dependencies. Run: pip install -r evaluate/requirements.txt"
        ) from error

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    try:
        from model.configuration_viet_han_bert import VietHanBertConfig
        from model.modeling_viet_han_bert import VietHanBertModel
        from model.tokenization import VietHanTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Could not import the VietHanBERT implementation or one of its "
            "dependencies. Run from the vi2ch-model checkout and install the "
            "pinned versions in evaluate/requirements.txt."
        ) from error

    download_args: dict[str, Any] = {"repo_id": model_id}
    if revision:
        download_args["revision"] = revision
    vi_vocab = hf_hub_download(filename="vocab_vi.json", **download_args)
    vi_vocab_path = Path(vi_vocab)
    resolved_revision = (
        vi_vocab_path.parent.name
        if vi_vocab_path.parent.parent.name == "snapshots"
        else revision
    )
    # Pin the rest of the bundle to the exact snapshot resolved by the first
    # download, so a concurrent update to the Hub's main branch cannot mix files.
    if resolved_revision:
        download_args["revision"] = resolved_revision
    han_vocab = hf_hub_download(filename="vocab_han.json", **download_args)
    config_path = hf_hub_download(filename="config.json", **download_args)
    weights_path = hf_hub_download(filename="model.safetensors", **download_args)

    tokenizer = VietHanTokenizer(vi_vocab, han_vocab)
    config = VietHanBertConfig.from_json_file(config_path)
    model = VietHanBertModel(config)
    model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
    model.to(device).eval()
    return model, tokenizer, config, resolved_revision


def _encode_sources(
    texts: Sequence[str], tokenizer, max_source_length: int, device
):
    import torch

    encoded = []
    for text in texts:
        ids = tokenizer.encode_vi(text)
        if len(ids) > max_source_length:
            ids = ids[:max_source_length]
            ids[-1] = tokenizer.vi_sep_id
        encoded.append(ids)
    width = max(map(len, encoded))
    input_ids = torch.full(
        (len(encoded), width), tokenizer.vi_pad_id, dtype=torch.long, device=device
    )
    for row_index, ids in enumerate(encoded):
        input_ids[row_index, : len(ids)] = torch.tensor(ids, device=device)
    return input_ids, (input_ids != tokenizer.vi_pad_id).long()


def _teacher_forced_stats(
    model, tokenizer, references: Sequence[str], input_ids, attention_mask, device
) -> tuple[float, int]:
    import torch
    import torch.nn.functional as functional

    encoded = [tokenizer.encode_han(text) for text in references]
    width = max(len(ids) - 1 for ids in encoded)
    decoder_ids = torch.full(
        (len(encoded), width), tokenizer.han_pad_id, dtype=torch.long, device=device
    )
    labels = torch.full(
        (len(encoded), width), -100, dtype=torch.long, device=device
    )
    for row_index, ids in enumerate(encoded):
        decoder = ids[:-1]
        target = ids[1:]
        decoder_ids[row_index, : len(decoder)] = torch.tensor(decoder, device=device)
        labels[row_index, : len(target)] = torch.tensor(target, device=device)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids,
        )["logits"]
        nll_sum = functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
    token_count = int((labels != -100).sum().item())
    return float(nll_sum.item()), token_count


def generate_batch(
    model,
    tokenizer,
    sources: Sequence[str],
    references: Sequence[str],
    device,
    max_source_length: int,
    max_new_tokens: int,
) -> tuple[list[str], float, int]:
    import torch

    input_ids, attention_mask = _encode_sources(
        sources, tokenizer, max_source_length, device
    )
    nll_sum, token_count = _teacher_forced_stats(
        model, tokenizer, references, input_ids, attention_mask, device
    )
    decoder_ids = torch.full(
        (len(sources), 1), tokenizer.han_bos_id, dtype=torch.long, device=device
    )
    finished = torch.zeros(len(sources), dtype=torch.bool, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_ids,
            )["logits"]
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, tokenizer.han_pad_id),
                next_tokens,
            )
            decoder_ids = torch.cat((decoder_ids, next_tokens[:, None]), dim=1)
            finished |= next_tokens == tokenizer.han_eos_id
            if bool(finished.all()):
                break

    predictions = [
        tokenizer.decode_han(token_ids) for token_ids in decoder_ids.cpu().tolist()
    ]
    return predictions, nll_sum, token_count


def run_inference(
    model,
    tokenizer,
    rows: Sequence[dict[str, str]],
    device,
    batch_size: int,
    max_source_length: int,
    max_new_tokens: int,
    log_every: int,
) -> tuple[list[str], float, int]:
    predictions: list[str] = []
    total_nll = 0.0
    total_tokens = 0
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_predictions, nll_sum, token_count = generate_batch(
            model,
            tokenizer,
            [row["vi"] for row in batch],
            [row["cn"] for row in batch],
            device,
            max_source_length,
            max_new_tokens,
        )
        predictions.extend(batch_predictions)
        total_nll += nll_sum
        total_tokens += token_count
        processed = start + len(batch)
        if processed == len(rows) or (log_every and processed % log_every < batch_size):
            elapsed = time.perf_counter() - started
            print(f"Generated {processed}/{len(rows)} samples ({processed / elapsed:.2f}/s)")
    return predictions, total_nll, total_tokens


def compute_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    ignore_punctuation: bool = False,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | bool]]]:
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError as error:
        raise RuntimeError(
            "Missing metric dependency. Run: pip install -r evaluate/requirements.txt"
        ) from error

    normalized_references = [
        normalize_han(text, ignore_punctuation=ignore_punctuation) for text in references
    ]
    normalized_predictions = [
        normalize_han(text, ignore_punctuation=ignore_punctuation) for text in predictions
    ]
    details: list[dict[str, float | int | bool]] = []
    total_substitutions = total_deletions = total_insertions = 0
    exact_count = 0
    fscore_sum = 0.0

    for reference, prediction in zip(
        normalized_references, normalized_predictions, strict=True
    ):
        counts = edit_counts(reference, prediction)
        exact = reference == prediction
        fscore = news_fscore(reference, prediction)
        exact_count += int(exact)
        fscore_sum += fscore
        total_substitutions += counts.substitutions
        total_deletions += counts.deletions
        total_insertions += counts.insertions
        details.append(
            {
                "exact_match": exact,
                "edit_distance": counts.distance,
                "substitutions": counts.substitutions,
                "deletions": counts.deletions,
                "insertions": counts.insertions,
                "news_fscore": fscore,
            }
        )

    sample_count = len(normalized_references)
    reference_characters = sum(map(len, normalized_references))
    prediction_characters = sum(map(len, normalized_predictions))
    edit_distance = total_substitutions + total_deletions + total_insertions
    cer = edit_distance / reference_characters if reference_characters else 0.0
    # tokenize="char" makes the intended BLEU tokenization reproducible.
    bleu = BLEU(tokenize="char", effective_order=True)
    chrf = CHRF(char_order=6, word_order=0, beta=2)
    suffix = "_no_punct" if ignore_punctuation else ""
    metrics: dict[str, float | int | str] = {
        f"sample_count{suffix}": sample_count,
        f"exact_match_count{suffix}": exact_count,
        f"exact_match_accuracy{suffix}": exact_count / sample_count,
        f"cer{suffix}": cer,
        f"character_accuracy{suffix}": 1.0 - cer,
        f"news_mean_fscore{suffix}": fscore_sum / sample_count,
        f"character_bleu{suffix}": bleu.corpus_score(
            normalized_predictions, [normalized_references]
        ).score,
        f"chrf2{suffix}": chrf.corpus_score(
            normalized_predictions, [normalized_references]
        ).score,
        f"reference_characters{suffix}": reference_characters,
        f"prediction_characters{suffix}": prediction_characters,
        f"substitutions{suffix}": total_substitutions,
        f"deletions{suffix}": total_deletions,
        f"insertions{suffix}": total_insertions,
    }
    if not ignore_punctuation:
        metrics["character_bleu_signature"] = str(bleu.get_signature())
        metrics["chrf2_signature"] = str(chrf.get_signature())
    return metrics, details


def write_predictions(
    path: Path,
    rows: Sequence[dict[str, str]],
    predictions: Sequence[str],
    details: Sequence[dict[str, float | int | bool]],
) -> None:
    fieldnames = [
        "vi",
        "cn_ref",
        "cn_pred",
        "exact_match",
        "edit_distance",
        "substitutions",
        "deletions",
        "insertions",
        "news_fscore",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, prediction, detail in zip(rows, predictions, details, strict=True):
            writer.writerow(
                {"vi": row["vi"], "cn_ref": row["cn"], "cn_pred": prediction, **detail}
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", help="Optional HF branch, tag, or commit hash")
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--max-source-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N rows (smoke test)")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.test_file, args.limit)
    if not rows:
        raise ValueError(f"No non-empty vi/cn rows found in {args.test_file}")
    device = resolve_device(args.device)
    print(f"Loading {args.model_id} on {device} ...")
    model, tokenizer, config, resolved_revision = load_model_bundle(
        args.model_id, args.revision, device
    )
    max_source_length = args.max_source_length or config.max_position_embeddings
    max_new_tokens = args.max_new_tokens or (
        config.max_target_position_embeddings - 1
    )
    if max_source_length > config.max_position_embeddings:
        raise ValueError(
            f"--max-source-length cannot exceed {config.max_position_embeddings}"
        )
    if max_new_tokens >= config.max_target_position_embeddings:
        raise ValueError(
            "--max-new-tokens must be smaller than max_target_position_embeddings "
            f"({config.max_target_position_embeddings})"
        )

    started = time.perf_counter()
    predictions, nll_sum, target_token_count = run_inference(
        model,
        tokenizer,
        rows,
        device,
        args.batch_size,
        max_source_length,
        max_new_tokens,
        args.log_every,
    )
    strict_metrics, details = compute_metrics(
        [row["cn"] for row in rows], predictions
    )
    no_punct_metrics, _ = compute_metrics(
        [row["cn"] for row in rows], predictions, ignore_punctuation=True
    )
    token_nll = nll_sum / target_token_count
    metrics: dict[str, Any] = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "test_file": str(args.test_file.resolve()),
        "device": str(device),
        "decoding": "greedy",
        "max_source_length": max_source_length,
        "max_new_tokens": max_new_tokens,
        **strict_metrics,
        **no_punct_metrics,
        "teacher_forced_token_nll": token_nll,
        "teacher_forced_perplexity": math.exp(token_nll),
        "teacher_forced_target_tokens": target_token_count,
        "elapsed_seconds": time.perf_counter() - started,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "predictions.csv"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")
    write_predictions(predictions_path, rows, predictions, details)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
