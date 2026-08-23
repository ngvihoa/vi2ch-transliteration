"""Small reusable helpers for model training and serialization."""

import json
from pathlib import Path
from typing import Mapping

import torch


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Move every tensor in a model batch to a device."""
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_loss(model, loader, device: torch.device | str) -> float:
    """Return average loss while preserving the model's train/eval state."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        outputs = model(**move_batch_to_device(batch, device))
        total_loss += outputs["loss"].item()
        total_batches += 1
    model.train(was_training)
    if not total_batches:
        raise ValueError("Cannot evaluate an empty data loader.")
    return total_loss / total_batches


def save_checkpoint(model, config, path: str | Path) -> Path:
    """Save model weights and configuration in a resumable checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, path)
    return path


def save_model_bundle(
    model,
    config,
    vocab_vi: Mapping[str, int],
    vocab_han: Mapping[str, int],
    output_dir: str | Path,
) -> Path:
    """Save final weights, config, and both vocabularies in one directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    payloads = {
        "config.json": config.to_dict(),
        "vocab_vi.json": vocab_vi,
        "vocab_han.json": vocab_han,
    }
    for filename, payload in payloads.items():
        with (output_dir / filename).open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    return output_dir
