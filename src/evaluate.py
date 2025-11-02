"""Evaluate a trained checkpoint on validation or test split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.data.dataset import TokenBlockDataset
from src.models.transformer import DecoderOnlyTransformer, ModelConfig


def evaluate_checkpoint(checkpoint_path: Path, split_path: Path, block_size: int, batch_size: int, num_workers: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError("Checkpoint does not contain configuration data")
    model_cfg = ModelConfig(**cfg["model"])

    model = DecoderOnlyTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = TokenBlockDataset(split_path, block_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, loss = model(x, y)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(min(20.0, avg_loss))
    print(f"Split: {split_path.name} | loss: {avg_loss:.4f} | perplexity: {ppl:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint file")
    parser.add_argument("--split", type=Path, default=Path("artifacts/datasets/val.npy"), help="Tokenized split to evaluate")
    parser.add_argument("--block-size", type=int, default=512, help="Sequence length used during training")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(args.checkpoint, args.split, args.block_size, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
