"""Train the decoder-only transformer on the prepared dataset."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import TokenBlockDataset
from src.models.transformer import DecoderOnlyTransformer, ModelConfig
from src.utils import count_parameters, load_yaml, set_seed


def build_dataloader(data_path: Path, block_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = TokenBlockDataset(data_path, block_size)
    drop_last = shuffle  # keep full evaluation batches while avoiding partial micro batches during training
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)


def evaluate(model: DecoderOnlyTransformer, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> float:
    model.eval()
    losses = []
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    elif max_batches is not None:
        total_batches = max_batches
    progress = tqdm(total=total_batches, desc="eval", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, loss = model(x, y)
            losses.append(loss.item())
            progress.update(1)
            if max_batches is not None and (idx + 1) >= max_batches:
                break
    model.train()
    progress.close()
    if not losses:
        raise RuntimeError("Validation loader did not yield any batches during evaluation.")
    return float(sum(losses) / len(losses))


def adjust_learning_rate(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        lr = base_lr * max(step, 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train(config_path: Path) -> None:
    cfg = load_yaml(config_path)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not found. Please ensure the 3090 GPU is visible before training.")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    optim_cfg = cfg["optimizer"]
    train_cfg = cfg["training"]

    dataset_dir = Path(data_cfg["dataset_dir"])
    train_loader = build_dataloader(dataset_dir / "train.npy", data_cfg["block_size"], data_cfg["batch_size"], data_cfg.get("num_workers", 4), shuffle=True)
    val_loader = build_dataloader(dataset_dir / "val.npy", data_cfg["block_size"], data_cfg["batch_size"], data_cfg.get("num_workers", 4), shuffle=False)

    model_config = ModelConfig(**model_cfg)
    model = DecoderOnlyTransformer(model_config).to(device)
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["lr"], betas=(0.9, 0.95), weight_decay=optim_cfg.get("weight_decay", 0.1))
    scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True))

    grad_clip = train_cfg.get("grad_clip", 1.0)
    total_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    eval_interval = train_cfg.get("eval_interval", 1000)
    log_interval = train_cfg.get("log_interval", 100)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    max_eval_batches = train_cfg.get("max_eval_batches")

    step = 0
    running_loss = 0.0
    start_time = time.time()
    micro_step = 0

    optimizer.zero_grad(set_to_none=True)
    progress_bar = tqdm(total=total_steps, desc="train", dynamic_ncols=True)

    while step < total_steps:
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=train_cfg.get("mixed_precision", True)):
                logits, loss = model(x, y)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

                step += 1
                progress_bar.update(1)
                lr = adjust_learning_rate(optimizer, optim_cfg["lr"], step, warmup_steps, total_steps)
                running_loss += loss.item() * grad_accum

                if step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    tokens_per_step = data_cfg["batch_size"] * data_cfg["block_size"] * grad_accum
                    elapsed = time.time() - start_time
                    print(f"step {step:06d} | loss {avg_loss:.4f} | lr {lr:.2e} | {tokens_per_step/elapsed:,.0f} tok/s")
                    running_loss = 0.0
                    start_time = time.time()

                if step % eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device, max_eval_batches)
                    ppl = math.exp(min(20.0, val_loss))
                    print(f"Eval step {step:06d} | val_loss {val_loss:.4f} | perplexity {ppl:.2f}")

                if step % train_cfg.get("checkpoint_interval", 5000) == 0:
                    ckpt_path = checkpoint_dir / f"model_step_{step:06d}.pt"
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "config": cfg,
                        "step": step,
                    }, ckpt_path)
                    print(f"Saved checkpoint to {ckpt_path}")

                if step >= total_steps:
                    break
        else:
            continue
        break

    progress_bar.close()

    final_ckpt = checkpoint_dir / "model_final.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg, "step": step}, final_ckpt)
    print(f"Training complete. Final checkpoint stored at {final_ckpt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_small.yaml"), help="Training configuration YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
