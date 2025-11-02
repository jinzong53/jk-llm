"""Run text generation with a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from src.models.transformer import DecoderOnlyTransformer, ModelConfig


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[DecoderOnlyTransformer, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError("Checkpoint missing configuration")
    model_cfg = ModelConfig(**cfg["model"])
    model = DecoderOnlyTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def generate_text(checkpoint: Path, tokenizer_path: Path, prompt: str, max_new_tokens: int, temperature: float, top_k: int | None) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    input_ids = sp.encode(prompt, out_type=int)
    if not input_ids:
        input_ids = [sp.bos_id() if sp.bos_id() >= 0 else sp.unk_id()]

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    output_ids = generated[0].tolist()
    text = sp.decode(output_ids)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/spm.model"), help="SentencePiece model path")
    parser.add_argument("--prompt", type=str, default="你好，世界", help="Prompt text for generation")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="How many tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 to disable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k = args.top_k if args.top_k > 0 else None
    text = generate_text(args.checkpoint, args.tokenizer, args.prompt, args.max_new_tokens, args.temperature, top_k)
    print(text)


if __name__ == "__main__":
    main()
