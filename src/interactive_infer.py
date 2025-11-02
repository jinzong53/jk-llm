"""Interactive/real-time inference REPL for the trained model.

Features:
- Load checkpoint and SentencePiece model (same as `src/infer.py`).
- REPL loop that accepts user input prompts and prints model responses.
- Optional streaming mode: prints tokens as they are sampled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import signal

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


def stream_generate(model: DecoderOnlyTransformer, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
    """Yield token ids one by one (on-device). This mirrors the model.generate logic but yields tokens so caller can stream output.

    Args:
        model: the loaded model in eval mode
        idx: input tensor of shape (1, seq_len)
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature
        top_k: optional top-k filtering

    Yields:
        next_token_id (int)
    """
    device = idx.device
    for _ in range(max_new_tokens):
        # keep only last context window
        idx_cond = idx[:, -model.config.n_ctx :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            min_vals = v[:, [-1]]
            logits = torch.where(logits < min_vals, torch.full_like(logits, float("-inf")), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_id = int(next_token[0, 0].item())
        idx = torch.cat((idx, next_token), dim=1)
        yield next_id


def generate_once(model: DecoderOnlyTransformer, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> list[int]:
    """Generate tokens in one shot using model.generate and return the full output ids as python list."""
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return out[0].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/spm.model"), help="SentencePiece model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="How many tokens to generate per reply")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 to disable)")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they are sampled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k = args.top_k if args.top_k > 0 else None
    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    # handle Ctrl-C to exit cleanly
    def _sigint_handler(signum, frame):
        print("\nExiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    print("Interactive inference REPL")
    print("Type your message and press Enter. Type 'exit' or 'quit' to leave.")
    print(f"Device: {device}; model context: {model.config.n_ctx}; stream: {args.stream}")

    while True:
        try:
            prompt = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye")
            break

        if not prompt:
            continue
        if prompt.strip().lower() in ("exit", "quit"):
            print("Bye")
            break

        input_ids = sp.encode(prompt, out_type=int)
        if not input_ids:
            input_ids = [sp.bos_id() if sp.bos_id() >= 0 else sp.unk_id()]

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        if args.stream:
            # stream tokens one by one
            printed = ""
            for tok_id in stream_generate(model, idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=top_k):
                # print piece form (may include special underline char)
                piece = sp.id_to_piece(tok_id)
                # write without newline
                sys.stdout.write(piece)
                sys.stdout.flush()
                printed += piece
            # final newline and optionally full decoded string
            sys.stdout.write("\n")
        else:
            out_ids = generate_once(model, idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=top_k)
            # out_ids contains prompt + generated tokens; get only generated portion
            gen_ids = out_ids[len(input_ids) :]
            text = sp.decode(gen_ids)
            print(text)


if __name__ == "__main__":
    main()
