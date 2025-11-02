"""Train a SentencePiece tokenizer on the cleaned corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def train_tokenizer(input_path: Path, output_dir: Path, vocab_size: int, model_type: str, character_coverage: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "spm"

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[],
        normalization_rule_name="nfkc",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/corpus/train.txt"), help="Training corpus (one document per line)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer"), help="Destination directory for tokenizer artifacts")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Number of subword units")
    parser.add_argument("--model-type", choices=["unigram", "bpe", "char", "word"], default="unigram", help="SentencePiece model type")
    parser.add_argument("--character-coverage", type=float, default=0.9995, help="Amount of characters covered by the model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_tokenizer(args.input, args.output, args.vocab_size, args.model_type, args.character_coverage)


if __name__ == "__main__":
    main()
