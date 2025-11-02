"""Tokenize the cleaned corpus into numpy arrays for model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


def encode_split(sp: spm.SentencePieceProcessor, path: Path, add_eos: bool) -> np.ndarray:
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc=f"Encoding {path.name}"):
            line = line.strip()
            if not line:
                continue
            encoded = sp.encode(line, out_type=int)
            if add_eos and sp.eos_id() >= 0:
                encoded.append(sp.eos_id())
            ids.extend(encoded)
    return np.asarray(ids, dtype=np.uint32)


def build_datasets(corpus_dir: Path, tokenizer_path: Path, output_dir: Path, add_eos: bool) -> None:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, int] = {
        "vocab_size": sp.get_piece_size(),
        "pad_id": sp.pad_id(),
        "unk_id": sp.unk_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
    }

    for split in ("train", "val", "test"):
        split_path = corpus_dir / f"{split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing corpus split: {split_path}")
        ids = encode_split(sp, split_path, add_eos)
        np.save(output_dir / f"{split}.npy", ids)

    with (output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("artifacts/corpus"), help="Directory containing train/val/test text files")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer/spm.model"), help="Path to the trained SentencePiece model")
    parser.add_argument("--output", type=Path, default=Path("artifacts/datasets"), help="Where to store the encoded numpy arrays")
    parser.add_argument("--append-eos", action="store_true", help="Append the tokenizer EOS token to every sequence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_datasets(args.corpus_dir, args.tokenizer, args.output, args.append_eos)


if __name__ == "__main__":
    main()
