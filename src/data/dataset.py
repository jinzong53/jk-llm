"""PyTorch dataset wrappers for tokenized corpora."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset):
    """Creates sliding blocks of tokens for next-token prediction."""

    def __init__(self, data_path: Path, block_size: int) -> None:
        super().__init__()
        self.tokens = np.load(data_path, mmap_mode="r")
        if self.tokens.ndim != 1:
            raise ValueError("Expected a flat array of token ids")
        if len(self.tokens) <= block_size:
            raise ValueError("Token sequence must be longer than block size")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.block_size
        x = np.asarray(self.tokens[start:end], dtype=np.int64)
        y = np.asarray(self.tokens[start + 1 : end + 1], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)
