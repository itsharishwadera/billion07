"""
dataset.py — Text dataset for language model training.

How language model data works:
  We tokenize the entire corpus into one long flat array of token ids.
  A "sample" is a slice of length (context_length + 1):
    - input  x = tokens[i   : i + context_length]
    - target y = tokens[i+1 : i + context_length + 1]
  The model is trained to predict y[t] given x[0..t].
  Every position is a training signal — no wasted tokens.

This module:
  1. Reads a text file and tokenizes it (or loads pre-tokenized binary cache).
  2. Splits into train / val.
  3. Exposes a PyTorch Dataset and a fast DataLoader factory.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import LLMTokenizer


def build_token_array(
    text_path: str,
    tokenizer: LLMTokenizer,
    cache_path: str = "data/tokens.npy",
    force_retokenize: bool = False,
) -> np.ndarray:
    """
    Tokenize a text file and return a flat uint16 numpy array of token ids.
    Caches to disk so tokenization only runs once.

    uint16 can store token ids up to 65535, which covers any vocab ≤ 64K.
    """
    if os.path.exists(cache_path) and not force_retokenize:
        print(f"Loading cached tokens from {cache_path}")
        return np.load(cache_path)

    print(f"Tokenizing {text_path} …")
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text, add_eos=False)
    arr = np.array(ids, dtype=np.uint16)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, arr)
    print(f"Saved {len(arr):,} tokens to {cache_path}")
    return arr


class TokenDataset(Dataset):
    """
    Slices a flat token array into (input, target) pairs of length context_length.

    Each item:
      x: LongTensor of shape (context_length,) — input tokens
      y: LongTensor of shape (context_length,) — target tokens (shifted by 1)
    """

    def __init__(self, tokens: np.ndarray, context_length: int):
        # We need at least context_length + 1 tokens to form one sample
        self.tokens = tokens
        self.context_length = context_length
        self.n_samples = len(tokens) - context_length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.context_length + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def build_dataloaders(
    text_path: str,
    tokenizer: LLMTokenizer,
    context_length: int,
    batch_size: int,
    train_split: float = 0.9,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a text corpus.

    The split is done on the flat token array (not on documents), which is
    standard practice for pre-training.  val set is the last (1-train_split)
    fraction of the corpus.
    """
    tokens = build_token_array(text_path, tokenizer)

    split_idx = int(len(tokens) * train_split)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(
        f"Dataset: {len(tokens):,} tokens total — "
        f"{len(train_tokens):,} train / {len(val_tokens):,} val"
    )

    train_ds = TokenDataset(train_tokens, context_length)
    val_ds = TokenDataset(val_tokens, context_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick sanity check: print first batch shapes
    from config import ModelConfig, TrainConfig

    mcfg = ModelConfig()
    tcfg = TrainConfig()

    tok = LLMTokenizer.from_file(tcfg.tokenizer_path)
    train_loader, val_loader = build_dataloaders(
        tcfg.data_path,
        tok,
        mcfg.context_length,
        tcfg.batch_size,
        tcfg.train_split,
    )
    x, y = next(iter(train_loader))
    print(f"x shape: {x.shape}")   # (batch_size, context_length)
    print(f"y shape: {y.shape}")   # (batch_size, context_length)
