"""
prepare_data.py — One-shot script to download TinyShakespeare and train the BPE tokenizer.

Run this once before training:
  python prepare_data.py

What it does:
  1. Downloads input.txt (~1MB of Shakespeare plays) from a public mirror.
  2. Trains an 8000-token BPE tokenizer on it.
  3. Prints a few stats so you can sanity-check everything is working.
"""

import os
import requests
from config import TrainConfig, ModelConfig
from tokenizer import LLMTokenizer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    tcfg = TrainConfig()
    mcfg = ModelConfig()

    # 1. Download dataset
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(tcfg.data_path):
        print(f"Downloading TinyShakespeare …")
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        with open(tcfg.data_path, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"Saved {len(r.text):,} chars → {tcfg.data_path}")
    else:
        print(f"Dataset already exists: {tcfg.data_path}")

    # 2. Train BPE tokenizer
    print(f"\nTraining BPE tokenizer (vocab_size={mcfg.vocab_size}) …")
    tok = LLMTokenizer.from_training(
        tcfg.data_path,
        tcfg.tokenizer_path,
        vocab_size=mcfg.vocab_size,
    )

    # 3. Stats
    sample = "To be, or not to be, that is the question:"
    ids = tok.encode(sample)
    print(f"\nSample: {sample!r}")
    print(f"  → {len(ids)} tokens: {ids[:10]} …")
    print(f"  → decoded: {tok.decode(ids)!r}")

    with open(tcfg.data_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    total_tokens = len(tok.encode(full_text))
    print(f"\nFull dataset: {total_tokens:,} tokens  ({total_tokens/1e6:.2f}M)")
    print(f"At batch_size={tcfg.batch_size}, context={mcfg.context_length}: "
          f"~{total_tokens // (tcfg.batch_size * mcfg.context_length)} steps/epoch")
    print("\nData preparation complete. You can now run: python train.py")


if __name__ == "__main__":
    main()
