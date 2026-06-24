"""
tokenizer.py — BPE tokenizer wrapper around HuggingFace `tokenizers`.

Why BPE (Byte-Pair Encoding)?
  Character-level models have small vocabularies but very long sequences.
  Word-level models can't handle unseen words or morphology.
  BPE finds the sweet spot: start with bytes/characters, iteratively merge the most
  frequent adjacent pair into a new token — until reaching vocab_size.
  Result: common words become single tokens, rare words split into subword pieces.
  "unhappiness" → ["un", "happiness"] or ["un", "happy", "ness"] etc.

This module:
  - Trains a BPE tokenizer on a text file and saves it.
  - Loads a saved tokenizer for encode/decode during training and inference.
  - Exposes a simple API: encode(str) → List[int], decode(List[int]) → str.
"""

import os
from typing import List, Union
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


def train_tokenizer(
    text_path: str,
    save_dir: str,
    vocab_size: int = 8000,
) -> Tokenizer:
    """
    Train a BPE tokenizer on `text_path` and save to `save_dir`.

    Special tokens:
      <|pad|>  — padding (id 0)
      <|eos|>  — end-of-sequence / document boundary (id 1)
      <|unk|>  — unknown (rarely triggered with byte-level BPE)

    Byte-level BPE (used by GPT-2) maps every possible byte to an initial
    token before merging, so it is truly lossless — no <unk> in practice.
    """
    os.makedirs(save_dir, exist_ok=True)

    special_tokens = ["<|pad|>", "<|eos|>", "<|unk|>"]

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,          # ignore pairs that appear < 2 times
        show_progress=True,
    )

    tokenizer.train(files=[text_path], trainer=trainer)

    # Save tokenizer files to disk
    save_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}  (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(save_dir: str) -> Tokenizer:
    """Load a previously trained tokenizer from `save_dir`."""
    path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No tokenizer found at {path}. Run train_tokenizer() first."
        )
    tokenizer = Tokenizer.from_file(path)
    return tokenizer


class LLMTokenizer:
    """
    Thin wrapper around a HuggingFace Tokenizer to expose a clean API
    used by dataset.py, train.py, and generate.py.
    """

    PAD_TOKEN = "<|pad|>"
    EOS_TOKEN = "<|eos|>"

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id = tokenizer.token_to_id(self.PAD_TOKEN)
        self.eos_id = tokenizer.token_to_id(self.EOS_TOKEN)

    # ------------------------------------------------------------------
    @classmethod
    def from_training(cls, text_path: str, save_dir: str, vocab_size: int = 8000):
        """Train from scratch and return an LLMTokenizer instance."""
        tok = train_tokenizer(text_path, save_dir, vocab_size)
        return cls(tok)

    @classmethod
    def from_file(cls, save_dir: str):
        """Load a saved tokenizer and return an LLMTokenizer instance."""
        tok = load_tokenizer(save_dir)
        return cls(tok)

    # ------------------------------------------------------------------
    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """
        Convert text → token ids.
        `add_eos=True` appends the EOS id (useful for document-level training).
        """
        ids = self._tok.encode(text).ids
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Union[List[int], "torch.Tensor"], skip_special: bool = True) -> str:
        """Convert token ids → text.  Strips special tokens by default."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a list of strings in parallel."""
        return [enc.ids for enc in self._tok.encode_batch(texts)]

    def __len__(self) -> int:
        return self.vocab_size


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tokenizer.py <text_file> [vocab_size]")
        sys.exit(1)

    text_file = sys.argv[1]
    vocab = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    tok = LLMTokenizer.from_training(text_file, "data/tokenizer", vocab_size=vocab)

    sample = "Hello, world! This is a test."
    ids = tok.encode(sample)
    print(f"Sample text : {sample!r}")
    print(f"Token ids   : {ids}")
    print(f"Decoded     : {tok.decode(ids)!r}")
    print(f"Vocab size  : {tok.vocab_size}")
