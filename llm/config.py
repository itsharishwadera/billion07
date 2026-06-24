"""
config.py — Central hyperparameter store.

Every number that controls model size, training behavior, or I/O lives here.
Nothing is hard-coded elsewhere; scripts import ModelConfig or TrainConfig.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # --- Vocabulary & context --------------------------------------------------
    vocab_size: int = 8000        # BPE vocab; set after tokenizer is trained
    context_length: int = 512     # Maximum sequence length (tokens)

    # --- Architecture ----------------------------------------------------------
    # With these defaults the model has ~10M parameters, comfortably fits 8GB VRAM.
    # To scale up: double d_model to 768, n_layers to 12, n_heads to 12 (~85M params).
    d_model: int = 512            # Embedding dimension (must be divisible by n_heads)
    n_layers: int = 6             # Number of transformer decoder blocks
    n_heads: int = 8              # Attention heads per layer (d_model / n_heads = 64)
    d_ff: int = 2048              # Feed-forward hidden dimension (usually 4 × d_model)

    # --- Regularisation --------------------------------------------------------
    dropout: float = 0.1          # Applied to attention weights and FFN activations

    # --- Misc ------------------------------------------------------------------
    bias: bool = False            # Whether Linear layers use bias (GPT-2 style: False)


@dataclass
class TrainConfig:
    # --- Data ------------------------------------------------------------------
    data_path: str = "data/input.txt"          # Raw text corpus
    tokenizer_path: str = "data/tokenizer"     # Directory for trained BPE tokenizer
    train_split: float = 0.9                   # Fraction used for training

    # --- Batching --------------------------------------------------------------
    batch_size: int = 32          # Sequences per gradient step
    # For 8GB VRAM: batch_size=32, context=512 is safe.  Increase to 64 with 16GB.

    # --- Optimiser -------------------------------------------------------------
    learning_rate: float = 3e-4   # AdamW lr; cosine-decay target is ~1e-5
    weight_decay: float = 0.1     # L2 regularisation on non-bias params
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0        # Gradient norm clipping

    # --- Schedule --------------------------------------------------------------
    warmup_steps: int = 200       # Linear warmup before cosine decay
    max_steps: int = 10_000       # Total training steps
    decay_lr: bool = True         # Use cosine LR decay

    # --- Logging & checkpointing -----------------------------------------------
    log_interval: int = 50        # Print loss every N steps
    eval_interval: int = 500      # Run validation every N steps
    eval_steps: int = 50          # Number of val batches to average
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000   # Save checkpoint every N steps
    resume_from: Optional[str] = None  # Path to a .pt checkpoint to resume from

    # --- Compute ---------------------------------------------------------------
    device: str = "cuda"          # "cuda" or "cpu"
    compile_model: bool = False   # torch.compile() — faster but needs PyTorch 2.x
    dtype: str = "bfloat16"       # "float32" | "bfloat16" | "float16"


@dataclass
class GenerateConfig:
    checkpoint_path: str = "checkpoints/best.pt"
    tokenizer_path: str = "data/tokenizer"
    max_new_tokens: int = 200
    temperature: float = 0.8      # > 1 = more random, < 1 = more focused
    top_k: int = 50               # Keep only top-k logits (0 = disabled)
    top_p: float = 0.9            # Nucleus sampling threshold (1.0 = disabled)
    device: str = "cuda"
