"""
model.py — GPT-style decoder-only transformer.

Layer stack per block:
  x → LayerNorm → MultiHeadSelfAttention → residual
    → LayerNorm → FeedForward           → residual

Top-level:
  token_ids → Embedding + PositionalEmbedding → N × Block → LayerNorm → LM head

Design notes:
- Pre-norm (LN before sublayer) is more stable than post-norm during training.
- No cross-attention; this is a pure language model, not an encoder-decoder.
- The LM head weight is tied to the token embedding weight (saves ~vocab_size × d_model
  parameters and often improves perplexity).
- Causal masking is applied inside attention so each token only attends to past tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ---------------------------------------------------------------------------
# 1. Causal self-attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head scaled dot-product attention with a causal (lower-triangular) mask.

    Why multi-head? Splitting d_model into H heads lets the model attend to
    different representation subspaces simultaneously.  Each head independently
    computes Q, K, V projections on d_model/H dimensions, then outputs are
    concatenated and projected back to d_model.

    Why causal mask? During training we do "teacher forcing": the full target
    sequence is fed in at once and we predict every next token in parallel.
    The mask prevents position i from seeing positions > i, which would be
    cheating — we'd be predicting a token using its own future.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        # Single matrix for Q, K, V projections — more efficient than three separate ones
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask: lower-triangular matrix of ones, registered as a buffer
        # (moves to GPU with the model but is not a learnable parameter)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_length, cfg.context_length))
            .view(1, 1, cfg.context_length, cfg.context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, d_model

        # Project to Q, K, V in one shot, then split
        qkv = self.qkv_proj(x)                              # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=2)            # each (B, T, C)

        # Reshape for multi-head: (B, H, T, d_head)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        # scale by 1/sqrt(d_head) to keep dot products from growing too large
        scale = 1.0 / math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) * scale            # (B, H, T, T)

        # Apply causal mask: fill future positions with -inf so softmax → 0
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values, then merge heads back
        out = attn @ v                                       # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.resid_drop(self.out_proj(out))


# ---------------------------------------------------------------------------
# 2. Position-wise feed-forward network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Two-layer MLP applied independently to each token position.

    The "thinking" layer: attention decides which tokens to mix,
    FFN transforms each token's representation after mixing.
    Width is d_ff (typically 4 × d_model), using GELU activation.

    GELU vs ReLU: GELU is smoother and empirically trains better for language models.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One full decoder block: pre-norm → attention → residual, pre-norm → FFN → residual.

    Pre-LayerNorm (normalize *before* the sublayer) is more stable than the
    original "post-norm" in Vaswani et al. 2017 — it became standard after GPT-2.

    Residual connections let gradients flow directly from loss to early layers,
    preventing the vanishing-gradient problem in deep networks.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention sub-layer
        x = x + self.ff(self.ln2(x))     # FFN sub-layer
        return x


# ---------------------------------------------------------------------------
# 4. Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    Decoder-only transformer (GPT architecture).

    Token embedding + learned positional embedding → N transformer blocks
    → final LayerNorm → linear projection to vocabulary logits.

    Weight tying: the LM head reuses the token embedding matrix.
    This reduces parameters and acts as a regulariser.  The idea: if the
    model learns that 'dog' and 'puppy' have similar embeddings (input),
    they should have similar logits (output) as next-token predictions.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(
            dict(
                tok_emb=nn.Embedding(cfg.vocab_size, cfg.d_model),
                pos_emb=nn.Embedding(cfg.context_length, cfg.d_model),
                drop=nn.Dropout(cfg.dropout),
                blocks=nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
                ln_f=nn.LayerNorm(cfg.d_model),  # final norm before LM head
            )
        )

        # LM head: projects d_model → vocab logits (no bias, no softmax — raw logits)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.transformer.tok_emb.weight = self.lm_head.weight

        # Initialise weights following GPT-2 paper conventions
        self.apply(self._init_weights)

        # Scale residual projections by 1/sqrt(2 * n_layers) so that the
        # output of each block has the same variance regardless of depth.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,                  # (B, T) token ids
        targets: Optional[torch.Tensor] = None,  # (B, T) for training
    ):
        B, T = idx.shape
        assert T <= self.cfg.context_length, (
            f"Sequence length {T} exceeds context_length {self.cfg.context_length}"
        )

        # Embeddings: token + positional
        pos = torch.arange(T, device=idx.device)          # (T,)
        x = self.transformer.tok_emb(idx)                  # (B, T, d_model)
        x = x + self.transformer.pos_emb(pos)              # broadcast over B
        x = self.transformer.drop(x)

        # Run through all transformer blocks
        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                           # (B, T, vocab_size)

        # Cross-entropy loss — only computed during training
        loss = None
        if targets is not None:
            # Flatten (B, T) → (B*T,) for F.cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # padding tokens (if any) are marked -1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,    # (1, T) seed token ids
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation: repeatedly predict the next token and append it.
        Temperature scales the logits before sampling — higher = more diverse output.
        top_k and top_p can be combined (apply top_k first, then nucleus).
        """
        for _ in range(max_new_tokens):
            # Crop context to context_length (sliding window)
            idx_cond = idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature         # (1, vocab_size)

            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                kth_val = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens once cumulative prob exceeds top_p
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx

    def num_parameters(self, exclude_embeddings: bool = True) -> int:
        """Count trainable parameters (optionally exclude embedding tables)."""
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embeddings:
            n -= self.transformer.tok_emb.weight.numel()
        return n


# Needed for the Optional type hint inside the class before Python 3.10
from typing import Optional


if __name__ == "__main__":
    cfg = ModelConfig()
    model = GPT(cfg)
    total = sum(p.numel() for p in model.parameters())
    non_emb = model.num_parameters(exclude_embeddings=True)
    print(f"Total parameters:           {total / 1e6:.2f}M")
    print(f"Non-embedding parameters:   {non_emb / 1e6:.2f}M")
    print(f"Config: {cfg.n_layers}L / {cfg.n_heads}H / d_model={cfg.d_model} / ctx={cfg.context_length}")

    # Quick forward pass sanity-check
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, _ = model(x)
    print(f"Logits shape: {logits.shape}")  # expected: (2, 64, 8000)
