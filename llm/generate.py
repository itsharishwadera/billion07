"""
generate.py — Interactive inference with a trained GPT model.

Sampling strategies:
  Greedy   temperature→0, top_k=1  — always picks the highest-prob token.
           Deterministic but repetitive.
  Top-k    keeps only the k most probable tokens before sampling.
           Cuts off the long tail of improbable tokens.
  Top-p    (nucleus) keeps the smallest set of tokens whose cumulative
           probability ≥ p.  Adaptive: uses fewer tokens when the model
           is confident, more when it is uncertain.
  Both     top_k and top_p can be combined — top_k is applied first.

Usage:
  python generate.py                          # interactive REPL
  python generate.py --prompt "Once upon a"  # one-shot
  python generate.py --temperature 1.2 --top_k 0 --top_p 0.95
"""

import argparse
import torch

from config import ModelConfig, GenerateConfig
from model import GPT
from tokenizer import LLMTokenizer


def load_model(cfg: GenerateConfig, model_cfg: ModelConfig = None) -> tuple[GPT, LLMTokenizer]:
    """Load tokenizer and model from a checkpoint."""
    tok = LLMTokenizer.from_file(cfg.tokenizer_path)

    # Load checkpoint
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)

    # Prefer the saved model config; fall back to a new ModelConfig
    saved_cfg = ckpt.get("model_cfg", model_cfg or ModelConfig())
    saved_cfg.vocab_size = tok.vocab_size   # ensure vocab is in sync

    model = GPT(saved_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    step = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", float("inf"))
    print(f"Loaded checkpoint from step {step}  (val_loss={val_loss:.4f}  params={n_params:.2f}M)")
    return model, tok


@torch.no_grad()
def generate(
    model: GPT,
    tok: LLMTokenizer,
    prompt: str,
    cfg: GenerateConfig,
) -> str:
    """Encode prompt, run autoregressive generation, decode output."""
    device = next(model.parameters()).device

    ids = tok.encode(prompt, add_eos=False)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    out_ids = model.generate(
        idx,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
    )

    # Decode only the newly generated tokens (strip the prompt)
    new_ids = out_ids[0, len(ids):]
    return tok.decode(new_ids)


def interactive(model: GPT, tok: LLMTokenizer, cfg: GenerateConfig):
    """Simple REPL: type a prompt, get a continuation."""
    print("\nGPT inference — type a prompt and press Enter.  Ctrl-C to quit.\n")
    print(f"  Settings: temp={cfg.temperature}  top_k={cfg.top_k}  top_p={cfg.top_p}  max_new={cfg.max_new_tokens}\n")
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not prompt:
            continue
        result = generate(model, tok, prompt, cfg)
        print(f"\n{prompt}{result}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="One-shot prompt (no interactive mode)")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    gcfg = GenerateConfig()
    if args.checkpoint:
        gcfg.checkpoint_path = args.checkpoint
    if args.tokenizer:
        gcfg.tokenizer_path = args.tokenizer
    if args.max_new_tokens:
        gcfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        gcfg.temperature = args.temperature
    if args.top_k is not None:
        gcfg.top_k = args.top_k
    if args.top_p is not None:
        gcfg.top_p = args.top_p
    if args.device:
        gcfg.device = args.device
    elif not torch.cuda.is_available():
        gcfg.device = "cpu"

    model, tok = load_model(gcfg)

    if args.prompt:
        result = generate(model, tok, args.prompt, gcfg)
        print(f"{args.prompt}{result}")
    else:
        interactive(model, tok, gcfg)
