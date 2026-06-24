"""
train.py — Training loop for the GPT language model.

Key features:
  - Cosine LR schedule with linear warmup
  - Gradient norm clipping (prevents exploding gradients)
  - Mixed-precision training (bfloat16) for ~2× speed on Ampere GPUs
  - Periodic validation loss evaluation
  - Checkpoint save/resume
  - CSV loss log for plotting

Usage:
  python train.py                     # train with defaults from config.py
  python train.py --resume checkpoints/step_5000.pt
"""

import os
import csv
import math
import time
import argparse
import torch
from torch.amp import GradScaler, autocast

from config import ModelConfig, TrainConfig
from model import GPT
from tokenizer import LLMTokenizer
from dataset import build_dataloaders


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup → cosine decay → min_lr floor
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig, min_lr_ratio: float = 0.1) -> float:
    """
    Why warmup? At step 0 the weights are random, so gradients are noisy.
    Starting with a tiny LR and ramping up prevents early catastrophic updates.

    Why cosine decay? Smoothly reduces LR as training converges, which often
    improves final perplexity compared to a fixed LR.
    """
    max_lr = cfg.learning_rate
    min_lr = max_lr * min_lr_ratio

    if not cfg.decay_lr:
        return max_lr

    # Linear warmup
    if step < cfg.warmup_steps:
        return max_lr * step / max(1, cfg.warmup_steps)

    # After max_steps, stay at min_lr
    if step >= cfg.max_steps:
        return min_lr

    # Cosine decay between warmup_steps and max_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_val_loss(
    model: GPT,
    val_loader,
    eval_steps: int,
    device: str,
    dtype: torch.dtype,
) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with autocast(device_type=device.split(":")[0], dtype=dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model: GPT, optimizer, step: int, val_loss: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "model_cfg": model.cfg,
        },
        path,
    )
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path: str, model: GPT, optimizer) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    step = ckpt["step"]
    val_loss = ckpt.get("val_loss", float("inf"))
    print(f"  [ckpt] Resumed from {path}  (step={step}, val_loss={val_loss:.4f})")
    return step, val_loss


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(mcfg: ModelConfig, tcfg: TrainConfig):
    # ---- device / dtype setup ------------------------------------------------
    device = tcfg.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, training on CPU (will be slow)")

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[tcfg.dtype]

    torch.manual_seed(42)
    if "cuda" in device:
        torch.cuda.manual_seed(42)

    # ---- tokenizer + data ----------------------------------------------------
    print("Loading tokenizer …")
    tok = LLMTokenizer.from_file(tcfg.tokenizer_path)
    mcfg.vocab_size = tok.vocab_size   # sync model config with actual vocab

    print("Building dataloaders …")
    train_loader, val_loader = build_dataloaders(
        tcfg.data_path,
        tok,
        mcfg.context_length,
        tcfg.batch_size,
        tcfg.train_split,
    )

    # Infinite iterator over train batches
    def cycle(loader):
        while True:
            yield from loader

    train_iter = cycle(train_loader)

    # ---- model ---------------------------------------------------------------
    print(f"Building model …")
    model = GPT(mcfg).to(device)
    if tcfg.compile_model and hasattr(torch, "compile"):
        print("  torch.compile() enabled")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.2f}M")

    # ---- optimizer -----------------------------------------------------------
    # Separate params into those that get weight decay and those that don't.
    # Biases and LayerNorm params should NOT be decayed (they're 1D vectors).
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": tcfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=tcfg.learning_rate,
        betas=(tcfg.beta1, tcfg.beta2),
        fused=("cuda" in device),   # fused kernel is faster on GPU
    )

    # Mixed-precision scaler (only useful for float16; bfloat16 doesn't need it)
    scaler = GradScaler(device=device, enabled=(tcfg.dtype == "float16"))

    # ---- resume --------------------------------------------------------------
    start_step = 0
    best_val_loss = float("inf")
    if tcfg.resume_from:
        start_step, best_val_loss = load_checkpoint(tcfg.resume_from, model, optimizer)

    # ---- logging setup -------------------------------------------------------
    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(tcfg.checkpoint_dir, "loss_log.csv")
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if start_step == 0:
        log_writer.writerow(["step", "train_loss", "val_loss", "lr"])

    # ---- training loop -------------------------------------------------------
    model.train()
    t0 = time.time()
    running_loss = 0.0

    print(f"\nStarting training from step {start_step} → {tcfg.max_steps}")
    print(f"  device={device}  dtype={tcfg.dtype}  batch={tcfg.batch_size}  ctx={mcfg.context_length}\n")

    for step in range(start_step, tcfg.max_steps):

        # Update learning rate
        lr = get_lr(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Fetch next batch
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Forward + backward with mixed precision
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.split(":")[0], dtype=dtype):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        # Gradient clipping: prevents exploding gradients by rescaling the
        # gradient vector so its L2 norm ≤ grad_clip
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # ---- logging ---------------------------------------------------------
        if (step + 1) % tcfg.log_interval == 0:
            avg_loss = running_loss / tcfg.log_interval
            elapsed = time.time() - t0
            tokens_per_sec = tcfg.log_interval * tcfg.batch_size * mcfg.context_length / elapsed
            print(
                f"step {step+1:6d}/{tcfg.max_steps}  "
                f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                f"{tokens_per_sec/1e3:.1f}K tok/s"
            )
            running_loss = 0.0
            t0 = time.time()

        # ---- validation ------------------------------------------------------
        if (step + 1) % tcfg.eval_interval == 0:
            val_loss = estimate_val_loss(model, val_loader, tcfg.eval_steps, device, dtype)
            print(f"  → val_loss={val_loss:.4f}  (best={best_val_loss:.4f})")
            log_writer.writerow([step + 1, avg_loss if (step+1) % tcfg.log_interval == 0 else "-", val_loss, lr])
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(tcfg.checkpoint_dir, "best.pt"),
                    model, optimizer, step + 1, val_loss,
                )

        # ---- checkpoint ------------------------------------------------------
        if (step + 1) % tcfg.checkpoint_interval == 0:
            save_checkpoint(
                os.path.join(tcfg.checkpoint_dir, f"step_{step+1}.pt"),
                model, optimizer, step + 1, 0.0,
            )

    log_file.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Loss log: {log_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    mcfg = ModelConfig()
    tcfg = TrainConfig()

    if args.resume:
        tcfg.resume_from = args.resume
    if args.max_steps:
        tcfg.max_steps = args.max_steps
    if args.batch_size:
        tcfg.batch_size = args.batch_size
    if args.lr:
        tcfg.learning_rate = args.lr

    train(mcfg, tcfg)
