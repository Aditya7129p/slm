"""
SLM Pre-Training  —  main entry point.

Run:
    python train.py

Stop cleanly:
    Ctrl+C  →  saves emergency checkpoint and exits gracefully.

Resume:
    python train.py          (auto-detects latest checkpoint)
    python train.py --resume checkpoints/step-0001000

HuggingFace token (optional but recommended):
    Set HF_TOKEN in environment  OR  put it in a .env file:
        echo HF_TOKEN=hf_xxxx > .env
    This enables higher rate limits and faster dataset downloads.
"""

import os
import sys
import gc
import time
import math
import argparse
import contextlib

# ── HuggingFace authentication ────────────────────────────────────────────────
# Load .env file if present (simple key=value, no external dependency needed)
def _load_dotenv(path: str = ".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass

_load_dotenv()

def _hf_login():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print(f"  [HF] Authenticated with HF_TOKEN ✔  (higher rate limits enabled)")
    except Exception as e:
        print(f"  [HF] Login failed: {e}  (continuing unauthenticated)")

_hf_login()
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

import torch

# ── Project imports ────────────────────────────────────────────────────────
from config import model_cfg, train_cfg
from model import SLM
from optimizer import build_optimizers, get_lr
from dataset import make_data_generator
from checkpoint import save_checkpoint, load_checkpoint, GracefulInterrupt
from logger import TrainingLogger
from inference import run_inference_check

# ── Optional: psutil for memory monitoring ─────────────────────────────────
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ─────────────────────────────────────────────────────────────────────────────
# Device + dtype resolution
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def _resolve_dtype(cfg_dtype: str, device: torch.device) -> torch.dtype:
    if cfg_dtype == "auto":
        if device.type in ("cuda", "mps"):
            return torch.bfloat16
        return torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg_dtype]


def _make_autocast(device: torch.device, dtype: torch.dtype):
    """Returns a context manager that enables autocast on GPU, no-op on CPU."""
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    if device.type == "mps":
        return torch.amp.autocast(device_type="mps", dtype=dtype)
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# Memory guard
# ─────────────────────────────────────────────────────────────────────────────
_MAX_RAM_GB = 7.0   # hard target — Windows takes the rest

def _check_memory(logger: TrainingLogger, step: int):
    """Warn if RAM usage is too high; force GC if critical."""
    if not _HAS_PSUTIL:
        return
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1024**3
    if used_gb > _MAX_RAM_GB * 0.95:   # within 5 % of limit
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ─────────────────────────────────────────────────────────────────────────────
# Gradient norm
# ─────────────────────────────────────────────────────────────────────────────
def _compute_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


# ─────────────────────────────────────────────────────────────────────────────
# LR update  (applies to both optimizers)
# ─────────────────────────────────────────────────────────────────────────────
def _set_lr(muon_opt, adamw_opt, lr_mult: float, cfg):
    for group in muon_opt.param_groups:
        group["lr"] = cfg.lr * lr_mult
    for group in adamw_opt.param_groups:
        # embed group has its own scale baked in at construction time
        # recompute from base
        group["lr"] = cfg.lr * cfg.embed_lr_scale * lr_mult


# ─────────────────────────────────────────────────────────────────────────────
# Load tokenizer (for inference checks)
# ─────────────────────────────────────────────────────────────────────────────
def _load_tokenizer(path: str):
    from tokenizers import Tokenizer
    tok_path = Path(path) / "tokenizer.json"
    if not tok_path.exists():
        return None
    return Tokenizer.from_file(str(tok_path))


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(resume_path: str = None):
    cfg       = train_cfg
    mcfg      = model_cfg

    # ── Directories ─────────────────────────────────────────────────────────
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    # ── Device & dtype ──────────────────────────────────────────────────────
    device    = _resolve_device(cfg.device)
    dtype     = _resolve_dtype(cfg.dtype, device)
    autocast  = _make_autocast(device, dtype)
    if device.type == "cuda":
        cfg.pin_memory = True
    print(f"[train] device={device}  dtype={dtype}  pin_memory={cfg.pin_memory}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = SLM(mcfg).to(device=device, dtype=dtype)
    model.train()

    n_params = model.num_parameters()

    # ── Optimizers ──────────────────────────────────────────────────────────
    muon_opt, adamw_opt = build_optimizers(model, cfg)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = _load_tokenizer(cfg.tokenizer_path)
    if tokenizer is None:
        print("[train] ⚠  Tokenizer not found — inference checks will be skipped.")
        print("[train]    Run:  python tokenizer_train.py")

    # ── Resume ──────────────────────────────────────────────────────────────
    start_step, tokens_consumed = 0, 0
    if cfg.resume or resume_path:
        start_step, tokens_consumed = load_checkpoint(
            model, muon_opt, adamw_opt, cfg, path=resume_path
        )

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = TrainingLogger(
        log_dir=cfg.log_dir,
        total_steps=cfg.max_steps,
        grad_accum_steps=cfg.grad_accum_steps,
        resume_step=start_step,
    )
    logger.start(n_params=n_params, vocab_size=mcfg.vocab_size)

    # ── Interrupt handler ───────────────────────────────────────────────────
    interrupt = GracefulInterrupt()

    # ── Data generator ──────────────────────────────────────────────────────
    data_gen = make_data_generator(cfg, skip_tokens=tokens_consumed)

    # ── Training loop ───────────────────────────────────────────────────────
    step = start_step
    grad_accum_loss = 0.0

    # Accumulation buffer
    accum_step = 0

    try:
        while step < cfg.max_steps:

            # ── Check for Ctrl+C ────────────────────────────────────────────
            if interrupt.interrupted:
                logger.log_interrupt(step)
                ckpt_path = save_checkpoint(
                    model, muon_opt, adamw_opt,
                    step=step,
                    tokens_consumed=tokens_consumed,
                    train_cfg=cfg,
                )
                logger.log_checkpoint(step, ckpt_path)
                break

            # ── Gradient accumulation inner loop ────────────────────────────
            muon_opt.zero_grad()
            adamw_opt.zero_grad()
            grad_accum_loss = 0.0

            for micro in range(cfg.grad_accum_steps):

                # Check interrupt inside micro loop too
                if interrupt.interrupted:
                    break

                # Get next batch
                try:
                    x, y, n_toks = next(data_gen)
                except StopIteration:
                    # Dataset exhausted (shouldn't happen with streaming, but handle it)
                    print("\n[train] Dataset exhausted — restarting stream.")
                    data_gen = make_data_generator(cfg, skip_tokens=0)
                    x, y, n_toks = next(data_gen)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Forward  (autocast is a no-op on CPU)
                with autocast:
                    _, loss = model(x, y)

                # Scale loss for accumulation
                scaled_loss = loss / cfg.grad_accum_steps
                scaled_loss.backward()

                grad_accum_loss += loss.item() / cfg.grad_accum_steps
                tokens_consumed += n_toks

                # Show micro-step progress
                if cfg.log_every <= 1:
                    logger.log_micro(micro + 1, micro_loss=loss.item())

            # Handle interrupt that fired inside micro loop
            if interrupt.interrupted:
                logger.log_interrupt(step)
                ckpt_path = save_checkpoint(
                    model, muon_opt, adamw_opt,
                    step=step,
                    tokens_consumed=tokens_consumed,
                    train_cfg=cfg,
                )
                logger.log_checkpoint(step, ckpt_path)
                break

            # ── Gradient clipping ────────────────────────────────────────────
            grad_norm = _compute_grad_norm(model)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            # ── LR schedule ──────────────────────────────────────────────────
            step += 1
            lr_mult = get_lr(step, cfg)
            _set_lr(muon_opt, adamw_opt, lr_mult, cfg)
            current_lr = cfg.lr * lr_mult

            # ── Optimizer step ───────────────────────────────────────────────
            muon_opt.step()
            adamw_opt.step()

            # ── Memory guard ─────────────────────────────────────────────────
            _check_memory(logger, step)

            # ── Logging ──────────────────────────────────────────────────────
            is_ckpt = (step % cfg.save_every == 0)
            is_inf  = (step % cfg.eval_every == 0) and tokenizer is not None

            if step % cfg.log_every == 0:
                logger.log_step(
                    step=step,
                    loss=grad_accum_loss,
                    lr=current_lr,
                    grad_norm=grad_norm,
                    tokens_total=tokens_consumed,
                    is_checkpoint=is_ckpt,
                    is_inference=is_inf,
                )

            # ── Periodic checkpoint ───────────────────────────────────────────
            if is_ckpt:
                ckpt_path = save_checkpoint(
                    model, muon_opt, adamw_opt,
                    step=step,
                    tokens_consumed=tokens_consumed,
                    train_cfg=cfg,
                )
                logger.log_checkpoint(step, ckpt_path)

            # ── Inference check ──────────────────────────────────────────────
            if is_inf:
                results = run_inference_check(
                    model, tokenizer,
                    prompts=cfg.inference_prompts,
                    max_new_tokens=cfg.inference_max_new_tokens,
                    temperature=cfg.inference_temperature,
                    top_k=cfg.inference_top_k,
                )
                for prompt, generated in zip(cfg.inference_prompts, results):
                    logger.log_inference(step, prompt, generated)
                model.train()   # ensure back in train mode

        # ── Training complete ────────────────────────────────────────────────
        if not interrupt.interrupted:
            # Final checkpoint
            ckpt_path = save_checkpoint(
                model, muon_opt, adamw_opt,
                step=step,
                tokens_consumed=tokens_consumed,
                train_cfg=cfg,
            )
            logger.log_checkpoint(step, ckpt_path)
            logger.log_done(step, tokens_consumed)

    except Exception as e:
        # Unexpected error — still try to save
        print(f"\n[train] ⚠  Unexpected error: {e}")
        try:
            ckpt_path = save_checkpoint(
                model, muon_opt, adamw_opt,
                step=step,
                tokens_consumed=tokens_consumed,
                train_cfg=cfg,
            )
            print(f"[train] Emergency checkpoint saved → {ckpt_path}")
        except Exception as save_err:
            print(f"[train] Could not save checkpoint: {save_err}")
        raise

    finally:
        interrupt.restore()
        logger.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SLM Pre-Training")
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Path to a specific checkpoint directory to resume from. "
             "If omitted, auto-resumes from the latest checkpoint (if any).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch, ignoring any existing checkpoints.",
    )
    args = parser.parse_args()

    if args.no_resume:
        train_cfg.resume = False

    train(resume_path=args.resume)


if __name__ == "__main__":
    main()
