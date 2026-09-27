"""
SLM Pre-Training  —  Kaggle 2× T4 DDP entry point.

Runs distributed training across 2 T4 GPUs using PyTorch DDP (DistributedDataParallel).

Launch command (run inside the Kaggle notebook):
    torchrun --nproc_per_node=2 train_kaggle.py

Or with explicit env (in case torchrun is not in PATH):
    python -m torch.distributed.run --nproc_per_node=2 train_kaggle.py

Resume from checkpoint:
    torchrun --nproc_per_node=2 train_kaggle.py --resume checkpoints_kaggle/step-0005000

Start from scratch:
    torchrun --nproc_per_node=2 train_kaggle.py --no-resume

HuggingFace token:
    Set HF_TOKEN in environment or .env file. Rank 0 prints auth status.

DDP design notes:
  - Each rank trains on its own data shard (different stream offset per rank).
  - Gradients are all-reduced across GPUs via NCCL before the optimizer step.
  - The Muon Newton-Schulz step is a local operation — no extra communication.
  - Only rank 0 writes checkpoints, logs, and inference checks.
  - All ranks must call optimizer.step() synchronously; non-rank-0 steps are
    a no-op in terms of side effects but required for NCCL synchronisation.

Session management:
  - SIGTERM (cell stop / kernel kill): caught alongside SIGINT — triggers an
    emergency checkpoint before exiting cleanly.
  - Auto-save timer: exits cleanly after SESSION_HOURS (default 8h 40min) so
    the notebook cell finishes and you can copy the checkpoint to persistent
    storage before the Kaggle session expires.
  - Override the timer via the --session-hours flag:
      torchrun --nproc_per_node=2 train_kaggle.py --session-hours 8.67
"""

import os
import sys
import gc
import time
import math
import signal
import argparse
import contextlib

# ── HuggingFace authentication ────────────────────────────────────────────────
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
        print("  [HF] Authenticated with HF_TOKEN ✔  (higher rate limits enabled)")
    except Exception as e:
        print(f"  [HF] Login failed: {e}  (continuing unauthenticated)")

# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ── Project imports ────────────────────────────────────────────────────────
from config_kaggle import model_cfg, train_cfg
from model import SLM
from optimizer import build_optimizers, get_lr
from dataset import make_data_generator
from checkpoint import save_checkpoint, load_checkpoint
from logger import TrainingLogger
from inference import run_inference_check

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown handler — catches SIGINT (Ctrl+C) AND SIGTERM (cell stop)
# ─────────────────────────────────────────────────────────────────────────────
class _ShutdownHandler:
    """
    Sets a shared flag when SIGINT or SIGTERM is received.

    Kaggle sends SIGTERM to child processes when you click the cell stop button
    or interrupt the kernel. We catch it here so torchrun worker processes can
    save a checkpoint before exiting instead of being killed mid-step.
    """

    def __init__(self):
        self.interrupted = False
        self._orig_sigint  = signal.getsignal(signal.SIGINT)
        self._orig_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT,  self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        # Only print on first signal to avoid console spam
        if not self.interrupted:
            print(f"\n[kaggle] {sig_name} received — finishing step and saving checkpoint …",
                  flush=True)
        self.interrupted = True

    def restore(self):
        signal.signal(signal.SIGINT,  self._orig_sigint)
        signal.signal(signal.SIGTERM, self._orig_sigterm)


# ─────────────────────────────────────────────────────────────────────────────
# Session timer — auto-exit before Kaggle's 12h wall-clock limit
# ─────────────────────────────────────────────────────────────────────────────
class _SessionTimer:
    """
    Triggers a clean checkpoint-and-exit after `hours` wall-clock time.

    Kaggle GPU sessions expire at 12h. We exit at 8h40m (configurable) so
    there is always ~3h20m left to copy checkpoints to persistent storage
    (Kaggle dataset output) before the session dies.
    """

    def __init__(self, hours: float):
        self._deadline = time.monotonic() + hours * 3600
        self._fired    = False

    @property
    def expired(self) -> bool:
        if not self._fired and time.monotonic() >= self._deadline:
            self._fired = True
            return True
        return self._fired

    def remaining_str(self) -> str:
        secs = max(0.0, self._deadline - time.monotonic())
        h, rem = divmod(int(secs), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"


# ─────────────────────────────────────────────────────────────────────────────
# DDP initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _init_ddp():
    """
    Initialise the process group.
    torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically.
    Returns (rank, local_rank, world_size).
    """
    dist.init_process_group(backend=train_cfg.ddp_backend)
    rank       = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _is_master(rank: int) -> bool:
    return rank == 0


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Device / dtype helpers
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_dtype(cfg_dtype: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg_dtype]


def _make_autocast(dtype: torch.dtype):
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _compute_grad_norm(model) -> float:
    """Compute gradient norm on the unwrapped model parameters."""
    raw = model.module if isinstance(model, DDP) else model
    total = 0.0
    for p in raw.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def _set_lr(muon_opt, adamw_opt, lr_mult: float, cfg):
    for group in muon_opt.param_groups:
        group["lr"] = cfg.lr * lr_mult
    for group in adamw_opt.param_groups:
        group["lr"] = cfg.lr * cfg.embed_lr_scale * lr_mult


def _load_tokenizer(path: str):
    from tokenizers import Tokenizer
    tok_path = Path(path) / "tokenizer.json"
    if not tok_path.exists():
        return None
    return Tokenizer.from_file(str(tok_path))


def _check_memory():
    if not _HAS_PSUTIL:
        return
    mem = psutil.virtual_memory()
    if mem.percent > 90:
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def _compute_val_loss(model, cfg, device, autocast, val_batches: int = 16) -> float:
    """Val loss on rank 0 only — uses a fresh stream starting from token 0."""
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()
    val_gen    = make_data_generator(cfg, skip_tokens=0)
    total_loss = 0.0
    count      = 0
    for _ in range(val_batches):
        try:
            x, y, _ = next(val_gen)
        except StopIteration:
            break
        x = x.to(device)
        y = y.to(device)
        with autocast:
            _, loss = raw(x, y)
        total_loss += loss.item()
        count      += 1
    raw.train()
    return total_loss / max(1, count)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(resume_path: str = None, session_hours: float = 8.667):
    cfg  = train_cfg
    mcfg = model_cfg

    # ── DDP init ─────────────────────────────────────────────────────────────
    rank, local_rank, world_size = _init_ddp()
    is_master = _is_master(rank)
    session_timer = _SessionTimer(session_hours)

    if is_master:
        _hf_login()
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    # Barrier — ensure dirs exist before all ranks proceed
    dist.barrier()

    device   = torch.device(f"cuda:{local_rank}")
    dtype    = _resolve_dtype(cfg.dtype)
    autocast = _make_autocast(dtype)

    if is_master:
        props = torch.cuda.get_device_properties(local_rank)
        print(f"[kaggle] rank={rank}  device={device}  dtype={dtype}")
        print(f"[kaggle] GPU: {props.name}  VRAM: {props.total_memory/1024**3:.1f} GB")
        print(f"[kaggle] world_size={world_size}  backend={cfg.ddp_backend}")
        print(f"[kaggle] session timer: {session_hours:.2f}h  "
              f"(auto-save + exit at T-{session_timer.remaining_str()})")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SLM(mcfg).to(device=device, dtype=dtype)
    n_params = model.num_parameters()

    # Wrap with DDP — gradients are all-reduced automatically on backward()
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=cfg.find_unused_params,
    )

    # ── Optimizers (operate on the unwrapped model parameters) ───────────────
    raw_model = model.module
    muon_opt, adamw_opt = build_optimizers(raw_model, cfg)

    # ── Tokenizer (rank 0 only for inference checks) ──────────────────────────
    tokenizer = None
    if is_master:
        tokenizer = _load_tokenizer(cfg.tokenizer_path)
        if tokenizer is None:
            print("[kaggle] ⚠  Tokenizer not found — inference checks will be skipped.")
            print("[kaggle]    Run:  python tokenizer_train.py --config config_kaggle")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step, tokens_consumed = 0, 0
    if cfg.resume or resume_path:
        # All ranks load the same checkpoint — DDP model state must be identical
        start_step, tokens_consumed = load_checkpoint(
            raw_model, muon_opt, adamw_opt, cfg, path=resume_path
        )

    # Broadcast step/tokens from rank 0 to ensure consistency
    state_tensor = torch.tensor([start_step, tokens_consumed], dtype=torch.long, device=device)
    dist.broadcast(state_tensor, src=0)
    start_step, tokens_consumed = int(state_tensor[0]), int(state_tensor[1])

    # ── Logger (rank 0 only) ──────────────────────────────────────────────────
    logger = None
    if is_master:
        logger = TrainingLogger(
            log_dir=cfg.log_dir,
            total_steps=cfg.max_steps,
            grad_accum_steps=cfg.grad_accum_steps,
            resume_step=start_step,
        )
        logger.start(n_params=n_params, vocab_size=mcfg.vocab_size)

    # ── Interrupt handler (rank 0 coordinates, others follow) ────────────────
    interrupt = _ShutdownHandler()

    # ── Data generator — each rank gets an offset to avoid duplicate data ─────
    # Offset each rank by rank × (tokens_per_rank) from current position.
    # We use a simple heuristic: each rank skips an equal shard of the stream.
    tokens_per_rank = tokens_consumed // max(1, world_size)
    rank_token_offset = tokens_consumed + rank * tokens_per_rank
    data_gen = make_data_generator(cfg, skip_tokens=rank_token_offset)

    step            = start_step
    grad_accum_loss = 0.0

    try:
        while step < cfg.max_steps:

            # ── Shutdown / timer check (broadcast from rank 0) ────────────────
            # Combine interrupt flag (SIGINT/SIGTERM) and session timer into one
            # tensor so we only need a single broadcast per step.
            # Bit 0 = user interrupt,  Bit 1 = session timer expired
            shutdown_reason = 0
            if interrupt.interrupted:
                shutdown_reason |= 1
            if is_master and session_timer.expired:
                shutdown_reason |= 2

            shutdown_tensor = torch.tensor([shutdown_reason], dtype=torch.int, device=device)
            dist.broadcast(shutdown_tensor, src=0)
            shutdown_reason = shutdown_tensor.item()

            if shutdown_reason != 0:
                reason_str = (
                    "SIGINT/SIGTERM received" if shutdown_reason & 1
                    else "session timer expired (8h40m)"
                )
                if is_master:
                    print(f"\n[kaggle] Stopping: {reason_str} — saving checkpoint …",
                          flush=True)
                    logger.log_interrupt(step)
                    ckpt_path = save_checkpoint(
                        raw_model, muon_opt, adamw_opt,
                        step=step, tokens_consumed=tokens_consumed,
                        train_cfg=cfg,
                    )
                    logger.log_checkpoint(step, ckpt_path)
                    print(f"[kaggle] Checkpoint saved → {ckpt_path}", flush=True)
                    print(f"[kaggle] Copy this to persistent storage to resume next session.",
                          flush=True)
                dist.barrier()
                break

            muon_opt.zero_grad()
            adamw_opt.zero_grad()
            grad_accum_loss = 0.0

            for micro in range(cfg.grad_accum_steps):
                # On all-but-last micro-step, disable DDP sync to avoid
                # premature all-reduce. Only sync on the final micro-step.
                sync_context = (
                    contextlib.nullcontext()
                    if micro == cfg.grad_accum_steps - 1
                    else model.no_sync()
                )

                try:
                    x, y, n_toks = next(data_gen)
                except StopIteration:
                    data_gen = make_data_generator(cfg, skip_tokens=0)
                    x, y, n_toks = next(data_gen)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with sync_context:
                    with autocast:
                        _, loss = model(x, y)
                    scaled_loss = loss / cfg.grad_accum_steps
                    scaled_loss.backward()

                grad_accum_loss += loss.item() / cfg.grad_accum_steps
                tokens_consumed += n_toks

                if is_master and cfg.log_every <= 1:
                    logger.log_micro(micro + 1, micro_loss=loss.item())

            # ── Average loss across ranks for accurate logging ─────────────────
            loss_tensor = torch.tensor([grad_accum_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            grad_accum_loss = loss_tensor.item()

            # ── Gradient clipping ─────────────────────────────────────────────
            grad_norm = _compute_grad_norm(model)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)

            # ── LR schedule ───────────────────────────────────────────────────
            step += 1
            lr_mult    = get_lr(step, cfg)
            _set_lr(muon_opt, adamw_opt, lr_mult, cfg)
            current_lr = cfg.lr * lr_mult

            # ── Optimizer step (all ranks must call this together) ─────────────
            muon_opt.step()
            adamw_opt.step()

            _check_memory()

            # ── Periodic timed checkpoint reminder (rank 0 only) ──────────────
            if is_master and step % 100 == 0:
                print(f"[kaggle] session time remaining: {session_timer.remaining_str()}",
                      flush=True)

            # ── Eval / checkpoint / logging (rank 0 only) ─────────────────────
            is_ckpt  = (step % cfg.save_every == 0)
            is_eval  = (step % cfg.eval_every == 0)
            is_inf   = is_eval and tokenizer is not None

            val_loss = None
            if is_master and is_eval:
                val_batches = getattr(cfg, "val_batches", 16)
                val_loss = _compute_val_loss(model, cfg, device, autocast, val_batches)

            if is_master:
                if step % cfg.log_every == 0:
                    logger.log_step(
                        step=step,
                        loss=grad_accum_loss,
                        lr=current_lr,
                        grad_norm=grad_norm,
                        tokens_total=tokens_consumed * world_size,
                        is_checkpoint=is_ckpt,
                        is_inference=is_inf,
                        val_loss=val_loss,
                    )

                if is_ckpt:
                    ckpt_path = save_checkpoint(
                        raw_model, muon_opt, adamw_opt,
                        step=step,
                        tokens_consumed=tokens_consumed,
                        train_cfg=cfg,
                    )
                    logger.log_checkpoint(step, ckpt_path)

                if is_inf:
                    results = run_inference_check(
                        raw_model, tokenizer,
                        prompts=cfg.inference_prompts,
                        max_new_tokens=cfg.inference_max_new_tokens,
                        temperature=cfg.inference_temperature,
                        top_k=cfg.inference_top_k,
                    )
                    for prompt, generated in zip(cfg.inference_prompts, results):
                        logger.log_inference(step, prompt, generated)
                    raw_model.train()

            # All ranks sync after each optimizer step
            dist.barrier()

        # ── Training complete ──────────────────────────────────────────────────
        if is_master and not interrupt.interrupted:
            ckpt_path = save_checkpoint(
                raw_model, muon_opt, adamw_opt,
                step=step,
                tokens_consumed=tokens_consumed,
                train_cfg=cfg,
            )
            logger.log_checkpoint(step, ckpt_path)
            logger.log_done(step, tokens_consumed * world_size)

    except Exception as e:
        print(f"\n[kaggle rank={rank}] ⚠  Unexpected error: {e}")
        if is_master:
            try:
                ckpt_path = save_checkpoint(
                    raw_model, muon_opt, adamw_opt,
                    step=step, tokens_consumed=tokens_consumed,
                    train_cfg=cfg,
                )
                print(f"[kaggle] Emergency checkpoint saved → {ckpt_path}")
            except Exception as save_err:
                print(f"[kaggle] Could not save checkpoint: {save_err}")
        raise

    finally:
        interrupt.restore()
        if is_master and logger is not None:
            logger.close()
        _cleanup_ddp()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SLM Pre-Training — Kaggle 2× T4 DDP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  torchrun --nproc_per_node=2 train_kaggle.py\n"
            "  torchrun --nproc_per_node=2 train_kaggle.py --no-resume\n"
            "  torchrun --nproc_per_node=2 train_kaggle.py --resume checkpoints_kaggle/step-0005000\n"
            "  torchrun --nproc_per_node=2 train_kaggle.py --session-hours 8.67\n"
        ),
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Path to a specific checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch, ignoring existing checkpoints.",
    )
    parser.add_argument(
        "--session-hours",
        type=float,
        default=8.667,
        metavar="HOURS",
        help=(
            "Auto-save checkpoint and exit after this many wall-clock hours. "
            "Default: 8.667 (8h 40min). Kaggle sessions last 12h; the gap "
            "gives time to copy checkpoints to persistent storage."
        ),
    )
    args = parser.parse_args()

    if args.no_resume:
        train_cfg.resume = False

    train(resume_path=args.resume, session_hours=args.session_hours)


if __name__ == "__main__":
    main()
