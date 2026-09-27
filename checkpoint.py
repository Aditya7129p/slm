"""
Checkpoint save / load with graceful Ctrl+C (SIGINT) handling.

Checkpoint layout  (checkpoints/step-NNNNNN/):
  model.pt        — model state dict
  muon.pt         — Muon optimizer state
  adamw.pt        — AdamW optimizer state
  train_state.json— step, tokens_consumed, rng states
"""

import os
import sys
import json
import signal
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple

import torch

from config import TrainConfig


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint paths
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_dir(base: str, step: int) -> Path:
    return Path(base) / f"step-{step:07d}"


def _latest_ckpt(base: str) -> Optional[Path]:
    """Find the highest-numbered checkpoint directory."""
    base = Path(base)
    if not base.exists():
        return None
    candidates = sorted(base.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if not candidates:
        return None
    return candidates[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    muon_opt,
    adamw_opt,
    step: int,
    tokens_consumed: int,
    train_cfg: TrainConfig,
    keep_last_n: int = 3,
) -> str:
    """
    Save a checkpoint and return its path.
    Keeps only the last `keep_last_n` checkpoints to save disk space.
    """
    base = train_cfg.checkpoint_dir
    ckpt_path = _ckpt_dir(base, step)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Save atomically via temp dir then rename (prevents corrupt checkpoints)
    tmp_path = ckpt_path.parent / f"_tmp_{step}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(model.state_dict(),     tmp_path / "model.pt")
        torch.save(muon_opt.state_dict(),  tmp_path / "muon.pt")
        torch.save(adamw_opt.state_dict(), tmp_path / "adamw.pt")

        state = {
            "step":             step,
            "tokens_consumed":  tokens_consumed,
            "torch_rng":        torch.get_rng_state().tolist(),
        }
        with open(tmp_path / "train_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Atomic rename
        if ckpt_path.exists():
            shutil.rmtree(ckpt_path)
        tmp_path.rename(ckpt_path)

    except Exception as e:
        # Clean up temp on failure
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise RuntimeError(f"Checkpoint save failed: {e}") from e

    # Prune old checkpoints
    _prune_old_checkpoints(base, keep_last_n)

    return str(ckpt_path)


def _prune_old_checkpoints(base: str, keep_last_n: int):
    base = Path(base)
    if not base.exists():
        return
    dirs = sorted(
        [d for d in base.glob("step-*") if d.is_dir()],
        key=lambda p: int(p.name.split("-")[1]),
    )
    for old in dirs[:-keep_last_n]:
        shutil.rmtree(old, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    model,
    muon_opt,
    adamw_opt,
    train_cfg: TrainConfig,
    path: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Load the latest (or specified) checkpoint into model and both optimizers.

    Returns (step, tokens_consumed).
    If no checkpoint found, returns (0, 0).
    """
    base = train_cfg.checkpoint_dir

    if path is not None:
        ckpt_path = Path(path)
    else:
        ckpt_path = _latest_ckpt(base)

    if ckpt_path is None or not ckpt_path.exists():
        return 0, 0

    print(f"  [checkpoint] Loading from {ckpt_path} …")

    map_loc = "cpu"   # always CPU

    model.load_state_dict(
        torch.load(ckpt_path / "model.pt", map_location=map_loc, weights_only=True)
    )
    muon_opt.load_state_dict(
        torch.load(ckpt_path / "muon.pt",  map_location=map_loc, weights_only=True)
    )
    adamw_opt.load_state_dict(
        torch.load(ckpt_path / "adamw.pt", map_location=map_loc, weights_only=True)
    )

    with open(ckpt_path / "train_state.json") as f:
        state = json.load(f)

    step            = state["step"]
    tokens_consumed = state["tokens_consumed"]

    if "torch_rng" in state:
        try:
            torch.set_rng_state(torch.tensor(state["torch_rng"], dtype=torch.uint8))
        except Exception:
            pass  # non-critical if RNG state is incompatible

    print(f"  [checkpoint] Resumed at step {step:,}  ({tokens_consumed/1e6:.1f}M tokens seen)")
    return step, tokens_consumed


# ─────────────────────────────────────────────────────────────────────────────
# Graceful Ctrl+C handler
# ─────────────────────────────────────────────────────────────────────────────

class GracefulInterrupt:
    """
    Intercepts SIGINT (Ctrl+C).

    Usage:
        handler = GracefulInterrupt()
        # In training loop:
        if handler.interrupted:
            save_checkpoint(...)
            break
    """

    def __init__(self):
        self.interrupted = False
        self._original   = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        self.interrupted = True
        # Don't re-raise; let the training loop detect the flag

    def restore(self):
        """Restore original SIGINT handler (call on clean exit)."""
        signal.signal(signal.SIGINT, self._original)

    def reset(self):
        """Clear the flag (if you want to allow multiple interrupts)."""
        self.interrupted = False
