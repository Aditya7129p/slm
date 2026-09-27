"""
Colab / GPU configuration for SLM.

This file is a SEPARATE config for Google Colab or any CUDA machine.
It inherits all defaults from config.py and overrides only what changes on GPU.

Usage in Colab:
    python train_colab.py

The existing  train.py / config.py  run on your local CPU is NEVER touched.
"""

from dataclasses import dataclass, field
from typing import Optional

# Import the base dataclasses (not the singleton instances)
from config import ModelConfig, TrainConfig


# ─────────────────────────────────────────────────────────────────────────────
# Colab Model config  — same architecture, just documenting it explicitly
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ColabModelConfig(ModelConfig):
    # Inherit everything from your local ModelConfig.
    # Override here if you want a larger model on Colab.
    pass   # identical architecture — keeps checkpoints compatible


# ─────────────────────────────────────────────────────────────────────────────
# Colab Train config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ColabTrainConfig(TrainConfig):
    # ── Device / Precision ────────────────────────────────────────────────────
    device: str = "auto"        # auto → cuda on Colab
    dtype:  str = "auto"        # auto → bfloat16 on cuda (free speedup on A100/T4)

    # ── Larger batches — GPU has much more memory ─────────────────────────────
    batch_size:       int = 16   # micro-batch per step (was 8 on CPU)
    grad_accum_steps: int = 4    # effective batch = 16 × 4 = 64 (was 16 × 8 = 128)

    # ── Faster saves — Colab disk is ephemeral, save often ───────────────────
    save_every: int = 250        # checkpoint every 250 steps (was 500)

    # ── Paths — keep separate so Colab checkpoints don't collide with local ──
    checkpoint_dir: str = "checkpoints_colab"
    log_dir:        str = "logs_colab"

    # ── Learning rate — same, Muon is robust across batch sizes ──────────────
    lr:              float = 3e-3
    warmup_steps:    int   = 100
    lr_decay_steps:  int   = 50_000

    # ── Resume from local checkpoint if uploaded to Colab ────────────────────
    # Set this to the path you uploaded, e.g. "checkpoints/step-0000563"
    # or leave True to auto-find latest inside checkpoint_dir
    resume: bool = True


# ── Singleton instances used by train_colab.py ───────────────────────────────
colab_model_cfg = ColabModelConfig()
colab_train_cfg = ColabTrainConfig()
