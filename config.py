"""
Central configuration for the 4M-parameter SLM.
All hyper-parameters are in one place — edit here, not elsewhere.
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ModelConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size: int = 8192          # small English-only BPE vocab
    context_len: int = 512          # sequence length
    d_model: int = 288              # embedding / hidden dimension  (↑ from 256)
    n_heads: int = 8                # attention heads  (head_dim = d_model // n_heads = 36)
    n_kv_heads: int = 2             # GQA key/value heads  (n_heads must be divisible by n_kv_heads)
    n_layers: int = 10              # transformer blocks  (↑ from 8 → better loss/step)
    d_ff_mult: float = 2.667        # FFN width = round(d_model * d_ff_mult / 64) * 64
    dropout: float = 0.0            # disabled for small model / fast training
    bias: bool = False              # no bias (cleaner, faster)
    norm_eps: float = 1e-6

    # ── RoPE ─────────────────────────────────────────────────────────────────
    rope_base: float = 10_000.0
    rope_scaling: Optional[float] = None   # set to e.g. 4.0 for long-context later

    @property
    def d_ff(self) -> int:
        raw = int(self.d_model * self.d_ff_mult)
        # Align to multiple of 64 for hardware efficiency
        return (raw // 64) * 64 or 64


@dataclass
class TrainConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"   # streaming-friendly subset
    dataset_split: str = "train"

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_path: str = "tokenizer"     # dir; trained once, then reused
    tokenizer_vocab_size: int = 8192

    # ── Batch / Sequence ──────────────────────────────────────────────────────
    batch_size: int = 8               # sequences per gradient-accumulation micro-step
    grad_accum_steps: int = 16        # effective batch = batch_size * grad_accum_steps  (↑ from 8)
    context_len: int = 512

    # ── Optimiser (Muon) ─────────────────────────────────────────────────────
    lr: float = 3e-3                  # Muon works best at higher LRs
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Nesterov momentum for Muon
    muon_momentum: float = 0.95
    # embed / lm_head use plain AdamW
    embed_lr_scale: float = 0.1

    # ── LR Schedule ──────────────────────────────────────────────────────────
    warmup_steps: int = 50            # shorter warmup → full LR sooner  (↓ from 100)
    lr_decay_steps: int = 50_000      # cosine decay over this many steps
    min_lr_ratio: float = 0.1         # min LR = lr * min_lr_ratio

    # ── Training Budget ───────────────────────────────────────────────────────
    max_steps: int = 50_000
    save_every: int = 500             # checkpoint every N steps
    eval_every: int = 50              # inference check every N steps
    log_every: int = 1                # log every step

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    resume: bool = True               # auto-resume from latest checkpoint

    # ── Device / Precision ────────────────────────────────────────────────────
    device: str = "auto"              # "auto" | "cuda" | "mps" | "cpu"
    pin_memory: bool = False          # set True automatically when CUDA is used
    num_workers: int = 0              # streaming dataset — no workers
    dtype: str = "auto"               # "auto" → bfloat16 on GPU/MPS, float32 on CPU

    # ── Inference check prompts ───────────────────────────────────────────────
    inference_prompts: list = field(default_factory=lambda: [
        "The theory of relativity states that",
        "Once upon a time in a kingdom",
        "The best way to learn mathematics is",
    ])
    inference_max_new_tokens: int = 64
    inference_temperature: float = 0.8
    inference_top_k: int = 40


# ── Singleton instances ────────────────────────────────────────────────────────
model_cfg = ModelConfig()
train_cfg = TrainConfig()
