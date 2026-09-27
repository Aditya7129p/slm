"""
Configuration for the ~150M-parameter SLM — Kaggle 2× T4 GPU (DDP).

Architecture: d_model=960, n_layers=16, n_heads=16, GQA kv=4, context=1024
  -> ~152 M trainable parameters  (weight-tied embed + lm_head)

Hardware target: Kaggle free tier — 2× NVIDIA T4 (14.6 GB usable VRAM each)
DDP strategy: torch.distributed, nproc_per_node=2, backend="nccl"

VRAM budget per GPU (bfloat16, gradient checkpointing ON):
  +-------------------------------------------------+-----------+
  | Model weights (bfloat16, 152M params)           | ~304 MB   |
  | Gradients    (bfloat16)                         | ~304 MB   |
  | Muon momentum buffers (~130M matrix params)     | ~260 MB   |
  | AdamW m + v states (embed/norm ~22M params)     |  ~88 MB   |
  | Activations B=2, T=1024 (grad checkpoint ON)    | ~400 MB   |
  | DDP gradient buckets                            |  ~50 MB   |
  | Dataset stream + tokenizer                      | ~200 MB   |
  | PyTorch / NCCL overhead                         | ~800 MB   |
  +-------------------------------------------------+-----------+
  | Total per GPU                                   | ~2.4 GB   |
  +-------------------------------------------------+-----------+
  -> Fits comfortably inside the 14.6 GB T4 usable VRAM.

Why the original batch_size=8 OOM'd:
  Without gradient checkpointing, each forward pass stores ALL intermediate
  activations for backprop.  For B=8, T=1024, D=960, L=16 that is roughly:
    8 × 1024 × 960 × 16 × ~8 tensors × 2 bytes ≈ 10+ GB of activations alone.
  Adding model weights (304 MB) + grads + optimizer states blows past 14.6 GB.

  With gradient checkpointing (torch.utils.checkpoint):
    Only block inputs are retained; activations are recomputed on the backward
    pass. Peak activation memory drops from ~10 GB to ~400 MB at the cost of
    ~30 % extra compute (one extra forward per block).

DDP notes:
  - Each GPU processes batch_size=2 sequences per micro-step.
  - grad_accum_steps=32 keeps effective tokens/step the same as before:
      2 GPUs × batch_size(2) × grad_accum(32) × context(1024)
      = 131,072 tokens per optimizer step.
  - Rank 0 handles checkpointing and logging; other ranks are silent.

Training budget:
  - max_steps=50,000 → ~6.5 B tokens total (Chinchilla-optimal for 150M).
  - At ~50 k tokens/s on 2×T4 with grad-ckpt overhead, expect ~36 h total.
  - Split into ~4 Kaggle sessions of ~9h each, auto-resuming via checkpoints.

To run on Kaggle (in the notebook):
    torchrun --nproc_per_node=2 train_kaggle.py
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size:  int   = 32_768   # same vocab as 100M — tokenizer reusable
    context_len: int   = 1024
    d_model:     int   = 960      # head_dim = 960 // 16 = 60
    n_heads:     int   = 16       # Q heads
    n_kv_heads:  int   = 4        # GQA: 4× KV savings  (16/4 = 4 rep factor)
    n_layers:    int   = 16       # transformer blocks
    d_ff_mult:   float = 2.667    # SwiGLU FFN: round(960 × 2.667 / 64) × 64 = 2560
    dropout:     float = 0.0      # disabled — DDP + large batches regularise well
    bias:        bool  = False
    norm_eps:    float = 1e-6

    # ── RoPE ─────────────────────────────────────────────────────────────────
    rope_base:    float          = 10_000.0
    rope_scaling: Optional[float] = None

    @property
    def d_ff(self) -> int:
        raw = int(self.d_model * self.d_ff_mult)
        return (raw // 64) * 64 or 64


@dataclass
class TrainConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_name:   str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split:  str = "train"

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Reuse the 100M tokenizer — same vocab size (32 768).
    tokenizer_path:       str = "tokenizer_100m"
    tokenizer_vocab_size: int = 32_768

    # ── Batch / Sequence ──────────────────────────────────────────────────────
    # batch_size=2: small per-GPU micro-batch keeps activation memory low.
    # grad_accum_steps=32: preserves the same effective batch as the original
    #   batch_size=8 / grad_accum=8 plan:
    #   2 GPUs × 2 seqs × 32 accum × 1024 tokens = 131,072 tokens/step.
    # Gradient checkpointing (enabled in train_kaggle.py) drops peak activation
    #   memory from ~10 GB to ~400 MB — fits cleanly on 14.6 GB T4 VRAM.
    batch_size:       int = 3    # per-GPU micro-batch (OOM-safe for T4)
    grad_accum_steps: int = 32   # effective batch unchanged: 2×2×32×1024 = 131,072
    context_len:      int = 1024

    # ── Optimiser (Muon + AdamW) ──────────────────────────────────────────────
    lr:             float = 3e-3
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0
    muon_momentum:  float = 0.95
    embed_lr_scale: float = 0.1

    # ── LR Schedule ──────────────────────────────────────────────────────────
    warmup_steps:   int   = 250   # stable warmup for 150M scale
    lr_decay_steps: int   = 50_000
    min_lr_ratio:   float = 0.05

    # ── Training Budget ───────────────────────────────────────────────────────
    max_steps:  int = 50_000
    save_every: int = 500
    eval_every: int = 200
    log_every:  int = 1

    # ── Validation ────────────────────────────────────────────────────────────
    val_batches: int = 16   # 16 × 2 × 1024 = 32,768 tokens per val pass

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str  = "checkpoints_kaggle"
    log_dir:        str  = "logs_kaggle"
    resume:         bool = True

    # ── Device / Precision ────────────────────────────────────────────────────
    device:      str  = "cuda"    # DDP launcher sets the correct device per rank
    pin_memory:  bool = True
    num_workers: int  = 2
    dtype:       str  = "bfloat16"  # T4 supports bfloat16 natively

    # ── DDP ──────────────────────────────────────────────────────────────────
    # These are read by train_kaggle.py; not used by the base train.py.
    ddp_backend:    str  = "nccl"   # NCCL for GPU-to-GPU communication
    find_unused_params: bool = False

    # ── Inference check prompts ───────────────────────────────────────────────
    inference_prompts: list = field(default_factory=lambda: [
        "The theory of relativity states that",
        "Once upon a time in a kingdom",
        "The best way to learn mathematics is",
    ])
    inference_max_new_tokens: int   = 64
    inference_temperature:    float = 0.8
    inference_top_k:          int   = 40


# ── Singleton instances ────────────────────────────────────────────────────────
model_cfg = ModelConfig()
train_cfg = TrainConfig()


# ── Quick param-count + VRAM check  (python config_kaggle.py) ─────────────────
if __name__ == "__main__":
    cfg  = model_cfg
    tcfg = train_cfg
    d    = cfg.d_model
    L    = cfg.n_layers
    V    = cfg.vocab_size
    H    = cfg.n_heads
    Hkv  = cfg.n_kv_heads
    hdim = d // H
    B    = tcfg.batch_size
    T    = cfg.context_len

    # Attention: q_proj + k_proj + v_proj + o_proj
    attn_per_layer = d * H * hdim + d * Hkv * hdim * 2 + H * hdim * d
    # SwiGLU FFN: gate + up + down
    dff = cfg.d_ff
    ffn_per_layer  = d * dff * 2 + dff * d
    # RMSNorm per block (weight only, no bias)
    norm_per_layer = d
    # Totals
    layer_params = L * (attn_per_layer + ffn_per_layer + norm_per_layer)
    embed_params = V * d   # tied with lm_head, counted once
    total        = layer_params + embed_params + d  # +d for final norm

    bytes_per = 2  # bfloat16
    weights_mb = total * bytes_per / 1024**2
    grads_mb   = weights_mb
    matrix_params = L * (attn_per_layer + ffn_per_layer)
    other_params  = total - matrix_params
    opt_mb = (matrix_params * bytes_per + other_params * bytes_per * 2) / 1024**2
    act_mb    = B * T * d * L * bytes_per / 1024**2
    ddp_mb    = 50    # DDP gradient buckets (approx)
    overhead  = 800   # PyTorch / NCCL / dataset
    total_mb  = weights_mb + grads_mb + opt_mb + act_mb + ddp_mb + overhead
    eff_toks_per_gpu = B * tcfg.grad_accum_steps * T
    eff_toks_global  = eff_toks_per_gpu * 2   # 2 GPUs

    SEP = "─" * 55
    print(f"\n{SEP}")
    print(f"  Architecture  (Kaggle 150M)")
    print(SEP)
    print(f"  d_model      : {d}")
    print(f"  n_layers     : {L}")
    print(f"  n_heads      : {H}  (kv={Hkv}, head_dim={hdim})")
    print(f"  d_ff         : {dff}")
    print(f"  vocab_size   : {V:,}")
    print(f"  context_len  : {T}")
    print(f"  dtype        : bfloat16")
    print(f"\n  Total params : {total:,}  ({total/1e6:.1f} M)")
    print(f"\n{SEP}")
    print(f"  VRAM estimate  (per GPU, batch={B}, T={T}, bfloat16)")
    print(SEP)
    print(f"  Weights      : {weights_mb:.0f} MB")
    print(f"  Gradients    : {grads_mb:.0f} MB")
    print(f"  Optimizer    : ~{opt_mb:.0f} MB")
    print(f"  Activations  : ~{act_mb:.0f} MB  (flash-attn, O(T·d))")
    print(f"  DDP buckets  : ~{ddp_mb} MB")
    print(f"  Overhead     : ~{overhead} MB  (PyTorch / NCCL / dataset)")
    print(f"  {'─'*35}")
    print(f"  Est. per GPU : ~{total_mb:.0f} MB  (~{total_mb/1024:.1f} GB)")
    print(f"  T4 VRAM      : 16 384 MB  (headroom: ~{16384 - total_mb:.0f} MB)")
    print(f"\n{SEP}")
    print(f"  Training  (2× T4 DDP)")
    print(SEP)
    print(f"  Per-GPU batch        : {B} seqs × {tcfg.grad_accum_steps} accum = {B * tcfg.grad_accum_steps} seqs")
    print(f"  Global eff. toks/step: {eff_toks_global:,}")
    print(f"  Max steps            : {tcfg.max_steps:,}")
    print(f"  Total tokens         : ~{eff_toks_global * tcfg.max_steps / 1e9:.2f} B")
    print()
