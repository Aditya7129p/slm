"""
Configuration for the ~98M-parameter SLM.
Optimised for training FROM SCRATCH on an Intel Core Ultra 5 225H laptop
with a strict 8 GB RAM budget (16 GB total, 8 GB reserved for OS + other apps).

Architecture: d_model=768, n_layers=13, n_heads=12, GQA kv=3, context=1024
  -> ~98 M trainable parameters  (weight-tied embed + lm_head)

RAM budget breakdown (bfloat16, worst-case with context_len=1024):
  +-----------------------------------------+---------+
  | Model weights (bfloat16)                | ~198 MB |
  | Gradients    (bfloat16)                 | ~198 MB |
  | Muon momentum buffers                   | ~148 MB |
  | AdamW m + v states (embed/norm only)    |  ~51 MB |
  | Activations B=1, T=1024 (flash O(T*d)) |  ~20 MB |
  | Dataset stream + tokenizer              | ~200 MB |
  | Python / PyTorch overhead               | ~800 MB |
  +-----------------------------------------+---------+
  | Total                                   | ~1.6 GB |
  +-----------------------------------------+---------+
  -> Well inside the 8 GB limit with ~6.4 GB headroom.

  NOTE: PyTorch >= 2.0 uses a memory-efficient math kernel for
  scaled_dot_product_attention on CPU, keeping attention memory
  O(T*d) instead of O(T^2). This is what makes context_len=1024
  viable at 8 GB. bfloat16 on Intel AVX-512 BF16 is ~1.5-2x faster
  than float32 — if you see NaN/inf, fall back to dtype="float32".

Speed notes (Intel Core Ultra 5 225H, 4P+8E cores, AVX-512 BF16):
  - Expect ~2-5 min/step on CPU (98M params, T=1024).
  - torch.set_num_threads is set in train.py to pin all P-cores.
  - Effective batch = batch_size(1) x grad_accum(64) x context_len(1024)
    = 65,536 tokens per optimizer step — healthy for loss convergence.
  - max_steps=50,000 -> ~3.3 B tokens total.

Loss-convergence improvements vs the 4M config:
  - 65K effective tokens/step (4x more than 4M config) -> less gradient noise.
  - context_len=1024 -> model sees long-range structure from step 1.
  - Warmup extended to 1000 steps -> stable LR ramp for 100M-scale init.
  - Cosine decay over full 50K steps -> LR stays high longer early on.
  - min_lr_ratio=0.05 -> sharper final annealing squeezes out last bits.
  - Dropout=0.1 -> prevents overfit on repeated streaming docs.
  - Validation loss every 200 steps to monitor the generalisation gap.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Architecture ──────────────────────────────────────────────────────────
    vocab_size:  int   = 32_768   # larger vocab -> better subword coverage
    context_len: int   = 1024     # full 1024-token context window
    d_model:     int   = 768      # embedding / hidden dimension
    n_heads:     int   = 12       # attention heads  (head_dim = 64)
    n_kv_heads:  int   = 3        # GQA: 4x KV savings  (12/3 = 4 rep factor)
    n_layers:    int   = 13       # transformer blocks  -> ~98 M params
    d_ff_mult:   float = 2.667    # SwiGLU width = round(d_model * mult / 64) * 64
    dropout:     float = 0.1      # light regularisation; set 0.0 to disable
    bias:        bool  = False    # no bias -> faster matmuls, cleaner gradients
    norm_eps:    float = 1e-6

    # ── RoPE ─────────────────────────────────────────────────────────────────
    rope_base:    float          = 10_000.0
    rope_scaling: Optional[float] = None   # set e.g. 4.0 for long-context later

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
    tokenizer_path:       str = "tokenizer_100m"  # separate dir — different vocab size (32 768 vs 8 192)
    tokenizer_vocab_size: int = 32_768

    # ── Batch / Sequence ──────────────────────────────────────────────────────
    # batch_size=1: keeps peak activation RAM minimal at T=1024.
    # grad_accum=64: effective batch = 64 x 1024 = 65,536 tokens/step.
    # Larger effective batch -> lower gradient variance -> faster loss drop.
    batch_size:       int = 1
    grad_accum_steps: int = 64
    context_len:      int = 1024

    # ── Optimiser (Muon + AdamW) ──────────────────────────────────────────────
    lr:             float = 3e-3   # Muon peak LR — robust at this scale
    weight_decay:   float = 0.1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0
    muon_momentum:  float = 0.95
    embed_lr_scale: float = 0.1    # embed/lm_head AdamW LR = lr * this

    # ── LR Schedule ──────────────────────────────────────────────────────────
    warmup_steps:   int   = 1_000  # longer warmup stabilises 100M-scale init
    lr_decay_steps: int   = 50_000 # cosine decay over the full run
    min_lr_ratio:   float = 0.05   # decay to 5% of peak LR at the end

    # ── Training Budget ───────────────────────────────────────────────────────
    max_steps:  int = 50_000
    save_every: int = 500          # checkpoint every 500 steps
    eval_every: int = 200          # val loss + inference check every 200 steps
    log_every:  int = 1            # log every optimizer step

    # ── Validation ────────────────────────────────────────────────────────────
    # Separate stream generator; never advances training position.
    # 16 x 1 x 1024 = 16,384 tokens evaluated per val pass.
    val_batches: int = 16

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str  = "checkpoints_100m"
    log_dir:        str  = "logs_100m"
    resume:         bool = True

    # ── Device / Precision ────────────────────────────────────────────────────
    # Intel Core Ultra 5 225H — CPU-only for PyTorch.
    # bfloat16 is natively accelerated on AVX-512 BF16 (Meteor Lake).
    # Fallback: set dtype="float32" if any NaN/inf appear.
    device:      str  = "cpu"
    pin_memory:  bool = False
    num_workers: int  = 0
    dtype:       str  = "bfloat16"  # ~1.5-2x faster on Intel AVX-512 BF16

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


# ── Quick param-count + RAM check  (python config_100m.py) ────────────────────
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
    # RMSNorm per block
    norm_per_layer = d
    # Totals
    layer_params = L * (attn_per_layer + ffn_per_layer + norm_per_layer)
    embed_params = V * d   # tied with lm_head, counted once
    total        = layer_params + embed_params + d  # +d for final norm

    bytes_per = 2 if tcfg.dtype == "bfloat16" else 4
    dtype_str = tcfg.dtype

    weights_mb = total * bytes_per / 1024**2
    grads_mb   = weights_mb
    # Muon keeps momentum only for 2-D weight matrices
    matrix_params = L * (attn_per_layer + ffn_per_layer)
    other_params  = total - matrix_params
    opt_mb = (matrix_params * bytes_per + other_params * bytes_per * 2) / 1024**2
    # Flash-attn: activation cost is O(T*d) per layer
    act_mb    = B * T * d * L * bytes_per / 1024**2
    overhead  = 800
    total_mb  = weights_mb + grads_mb + opt_mb + act_mb + overhead
    eff_toks  = B * tcfg.grad_accum_steps * T

    SEP = "─" * 50
    print(f"\n{SEP}")
    print(f"  Architecture")
    print(SEP)
    print(f"  d_model      : {d}")
    print(f"  n_layers     : {L}")
    print(f"  n_heads      : {H}  (kv={Hkv}, head_dim={hdim})")
    print(f"  d_ff         : {dff}")
    print(f"  vocab_size   : {V:,}")
    print(f"  context_len  : {T}")
    print(f"  dtype        : {dtype_str}")
    print(f"\n  Total params : {total:,}  ({total/1e6:.1f} M)")
    print(f"\n{SEP}")
    print(f"  RAM estimate  (batch={B}, T={T}, {dtype_str})")
    print(SEP)
    print(f"  Weights      : {weights_mb:.0f} MB")
    print(f"  Gradients    : {grads_mb:.0f} MB")
    print(f"  Optimizer    : ~{opt_mb:.0f} MB")
    print(f"  Activations  : ~{act_mb:.0f} MB  (flash-attn, O(T·d))")
    print(f"  Overhead     : ~{overhead} MB  (Python / PyTorch / dataset)")
    print(f"  {'─'*30}")
    print(f"  Est. total   : ~{total_mb:.0f} MB  (~{total_mb/1024:.1f} GB)")
    print(f"\n{SEP}")
    print(f"  Training")
    print(SEP)
    print(f"  Eff. tokens/step : {eff_toks:,}")
    print(f"  Max steps        : {tcfg.max_steps:,}")
    print(f"  Total tokens     : ~{eff_toks * tcfg.max_steps / 1e9:.2f} B")
    print()
