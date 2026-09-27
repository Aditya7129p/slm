# SLM — Small Language Model

A transformer trained **from scratch**, available in four configurations:

| Config | Parameters | Context | Vocab | Target Hardware |
|---|---|---|---|---|
| `config.py` (default) | **~4 M** | 512 tokens | 8 192 | Any CPU, ≤ 7 GB RAM |
| `config_100m.py` | **~98 M** | 1 024 tokens | 32 768 | Intel Core Ultra 5 225H, ≤ 8 GB RAM |
| `config_colab.py` | **~4 M** | 512 tokens | 8 192 | Google Colab T4 / A100 |
| `config_kaggle.py` | **~152 M** | 1 024 tokens | 32 768 | Kaggle 2× T4 (DDP) |

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Layout](#project-layout)
3. [Quick Start](#quick-start)
4. [Configs](#configs)
5. [Training](#training)
6. [Resume & Checkpoints](#resume--checkpoints)
7. [Inference](#inference)
8. [Logger & Logs](#logger--logs)
9. [Memory Budget](#memory-budget)
10. [Design Decisions](#design-decisions)

---

## Architecture

All configs share the same architecture family. Only the scale changes.

### 4M config (`config.py`)

| Hyperparameter | Value |
|---|---|
| Parameters | **~4 M** |
| Vocabulary | 8 192 tokens (BPE) |
| Context length | 512 tokens |
| `d_model` | 288 |
| Layers | 10 |
| Q heads | 8 |
| KV heads (GQA) | 2 |
| `d_ff` | 768 |

### 100M config (`config_100m.py`)

| Hyperparameter | Value |
|---|---|
| Parameters | **~98 M** |
| Vocabulary | 32 768 tokens (BPE) |
| Context length | 1 024 tokens |
| `d_model` | 768 |
| Layers | 13 |
| Q heads | 12 |
| KV heads (GQA) | 3 |
| `d_ff` | 2 048 |
| dtype | bfloat16 (AVX-512 BF16) |

### 150M config (`config_kaggle.py`) — Kaggle 2× T4 DDP

| Hyperparameter | Value |
|---|---|
| Parameters | **~152 M** |
| Vocabulary | 32 768 tokens (BPE) |
| Context length | 1 024 tokens |
| `d_model` | 960 |
| Layers | 16 |
| Q heads | 16 |
| KV heads (GQA) | 4 |
| `d_ff` | 2 560 |
| DDP | 2× T4, NCCL backend |
| Global eff. batch | 131 072 tokens/step |
| dtype | bfloat16 |

### Techniques used by all configs

| Technique | Purpose |
|---|---|
| **RMSNorm** | No mean-centering — ~15 % cheaper than LayerNorm |
| **RoPE** | Relative positions, extrapolates to longer sequences |
| **GQA** | Q-heads share fewer KV-heads — 4× KV memory savings |
| **SwiGLU** | Gated activation — consistently beats GELU on LM benchmarks |
| **Parallel blocks** | Attn + FFN on same pre-norm state (PaLM-style), one fewer norm |
| **Muon optimizer** | Newton-Schulz orthogonalised SGD for 2-D weight matrices |
| **Weight tying** | `embed` and `lm_head` share the same matrix |
| **Flash attention** | `F.scaled_dot_product_attention` — O(T) memory, not O(T²) |

---

## Project Layout

```
slm/
├── config.py            ← Default 4M config (single source of truth)
├── config_100m.py       ← 100M config for Intel Core Ultra 5 225H
├── config_colab.py      ← Colab/GPU config (inherits 4M)
├── config_kaggle.py     ← 150M config for Kaggle 2× T4 DDP
│
├── model.py             ← SLM architecture (RoPE · GQA · SwiGLU · RMSNorm)
├── optimizer.py         ← Muon optimizer + AdamW companion + LR schedule
├── dataset.py           ← FineWeb-Edu streaming pipeline
├── logger.py            ← ANSI terminal logger + JSONL writer
├── checkpoint.py        ← Save / load + graceful Ctrl+C handler
├── inference.py         ← Generation (top-k, temperature, streaming)
├── train.py             ← Main training loop  (--config selects config)
├── train_colab.py       ← Colab single-GPU entry point
├── train_kaggle.py      ← Kaggle 2× T4 DDP entry point (torchrun)
├── tokenizer_train.py   ← BPE tokenizer training  (--config selects config)
├── setup_check.py       ← Pre-flight verification script
├── colab_train.ipynb    ← Google Colab notebook
├── kaggle_notebook.ipynb← Kaggle notebook (2× T4 DDP)
├── requirements.txt
│
├── tokenizer/           ← Created by tokenizer_train.py  (4M config, vocab 8 192)
├── tokenizer_100m/      ← Shared by 100M + 150M configs  (vocab 32 768)
│
├── checkpoints/         ← Default 4M checkpoints
├── checkpoints_100m/    ← 100M checkpoints
├── checkpoints_kaggle/  ← 150M Kaggle DDP checkpoints
│
├── logs/                ← Default 4M JSONL logs
├── logs_100m/           ← 100M JSONL logs
└── logs_kaggle/         ← 150M Kaggle JSONL logs
```

---

## Quick Start

### 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

> **PyTorch CPU build** is sufficient — no CUDA needed.  
> For faster matmuls (MKL / AVX-512), use the official CPU wheel:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 2 — Verify setup

```powershell
python setup_check.py
```

### 3 — Train the tokenizer *(once per config)*

```powershell
# Default 4M config — vocab 8 192, saves to ./tokenizer/
python tokenizer_train.py

# 100M config — vocab 32 768, saves to path set in config_100m.py
python tokenizer_train.py --config config_100m

# Overwrite an existing tokenizer
python tokenizer_train.py --config config_100m --force
```

### 4 — Start training

```powershell
# Default 4M model
python train.py

# 100M model (Intel Core Ultra 5 225H)
python train.py --config config_100m

# 150M model on Kaggle (2× T4, DDP) — run inside kaggle_notebook.ipynb
torchrun --nproc_per_node=2 train_kaggle.py
```

---

## Configs

### Selecting a config

Both `train.py` and `tokenizer_train.py` accept a `--config` / `-c` flag that takes any config **module name** (filename without `.py`) from the project root:

```powershell
python train.py --config config          # 4M  (default)
python train.py --config config_100m     # 100M
python train.py --config config_colab    # Colab
```

The config module must define `model_cfg` and `train_cfg` dataclass instances. Each config is fully self-contained — it sets its own `checkpoint_dir`, `log_dir`, `tokenizer_path`, `dtype`, and all hyperparameters.

### Kaggle 150M DDP config (`config_kaggle.py`)

| Field | Value | Description |
|---|---|---|
| `d_model` | 960 | Hidden dimension |
| `n_layers` | 16 | Transformer blocks |
| `n_heads` | 16 | Query heads |
| `n_kv_heads` | 4 | GQA KV heads (4× savings) |
| `batch_size` | 8 | Per-GPU micro-batch |
| `grad_accum_steps` | 8 | Steps before optimizer update |
| `dtype` | `"bfloat16"` | Native T4 support |
| `ddp_backend` | `"nccl"` | GPU-to-GPU communication |
| `checkpoint_dir` | `checkpoints_kaggle` | Separate from other configs |
| `tokenizer_path` | `tokenizer_100m` | Reuses 32 768-vocab tokenizer |

### Creating a new config

Copy any existing config file, rename it (e.g. `config_experiment.py`), and modify the values. Then:

```powershell
python train.py --config config_experiment
```

### `ModelConfig` fields

| Field | 4M default | 100M default | Description |
|---|---|---|---|
| `vocab_size` | 8 192 | 32 768 | BPE vocabulary size |
| `context_len` | 512 | 1 024 | Maximum sequence length |
| `d_model` | 288 | 768 | Hidden / embedding dimension |
| `n_heads` | 8 | 12 | Query attention heads |
| `n_kv_heads` | 2 | 3 | KV heads (GQA) |
| `n_layers` | 10 | 13 | Transformer blocks |
| `d_ff_mult` | 2.667 | 2.667 | FFN width multiplier |
| `dropout` | 0.0 | 0.1 | Dropout (0 = disabled) |
| `rope_base` | 10 000 | 10 000 | RoPE frequency base |

### `TrainConfig` fields

| Field | 4M default | 100M default | Description |
|---|---|---|---|
| `batch_size` | 8 | 1 | Sequences per micro-step |
| `grad_accum_steps` | 16 | 64 | Steps before optimizer update |
| `context_len` | 512 | 1 024 | Must match `ModelConfig.context_len` |
| `lr` | 3e-3 | 3e-3 | Peak LR (Muon) |
| `warmup_steps` | 50 | 1 000 | Linear LR warmup |
| `lr_decay_steps` | 50 000 | 50 000 | Cosine decay over N steps |
| `min_lr_ratio` | 0.1 | 0.05 | Floor = `lr × min_lr_ratio` |
| `max_steps` | 50 000 | 50 000 | Total optimizer steps |
| `save_every` | 500 | 500 | Checkpoint interval |
| `eval_every` | 50 | 200 | Val loss + inference interval |
| `val_batches` | — | 16 | Micro-batches per val pass |
| `grad_clip` | 1.0 | 1.0 | Gradient clipping norm |
| `dtype` | `"auto"` | `"bfloat16"` | `"auto"` / `"float32"` / `"bfloat16"` |
| `checkpoint_dir` | `checkpoints` | `checkpoints_100m` | Where to save checkpoints |
| `log_dir` | `logs` | `logs_100m` | Where to write JSONL logs |
| `tokenizer_path` | `tokenizer` | `tokenizer_100m` | Tokenizer directory |

---

## Training

```powershell
# ── Default 4M config ────────────────────────────────────────────
python train.py                                          # auto-resume
python train.py --no-resume                              # from scratch
python train.py --resume checkpoints/step-0001000        # specific checkpoint

# ── 100M config (Intel Core Ultra 5 225H) ────────────────────────
python train.py --config config_100m
python train.py --config config_100m --no-resume
python train.py --config config_100m --resume checkpoints_100m/step-0000500

# ── 150M config (Kaggle 2× T4 DDP) ──────────────────────────────
# Run inside kaggle_notebook.ipynb, or:
torchrun --nproc_per_node=2 train_kaggle.py
torchrun --nproc_per_node=2 train_kaggle.py --no-resume
torchrun --nproc_per_node=2 train_kaggle.py --resume checkpoints_kaggle/step-0005000
```

### What the terminal shows

```
══════════════════════════════════════════════════════════════════
                    ✦  SLM Pre-Training  ✦
══════════════════════════════════════════════════════════════════

  Parameters  :  98,234,368
  Vocab size  :  32,768
  Max steps   :  50,000
  Grad accum  :  64
  Log file    :  logs_100m/log-1.jsonl
  Memory      :  ███░░░░░░░░░░░░░░░░░ 2.1/15.4GB (14%)

──────────────────────────────────────────────────────────────────
  STEP           LOSS       VAL LOSS   LR         GRAD NORM  …  TREND
  ──────────────────────────────────────────────────────────────
  [   1/50000]   10.8123    —          3.00e-06   0.9821     …  ▁
  [   2/50000]   10.5431    —          6.00e-06   0.9134     …  ▁▂
  [ 200/50000] 🔍 8.2341    8.4102     1.20e-03   0.7221     …  ▁▂▃▄▅▆
```

- **LOSS** — train loss, colour-coded green (decreasing) / yellow (plateau) / red (rising)
- **VAL LOSS** — held-out validation loss, computed every `eval_every` steps; shows last known value (dimmed) between evaluations
- **Trend sparkline** — last 16 train loss values as Unicode block characters
- **Memory gauge** — live RAM bar, turns yellow > 65 %, red > 85 %
- **💾** — checkpoint being saved at this step
- **🔍** — inference check and validation pass being run at this step

### Validation loss

A separate data generator (starting from token 0) is used for validation — the training position is never advanced. Each eval pass processes `val_batches × batch_size × context_len` tokens. Both train loss and val loss are recorded in the JSONL log.

---

## Resume & Checkpoints

### Automatic resume

`resume = True` (default in all configs). On startup, `train.py` finds the **highest-numbered** checkpoint directory inside `checkpoint_dir` and loads:

```
model.pt          model weights
muon.pt           Muon momentum buffers
adamw.pt          AdamW moment estimates
train_state.json  step, tokens_consumed, torch RNG state
```

Dataset position is recovered from `tokens_consumed` via a document-skip heuristic.

### Ctrl+C — emergency save

Press **Ctrl+C** at any point:

```
  ⚡  Ctrl+C detected — saving emergency checkpoint …
  💾 Checkpoint saved → checkpoints_100m/step-0003217  (step 3217)
```

The interrupt is caught between micro-steps so no partial gradient update is lost. Re-run the same command to continue from that exact step.

### Checkpoint retention

Only the **3 most recent** checkpoints are kept (configurable via `keep_last_n` in [`checkpoint.py`](checkpoint.py)).

---

## Inference

### Interactive REPL

```powershell
python inference.py
```

### Single-shot

```powershell
python inference.py --prompt "Once upon a time" --max-tokens 200
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint`, `-c` | latest | Path to checkpoint directory |
| `--prompt`, `-p` | — | Single prompt (non-interactive) |
| `--max-tokens`, `-m` | 200 | Max new tokens to generate |
| `--temperature`, `-t` | 0.8 | Sampling temperature |
| `--top-k`, `-k` | 40 | Top-K filter |
| `--top-p` | 0.95 | Nucleus sampling threshold |
| `--rep-penalty`, `-r` | 1.1 | Repetition penalty |
| `--no-stream` | off | Print all at once instead of streaming |

---

## Logger & Logs

### Terminal columns

| Column | Description |
|---|---|
| `STEP` | Current step / total, with 💾 / 🔍 badges |
| `LOSS` | Train loss — green if falling, yellow if flat, red if rising |
| `VAL LOSS` | Validation loss — same colour coding; dimmed when stale |
| `LR` | Current learning rate |
| `GRAD NORM` | Pre-clip gradient norm |
| `TOKENS` | Cumulative tokens consumed |
| `ELAPSED` | Wall-clock time since run start |
| `ETA` | Estimated time remaining |
| `TREND` | 16-char sparkline of recent train loss |

### JSONL log files

Each run writes a new `logs/log-N.jsonl` (or `logs_100m/log-N.jsonl`). Every line is a JSON object.

**Step record:**
```json
{
  "step": 200,
  "loss": 8.23410,
  "val_loss": 8.41020,
  "lr": 0.0012,
  "grad_norm": 0.7221,
  "tokens_total": 13107200,
  "elapsed_s": 843.1,
  "is_checkpoint": false,
  "is_inference": true
}
```

> `val_loss` is `null` on steps where validation was not run.

**Inference record:**
```json
{
  "step": 200,
  "type": "inference",
  "prompt": "The theory of relativity states that",
  "generated": "The theory of relativity states that space and time are ..."
}
```

**Parse logs:**
```python
import json

with open("logs_100m/log-1.jsonl") as f:
    records = [json.loads(line) for line in f]

train_loss = [(r["step"], r["loss"]) for r in records if "loss" in r]
val_loss   = [(r["step"], r["val_loss"]) for r in records
              if r.get("val_loss") is not None]
```

---

## Memory Budget

### 4M config (float32, batch=8, seq=512)

| Component | Size |
|---|---|
| Model weights | ~11 MB |
| Gradients | ~11 MB |
| Muon momentum | ~8 MB |
| AdamW m + v | ~13 MB |
| Activations | ~20 MB |
| PyTorch + datasets overhead | ~400 MB |
| **Total estimate** | **~460 MB** |

### 100M config (bfloat16, batch=1, seq=1024)

| Component | Size |
|---|---|
| Model weights | ~198 MB |
| Gradients | ~198 MB |
| Muon momentum | ~148 MB |
| AdamW m + v (embed/norm only) | ~51 MB |
| Activations (flash-attn, O(T·d)) | ~20 MB |
| Python / PyTorch / dataset overhead | ~800 MB |
| **Total estimate** | **~1 415 MB (~1.4 GB)** |

> The 100M config uses **bfloat16** throughout, cutting weight + gradient memory in half vs float32.  
> Intel Core Ultra 5 225H (Meteor Lake) has native **AVX-512 BF16** units giving ~1.5–2× throughput over float32.  
> If you see `NaN` or `inf` losses, add `dtype = "float32"` to your config.

### Why context_len=1024 fits in 8 GB

PyTorch ≥ 2.0 uses a **memory-efficient attention kernel** for `scaled_dot_product_attention` on CPU, keeping attention memory O(T·d) instead of O(T²). Without this, a T=1024, B=1, 12-head model would need ~50 MB per layer just for attention scores; with the kernel it's negligible.

---

## Design Decisions

### Why Muon instead of AdamW for matrix weights?

Muon orthogonalises the gradient update via Newton-Schulz iterations, approximating steepest descent in spectral norm. This gives better loss-per-step than AdamW at the same wall-clock cost — especially on CPU where the Newton-Schulz passes over small matrices are cheap. Embed / norm parameters still use AdamW; Muon only applies to 2-D weight matrices.

### Why GQA?

At 4× KV compression, KV projection and KV cache shrink substantially with minimal perplexity cost. In the 100M config, 12 Q-heads share 3 KV-heads (4× rep factor), saving ~25 % of attention parameters and speeding up both training and inference.

### Why Parallel blocks (PaLM-style)?

Standard Llama-style blocks apply norm twice per block:
```
x = x + Attn(norm(x))
x = x + FFN(norm(x))
```
Parallel blocks apply it once:
```
x = x + Attn(norm(x)) + FFN(norm(x))
```
One fewer RMSNorm call per block, and better gradient flow on small-to-medium models.

### Why SwiGLU?

SwiGLU consistently outperforms ReLU and GELU on language modelling. The gating mechanism (`silu(gate(x)) * up(x)`) acts as a learned per-position feature selector inside the FFN.

### Why RMSNorm?

RMSNorm skips mean-centering, making it ~15 % faster than LayerNorm with negligible effect on training stability at this scale.

### Why streaming dataset?

FineWeb-Edu is ~250 GB uncompressed. Streaming processes one document at a time with no local cache beyond HuggingFace's buffer (~a few hundred MB), keeping disk usage near zero.

### Why bfloat16 on the 100M config?

Intel Core Ultra 5 225H (Meteor Lake) includes AVX-512 BF16 VNNI instructions that accelerate bfloat16 matrix multiplications natively. PyTorch's MKL/oneDNN backend uses these automatically, giving ~1.5–2× throughput over float32 with the same numerical range (8-bit exponent, vs float16's 5-bit exponent — no loss scaling needed).

---

## License

MIT — do whatever you want with it.
