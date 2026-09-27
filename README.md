# SLM — Small Language Model

A **7.3 M-parameter** transformer trained from scratch on your CPU.
Built for Intel Core Ultra 5 225H + Iris iGPU, engineered to stay under **7 GB RAM**.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Layout](#project-layout)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Resume & Checkpoints](#resume--checkpoints)
7. [Inference](#inference)
8. [Logger & Logs](#logger--logs)
9. [Memory Budget](#memory-budget)
10. [Design Decisions](#design-decisions)

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | **7 342 336** |
| Vocabulary | 8 192 tokens (BPE, English-only) |
| Context length | 512 tokens |
| Embedding dim (`d_model`) | 256 |
| Transformer layers | 8 |
| Attention heads (Q) | 8 |
| KV heads (GQA) | 2 |
| FFN hidden dim (`d_ff`) | 640 |
| Positional encoding | RoPE |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Residual style | Parallel Attn + FFN (PaLM-style) |
| Weight tying | embed ↔ lm\_head |
| Bias | None |

### Bleeding-edge techniques used

```
RMSNorm          — no mean-centering, cheaper than LayerNorm
RoPE             — relative positions, extrapolates to longer sequences
GQA (4× KV)     — 8 Q-heads share 2 KV-heads; 4× KV memory savings
SwiGLU           — gated activation; consistently outperforms GELU FFN
Parallel blocks  — Attn and FFN run on the same pre-norm state (PaLM)
Muon optimizer   — Newton-Schulz orthogonalised SGD for matrix weights
Weight tying     — embed and lm_head share the same matrix (~2M saved)
```

---

## Project Layout

```
slm/
├── config.py            ← All hyperparameters (single source of truth)
├── model.py             ← SLM architecture (RoPE · GQA · SwiGLU · RMSNorm)
├── optimizer.py         ← Muon optimizer + AdamW companion + LR schedule
├── dataset.py           ← FineWeb-Edu streaming pipeline
├── logger.py            ← ANSI terminal logger + JSONL writer
├── checkpoint.py        ← Save / load + graceful Ctrl+C handler
├── inference.py         ← Generation (top-k/p, rep-penalty, streaming)
├── train.py             ← Main training loop
├── tokenizer_train.py   ← BPE tokenizer training (run once)
├── setup_check.py       ← Pre-flight verification script
├── requirements.txt
│
├── tokenizer/           ← Created by tokenizer_train.py
│   ├── tokenizer.json
│   └── config.json
│
├── checkpoints/         ← Created during training
│   └── step-0000500/
│       ├── model.pt
│       ├── muon.pt
│       ├── adamw.pt
│       └── train_state.json
│
└── logs/                ← Created during training
    ├── log-1.jsonl
    ├── log-2.jsonl
    └── ...
```

---

## Quick Start

### 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

> **PyTorch CPU build** is sufficient — no CUDA needed.
> For faster matmuls, install the MKL-optimised wheel:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 2 — Verify setup

```powershell
python setup_check.py
```

Expected output:
```
✔  Python 3.x.x
✔  PyTorch 2.x.x
✔  MKL available
✔  tokenizers x.x
✔  datasets x.x
✔  psutil x.x
ℹ  Total RAM : 15.4 GB
ℹ  TOTAL estimate : ~240 MB  (0.23 GB)
✔  Fits within 7.0 GB budget
⚠  Tokenizer NOT found          ← expected on first run
```

### 3 — Train the tokenizer  *(once)*

```powershell
python tokenizer_train.py
```

This streams ~80 MB of FineWeb-Edu text and trains an 8 192-token BPE vocabulary.
Takes roughly **5–10 minutes** on a home internet connection.

### 4 — Start training

```powershell
python train.py
```

---

## Configuration

All knobs are in [`config.py`](config.py). The two dataclasses:

### `ModelConfig`

| Field | Default | Description |
|---|---|---|
| `vocab_size` | 8192 | BPE vocabulary size |
| `context_len` | 512 | Maximum sequence length |
| `d_model` | 256 | Hidden / embedding dimension |
| `n_heads` | 8 | Query attention heads |
| `n_kv_heads` | 2 | Key/Value heads (GQA) |
| `n_layers` | 8 | Transformer blocks |
| `d_ff_mult` | 2.667 | FFN width multiplier |
| `rope_base` | 10 000 | RoPE frequency base |

### `TrainConfig`

| Field | Default | Description |
|---|---|---|
| `batch_size` | 4 | Sequences per micro-step |
| `grad_accum_steps` | 8 | Effective batch = 4 × 8 = **32** |
| `context_len` | 512 | Sequence length |
| `lr` | 3e-3 | Peak learning rate (Muon) |
| `weight_decay` | 0.1 | Decoupled L2 penalty |
| `muon_momentum` | 0.95 | Nesterov momentum for Muon |
| `warmup_steps` | 100 | Linear LR warmup |
| `lr_decay_steps` | 50 000 | Cosine decay over N steps |
| `max_steps` | 50 000 | Total optimizer steps |
| `save_every` | 500 | Checkpoint interval |
| `eval_every` | 50 | Inference check interval |
| `grad_clip` | 1.0 | Gradient clipping norm |
| `dataset_config` | `sample-10BT` | FineWeb-Edu subset |

---

## Training

```powershell
# Start fresh
python train.py

# Start fresh, ignoring any existing checkpoints
python train.py --no-resume

# Resume from a specific checkpoint
python train.py --resume checkpoints/step-0001000
```

### What the terminal shows

```
══════════════════════════════════════════════════════════════════
                    ✦  SLM Pre-Training  ✦
══════════════════════════════════════════════════════════════════

  Parameters  :  7,342,336
  Vocab size  :  8,192
  Max steps   :  50,000
  Grad accum  :  8
  Log file    :  logs/log-1.jsonl
  Memory      :  ████████░░░░░░░░░░░░ 4.1/15.4GB (27%)

──────────────────────────────────────────────────────────────────
  STEP              LOSS          LR          GRAD NORM   TOKENS   …  TREND
  ──────────────────────────────────────────────────────────────────
  [     1/50000]    9.01234       3.00e-05    0.9821      16.4K    …  ▁
  [     2/50000]    8.84521       6.00e-05    0.9134      32.8K    …  ▁▂
  ...
  [    50/50000] 🔍  7.23100       1.50e-03    0.7442      819.2K   …  ▁▂▃▄▄▅

  ┌────────────────────────────────────────────────┐
  │  🔍  Inference Check @ step 50
  │
  │  PROMPT:  The theory of relativity states that
  │  MODEL :  the universe is a very large number of ...
  └────────────────────────────────────────────────┘
```

- **Loss sparkline** — last 20 steps shown as Unicode block characters
- **Memory gauge** — live RAM bar, turns yellow >65 %, red >85 %
- **💾** next to step = checkpoint being saved
- **🔍** next to step = inference check being run
- **↩ Resuming** banner shown when continuing a previous run

---

## Resume & Checkpoints

### Automatic resume

`train_cfg.resume = True` (default). On startup, `train.py` finds the **highest-numbered** `checkpoints/step-XXXXXXX/` directory and loads:

```
model.pt          model weights
muon.pt           Muon optimizer momentum buffers
adamw.pt          AdamW moment estimates
train_state.json  step, tokens_consumed, torch RNG state
```

Dataset position is recovered from `tokens_consumed` via document-skip heuristic (avg 512 tokens/doc).

### Ctrl+C — emergency save

Press **Ctrl+C** at any point during training:

```
  ⚡  Ctrl+C detected — saving emergency checkpoint …
  💾 Checkpoint saved → checkpoints/step-0003217  (step 3217)
```

The interrupt is caught between micro-steps so **no gradient update is lost**.
Re-run `python train.py` to continue exactly from that step.

### Checkpoint retention

Only the **3 most recent** checkpoints are kept (configurable via `keep_last_n` in [`checkpoint.py`](checkpoint.py)) to save disk space.

### Manual checkpoint management

```powershell
# List checkpoints
Get-ChildItem checkpoints

# Resume from a specific one
python train.py --resume checkpoints/step-0002000
```

---

## Inference

### Interactive REPL

```powershell
python inference.py
```

```
SLM Interactive Inference  (step 50,000)
temperature=0.8  top_k=40  top_p=0.95

>>> The theory of relativity states that
The theory of relativity states that space and time are not absolute ...

[87 tokens  2.34s  37.2 tok/s]

>>> quit
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
| `--max-tokens`, `-m` | 200 | Max new tokens |
| `--temperature`, `-t` | 0.8 | Sampling temperature |
| `--top-k`, `-k` | 40 | Top-K filter |
| `--top-p` | 0.95 | Nucleus sampling threshold |
| `--rep-penalty`, `-r` | 1.1 | Repetition penalty |
| `--no-stream` | off | Print full output at once |

---

## Logger & Logs

### Terminal logger (`logger.py`)

The logger writes to `stdout` with full ANSI colour support (Windows 10+):

- **Step row** — step/total, loss (colour-coded by trend), LR, grad norm, tokens, elapsed, ETA, sparkline
- **Memory warning** — printed every 10 steps if RAM > 80 %
- **Inference block** — framed box showing prompt and generated text
- **Checkpoint notice** — path and step number

### JSONL log files (`logs/log-N.jsonl`)

Each training run creates a **new** `log-N.jsonl` file (N increments automatically).
Each line is a JSON object:

**Step record:**
```json
{"step": 100, "loss": 7.12345, "lr": 0.003, "grad_norm": 0.8821,
 "tokens_total": 1638400, "elapsed_s": 142.3,
 "is_checkpoint": false, "is_inference": true}
```

**Inference record:**
```json
{"step": 100, "type": "inference",
 "prompt": "The theory of relativity states that",
 "generated": "The theory of relativity states that space and time ..."}
```

Parse logs with standard tools:

```python
import json
with open("logs/log-1.jsonl") as f:
    records = [json.loads(line) for line in f]
losses = [r["loss"] for r in records if "loss" in r]
```

---

## Memory Budget

Target: **≤ 7 GB** process RAM (Windows uses ~7.7 GB baseline on a 15.4 GB system).

| Component | Size |
|---|---|
| Model weights (float32) | ~28 MB |
| Gradients | ~28 MB |
| Muon momentum buffers | ~22 MB |
| AdamW m + v buffers | ~34 MB |
| Activations (batch=4, seq=512) | ~20 MB |
| PyTorch overhead + datasets cache | ~200 MB |
| **Total estimate** | **~330 MB** |

The model itself is extremely lightweight. The bulk of your 7 GB budget is consumed by:
- Python interpreter + stdlib (~100 MB)
- HuggingFace `datasets` + streaming cache (~200–400 MB)
- Windows process overhead (~50 MB)

To **reduce** memory further:
```python
# config.py
train_cfg.batch_size = 2        # halves activation memory
train_cfg.context_len = 256     # quarters activation memory
train_cfg.grad_accum_steps = 16 # compensate for smaller batch
```

---

## Design Decisions

### Why Muon instead of AdamW for matrix weights?

Muon orthogonalises the gradient update via Newton-Schulz iterations, approximating steepest descent in spectral norm. This gives better loss-per-step than AdamW at the same wall-clock cost on CPU, especially for small models where the Newton-Schulz iterations (5 passes over a small matrix) are cheap.

Embed / norm / bias parameters still use AdamW — Muon only makes sense for 2-D weight matrices.

### Why GQA (2 KV heads for 8 Q heads)?

At 4× KV compression, the KV projection and KV cache shrink substantially with minimal perplexity cost. Especially beneficial during inference where KV tensors grow with sequence length.

### Why Parallel blocks?

In the standard Llama/GPT-2 style, each block does:
```
x = x + Attn(norm(x))
x = x + FFN(norm(x))
```
The parallel (PaLM) style does:
```
x = x + Attn(norm(x)) + FFN(norm(x))
```
One `norm` call per block instead of two, and better gradient flow for small models.

### Why SwiGLU?

SwiGLU consistently outperforms ReLU and GELU on language modelling benchmarks. The gating mechanism (`silu(gate(x)) * up(x)`) acts as a learned feature selector at each FFN position.

### Why RMSNorm instead of LayerNorm?

RMSNorm skips the mean-centering step, making it ~15 % faster. For small models the difference in training stability is negligible, but the speed improvement is free.

### Why streaming dataset?

FineWeb-Edu is ~250 GB uncompressed. Downloading it fully is impractical. Streaming processes one document at a time with no local cache beyond what HuggingFace buffers (~a few hundred MB), keeping disk usage near zero.

---

## License

MIT — do whatever you want with it.
