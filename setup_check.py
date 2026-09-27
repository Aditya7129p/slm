"""
Pre-flight checks for the SLM training setup.

Run this before training:  python setup_check.py

It verifies:
  • Python & PyTorch versions
  • Available RAM
  • Memory budget estimation for the configured model
  • Tokenizer presence
  • Checkpoint directory writability
"""

import sys
import os
import math

# ── Colours (same minimal set as logger) ──────────────────────────────────────
os.system("")
R = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"

def ok(msg):  print(f"  {GREEN}✔{R}  {msg}")
def warn(msg):print(f"  {YELLOW}⚠{R}  {msg}")
def err(msg): print(f"  {RED}✘{R}  {msg}")
def info(msg):print(f"  {CYAN}ℹ{R}  {msg}")
def hdr(msg): print(f"\n{BOLD}{msg}{R}")


def check_python():
    hdr("Python")
    v = sys.version_info
    if v >= (3, 9):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        err(f"Python {v.major}.{v.minor}.{v.micro} — need 3.9+")


def check_torch():
    hdr("PyTorch")
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")
        if torch.backends.mkl.is_available():
            ok("MKL available (faster CPU matmuls)")
        else:
            warn("MKL not available — training will be slower")
        # Check bf16 support (optional but nice on newer CPUs)
        if hasattr(torch, "bfloat16"):
            try:
                t = torch.tensor([1.0]).to(torch.bfloat16)
                info("bfloat16 dtype supported")
            except Exception:
                pass
    except ImportError:
        err("PyTorch not installed — run: pip install torch")


def check_deps():
    hdr("Dependencies")
    deps = {
        "tokenizers": "0.19",
        "datasets":   "2.18",
        "psutil":     "5.9",
    }
    for pkg, min_ver in deps.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            ok(f"{pkg} {ver}")
        except ImportError:
            warn(f"{pkg} not installed — run: pip install {pkg}")


def check_memory():
    hdr("Memory Budget")
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1024**3
        avail_gb = mem.available / 1024**3
        used_gb  = mem.used / 1024**3

        info(f"Total RAM : {total_gb:.1f} GB")
        info(f"Used  RAM : {used_gb:.1f} GB")
        info(f"Avail RAM : {avail_gb:.1f} GB")

        # Estimate model memory
        from config import model_cfg, train_cfg
        n_params = (
            model_cfg.vocab_size * model_cfg.d_model        # embed/head (tied)
            + model_cfg.n_layers * (
                4 * model_cfg.d_model * model_cfg.d_model   # rough attn
                + 2 * model_cfg.d_model * model_cfg.d_ff    # FFN
            )
        )
        weights_mb  = n_params * 4 / 1024**2           # float32
        grads_mb    = weights_mb                        # same as weights
        optim_mb    = weights_mb * 4                    # Muon state (m) + AdamW (m,v) ≈ 4×
        batch_mb    = (train_cfg.batch_size * train_cfg.context_len * 4 * 2
                       * train_cfg.grad_accum_steps) / 1024**2
        total_est_mb = weights_mb + grads_mb + optim_mb + batch_mb

        print()
        info(f"Model weights  : ~{weights_mb:.0f} MB")
        info(f"Gradients      : ~{grads_mb:.0f} MB")
        info(f"Optimizer state: ~{optim_mb:.0f} MB")
        info(f"Activations    : ~{batch_mb:.0f} MB")
        info(f"TOTAL estimate : ~{total_est_mb:.0f} MB  ({total_est_mb/1024:.2f} GB)")
        print()

        budget_gb = 7.0
        if total_est_mb / 1024 < budget_gb * 0.9:
            ok(f"Fits within {budget_gb} GB budget  ✦")
        elif total_est_mb / 1024 < budget_gb:
            warn(f"Close to {budget_gb} GB budget — monitor with logger")
        else:
            err(f"Exceeds {budget_gb} GB budget! Reduce batch_size or context_len in config.py")

    except ImportError:
        warn("psutil not installed — cannot estimate memory")


def check_tokenizer():
    hdr("Tokenizer")
    from config import train_cfg
    from pathlib import Path
    tok_path = Path(train_cfg.tokenizer_path) / "tokenizer.json"
    if tok_path.exists():
        ok(f"Tokenizer found at {tok_path}")
    else:
        warn(f"Tokenizer NOT found at {tok_path}")
        info("Run:  python tokenizer_train.py   (takes ~5-10 min)")


def check_dirs():
    hdr("Directories")
    from config import train_cfg
    from pathlib import Path
    for d in [train_cfg.checkpoint_dir, train_cfg.log_dir]:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            ok(f"{d}/  (writable)")
        except Exception as e:
            err(f"{d}/  —  {e}")


def main():
    print(f"\n{CYAN}{BOLD}{'═'*50}")
    print("  SLM Setup Check")
    print(f"{'═'*50}{R}")

    check_python()
    check_torch()
    check_deps()
    check_memory()
    check_tokenizer()
    check_dirs()

    print(f"\n{CYAN}{'─'*50}{R}")
    print(f"  {BOLD}Next steps:{R}")
    print(f"  1.  pip install -r requirements.txt")
    print(f"  2.  python tokenizer_train.py      {DIM}(once){R}")
    print(f"  3.  python setup_check.py          {DIM}(verify){R}")
    print(f"  4.  python train.py                {DIM}(start training){R}")
    print(f"  5.  python inference.py            {DIM}(chat with your model){R}")
    print()


if __name__ == "__main__":
    main()
