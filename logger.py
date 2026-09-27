"""
Beautiful terminal logger for SLM training.

Features:
  • ANSI-coloured status bar with live metrics
  • Progress bar for grad-accumulation steps
  • Loss history sparkline
  • Memory gauge (RAM used / available)
  • JSONL log writer  →  logs/log-N.jsonl
  • No external dependencies beyond standard lib + psutil
"""

import os
import sys
import json
import time
import math
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Optional, List

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────
# Windows 10+ supports ANSI via virtual terminal processing
os.system("")   # enable ANSI on Windows

class C:  # Colours
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Foreground
    BLACK   = "\033[30m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    # Background
    BG_BLUE  = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_RED   = "\033[41m"
    BG_DARK  = "\033[40m"

def _c(text, *codes):
    return "".join(codes) + str(text) + C.RESET


# ─────────────────────────────────────────────────────────────────────────────
# Sparkline (Unicode block chars)
# ─────────────────────────────────────────────────────────────────────────────
_SPARK = " ▁▂▃▄▅▆▇█"

def _sparkline(values: List[float], width: int = 20) -> str:
    if not values:
        return " " * width
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    r = hi - lo if hi > lo else 1
    chars = [_SPARK[int((v - lo) / r * (len(_SPARK) - 1))] for v in vals]
    # Pad left if shorter than width
    return " " * (width - len(chars)) + "".join(chars)


# ─────────────────────────────────────────────────────────────────────────────
# Memory bar
# ─────────────────────────────────────────────────────────────────────────────
def _mem_bar(width: int = 20) -> str:
    if not _HAS_PSUTIL:
        return "psutil N/A"
    mem = psutil.virtual_memory()
    used_gb  = mem.used  / 1024**3
    total_gb = mem.total / 1024**3
    pct      = mem.percent / 100.0
    filled   = int(pct * width)
    bar      = "█" * filled + "░" * (width - filled)
    colour   = C.RED if pct > 0.85 else (C.YELLOW if pct > 0.65 else C.GREEN)
    return f"{colour}{bar}{C.RESET} {used_gb:.1f}/{total_gb:.1f}GB ({mem.percent:.0f}%)"


# ─────────────────────────────────────────────────────────────────────────────
# Gradient-accumulation mini-bar
# ─────────────────────────────────────────────────────────────────────────────
def _accum_bar(micro_step: int, total_steps: int, width: int = 12) -> str:
    filled = int((micro_step / total_steps) * width)
    bar    = "▓" * filled + "░" * (width - filled)
    return f"{C.CYAN}{bar}{C.RESET} {micro_step}/{total_steps}"


# ─────────────────────────────────────────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def _fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def _eta(elapsed: float, step: int, total_steps: int) -> str:
    if step == 0:
        return "?"
    per_step = elapsed / step
    remaining = (total_steps - step) * per_step
    return _fmt_time(remaining)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL log file management
# ─────────────────────────────────────────────────────────────────────────────
def _get_log_path(log_dir: str) -> Path:
    """Find the next available log-N.jsonl filename."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (log_dir / f"log-{idx}.jsonl").exists():
        idx += 1
    return log_dir / f"log-{idx}.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Main Logger
# ─────────────────────────────────────────────────────────────────────────────
class TrainingLogger:
    """
    Single-instance logger for training.

    Usage:
        logger = TrainingLogger(log_dir="logs", total_steps=50000, grad_accum=8)
        logger.start()
        for step in range(total_steps):
            logger.log_micro(micro_step=...)
            logger.log_step(step=step, loss=..., lr=..., grad_norm=..., tokens=...)
        logger.close()
    """

    HEADER_INTERVAL = 25   # reprint column headers every N steps

    def __init__(
        self,
        log_dir: str,
        total_steps: int,
        grad_accum_steps: int,
        resume_step: int = 0,
    ):
        self.log_dir         = log_dir
        self.total_steps     = total_steps
        self.grad_accum      = grad_accum_steps
        self.step            = resume_step
        self.start_time      = time.time()
        self.last_log_time   = self.start_time
        self._loss_history: List[float] = []
        self._log_path       = _get_log_path(log_dir)
        self._log_file       = open(self._log_path, "a", buffering=1)
        self._header_printed = 0

    # ── Banner ────────────────────────────────────────────────────────────────
    def start(self, n_params: int = 0, vocab_size: int = 0):
        term_w = shutil.get_terminal_size((100, 24)).columns
        line   = "═" * term_w

        print()
        print(_c(line, C.CYAN, C.BOLD))
        title = "  ✦  SLM Pre-Training  ✦"
        print(_c(title.center(term_w), C.CYAN, C.BOLD))
        print(_c(line, C.CYAN, C.BOLD))
        print()
        if n_params:
            print(f"  {_c('Parameters  :', C.DIM)}  {_c(f'{n_params:,}', C.YELLOW, C.BOLD)}")
        if vocab_size:
            print(f"  {_c('Vocab size  :', C.DIM)}  {_c(f'{vocab_size:,}', C.YELLOW, C.BOLD)}")
        print(f"  {_c('Max steps   :', C.DIM)}  {_c(f'{self.total_steps:,}', C.YELLOW, C.BOLD)}")
        print(f"  {_c('Grad accum  :', C.DIM)}  {_c(str(self.grad_accum), C.YELLOW, C.BOLD)}")
        print(f"  {_c('Log file    :', C.DIM)}  {_c(str(self._log_path), C.CYAN)}")
        print(f"  {_c('Memory      :', C.DIM)}  {_mem_bar()}")
        if self.step > 0:
            print(f"\n  {_c('↩ Resuming from step', C.GREEN, C.BOLD)}  {_c(str(self.step), C.GREEN, C.BOLD)}")
        print()
        print(_c("─" * term_w, C.DIM))
        sys.stdout.flush()

    # ── Column header ─────────────────────────────────────────────────────────
    def _print_header(self):
        hdr = (
            f"  {_c('STEP', C.DIM, C.BOLD):<18}"
            f"  {_c('LOSS', C.DIM, C.BOLD):<14}"
            f"  {_c('LR', C.DIM, C.BOLD):<14}"
            f"  {_c('GRAD NORM', C.DIM, C.BOLD):<14}"
            f"  {_c('TOKENS', C.DIM, C.BOLD):<14}"
            f"  {_c('ELAPSED', C.DIM, C.BOLD):<12}"
            f"  {_c('ETA', C.DIM, C.BOLD):<12}"
            f"  {_c('TREND', C.DIM, C.BOLD)}"
        )
        print(hdr)
        print(_c("  " + "─" * (shutil.get_terminal_size((120,24)).columns - 2), C.DIM))

    # ── Micro-step progress  (in-place overwrite) ─────────────────────────────
    def log_micro(self, micro_step: int, micro_loss: Optional[float] = None):
        bar  = _accum_bar(micro_step, self.grad_accum)
        loss = f"  {_c(f'loss≈{micro_loss:.4f}', C.DIM)}" if micro_loss is not None else ""
        line = f"\r  {_c('Accumulating', C.DIM)} {bar}{loss}  "
        sys.stdout.write(line)
        sys.stdout.flush()

    # ── Full optimizer step ───────────────────────────────────────────────────
    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_total: int,
        is_checkpoint: bool = False,
        is_inference: bool = False,
    ):
        self.step = step
        self._loss_history.append(loss)
        now     = time.time()
        elapsed = now - self.start_time

        # Clear micro-step line
        sys.stdout.write("\r" + " " * shutil.get_terminal_size((120,24)).columns + "\r")

        # Column header every N steps
        if (step - (step % self.HEADER_INTERVAL)) > self._header_printed or step == 1:
            self._print_header()
            self._header_printed = step

        # Loss colour
        if len(self._loss_history) >= 5:
            trend = self._loss_history[-1] - sum(self._loss_history[-5:-1]) / 4
            loss_col = C.GREEN if trend < 0 else (C.YELLOW if trend < 0.01 else C.RED)
        else:
            loss_col = C.WHITE

        spark = _sparkline(self._loss_history, width=16)

        # Step counter  (colour = red if saving, cyan if inference check)
        step_tag = ""
        if is_checkpoint:
            step_tag = _c(" 💾", C.YELLOW)
        if is_inference:
            step_tag += _c(" 🔍", C.CYAN)

        pct = step / max(1, self.total_steps) * 100

        row = (
            f"  {_c(f'[{step:>6}/{self.total_steps}]', C.BOLD)}{step_tag}  "
            f"  {_c(f'{loss:.5f}', loss_col, C.BOLD):<22}"
            f"  {_c(f'{lr:.2e}', C.BLUE):<22}"
            f"  {_c(f'{grad_norm:.4f}', C.MAGENTA):<22}"
            f"  {_c(_fmt_tokens(tokens_total), C.YELLOW):<22}"
            f"  {_c(_fmt_time(elapsed), C.DIM):<20}"
            f"  {_c(_eta(elapsed, step, self.total_steps), C.DIM):<20}"
            f"  {_c(spark, C.CYAN)}"
        )
        print(row)
        sys.stdout.flush()

        # Memory warning every 10 steps
        if step % 10 == 0 and _HAS_PSUTIL:
            mem = psutil.virtual_memory()
            if mem.percent > 80:
                print(f"  {_c('⚠ HIGH MEMORY', C.RED, C.BOLD)}  {_mem_bar()}")

        # Write JSONL
        record = {
            "step":          step,
            "loss":          round(loss, 6),
            "lr":            round(lr, 8),
            "grad_norm":     round(grad_norm, 6),
            "tokens_total":  tokens_total,
            "elapsed_s":     round(elapsed, 1),
            "is_checkpoint": is_checkpoint,
            "is_inference":  is_inference,
        }
        self._log_file.write(json.dumps(record) + "\n")

    # ── Inference result display ──────────────────────────────────────────────
    def log_inference(self, step: int, prompt: str, generated: str):
        term_w = shutil.get_terminal_size((100, 24)).columns
        print()
        print(_c("  ┌" + "─" * (term_w - 4) + "┐", C.CYAN, C.DIM))
        print(_c(f"  │  🔍  Inference Check @ step {step}", C.CYAN, C.BOLD))
        print(_c("  │", C.CYAN, C.DIM))
        prompt_lines  = prompt.split("\n")
        gen_lines     = generated.split("\n")
        print(_c(f"  │  {_c('PROMPT:', C.DIM, C.BOLD)}  {prompt_lines[0]}", C.CYAN))
        print(_c(f"  │  {_c('MODEL :', C.DIM, C.BOLD)}  {gen_lines[0][:term_w - 18]}", C.CYAN))
        if len(gen_lines) > 1:
            for ln in gen_lines[1:3]:
                print(_c(f"  │           {ln[:term_w - 13]}", C.CYAN, C.DIM))
        print(_c("  └" + "─" * (term_w - 4) + "┘", C.CYAN, C.DIM))
        print()
        sys.stdout.flush()

        record = {
            "step":      step,
            "type":      "inference",
            "prompt":    prompt,
            "generated": generated,
        }
        self._log_file.write(json.dumps(record) + "\n")

    # ── Checkpoint notice ─────────────────────────────────────────────────────
    def log_checkpoint(self, step: int, path: str):
        print(f"\n  {_c('💾 Checkpoint saved', C.YELLOW, C.BOLD)} → {_c(path, C.CYAN)}  (step {step})\n")
        sys.stdout.flush()

    # ── Interrupt notice ──────────────────────────────────────────────────────
    def log_interrupt(self, step: int):
        term_w = shutil.get_terminal_size((100, 24)).columns
        print()
        print(_c("  ⚡  Ctrl+C detected — saving emergency checkpoint …", C.YELLOW, C.BOLD))
        sys.stdout.flush()

    # ── Training complete ─────────────────────────────────────────────────────
    def log_done(self, step: int, tokens_total: int):
        elapsed = time.time() - self.start_time
        term_w  = shutil.get_terminal_size((100, 24)).columns
        print()
        print(_c("═" * term_w, C.GREEN, C.BOLD))
        print(_c("  ✅  Training complete!".center(term_w), C.GREEN, C.BOLD))
        print(_c("═" * term_w, C.GREEN, C.BOLD))
        print(f"  Steps trained : {step:,}")
        print(f"  Tokens seen   : {_fmt_tokens(tokens_total)}")
        print(f"  Total time    : {_fmt_time(elapsed)}")
        if self._loss_history:
            print(f"  Final loss    : {self._loss_history[-1]:.5f}")
        print()
        sys.stdout.flush()

    # ── Close ─────────────────────────────────────────────────────────────────
    def close(self):
        try:
            self._log_file.close()
        except Exception:
            pass
