"""
Inference module — used both during training (periodic checks) and standalone.

Supports:
  • Greedy / temperature sampling
  • Top-K filtering
  • Top-P (nucleus) filtering
  • Repetition penalty
  • Streaming output (character by character) for interactive use
"""

import sys
import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import ModelConfig, TrainConfig, model_cfg, train_cfg
from model import SLM
from checkpoint import load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    vals, _ = torch.topk(logits, k)
    min_val  = vals[..., -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_val, float("-inf"))


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs  = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens that push cumulative prob above threshold
    remove    = cumprobs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove] = float("-inf")
    # Scatter back
    return logits.scatter(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(
    model: SLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    stream: bool = False,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model            : SLM in eval mode
        tokenizer        : loaded Tokenizer
        prompt           : input text
        max_new_tokens   : tokens to generate
        temperature      : sampling temperature (1.0 = no scaling)
        top_k            : keep only top-k logits
        top_p            : nucleus sampling threshold
        repetition_penalty: penalise repeated tokens (1.0 = off)
        stream           : if True, print tokens as they are generated

    Returns: full generated string (prompt + continuation)
    """
    model.eval()
    ctx_len  = model.cfg.context_len
    eos_id   = tokenizer.token_to_id("<eos>") or 3

    enc      = tokenizer.encode(prompt)
    ids      = enc.ids
    in_tensor = torch.tensor([ids], dtype=torch.long)

    if stream:
        sys.stdout.write(prompt)
        sys.stdout.flush()

    generated_ids: List[int] = []

    for _ in range(max_new_tokens):
        # Truncate context if necessary
        context = in_tensor[:, -ctx_len:]
        logits, _ = model(context)
        next_logits = logits[0, -1, :].float()   # (V,)

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            for prev_id in set(generated_ids[-64:]):
                if next_logits[prev_id] > 0:
                    next_logits[prev_id] /= repetition_penalty
                else:
                    next_logits[prev_id] *= repetition_penalty

        # Temperature
        if temperature != 1.0:
            next_logits = next_logits / max(temperature, 1e-8)

        # Top-K
        next_logits = _top_k_logits(next_logits, top_k)

        # Top-P
        next_logits = _top_p_logits(next_logits, top_p)

        probs    = F.softmax(next_logits, dim=-1)
        next_id  = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_id)
        in_tensor = torch.cat([in_tensor, torch.tensor([[next_id]])], dim=1)

        if stream:
            token_str = tokenizer.decode([next_id])
            sys.stdout.write(token_str)
            sys.stdout.flush()

        if next_id == eos_id:
            break

    if stream:
        print()

    full_ids = enc.ids + generated_ids
    return tokenizer.decode(full_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Inline inference check (called during training every `eval_every` steps)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_check(
    model: SLM,
    tokenizer: Tokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 40,
) -> List[str]:
    """Run inference on all prompts and return generated strings."""
    model.eval()
    results = []
    for prompt in prompts:
        try:
            text = generate(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stream=False,
            )
            results.append(text)
        except Exception as e:
            results.append(f"[inference error: {e}]")
    model.train()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Standalone interactive CLI
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_for_inference(ckpt_path: Optional[str] = None) -> tuple:
    """Load model + tokenizer from checkpoint for standalone use."""
    from tokenizers import Tokenizer as HFTokenizer

    tok_path = Path(train_cfg.tokenizer_path) / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run tokenizer_train.py first.")

    tokenizer = HFTokenizer.from_file(str(tok_path))
    model     = SLM(model_cfg)

    from optimizer import build_optimizers
    muon, adamw = build_optimizers(model, train_cfg)

    step, tokens = load_checkpoint(model, muon, adamw, train_cfg, path=ckpt_path)
    if step == 0:
        print("[inference] No checkpoint found — using random weights (output will be gibberish).")
    else:
        print(f"[inference] Loaded checkpoint at step {step:,}  ({tokens/1e6:.1f}M tokens)")

    model.eval()
    return model, tokenizer, step


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SLM Inference")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to specific checkpoint directory. Default: latest.")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Single prompt (non-interactive mode).")
    parser.add_argument("--max-tokens", "-m", type=int, default=200,
                        help="Max new tokens to generate.")
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top-k", "-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rep-penalty", "-r", type=float, default=1.1)
    parser.add_argument("--no-stream", action="store_true",
                        help="Print full output at once instead of streaming.")
    args = parser.parse_args()

    model, tokenizer, step = _load_model_for_inference(args.checkpoint)

    os.system("")  # enable ANSI
    CYAN  = "\033[96m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"
    DIM   = "\033[2m"

    print(f"\n{CYAN}{BOLD}SLM Interactive Inference  (step {step:,}){RESET}")
    print(f"{DIM}temperature={args.temperature}  top_k={args.top_k}  top_p={args.top_p}{RESET}\n")

    if args.prompt:
        # Single-shot mode
        t0 = time.time()
        out = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            stream=not args.no_stream,
        )
        if args.no_stream:
            print(out)
        dt = time.time() - t0
        enc = tokenizer.encode(out)
        n_toks = len(enc.ids)
        print(f"\n{DIM}[{n_toks} tokens in {dt:.2f}s  ({n_toks/dt:.1f} tok/s)]{RESET}")
        return

    # Interactive REPL
    print(f"Type your prompt and press Enter. {DIM}Ctrl+C or 'quit' to exit.{RESET}\n")
    while True:
        try:
            prompt = input(f"{CYAN}>>> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        t0 = time.time()
        out = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            stream=not args.no_stream,
        )
        if args.no_stream:
            print(f"\n{out}")
        dt = time.time() - t0
        enc = tokenizer.encode(out)
        n_toks = len(enc.ids)
        print(f"\n{DIM}[{n_toks} tokens  {dt:.2f}s  {n_toks/dt:.1f} tok/s]{RESET}\n")


if __name__ == "__main__":
    import os
    main()
