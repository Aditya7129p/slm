"""
4-million-parameter language model.

Architecture highlights (all bleeding-edge):
  • RMSNorm          — no mean-centering overhead, stable training
  • Rotary Position Embedding (RoPE) — relative positions, no learned PE
  • Grouped-Query Attention (GQA)    — 8 Q-heads, 2 KV-heads  → 4× KV savings
  • SwiGLU Feed-Forward              — gated activation, ~10 % better than GELU
  • Parallel Attention + FFN         — fuses residual paths (like PaLM)
  • No bias anywhere                 — faster matmuls, cleaner gradients
  • Weight tying: embed ↔ lm_head   — saves ~2 M params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Embedding
# ─────────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0, max_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (T, D)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int):
        if seq_len > self._cached_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    # q/k: (B, n_heads, T, head_dim)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ─────────────────────────────────────────────────────────────────────────────
# Grouped-Query Attention  (GQA)
# ─────────────────────────────────────────────────────────────────────────────
class GQA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep      = cfg.n_heads // cfg.n_kv_heads
        self.head_dim   = cfg.d_model // cfg.n_heads

        self.q_proj  = nn.Linear(cfg.d_model, cfg.n_heads    * self.head_dim, bias=cfg.bias)
        self.k_proj  = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj  = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.o_proj  = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model,    bias=cfg.bias)

        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, cos, sin)

        # Expand KV heads to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (uses flash-attn kernel when available)
        attn_dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=attn_dropout,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU FFN
# ─────────────────────────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    """
    FFN(x) = (xW_gate ⊙ silu(xW_up)) W_down
    Two parallel projections, no bias.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d_ff = cfg.d_ff
        self.gate = nn.Linear(cfg.d_model, d_ff, bias=cfg.bias)
        self.up   = nn.Linear(cfg.d_model, d_ff, bias=cfg.bias)
        self.down = nn.Linear(d_ff, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block  — Parallel Attention + FFN  (PaLM-style)
# ─────────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    """
    Parallel formulation:
        h = x + Attn(norm(x)) + FFN(norm(x))
    Single norm pass per block  →  faster & slightly better gradient flow.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQA(cfg)
        self.ffn  = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        # Parallel: both branches see the same pre-norm hidden state
        return x + self.attn(h, cos, sin, mask) + self.ffn(h)


# ─────────────────────────────────────────────────────────────────────────────
# SLM — the full model
# ─────────────────────────────────────────────────────────────────────────────
class SLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed    = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks   = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: saves ~2 M parameters
        self.lm_head.weight = self.embed.weight

        # Single shared RoPE (all layers use same freqs)
        self.rope = RotaryEmbedding(
            dim=cfg.d_model // cfg.n_heads,
            base=cfg.rope_base,
            max_len=cfg.context_len * 2,
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        std = 0.02
        for name, p in self.named_parameters():
            if p.dim() < 2:
                nn.init.ones_(p)   # norm scales → 1
            elif "embed" in name:
                nn.init.normal_(p, mean=0.0, std=std)
            elif "o_proj" in name or "down" in name:
                # Scale residual projections by 1/√(2*n_layers) (GPT-2 trick)
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * self.cfg.n_layers))
            else:
                nn.init.normal_(p, mean=0.0, std=std)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        idx: torch.Tensor,                # (B, T) int64
        targets: Optional[torch.Tensor] = None,   # (B, T) int64
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, T = idx.shape
        assert T <= self.cfg.context_len, f"Sequence length {T} > context_len {self.cfg.context_len}"

        x = self.embed(idx)               # (B, T, D)
        cos, sin = self.rope(T)           # (1, 1, T, head_dim)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm_out(x)
        logits = self.lm_head(x)          # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ── Parameter count ───────────────────────────────────────────────────────
    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (not only_trainable or p.requires_grad))

    # ── Grouped parameters for different LR ──────────────────────────────────
    def get_param_groups(self, base_lr: float, embed_lr_scale: float = 0.1):
        """
        Returns two groups:
          1. 'matrix' params  →  Muon  (all 2-D weight matrices except embed/head)
          2. 'other'  params  →  AdamW (embed, norms, biases, lm_head via tie)
        """
        matrix_params = []
        other_params  = []

        exclude_names = {"embed.weight", "lm_head.weight"}   # tied; treat as embed

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name in exclude_names:
                other_params.append(p)
            elif p.dim() == 2:
                matrix_params.append(p)
            else:
                other_params.append(p)

        return [
            {"params": matrix_params, "lr": base_lr,               "name": "matrix"},
            {"params": other_params,  "lr": base_lr * embed_lr_scale, "name": "other"},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Quick size check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import model_cfg
    m = SLM(model_cfg)
    total  = m.num_parameters(only_trainable=False)
    train  = m.num_parameters(only_trainable=True)
    print(f"Total params     : {total:>10,}")
    print(f"Trainable params : {train:>10,}")

    # Memory estimate (float32)
    mem_mb = (total * 4) / 1024**2
    print(f"Weights memory   : {mem_mb:.1f} MB  (float32)")

    # Forward pass
    dummy_in  = torch.randint(0, model_cfg.vocab_size, (2, model_cfg.context_len))
    dummy_tgt = torch.randint(0, model_cfg.vocab_size, (2, model_cfg.context_len))
    logits, loss = m(dummy_in, dummy_tgt)
    print(f"Logits shape     : {logits.shape}")
    print(f"Loss             : {loss.item():.4f}")
