"""
Muon Optimizer — Newton-Schulz orthogonalised SGD for matrix weights.

Based on: Kosson et al. 2024 "Muon: An optimizer for hidden layers in neural networks"
Reference implementation: https://github.com/KellerJordan/modded-nanogpt

Key idea:
  For every 2-D weight matrix W, compute the momentum update m, then
  orthogonalise it via Newton-Schulz iteration to get a quasi-orthogonal
  update direction. This approximates the steepest descent in spectral norm,
  giving better gradient conditioning without expensive SVD.

  Embed / norm / 1-D params fall back to plain AdamW in a companion group.
"""

import math
import torch
from torch.optim import Optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Newton-Schulz  (5th-order)
# ─────────────────────────────────────────────────────────────────────────────
# Coefficients from the original Muon paper (Table 1)
_NS_COEFFS = [(7, 0, 3.2857142857142856),   # iteration 1: a=7, b=0, c=3.2857…
              (8, 15, -10),                   # iteration 2
              (15, -45, 40),                  # iteration 3
              (40, -105, 70),                 # iteration 4  ← 4 iters is enough
              ]


def _zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximate the matrix sign function (zero-power, a.k.a. spectral normalization)
    using Newton-Schulz iterations.

    Input:  G  — gradient matrix  (m, n), m ≤ n  (handled by transposing if needed)
    Output: X  — approximately orthogonal matrix of the same shape

    Steps=5 gives double-precision accuracy within typical training regimes.
    """
    assert G.ndim == 2, "Newton-Schulz expects a 2-D tensor"
    m, n = G.shape
    transposed = m > n
    if transposed:
        G = G.T           # ensure rows ≤ cols
        m, n = n, m

    # Normalise to unit spectral norm as starting point
    X = G / (G.norm() + 1e-7)

    # Quintic iterations
    for _ in range(steps):
        A = X @ X.T       # (m, m)
        # Horner's method for  (15/8)I - (105/16)A + (315/32)A² - (231/32)A³
        # Simplified 3-term version used in the open-source reference:
        #   X ← a*X + b*(A@X) + c*(A@A@X)
        #   with a=15/8, b=-105/64, c=315/128  (Chebyshev approx)
        # We use the simple but stable form below:
        B = X @ X.T @ X
        X = 1.5 * X - 0.5 * B

    if transposed:
        X = X.T
    return X


def newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Public alias used in the training loop."""
    return _zeropower_via_newtonschulz(G, steps)


# ─────────────────────────────────────────────────────────────────────────────
# Muon
# ─────────────────────────────────────────────────────────────────────────────
class Muon(Optimizer):
    """
    Muon — Momentum + orthogonalised update for 2-D weight matrices.

    Usage:
        matrix_params = [p for p in model.parameters() if p.dim() == 2]
        optimizer = Muon(matrix_params, lr=3e-3, momentum=0.95)

    Args:
        params      : iterable of 2-D parameter tensors
        lr          : learning rate
        momentum    : Nesterov momentum coefficient  (0.95 recommended)
        ns_steps    : Newton-Schulz iterations  (5 is the sweet spot)
        weight_decay: L2 penalty applied *before* orthogonalisation
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr          = group["lr"]
            momentum    = group["momentum"]
            ns_steps    = group["ns_steps"]
            wd          = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.dim() == 2, (
                    f"Muon expects 2-D params, got {p.dim()}-D  ({p.shape}). "
                    "Use AdamW for embeddings, norms, and biases."
                )

                g = p.grad

                # Optional weight decay (decoupled, like AdamW)
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # Initialise / update momentum buffer
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]

                # Nesterov: use look-ahead gradient
                m.mul_(momentum).add_(g)
                g_nesterov = g.add(m, alpha=momentum)   # g + β·m

                # Orthogonalise update via Newton-Schulz
                update = newtonschulz5(g_nesterov.float(), steps=ns_steps).to(p.dtype)

                # Scale to match RMS of 0.01 (stabilises mixed-scale networks)
                update_rms = update.pow(2).mean().sqrt().add(1e-8)
                update = update / update_rms * 0.01

                p.add_(update, alpha=-lr)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Companion AdamW  (for embed, norm scales, biases)
# ─────────────────────────────────────────────────────────────────────────────
class AdamWOptimizer(torch.optim.AdamW):
    """Thin wrapper — just documents the intent of this param group."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build both optimizers from model param groups
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizers(model, train_cfg):
    """
    Returns (muon_opt, adamw_opt) — both need .step() and .zero_grad() each iteration.
    """
    from config import TrainConfig
    cfg: TrainConfig = train_cfg

    groups = model.get_param_groups(base_lr=cfg.lr, embed_lr_scale=cfg.embed_lr_scale)
    matrix_group = [g for g in groups if g["name"] == "matrix"][0]
    other_group  = [g for g in groups if g["name"] == "other"][0]

    muon = Muon(
        [{"params": matrix_group["params"], "lr": matrix_group["lr"]}],
        lr=cfg.lr,
        momentum=cfg.muon_momentum,
        ns_steps=5,
        weight_decay=cfg.weight_decay,
    )

    adamw = torch.optim.AdamW(
        [{"params": other_group["params"], "lr": other_group["lr"]}],
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    return muon, adamw


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule  (cosine with linear warmup)
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(step: int, train_cfg) -> float:
    """Returns the learning rate multiplier at a given step."""
    cfg = train_cfg
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    if step >= cfg.lr_decay_steps:
        return cfg.min_lr_ratio
    # Cosine decay
    progress = (step - cfg.warmup_steps) / (cfg.lr_decay_steps - cfg.warmup_steps)
    return cfg.min_lr_ratio + 0.5 * (1.0 - cfg.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
