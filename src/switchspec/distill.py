"""Approach A: switch-weighted distillation loss.

w_t = 1 + lambda * s_t + mu * 1[t in W_k]                       (Eq. 3)
L   = sum_t w_t * KL(p_t || q_t) / sum_t w_t   +   beta * mean_{t: s_t=1} TV(p_t, q_t)

p = target (teacher), q = drafter (student). Setting lambda = mu = beta = 0
recovers plain (unweighted) distillation, i.e. the Paudel et al. B3 variant.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def switch_weights(switch_mask: torch.Tensor, window_mask: torch.Tensor,
                   lam: float, mu: float) -> torch.Tensor:
    return 1.0 + lam * switch_mask.float() + mu * window_mask.float()


def weighted_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                          switch_mask: torch.Tensor, window_mask: torch.Tensor,
                          valid_mask: torch.Tensor, lam: float = 0.0, mu: float = 0.0,
                          beta_tv: float = 0.0, temperature: float = 1.0) -> dict:
    """All tensors shaped [B, T, V] (logits) or [B, T] (masks); logits already aligned
    so position t predicts token t+1. Returns dict with 'loss' and components."""
    V = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s_logp = F.log_softmax(student_logits[..., :V].float() / temperature, -1)
    t_logp = F.log_softmax(teacher_logits[..., :V].float() / temperature, -1)
    t_p = t_logp.exp()
    kl_t = (t_p * (t_logp - s_logp)).sum(-1)                 # [B, T]
    w = switch_weights(switch_mask, window_mask, lam, mu) * valid_mask.float()
    kl_loss = (w * kl_t).sum() / w.sum().clamp_min(1e-8)
    out = {"kl": kl_loss}
    loss = kl_loss
    if beta_tv > 0:
        sw = switch_mask.bool() & valid_mask.bool()
        if sw.any():
            tv_t = 0.5 * (t_p - s_logp.exp()).abs().sum(-1)
            tv_loss = tv_t[sw].mean()
            loss = loss + beta_tv * tv_loss
            out["tv_switch"] = tv_loss
    out["loss"] = loss
    return out
