"""Metrics: expected speedup, acceptance by position category, CIs, TV/KL."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import torch


def expected_speedup(alpha: float, gamma: int, c: float) -> float:
    """Leviathan et al. (2023): f = (1 - alpha^(gamma+1)) / ((1 - alpha)(gamma*c + 1))."""
    if alpha >= 1.0:
        return (gamma + 1) / (gamma * c + 1)
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))


def expected_tokens_per_call(alpha: float, gamma: int) -> float:
    return (gamma + 1) if alpha >= 1 else (1 - alpha ** (gamma + 1)) / (1 - alpha)


def tv(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p - q).abs().sum(-1)


def kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(p || q) along the last dim."""
    mask = p > 0
    return torch.where(mask, p * (p.clamp_min(eps).log() - q.clamp_min(eps).log()),
                       torch.zeros_like(p)).sum(-1)


def cluster_bootstrap_ci(values: Sequence[float], groups: Sequence, n_boot: int = 1000,
                         ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a bootstrap CI that resamples whole groups (e.g. sentences),
    since positions within a sentence are correlated."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    by_group = defaultdict(list)
    for v, g in zip(values, groups):
        by_group[g].append(v)
    sums = np.array([sum(v) for v in by_group.values()])
    cnts = np.array([len(v) for v in by_group.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(values.mean()), float(lo), float(hi)


def summarise_by_category(rows: Iterable[dict], value_key: str, cat_key: str = "category",
                          group_key: str = "sent_id", **kw) -> dict:
    """rows: dicts with value_key, cat_key, group_key. Returns per-category mean + CI."""
    buckets = defaultdict(lambda: ([], []))
    for r in rows:
        v, g = buckets[r[cat_key]]
        v.append(r[value_key])
        g.append(r[group_key])
    out = {}
    for cat, (v, g) in sorted(buckets.items()):
        m, lo, hi = cluster_bootstrap_ci(v, g, **kw)
        out[cat] = {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(v)}
    return out
