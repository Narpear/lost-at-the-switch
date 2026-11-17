"""Instrumented speculative sampling (Leviathan et al. 2023 / Chen et al. 2023).

Batch size 1 only. Every drafted token is logged with whether it was accepted,
the drafter/target probabilities, and the total-variation distance between the
two (warped) next-token distributions at that position. 1 - TV is the
*expected* acceptance probability at that position, which is a much lower
variance signal than the 0/1 accept bit.

This loop exists for per-token logging, not for wall-clock speed. Use vLLM for
timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Protocol, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Sampling / warping
# --------------------------------------------------------------------------- #
@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0          # 0 = off
    top_p: float = 1.0      # 1.0 = off

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


def warp_to_probs(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """logits [..., V] -> probs [..., V] (float64) after temperature/top-k/top-p.

    The same warp is applied to drafter and target, so speculative sampling is
    lossless w.r.t. the *warped* target distribution (same convention as HF).
    """
    logits = logits.double()
    if cfg.greedy:
        idx = logits.argmax(-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, idx, 1.0)
    logits = logits / cfg.temperature
    V = logits.shape[-1]
    if cfg.top_k and cfg.top_k < V:
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = sorted_logits.softmax(-1)
        cum = sorted_probs.cumsum(-1)
        remove_sorted = (cum - sorted_probs) > cfg.top_p   # always keeps the top token
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits.softmax(-1)


def sample(probs: torch.Tensor, gen: torch.Generator) -> int:
    return int(torch.multinomial(probs, 1, generator=gen).item())


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p - q).abs().sum(-1)


# --------------------------------------------------------------------------- #
# Verification step (pure math, unit-tested separately)
# --------------------------------------------------------------------------- #
def verify(draft_tokens: Sequence[int], q: torch.Tensor, p: torch.Tensor,
           gen: torch.Generator) -> tuple[int, int, list[bool]]:
    """Speculative sampling acceptance.

    draft_tokens: gamma drafted ids
    q: [gamma, V] drafter probs at each drafted position
    p: [gamma+1, V] target probs (last row = bonus position)
    Returns (n_accepted, next_token, accepted_mask).
    """
    g = len(draft_tokens)
    accepted: list[bool] = []
    for i in range(g):
        x = draft_tokens[i]
        px, qx = p[i, x].item(), q[i, x].item()
        r = torch.rand((), generator=gen, dtype=torch.float64).item()
        if qx > 0 and r * qx < px:          # r < p/q without dividing
            accepted.append(True)
            continue
        accepted.append(False)
        residual = (p[i] - q[i]).clamp_min(0)
        s = residual.sum()
        residual = p[i] if s <= 0 else residual / s
        return i, sample(residual, gen), accepted
    return g, sample(p[g], gen), accepted


# --------------------------------------------------------------------------- #
# Model wrappers
# --------------------------------------------------------------------------- #
class CachedLM:
    """Wraps an HF causal LM with an incremental KV cache that can be rolled back.

    cache_mode="crop": DynamicCache.crop() on rejection (fast; valid for
        standard attention models).
    cache_mode="recompute": drop the cache on rejection and re-prefill
        (slow but always correct, e.g. for hybrid/linear-attention models whose
        recurrent state cannot be cropped). Run scripts/sanity_check.py to see
        which mode is valid for a given model.
    """

    def __init__(self, model, vocab_limit: int, cache_mode: str = "crop"):
        assert cache_mode in {"crop", "recompute"}
        self.model = model
        self.V = vocab_limit
        self.cache_mode = cache_mode
        self.device = next(model.parameters()).device
        self.reset()

    def reset(self):
        self.cache = None
        self.cached = 0
        self.n_forward = 0

    @torch.no_grad()
    def forward_new(self, seq: Sequence[int]) -> torch.Tensor:
        """Feed seq[self.cached:], return logits [n_new, V] (float32, on CPU)."""
        new = list(seq[self.cached:])
        assert new, "nothing new to feed"
        ids = torch.tensor([new], device=self.device)
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.cached = len(seq)
        self.n_forward += 1
        return out.logits[0, :, : self.V].float().cpu()

    def rollback(self, n: int):
        """Keep only the first n positions in the cache."""
        if self.cached <= n:
            return
        if self.cache_mode == "crop":
            # negative = "remove this many tokens"; works on transformers 4.x and 5.x
            self.cache.crop(n - self.cached)
            self.cached = n
        else:
            self.cache = None
            self.cached = 0


class Drafter(Protocol):
    def reset(self) -> None: ...
    def rollback(self, n: int) -> None: ...
    def draft(self, seq: list[int], gamma: int, cfg: SamplingConfig,
              gen: torch.Generator, stop_fn: Optional[Callable] = None
              ) -> tuple[list[int], torch.Tensor]: ...


StopFn = Callable[[torch.Tensor, int], bool]  # (q_row, step) -> stop drafting?


class NeuralDrafter:
    def __init__(self, lm: CachedLM):
        self.lm = lm

    def reset(self):
        self.lm.reset()

    def rollback(self, n: int):
        self.lm.rollback(n)

    def draft(self, seq, gamma, cfg, gen, stop_fn: Optional[StopFn] = None):
        toks, qs = [], []
        cur = list(seq)
        for step in range(gamma):
            q = warp_to_probs(self.lm.forward_new(cur)[-1], cfg)
            t = sample(q, gen)
            toks.append(t)
            qs.append(q)
            cur.append(t)
            if stop_fn is not None and stop_fn(q, step):   # Approach B hook
                break
        V = self.lm.V
        return toks, (torch.stack(qs) if qs else torch.empty(0, V, dtype=torch.float64))


def entropy_stop(threshold: float) -> StopFn:
    """Toy Approach-B policy: stop drafting when drafter entropy exceeds threshold (nats)."""
    def fn(q: torch.Tensor, step: int) -> bool:
        nz = q[q > 0]
        return float(-(nz * nz.log()).sum()) > threshold
    return fn


# --------------------------------------------------------------------------- #
# Generation loop
# --------------------------------------------------------------------------- #
@dataclass
class DraftRecord:
    step: int            # target call index
    out_pos: int         # index of this token in the generated continuation
    draft_idx: int       # 0..gamma-1 within the draft block
    token: int
    accepted: bool
    p_tok: float
    q_tok: float
    tv: float            # TV(p, q) at this position; 1 - tv = expected acceptance


@dataclass
class SpecResult:
    tokens: list[int]
    records: list[DraftRecord] = field(default_factory=list)
    n_target_calls: int = 0
    n_draft_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Accepted / verified drafts: estimates the per-token alpha of Eq. 1."""
        n = len(self.records)
        return sum(r.accepted for r in self.records) / n if n else float("nan")

    @property
    def draft_efficiency(self) -> float:
        """Accepted / all drafted tokens (incl. ones discarded after a rejection).
        Check which definition Paudel et al. report before comparing numbers."""
        n_acc = sum(r.accepted for r in self.records)
        return n_acc / self.n_draft_tokens if self.n_draft_tokens else float("nan")

    @property
    def tokens_per_target_call(self) -> float:
        return len(self.tokens) / self.n_target_calls if self.n_target_calls else float("nan")

    def to_dict(self):
        return {"tokens": self.tokens, "n_target_calls": self.n_target_calls,
                "records": [asdict(r) for r in self.records]}


def speculative_generate(target: CachedLM, drafter: Drafter, prompt_ids: Sequence[int],
                         max_new_tokens: int, gamma: int, cfg: SamplingConfig,
                         seed: int = 0, eos_ids: Sequence[int] = (),
                         stop_fn: Optional[StopFn] = None) -> SpecResult:
    gen = torch.Generator().manual_seed(seed)
    target.reset()
    drafter.reset()
    seq = list(prompt_ids)
    P = len(seq)
    res = SpecResult(tokens=[])
    eos = set(eos_ids)

    while len(seq) - P < max_new_tokens:
        remaining = max_new_tokens - (len(seq) - P)
        g = min(gamma, remaining - 1)          # bonus token always fits
        drafts, q = drafter.draft(seq, g, cfg, gen, stop_fn) if g > 0 else (
            [], torch.empty(0, target.V, dtype=torch.float64))
        g = len(drafts)
        logits = target.forward_new(seq + drafts)[-(g + 1):]
        p = warp_to_probs(logits, cfg)
        res.n_target_calls += 1
        res.n_draft_tokens += g

        n_acc, nxt, mask = verify(drafts, q, p, gen)
        out_base = len(seq) - P
        for i, acc in enumerate(mask):
            res.records.append(DraftRecord(
                step=res.n_target_calls - 1, out_pos=out_base + i, draft_idx=i,
                token=drafts[i], accepted=acc, p_tok=float(p[i, drafts[i]]),
                q_tok=float(q[i, drafts[i]]), tv=float(total_variation(p[i], q[i]))))
        new_tokens = drafts[:n_acc] + [nxt]
        seq.extend(new_tokens)
        target.rollback(len(seq) - 1)
        drafter.rollback(len(seq) - 1)
        if eos and any(t in eos for t in new_tokens):
            break

    out = seq[P:][:max_new_tokens]
    if eos:
        for i, t in enumerate(out):
            if t in eos:
                out = out[: i + 1]
                break
    res.tokens = out
    return res


def autoregressive_generate(target: CachedLM, prompt_ids: Sequence[int], max_new_tokens: int,
                            cfg: SamplingConfig, seed: int = 0,
                            eos_ids: Sequence[int] = ()) -> list[int]:
    """Plain target-only sampling with the same warp (reference for losslessness tests)."""
    gen = torch.Generator().manual_seed(seed)
    target.reset()
    seq = list(prompt_ids)
    out = []
    for _ in range(max_new_tokens):
        p = warp_to_probs(target.forward_new(seq)[-1], cfg)
        t = sample(p, gen)
        out.append(t)
        seq.append(t)
        if t in set(eos_ids):
            break
    return out
