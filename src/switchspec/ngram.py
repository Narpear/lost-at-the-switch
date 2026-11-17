"""Token-level n-gram drafter over the target tokenizer (baseline B2, Approach C).

Paudel et al. (2026, Sec. 6) store these as Python dicts. We do the same, with
backoff to shorter contexts and finally to unigrams for unseen prefixes.
Losslessness does not depend on the drafter, so any q is valid.
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from typing import Iterable, Optional, Sequence

import torch

from .specdec import SamplingConfig, sample


class NgramDrafter:
    def __init__(self, n: int, vocab_size: int):
        assert n >= 1
        self.n = n
        self.V = vocab_size
        # tables[k] maps a k-token context tuple -> Counter(next_token); k = 0..n-1
        self.tables: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(n)]
        self.n_forward = 0

    # ---- building ----------------------------------------------------------- #
    def fit(self, sequences: Iterable[Sequence[int]]) -> "NgramDrafter":
        for seq in sequences:
            seq = list(seq)
            for i, tok in enumerate(seq):
                for k in range(self.n):
                    if i - k < 0:
                        break
                    ctx = tuple(seq[i - k:i])
                    self.tables[k][ctx][tok] += 1
        return self

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"n": self.n, "V": self.V,
                         "tables": [dict(t) for t in self.tables]}, f)

    @classmethod
    def load(cls, path: str) -> "NgramDrafter":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["n"], d["V"])
        obj.tables = [defaultdict(Counter, t) for t in d["tables"]]
        return obj

    # ---- drafting ----------------------------------------------------------- #
    def dist(self, context: Sequence[int], cfg: SamplingConfig) -> torch.Tensor:
        counts: Optional[Counter] = None
        for k in range(self.n - 1, -1, -1):
            ctx = tuple(context[len(context) - k:]) if k else ()
            if k and len(context) < k:
                continue
            c = self.tables[k].get(ctx)
            if c:
                counts = c
                break
        q = torch.zeros(self.V, dtype=torch.float64)
        if not counts:
            q.fill_(1.0 / self.V)
            return q
        ids = torch.tensor(list(counts.keys()))
        vals = torch.tensor(list(counts.values()), dtype=torch.float64)
        if cfg.greedy:
            q[ids[vals.argmax()]] = 1.0
        else:
            q[ids] = vals / vals.sum()
        return q

    def reset(self):
        self.n_forward = 0

    def rollback(self, n: int):
        pass  # stateless

    def draft(self, seq, gamma, cfg, gen, stop_fn=None):
        cur = list(seq)
        toks, qs = [], []
        for step in range(gamma):
            q = self.dist(cur, cfg)
            t = sample(q, gen)
            toks.append(t)
            qs.append(q)
            cur.append(t)
            if stop_fn is not None and stop_fn(q, step):
                break
        return toks, (torch.stack(qs) if qs else torch.empty(0, self.V, dtype=torch.float64))
