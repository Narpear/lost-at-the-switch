"""Run this first on every new machine / model pair.

Checks:
  1. drafter & target share a tokenizer (exact verification needs it)
  2. KV-cache crop gives the same logits as recomputing (decides cache_mode)
  3. T=0 speculative decoding == target greedy, token for token
  4. measures c = drafter forward time / target forward time (for Eq. 2)

Usage: python scripts/sanity_check.py --pair qwen [--device cuda]
"""
import argparse
import time

import torch

from _common import get_pair, sampling_from
from switchspec.specdec import CachedLM, autoregressive_generate, speculative_generate


def time_forward(lm: CachedLM, ids, n=20):
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    lm.reset(); lm.forward_new(ids)          # warmup / prefill
    ts = []
    for i in range(n):
        lm.rollback(len(ids) - 1)
        sync(); t0 = time.perf_counter()
        lm.forward_new(ids)
        sync(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--pair", default="qwen")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg, (tok, target, drafter, V) = get_pair(a.config, a.pair, a.device)
    print(f"[1] shared tokenizer OK, vocab limit = {V}")

    ids = tok("Kal main office ja rahi hoon but traffic is so bad yaar", add_special_tokens=True)["input_ids"]
    for name, lm in [("target", target), ("drafter", drafter.lm)]:
        mode = lm.cache_mode
        lm.cache_mode = "crop"
        lm.reset(); lm.forward_new(ids); lm.rollback(len(ids) // 2)
        try:
            a_ = lm.forward_new(ids + ids[:3])
            lm.reset()
            ref = lm.forward_new(ids + ids[:3])[len(ids) // 2:]
            diff = (a_ - ref).abs().max().item()
            ok = diff < 0.05 * ref.abs().max().item()
            print(f"[2] {name}: crop vs recompute max |dlogit| = {diff:.4f} -> {'OK' if ok else 'MISMATCH: use cache_mode: recompute'}")
        except Exception as e:  # noqa
            print(f"[2] {name}: crop failed ({e!r}) -> use cache_mode: recompute")
        lm.cache_mode = mode

    g = sampling_from(cfg, temperature=0)
    for gamma in cfg["gammas"]:
        ref = autoregressive_generate(target, ids, 48, g)
        res = speculative_generate(target, drafter, ids, 48, gamma, g)
        status = "OK" if ref == res.tokens else "MISMATCH"
        first_diff = next((i for i, (x, y) in enumerate(zip(ref, res.tokens)) if x != y), None)
        print(f"[3] gamma={gamma}: greedy match {status} (first diff at {first_diff}); "
              f"alpha={res.acceptance_rate:.3f}, TPT={res.tokens_per_target_call:.2f}")
    print("    (bf16 can cause rare near-tie argmax flips; if a mismatch appears, rerun in float32 to confirm)")

    tt, td = time_forward(target, ids), time_forward(drafter.lm, ids)
    print(f"[4] single-token forward: target {tt*1e3:.1f} ms, drafter {td*1e3:.1f} ms, c = {td/tt:.3f}")


if __name__ == "__main__":
    main()
