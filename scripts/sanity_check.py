"""Run this first on every new machine / model pair.

Checks:
  1. drafter & target share a tokenizer (exact verification needs it)
  2. cache rollback (crop or snapshot, chosen automatically) gives the same logits as recomputing
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
    """Median time of one decode step (one new token on a warm cache).
    The sequence grows by one token per step, so no rollback is involved: a rollback
    with no saved snapshot drops a hybrid model's whole cache and re-prefills, which
    is what the first version of this function accidentally timed."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    lm.reset()
    seq = list(ids)
    lm.forward_new(seq)                       # prefill
    for _ in range(3):                        # warm up the decode path
        seq.append(seq[-1]); lm.forward_new(seq)
    ts = []
    for _ in range(n):
        seq.append(seq[-1])
        before = lm.cached
        sync(); t0 = time.perf_counter()
        lm.forward_new(seq)
        sync(); ts.append(time.perf_counter() - t0)
        assert lm.cached - before == 1, "timing must feed exactly one token"
    return sorted(ts)[n // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--pair", default="qwen")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--timing-only", action="store_true", help="only measure c (step [4])")
    a = ap.parse_args()
    cfg, (tok, target, drafter, V) = get_pair(a.config, a.pair, a.device)
    if a.timing_only:
        ids = tok("Kal main office ja rahi hoon but traffic is so bad yaar, kya karun", add_special_tokens=True)["input_ids"]
        tt, td = time_forward(target, ids), time_forward(drafter.lm, ids)
        print(f"[4] single-token decode step: target {tt*1e3:.1f} ms, drafter {td*1e3:.1f} ms, c = {td/tt:.3f}")
        return
    print(f"[1] shared tokenizer OK, vocab limit = {V}")

    ids = tok("Kal main office ja rahi hoon but traffic is so bad yaar, kya karun", add_special_tokens=True)["input_ids"]
    k = len(ids) // 2
    for name, lm in [("target", target), ("drafter", drafter.lm)]:
        try:
            test = CachedLM(lm.model, V, lm.requested_mode)
            test.forward_new(ids[:k]); test.forward_new(ids[:k + 4]); test.rollback(k + 2)
            start = test.cached
            a = test.forward_new(ids)
            ref = CachedLM(lm.model, V, "recompute").forward_new(ids)[start:]
            diff = (a - ref).abs().max().item()
            same_argmax = (a.argmax(-1) == ref.argmax(-1)).float().mean().item()
            ok = diff < 0.05 * ref.abs().max().item()
            print(f"[2] {name}: cache_mode={test.cache_mode}, rollback vs recompute max |dlogit| = {diff:.4f}, "
                  f"argmax agreement = {same_argmax:.0%} -> {'OK' if ok else 'MISMATCH: set cache_mode: recompute'}")
        except Exception as e:  # noqa
            print(f"[2] {name}: rollback failed ({e!r}) -> set cache_mode: recompute and tell Claude")

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
    print(f"[4] single-token decode step: target {tt*1e3:.1f} ms, drafter {td*1e3:.1f} ms, c = {td/tt:.3f}")


if __name__ == "__main__":
    main()
