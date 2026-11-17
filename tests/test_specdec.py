"""Correctness of the speculative decoding loop.

These are the tests the proposal's 'Controls' paragraph asks for:
  * T = 0 must exactly match greedy decoding of the target.
  * T > 0 must sample from the same distribution as the target.
"""
import itertools

import pytest
import torch

from switchspec.specdec import (CachedLM, NeuralDrafter, SamplingConfig, autoregressive_generate,
                                entropy_stop, speculative_generate, verify, warp_to_probs)
from switchspec.ngram import NgramDrafter
from conftest import wrap


# ---- 1. the accept/reject rule alone reproduces p ------------------------- #
@pytest.mark.parametrize("seed", [0, 1])
def test_verify_single_step_is_lossless(seed):
    torch.manual_seed(seed)
    V, N = 6, 40000
    p = torch.rand(V, dtype=torch.float64) ** 3; p /= p.sum()
    q = torch.rand(V, dtype=torch.float64) ** 3; q /= q.sum()
    gen = torch.Generator().manual_seed(seed)
    counts = torch.zeros(V)
    for _ in range(N):
        x = int(torch.multinomial(q, 1, generator=gen))
        n_acc, nxt, _ = verify([x], q[None], torch.stack([p, p]), gen)
        counts[x if n_acc == 1 else nxt] += 1
    emp = counts / N
    assert (emp - p).abs().max() < 0.012, (emp, p)


def test_verify_greedy_one_hot():
    gen = torch.Generator().manual_seed(0)
    p = torch.eye(4, dtype=torch.float64)[[2, 1]]
    q = torch.eye(4, dtype=torch.float64)[[2]]
    assert verify([2], q, p, gen)[:2] == (1, 1)            # agree -> accept, bonus = argmax
    q = torch.eye(4, dtype=torch.float64)[[3]]
    assert verify([3], q, p, gen)[:2] == (0, 2)            # disagree -> target argmax


def test_warp_topk_topp():
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    p = warp_to_probs(logits, SamplingConfig(1.0, top_k=2))
    assert (p[2:] == 0).all() and abs(p.sum() - 1) < 1e-9
    p = warp_to_probs(logits, SamplingConfig(1.0, top_p=0.5))
    assert p[0] == 1.0                                       # top token alone has mass > 0.5
    p = warp_to_probs(logits, SamplingConfig(0.0))
    assert p.tolist() == [1, 0, 0, 0]


# ---- 2. KV-cache rollback is correct -------------------------------------- #
def test_cache_crop_matches_recompute(toy_pair):
    V, target, _ = toy_pair
    lm = CachedLM(target, V, "crop")
    seq = list(range(1, 20))
    lm.forward_new(seq[:12])
    lm.rollback(7)
    a = lm.forward_new(seq)               # feeds seq[7:]
    ref = CachedLM(target, V, "crop").forward_new(seq)[7:]
    assert torch.allclose(a, ref, atol=1e-4)


# ---- 3. T = 0 exactly matches target greedy ------------------------------- #
@pytest.mark.parametrize("gamma", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("mode", ["crop", "recompute"])
def test_greedy_matches_target(toy_pair, gamma, mode):
    V, target, drafter = toy_pair
    t, d = wrap(target, drafter, V, mode)
    cfg = SamplingConfig(temperature=0)
    for prompt in ([1, 5, 9], [3], [7, 7, 2, 30, 11]):
        ref = autoregressive_generate(t, prompt, 24, cfg)
        res = speculative_generate(t, d, prompt, 24, gamma, cfg)
        assert res.tokens == ref
        # sanity: also matches HF's own greedy generate
        hf = target.generate(torch.tensor([prompt]), max_new_tokens=24, do_sample=False,
                             pad_token_id=0, eos_token_id=None)[0, len(prompt):].tolist()
        assert res.tokens == hf


def test_self_drafting_accepts_everything(toy_pair):
    V, target, _ = toy_pair
    t = CachedLM(target, V)
    d = NeuralDrafter(CachedLM(target, V))
    res = speculative_generate(t, d, [1, 2], 20, 4, SamplingConfig(temperature=1.0), seed=3)
    assert res.acceptance_rate == 1.0
    assert max(r.tv for r in res.records) < 1e-3   # float noise: chunked vs incremental


# ---- 4. T > 0 matches the target's sequence distribution ------------------ #
def exact_joint(target, V, prompt, L, cfg):
    """Exact distribution over all V^L continuations under the warped target."""
    probs = {}
    for cont in itertools.product(range(V), repeat=L):
        ids = torch.tensor([list(prompt) + list(cont)])
        with torch.no_grad():
            logits = target(ids).logits[0, len(prompt) - 1: len(prompt) + L - 1, :V]
        p = warp_to_probs(logits, cfg)
        probs[cont] = float(torch.prod(p[torch.arange(L), torch.tensor(cont)]))
    return probs


def tv_dict(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


@pytest.mark.parametrize("drafter_kind", ["neural", "ngram"])
def test_sampling_is_lossless(small_vocab_pair, drafter_kind):
    V, target, drafter = small_vocab_pair
    cfg = SamplingConfig(temperature=1.0, top_k=4)
    prompt, L, N = [1, 2], 3, 3000
    exact = exact_joint(target, V, prompt, L, cfg)
    t = CachedLM(target, V)
    if drafter_kind == "neural":
        d = NeuralDrafter(CachedLM(drafter, V))
    else:
        d = NgramDrafter(2, V).fit([[1, 2, 3, 4, 5, 0, 1, 2, 2, 3]])
    spec, ref = {}, {}
    for s in range(N):
        k = tuple(speculative_generate(t, d, prompt, L, gamma=2, cfg=cfg, seed=s).tokens)
        spec[k] = spec.get(k, 0) + 1 / N
        k = tuple(autoregressive_generate(t, prompt, L, cfg, seed=10_000 + s))
        ref[k] = ref.get(k, 0) + 1 / N
    d_spec, d_ref = tv_dict(spec, exact), tv_dict(ref, exact)
    # spec decoding should be no further from the truth than plain sampling (+ noise)
    assert d_spec < d_ref + 0.03, (d_spec, d_ref)
    assert d_spec < 0.08


# ---- 5. bookkeeping ------------------------------------------------------- #
def test_records_and_counters(toy_pair):
    V, target, drafter = toy_pair
    t, d = wrap(target, drafter, V)
    res = speculative_generate(t, d, [4, 4], 30, 3, SamplingConfig(1.0), seed=1)
    assert len(res.tokens) == 30
    # drafts after the first rejection are never verified, so they have no record
    assert len(res.records) <= res.n_draft_tokens
    n_acc = sum(r.accepted for r in res.records)
    assert res.draft_efficiency <= res.acceptance_rate
    assert len(res.tokens) == n_acc + res.n_target_calls  # each call adds 1 non-draft token
    assert res.tokens_per_target_call >= 1


def test_eos_stops(toy_pair):
    V, target, drafter = toy_pair
    t, d = wrap(target, drafter, V)
    ref = autoregressive_generate(t, [1], 30, SamplingConfig(0))
    eos = ref[5]
    res = speculative_generate(t, d, [1], 30, 3, SamplingConfig(0), eos_ids=[eos])
    assert res.tokens[-1] == eos and eos not in res.tokens[:-1]


def test_adaptive_stop_reduces_draft_length(toy_pair):
    V, target, drafter = toy_pair
    t, d = wrap(target, drafter, V)
    cfg = SamplingConfig(1.0)
    full = speculative_generate(t, d, [1, 2], 30, 4, cfg, seed=0)
    short = speculative_generate(t, d, [1, 2], 30, 4, cfg, seed=0, stop_fn=entropy_stop(0.0))
    assert max(r.draft_idx for r in short.records) == 0
    assert short.n_draft_tokens < full.n_draft_tokens
