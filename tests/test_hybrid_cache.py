"""Hybrid (Qwen3.5-style) models: linear-attention state can't be cropped, so CachedLM uses
snapshot/restore. These tests guard the bug found on the real Qwen3.5 checkpoint."""
import pytest
import torch

from switchspec.specdec import (CachedLM, NeuralDrafter, SamplingConfig, autoregressive_generate,
                                speculative_generate)


def test_auto_resolves_to_snapshot(hybrid_pair):
    V, target, _ = hybrid_pair
    lm = CachedLM(target, V)
    lm.forward_new([1, 2, 3])
    assert lm.cache_mode == "snapshot"


def test_crop_is_refused_on_hybrid(hybrid_pair):
    V, target, _ = hybrid_pair
    lm = CachedLM(target, V, "crop")
    lm.forward_new([1, 2, 3, 4]); lm.forward_new([1, 2, 3, 4, 5, 6])
    with pytest.raises(RuntimeError):
        lm.rollback(5)


def test_snapshot_restore_matches_recompute(hybrid_pair):
    V, target, _ = hybrid_pair
    seq = list(range(1, 20))
    lm = CachedLM(target, V)
    lm.forward_new(seq[:6]); lm.forward_new(seq[:12]); lm.rollback(9)
    start = lm.cached
    assert start == 6
    a = lm.forward_new(seq)
    ref = CachedLM(target, V, "recompute").forward_new(seq)[start:]
    assert torch.allclose(a, ref, atol=1e-4)


@pytest.mark.parametrize("gamma", [1, 2, 4])
def test_hybrid_greedy_matches_target_with_partial_acceptance(hybrid_pair, gamma):
    V, target, drafter = hybrid_pair
    cfg = SamplingConfig(temperature=0)
    recs = []
    for prompt in ([1, 2, 3], [7, 30, 2, 11]):
        ref = autoregressive_generate(CachedLM(target, V, "recompute"), prompt, 30, cfg)
        res = speculative_generate(CachedLM(target, V), NeuralDrafter(CachedLM(drafter, V)),
                                   prompt, 30, gamma, cfg)
        assert res.tokens == ref
        recs += res.records
    n_acc = sum(r.accepted for r in recs)
    assert 0 < n_acc < len(recs), "test should exercise both accept and reject paths"


def test_text_vocab_size_handles_multimodal_config():
    """Qwen3_5Config (multimodal) has no top-level vocab_size; it lives in text_config."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
    from switchspec.models import text_vocab_size
    cfg = Qwen3_5Config(text_config={"vocab_size": 123})
    assert text_vocab_size(cfg) == 123
    assert text_vocab_size(Qwen3_5TextConfig(vocab_size=77)) == 77
