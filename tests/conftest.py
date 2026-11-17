"""Tiny randomly initialised models so the whole suite runs on CPU with no downloads."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from switchspec.specdec import CachedLM, NeuralDrafter


def tiny_llama(vocab: int, hidden: int, layers: int, seed: int, sharp: float = 1.0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():  # make distributions peakier than the near-uniform init
        m.lm_head.weight.mul_(sharp)
    return m


@pytest.fixture(scope="session")
def toy_pair():
    V = 32
    target = tiny_llama(V, 32, 2, seed=0, sharp=40.0)
    drafter = tiny_llama(V, 16, 1, seed=1, sharp=40.0)
    return V, target, drafter


@pytest.fixture(scope="session")
def small_vocab_pair():
    V = 6
    target = tiny_llama(V, 16, 1, seed=2, sharp=15.0)
    drafter = tiny_llama(V, 16, 1, seed=3, sharp=15.0)
    return V, target, drafter


def wrap(target, drafter, V, mode="crop"):
    return CachedLM(target, V, mode), NeuralDrafter(CachedLM(drafter, V, mode))


def tiny_qwen35(vocab: int, seed: int, sharp: float = 30.0, noise_from=None, noise: float = 0.0):
    """Tiny hybrid model with Qwen3.5's layout (gated-delta linear attention + full attention).
    If noise_from is given, returns a perturbed copy of it (a drafter with partial agreement)."""
    import copy
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(seed)
    if noise_from is not None:
        m = copy.deepcopy(noise_from)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(noise * torch.randn_like(p) * p.std().clamp_min(1e-3))
        return m
    cfg = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_conv_kernel_dim=4,
                            layer_types=["linear_attention"] * 3 + ["full_attention"],
                            max_position_embeddings=256)
    m = Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(sharp)
    return m


@pytest.fixture(scope="session")
def hybrid_pair():
    V = 48
    target = tiny_qwen35(V, seed=0)
    drafter = tiny_qwen35(V, seed=1, noise_from=target, noise=0.3)
    return V, target, drafter
