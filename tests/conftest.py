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
