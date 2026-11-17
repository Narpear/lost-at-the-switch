"""Loading drafter/target pairs from configs/models.yaml."""
from __future__ import annotations

import yaml
import torch

from .specdec import CachedLM, NeuralDrafter


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def shared_vocab_limit(tok, drafter_model, target_model) -> int:
    """Logit dims can be padded beyond the real vocab and differ between sizes;
    only compare over the shared, real part."""
    return min(len(tok), drafter_model.config.vocab_size, target_model.config.vocab_size)


def load_pair(pair_cfg: dict, device: str = "cuda", dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(pair_cfg["target"])
    tok_d = AutoTokenizer.from_pretrained(pair_cfg["drafter"])
    if tok.get_vocab() != tok_d.get_vocab():
        raise ValueError("Drafter and target tokenizers differ; exact verification needs a shared tokenizer.")
    target = AutoModelForCausalLM.from_pretrained(pair_cfg["target"], dtype=dt, device_map=device).eval()
    drafter = AutoModelForCausalLM.from_pretrained(pair_cfg["drafter"], dtype=dt, device_map=device).eval()
    V = shared_vocab_limit(tok, drafter, target)
    mode = pair_cfg.get("cache_mode", "auto")
    return tok, CachedLM(target, V, mode), NeuralDrafter(CachedLM(drafter, V, mode)), V
