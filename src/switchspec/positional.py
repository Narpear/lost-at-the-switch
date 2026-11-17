"""Teacher-forced positional analysis.

Run drafter and target over a gold code-switched sentence in one forward pass
each, and record TV(p_t, q_t) at every token. 1 - TV_t is exactly the expected
probability that a token drafted at position t would be accepted (Leviathan
et al. 2023), and gold LID labels are available at every position because the
text is the dataset text, not model output. This is the cleanest way to answer
"does acceptance drop at switch points?".
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .align import label_tokens
from .lid import Sentence, switch_categories
from .specdec import SamplingConfig, warp_to_probs
from .metrics import kl, tv


@torch.no_grad()
def full_logits(model, ids: Sequence[int], vocab_limit: int) -> torch.Tensor:
    dev = next(model.parameters()).device
    out = model(input_ids=torch.tensor([list(ids)], device=dev), use_cache=False)
    return out.logits[0, :, :vocab_limit].float().cpu()


def positional_records(sent: Sentence, sent_id, tokenizer, target, drafter, vocab_limit: int,
                       cfg: SamplingConfig, prefix_ids: Optional[Sequence[int]] = None,
                       k_before: int = 1, k_after: int = 2) -> list[dict]:
    cats = switch_categories(sent.tags, k_before, k_after)
    labels = label_tokens(sent.words, cats, sent.tags, tokenizer)
    prefix = list(prefix_ids or [])
    if not prefix:
        bos = tokenizer.bos_token_id
        prefix = [bos] if bos is not None else []
    if not prefix:
        # No BOS and no prompt: the first sentence token has no prediction; skip it.
        prefix, labels = [labels[0].token_id], labels[1:]
    ids = prefix + [l.token_id for l in labels]
    lt = full_logits(target, ids, vocab_limit)
    ld = full_logits(drafter, ids, vocab_limit)
    P = len(prefix)
    # logits at index t-1 predict token t
    p = warp_to_probs(lt[P - 1: len(ids) - 1], cfg)
    q = warp_to_probs(ld[P - 1: len(ids) - 1], cfg)
    tvs, kls = tv(p, q), kl(p, q)
    rows = []
    for j, lab in enumerate(labels):
        rows.append({
            "sent_id": sent_id, "tok_idx": j, "token_id": lab.token_id,
            "word_idx": lab.word_idx, "category": lab.category, "tag": lab.tag,
            "word_start": lab.word_start, "tv": float(tvs[j]), "exp_accept": 1 - float(tvs[j]),
            "kl": float(kls[j]),
            "p_gold": float(p[j, lab.token_id]), "q_gold": float(q[j, lab.token_id]),
        })
    return rows
