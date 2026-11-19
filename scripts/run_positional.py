"""Experiment 1: teacher-forced expected acceptance (1 - TV) by position category.

For each gold LID-tagged sentence, one forward pass per model; per-token rows
(category, tv, kl, word_start, ...) go to a JSONL, and a per-category summary
with sentence-clustered 95% CIs is printed and saved.

Usage:
  python scripts/run_positional.py --pair qwen --data data/raw/lince/lid_hineng/dev.conll \
      --out results/positional/qwen_hineng_dev.jsonl [--limit 500] [--drafter-ckpt path]
"""
import argparse
import json

import torch
from tqdm import tqdm

from _common import get_pair, sampling_from, write_jsonl
from switchspec.lid import read_lince_conll
from switchspec.metrics import summarise_by_category
from switchspec.positional import positional_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--pair", default="qwen")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-code-switched", action="store_true")
    ap.add_argument("--only-monolingual", action="store_true",
                    help="same-domain control: sentences whose language words are all one language")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--drafter-ckpt", default=None, help="distilled drafter dir (Approach A / B3)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    import os
    if a.skip_existing and os.path.exists(a.out):
        print(f"{a.out} exists, skipping"); return

    cfg, (tok, target, drafter, V) = get_pair(a.config, a.pair, a.device)
    if a.drafter_ckpt:
        from transformers import AutoModelForCausalLM
        drafter.lm.model = AutoModelForCausalLM.from_pretrained(
            a.drafter_ckpt, dtype=next(target.model.parameters()).dtype, device_map=a.device).eval()
    sc = sampling_from(cfg, a.temperature)
    win = cfg["switch_window"]
    sents = read_lince_conll(a.data)
    if a.only_monolingual:
        from switchspec.lid import is_monolingual
        sents = [s for s in sents if is_monolingual(s.tags)]
    if a.only_code_switched:
        sents = [s for s in sents if len({t for t in s.tags if t in ("lang1", "lang2")}) > 1]
    sents = sents[: a.limit]

    rows = []
    for i, s in enumerate(tqdm(sents)):
        rows += positional_records(s, i, tok, target.model, drafter.lm.model, V, sc,
                                   k_before=win["k_before"], k_after=win["k_after"])
    write_jsonl(a.out, rows)
    summary = {
        "exp_accept_by_category": summarise_by_category(rows, "exp_accept"),
        "exp_accept_by_tag": summarise_by_category(rows, "exp_accept", cat_key="tag"),
        "exp_accept_switch_by_word_start": summarise_by_category(
            [dict(r, ws=f"{r['category']}|start={r['word_start']}") for r in rows], "exp_accept", cat_key="ws"),
    }
    with open(a.out.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["exp_accept_by_category"], indent=2))


if __name__ == "__main__":
    main()
