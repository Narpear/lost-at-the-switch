"""Baseline B2 / Approach C: build an n-gram drafter over the target tokenizer.

Input is either LinCE-format files (*.conll) or plain text (one example per line).
For Approach C, build separate tables from romanised vs Devanagari text.

Usage:
  python scripts/build_ngram.py --pair qwen --n 2 --inputs data/raw/lince/lid_hineng/train.conll \
      --out results/ngram/qwen_hineng_n2.pkl
"""
import argparse
from pathlib import Path

from _common import load_config
from switchspec.lid import read_lince_conll
from switchspec.models import text_vocab_size
from switchspec.ngram import NgramDrafter


def iter_texts(paths):
    for p in paths:
        if p.endswith(".conll"):
            for s in read_lince_conll(p):
                yield " ".join(s.words)
        else:
            with open(p, encoding="utf-8") as f:
                yield from (l.strip() for l in f if l.strip())


def main():
    from transformers import AutoConfig, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--pair", default="qwen")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pc = load_config(a.config)["pairs"][a.pair]
    tok = AutoTokenizer.from_pretrained(pc["target"])
    V = min(len(tok), text_vocab_size(AutoConfig.from_pretrained(pc["target"])),
            text_vocab_size(AutoConfig.from_pretrained(pc["drafter"])))
    seqs = (tok(t, add_special_tokens=True)["input_ids"] for t in iter_texts(a.inputs))
    d = NgramDrafter(a.n, V).fit(seqs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.save(a.out)
    print(f"saved {a.n}-gram drafter with {sum(len(t) for t in d.tables)} contexts -> {a.out}")


if __name__ == "__main__":
    main()
