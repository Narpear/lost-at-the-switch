"""Rough check: do generations stay code-switched?

Tags each generated word with the language it most often has in LinCE train
(words never seen there, or mostly tagged other/ne, are 'unknown'), then reports
per run file: how much of the output is recognisable as lang1 / lang2, and how many
switches between them there are per 100 words. A word-lookup proxy, not a real LID
tagger: good for comparing prompt styles, not for final numbers.

Usage:
  python scripts/cs_check.py --train data/raw/lince/lid_hineng/train.conll \
      results/specdec/qwen_b1_hineng.jsonl results/specdec/qwen_b1_hineng_instruct.jsonl [--show 5]
  # what real code-switched text scores, for comparison:
  python scripts/cs_check.py --train data/raw/lince/lid_hineng/train.conll --field reference \
      data/processed/lid_hineng_dev_prompts.jsonl
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from _common import read_jsonl
from switchspec.lid import read_lince_conll

WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def build_lexicon(train_path: str) -> dict[str, str]:
    counts = defaultdict(Counter)
    for s in read_lince_conll(train_path):
        for w, t in zip(s.words, s.tags):
            counts[w.lower()][t] += 1
    lex = {}
    for w, c in counts.items():
        tag, n = c.most_common(1)[0]
        if tag in ("lang1", "lang2") and n / sum(c.values()) >= 0.7:
            lex[w] = tag
    return lex


def score(text: str, lex: dict) -> dict:
    words = WORD.findall(text.lower())
    tags = [lex.get(w, "unk") for w in words]
    known = [t for t in tags if t != "unk"]
    switches = sum(1 for a, b in zip(known, known[1:]) if a != b)
    n = max(len(words), 1)
    return {"words": len(words), "lang1": tags.count("lang1") / n, "lang2": tags.count("lang2") / n,
            "unknown": tags.count("unk") / n, "switches_per_100w": 100 * switches / n,
            "has_both": int("lang1" in known and "lang2" in known)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run_specdec.py output .jsonl files")
    ap.add_argument("--train", required=True, help="LinCE train.conll of the same language pair")
    ap.add_argument("--show", type=int, default=0, help="print this many generations per file")
    ap.add_argument("--field", default="text",
                    help="'text' for run outputs; 'reference' on a prompts file scores the real LinCE continuations")
    a = ap.parse_args()
    lex = build_lexicon(a.train)
    print(f"lexicon: {len(lex)} words from {a.train}\n")
    for path in a.runs:
        rows = [dict(r, text=r[a.field]) for r in read_jsonl(path) if a.field in r]
        seen, scores = set(), []
        for r in rows:                     # one generation per prompt (first gamma/seed)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            scores.append(score(r["text"], lex))
        if not scores:
            print(f"{path}: no generations"); continue
        avg = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0]}
        print(f"{path}  ({len(scores)} prompts)")
        print("  lang1 {lang1:.0%}  lang2 {lang2:.0%}  unknown {unknown:.0%}  "
              "switches/100 words {switches_per_100w:.1f}  outputs with both languages {has_both:.0%}".format(**avg))
        for r in rows[: a.show]:
            print("  ---", r["id"].split(":")[-1], json.dumps(r["text"][:300], ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
