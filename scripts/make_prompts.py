"""Build generation prompts from a LinCE-format file.

Default task = continuation: the first half of each code-switched sentence is
the prompt. ASSUMPTION: this is a placeholder task design -- decide as a team
whether you want raw continuation, chat-templated "reply in Hinglish", or
PHINC/MixMT "translate into Hinglish" prompts (Table 1), and add them here.

Usage: python scripts/make_prompts.py data/raw/lince/lid_hineng/dev.conll data/processed/hineng_dev_prompts.jsonl
"""
import argparse

from _common import write_jsonl
from switchspec.lid import read_lince_conll

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--min-words", type=int, default=8)
a = ap.parse_args()
rows = []
for i, s in enumerate(read_lince_conll(a.src)):
    langs = {t for t in s.tags if t in ("lang1", "lang2")}
    if len(langs) < 2 or len(s.words) < a.min_words:
        continue
    h = len(s.words) // 2
    rows.append({"id": f"{a.src}:{i}", "task": "continuation", "prompt": " ".join(s.words[:h]),
                 "reference": " ".join(s.words[h:])})
write_jsonl(a.out, rows)
print(f"wrote {len(rows)} prompts to {a.out}")
