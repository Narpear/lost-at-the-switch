"""Build generation prompts from a LinCE-format file.

Both styles use the first half of each code-switched dev sentence (>= --min-words
words), with the same ids, so runs on the two styles can be compared prompt by prompt.

  --style continuation  the raw half-sentence; the model just keeps writing.
                        Generations drift off code-switching within a few tokens.
  --style instruct      an instruction to continue in the same mixed style, meant to be
                        run with run_specdec.py --chat-template (as Paudel et al. do).

Usage:
  python scripts/make_prompts.py data/raw/lince/lid_hineng/dev.conll data/processed/lid_hineng_dev_prompts.jsonl
  python scripts/make_prompts.py data/raw/lince/lid_hineng/dev.conll data/processed/lid_hineng_dev_prompts_instruct.jsonl --style instruct
"""
import argparse

from _common import write_jsonl
from switchspec.lid import read_lince_conll

# lang2 name and how it is written in LinCE, by dataset id in the path
LANG2 = {"hineng": ("Hindi", " written in Roman script"),
         "nepeng": ("Nepali", " written in Roman script"),
         "spaeng": ("Spanish", "")}

INSTRUCT = ("Here is the start of a message that mixes {lang} and English{script}. "
            "Continue the message in the same style, switching between {lang} and English "
            "the way it does. Write only the continuation.\n\n{prefix}")


def lang_for(path: str):
    for key, val in LANG2.items():
        if key in path:
            return val
    raise SystemExit(f"can't tell the language pair from {path}; expected one of {list(LANG2)} in the path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--style", choices=["continuation", "instruct"], default="continuation")
    a = ap.parse_args()
    lang, script = lang_for(a.src) if a.style == "instruct" else (None, None)
    rows = []
    for i, s in enumerate(read_lince_conll(a.src)):
        langs = {t for t in s.tags if t in ("lang1", "lang2")}
        if len(langs) < 2 or len(s.words) < a.min_words:
            continue
        h = len(s.words) // 2
        prefix = " ".join(s.words[:h])
        prompt = prefix if a.style == "continuation" else INSTRUCT.format(lang=lang, script=script, prefix=prefix)
        rows.append({"id": f"{a.src}:{i}", "task": a.style, "prompt": prompt, "prefix": prefix,
                     "reference": " ".join(s.words[h:])})
    write_jsonl(a.out, rows)
    print(f"wrote {len(rows)} {a.style} prompts to {a.out}")


if __name__ == "__main__":
    main()
