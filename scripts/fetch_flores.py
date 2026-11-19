"""FLORES-200 monolingual controls + English->X translation prompts.

Writes, for each language:
  data/raw/flores/<lang>.conll         every word tagged lang1 (English) or lang2 (others),
                                       so run_positional.py works unchanged (all "inside")
  data/processed/flores_translate_<lang>.jsonl   prompts "Translate ... into <Language>"
                                       for B1 reproduction (Paudel et al. do translation)

Source: openlanguagedata/flores_plus on Hugging Face (gated: click "Agree and access"
on the dataset page first, with the same HF account as your HF_TOKEN).
ASSUMPTION: config names are FLORES codes (eng_Latn, ...) with splits dev/devtest and a
"text" column. If that fails, the script falls back to filtering the default config.

Usage: python scripts/fetch_flores.py [--split devtest] [--langs eng_Latn spa_Latn hin_Deva npi_Deva]
"""
import argparse
import json
import os

NAMES = {"eng_Latn": "English", "spa_Latn": "Spanish", "hin_Deva": "Hindi", "npi_Deva": "Nepali"}


def load_lang(lang: str, split: str) -> list[str]:
    from datasets import load_dataset
    try:
        ds = load_dataset("openlanguagedata/flores_plus", lang, split=split)
        return list(ds["text"])
    except Exception as e:  # noqa
        print(f"  {lang}: per-language config failed ({type(e).__name__}: {str(e)[:100]}); trying default config")
    iso3, script = lang.split("_")
    ds = load_dataset("openlanguagedata/flores_plus", split=split)
    ds = ds.filter(lambda ex: ex["iso_639_3"] == iso3 and ex["iso_15924"] == script)
    return list(ds["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="devtest")
    ap.add_argument("--langs", nargs="+", default=list(NAMES))
    ap.add_argument("--out-raw", default="data/raw/flores")
    ap.add_argument("--out-prompts", default="data/processed")
    a = ap.parse_args()
    os.makedirs(a.out_raw, exist_ok=True)
    os.makedirs(a.out_prompts, exist_ok=True)

    texts = {lang: load_lang(lang, a.split) for lang in a.langs}
    for lang, sents in texts.items():
        tag = "lang1" if lang.startswith("eng") else "lang2"
        with open(f"{a.out_raw}/{lang}.conll", "w", encoding="utf-8") as f:
            for i, s in enumerate(sents):
                f.write(f"# sent_enum = {i}\n")
                for w in s.split():
                    f.write(f"{w}\t{tag}\n")
                f.write("\n")
        print(f"{lang}: {len(sents)} sentences -> {a.out_raw}/{lang}.conll")

    if "eng_Latn" in texts:
        eng = texts["eng_Latn"]
        for lang, sents in texts.items():
            if lang == "eng_Latn":
                continue
            path = f"{a.out_prompts}/flores_translate_{lang}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for i, (src, ref) in enumerate(zip(eng, sents)):
                    f.write(json.dumps({
                        "id": f"flores_{a.split}_{lang}_{i}", "task": "translation",
                        "prompt": f"Translate the following English text into {NAMES.get(lang, lang)}. "
                                  f"Reply with only the translation.\n\n{src}",
                        "reference": ref}, ensure_ascii=False) + "\n")
            print(f"  prompts -> {path}")


if __name__ == "__main__":
    main()
