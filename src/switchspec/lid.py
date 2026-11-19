"""Word-level language ID handling and switch-point labelling.

Conventions (from the proposal):
  * lang1 = English, lang2 = the other language.
  * Tags that belong to neither language (ne, other, mixed, ambiguous, fw, unk,
    punctuation/emoji/numbers) are skipped entirely: they neither create nor
    break a switch.
  * A switch word is the first language-tagged word after the language changes.
  * The window W_k is the k language-tagged words before and after a switch
    (excluding the switch word itself).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

LANG_TAGS = ("lang1", "lang2")

# Map other datasets' tag sets onto LinCE's. Extend as needed.
TAG_MAP = {
    "lang1": "lang1", "en": "lang1", "eng": "lang1", "english": "lang1",
    "lang2": "lang2", "hi": "lang2", "hin": "lang2", "es": "lang2", "spa": "lang2",
    "ne": "ne",  # NOTE: LinCE "ne" = named entity, not Nepali
    "nep": "lang2",
}

# Categories assigned to every word
SWITCH, NEAR, INSIDE, SKIP = "switch", "near_switch", "inside", "skip"


@dataclass
class Sentence:
    words: list[str]
    tags: list[str]                     # normalised LinCE-style tags
    meta: dict = field(default_factory=dict)


def normalise_tag(tag: str) -> str:
    t = tag.strip().lower()
    return TAG_MAP.get(t, t)


def read_lince_conll(path: str, tag_map: Optional[dict] = None) -> list[Sentence]:
    """Read a LinCE LID file: 'word<TAB>tag' per line, blank line between sentences,
    optional '# sent_enum = N' comment lines. Unlabelled (test) files get tag 'unk'."""
    sents, words, tags, meta = [], [], [], {}
    norm = (lambda t: (tag_map or {}).get(t, normalise_tag(t)))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# "):
                k, _, v = line[2:].partition("=")
                meta[k.strip()] = v.strip()
                continue
            if not line.strip():
                if words:
                    sents.append(Sentence(words, tags, meta))
                words, tags, meta = [], [], {}
                continue
            parts = line.split("\t")
            words.append(parts[0])
            tags.append(norm(parts[1]) if len(parts) > 1 else "unk")
    if words:
        sents.append(Sentence(words, tags, meta))
    return sents


def switch_categories(tags: list[str], k_before: int = 1, k_after: int = 2) -> list[str]:
    """Assign switch / near_switch / inside / skip to each word."""
    lang_idx = [i for i, t in enumerate(tags) if t in LANG_TAGS]
    cats = [SKIP] * len(tags)
    switch_pos = set()          # positions within lang_idx
    for j in range(1, len(lang_idx)):
        if tags[lang_idx[j]] != tags[lang_idx[j - 1]]:
            switch_pos.add(j)
    near_pos = set()
    for j in switch_pos:
        for d in range(1, k_before + 1):
            if j - d >= 0:
                near_pos.add(j - d)
        for d in range(1, k_after + 1):
            if j + d < len(lang_idx):
                near_pos.add(j + d)
    for j, i in enumerate(lang_idx):
        cats[i] = SWITCH if j in switch_pos else NEAR if j in near_pos else INSIDE
    return cats


def switch_distance(tags: list[str]) -> list:
    """Signed distance, in language-tagged words, from each word to the nearest switch:
    0 = the switch word, -1 = the language word just before it, +1 = just after, ...
    None for words outside lang1/lang2 and for sentences with no switch.
    Ties go to the switch before the word (positive distance)."""
    lang_idx = [i for i, t in enumerate(tags) if t in LANG_TAGS]
    sw = [j for j in range(1, len(lang_idx)) if tags[lang_idx[j]] != tags[lang_idx[j - 1]]]
    out = [None] * len(tags)
    if not sw:
        return out
    for j, i in enumerate(lang_idx):
        d = min((j - s for s in sw), key=lambda x: (abs(x), x < 0))
        out[i] = d
    return out


def is_monolingual(tags: list[str]) -> bool:
    langs = {t for t in tags if t in LANG_TAGS}
    return len(langs) == 1


def switch_stats(sents: Iterable[Sentence]) -> dict:
    n_words = n_lang = n_switch = n_cs_sents = n_sents = 0
    for s in sents:
        n_sents += 1
        cats = switch_categories(s.tags)
        n_words += len(cats)
        n_lang += sum(c != SKIP for c in cats)
        k = sum(c == SWITCH for c in cats)
        n_switch += k
        n_cs_sents += k > 0
    return {"sentences": n_sents, "code_switched_sentences": n_cs_sents,
            "words": n_words, "language_words": n_lang, "switch_words": n_switch,
            "switches_per_cs_sentence": n_switch / max(n_cs_sents, 1)}
