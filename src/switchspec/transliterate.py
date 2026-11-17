"""Native-script twins and transliteration quality checks.

A twin of a romanised code-switched sentence is identical except lang2 words
are transliterated to Devanagari (English stays Latin, as on a Devanagari
keyboard). The fully-native control also transliterates lang1 words.

The transliterator is any callable str -> str, so tests can use a fake one.
IndicXlit is wrapped lazily because its package pins old dependencies; install
it in a separate environment if it conflicts (see README).
"""
from __future__ import annotations

from typing import Callable, Iterable

from .lid import Sentence

Xlit = Callable[[str], str]


def make_twin(sent: Sentence, to_native: Xlit, fully_native: bool = False,
              en_to_native: Xlit | None = None) -> Sentence:
    words = []
    for w, t in zip(sent.words, sent.tags):
        if t == "lang2":
            words.append(to_native(w))
        elif t == "lang1" and fully_native:
            words.append((en_to_native or to_native)(w))
        else:
            words.append(w)
    return Sentence(words, list(sent.tags), {**sent.meta, "twin": "full" if fully_native else "mixed"})


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return levenshtein(ref, hyp) / max(len(ref), 1)


def round_trip_cer(words: Iterable[str], to_native: Xlit, to_roman: Xlit) -> float:
    """Character error rate of roman -> native -> roman, over the concatenated words."""
    refs, hyps = [], []
    for w in words:
        refs.append(w.lower())
        hyps.append(to_roman(to_native(w)).lower())
    ref, hyp = " ".join(refs), " ".join(hyps)
    return cer(ref, hyp)


class IndicXlit:
    """Thin wrapper around ai4bharat-transliteration's XlitEngine (lazy import)."""

    def __init__(self, lang: str = "hi"):
        from ai4bharat.transliteration import XlitEngine  # noqa: lazy
        self.lang = lang
        self.en2x = XlitEngine(lang, beam_width=4, rescore=True)
        self.x2en = XlitEngine(src_script_type="indic", beam_width=4, rescore=False)

    def to_native(self, w: str) -> str:
        return self.en2x.translit_word(w, topk=1)[self.lang][0]

    def to_roman(self, w: str) -> str:
        return self.x2en.translit_word(w, lang_code=self.lang, topk=1)[0]
