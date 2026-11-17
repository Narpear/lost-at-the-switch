"""Map word-level switch categories onto tokenizer tokens (needs a *fast* tokenizer)."""
from __future__ import annotations

from dataclasses import dataclass

from .lid import SKIP


@dataclass
class TokenLabel:
    token_id: int
    word_idx: int        # -1 if the token is pure whitespace
    category: str
    word_start: bool     # True for the first subtoken of a word
    tag: str


def label_tokens(words: list[str], cats: list[str], tags: list[str], tokenizer,
                 sep: str = " ") -> list[TokenLabel]:
    """Tokenise sep.join(words) and give each token its word's category.

    Byte-level BPE tokens like ' word' carry a leading space in their char span,
    so a token is assigned to the word of its first non-space character.
    """
    text, char2word, starts = "", [], []
    for i, w in enumerate(words):
        if i:
            text += sep
            char2word += [-1] * len(sep)
        starts.append(len(text))
        text += w
        char2word += [i] * len(w)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    out = []
    word_seen = set()
    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        w = next((char2word[c] for c in range(s, e) if char2word[c] >= 0), -1)
        if w < 0:
            out.append(TokenLabel(tid, -1, SKIP, False, "space"))
            continue
        first = w not in word_seen
        word_seen.add(w)
        out.append(TokenLabel(tid, w, cats[w], first, tags[w]))
    return out
