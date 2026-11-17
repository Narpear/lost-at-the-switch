import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast

from switchspec.align import label_tokens
from switchspec.lid import (INSIDE, NEAR, SKIP, SWITCH, Sentence, read_lince_conll,
                            switch_categories, switch_stats)


def test_switch_categories_basic():
    #        main  kal   office  ja    rahi  hoon  ,      but   traffic  is    bad
    tags = ["lang2", "lang2", "lang1", "lang2", "lang2", "lang2", "other", "lang1", "lang1", "lang1", "lang1"]
    cats = switch_categories(tags, k_before=1, k_after=1)
    assert cats[2] == SWITCH and cats[3] == SWITCH and cats[7] == SWITCH
    assert cats[6] == SKIP                      # punctuation ignored
    assert cats[1] == NEAR and cats[4] == NEAR and cats[5] == NEAR and cats[8] == NEAR
    assert cats[0] == INSIDE and cats[9] == INSIDE and cats[10] == INSIDE


def test_skip_tags_do_not_break_or_create_switches():
    tags = ["lang1", "ne", "lang1", "other", "lang1"]
    assert SWITCH not in switch_categories(tags)
    tags = ["lang1", "ne", "lang2"]
    assert switch_categories(tags, 0, 0) == [INSIDE, SKIP, SWITCH]


def test_read_lince(tmp_path):
    f = tmp_path / "x.conll"
    f.write_text("# sent_enum = 1\nyaar\tlang2\nthis\tlang1\n\n# sent_enum = 2\nok\tlang1\n")
    s = read_lince_conll(str(f))
    assert len(s) == 2 and s[0].words == ["yaar", "this"] and s[0].meta["sent_enum"] == "1"
    st = switch_stats(s)
    assert st["switch_words"] == 1 and st["code_switched_sentences"] == 1


@pytest.fixture(scope="module")
def bpe_tok():
    """Byte-level BPE like Qwen/Llama, trained in-memory."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["main kal office ja rahi hoon but traffic is bad"] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tok)


def test_label_tokens_alignment(bpe_tok):
    words = "main kal officeXYZ ja".split()
    tags = ["lang2", "lang2", "lang1", "lang2"]
    cats = switch_categories(tags, 0, 0)
    labs = label_tokens(words, cats, tags, bpe_tok)
    decoded = bpe_tok.decode([l.token_id for l in labs])
    assert decoded == " ".join(words)
    by_word = {}
    for l in labs:
        by_word.setdefault(l.word_idx, []).append(l)
    assert set(by_word) == {0, 1, 2, 3}
    assert all(l.category == SWITCH for l in by_word[2])
    assert len(by_word[2]) > 1                        # unseen word split into subtokens
    assert [l.word_start for l in by_word[2]] == [True] + [False] * (len(by_word[2]) - 1)
