"""Create tiny local models + a toy LinCE file so every script can be smoke-tested
on a laptop CPU in ~1 minute, with no downloads. Use with --config configs/toy.yaml.

Usage: python scripts/make_toy_setup.py
"""
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "toy"
LINES = [
    ("yaar main kal office ja rahi hoon but traffic is so bad",
     "lang2 lang2 lang2 lang1 lang2 lang2 lang2 lang1 lang1 lang1 lang1 lang1"),
    ("this movie was too good , ekdum mast tha",
     "lang1 lang1 lang1 lang1 lang1 other lang2 lang2 lang2"),
    ("kya scene hai bro , lets go for chai",
     "lang2 lang1 lang2 lang1 other lang1 lang1 lang1 lang2"),
    ("I am so tired aaj bahut kaam tha",
     "lang1 lang1 lang1 lang1 lang2 lang2 lang2 lang2"),
]


def main():
    TOY.mkdir(exist_ok=True)
    with open(TOY / "toy.conll", "w") as f:
        for k in range(10):
            for text, tags in LINES:
                f.write(f"# sent_enum = {k}\n")
                for w, t in zip(text.split(), tags.split()):
                    f.write(f"{w}\t{t}\n")
                f.write("\n")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([t for t, _ in LINES] * 20, trainers.BpeTrainer(
        vocab_size=320, special_tokens=["<s>", "</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>")
    hf_tok.chat_template = ("{% for m in messages %}<s>{{ m['content'] }}{% endfor %}"
                            "{% if add_generation_prompt %} reply :{% endif %}")
    V = len(hf_tok)
    for name, hidden, layers, seed in [("target", 64, 2, 0), ("drafter", 32, 1, 1)]:
        torch.manual_seed(seed)
        cfg = LlamaConfig(vocab_size=V, hidden_size=hidden, intermediate_size=2 * hidden,
                          num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=2,
                          max_position_embeddings=512, bos_token_id=0, eos_token_id=1)
        m = LlamaForCausalLM(cfg)
        m.save_pretrained(TOY / name)
        hf_tok.save_pretrained(TOY / name)
    print(f"toy models + data written to {TOY}")


if __name__ == "__main__":
    main()
