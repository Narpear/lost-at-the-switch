import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from switchspec.models import load_config, load_pair  # noqa: E402
from switchspec.specdec import SamplingConfig  # noqa: E402


def get_pair(config_path: str, pair: str, device: str):
    cfg = load_config(config_path)
    dtype = "float32" if device == "cpu" else "bfloat16"
    return cfg, load_pair(cfg["pairs"][pair], device=device, dtype=dtype)


def sampling_from(cfg: dict, temperature=None) -> SamplingConfig:
    s = dict(cfg["sampling"])
    if temperature is not None:
        s["temperature"] = temperature
    return SamplingConfig(**s)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def encode_prompt(tok, text: str, chat: bool) -> list[int]:
    """Raw text (add BOS if the tokenizer uses one), or the model's chat template with a
    generation prompt (as Paudel et al. do). Qwen3.x thinking is switched off."""
    if not chat:
        return tok(text, add_special_tokens=True)["input_ids"]
    msgs = [{"role": "user", "content": text}]
    try:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      enable_thinking=False)
    except TypeError:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if isinstance(out, dict) or hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)
