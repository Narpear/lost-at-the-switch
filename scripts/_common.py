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
