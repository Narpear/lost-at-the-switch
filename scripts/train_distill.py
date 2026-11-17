"""Approach A (and baseline B3 with --lam 0 --mu 0 --beta-tv 0): distil the target into the drafter
on LID-tagged code-switched text, with extra weight on switch tokens and their window.

This is the "general distillation" variant (target scores existing text). For the
task-specific B3 variant, first generate target outputs with run_specdec.py /
greedy decoding, tag them, and pass them here as LinCE-format files.

Usage:
  python scripts/train_distill.py --pair qwen --train data/raw/lince/lid_hineng/train.conll \
      --out checkpoints/qwen_A_l2_m1 --lam 2.0 --mu 1.0 --beta-tv 0.5
"""
import argparse
import random

import torch
from tqdm import tqdm

from _common import load_config
from switchspec.align import label_tokens
from switchspec.distill import weighted_distill_loss
from switchspec.lid import NEAR, SWITCH, read_lince_conll, switch_categories


def encode(sents, tok, win):
    out = []
    for s in sents:
        cats = switch_categories(s.tags, win["k_before"], win["k_after"])
        labs = label_tokens(s.words, cats, s.tags, tok)
        if len(labs) < 2:
            continue
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        ids = bos + [l.token_id for l in labs]
        # masks are over *predicted* tokens, i.e. aligned with logits[:-1]
        pad = [0] * len(bos)
        sw = pad + [l.category == SWITCH for l in labs]
        nw = pad + [l.category == NEAR for l in labs]
        out.append((ids, sw[1:], nw[1:]))
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--pair", default="qwen")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--beta-tv", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    cfg = load_config(a.config)
    pc, win = cfg["pairs"][a.pair], cfg["switch_window"]
    dt = torch.float32 if a.device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(pc["target"])
    teacher = AutoModelForCausalLM.from_pretrained(pc["target"], dtype=dt, device_map=a.device).eval()
    student = AutoModelForCausalLM.from_pretrained(pc["drafter"], dtype=torch.float32 if a.device == "cpu" else dt,
                                                   device_map=a.device)
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr)

    data = encode([s for p in a.train for s in read_lince_conll(p)], tok, win)
    print(f"{len(data)} training sequences; switch tokens = {sum(sum(d[1]) for d in data)}")
    step = 0
    for ep in range(a.epochs):
        random.shuffle(data)
        for i, (ids, sw, nw) in enumerate(tqdm(data, desc=f"epoch {ep}")):
            x = torch.tensor([ids], device=a.device)
            with torch.no_grad():
                t_logits = teacher(x).logits[:, :-1]
            s_logits = student(x).logits[:, :-1]
            m = lambda v: torch.tensor([v], device=a.device, dtype=torch.float32)
            out = weighted_distill_loss(s_logits, t_logits, m(sw), m(nw), torch.ones_like(m(sw)),
                                        a.lam, a.mu, a.beta_tv)
            (out["loss"] / a.grad_accum).backward()
            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    tqdm.write(f"step {step} loss {out['loss'].item():.4f}")
                if a.max_steps and step >= a.max_steps:
                    break
    student.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved drafter to {a.out}")


if __name__ == "__main__":
    main()
