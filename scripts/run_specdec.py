"""Experiment 2: actual speculative generation with per-token logging (B1, B2, B3, A, B, C).

Usage:
  # B1: off-the-shelf neural drafter, sweep gamma
  python scripts/run_specdec.py --pair qwen --prompts data/processed/hineng_dev_prompts.jsonl \
      --out results/specdec/qwen_b1_hineng.jsonl
  # B2: n-gram drafter
  ... --ngram results/ngram/qwen_hineng_n2.pkl
  # A / B3: distilled drafter
  ... --drafter-ckpt checkpoints/approachA
  # B (toy policy): stop drafting when drafter entropy > 2.0 nats
  ... --entropy-stop 2.0

Each output line = one (prompt, gamma, seed) run with generated text and every
verified draft token. Tag the generated text with the LID tagger afterwards to
get switch positions for generated (not gold) text.
"""
import argparse
import json
from statistics import mean

import torch
from tqdm import tqdm

from _common import encode_prompt, get_pair, read_jsonl, sampling_from
from switchspec.metrics import expected_speedup
from switchspec.ngram import NgramDrafter
from switchspec.specdec import entropy_stop, speculative_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--pair", default="qwen")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gammas", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--ngram", default=None, help="pickled NgramDrafter (B2 / Approach C)")
    ap.add_argument("--drafter-ckpt", default=None)
    ap.add_argument("--entropy-stop", type=float, default=None)
    ap.add_argument("--c", type=float, default=None, help="drafter/target cost ratio from sanity_check.py")
    ap.add_argument("--chat-template", action="store_true", help="wrap prompts in the model's chat template")
    ap.add_argument("--skip-existing", action="store_true", help="exit if this run already finished")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    import os
    if a.skip_existing and os.path.exists(a.out.replace(".jsonl", ".summary.json")):
        print(f"{a.out} already finished, skipping"); return

    cfg, (tok, target, drafter, V) = get_pair(a.config, a.pair, a.device)
    if a.drafter_ckpt:
        from transformers import AutoModelForCausalLM
        drafter.lm.model = AutoModelForCausalLM.from_pretrained(
            a.drafter_ckpt, dtype=next(target.model.parameters()).dtype, device_map=a.device).eval()
    if a.ngram:
        drafter = NgramDrafter.load(a.ngram)
        assert drafter.V == V, "n-gram table built with a different vocab size"
    sc = sampling_from(cfg, a.temperature)
    gammas = a.gammas or cfg["gammas"]
    seeds = a.seeds if a.seeds is not None else cfg["seeds"][:1]
    max_new = a.max_new_tokens or cfg["max_new_tokens"]
    stop = entropy_stop(a.entropy_stop) if a.entropy_stop is not None else None
    eos = sorted({t for t in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                              tok.convert_tokens_to_ids("<|eot_id|>")]
                  if isinstance(t, int) and t >= 0 and t != tok.unk_token_id})

    prompts = read_jsonl(a.prompts)[: a.limit]
    # Resume: rows are appended as they finish, so a Colab disconnect loses at most one run.
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(a.out) if os.path.exists(a.out) else []
    done = {(r["id"], r["gamma"], r["seed"]) for r in rows}
    if done:
        print(f"resuming: {len(done)} runs already in {a.out}")
    with open(a.out, "a", encoding="utf-8") as fout:
        for pr in tqdm(prompts):
            ids = encode_prompt(tok, pr["prompt"], a.chat_template)
            for gamma in gammas:
                for seed in seeds:
                    if (pr["id"], gamma, seed) in done:
                        continue
                    r = speculative_generate(target, drafter, ids, max_new, gamma, sc, seed, eos, stop)
                    row = {"id": pr["id"], "gamma": gamma, "seed": seed,
                           "text": tok.decode(r.tokens), "alpha": r.acceptance_rate,
                           "draft_efficiency": r.draft_efficiency, "tpt": r.tokens_per_target_call,
                           **r.to_dict()}
                    rows.append(row)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()

    summary = {}
    for gamma in gammas:
        rs = [r for r in rows if r["gamma"] == gamma]
        recs = [x for r in rs for x in r["records"]]
        alpha = mean(x["accepted"] for x in recs) if recs else float("nan")
        per_ex = [r["draft_efficiency"] for r in rs if r["draft_efficiency"] == r["draft_efficiency"]]
        s = {"alpha_pooled": alpha,                                   # accepted / verified, pooled
             "acceptance_paudel": mean(per_ex) if per_ex else None,   # accepted / proposed, per-example mean
             "mean_exp_accept": mean(1 - x["tv"] for x in recs) if recs else None,
             "tpt": sum(len(r["tokens"]) for r in rs) / sum(r["n_target_calls"] for r in rs),
             "n_runs": len(rs)}
        if a.c is not None:
            s["expected_speedup"] = expected_speedup(alpha, gamma, a.c)
        summary[gamma] = s
    with open(a.out.replace(".jsonl", ".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
