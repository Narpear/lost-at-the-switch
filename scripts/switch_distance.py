"""Expected acceptance by signed distance to the nearest switch, from an existing
run_positional.py output (no GPU, no rerun).

Splits the "near_switch" bucket into before (-1) and after (+1, +2) the switch, so we
can see whether the dip is the drafter anticipating the switch or recovering from it.

The rows only store each sentence's position in the filtered list, so pass the SAME
--data / --only-code-switched / --only-monolingual / --limit as the run_positional call.
Every row's language tag is checked against the data, so a mismatch fails loudly.

Usage:
  python scripts/switch_distance.py --data data/raw/lince/lid_hineng/dev.conll \
      --rows results/positional/qwen_hineng_dev.jsonl --only-code-switched --limit 800
"""
import argparse
import os

from _common import read_jsonl
from switchspec.lid import is_monolingual, read_lince_conll, switch_distance
from switchspec.metrics import cluster_bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--only-code-switched", action="store_true")
    ap.add_argument("--only-monolingual", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-distance", type=int, default=3)
    ap.add_argument("--first-subtoken-only", action="store_true",
                    help="only score the first token of each word")
    a = ap.parse_args()

    sents = read_lince_conll(a.data)
    if a.only_monolingual:
        sents = [s for s in sents if is_monolingual(s.tags)]
    if a.only_code_switched:
        sents = [s for s in sents if len({t for t in s.tags if t in ("lang1", "lang2")}) > 1]
    sents = sents[: a.limit]
    dist = [switch_distance(s.tags) for s in sents]

    buckets = {}
    for r in read_jsonl(a.rows):
        if r["word_idx"] < 0 or (a.first_subtoken_only and not r["word_start"]):
            continue
        s = sents[r["sent_id"]]
        if s.tags[r["word_idx"]] != r["tag"]:
            raise SystemExit(f"row/data mismatch at sentence {r['sent_id']}, word {r['word_idx']}: "
                             "pass the same filters and --limit as the run_positional call")
        d = dist[r["sent_id"]][r["word_idx"]]
        if d is None:
            continue
        key = max(-a.max_distance, min(a.max_distance, d))
        buckets.setdefault(key, ([], []))
        buckets[key][0].append(r["exp_accept"]); buckets[key][1].append(r["sent_id"])

    name = os.path.basename(a.rows).replace(".jsonl", "")
    lines = [f"Expected acceptance by distance to the nearest switch: {name}"
             + (" (first subtokens only)" if a.first_subtoken_only else ""),
             "| distance (words) | exp_accept | 95% CI | n tokens |", "|---|---|---|---|"]
    m = a.max_distance
    for k in sorted(buckets):
        mean, lo, hi = cluster_bootstrap_ci(*buckets[k])
        label = {0: "0 (switch word)", -m: f"<= -{m}", m: f">= +{m}"}.get(k, f"{k:+d}")
        lines.append(f"| {label} | {mean:.3f} | {lo:.3f}-{hi:.3f} | {len(buckets[k][0])} |")
    text = "\n".join(lines)
    print(text)
    os.makedirs("results/report", exist_ok=True)
    suffix = "_firstsub" if a.first_subtoken_only else ""
    with open(f"results/report/switch_distance_{name}{suffix}.md", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
