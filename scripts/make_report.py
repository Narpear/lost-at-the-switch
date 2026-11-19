"""Collect result summaries into tables + one figure for the check-in / report.

Reads   results/positional/*.summary.json   (run_positional.py)
        results/specdec/*.summary.json      (run_specdec.py)
Writes  results/report/positional.csv, positional.md, positional.png
        results/report/specdec.csv, specdec.md

Usage: python scripts/make_report.py [--results results]
"""
import argparse
import csv
import glob
import json
import os

CATS = ["switch", "near_switch", "inside"]
COLORS = {"switch": "#2a78d6", "near_switch": "#eb6834", "inside": "#1baf7a"}  # validated categorical slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        name = os.path.basename(p).replace(".summary.json", "")
        with open(p) as f:
            out[name] = json.load(f)
    return out


def write_table(rows, header, stem):
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(f"{stem}.md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def positional(results, out):
    runs = load(f"{results}/positional/*.summary.json")
    if not runs:
        print("no positional summaries yet"); return
    rows = []
    for name, s in runs.items():
        for cat, v in s["exp_accept_by_category"].items():
            rows.append([name, cat, fmt(v["mean"]), fmt(v["ci_low"]), fmt(v["ci_high"]), v["n"]])
    write_table(rows, ["run", "category", "exp_accept", "ci_low", "ci_high", "n_tokens"], f"{out}/positional")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(runs)
    fig, axes = plt.subplots(1, len(names), figsize=(max(4, 2.6 * len(names)), 3.6),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    for ax, name in zip(axes[0], names):
        cats = [c for c in CATS if c in runs[name]["exp_accept_by_category"]]
        vals = [runs[name]["exp_accept_by_category"][c] for c in cats]
        x = range(len(cats))
        m = [v["mean"] for v in vals]
        err = [[v["mean"] - v["ci_low"] for v in vals], [v["ci_high"] - v["mean"] for v in vals]]
        ax.bar(x, m, width=0.7, color=[COLORS[c] for c in cats], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x, m, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for xi, (mi, v) in enumerate(zip(m, vals)):  # direct value labels (contrast relief)
            ax.text(xi, v["ci_high"] + 0.02, f"{mi:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(list(x)); ax.set_xticklabels([c.replace("_", "-") for c in cats], fontsize=8, color=INK2)
        ax.set_title(name.replace("_", " "), fontsize=9, color=INK)
        ax.set_facecolor(SURFACE); ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    axes[0][0].set_ylabel("expected acceptance (1 − TV)", fontsize=9, color=INK2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLORS[c], label=c.replace("_", "-")) for c in CATS],
               loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Teacher-forced expected acceptance by position (95% CI, sentence bootstrap)",
                 fontsize=10, color=INK, y=1.09)
    fig.tight_layout()
    fig.savefig(f"{out}/positional.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}/positional.png")


def specdec(results, out):
    runs = load(f"{results}/specdec/*.summary.json")
    if not runs:
        print("no specdec summaries yet"); return
    rows = []
    for name, s in runs.items():
        for gamma, v in s.items():
            rows.append([name, gamma, fmt(v.get("acceptance_paudel")), fmt(v.get("alpha_pooled", v.get("alpha"))),
                         fmt(v.get("mean_exp_accept")), fmt(v.get("tpt"), 2), fmt(v.get("expected_speedup"), 2),
                         v.get("n_runs", "")])
    write_table(rows, ["run", "gamma", "acc_per_example(Paudel)", "alpha_pooled", "mean_1-TV",
                       "tokens_per_target_call", "expected_speedup", "n_runs"], f"{out}/specdec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    out = f"{a.results}/report"
    os.makedirs(out, exist_ok=True)
    positional(a.results, out)
    specdec(a.results, out)
    for f in sorted(glob.glob(f"{out}/*.md")):
        print(f"\n== {f}\n" + open(f).read())


if __name__ == "__main__":
    main()
