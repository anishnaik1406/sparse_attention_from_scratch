"""Merge quality runs that were executed as separate invocations.

`train_char_gpt.py` overwrites results/quality.csv each time it runs. On a slow
machine it is easier to train one or two patterns per invocation and stitch the
outputs together than to hold a single process open for an hour. Each partial
run is copied into results/parts/ and merged here.

    python experiments/merge_results.py

Ordinary use (one invocation, all patterns) does not need this.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(REPO, "results", "parts")
ORDER = ["dense", "sliding_window", "bigbird", "dilated", "mixed"]


def main():
    rows, curves, config = {}, {}, None
    for f in sorted(glob.glob(os.path.join(PARTS, "q*.csv"))):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["pattern"]] = r
    for f in sorted(glob.glob(os.path.join(PARTS, "c*.json"))):
        with open(f) as fh:
            d = json.load(fh)
            config = config or d["config"]
            curves.update(d["curves"])

    if not rows:
        print("nothing in results/parts/", file=sys.stderr)
        return 1

    ordered = [rows[p] for p in ORDER if p in rows] + [
        r for p, r in rows.items() if p not in ORDER
    ]
    out = os.path.join(REPO, "results", "quality.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ordered[0].keys()))
        w.writeheader()
        w.writerows(ordered)
    with open(os.path.join(REPO, "results", "loss_curves.json"), "w") as fh:
        json.dump({"config": config, "curves": curves}, fh, indent=2)

    base = rows.get("dense")
    print("pattern          density   val loss   val bpc   Δ vs dense")
    for r in ordered:
        d = (
            f"{float(r['val_loss']) - float(base['val_loss']):+.4f}"
            if base
            else "n/a"
        )
        print(
            f"{r['pattern']:15s} {float(r['density']):.3f}   "
            f"{float(r['val_loss']):.4f}   {float(r['val_bpc']):.4f}   {d}"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for p in ORDER:
            if p in curves:
                c = curves[p]
                ax.plot([e["step"] for e in c], [e["val"] for e in c],
                        marker="o", label=p)
        ax.set_xlabel("step")
        ax.set_ylabel("val loss (nats/char)")
        ax.set_title("TinyShakespeare, 2-layer char GPT, context 256, attn block 16",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(REPO, "plots", "quality_loss.png"), dpi=140)
        print("wrote plots/quality_loss.png")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
