"""Deliverable 1.6 - quality evaluation.

Trains the same 2-layer char-level GPT on TinyShakespeare once per attention
pattern and reports train/val loss (and bits-per-character, which is the number
that is actually comparable across vocabularies).

Everything except the block mask is held fixed: seed, init, batch order,
optimiser, steps. Each run re-seeds before building the model, so the dense and
sparse models start from bit-identical weights and see identical batches.

    python experiments/train_char_gpt.py --steps 2000
    python experiments/train_char_gpt.py --patterns dense,sliding_window --steps 500

Writes results/quality.csv, results/loss_curves.json and
plots/quality_loss.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CharGPT  # noqa: E402
from src.patterns import build_block_mask, density  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def get_data(path=None):
    path = path or os.path.join(REPO, "data", "tinyshakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading TinyShakespeare -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars), chars


def batcher(data, context, batch_size, device, seed):
    """Deterministic batch stream: the same seed gives the same batches, so all
    patterns see identical data in identical order."""
    g = torch.Generator().manual_seed(seed)
    while True:
        ix = torch.randint(len(data) - context - 1, (batch_size,), generator=g)
        x = torch.stack([data[i : i + context] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + 1 + context] for i in ix]).to(device)
        yield x, y


@torch.no_grad()
def evaluate(model, data, context, batch_size, device, iters, seed=1234):
    model.eval()
    stream = batcher(data, context, batch_size, device, seed)
    tot = 0.0
    for _ in range(iters):
        x, y = next(stream)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def train_one(pattern, args, train_data, val_data, vocab, device):
    torch.manual_seed(args.seed)  # identical init across patterns
    head_patterns = None
    if pattern == "mixed":  # stretch: per-head pattern mixing
        head_patterns = ["sliding_window", "sliding_window", "bigbird", "dilated"][
            : args.heads
        ]
    model = CharGPT(
        vocab,
        context=args.context,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        attn_block=args.attn_block,
        pattern="dense" if head_patterns else pattern,
        head_patterns=head_patterns,
        dropout=args.dropout,
    ).to(device)

    nparams = sum(p.numel() for p in model.parameters())
    bm = build_block_mask(
        pattern if pattern != "mixed" else "bigbird",
        args.context // args.attn_block,
        causal=True,
    )
    dens = density(bm)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )
    stream = batcher(train_data, args.context, args.batch, device, args.seed + 7)

    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        x, y = next(stream)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate(model, val_data, args.context, args.batch, device, args.eval_iters)
            curve.append({"step": step, "train": loss.item(), "val": vl})
            print(
                f"  [{pattern:14s}] step {step:5d}/{args.steps}  "
                f"train {loss.item():.4f}  val {vl:.4f}  ({time.time()-t0:.0f}s)"
            )
        assert not math.isnan(loss.item()), "NaN loss - check the masked softmax"

    final_val = evaluate(model, val_data, args.context, args.batch, device,
                         args.eval_iters * 2)
    final_train = evaluate(model, train_data, args.context, args.batch, device,
                           args.eval_iters)
    return {
        "pattern": pattern,
        "params": nparams,
        "density": round(dens, 4),
        "train_loss": round(final_train, 4),
        "val_loss": round(final_val, 4),
        "val_bpc": round(final_val / math.log(2), 4),
        "minutes": round((time.time() - t0) / 60, 2),
    }, curve, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", default="dense,sliding_window,bigbird,dilated")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--attn-block", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--sample", action="store_true", help="print a sample per pattern")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    train_data, val_data, vocab, chars = get_data()
    print(f"device={device} vocab={vocab} train={len(train_data)} val={len(val_data)}")

    rows, curves = [], {}
    for pattern in args.patterns.split(","):
        row, curve, model = train_one(pattern, args, train_data, val_data, vocab, device)
        rows.append(row)
        curves[pattern] = curve
        print(f"  -> {row}")
        if args.sample:
            idx = torch.zeros((1, 1), dtype=torch.long, device=device)
            out = model.generate(idx, 300, temperature=0.8, top_k=40)[0].tolist()
            print("  sample:", "".join(chars[i] for i in out).replace("\n", " / ")[:300])

    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    with open(os.path.join(REPO, "results", "quality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(REPO, "results", "loss_curves.json"), "w") as f:
        json.dump({"config": vars(args), "curves": curves}, f, indent=2)

    base = next((r for r in rows if r["pattern"] == "dense"), None)
    print("\npattern          density   val loss   val bpc   Δ vs dense")
    for r in rows:
        d = f"{r['val_loss'] - base['val_loss']:+.4f}" if base else "n/a"
        print(f"{r['pattern']:15s} {r['density']:.3f}   {r['val_loss']:.4f}   "
              f"{r['val_bpc']:.4f}   {d}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for p, c in curves.items():
            ax.plot([e["step"] for e in c], [e["val"] for e in c], marker="o", label=p)
        ax.set_xlabel("step")
        ax.set_ylabel("val loss (nats/char)")
        ax.set_title(f"TinyShakespeare, 2-layer char GPT, context {args.context}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(os.path.join(REPO, "plots"), exist_ok=True)
        fig.savefig(os.path.join(REPO, "plots", "quality_loss.png"), dpi=140)
        print("wrote plots/quality_loss.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
