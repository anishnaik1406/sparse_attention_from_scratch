"""Deliverable 1.4, made visible.

Runs the same inputs through the naive softmax and the safe one and prints what
each produces, so the failure is something you can see rather than something
the writeup asserts.

    python experiments/nan_demo.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dense_attention import masked_softmax  # noqa: E402
from src.patterns import build_block_mask  # noqa: E402
from src.sparse_attention import (  # noqa: E402
    block_sparse_attention,
    build_plan,
    explain_nan_sources,
    plan_token_mask,
)


def rule(t):
    print("\n" + t + "\n" + "-" * len(t))


def main():
    torch.manual_seed(0)

    rule("1. the minimal case: one row with nothing allowed")
    scores = torch.tensor([[0.3, -1.2, 2.0], [0.1, 0.4, -0.7]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    naive = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    safe = masked_softmax(scores, mask)
    print("naive :", naive.tolist())
    print("safe  :", safe.tolist())
    print(f"naive contains NaN: {torch.isnan(naive).any().item()}")

    rule("2. why: the subtraction, step by step")
    row = scores[1].masked_fill(~mask[1], float("-inf"))
    print("filled row      :", row.tolist())
    print("row max         :", row.max().item())
    print("row - max       :", (row - row.max()).tolist(), " <- inf - inf")
    print("exp(row - max)  :", torch.exp(row - row.max()).tolist())
    print("sum             :", torch.exp(row - row.max()).sum().item(), " -> 0/0")

    rule("3. where it shows up for real: a pattern with no local term")
    n, bs = 128, 16
    bm = build_block_mask(
        "dilated", n // bs, causal=True,
        window_blocks=1, dilation=2, n_taps=2, include_self=False,
    )
    plan = build_plan(bm, bs, causal=True)
    tok = plan_token_mask(plan)
    empty = ~tok.any(-1)
    print(f"query blocks with an empty allowed set : {(bm.sum(1) == 0).sum().item()}/{bm.shape[0]}")
    print(f"query rows with an empty allowed set   : {empty.sum().item()}/{n}")

    q, k, v = (torch.randn(1, 1, n, 16) for _ in range(3))
    out = block_sparse_attention(q, k, v, plan)
    naive_p = ((q @ k.transpose(-2, -1)) / math.sqrt(16)).masked_fill(
        ~tok, float("-inf")
    ).softmax(-1)
    naive_out = naive_p @ v
    print(f"safe  kernel: NaNs = {torch.isnan(out).sum().item()}, "
          f"empty rows are exactly zero = {bool((out[0,0][empty] == 0).all())}")
    print(f"naive kernel: NaNs = {torch.isnan(naive_out).sum().item()}")

    rule("4. one NaN row poisons everything downstream")
    proj = torch.nn.Linear(16, 16)
    print("safe  -> loss:", proj(out).pow(2).mean().item())
    print("naive -> loss:", proj(naive_out).pow(2).mean().item(), "  (unrecoverable)")

    rule("5. -inf vs finfo.min, on a row that is NOT empty")
    s = torch.tensor([[5.0, 1.0, 2.0]])
    m = torch.tensor([[True, False, False]])
    a = s.masked_fill(~m, float("-inf"))
    b = s.masked_fill(~m, torch.finfo(s.dtype).min)
    print("with -inf     :", (a - a.max()).tolist(), "-> exp:", torch.exp(a - a.max()).tolist())
    print("with finfo.min:", (b - b.max()).tolist()[0][:1], "... finite everywhere")
    print("safe result   :", masked_softmax(s, m).tolist())

    rule("when this arises (from src/sparse_attention.py)")
    print(explain_nan_sources())


if __name__ == "__main__":
    main()
