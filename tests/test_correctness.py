"""Deliverable 1.3 + 1.4 - correctness harness.

Run directly (`python tests/test_correctness.py`) for a pass/fail table, or
under pytest. Every check compares the sparse kernel against the *manual dense*
implementation fed the identical token mask, so a pass means "the sparse path
computes exactly the attention the pattern describes", not "the outputs look
similar".

Checks
  A  sparse == dense under the same mask, all patterns, fp64 and fp32
  B  the dense pattern makes the sparse kernel reduce to plain causal attention
  C  masked positions genuinely receive zero probability (no leakage through
     gather padding slots)
  D  rows are proper distributions (sum to 1) except deliberately empty ones
  E  NaN handling: fully-masked rows -> zeros, and the naive -inf softmax that
     we are *not* using does produce NaN on the same input (so the test proves
     the hazard is real rather than hypothetical)
  F  padding: a sequence whose length is not a multiple of the block size
  G  shape/dtype/determinism sanity across B, H, D, N
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dense_attention import dense_attention, masked_softmax  # noqa: E402
from src.patterns import build_block_mask, density, expand_block_mask  # noqa: E402
from src.sparse_attention import (  # noqa: E402
    block_sparse_attention,
    build_plan,
    get_plan,
    plan_token_mask,
)

TOL = {torch.float64: 1e-12, torch.float32: 2e-5}
PATTERNS = ["dense", "sliding_window", "bigbird", "dilated"]

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def qkv(B, H, N, D, dtype, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(B, H, N, D, generator=g, dtype=torch.float64).to(dtype) * scale
        for _ in range(3)
    ]


# --------------------------------------------------------------------- A, B
def test_sparse_matches_dense():
    for dtype in (torch.float64, torch.float32):
        for name in PATTERNS:
            B, H, N, D, bs = 2, 4, 256, 32, 32
            q, k, v = qkv(B, H, N, D, dtype, seed=1)
            plan = get_plan(name, N, bs, causal=True)
            out = block_sparse_attention(q, k, v, plan)
            ref = dense_attention(q, k, v, plan_token_mask(plan))
            err = (out - ref).abs().max().item()
            assert check(
                f"A sparse==dense  {name:14s} {str(dtype).split('.')[-1]}",
                err <= TOL[dtype] and not torch.isnan(out).any(),
                f"max|Δ|={err:.2e}, density={density(plan.block_mask):.3f}",
            )


def test_dense_pattern_is_plain_causal_attention():
    B, H, N, D, bs = 1, 2, 128, 16, 32
    q, k, v = qkv(B, H, N, D, torch.float64, seed=2)
    plan = get_plan("dense", N, bs, causal=True)
    out = block_sparse_attention(q, k, v, plan)
    ref = dense_attention(q, k, v, torch.ones(N, N, dtype=torch.bool).tril())
    err = (out - ref).abs().max().item()
    assert check("B dense pattern == plain causal attention", err < 1e-12, f"max|Δ|={err:.1e}")


# ------------------------------------------------------------------------ C
def test_no_leakage_through_padding_slots():
    """The gather pads short rows with block 0. If those slots were not masked,
    early tokens would be attended to twice and every row would still sum to 1,
    so this is invisible unless you check the probabilities directly."""
    B, H, N, D, bs = 1, 1, 128, 8, 16
    q, k, v = qkv(B, H, N, D, torch.float64, seed=3)
    ok = True
    for name in PATTERNS:
        plan = get_plan(name, N, bs, causal=True)
        tok = plan_token_mask(plan)
        # recover the effective probability matrix by attending to identity V
        eye = torch.eye(N, dtype=torch.float64).expand(B, H, N, N).contiguous()
        probs = block_sparse_attention(q, k, eye, plan)[0, 0]  # [N, N]
        leaked = probs[~tok].abs().max().item()
        rowsum = probs.sum(-1)
        ok &= leaked < 1e-15 and (rowsum - 1).abs().max().item() < 1e-12
    assert check("C no probability mass on masked positions", ok)


# ------------------------------------------------------------------------ D
def test_rows_are_distributions():
    B, H, N, D, bs = 2, 2, 192, 16, 32
    q, k, v = qkv(B, H, N, D, torch.float64, seed=4)
    ok = True
    for name in PATTERNS:
        plan = get_plan(name, N, bs, causal=True)
        eye = torch.eye(N, dtype=torch.float64).expand(B, H, N, N).contiguous()
        probs = block_sparse_attention(q, k, eye, plan)
        ok &= bool((probs >= -1e-15).all()) and (probs.sum(-1) - 1).abs().max() < 1e-12
    assert check("D probabilities non-negative and normalised", ok)


# ------------------------------------------------------------------------ E
def test_nan_handling():
    """A pattern with no self/local term leaves the first blocks with an empty
    allowed set under causality. That is the real-world NaN trigger."""
    N, bs = 128, 16
    nb = N // bs
    bm = build_block_mask(
        "dilated", nb, causal=True, window_blocks=1, dilation=2, n_taps=2, include_self=False
    )
    empty_rows = int((bm.sum(1) == 0).sum())
    plan = build_plan(bm, bs, causal=True)
    q, k, v = qkv(1, 1, N, 16, torch.float64, seed=5)
    out = block_sparse_attention(q, k, v, plan)

    tok = plan_token_mask(plan)
    empty_tok = ~tok.any(-1)
    finite = not torch.isnan(out).any() and not torch.isinf(out).any()
    zeroed = out[0, 0][empty_tok].abs().max().item() if empty_tok.any() else 0.0

    # and the naive version really does blow up on the same input
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(16)
    naive = scores.masked_fill(~tok, float("-inf")).softmax(-1)
    naive_nan = bool(torch.isnan(naive).any())

    assert check(
        "E fully-masked rows give zeros, not NaN",
        finite and zeroed == 0.0 and naive_nan,
        f"{empty_rows} empty query blocks, {int(empty_tok.sum())} empty rows; "
        f"naive -inf softmax NaN = {naive_nan}",
    )


def test_nan_handling_extreme_scores():
    """finfo.min fill + subtract-max must also survive huge logits (the same
    failure mode a fused kernel hits in fp16)."""
    N, bs = 64, 16
    q, k, v = qkv(1, 2, N, 8, torch.float32, seed=6, scale=50.0)
    plan = get_plan("sliding_window", N, bs, causal=True)
    out = block_sparse_attention(q, k, v, plan)
    ref = dense_attention(q, k, v, plan_token_mask(plan))
    ok = not torch.isnan(out).any() and (out - ref).abs().max().item() < 1e-4
    assert check("E' large-magnitude scores stay finite", ok)


def test_masked_softmax_unit():
    s = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    m = torch.tensor([[True, True, False], [False, False, False]])
    p = masked_softmax(s, m)
    ok = (
        not torch.isnan(p).any()
        and abs(p[0].sum().item() - 1.0) < 1e-6
        and p[1].abs().sum().item() == 0.0
        and p[0, 2].item() == 0.0
    )
    assert check("E'' masked_softmax unit case", ok)


# ------------------------------------------------------------------------ F
def test_padded_sequence_length():
    """N=300 with block_size 64 -> pad to 320, mark the 20 pad keys invalid,
    slice the output back. Pad queries must not contaminate real ones."""
    N, bs, D = 300, 64, 16
    npad = (-N) % bs
    Np = N + npad
    q, k, v = qkv(1, 2, Np, D, torch.float64, seed=7)
    key_valid = torch.zeros(Np, dtype=torch.bool)
    key_valid[:N] = True
    bm = build_block_mask("sliding_window", Np // bs, causal=True, window_blocks=2)
    plan = build_plan(bm, bs, causal=True, key_valid=key_valid)
    out = block_sparse_attention(q, k, v, plan)[:, :, :N]

    tok = plan_token_mask(plan) & key_valid[None, :]
    ref = dense_attention(q, k, v, tok)[:, :, :N]
    ok = not torch.isnan(out).any() and (out - ref).abs().max().item() < 1e-12
    assert check("F non-multiple sequence length via padding", ok, f"N={N} -> {Np}")


# ------------------------------------------------------------------------ G
def test_shapes_and_determinism():
    ok = True
    for (B, H, N, D, bs) in [(1, 1, 64, 8, 16), (3, 8, 512, 64, 64), (2, 2, 128, 32, 32)]:
        q, k, v = qkv(B, H, N, D, torch.float32, seed=8)
        plan = get_plan("bigbird", N, bs, causal=True)
        o1 = block_sparse_attention(q, k, v, plan)
        o2 = block_sparse_attention(q, k, v, plan)
        ok &= o1.shape == (B, H, N, D) and torch.equal(o1, o2)
    assert check("G shapes and determinism across configs", ok)


def test_gradients_are_finite():
    N, bs = 128, 32
    q, k, v = qkv(1, 2, N, 16, torch.float64, seed=9)
    q.requires_grad_(True)
    plan = get_plan("bigbird", N, bs, causal=True)
    block_sparse_attention(q, k, v, plan).sum().backward()
    assert check("G' gradients finite", bool(torch.isfinite(q.grad).all()))


def main():
    torch.manual_seed(0)
    tests = [
        test_sparse_matches_dense,
        test_dense_pattern_is_plain_causal_attention,
        test_no_leakage_through_padding_slots,
        test_rows_are_distributions,
        test_nan_handling,
        test_nan_handling_extreme_scores,
        test_masked_softmax_unit,
        test_padded_sequence_length,
        test_shapes_and_determinism,
        test_gradients_are_finite,
    ]
    for t in tests:
        try:
            t()
        except AssertionError:
            pass
        except Exception as e:  # noqa: BLE001
            check(f"{t.__name__} raised", False, repr(e))

    n_pass = sum(ok for _, ok, _ in _results)
    print("\n" + "-" * 60)
    print(f"{n_pass}/{len(_results)} checks passed")
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
