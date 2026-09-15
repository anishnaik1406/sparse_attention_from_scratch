"""Deliverable 1.1 - manual dense attention.

This is the reference implementation everything else is checked against.
Written out by hand: matmul -> scale -> mask -> softmax -> matmul.
No F.scaled_dot_product_attention anywhere in this file (or in the repo,
outside of the benchmark script where it is used purely as a speed
reference and is clearly labelled as such).
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def masked_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Softmax over the last dim that is safe when a row is fully masked.

    Deliverable 1.4 lives here.

    `scores` : [..., Q, K] raw (already scaled) attention logits.
    `mask`   : broadcastable bool tensor, True = keep, False = drop.

    The naive thing is `scores.masked_fill(~mask, -inf).softmax(-1)`. If a row
    of `mask` is all False that gives softmax([-inf, -inf, ...]):

        max      = -inf
        x - max  = -inf - (-inf) = nan
        exp(nan) = nan  -> the whole row is NaN, and one NaN row poisons the
                           output projection, the residual stream, the loss,
                           and every gradient in the model.

    So we do the subtract-max softmax by hand and special-case empty rows:
      * fill masked logits with a large finite negative number, not -inf,
        so that no inf - inf ever happens;
      * compute the row max only over kept entries;
      * for rows with nothing kept, force max=0, numerator=0, denominator=1,
        which yields an all-zero attention row and therefore an all-zero
        (rather than NaN) context vector for that query.

    Returning zeros is a deliberate choice, not a hack: a query with no
    permitted keys has no information to read, and zero is the identity for
    the subsequent `p @ V`. The alternative (uniform attention over the
    forbidden set) would silently leak exactly the information the mask was
    supposed to remove.
    """
    if mask is None:
        return torch.softmax(scores, dim=-1)

    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask, neg)

    row_has_any = mask.any(dim=-1, keepdim=True)  # [..., Q, 1]
    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(row_has_any, row_max, torch.zeros_like(row_max))

    p = torch.exp(scores - row_max)
    p = p.masked_fill(~mask, 0.0)  # kills the exp(neg - 0) residue on empty rows

    denom = p.sum(dim=-1, keepdim=True)
    denom = torch.where(row_has_any, denom, torch.ones_like(denom))
    return p / denom


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_probs: bool = False,
):
    """Full O(N^2) attention, done by hand.

    q, k, v : [B, H, N, D]
    mask    : None, or bool [N, N] / [1, 1, N, N] / [B, H, N, N]; True = keep.

    Returns [B, H, N, D] (and the [B, H, N, N] probability matrix if asked).
    """
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # [B, H, N, N]

    if mask is not None:
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(0)

    p = masked_softmax(scores, mask)
    out = p @ v
    return (out, p) if return_probs else out


def causal_mask(n: int, device=None) -> torch.Tensor:
    """[N, N] bool, True where a query may see a key."""
    return torch.ones(n, n, dtype=torch.bool, device=device).tril()
