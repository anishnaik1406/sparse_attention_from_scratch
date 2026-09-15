"""Deliverables 1.2 / 1.4 - the actual sparse attention kernel.

The whole point of this file is that it never builds the [N, N] score matrix.

Given a block mask [nb, nb], each query block has a (short, ragged) list of key
blocks it is allowed to read. We pad those lists to a common length `kmax`,
gather the corresponding K/V blocks, and compute a score tensor of shape

    [B, H, nb, block, kmax * block]

which is O(N * kmax * block) instead of O(N^2). For the configurations used in
the benchmark, kmax stays roughly constant as N grows, so the cost is linear in
N while dense is quadratic.

Two things that are easy to get wrong and are handled explicitly:

  * padding slots. Query blocks with fewer than kmax allowed key blocks get
    their list padded with block 0. Those slots are marked invalid in the fine
    mask so they contribute nothing. Forget this and every short row silently
    attends twice to the prefix.

  * fully-masked rows -> NaN (deliverable 1.4). See masked_softmax in
    dense_attention.py for the mechanism; see the docstring of
    `explain_nan_sources` below for when it actually happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .dense_attention import masked_softmax
from .patterns import build_block_mask, expand_block_mask


# --------------------------------------------------------------------------
# plan construction
# --------------------------------------------------------------------------


@dataclass
class SparsePlan:
    """Everything about a pattern that does not depend on the tensor values.

    Built once per (pattern, seq_len, block_size, causal) and cached, because
    building it involves a Python loop over blocks and would otherwise show up
    in the benchmark as if it were attention cost.
    """

    block_size: int
    n_blocks: int
    idx: torch.Tensor  # [nb, kmax]  long, key-block ids per query block
    fine_mask: torch.Tensor  # [nb, block, kmax*block] bool, True = keep
    block_mask: torch.Tensor  # [nb, nb] bool, the pattern it came from
    kmax: int

    @property
    def seq_len(self) -> int:
        return self.n_blocks * self.block_size

    def to(self, device):
        return SparsePlan(
            self.block_size,
            self.n_blocks,
            self.idx.to(device),
            self.fine_mask.to(device),
            self.block_mask.to(device),
            self.kmax,
        )


def build_plan(
    block_mask: torch.Tensor,
    block_size: int,
    causal: bool = True,
    key_valid: Optional[torch.Tensor] = None,
) -> SparsePlan:
    """Turn a [nb, nb] block mask into a gather plan.

    key_valid : optional [N] bool marking real (non-padding) key positions.
    """
    nb = block_mask.shape[0]
    n = nb * block_size

    counts = block_mask.sum(dim=1)
    kmax = int(counts.max().item())
    if kmax == 0:
        raise ValueError("block mask is empty; every query would be a NaN row")

    idx = torch.zeros(nb, kmax, dtype=torch.long)
    slot_valid = torch.zeros(nb, kmax, dtype=torch.bool)
    for i in range(nb):
        allowed = torch.nonzero(block_mask[i], as_tuple=False).flatten()
        c = allowed.numel()
        if c:
            idx[i, :c] = allowed
            slot_valid[i, :c] = True
        # remaining slots keep idx=0 / valid=False -> gathered but masked out

    ar = torch.arange(block_size)
    q_abs = (torch.arange(nb)[:, None] * block_size + ar[None, :])  # [nb, b]
    k_abs = (idx[:, :, None] * block_size + ar[None, None, :])  # [nb, kmax, b]
    k_abs_flat = k_abs.reshape(nb, kmax * block_size)

    keep = slot_valid[:, :, None].expand(nb, kmax, block_size).reshape(nb, -1)
    keep = keep[:, None, :].expand(nb, block_size, kmax * block_size).clone()

    if causal:
        keep &= q_abs[:, :, None] >= k_abs_flat[:, None, :]

    if key_valid is not None:
        kv = key_valid[k_abs_flat]  # [nb, kmax*b]
        keep &= kv[:, None, :]

    return SparsePlan(block_size, nb, idx, keep, block_mask.clone(), kmax)


_PLAN_CACHE: dict = {}


def get_plan(
    pattern: str,
    seq_len: int,
    block_size: int,
    causal: bool = True,
    device=None,
    **pattern_kw,
) -> SparsePlan:
    if seq_len % block_size:
        raise ValueError(
            f"seq_len {seq_len} not divisible by block_size {block_size}; "
            "use pad_to_multiple() first"
        )
    key = (pattern, seq_len, block_size, causal, tuple(sorted(pattern_kw.items())))
    if key not in _PLAN_CACHE:
        bm = build_block_mask(pattern, seq_len // block_size, causal=causal, **pattern_kw)
        _PLAN_CACHE[key] = build_plan(bm, block_size, causal=causal)
    plan = _PLAN_CACHE[key]
    return plan.to(device) if device is not None else plan


def plan_token_mask(plan: SparsePlan, causal: bool = True) -> torch.Tensor:
    """The [N, N] token mask this plan is equivalent to.

    Only used by the correctness harness, to feed the dense reference exactly
    the same mask. Never called on the fast path.
    """
    m = expand_block_mask(plan.block_mask, plan.block_size)
    if causal:
        m = m & torch.ones_like(m).tril()
    return m


# --------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: SparsePlan,
) -> torch.Tensor:
    """q, k, v: [B, H, N, D] -> [B, H, N, D]. Never materializes [N, N]."""
    B, H, N, D = q.shape
    Dv = v.shape[-1]  # V may carry a different head dim (the harness exploits this)
    b, nb, kmax = plan.block_size, plan.n_blocks, plan.kmax
    assert N == nb * b, f"plan is for seq_len {nb * b}, got {N}"

    qb = q.reshape(B, H, nb, b, D)
    kb = k.reshape(B, H, nb, b, D)
    vb = v.reshape(B, H, nb, b, Dv)

    # gather the allowed key/value blocks for every query block at once
    kg = kb[:, :, plan.idx].reshape(B, H, nb, kmax * b, D)
    vg = vb[:, :, plan.idx].reshape(B, H, nb, kmax * b, Dv)

    scores = (qb @ kg.transpose(-2, -1)) / math.sqrt(D)  # [B,H,nb,b,kmax*b]
    p = masked_softmax(scores, plan.fine_mask)  # broadcasts over B, H
    out = p @ vg  # [B,H,nb,b,Dv]
    return out.reshape(B, H, N, Dv)


def pad_to_multiple(x: torch.Tensor, block_size: int, dim: int = -2):
    """Pad the sequence dim up to a multiple of block_size. Returns (x, n_orig)."""
    n = x.shape[dim]
    rem = (-n) % block_size
    if rem == 0:
        return x, n
    shape = list(x.shape)
    shape[dim % x.dim()] = rem
    return torch.cat([x, x.new_zeros(shape)], dim=dim), n


def explain_nan_sources() -> str:
    """Deliverable 1.4, the 'explain when it arises' half.

    Concretely, in this implementation a query row ends up with an empty
    allowed set in four situations:

    1. Padding rows. A sequence of length 300 with block_size 64 is padded to
       320. The 20 pad queries at the end are real rows in the tensor; if the
       pad keys are marked invalid, some of them can have nothing legal to
       read. They are sliced off afterwards, but a NaN there propagates into
       the loss through the padded region before you get the chance.

    2. A pattern that excludes the diagonal block under causality. Strided /
       dilated patterns of the form {i - d, i - 2d, ...} with no local term
       give query block 0 (and any block with index < d) an empty row. See
       `dilated_block_mask(..., include_self=False)`.

    3. Block boundaries in the *first* block. Any causal pattern makes the
       very first query position see only itself; combine that with a mask
       that also drops self-attention (some retrieval-style masks do) and
       row 0 is empty.

    4. Padding slots in the gather. Every query block whose allowed-block
       count is below kmax carries dead slots. Those are masked out, which is
       correct, but it means the *masked* fraction of the score tensor is
       large and rows near the start of the sequence are mostly -inf. If the
       fill value were -inf rather than finfo.min, `max - max` on such a row
       is inf - inf = NaN even when the row is not fully empty in principle.

    The fix in all four cases is the same: never fill with -inf, compute the
    row max only over kept entries, and force (numerator, denominator) to
    (0, 1) on empty rows so the context vector is zero rather than NaN.
    """
    return explain_nan_sources.__doc__


# --------------------------------------------------------------------------
# nn.Module wrapper used by the char-GPT (deliverable 1.6)
# --------------------------------------------------------------------------


class SparseSelfAttention(nn.Module):
    """Causal multi-head self-attention with a configurable sparsity pattern.

    pattern="dense" runs the same gather kernel with a full block mask, so the
    dense/sparse comparison in 1.6 differs only in the mask, not in the code
    path. head_patterns (stretch goal) assigns a different pattern per head.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int = 32,
        pattern: str = "dense",
        head_patterns: Optional[Sequence[str]] = None,
        dropout: float = 0.0,
        pattern_kw: Optional[dict] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.block_size = block_size
        self.pattern = pattern
        self.head_patterns = list(head_patterns) if head_patterns else None
        if self.head_patterns is not None:
            assert len(self.head_patterns) == n_heads
        self.pattern_kw = pattern_kw or {}

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _plans(self, n, device):
        names = self.head_patterns or [self.pattern]
        return [
            get_plan(nm, n, self.block_size, causal=True, device=device, **self.pattern_kw)
            for nm in names
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        assert T % self.block_size == 0, "pad the input to a multiple of block_size"
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, H, T, Dh]

        plans = self._plans(T, x.device)
        if len(plans) == 1:
            out = block_sparse_attention(q, k, v, plans[0])
        else:  # per-head pattern mixing
            outs = [
                block_sparse_attention(
                    q[:, h : h + 1], k[:, h : h + 1], v[:, h : h + 1], plans[h]
                )
                for h in range(self.n_heads)
            ]
            out = torch.cat(outs, dim=1)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out))
