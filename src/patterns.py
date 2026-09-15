"""Deliverable 1.2 (+ stretch) - sparsity patterns.

Every pattern is expressed as a *block* mask of shape [n_blocks, n_blocks],
where entry (i, j) says "query block i is allowed to look at key block j".
Token-level structure that is finer than a block (causality inside the
diagonal block, padding) is applied later, in sparse_attention.py.

Why block granularity, when the assignment talks about sliding windows in
tokens? Because a token-granular mask buys you nothing: you still have to
touch every (q, k) pair to know it is zero. Sparsity only becomes speed when
it is structured coarsely enough that whole tiles of the score matrix can be
skipped without being computed. This is exactly what Longformer, BigBird and
FlashAttention's causal block-skipping do. Block size is a knob:
block_size = 1 recovers the token-granular mask (and all of the cost).

Patterns implemented
--------------------
  sliding_window : local band of `window_blocks` blocks around the diagonal
  bigbird        : local band + global rows/columns + random blocks
  dilated        : stretch goal, strided/dilated band (Sparse Transformer-ish)

`expand_block_mask` turns any of these into the [N, N] token mask that the
dense reference is fed, so that "sparse == dense under the same mask" is a
meaningful, exactly-checkable statement.
"""

from __future__ import annotations

import torch


def expand_block_mask(block_mask: torch.Tensor, block_size: int) -> torch.Tensor:
    """[nb, nb] block mask -> [N, N] token mask (N = nb * block_size)."""
    return block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=1
    )


def _apply_causal(block_mask: torch.Tensor, causal: bool) -> torch.Tensor:
    """Drop blocks strictly above the diagonal.

    The diagonal block is kept: it is only *partially* masked, and that part is
    handled at token level later.
    """
    if not causal:
        return block_mask
    nb = block_mask.shape[0]
    tri = torch.ones(nb, nb, dtype=torch.bool).tril()
    return block_mask & tri


def sliding_window_block_mask(
    n_blocks: int, window_blocks: int = 3, causal: bool = True
) -> torch.Tensor:
    """Local band. `window_blocks` blocks wide, centred on the diagonal.

    With causal=True the band becomes one-sided: block i sees blocks
    [i - window_blocks + 1, i].
    """
    idx = torch.arange(n_blocks)
    dist = idx[:, None] - idx[None, :]
    if causal:
        m = (dist >= 0) & (dist < window_blocks)
    else:
        half = window_blocks // 2
        m = dist.abs() <= half
    return _apply_causal(m, causal)


def bigbird_block_mask(
    n_blocks: int,
    window_blocks: int = 3,
    n_global: int = 1,
    n_random: int = 2,
    causal: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """BigBird-style: local window + global tokens + random blocks.

    global : the first `n_global` blocks are attended to by everyone
             (column-global) and attend to everyone (row-global). In the causal
             setting the row-global part is clipped by causality, so in
             practice the useful half is the column: every query can read the
             prefix. Those blocks are the "sink" that everything falls back to.
    random : `n_random` extra key blocks per query block, drawn once with a
             fixed seed so the pattern is deterministic and reproducible
             (BigBird re-draws per layer; a fixed draw makes the correctness
             harness meaningful).
    """
    g = torch.Generator().manual_seed(seed)
    m = sliding_window_block_mask(n_blocks, window_blocks, causal=causal)

    if n_global > 0:
        m[:, :n_global] = True  # everyone reads the prefix
        m[:n_global, :] = True  # the prefix reads everyone (clipped below)

    if n_random > 0:
        for i in range(n_blocks):
            # only sample from blocks that causality allows
            hi = i + 1 if causal else n_blocks
            if hi <= 0:
                continue
            choice = torch.randint(low=0, high=hi, size=(n_random,), generator=g)
            m[i, choice] = True

    return _apply_causal(m, causal)


def dilated_block_mask(
    n_blocks: int,
    window_blocks: int = 2,
    dilation: int = 2,
    n_taps: int = 3,
    causal: bool = True,
    include_self: bool = True,
) -> torch.Tensor:
    """Stretch: strided / dilated attention.

    Query block i sees the last `window_blocks` blocks plus blocks at
    i - dilation, i - 2*dilation, ... (`n_taps` of them). Same budget as a
    window, much longer reach, but the coverage has holes.

    `include_self=False` produces a pattern that can leave a query block with
    *nothing* to attend to under causality (block 0 has no past). That is not
    an accident, it is the fixture used to exercise the NaN path in 1.4.
    """
    idx = torch.arange(n_blocks)
    dist = idx[:, None] - idx[None, :]
    m = torch.zeros(n_blocks, n_blocks, dtype=torch.bool)
    lo = 0 if include_self else 1
    m |= (dist >= lo) & (dist < max(window_blocks, lo))
    for t in range(1, n_taps + 1):
        m |= dist == t * dilation
    if not causal:
        for t in range(1, n_taps + 1):
            m |= dist == -t * dilation
    return _apply_causal(m, causal)


def dense_block_mask(n_blocks: int, causal: bool = True) -> torch.Tensor:
    """Every block sees every allowed block. Used to prove the sparse kernel
    reduces exactly to dense attention when the pattern is not sparse."""
    m = torch.ones(n_blocks, n_blocks, dtype=torch.bool)
    return _apply_causal(m, causal)


PATTERNS = {
    "dense": dense_block_mask,
    "sliding_window": sliding_window_block_mask,
    "bigbird": bigbird_block_mask,
    "dilated": dilated_block_mask,
}


def build_block_mask(name: str, n_blocks: int, causal: bool = True, **kw):
    if name not in PATTERNS:
        raise KeyError(f"unknown pattern {name!r}, have {sorted(PATTERNS)}")
    return PATTERNS[name](n_blocks, causal=causal, **kw)


def density(block_mask: torch.Tensor, causal: bool = True) -> float:
    """Fraction of the *causally reachable* score matrix that survives."""
    nb = block_mask.shape[0]
    ref = torch.ones(nb, nb, dtype=torch.bool)
    if causal:
        ref = ref.tril()
    return (block_mask & ref).sum().item() / ref.sum().item()
