"""Deliverable 1.6 - a 2-layer character-level GPT.

Deliberately minimal, and deliberately *identical* across conditions: the only
thing that changes between the dense run and the sparse runs is the block mask
handed to SparseSelfAttention. Same parameter count, same init, same data
order, same seed. If the loss moves, the pattern moved it.

Sizes follow the assignment's "about ten minutes on a T4" budget:
2 layers, 4 heads, d_model 128, context 256.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_attention import SparseSelfAttention


class MLP(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_block, pattern, head_patterns, dropout, pattern_kw):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SparseSelfAttention(
            d_model,
            n_heads,
            block_size=attn_block,
            pattern=pattern,
            head_patterns=head_patterns,
            dropout=dropout,
            pattern_kw=pattern_kw,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        attn_block: int = 32,
        pattern: str = "dense",
        head_patterns: Optional[Sequence[str]] = None,
        dropout: float = 0.1,
        pattern_kw: Optional[dict] = None,
    ):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, attn_block, pattern, head_patterns, dropout, pattern_kw)
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.context :]
            if ctx.shape[1] < self.context:
                # left-pad so T stays a multiple of the attention block size;
                # causal masking means the pad tokens can only be read by
                # positions we discard, so this does not leak.
                pad = self.context - ctx.shape[1]
                ctx = torch.cat([torch.zeros_like(ctx[:, :1]).repeat(1, pad), ctx], dim=1)
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
