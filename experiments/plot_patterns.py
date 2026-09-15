"""Visualise the block masks and how their density scales with sequence length.

Not a required deliverable, but the density-vs-N plot is the cleanest way to see
why the sliding window is O(N) and the dense mask is O(N^2): the window keeps a
constant number of blocks per row, so its share of the causal triangle falls
like 1/N.

    python experiments/plot_patterns.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.patterns import build_block_mask, density  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = ["dense", "sliding_window", "bigbird", "dilated"]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.join(REPO, "plots"), exist_ok=True)

    nb = 32
    fig, axes = plt.subplots(1, len(PATTERNS), figsize=(3 * len(PATTERNS), 3.3))
    for ax, name in zip(axes, PATTERNS):
        m = build_block_mask(name, nb, causal=True)
        ax.imshow(m.numpy(), cmap="Greys", interpolation="nearest")
        ax.set_title(f"{name}\ndensity {density(m):.2f}", fontsize=9)
        ax.set_xlabel("key block")
        if ax is axes[0]:
            ax.set_ylabel("query block")
    fig.suptitle(f"causal block masks, {nb} blocks", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "plots", "patterns.png"), dpi=140)
    print("wrote plots/patterns.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    nbs = [8, 16, 32, 64, 128, 256]
    for name in PATTERNS:
        ax.plot(nbs, [density(build_block_mask(name, n, causal=True)) for n in nbs],
                marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of blocks (∝ sequence length)")
    ax.set_ylabel("fraction of the causal triangle kept")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "plots", "pattern_density.png"), dpi=140)
    print("wrote plots/pattern_density.png")


if __name__ == "__main__":
    main()
