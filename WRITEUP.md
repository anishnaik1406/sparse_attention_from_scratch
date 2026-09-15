# Sparse Attention from Scratch — Writeup

## 1. What I built and why it is shaped this way

Dense attention is written out by hand (`matmul → scale → mask → softmax →
matmul`) and is the reference for everything else. On top of it sits a
block-gather sparse kernel and three patterns: a causal sliding window, a
BigBird-style local+global+random mask, and a dilated/strided mask as the third
pattern (stretch).

The one design decision worth defending is that **patterns are defined at block
granularity, not token granularity**. A token-level mask buys nothing: you
still compute all N² scores and then throw most of them away, so a "sparse"
attention implemented as `scores.masked_fill(~mask, -inf)` is strictly slower
than dense attention, not faster. Sparsity only turns into speed when the
structure is coarse enough that entire tiles of the score matrix are never
touched. So a pattern here is a `[nb, nb]` boolean over blocks; the fine
structure (causality inside the diagonal block, padding) is applied afterwards
on a tensor that is already small.

`build_plan` compiles a block mask into a gather plan: per query block, the
list of allowed key blocks, padded to a common length `kmax`, plus a fine mask
that kills padding slots and enforces causality. The forward pass gathers K/V
and computes

```
[B, H, nb, block, kmax·block]     instead of     [B, H, N, N]
```

`kmax` is roughly constant in N for all three patterns, so the cost is linear.
The plan is built once and cached, because building it involves a Python loop
and would otherwise show up in the benchmark as if it were attention cost —
that alone would have made the sparse numbers look better than they are.

## 2. Correctness: what "correct" was defined to mean

The trap in checking a sparse kernel is comparing it to *unmasked* dense
attention and declaring victory when the outputs are "close". They should not
be close; they should be different, because the mask changed the function.

So `plan_token_mask` reconstructs the exact `[N, N]` mask that a plan is
equivalent to, and the harness asks for **bit-level agreement** between the
sparse kernel and manual dense attention fed that same mask. In float64 the
max absolute difference is 0.0e+00 for all four patterns; in float32 it is also
0.0e+00, because the gather changes the order of nothing — the same products
are summed in the same order.

Three checks beyond that mattered more than I expected:

* **No leakage through gather padding slots.** Query blocks with fewer than
  `kmax` allowed blocks get their list padded with block 0. If those slots are
  not masked, every short row attends to the prefix twice — and the rows still
  sum to 1, so the bug is invisible in the loss and invisible in the output
  norms. The harness recovers the effective probability matrix by attending to
  an identity V and asserts that mass on masked positions is exactly zero.
* **Rows are distributions** (non-negative, sum to 1) except where deliberately
  empty.
* **Gradients are finite**, since the masked softmax is where a NaN would first
  appear in a backward pass even if the forward looks clean.

17/17 checks pass (`python tests/test_correctness.py`).

## 3. NaN handling (1.4)

The failure is one line deep. `scores.masked_fill(~mask, -inf).softmax(-1)` on
a row where the mask is all `False` computes `max = -inf`, then `x - max` =
`-inf - (-inf)` = `NaN`, then `exp(NaN)/sum(NaN)` = NaN for the whole row. One
NaN row does not stay local: it goes through `p @ V`, the output projection and
the residual stream, and after one optimizer step every parameter in the model
is NaN. `experiments/nan_demo.py` runs both versions side by side; the naive
path produces 512 NaNs and a NaN loss on the same input where the safe path is
finite.

The fix in `masked_softmax`:

1. fill masked logits with `torch.finfo(dtype).min`, not `-inf`, so no
   `inf - inf` can ever occur even on rows that are only mostly masked;
2. take the row max over kept entries only, and force it to 0 on empty rows;
3. zero the numerator on masked entries and force the denominator to 1 on empty
   rows, so an empty row yields an all-zero attention distribution.

Returning **zeros** rather than a uniform distribution is a deliberate choice.
A query with no permitted keys has no information it is allowed to read, and
zero is the identity for the following `p @ V`. Uniform attention over the
forbidden set would be numerically well-behaved and semantically wrong: it
leaks exactly the information the mask existed to remove.

**When empty rows actually arise.** Not hypothetically — four ways, all of them
reachable in this repo:

* **Patterns without a local term.** A strided mask `{i−d, i−2d, …}` under
  causality leaves every query block with index `< d` nothing to attend to.
  `dilated_block_mask(include_self=False)` is the fixture; at N=128, block 16
  it produces 2 empty query blocks and 32 empty rows.
* **Padding.** A length-300 sequence with block 64 is padded to 320. The pad
  queries are real rows in the tensor and can end up with no valid keys; they
  are sliced off afterwards, but a NaN reaches the loss before you get there.
  Harness check F covers this.
* **The first block under causality.** Position 0 sees only itself. Combine
  that with any mask that also drops self-attention and row 0 is empty.
* **Mostly-masked rows with `-inf` fill.** Even when a row is not empty, if the
  fill value is `-inf` and the row max happens to be `-inf` for a
  sub-computation (a fully-masked *tile* in a tiled implementation), you get the
  same `inf - inf`. Using `finfo.min` removes the whole class.

The block-boundary version the task hints at is the tiled case: with block
granularity, a query block near the start of the sequence can gather a key
block that causality then masks completely. That tile is all-`-inf` on its own,
and any implementation that computes a per-tile max — which every fused
attention kernel does — hits the empty-row arithmetic there even though the
full row is fine. This is the same rescale hazard as task 4's online softmax,
seen from the correctness side rather than the performance side.

## 4. Cost: benchmark (1.5)

Hardware for the numbers below (`results/hardware.json`): single-core
Intel Xeon @ 2.10 GHz, 1 thread, PyTorch 2.14, CPU, float32, B=1, H=2, D=64,
block size 64, forward pass only, median of 3 timed reps after warmup.
**These are relative numbers on a deliberately slow machine** — the shapes of
the curves are the result, not the milliseconds. Rerun on a T4 with
`--device cuda --heads 8` before quoting anything.

| N | dense | sliding window | bigbird | dilated |
|---:|---:|---:|---:|---:|
| 512 | 12.1 ms | 4.4 ms (2.8×) | 8.5 ms (1.4×) | 8.8 ms (1.4×) |
| 1024 | 47.6 ms | 5.6 ms (8.5×) | 19.3 ms (2.5×) | 15.3 ms (3.1×) |
| 2048 | 218.6 ms | 15.6 ms (14.1×) | 34.4 ms (6.4×) | 26.4 ms (8.3×) |
| 4096 | 935.9 ms | 37.3 ms (25.1×) | 75.0 ms (12.5×) | 57.0 ms (16.4×) |
| 8192 | skipped (score matrix alone = 0.54 GB) | 80.4 ms | 160.1 ms | 127.0 ms |

Dense multiplies by ≈4.0 per doubling (12.1 → 47.6 → 218.6 → 935.9); every
sparse pattern multiplies by ≈2.0 once N ≥ 1024 (sliding window
15.6 → 37.3 → 80.4). That is the O(N²) vs O(N) split, measured rather than
asserted.

Materialised score entries, which is the hardware-independent version of the
memory claim: dense grows as `B·H·N²` (67 M entries at N=8192, B=1, H=2),
sliding window as `B·H·N·kmax·block` (3.1 M at the same point, 21× fewer).
Measured peak memory tracks this on CUDA exactly; on CPU it is a sampled RSS
delta and is noisy, which is why both columns are in the CSV.

**Where sparse does not win.** At N=512 the sliding window is only 2.8× faster
and BigBird only 1.4×, because at that size the dense matmul is one large,
BLAS-friendly, compute-bound operation while the sparse path spends its time on
a gather (memory-bound, no arithmetic) and on applying a fine mask whose tensor
is a substantial fraction of the score tensor it is masking. The crossover
where structure beats raw matmul efficiency is around N ≈ 1024 here. That
ordering — dense wins at short context, sparse wins at long — is the honest
summary, and it is why nobody sparsifies a 512-token model.

BigBird is consistently ~2× slower than the sliding window at equal N despite
being only ~1.7× denser. The extra cost is the gather: its allowed blocks are
scattered (global at the front, random in the middle), so the gather has poor
locality, whereas the window's blocks are contiguous and prefetch cleanly.
Density predicts FLOPs; it does not predict time.

The kernel also still loses to what a fused implementation would do. The gather
*copies* K and V into a `[nb, kmax·block, D]` buffer, so bytes moved are higher
than necessary; a Triton kernel would loop over the same blocks in-place and
never write that buffer to HBM. `masked_softmax` also allocates several
temporaries the size of the score tensor. Both are fixable and both are
essentially task 4.

## 5. Quality: what each pattern costs in loss (1.6)

Setup: 2-layer character-level GPT on TinyShakespeare, d_model 128, 4 heads,
context 256, attention block 16 (so 16 blocks), 437 K parameters, batch 8,
400 steps, AdamW with OneCycle, dropout 0.1. Every condition uses the same
seed, the same initial weights and the same batch order; the *only* difference
is the block mask. `mixed` is the per-head stretch goal: two window heads, one
BigBird head, one dilated head.

| pattern | density | val loss (nats/char) | val bpc | Δ vs dense | train min |
|---|---:|---:|---:|---:|---:|
| dense | 1.000 | 2.4358 | 3.5141 | +0.0000 | 2.09 |
| sliding_window | 0.331 | 2.3846 | 3.4402 | **−0.0512** | 0.74 |
| bigbird | 0.522 | 2.3975 | 3.4589 | −0.0383 | 1.01 |
| dilated | 0.493 | 2.4226 | 3.4950 | −0.0132 | 0.96 |
| mixed (per-head) | 0.522 | 2.3905 | 3.4487 | −0.0453 | 0.94 |

**Every sparse pattern beat dense.** That is the opposite of the expected
result and it needs explaining rather than celebrating.

It is not extra capacity — parameter counts are identical to the digit, and
sparsity strictly removes paths. Three things are going on:

1. **The locality prior is correct for this task, and 400 steps is short.**
   Next-character prediction on Shakespeare is dominated by the last few
   tokens: spelling, the current word, the speaker tag two lines up. The dense
   model has to *learn* to ignore distant tokens; the windowed model gets that
   for free from its mask. Under a short budget, a correct hard-coded prior
   beats a soft one that still has to be discovered. I would expect dense to
   catch up and pass at convergence, and the ordering here should be read as a
   statement about optimisation speed, not about model quality.

2. **Regularisation.** Removing 67 % of the attention edges is a strong
   structural constraint on a 437 K-parameter model, and val loss is what
   improved. The train/val gaps in `results/quality.csv` are consistent with
   this: sparse models have slightly *higher* train loss relative to their val
   loss than dense does.

3. **Ordering across patterns is itself informative.** The ranking is
   window < mixed < BigBird < dilated < dense. The sparsest pattern wins and the
   pattern with holes in its local coverage (dilated, which trades local density
   for reach) does worst of the sparse three — exactly what you would predict if
   local coverage is what carries the signal here and long-range capacity is
   dead weight.

**The caveat that matters most.** These are single-seed runs at 400 steps and
the spread is ~0.05 nats. I did not run multiple seeds, so I cannot claim the
gaps between the *sparse* patterns are real; only the sparse-vs-dense direction
is large enough and consistent enough across all four patterns to be worth
anything. Do not read the 0.013 gap between dilated and dense as a measurement.

**And the deeper caveat.** Character-level LM on Shakespeare is close to the
best possible case for a sliding window and a bad test of what sparsity
actually costs. The mutual information between character *t* and character
*t−400* is nearly zero once you condition on the intervening text, so a model
that provably cannot see past 48 tokens loses nothing measurable. The
conclusion is not "sparse attention is free"; it is that **perplexity on a
locally-predictable corpus does not measure the thing sparsity destroys**. The
same point recurs in task 3 of this assignment: perplexity barely moves under
bad KV-cache eviction while needle-in-a-haystack retrieval collapses. To see
the cost you need a dependency that cannot be guessed locally — selective
copying, induction heads, a fact planted 3000 tokens back. I did not build that
here, and it is the first thing I would add.

## 6. What each pattern loses (1.7)

**Sliding window.** Loses everything non-local, and loses it in a
structured way: with a window of `w` blocks and `L` layers, the receptive field
is `L·w` blocks, so connecting positions N apart needs `N/(w·block)` layers.
Two layers with a 96-token window means a 512-token model whose first token
provably cannot influence its last. No amount of training fixes this; it is a
property of the mask, not of the weights. What it keeps is exactly what
local statistics need, which is why it is so hard to catch on an LM loss.

**Dilated / strided.** Buys reach at constant budget — taps at
`i−d, i−2d, …` reach `n_taps·d` blocks back for the price of `n_taps` blocks —
but the coverage has periodic holes. A dependency at a lag that is not a
multiple of `d` is only reachable by composing layers, so its cost is paid in
depth. In a 2-layer model, a lot of it is simply unreachable. It is also the
pattern that produces empty rows if you forget the local term, which is how it
ended up as the NaN fixture.

**BigBird (local + global + random).** Local keeps the statistics, random
blocks make the attention graph an expander so that in expectation any two
positions are within a couple of hops, and global tokens give a guaranteed
short path. The theory says random is what provides the connectivity guarantee;
in practice at this scale the random blocks mostly add variance, and the
globals do the work. Worth stating plainly rather than repeating the paper's
claim.

**Why global tokens matter disproportionately.** Two separate reasons, and they
are often conflated:

1. *Routing.* A handful of columns every query can read turns a band graph
   (diameter O(N/w)) into a band-plus-star (diameter 2), independent of N. A
   couple of blocks of extra compute replace the O(N/w) layers a pure window
   would need to move information end to end. That is a structural change, not
   a marginal capacity increase, and it is why the global term earns far more
   than its density share.

2. *The softmax denominator.* Softmax rows must sum to 1. A query whose allowed
   set contains nothing relevant is still forced to distribute a full unit of
   probability mass over whatever it can see, injecting a weighted average of
   irrelevant values into the residual stream. A globally-visible token gives
   the model somewhere to dump that mass — a learned no-op. This is the same
   attention-sink phenomenon that task 3 is about, and it is why the first few
   tokens absorb enormous attention mass "regardless of content": their content
   is irrelevant precisely because their function is to be a null destination.
   Remove them and you do not just lose routing, you corrupt every query that
   had nothing to say.

Under **causal** masking only the column half of "global" survives — a prefix
token cannot attend to the future — so in a decoder the global term degenerates
to "everyone can read the prefix". That is exactly StreamingLLM's attention
sink, arrived at from the other direction.

## 7. Limitations and what I would do next

* No long-range evaluation task, which is the gap that matters most (§5). A
  selective-copy or induction-head probe would take an afternoon and would
  actually separate the patterns.
* Forward-pass benchmark only, as specified; no hand-written backward.
* The gather copies K/V instead of streaming tiles. Fusing the mask application
  into the score computation and skipping fully-masked tiles entirely, rather
  than computing and masking them, is the obvious next step and the bridge to
  task 4.
* BigBird's random blocks are drawn once with a fixed seed for reproducibility;
  the paper resamples per layer, which likely matters more at depth than at 2
  layers.
* Only tested up to 8192 on CPU with H=2. The dense curve should be pushed to
  actual OOM on a T4 rather than stopped by a memory budget flag.
