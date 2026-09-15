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

Hardware (`results/hardware.json`): Tesla T4, 15.6 GB, driver 580.82.07,
PyTorch 2.11 + CUDA 12.8, float32, B=1, H=8, D=64, block size 64, forward pass
only, median of 5 timed reps after warmup. Peak memory is
`torch.cuda.max_memory_allocated`, so these are exact allocator numbers rather
than the sampled RSS proxy the CPU path falls back to.

| N | dense | sliding window | bigbird | dilated |
|---:|---:|---:|---:|---:|
| 512 | 1.12 ms | 0.67 ms (1.7×) | 0.88 ms (1.3×) | 0.88 ms (1.3×) |
| 1024 | 3.59 ms | 0.98 ms (3.7×) | 1.77 ms (2.0×) | 1.54 ms (2.3×) |
| 2048 | 13.62 ms | 1.80 ms (7.6×) | 3.01 ms (4.5×) | 2.56 ms (5.3×) |
| 4096 | 49.57 ms | 3.04 ms (16.3×) | 5.89 ms (8.4×) | 4.97 ms (10.0×) |
| 8192 | 204.24 ms | 6.03 ms (33.9×) | 11.74 ms (17.4×) | 9.81 ms (20.8×) |

Dense multiplies by ≈3.8 per doubling (1.12 → 3.59 → 13.62 → 49.57 → 204.24);
sliding window multiplies by ≈1.9 once N ≥ 1024 (0.98 → 1.80 → 3.04 → 6.03).
That is the O(N²) vs O(N) split, measured rather than asserted, and the gap
compounds: 1.7× at 512 becomes 34× at 8192.

Peak memory shows the same split more starkly, because there is no constant
overhead hiding in it:

| N | dense | sliding window | ratio |
|---:|---:|---:|---:|
| 512 | 45.8 MB | 31.7 MB | 1.4× |
| 1024 | 151.3 MB | 53.0 MB | 2.9× |
| 2048 | 566.8 MB | 97.5 MB | 5.8× |
| 4096 | 2215.5 MB | 186.4 MB | 11.9× |
| 8192 | 8784.6 MB | 365.2 MB | 24.1× |

Dense at 8192 needs 8.8 GB for a single forward pass at batch 1. It fits on a
T4's 15.6 GB, but only just, and one more doubling would not. The sliding
window needs 365 MB for the same computation. Materialised score entries, the
hardware-independent version of the same claim, are 537 M for dense against
12.6 M for the window — a 43× reduction that the 24× memory ratio understates
because Q, K and V themselves are a fixed cost both paths pay.

**Where sparse does not win.** At N=512 the sliding window is only 1.7× faster
and BigBird only 1.3×, because at that size the dense matmul is one large,
BLAS-friendly, compute-bound operation while the sparse path spends its time on
a gather (memory-bound, no arithmetic) and on applying a fine mask whose tensor
is a substantial fraction of the score tensor it is masking. On a GPU this is
more pronounced than on CPU: 512×512 attention barely occupies a T4, so the
sparse version is saving work the hardware had spare anyway.

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
context 256, attention block 16 (so 16 blocks), 437 K parameters, batch 32,
2000 steps, AdamW with OneCycle, dropout 0.1, on a Colab T4. Every condition
uses the same seed, the same initial weights and the same batch order; the
*only* difference is the block mask.

| pattern | density | val loss (nats/char) | val bpc | Δ vs dense | train min |
|---|---:|---:|---:|---:|---:|
| dense | 1.000 | 1.6494 | 2.3796 | +0.0000 | 1.88 |
| sliding_window | 0.331 | 1.6058 | 2.3167 | **−0.0436** | 0.70 |
| bigbird | 0.522 | 1.6258 | 2.3456 | −0.0236 | 0.99 |
| dilated | 0.493 | 1.6259 | 2.3457 | −0.0235 | 0.90 |

**Every sparse pattern beat dense**, and the effect survived a 5× increase in
training budget — an earlier 400-step run showed the same ordering with the
same rough spread. That rules out the first explanation I reached for, which
was that dense simply hadn't had time to learn what the window mask hard-codes.

It is not extra capacity: parameter counts are identical to the digit, and
sparsity strictly removes attention paths. What is left is regularisation.
Removing 67 % of the attention edges is a strong structural constraint on a
437 K-parameter model, and val loss is what improved while the gap to train
loss stayed comparable. The locality prior encoded by the mask is simply
correct for this task, and a correct hard constraint beats a soft one the model
has to discover and maintain.

The ordering across patterns is consistent with that reading: the sparsest
pattern wins outright, and BigBird and dilated land within 0.0001 of each other
at similar densities. Whatever is helping scales with how much attention is
removed, not with which long-range structure the pattern adds — which is
another way of saying the long-range capacity is dead weight on this task.

**The caveat that matters most.** These are single-seed runs and
the spread is ~0.05 nats. I did not run multiple seeds, so I cannot claim the
gaps between the *sparse* patterns are real; only the sparse-vs-dense direction
is large enough and consistent enough across all four patterns to be worth
anything. Do not read the 0.0001 gap between BigBird and dilated as a measurement.

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
* Tested up to 8192 on a T4, where dense still fits (8.8 GB of 15.6 GB). The
  dense curve should be pushed to actual OOM at 16384 to show the hard wall,
  not just the slope.
