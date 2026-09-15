Built with AI assistance, which the task brief permits. All experiments in
`results/` were run by me on a Colab T4, and the numbers in this README and
in WRITEUP.md come from those runs.
The commit history here is short: the implementation was developed with AI assistance in a single session and committed in stages as results came in, rather than incrementally over two weeks.
# Sparse Attention from Scratch

Task 1 of the Postman AI/ML recruitment task. Dense attention written out by
hand, three sparsity patterns built on a block-gather kernel that never
materialises the `[N, N]` score matrix, a correctness harness that checks the
sparse path against the dense one exactly, a benchmark showing the O(N) vs
O(N²) split, and a small char-level GPT to see what each pattern costs in loss.

No `F.scaled_dot_product_attention` anywhere except in the benchmark, where it
appears only as a labelled speed reference.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, matplotlib
```

CPU is enough for everything here. A free Colab T4 makes the 1.6 training run
take minutes instead of tens of minutes.

## Run

```bash
python tests/test_correctness.py         # 1.3 / 1.4  pass-fail table
python experiments/nan_demo.py           # 1.4        the NaN, shown not asserted
python experiments/plot_patterns.py      # 1.2        mask pictures + density scaling
python benchmarks/benchmark.py           # 1.5        time + memory, 512 -> 8192
python experiments/train_char_gpt.py --context 256 --attn-block 16 --steps 2000  # 1.6
bash run_all.sh                          # all of the above
```

Useful flags:

```bash
python benchmarks/benchmark.py --device cuda --heads 8 --lengths 512,1024,2048,4096,8192
python benchmarks/benchmark.py --mem-budget-gb 6      # how big dense is allowed to get
python experiments/train_char_gpt.py --patterns dense,sliding_window,mixed --sample
```

`results/` gets the CSVs and `results/hardware.json`; `plots/` gets the figures.

## Repository layout

```
src/dense_attention.py     1.1  manual dense attention + the safe masked softmax (1.4)
src/patterns.py            1.2  sliding window, BigBird-style, dilated (stretch), dense
src/sparse_attention.py    1.2  block-gather sparse kernel, plan builder, nn.Module
src/model.py               1.6  2-layer char GPT
tests/test_correctness.py  1.3  17 checks, sparse vs dense under an identical mask
benchmarks/benchmark.py    1.5  wall-clock + peak memory sweep, hardware reported
experiments/nan_demo.py    1.4  side-by-side naive vs safe softmax
experiments/plot_patterns.py    mask visualisation, density vs sequence length
experiments/train_char_gpt.py   1.6  one training run per pattern, held-out loss
WRITEUP.md                 1.7  what each pattern loses, and why
```

## Deliverable map

| Item | Where | Status |
|---|---|---|
| 1.1 manual dense attention | `src/dense_attention.py::dense_attention` | done |
| 1.2 two sparsity patterns | `src/patterns.py` — sliding window, BigBird (local+global+random) | done, plus a third (dilated) |
| 1.3 correctness harness | `tests/test_correctness.py` | done, 17/17 |
| 1.4 NaN handling | `src/dense_attention.py::masked_softmax`, `experiments/nan_demo.py` | done |
| 1.5 benchmark 512→8192 | `benchmarks/benchmark.py`, `plots/benchmark_*.png` | done |
| 1.6 char-GPT quality eval | `experiments/train_char_gpt.py`, `results/quality.csv` | done |
| 1.7 writeup | `WRITEUP.md` | done |
| stretch: third pattern | `dilated_block_mask` | done |
| stretch: per-head mixing | `SparseSelfAttention(head_patterns=[...])`, `--patterns mixed` | done |

## How the sparse kernel works

Patterns are defined at *block* granularity: a `[nb, nb]` boolean saying which
key blocks each query block may read. Token-level structure (causality inside
the diagonal block, padding) is applied afterwards on a much smaller tensor.

Block granularity is the point. A token-granular mask saves no work — you still
compute every score before zeroing it. Coarse structure is what lets whole tiles
be skipped, which is the same reason FlashAttention skips fully-masked causal
tiles.

`build_plan` turns the block mask into a gather plan: for each query block, the
list of allowed key blocks, padded to a common length `kmax`, plus a fine mask
that kills the padding slots and enforces causality. The forward pass then
gathers K and V and computes a score tensor of shape

```
[B, H, nb, block, kmax * block]      instead of      [B, H, N, N]
```

`kmax` stays roughly constant as N grows for all three sparse patterns, so cost
is linear in N.

`plan_token_mask` reconstructs the equivalent `[N, N]` mask. It is used only by
the correctness harness, so "sparse matches dense" is an exact statement about
the same mathematical object rather than a similarity check.

## Results at a glance

Benchmark (Tesla T4, B=1 H=8 D=64, block 64, forward only — relative numbers,
see `results/hardware.json`):

| N | dense | sliding window | bigbird | dilated |
|---:|---:|---:|---:|---:|
| 512 | 1.12 ms | 0.67 ms | 0.88 ms | 0.88 ms |
| 1024 | 3.59 ms | 0.98 ms | 1.77 ms | 1.54 ms |
| 2048 | 13.62 ms | 1.80 ms | 3.01 ms | 2.56 ms |
| 4096 | 49.57 ms | 3.04 ms | 5.89 ms | 4.97 ms |
| 8192 | 204.24 ms | 6.03 ms | 11.74 ms | 9.81 ms |

Dense ×3.8 per doubling, sparse ×1.9. Peak memory at 8192: 8.8 GB dense vs
365 MB for the sliding window.

## Reading the numbers

* **Timings are relative.** Compare sparse against dense from the same run on
  the same machine. `results/hardware.json` records what that machine was.
* **On CUDA**, peak memory comes from `torch.cuda.max_memory_allocated` and is
  exact — that is the path every number here came from. **On CPU** it falls
  back to a sampled RSS delta and is noisy.
* Dense is skipped, not silently dropped, when its score matrix alone would
  exceed `--mem-budget-gb`. On the T4 run nothing was skipped: dense reached
  8192 at 8.8 GB of the card's 15.6 GB.

## Known limitations

* Forward pass only for the benchmark, as specified. The kernel is
  autograd-differentiable (the harness checks gradients are finite) but there
  is no hand-written backward.
* The gather materialises K/V blocks, so it is memory-lighter than dense but
  heavier than a fused kernel that streams tiles from HBM. That is task 4's
  problem, not this one.
* `seq_len` must be a multiple of `block_size`; `build_plan(..., key_valid=)`
  handles the padded case and the harness covers it (check F).
* Random blocks in the BigBird pattern are drawn once from a fixed seed rather
  than resampled per layer, so the pattern is reproducible and testable.
* The correctness harness runs on CPU only, which let a device-mismatch bug in
  the benchmark's `density()` call through until the first CUDA run.
  Parameterising the harness over devices would have caught it.
