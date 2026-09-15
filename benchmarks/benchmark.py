"""Deliverable 1.5 - wall-clock and peak-memory benchmark, forward pass only.

Sweeps sequence length 512 -> 8192 for dense attention and each sparsity
pattern, writes results/benchmark.csv and plots/benchmark_*.png, and prints the
hardware it ran on. All numbers are relative: the interesting quantity is
sparse-vs-dense on the *same* machine in the *same* process, not absolute
throughput.

Usage
    python benchmarks/benchmark.py                      # defaults
    python benchmarks/benchmark.py --max-len 4096 --heads 4 --device cuda

Notes on measurement
  * Timing: warmup reps are discarded, then `--reps` timed reps; we report the
    median, because on a shared machine (Colab) the mean is dominated by the
    occasional descheduled run.
  * CUDA memory: torch.cuda.max_memory_allocated after reset_peak_memory_stats.
    Exact and allocator-level.
  * CPU memory: torch tensors are invisible to tracemalloc, so we poll
    /proc/self/statm from a background thread and take (peak RSS - baseline).
    Coarse, but it captures the [N, N] blowup, which is the whole point.
  * `score_elems` is the analytical number of attention-score entries the
    implementation materialises. It is hardware-independent and is the honest
    version of the memory claim; the RSS numbers are there to show it shows up
    in practice.
  * Dense runs are skipped once they exceed --mem-budget-gb of predicted score
    memory, and that is recorded rather than silently dropped: "dense OOMs
    here" is a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dense_attention import dense_attention  # noqa: E402
from src.patterns import density  # noqa: E402
from src.sparse_attention import block_sparse_attention, get_plan  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ hardware
def hardware_info(device: str) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": device,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if device == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
        try:
            info["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip()
        except Exception:  # noqa: BLE001
            pass
    return info


# -------------------------------------------------------------------- memory
def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class RSSProbe:
    """Poll RSS in a thread; peak - baseline approximates peak CPU allocation."""

    def __init__(self, interval=0.002):
        self.interval, self.peak, self._stop = interval, 0, False

    def __enter__(self):
        self.base = _rss_bytes()
        self.peak = self.base
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, _rss_bytes())
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop = True
        self.t.join()
        self.delta = max(0, self.peak - self.base)


def measure(fn, device: str, warmup: int, reps: int):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    with RSSProbe() as probe:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    peak = (
        torch.cuda.max_memory_allocated()
        if device == "cuda"
        else probe.delta
    )
    return times[len(times) // 2], peak


# ------------------------------------------------------------------ sweeping
def run(args):
    device = args.device
    dtype = getattr(torch, args.dtype)
    hw = hardware_info(device)
    print(json.dumps(hw, indent=2))

    lengths = [int(x) for x in args.lengths.split(",")]
    patterns = args.patterns.split(",")
    rows = []

    for n in lengths:
        # analytical score-matrix size, in elements, per (batch, head)
        dense_elems = args.batch * args.heads * n * n
        dense_gb = dense_elems * torch.finfo(dtype).bits / 8 / 1e9

        def make():
            g = torch.Generator(device="cpu").manual_seed(0)
            t = [
                torch.randn(
                    args.batch, args.heads, n, args.dim, generator=g, dtype=torch.float32
                ).to(device=device, dtype=dtype)
                for _ in range(3)
            ]
            return t

        q, k, v = make()
        cmask = torch.ones(n, n, dtype=torch.bool, device=device).tril()

        # ---- dense reference
        if dense_gb <= args.mem_budget_gb:
            t, mem = measure(
                lambda: dense_attention(q, k, v, cmask), device, args.warmup, args.reps
            )
            rows.append(
                dict(
                    n=n, pattern="dense", seconds=t, peak_bytes=mem,
                    score_elems=dense_elems, density=1.0, kmax=n // args.block,
                )
            )
            print(f"  n={n:6d} dense           {t*1e3:9.2f} ms  peak {mem/1e6:8.1f} MB")
            dense_t = t
        else:
            dense_t = float("nan")
            rows.append(
                dict(n=n, pattern="dense", seconds=float("nan"), peak_bytes=float("nan"),
                     score_elems=dense_elems, density=1.0, kmax=n // args.block)
            )
            print(f"  n={n:6d} dense           SKIPPED (needs ~{dense_gb:.1f} GB of scores)")
        del cmask

        # ---- sparse patterns
        for p in patterns:
            plan = get_plan(p, n, args.block, causal=True, device=device)
            elems = args.batch * args.heads * n * plan.kmax * args.block
            t, mem = measure(
                lambda: block_sparse_attention(q, k, v, plan), device, args.warmup, args.reps
            )
            rows.append(
                dict(n=n, pattern=p, seconds=t, peak_bytes=mem, score_elems=elems,
                     density=density(plan.block_mask), kmax=plan.kmax)
            )
            spd = f"{dense_t/t:5.2f}x" if dense_t == dense_t else "  n/a"
            print(
                f"  n={n:6d} {p:14s} {t*1e3:9.2f} ms  peak {mem/1e6:8.1f} MB "
                f" score-elems {elems/1e6:8.1f}M  vs dense {spd}"
            )
        del q, k, v
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    out_csv = os.path.join(REPO, "results", "benchmark.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(REPO, "results", "hardware.json"), "w") as f:
        json.dump({**hw, "config": vars(args)}, f, indent=2)
    print(f"\nwrote {out_csv}")
    plot(rows, hw)
    return rows


def plot(rows, hw):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    os.makedirs(os.path.join(REPO, "plots"), exist_ok=True)
    pats = []
    for r in rows:
        if r["pattern"] not in pats:
            pats.append(r["pattern"])
    ns = sorted({r["n"] for r in rows})
    dev = hw.get("gpu", hw.get("cpu", "unknown"))
    sub = f"{dev} | torch {hw['torch']} | {hw['device']}"

    def series(p, field):
        d = {r["n"]: r[field] for r in rows if r["pattern"] == p}
        return [d.get(n, float("nan")) for n in ns]

    for field, ylabel, fname, scale in [
        ("seconds", "forward pass, ms (median)", "benchmark_time.png", 1e3),
        ("peak_bytes", "peak memory, MB", "benchmark_memory.png", 1e-6),
        ("score_elems", "materialised score entries", "benchmark_score_elems.png", 1.0),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for p in pats:
            ys = [
                y * scale if (y == y and y > 0) else float("nan")
                for y in series(p, field)
            ]  # RSS sampling occasionally reads 0 when the allocator reuses pages
            ax.plot(ns, ys, marker="o", label=p, lw=2 if p == "dense" else 1.5,
                    ls="--" if p == "dense" else "-")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs sequence length\n{sub}", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(REPO, "plots", fname)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--patterns", default="sliding_window,bigbird,dilated")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mem-budget-gb", type=float, default=6.0,
                    help="skip dense when the score matrix alone would exceed this")
    args = ap.parse_args()
    with torch.no_grad():
        run(args)


if __name__ == "__main__":
    main()
