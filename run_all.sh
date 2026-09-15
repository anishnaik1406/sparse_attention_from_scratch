#!/usr/bin/env bash
# Reproduce every deliverable. ~15 min on a T4, longer on CPU.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1.3/1.4 correctness harness =="
python tests/test_correctness.py

echo
echo "== 1.4 NaN demonstration =="
python experiments/nan_demo.py

echo
echo "== 1.2 pattern visualisation =="
python experiments/plot_patterns.py

echo
echo "== 1.5 benchmark (512 -> 8192, forward only) =="
python benchmarks/benchmark.py "$@"

echo
echo "== 1.6 quality evaluation on TinyShakespeare =="
python experiments/train_char_gpt.py --context 512 --steps 2000

echo
echo "done. results/ has the CSVs, plots/ has the figures."
