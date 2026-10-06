# Benchmark results

MacBook Air M4 (4P + 6E CPU cores, 10-core GPU, 16 GB, macOS 26), Gemma 3 4B IT BF16.

* `m4-air-*.json` - output of `gemma3-bench -p 64,256,1024 -n 128 -d 2048 -c 4096`,
  run back-to-back for this branch (`new-*`) and the original code (`base-*`, the same
  harness linked against commit 76fe597). Raw console output: `m4-air-suite.log`.
* The fanless Air throttles during long runs, so absolute numbers drift by up to ~25%.
  CPU generation (`decode`) for `new-cpu` / `base-cpu` comes from alternating short runs
  of both builds (best of two each), which cancels that drift; the suite's original
  values are kept as `decode_suite`.
* `validation-cpu.txt` - `tools/compare_logits.py` against the NumPy reference.

Plot with `python scripts/plot_bench.py --out docs/images/speedup --pair CPU bench/results/m4-air-base-cpu.json bench/results/m4-air-new-cpu.json --pair "Metal GPU" bench/results/m4-air-base-mps.json bench/results/m4-air-new-mps.json`.
