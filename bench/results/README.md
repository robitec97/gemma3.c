# Benchmark results

MacBook Air M4 (4P + 6E CPU cores, 10-core GPU, 16 GB, macOS 26), Gemma 3 4B IT BF16.

* `m4-air-{mps,blas,cpu}.json` - output of `gemma3-bench -p 64,256,1024 -n 128 -d 2048 -c 4096`
  for `make mps`, `make blas` and `make`. Raw console output: `m4-air-suite.log`.
* The fanless Air throttles during long runs, so absolute numbers drift by up to ~25%.
  CPU generation (`decode`) in `m4-air-cpu.json` is the best of two short runs, which
  limits that drift; the value from the full suite run is kept as `decode_suite`.
* `validation-cpu.txt` - `tools/compare_logits.py` against the NumPy reference.

Plot with `python scripts/plot_bench.py --out docs/images/performance --result "make mps (Metal GPU)" bench/results/m4-air-mps.json --result "make blas (Accelerate)" bench/results/m4-air-blas.json --result "make (CPU, 10 threads)" bench/results/m4-air-cpu.json`.
