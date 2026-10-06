#!/usr/bin/env bash
# Build each backend and run the end-to-end benchmark, saving JSON results.
#
# Usage: scripts/bench.sh [model_dir] [modes...]
#   scripts/bench.sh                         # ./gemma-3-4b-it, native (+ mps on macOS)
#   scripts/bench.sh ~/models/gemma-3-4b-it native blas mps
#
# Results go to bench/results/<host>-<mode>.json and are printed as a table.
# Close other heavy applications first: decoding is memory-bandwidth bound.
set -euo pipefail

MODEL=${1:-gemma-3-4b-it}
shift || true
MODES=("$@")
if [ ${#MODES[@]} -eq 0 ]; then
    MODES=(native)
    [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] && MODES+=(mps)
fi

PROMPTS=${PROMPTS:-64,256,1024}
GEN=${GEN:-128}
REPEATS=${REPEATS:-2}
HOST=$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)
mkdir -p bench/results

for mode in "${MODES[@]}"; do
    echo "=== building MODE=$mode"
    cc=cc; [ "$mode" = "mps" ] && cc=clang
    make --no-print-directory clean >/dev/null
    make --no-print-directory gemma3-bench MODE="$mode" CC="$cc" >/dev/null
    out="bench/results/$HOST-$mode.json"
    ./gemma3-bench -m "$MODEL" -p "$PROMPTS" -n "$GEN" -r "$REPEATS" -c 4096 \
        --label "$HOST $mode" --json "$out" 2>/dev/null
    echo "saved $out"
done
