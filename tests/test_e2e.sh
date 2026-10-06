#!/usr/bin/env bash
# End-to-end checks against a real model (run via `make test-model`).
# Usage: tests/test_e2e.sh ./gemma3 ./gemma-3-4b-it
set -u
BIN=${1:-./gemma3}
MODEL=${2:-gemma-3-4b-it}
pass=0; fail=0

check() {  # name, expected-substring, actual
    if [[ "$3" == *"$2"* ]]; then
        echo "  ok   $1"; pass=$((pass + 1))
    else
        echo "  FAIL $1"; echo "       expected to contain: $2"; echo "       got: ${3:0:300}"; fail=$((fail + 1))
    fi
}

echo "End-to-end tests ($BIN, model $MODEL)"

out=$("$BIN" -m "$MODEL" --tokenize -p $'<start_of_turn>user\nHello  world<end_of_turn>' 2>&1)
check "chat markers are single tokens" "Token IDs: [2, 105, 2364, 107, 9259, 138, 12392, 106]" "$out"

out=$("$BIN" -m "$MODEL" --logits -p "The capital of France is" 2>&1)
check "top-1 next token after 'The capital of France is'" "1       9079" "$out"

out=$("$BIN" -m "$MODEL" -q --greedy -n 32 -p "What is the capital of France? Answer in one sentence." 2>&1)
check "greedy chat answer" "Paris" "$out"

out=$(echo "What is 12 times 12? Reply with just the number." | "$BIN" -m "$MODEL" -q --greedy -n 16 2>&1)
check "prompt from stdin" "144" "$out"

out=$("$BIN" -m "$MODEL" -q --greedy -n 24 --stats -p "Say hello." 2>&1)
check "--stats line" "tok/s" "$out"

out=$("$BIN" -m "$MODEL" -q -n 8 -c 16 -p "Please tell me a long story about a dragon who learns to program in C" 2>&1)
check "prompt longer than context is rejected" "context size" "$out"

out=$(printf 'My name is Ada.\nWhat is my name?\n/exit\n' | "$BIN" -m "$MODEL" -i --greedy -n 24 --no-color --stats 2>&1)
check "interactive mode remembers previous turns" "Ada" "$(echo "$out" | tail -n +4)"
check "interactive mode reuses the KV cache" "cached" "$out"

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
